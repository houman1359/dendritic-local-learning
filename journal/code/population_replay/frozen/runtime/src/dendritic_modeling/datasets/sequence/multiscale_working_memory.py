"""Counterfactually paired multiscale working-memory sequence tasks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any

import torch
from torch.utils.data import Dataset


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_nonnegative(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _balanced_categories(
    n_items: int,
    n_categories: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return seeded categories whose counts differ by at most one."""

    quotient, remainder = divmod(n_items, n_categories)
    counts = torch.full((n_categories,), quotient, dtype=torch.long)
    if remainder:
        tie_order = torch.randperm(n_categories, generator=generator)
        counts[tie_order[:remainder]] += 1
    categories = torch.repeat_interleave(torch.arange(n_categories), counts)
    return categories[torch.randperm(n_items, generator=generator)]


def _sha_seed(base_seed: int, namespace: str) -> int:
    """Derive one deterministic, independently named Torch seed."""

    digest = hashlib.sha256(
        f"speeded-distractor-dms-v1|{int(base_seed)}|{namespace}".encode()
    ).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)
    return seed or 1


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _update_hash_with_tensor(
    digest: Any,
    *,
    name: str,
    tensor: torch.Tensor,
) -> None:
    value = tensor.detach().cpu().contiguous()
    descriptor = {
        "dtype": str(value.dtype),
        "name": name,
        "shape": list(value.shape),
    }
    digest.update(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(memoryview(value.numpy()).cast("B"))


class SpeededDistractorDMSDataset(Dataset):
    """Speeded distractor-interleaved delayed match-to-sample task.

    Samples are stored as consecutive counterfactual pairs. Pair members are
    bit-identical before the probe and differ only in probe identity and final
    memory label. Randomness is owned by a private CPU ``torch.Generator``;
    construction never reads from or advances ambient Torch RNG state.

    The packed target planes are ``fast_label``, ``fast_mask``,
    ``memory_label``, and ``memory_mask``.
    """

    N_IDENTITIES = 8
    FIXATION_END = 10
    SAMPLE_START = 10
    SAMPLE_END = 20
    RETENTION_START = 20
    DISTRACTOR_DURATION = 5
    PROBE_DURATION = 5
    MEMORY_RESPONSE_DURATION = 12
    EARLIEST_DISTRACTOR_ONSET = 32
    DISTRACTOR_PROBE_CLEARANCE = 20
    MIN_DISTRACTOR_ONSET_GAP = 16
    FAST_RESPONSE_LAG = 2
    FAST_RESPONSE_DEADLINE = 12

    FAST_LABEL = 0
    FAST_MASK = 1
    MEMORY_LABEL = 2
    MEMORY_MASK = 3
    TARGET_PLANES = 4
    TARGET_PLANE_NAMES = (
        "fast_label",
        "fast_mask",
        "memory_label",
        "memory_mask",
    )
    DISTRACTOR_IDENTITY_MODES = (
        "sample_conditioned_lures",
        "independent_uniform",
    )
    INPUT_INTERVENTIONS = (
        "initial_sample_cue_erased",
        "sample_identity_history_randomized",
        "probe_identity_erased",
        "salience_erased",
    )
    POINT_INTERVENTIONS = (
        "initial_sample_cue_erased",
        "probe_identity_erased",
        "salience_erased",
    )
    INDEPENDENT_PRIMARY_INFORMATION_CONTROLS = INPUT_INTERVENTIONS
    SAMPLE_CONDITIONED_INFORMATION_CONTROLS = (
        "sample_identity_history_randomized",
        "probe_identity_erased",
        "salience_erased",
    )
    RNG_NAMESPACES = (
        "geometry",
        "lure_counts",
        "lure_positions",
        "lure_subtypes",
        "distractor_identities",
        "vigilance_labels",
        "salience_evidence",
        "sensory_noise",
    )
    CONTENT_SCHEMA = "speeded-distractor-dms-content-v1"
    SPLIT_RECEIPT_SCHEMA = "speeded-distractor-dms-split-receipt-v1"
    INTERVENTION_RECEIPT_SCHEMA = "speeded-distractor-dms-input-intervention-receipt-v1"

    is_counterfactual_paired = True

    def __init__(
        self,
        n_samples: int | None = None,
        *,
        n_pairs: int | None = None,
        delay_support: Sequence[int] = (128, 160, 192, 224),
        n_distractors: int = 4,
        lure_fraction: float = 0.25,
        distractor_identity_mode: str = "sample_conditioned_lures",
        salience_mu: float = 0.8,
        circular_noise_std: float = 0.04,
        sequence_duration: int = 360,
        positive_input_encoding: bool = True,
        seed: int = 0,
        intervention_seed: int | None = None,
        split_name: str = "unspecified",
    ) -> None:
        if n_samples is not None and n_pairs is not None:
            raise ValueError("specify exactly one of n_samples or n_pairs, not both")
        if n_samples is None and n_pairs is None:
            n_samples = 10000
        if n_pairs is not None:
            self.n_pairs = _positive_integer(n_pairs, name="n_pairs")
            self.n_samples = 2 * self.n_pairs
        else:
            self.n_samples = _positive_integer(n_samples, name="n_samples")
            if self.n_samples % 2:
                raise ValueError("n_samples must be even for counterfactual pairs")
            self.n_pairs = self.n_samples // 2

        if len(delay_support) == 0:
            raise ValueError("delay_support must not be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in delay_support
        ):
            raise TypeError("delay_support must contain integers")
        self.delay_support = tuple(int(value) for value in delay_support)
        if any(value <= 0 for value in self.delay_support):
            raise ValueError("delay_support must contain positive values")
        if len(set(self.delay_support)) != len(self.delay_support):
            raise ValueError("delay_support must not contain duplicates")

        self.n_distractors = _positive_integer(n_distractors, name="n_distractors")
        if self.n_distractors % 2:
            raise ValueError(
                "n_distractors must be even so vigilance classes are exactly balanced"
            )
        self.lure_fraction = _finite_nonnegative(lure_fraction, name="lure_fraction")
        if self.lure_fraction > 1.0:
            raise ValueError("lure_fraction must not exceed one")
        if distractor_identity_mode not in self.DISTRACTOR_IDENTITY_MODES:
            choices = ", ".join(self.DISTRACTOR_IDENTITY_MODES)
            raise ValueError(f"distractor_identity_mode must be one of: {choices}")
        self.distractor_identity_mode = distractor_identity_mode
        if self.distractor_identity_mode == "independent_uniform" and (
            self.lure_fraction != 0.0
        ):
            raise ValueError(
                "independent_uniform distractor identities require lure_fraction=0"
            )
        self.salience_mu = _finite_nonnegative(salience_mu, name="salience_mu")
        self.circular_noise_std = _finite_nonnegative(
            circular_noise_std, name="circular_noise_std"
        )
        self.sequence_duration = _positive_integer(
            sequence_duration, name="sequence_duration"
        )
        if not isinstance(positive_input_encoding, bool):
            raise TypeError("positive_input_encoding must be a bool")
        self.positive_input_encoding = positive_input_encoding
        self.input_dim = 13 if positive_input_encoding else 12
        salience_names = (
            ("salience_positive", "salience_negative")
            if positive_input_encoding
            else ("salience_signed",)
        )
        self.input_channel_names = tuple(
            [f"circular_code_{index}" for index in range(self.N_IDENTITIES)]
            + list(salience_names)
            + ["sample_role", "distractor_role", "probe_role"]
        )
        self.target_plane_names = self.TARGET_PLANE_NAMES
        self.salience_slice = slice(8, 10) if positive_input_encoding else slice(8, 9)
        self.sample_role_channel = 10 if positive_input_encoding else 9
        self.distractor_role_channel = 11 if positive_input_encoding else 10
        self.probe_role_channel = 12 if positive_input_encoding else 11

        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer")
        self.seed = int(seed)
        if intervention_seed is not None and (
            isinstance(intervention_seed, bool)
            or not isinstance(intervention_seed, Integral)
        ):
            raise TypeError("intervention_seed must be an integer or None")
        self.intervention_seed = (
            None if intervention_seed is None else int(intervention_seed)
        )
        self.intervention_rng_seed = (
            None
            if self.intervention_seed is None
            else _sha_seed(self.intervention_seed, "sample_information_randomization")
        )
        if not isinstance(split_name, str) or not split_name.strip():
            raise ValueError("split_name must be a non-empty string")
        self.split_name = split_name
        self.rng_namespace_seeds = {
            namespace: _sha_seed(self.seed, namespace)
            for namespace in self.RNG_NAMESPACES
        }
        self._generators = {
            namespace: torch.Generator(device="cpu").manual_seed(namespace_seed)
            for namespace, namespace_seed in self.rng_namespace_seeds.items()
        }

        self._assert_timeline_feasible()
        self._generate()
        self._build_input_intervention_schedule()
        self._assert_realized_contract()
        self._build_provenance()

    def _scientific_configuration(self) -> dict[str, Any]:
        return {
            "schema": self.CONTENT_SCHEMA,
            "n_samples": self.n_samples,
            "n_pairs": self.n_pairs,
            "sequence_duration": self.sequence_duration,
            "delay_support": list(self.delay_support),
            "n_distractors": self.n_distractors,
            "lure_fraction": self.lure_fraction,
            "distractor_identity_mode": self.distractor_identity_mode,
            "salience_mu": self.salience_mu,
            "circular_noise_std": self.circular_noise_std,
            "positive_input_encoding": self.positive_input_encoding,
            "intervention_seed": self.intervention_seed,
            "input_channel_names": list(self.input_channel_names),
            "target_plane_names": list(self.target_plane_names),
        }

    def _assert_timeline_feasible(self) -> None:
        max_delay = max(self.delay_support)
        last_meaningful_step = (
            self.RETENTION_START
            + max_delay
            + self.PROBE_DURATION
            + self.MEMORY_RESPONSE_DURATION
        )
        if last_meaningful_step > self.sequence_duration:
            raise ValueError(
                "sequence_duration cannot contain the longest probe and memory window"
            )

        for delay in self.delay_support:
            latest = self.RETENTION_START + delay - self.DISTRACTOR_PROBE_CLEARANCE
            adjusted_latest = latest - (self.MIN_DISTRACTOR_ONSET_GAP - 1) * (
                self.n_distractors - 1
            )
            n_adjusted_slots = adjusted_latest - self.EARLIEST_DISTRACTOR_ONSET + 1
            if n_adjusted_slots < self.n_distractors:
                raise ValueError(
                    f"delay {delay} cannot fit {self.n_distractors} distractors "
                    "with the frozen onset constraints"
                )

    def _population_code(self) -> torch.Tensor:
        identity = torch.arange(self.N_IDENTITIES, dtype=torch.float32)[:, None]
        channel = torch.arange(self.N_IDENTITIES, dtype=torch.float32)[None, :]
        angle = 2.0 * math.pi * (identity - channel) / self.N_IDENTITIES
        return torch.exp(4.0 * (torch.cos(angle) - 1.0))

    def _sample_onsets(
        self,
        delay: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        latest = self.RETENTION_START + delay - self.DISTRACTOR_PROBE_CLEARANCE
        adjustment = self.MIN_DISTRACTOR_ONSET_GAP - 1
        adjusted_latest = latest - adjustment * (self.n_distractors - 1)
        candidates = torch.arange(self.EARLIEST_DISTRACTOR_ONSET, adjusted_latest + 1)
        chosen = (
            candidates[
                torch.randperm(candidates.numel(), generator=generator)[
                    : self.n_distractors
                ]
            ]
            .sort()
            .values
        )
        return chosen + adjustment * torch.arange(self.n_distractors)

    def _lure_type_schedule(self) -> torch.Tensor:
        """Return pair-level lure types with exact aggregate fraction."""

        total_events = self.n_pairs * self.n_distractors
        total_lures = min(
            total_events,
            max(0, math.floor(self.lure_fraction * total_events + 0.5)),
        )
        base_count, extra_pairs = divmod(total_lures, self.n_pairs)
        lure_counts = torch.full((self.n_pairs,), base_count, dtype=torch.long)
        if extra_pairs:
            order = torch.randperm(
                self.n_pairs,
                generator=self._generators["lure_counts"],
            )
            lure_counts[order[:extra_pairs]] += 1
        if bool((lure_counts > self.n_distractors).any()):
            raise RuntimeError("scheduled lure count exceeds distractors per trial")

        lure_types = torch.zeros(
            self.n_pairs,
            self.n_distractors,
            dtype=torch.long,
        )
        lure_slots: list[tuple[int, int]] = []
        position_generator = self._generators["lure_positions"]
        for pair_index, count in enumerate(lure_counts.tolist()):
            positions = torch.randperm(
                self.n_distractors,
                generator=position_generator,
            )[:count]
            lure_slots.extend((pair_index, int(position)) for position in positions)

        if lure_slots:
            subtypes = _balanced_categories(
                len(lure_slots),
                2,
                generator=self._generators["lure_subtypes"],
            )
            for (pair_index, event_index), subtype in zip(
                lure_slots,
                subtypes.tolist(),
                strict=True,
            ):
                lure_types[pair_index, event_index] = int(subtype) + 1
        return lure_types

    def _vigilance_schedule(
        self,
        *,
        delay_indices: torch.Tensor,
        sample_identities: torch.Tensor,
        lure_types: torch.Tensor,
    ) -> torch.Tensor:
        """Balance labels within frozen scientific strata without a trial quota."""

        strata: dict[tuple[int, int, int, int], list[tuple[int, int]]] = {}
        for pair_index in range(self.n_pairs):
            for event_index in range(self.n_distractors):
                key = (
                    int(delay_indices[pair_index]),
                    int(sample_identities[pair_index]),
                    int(lure_types[pair_index, event_index]),
                    event_index,
                )
                strata.setdefault(key, []).append((pair_index, event_index))

        generator = self._generators["vigilance_labels"]
        target_flags = torch.zeros_like(lure_types, dtype=torch.bool)
        keys_by_lure_type: dict[int, list[tuple[int, int, int, int]]] = {
            0: [],
            1: [],
            2: [],
        }
        for key in sorted(strata):
            keys_by_lure_type[key[2]].append(key)

        type_sizes = {
            lure_type: sum(len(strata[key]) for key in keys)
            for lure_type, keys in keys_by_lure_type.items()
        }
        odd_types = [
            lure_type for lure_type, size in type_sizes.items() if size % 2 == 1
        ]
        type_round_up = set()
        if odd_types:
            order = torch.randperm(len(odd_types), generator=generator)
            type_round_up = {
                odd_types[int(index)] for index in order[: len(odd_types) // 2]
            }

        for lure_type, keys in keys_by_lure_type.items():
            if not keys:
                continue
            type_target_total = type_sizes[lure_type] // 2
            if lure_type in type_round_up:
                type_target_total += 1
            base_targets = sum(len(strata[key]) // 2 for key in keys)
            odd_keys = [key for key in keys if len(strata[key]) % 2 == 1]
            extras_needed = type_target_total - base_targets
            if extras_needed < 0 or extras_needed > len(odd_keys):
                raise RuntimeError("vigilance stratum balancing became infeasible")
            extra_keys: set[tuple[int, int, int, int]] = set()
            if extras_needed:
                order = torch.randperm(len(odd_keys), generator=generator)
                extra_keys = {odd_keys[int(index)] for index in order[:extras_needed]}

            for key in keys:
                slots = strata[key]
                n_targets = len(slots) // 2 + int(key in extra_keys)
                within_order = torch.randperm(len(slots), generator=generator)
                for chosen in within_order[:n_targets].tolist():
                    pair_index, event_index = slots[int(chosen)]
                    target_flags[pair_index, event_index] = True

        if int(target_flags.sum()) * 2 != target_flags.numel():
            raise RuntimeError("global vigilance labels are not exactly balanced")
        return target_flags

    def _event_identity_schedule(
        self,
        *,
        sample_identities: torch.Tensor,
        lure_types: torch.Tensor,
    ) -> torch.Tensor:
        generator = self._generators["distractor_identities"]
        if self.distractor_identity_mode == "independent_uniform":
            identities = _balanced_categories(
                lure_types.numel(),
                self.N_IDENTITIES,
                generator=generator,
            ).reshape_as(lure_types)
            return identities

        identities = torch.empty_like(lure_types)
        for pair_index in range(self.n_pairs):
            sample_identity = int(sample_identities[pair_index])
            excluded = {
                sample_identity,
                (sample_identity - 1) % self.N_IDENTITIES,
                (sample_identity + 1) % self.N_IDENTITIES,
            }
            non_lures = torch.tensor(
                [value for value in range(self.N_IDENTITIES) if value not in excluded]
            )
            for event_index, lure_type in enumerate(lure_types[pair_index].tolist()):
                if lure_type == 1:
                    identities[pair_index, event_index] = sample_identity
                elif lure_type == 2:
                    direction = (
                        -1 if bool(torch.randint(0, 2, (), generator=generator)) else 1
                    )
                    identities[pair_index, event_index] = (
                        sample_identity + direction
                    ) % self.N_IDENTITIES
                else:
                    choice = torch.randint(
                        0,
                        non_lures.numel(),
                        (),
                        generator=generator,
                    )
                    identities[pair_index, event_index] = non_lures[choice]
        return identities

    def _noisy_code(
        self,
        identity: int,
        duration: int,
        population_codes: torch.Tensor,
        *,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(noise.shape) != (duration, self.N_IDENTITIES):
            raise ValueError("sensory noise has the wrong shape")
        activity = torch.clamp(population_codes[identity] + noise, min=0.0)
        return activity, noise

    def _write_salience(
        self,
        trial: torch.Tensor,
        start: int,
        evidence: torch.Tensor,
    ) -> None:
        if self.positive_input_encoding:
            trial[start : start + self.DISTRACTOR_DURATION, 8] = evidence.clamp_min(0)
            trial[start : start + self.DISTRACTOR_DURATION, 9] = (-evidence).clamp_min(
                0
            )
        else:
            trial[start : start + self.DISTRACTOR_DURATION, 8] = evidence

    def _generate(self) -> None:
        self.inputs = torch.zeros(
            self.n_samples, self.sequence_duration, self.input_dim
        )
        self.targets = torch.zeros(
            self.n_samples,
            self.sequence_duration,
            self.TARGET_PLANES,
        )
        self.pair_ids = torch.arange(self.n_pairs).repeat_interleave(2)
        self.sample_identities = torch.empty(self.n_samples, dtype=torch.long)
        self.probe_identities = torch.empty(self.n_samples, dtype=torch.long)
        self.match_labels = torch.empty(self.n_samples, dtype=torch.long)
        self.delays = torch.empty(self.n_samples, dtype=torch.long)
        self.distractor_onsets = torch.empty(
            self.n_samples, self.n_distractors, dtype=torch.long
        )
        self.distractor_identities = torch.empty_like(self.distractor_onsets)
        self.distractor_target_flags = torch.empty(
            self.n_samples, self.n_distractors, dtype=torch.bool
        )
        self.lure_flags = torch.empty_like(self.distractor_target_flags)
        self.lure_types = torch.empty_like(self.distractor_onsets)
        self.sensory_noise_realization_ids = torch.empty(
            self.n_samples,
            dtype=torch.long,
        )
        self.probe_onsets = torch.empty(self.n_samples, dtype=torch.long)
        self.seq_lengths = torch.empty(self.n_samples, dtype=torch.long)
        self.fast_response_starts = torch.empty_like(self.distractor_onsets)
        self.fast_response_ends = torch.empty_like(self.distractor_onsets)
        self.memory_response_starts = torch.empty(self.n_samples, dtype=torch.long)
        self.memory_response_ends = torch.empty(self.n_samples, dtype=torch.long)

        geometry_generator = self._generators["geometry"]
        delay_indices = _balanced_categories(
            self.n_pairs,
            len(self.delay_support),
            generator=geometry_generator,
        )
        sample_identities = _balanced_categories(
            self.n_pairs,
            self.N_IDENTITIES,
            generator=geometry_generator,
        )
        nonmatch_directions = _balanced_categories(
            self.n_pairs,
            2,
            generator=geometry_generator,
        )
        match_positions = _balanced_categories(
            self.n_pairs,
            2,
            generator=geometry_generator,
        )
        pair_lure_types = self._lure_type_schedule()
        pair_target_flags = self._vigilance_schedule(
            delay_indices=delay_indices,
            sample_identities=sample_identities,
            lure_types=pair_lure_types,
        )
        pair_event_identities = self._event_identity_schedule(
            sample_identities=sample_identities,
            lure_types=pair_lure_types,
        )
        population_codes = self._population_code()
        sensory_generator = self._generators["sensory_noise"]
        salience_generator = self._generators["salience_evidence"]
        self.pair_sample_sensory_noise = torch.empty(
            self.n_pairs,
            self.SAMPLE_END - self.SAMPLE_START,
            self.N_IDENTITIES,
        )
        self.pair_distractor_sensory_noise = torch.empty(
            self.n_pairs,
            self.n_distractors,
            self.DISTRACTOR_DURATION,
            self.N_IDENTITIES,
        )
        self.pair_probe_sensory_noise = torch.empty(
            self.n_pairs,
            self.PROBE_DURATION,
            self.N_IDENTITIES,
        )
        self.pair_salience_evidence = torch.empty(
            self.n_pairs,
            self.n_distractors,
            self.DISTRACTOR_DURATION,
        )
        noise_hashes: list[str] = []

        for pair_index in range(self.n_pairs):
            delay = self.delay_support[int(delay_indices[pair_index])]
            sample_identity = int(sample_identities[pair_index])
            probe_onset = self.RETENTION_START + delay
            memory_start = probe_onset + self.PROBE_DURATION
            memory_end = memory_start + self.MEMORY_RESPONSE_DURATION
            onsets = self._sample_onsets(
                delay,
                generator=geometry_generator,
            )
            event_identities = pair_event_identities[pair_index]
            target_flags = pair_target_flags[pair_index]
            lure_types = pair_lure_types[pair_index]

            sample_noise = (
                torch.randn(
                    self.SAMPLE_END - self.SAMPLE_START,
                    self.N_IDENTITIES,
                    generator=sensory_generator,
                )
                * self.circular_noise_std
            )
            distractor_noise = (
                torch.randn(
                    self.n_distractors,
                    self.DISTRACTOR_DURATION,
                    self.N_IDENTITIES,
                    generator=sensory_generator,
                )
                * self.circular_noise_std
            )
            probe_noise = (
                torch.randn(
                    self.PROBE_DURATION,
                    self.N_IDENTITIES,
                    generator=sensory_generator,
                )
                * self.circular_noise_std
            )
            evidence_noise = torch.randn(
                self.n_distractors,
                self.DISTRACTOR_DURATION,
                generator=salience_generator,
            )
            evidence_signs = target_flags.to(torch.float32).mul(2.0).sub(1.0)
            salience_evidence = (
                evidence_noise + evidence_signs[:, None] * self.salience_mu
            )
            self.pair_sample_sensory_noise[pair_index] = sample_noise
            self.pair_distractor_sensory_noise[pair_index] = distractor_noise
            self.pair_probe_sensory_noise[pair_index] = probe_noise
            self.pair_salience_evidence[pair_index] = salience_evidence

            noise_digest = hashlib.sha256()
            for name, value in (
                ("sample", sample_noise),
                ("distractor", distractor_noise),
                ("probe", probe_noise),
            ):
                _update_hash_with_tensor(noise_digest, name=name, tensor=value)
            noise_sha256 = noise_digest.hexdigest()
            noise_hashes.append(noise_sha256)
            noise_identifier = int(noise_sha256[:16], 16) % (2**63 - 1) or 1

            base_input = torch.zeros(self.sequence_duration, self.input_dim)
            sample_activity, _ = self._noisy_code(
                sample_identity,
                self.SAMPLE_END - self.SAMPLE_START,
                population_codes,
                noise=sample_noise,
            )
            base_input[self.SAMPLE_START : self.SAMPLE_END, :8] = sample_activity
            base_input[
                self.SAMPLE_START : self.SAMPLE_END, self.sample_role_channel
            ] = 1.0

            base_target = torch.zeros(self.sequence_duration, self.TARGET_PLANES)
            base_target[self.RETENTION_START : probe_onset, self.FAST_MASK] = 1.0
            for event_index, onset_tensor in enumerate(onsets):
                onset = int(onset_tensor)
                event_activity, _ = self._noisy_code(
                    int(event_identities[event_index]),
                    self.DISTRACTOR_DURATION,
                    population_codes,
                    noise=distractor_noise[event_index],
                )
                base_input[onset : onset + self.DISTRACTOR_DURATION, :8] = (
                    event_activity
                )
                base_input[
                    onset : onset + self.DISTRACTOR_DURATION,
                    self.distractor_role_channel,
                ] = 1.0
                evidence = salience_evidence[event_index]
                self._write_salience(base_input, onset, evidence)
                response_start = onset + self.FAST_RESPONSE_LAG
                response_end = onset + self.FAST_RESPONSE_DEADLINE + 1
                if bool(target_flags[event_index]):
                    base_target[response_start:response_end, self.FAST_LABEL] = 1.0

            direction = -1 if int(nonmatch_directions[pair_index]) == 0 else 1
            nonmatch_identity = (sample_identity + direction) % self.N_IDENTITIES
            probes = (sample_identity, nonmatch_identity)
            labels = (1, 0)
            if int(match_positions[pair_index]) == 1:
                probes = (nonmatch_identity, sample_identity)
                labels = (0, 1)
            for member_index, (probe_identity, label) in enumerate(
                zip(probes, labels, strict=True)
            ):
                sample_index = 2 * pair_index + member_index
                trial_input = base_input.clone()
                probe_activity, _ = self._noisy_code(
                    probe_identity,
                    self.PROBE_DURATION,
                    population_codes,
                    noise=probe_noise,
                )
                trial_input[probe_onset : probe_onset + self.PROBE_DURATION, :8] = (
                    probe_activity
                )
                trial_input[
                    probe_onset : probe_onset + self.PROBE_DURATION,
                    self.probe_role_channel,
                ] = 1.0
                trial_target = base_target.clone()
                trial_target[memory_start:memory_end, self.MEMORY_MASK] = 1.0
                trial_target[memory_start:memory_end, self.MEMORY_LABEL] = float(label)

                self.inputs[sample_index] = trial_input
                self.targets[sample_index] = trial_target
                self.sample_identities[sample_index] = sample_identity
                self.probe_identities[sample_index] = probe_identity
                self.match_labels[sample_index] = label
                self.delays[sample_index] = delay
                self.distractor_onsets[sample_index] = onsets
                self.distractor_identities[sample_index] = event_identities
                self.distractor_target_flags[sample_index] = target_flags
                self.lure_flags[sample_index] = lure_types > 0
                self.lure_types[sample_index] = lure_types
                self.sensory_noise_realization_ids[sample_index] = noise_identifier
                self.probe_onsets[sample_index] = probe_onset
                self.seq_lengths[sample_index] = memory_end
                self.fast_response_starts[sample_index] = (
                    onsets + self.FAST_RESPONSE_LAG
                )
                self.fast_response_ends[sample_index] = (
                    onsets + self.FAST_RESPONSE_DEADLINE + 1
                )
                self.memory_response_starts[sample_index] = memory_start
                self.memory_response_ends[sample_index] = memory_end
        self.sensory_noise_sha256_by_pair = tuple(noise_hashes)

    def _build_input_intervention_schedule(self) -> None:
        """Freeze the exhaustive identity-history rotation order per pair."""

        if self.intervention_rng_seed is None:
            self.sample_identity_rotation_order = None
            self.input_intervention_receipt = None
            self.input_intervention_receipt_sha256 = None
            return

        generator = torch.Generator(device="cpu").manual_seed(
            self.intervention_rng_seed
        )
        common_order = torch.randperm(
            self.N_IDENTITIES,
            generator=generator,
        )
        order = common_order.expand(self.n_pairs, -1).clone()
        expected = torch.arange(self.N_IDENTITIES).expand(self.n_pairs, -1)
        if not torch.equal(order.sort(dim=1).values, expected):
            raise RuntimeError("identity-history rotation rows must be permutations")
        counts = torch.bincount(order.reshape(-1), minlength=self.N_IDENTITIES)
        if not torch.all(counts == self.n_pairs):
            raise RuntimeError("identity-history rotations must be exhaustive per pair")
        self.sample_identity_rotation_order = order

        digest = hashlib.sha256()
        _update_hash_with_tensor(
            digest,
            name="sample_identity_rotation_order",
            tensor=order,
        )
        self.input_intervention_receipt = {
            "schema": self.INTERVENTION_RECEIPT_SCHEMA,
            "intervention_seed": self.intervention_seed,
            "intervention_rng_seed": self.intervention_rng_seed,
            "randomized_mode": "sample_identity_history_randomized",
            "aggregation": ("balanced_accuracy_over_expanded_trial_by_eight_dose_bank"),
            "offset_support": list(range(self.N_IDENTITIES)),
            "doses_per_trial": self.N_IDENTITIES,
            "pair_shared_order": True,
            "common_offset_order": common_order.tolist(),
            "exhaustive_per_pair": True,
            "rotation_order_sha256": digest.hexdigest(),
            "initial_sample_cue_erasure": {
                "erased_channels": list(range(self.N_IDENTITIES)),
                "erased_steps": [self.SAMPLE_START, self.SAMPLE_END],
                "sample_role_preserved": True,
                "eligibility_gate": (
                    self.distractor_identity_mode == "independent_uniform"
                ),
            },
            "probe_identity_erasure": {
                "erased_channels": list(range(self.N_IDENTITIES)),
                "probe_role_preserved": True,
                "pair_inputs_collapse": True,
            },
        }
        self.input_intervention_receipt_sha256 = _canonical_sha256(
            self.input_intervention_receipt
        )

    def _assert_realized_contract(self) -> None:
        if (
            not torch.isfinite(self.inputs).all()
            or not torch.isfinite(self.targets).all()
        ):
            raise RuntimeError("generated inputs and targets must be finite")
        if self.positive_input_encoding and bool((self.inputs < 0).any()):
            raise RuntimeError("positive input encoding generated a negative value")

        pair_view = self.inputs.reshape(
            self.n_pairs, 2, self.sequence_duration, self.input_dim
        )
        target_pair_view = self.targets.reshape(
            self.n_pairs,
            2,
            self.sequence_duration,
            self.TARGET_PLANES,
        )
        shared_pair_metadata = (
            "pair_ids",
            "sample_identities",
            "delays",
            "distractor_onsets",
            "distractor_identities",
            "distractor_target_flags",
            "lure_flags",
            "lure_types",
            "sensory_noise_realization_ids",
            "probe_onsets",
            "seq_lengths",
            "fast_response_starts",
            "fast_response_ends",
            "memory_response_starts",
            "memory_response_ends",
        )
        for name in shared_pair_metadata:
            value = getattr(self, name)
            if not torch.equal(value[::2], value[1::2]):
                raise RuntimeError(f"counterfactual pair metadata differs for {name}")

        population_codes = self._population_code()
        for pair_index in range(self.n_pairs):
            first = 2 * pair_index
            second = first + 1
            probe_onset = int(self.probe_onsets[first])
            probe_end = probe_onset + self.PROBE_DURATION
            if not torch.equal(
                pair_view[pair_index, 0, :probe_onset],
                pair_view[pair_index, 1, :probe_onset],
            ):
                raise RuntimeError("counterfactual pair differs before probe onset")
            if not torch.equal(
                pair_view[pair_index, 0, probe_end:],
                pair_view[pair_index, 1, probe_end:],
            ):
                raise RuntimeError("counterfactual pair differs after the probe")
            if not torch.equal(
                pair_view[pair_index, 0, :, 8:],
                pair_view[pair_index, 1, :, 8:],
            ):
                raise RuntimeError("counterfactual pair differs outside probe identity")
            if not torch.equal(
                target_pair_view[pair_index, 0, :, (0, 1, 3)],
                target_pair_view[pair_index, 1, :, (0, 1, 3)],
            ):
                raise RuntimeError("counterfactual pair supervision geometry differs")

            sample_identity = int(self.sample_identities[first])
            expected_sample = torch.clamp(
                population_codes[sample_identity]
                + self.pair_sample_sensory_noise[pair_index],
                min=0.0,
            )
            if not torch.equal(
                self.inputs[first, self.SAMPLE_START : self.SAMPLE_END, :8],
                expected_sample,
            ):
                raise RuntimeError("sample sensory noise reconstruction failed")

            for event_index, onset_tensor in enumerate(self.distractor_onsets[first]):
                onset = int(onset_tensor)
                event_identity = int(self.distractor_identities[first, event_index])
                expected_event = torch.clamp(
                    population_codes[event_identity]
                    + self.pair_distractor_sensory_noise[pair_index, event_index],
                    min=0.0,
                )
                if not torch.equal(
                    self.inputs[
                        first,
                        onset : onset + self.DISTRACTOR_DURATION,
                        :8,
                    ],
                    expected_event,
                ):
                    raise RuntimeError("distractor sensory noise reconstruction failed")
                evidence = self.pair_salience_evidence[pair_index, event_index]
                if self.positive_input_encoding:
                    expected_salience = torch.stack(
                        (evidence.clamp_min(0), (-evidence).clamp_min(0)),
                        dim=-1,
                    )
                else:
                    expected_salience = evidence[:, None]
                if not torch.equal(
                    self.inputs[
                        first,
                        onset : onset + self.DISTRACTOR_DURATION,
                        self.salience_slice,
                    ],
                    expected_salience,
                ):
                    raise RuntimeError("salience evidence reconstruction failed")

            for sample_index in (first, second):
                probe_identity = int(self.probe_identities[sample_index])
                expected_probe = torch.clamp(
                    population_codes[probe_identity]
                    + self.pair_probe_sensory_noise[pair_index],
                    min=0.0,
                )
                if not torch.equal(
                    self.inputs[sample_index, probe_onset:probe_end, :8],
                    expected_probe,
                ):
                    raise RuntimeError("probe sensory noise reconstruction failed")

        labels_by_pair = self.match_labels.reshape(self.n_pairs, 2)
        if not torch.all(labels_by_pair.sum(dim=1) == 1):
            raise RuntimeError("each pair must contain one match and one non-match")
        if (
            abs(
                int((labels_by_pair[:, 0] == 1).sum())
                - int((labels_by_pair[:, 0] == 0).sum())
            )
            > 1
        ):
            raise RuntimeError("pair order reveals the memory label")
        for delay in self.delay_support:
            labels = self.match_labels[self.delays == delay]
            if (
                labels.numel()
                and abs(int((labels == 1).sum()) - int((labels == 0).sum())) > 1
            ):
                raise RuntimeError("memory classes are not balanced within delay")

        pair_delays = self.delays[::2]
        delay_counts = torch.tensor(
            [int((pair_delays == delay).sum()) for delay in self.delay_support]
        )
        if int(delay_counts.max() - delay_counts.min()) > 1:
            raise RuntimeError("delay counts differ by more than one")
        pair_identities = self.sample_identities[::2]
        identity_counts = torch.bincount(pair_identities, minlength=self.N_IDENTITIES)
        if int(identity_counts.max() - identity_counts.min()) > 1:
            raise RuntimeError("sample identity counts differ by more than one")
        pair_lure_types = self.lure_types[::2]
        pair_targets = self.distractor_target_flags[::2]
        total_pair_events = pair_targets.numel()
        if int(pair_targets.sum()) * 2 != total_pair_events:
            raise RuntimeError("vigilance classes are not globally balanced")

        expected_lures = math.floor(self.lure_fraction * total_pair_events + 0.5)
        if int((pair_lure_types > 0).sum()) != expected_lures:
            raise RuntimeError("realized lure fraction differs from its exact schedule")
        if (
            abs(int((pair_lure_types == 1).sum()) - int((pair_lure_types == 2).sum()))
            > 1
        ):
            raise RuntimeError("exact and adjacent lure counts are not balanced")
        for lure_type in (0, 1, 2):
            selected_targets = pair_targets[pair_lure_types == lure_type]
            if (
                selected_targets.numel()
                and abs(
                    int(selected_targets.sum())
                    - int(selected_targets.numel() - selected_targets.sum())
                )
                > 1
            ):
                raise RuntimeError("lure subtype predicts the vigilance class")

        maximum_factor_imbalance = 0
        pair_delay_indices = torch.tensor(
            [self.delay_support.index(int(value)) for value in pair_delays]
        )
        for delay_index in range(len(self.delay_support)):
            for sample_identity in range(self.N_IDENTITIES):
                for lure_type in (0, 1, 2):
                    for event_index in range(self.n_distractors):
                        selected_pairs = (
                            (pair_delay_indices == delay_index)
                            & (pair_identities == sample_identity)
                            & (pair_lure_types[:, event_index] == lure_type)
                        )
                        selected_targets = pair_targets[
                            selected_pairs,
                            event_index,
                        ]
                        if not selected_targets.numel():
                            continue
                        imbalance = abs(
                            int(selected_targets.sum())
                            - int(selected_targets.numel() - selected_targets.sum())
                        )
                        maximum_factor_imbalance = max(
                            maximum_factor_imbalance,
                            imbalance,
                        )
        if maximum_factor_imbalance > 1:
            raise RuntimeError("a vigilance counterbalance stratum is imbalanced")
        self.maximum_vigilance_factor_imbalance = maximum_factor_imbalance

        pair_event_identities = self.distractor_identities[::2]
        event_identity_counts = torch.bincount(
            pair_event_identities.reshape(-1),
            minlength=self.N_IDENTITIES,
        )
        if self.distractor_identity_mode == "independent_uniform" and (
            int(event_identity_counts.max() - event_identity_counts.min()) > 1
        ):
            raise RuntimeError("independent distractor identities are not balanced")
        self.distractor_identity_counts = event_identity_counts

        nonmatch_selection = self.match_labels == 0
        nonmatch_delta = (
            self.probe_identities[nonmatch_selection]
            - self.sample_identities[nonmatch_selection]
        ) % self.N_IDENTITIES
        if not torch.all((nonmatch_delta == 1) | (nonmatch_delta == 7)):
            raise RuntimeError("non-match probes must be adjacent to the sample")
        if abs(int((nonmatch_delta == 1).sum()) - int((nonmatch_delta == 7).sum())) > 1:
            raise RuntimeError("adjacent non-match directions are not balanced")

        gaps = torch.diff(self.distractor_onsets, dim=1)
        if gaps.numel() and bool((gaps < self.MIN_DISTRACTOR_ONSET_GAP).any()):
            raise RuntimeError("distractor response windows can overlap")
        if bool((self.distractor_onsets[:, 0] < self.EARLIEST_DISTRACTOR_ONSET).any()):
            raise RuntimeError("distractor begins before its earliest valid onset")
        if bool(
            (
                self.distractor_onsets[:, -1]
                > self.probe_onsets - self.DISTRACTOR_PROBE_CLEARANCE
            ).any()
        ):
            raise RuntimeError("distractor violates probe clearance")
        if bool((self.seq_lengths > self.sequence_duration).any()):
            raise RuntimeError("a meaningful sequence extends beyond the tensor")

        masks = self.targets[..., (self.FAST_MASK, self.MEMORY_MASK)]
        if not torch.all((masks == 0) | (masks == 1)):
            raise RuntimeError("packed target masks must be binary")
        for sample_index in range(self.n_samples):
            probe_onset = int(self.probe_onsets[sample_index])
            memory_start = int(self.memory_response_starts[sample_index])
            memory_end = int(self.memory_response_ends[sample_index])
            if not torch.all(
                self.targets[
                    sample_index,
                    self.RETENTION_START : probe_onset,
                    self.FAST_MASK,
                ]
                == 1
            ):
                raise RuntimeError("fast mask does not cover the full retention")
            if not torch.all(
                self.targets[sample_index, memory_start:memory_end, self.MEMORY_MASK]
                == 1
            ):
                raise RuntimeError("memory mask does not cover its frozen window")
            seq_length = int(self.seq_lengths[sample_index])
            if bool(self.targets[sample_index, seq_length:].any()):
                raise RuntimeError("padding must contain zero target supervision")
            if bool(self.inputs[sample_index, seq_length:].any()):
                raise RuntimeError("padding inputs must be zero")
            for condition in self.POINT_INTERVENTIONS:
                transformed = self.intervention_transform(sample_index, condition)
                allowed = self._point_intervention_mask(sample_index, condition)
                if not torch.equal(
                    transformed[~allowed],
                    self.inputs[sample_index][~allowed],
                ):
                    raise RuntimeError(f"{condition} changed a non-targeted input")
                if bool(transformed[allowed].any()):
                    raise RuntimeError(f"{condition} did not erase its target")

        if self.sample_identity_rotation_order is not None:
            if self.input_intervention_receipt is None or (
                _canonical_sha256(self.input_intervention_receipt)
                != self.input_intervention_receipt_sha256
            ):
                raise RuntimeError("input intervention receipt is not hash-bound")
            for pair_index in range(self.n_pairs):
                zero_dose = int(
                    torch.nonzero(
                        self.sample_identity_rotation_order[pair_index] == 0,
                        as_tuple=False,
                    ).item()
                )
                for sample_index in (2 * pair_index, 2 * pair_index + 1):
                    transformed = self.sample_identity_history_randomized(
                        sample_index,
                        zero_dose,
                    )
                    if not torch.equal(transformed, self.inputs[sample_index]):
                        raise RuntimeError(
                            "zero-offset identity-history dose changed the input"
                        )

    def _counterbalance_summary(self) -> dict[str, Any]:
        pair_lure_types = self.lure_types[::2]
        pair_targets = self.distractor_target_flags[::2]
        total_events = pair_targets.numel()
        realized_lure_count = int((pair_lure_types > 0).sum())
        lure_type_names = {0: "non_lure", 1: "exact", 2: "adjacent"}
        lure_type_counts = {
            lure_type_names[lure_type]: int((pair_lure_types == lure_type).sum())
            for lure_type in lure_type_names
        }
        lure_type_target_counts = {
            lure_type_names[lure_type]: int(
                pair_targets[pair_lure_types == lure_type].sum()
            )
            for lure_type in lure_type_names
        }
        return {
            "schema": "speeded-distractor-dms-counterbalance-v1",
            "distractor_identity_mode": self.distractor_identity_mode,
            "distractor_identity_counts": self.distractor_identity_counts.tolist(),
            "n_pair_events": total_events,
            "target_events": int(pair_targets.sum()),
            "non_target_events": int(total_events - pair_targets.sum()),
            "requested_lure_fraction": self.lure_fraction,
            "realized_lure_count": realized_lure_count,
            "realized_lure_fraction": realized_lure_count / total_events,
            "lure_type_counts": lure_type_counts,
            "lure_type_target_counts": lure_type_target_counts,
            "maximum_factor_imbalance": self.maximum_vigilance_factor_imbalance,
            "per_trial_target_count_support": sorted(
                int(value) for value in torch.unique(pair_targets.sum(dim=1)).tolist()
            ),
        }

    def _build_provenance(self) -> None:
        self.counterbalance_receipt = self._counterbalance_summary()
        self.counterbalance_sha256 = _canonical_sha256(self.counterbalance_receipt)
        scientific_metadata = {
            **self._scientific_configuration(),
            "counterbalance_receipt": self.counterbalance_receipt,
            "input_intervention_receipt": self.input_intervention_receipt,
            "sensory_noise_sha256_by_pair": list(self.sensory_noise_sha256_by_pair),
        }
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                scientific_metadata,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        tensor_names = (
            "inputs",
            "targets",
            "pair_ids",
            "sample_identities",
            "probe_identities",
            "match_labels",
            "delays",
            "distractor_onsets",
            "distractor_identities",
            "distractor_target_flags",
            "lure_flags",
            "lure_types",
            "sensory_noise_realization_ids",
            "probe_onsets",
            "seq_lengths",
            "fast_response_starts",
            "fast_response_ends",
            "memory_response_starts",
            "memory_response_ends",
            "pair_sample_sensory_noise",
            "pair_distractor_sensory_noise",
            "pair_probe_sensory_noise",
            "pair_salience_evidence",
        )
        for name in tensor_names:
            _update_hash_with_tensor(digest, name=name, tensor=getattr(self, name))
        if self.sample_identity_rotation_order is not None:
            _update_hash_with_tensor(
                digest,
                name="sample_identity_rotation_order",
                tensor=self.sample_identity_rotation_order,
            )
        self.content_sha256 = digest.hexdigest()
        self.split_receipt = {
            "schema": self.SPLIT_RECEIPT_SCHEMA,
            "split_name": self.split_name,
            "seed": self.seed,
            "rng_namespace_seeds": dict(self.rng_namespace_seeds),
            "n_samples": self.n_samples,
            "n_pairs": self.n_pairs,
            "content_sha256": self.content_sha256,
            "counterbalance_sha256": self.counterbalance_sha256,
            "input_intervention_receipt_sha256": (
                self.input_intervention_receipt_sha256
            ),
        }
        self.split_receipt_sha256 = _canonical_sha256(self.split_receipt)
        self.split_metadata = {
            **self._scientific_configuration(),
            **self.split_receipt,
            "split_receipt_sha256": self.split_receipt_sha256,
        }

    def _validated_sample_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("sample index must be an integer")
        resolved = int(index)
        if resolved < 0 or resolved >= self.n_samples:
            raise IndexError("sample index is outside the dataset")
        return resolved

    def _validated_point_intervention(self, condition: str) -> str:
        if condition not in self.POINT_INTERVENTIONS:
            choices = ", ".join(self.POINT_INTERVENTIONS)
            raise ValueError(f"point intervention must be one of: {choices}")
        return condition

    def _point_intervention_mask(
        self,
        index: int,
        condition: str,
    ) -> torch.Tensor:
        sample_index = self._validated_sample_index(index)
        mode = self._validated_point_intervention(condition)
        mask = torch.zeros(
            self.sequence_duration,
            self.input_dim,
            dtype=torch.bool,
        )
        if mode == "initial_sample_cue_erased":
            mask[self.SAMPLE_START : self.SAMPLE_END, :8] = True
        elif mode == "probe_identity_erased":
            probe_onset = int(self.probe_onsets[sample_index])
            probe_end = probe_onset + self.PROBE_DURATION
            mask[probe_onset:probe_end, :8] = True
        else:
            mask[:, self.salience_slice] = True
        return mask

    def intervention_transform(self, index: int, condition: str) -> torch.Tensor:
        """Return one cloned input with one frozen point intervention applied."""

        sample_index = self._validated_sample_index(index)
        mode = self._validated_point_intervention(condition)
        transformed = self.inputs[sample_index].clone()
        transformed[self._point_intervention_mask(sample_index, mode)] = 0.0
        return transformed

    def intervention_transform_batch(
        self,
        indices: Sequence[int] | torch.Tensor,
        condition: str,
    ) -> torch.Tensor:
        """Apply one point intervention to an explicit sequence of indices."""

        mode = self._validated_point_intervention(condition)
        if isinstance(indices, torch.Tensor):
            if indices.ndim != 1:
                raise ValueError("intervention indices must be one-dimensional")
            index_values = indices.detach().cpu().tolist()
        else:
            index_values = list(indices)
        if not index_values:
            return torch.empty(
                0,
                self.sequence_duration,
                self.input_dim,
                dtype=self.inputs.dtype,
            )
        return torch.stack(
            [self.intervention_transform(index, mode) for index in index_values],
            dim=0,
        )

    def _validated_rotation_dose(self, dose_index: int) -> int:
        if isinstance(dose_index, bool) or not isinstance(dose_index, Integral):
            raise TypeError("rotation dose index must be an integer")
        result = int(dose_index)
        if result < 0 or result >= self.N_IDENTITIES:
            raise IndexError("rotation dose index is outside the exhaustive support")
        if self.sample_identity_rotation_order is None:
            raise RuntimeError(
                "sample identity history randomization requires intervention_seed"
            )
        return result

    def _rerender_identity_history(
        self,
        sample_index: int,
        offset: int,
    ) -> torch.Tensor:
        transformed = self.inputs[sample_index].clone()
        pair_index = int(self.pair_ids[sample_index])
        population_codes = self._population_code()
        sample_identity = (
            int(self.sample_identities[sample_index]) + offset
        ) % self.N_IDENTITIES
        transformed[self.SAMPLE_START : self.SAMPLE_END, :8] = torch.clamp(
            population_codes[sample_identity]
            + self.pair_sample_sensory_noise[pair_index],
            min=0.0,
        )
        for event_index, onset_tensor in enumerate(
            self.distractor_onsets[sample_index]
        ):
            onset = int(onset_tensor)
            identity = (
                int(self.distractor_identities[sample_index, event_index]) + offset
            ) % self.N_IDENTITIES
            transformed[onset : onset + self.DISTRACTOR_DURATION, :8] = torch.clamp(
                population_codes[identity]
                + self.pair_distractor_sensory_noise[pair_index, event_index],
                min=0.0,
            )
        return transformed

    def sample_identity_history_randomized(
        self,
        index: int,
        dose_index: int,
    ) -> torch.Tensor:
        """Return one dose of the exhaustive identity-history control.

        Every circular identity signal before the probe is shifted by the same
        pair-shared offset. Timing, role channels, salience, stored sensory-noise
        draws, and the probe are unchanged. Evaluation over all eight doses
        marginalizes the sample--probe identity relation. The paper evaluator
        computes balanced accuracy over the expanded trial-by-dose bank; it
        does not threshold an averaged logit.
        """

        sample_index = self._validated_sample_index(index)
        dose = self._validated_rotation_dose(dose_index)
        pair_index = int(self.pair_ids[sample_index])
        offset = int(self.sample_identity_rotation_order[pair_index, dose])
        return self._rerender_identity_history(sample_index, offset)

    def sample_identity_history_randomized_batch(
        self,
        indices: Sequence[int] | torch.Tensor,
        dose_index: int,
    ) -> torch.Tensor:
        """Apply one exhaustive rotation dose to explicit sample indices."""

        dose = self._validated_rotation_dose(dose_index)
        if isinstance(indices, torch.Tensor):
            if indices.ndim != 1:
                raise ValueError("intervention indices must be one-dimensional")
            index_values = indices.detach().cpu().tolist()
        else:
            index_values = list(indices)
        if not index_values:
            return torch.empty(
                0,
                self.sequence_duration,
                self.input_dim,
                dtype=self.inputs.dtype,
            )
        return torch.stack(
            [
                self.sample_identity_history_randomized(index, dose)
                for index in index_values
            ],
            dim=0,
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index], self.seq_lengths[index]


__all__ = ["SpeededDistractorDMSDataset"]
