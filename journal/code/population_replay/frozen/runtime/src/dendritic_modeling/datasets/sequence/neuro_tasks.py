"""Sequence dataset task implementations."""

from __future__ import annotations

import logging
import math
import os
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_STRINGER_SESSION_RE = re.compile(
    r"(?P<expt>.+)_(?P<mouse>[^_]+)_(?P<date>\d{4}_\d{2}_\d{2})_(?P<block>\d+)\.npy$"
)


def _resolve_stringer_session_file(data_dir: str, session_id: str) -> tuple[str, int]:
    """Resolve a local Stringer session id to response filename and running index."""
    data_dir_path = Path(data_dir)
    session_key = str(session_id)
    if not session_key.strip():
        raise ValueError("stringer_v1 session_id must be non-empty")

    direct_name = session_key if session_key.endswith(".npy") else f"{session_key}.npy"
    direct_path = data_dir_path / direct_name
    if direct_path.exists():
        candidates = [direct_path]
    else:
        candidates = [
            path
            for path in sorted(data_dir_path.glob(f"*{session_key}*.npy"))
            if path.name not in {"all_running.npy", "all_depths.npy", "database.npy"}
        ]

    if not candidates:
        raise FileNotFoundError(
            f"No local Stringer response file matched session_id={session_id!r} "
            f"in {data_dir_path}"
        )
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates[:8])
        raise ValueError(
            f"Stringer session_id={session_id!r} matched multiple files: {names}"
        )

    grating_file = candidates[0].name
    match = _STRINGER_SESSION_RE.match(grating_file)
    if match is None:
        raise ValueError(
            f"Could not infer Stringer session metadata from filename {grating_file!r}; "
            "use the standard gratings_drifting_<mouse>_<date>_<block>.npy naming."
        )

    database_path = data_dir_path / "database.npy"
    if not database_path.exists():
        raise FileNotFoundError(
            f"Stringer session_id resolution requires {database_path}"
        )
    database = [
        dict(entry) for entry in np.load(database_path, allow_pickle=True).tolist()
    ]
    groups = match.groupdict()
    for idx, entry in enumerate(database):
        if (
            str(entry.get("expt")) == groups["expt"]
            and str(entry.get("mouse_name")) == groups["mouse"]
            and str(entry.get("date")) == groups["date"]
            and str(entry.get("block")) == groups["block"]
        ):
            return grating_file, int(idx)

    raise ValueError(
        f"Could not find metadata entry for Stringer session file {grating_file!r}"
    )


class OculomotorDelayedResponseDataset(Dataset):
    """Delayed-response task with a remembered angular cue.

    A brief cue indicates one of ``n_targets`` angular locations. The network
    must retain that location over a silent delay and report it after a go cue.
    This is a simple delayed-response benchmark motivated by oculomotor memory
    tasks used in systems neuroscience.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        n_targets: int = 8,
        input_tuning_dim: int = 16,
        fixation_duration: int = 6,
        cue_duration: int = 8,
        min_delay: int = 10,
        max_delay: int = 100,
        go_duration: int = 8,
        tuning_kappa: float = 2.0,
        cue_noise_std: float = 0.05,
        positive_input_encoding: bool = False,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.n_targets = n_targets
        self.input_tuning_dim = input_tuning_dim
        self.fixation_duration = fixation_duration
        self.cue_duration = cue_duration
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.go_duration = go_duration
        self.positive_input_encoding = bool(positive_input_encoding)
        self.input_dim = input_tuning_dim + 2
        self.max_total_len = fixation_duration + cue_duration + max_delay + go_duration

        self.inputs = torch.zeros(n_samples, self.max_total_len, self.input_dim)
        self.targets = torch.zeros(n_samples, dtype=torch.long)
        self.delays = torch.zeros(n_samples, dtype=torch.long)

        pref_angles = torch.linspace(
            0.0, 2.0 * math.pi, steps=input_tuning_dim + 1, dtype=torch.float32
        )[:-1]
        target_angles = torch.linspace(
            0.0, 2.0 * math.pi, steps=n_targets + 1, dtype=torch.float32
        )[:-1]

        for i in range(n_samples):
            target_idx = torch.randint(0, n_targets, (1,)).item()
            delay = torch.randint(min_delay, max_delay + 1, (1,)).item()
            total_len = fixation_duration + cue_duration + delay + go_duration

            angle = target_angles[target_idx]
            cue_pattern = torch.exp(tuning_kappa * torch.cos(pref_angles - angle))
            cue_pattern = cue_pattern / cue_pattern.max().clamp(min=1e-8)

            self.targets[i] = target_idx
            self.delays[i] = delay

            # Fixation stays on through the silent memory period.
            self.inputs[
                i, : fixation_duration + cue_duration + delay, input_tuning_dim
            ] = 1.0

            cue_start = fixation_duration
            cue_end = fixation_duration + cue_duration
            self.inputs[i, cue_start:cue_end, :input_tuning_dim] = (
                cue_pattern.unsqueeze(0)
            )
            self.inputs[
                i, cue_start:cue_end, :input_tuning_dim
            ] += cue_noise_std * torch.randn(cue_duration, input_tuning_dim)
            if self.positive_input_encoding:
                self.inputs[i, cue_start:cue_end, :input_tuning_dim].clamp_(min=0.0)

            go_start = fixation_duration + cue_duration + delay
            self.inputs[i, go_start:total_len, input_tuning_dim + 1] = 1.0

        self.seq_lengths = (
            self.fixation_duration + self.cue_duration + self.delays + self.go_duration
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq_lengths = self.seq_lengths[idx]
        if torch.is_tensor(seq_lengths) and seq_lengths.dim() == 0:
            seq_lengths = int(seq_lengths.item())
        return self.inputs[idx], self.targets[idx], seq_lengths


ROMO_PAIR_SAMPLING_INDEPENDENT_UNEQUAL_V1 = "independent_unequal_v1"
ROMO_PAIR_SAMPLING_BALANCED_SECOND_STIMULUS_V1 = "balanced_second_stimulus_v1"
ROMO_PAIR_SAMPLING_POLICIES = frozenset(
    {
        ROMO_PAIR_SAMPLING_INDEPENDENT_UNEQUAL_V1,
        ROMO_PAIR_SAMPLING_BALANCED_SECOND_STIMULUS_V1,
    }
)


class RomoDelayComparisonDataset(Dataset):
    """Delayed comparison of two vibrotactile frequencies.

    The first stimulus frequency must be retained over a silent delay and then
    compared against a second frequency. The target is binary: whether the
    second frequency is higher than the first.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        frequency_values: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
        stimulus_duration: int = 20,
        min_delay: int = 10,
        max_delay: int = 100,
        response_duration: int = 8,
        noise_std: float = 0.08,
        positive_input_encoding: bool = False,
        tuning_dim: int = 8,
        tuning_sigma: float | None = None,
        pair_sampling: str = ROMO_PAIR_SAMPLING_INDEPENDENT_UNEQUAL_V1,
    ):
        super().__init__()
        if pair_sampling not in ROMO_PAIR_SAMPLING_POLICIES:
            raise ValueError(
                "pair_sampling must be one of "
                f"{sorted(ROMO_PAIR_SAMPLING_POLICIES)}, got {pair_sampling!r}"
            )
        if len(frequency_values) < 2:
            raise ValueError("frequency_values must contain at least two entries")
        if pair_sampling == ROMO_PAIR_SAMPLING_BALANCED_SECOND_STIMULUS_V1:
            if n_samples <= 0 or n_samples % 2 != 0:
                raise ValueError(
                    "balanced_second_stimulus_v1 requires a positive even n_samples"
                )
            if len(frequency_values) < 3:
                raise ValueError(
                    "balanced_second_stimulus_v1 requires at least three frequencies"
                )
            if any(
                float(right) <= float(left)
                for left, right in pairwise(frequency_values)
            ):
                raise ValueError(
                    "balanced_second_stimulus_v1 requires strictly increasing "
                    "frequency_values"
                )
        self.n_samples = n_samples
        self.frequency_values = tuple(float(v) for v in frequency_values)
        self.stimulus_duration = stimulus_duration
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.response_duration = response_duration
        self.positive_input_encoding = bool(positive_input_encoding)
        self.tuning_dim = int(tuning_dim)
        self.pair_sampling = str(pair_sampling)
        self.input_dim = (self.tuning_dim if self.positive_input_encoding else 1) + 3
        self.max_total_len = 2 * stimulus_duration + max_delay + response_duration

        self.inputs = torch.zeros(n_samples, self.max_total_len, self.input_dim)
        self.targets = torch.zeros(n_samples, dtype=torch.long)
        self.delays = torch.zeros(n_samples, dtype=torch.long)
        # Preserve the latent comparison variables so evaluation banks can
        # report accuracy as a function of delay and comparison difficulty
        # without trying to infer frequencies back from noisy input traces.
        self.frequency1_indices = torch.zeros(n_samples, dtype=torch.long)
        self.frequency2_indices = torch.zeros(n_samples, dtype=torch.long)
        self.frequency1_values = torch.zeros(n_samples, dtype=torch.float32)
        self.frequency2_values = torch.zeros(n_samples, dtype=torch.float32)
        # ``pair_ids`` is -1 for the historical independent sampler. Under the
        # shortcut-resistant policy, exactly two rows share each non-negative
        # pair ID: one lower-first positive trial and one higher-first negative
        # trial with byte-identical input from delay onset onward.
        self.pair_ids = torch.full((n_samples,), -1, dtype=torch.long)
        self.comparison_distance_indices = torch.zeros(n_samples, dtype=torch.long)

        phase_grid = torch.arange(stimulus_duration, dtype=torch.float32)
        if self.positive_input_encoding:
            freq_min = min(self.frequency_values)
            freq_max = max(self.frequency_values)
            self.frequency_pref = torch.linspace(
                float(freq_min), float(freq_max), steps=self.tuning_dim
            )
            spacing = (
                float(freq_max - freq_min) / max(self.tuning_dim - 1, 1)
                if self.tuning_dim > 1
                else 1.0
            )
            self.tuning_sigma = (
                float(tuning_sigma) if tuning_sigma is not None else spacing
            )

        def frequency_signal(frequency: float) -> torch.Tensor:
            if self.positive_input_encoding:
                return self._positive_frequency_code(frequency, noise_std)
            signal = torch.sin(
                2.0 * math.pi * frequency * phase_grid / stimulus_duration
            )
            signal = signal + noise_std * torch.randn_like(signal)
            return signal.unsqueeze(-1)

        def write_trial(
            index: int,
            *,
            f1_idx: int,
            f2_idx: int,
            delay: int,
            sig1: torch.Tensor,
            sig2: torch.Tensor,
            pair_id: int = -1,
        ) -> None:
            f1 = self.frequency_values[f1_idx]
            f2 = self.frequency_values[f2_idx]
            total_len = 2 * stimulus_duration + delay + response_duration
            self.targets[index] = int(f2 > f1)
            self.delays[index] = delay
            self.frequency1_indices[index] = int(f1_idx)
            self.frequency2_indices[index] = int(f2_idx)
            self.frequency1_values[index] = float(f1)
            self.frequency2_values[index] = float(f2)
            self.pair_ids[index] = int(pair_id)
            self.comparison_distance_indices[index] = abs(f2_idx - f1_idx)

            s1_start = 0
            s1_end = stimulus_duration
            s2_start = stimulus_duration + delay
            s2_end = s2_start + stimulus_duration
            rsp_start = s2_end

            stim_dim = self.tuning_dim if self.positive_input_encoding else 1
            self.inputs[index, s1_start:s1_end, :stim_dim] = sig1
            self.inputs[index, s1_start:s1_end, stim_dim] = 1.0
            self.inputs[index, s2_start:s2_end, :stim_dim] = sig2
            self.inputs[index, s2_start:s2_end, stim_dim + 1] = 1.0
            self.inputs[index, rsp_start:total_len, stim_dim + 2] = 1.0

        if pair_sampling == ROMO_PAIR_SAMPLING_INDEPENDENT_UNEQUAL_V1:
            for i in range(n_samples):
                f1_idx = torch.randint(0, len(self.frequency_values), (1,)).item()
                f2_idx = torch.randint(0, len(self.frequency_values), (1,)).item()
                while f2_idx == f1_idx:
                    f2_idx = torch.randint(0, len(self.frequency_values), (1,)).item()
                delay = torch.randint(min_delay, max_delay + 1, (1,)).item()
                write_trial(
                    i,
                    f1_idx=f1_idx,
                    f2_idx=f2_idx,
                    delay=delay,
                    sig1=frequency_signal(self.frequency_values[f1_idx]),
                    sig2=frequency_signal(self.frequency_values[f2_idx]),
                )
        else:
            n_frequencies = len(self.frequency_values)
            for pair_id in range(n_samples // 2):
                f2_idx = torch.randint(1, n_frequencies - 1, (1,)).item()
                max_distance = min(f2_idx, n_frequencies - 1 - f2_idx)
                distance = torch.randint(1, max_distance + 1, (1,)).item()
                delay = torch.randint(min_delay, max_delay + 1, (1,)).item()
                shared_sig2 = frequency_signal(self.frequency_values[f2_idx])
                lower_index = 2 * pair_id
                higher_index = lower_index + 1
                lower_f1_idx = f2_idx - distance
                higher_f1_idx = f2_idx + distance
                write_trial(
                    lower_index,
                    f1_idx=lower_f1_idx,
                    f2_idx=f2_idx,
                    delay=delay,
                    sig1=frequency_signal(self.frequency_values[lower_f1_idx]),
                    sig2=shared_sig2,
                    pair_id=pair_id,
                )
                write_trial(
                    higher_index,
                    f1_idx=higher_f1_idx,
                    f2_idx=f2_idx,
                    delay=delay,
                    sig1=frequency_signal(self.frequency_values[higher_f1_idx]),
                    sig2=shared_sig2,
                    pair_id=pair_id,
                )

            # Pair construction is exact, while this final permutation prevents
            # adjacent opposite-label rows from becoming a loader-order cue.
            permutation = torch.randperm(n_samples)
            for field_name in (
                "inputs",
                "targets",
                "delays",
                "frequency1_indices",
                "frequency2_indices",
                "frequency1_values",
                "frequency2_values",
                "pair_ids",
                "comparison_distance_indices",
            ):
                value = getattr(self, field_name)
                setattr(self, field_name, value[permutation])

        self.seq_lengths = (
            2 * self.stimulus_duration + self.delays + self.response_duration
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq_lengths = self.seq_lengths[idx]
        if torch.is_tensor(seq_lengths) and seq_lengths.dim() == 0:
            seq_lengths = int(seq_lengths.item())
        return self.inputs[idx], self.targets[idx], seq_lengths

    def _positive_frequency_code(
        self, frequency: float, noise_std: float
    ) -> torch.Tensor:
        tuning = torch.exp(
            -0.5
            * ((self.frequency_pref - float(frequency)) / max(self.tuning_sigma, 1e-6))
            ** 2
        )
        tuning = tuning / tuning.max().clamp(min=1e-8)
        signal = tuning.unsqueeze(0).repeat(self.stimulus_duration, 1)
        if noise_std > 0:
            signal = signal + noise_std * torch.randn_like(signal)
        return signal.clamp_min(0.0)


class GainModulatedContextualComparisonDataset(Dataset):
    """Context-cued delayed comparison under multiplicative nuisance gain.

    Each trial contains:
    1. a brief cue selecting one relevant pathway,
    2. a first short stimulus on every pathway,
    3. a silent delay with optional distractor bursts,
    4. a second short stimulus on every pathway,
    5. a response window.

    The target is binary: whether the relevant pathway's second latent value is
    larger than its first. All pathways are corrupted by a trial-global gain and
    pathway-specific gains, making the task a sequence-level nuisance-robust
    delayed comparison problem rather than a pure memory benchmark.

    Inputs are arranged as:
        [context one-hot (n_pathways) |
         pathway features (n_pathways * pathway_dim) |
         phase markers (sample1, sample2, response)]

    The contiguous pathway blocks are intentional so configs can test explicit
    clustered vs. dispersed pathway routing with the ``PathwayRouter`` encoder.
    """

    def __init__(
        self,
        n_samples: int = 20000,
        n_pathways: int = 4,
        pathway_dim: int = 4,
        cue_duration: int = 8,
        stimulus_duration: int = 16,
        min_delay: int = 20,
        max_delay: int = 80,
        response_duration: int = 8,
        amplitude_values: tuple[float, ...] = (0.45, 0.6, 0.75, 0.9, 1.05, 1.2),
        comparison_margin: float = 0.15,
        global_gain_min: float = 0.85,
        global_gain_max: float = 1.15,
        pathway_gain_min: float = 0.8,
        pathway_gain_max: float = 1.2,
        gain_drift_std: float = 0.03,
        noise_std: float = 0.08,
        cue_noise_std: float = 0.02,
        distractor_burst_prob: float = 0.35,
        distractor_burst_scale: float = 1.5,
        distractor_burst_duration: int = 4,
        positive_input_encoding: bool = False,
    ):
        super().__init__()
        if n_pathways < 2:
            raise ValueError("n_pathways must be >= 2")
        if pathway_dim < 1:
            raise ValueError("pathway_dim must be >= 1")
        if min_delay < 0 or max_delay < min_delay:
            raise ValueError(
                f"Expected 0 <= min_delay <= max_delay, got {min_delay}, {max_delay}"
            )
        if len(amplitude_values) < 2:
            raise ValueError("amplitude_values must contain at least two entries")
        if global_gain_min <= 0 or pathway_gain_min <= 0:
            raise ValueError("Gain ranges must be positive")

        self.n_samples = int(n_samples)
        self.n_pathways = int(n_pathways)
        self.pathway_dim = int(pathway_dim)
        self.cue_duration = int(cue_duration)
        self.stimulus_duration = int(stimulus_duration)
        self.min_delay = int(min_delay)
        self.max_delay = int(max_delay)
        self.response_duration = int(response_duration)
        self.comparison_margin = float(comparison_margin)
        self.global_gain_min = float(global_gain_min)
        self.global_gain_max = float(global_gain_max)
        self.pathway_gain_min = float(pathway_gain_min)
        self.pathway_gain_max = float(pathway_gain_max)
        self.gain_drift_std = float(gain_drift_std)
        self.noise_std = float(noise_std)
        self.cue_noise_std = float(cue_noise_std)
        self.distractor_burst_prob = float(distractor_burst_prob)
        self.distractor_burst_scale = float(distractor_burst_scale)
        self.distractor_burst_duration = max(1, int(distractor_burst_duration))
        self.positive_input_encoding = bool(positive_input_encoding)

        self.context_dim = self.n_pathways
        self.phase_marker_dim = 3
        self.pathway_feature_start = self.context_dim
        self.pathway_feature_end = self.context_dim + self.n_pathways * self.pathway_dim
        self.phase_marker_start = self.pathway_feature_end
        self.input_dim = self.phase_marker_start + self.phase_marker_dim
        self.max_total_len = (
            self.cue_duration
            + 2 * self.stimulus_duration
            + self.max_delay
            + self.response_duration
        )

        amplitude_tensor = torch.tensor(amplitude_values, dtype=torch.float32)
        if amplitude_tensor.numel() < 2:
            raise ValueError("Need at least two amplitude values for comparison")
        self.amplitude_values = amplitude_tensor

        self.inputs = torch.zeros(self.n_samples, self.max_total_len, self.input_dim)
        self.targets = torch.zeros(self.n_samples, dtype=torch.long)
        self.delays = torch.zeros(self.n_samples, dtype=torch.long)
        self.relevant_pathways = torch.zeros(self.n_samples, dtype=torch.long)
        self.seq_lengths = torch.zeros(self.n_samples, dtype=torch.long)
        self.sample1_values = torch.zeros(self.n_samples, self.n_pathways)
        self.sample2_values = torch.zeros(self.n_samples, self.n_pathways)
        self.global_gains = torch.zeros(self.n_samples, dtype=torch.float32)
        self.pathway_gain_sample1 = torch.zeros(self.n_samples, self.n_pathways)
        self.pathway_gain_sample2 = torch.zeros(self.n_samples, self.n_pathways)
        self.delay_burst_flags = torch.zeros(self.n_samples, dtype=torch.bool)

        self.pathway_templates = self._build_pathway_templates()
        self.pathway_slices = [
            slice(
                self.pathway_feature_start + pathway_idx * self.pathway_dim,
                self.pathway_feature_start + (pathway_idx + 1) * self.pathway_dim,
            )
            for pathway_idx in range(self.n_pathways)
        ]

        for i in range(self.n_samples):
            relevant = torch.randint(0, self.n_pathways, (1,)).item()
            delay = torch.randint(self.min_delay, self.max_delay + 1, (1,)).item()
            total_len = (
                self.cue_duration
                + 2 * self.stimulus_duration
                + delay
                + self.response_duration
            )
            cue_end = self.cue_duration
            s1_end = cue_end + self.stimulus_duration
            s2_start = s1_end + delay
            s2_end = s2_start + self.stimulus_duration
            response_start = s2_end

            global_gain = torch.empty(1).uniform_(
                self.global_gain_min, self.global_gain_max
            )[0]
            pathway_gain_base = torch.empty(self.n_pathways).uniform_(
                self.pathway_gain_min, self.pathway_gain_max
            )
            sample2_drift = torch.exp(
                0.25 * torch.randn(self.n_pathways, dtype=torch.float32)
            )
            pathway_gain_s1 = pathway_gain_base
            pathway_gain_s2 = torch.clamp(
                pathway_gain_base * sample2_drift,
                min=self.pathway_gain_min,
                max=self.pathway_gain_max,
            )

            target = int(torch.rand(1).item() > 0.5)
            relevant_v1, relevant_v2 = self._sample_relevant_pair(target)

            values1 = self._sample_pathway_values()
            values2 = self._sample_pathway_values()
            values1[relevant] = relevant_v1
            values2[relevant] = relevant_v2

            self.targets[i] = target
            self.delays[i] = delay
            self.relevant_pathways[i] = relevant
            self.seq_lengths[i] = total_len
            self.sample1_values[i] = values1
            self.sample2_values[i] = values2
            self.global_gains[i] = global_gain
            self.pathway_gain_sample1[i] = pathway_gain_s1
            self.pathway_gain_sample2[i] = pathway_gain_s2

            cue = torch.zeros(self.cue_duration, self.context_dim)
            cue[:, relevant] = 1.0
            cue = cue + self.cue_noise_std * torch.randn_like(cue)
            if self.positive_input_encoding:
                cue = cue.clamp_min(0.0)
            self.inputs[i, :cue_end, : self.context_dim] = cue

            self.inputs[i, cue_end:s1_end, self.phase_marker_start] = 1.0
            self.inputs[i, s2_start:s2_end, self.phase_marker_start + 1] = 1.0
            self.inputs[i, response_start:total_len, self.phase_marker_start + 2] = 1.0

            for pathway_idx in range(self.n_pathways):
                template = self.pathway_templates[pathway_idx]
                gain_trace1 = self._build_gain_trace(pathway_gain_s1[pathway_idx])
                gain_trace2 = self._build_gain_trace(pathway_gain_s2[pathway_idx])

                seg1 = (
                    global_gain
                    * gain_trace1.unsqueeze(-1)
                    * values1[pathway_idx]
                    * template
                )
                seg2 = (
                    global_gain
                    * gain_trace2.unsqueeze(-1)
                    * values2[pathway_idx]
                    * template
                )
                seg1 = seg1 + self.noise_std * torch.randn_like(seg1)
                seg2 = seg2 + self.noise_std * torch.randn_like(seg2)
                if self.positive_input_encoding:
                    seg1 = seg1.clamp_min(0.0)
                    seg2 = seg2.clamp_min(0.0)

                block = self.pathway_slices[pathway_idx]
                self.inputs[i, cue_end:s1_end, block] = seg1
                self.inputs[i, s2_start:s2_end, block] = seg2

            if (
                delay >= self.distractor_burst_duration
                and torch.rand(1).item() < self.distractor_burst_prob
            ):
                burst_rel = torch.randint(0, self.n_pathways - 1, (1,)).item()
                burst_pathway = burst_rel if burst_rel < relevant else burst_rel + 1
                burst_start = torch.randint(
                    0, delay - self.distractor_burst_duration + 1, (1,)
                ).item()
                burst_end = burst_start + self.distractor_burst_duration
                delay_start = s1_end + burst_start
                delay_end = s1_end + burst_end
                burst_template = self.pathway_templates[burst_pathway][
                    : self.distractor_burst_duration
                ]
                burst_gain = self._build_gain_trace(
                    pathway_gain_s2[burst_pathway],
                    length=self.distractor_burst_duration,
                )
                burst = (
                    global_gain
                    * burst_gain.unsqueeze(-1)
                    * self.distractor_burst_scale
                    * burst_template
                )
                burst = burst + self.noise_std * torch.randn_like(burst)
                if self.positive_input_encoding:
                    burst = burst.clamp_min(0.0)
                self.inputs[
                    i, delay_start:delay_end, self.pathway_slices[burst_pathway]
                ] += burst
                self.delay_burst_flags[i] = True

    def _build_pathway_templates(self) -> torch.Tensor:
        time = torch.linspace(0.0, 1.0, self.stimulus_duration)
        templates = torch.zeros(
            self.n_pathways, self.stimulus_duration, self.pathway_dim
        )
        for pathway_idx in range(self.n_pathways):
            base_phase = 2.0 * math.pi * pathway_idx / self.n_pathways
            for feature_idx in range(self.pathway_dim):
                freq = 1.0 + 0.35 * feature_idx
                waveform = torch.sin(2.0 * math.pi * freq * time + base_phase)
                waveform = waveform + 0.35 * torch.cos(
                    2.0 * math.pi * (0.5 + 0.2 * pathway_idx) * time + 0.7 * feature_idx
                )
                if self.positive_input_encoding:
                    waveform = waveform - waveform.min()
                    waveform = waveform / waveform.max().clamp(min=1e-6)
                else:
                    waveform = waveform / waveform.std().clamp(min=1e-6)
                templates[pathway_idx, :, feature_idx] = waveform
        return templates

    def _sample_pathway_values(self) -> torch.Tensor:
        indices = torch.randint(
            0, len(self.amplitude_values), (self.n_pathways,), dtype=torch.long
        )
        return self.amplitude_values.index_select(0, indices).clone()

    def _sample_relevant_pair(self, target: int) -> tuple[float, float]:
        valid_pairs: list[tuple[float, float]] = []
        values = self.amplitude_values.tolist()
        for v1 in values:
            for v2 in values:
                if abs(v2 - v1) < self.comparison_margin:
                    continue
                if int(v2 > v1) == int(target):
                    valid_pairs.append((float(v1), float(v2)))
        if not valid_pairs:
            raise ValueError(
                "No valid amplitude pairs satisfy the requested comparison_margin"
            )
        pair_idx = torch.randint(0, len(valid_pairs), (1,)).item()
        return valid_pairs[pair_idx]

    def _build_gain_trace(self, base_gain: torch.Tensor, length: int | None = None):
        length = self.stimulus_duration if length is None else int(length)
        if self.gain_drift_std <= 0.0 or length <= 1:
            return torch.full((length,), float(base_gain), dtype=torch.float32)

        log_drift = self.gain_drift_std * torch.randn(length, dtype=torch.float32)
        trace = float(base_gain) * torch.exp(torch.cumsum(log_drift, dim=0) / length)
        return trace.clamp(
            min=0.5 * self.pathway_gain_min,
            max=1.5 * self.pathway_gain_max,
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        seq_lengths = self.seq_lengths[idx]
        if torch.is_tensor(seq_lengths) and seq_lengths.dim() == 0:
            seq_lengths = int(seq_lengths.item())
        return self.inputs[idx], self.targets[idx], seq_lengths


class StringerV1Dataset(Dataset):
    """Stringer et al. (2019) V1 population responses to drifting gratings.

    Loads real calcium imaging data from mouse V1 (11,311 neurons, 4,442 trials).
    Selects top neurons by response variance and bins orientations into discrete
    classes for feedforward classification.

    Returns (response_vector, label) where response_vector: [n_neurons] and
    label: int in [0, n_classes).

    Running speed metadata is stored for post-hoc cross-context analysis
    (e.g., comparing representations during locomotion vs. stillness).

    Data source: Stringer et al., "Spontaneous behaviors drive multidimensional,
    brainwide activity", Science 364, 2019. CC0 license on Figshare.

    Args:
        data_dir: Directory containing a Stringer drifting-gratings response
                  file and all_running.npy.
        n_neurons: Number of neurons to select by response variance.
        neuron_selection_scope: Trials used to rank neurons by response variance.
            ``"all_trials"`` preserves the legacy transductive behavior.
            ``"train_split"`` ranks neurons using only the training side of the
            deterministic train/test split, for prospective held-out evaluation.
            For a state-specific ``split="all"`` training set, it ranks only
            the selected training state; builders can then pass those indices
            unchanged to a disjoint held-out state.
        neuron_selection_indices: Optional explicit ordered neuron indices.
            This is an internal builder hook for reusing a training-state
            selection in a disjoint state-specific test set.
        n_classes: Number of orientation bins (orientations span [0, 2*pi)).
        split: "train", "test", or "all".
        train_fraction: Fraction of trials for training (rest = test).
        normalize: Z-score normalize each neuron's responses.
        seed: Random seed for reproducible train/test split.
        session_idx: Index into all_running.npy for this session (default 20 = GT3).
        grating_file: Response filename inside data_dir. Defaults to the GT3
            session used by the original single-session examples.
        running_state: Keep all trials, or only the low/high running subset.
        running_quantile: Quantile used to define low/high states. `0.5` gives
            a median split, `1/3` gives lower/upper tertiles.
            State-transfer builders may additionally reserve a deterministic
            within-state holdout before their train/validation split.
        gain_group_count: Number of pathway-specific multiplicative gain factors.
        gain_grouping: How neurons are grouped for pathway-specific gains.
            Supported: "preferred_class", "random", "contiguous".
        global_gain_min/global_gain_max: Range for a per-trial global positive gain.
        group_gain_min/group_gain_max: Range for per-group positive gains.
        gain_seed: Seed for the deterministic gain augmentation.
        poisson_sampling: If True, apply Poisson shot noise after multiplicative gain.
        poisson_rate_scale: Scale factor before Poisson sampling.
        running_residualize: If True, regress running speed out of responses
            within each orientation class after neuron selection. This removes
            the running-linked gain component while preserving class means.
        running_residualize_clip: Clip residualized responses at zero so shunting
            models still receive nonnegative rate-like inputs.
    """

    _GRATING_FILE = "gratings_drifting_GT3_2019_04_05_1.npy"
    _RUNNING_FILE = "all_running.npy"

    def __init__(
        self,
        data_dir: str,
        n_neurons: int = 500,
        neuron_selection_scope: str = "all_trials",
        neuron_selection_indices: np.ndarray | list[int] | None = None,
        n_classes: int = 8,
        split: str = "train",
        train_fraction: float = 0.9,
        normalize: bool = True,
        seed: int = 42,
        session_idx: int = 20,
        running_state: str = "all",
        running_quantile: float = 0.5,
        gain_group_count: int = 1,
        gain_grouping: str = "preferred_class",
        global_gain_min: float = 1.0,
        global_gain_max: float = 1.0,
        group_gain_min: float = 1.0,
        group_gain_max: float = 1.0,
        gain_seed: int | None = None,
        poisson_sampling: bool = False,
        poisson_rate_scale: float = 1.0,
        running_residualize: bool = False,
        running_residualize_clip: bool = True,
        normalization_stats: tuple[np.ndarray, np.ndarray] | None = None,
        grating_file: str | None = None,
    ):
        super().__init__()
        if split not in ("train", "test", "all"):
            raise ValueError(f"split must be 'train', 'test', or 'all', got '{split}'")
        if neuron_selection_scope not in ("all_trials", "train_split"):
            raise ValueError(
                "neuron_selection_scope must be 'all_trials' or 'train_split', "
                f"got {neuron_selection_scope!r}"
            )
        if (
            neuron_selection_scope == "train_split"
            and split == "all"
            and running_state == "all"
            and neuron_selection_indices is None
        ):
            raise ValueError(
                "neuron_selection_scope='train_split' requires split='train' or "
                "split='test', unless split='all' is a declared low/high "
                "training state or explicit training-state indices are supplied"
            )
        if running_state not in ("all", "low", "high"):
            raise ValueError(
                f"running_state must be 'all', 'low', or 'high', got '{running_state}'"
            )
        if not (0.0 < running_quantile <= 0.5):
            raise ValueError(
                f"running_quantile must be in (0, 0.5], got {running_quantile}"
            )
        if gain_group_count < 1:
            raise ValueError(f"gain_group_count must be >= 1, got {gain_group_count}")
        if poisson_rate_scale <= 0:
            raise ValueError(
                f"poisson_rate_scale must be > 0, got {poisson_rate_scale}"
            )

        # --- Load raw data ---
        grating_file = self._GRATING_FILE if grating_file is None else str(grating_file)
        if os.path.basename(grating_file) != grating_file:
            raise ValueError(
                "Stringer grating_file must be a filename inside data_dir, "
                f"got {grating_file!r}"
            )
        grating_path = os.path.join(data_dir, grating_file)
        running_path = os.path.join(data_dir, self._RUNNING_FILE)

        if not os.path.exists(grating_path):
            raise FileNotFoundError(
                f"Stringer V1 data not found at {grating_path}. "
                f"Download from https://figshare.com/articles/dataset/9598406"
            )

        gdata = np.load(grating_path, allow_pickle=True).item()
        sresp = gdata["sresp"]  # (n_neurons_total, n_trials)
        istim = gdata["istim"]  # (n_trials,) orientations in radians [0, 2*pi)

        n_neurons_total, n_trials = sresp.shape
        logger.info(f"Loaded Stringer V1: {n_neurons_total} neurons, {n_trials} trials")

        # --- Load running speed ---
        if os.path.exists(running_path):
            all_running = np.load(running_path, allow_pickle=True)
            running = all_running[session_idx]  # (n_trials,)
            running_speed_full = running.astype(np.float32)
        else:
            logger.warning(
                f"Running speed file not found at {running_path}, "
                f"setting all running speeds to 0"
            )
            running_speed_full = np.zeros(n_trials, dtype=np.float32)

        # --- Bin orientations into classes ---
        # istim is in [0, 2*pi), bin into n_classes equal bins
        bin_edges = np.linspace(0, 2 * np.pi, n_classes + 1)
        labels = np.digitize(istim, bin_edges) - 1
        labels = np.clip(labels, 0, n_classes - 1)  # handle edge case at 2*pi
        full_labels = labels.copy()

        if normalize and (
            gain_group_count > 1
            or global_gain_min != 1.0
            or global_gain_max != 1.0
            or group_gain_min != 1.0
            or group_gain_max != 1.0
            or poisson_sampling
            or running_residualize
        ):
            raise ValueError(
                "Stringer gain augmentation/residualization requires normalize=False "
                "so the inputs remain nonnegative and rate-like."
            )

        # --- Running-state filter ---
        low_thresh = np.quantile(running_speed_full, running_quantile)
        high_thresh = np.quantile(running_speed_full, 1.0 - running_quantile)
        if running_state == "low":
            keep_mask = running_speed_full <= low_thresh
        elif running_state == "high":
            keep_mask = running_speed_full >= high_thresh
        else:
            keep_mask = np.ones(n_trials, dtype=bool)

        available_idx = np.flatnonzero(keep_mask)

        # --- Train/test split ---
        if split == "all":
            idx = np.arange(len(labels))
        else:
            rng = np.random.RandomState(seed)
            perm = rng.permutation(len(available_idx))
            n_train = int(len(available_idx) * train_fraction)
            if split == "train":
                idx = perm[:n_train]
            else:
                idx = perm[n_train:]

        if split == "all":
            selected_global_idx = available_idx
        else:
            selected_global_idx = available_idx[idx]

        # --- Select top neurons by variance ---
        # For prospective evaluation, train and test constructors use the same
        # seed and therefore derive the same training indices and neuron order.
        # This keeps candidate pools nested while preventing held-out response
        # variance from affecting feature selection.
        n_select = min(n_neurons, n_neurons_total)
        if neuron_selection_indices is not None:
            explicit_indices = np.asarray(neuron_selection_indices, dtype=np.int64)
            if explicit_indices.ndim != 1 or len(explicit_indices) < n_select:
                raise ValueError(
                    "neuron_selection_indices must be a one-dimensional ordered "
                    f"array with at least {n_select} entries"
                )
            if (
                np.any(explicit_indices < 0)
                or np.any(explicit_indices >= n_neurons_total)
                or len(np.unique(explicit_indices)) != len(explicit_indices)
            ):
                raise ValueError(
                    "neuron_selection_indices must contain unique in-range indices"
                )
            top_idx = explicit_indices[:n_select]
        else:
            if neuron_selection_scope == "train_split":
                selection_global_idx = (
                    available_idx[perm[:n_train]] if split != "all" else available_idx
                )
            else:
                selection_global_idx = np.arange(n_trials)
            neuron_var = sresp[:, selection_global_idx].var(axis=1)
            ranked_idx = np.argsort(neuron_var, kind="stable")[::-1]
            top_idx = ranked_idx[:n_select]
        self._neuron_indices = top_idx
        self.neuron_selection_scope = neuron_selection_scope
        raw_responses = sresp[top_idx, :].T.astype(np.float32)  # (n_trials, n_select)

        # --- Normalize ---
        if normalize:
            if normalization_stats is not None:
                mu, std = normalization_stats
            else:
                if split == "all":
                    stats_global_idx = available_idx
                else:
                    stats_global_idx = available_idx[perm[:n_train]]
                mu = raw_responses[stats_global_idx].mean(axis=0, keepdims=True)
                std = raw_responses[stats_global_idx].std(axis=0, keepdims=True) + 1e-8
            mu = np.asarray(mu, dtype=np.float32).reshape(1, -1)
            std = np.asarray(std, dtype=np.float32).reshape(1, -1)
            responses = (raw_responses - mu) / std
            self.normalization_stats = (mu.copy(), std.copy())
        else:
            responses = raw_responses.copy()
            self.normalization_stats = None

        labels = labels[available_idx]
        running_speed = running_speed_full[available_idx]

        if split == "all":
            idx = np.arange(len(labels))

        selected_global_idx = available_idx[idx]
        selected_responses = responses[selected_global_idx]
        selected_labels = labels[idx]
        selected_running = running_speed[idx]

        if running_residualize:
            selected_responses = self._residualize_running_gain(
                selected_responses,
                selected_running,
                selected_labels,
                n_classes=n_classes,
                clip_nonnegative=running_residualize_clip,
            )

        gain_groups = self._build_gain_groups(
            raw_responses=raw_responses,
            full_labels=full_labels,
            n_classes=n_classes,
            gain_group_count=gain_group_count,
            gain_grouping=gain_grouping,
            seed=(gain_seed if gain_seed is not None else seed),
        )
        selected_responses = self._apply_gain_augmentation(
            responses=selected_responses,
            sample_indices=selected_global_idx,
            gain_groups=gain_groups,
            gain_group_count=gain_group_count,
            global_gain_min=global_gain_min,
            global_gain_max=global_gain_max,
            group_gain_min=group_gain_min,
            group_gain_max=group_gain_max,
            gain_seed=(gain_seed if gain_seed is not None else seed),
            poisson_sampling=poisson_sampling,
            poisson_rate_scale=poisson_rate_scale,
        )

        self.responses = torch.tensor(selected_responses, dtype=torch.float32)
        self.labels = torch.tensor(selected_labels, dtype=torch.long)
        self.grating_file = grating_file
        self.running_residualized = bool(running_residualize)
        self._running_speed = selected_running.astype(np.float32)
        self._running_mask = torch.tensor(
            self._running_speed > np.median(running_speed_full),
            dtype=torch.bool,
        )
        self._split_indices = selected_global_idx

        logger.info(
            f"StringerV1Dataset({split}, state={running_state}, file={grating_file}): "
            f"{len(self)} trials, {n_select} neurons, {n_classes} classes"
        )

    @staticmethod
    def _build_gain_groups(
        raw_responses: np.ndarray,
        full_labels: np.ndarray,
        n_classes: int,
        gain_group_count: int,
        gain_grouping: str,
        seed: int,
    ) -> np.ndarray:
        n_neurons = raw_responses.shape[1]
        if gain_group_count <= 1:
            return np.zeros(n_neurons, dtype=np.int64)

        if gain_grouping == "random":
            rng = np.random.RandomState(seed)
            perm = rng.permutation(n_neurons)
            group_ids = np.zeros(n_neurons, dtype=np.int64)
            group_ids[perm] = np.arange(n_neurons) % gain_group_count
            return group_ids

        if gain_grouping == "contiguous":
            return np.floor(
                np.arange(n_neurons, dtype=np.float32) * gain_group_count / n_neurons
            ).astype(np.int64)

        if gain_grouping != "preferred_class":
            raise ValueError(
                f"Unsupported gain_grouping={gain_grouping!r}. "
                "Use 'preferred_class', 'random', or 'contiguous'."
            )

        class_means = np.zeros((n_classes, n_neurons), dtype=np.float32)
        for c in range(n_classes):
            mask = full_labels == c
            if not np.any(mask):
                continue
            class_means[c] = raw_responses[mask].mean(axis=0)

        preferred_class = class_means.argmax(axis=0)
        return np.floor(preferred_class * gain_group_count / n_classes).astype(np.int64)

    @staticmethod
    def _apply_gain_augmentation(
        responses: np.ndarray,
        sample_indices: np.ndarray,
        gain_groups: np.ndarray,
        gain_group_count: int,
        global_gain_min: float,
        global_gain_max: float,
        group_gain_min: float,
        group_gain_max: float,
        gain_seed: int,
        poisson_sampling: bool,
        poisson_rate_scale: float,
    ) -> np.ndarray:
        if (
            gain_group_count <= 1
            and global_gain_min == 1.0
            and global_gain_max == 1.0
            and group_gain_min == 1.0
            and group_gain_max == 1.0
            and not poisson_sampling
        ):
            return responses

        if min(global_gain_min, global_gain_max, group_gain_min, group_gain_max) <= 0:
            raise ValueError("All gain ranges must be strictly positive.")

        rng = np.random.RandomState(gain_seed)
        max_trial_idx = int(sample_indices.max()) + 1
        global_gains = rng.uniform(global_gain_min, global_gain_max, size=max_trial_idx)
        group_gains = rng.uniform(
            group_gain_min,
            group_gain_max,
            size=(max_trial_idx, gain_group_count),
        )

        neuron_gains = group_gains[sample_indices][:, gain_groups]
        total_gain = global_gains[sample_indices, None] * neuron_gains
        augmented = np.clip(
            responses * total_gain.astype(np.float32), a_min=0.0, a_max=None
        )

        if poisson_sampling:
            rate = np.clip(augmented * poisson_rate_scale, a_min=0.0, a_max=None)
            augmented = rng.poisson(rate).astype(np.float32) / float(poisson_rate_scale)

        return augmented.astype(np.float32)

    @staticmethod
    def _residualize_running_gain(
        responses: np.ndarray,
        running: np.ndarray,
        labels: np.ndarray,
        *,
        n_classes: int,
        clip_nonnegative: bool,
    ) -> np.ndarray:
        """Remove per-class linear running dependence while preserving means."""
        out = responses.astype(np.float64, copy=True)
        running = np.asarray(running, dtype=np.float64).reshape(-1)
        labels = np.asarray(labels).reshape(-1)
        if out.shape[0] != running.shape[0] or out.shape[0] != labels.shape[0]:
            raise ValueError("responses, running, and labels must have matching trials")

        for class_idx in range(n_classes):
            mask = labels == class_idx
            if int(mask.sum()) < 5:
                continue
            r = running[mask]
            x = out[mask]
            r_centered = r - r.mean()
            denom = float(np.dot(r_centered, r_centered)) + 1e-8
            x_mean = x.mean(axis=0, keepdims=True)
            x_centered = x - x_mean
            beta = (r_centered[:, None] * x_centered).sum(axis=0, keepdims=True) / denom
            out[mask] = x_mean + x_centered - r_centered[:, None] * beta

        if clip_nonnegative:
            out = np.clip(out, a_min=0.0, a_max=None)
        return out.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.responses[idx], self.labels[idx]

    @property
    def running_mask(self):
        """Boolean mask: True for high-running trials (above median speed)."""
        return self._running_mask

    @property
    def neuron_indices(self):
        """Indices of selected neurons in original sresp array."""
        return self._neuron_indices


__all__ = [
    "ROMO_PAIR_SAMPLING_BALANCED_SECOND_STIMULUS_V1",
    "ROMO_PAIR_SAMPLING_INDEPENDENT_UNEQUAL_V1",
    "ROMO_PAIR_SAMPLING_POLICIES",
    "GainModulatedContextualComparisonDataset",
    "OculomotorDelayedResponseDataset",
    "RomoDelayComparisonDataset",
    "StringerV1Dataset",
    "_resolve_stringer_session_file",
]
