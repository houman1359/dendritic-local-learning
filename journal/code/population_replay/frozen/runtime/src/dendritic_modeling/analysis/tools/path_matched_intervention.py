"""Path-matched, fixed-count interventions across dendritic depth.

This analyzer measures the causal effect of local excitatory and inhibitory
currents, or post-reactivation compartment outputs, without confounding
dendritic depth with the number of compartments replaced.  A label-independent
support draw selects one complete soma-to-distal path for every soma.  Prefixes
of those paths give the same number of compartment coordinates at every
dendritic depth.

Current interventions act on the output of the actual E/I current producer;
post-gate interventions act on the branch-layer output after its configured
reactivation.  All unselected coordinates are copied unchanged.  Dataset-wide
evaluation-signal shuffles use one donor permutation for every depth and, for
joint E/I interventions, for both streams.  Fixed-reference interventions are
calibrated on a separate reference split (normally training) and then held
fixed on the evaluation split.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from dendritic_modeling.analysis.utils.dendritic_depth import (
    SOMA_RELATIVE_DEPTH_REFERENCE,
    soma_relative_dendritic_depth,
)
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    iter_analysis_batches,
)
from dendritic_modeling.config.analysis import (
    EvaluationRuntimeConfig,
    PathMatchedInterventionAnalysisParams,
)
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.utils.general import save_dict
from dendritic_modeling.utils.hooks import iter_named_modules_of_type

_TARGET_SIGNALS: dict[str, tuple[str, ...]] = {
    "excitation_current": ("excitation_current",),
    "inhibition_current": ("inhibition_current",),
    "joint_EI_currents": ("excitation_current", "inhibition_current"),
    "post_gate_output": ("post_gate_output",),
}


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _mixed_radix_index(path: Sequence[int], radices: Sequence[int]) -> int:
    if len(path) != len(radices):
        raise ValueError("a path needs one digit per dendritic depth")
    value = 0
    for digit, radix in zip(path, radices, strict=True):
        digit, radix = int(digit), int(radix)
        if radix <= 0 or digit < 0 or digit >= radix:
            raise ValueError(f"invalid path digit {digit} for radix {radix}")
        value = value * radix + digit
    return value


def _mixed_radix_digits(value: int, radices: Sequence[int]) -> list[int]:
    value = int(value)
    if value < 0:
        raise ValueError("mixed-radix values must be non-negative")
    digits = [0] * len(radices)
    for position in range(len(radices) - 1, -1, -1):
        radix = int(radices[position])
        if radix <= 0:
            raise ValueError("branch factors must be positive")
        digits[position] = value % radix
        value //= radix
    if value:
        raise ValueError("mixed-radix value exceeds the requested width")
    return digits


def build_nested_path_supports(
    *,
    n_somas: int,
    branch_factors: Sequence[int],
    draws: int,
    seed: int,
    namespace: str = "path-matched-depth-intervention-v1",
) -> dict[str, Any]:
    """Build deterministic, label-independent supports matched across depth.

    Every non-somatic support contains exactly one coordinate owned by each
    soma.  A draw first selects complete paths and then uses their prefixes, so
    coordinate identities are anatomically nested from proximal to distal.
    When possible, proximal branches are visited before a branch is reused.
    """

    n_somas = int(n_somas)
    factors = [int(value) for value in branch_factors]
    draws, seed = int(draws), int(seed)
    namespace = str(namespace)
    if n_somas < 1 or not factors or any(value < 1 for value in factors):
        raise ValueError("path supports require somas and positive branch factors")
    if draws < 1:
        raise ValueError("support draw count must be positive")
    path_count = math.prod(factors)
    if draws > path_count:
        raise ValueError(f"requested {draws} unique paths but only {path_count} exist")

    proximal_factor = factors[0]
    descendant_radices = factors[1:]
    descendant_count = math.prod(descendant_radices) if descendant_radices else 1
    complete_paths: dict[tuple[int, int], list[int]] = {}
    for soma in range(n_somas):
        identity = f"{namespace}|seed={seed}|soma={soma}"
        identity_seed = int.from_bytes(
            hashlib.sha256(identity.encode("utf-8")).digest()[:8], "little"
        )
        rng = np.random.default_rng(identity_seed)
        offset = int(rng.integers(proximal_factor))
        proximal_schedule = (
            np.arange(draws, dtype=np.int64) + offset
        ) % proximal_factor
        rng.shuffle(proximal_schedule)
        descendant_permutations = {
            proximal: rng.permutation(descendant_count).tolist()
            for proximal in range(proximal_factor)
        }
        descendant_cursors = dict.fromkeys(range(proximal_factor), 0)
        for draw, proximal_value in enumerate(proximal_schedule):
            proximal = int(proximal_value)
            cursor = int(descendant_cursors[proximal])
            descendant_index = descendant_permutations[proximal][cursor]
            descendant_cursors[proximal] = cursor + 1
            complete_paths[(draw, soma)] = [
                proximal,
                *_mixed_radix_digits(descendant_index, descendant_radices),
            ]

    depths: dict[str, list[dict[str, Any]]] = {}
    for depth in range(1, len(factors) + 1):
        per_soma_width = math.prod(factors[:depth])
        entries: list[dict[str, Any]] = []
        for draw in range(draws):
            coordinates = []
            for soma in range(n_somas):
                path = complete_paths[(draw, soma)][:depth]
                flat_index = soma * per_soma_width + _mixed_radix_index(
                    path, factors[:depth]
                )
                coordinates.append(
                    {
                        "flat_index": int(flat_index),
                        "soma_owner": int(soma),
                        "path_from_soma": [int(value) for value in path],
                    }
                )
            coordinates.sort(key=lambda item: int(item["flat_index"]))
            indices = np.asarray(
                [item["flat_index"] for item in coordinates], dtype=np.int64
            )
            if len(indices) != n_somas or len(set(indices.tolist())) != n_somas:
                raise RuntimeError(f"invalid path support at dendritic depth {depth}")
            entries.append(
                {
                    "support_draw": int(draw),
                    "selected_indices": indices.tolist(),
                    "selected_indices_sha256": _sha256_array(indices),
                    "coordinates": coordinates,
                    "coordinates_sha256": _sha256_json(coordinates),
                }
            )
        depths[str(depth)] = entries

    support = {
        "schema_version": 1,
        "selection_policy": (
            "one prespecified complete soma-to-distal path per soma and draw; "
            "depth supports are nested path prefixes; selection excludes data, "
            "labels, activations, mechanism, checkpoint, and experiment seed"
        ),
        "technical_draws_are_model_replicates": False,
        "selection_seed": seed,
        "selection_namespace": namespace,
        "draw_count": draws,
        "n_somas": n_somas,
        "branch_factors": factors,
        "coordinates_per_depth_per_draw": n_somas,
        "depths": depths,
    }
    selected_index_supports = {
        depth: [entry["selected_indices"] for entry in entries]
        for depth, entries in depths.items()
    }
    support["selected_index_supports_sha256"] = _sha256_json(selected_index_supports)
    support["support_plan_sha256"] = _sha256_json(support)
    return support


def build_nested_path_dose_supports(
    *,
    n_somas: int,
    branch_factors: Sequence[int],
    draws: int,
    paths_per_soma: Sequence[int],
    seed: int,
    namespace: str,
) -> dict[str, Any]:
    """Build nested, fixed-dose path supports for intervention curves.

    Within every technical draw and soma, the largest dose selects distinct
    proximal branches and one complete descendant path under each.  Smaller
    doses are prefixes of that same ordered set.  Thus both dendritic depth and
    intervention dose are anatomically nested without using data or labels.
    """

    n_somas = int(n_somas)
    factors = [int(value) for value in branch_factors]
    draws = int(draws)
    doses = sorted({int(value) for value in paths_per_soma})
    if n_somas < 1 or draws < 1 or not factors:
        raise ValueError("n_somas, draws, and branch_factors must be positive")
    if (
        any(value < 1 for value in factors)
        or not doses
        or any(value < 1 for value in doses)
    ):
        raise ValueError("branch factors and path doses must be positive")
    if doses[-1] > factors[0]:
        raise ValueError(
            "the largest paths_per_soma dose cannot exceed the proximal "
            f"branch factor ({factors[0]})"
        )

    complete_paths: dict[tuple[int, int], list[list[int]]] = {}
    for draw in range(draws):
        for soma in range(n_somas):
            identity = f"{namespace}|seed={seed}|draw={draw}|soma={soma}|dose"
            identity_seed = int.from_bytes(
                hashlib.sha256(identity.encode("utf-8")).digest()[:8], "little"
            )
            rng = np.random.default_rng(identity_seed)
            proximal_order = rng.permutation(factors[0])[: doses[-1]]
            paths = []
            for proximal in proximal_order:
                descendants = [int(rng.integers(radix)) for radix in factors[1:]]
                paths.append([int(proximal), *descendants])
            complete_paths[(draw, soma)] = paths

    depths: dict[str, list[dict[str, Any]]] = {}
    for depth in range(1, len(factors) + 1):
        per_soma_width = math.prod(factors[:depth])
        entries: list[dict[str, Any]] = []
        for draw in range(draws):
            for dose in doses:
                coordinates = []
                for soma in range(n_somas):
                    for complete_path in complete_paths[(draw, soma)][:dose]:
                        path = complete_path[:depth]
                        flat_index = soma * per_soma_width + _mixed_radix_index(
                            path, factors[:depth]
                        )
                        coordinates.append(
                            {
                                "flat_index": int(flat_index),
                                "soma_owner": int(soma),
                                "path_from_soma": [int(value) for value in path],
                            }
                        )
                coordinates.sort(key=lambda item: int(item["flat_index"]))
                indices = np.asarray(
                    [item["flat_index"] for item in coordinates], dtype=np.int64
                )
                expected_count = n_somas * dose
                if len(indices) != expected_count or len(set(indices.tolist())) != len(
                    indices
                ):
                    raise RuntimeError(
                        "invalid dose support at dendritic depth "
                        f"{depth}, draw {draw}, dose {dose}"
                    )
                entries.append(
                    {
                        "support_draw": int(draw),
                        "paths_per_soma": int(dose),
                        "selected_indices": indices.tolist(),
                        "selected_indices_sha256": _sha256_array(indices),
                        "coordinates": coordinates,
                        "coordinates_sha256": _sha256_json(coordinates),
                    }
                )
        depths[str(depth)] = entries

    support = {
        "schema_version": 2,
        "selection_policy": (
            "prespecified distinct proximal paths per soma and draw; smaller "
            "doses and shallower depth supports are nested prefixes; selection "
            "excludes data, labels, activations, mechanism, checkpoint, and seed"
        ),
        "technical_draws_are_model_replicates": False,
        "selection_seed": int(seed),
        "selection_namespace": str(namespace),
        "draw_count": draws,
        "n_somas": n_somas,
        "branch_factors": factors,
        "paths_per_soma": doses,
        "depths": depths,
    }
    selected_index_supports = {
        depth: [
            {
                "support_draw": entry["support_draw"],
                "paths_per_soma": entry["paths_per_soma"],
                "selected_indices": entry["selected_indices"],
            }
            for entry in entries
        ]
        for depth, entries in depths.items()
    }
    support["selected_index_supports_sha256"] = _sha256_json(selected_index_supports)
    support["support_plan_sha256"] = _sha256_json(support)
    return support


@dataclass(frozen=True)
class _SignalSite:
    depth: int
    signal: str
    module_name: str
    producer: torch.nn.Module
    coordinate_count: int


def _tree_inventory(
    model: torch.nn.Module,
    *,
    module_name_prefix: str | None,
    expected_n_somas: int | None,
    expected_branch_factors: Sequence[int],
) -> tuple[dict[int, tuple[str, DendriticBranchLayer]], int, list[int]]:
    grouped: dict[int, list[tuple[str, DendriticBranchLayer]]] = defaultdict(list)
    for name, module in iter_named_modules_of_type(model, DendriticBranchLayer):
        if module_name_prefix and not name.startswith(module_name_prefix):
            continue
        grouped[soma_relative_dendritic_depth(module)].append((name, module))
    if not grouped:
        selector = module_name_prefix if module_name_prefix else "<all modules>"
        raise RuntimeError(f"no dendritic branch modules matched {selector!r}")
    depths = sorted(grouped)
    if depths != list(range(max(depths) + 1)):
        raise RuntimeError(f"dendritic depths must be contiguous from soma: {depths}")
    if any(len(grouped[depth]) != 1 for depth in depths):
        counts = {depth: len(grouped[depth]) for depth in depths}
        raise ValueError(
            "path-matched intervention requires one tree after module selection; "
            f"module counts by depth are {counts}. Set module_name_prefix."
        )
    modules = {depth: grouped[depth][0] for depth in depths}
    widths = {
        depth: int(module.branch_config.output_dim)
        for depth, (_name, module) in modules.items()
    }
    n_somas = widths[0]
    branch_factors = []
    for depth in depths[1:]:
        previous = widths[depth - 1]
        current = widths[depth]
        factor, remainder = divmod(current, previous)
        if remainder or factor < 1:
            raise RuntimeError(
                f"depth {depth} width {current} is not an integer expansion of "
                f"depth {depth - 1} width {previous}"
            )
        branch_factors.append(int(factor))
    if expected_n_somas is not None and n_somas != int(expected_n_somas):
        raise ValueError(f"expected {expected_n_somas} somas, found {n_somas}")
    expected = [int(value) for value in expected_branch_factors]
    if expected and branch_factors != expected:
        raise ValueError(f"expected branch factors {expected}, found {branch_factors}")
    return modules, n_somas, branch_factors


def inventory_nested_tree(
    model: torch.nn.Module,
    *,
    module_name_prefix: str | None = None,
    expected_n_somas: int | None = None,
    expected_branch_factors: Sequence[int] = (),
) -> tuple[dict[int, tuple[str, DendriticBranchLayer]], int, list[int]]:
    """Inventory one selected tree for path-matched analyses.

    This public facade keeps architecture validation identical for the
    intervention and information analyzers.
    """

    return _tree_inventory(
        model,
        module_name_prefix=module_name_prefix,
        expected_n_somas=expected_n_somas,
        expected_branch_factors=expected_branch_factors,
    )


def _signal_sites(
    modules: dict[int, tuple[str, DendriticBranchLayer]],
) -> dict[tuple[int, str], _SignalSite]:
    sites: dict[tuple[int, str], _SignalSite] = {}
    for depth, (module_name, module) in modules.items():
        producers = {
            "excitation_current": module.branch_excitation,
            "inhibition_current": module.branch_inhibition,
        }
        for signal, producer in producers.items():
            if producer is None:
                continue
            sites[(depth, signal)] = _SignalSite(
                depth=int(depth),
                signal=signal,
                module_name=module_name,
                producer=producer,
                coordinate_count=int(module.branch_config.output_dim),
            )
        sites[(depth, "post_gate_output")] = _SignalSite(
            depth=int(depth),
            signal="post_gate_output",
            module_name=module_name,
            producer=module,
            coordinate_count=int(module.branch_config.output_dim),
        )
    return sites


def _union_supports(
    support_plan: dict[str, Any],
) -> tuple[dict[int, np.ndarray], dict[tuple[int, int, int], np.ndarray]]:
    union_indices: dict[int, np.ndarray] = {}
    union_positions: dict[tuple[int, int, int], np.ndarray] = {}
    for depth_text, entries in support_plan["depths"].items():
        depth = int(depth_text)
        union = np.unique(
            np.concatenate(
                [
                    np.asarray(entry["selected_indices"], dtype=np.int64)
                    for entry in entries
                ]
            )
        )
        lookup = {int(value): position for position, value in enumerate(union)}
        union_indices[depth] = union
        for entry in entries:
            draw = int(entry["support_draw"])
            dose = int(entry.get("paths_per_soma", 1))
            union_positions[(depth, draw, dose)] = np.asarray(
                [lookup[int(value)] for value in entry["selected_indices"]],
                dtype=np.int64,
            )
    return union_indices, union_positions


def _capture_signal_values(
    *,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    sites: dict[tuple[int, str], _SignalSite],
    union_indices: dict[int, np.ndarray],
    runtime: EvaluationRuntimeConfig | None,
    max_samples: int | None,
    device: torch.device,
    compute_metrics: bool,
) -> tuple[dict[tuple[int, str], np.ndarray], dict[str, Any]]:
    chunks: dict[tuple[int, str], list[torch.Tensor]] = defaultdict(list)
    handles = []
    for key, site in sites.items():
        indices = torch.as_tensor(union_indices[site.depth], dtype=torch.long)

        def _capture(_module, _inputs, output, *, capture_key=key, selected=indices):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"expected tensor signal at {capture_key}")
            flat = output.detach().reshape(output.shape[0], -1)
            chosen = selected.to(flat.device)
            chunks[capture_key].append(
                flat.index_select(1, chosen).to(dtype=torch.float32).cpu()
            )
            return None

        handles.append(site.producer.register_forward_hook(_capture))

    correct = 0
    count = 0
    cross_entropy_sum = 0.0
    n_classes: int | None = None
    try:
        with torch.no_grad():
            for batch in iter_analysis_batches(
                dataset,
                runtime,
                explicit_max_samples=max_samples,
                device=device,
            ):
                if len(batch) < 2:
                    raise TypeError(
                        "intervention analysis requires (input, label) data"
                    )
                inputs = batch[0].to(device)
                labels = batch[1].to(device).reshape(-1).long()
                logits = model(inputs)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise TypeError("intervention analysis requires 2-D class logits")
                if int(logits.shape[0]) != int(labels.shape[0]):
                    raise RuntimeError("model output and label counts differ")
                observed_classes = int(logits.shape[1])
                if n_classes is None:
                    n_classes = observed_classes
                elif n_classes != observed_classes:
                    raise RuntimeError("class-logit width changed across batches")
                count += int(labels.numel())
                if compute_metrics:
                    correct += int((logits.argmax(dim=1) == labels).sum().item())
                    cross_entropy_sum += float(
                        F.cross_entropy(logits, labels, reduction="sum").item()
                    )
    finally:
        for handle in handles:
            handle.remove()
    if count < 1:
        raise RuntimeError("intervention dataset is empty")
    values = {key: torch.cat(parts, dim=0).numpy() for key, parts in chunks.items()}
    for key in sites:
        if key not in values or int(values[key].shape[0]) != count:
            raise RuntimeError(f"signal capture count mismatch for {key}")
    metrics = {
        "sample_count": int(count),
        "n_classes": int(n_classes) if n_classes is not None else None,
        "accuracy": float(correct / count) if compute_metrics else None,
        "mean_cross_entropy": (
            float(cross_entropy_sum / count) if compute_metrics else None
        ),
    }
    return values, metrics


def _replacement_hook(
    *,
    method: str,
    selected_indices: np.ndarray,
    selected_union_positions: np.ndarray,
    fixed_reference_values: np.ndarray,
    shuffled_evaluation_values: np.ndarray,
):
    cursor = 0
    global_indices = torch.as_tensor(selected_indices, dtype=torch.long)
    union_positions = np.asarray(selected_union_positions, dtype=np.int64)
    fixed = np.asarray(fixed_reference_values, dtype=np.float32)[union_positions]
    shuffled = np.asarray(shuffled_evaluation_values, dtype=np.float32)[
        :, union_positions
    ]

    def _hook(_module, _inputs, output):
        nonlocal cursor
        if not isinstance(output, torch.Tensor):
            raise TypeError("path-matched intervention requires a tensor output")
        flat = output.reshape(output.shape[0], -1)
        batch_size = int(flat.shape[0])
        indices = global_indices.to(flat.device)
        if method == "zero":
            replacement = torch.zeros(
                (batch_size, len(global_indices)),
                device=flat.device,
                dtype=flat.dtype,
            )
        elif method == "fixed_train_mean":
            replacement = torch.as_tensor(fixed, device=flat.device, dtype=flat.dtype)
            replacement = replacement.reshape(1, -1).expand(batch_size, -1)
        elif method == "global_test_shuffle":
            donor = shuffled[cursor : cursor + batch_size]
            if int(donor.shape[0]) != batch_size:
                raise RuntimeError("global shuffle donor pool was exhausted early")
            replacement = torch.as_tensor(donor, device=flat.device, dtype=flat.dtype)
        else:
            raise ValueError(f"unknown intervention method: {method}")
        modified = flat.clone()
        modified[:, indices] = replacement
        cursor += batch_size
        return modified.reshape_as(output)

    return _hook, lambda: int(cursor)


def _evaluate_with_intervention(
    *,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    runtime: EvaluationRuntimeConfig | None,
    max_samples: int | None,
    device: torch.device,
    hook_specs: list[tuple[_SignalSite, Any]],
) -> dict[str, Any]:
    handles = [site.producer.register_forward_hook(hook) for site, hook in hook_specs]
    correct = 0
    count = 0
    cross_entropy_sum = 0.0
    try:
        with torch.no_grad():
            for batch in iter_analysis_batches(
                dataset,
                runtime,
                explicit_max_samples=max_samples,
                device=device,
            ):
                inputs = batch[0].to(device)
                labels = batch[1].to(device).reshape(-1).long()
                logits = model(inputs)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise TypeError("intervention analysis requires 2-D class logits")
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                cross_entropy_sum += float(
                    F.cross_entropy(logits, labels, reduction="sum").item()
                )
                count += int(labels.numel())
    finally:
        for handle in handles:
            handle.remove()
    if count < 1:
        raise RuntimeError("intervention evaluation dataset is empty")
    return {
        "sample_count": int(count),
        "accuracy": float(correct / count),
        "mean_cross_entropy": float(cross_entropy_sum / count),
    }


class PathMatchedInterventionAnalyzer:
    """Run equal-coordinate current or output interventions across depth."""

    def __init__(self, params: PathMatchedInterventionAnalysisParams):
        self.params = params

    def analyze(
        self,
        model: torch.nn.Module,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "final",
        training: bool = False,
        runtime: EvaluationRuntimeConfig | None = None,
        reference_dataset: torch.utils.data.Dataset | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate interventions using training references and held-out test data."""

        if training:
            raise ValueError("path-matched intervention is a final-only analysis")
        if reference_dataset is None and "fixed_train_mean" in self.params.methods:
            raise ValueError("fixed_train_mean requires a separate reference dataset")
        if reference_dataset is None:
            reference_dataset = test_dataset

        with analysis_device_context(model, device) as analysis_device:
            modules, n_somas, branch_factors = _tree_inventory(
                model,
                module_name_prefix=self.params.module_name_prefix,
                expected_n_somas=self.params.expected_n_somas,
                expected_branch_factors=self.params.expected_branch_factors,
            )
            if self.params.paths_per_soma:
                support_plan = build_nested_path_dose_supports(
                    n_somas=n_somas,
                    branch_factors=branch_factors,
                    draws=self.params.support_draws,
                    paths_per_soma=self.params.paths_per_soma,
                    seed=self.params.support_seed,
                    namespace=self.params.support_namespace,
                )
            else:
                support_plan = build_nested_path_supports(
                    n_somas=n_somas,
                    branch_factors=branch_factors,
                    draws=self.params.support_draws,
                    seed=self.params.support_seed,
                    namespace=self.params.support_namespace,
                )
            union_indices, union_positions = _union_supports(support_plan)
            sites = _signal_sites(modules)
            requested_depths = (
                set(self.params.depths)
                if self.params.depths
                else set(range(1, len(branch_factors) + 1))
            )
            required_keys = {
                (depth, signal)
                for target in self.params.targets
                for signal in _TARGET_SIGNALS[target]
                for depth in requested_depths
            }
            missing = sorted(required_keys - set(sites))
            if missing:
                raise ValueError(
                    f"requested intervention signals are unavailable: {missing}"
                )
            selected_sites = {key: sites[key] for key in sorted(required_keys)}

            fixed_values, reference_metrics = _capture_signal_values(
                model=model,
                dataset=reference_dataset,
                sites=selected_sites,
                union_indices=union_indices,
                runtime=runtime,
                max_samples=self.params.max_samples,
                device=analysis_device,
                compute_metrics=False,
            )
            fixed_means = {
                key: values.mean(axis=0, dtype=np.float64).astype(np.float32)
                for key, values in fixed_values.items()
            }
            evaluation_values, baseline = _capture_signal_values(
                model=model,
                dataset=test_dataset,
                sites=selected_sites,
                union_indices=union_indices,
                runtime=runtime,
                max_samples=self.params.max_samples,
                device=analysis_device,
                compute_metrics=True,
            )
            chance = (
                float(self.params.chance_accuracy)
                if self.params.chance_accuracy is not None
                else 1.0 / float(baseline["n_classes"])
            )
            baseline_accuracy = float(baseline["accuracy"])
            baseline_ce = float(baseline["mean_cross_entropy"])
            if baseline_accuracy <= chance:
                raise RuntimeError(
                    "normalized above-chance loss requires baseline accuracy above chance"
                )
            rng = np.random.default_rng(int(self.params.shuffle_seed))
            permutation = rng.permutation(int(baseline["sample_count"])).astype(
                np.int64
            )

            records: list[dict[str, Any]] = []
            for target in self.params.targets:
                signals = _TARGET_SIGNALS[target]
                for depth in sorted(requested_depths):
                    support_entries = support_plan["depths"][str(depth)]
                    for support_entry in support_entries:
                        draw = int(support_entry["support_draw"])
                        paths_per_soma = int(support_entry.get("paths_per_soma", 1))
                        selected = np.asarray(
                            support_entry["selected_indices"], dtype=np.int64
                        )
                        positions = union_positions[(depth, draw, paths_per_soma)]
                        for method in self.params.methods:
                            hook_specs = []
                            cursors = []
                            for signal in signals:
                                key = (depth, signal)
                                site = sites[key]
                                shuffled = evaluation_values[key][permutation]
                                hook, cursor = _replacement_hook(
                                    method=method,
                                    selected_indices=selected,
                                    selected_union_positions=positions,
                                    fixed_reference_values=fixed_means[key],
                                    shuffled_evaluation_values=shuffled,
                                )
                                hook_specs.append((site, hook))
                                cursors.append(cursor)
                            metrics = _evaluate_with_intervention(
                                model=model,
                                dataset=test_dataset,
                                runtime=runtime,
                                max_samples=self.params.max_samples,
                                device=analysis_device,
                                hook_specs=hook_specs,
                            )
                            if int(metrics["sample_count"]) != int(
                                baseline["sample_count"]
                            ):
                                raise RuntimeError(
                                    "intervention changed the evaluation sample count"
                                )
                            if any(
                                cursor() != int(baseline["sample_count"])
                                for cursor in cursors
                            ):
                                raise RuntimeError(
                                    "an intervention hook did not consume all examples"
                                )
                            accuracy = float(metrics["accuracy"])
                            cross_entropy = float(metrics["mean_cross_entropy"])
                            accuracy_drop = baseline_accuracy - accuracy
                            records.append(
                                {
                                    "soma_relative_depth": int(depth),
                                    "target": target,
                                    "target_signals": list(signals),
                                    "method": method,
                                    "support_draw": draw,
                                    "paths_per_soma": paths_per_soma,
                                    "selected_fraction_of_depth": float(
                                        len(selected)
                                        / sites[(depth, signals[0])].coordinate_count
                                    ),
                                    "technical_draw_is_model_replicate": False,
                                    "support_plan_sha256": support_plan[
                                        "support_plan_sha256"
                                    ],
                                    "selected_indices_sha256": support_entry[
                                        "selected_indices_sha256"
                                    ],
                                    "selected_compartment_coordinate_count": len(
                                        selected
                                    ),
                                    "producer_stream_count": len(signals),
                                    "replaced_tensor_coordinate_count": int(
                                        len(selected) * len(signals)
                                    ),
                                    "available_compartment_coordinate_count": int(
                                        sites[(depth, signals[0])].coordinate_count
                                    ),
                                    "intervention_scope": (
                                        "path_matched_fixed_count_compartment_coordinates"
                                    ),
                                    "baseline_accuracy": baseline_accuracy,
                                    "accuracy": accuracy,
                                    "accuracy_drop": float(accuracy_drop),
                                    "chance_accuracy": float(chance),
                                    "normalized_above_chance_accuracy_loss": float(
                                        accuracy_drop / (baseline_accuracy - chance)
                                    ),
                                    "baseline_mean_cross_entropy": baseline_ce,
                                    "mean_cross_entropy": cross_entropy,
                                    "mean_cross_entropy_increase": float(
                                        cross_entropy - baseline_ce
                                    ),
                                    "module_names": [
                                        sites[(depth, signal)].module_name
                                        for signal in signals
                                    ],
                                }
                            )

        includes_post_gate_output = "post_gate_output" in self.params.targets
        result = {
            "schema_version": 1,
            "analysis_type": "path_matched_depth_intervention",
            "depth_reference": SOMA_RELATIVE_DEPTH_REFERENCE,
            "estimand": (
                (
                    "accuracy and cross-entropy change after replacing the same "
                    "number of anatomically matched compartment-current or "
                    "post-reactivation output coordinates at each non-somatic "
                    "dendritic depth"
                )
                if includes_post_gate_output
                else (
                    "accuracy and cross-entropy change after replacing the same "
                    "number of anatomically matched compartment-current coordinates "
                    "at each dendritic depth"
                )
            ),
            "interpretation": (
                "support draws are prespecified technical coordinate draws, not "
                "independent trained-model replicates"
            ),
            "reference_split": "training",
            "evaluation_split": "test",
            "module_name_prefix": self.params.module_name_prefix,
            "tree": {
                "n_somas": int(n_somas),
                "branch_factors": branch_factors,
                "module_names_by_depth": {
                    str(depth): name for depth, (name, _module) in modules.items()
                },
            },
            "support_plan": support_plan,
            "shuffle": {
                "policy": (
                    "one dataset-wide donor permutation shared across depths and "
                    "across E/I streams in joint interventions"
                ),
                "seed": int(self.params.shuffle_seed),
                "permutation_sha256": _sha256_array(permutation),
            },
            "reference_sample_count": int(reference_metrics["sample_count"]),
            "baseline": baseline,
            "records": records,
        }
        if includes_post_gate_output:
            result["post_gate_output_semantics"] = {
                "stage": (
                    "DendriticBranchLayer output after voltage computation and "
                    "configured reactivation"
                ),
                "propagation": (
                    "replacement occurs before the output is consumed by the next "
                    "more-proximal branch layer"
                ),
                "depth_convention": (
                    "soma-relative: depth 1 is the proximal dendritic layer and "
                    "larger values are more distal; depth 0 is the soma and is not "
                    "part of this intervention"
                ),
            }
        if save_path is not None:
            save_dict(result, save_path, f"{filename}.json")
        return result


__all__ = [
    "PathMatchedInterventionAnalyzer",
    "build_nested_path_dose_supports",
    "build_nested_path_supports",
    "inventory_nested_tree",
]
