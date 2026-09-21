"""Matched-width class information along a selected dendritic tree.

This analyzer complements :mod:`path_matched_intervention`.  Both use the
same label-independent nested soma-to-distal path supports, but this module
measures the joint class information carried by currents and compartment
outputs on the validation split.  It deliberately does not interpret path
draws as independently trained models or average them inside the analyzer.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from dendritic_modeling.analysis.core.information_parts.information_hooks_mixin import (
    InformationHooksMixin,
)
from dendritic_modeling.analysis.tools.class_information import (
    matched_support_class_information,
)
from dendritic_modeling.analysis.tools.path_matched_intervention import (
    build_nested_path_supports,
    inventory_nested_tree,
)
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    iter_analysis_batches,
)
from dendritic_modeling.config.analysis import (
    EvaluationRuntimeConfig,
    PathMatchedInformationAnalysisParams,
)
from dendritic_modeling.utils.general import save_dict

_SIGNAL_COMPONENTS: dict[str, tuple[str, ...]] = {
    "excitation_current": ("pre_gate_excitation_current",),
    "inhibition_current": ("pre_gate_inhibition_current",),
    "joint_EI_currents": (
        "pre_gate_excitation_current",
        "pre_gate_inhibition_current",
    ),
    "upstream_child_current": ("pre_gate_upstream_current",),
    "pre_gate_voltage": ("pre_gate_voltage",),
    "post_gate_output": ("post_gate_output",),
}
_SIGNAL_ORDER = {name: index for index, name in enumerate(_SIGNAL_COMPONENTS)}


def _sha256_array(values: np.ndarray) -> str:
    """Hash one typed array for an auditable somatic support record."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _somatic_support(n_somas: int) -> dict[str, Any]:
    """Return the single full-soma support used at depth zero."""

    indices = np.arange(int(n_somas), dtype=np.int64)
    return {
        "support_draw": 0,
        "selected_indices": indices.tolist(),
        "selected_indices_sha256": _sha256_array(indices),
        "coordinates": [
            {
                "flat_index": int(index),
                "soma_owner": int(index),
                "path_from_soma": [],
            }
            for index in indices
        ],
    }


def _support_views(
    support_plan: dict[str, Any],
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[int, np.ndarray],
    dict[tuple[int, int], np.ndarray],
]:
    """Add the soma support and map full indices to captured-union positions."""

    supports = {
        0: [_somatic_support(int(support_plan["n_somas"]))],
        **{int(depth): entries for depth, entries in support_plan["depths"].items()},
    }
    union_indices: dict[int, np.ndarray] = {}
    union_positions: dict[tuple[int, int], np.ndarray] = {}
    for depth, entries in supports.items():
        union = np.unique(
            np.concatenate(
                [
                    np.asarray(entry["selected_indices"], dtype=np.int64)
                    for entry in entries
                ]
            )
        )
        lookup = {int(index): position for position, index in enumerate(union)}
        union_indices[depth] = union
        for entry in entries:
            draw = int(entry["support_draw"])
            union_positions[(depth, draw)] = np.asarray(
                [lookup[int(index)] for index in entry["selected_indices"]],
                dtype=np.int64,
            )
    return supports, union_indices, union_positions


def estimate_path_matched_class_information(
    features: dict[tuple[int, str], np.ndarray],
    labels: np.ndarray,
    *,
    supports: dict[int, list[dict[str, Any]]],
    union_positions: dict[tuple[int, int], np.ndarray],
    signals: list[str],
    depths: list[int],
    n_neighbors: int,
    n_label_shuffles: int,
    seed: int,
    support_plan_sha256: str,
    selected_index_supports_sha256: str,
    support_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Estimate every available signal on prespecified matched supports.

    ``features`` contains only the union of selected coordinates for each raw
    hook signal.  The same support positions are applied to E, I, child-current,
    and output representations.  Joint E+I concatenates the two selected
    vectors before one multivariate estimate.
    """

    targets = np.asarray(labels).reshape(-1)
    records: list[dict[str, Any]] = []
    available_depths: dict[str, list[int]] = {}
    for signal in signals:
        component_names = _SIGNAL_COMPONENTS[signal]
        signal_depths: list[int] = []
        for depth in depths:
            components = [features.get((depth, name)) for name in component_names]
            if any(values is None for values in components):
                continue
            arrays = [np.asarray(values) for values in components]
            if any(values.ndim != 2 for values in arrays):
                raise ValueError(f"captured information signal is not 2-D: {signal}")
            if any(values.shape[0] != targets.shape[0] for values in arrays):
                raise ValueError(f"captured sample count differs for {signal}")
            widths = {int(values.shape[1]) for values in arrays}
            if len(widths) != 1:
                raise ValueError(
                    f"component widths differ for {signal} at depth {depth}"
                )
            signal_depths.append(int(depth))
            for support_entry in supports[depth]:
                draw = int(support_entry["support_draw"])
                positions = union_positions[(depth, draw)]
                selected = np.concatenate(
                    [values[:, positions] for values in arrays], axis=1
                )
                seed_offset = (
                    int(depth) * 100_003 + draw * 1_009 + _SIGNAL_ORDER[signal] * 10_007
                )
                estimate = matched_support_class_information(
                    selected,
                    targets,
                    n_neighbors=int(n_neighbors),
                    n_label_shuffles=int(n_label_shuffles),
                    seed=int(seed) + seed_offset,
                    standardize=True,
                )
                records.append(
                    {
                        "soma_relative_depth": int(depth),
                        "signal": signal,
                        "source_signals": list(component_names),
                        "support_draw": draw,
                        "path_draw": draw,
                        "support_plan_sha256": support_plan_sha256,
                        "selected_index_supports_sha256": (
                            selected_index_supports_sha256
                        ),
                        "feature_selection_policy": (
                            "shared_nested_anatomical_coordinate_support"
                        ),
                        "feature_selection_seed": int(support_seed),
                        "selected_features": list(support_entry["selected_indices"]),
                        "selected_features_sha256": str(
                            support_entry["selected_indices_sha256"]
                        ),
                        "selected_coordinate_count_per_component": len(positions),
                        "joint_feature_count": int(selected.shape[1]),
                        "technical_draw_role": (
                            "coordinate-support sensitivity; not an independent "
                            "trained-model replicate"
                        ),
                        **estimate,
                    }
                )
        if signal_depths:
            available_depths[signal] = signal_depths
    missing_signals = sorted(set(signals) - set(available_depths))
    if missing_signals:
        raise ValueError(
            f"requested information signals are unavailable: {missing_signals}"
        )
    return records, available_depths


class PathMatchedInformationAnalyzer(InformationHooksMixin):
    """Estimate validation-set class information on nested path supports."""

    def __init__(self, params: PathMatchedInformationAnalysisParams):
        self.params = params
        self.compute_lda_weights = False

    def _capture_features(
        self,
        *,
        model: torch.nn.Module,
        validation_dataset: torch.utils.data.Dataset,
        modules: dict[int, tuple[str, torch.nn.Module]],
        union_indices: dict[int, np.ndarray],
        device: torch.device,
        runtime: EvaluationRuntimeConfig | None,
    ) -> tuple[dict[tuple[int, str], np.ndarray], np.ndarray]:
        chunks: dict[tuple[int, str], list[torch.Tensor]] = defaultdict(list)
        labels: list[torch.Tensor] = []
        handles = self.attach_forward_hooks(model)
        try:
            with torch.no_grad():
                for batch in iter_analysis_batches(
                    validation_dataset,
                    runtime,
                    self.params.max_samples,
                    device=device,
                ):
                    if len(batch) < 2:
                        raise TypeError(
                            "path-matched information requires (input, label) data"
                        )
                    inputs = batch[0].to(device)
                    batch_labels = batch[1].reshape(-1).long().cpu()
                    _ = model(inputs)
                    labels.append(batch_labels)
                    for depth, (module_name, _module) in modules.items():
                        layer_data = self.data_dict.get(module_name)
                        if layer_data is None:
                            raise RuntimeError(
                                f"information hooks did not capture {module_name}"
                            )
                        selected = torch.as_tensor(
                            union_indices[depth],
                            dtype=torch.long,
                        )
                        for raw_name in {
                            component
                            for signal in self.params.signals
                            for component in _SIGNAL_COMPONENTS[signal]
                        }:
                            if raw_name == "pre_gate_excitation_current" and not bool(
                                layer_data.get("has_exc_synapses", False)
                            ):
                                continue
                            if raw_name == "pre_gate_inhibition_current" and not bool(
                                layer_data.get("has_inh_synapses", False)
                            ):
                                continue
                            value = layer_data.get(raw_name)
                            if not isinstance(value, torch.Tensor):
                                continue
                            flattened = value.detach().reshape(value.shape[0], -1)
                            if int(selected.max()) >= int(flattened.shape[1]):
                                raise RuntimeError(
                                    f"support exceeds {raw_name} width at depth {depth}"
                                )
                            chunks[(depth, raw_name)].append(
                                flattened.index_select(
                                    1,
                                    selected.to(flattened.device),
                                )
                                .to(dtype=torch.float32)
                                .cpu()
                            )
        finally:
            self.remove_forward_hooks(handles)
        if not labels:
            raise RuntimeError("validation dataset is empty")
        label_array = torch.cat(labels, dim=0).numpy()
        features = {
            key: torch.cat(parts, dim=0).numpy() for key, parts in chunks.items()
        }
        incomplete = {
            key: int(values.shape[0])
            for key, values in features.items()
            if int(values.shape[0]) != int(label_array.shape[0])
        }
        if incomplete:
            raise RuntimeError(f"incomplete hook captures: {incomplete}")
        return features, label_array

    def analyze(
        self,
        model: torch.nn.Module,
        validation_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "final",
        training: bool = False,
        runtime: EvaluationRuntimeConfig | None = None,
        seed_offset: int = 0,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Run the final-only analysis without receiving test data."""

        if training:
            raise ValueError("path-matched information is a final-only analysis")
        with analysis_device_context(model, device) as analysis_device:
            modules, n_somas, branch_factors = inventory_nested_tree(
                model,
                module_name_prefix=self.params.module_name_prefix,
                expected_n_somas=self.params.expected_n_somas,
                expected_branch_factors=self.params.expected_branch_factors,
            )
            support_plan = build_nested_path_supports(
                n_somas=n_somas,
                branch_factors=branch_factors,
                draws=self.params.support_draws,
                seed=self.params.support_seed,
                namespace=self.params.support_namespace,
            )
            supports, union_indices, union_positions = _support_views(support_plan)
            requested_depths = (
                self.params.depths
                if self.params.depths
                else list(range(len(branch_factors) + 1))
            )
            unknown_depths = sorted(set(requested_depths) - set(supports))
            if unknown_depths:
                raise ValueError(
                    f"requested information depths do not exist: {unknown_depths}"
                )
            selected_union = {depth: union_indices[depth] for depth in requested_depths}
            features, labels = self._capture_features(
                model=model,
                validation_dataset=validation_dataset,
                modules={depth: modules[depth] for depth in requested_depths},
                union_indices=selected_union,
                device=analysis_device,
                runtime=runtime,
            )
            records, available_depths = estimate_path_matched_class_information(
                features,
                labels,
                supports={depth: supports[depth] for depth in requested_depths},
                union_positions=union_positions,
                signals=self.params.signals,
                depths=requested_depths,
                n_neighbors=self.params.n_neighbors,
                n_label_shuffles=self.params.n_label_shuffles,
                seed=self.params.information_seed + int(seed_offset),
                support_plan_sha256=str(support_plan["support_plan_sha256"]),
                selected_index_supports_sha256=str(
                    support_plan["selected_index_supports_sha256"]
                ),
                support_seed=self.params.support_seed,
            )

        result = {
            "schema_version": "path_matched_information_v1",
            "estimand": (
                "joint mutual information I(X; class) on matched anatomical "
                "coordinate supports"
            ),
            "analysis_split": "validation",
            "test_examples_or_labels_used": False,
            "feature_selection_uses_labels_or_activations": False,
            "information_units": "bits",
            "preprocessing": (
                "per-coordinate validation-sample z-score using population "
                "standard deviation; constant coordinates set to zero"
            ),
            "technical_draw_aggregation": (
                "preserve every path draw here; downstream collectors take the "
                "arithmetic mean within checkpoint before model-level inference"
            ),
            "technical_draws_are_model_replicates": False,
            "somatic_support_draw_count": 1,
            "non_somatic_support_draw_count": int(self.params.support_draws),
            "information_seed_base": int(self.params.information_seed),
            "information_seed_offset": int(seed_offset),
            "information_seed_effective": int(
                self.params.information_seed + int(seed_offset)
            ),
            "sample_count": int(labels.shape[0]),
            "tree": {
                "n_somas": int(n_somas),
                "branch_factors": branch_factors,
                "module_names_by_depth": {
                    str(depth): name for depth, (name, _module) in modules.items()
                },
            },
            "available_depths_by_signal": available_depths,
            "support_plan": support_plan,
            "somatic_support": supports[0][0],
            "records": records,
        }
        if save_path is not None:
            save_dict(result, save_path, f"{filename}.json")
        return result


__all__ = [
    "PathMatchedInformationAnalyzer",
    "estimate_path_matched_class_information",
]
