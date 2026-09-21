"""Whole local-inhibitory-population interventions across network position.

This analyzer is intentionally distinct from dendritic-depth intervention
tools. It replaces the complete post-reactivation output of one inhibitory
population in one feedforward ``PopulationLayer``. The graph is validated so
that the population has exactly one outgoing, same-step I-to-E route; in that
restricted graph the population-output intervention is also a route-specific
local-I intervention. Only the excitatory population is serialized between
network layers.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    iter_analysis_batches,
)
from dendritic_modeling.config.analysis import (
    EvaluationRuntimeConfig,
    LocalInhibitoryPopulationInterventionAnalysisParams,
)
from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationNetwork,
)
from dendritic_modeling.utils.general import save_dict

NETWORK_POSITION_REFERENCE = (
    "network_layer_index_zero_based_input_to_readout; distinct from "
    "soma_relative_dendritic_depth"
)


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class _LocalISite:
    layer_index: int
    layer_name: str
    population_name: str
    output_dim: int
    target_population: str
    module: torch.nn.Module


def _connection_field(connection: object, name: str, default: object = None) -> object:
    if isinstance(connection, dict):
        return connection.get(name, default)
    return getattr(connection, name, default)


def _local_i_inventory(
    model: torch.nn.Module,
    *,
    expected_network_layers: int | None,
    expected_inhibitory_width: int | None,
) -> list[_LocalISite]:
    core = getattr(model, "core_network", None)
    if not isinstance(core, PopulationNetwork):
        raise TypeError("local-I intervention requires a PopulationNetwork core")
    if core.is_recurrent:
        raise ValueError("local-I population intervention requires a feedforward stack")
    if expected_network_layers is not None and len(core.layers) != int(
        expected_network_layers
    ):
        raise ValueError(
            f"expected {expected_network_layers} network layers, found "
            f"{len(core.layers)}"
        )

    sites: list[_LocalISite] = []
    for layer_index, layer in enumerate(core.layers):
        polarities = {
            str(definition.name): str(definition.polarity)
            for definition in layer.population_definitions
        }
        inhibitory_names = [
            name for name, polarity in polarities.items() if polarity == "inhibitory"
        ]
        if len(inhibitory_names) != 1:
            raise ValueError(
                f"layer {layer.config.name!r} must contain exactly one inhibitory "
                f"population, found {inhibitory_names}"
            )
        population_name = inhibitory_names[0]
        if layer.readout_population == population_name:
            raise ValueError("the local inhibitory population cannot be the readout")
        outgoing = [
            connection
            for connection in layer.config.connections
            if bool(_connection_field(connection, "enabled", True))
            and str(_connection_field(connection, "source")) == population_name
        ]
        if len(outgoing) != 1:
            raise ValueError(
                f"{layer.config.name}.{population_name} must have exactly one "
                f"outgoing route, found {len(outgoing)}"
            )
        route = outgoing[0]
        target = str(_connection_field(route, "target"))
        if (
            str(_connection_field(route, "timing", "same_step")) != "same_step"
            or str(_connection_field(route, "pathway", "ff_inhibitory"))
            != "ff_inhibitory"
            or polarities.get(target) != "excitatory"
        ):
            raise ValueError(
                f"{layer.config.name}.{population_name} must route only through "
                "same-step ff_inhibitory to a local excitatory population"
            )
        qualified_source = f"{layer.config.name}.{population_name}"
        qualified_consumers = []
        for consumer_index, consumer_layer in enumerate(core.layers):
            for connection in consumer_layer.config.connections:
                if not bool(_connection_field(connection, "enabled", True)):
                    continue
                if str(_connection_field(connection, "source")) == qualified_source:
                    qualified_consumers.append(
                        (
                            int(consumer_index),
                            str(_connection_field(connection, "target")),
                            str(_connection_field(connection, "pathway")),
                        )
                    )
        if qualified_consumers:
            raise ValueError(
                f"{qualified_source} has qualified cross-layer consumers "
                f"{qualified_consumers}; its output is not a local-I-only route"
            )
        population = layer.populations[population_name]
        output_dim = int(population.output_dim)
        if expected_inhibitory_width is not None and output_dim != int(
            expected_inhibitory_width
        ):
            raise ValueError(
                f"expected inhibitory width {expected_inhibitory_width}, found "
                f"{output_dim} at layer {layer_index}"
            )
        sites.append(
            _LocalISite(
                layer_index=int(layer_index),
                layer_name=str(layer.config.name),
                population_name=population_name,
                output_dim=output_dim,
                target_population=target,
                module=population,
            )
        )
    if not sites:
        raise RuntimeError("PopulationNetwork contains no analyzable layers")
    return sites


def _capture_baseline(
    *,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    sites: list[_LocalISite],
    runtime: EvaluationRuntimeConfig | None,
    max_samples: int | None,
    device: torch.device,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, Any]]:
    chunks: dict[int, list[torch.Tensor]] = defaultdict(list)
    handles = []
    for site in sites:

        def _capture(_module, _inputs, output, *, index=site.layer_index):
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise TypeError("local inhibitory population output must be 2-D")
            chunks[index].append(output.detach().to(dtype=torch.float32).cpu())
            return None

        handles.append(site.module.register_forward_hook(_capture))

    labels_parts: list[torch.Tensor] = []
    correct = 0
    sample_count = 0
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
                    raise TypeError("local-I intervention requires (input, label) data")
                inputs = batch[0].to(device)
                labels = batch[1].to(device).reshape(-1).long()
                logits = model(inputs)
                if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                    raise TypeError("local-I intervention requires 2-D class logits")
                if logits.shape[0] != labels.shape[0]:
                    raise RuntimeError("model output and label counts differ")
                if n_classes is None:
                    n_classes = int(logits.shape[1])
                elif n_classes != int(logits.shape[1]):
                    raise RuntimeError("class-logit width changed across batches")
                labels_parts.append(labels.detach().cpu())
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                cross_entropy_sum += float(
                    F.cross_entropy(logits, labels, reduction="sum").item()
                )
                sample_count += int(labels.numel())
    finally:
        for handle in reversed(handles):
            handle.remove()
    if sample_count < 1 or n_classes is None:
        raise RuntimeError("local-I intervention dataset is empty")
    values = {index: torch.cat(parts, dim=0).numpy() for index, parts in chunks.items()}
    for site in sites:
        array = values.get(site.layer_index)
        if array is None or array.shape != (sample_count, site.output_dim):
            raise RuntimeError(
                f"local-I capture mismatch at network layer {site.layer_index}"
            )
        if not np.isfinite(array).all():
            raise FloatingPointError("local-I baseline output is non-finite")
    labels = torch.cat(labels_parts).numpy().astype(np.int64, copy=False)
    return (
        values,
        labels,
        {
            "sample_count": int(sample_count),
            "n_classes": int(n_classes),
            "accuracy": float(correct / sample_count),
            "mean_cross_entropy": float(cross_entropy_sum / sample_count),
        },
    )


def _replacement_hook(method: str, donor_values: np.ndarray):
    cursor = 0
    donor_values = np.asarray(donor_values, dtype=np.float32)

    def _hook(_module, _inputs, output):
        nonlocal cursor
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise TypeError("local inhibitory population output must be 2-D")
        batch_size = int(output.shape[0])
        if method == "zero":
            replacement = torch.zeros_like(output)
        else:
            donor = donor_values[cursor : cursor + batch_size]
            if donor.shape != tuple(output.shape):
                raise RuntimeError("local-I donor pool shape changed or was exhausted")
            replacement = torch.as_tensor(
                donor,
                device=output.device,
                dtype=output.dtype,
            )
        cursor += batch_size
        return replacement

    return _hook, lambda: int(cursor)


def _evaluate_intervention(
    *,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    site: _LocalISite,
    method: str,
    donor_values: np.ndarray,
    runtime: EvaluationRuntimeConfig | None,
    max_samples: int | None,
    device: torch.device,
) -> tuple[dict[str, float | int], int]:
    hook, cursor = _replacement_hook(method, donor_values)
    handle = site.module.register_forward_hook(hook)
    correct = 0
    sample_count = 0
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
                correct += int((logits.argmax(dim=1) == labels).sum().item())
                cross_entropy_sum += float(
                    F.cross_entropy(logits, labels, reduction="sum").item()
                )
                sample_count += int(labels.numel())
    finally:
        handle.remove()
    if sample_count < 1:
        raise RuntimeError("local-I intervention dataset is empty")
    return {
        "sample_count": int(sample_count),
        "accuracy": float(correct / sample_count),
        "mean_cross_entropy": float(cross_entropy_sum / sample_count),
    }, cursor()


def _permutations(
    labels: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[tuple[str, int], np.ndarray]:
    permutations: dict[tuple[str, int], np.ndarray] = {}
    sample_count = len(labels)
    for draw in range(draws):
        rng = np.random.default_rng(int(seed) + draw * 1009)
        global_order = rng.permutation(sample_count).astype(np.int64)
        within_order = np.arange(sample_count, dtype=np.int64)
        for class_label in np.unique(labels):
            indices = np.flatnonzero(labels == class_label)
            within_order[indices] = rng.permutation(indices)
        if not np.array_equal(labels[within_order], labels):
            raise RuntimeError("within-class donor permutation changed labels")
        permutations[("global_test_shuffle", draw)] = global_order
        permutations[("within_class_test_shuffle", draw)] = within_order
    return permutations


class LocalInhibitoryPopulationInterventionAnalyzer:
    """Replace each complete local inhibitory soma population in turn."""

    def __init__(self, params: LocalInhibitoryPopulationInterventionAnalysisParams):
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
        **_kwargs: Any,
    ) -> dict[str, Any]:
        if training:
            raise ValueError("local-I population intervention is final-only")
        with analysis_device_context(model, device) as analysis_device:
            sites = _local_i_inventory(
                model,
                expected_network_layers=self.params.expected_network_layers,
                expected_inhibitory_width=self.params.expected_inhibitory_width,
            )
            baseline_values, labels, baseline = _capture_baseline(
                model=model,
                dataset=test_dataset,
                sites=sites,
                runtime=runtime,
                max_samples=self.params.max_samples,
                device=analysis_device,
            )
            permutations = _permutations(
                labels,
                draws=self.params.permutation_draws,
                seed=self.params.shuffle_seed,
            )
            chance = 1.0 / float(baseline["n_classes"])
            baseline_accuracy = float(baseline["accuracy"])
            baseline_ce = float(baseline["mean_cross_entropy"])
            records: list[dict[str, Any]] = []
            for site in sites:
                for method in self.params.methods:
                    draws = (
                        range(self.params.permutation_draws)
                        if "shuffle" in method
                        else range(1)
                    )
                    for draw in draws:
                        permutation = (
                            permutations[(method, draw)]
                            if "shuffle" in method
                            else np.arange(len(labels), dtype=np.int64)
                        )
                        metrics, consumed = _evaluate_intervention(
                            model=model,
                            dataset=test_dataset,
                            site=site,
                            method=method,
                            donor_values=baseline_values[site.layer_index][permutation],
                            runtime=runtime,
                            max_samples=self.params.max_samples,
                            device=analysis_device,
                        )
                        if consumed != int(baseline["sample_count"]):
                            raise RuntimeError(
                                "local-I hook did not consume all examples"
                            )
                        accuracy = float(metrics["accuracy"])
                        cross_entropy = float(metrics["mean_cross_entropy"])
                        accuracy_drop = baseline_accuracy - accuracy
                        denominator = baseline_accuracy - chance
                        records.append(
                            {
                                "network_layer_index": site.layer_index,
                                "network_layer_number": site.layer_index + 1,
                                "network_layer_name": site.layer_name,
                                "network_position_reference": NETWORK_POSITION_REFERENCE,
                                "source_population": site.population_name,
                                "source_polarity": "inhibitory",
                                "target_population": site.target_population,
                                "route": "same_step_ff_inhibitory_local_I_to_E",
                                "target_semantics": (
                                    "complete post-reactivation inhibitory soma-"
                                    "population output; exactly one local I-to-E route"
                                ),
                                "replacement_coordinate_count": site.output_dim,
                                "method": method,
                                "permutation_draw": (
                                    int(draw) if "shuffle" in method else None
                                ),
                                "technical_draw_is_model_replicate": False,
                                "baseline_accuracy": baseline_accuracy,
                                "accuracy": accuracy,
                                "accuracy_drop": float(accuracy_drop),
                                "chance_accuracy": chance,
                                "normalized_above_chance_accuracy_loss": (
                                    float(accuracy_drop / denominator)
                                    if denominator > 0
                                    else None
                                ),
                                "baseline_mean_cross_entropy": baseline_ce,
                                "mean_cross_entropy": cross_entropy,
                                "mean_cross_entropy_increase": float(
                                    cross_entropy - baseline_ce
                                ),
                                "permutation_preserves_class": (
                                    bool(np.array_equal(labels[permutation], labels))
                                    if "shuffle" in method
                                    else None
                                ),
                                "permutation_class_agreement": (
                                    float(np.mean(labels[permutation] == labels))
                                    if "shuffle" in method
                                    else None
                                ),
                                "permutation_sha256": (
                                    _sha256_array(permutation)
                                    if "shuffle" in method
                                    else None
                                ),
                                "baseline_population_output_sha256": _sha256_array(
                                    baseline_values[site.layer_index]
                                ),
                            }
                        )

        paired = []
        if {
            "global_test_shuffle",
            "within_class_test_shuffle",
        }.issubset(self.params.methods):
            by_key = {
                (
                    int(record["network_layer_index"]),
                    str(record["method"]),
                    int(record["permutation_draw"]),
                ): record
                for record in records
                if record["permutation_draw"] is not None
            }
            for site in sites:
                for draw in range(self.params.permutation_draws):
                    global_record = by_key[
                        (site.layer_index, "global_test_shuffle", draw)
                    ]
                    within_record = by_key[
                        (site.layer_index, "within_class_test_shuffle", draw)
                    ]
                    paired.append(
                        {
                            "network_layer_index": site.layer_index,
                            "network_layer_number": site.layer_index + 1,
                            "permutation_draw": draw,
                            "class_aligned_mean_cross_entropy_excess": float(
                                global_record["mean_cross_entropy_increase"]
                                - within_record["mean_cross_entropy_increase"]
                            ),
                            "class_aligned_accuracy_drop_excess": float(
                                global_record["accuracy_drop"]
                                - within_record["accuracy_drop"]
                            ),
                            "is_unique_information_component": False,
                        }
                    )
        result = {
            "schema_version": 1,
            "analysis_type": "local_inhibitory_population_intervention",
            "network_position_reference": NETWORK_POSITION_REFERENCE,
            "estimand": (
                "frozen-classifier change after replacing one complete local "
                "inhibitory soma population at one feedforward network position"
            ),
            "not_estimands": [
                "within-tree dendritic-depth intervention",
                "unique information decomposition",
                "lateral or recurrent inhibition",
            ],
            "evaluation_split": "test",
            "baseline": baseline,
            "records": records,
            "paired_global_minus_within_class_records": paired,
            "technical_draws_are_model_replicates": False,
            "within_class_shuffle_uses_evaluation_labels": True,
            "primary_causal_metric": (
                "global-minus-within-class mean-cross-entropy increase"
            ),
            "shuffle_seed": int(self.params.shuffle_seed),
            "permutation_draws": int(self.params.permutation_draws),
        }
        if save_path is not None:
            save_dict(result, save_path, f"{filename}.json")
        return result


__all__ = [
    "NETWORK_POSITION_REFERENCE",
    "LocalInhibitoryPopulationInterventionAnalyzer",
]
