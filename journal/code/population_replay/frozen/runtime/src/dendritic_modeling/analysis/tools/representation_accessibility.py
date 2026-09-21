"""Held-out accessibility and information of stacked soma populations.

The analyzer captures post-gate soma outputs from every feedforward
``PopulationLayer``. Linear and nonlinear probes are fit only on validation
representations and evaluated on untouched test representations. Optional
mutual-information estimates use validation representations with explicit
label-shuffle baselines, leaving test examples for held-out probe evaluation.

Probe accuracy and mutual information are intentionally reported as different
estimands. Per-soma scalar information is also kept separate from joint
population information because averaging scalar estimates is sensitive to
redundancy and is not the information in the complete population code.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from dendritic_modeling.analysis.tools.class_accessibility import (
    validation_to_test_probe_decode,
)
from dendritic_modeling.analysis.tools.class_information import (
    matched_support_class_information,
)
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    iter_analysis_batches,
)
from dendritic_modeling.config.analysis import (
    EvaluationRuntimeConfig,
    RepresentationAccessibilityAnalysisParams,
    RepresentationProbeConfig,
    RepresentationProjectionConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_layer import (
    PopulationLayer,
)
from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationNetwork,
)
from dendritic_modeling.utils.general import save_dict

NETWORK_DEPTH_REFERENCE = "zero-based stacked PopulationNetwork layer index"
FEATURE_SOURCE = "post-gate soma output"
PROJECTION_SCHEME = "balanced_countsketch_v1"


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


def _derived_seed(base_seed: int, *coordinates: int) -> int:
    sequence = np.random.SeedSequence(
        [int(base_seed), *(int(value) for value in coordinates), 0x5A17C0DE]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def population_layer_inventory(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Return the ordered feedforward population-layer contract."""

    core = getattr(model, "core_network", None)
    if not isinstance(core, PopulationNetwork):
        raise TypeError("representation analysis requires a PopulationNetwork core")
    if core.is_recurrent:
        raise ValueError(
            "representation analysis currently requires a feedforward stack"
        )

    records: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(core.layers):
        if not isinstance(layer, PopulationLayer) or layer.is_recurrent:
            raise ValueError(
                "every analyzed layer must be a feedforward PopulationLayer"
            )
        polarities = {
            str(definition.name): str(definition.polarity)
            for definition in layer.population_definitions
        }
        populations = []
        for population_name, population in layer.populations.items():
            polarity = polarities.get(str(population_name))
            if polarity not in {"excitatory", "inhibitory"}:
                raise ValueError(
                    f"unknown polarity for {layer.config.name}.{population_name}"
                )
            populations.append(
                {
                    "population_name": str(population_name),
                    "polarity": polarity,
                    "output_dim": int(population.output_dim),
                    "module_type": type(population).__name__,
                }
            )
        records.append(
            {
                "network_layer_index": int(layer_index),
                "network_layer_number": int(layer_index) + 1,
                "network_layer_name": str(layer.config.name),
                "readout_population": str(layer.readout_population),
                "output_dim": int(layer.output_dim),
                "populations": populations,
            }
        )
    if not records:
        raise RuntimeError("PopulationNetwork contains no layers")
    return records


def _flatten_feature(value: torch.Tensor) -> np.ndarray:
    values = value.detach().to(dtype=torch.float32).cpu().numpy()
    return values.reshape(values.shape[0], -1)


def collect_layer_population_features(
    *,
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: str | torch.device,
    runtime: EvaluationRuntimeConfig | None,
    max_samples: int | None,
) -> tuple[dict[tuple[int, str], np.ndarray], np.ndarray, dict[str, Any]]:
    """Capture named post-gate soma populations for one dataset split."""

    inventory = population_layer_inventory(model)
    if len(dataset) < 1:
        raise ValueError("representation dataset must not be empty")
    core = model.core_network
    chunks: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    calls = dict.fromkeys(range(len(inventory)), 0)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    observed_batches = 0

    for layer_index, layer in enumerate(core.layers):

        def _hook(_module, _inputs, _output, *, index=layer_index):
            calls[index] += 1
            expected = {
                record["population_name"] for record in inventory[index]["populations"]
            }
            observed = set(_module._last_outputs)
            if observed != expected:
                raise RuntimeError(
                    f"layer {index} soma-output keys changed: {observed} != {expected}"
                )
            for population_name in sorted(expected):
                output = _module._last_outputs[population_name]
                if not isinstance(output, torch.Tensor):
                    raise TypeError("population soma output must be a tensor")
                chunks[(index, population_name)].append(_flatten_feature(output))

        handles.append(layer.register_forward_hook(_hook))

    try:
        with analysis_device_context(model, device) as analysis_device:
            with torch.no_grad():
                for batch in iter_analysis_batches(
                    dataset,
                    runtime,
                    max_samples,
                    device=analysis_device,
                ):
                    if len(batch) < 2:
                        raise ValueError(
                            "representation analysis requires input and class label"
                        )
                    observed_batches += 1
                    batch_labels = torch.as_tensor(batch[1]).reshape(-1)
                    logits = model(batch[0].to(analysis_device))
                    if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
                        raise TypeError(
                            "classifier output must have shape [samples, classes]"
                        )
                    if logits.shape[0] != batch_labels.shape[0]:
                        raise RuntimeError(
                            "classifier output/label counts do not match"
                        )
                    labels.append(batch_labels.cpu().numpy())
                    predictions.append(logits.argmax(dim=-1).cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()

    if not labels:
        raise RuntimeError("representation analysis observed no dataset batches")
    label_values = np.concatenate(labels).astype(np.int64, copy=False)
    prediction_values = np.concatenate(predictions).astype(np.int64, copy=False)
    features = {
        key: np.concatenate(values, axis=0).astype(np.float32, copy=False)
        for key, values in chunks.items()
    }
    expected_keys = {
        (int(layer["network_layer_index"]), str(population["population_name"]))
        for layer in inventory
        for population in layer["populations"]
    }
    if set(features) != expected_keys:
        raise RuntimeError("layer-population feature capture is incomplete")
    for key, values in features.items():
        if values.shape[0] != label_values.shape[0]:
            raise RuntimeError(f"feature/label count mismatch for {key}")
        if not np.isfinite(values).all():
            raise FloatingPointError(f"non-finite layer representation for {key}")

    if set(calls.values()) != {observed_batches}:
        raise RuntimeError(f"population-layer forward counts changed: {calls}")
    return (
        features,
        label_values,
        {
            "n_samples": int(label_values.shape[0]),
            "n_classes": int(np.unique(label_values).size),
            "model_accuracy": float(np.mean(prediction_values == label_values)),
            "labels_sha256": _sha256_array(label_values),
            "predictions_sha256": _sha256_array(prediction_values),
            "representation_sha256": {
                f"layer={key[0]};population={key[1]}": _sha256_array(values)
                for key, values in sorted(features.items())
            },
        },
    )


def balanced_countsketch_projection(
    n_features: int,
    output_dim: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a label-independent signed hash with balanced bucket occupancy."""

    n_features = int(n_features)
    output_dim = int(output_dim)
    seed = int(seed)
    if n_features < 1 or output_dim < 1:
        raise ValueError("projection dimensions must be positive")
    if n_features <= output_dim:
        buckets = np.arange(n_features, dtype=np.int64)
        signs = np.ones(n_features, dtype=np.int8)
        scheme = "identity_v1"
        decoded_features = n_features
    else:
        rng = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([seed, n_features, output_dim, 0x5A17C0DE])
            )
        )
        order = rng.permutation(n_features)
        buckets = np.empty(n_features, dtype=np.int64)
        buckets[order] = np.arange(n_features, dtype=np.int64) % output_dim
        signs = (2 * rng.integers(0, 2, n_features, dtype=np.int8) - 1).astype(
            np.int8,
            copy=False,
        )
        scheme = PROJECTION_SCHEME
        decoded_features = output_dim
    contract = {
        "feature_projection": scheme,
        "feature_projection_seed": seed,
        "input_feature_count": n_features,
        "decoded_feature_count": decoded_features,
        "bucket_occupancies": np.bincount(
            buckets,
            minlength=decoded_features,
        ).tolist(),
    }
    contract["feature_projection_sha256"] = _sha256_json(
        {
            **contract,
            "buckets": buckets.tolist(),
            "signs": signs.tolist(),
        }
    )
    return buckets, signs, contract


def apply_balanced_countsketch(
    values: np.ndarray,
    output_dim: int,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project a finite two-dimensional representation without using labels."""

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("CountSketch input must be a finite feature matrix")
    buckets, signs, metadata = balanced_countsketch_projection(
        values.shape[1],
        output_dim,
        seed=seed,
    )
    if values.shape[1] <= output_dim:
        return values.copy(), metadata
    projected = np.zeros((values.shape[0], output_dim), dtype=np.float32)
    for source in range(values.shape[1]):
        projected[:, buckets[source]] += values[:, source] * signs[source]
    if not np.isfinite(projected).all():
        raise FloatingPointError("CountSketch produced non-finite values")
    return projected, metadata


def _apply_projection(
    values: np.ndarray,
    projection: RepresentationProjectionConfig,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if projection.type == "identity":
        metadata = {
            "feature_projection": "identity_v1",
            "feature_projection_seed": int(seed),
            "input_feature_count": int(values.shape[1]),
            "decoded_feature_count": int(values.shape[1]),
            "bucket_occupancies": [1] * int(values.shape[1]),
        }
        metadata["feature_projection_sha256"] = _sha256_json(metadata)
        return values.copy(), metadata
    if projection.output_dim is None:
        raise ValueError("CountSketch projection requires output_dim")
    return apply_balanced_countsketch(values, projection.output_dim, seed=seed)


def _population_names_by_layer(
    inventory: Sequence[dict[str, Any]],
) -> dict[int, dict[str, list[str]]]:
    grouped: dict[int, dict[str, list[str]]] = {}
    for layer in inventory:
        layer_index = int(layer["network_layer_index"])
        grouped[layer_index] = {"excitatory": [], "inhibitory": [], "joint": []}
        for population in layer["populations"]:
            name = str(population["population_name"])
            polarity = str(population["polarity"])
            grouped[layer_index][polarity].append(name)
            grouped[layer_index]["joint"].append(name)
        grouped[layer_index]["readout"] = [str(layer["readout_population"])]
    return grouped


def build_layer_representations(
    features: dict[tuple[int, str], np.ndarray],
    *,
    inventory: Sequence[dict[str, Any]],
    requested: Sequence[str],
) -> tuple[
    dict[tuple[int, str], np.ndarray],
    dict[tuple[int, str], list[str]],
    list[dict[str, Any]],
]:
    """Construct declared E, I, joint, readout, or named population matrices."""

    names_by_layer = _population_names_by_layer(inventory)
    matrices: dict[tuple[int, str], np.ndarray] = {}
    members: dict[tuple[int, str], list[str]] = {}
    missing: list[dict[str, Any]] = []
    for layer_index in sorted(names_by_layer):
        available_names = set(names_by_layer[layer_index]["joint"])
        for requested_name in requested:
            requested_name = str(requested_name)
            if requested_name.startswith("population:"):
                population_names = [requested_name.split(":", 1)[1]]
            else:
                population_names = names_by_layer[layer_index].get(requested_name, [])
            population_names = [
                name for name in population_names if name in available_names
            ]
            if not population_names:
                missing.append(
                    {
                        "network_layer_index": int(layer_index),
                        "representation": requested_name,
                        "reason": "no matching soma population",
                    }
                )
                continue
            key = (int(layer_index), requested_name)
            values = [features[(layer_index, name)] for name in population_names]
            matrices[key] = (
                values[0].copy() if len(values) == 1 else np.concatenate(values, axis=1)
            )
            members[key] = list(population_names)
    return matrices, members, missing


def _probe_kwargs(probe: RepresentationProbeConfig) -> dict[str, Any]:
    return {
        "probe_type": probe.type,
        "n_label_shuffles": probe.n_label_shuffles,
        "ridge_alpha": probe.ridge_alpha,
        "mlp_hidden_dims": tuple(probe.hidden_dims),
        "mlp_activation": probe.activation,
        "mlp_alpha": probe.alpha,
        "mlp_learning_rate_init": probe.learning_rate_init,
        "mlp_batch_size": probe.batch_size,
        "mlp_max_iter": probe.max_iter,
        "mlp_early_stopping": probe.early_stopping,
        "mlp_validation_fraction": probe.validation_fraction,
        "mlp_n_iter_no_change": probe.n_iter_no_change,
    }


def _spectrum_summary(eigenvalues: np.ndarray) -> dict[str, float]:
    """Return scale-free effective-dimension statistics for one spectrum."""

    values = np.asarray(eigenvalues, dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return {
            "participation_ratio": 0.0,
            "entropy_effective_rank": 0.0,
            "top_eigenvalue_fraction": 0.0,
        }
    probabilities = values / values.sum()
    participation_ratio = 1.0 / float(np.square(probabilities).sum())
    entropy_effective_rank = float(
        np.exp(-np.sum(probabilities * np.log(probabilities)))
    )
    return {
        "participation_ratio": participation_ratio,
        "entropy_effective_rank": entropy_effective_rank,
        "top_eigenvalue_fraction": float(probabilities.max()),
    }


def population_geometry(
    values: np.ndarray,
    *,
    variance_epsilon: float,
) -> dict[str, Any]:
    """Measure covariance and correlation effective dimension without labels."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError("population geometry requires a samples-by-features matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("population geometry requires finite values")

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    variances = centered.var(axis=0, ddof=1)
    nonconstant = variances > float(variance_epsilon)
    covariance_singular_values = np.linalg.svd(
        centered,
        full_matrices=False,
        compute_uv=False,
    )
    covariance_eigenvalues = np.square(covariance_singular_values)
    covariance = _spectrum_summary(covariance_eigenvalues)

    if np.any(nonconstant):
        standardized = centered[:, nonconstant] / np.sqrt(variances[nonconstant])
        correlation_singular_values = np.linalg.svd(
            standardized,
            full_matrices=False,
            compute_uv=False,
        )
        correlation_eigenvalues = np.square(correlation_singular_values)
    else:
        correlation_eigenvalues = np.asarray([], dtype=np.float64)
    correlation = _spectrum_summary(correlation_eigenvalues)

    native_width = int(matrix.shape[1])
    nonconstant_width = int(nonconstant.sum())
    return {
        "native_feature_count": native_width,
        "nonconstant_feature_count": nonconstant_width,
        "constant_feature_count": native_width - nonconstant_width,
        "covariance_participation_ratio": covariance["participation_ratio"],
        "covariance_entropy_effective_rank": covariance["entropy_effective_rank"],
        "covariance_top_eigenvalue_fraction": covariance["top_eigenvalue_fraction"],
        "covariance_participation_ratio_fraction": (
            covariance["participation_ratio"] / native_width
        ),
        "correlation_participation_ratio": correlation["participation_ratio"],
        "correlation_entropy_effective_rank": correlation["entropy_effective_rank"],
        "correlation_top_eigenvalue_fraction": correlation["top_eigenvalue_fraction"],
        "correlation_participation_ratio_fraction": (
            correlation["participation_ratio"] / nonconstant_width
            if nonconstant_width
            else 0.0
        ),
    }


class RepresentationAccessibilityAnalyzer:
    """Config-driven held-out analysis of stacked soma populations."""

    def __init__(self, params: RepresentationAccessibilityAnalysisParams):
        self.params = params

    def analyze(
        self,
        model: torch.nn.Module,
        train_ds: torch.utils.data.Dataset | None = None,
        valid_ds: torch.utils.data.Dataset | None = None,
        test_ds: torch.utils.data.Dataset | None = None,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "final",
        training: bool = False,
        runtime: EvaluationRuntimeConfig | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Fit probes on validation representations and evaluate test once."""

        del train_ds
        if training:
            raise ValueError("representation accessibility is a final-only analysis")
        if valid_ds is None or test_ds is None:
            raise ValueError("representation analysis requires valid and test splits")

        inventory = population_layer_inventory(model)
        validation_features, validation_labels, validation_summary = (
            collect_layer_population_features(
                model=model,
                dataset=valid_ds,
                device=device,
                runtime=runtime,
                max_samples=self.params.max_samples,
            )
        )
        test_features, test_labels, test_summary = collect_layer_population_features(
            model=model,
            dataset=test_ds,
            device=device,
            runtime=runtime,
            max_samples=self.params.max_samples,
        )
        if set(validation_features) != set(test_features):
            raise RuntimeError("validation/test population inventories differ")

        validation_representations, members, validation_missing = (
            build_layer_representations(
                validation_features,
                inventory=inventory,
                requested=self.params.representations,
            )
        )
        test_representations, test_members, test_missing = build_layer_representations(
            test_features,
            inventory=inventory,
            requested=self.params.representations,
        )
        if set(validation_representations) != set(test_representations):
            raise RuntimeError("validation/test representation keys differ")
        if members != test_members or validation_missing != test_missing:
            raise RuntimeError("validation/test representation contracts differ")

        probe_records: list[dict[str, Any]] = []
        joint_information_records: list[dict[str, Any]] = []
        for representation_index, key in enumerate(sorted(validation_representations)):
            validation_values = validation_representations[key]
            test_values = test_representations[key]
            if validation_values.shape[1] != test_values.shape[1]:
                raise RuntimeError(f"validation/test feature widths differ for {key}")
            for projection_index, projection in enumerate(self.params.projections):
                draws = int(projection.draws)
                if projection.type == "identity" or (
                    projection.output_dim is not None
                    and validation_values.shape[1] <= projection.output_dim
                ):
                    draws = 1
                for projection_draw in range(draws):
                    projection_seed = _derived_seed(
                        self.params.seed,
                        key[0],
                        representation_index,
                        projection_index,
                        projection_draw,
                    )
                    validation_projected, projection_metadata = _apply_projection(
                        validation_values,
                        projection,
                        seed=projection_seed,
                    )
                    test_projected, test_projection_metadata = _apply_projection(
                        test_values,
                        projection,
                        seed=projection_seed,
                    )
                    if projection_metadata != test_projection_metadata:
                        raise RuntimeError(
                            "validation/test projection contracts differ"
                        )
                    for probe_index, probe in enumerate(self.params.probes):
                        if (
                            probe.max_projection_draws is not None
                            and projection_draw >= probe.max_projection_draws
                        ):
                            continue
                        probe_draws = 1 if probe.type == "ridge" else int(probe.draws)
                        for probe_draw in range(probe_draws):
                            probe_seed = _derived_seed(
                                projection_seed,
                                probe_index,
                                probe_draw,
                            )
                            decoded = validation_to_test_probe_decode(
                                validation_projected,
                                validation_labels,
                                test_projected,
                                test_labels,
                                seed=probe_seed,
                                **_probe_kwargs(probe),
                            )
                            observed = float(decoded["heldout_test_balanced_accuracy"])
                            null = decoded[
                                "heldout_test_label_shuffle_balanced_accuracy"
                            ]
                            probe_records.append(
                                {
                                    "network_layer_index": int(key[0]),
                                    "network_layer_number": int(key[0]) + 1,
                                    "representation": str(key[1]),
                                    "population_names": members[key],
                                    "native_feature_count": int(
                                        validation_values.shape[1]
                                    ),
                                    "projection_name": projection.name,
                                    "projection_draw": int(projection_draw),
                                    "probe_name": probe.name,
                                    "probe_draw": int(probe_draw),
                                    "class_accessibility_metric": (
                                        "heldout test balanced accuracy above "
                                        "validation-label shuffle null"
                                    ),
                                    "class_accessibility_above_null": (
                                        observed - float(null)
                                        if null is not None
                                        else None
                                    ),
                                    "is_mutual_information": False,
                                    **projection_metadata,
                                    **decoded,
                                }
                            )

                    information = self.params.information
                    if (
                        information.enabled
                        and information.compute_joint
                        and projection.name in information.projection_names
                    ):
                        info_seed = _derived_seed(projection_seed, 0x1F0)
                        info = matched_support_class_information(
                            validation_projected,
                            validation_labels,
                            n_neighbors=information.n_neighbors,
                            n_label_shuffles=information.n_label_shuffles,
                            seed=info_seed,
                            standardize=information.standardize,
                        )
                        joint_information_records.append(
                            {
                                "network_layer_index": int(key[0]),
                                "network_layer_number": int(key[0]) + 1,
                                "representation": str(key[1]),
                                "population_names": members[key],
                                "native_feature_count": int(validation_values.shape[1]),
                                "projection_name": projection.name,
                                "projection_draw": int(projection_draw),
                                "information_split": "valid",
                                "information_scope": "joint projected population",
                                **projection_metadata,
                                **info,
                            }
                        )

        per_soma_records = self._per_soma_information(
            validation_representations,
            validation_labels,
            members,
        )
        per_soma_summary = self._summarize_per_soma_information(per_soma_records)
        subset_probe_records, subset_probe_missing = self._subset_probe_records(
            validation_representations=validation_representations,
            validation_labels=validation_labels,
            test_representations=test_representations,
            test_labels=test_labels,
            members=members,
        )
        geometry_records = self._population_geometry_records(
            validation_representations,
            members,
        )
        result = {
            "schema_version": 1,
            "analysis_type": "representation_accessibility",
            "feature_source": FEATURE_SOURCE,
            "network_depth_reference": NETWORK_DEPTH_REFERENCE,
            "fit_split": "valid",
            "evaluation_split": "test",
            "information_split": "valid",
            "test_used_for_probe_selection": False,
            "test_used_for_information": False,
            "interpretation": {
                "probe_accuracy": (
                    "held-out class accessibility under the declared probe; not MI"
                ),
                "joint_information": (
                    "joint I(X; C) on the declared label-independent feature support"
                ),
                "per_soma_information": (
                    "separate scalar I(x_i; C) values; their mean is not joint MI"
                ),
                "subset_decoding": (
                    "held-out accessibility of label-independently sampled nested "
                    "population-coordinate subsets; E- or I-only coordinates are "
                    "individual somas, and technical subset draws are not "
                    "trained-model replicates"
                ),
                "population_geometry": (
                    "validation-only unsupervised covariance/correlation spectral "
                    "dimension; not class information"
                ),
            },
            "seed": int(self.params.seed),
            "inventory": inventory,
            "missing_representations": validation_missing,
            "validation_summary": validation_summary,
            "test_summary": test_summary,
            "probe_records": probe_records,
            "joint_information_records": joint_information_records,
            "per_soma_information_records": per_soma_records,
            "per_soma_information_summary": per_soma_summary,
            "subset_probe_records": subset_probe_records,
            "subset_probe_missing": subset_probe_missing,
            "population_geometry_records": geometry_records,
        }
        if save_path is not None:
            save_dict(result, save_path, f"{filename}.json")
        return result

    def _subset_probe_records(
        self,
        *,
        validation_representations: dict[tuple[int, str], np.ndarray],
        validation_labels: np.ndarray,
        test_representations: dict[tuple[int, str], np.ndarray],
        test_labels: np.ndarray,
        members: dict[tuple[int, str], list[str]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        subset = self.params.subset_decoding
        if not subset.enabled:
            return [], []

        allowed = set(subset.representations)
        records: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        requested_keys = [
            key for key in sorted(validation_representations) if key[1] in allowed
        ]
        present_representations = {key[1] for key in requested_keys}
        for representation in sorted(allowed - present_representations):
            missing.append(
                {
                    "representation": representation,
                    "reason": "representation was not captured",
                }
            )

        for representation_index, key in enumerate(requested_keys):
            validation_values = validation_representations[key]
            test_values = test_representations[key]
            native_width = int(validation_values.shape[1])
            valid_sizes = [
                int(size) for size in subset.subset_sizes if int(size) <= native_width
            ]
            omitted_sizes = [
                int(size) for size in subset.subset_sizes if int(size) > native_width
            ]
            if omitted_sizes:
                missing.append(
                    {
                        "network_layer_index": int(key[0]),
                        "network_layer_number": int(key[0]) + 1,
                        "representation": str(key[1]),
                        "native_feature_count": native_width,
                        "omitted_subset_sizes": omitted_sizes,
                        "reason": "requested subset exceeds native width",
                    }
                )
            for subset_draw in range(int(subset.draws)):
                selection_seed = _derived_seed(
                    self.params.seed,
                    key[0],
                    representation_index,
                    subset_draw,
                    0x5AB5E7,
                )
                generator = np.random.Generator(np.random.PCG64(selection_seed))
                permutation = generator.permutation(native_width)
                for subset_size in valid_sizes:
                    if subset_size == native_width and subset_draw > 0:
                        continue
                    selected = np.sort(permutation[:subset_size])
                    probe_seed = _derived_seed(selection_seed, subset_size, 0xDEC0DE)
                    decoded = validation_to_test_probe_decode(
                        validation_values[:, selected],
                        validation_labels,
                        test_values[:, selected],
                        test_labels,
                        seed=probe_seed,
                        **_probe_kwargs(subset.probe),
                    )
                    observed = float(decoded["heldout_test_balanced_accuracy"])
                    null = decoded["heldout_test_label_shuffle_balanced_accuracy"]
                    records.append(
                        {
                            "network_layer_index": int(key[0]),
                            "network_layer_number": int(key[0]) + 1,
                            "representation": str(key[1]),
                            "population_names": members[key],
                            "native_feature_count": native_width,
                            "subset_size": int(subset_size),
                            "subset_draw": int(subset_draw),
                            "subset_selection_seed": int(selection_seed),
                            "selected_coordinate_indices": selected.tolist(),
                            "subset_sampling": (
                                "label-independent nested uniform permutation of "
                                "native representation coordinates without replacement"
                            ),
                            "probe_name": subset.probe.name,
                            "class_accessibility_metric": (
                                "heldout test balanced accuracy above "
                                "validation-label shuffle null"
                            ),
                            "class_accessibility_above_null": (
                                observed - float(null) if null is not None else None
                            ),
                            "is_mutual_information": False,
                            **decoded,
                        }
                    )
        return records, missing

    def _population_geometry_records(
        self,
        validation_representations: dict[tuple[int, str], np.ndarray],
        members: dict[tuple[int, str], list[str]],
    ) -> list[dict[str, Any]]:
        geometry = self.params.geometry
        if not geometry.enabled:
            return []
        allowed = set(geometry.representations)
        records: list[dict[str, Any]] = []
        for key in sorted(validation_representations):
            if key[1] not in allowed:
                continue
            records.append(
                {
                    "network_layer_index": int(key[0]),
                    "network_layer_number": int(key[0]) + 1,
                    "representation": str(key[1]),
                    "population_names": members[key],
                    "measurement_split": "valid",
                    "label_usage": "none",
                    "geometry_scope": "complete native soma population",
                    **population_geometry(
                        validation_representations[key],
                        variance_epsilon=geometry.variance_epsilon,
                    ),
                }
            )
        return records

    def _per_soma_information(
        self,
        validation_representations: dict[tuple[int, str], np.ndarray],
        validation_labels: np.ndarray,
        members: dict[tuple[int, str], list[str]],
    ) -> list[dict[str, Any]]:
        information = self.params.information
        if not information.enabled or not information.compute_per_soma:
            return []
        records: list[dict[str, Any]] = []
        allowed = set(information.per_soma_representations)
        for key in sorted(validation_representations):
            if key[1] not in allowed:
                continue
            values = validation_representations[key]
            for soma_index in range(values.shape[1]):
                seed = _derived_seed(self.params.seed, key[0], soma_index, 0x50A4)
                info = matched_support_class_information(
                    values[:, soma_index : soma_index + 1],
                    validation_labels,
                    n_neighbors=information.n_neighbors,
                    n_label_shuffles=information.n_label_shuffles,
                    seed=seed,
                    standardize=information.standardize,
                )
                records.append(
                    {
                        "network_layer_index": int(key[0]),
                        "network_layer_number": int(key[0]) + 1,
                        "representation": str(key[1]),
                        "population_names": members[key],
                        "soma_index_within_representation": int(soma_index),
                        "information_split": "valid",
                        "information_scope": "single soma scalar",
                        **info,
                    }
                )
        return records

    @staticmethod
    def _summarize_per_soma_information(
        records: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[
                (int(record["network_layer_index"]), str(record["representation"]))
            ].append(record)
        summaries = []
        for key, values in sorted(grouped.items()):
            observed = np.asarray(
                [record["observed_mi_bits"] for record in values], dtype=float
            )
            excess = np.asarray(
                [record["excess_mi_over_label_shuffle_bits"] for record in values],
                dtype=float,
            )
            summaries.append(
                {
                    "network_layer_index": int(key[0]),
                    "network_layer_number": int(key[0]) + 1,
                    "representation": str(key[1]),
                    "information_split": "valid",
                    "n_somas": int(observed.size),
                    "mean_scalar_mi_bits": float(observed.mean()),
                    "median_scalar_mi_bits": float(np.median(observed)),
                    "mean_scalar_excess_mi_bits": float(excess.mean()),
                    "median_scalar_excess_mi_bits": float(np.median(excess)),
                    "interpretation": (
                        "summary across separately estimated somas; not joint population MI"
                    ),
                }
            )
        return summaries


__all__ = [
    "FEATURE_SOURCE",
    "NETWORK_DEPTH_REFERENCE",
    "PROJECTION_SCHEME",
    "RepresentationAccessibilityAnalyzer",
    "apply_balanced_countsketch",
    "balanced_countsketch_projection",
    "build_layer_representations",
    "collect_layer_population_features",
    "population_geometry",
    "population_layer_inventory",
]
