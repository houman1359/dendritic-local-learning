"""Split-safe source-tuning and sparse-support alignment analysis.

This module relates the *realized* sparse operator of a feedforward
``PopulationNetwork`` to the class tuning of the coordinates that actually
feed that operator.  It deliberately keeps three quantities separate:

1. source activity, estimated on an evaluation split;
2. the exact active contact mask; and
3. the corresponding effective synaptic conductance.

Target-soma preferred classes are selected on an independent reference split.
Only streaming sufficient statistics are retained.  In particular, this
analyzer never stores sample-by-synapse or sample-by-source arrays.

The coordinate contract is also explicit.  A first-layer source may be marked
as image pixels by configuration.  Sources in deeper network layers are native
source-neuron coordinates, not pixels and not composed receptive fields.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    iter_analysis_batches,
)
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import PopulationNetwork
from dendritic_modeling.networks.architectures.recurrent.population_constants import (
    _EXTERNAL_INPUT_SOURCES,
    _FF_EXC,
    _FF_INH,
    _SAME_STEP,
)
from dendritic_modeling.networks.architectures.recurrent.population_sources import (
    _split_qualified_source,
)
from dendritic_modeling.utils.hooks import remove_hook_handles

_PATHWAY_MODULE_ATTRIBUTES = {
    _FF_EXC: "branch_excitation",
    _FF_INH: "branch_inhibition",
}


@dataclass(frozen=True)
class SourceTuningOptions:
    """Runtime-independent options for source-tuning analysis.

    ``image_source_layers`` is intentionally empty by default.  Image-space
    semantics must be declared rather than inferred from a coincidentally
    matching feature dimension (for example, after an encoder projection).
    """

    pathways: tuple[str, ...] = (_FF_EXC, _FF_INH)
    target_polarities: tuple[str, ...] = ("excitatory", "inhibitory")
    image_shape: tuple[int, ...] | None = None
    image_source_layers: tuple[int, ...] = ()
    image_source_names: tuple[str, ...] = ("input", "input_e", "input_i")
    image_binarization_threshold: float = 0.5
    continuous_activity_threshold: float = 0.0
    input_batch_index: int = 0
    label_batch_index: int = 1
    max_reference_samples: int | None = None
    max_evaluation_samples: int | None = None
    require_independent_splits: bool = True
    retain_coordinate_profiles: bool = False
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        unknown_pathways = set(self.pathways) - set(_PATHWAY_MODULE_ATTRIBUTES)
        if unknown_pathways:
            raise ValueError(
                "source tuning currently supports feedforward pathways only; "
                f"got {sorted(unknown_pathways)}"
            )
        unknown_polarities = set(self.target_polarities) - {
            "excitatory",
            "inhibitory",
        }
        if unknown_polarities:
            raise ValueError(f"unknown target polarities: {sorted(unknown_polarities)}")
        if self.image_shape is not None:
            normalized_image_shape = tuple(int(value) for value in self.image_shape)
            if len(normalized_image_shape) not in {2, 3} or any(
                value < 1 for value in normalized_image_shape
            ):
                raise ValueError(
                    "image_shape must be [height, width] or "
                    "[channels, height, width] with positive dimensions"
                )
            object.__setattr__(self, "image_shape", normalized_image_shape)
        if not math.isfinite(self.image_binarization_threshold):
            raise ValueError("image_binarization_threshold must be finite")
        if not math.isfinite(self.continuous_activity_threshold):
            raise ValueError("continuous_activity_threshold must be finite")
        if self.max_reference_samples is not None and self.max_reference_samples < 1:
            raise ValueError("max_reference_samples must be positive")
        if self.max_evaluation_samples is not None and self.max_evaluation_samples < 1:
            raise ValueError("max_evaluation_samples must be positive")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")


@dataclass(frozen=True)
class TargetPopulationRecord:
    """One target population whose somatic class preference is estimated."""

    key: str
    network_layer_index: int
    network_layer_name: str
    target_population: str
    target_polarity: str
    n_soma: int
    module: torch.nn.Module = field(repr=False, compare=False)


@dataclass(frozen=True)
class SourceSupportBlock:
    """One source-coordinate block in one dendritic synaptic operator."""

    key: str
    target_key: str
    network_layer_index: int
    network_layer_name: str
    target_population: str
    target_polarity: str
    n_soma: int
    dendritic_depth: int
    pathway: str
    source_name: str
    source_polarity: str
    source_kind: str
    source_layer_name: str | None
    source_population_name: str | None
    coordinate_space: str
    image_shape: tuple[int, ...] | None
    source_start: int
    source_stop: int
    owner_index: torch.Tensor = field(repr=False, compare=False)
    synapse: torch.nn.Module = field(repr=False, compare=False)

    @property
    def source_dim(self) -> int:
        """Number of native coordinates in this source block."""
        return self.source_stop - self.source_start

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe identity and coordinate metadata."""
        return {
            "source_block_key": self.key,
            "target_key": self.target_key,
            "network_layer_index": self.network_layer_index,
            "network_layer_name": self.network_layer_name,
            "target_population": self.target_population,
            "target_polarity": self.target_polarity,
            "n_soma": self.n_soma,
            "dendritic_depth": self.dendritic_depth,
            "pathway": self.pathway,
            "source_name": self.source_name,
            "source_polarity": self.source_polarity,
            "source_kind": self.source_kind,
            "source_layer_name": self.source_layer_name,
            "source_population_name": self.source_population_name,
            "coordinate_space": self.coordinate_space,
            "image_shape": (
                None if self.image_shape is None else list(self.image_shape)
            ),
            "source_start": self.source_start,
            "source_stop": self.source_stop,
            "source_dim": self.source_dim,
        }


@dataclass(frozen=True)
class SourceTuningInventory:
    """Target populations and source blocks resolved from a network."""

    targets: tuple[TargetPopulationRecord, ...]
    source_blocks: tuple[SourceSupportBlock, ...]


class StreamingClassFeatureStatistics:
    """Per-class feature moments accumulated without retaining examples."""

    def __init__(self, feature_dim: int, activity_threshold: float):
        if int(feature_dim) < 1:
            raise ValueError("feature_dim must be positive")
        if not math.isfinite(float(activity_threshold)):
            raise ValueError("activity_threshold must be finite")
        self.feature_dim = int(feature_dim)
        self.activity_threshold = float(activity_threshold)
        self.counts: dict[int, int] = defaultdict(int)
        self.sums: dict[int, torch.Tensor] = {}
        self.sum_squares: dict[int, torch.Tensor] = {}
        self.active_counts: dict[int, torch.Tensor] = {}

    def update(self, values: torch.Tensor, labels: torch.Tensor) -> None:
        """Add one batch using on-device reductions and CPU sufficient stats."""
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(
                "source values must have shape [batch, feature_dim], got "
                f"{tuple(values.shape)} for feature_dim={self.feature_dim}"
            )
        labels = labels.reshape(-1)
        if labels.shape[0] != values.shape[0]:
            raise ValueError(
                "source values and labels have different sample counts: "
                f"{values.shape[0]} vs {labels.shape[0]}"
            )
        if labels.is_floating_point() or labels.is_complex():
            rounded = labels.round()
            if not bool(torch.equal(labels, rounded)):
                raise ValueError("class labels must be integer-valued")
            labels = rounded
        labels = labels.to(device=values.device, dtype=torch.long)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("source-tuning values must be finite")

        values64 = values.detach().to(dtype=torch.float64)
        for class_tensor in labels.unique(sorted=True):
            class_id = int(class_tensor.item())
            selected = values64[labels == class_tensor]
            if selected.numel() == 0:
                continue
            batch_count = int(selected.shape[0])
            batch_sum = selected.sum(dim=0).cpu()
            batch_sum_squares = selected.square().sum(dim=0).cpu()
            batch_active = (
                selected.gt(self.activity_threshold).sum(dim=0).to(torch.float64).cpu()
            )
            self.counts[class_id] += batch_count
            if class_id not in self.sums:
                self.sums[class_id] = torch.zeros(self.feature_dim, dtype=torch.float64)
                self.sum_squares[class_id] = torch.zeros(
                    self.feature_dim, dtype=torch.float64
                )
                self.active_counts[class_id] = torch.zeros(
                    self.feature_dim, dtype=torch.float64
                )
            self.sums[class_id].add_(batch_sum)
            self.sum_squares[class_id].add_(batch_sum_squares)
            self.active_counts[class_id].add_(batch_active)

    @property
    def total_count(self) -> int:
        """Total number of streamed examples."""
        return sum(self.counts.values())

    @property
    def classes(self) -> tuple[int, ...]:
        """Observed class labels in ascending order."""
        return tuple(sorted(self.counts))

    def class_conditional_means(
        self,
        classes: Iterable[int] | None = None,
    ) -> tuple[tuple[int, ...], torch.Tensor]:
        """Return equally represented class means without retaining samples.

        The returned matrix has shape ``[class, feature]``.  Sample counts do
        not enter the class axis, which is important when the split has class
        imbalance and the scientific quantity compares class-response shapes.
        """
        class_ids = self.classes if classes is None else tuple(int(c) for c in classes)
        if not class_ids:
            raise ValueError("class-conditional means require at least one class")
        missing = [class_id for class_id in class_ids if class_id not in self.counts]
        if missing:
            raise ValueError(f"class-conditional means lack classes: {missing}")
        means = torch.stack(
            [self.sums[class_id] / self.counts[class_id] for class_id in class_ids],
            dim=0,
        )
        return class_ids, means

    def _pooled(self, classes: Iterable[int]) -> tuple[int, torch.Tensor, ...]:
        selected = [int(value) for value in classes if int(value) in self.counts]
        count = sum(self.counts[value] for value in selected)
        if count == 0:
            zeros = torch.zeros(self.feature_dim, dtype=torch.float64)
            return 0, zeros, zeros.clone(), zeros.clone()
        sums = torch.stack([self.sums[value] for value in selected]).sum(dim=0)
        squares = torch.stack([self.sum_squares[value] for value in selected]).sum(
            dim=0
        )
        active = torch.stack([self.active_counts[value] for value in selected]).sum(
            dim=0
        )
        return count, sums, squares, active

    def preferred_vs_rest(self, preferred_class: int) -> dict[str, torch.Tensor | int]:
        """Return vector sufficient-statistic contrasts for one preferred class."""
        preferred_class = int(preferred_class)
        preferred = self._pooled([preferred_class])
        rest = self._pooled(
            class_id for class_id in self.classes if class_id != preferred_class
        )
        n_preferred, sum_preferred, sq_preferred, active_preferred = preferred
        n_rest, sum_rest, sq_rest, active_rest = rest
        if n_preferred < 1 or n_rest < 1:
            raise ValueError(
                "preferred-vs-rest tuning requires the preferred class and at "
                "least one other observed class"
            )

        preferred_mean = sum_preferred / n_preferred
        rest_mean = sum_rest / n_rest
        preferred_probability = active_preferred / n_preferred
        rest_probability = active_rest / n_rest
        mean_contrast = preferred_mean - rest_mean
        probability_contrast = preferred_probability - rest_probability

        preferred_variance = _sample_variance(n_preferred, sum_preferred, sq_preferred)
        rest_variance = _sample_variance(n_rest, sum_rest, sq_rest)
        degrees = max(n_preferred + n_rest - 2, 0)
        if degrees > 0:
            pooled_variance = (
                max(n_preferred - 1, 0) * preferred_variance
                + max(n_rest - 1, 0) * rest_variance
            ) / degrees
        else:
            pooled_variance = torch.zeros_like(mean_contrast)
        standardized = torch.where(
            pooled_variance > 0,
            mean_contrast / pooled_variance.sqrt(),
            torch.zeros_like(mean_contrast),
        )

        return {
            "n_preferred": n_preferred,
            "n_rest": n_rest,
            "preferred_mean": preferred_mean,
            "rest_mean": rest_mean,
            "mean_contrast": mean_contrast,
            "preferred_activity_probability": preferred_probability,
            "rest_activity_probability": rest_probability,
            "activity_probability_contrast": probability_contrast,
            "standardized_mean_difference": standardized,
        }


def _sample_variance(
    count: int,
    values_sum: torch.Tensor,
    values_sum_squares: torch.Tensor,
) -> torch.Tensor:
    """Numerically clipped unbiased sample variance from sufficient statistics."""
    if count < 2:
        return torch.zeros_like(values_sum)
    centered_sum_squares = values_sum_squares - values_sum.square() / count
    return (centered_sum_squares / (count - 1)).clamp_min(0)


def _source_description(
    network: PopulationNetwork,
    *,
    layer_index: int,
    source_name: str,
    source_dim: int,
    options: SourceTuningOptions,
) -> dict[str, str | None]:
    """Describe the native coordinates feeding one configured source block."""
    qualified = _split_qualified_source(source_name)
    if qualified is not None:
        source_layer, source_population = qualified
        return {
            "source_kind": "qualified_population",
            "source_layer_name": source_layer,
            "source_population_name": source_population,
            "coordinate_space": "native_source_neurons",
        }

    if source_name not in _EXTERNAL_INPUT_SOURCES:
        return {
            "source_kind": "same_layer_population",
            "source_layer_name": network.layer_names[layer_index],
            "source_population_name": source_name,
            "coordinate_space": "native_source_neurons",
        }

    if layer_index > 0:
        upstream_layer = network.layers[layer_index - 1]
        return {
            "source_kind": "previous_layer_readout_population",
            "source_layer_name": network.layer_names[layer_index - 1],
            "source_population_name": upstream_layer.readout_population,
            "coordinate_space": "native_source_neurons",
        }

    is_declared_image = (
        options.image_shape is not None
        and layer_index in options.image_source_layers
        and source_name in options.image_source_names
    )
    if is_declared_image:
        image_dim = math.prod(options.image_shape or ())
        if source_dim != image_dim:
            raise ValueError(
                f"declared image source {source_name!r} has dimension {source_dim}, "
                f"but image_shape={options.image_shape} has {image_dim} pixels"
            )
    return {
        "source_kind": "external_input",
        "source_layer_name": None,
        "source_population_name": None,
        "coordinate_space": (
            "image_pixels" if is_declared_image else "external_features"
        ),
    }


def inventory_population_network_source_support(
    model: torch.nn.Module,
    options: SourceTuningOptions,
) -> SourceTuningInventory:
    """Resolve feedforward source blocks without computing activity statistics."""
    core = getattr(model, "core_network", model)
    if not isinstance(core, PopulationNetwork):
        raise TypeError(
            "source-tuning support analysis currently requires PopulationNetwork; "
            f"got {type(core).__name__}"
        )
    if core.is_recurrent:
        raise ValueError(
            "source-tuning support analysis currently requires a feedforward "
            "PopulationNetwork; temporal source tuning needs an explicit time policy"
        )

    targets: list[TargetPopulationRecord] = []
    blocks: list[SourceSupportBlock] = []
    for layer_index, layer in enumerate(core.layers):
        layer_name = str(layer.config.name)
        polarity_by_population = {
            str(item.name): str(item.polarity).lower()
            for item in layer.population_definitions
        }
        for target_name, population in layer.populations.items():
            target_polarity = polarity_by_population[str(target_name)]
            if target_polarity not in options.target_polarities:
                continue
            n_soma = int(population.n_soma)
            target_key = f"layers.{layer_index}.{layer_name}.{target_name}"
            targets.append(
                TargetPopulationRecord(
                    key=target_key,
                    network_layer_index=layer_index,
                    network_layer_name=layer_name,
                    target_population=str(target_name),
                    target_polarity=target_polarity,
                    n_soma=n_soma,
                    module=population,
                )
            )

            for pathway in options.pathways:
                source_names = layer._stream_sources(
                    str(target_name), _SAME_STEP, pathway
                )
                if not source_names:
                    continue
                source_dims = [layer._source_dim(name) for name in source_names]
                stream_dim = sum(source_dims)
                offsets = [0]
                for source_dim in source_dims:
                    offsets.append(offsets[-1] + int(source_dim))

                module_attribute = _PATHWAY_MODULE_ATTRIBUTES[pathway]
                owner_indices = population._output_owner_index_per_level
                for level_position, branch_layer in enumerate(population.branch_layers):
                    synapse = getattr(branch_layer, module_attribute, None)
                    if synapse is None:
                        continue
                    in_features = int(getattr(synapse, "in_features", stream_dim))
                    if in_features != stream_dim:
                        raise ValueError(
                            f"{target_key} {pathway} source dimension is {stream_dim}, "
                            f"but its synapse expects {in_features} coordinates"
                        )
                    owner_index = owner_indices[level_position].detach().cpu().long()
                    dendritic_depth = int(
                        getattr(branch_layer, "layer_idx", level_position)
                    )
                    for source_position, source_name in enumerate(source_names):
                        source_dim = int(source_dims[source_position])
                        description = _source_description(
                            core,
                            layer_index=layer_index,
                            source_name=source_name,
                            source_dim=source_dim,
                            options=options,
                        )
                        source_start = offsets[source_position]
                        source_stop = offsets[source_position + 1]
                        key = (
                            f"{target_key}.depth_{dendritic_depth}.{pathway}."
                            f"source_{source_position}_{source_name}"
                        )
                        blocks.append(
                            SourceSupportBlock(
                                key=key,
                                target_key=target_key,
                                network_layer_index=layer_index,
                                network_layer_name=layer_name,
                                target_population=str(target_name),
                                target_polarity=target_polarity,
                                n_soma=n_soma,
                                dendritic_depth=dendritic_depth,
                                pathway=pathway,
                                source_name=str(source_name),
                                source_polarity=str(
                                    layer._source_polarity(source_name)
                                ),
                                source_kind=str(description["source_kind"]),
                                source_layer_name=description["source_layer_name"],
                                source_population_name=description[
                                    "source_population_name"
                                ],
                                coordinate_space=str(description["coordinate_space"]),
                                image_shape=(
                                    options.image_shape
                                    if description["coordinate_space"] == "image_pixels"
                                    else None
                                ),
                                source_start=source_start,
                                source_stop=source_stop,
                                owner_index=owner_index,
                                synapse=synapse,
                            )
                        )
    if not targets:
        raise ValueError("no target populations matched source-tuning options")
    if not blocks:
        raise ValueError("no feedforward source-support blocks matched options")
    return SourceTuningInventory(tuple(targets), tuple(blocks))


def _weighted_mean(
    values: torch.Tensor,
    weights: torch.Tensor,
    *,
    epsilon: float,
) -> float | None:
    total = float(weights.sum().item())
    if total <= epsilon:
        return None
    return float(torch.dot(values, weights / total).item())


def centered_class_profile_alignment(
    target_class_means: torch.Tensor,
    source_class_means: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, bool, torch.Tensor]:
    """Align each source coordinate's class profile with a target soma.

    Both profiles are centered across equally weighted classes before cosine
    similarity is computed, making the result the Pearson correlation across
    class-conditional means.  A degenerate target or source profile is assigned
    alignment zero and reported separately rather than producing ``NaN``.

    Args:
        target_class_means: Target-soma means with shape ``[class]``.
        source_class_means: Source-coordinate means with shape
            ``[class, source_coordinate]``.
        epsilon: Strictly positive norm threshold for degeneracy.

    Returns:
        Alignment for every source coordinate, whether the target profile is
        degenerate, and a Boolean source-coordinate degeneracy mask.
    """
    target = target_class_means.detach().to(dtype=torch.float64).reshape(-1)
    sources = source_class_means.detach().to(dtype=torch.float64)
    if sources.ndim != 2 or sources.shape[0] != target.numel():
        raise ValueError(
            "target and source class profiles must have shapes [class] and "
            "[class, source_coordinate]"
        )
    if target.numel() < 2:
        raise ValueError("class-profile alignment requires at least two classes")
    if sources.shape[1] < 1:
        raise ValueError("class-profile alignment requires source coordinates")
    if not math.isfinite(float(epsilon)) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not bool(torch.isfinite(target).all()) or not bool(
        torch.isfinite(sources).all()
    ):
        raise ValueError("class-profile means must be finite")

    target_centered = target - target.mean()
    source_centered = sources - sources.mean(dim=0, keepdim=True)
    target_norm = torch.linalg.vector_norm(target_centered)
    source_norms = torch.linalg.vector_norm(source_centered, dim=0)
    target_degenerate = bool(target_norm <= epsilon)
    source_degenerate = source_norms <= epsilon
    alignment = torch.zeros(sources.shape[1], dtype=torch.float64)
    valid = ~source_degenerate
    if not target_degenerate and bool(valid.any()):
        numerators = torch.matmul(target_centered, source_centered[:, valid])
        denominators = target_norm * source_norms[valid]
        alignment[valid] = (numerators / denominators).clamp(-1.0, 1.0)
    return alignment, target_degenerate, source_degenerate


def summarize_support_alignment(
    *,
    tuning: Mapping[str, torch.Tensor | int],
    candidate_frequency: torch.Tensor,
    contact_frequency: torch.Tensor,
    effective_conductance: torch.Tensor,
    epsilon: float,
) -> dict[str, float | int | None]:
    """Reduce coordinate tuning under uniform, contact, and conductance weights."""
    candidate_frequency = candidate_frequency.to(torch.float64)
    contact_frequency = contact_frequency.to(torch.float64)
    effective_conductance = effective_conductance.to(torch.float64)
    vectors = {
        key: value.to(torch.float64)
        for key, value in tuning.items()
        if isinstance(value, torch.Tensor)
    }
    expected_dim = candidate_frequency.numel()
    if any(value.numel() != expected_dim for value in vectors.values()):
        raise ValueError("tuning and support profiles must share a source dimension")
    if contact_frequency.numel() != expected_dim:
        raise ValueError("contact and candidate profiles must share a source dimension")
    if effective_conductance.numel() != expected_dim:
        raise ValueError(
            "conductance and candidate profiles must share a source dimension"
        )
    if bool((effective_conductance < -epsilon).any()):
        raise ValueError(
            "effective-conductance alignment requires non-negative conductances"
        )
    effective_conductance = effective_conductance.clamp_min(0)
    eligible = candidate_frequency > 0
    uniform_weights = eligible.to(torch.float64)
    weighting_profiles = {
        "source_uniform": uniform_weights,
        "contact_weighted": contact_frequency,
        "conductance_weighted": effective_conductance,
    }
    output: dict[str, float | int | None] = {
        "n_source_coordinates": expected_dim,
        "n_eligible_source_coordinates": int(eligible.sum().item()),
    }
    metric_names = (
        "preferred_mean",
        "rest_mean",
        "mean_contrast",
        "preferred_activity_probability",
        "rest_activity_probability",
        "activity_probability_contrast",
        "standardized_mean_difference",
        "class_profile_alignment",
    )
    for weighting_name, weighting_profile in weighting_profiles.items():
        for metric_name in metric_names:
            output[f"{weighting_name}_{metric_name}"] = _weighted_mean(
                vectors[metric_name], weighting_profile, epsilon=epsilon
            )
    for metric_name in (
        "mean_contrast",
        "activity_probability_contrast",
        "standardized_mean_difference",
        "class_profile_alignment",
    ):
        baseline = output[f"source_uniform_{metric_name}"]
        for weighting_name in ("contact_weighted", "conductance_weighted"):
            weighted = output[f"{weighting_name}_{metric_name}"]
            output[f"{weighting_name}_{metric_name}_excess_over_uniform"] = (
                None
                if baseline is None or weighted is None
                else float(weighted - baseline)
            )
    return output


def _source_stream_identity(block: SourceSupportBlock) -> tuple[Any, ...]:
    """Identity of one actual source tensor slice, independent of tree depth."""
    return (
        block.network_layer_index,
        block.target_key,
        block.pathway,
        block.source_name,
        block.source_start,
        block.source_stop,
        block.coordinate_space,
    )


def _group_source_stream_blocks(
    blocks: Iterable[SourceSupportBlock],
) -> tuple[tuple[SourceSupportBlock, ...], ...]:
    """Group depth-repeated operators that consume the same source stream.

    Dendritic levels retain distinct sparse masks and conductances, but their
    source activity tensor is identical.  Grouping avoids streaming the same
    class statistics once per tree depth.  Metadata assertions make that reuse
    explicit rather than relying on coincidentally equal feature dimensions.
    """
    grouped: dict[tuple[Any, ...], list[SourceSupportBlock]] = defaultdict(list)
    for block in blocks:
        grouped[_source_stream_identity(block)].append(block)

    invariant_fields = (
        "network_layer_index",
        "network_layer_name",
        "target_key",
        "target_population",
        "target_polarity",
        "n_soma",
        "pathway",
        "source_name",
        "source_polarity",
        "source_kind",
        "source_layer_name",
        "source_population_name",
        "coordinate_space",
        "image_shape",
        "source_start",
        "source_stop",
        "source_dim",
    )
    output: list[tuple[SourceSupportBlock, ...]] = []
    for members in grouped.values():
        representative = members[0]
        for member in members[1:]:
            mismatched = [
                field_name
                for field_name in invariant_fields
                if getattr(member, field_name) != getattr(representative, field_name)
            ]
            if mismatched:
                raise ValueError(
                    "cannot reuse source statistics across blocks with mismatched "
                    f"metadata: {mismatched}"
                )
        output.append(tuple(members))
    return tuple(output)


class SourceTuningSupportAnalyzer(AbstractAnalyzer):
    """Analyze split-safe source tuning against exact sparse synaptic support."""

    def __init__(
        self,
        options: SourceTuningOptions | None = None,
        runtime: EvaluationRuntimeConfig | None = None,
    ):
        super().__init__("SourceTuningSupportAnalyzer")
        self.options = options or SourceTuningOptions()
        self.runtime = runtime or EvaluationRuntimeConfig()

    def analyze(
        self,
        model: BaseModel,
        data: torch.utils.data.Dataset,
        device: str = "cpu",
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze evaluation ``data`` using preferences from ``reference_data``.

        Args:
            model: Frozen model containing a feedforward ``PopulationNetwork``.
            data: Independent evaluation dataset used for source tuning.
            device: Analysis device.
            **kwargs: Must include ``reference_data``. Optional
                ``reference_split_name`` and ``evaluation_split_name`` are
                retained as provenance.
        """
        reference_data = kwargs.get("reference_data")
        if reference_data is None:
            raise ValueError(
                "source-tuning analysis requires reference_data for independent "
                "preferred-class selection"
            )
        if self.options.require_independent_splits and reference_data is data:
            raise ValueError(
                "reference_data and evaluation data must be independent objects"
            )
        reference_split_name = str(kwargs.get("reference_split_name", "reference"))
        evaluation_split_name = str(kwargs.get("evaluation_split_name", "evaluation"))
        if (
            self.options.require_independent_splits
            and reference_split_name == evaluation_split_name
        ):
            raise ValueError(
                "reference and evaluation split names must be distinct when "
                "require_independent_splits is true"
            )
        if any(parameter.requires_grad for parameter in model.parameters()):
            self.logger.debug(
                "Analyzing model in eval/no-grad mode; trainable parameters are "
                "treated as a frozen checkpoint."
            )

        inventory = inventory_population_network_source_support(model, self.options)
        runtime = kwargs.get("runtime") or self.runtime

        with analysis_device_context(model, device) as analysis_device:
            reference_stats = self._collect_reference_preferences(
                model,
                reference_data,
                inventory,
                analysis_device,
                runtime,
            )
            (
                preference_records,
                preferred_classes,
                target_class_profiles,
            ) = self._select_preferences(inventory.targets, reference_stats)
            source_stats = self._collect_source_tuning(
                model,
                data,
                inventory,
                analysis_device,
                runtime,
            )
            reference_class_counts = _consistent_class_counts(reference_stats.values())
            evaluation_class_counts = _consistent_class_counts(source_stats.values())
            if set(reference_class_counts) != set(evaluation_class_counts):
                raise ValueError(
                    "reference and evaluation class sets must match for "
                    "class-profile alignment"
                )
            records, profiles = self._align_support(
                inventory,
                preferred_classes,
                target_class_profiles,
                source_stats,
            )

        result = {
            "schema_version": 1,
            "analysis_type": "source_tuning_support",
            "require_independent_splits": self.options.require_independent_splits,
            "quantity_contract": self._quantity_contract(),
            "reference": {
                "split": reference_split_name,
                "n_samples": _consistent_total_count(reference_stats.values()),
                "class_counts": reference_class_counts,
                "preferred_class_selection": "maximum class-conditional soma mean",
                "preference_records": preference_records,
            },
            "evaluation": {
                "split": evaluation_split_name,
                "n_samples": _consistent_total_count(source_stats.values()),
                "class_counts": evaluation_class_counts,
            },
            "n_alignment_records": len(records),
            "alignment_records": records,
            "coordinate_profiles": profiles,
        }
        save_path = kwargs.get("save_path")
        if save_path is not None:
            filename = Path(str(kwargs.get("filename", "final"))).stem
            self.save_results(result, str(Path(save_path) / f"{filename}.json"))
        return result

    def _collect_reference_preferences(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        inventory: SourceTuningInventory,
        device: torch.device,
        runtime: EvaluationRuntimeConfig,
    ) -> dict[str, StreamingClassFeatureStatistics]:
        stats = {
            target.key: StreamingClassFeatureStatistics(target.n_soma, 0.0)
            for target in inventory.targets
        }
        current_labels: dict[str, torch.Tensor | None] = {"value": None}
        handles = []
        try:
            for target in inventory.targets:
                hook = partial(
                    _target_output_hook,
                    key=target.key,
                    stats=stats,
                    current_labels=current_labels,
                )
                handles.append(target.module.register_forward_hook(hook))
            self._stream_dataset(
                model,
                dataset,
                device,
                current_labels,
                self.options.max_reference_samples,
                runtime,
            )
        finally:
            current_labels["value"] = None
            remove_hook_handles(handles)
        return stats

    def _collect_source_tuning(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        inventory: SourceTuningInventory,
        device: torch.device,
        runtime: EvaluationRuntimeConfig,
    ) -> dict[str, StreamingClassFeatureStatistics]:
        stats: dict[str, StreamingClassFeatureStatistics] = {}
        blocks_by_module: dict[torch.nn.Module, list[SourceSupportBlock]] = defaultdict(
            list
        )
        for stream_blocks in _group_source_stream_blocks(inventory.source_blocks):
            representative = stream_blocks[0]
            stream_stats = StreamingClassFeatureStatistics(
                representative.source_dim,
                (
                    self.options.image_binarization_threshold
                    if representative.coordinate_space == "image_pixels"
                    else self.options.continuous_activity_threshold
                ),
            )
            for block in stream_blocks:
                stats[block.key] = stream_stats
            blocks_by_module[representative.synapse].append(representative)

        current_labels: dict[str, torch.Tensor | None] = {"value": None}
        handles = []
        try:
            for module, blocks in blocks_by_module.items():
                hook = partial(
                    _source_input_pre_hook,
                    blocks=tuple(blocks),
                    stats=stats,
                    current_labels=current_labels,
                )
                handles.append(module.register_forward_pre_hook(hook))
            self._stream_dataset(
                model,
                dataset,
                device,
                current_labels,
                self.options.max_evaluation_samples,
                runtime,
            )
        finally:
            current_labels["value"] = None
            remove_hook_handles(handles)
        return stats

    def _stream_dataset(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        device: torch.device,
        current_labels: dict[str, torch.Tensor | None],
        max_samples: int | None,
        runtime: EvaluationRuntimeConfig,
    ) -> None:
        with torch.no_grad():
            for batch in iter_analysis_batches(
                dataset,
                runtime,
                max_samples,
                device=device,
            ):
                if len(batch) <= max(
                    self.options.input_batch_index,
                    self.options.label_batch_index,
                ):
                    raise ValueError("analysis batch does not contain input and label")
                inputs = batch[self.options.input_batch_index]
                labels = batch[self.options.label_batch_index]
                if not isinstance(inputs, torch.Tensor) or not isinstance(
                    labels, torch.Tensor
                ):
                    raise TypeError("analysis input and label must be tensors")
                labels = labels.reshape(-1).to(device)
                current_labels["value"] = labels
                _ = model(inputs.to(device))
                current_labels["value"] = None

    @staticmethod
    def _select_preferences(
        targets: tuple[TargetPopulationRecord, ...],
        stats: Mapping[str, StreamingClassFeatureStatistics],
    ) -> tuple[
        list[dict[str, Any]],
        dict[tuple[str, int], int],
        dict[tuple[str, int], tuple[tuple[int, ...], torch.Tensor]],
    ]:
        records: list[dict[str, Any]] = []
        preferences: dict[tuple[str, int], int] = {}
        class_profiles: dict[tuple[str, int], tuple[tuple[int, ...], torch.Tensor]] = {}
        for target in targets:
            target_stats = stats[target.key]
            if len(target_stats.classes) < 2:
                raise ValueError(
                    f"reference split for {target.key} must contain at least two classes"
                )
            class_ids, class_means = target_stats.class_conditional_means()
            for soma_index in range(target.n_soma):
                values = class_means[:, soma_index]
                order = torch.argsort(values, descending=True, stable=True)
                preferred_class = target_stats.classes[int(order[0].item())]
                runner_up_class = target_stats.classes[int(order[1].item())]
                preferred_mean = float(values[order[0]].item())
                runner_up_mean = float(values[order[1]].item())
                preferences[(target.key, soma_index)] = preferred_class
                class_profiles[(target.key, soma_index)] = (
                    class_ids,
                    values.clone(),
                )
                records.append(
                    {
                        "target_key": target.key,
                        "network_layer_index": target.network_layer_index,
                        "network_layer_name": target.network_layer_name,
                        "target_population": target.target_population,
                        "target_polarity": target.target_polarity,
                        "target_soma_index": soma_index,
                        "preferred_class": preferred_class,
                        "preferred_reference_mean": preferred_mean,
                        "runner_up_class": runner_up_class,
                        "runner_up_reference_mean": runner_up_mean,
                        "preference_margin": preferred_mean - runner_up_mean,
                    }
                )
        return records, preferences, class_profiles

    def _align_support(
        self,
        inventory: SourceTuningInventory,
        preferred_classes: Mapping[tuple[str, int], int],
        target_class_profiles: Mapping[
            tuple[str, int], tuple[tuple[int, ...], torch.Tensor]
        ],
        source_stats: Mapping[str, StreamingClassFeatureStatistics],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        records: list[dict[str, Any]] = []
        profiles: dict[str, dict[str, Any]] = {}
        snapshots: dict[torch.nn.Module, Any] = {}
        for block in inventory.source_blocks:
            if block.synapse not in snapshots:
                snapshots[block.synapse] = effective_synapse_snapshot(
                    block.synapse,
                    prefer_cached_forward_mask=False,
                )
            snapshot = snapshots[block.synapse]
            source_slice = slice(block.source_start, block.source_stop)
            candidate = snapshot.candidate_mask[:, source_slice].detach().cpu()
            active = snapshot.active_mask[:, source_slice].detach().cpu()
            effective = snapshot.effective_weight[:, source_slice].detach().cpu()
            if candidate.shape[0] != block.owner_index.numel():
                raise ValueError(
                    f"owner index for {block.key} has {block.owner_index.numel()} "
                    f"rows, but synapse has {candidate.shape[0]} outputs"
                )
            for soma_index in range(block.n_soma):
                owned = block.owner_index == soma_index
                n_branches = int(owned.sum().item())
                if n_branches < 1:
                    raise ValueError(
                        f"target soma {soma_index} owns no branches in {block.key}"
                    )
                candidate_soma = candidate[owned].to(torch.float64)
                active_soma = active[owned].to(torch.float64)
                effective_soma = effective[owned].to(torch.float64)
                candidate_frequency = candidate_soma.mean(dim=0)
                contact_frequency = active_soma.mean(dim=0)
                conductance = effective_soma.mean(dim=0)
                target_identity = (block.target_key, soma_index)
                preferred_class = preferred_classes[target_identity]
                block_stats = source_stats[block.key]
                tuning = block_stats.preferred_vs_rest(preferred_class)
                class_ids, target_class_means = target_class_profiles[target_identity]
                source_class_ids, source_class_means = (
                    block_stats.class_conditional_means(class_ids)
                )
                if source_class_ids != class_ids:
                    raise RuntimeError(
                        "source and target class-profile axes are inconsistent"
                    )
                (
                    class_profile_alignment,
                    target_profile_degenerate,
                    source_profile_degenerate,
                ) = centered_class_profile_alignment(
                    target_class_means,
                    source_class_means,
                    epsilon=self.options.epsilon,
                )
                tuning = {
                    **tuning,
                    "class_profile_alignment": class_profile_alignment,
                }
                alignment = summarize_support_alignment(
                    tuning=tuning,
                    candidate_frequency=candidate_frequency,
                    contact_frequency=contact_frequency,
                    effective_conductance=conductance,
                    epsilon=self.options.epsilon,
                )
                profile_key = f"{block.key}.soma_{soma_index}"
                record = {
                    **block.metadata(),
                    "target_soma_index": soma_index,
                    "preferred_class": preferred_class,
                    "activity_threshold": source_stats[block.key].activity_threshold,
                    "activity_threshold_kind": (
                        "image_binarization"
                        if block.coordinate_space == "image_pixels"
                        else "continuous_activity"
                    ),
                    "n_branches": n_branches,
                    "n_preferred_evaluation_samples": int(tuning["n_preferred"]),
                    "n_rest_evaluation_samples": int(tuning["n_rest"]),
                    "n_class_profile_classes": len(class_ids),
                    "n_degenerate_target_class_profiles": int(
                        target_profile_degenerate
                    ),
                    "n_degenerate_source_class_profiles": int(
                        source_profile_degenerate.sum().item()
                    ),
                    "n_eligible_degenerate_source_class_profiles": int(
                        (source_profile_degenerate & candidate_frequency.gt(0))
                        .sum()
                        .item()
                    ),
                    "candidate_contact_count": int(candidate_soma.ne(0).sum().item()),
                    "active_contact_count": int(active_soma.ne(0).sum().item()),
                    "active_fraction_of_candidates": (
                        None
                        if not bool(candidate_soma.ne(0).any())
                        else float(
                            active_soma.ne(0).sum().item()
                            / candidate_soma.ne(0).sum().item()
                        )
                    ),
                    "total_effective_conductance": float(effective_soma.sum().item()),
                    "profile_key": profile_key,
                    **alignment,
                }
                records.append(record)
                if self.options.retain_coordinate_profiles:
                    profiles[profile_key] = {
                        "coordinate_axis": block.coordinate_space,
                        "source_dim": block.source_dim,
                        "candidate_frequency": candidate_frequency,
                        "exact_contact_frequency": contact_frequency,
                        "mean_effective_conductance": conductance,
                        **{
                            key: value
                            for key, value in tuning.items()
                            if isinstance(value, torch.Tensor)
                        },
                    }
        return records, profiles

    def _quantity_contract(self) -> dict[str, Any]:
        return {
            "preferred_class": (
                "argmax class-conditional target-soma mean on the independent "
                "reference split"
            ),
            "source_tuning": (
                "class-conditional moments of the actual tensor coordinates "
                "entering each synaptic pathway, estimated on the evaluation split"
            ),
            "rest_pooling": "all non-preferred evaluation examples pooled by sample",
            "activity_probability": (
                "P(source_coordinate > configured threshold | class group)"
            ),
            "mean_contrast": "preferred mean minus rest mean",
            "activity_probability_contrast": (
                "preferred activity probability minus rest activity probability"
            ),
            "standardized_mean_difference": (
                "preferred-rest mean contrast divided by pooled within-group "
                "sample standard deviation; zero when pooled variance is zero"
            ),
            "class_profile_alignment": (
                "Pearson correlation across equally weighted class-conditional "
                "means between one target soma on the reference split and each "
                "source coordinate on the evaluation split; equivalently cosine "
                "similarity after centering both class profiles. This avoids "
                "weighting classes by their possibly imbalanced sample counts."
            ),
            "class_profile_degeneracy": (
                "a centered target or source class profile with L2 norm at or "
                "below epsilon has alignment defined as zero; each record reports "
                "target, source, and eligible-source degeneracy counts"
            ),
            "source_uniform": (
                "uniform average across source coordinates eligible under the "
                "candidate connection mask"
            ),
            "contact_weighted": (
                "coordinate average weighted by exact active-contact frequency "
                "across branches owned by the target soma"
            ),
            "conductance_weighted": (
                "coordinate average weighted by mean effective non-negative "
                "conductance across branches owned by the target soma"
            ),
            "image_coordinates": (
                "only explicitly declared first-layer sources are labeled pixels; "
                "image_shape is [H,W] or [C,H,W], its product must equal the "
                "actual source dimension, and coordinates remain in model source "
                "order"
            ),
            "hidden_coordinates": (
                "later-layer sources are native source neurons, never pixels or "
                "implicitly composed receptive fields"
            ),
            "stored_state": (
                "class counts, sums, squared sums, and threshold exceedance counts; "
                "derived one-dimensional coordinate profiles; no sample-by-synapse "
                "or sample-by-source arrays"
            ),
        }


def _target_output_hook(
    _module: torch.nn.Module,
    _inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    *,
    key: str,
    stats: Mapping[str, StreamingClassFeatureStatistics],
    current_labels: Mapping[str, torch.Tensor | None],
) -> None:
    labels = current_labels["value"]
    if labels is None:
        raise RuntimeError("target output hook ran without current labels")
    if not isinstance(output, torch.Tensor):
        raise TypeError("feedforward target population must return a tensor")
    stats[key].update(output, labels)


def _source_input_pre_hook(
    _module: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    *,
    blocks: tuple[SourceSupportBlock, ...],
    stats: Mapping[str, StreamingClassFeatureStatistics],
    current_labels: Mapping[str, torch.Tensor | None],
) -> None:
    labels = current_labels["value"]
    if labels is None:
        raise RuntimeError("source input hook ran without current labels")
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        raise TypeError("synaptic source hook requires a tensor positional input")
    source_values = inputs[0]
    if source_values.ndim != 2:
        raise ValueError(
            "feedforward synaptic source must have shape [batch, source], got "
            f"{tuple(source_values.shape)}"
        )
    for block in blocks:
        stats[block.key].update(
            source_values[:, block.source_start : block.source_stop], labels
        )


def _consistent_total_count(
    stats: Iterable[StreamingClassFeatureStatistics],
) -> int:
    counts = {item.total_count for item in stats}
    if len(counts) != 1:
        raise RuntimeError(
            f"hooked streams observed inconsistent sample counts: {counts}"
        )
    return counts.pop()


def _consistent_class_counts(
    stats: Iterable[StreamingClassFeatureStatistics],
) -> dict[int, int]:
    count_records = {tuple(sorted(item.counts.items())) for item in stats}
    if len(count_records) != 1:
        raise RuntimeError("hooked streams observed inconsistent class counts")
    return dict(count_records.pop())


__all__ = [
    "SourceSupportBlock",
    "SourceTuningInventory",
    "SourceTuningOptions",
    "SourceTuningSupportAnalyzer",
    "StreamingClassFeatureStatistics",
    "TargetPopulationRecord",
    "centered_class_profile_alignment",
    "inventory_population_network_source_support",
    "summarize_support_alignment",
]
