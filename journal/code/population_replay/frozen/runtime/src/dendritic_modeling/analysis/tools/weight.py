"""
Unified Weight Analysis Module.

This module provides comprehensive analysis of synaptic weights across the dendritic network,
including both global statistics and hierarchical per-layer analysis with plotting capabilities.
Provides comprehensive weight analysis with both global and hierarchical statistics.
"""

import csv
import logging
import math
import os
import re
import time
from collections.abc import Iterator
from typing import Any, Optional

import numpy as np
import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.analysis.utils.einet_core import has_einet_core
from dendritic_modeling.analysis.utils.recurrent_introspection import (
    iter_recurrent_populations,
)
from dendritic_modeling.analysis.utils.runtime import iter_analysis_batches
from dendritic_modeling.config import EvaluationRuntimeConfig, WeightAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import iter_named_modules_of_type

logger = logging.getLogger(__name__)


_SOURCE_SUPPORT_PATHS = {
    "ff_excitatory": ("branch_excitation", "excitatory"),
    "ff_inhibitory": ("branch_inhibition", "inhibitory"),
    "rec_excitatory": ("branch_recurrent", "excitatory"),
    "rec_inhibitory": ("branch_rec_inhibition", "inhibitory"),
}


def _safe_profile_key(*parts: object) -> str:
    """Return a stable NPZ-compatible key from structural identifiers."""
    return "__".join(
        re.sub(r"[^A-Za-z0-9_.-]+", "-", str(part)).strip("-") for part in parts
    )


def _mean_pairwise_jaccard(mask: torch.Tensor) -> float | None:
    """Mean Jaccard overlap between distinct binary branch supports."""
    mask = mask.to(dtype=torch.float64)
    n_rows = int(mask.shape[0])
    if n_rows < 2:
        return None
    intersection = mask @ mask.T
    row_sizes = mask.sum(dim=1)
    union = row_sizes[:, None] + row_sizes[None, :] - intersection
    valid = ~torch.eye(n_rows, dtype=torch.bool, device=mask.device)
    pairwise = torch.where(
        union > 0,
        intersection / union.clamp_min(torch.finfo(mask.dtype).eps),
        torch.zeros_like(union),
    )
    return float(pairwise[valid].mean().item())


def _source_support_statistics(
    active_mask: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> dict[str, float | None]:
    """Compute dimensionless source-support statistics for one soma."""
    active_mask = active_mask.ne(0)
    candidate_mask = candidate_mask.ne(0)
    source_dim = int(active_mask.shape[-1])
    active_counts = active_mask.sum(dim=0, dtype=torch.float64)
    candidate_union = candidate_mask.any(dim=0)
    active_union = active_counts > 0
    eligible_count = int(candidate_union.sum().item())
    active_source_count = int(active_union.sum().item())
    total_contacts = float(active_counts.sum().item())

    if total_contacts > 0:
        distribution = active_counts / total_contacts
        nonzero = distribution[distribution > 0]
        entropy_bits = float((-(nonzero * torch.log2(nonzero))).sum().item())
    else:
        entropy_bits = 0.0
    max_entropy_bits = math.log2(source_dim) if source_dim > 1 else 0.0
    normalized_entropy = (
        entropy_bits / max_entropy_bits if max_entropy_bits > 0 else 0.0
    )
    effective_source_fraction = (
        (2.0**entropy_bits) / source_dim if source_dim > 0 else 0.0
    )

    return {
        "source_coverage_fraction": active_source_count / max(source_dim, 1),
        "eligible_source_fraction": eligible_count / max(source_dim, 1),
        "source_coverage_of_eligible": active_source_count / max(eligible_count, 1),
        "source_entropy_bits": entropy_bits,
        "normalized_source_entropy": normalized_entropy,
        "effective_source_fraction": effective_source_fraction,
        "mean_pairwise_branch_jaccard": _mean_pairwise_jaccard(active_mask),
    }


def _center_region_mask(
    image_shape: tuple[int, int],
    bounds: tuple[int, int, int, int] | None,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Return a flattened center mask and its half-open image bounds."""
    height, width = image_shape
    if bounds is None:
        center_height = max(1, height // 2)
        center_width = max(1, width // 2)
        r0 = (height - center_height) // 2
        c0 = (width - center_width) // 2
        bounds = (r0, r0 + center_height, c0, c0 + center_width)
    r0, r1, c0, c1 = bounds
    mask = torch.zeros((height, width), dtype=torch.float64)
    mask[r0:r1, c0:c1] = 1.0
    return mask.reshape(-1), bounds


def _safe_alignment_ratio(numerator: float, denominator: float) -> float | None:
    """Return a finite enrichment ratio, leaving zero baselines undefined."""
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _source_activity_alignment_statistics(
    profile: torch.Tensor,
    preferred_on_probability: torch.Tensor,
    rest_on_probability: torch.Tensor,
    availability_mask: torch.Tensor | None = None,
) -> dict[str, float | bool | None]:
    """Align one source profile with preferred- and rest-class pixel activity.

    The profile is normalized to unit mass before alignment. Consequently the
    foreground values are the expected fractions of contacts or conductance on
    pixels satisfying ``x > threshold``. Uniform values are the corresponding
    occupancies over all available pixels and provide the occupancy control.
    Supplying an availability mask computes the same estimands within a region.
    """
    profile = profile.to(dtype=torch.float64).reshape(-1)
    preferred = preferred_on_probability.to(dtype=torch.float64).reshape(-1)
    rest = rest_on_probability.to(dtype=torch.float64).reshape(-1)
    if profile.shape != preferred.shape or profile.shape != rest.shape:
        raise ValueError("source profile and activity probabilities must align")
    if not (
        torch.isfinite(profile).all()
        and torch.isfinite(preferred).all()
        and torch.isfinite(rest).all()
    ):
        raise ValueError("source-region alignment inputs must be finite")
    if bool((profile < 0).any()):
        raise ValueError("source-region alignment requires non-negative profiles")

    if availability_mask is None:
        availability = torch.ones_like(profile)
    else:
        availability = availability_mask.to(dtype=torch.float64).reshape(-1)
        if availability.shape != profile.shape:
            raise ValueError("source profile and availability mask must align")
        if (
            not torch.isfinite(availability).all()
            or bool((availability < 0).any())
            or float(availability.sum().item()) <= 0.0
        ):
            raise ValueError("availability mask must have finite positive mass")
    profile = profile * availability
    available_mass = float(availability.sum().item())
    uniform_preferred_foreground = float(
        torch.dot(preferred, availability).item() / available_mass
    )
    uniform_rest_foreground = float(
        torch.dot(rest, availability).item() / available_mass
    )
    uniform_preferred_background = 1.0 - uniform_preferred_foreground
    uniform_rest_background = 1.0 - uniform_rest_foreground

    total_mass = float(profile.sum().item())
    if total_mass <= 0.0:
        return {
            "profile_has_mass": False,
            "profile_total_mass": total_mass,
            "preferred_foreground_fraction": None,
            "preferred_background_fraction": None,
            "rest_foreground_fraction": None,
            "rest_background_fraction": None,
            "preferred_rest_foreground_contrast": None,
            "preferred_rest_background_contrast": None,
            "uniform_preferred_foreground_occupancy": uniform_preferred_foreground,
            "uniform_preferred_background_occupancy": uniform_preferred_background,
            "uniform_rest_foreground_occupancy": uniform_rest_foreground,
            "uniform_rest_background_occupancy": uniform_rest_background,
            "preferred_foreground_occupancy_excess": None,
            "preferred_background_occupancy_excess": None,
            "rest_foreground_occupancy_excess": None,
            "rest_background_occupancy_excess": None,
            "preferred_foreground_occupancy_enrichment": None,
            "preferred_background_occupancy_enrichment": None,
            "rest_foreground_occupancy_enrichment": None,
            "rest_background_occupancy_enrichment": None,
        }

    normalized = profile / total_mass
    preferred_foreground = float(torch.dot(normalized, preferred).item())
    rest_foreground = float(torch.dot(normalized, rest).item())
    preferred_background = 1.0 - preferred_foreground
    rest_background = 1.0 - rest_foreground
    return {
        "profile_has_mass": True,
        "profile_total_mass": total_mass,
        "preferred_foreground_fraction": preferred_foreground,
        "preferred_background_fraction": preferred_background,
        "rest_foreground_fraction": rest_foreground,
        "rest_background_fraction": rest_background,
        "preferred_rest_foreground_contrast": (preferred_foreground - rest_foreground),
        "preferred_rest_background_contrast": (preferred_background - rest_background),
        "uniform_preferred_foreground_occupancy": uniform_preferred_foreground,
        "uniform_preferred_background_occupancy": uniform_preferred_background,
        "uniform_rest_foreground_occupancy": uniform_rest_foreground,
        "uniform_rest_background_occupancy": uniform_rest_background,
        "preferred_foreground_occupancy_excess": (
            preferred_foreground - uniform_preferred_foreground
        ),
        "preferred_background_occupancy_excess": (
            preferred_background - uniform_preferred_background
        ),
        "rest_foreground_occupancy_excess": (rest_foreground - uniform_rest_foreground),
        "rest_background_occupancy_excess": (rest_background - uniform_rest_background),
        "preferred_foreground_occupancy_enrichment": _safe_alignment_ratio(
            preferred_foreground,
            uniform_preferred_foreground,
        ),
        "preferred_background_occupancy_enrichment": _safe_alignment_ratio(
            preferred_background,
            uniform_preferred_background,
        ),
        "rest_foreground_occupancy_enrichment": _safe_alignment_ratio(
            rest_foreground,
            uniform_rest_foreground,
        ),
        "rest_background_occupancy_enrichment": _safe_alignment_ratio(
            rest_background,
            uniform_rest_background,
        ),
    }


def _spatial_region_alignment_statistics(
    profile: torch.Tensor,
    center_mask: torch.Tensor,
) -> dict[str, float | bool | None]:
    """Return center/surround profile mass normalized by available pixels."""
    profile = profile.to(dtype=torch.float64).reshape(-1)
    center = center_mask.to(dtype=torch.float64).reshape(-1)
    if profile.shape != center.shape:
        raise ValueError("source profile and center mask must align")
    if not torch.isfinite(profile).all() or bool((profile < 0).any()):
        raise ValueError("spatial alignment requires a finite non-negative profile")
    center_pixel_fraction = float(center.mean().item())
    surround_pixel_fraction = 1.0 - center_pixel_fraction
    total_mass = float(profile.sum().item())
    if total_mass <= 0.0:
        return {
            "profile_has_mass": False,
            "profile_total_mass": total_mass,
            "center_support_fraction": None,
            "surround_support_fraction": None,
            "center_pixel_fraction": center_pixel_fraction,
            "surround_pixel_fraction": surround_pixel_fraction,
            "center_support_excess": None,
            "surround_support_excess": None,
            "center_support_enrichment": None,
            "surround_support_enrichment": None,
            "center_surround_enrichment_contrast": None,
        }
    normalized = profile / total_mass
    center_support = float(torch.dot(normalized, center).item())
    surround_support = 1.0 - center_support
    center_enrichment = _safe_alignment_ratio(
        center_support,
        center_pixel_fraction,
    )
    surround_enrichment = _safe_alignment_ratio(
        surround_support,
        surround_pixel_fraction,
    )
    return {
        "profile_has_mass": True,
        "profile_total_mass": total_mass,
        "center_support_fraction": center_support,
        "surround_support_fraction": surround_support,
        "center_pixel_fraction": center_pixel_fraction,
        "surround_pixel_fraction": surround_pixel_fraction,
        "center_support_excess": center_support - center_pixel_fraction,
        "surround_support_excess": surround_support - surround_pixel_fraction,
        "center_support_enrichment": center_enrichment,
        "surround_support_enrichment": surround_enrichment,
        "center_surround_enrichment_contrast": (
            None
            if center_enrichment is None or surround_enrichment is None
            else center_enrichment - surround_enrichment
        ),
    }


def _weight_tensor_statistics(
    weights: torch.Tensor,
    *,
    compute_mean: bool,
    compute_variance: bool,
    compute_min_max: bool,
    compute_percentiles: bool,
) -> dict[str, float]:
    """Compute configured weight statistics with the historical torch reductions."""
    stats = {}

    if compute_mean:
        stats["mean"] = weights.mean().item()

    if compute_variance:
        stats["variance"] = weights.var().item()
        stats["std"] = weights.std().item()

    if compute_min_max:
        stats["min"] = weights.min().item()
        stats["max"] = weights.max().item()

    if compute_percentiles:
        stats["median"] = weights.median().item()
        stats["q1"] = weights.quantile(0.25).item()
        stats["q3"] = weights.quantile(0.75).item()

    return stats


def _branch_synapse_statistics(
    prefix: str,
    active_weights: torch.Tensor,
    *,
    compute_mean: bool,
    compute_variance: bool,
    compute_min_max: bool,
    compute_percentiles: bool,
) -> dict[str, float | int]:
    """Compute active-synapse branch statistics for one synapse type."""
    stats: dict[str, float | int] = {f"{prefix}_n_active_synapses": len(active_weights)}
    if len(active_weights) == 0:
        return stats

    if compute_mean:
        stats[f"{prefix}_weight_mean"] = active_weights.mean().item()
    if compute_variance:
        stats[f"{prefix}_weight_var"] = active_weights.var().item()
    if compute_min_max:
        stats[f"{prefix}_weight_min"] = active_weights.min().item()
        stats[f"{prefix}_weight_max"] = active_weights.max().item()
    if compute_percentiles:
        stats[f"{prefix}_weight_median"] = active_weights.median().item()

    return stats


def _iter_dendritic_populations(
    model: BaseModel,
    *,
    include_inhibitory: bool = True,
):
    """Yield unique population modules that expose dendritic branch layers."""
    core = getattr(model, "core_network", model)
    seen: set[int] = set()

    for record in iter_recurrent_populations(core):
        if not include_inhibitory and record.polarity == "inhibitory":
            continue
        population = record.population
        if not hasattr(population, "branch_layers"):
            continue
        ident = id(population)
        if ident in seen:
            continue
        seen.add(ident)
        yield population

    for layer in getattr(core, "layers", []) or []:
        for attr, polarity in (
            ("excitatory_cells", "excitatory"),
            ("inhibitory_cells", "inhibitory"),
            ("e_population", "excitatory"),
            ("i_population", "inhibitory"),
        ):
            if not include_inhibitory and polarity == "inhibitory":
                continue
            population = getattr(layer, attr, None)
            if not hasattr(population, "branch_layers"):
                continue
            ident = id(population)
            if ident in seen:
                continue
            seen.add(ident)
            yield population


class WeightAnalyzer(AbstractAnalyzer):
    """
    Unified analyzer for comprehensive synaptic weight analysis.

    Provides both global weight statistics and hierarchical per-layer analysis
    with enhanced plotting capabilities including min/max and dendritic strength plots.
    """

    def __init__(self, params: WeightAnalysisParams):
        """Initialize the unified weight analyzer."""
        super().__init__("WeightAnalyzer")

        self.synapse_weights = getattr(params, "synapse_weights", True)
        self.branch_weights = getattr(params, "branch_weights", True)
        self.computation_level = getattr(params, "computation_level", "single_branch")
        self.branch_aggregation = getattr(params, "branch_aggregation", "mean")

        # Enhanced analysis modes
        self.per_layer_analysis = getattr(params, "per_layer_analysis", True)
        self.per_einet_analysis = getattr(params, "per_einet_analysis", True)
        self.per_neuron_analysis = getattr(params, "per_neuron_analysis", False)

        self.analyze_excitatory = getattr(params, "analyze_excitatory", True)
        self.analyze_inhibitory = getattr(params, "analyze_inhibitory", True)
        self.analyze_branch_output = getattr(params, "analyze_branch_output", True)

        self.compute_mean = getattr(params, "compute_mean", True)
        self.compute_variance = getattr(params, "compute_variance", True)
        self.compute_percentiles = getattr(params, "compute_percentiles", True)
        self.compute_min_max = getattr(params, "compute_min_max", True)
        self.compute_dendritic_strength = getattr(
            params, "compute_dendritic_strength", True
        )

        self.weight_threshold = getattr(params, "weight_threshold", 1e-6)
        self.compute_source_support = bool(
            getattr(params, "compute_source_support", False)
        )
        self.source_support_target_polarities = tuple(
            str(value).lower()
            for value in getattr(
                params,
                "source_support_target_polarities",
                ["excitatory"],
            )
        )
        self.source_support_pathways = tuple(
            str(value).lower()
            for value in getattr(
                params,
                "source_support_pathways",
                ["ff_excitatory", "ff_inhibitory"],
            )
        )
        image_shape = getattr(params, "source_support_image_shape", None)
        self.source_support_image_shape = (
            None if image_shape is None else tuple(int(value) for value in image_shape)
        )
        self.source_support_plot_soma_indices = tuple(
            int(value)
            for value in getattr(params, "source_support_plot_soma_indices", [])
        )
        self.source_support_fig_save_format = str(
            getattr(params, "source_support_fig_save_format", "pdf")
        ).lstrip(".")
        self.compute_source_region_alignment = bool(
            getattr(params, "compute_source_region_alignment", False)
        )
        self.source_region_soma_to_class = tuple(
            int(value) for value in getattr(params, "source_region_soma_to_class", [])
        )
        self.source_region_reference_split = str(
            getattr(params, "source_region_reference_split", "validation")
        ).lower()
        self.source_region_image_network_layers = tuple(
            int(value)
            for value in getattr(params, "source_region_image_network_layers", [0])
        )
        self.source_region_foreground_thresholds = tuple(
            float(value)
            for value in getattr(
                params,
                "source_region_foreground_thresholds",
                [0.0, 0.5],
            )
        )
        center_bounds = getattr(params, "source_region_center_bounds", None)
        self.source_region_center_bounds = (
            None
            if center_bounds is None
            else tuple(int(value) for value in center_bounds)
        )
        max_reference_samples = getattr(
            params,
            "source_region_max_reference_samples",
            None,
        )
        self.source_region_max_reference_samples = (
            None if max_reference_samples is None else int(max_reference_samples)
        )

        self.logger.info(
            f"Initialized unified WeightAnalyzer with computation_level: {self.computation_level}"
        )

    def analyze(
        self,
        model: BaseModel,
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
        reference_dataset: torch.utils.data.Dataset | None = None,
        reference_split: str | None = None,
        runtime: EvaluationRuntimeConfig | None = None,
    ):
        """
        Perform comprehensive weight analysis including global and hierarchical statistics.

        Args:
            model: The model to analyze (must have ExcitationInhibitionNetwork)
            save_path: Path to save results (optional)
            filename: Filename for saved results
            training: If True, only save data (no plots). If False, generate plots.
            reference_dataset: Configured reference split used only for
                source-region alignment.
            reference_split: Provenance label for the configured reference split.
            runtime: Shared deterministic analysis iteration policy.

        Returns:
            Dictionary containing all weight analysis results
        """
        if not has_einet_core(model):
            return None

        self.logger.info("Starting comprehensive weight analysis...")
        self.logger.info("Weight analysis configuration:")
        self.logger.info(f"  per_layer_analysis: {self.per_layer_analysis}")
        self.logger.info(f"  compute_min_max: {self.compute_min_max}")
        self.logger.info(
            f"  compute_dendritic_strength: {self.compute_dendritic_strength}"
        )
        self.logger.info(f"  save_path: {save_path}")
        start_time = time.time()

        results = {}

        if self.synapse_weights or self.branch_weights:
            global_stats = self._compute_global_statistics(model)
            results["global_statistics"] = global_stats

        if self.per_layer_analysis:
            layer_stats = self._compute_layer_statistics(model)

            # Inject synthetic soma ONLY if somatic_synapses=False and there is no layer_idx=0
            somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
            if not somatic_synapses and layer_stats:
                # Check if synthetic soma is needed
                if "branch_layers.0" not in layer_stats:
                    layer_stats["branch_layers.0"] = {}  # Synthetic soma
                    self.logger.info(
                        "Added synthetic soma layer (branch_layers.0) for somatic_synapses=False"
                    )
                # Dynamically determine expected layers from network structure
                # Get the actual number of dendritic layers from the first EI layer
                expected_max_idx = 0
                for population in _iter_dendritic_populations(model):
                    expected_max_idx = max(
                        expected_max_idx, len(population.branch_layers)
                    )

                # Remove any layers beyond the expected range
                for key in list(layer_stats.keys()):
                    if "branch_layers." in key:
                        idx = int(key.split(".")[-1])
                        if idx > expected_max_idx:
                            del layer_stats[key]  # Remove extra
                            self.logger.info(
                                f"Removed extra layer {key} (idx={idx} > expected_max_idx={expected_max_idx})"
                            )

            results["layer_statistics"] = layer_stats
            self.logger.info(
                f"Generated layer_statistics with {len(layer_stats)} layers: {list(layer_stats.keys())}"
            )

            # Also store raw per-branch statistics for compatibility with the original analyzer
            branch_level_stats = self._compute_branch_level_statistics(model)
            if branch_level_stats:
                results["branch_level_statistics"] = branch_level_stats

        if self.compute_min_max or self.compute_dendritic_strength:
            enhanced_stats = self._compute_enhanced_layer_statistics(model)

            # Inject synthetic Soma for enhanced statistics as well
            somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
            if not somatic_synapses and enhanced_stats:
                if "branch_layers.0" not in enhanced_stats:
                    enhanced_stats = {"branch_layers.0": {}} | enhanced_stats

            results["enhanced_layer_statistics"] = enhanced_stats

        source_support_profiles: dict[str, np.ndarray] = {}
        source_support_profile_metadata: list[dict[str, Any]] = []
        if self.compute_source_support:
            (
                source_support,
                source_support_profiles,
                source_support_profile_metadata,
            ) = self._compute_source_support(model)
            results["source_support"] = source_support
            if self.compute_source_region_alignment:
                if reference_dataset is None:
                    raise ValueError(
                        "source-region alignment requires the configured reference "
                        "dataset"
                    )
                source_region_alignment = self._compute_source_region_alignment(
                    profiles=source_support_profiles,
                    profile_metadata=source_support_profile_metadata,
                    reference_dataset=reference_dataset,
                    reference_split=(
                        self.source_region_reference_split
                        if reference_split is None
                        else str(reference_split)
                    ),
                    runtime=runtime,
                )
                results["source_region_alignment"] = source_region_alignment

        elapsed_time = time.time() - start_time
        self.logger.info(f"Weight analysis completed in {elapsed_time:.2f} seconds")

        # Save results and generate plots
        if save_path is not None:
            self.logger.info(
                f"Saving weight analysis results to {save_path}/{filename}"
            )
            if self.compute_source_support:
                self._save_source_support_outputs(
                    results["source_support"],
                    source_support_profiles,
                    source_support_profile_metadata,
                    save_path=save_path,
                    filename=filename,
                    plot=not training,
                )
                if self.compute_source_region_alignment:
                    self._save_source_region_alignment_outputs(
                        results["source_region_alignment"],
                        save_path=save_path,
                        filename=filename,
                    )
            save_dict(results, save_path, filename)

            # Only generate plots during final analysis (not during training)
            if not training:
                self.logger.info("Starting plot generation...")
                self._generate_all_plots(results, save_path, model)
                self.logger.info("Plot generation completed!")
            else:
                self.logger.info("Training mode: plots skipped, only data saved")

        return results

    def _compute_source_support(
        self,
        model: BaseModel,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
        """Aggregate realized sparse supports without conflating them with weights."""
        core = getattr(model, "core_network", model)
        rows: list[dict[str, Any]] = []
        profiles: dict[str, np.ndarray] = {}
        profile_metadata: list[dict[str, Any]] = []
        pooled: dict[
            tuple[int, str, str, str],
            dict[str, list[list[torch.Tensor]] | int],
        ] = {}

        for population_record in iter_recurrent_populations(core):
            target_polarity = str(population_record.polarity).lower()
            if target_polarity not in self.source_support_target_polarities:
                continue
            population = population_record.population
            branch_layers = list(getattr(population, "branch_layers", []) or [])
            n_soma = int(population_record.n_neurons)
            if n_soma < 1 or not branch_layers:
                continue

            network_layer_index = (
                -1
                if population_record.layer_index is None
                else int(population_record.layer_index)
            )
            population_name = str(
                population_record.population_name or population_record.key
            )

            for fallback_depth, branch_layer in enumerate(branch_layers):
                depth = int(getattr(branch_layer, "layer_idx", fallback_depth))
                for pathway in self.source_support_pathways:
                    module_attr, source_polarity = _SOURCE_SUPPORT_PATHS[pathway]
                    synapse = getattr(branch_layer, module_attr, None)
                    if synapse is None:
                        continue
                    snapshot = effective_synapse_snapshot(synapse)
                    active = snapshot.active_mask.detach().cpu()
                    candidate = snapshot.candidate_mask.detach().cpu()
                    effective = snapshot.effective_weight.detach().cpu()
                    if active.ndim != 2 or active.shape[0] % n_soma != 0:
                        raise ValueError(
                            "Source-support analysis requires branch rows grouped "
                            f"evenly by soma; got {tuple(active.shape)} for "
                            f"{population_record.key} with {n_soma} somas"
                        )
                    branches_per_soma = int(active.shape[0] // n_soma)
                    source_dim = int(active.shape[1])
                    active_by_soma = active.reshape(
                        n_soma, branches_per_soma, source_dim
                    )
                    candidate_by_soma = candidate.reshape(
                        n_soma, branches_per_soma, source_dim
                    )
                    effective_by_soma = effective.reshape(
                        n_soma, branches_per_soma, source_dim
                    )

                    metadata = {
                        "network_layer_index": network_layer_index,
                        "population_name": population_name,
                        "target_polarity": target_polarity,
                        "dendritic_depth": depth,
                        "depth_label": str(depth),
                        "pathway": pathway,
                        "source_polarity": source_polarity,
                        "n_soma": n_soma,
                        "branches_per_soma": branches_per_soma,
                        "source_dim": source_dim,
                        "selection_policy": snapshot.selection_policy,
                        "mask_source": snapshot.mask_source,
                    }
                    self._record_source_support_block(
                        metadata=metadata,
                        active_by_soma=active_by_soma,
                        candidate_by_soma=candidate_by_soma,
                        effective_by_soma=effective_by_soma,
                        rows=rows,
                        profiles=profiles,
                        profile_metadata=profile_metadata,
                    )

                    pool_key = (
                        network_layer_index,
                        population_name,
                        target_polarity,
                        pathway,
                    )
                    pool = pooled.setdefault(
                        pool_key,
                        {
                            "active": [[] for _ in range(n_soma)],
                            "candidate": [[] for _ in range(n_soma)],
                            "effective": [[] for _ in range(n_soma)],
                            "source_dim": source_dim,
                            "source_polarity": source_polarity,
                            "selection_policy": snapshot.selection_policy,
                            "mask_source": snapshot.mask_source,
                        },
                    )
                    if int(pool["source_dim"]) != source_dim:
                        raise ValueError(
                            "Cannot aggregate source support across dendritic "
                            "depths with different source dimensions"
                        )
                    for soma_idx in range(n_soma):
                        pool["active"][soma_idx].append(active_by_soma[soma_idx])
                        pool["candidate"][soma_idx].append(candidate_by_soma[soma_idx])
                        pool["effective"][soma_idx].append(effective_by_soma[soma_idx])

        for (
            network_layer_index,
            population_name,
            target_polarity,
            pathway,
        ), pool in pooled.items():
            active_lists = pool["active"]
            n_soma = len(active_lists)
            active_by_soma = [
                torch.cat(active_lists[soma_idx], dim=0) for soma_idx in range(n_soma)
            ]
            candidate_by_soma = [
                torch.cat(pool["candidate"][soma_idx], dim=0)
                for soma_idx in range(n_soma)
            ]
            effective_by_soma = [
                torch.cat(pool["effective"][soma_idx], dim=0)
                for soma_idx in range(n_soma)
            ]
            branches_per_soma = int(active_by_soma[0].shape[0])
            metadata = {
                "network_layer_index": network_layer_index,
                "population_name": population_name,
                "target_polarity": target_polarity,
                "dendritic_depth": -1,
                "depth_label": "all",
                "pathway": pathway,
                "source_polarity": str(pool["source_polarity"]),
                "n_soma": n_soma,
                "branches_per_soma": branches_per_soma,
                "source_dim": int(pool["source_dim"]),
                "selection_policy": str(pool["selection_policy"]),
                "mask_source": str(pool["mask_source"]),
            }
            self._record_source_support_block(
                metadata=metadata,
                active_by_soma=torch.stack(active_by_soma, dim=0),
                candidate_by_soma=torch.stack(candidate_by_soma, dim=0),
                effective_by_soma=torch.stack(effective_by_soma, dim=0),
                rows=rows,
                profiles=profiles,
                profile_metadata=profile_metadata,
            )

        summary: dict[str, Any] = {
            "quantity_contract": {
                "contact_frequency": (
                    "fraction of branches owned by a soma whose exact realized "
                    "sparse mask selects each source coordinate"
                ),
                "mean_effective_conductance": (
                    "effective positive synaptic conductance averaged over all "
                    "branches owned by the soma, with unselected contacts equal to zero"
                ),
                "dendritic_depth_minus_one": (
                    "all dendritic depths pooled by concatenating their branch rows"
                ),
                "not_activity_or_information": True,
            },
            "n_scalar_records": len(rows),
            "n_profile_arrays": len(profiles),
            "records": rows,
            "profile_metadata": profile_metadata,
            "profiles_file": None,
            "records_file": None,
        }
        return summary, profiles, profile_metadata

    def _compute_source_region_alignment(
        self,
        *,
        profiles: dict[str, np.ndarray],
        profile_metadata: list[dict[str, Any]],
        reference_dataset: torch.utils.data.Dataset,
        reference_split: str,
        runtime: EvaluationRuntimeConfig | None,
    ) -> dict[str, Any]:
        """Align image-space source profiles to reference-set source regions."""
        image_shape = self.source_support_image_shape
        if image_shape is None:  # guarded by configuration validation
            raise ValueError("source-region alignment requires an image shape")
        image_size = int(np.prod(image_shape))
        image_metadata: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for item in profile_metadata:
            network_layer_index = int(item["network_layer_index"])
            skip_reason: str | None = None
            if network_layer_index not in self.source_region_image_network_layers:
                skip_reason = "network_layer_not_declared_image_source"
            elif int(item["source_dim"]) != image_size:
                skip_reason = "source_dimension_not_image_space"
            if skip_reason is not None:
                skipped.append(
                    {
                        **{
                            key: value
                            for key, value in item.items()
                            if not key.endswith("_key")
                        },
                        "expected_image_source_dim": image_size,
                        "declared_image_network_layers": list(
                            self.source_region_image_network_layers
                        ),
                        "skip_reason": skip_reason,
                    }
                )
                continue
            image_metadata.append(item)

        quantity_contract = {
            "foreground_definition": "reference example pixel x_p > threshold",
            "input_space": (
                "x is the model input after configured dataset preprocessing; "
                "no inverse normalization is applied"
            ),
            "preferred_probability": (
                "q_pref(p) = P_ref[x_p > threshold | Y = soma_class]"
            ),
            "rest_probability": (
                "q_rest(p) = P_ref[x_p > threshold | Y != soma_class]"
            ),
            "profile_alignment": (
                "A = sum_p normalized_source_profile(p) q(p); reported "
                "separately for exact contact frequency and mean effective "
                "conductance"
            ),
            "preferred_rest_contrast": "T = A_pref - A_rest",
            "occupancy_control": (
                "image-wide mean q; enrichment is aligned fraction divided by "
                "this uniform-source occupancy"
            ),
            "center_surround_control": (
                "source-profile mass in the configured center/surround divided "
                "by the corresponding fraction of available image pixels"
            ),
            "within_region_activity_alignment": (
                "center and surround repeat A_pref, A_rest, and T after masking "
                "and renormalizing the source profile within that region"
            ),
            "reference_split": reference_split,
            "declared_image_network_layers": list(
                self.source_region_image_network_layers
            ),
            "structural_not_activity_or_information": True,
        }
        if not image_metadata:
            return {
                "quantity_contract": quantity_contract,
                "reference": {
                    "split": reference_split,
                    "status": "not_evaluated_no_image_space_sources",
                    "image_shape": list(image_shape),
                },
                "n_source_activity_records": 0,
                "n_spatial_region_records": 0,
                "n_skipped_profile_blocks": len(skipped),
                "source_activity_records": [],
                "spatial_region_records": [],
                "skipped_profile_blocks": skipped,
                "source_activity_records_file": None,
                "spatial_region_records_file": None,
                "skipped_profile_blocks_file": None,
            }

        mapping = self.source_region_soma_to_class
        for item in image_metadata:
            n_soma = int(item["n_soma"])
            if len(mapping) != n_soma:
                raise ValueError(
                    "source_region_soma_to_class must contain exactly one class "
                    f"per soma; got {len(mapping)} entries for {n_soma} somas in "
                    f"population {item['population_name']!r}"
                )

        reference = self._estimate_source_activity_reference(
            reference_dataset=reference_dataset,
            image_size=image_size,
            required_classes=sorted(set(mapping)),
            runtime=runtime,
        )
        center_mask, center_bounds = _center_region_mask(
            image_shape,
            self.source_region_center_bounds,
        )
        source_activity_rows: list[dict[str, Any]] = []
        spatial_rows: list[dict[str, Any]] = []
        quantities = (
            ("contact_frequency", "contact_frequency_key"),
            ("mean_effective_conductance", "mean_effective_conductance_key"),
        )
        for item in image_metadata:
            structural = {
                key: value for key, value in item.items() if not key.endswith("_key")
            }
            n_soma = int(item["n_soma"])
            for quantity, key_field in quantities:
                profile_key = str(item[key_field])
                values = torch.from_numpy(np.asarray(profiles[profile_key]))
                if tuple(values.shape) != (n_soma, image_size):
                    raise ValueError(
                        f"source profile {profile_key!r} has shape "
                        f"{tuple(values.shape)}; expected {(n_soma, image_size)}"
                    )
                for soma_index, soma_class in enumerate(mapping):
                    common = {
                        **structural,
                        "soma_index": soma_index,
                        "soma_class": soma_class,
                        "quantity": quantity,
                        "source_profile_key": profile_key,
                        "reference_split": reference_split,
                        "reference_sample_count": reference["n_samples"],
                        "preferred_class_sample_count": reference["class_counts"][
                            soma_class
                        ],
                        "rest_class_sample_count": (
                            reference["n_samples"]
                            - reference["class_counts"][soma_class]
                        ),
                    }
                    profile = values[soma_index]
                    spatial_rows.append(
                        {
                            **common,
                            "center_row_start": center_bounds[0],
                            "center_row_stop": center_bounds[1],
                            "center_col_start": center_bounds[2],
                            "center_col_stop": center_bounds[3],
                            **_spatial_region_alignment_statistics(
                                profile,
                                center_mask,
                            ),
                        }
                    )
                    for threshold in self.source_region_foreground_thresholds:
                        threshold_reference = reference["thresholds"][threshold]
                        preferred = threshold_reference["preferred"][soma_class]
                        rest = threshold_reference["rest"][soma_class]
                        full_stats = _source_activity_alignment_statistics(
                            profile,
                            preferred,
                            rest,
                        )
                        center_stats = {
                            f"center_{key}": value
                            for key, value in _source_activity_alignment_statistics(
                                profile,
                                preferred,
                                rest,
                                center_mask,
                            ).items()
                        }
                        surround_stats = {
                            f"surround_{key}": value
                            for key, value in _source_activity_alignment_statistics(
                                profile,
                                preferred,
                                rest,
                                1.0 - center_mask,
                            ).items()
                        }
                        source_activity_rows.append(
                            {
                                **common,
                                "foreground_threshold": threshold,
                                **full_stats,
                                **center_stats,
                                **surround_stats,
                            }
                        )

        return {
            "quantity_contract": quantity_contract,
            "reference": {
                "split": reference_split,
                "status": "evaluated",
                "image_shape": list(image_shape),
                "n_samples": reference["n_samples"],
                "class_counts": {
                    str(key): value
                    for key, value in sorted(reference["class_counts"].items())
                },
                "foreground_thresholds": list(self.source_region_foreground_thresholds),
                "center_bounds": list(center_bounds),
            },
            "n_source_activity_records": len(source_activity_rows),
            "n_spatial_region_records": len(spatial_rows),
            "n_skipped_profile_blocks": len(skipped),
            "source_activity_records": source_activity_rows,
            "spatial_region_records": spatial_rows,
            "skipped_profile_blocks": skipped,
            "source_activity_records_file": None,
            "spatial_region_records_file": None,
            "skipped_profile_blocks_file": None,
        }

    def _estimate_source_activity_reference(
        self,
        *,
        reference_dataset: torch.utils.data.Dataset,
        image_size: int,
        required_classes: list[int],
        runtime: EvaluationRuntimeConfig | None,
    ) -> dict[str, Any]:
        """Estimate per-class pixel-on probabilities without storing images."""
        class_counts = dict.fromkeys(required_classes, 0)
        total_count = 0
        total_on = {
            threshold: torch.zeros(image_size, dtype=torch.float64)
            for threshold in self.source_region_foreground_thresholds
        }
        class_on = {
            threshold: {
                class_label: torch.zeros(image_size, dtype=torch.float64)
                for class_label in required_classes
            }
            for threshold in self.source_region_foreground_thresholds
        }
        for batch in iter_analysis_batches(
            reference_dataset,
            runtime,
            self.source_region_max_reference_samples,
            device="cpu",
        ):
            if len(batch) < 2:
                raise ValueError(
                    "source-region alignment requires (input, class label) data"
                )
            labels = torch.as_tensor(batch[1]).reshape(-1).cpu()
            if labels.numel() < 1:
                continue
            if labels.is_floating_point() and not torch.equal(
                labels,
                labels.round(),
            ):
                raise ValueError("source-region class labels must be integral")
            labels = labels.to(dtype=torch.long)
            inputs = torch.as_tensor(batch[0]).detach().cpu()
            if int(inputs.shape[0]) != int(labels.shape[0]):
                raise ValueError("source-region input and label counts do not match")
            flattened = inputs.reshape(labels.shape[0], -1).to(dtype=torch.float64)
            if int(flattened.shape[1]) != image_size:
                raise ValueError(
                    "source-region reference inputs do not match "
                    f"source_support_image_shape: got {flattened.shape[1]} "
                    f"features, expected {image_size}"
                )
            if not torch.isfinite(flattened).all():
                raise ValueError(
                    "source-region reference inputs contain non-finite values"
                )
            total_count += int(labels.numel())
            for threshold in self.source_region_foreground_thresholds:
                is_on = flattened > threshold
                total_on[threshold] += is_on.sum(dim=0, dtype=torch.float64)
                for class_label in required_classes:
                    class_mask = labels == class_label
                    if bool(class_mask.any()):
                        class_on[threshold][class_label] += is_on[class_mask].sum(
                            dim=0,
                            dtype=torch.float64,
                        )
            for class_label in required_classes:
                class_counts[class_label] += int((labels == class_label).sum().item())

        if total_count < 1:
            raise ValueError("source-region reference dataset is empty")
        thresholds: dict[float, dict[str, dict[int, torch.Tensor]]] = {}
        for threshold in self.source_region_foreground_thresholds:
            preferred: dict[int, torch.Tensor] = {}
            rest: dict[int, torch.Tensor] = {}
            for class_label in required_classes:
                preferred_count = class_counts[class_label]
                rest_count = total_count - preferred_count
                if preferred_count < 1 or rest_count < 1:
                    raise ValueError(
                        "source-region reference requires both preferred and rest "
                        f"examples for class {class_label}; counts are "
                        f"{preferred_count} and {rest_count}"
                    )
                preferred[class_label] = (
                    class_on[threshold][class_label] / preferred_count
                )
                rest[class_label] = (
                    total_on[threshold] - class_on[threshold][class_label]
                ) / rest_count
            thresholds[threshold] = {
                "preferred": preferred,
                "rest": rest,
            }
        return {
            "n_samples": total_count,
            "class_counts": class_counts,
            "thresholds": thresholds,
        }

    @staticmethod
    def _record_source_support_block(
        *,
        metadata: dict[str, Any],
        active_by_soma: torch.Tensor,
        candidate_by_soma: torch.Tensor,
        effective_by_soma: torch.Tensor,
        rows: list[dict[str, Any]],
        profiles: dict[str, np.ndarray],
        profile_metadata: list[dict[str, Any]],
    ) -> None:
        n_soma = int(metadata["n_soma"])
        contact_frequency = active_by_soma.ne(0).to(torch.float64).mean(dim=1)
        mean_effective = effective_by_soma.to(torch.float64).mean(dim=1)
        prefix = _safe_profile_key(
            f"network-layer-{metadata['network_layer_index']}",
            f"population-{metadata['population_name']}",
            f"target-{metadata['target_polarity']}",
            f"depth-{metadata['depth_label']}",
            f"pathway-{metadata['pathway']}",
        )
        contact_key = f"{prefix}__contact-frequency"
        effective_key = f"{prefix}__mean-effective-conductance"
        profiles[contact_key] = contact_frequency.numpy()
        profiles[effective_key] = mean_effective.numpy()
        profile_metadata.append(
            {
                **metadata,
                "contact_frequency_key": contact_key,
                "mean_effective_conductance_key": effective_key,
            }
        )

        for soma_idx in range(n_soma):
            active = active_by_soma[soma_idx]
            candidate = candidate_by_soma[soma_idx]
            effective = effective_by_soma[soma_idx]
            active_values = effective[active.ne(0)]
            rows.append(
                {
                    **metadata,
                    "soma_index": soma_idx,
                    "realized_contacts_per_branch_mean": float(
                        active.ne(0).sum(dim=1).to(torch.float64).mean().item()
                    ),
                    "active_effective_conductance_mean": (
                        float(active_values.to(torch.float64).mean().item())
                        if active_values.numel()
                        else 0.0
                    ),
                    **_source_support_statistics(active, candidate),
                }
            )

    def _save_source_support_outputs(
        self,
        summary: dict[str, Any],
        profiles: dict[str, np.ndarray],
        profile_metadata: list[dict[str, Any]],
        *,
        save_path: str,
        filename: str,
        plot: bool,
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        profiles_name = f"{filename}_source_support_profiles.npz"
        records_name = f"{filename}_source_support_records.csv"
        summary["profiles_file"] = profiles_name
        summary["records_file"] = records_name
        np.savez_compressed(os.path.join(save_path, profiles_name), **profiles)

        records = summary["records"]
        if records:
            with open(
                os.path.join(save_path, records_name),
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)

        if plot and self.source_support_image_shape is not None:
            figure_files = self._plot_source_support_image_grids(
                profiles,
                profile_metadata,
                save_path=save_path,
                filename=filename,
            )
            if figure_files:
                summary["figure_files"] = figure_files

    @staticmethod
    def _save_source_region_alignment_outputs(
        summary: dict[str, Any],
        *,
        save_path: str,
        filename: str,
    ) -> None:
        """Save tidy source-activity, spatial-control, and skip records."""
        os.makedirs(save_path, exist_ok=True)
        outputs = (
            (
                "source_activity_records",
                "source_activity_records_file",
                f"{filename}_source_activity_alignment_records.csv",
            ),
            (
                "spatial_region_records",
                "spatial_region_records_file",
                f"{filename}_spatial_region_alignment_records.csv",
            ),
            (
                "skipped_profile_blocks",
                "skipped_profile_blocks_file",
                f"{filename}_source_region_skipped_blocks.csv",
            ),
        )
        for records_key, filename_key, output_name in outputs:
            records = summary[records_key]
            if not records:
                continue
            fields = sorted({str(key) for record in records for key in record})
            with open(
                os.path.join(save_path, output_name),
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(records)
            summary[filename_key] = output_name

    def _plot_source_support_image_grids(
        self,
        profiles: dict[str, np.ndarray],
        profile_metadata: list[dict[str, Any]],
        *,
        save_path: str,
        filename: str,
    ) -> list[str]:
        """Plot all-depth image-space support; scientific figures may restyle it."""
        import matplotlib.pyplot as plt

        image_shape = self.source_support_image_shape
        assert image_shape is not None
        image_size = int(np.prod(image_shape))
        selected = [
            item
            for item in profile_metadata
            if item["depth_label"] == "all" and int(item["source_dim"]) == image_size
        ]
        groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for item in selected:
            key = (int(item["network_layer_index"]), str(item["population_name"]))
            groups.setdefault(key, []).append(item)

        figure_files: list[str] = []
        for (network_layer_index, population_name), items in sorted(groups.items()):
            by_pathway = {str(item["pathway"]): item for item in items}
            pathways = [
                pathway
                for pathway in self.source_support_pathways
                if pathway in by_pathway
            ]
            if not pathways:
                continue
            n_soma = int(items[0]["n_soma"])
            soma_indices = (
                list(self.source_support_plot_soma_indices)
                if self.source_support_plot_soma_indices
                else list(range(n_soma))
            )
            soma_indices = [index for index in soma_indices if index < n_soma]
            if not soma_indices:
                continue

            for quantity, key_field, label in (
                (
                    "contact_frequency",
                    "contact_frequency_key",
                    "Contact frequency",
                ),
                (
                    "mean_effective_conductance",
                    "mean_effective_conductance_key",
                    "Mean effective conductance",
                ),
            ):
                fig, axes = plt.subplots(
                    len(pathways),
                    len(soma_indices),
                    figsize=(
                        max(3.0, 0.75 * len(soma_indices)),
                        max(1.5, 0.85 * len(pathways)),
                    ),
                    squeeze=False,
                    constrained_layout=True,
                )
                for row_idx, pathway in enumerate(pathways):
                    metadata = by_pathway[pathway]
                    values = profiles[str(metadata[key_field])]
                    vmax = float(np.nanmax(values))
                    for col_idx, soma_idx in enumerate(soma_indices):
                        ax = axes[row_idx, col_idx]
                        ax.imshow(
                            values[soma_idx].reshape(image_shape),
                            cmap="magma",
                            vmin=0.0,
                            vmax=max(vmax, np.finfo(float).eps),
                            interpolation="nearest",
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if row_idx == 0:
                            ax.set_title(f"soma {soma_idx}", fontsize=7)
                        if col_idx == 0:
                            ax.set_ylabel(pathway.replace("_", " "), fontsize=7)
                fig.suptitle(label, fontsize=9)
                stem = _safe_profile_key(
                    filename,
                    "source-support",
                    f"network-layer-{network_layer_index}",
                    f"population-{population_name}",
                    quantity,
                )
                figure_name = f"{stem}.{self.source_support_fig_save_format}"
                fig.savefig(
                    os.path.join(save_path, figure_name),
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)
                figure_files.append(figure_name)
        return figure_files

    def _compute_global_statistics(self, model: BaseModel) -> dict[str, Any]:
        """Compute global weight statistics across the entire network."""
        self.logger.info("Computing global weight statistics...")

        # Collect weights from all dendritic branch layers
        exc_weights = []
        inh_weights = []
        branch_weights = []

        for _name, module in iter_named_modules_of_type(model, DendriticBranchLayer):
            # Collect excitatory synapse weights
            if (
                self.synapse_weights
                and self.analyze_excitatory
                and module.branch_excitation is not None
            ):
                exc_w = self._active_weights(
                    module.branch_excitation.pruned_weight().flatten()
                ).cpu()
                exc_weights.append(exc_w)

            # Collect inhibitory synapse weights
            if (
                self.synapse_weights
                and self.analyze_inhibitory
                and module.branch_inhibition is not None
            ):
                inh_w = self._active_weights(
                    module.branch_inhibition.pruned_weight().flatten()
                ).cpu()
                inh_weights.append(inh_w)

            # Collect branch-to-output weights
            if (
                self.branch_weights
                and self.analyze_branch_output
                and hasattr(module, "branches_to_output")
            ):
                branch_w = module.branches_to_output.weight().cpu().flatten()
                branch_weights.append(branch_w)

        global_stats = {}

        # Compute statistics for excitatory weights
        if len(exc_weights) > 0:
            exc_weights = torch.cat(exc_weights, dim=0)
            global_stats["excitatory_weights"] = self._compute_weight_statistics(
                exc_weights
            )

        # Compute statistics for inhibitory weights
        if len(inh_weights) > 0:
            inh_weights = torch.cat(inh_weights, dim=0)
            global_stats["inhibitory_weights"] = self._compute_weight_statistics(
                inh_weights
            )

        # Compute statistics for branch weights
        if len(branch_weights) > 0:
            branch_weights = torch.cat(branch_weights, dim=0)
            global_stats["branch_weights"] = self._compute_weight_statistics(
                branch_weights
            )

        return global_stats

    def _compute_weight_statistics(self, weights: torch.Tensor) -> dict[str, float]:
        """Compute comprehensive statistics for a weight tensor."""
        return _weight_tensor_statistics(
            weights,
            compute_mean=self.compute_mean,
            compute_variance=self.compute_variance,
            compute_min_max=self.compute_min_max,
            compute_percentiles=self.compute_percentiles,
        )

    def _active_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Return weights above the configured active-synapse threshold."""
        return weights[weights > self.weight_threshold]

    @staticmethod
    def _branch_layer_depth(
        branch_layer: DendriticBranchLayer, fallback_depth: int
    ) -> int:
        return getattr(branch_layer, "layer_idx", fallback_depth)

    def _iter_branch_layers(
        self,
        model: BaseModel,
        *,
        include_inhibitory: bool,
        require_excitatory_dendrinet: bool = False,
        require_layer_idx: bool = False,
    ) -> Iterator[tuple[int, DendriticBranchLayer]]:
        del require_excitatory_dendrinet
        for population in _iter_dendritic_populations(
            model, include_inhibitory=include_inhibitory
        ):
            yield from self._iter_dendrinet_branch_layers(
                population, require_layer_idx=require_layer_idx
            )

    def _iter_dendrinet_branch_layers(
        self, dendrinet, *, require_layer_idx: bool = False
    ) -> Iterator[tuple[int, DendriticBranchLayer]]:
        for depth_idx, branch_layer in enumerate(dendrinet.branch_layers):
            if require_layer_idx and getattr(branch_layer, "layer_idx", None) is None:
                continue
            yield self._branch_layer_depth(branch_layer, depth_idx), branch_layer

    def _compute_layer_statistics(self, model: BaseModel) -> dict[str, Any]:
        """Compute hierarchical per-branch-depth weight statistics across the network.

        This aggregates statistics for each dendritic branch depth index across all
        EI layers, matching the structure expected by the original synaptic weight
        analyzer and plotting utilities (keys like "branch_layers.{idx}").
        """
        self.logger.info("Computing hierarchical layer statistics...")

        # Accumulate branch statistics per dendritic depth across EI layers
        depth_to_branch_list: dict[int, list[dict[str, Any]]] = {}

        for layer_idx_attr, branch_layer in self._iter_branch_layers(
            model, include_inhibitory=True
        ):
            stats_dict = self._compute_branch_weight_statistics(
                branch_layer, module_name=f"branch_layers.{layer_idx_attr}"
            )
            if stats_dict and "branches" in stats_dict:
                if layer_idx_attr not in depth_to_branch_list:
                    depth_to_branch_list[layer_idx_attr] = []
                depth_to_branch_list[layer_idx_attr].extend(stats_dict["branches"])

        # Now aggregate per-depth branch statistics to layer-level metrics
        layer_results: dict[str, Any] = {}
        for depth_idx, branches in depth_to_branch_list.items():
            layer_name = f"branch_layers.{depth_idx}"
            aggregated_stats = self._aggregate_layer_statistics({"branches": branches})
            if aggregated_stats:
                layer_results[layer_name] = aggregated_stats
                self.logger.info(f"Layer {layer_name}: {list(aggregated_stats.keys())}")

        return layer_results

    def _compute_branch_weight_statistics(
        self, module: DendriticBranchLayer, module_name: str
    ) -> dict[str, Any]:
        """Compute weight statistics for a single dendritic branch layer."""
        if not isinstance(module, DendriticBranchLayer):
            return {}

        branch_results = []

        # Determine the maximum number of branches
        max_branches = 0
        if self.analyze_excitatory and module.branch_excitation is not None:
            max_branches = max(
                max_branches, module.branch_excitation.pruned_weight().shape[0]
            )
        if self.analyze_inhibitory and module.branch_inhibition is not None:
            max_branches = max(
                max_branches, module.branch_inhibition.pruned_weight().shape[0]
            )

        # Initialize branch results for all branches
        for branch_idx in range(max_branches):
            branch_stats = {"branch_idx": branch_idx, "layer": module_name}
            branch_results.append(branch_stats)

        # Analyze excitatory weights per branch
        if self.analyze_excitatory and module.branch_excitation is not None:
            exc_weights = (
                module.branch_excitation.pruned_weight()
            )  # [n_branches, in_features]

            for branch_idx in range(exc_weights.shape[0]):
                if branch_idx < len(branch_results):
                    self._record_branch_synapse_statistics(
                        branch_results[branch_idx],
                        prefix="exc",
                        weights=exc_weights[branch_idx, :],
                    )

        # Analyze inhibitory weights per branch
        if self.analyze_inhibitory and module.branch_inhibition is not None:
            inh_weights = module.branch_inhibition.pruned_weight()

            for branch_idx in range(inh_weights.shape[0]):
                if branch_idx < len(branch_results):
                    self._record_branch_synapse_statistics(
                        branch_results[branch_idx],
                        prefix="inh",
                        weights=inh_weights[branch_idx, :],
                    )

        return {"branches": branch_results}

    def _record_branch_synapse_statistics(
        self, branch_stats: dict[str, Any], prefix: str, weights: torch.Tensor
    ) -> None:
        """Add active-synapse weight statistics for one branch and synapse type."""
        active_weights = weights[weights > self.weight_threshold]
        branch_stats.update(
            _branch_synapse_statistics(
                prefix,
                active_weights,
                compute_mean=self.compute_mean,
                compute_variance=self.compute_variance,
                compute_min_max=self.compute_min_max,
                compute_percentiles=self.compute_percentiles,
            )
        )

    def _compute_branch_level_statistics(self, model: BaseModel) -> dict[str, Any]:
        """Compute and return raw per-branch statistics grouped by depth.

        This mirrors the original analyzer's behavior of keeping branch-level data
        before aggregation, to enable downstream tools to recompute or plot with
        different aggregations if desired.
        """
        depth_to_branch_list: dict[int, list[dict[str, Any]]] = {}

        for layer_idx_attr, branch_layer in self._iter_branch_layers(
            model, include_inhibitory=True
        ):
            stats_dict = self._compute_branch_weight_statistics(
                branch_layer, module_name=f"branch_layers.{layer_idx_attr}"
            )
            if not stats_dict or "branches" not in stats_dict:
                continue
            if layer_idx_attr not in depth_to_branch_list:
                depth_to_branch_list[layer_idx_attr] = []
            depth_to_branch_list[layer_idx_attr].extend(stats_dict["branches"])

        # Format similar to layer_statistics but with raw branches list
        formatted: dict[str, Any] = {}
        for depth_idx, branches in depth_to_branch_list.items():
            formatted[f"branch_layers.{depth_idx}"] = {"branches": branches}
        return formatted

    def _aggregate_layer_statistics(
        self, layer_data: dict[str, Any]
    ) -> dict[str, float]:
        """Aggregate branch statistics to layer level."""
        branch_results = layer_data.get("branches", [])
        if not branch_results:
            return {}

        layer_stats = {}

        # Collect all branch statistics
        metrics = [
            "exc_weight_mean",
            "exc_weight_var",
            "exc_weight_min",
            "exc_weight_max",
            "exc_n_active_synapses",
            "inh_weight_mean",
            "inh_weight_var",
            "inh_weight_min",
            "inh_weight_max",
            "inh_n_active_synapses",
        ]

        for metric in metrics:
            values = [
                branch.get(metric) for branch in branch_results if metric in branch
            ]
            if values:
                values = [v for v in values if v is not None]
                if values:
                    layer_stats[f"{metric}_mean"] = np.mean(values)
                    layer_stats[f"{metric}_std"] = np.std(values)
                    if self.compute_min_max:
                        layer_stats[f"{metric}_layer_min"] = np.min(values)
                        layer_stats[f"{metric}_layer_max"] = np.max(values)

        return layer_stats

    def _compute_enhanced_layer_statistics(self, model: BaseModel) -> dict[str, Any]:
        """Compute enhanced per-layer statistics including dendritic strength."""
        self.logger.info("Computing enhanced layer statistics...")

        enhanced_stats: dict[str, Any] = {}

        # Aggregate enhanced statistics per dendritic depth across EI layers
        # Build a dict keyed by actual branch layer_idx to accumulate values
        depth_to_strength: dict[int, float] = {}
        depth_to_weights: dict[int, list[torch.Tensor]] = {}

        for layer_idx_attr, branch_layer in self._iter_branch_layers(
            model,
            include_inhibitory=True,
            require_excitatory_dendrinet=True,
            require_layer_idx=True,
        ):
            self._accumulate_enhanced_branch_statistics(
                branch_layer,
                layer_idx_attr,
                depth_to_strength,
                depth_to_weights,
            )

        # Build enhanced_stats from accumulators
        for layer_idx_attr in sorted(
            set(depth_to_strength.keys()) | set(depth_to_weights.keys())
        ):
            layer_name = f"branch_layers.{layer_idx_attr}"
            enhanced_stats[layer_name] = {}
            if self.compute_dendritic_strength and layer_idx_attr in depth_to_strength:
                enhanced_stats[layer_name]["dendritic_strength"] = depth_to_strength[
                    layer_idx_attr
                ]
            if self.compute_min_max and depth_to_weights.get(layer_idx_attr):
                all_weights = torch.cat(depth_to_weights[layer_idx_attr])
                if all_weights.numel() > 0:
                    enhanced_stats[layer_name][
                        "layer_weight_min"
                    ] = all_weights.min().item()
                    enhanced_stats[layer_name][
                        "layer_weight_max"
                    ] = all_weights.max().item()

        return enhanced_stats

    def _accumulate_enhanced_branch_statistics(
        self,
        branch_layer: DendriticBranchLayer,
        layer_idx_attr: int,
        depth_to_strength: dict[int, float],
        depth_to_weights: dict[int, list[torch.Tensor]],
    ) -> None:
        if layer_idx_attr not in depth_to_strength:
            depth_to_strength[layer_idx_attr] = 0.0
        if layer_idx_attr not in depth_to_weights:
            depth_to_weights[layer_idx_attr] = []

        if self.compute_dendritic_strength:
            if self.analyze_excitatory and branch_layer.branch_excitation is not None:
                active_exc = self._active_weights(
                    branch_layer.branch_excitation.pruned_weight()
                )
                depth_to_strength[layer_idx_attr] += active_exc.sum().item()

            if self.analyze_inhibitory and branch_layer.branch_inhibition is not None:
                active_inh = self._active_weights(
                    branch_layer.branch_inhibition.pruned_weight()
                )
                depth_to_strength[layer_idx_attr] += active_inh.sum().item()

        if self.compute_min_max:
            if self.analyze_excitatory and branch_layer.branch_excitation is not None:
                active_exc = self._active_weights(
                    branch_layer.branch_excitation.pruned_weight().flatten()
                )
                if active_exc.numel() > 0:
                    depth_to_weights[layer_idx_attr].append(active_exc)
            if self.analyze_inhibitory and branch_layer.branch_inhibition is not None:
                active_inh = self._active_weights(
                    branch_layer.branch_inhibition.pruned_weight().flatten()
                )
                if active_inh.numel() > 0:
                    depth_to_weights[layer_idx_attr].append(active_inh)

    def _generate_all_plots(
        self, results: dict[str, Any], save_path: str, model: BaseModel
    ):
        """Generate all weight analysis plots."""
        self.logger.info("Generating weight analysis plots...")
        self.logger.info(f"Results keys: {list(results.keys())}")
        self.logger.info(f"Save path: {save_path}")

        try:
            # Get somatic synapses setting
            somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
            self.logger.info(f"Somatic synapses: {somatic_synapses}")

            # Generate hierarchical plots (original synaptic_weight_expectation plots)
            if "layer_statistics" in results:
                self.logger.info("Generating hierarchical weight plots...")
                self._plot_hierarchical_weights(
                    results["layer_statistics"], save_path, somatic_synapses
                )
                self.logger.info("Hierarchical plots completed")
            else:
                self.logger.warning("No layer_statistics found in results")

            # Generate enhanced plots (min/max and dendritic strength)
            if "enhanced_layer_statistics" in results:
                self.logger.info("Generating enhanced statistics plots...")
                self._plot_enhanced_statistics(
                    results["enhanced_layer_statistics"], save_path, somatic_synapses
                )
                self.logger.info("Enhanced plots completed")
            else:
                self.logger.info(
                    "enhanced_layer_statistics not present; skipping optional enhanced plots"
                )

            # Generate global distribution plots
            if "global_statistics" in results:
                self.logger.info("Generating global distribution plots...")
                self._plot_global_distributions(results["global_statistics"], save_path)
                self.logger.info("Global plots completed")
            else:
                self.logger.warning("No global_statistics found in results")

        except Exception as e:
            self.logger.error(f"Error generating weight plots: {e}")
            import traceback

            traceback.print_exc()

    def _plot_hierarchical_weights(
        self, layer_stats: dict[str, Any], save_path: str, somatic_synapses: bool
    ):
        """Generate all original synaptic weight plots plus the simplified ones."""
        try:
            from dendritic_modeling.plotting.visualizations.synaptic_weight_plots import (
                plot_synaptic_weight_analysis,
            )

            # Generate all the original plots that synaptic_weight_expectation analyzer created
            plot_synaptic_weight_analysis(
                layer_stats,
                save_path=save_path,
                somatic_synapses=somatic_synapses,
            )

        except ImportError as e:
            self.logger.warning(f"Could not import weight plotting functions: {e}")

    def _plot_enhanced_statistics(
        self, enhanced_stats: dict[str, Any], save_path: str, somatic_synapses: bool
    ):
        """Generate enhanced plots for min/max and dendritic strength."""
        try:
            import os

            import matplotlib.pyplot as plt

            from dendritic_modeling.plotting.visualizations.plotting_utils import (
                convert_layer_names,
                get_color_scheme,
                setup_basic_plot,
            )

            # Convert layer names for proper ordering
            converted_results = convert_layer_names(
                enhanced_stats, somatic_synapses=somatic_synapses
            )
            layer_names = list(converted_results.keys())
            colors = get_color_scheme()

            # Create 2-panel plot for enhanced statistics
            _, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Collect data
            depths = []
            min_weights = []
            max_weights = []
            dendritic_strengths = []

            for layer_name in layer_names:
                layer_data = converted_results[layer_name]
                if (
                    "layer_weight_min" in layer_data
                    and "layer_weight_max" in layer_data
                ):
                    depths.append(layer_name)
                    min_weights.append(layer_data["layer_weight_min"])
                    max_weights.append(layer_data["layer_weight_max"])
                    dendritic_strengths.append(
                        layer_data.get("dendritic_strength", 0.0)
                    )

            if depths:
                x_pos = np.arange(len(depths))
                width = 0.35

                # Panel 1: Min/Max weights per layer
                ax1 = axes[0]
                ax1.bar(
                    x_pos - width / 2,
                    min_weights,
                    width,
                    label="Min Weight",
                    color=colors["excitatory"],
                    alpha=0.7,
                )
                ax1.bar(
                    x_pos + width / 2,
                    max_weights,
                    width,
                    label="Max Weight",
                    color=colors["inhibitory"],
                    alpha=0.7,
                )

                setup_basic_plot(
                    ax1,
                    "Min/Max Weights per Layer",
                    "Layer (Soma to Distal)",
                    "Weight Value",
                    grid=True,
                )
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(depths, rotation=45)
                ax1.legend()

                # Panel 2: Dendritic strength per layer
                ax2 = axes[1]
                ax2.bar(
                    x_pos,
                    dendritic_strengths,
                    width * 2,
                    color=colors["combined"],
                    alpha=0.7,
                )

                setup_basic_plot(
                    ax2,
                    "Total Dendritic Strength per Layer",
                    "Layer (Soma to Distal)",
                    "Total Weight Sum",
                    grid=True,
                )
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(depths, rotation=45)

            plt.suptitle(
                "Enhanced Weight Analysis: Min/Max and Dendritic Strength", fontsize=16
            )
            plt.tight_layout()

            # Save plot
            filename = os.path.join(save_path, "enhanced_weight_analysis.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.info(f"Saved enhanced weight analysis plot: {filename}")

        except Exception as e:
            self.logger.error(f"Error generating enhanced plots: {e}")

    def _plot_global_distributions(self, global_stats: dict[str, Any], save_path: str):
        """Generate global weight distribution plots."""
        try:
            import os

            import matplotlib.pyplot as plt

            # Create histogram plots for global distributions
            _, axes = plt.subplots(1, 3, figsize=(15, 5))

            weight_types = [
                "excitatory_weights",
                "inhibitory_weights",
                "branch_weights",
            ]
            titles = [
                "Excitatory Weight Distribution",
                "Inhibitory Weight Distribution",
                "Branch Weight Distribution",
            ]

            for i, (weight_type, title) in enumerate(zip(weight_types, titles)):
                if weight_type in global_stats:
                    stats = global_stats[weight_type]
                    ax = axes[i]

                    # Create a simple bar plot of statistics
                    metrics = ["mean", "std", "min", "max", "median"]
                    values = [stats.get(metric, 0) for metric in metrics]

                    ax.bar(metrics, values, alpha=0.7)
                    ax.set_title(title)
                    ax.set_ylabel("Weight Value")
                    ax.tick_params(axis="x", rotation=45)
                else:
                    axes[i].text(
                        0.5,
                        0.5,
                        f"No {weight_type.replace('_', ' ')}",
                        ha="center",
                        va="center",
                        transform=axes[i].transAxes,
                    )
                    axes[i].set_title(titles[i])

            plt.suptitle("Global Weight Statistics", fontsize=16)
            plt.tight_layout()

            # Save plot
            filename = os.path.join(save_path, "global_weight_distributions.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.info(f"Saved global weight distributions plot: {filename}")

        except Exception as e:
            self.logger.error(f"Error generating global distribution plots: {e}")


__all__ = ["WeightAnalyzer"]
