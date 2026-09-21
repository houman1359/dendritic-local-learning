"""Compartment statistics analysis.

This module collects dendritic synapse weights, branch weights, inputs, and
activations, then summarizes them at branch, layer, and global granularity with
optional entropy and per-class statistics.
"""

import logging
import os
import time
import traceback
import zlib
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import differential_entropy

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    materialize_dataset,
)
from dendritic_modeling.config import CompartmentStatisticsAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer, TopKLinear
from dendritic_modeling.plotting.visualizations.plotting_utils import (
    convert_layer_names,
    get_color_scheme,
    setup_basic_plot,
)
from dendritic_modeling.plotting.visualizations.synaptic_weight_plots import (
    plot_synaptic_weight_analysis,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_pathways import (
    iter_topk_path_modules,
)
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    iter_named_modules_of_type,
    register_hook_groups,
    register_named_forward_hook_groups,
    run_with_forward_hooks,
)

logger = logging.getLogger(__name__)


def _empty_tensor_statistics(
    prefix: str,
    *,
    compute_mean: bool,
    compute_variance: bool,
    compute_min_max: bool,
    compute_percentiles: bool,
) -> dict[str, float]:
    """Return the legacy zero-valued statistics for empty tensors."""
    stats: dict[str, float] = {}
    if compute_mean:
        stats[f"{prefix}_mean"] = 0.0
    if compute_variance:
        stats[f"{prefix}_var"] = 0.0
        stats[f"{prefix}_std"] = 0.0
    if compute_min_max:
        stats[f"{prefix}_min"] = 0.0
        stats[f"{prefix}_max"] = 0.0
    if compute_percentiles:
        stats[f"{prefix}_q1"] = 0.0
        stats[f"{prefix}_q2"] = 0.0
        stats[f"{prefix}_q3"] = 0.0
    return stats


def _tensor_statistics(
    data_tensor: torch.Tensor,
    prefix: str,
    *,
    compute_mean: bool,
    compute_variance: bool,
    compute_min_max: bool,
    compute_percentiles: bool,
) -> dict[str, float]:
    """Compute configured summary statistics with the historical torch reductions."""
    if not isinstance(data_tensor, torch.Tensor) or data_tensor.numel() == 0:
        return _empty_tensor_statistics(
            prefix,
            compute_mean=compute_mean,
            compute_variance=compute_variance,
            compute_min_max=compute_min_max,
            compute_percentiles=compute_percentiles,
        )

    stats: dict[str, float] = {}
    if compute_mean:
        stats[f"{prefix}_mean"] = data_tensor.mean().item()

    if compute_variance:
        stats[f"{prefix}_var"] = data_tensor.var().item()
        stats[f"{prefix}_std"] = data_tensor.std().item()

    if compute_min_max:
        stats[f"{prefix}_min"] = data_tensor.min().item()
        stats[f"{prefix}_max"] = data_tensor.max().item()

    if compute_percentiles:
        stats[f"{prefix}_q1"] = data_tensor.quantile(0.25).item()
        stats[f"{prefix}_q2"] = data_tensor.median().item()
        stats[f"{prefix}_q3"] = data_tensor.quantile(0.75).item()

    return stats


class CompartmentStatisticsAnalyzer(ForwardHookRemovalMixin, AbstractAnalyzer):
    """
    Analyzer for compartment-level dendritic statistics.

    Provides branch, layer, and global summaries for synapse weights, branch
    weights, inputs, activations, entropy, and optional per-class statistics.
    """

    def __init__(self, params: CompartmentStatisticsAnalysisParams):
        """Initialize the unified compartment statistics analyzer."""
        super().__init__("CompartmentStatisticsAnalyzer")

        self.synapse_weights = getattr(params, "synapse_weights", True)
        self.branch_weights = getattr(params, "branch_weights", True)
        self.inputs = getattr(params, "inputs", True)
        self.activations = getattr(params, "activations", True)

        # Enhanced analysis modes
        self.global_analysis = getattr(params, "global_analysis", True)
        self.layer_analysis = getattr(params, "layer_analysis", True)
        self.branch_analysis = getattr(params, "branch_analysis", True)

        self.compute_mean = getattr(params, "compute_mean", True)
        self.compute_variance = getattr(params, "compute_variance", True)
        self.compute_percentiles = getattr(params, "compute_percentiles", True)
        self.compute_min_max = getattr(params, "compute_min_max", True)

        self.compute_entropy = getattr(params, "compute_entropy", True)
        self.dequantize = getattr(params, "dequantize", True)
        # self.n_neighbors = getattr(params, "n_neighbors", 10)

        self.per_class_analysis = getattr(params, "per_class_analysis", False)
        self.n_samples = getattr(params, "n_samples", None)
        self.seed = int(getattr(params, "seed", 0) or 0)
        self.epsilon = torch.finfo(torch.float32).eps

    def collect_raw_data(
        self, model: BaseModel, x: torch.Tensor, y: torch.Tensor = None
    ):
        """Collect raw data from the model."""
        self.raw_dict = {}
        self.class_labels = (
            y.detach().cpu() if self.per_class_analysis and y is not None else None
        )

        def _attach_hooks():
            return register_named_forward_hook_groups(
                model,
                DendriticBranchLayer,
                self._collect_layer_raw_data,
            )

        def _run_model():
            if self.inputs or self.activations:
                batch_x = x
                if self.n_samples is not None:
                    generator = torch.Generator(device=batch_x.device)
                    generator.manual_seed(self.seed)
                    indices = torch.randperm(
                        batch_x.shape[0], generator=generator, device=batch_x.device
                    )[: self.n_samples]
                    batch_x = batch_x[indices]
                    if self.class_labels is not None:
                        self.class_labels = self.class_labels[indices.cpu()]
                with torch.no_grad():
                    _ = model(batch_x)

        run_with_forward_hooks(
            attach=_attach_hooks,
            remove=self.remove_forward_hooks,
            body=_run_model,
        )

    def _collect_layer_raw_data(
        self,
        key: str,
        module: DendriticBranchLayer,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        self._initialize_branch_records(key, module)
        synapse_modules = self._iter_synapse_modules(module)
        self._collect_layer_static_data(key, module, synapse_modules)

        if not (self.inputs or self.activations):
            return []
        return self._attach_synapse_capture_hooks(key, synapse_modules)

    def _initialize_branch_records(
        self, key: str, module: DendriticBranchLayer
    ) -> None:
        self.raw_dict[key] = {}
        for branch_idx in range(module.n_branches):
            self.raw_dict[key][branch_idx] = {"depth": module.layer_idx}

    def _iter_synapse_modules(
        self, module: DendriticBranchLayer
    ) -> list[tuple[str, TopKLinear | None]]:
        return [
            (path.pathway, synapse_module)
            for path, synapse_module in iter_topk_path_modules(module)
        ]

    def _collect_layer_static_data(
        self,
        key: str,
        module: DendriticBranchLayer,
        synapse_modules: list[tuple[str, TopKLinear | None]],
    ) -> None:
        if self.synapse_weights:
            for synapse_type, synapse_module in synapse_modules:
                if synapse_module is None:
                    continue
                weights = synapse_module.pruned_weight().detach().cpu()
                for branch_idx in range(module.n_branches):
                    w = weights[branch_idx]
                    w = w[w > self.epsilon]
                    self.raw_dict[key][branch_idx][f"{synapse_type}_weights"] = w

        if self.branch_weights and hasattr(module, "branches_to_output"):
            branch_weights = module.branches_to_output.weight().detach().cpu()
            for branch_idx in range(module.n_branches):
                branch_w = branch_weights[branch_idx]
                branch_w = branch_w[branch_w > self.epsilon]
                self.raw_dict[key][branch_idx]["branch_weights"] = branch_w

    def _register_synapse_capture_hook(
        self,
        key: str,
        synapse_type: str,
        synapse_module: TopKLinear,
    ) -> torch.utils.hooks.RemovableHandle:
        return synapse_module.register_forward_hook(
            lambda module, input, output, layer_key=key, label=synapse_type: self.forward_hook(
                module, input, output, layer_key, label
            )
        )

    def _attach_synapse_capture_hooks(
        self,
        key: str,
        synapse_modules: list[tuple[str, TopKLinear | None]],
    ) -> list[torch.utils.hooks.RemovableHandle]:
        def _register_synapse_hook(
            synapse_entry: tuple[str, TopKLinear | None],
        ) -> list[torch.utils.hooks.RemovableHandle]:
            synapse_type, synapse_module = synapse_entry
            if synapse_module is None:
                return []
            return [
                self._register_synapse_capture_hook(key, synapse_type, synapse_module)
            ]

        return register_hook_groups(
            synapse_modules,
            _register_synapse_hook,
        )

    def forward_hook(
        self,
        module: TopKLinear,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        key: str,
        synapse_type: str,
    ):
        if self.inputs:
            weight_mask = module.weight_mask()

            for branch_idx in range(module.out_features):
                branch_mask = weight_mask[branch_idx]
                branch_x = input[0][..., branch_mask > self.epsilon].detach().cpu()

                if self.per_class_analysis and self.class_labels is not None:
                    # Store inputs per class
                    if (
                        f"{synapse_type}_inputs_per_class"
                        not in self.raw_dict[key][branch_idx]
                    ):
                        self.raw_dict[key][branch_idx][
                            f"{synapse_type}_inputs_per_class"
                        ] = {}

                    for class_idx in torch.unique(self.class_labels):
                        class_mask = self.class_labels == class_idx
                        class_inputs = branch_x[class_mask]
                        if len(class_inputs) > 0:
                            class_key = int(class_idx.item())
                            if (
                                class_key
                                not in self.raw_dict[key][branch_idx][
                                    f"{synapse_type}_inputs_per_class"
                                ]
                            ):
                                self.raw_dict[key][branch_idx][
                                    f"{synapse_type}_inputs_per_class"
                                ][class_key] = []
                            self.raw_dict[key][branch_idx][
                                f"{synapse_type}_inputs_per_class"
                            ][class_key].append(class_inputs)
                else:
                    self.raw_dict[key][branch_idx][f"{synapse_type}_inputs"] = branch_x

        if self.activations:
            for branch_idx in range(module.out_features):
                branch_activation = output[..., branch_idx].detach().cpu()

                if self.per_class_analysis and self.class_labels is not None:
                    # Store activations per class
                    if (
                        f"{synapse_type}_activations_per_class"
                        not in self.raw_dict[key][branch_idx]
                    ):
                        self.raw_dict[key][branch_idx][
                            f"{synapse_type}_activations_per_class"
                        ] = {}

                    for class_idx in torch.unique(self.class_labels):
                        class_mask = self.class_labels == class_idx
                        class_activations = branch_activation[class_mask]
                        if len(class_activations) > 0:
                            class_key = int(class_idx.item())
                            if (
                                class_key
                                not in self.raw_dict[key][branch_idx][
                                    f"{synapse_type}_activations_per_class"
                                ]
                            ):
                                self.raw_dict[key][branch_idx][
                                    f"{synapse_type}_activations_per_class"
                                ][class_key] = []
                            self.raw_dict[key][branch_idx][
                                f"{synapse_type}_activations_per_class"
                            ][class_key].append(class_activations)
                else:
                    self.raw_dict[key][branch_idx][
                        f"{synapse_type}_activations"
                    ] = branch_activation

    def _compute_statistics(
        self,
        data_tensor: torch.Tensor,
        prefix: str,
        stats_dict: dict[str, float],
    ) -> dict[str, float]:
        """Compute comprehensive statistics for a weight tensor."""
        stats_dict.update(
            _tensor_statistics(
                data_tensor,
                prefix,
                compute_mean=self.compute_mean,
                compute_variance=self.compute_variance,
                compute_min_max=self.compute_min_max,
                compute_percentiles=self.compute_percentiles,
            )
        )
        return stats_dict

    def _entropy_samples_for_prefix(
        self,
        data_tensor: torch.Tensor,
        prefix: str,
    ) -> np.ndarray | None:
        if "inputs" in prefix:
            return data_tensor.flatten(0, -2).mean(dim=-1).numpy()
        if "activations" in prefix:
            return data_tensor.flatten(0, -1).numpy()
        return None

    def _per_class_entropy(
        self,
        data_for_entropy: np.ndarray,
        seed_key: str,
    ):
        if np.var(data_for_entropy) < 1e-10:
            seed = zlib.adler32(seed_key.encode())
            rng = np.random.default_rng(seed)
            data_for_entropy = data_for_entropy + rng.normal(
                0.0,
                1e-8,
                data_for_entropy.shape,
            )
        return differential_entropy(data_for_entropy, method="vasicek")

    def _dequantize_entropy_samples(
        self,
        data_for_entropy: np.ndarray,
        seed_key: str,
    ) -> np.ndarray:
        u = np.unique(data_for_entropy)
        if u.size >= 2:
            diffs = np.diff(u)
            diffs = diffs[diffs > 0]
            step = float(np.median(diffs)) if diffs.size > 0 else 1e-8
        else:
            step = 1e-8

        if not np.isfinite(step) or step <= 0:
            step = 1e-8

        seed = zlib.adler32(seed_key.encode())
        rng = np.random.default_rng(seed)
        return data_for_entropy + ((rng.random(data_for_entropy.shape) - 0.5) * step)

    def _generic_entropy(
        self,
        data_for_entropy: np.ndarray,
        prefix: str,
        layer_key: str,
        branch_idx: int,
    ) -> float:
        data_for_entropy = np.asarray(data_for_entropy, dtype=float)
        data_for_entropy = data_for_entropy[np.isfinite(data_for_entropy)]

        # Degenerate / empty distributions should not produce NaNs.
        if data_for_entropy.size < 2 or float(np.var(data_for_entropy)) < 1e-10:
            return 0.0

        if self.dequantize:
            data_for_entropy = self._dequantize_entropy_samples(
                data_for_entropy,
                f"{layer_key}:{branch_idx}:{prefix}",
            )

        try:
            ent = float(differential_entropy(data_for_entropy, method="vasicek"))
        except Exception:
            ent = 0.0

        if not np.isfinite(ent):
            logger.warning(
                "Non-finite entropy for %s (layer=%s branch=%s); setting to 0.0",
                prefix,
                layer_key,
                branch_idx,
            )
            ent = 0.0
        return ent

    def _compute_branch_statistics(self):
        branch_statistics = {}
        for layer_key, layer_data in self.raw_dict.items():
            branch_statistics[layer_key] = {}
            layer_data: dict[int, dict[str, torch.Tensor]]
            for branch_idx, branch_data in layer_data.items():
                branch_stats = {}

                for prefix, data_tensor in branch_data.items():
                    if prefix == "depth":
                        branch_stats[prefix] = (
                            data_tensor  # data_tensor is the depth index
                        )
                    elif "per_class" in prefix:
                        # Handle per-class data
                        if self.per_class_analysis:
                            for class_idx, class_data_list in data_tensor.items():
                                # Concatenate all data for this class
                                class_data = torch.cat(class_data_list, dim=0)
                                class_prefix = f"{prefix}_class_{class_idx}"
                                branch_stats = self._compute_statistics(
                                    class_data, class_prefix, branch_stats
                                )

                                if self.compute_entropy:
                                    try:
                                        data_for_entropy = (
                                            self._entropy_samples_for_prefix(
                                                class_data,
                                                prefix,
                                            )
                                        )
                                        if data_for_entropy is not None:
                                            branch_stats[f"{class_prefix}_entropy"] = (
                                                self._per_class_entropy(
                                                    data_for_entropy,
                                                    f"{layer_key}:{branch_idx}:{prefix}:class_{class_idx}",
                                                )
                                            )
                                    except (ValueError, RuntimeWarning):
                                        # Handle entropy calculation errors gracefully
                                        branch_stats[f"{class_prefix}_entropy"] = 0.0
                        # Per-class payloads are dictionaries, so skip the generic entropy path.
                        continue
                    else:
                        branch_stats = self._compute_statistics(
                            data_tensor, prefix, branch_stats
                        )

                    if self.compute_entropy:
                        data_for_entropy = self._entropy_samples_for_prefix(
                            data_tensor,
                            prefix,
                        )

                        if data_for_entropy is not None:
                            branch_stats[f"{prefix}_entropy"] = self._generic_entropy(
                                data_for_entropy,
                                prefix,
                                layer_key,
                                branch_idx,
                            )
                branch_statistics[layer_key][branch_idx] = branch_stats

        return branch_statistics

    def _aggregate_branch_stat_values(
        self,
        branch_records,
    ) -> dict[str, list[Any]]:
        aggregated: dict[str, list[Any]] = {}
        for branch_data in branch_records:
            for field, value in branch_data.items():
                if field not in aggregated.keys() and field != "depth":
                    aggregated[field] = []
                if field != "depth":
                    aggregated[field].append(value)
        return aggregated

    def _compute_aggregated_statistics(
        self,
        aggregated_values: dict[str, list[Any]],
        stats_dict: dict[str, float],
    ) -> dict[str, float]:
        for prefix, values in aggregated_values.items():
            data_tensor = torch.tensor(values)
            stats_dict = self._compute_statistics(data_tensor, prefix, stats_dict)
        return stats_dict

    def _compute_layer_statistics(
        self, branch_statistics: dict[str, dict[int, dict[str, float]]]
    ):
        """Compute layer statistics from branch statistics."""
        layer_statistics = {}
        for layer_key, layer_data in branch_statistics.items():
            layer_aggregated = self._aggregate_branch_stat_values(layer_data.values())
            for branch_data in layer_data.values():
                depth = branch_data["depth"]
            layer_stats = {"depth": depth}
            layer_stats = self._compute_aggregated_statistics(
                layer_aggregated,
                layer_stats,
            )

            layer_statistics[layer_key] = layer_stats
        return layer_statistics

    def _compute_global_statistics(
        self, branch_statistics: dict[str, dict[int, dict[str, float]]]
    ):
        """Compute global statistics from branch statistics."""
        global_aggregated: dict[str, list[Any]] = {}
        for layer_data in branch_statistics.values():
            layer_aggregated = self._aggregate_branch_stat_values(layer_data.values())
            for field, values in layer_aggregated.items():
                if field not in global_aggregated:
                    global_aggregated[field] = []
                global_aggregated[field].extend(values)

        return self._compute_aggregated_statistics(global_aggregated, {})

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ):
        """
        Perform compartment statistics analysis with optional hierarchical summaries.

        Args:
            model: The model to analyze (must have ExcitationInhibitionNetwork)
            save_path: Path to save results (optional)
            filename: Filename for saved results
            training: If True, only save data (no plots). If False, generate plots.

        Returns:
            Dictionary containing compartment statistics analysis results.
        """
        core = getattr(model, "core_network", model)
        if not any(iter_named_modules_of_type(core, DendriticBranchLayer)):
            return None

        start_time = time.time()

        with analysis_device_context(model, device) as analysis_device:
            items = materialize_dataset(
                test_dataset,
                runtime,
                explicit_max_samples=self.n_samples,
                device=analysis_device,
            )
            x: torch.Tensor = items[0].to(analysis_device)
            y: torch.Tensor = (
                items[1].to(analysis_device) if self.per_class_analysis else None
            )

            self.collect_raw_data(model, x, y)
            branch_statistics = self._compute_branch_statistics()

        results = {}

        if self.global_analysis:
            global_statistics = self._compute_global_statistics(branch_statistics)
            results["global_statistics"] = global_statistics

        if self.layer_analysis:
            layer_statistics = self._compute_layer_statistics(branch_statistics)
            results["layer_statistics"] = layer_statistics

        if self.branch_analysis:
            results["branch_statistics"] = branch_statistics

        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Compartment statistics analysis completed in {elapsed_time:.2f} seconds"
        )

        # Save results and generate plots
        if save_path is not None:
            self.logger.info(
                f"Saving compartment statistics results to {save_path}/{filename}"
            )
            save_dict(results, save_path, filename)

            fields = []
            if self.synapse_weights:
                fields.append("weights")
            if self.inputs:
                fields.append("inputs")
            if self.activations:
                fields.append("activations")

            summ_stats = []
            if self.compute_mean:
                summ_stats.append("mean")
            if self.compute_variance:
                summ_stats.append("std")

            # try:
            #     plot_ratio_comparison(
            #         branch_statistics=branch_statistics,
            #         save_path=save_path
            #     )
            # except Exception as e:
            #     self.logger.error(f"Error generating ratio comparison plot: {e}")
            #     traceback.print_exc()

            try:
                plot_ratio_pairwise_comparison(
                    branch_statistics=branch_statistics,
                    weights_ratio=self.synapse_weights,
                    inputs_ratio=self.inputs,
                    activations_ratio=self.activations,
                    save_path=save_path,
                )
            except Exception as e:
                self.logger.error(
                    f"Error generating ratio pairwise comparison plot: {e}"
                )
                traceback.print_exc()

            # Only generate plots during final analysis (not during training)
            if not training:
                self.logger.info("Starting plot generation...")
                self._generate_all_plots(results, save_path, model)
                self.logger.info("Plot generation completed!")
            else:
                self.logger.info("Training mode: plots skipped, only data saved")

        return results

    def _generate_all_plots(
        self, results: dict[str, Any], save_path: str, model: BaseModel
    ):
        """Generate all compartment statistics plots."""
        self.logger.info("Generating compartment statistics plots...")
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

            traceback.print_exc()

    def _plot_hierarchical_weights(
        self, layer_stats: dict[str, Any], save_path: str, somatic_synapses: bool
    ):
        """Generate all original synaptic weight plots plus the simplified ones."""

        # Generate all the original plots that synaptic_weight_expectation analyzer created
        plot_synaptic_weight_analysis(
            layer_stats,
            save_path=save_path,
            somatic_synapses=somatic_synapses,
        )

    def _plot_enhanced_statistics(
        self, enhanced_stats: dict[str, Any], save_path: str, somatic_synapses: bool
    ):
        """Generate enhanced plots for min/max and dendritic strength."""
        try:
            # Convert layer names for proper ordering
            converted_results = convert_layer_names(
                enhanced_stats, somatic_synapses=somatic_synapses
            )
            layer_names = list(converted_results.keys())
            colors = get_color_scheme()

            # Create 2-panel plot for enhanced statistics
            _fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
            # Create histogram plots for global distributions
            _fig, axes = plt.subplots(1, 3, figsize=(15, 5))

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


def plot_compartment_weight_analysis(
    layer_statistics: dict[str, dict[str, float]],
    save_path: str,
    somatic_synapses: bool = True,
):
    """Plot compartment weight analysis following the exact weight analyzer format."""
    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        convert_layer_names,
    )

    logger.info("Generating compartment weight analysis plots...")

    try:
        # Convert layer names to uniform format (soma to distal)
        converted_results = convert_layer_names(
            layer_statistics, somatic_synapses=somatic_synapses
        )
        if not converted_results:
            logger.warning("No layer results after conversion")
            return

        layer_names = list(converted_results.keys())
        logger.info(f"Plotting compartment weight analysis for layers: {layer_names}")

        # Create individual weight strength plot (following _plot_total_synaptic_strength format)
        _plot_compartment_weight_strength(converted_results, layer_names, save_path)

        # Create weight comparison plots (following _plot_weight_comparisons format)
        _plot_compartment_weight_comparisons(converted_results, layer_names, save_path)

        logger.info("Compartment weight analysis plots completed successfully!")

    except Exception as e:
        logger.error(f"Error generating compartment weight plots: {e}")
        import traceback

        traceback.print_exc()


def plot_compartment_input_analysis(
    layer_statistics: dict[str, dict[str, float]],
    save_path: str,
    somatic_synapses: bool = True,
):
    """Plot compartment input analysis following the exact weight analyzer format."""
    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        convert_layer_names,
    )

    logger.info("Generating compartment input analysis plots...")

    try:
        # Convert layer names to uniform format (soma to distal)
        converted_results = convert_layer_names(
            layer_statistics, somatic_synapses=somatic_synapses
        )
        if not converted_results:
            logger.warning("No layer results after conversion")
            return

        layer_names = list(converted_results.keys())
        logger.info(f"Plotting compartment input analysis for layers: {layer_names}")

        # Create input strength plot
        _plot_compartment_input_strength(converted_results, layer_names, save_path)

        # Create input comparison plots
        _plot_compartment_input_comparisons(converted_results, layer_names, save_path)

        logger.info("Compartment input analysis plots completed successfully!")

    except Exception as e:
        logger.error(f"Error generating compartment input plots: {e}")
        import traceback

        traceback.print_exc()


def plot_compartment_activation_analysis(
    layer_statistics: dict[str, dict[str, float]],
    save_path: str,
    somatic_synapses: bool = True,
):
    """Plot compartment activation analysis following the exact weight analyzer format."""
    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        convert_layer_names,
    )

    logger.info("Generating compartment activation analysis plots...")

    try:
        # Convert layer names to uniform format (soma to distal)
        converted_results = convert_layer_names(
            layer_statistics, somatic_synapses=somatic_synapses
        )
        if not converted_results:
            logger.warning("No layer results after conversion")
            return

        layer_names = list(converted_results.keys())
        logger.info(
            f"Plotting compartment activation analysis for layers: {layer_names}"
        )

        # Create activation strength plot
        _plot_compartment_activation_strength(converted_results, layer_names, save_path)

        # Create activation comparison plots
        _plot_compartment_activation_comparisons(
            converted_results, layer_names, save_path
        )

        logger.info("Compartment activation analysis plots completed successfully!")

    except Exception as e:
        logger.error(f"Error generating compartment activation plots: {e}")
        import traceback

        traceback.print_exc()


def plot_compartment_entropy_analysis(
    layer_statistics: dict[str, dict[str, float]],
    save_path: str,
    somatic_synapses: bool = True,
):
    """Plot compartment entropy analysis following the exact weight analyzer format."""
    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        convert_layer_names,
    )

    logger.info("Generating compartment entropy analysis plots...")

    try:
        # Convert layer names to uniform format (soma to distal)
        converted_results = convert_layer_names(
            layer_statistics, somatic_synapses=somatic_synapses
        )
        if not converted_results:
            logger.warning("No layer results after conversion")
            return

        layer_names = list(converted_results.keys())
        logger.info(f"Plotting compartment entropy analysis for layers: {layer_names}")

        # Create entropy comparison plot
        _plot_compartment_entropy_comparisons(converted_results, layer_names, save_path)

        logger.info("Compartment entropy analysis plots completed successfully!")

    except Exception as e:
        logger.error(f"Error generating compartment entropy plots: {e}")
        import traceback

        traceback.print_exc()


def plot_compartment_ei_comparisons(
    layer_statistics: dict[str, dict[str, float]],
    save_path: str,
    somatic_synapses: bool = True,
):
    """Plot E/I balance comparisons following the exact weight analyzer format."""
    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        convert_layer_names,
    )

    logger.info("Generating compartment E/I comparison plots...")

    try:
        # Convert layer names to uniform format (soma to distal)
        converted_results = convert_layer_names(
            layer_statistics, somatic_synapses=somatic_synapses
        )
        if not converted_results:
            logger.warning("No layer results after conversion")
            return

        layer_names = list(converted_results.keys())
        logger.info(f"Plotting compartment E/I comparisons for layers: {layer_names}")

        # Create E/I ratio comparison plot
        _plot_compartment_ei_ratios(converted_results, layer_names, save_path)

        logger.info("Compartment E/I comparison plots completed successfully!")

    except Exception as e:
        logger.error(f"Error generating compartment E/I plots: {e}")
        import traceback

        traceback.print_exc()


# Individual plotting functions following weight analyzer format
def _plot_compartment_weight_strength(
    converted_results: dict[str, Any],
    layer_names: list[str],
    save_path: str,
):
    """Plot compartment weight strength across layers (following weight analyzer format)."""
    import os

    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        get_color_scheme,
        setup_basic_plot,
    )

    colors = get_color_scheme()
    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Collect data for both excitatory and inhibitory weights
    depths = []
    exc_weights = []
    inh_weights = []
    exc_errors = []
    inh_errors = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]

        # Get weight means and stds
        exc_weight = layer_data.get("exc_weights_mean_mean", 0)
        inh_weight = layer_data.get("inh_weights_mean_mean", 0)
        exc_weight_std = layer_data.get("exc_weights_mean_std", 0)
        inh_weight_std = layer_data.get("inh_weights_mean_std", 0)

        depths.append(layer_name)
        exc_weights.append(exc_weight)
        inh_weights.append(inh_weight)
        exc_errors.append(exc_weight_std)
        inh_errors.append(inh_weight_std)

    if depths and (exc_weights or inh_weights):
        x_pos = np.arange(len(depths))
        width = 0.35

        # Plot excitatory and inhibitory bars side by side
        ax.bar(
            x_pos - width / 2,
            exc_weights,
            width,
            label="Excitatory Weights",
            color=colors["excitatory"],
            alpha=0.7,
        )
        ax.bar(
            x_pos + width / 2,
            inh_weights,
            width,
            label="Inhibitory Weights",
            color=colors["inhibitory"],
            alpha=0.7,
        )

        setup_basic_plot(
            ax,
            "Compartment Weight Strength (Mean Weight Values)",
            "Layer (Soma to Distal)",
            "Weight Strength",
            grid=True,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(depths, rotation=45)
        ax.legend()

    plt.tight_layout()

    # Save plot
    filename = os.path.join(save_path, "compartment_weight_strength.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved compartment weight strength plot: {filename}")


def _plot_compartment_weight_comparisons(
    converted_results: dict[str, Any],
    layer_names: list[str],
    save_path: str,
):
    """Plot weight comparisons across layers (following weight analyzer format)."""
    import os

    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        get_color_scheme,
        setup_basic_plot,
    )

    colors = get_color_scheme()
    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Collect excitatory vs inhibitory weight data
    exc_weights = []
    inh_weights = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]
        exc_weight = layer_data.get("exc_weights_mean_mean", 0)
        inh_weight = layer_data.get("inh_weights_mean_mean", 0)
        exc_weights.append(exc_weight)
        inh_weights.append(inh_weight)

    if exc_weights and inh_weights:
        # Create scatter plot with layer labels
        for i, layer_name in enumerate(layer_names):
            ax.scatter(
                exc_weights[i],
                inh_weights[i],
                color=colors["excitatory"] if i % 2 == 0 else colors["inhibitory"],
                s=100,
                alpha=0.7,
                label=layer_name,
            )

        # Add diagonal reference line
        if exc_weights and inh_weights:
            min_val = min(min(exc_weights), min(inh_weights))
            max_val = max(max(exc_weights), max(inh_weights))
            ax.plot(
                [min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="E=I"
            )

        setup_basic_plot(
            ax,
            "Excitatory vs Inhibitory Weight Comparison",
            "Excitatory Weight Mean",
            "Inhibitory Weight Mean",
            grid=True,
        )
        ax.legend()

    plt.tight_layout()

    # Save plot
    filename = os.path.join(save_path, "compartment_weight_comparisons.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved compartment weight comparisons plot: {filename}")


def _plot_compartment_input_strength(
    converted_results: dict[str, Any],
    layer_names: list[str],
    save_path: str,
):
    """Plot compartment input strength across layers."""
    import os

    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        get_color_scheme,
        setup_basic_plot,
    )

    colors = get_color_scheme()
    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Collect data for both excitatory and inhibitory inputs
    depths = []
    exc_inputs = []
    inh_inputs = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]

        # Get input means
        exc_input = layer_data.get("exc_inputs_mean_mean", 0)
        inh_input = layer_data.get("inh_inputs_mean_mean", 0)

        depths.append(layer_name)
        exc_inputs.append(exc_input)
        inh_inputs.append(inh_input)

    if depths and (exc_inputs or inh_inputs):
        x_pos = np.arange(len(depths))
        width = 0.35

        # Plot excitatory and inhibitory bars side by side
        ax.bar(
            x_pos - width / 2,
            exc_inputs,
            width,
            label="Excitatory Inputs",
            color=colors["excitatory"],
            alpha=0.7,
        )
        ax.bar(
            x_pos + width / 2,
            inh_inputs,
            width,
            label="Inhibitory Inputs",
            color=colors["inhibitory"],
            alpha=0.7,
        )

        setup_basic_plot(
            ax,
            "Compartment Input Strength (Mean Input Values)",
            "Layer (Soma to Distal)",
            "Input Strength",
            grid=True,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(depths, rotation=45)
        ax.legend()

    plt.tight_layout()

    # Save plot
    filename = os.path.join(save_path, "compartment_input_strength.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved compartment input strength plot: {filename}")


def _plot_compartment_input_comparisons(
    converted_results: dict[str, Any],
    layer_names: list[str],
    save_path: str,
):
    """Plot input comparisons across layers."""
    import os

    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        get_color_scheme,
        setup_basic_plot,
    )

    colors = get_color_scheme()
    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Collect excitatory vs inhibitory input data
    exc_inputs = []
    inh_inputs = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]
        exc_input = layer_data.get("exc_inputs_mean_mean", 0)
        inh_input = layer_data.get("inh_inputs_mean_mean", 0)
        exc_inputs.append(exc_input)
        inh_inputs.append(inh_input)

    if exc_inputs and inh_inputs:
        # Create scatter plot with layer labels
        for i, layer_name in enumerate(layer_names):
            ax.scatter(
                exc_inputs[i],
                inh_inputs[i],
                color=colors["excitatory"] if i % 2 == 0 else colors["inhibitory"],
                s=100,
                alpha=0.7,
                label=layer_name,
            )

        # Add diagonal reference line
        if exc_inputs and inh_inputs:
            min_val = min(min(exc_inputs), min(inh_inputs))
            max_val = max(max(exc_inputs), max(inh_inputs))
            ax.plot(
                [min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="E=I"
            )

        setup_basic_plot(
            ax,
            "Excitatory vs Inhibitory Input Comparison",
            "Excitatory Input Mean",
            "Inhibitory Input Mean",
            grid=True,
        )
        ax.legend()

    plt.tight_layout()

    # Save plot
    filename = os.path.join(save_path, "compartment_input_comparisons.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved compartment input comparisons plot: {filename}")


def _plot_compartment_activation_strength(
    converted_results: dict[str, Any],
    layer_names: list[str],
    save_path: str,
):
    """Plot compartment activation strength across layers."""
    import os

    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        get_color_scheme,
        setup_basic_plot,
    )

    colors = get_color_scheme()
    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Collect data for both excitatory and inhibitory activations
    depths = []
    exc_activations = []
    inh_activations = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]

        # Get activation means
        exc_activation = layer_data.get("exc_activations_mean_mean", 0)
        inh_activation = layer_data.get("inh_activations_mean_mean", 0)

        depths.append(layer_name)
        exc_activations.append(exc_activation)
        inh_activations.append(inh_activation)

    if depths and (exc_activations or inh_activations):
        x_pos = np.arange(len(depths))
        width = 0.35

        # Plot excitatory and inhibitory bars side by side
        ax.bar(
            x_pos - width / 2,
            exc_activations,
            width,
            label="Excitatory Activations",
            color=colors["excitatory"],
            alpha=0.7,
        )
        ax.bar(
            x_pos + width / 2,
            inh_activations,
            width,
            label="Inhibitory Activations",
            color=colors["inhibitory"],
            alpha=0.7,
        )

        setup_basic_plot(
            ax,
            "Compartment Activation Strength (Mean Activation Values)",
            "Layer (Soma to Distal)",
            "Activation Strength",
            grid=True,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(depths, rotation=45)
        ax.legend()

    plt.tight_layout()

    # Save plot
    filename = os.path.join(save_path, "compartment_activation_strength.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved compartment activation strength plot: {filename}")


def _plot_compartment_activation_comparisons(
    converted_results: dict[str, Any],
    layer_names: list[str],
    save_path: str,
):
    """Plot activation comparisons across layers."""
    import os

    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        get_color_scheme,
        setup_basic_plot,
    )

    colors = get_color_scheme()
    _, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Collect excitatory vs inhibitory activation data
    exc_activations = []
    inh_activations = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]
        exc_activation = layer_data.get("exc_activations_mean_mean", 0)
        inh_activation = layer_data.get("inh_activations_mean_mean", 0)
        exc_activations.append(exc_activation)
        inh_activations.append(inh_activation)

    if exc_activations and inh_activations:
        # Create scatter plot with layer labels
        for i, layer_name in enumerate(layer_names):
            ax.scatter(
                exc_activations[i],
                inh_activations[i],
                color=colors["excitatory"] if i % 2 == 0 else colors["inhibitory"],
                s=100,
                alpha=0.7,
                label=layer_name,
            )

        # Add diagonal reference line
        if exc_activations and inh_activations:
            min_val = min(min(exc_activations), min(inh_activations))
            max_val = max(max(exc_activations), max(inh_activations))
            ax.plot(
                [min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="E=I"
            )

        setup_basic_plot(
            ax,
            "Excitatory vs Inhibitory Activation Comparison",
            "Excitatory Activation Mean",
            "Inhibitory Activation Mean",
            grid=True,
        )
        ax.legend()

    plt.tight_layout()

    # Save plot
    filename = os.path.join(save_path, "compartment_activation_comparisons.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved compartment activation comparisons plot: {filename}")


def _plot_compartment_entropy_comparisons(
    converted_results: dict[str, Any],
    layer_names: list[str],
    save_path: str,
):
    """Plot entropy comparisons across layers."""
    import os

    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        get_color_scheme,
        setup_basic_plot,
    )

    colors = get_color_scheme()
    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Collect entropy data
    depths = []
    exc_input_entropies = []
    inh_input_entropies = []
    exc_act_entropies = []
    inh_act_entropies = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]

        # Get entropy means
        exc_input_entropy = layer_data.get("exc_inputs_entropy_mean", 0)
        inh_input_entropy = layer_data.get("inh_inputs_entropy_mean", 0)
        exc_act_entropy = layer_data.get("exc_activations_entropy_mean", 0)
        inh_act_entropy = layer_data.get("inh_activations_entropy_mean", 0)

        depths.append(layer_name)
        exc_input_entropies.append(exc_input_entropy)
        inh_input_entropies.append(inh_input_entropy)
        exc_act_entropies.append(exc_act_entropy)
        inh_act_entropies.append(inh_act_entropy)

    if depths:
        x_pos = np.arange(len(depths))

        # Plot entropy lines
        ax.plot(
            x_pos,
            exc_input_entropies,
            "o-",
            color=colors["excitatory"],
            label="Exc Input Entropy",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            x_pos,
            inh_input_entropies,
            "s-",
            color=colors["inhibitory"],
            label="Inh Input Entropy",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            x_pos,
            exc_act_entropies,
            "^--",
            color=colors["excitatory"],
            label="Exc Activation Entropy",
            linewidth=2,
            markersize=8,
            alpha=0.6,
        )
        ax.plot(
            x_pos,
            inh_act_entropies,
            "v--",
            color=colors["inhibitory"],
            label="Inh Activation Entropy",
            linewidth=2,
            markersize=8,
            alpha=0.6,
        )

        setup_basic_plot(
            ax,
            "Compartment Entropy Analysis (Information Content)",
            "Layer (Soma to Distal)",
            "Entropy",
            grid=True,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(depths, rotation=45)
        ax.legend()

    plt.tight_layout()

    # Save plot
    filename = os.path.join(save_path, "compartment_entropy_analysis.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved compartment entropy analysis plot: {filename}")


def _plot_compartment_ei_ratios(
    converted_results: dict[str, Any],
    layer_names: list[str],
    save_path: str,
):
    """Plot E/I ratios across layers."""
    import os

    from dendritic_modeling.plotting.visualizations.plotting_utils import (
        get_color_scheme,
        setup_basic_plot,
    )

    colors = get_color_scheme()
    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Collect E/I ratio data
    depths = []
    weight_ratios = []
    input_ratios = []
    activation_ratios = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]

        # Calculate E/I ratios
        exc_w = layer_data.get("exc_weights_mean_mean", 0)
        inh_w = layer_data.get("inh_weights_mean_mean", 1e-8)
        weight_ratio = exc_w / inh_w

        exc_i = layer_data.get("exc_inputs_mean_mean", 0)
        inh_i = layer_data.get("inh_inputs_mean_mean", 1e-8)
        input_ratio = exc_i / inh_i

        exc_a = layer_data.get("exc_activations_mean_mean", 0)
        inh_a = layer_data.get("inh_activations_mean_mean", 1e-8)
        activation_ratio = exc_a / inh_a

        depths.append(layer_name)
        weight_ratios.append(weight_ratio)
        input_ratios.append(input_ratio)
        activation_ratios.append(activation_ratio)

    if depths:
        x_pos = np.arange(len(depths))

        # Plot E/I ratio lines
        ax.plot(
            x_pos,
            weight_ratios,
            "o-",
            color=colors["primary"],
            label="Weight E/I",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            x_pos,
            input_ratios,
            "s-",
            color=colors["excitatory"],
            label="Input E/I",
            linewidth=2,
            markersize=8,
        )
        ax.plot(
            x_pos,
            activation_ratios,
            "^-",
            color=colors["inhibitory"],
            label="Activation E/I",
            linewidth=2,
            markersize=8,
        )

        # Add reference line at E=I
        ax.axhline(y=1, color="k", linestyle="--", alpha=0.5, label="E=I")

        setup_basic_plot(
            ax,
            "Compartment E/I Balance Analysis",
            "Layer (Soma to Distal)",
            "E/I Ratio",
            grid=True,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(depths, rotation=45)
        ax.legend()

    plt.tight_layout()

    # Save plot
    filename = os.path.join(save_path, "compartment_ei_ratios.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved compartment E/I ratios plot: {filename}")


def _aggregate_branch_to_layer_stats(branch_statistics):
    """Convert branch statistics to layer statistics format."""
    layer_stats = {}

    # Group by depth
    depth_groups = {}
    for _layer_key, layer_data in branch_statistics.items():
        for _branch_idx, branch_data in layer_data.items():
            depth = branch_data["depth"]
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(branch_data)

    # Aggregate by depth (layer)
    for depth, branches in depth_groups.items():
        layer_name = f"branch_layers.{depth}"
        layer_stats[layer_name] = {}

        # Collect all metrics
        all_metrics = set()
        for branch in branches:
            all_metrics.update(branch.keys())
        all_metrics.discard("depth")

        # Aggregate each metric
        for metric in all_metrics:
            values = [branch.get(metric, 0) for branch in branches if metric in branch]
            if values:
                layer_stats[layer_name][f"{metric}_mean"] = np.mean(values)
                layer_stats[layer_name][f"{metric}_std"] = np.std(values)
                layer_stats[layer_name][f"{metric}_layer_min"] = np.min(values)
                layer_stats[layer_name][f"{metric}_layer_max"] = np.max(values)

    return layer_stats


def plot_per_class_analysis(
    branch_statistics: dict[str, dict[int, dict[str, float]]], save_path: str
):
    """Generate per-class analysis plots."""
    # Extract per-class data
    per_class_data = {}
    classes = set()

    for _layer_key, layer_data in branch_statistics.items():
        for _branch_idx, branch_data in layer_data.items():
            depth = branch_data["depth"]
            if f"depth {depth}" not in per_class_data:
                per_class_data[f"depth {depth}"] = {}

            # Find all per-class metrics
            for key, value in branch_data.items():
                if "_class_" in key:
                    parts = key.split("_class_")
                    if len(parts) == 2:
                        metric_name = parts[0]
                        class_idx = int(
                            parts[1].split("_")[0]
                        )  # Handle cases like "class_0_mean"
                        classes.add(class_idx)

                        if metric_name not in per_class_data[f"depth {depth}"]:
                            per_class_data[f"depth {depth}"][metric_name] = {}
                        if (
                            class_idx
                            not in per_class_data[f"depth {depth}"][metric_name]
                        ):
                            per_class_data[f"depth {depth}"][metric_name][
                                class_idx
                            ] = []

                        per_class_data[f"depth {depth}"][metric_name][class_idx].append(
                            value
                        )

    if not per_class_data or not classes:
        return

    classes = sorted(classes)

    # Create per-class comparison plots
    metrics_to_plot = [
        "exc_inputs_per_class_mean",
        "inh_inputs_per_class_mean",
        "exc_activations_per_class_mean",
        "inh_activations_per_class_mean",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot):
        if i >= len(axes):
            break

        ax = axes[i]

        # Plot data for each depth
        for depth_key in sorted(per_class_data.keys()):
            if metric in per_class_data[depth_key]:
                class_means = []
                class_stds = []
                for class_idx in classes:
                    if class_idx in per_class_data[depth_key][metric]:
                        values = per_class_data[depth_key][metric][class_idx]
                        class_means.append(np.mean(values))
                        class_stds.append(np.std(values))
                    else:
                        class_means.append(0)
                        class_stds.append(0)

                ax.errorbar(
                    classes,
                    class_means,
                    yerr=class_stds,
                    label=depth_key.title(),
                    marker="o",
                    alpha=0.7,
                )

        ax.set_xlabel("Class Index")
        ax.set_ylabel("Mean Value")
        ax.set_title(metric.replace("_", " ").title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(classes)

    plt.suptitle("Per-Class Analysis: Compartment Statistics", fontsize=16)
    plt.tight_layout()
    fig.savefig(
        os.path.join(save_path, "per_class_analysis.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def plot_ratio_comparison(
    branch_statistics: dict[str, dict[int, dict[str, float]]], save_path: str
):
    agg_data: dict[str, dict[str, list[float]]] = {}
    for _layer_key, layer_data in branch_statistics.items():
        for branch_data in layer_data.values():
            depth = branch_data["depth"]
            if f"depth {depth}" not in agg_data.keys():
                agg_data[f"depth {depth}"] = {}

            if "ei_expected_act_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"]["ei_expected_act_ratio"] = []
            if "ei_act_mean_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"]["ei_act_mean_ratio"] = []

            if "ei_acts_eff_std_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"]["ei_acts_eff_std_ratio"] = []
            if "ei_acts_emp_std_ratio" not in agg_data[f"depth {depth}"].keys():
                agg_data[f"depth {depth}"]["ei_acts_emp_std_ratio"] = []

            ei_expected_act_ratio = (
                branch_data["exc_inputs_mean"] * branch_data["exc_weights_mean"]
            ) / (
                branch_data["inh_inputs_mean"] * branch_data["inh_weights_mean"] + 1e-8
            )
            agg_data[f"depth {depth}"]["ei_expected_act_ratio"].append(
                ei_expected_act_ratio
            )

            ei_act_mean_ratio = branch_data["exc_activations_mean"] / (
                branch_data["inh_activations_mean"] + 1e-8
            )
            agg_data[f"depth {depth}"]["ei_act_mean_ratio"].append(ei_act_mean_ratio)

            ei_acts_entropy_diff = (
                branch_data["exc_activations_entropy"]
                - branch_data["inh_activations_entropy"]
            )
            ei_acts_eff_std_ratio = np.exp(ei_acts_entropy_diff)
            agg_data[f"depth {depth}"]["ei_acts_eff_std_ratio"].append(
                ei_acts_eff_std_ratio
            )
            agg_data[f"depth {depth}"]["ei_acts_emp_std_ratio"].append(
                branch_data["exc_activations_std"]
                / (branch_data["inh_activations_std"] + 1e-8)
            )

    indeps = ["ei_expected_act_ratio", "ei_act_mean_ratio"]
    deps = ["ei_acts_eff_std_ratio", "ei_acts_emp_std_ratio"]

    # Build long-form DataFrame from agg_data
    rows = []
    for depth, data in agg_data.items():
        for dep in deps:  # y
            for indep in indeps:  # x
                x = np.asarray(data[indep])
                y = np.asarray(data[dep])
                rows.append(
                    pd.DataFrame(
                        {"x": x, "y": y, "depth": depth, "row": dep, "col": indep}
                    )
                )
    df = pd.concat(rows, ignore_index=True)

    # Labels
    xlabels = {
        "ei_expected_act_ratio": r"$\frac{X_{exc}}{X_{inh}}*\frac{W_{exc}}{W_{inh}}$",
        "ei_act_mean_ratio": r"$\frac{E}{I}$",
    }
    ylabels = {
        "ei_acts_eff_std_ratio": r"$\sqrt{\frac{N(E)}{N(I)}}$",
        "ei_acts_emp_std_ratio": r"$\frac{\sigma_{E}}{\sigma_{I}}$",
    }

    # Scatter faceted by (row=dep, col=indep) with hue=depth
    g = sns.relplot(
        data=df,
        x="x",
        y="y",
        hue="depth",
        row="row",
        col="col",
        kind="scatter",
        row_order=deps,
        col_order=indeps,
        facet_kws={"sharex": "col", "sharey": "row"},
        height=4,
        aspect=1.2,
        s=10,
        alpha=0.5,
    )

    # Add one global regression line per facet and annotate slope/intercept
    def add_regline_and_stats(data: pd.DataFrame, color: Optional[str] = None, **kws):
        ax = plt.gca()
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            slope, intercept = np.polyfit(x[m], y[m], 1)
            xx = np.linspace(x[m].min(), x[m].max(), 200)
            ax.plot(xx, slope * xx + intercept, color="k", lw=2)
            ax.text(
                0.02,
                0.98,
                f"y={slope:.2g}x+{intercept:.2g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="k",
            )

    g.map_dataframe(add_regline_and_stats)

    # Axis labels per row/col (only outer axes)
    for i, row_key in enumerate(deps):
        for j, col_key in enumerate(indeps):
            ax = g.axes[i, j]
            ax.grid(True)
            if i == len(deps) - 1:
                ax.set_xlabel(xlabels[col_key])
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(ylabels[row_key])
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    g.figure.savefig(
        os.path.join(save_path, "another_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(g.figure)


def plot_ratio_pairwise_comparison(
    branch_statistics: dict[str, dict[int, dict[str, float]]],
    weights_ratio: bool = False,
    inputs_ratio: bool = False,
    activations_ratio: bool = False,
    save_path: Optional[str] = None,
):
    if save_path is None:
        logger.warning("save_path is None; skipping ratio pairwise comparison plot")
        return

    agg_data: dict[str, dict[str, list[float]]] = {}
    for _layer_key, layer_data in branch_statistics.items():
        for branch_data in layer_data.values():
            depth = branch_data["depth"]
            if f"depth {depth}" not in agg_data.keys():
                agg_data[f"depth {depth}"] = {}

            if (
                weights_ratio
                and "ei_w_mean_ratio" not in agg_data[f"depth {depth}"].keys()
            ):
                agg_data[f"depth {depth}"]["ei_w_mean_ratio"] = []
            if (
                weights_ratio
                and "ei_w_std_ratio" not in agg_data[f"depth {depth}"].keys()
            ):
                agg_data[f"depth {depth}"]["ei_w_std_ratio"] = []

            if (
                inputs_ratio
                and "ei_x_mean_ratio" not in agg_data[f"depth {depth}"].keys()
            ):
                agg_data[f"depth {depth}"]["ei_x_mean_ratio"] = []
            if (
                inputs_ratio
                and "ei_x_std_ratio" not in agg_data[f"depth {depth}"].keys()
            ):
                agg_data[f"depth {depth}"]["ei_x_std_ratio"] = []

            if (
                weights_ratio
                and inputs_ratio
                and "ei_wx_mean_ratio" not in agg_data[f"depth {depth}"].keys()
            ):
                agg_data[f"depth {depth}"]["ei_wx_mean_ratio"] = []
            if (
                weights_ratio
                and inputs_ratio
                and "ei_wx_std_ratio" not in agg_data[f"depth {depth}"].keys()
            ):
                agg_data[f"depth {depth}"]["ei_wx_std_ratio"] = []

            if (
                activations_ratio
                and "ei_a_mean_ratio" not in agg_data[f"depth {depth}"].keys()
            ):
                agg_data[f"depth {depth}"]["ei_a_mean_ratio"] = []
            if (
                activations_ratio
                and "ei_a_std_ratio" not in agg_data[f"depth {depth}"].keys()
            ):
                agg_data[f"depth {depth}"]["ei_a_std_ratio"] = []

            if weights_ratio:
                ei_w_mean_ratio = branch_data["exc_weights_mean"] / (
                    branch_data["inh_weights_mean"] + 1e-8
                )
                agg_data[f"depth {depth}"]["ei_w_mean_ratio"].append(ei_w_mean_ratio)

                ei_w_std_ratio = branch_data["exc_weights_std"] / (
                    branch_data["inh_weights_std"] + 1e-8
                )
                agg_data[f"depth {depth}"]["ei_w_std_ratio"].append(ei_w_std_ratio)

            if inputs_ratio:
                ei_x_mean_ratio = branch_data["exc_inputs_mean"] / (
                    branch_data["inh_inputs_mean"] + 1e-8
                )
                agg_data[f"depth {depth}"]["ei_x_mean_ratio"].append(ei_x_mean_ratio)

                ei_x_std_ratio = branch_data["exc_inputs_std"] / (
                    branch_data["inh_inputs_std"] + 1e-8
                )
                agg_data[f"depth {depth}"]["ei_x_std_ratio"].append(ei_x_std_ratio)

            if weights_ratio and inputs_ratio:
                ei_wx_mean_ratio = ei_w_mean_ratio * ei_x_mean_ratio
                agg_data[f"depth {depth}"]["ei_wx_mean_ratio"].append(ei_wx_mean_ratio)

                ei_wx_std_ratio = ei_w_std_ratio * ei_x_std_ratio
                agg_data[f"depth {depth}"]["ei_wx_std_ratio"].append(ei_wx_std_ratio)

            if activations_ratio:
                ei_a_mean_ratio = branch_data["exc_activations_mean"] / (
                    branch_data["inh_activations_mean"] + 1e-8
                )
                agg_data[f"depth {depth}"]["ei_a_mean_ratio"].append(ei_a_mean_ratio)

                ei_a_std_ratio = branch_data["exc_activations_std"] / (
                    branch_data["inh_activations_std"] + 1e-8
                )
                agg_data[f"depth {depth}"]["ei_a_std_ratio"].append(ei_a_std_ratio)

    available = []
    if weights_ratio:
        available.extend(["ei_w_mean_ratio", "ei_w_std_ratio"])
    if inputs_ratio:
        available.extend(["ei_x_mean_ratio", "ei_x_std_ratio"])
    if weights_ratio and inputs_ratio:
        available.extend(["ei_wx_mean_ratio", "ei_wx_std_ratio"])
    if activations_ratio:
        available.extend(["ei_a_mean_ratio", "ei_a_std_ratio"])

    # Build long-form DataFrame from agg_data
    rows = []
    for depth, data in agg_data.items():
        for dep in available:  # y
            for indep in available:  # x
                x = np.asarray(data[indep])
                y = np.asarray(data[dep])
                rows.append(
                    pd.DataFrame(
                        {"x": x, "y": y, "depth": depth, "row": dep, "col": indep}
                    )
                )
    df = pd.concat(rows, ignore_index=True)

    label_map = {
        "ei_w_mean_ratio": r"$\frac{\bar W_E}{\bar W_I}$",
        "ei_w_std_ratio": r"$\frac{\sigma(W_E)}{\sigma(W_I)}$",
        "ei_x_mean_ratio": r"$\frac{\bar X_E}{\bar X_I}$",
        "ei_x_std_ratio": r"$\frac{\sigma(X_E)}{\sigma(X_I)}$",
        "ei_wx_mean_ratio": r"$\frac{\bar W_E}{\bar W_I}\cdot\frac{\bar X_E}{\bar X_I}$",
        "ei_wx_std_ratio": r"$\frac{\sigma(W_E)}{\sigma(W_I)}\cdot\frac{\sigma(X_E)}{\sigma(X_I)}$",
        "ei_a_mean_ratio": r"$\frac{\bar A_E}{\bar A_I}$",
        "ei_a_std_ratio": r"$\frac{\sigma(A_E)}{\sigma(A_I)}$",
    }

    # Scatter faceted by (row=dep, col=indep) with hue=depth
    g = sns.relplot(
        data=df,
        x="x",
        y="y",
        hue="depth",
        row="row",
        col="col",
        kind="scatter",
        row_order=available,
        col_order=available,
        facet_kws={"sharex": "col", "sharey": "row"},
        height=4,
        aspect=1.2,
        s=10,
        alpha=0.5,
    )

    # Add one global regression line per facet and annotate slope/intercept
    def add_regline_and_stats(data: pd.DataFrame, color: Optional[str] = None, **kws):
        ax = plt.gca()
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            slope, intercept = np.polyfit(x[m], y[m], 1)
            xx = np.linspace(x[m].min(), x[m].max(), 200)
            ax.plot(xx, slope * xx + intercept, color="k", lw=2)
            ax.text(
                0.02,
                0.98,
                f"y={slope:.2g}x+{intercept:.2g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="k",
            )

    g.map_dataframe(add_regline_and_stats)

    # Axis labels per row/col (only outer axes)
    for i, row_key in enumerate(available):
        for j, col_key in enumerate(available):
            ax = g.axes[i, j]
            ax.grid(True)
            if i == len(available) - 1:
                ax.set_xlabel(label_map[col_key])
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(label_map[row_key])
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    g.figure.savefig(
        os.path.join(save_path, "ratio_pairwise_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(g.figure)


__all__ = ["CompartmentStatisticsAnalyzer"]
