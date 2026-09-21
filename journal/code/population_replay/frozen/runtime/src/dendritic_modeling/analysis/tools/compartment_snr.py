"""Compartment signal-to-noise analysis.

This module computes discriminability metrics for dendritic inputs, branch
activations, and layer outputs, with global, per-layer, and per-branch
aggregation plus plotting helpers for final and training-time summaries.
"""

import logging
import os
import time
from typing import Optional, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.tools.epoch_aggregation import (
    initialize_epoch_aggregation_structure,
    load_epoch_aggregation,
    populate_epoch_aggregation,
)
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    materialize_dataset,
)
from dendritic_modeling.config import CompartmentSNRAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel, Classifier
from dendritic_modeling.networks import (
    DendriNet,
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
    ParametricActivation,
    TopKLinear,
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


METRIC_MAPPING = {
    "d_max": "Max Discriminability",
    "d_total": "Total Discriminability",
    "n_sig_dirs": "Number of Significant Directions",
    "ratio_sig_dirs": "Ratio of Significant Directions",
}


def _aggregate_non_depth_values(records) -> dict[str, list[float]]:
    """Collect non-depth scalar fields across branch or layer records."""
    aggregated: dict[str, list[float]] = {}
    for record in records:
        for field, value in record.items():
            if field not in aggregated.keys() and field != "depth":
                aggregated[field] = []
            if field != "depth":
                aggregated[field].append(value)
    return aggregated


def _summarize_population_values(
    aggregated_values: dict[str, list[float]],
    stats_dict: dict[str, float],
) -> dict[str, float]:
    """Summarize aggregated scalar metrics using population reductions."""
    for field, values in aggregated_values.items():
        data_tensor = torch.tensor(values)
        stats_dict[f"{field}_mean"] = data_tensor.mean().item()
        stats_dict[f"{field}_std"] = data_tensor.std(correction=0).item()
    return stats_dict


class CompartmentSNRAnalyzer(ForwardHookRemovalMixin, AbstractAnalyzer):
    """
    Unified analyzer for comprehensive compartment SNR analysis.

    Provides both global SNR statistics and hierarchical per-layer analysis
    with enhanced plotting capabilities including SNR vs. depth plots.
    """

    def __init__(self, params: CompartmentSNRAnalysisParams):
        """Initialize the unified compartment SNR analyzer."""
        super().__init__("CompartmentSNRAnalyzer")
        # Enhanced analysis modes
        self.input_analysis = getattr(params, "input_analysis", True)
        self.layer_analysis = getattr(params, "layer_analysis", True)
        self.branch_analysis = getattr(params, "branch_analysis", True)
        self.input_capture_mode = (
            str(getattr(params, "input_capture_mode", "branch"))
            .strip()
            .lower()
            .replace("-", "_")
        )
        if self.input_capture_mode in {"branch_inputs", "branch_input"}:
            self.input_capture_mode = "branch"
        elif self.input_capture_mode in {"network_inputs", "network_input"}:
            self.input_capture_mode = "network"
        if self.input_capture_mode not in {"branch", "network"}:
            raise ValueError(
                "input_capture_mode must be 'branch' or 'network', "
                f"got {self.input_capture_mode!r}"
            )

        self.global_aggregation = getattr(params, "global_aggregation", True)
        self.layer_aggregation = getattr(params, "layer_aggregation", True)
        self.branch_aggregation = getattr(params, "branch_aggregation", True)

        self.epsilon = getattr(params, "epsilon", torch.finfo(torch.float32).eps)

    def collect_raw_data(self, model: BaseModel, x: torch.Tensor):
        """Collect raw data from the model."""
        self.raw_dict = {}

        core_network = getattr(model, "core_network", model)
        direct_ei_core = isinstance(core_network, ExcitationInhibitionNetwork)
        branch_root = (
            core_network.layers[-1].excitatory_cells if direct_ei_core else core_network
        )

        def _attach_hooks():
            handles = []
            if (
                self.input_analysis
                and self.input_capture_mode == "network"
                and direct_ei_core
            ):
                handles.extend(self._attach_network_input_hooks(core_network))
            handles.extend(
                register_named_forward_hook_groups(
                    branch_root,
                    DendriticBranchLayer,
                    self._collect_layer_raw_data,
                )
            )
            return handles

        def _run_model():
            with torch.no_grad():
                _ = model(x)

        run_with_forward_hooks(
            attach=_attach_hooks,
            remove=self.remove_forward_hooks,
            body=_run_model,
        )

    def _attach_network_input_hooks(
        self,
        core_network: ExcitationInhibitionNetwork,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        handles = [
            core_network.register_forward_hook(
                lambda module, input, output: self.network_input_hook(
                    module, input, output
                )
            )
        ]
        inhibitory_cells = core_network.layers[-1].inhibitory_cells
        if isinstance(inhibitory_cells, DendriNet):
            handles.append(
                inhibitory_cells.register_forward_hook(
                    lambda module, input, output: self.network_input_hook(
                        module, input, output
                    )
                )
            )
        return handles

    def _collect_layer_raw_data(
        self,
        key: str,
        module: DendriticBranchLayer,
    ) -> list[torch.utils.hooks.RemovableHandle]:
        self._initialize_branch_records(key, module)
        return self._attach_layer_capture_hooks(key, module)

    def _initialize_branch_records(
        self, key: str, module: DendriticBranchLayer
    ) -> None:
        self.raw_dict[key] = {}
        for branch_idx in range(module.n_branches):
            self.raw_dict[key][branch_idx] = {"depth": module.layer_idx}

    def _register_activation_hook(
        self,
        module: torch.nn.Module,
        key: str,
        activation_type: str,
    ) -> torch.utils.hooks.RemovableHandle:
        return module.register_forward_hook(
            lambda module, input, output, layer_key=key: self.forward_hook(
                module, input, output, layer_key, activation_type
            )
        )

    def _attach_layer_capture_hooks(
        self, key: str, module: DendriticBranchLayer
    ) -> list[torch.utils.hooks.RemovableHandle]:
        def _hook_specs():
            if self.input_analysis and self.input_capture_mode == "branch":
                yield ("input", module, "")
            if module.branch_excitation is not None:
                yield ("activation", module.branch_excitation, "exc")
            if module.branch_inhibition is not None:
                yield ("activation", module.branch_inhibition, "inh")
            if hasattr(module, "branches_to_output"):
                yield ("activation", module.branches_to_output, "upstream")
            yield ("activation", module.reactivation, "vinf")
            yield ("activation", module.reactivation, "vout")

        def _register_hook(
            hook_spec: tuple[str, torch.nn.Module, str],
        ) -> list[torch.utils.hooks.RemovableHandle]:
            hook_kind, hook_module, activation_type = hook_spec
            if hook_kind == "input":
                return [
                    hook_module.register_forward_pre_hook(
                        lambda module, input, layer_key=key: self.branch_input_hook(
                            module, input, layer_key
                        )
                    )
                ]
            return [self._register_activation_hook(hook_module, key, activation_type)]

        return register_hook_groups(
            _hook_specs(),
            _register_hook,
        )

    def _capture_input_once(self, name: str, tensor: torch.Tensor | None) -> None:
        if tensor is None or hasattr(self, name):
            return
        if not torch.is_tensor(tensor):
            return
        setattr(self, name, tensor.detach().cpu())

    def network_input_hook(
        self,
        module: Union[ExcitationInhibitionNetwork, DendriNet],
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Capture network-level inputs as a separate optional SNR mode."""
        if isinstance(module, ExcitationInhibitionNetwork):
            if input:
                self._capture_input_once("network_excitatory_inputs", input[0])
        elif isinstance(module, DendriNet):
            self._capture_input_once("network_inhibitory_outputs", output)

    def branch_input_hook(
        self,
        module: DendriticBranchLayer,
        input: tuple[torch.Tensor, ...],
        key: str,
    ) -> None:
        """Capture the actual tensors entering a branch layer.

        This is intentionally attached to ``DendriticBranchLayer`` rather than
        to the whole E/I network.  It therefore records direct inhibitory
        streams from ``input_mode=1`` as inhibitory inputs, while explicit
        inhibitory-cell outputs are captured only when they are actually passed
        into the analyzed excitatory dendrites.
        """
        if not input:
            return
        self._capture_input_once("excitatory_inputs", input[0])
        if len(input) > 1:
            self._capture_input_once("inhibitory_inputs", input[1])
        if len(input) > 2:
            self._capture_input_once("upstream_inputs", input[2])

    def forward_hook(
        self,
        module: Union[TopKLinear, ParametricActivation, torch.nn.Identity],
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        key: str,
        activation_type: str,
    ):
        if self.input_analysis and self.input_capture_mode == "branch":
            if activation_type == "exc" and not hasattr(self, "excitatory_inputs"):
                self._capture_input_once("excitatory_inputs", input[0])
            elif activation_type == "inh" and not hasattr(self, "inhibitory_inputs"):
                self._capture_input_once("inhibitory_inputs", input[0])

        for branch_idx in range(output.shape[-1]):
            if activation_type == "vinf":
                branch_activation = input[0][..., branch_idx].detach().cpu()
            else:  # "exc", "inh", "vout"
                branch_activation = output[..., branch_idx].detach().cpu()

            self.raw_dict[key][branch_idx][activation_type] = branch_activation

    def _input_analysis(self, labels: torch.Tensor):
        results = {}

        if hasattr(self, "excitatory_inputs"):
            results = self._compute_discriminability_multivariate(
                data_tensor=self.excitatory_inputs,
                labels=labels,
                prefix="exc_inputs",
                stats_dict=results,
            )
            delattr(self, "excitatory_inputs")

        if hasattr(self, "inhibitory_inputs"):
            results = self._compute_discriminability_multivariate(
                data_tensor=self.inhibitory_inputs,
                labels=labels,
                prefix="inh_inputs",
                stats_dict=results,
            )
            delattr(self, "inhibitory_inputs")

        if hasattr(self, "upstream_inputs"):
            results = self._compute_discriminability_multivariate(
                data_tensor=self.upstream_inputs,
                labels=labels,
                prefix="upstream_inputs",
                stats_dict=results,
            )
            delattr(self, "upstream_inputs")

        if hasattr(self, "network_excitatory_inputs"):
            results = self._compute_discriminability_multivariate(
                data_tensor=self.network_excitatory_inputs,
                labels=labels,
                prefix="network_exc_inputs",
                stats_dict=results,
            )
            delattr(self, "network_excitatory_inputs")

        if hasattr(self, "network_inhibitory_outputs"):
            results = self._compute_discriminability_multivariate(
                data_tensor=self.network_inhibitory_outputs,
                labels=labels,
                prefix="network_inh_outputs",
                stats_dict=results,
            )
            delattr(self, "network_inhibitory_outputs")

        return results

    def _layer_analysis(self, labels: torch.Tensor):
        """Analyze layers of the dendritic network."""

        layer_statistics = {}
        for layer_key, layer_data in self.raw_dict.items():
            layer_data: dict[int, dict[str, torch.Tensor]]
            layer_agg: dict[str, list[torch.Tensor] | torch.Tensor] = {}
            for _branch_idx, branch_data in layer_data.items():
                for prefix, data_tensor in branch_data.items():
                    if prefix != "depth":
                        if prefix not in layer_agg.keys():
                            layer_agg[prefix] = []
                        layer_agg[prefix].append(data_tensor)

            layer_stats = {"depth": branch_data["depth"]}
            for prefix, tensor_list in layer_agg.items():
                data_tensor = torch.stack(tensor_list, dim=-1)
                layer_stats = self._compute_discriminability_multivariate(
                    data_tensor, labels, prefix, layer_stats
                )
            layer_statistics[layer_key] = layer_stats

        results = {}

        if self.global_aggregation:
            global_statistics = self._global_layer_aggregation(layer_statistics)
            results["global_statistics"] = global_statistics

        if self.layer_aggregation:
            results["layer_statistics"] = layer_statistics

        return results

    def _compute_discriminability_multivariate(
        self,
        data_tensor: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
        stats_dict: dict[str, float],
    ) -> dict[str, float]:
        """Compute discriminability for a multivariate activation tensor."""
        if data_tensor.ndim == 1:
            data_tensor = data_tensor[:, None]
        else:
            data_tensor = data_tensor.reshape(data_tensor.shape[0], -1)
        n_samples = data_tensor.shape[0]
        n_dims = data_tensor.shape[1]

        if n_dims == 1:
            total_mean = data_tensor.mean(dim=0)
            total_var = data_tensor[:, 0].var(correction=0)
            between_var = data_tensor.new_tensor(0.0)
            for label in torch.unique(labels):
                class_data = data_tensor[labels == label]
                class_prob = class_data.shape[0] / max(n_samples, 1)
                diff = class_data.mean(dim=0) - total_mean
                between_var = between_var + class_prob * diff[0].pow(2)

            d2 = (between_var / (total_var + self.epsilon)).item()
            stats_dict[f"{prefix}_d_max"] = d2
            stats_dict[f"{prefix}_d_total"] = d2
            stats_dict[f"{prefix}_n_sig_dirs"] = float(d2 > self.epsilon)
            stats_dict[f"{prefix}_ratio_sig_dirs"] = float(d2 > self.epsilon)
            return stats_dict

        # cov = torch.cov(data_tensor.T, correction=0)
        # eigvals_cov, eigvecs_cov = torch.linalg.eigh(cov)
        # inv_sqrt_e = torch.diag(
        #     torch.clamp(eigvals_cov, min=1e-3).rsqrt()
        # )
        # cov_inv_sqrt = eigvecs_cov @ inv_sqrt_e @ eigvecs_cov.T

        # total_mean = data_tensor.mean(dim=0)

        # n_samples = data_tensor.shape[0]
        # n_dims = data_tensor.shape[1]

        # cov_b = torch.zeros_like(cov)

        # for label in torch.unique(labels):
        #     class_data = data_tensor[labels == label]

        #     class_mean = class_data.mean(dim=0)
        #     class_prob = class_data.shape[0] / n_samples

        #     diff_means = (class_mean - total_mean)[:, None]
        #     cov_b += class_prob * (diff_means @ diff_means.T)

        # F = cov_inv_sqrt @ cov_b @ cov_inv_sqrt.T
        # eigvals_F = torch.linalg.eigvalsh(F)
        # eigvals_F = torch.where(eigvals_F > self.epsilon, eigvals_F, 0)

        # d_max = eigvals_F[-1].item()
        # d_total = eigvals_F.sum().item()

        # n_sig_dims = (eigvals_F > self.epsilon).sum().float().item()
        # ratio_sig_dims = n_sig_dims / n_dims

        # stats_dict[f"{prefix}_d_max"] = d_max
        # stats_dict[f"{prefix}_d_total"] = d_total
        # stats_dict[f"{prefix}_n_sig_dims"] = n_sig_dims
        # stats_dict[f"{prefix}_ratio_sig_dims"] = ratio_sig_dims

        cov = torch.cov(data_tensor.T, correction=0)
        inv_cov: torch.Tensor = torch.linalg.pinv(cov)

        cov_b = torch.zeros_like(cov)
        total_mean = data_tensor.mean(dim=0)

        for label in torch.unique(labels):
            class_data = data_tensor[labels == label]

            class_mean = class_data.mean(dim=0)
            class_prob = class_data.shape[0] / n_samples

            diff_means = (class_mean - total_mean)[:, None]
            cov_b += class_prob * (diff_means @ diff_means.T)

        F = inv_cov @ cov_b

        eigvals_F: torch.Tensor = torch.linalg.eigvals(F)
        eigvals_F = eigvals_F.real

        d_max = torch.max(eigvals_F).item()
        d_total = eigvals_F.sum().item()

        n_sig_dirs = (eigvals_F > self.epsilon).sum().float().item()
        ratio_sig_dirs = n_sig_dirs / n_dims

        stats_dict[f"{prefix}_d_max"] = d_max
        stats_dict[f"{prefix}_d_total"] = d_total
        stats_dict[f"{prefix}_n_sig_dirs"] = n_sig_dirs
        stats_dict[f"{prefix}_ratio_sig_dirs"] = ratio_sig_dirs

        return stats_dict

    def _global_layer_aggregation(
        self, layer_statistics: dict[str, dict[str, int | float]]
    ):
        """Compute global statistics from layer statistics."""
        global_aggregated = self._aggregate_stat_values(layer_statistics.values())
        return self._summarize_aggregated_values(global_aggregated, {})

    def _branch_analysis(self, labels: torch.Tensor):
        """Analyze branches of the dendritic network."""

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
                    else:
                        branch_stats = self._compute_discriminability_univariate(
                            data_tensor, labels, prefix, branch_stats
                        )

                branch_statistics[layer_key][branch_idx] = branch_stats

        results = {}

        if self.global_aggregation:
            global_statistics = self._global_branch_aggregation(branch_statistics)
            results["global_statistics"] = global_statistics

        if self.layer_aggregation:
            layer_statistics = self._layer_branch_aggregation(branch_statistics)
            results["layer_statistics"] = layer_statistics

        if self.branch_aggregation:
            results["branch_statistics"] = branch_statistics

        return results

    def _compute_discriminability_univariate(
        self,
        data_tensor: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
        stats_dict: dict[str, float],
    ) -> dict[str, float]:
        """Compute discriminability for a univariate activation tensor."""
        total_mean = data_tensor.mean().item()
        total_var = data_tensor.var(correction=0).item()

        n_samples = data_tensor.shape[0]
        var_E_C = 0

        for label in torch.unique(labels):
            class_data = data_tensor[labels == label]

            class_mean = class_data.mean().item()
            class_prob = class_data.shape[0] / n_samples
            var_E_C += class_prob * (class_mean - total_mean) ** 2

        stats_dict[f"{prefix}_d'^2"] = var_E_C / (total_var + self.epsilon)

        return stats_dict

    def _compute_discriminality_univariate(
        self,
        data_tensor: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
        stats_dict: dict[str, float],
    ) -> dict[str, float]:
        """Backward-compatible alias for the corrected method name."""
        return self._compute_discriminability_univariate(
            data_tensor,
            labels,
            prefix,
            stats_dict,
        )

    def _global_branch_aggregation(
        self, branch_statistics: dict[str, dict[int, dict[str, float]]]
    ):
        """Compute global statistics from branch statistics."""
        global_aggregated: dict[str, list[float]] = {}
        for layer_data in branch_statistics.values():
            layer_aggregated = self._aggregate_stat_values(layer_data.values())
            for field, values in layer_aggregated.items():
                if field not in global_aggregated:
                    global_aggregated[field] = []
                global_aggregated[field].extend(values)

        return self._summarize_aggregated_values(global_aggregated, {})

    def _layer_branch_aggregation(
        self, branch_statistics: dict[str, dict[int, dict[str, float]]]
    ):
        """Compute layer statistics from branch statistics."""
        layer_statistics = {}
        for layer_key, layer_data in branch_statistics.items():
            layer_aggregated = self._aggregate_stat_values(layer_data.values())
            for branch_data in layer_data.values():
                depth = branch_data["depth"]
            layer_stats = self._summarize_aggregated_values(
                layer_aggregated,
                {"depth": depth},
            )

            layer_statistics[layer_key] = layer_stats
        return layer_statistics

    def _aggregate_stat_values(self, records) -> dict[str, list[float]]:
        return _aggregate_non_depth_values(records)

    def _summarize_aggregated_values(
        self,
        aggregated_values: dict[str, list[float]],
        stats_dict: dict[str, float],
    ) -> dict[str, float]:
        return _summarize_population_values(aggregated_values, stats_dict)

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
        Perform compartment SNR analysis with optional hierarchical summaries.

        Args:
            model: Classifier whose core contains DendriticBranchLayer modules.
                The optional network input-capture mode is only available for
                explicit ExcitationInhibitionNetwork cores.
            test_dataset: Dataset used to collect activations and labels.
            device: Device for analysis execution.
            save_path: Path to save results (optional).
            filename: Filename for saved results.
            training: If True, only save data (no plots). If False, generate plots.
            runtime: Optional evaluation/runtime controls for dataset materialization.

        Returns:
            Dictionary containing compartment SNR analysis results.
        """
        core_network = getattr(model, "core_network", model)
        if not any(iter_named_modules_of_type(core_network, DendriticBranchLayer)):
            self.logger.warning(
                "Skipping SNR analysis: model has no DendriticBranchLayer modules"
            )
            return None

        if not isinstance(model, Classifier):
            self.logger.warning("Skipping SNR analysis: model is not a Classifier")
            return None

        start_time = time.time()

        with analysis_device_context(model, device) as analysis_device:
            items = materialize_dataset(
                test_dataset,
                runtime,
                device=analysis_device,
            )
            x: torch.Tensor = items[0].to(analysis_device)
            labels: torch.Tensor = items[1].cpu()

            self.collect_raw_data(model, x)

        results = {}

        if self.input_analysis and not training:
            input_analysis = self._input_analysis(labels)
            results["input_analysis"] = input_analysis

        if self.layer_analysis:
            layer_analysis = self._layer_analysis(labels)
            results["layer_analysis"] = layer_analysis

        if self.branch_analysis:
            branch_analysis = self._branch_analysis(labels)
            results["branch_analysis"] = branch_analysis

        elapsed_time = time.time() - start_time
        self.logger.info(f"SNR analysis completed in {elapsed_time:.2f} seconds")

        # Save results and generate plots
        if save_path is not None:
            if training:
                save_path = os.path.join(save_path, "epochs")

            self.logger.info(f"Saving SNR analysis results to {save_path}/{filename}")
            save_dict(results, save_path, f"{filename}.json")

            if not training:
                epochs_data, epoch_numbers = self.load_epochs_data(save_path)
                metrics = list(METRIC_MAPPING.keys())

                if self.layer_analysis:
                    layer_save_path = os.path.join(save_path, "layer_analysis")
                    if self.global_aggregation:
                        if epochs_data:
                            pass

                    if self.layer_aggregation:
                        for metric in metrics:
                            if epochs_data:
                                plot_layer_analysis_layer_agg_vs_training(
                                    epochs_data["layer_analysis"]["layer_statistics"],
                                    epoch_numbers,
                                    layer_save_path,
                                    metric=metric,
                                )

                            plot_layer_analysis_layer_agg_final(
                                results["layer_analysis"]["layer_statistics"],
                                layer_save_path,
                                metric=metric,
                            )

                    if self.branch_aggregation:
                        if epochs_data:
                            pass

                if self.branch_analysis:
                    branch_save_path = os.path.join(save_path, "branch_analysis")
                    if self.global_aggregation:
                        if epochs_data:
                            plot_branch_analysis_global_agg_vs_training(
                                epochs_data["branch_analysis"]["global_statistics"],
                                epoch_numbers,
                                branch_save_path,
                            )

                    if self.layer_aggregation:
                        if epochs_data:
                            plot_branch_analysis_layer_agg_vs_training(
                                epochs_data["branch_analysis"]["layer_statistics"],
                                epoch_numbers,
                                branch_save_path,
                            )

                        plot_branch_analysis_layer_agg_final(
                            results["branch_analysis"]["layer_statistics"],
                            branch_save_path,
                        )

                    if self.branch_aggregation:
                        if epochs_data:
                            pass

                        plot_branch_analysis_branch_agg_final(
                            results["branch_analysis"]["branch_statistics"],
                            branch_save_path,
                        )

        return results

    def _initialize_agg_structure(self, data: dict) -> dict:
        """
        Recursively initialize an aggregate dictionary structure with empty lists at leaf nodes.

        Args:
            data: The dictionary structure to replicate

        Returns:
            Dictionary with same structure but empty lists at leaf nodes
        """
        return initialize_epoch_aggregation_structure(
            data,
            preserved_leaf_keys={"depth"},
        )

    def _populate_agg_dict(self, agg_dict: dict, epoch_data: dict):
        """
        Recursively populate aggregate dictionary by appending leaf values from epoch_data.

        Args:
            agg_dict: The aggregate dictionary to populate (modified in place)
            epoch_data: The epoch data dictionary to extract values from
        """
        populate_epoch_aggregation(
            agg_dict,
            epoch_data,
            preserved_leaf_keys={"depth"},
        )

    def load_epochs_data(self, save_path: str):
        return load_epoch_aggregation(save_path, preserved_leaf_keys={"depth"})


def plot_branch_analysis_global_agg_vs_training(
    global_data: dict, epoch_numbers: list, save_path: str
):
    """
    Plot mean +/- 1 standard deviation for all activation types across training epochs for branch analysis global aggregation.

    Args:
        global_data: Aggregated dictionary with lists of values for each metric
        epoch_numbers: List of epoch numbers corresponding to the data
        save_path: Path to save the plot
    """
    activation_types = ["exc", "inh", "vinf", "vout"]
    colors = {
        "exc": "#e74c3c",
        "inh": "#3498db",
        "vinf": "#9b59b6",
        "vout": "#2c3e50",
    }

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    for act_type in activation_types:
        mean_key = f"{act_type}_d'^2_mean"
        std_key = f"{act_type}_d'^2_std"

        # Extract mean and std lists from global_data
        if mean_key in global_data and std_key in global_data:
            mean_values = global_data[mean_key]
            std_values = global_data[std_key]

            # Convert to numpy arrays for easier manipulation
            mean_array = np.array(mean_values)
            std_array = np.array(std_values)

            # Plot mean line
            ax.plot(
                epoch_numbers,
                mean_array,
                label=act_type,
                color=colors[act_type],
                linewidth=2,
            )

            # Plot mean +/- std as filled region
            ax.fill_between(
                epoch_numbers,
                mean_array - std_array,
                mean_array + std_array,
                alpha=0.2,
                color=colors[act_type],
            )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(r"d'$^2$", fontsize=12)
    ax.set_title(
        "Global Statistics: Mean ± 1 Standard Deviation Across Training", fontsize=14
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, "branch_analysis_global_agg_vs_training.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_branch_analysis_layer_agg_vs_training(
    layer_data: dict, epoch_numbers: list, save_path: str
):
    """
    Plot d'^2 for excitation, inhibition, and vout across training epochs for branch analysis layer aggregation.

    Args:
        layer_data: Aggregated dictionary with layer statistics for branch analysis
                   Structure: {layer_key: {metric_key: [list of values across epochs]}}
        epoch_numbers: List of epoch numbers corresponding to the data
        save_path: Path to save the plot
    """
    # Get all layer keys and sort them for consistent ordering
    layer_keys = sorted(layer_data.keys())

    # Generate colors for each layer using a colormap
    # Use a colormap that provides distinct colors
    cmap = plt.cm.get_cmap("tab10")
    n_layers = len(layer_keys)
    if n_layers == 1:
        layer_colors = {layer_keys[0]: cmap(0)}
    else:
        layer_colors = {
            layer_key: cmap(i / (n_layers - 1))
            for i, layer_key in enumerate(layer_keys)
        }

    # Create figure with 3 subplots (rows)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

    # Activation types and their corresponding row indices
    activation_types = ["exc", "inh", "vout"]

    for row_idx, act_type in enumerate(activation_types):
        ax = axes[row_idx]

        # Plot each layer
        for layer_key in layer_keys:
            mean_key = f"{act_type}_d'^2_mean"
            std_key = f"{act_type}_d'^2_std"

            if mean_key in layer_data[layer_key] and std_key in layer_data[layer_key]:
                mean_values = layer_data[layer_key][mean_key]
                std_values = layer_data[layer_key][std_key]

                # Convert to numpy arrays
                mean_array = np.array(mean_values)
                std_array = np.array(std_values)

                # Get layer color
                color = layer_colors[layer_key]

                # Plot mean line with layer label
                ax.plot(
                    epoch_numbers,
                    mean_array,
                    label=f"depth {layer_data[layer_key]['depth']}",
                    color=color,
                    linewidth=2,
                )

                # Plot mean +/- std as filled region
                ax.fill_between(
                    epoch_numbers,
                    mean_array - std_array,
                    mean_array + std_array,
                    alpha=0.2,
                    color=color,
                )

        # Set labels and title for each subplot
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(r"d'$^2$", fontsize=12)

        # Set title based on activation type
        act_labels = {"exc": "Excitation", "inh": "Inhibition", "vout": "Vout"}
        ax.set_title(
            f"Layer Statistics: {act_labels[act_type]} d'$^2$ Across Training",
            fontsize=12,
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, "branch_analysis_layer_agg_vs_training.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_branch_analysis_layer_agg_final(
    layer_statistics: dict, save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot layer statistics showing d'^2 for vout, excitation, inhibition, and upstream across layers for branch analysis layer aggregation.

    This function produces a figure analogous to plot_layer_ablation but shows d'^2 values
    instead of accuracy/log-likelihood decrease.

    Args:
        layer_statistics: Dictionary containing layer statistics for branch analysis with structure:
                         {layer_key: {depth: int, exc_d'^2_mean: float,
                                     inh_d'^2_mean: float, vout_d'^2_mean: float,
                                     upstream_d'^2_mean: float, ...}}
        save_path: Path to save plot for branch analysis layer aggregation. If None, returns figure

    Returns:
        Figure if save_path is None, otherwise None
    """
    # Extract activation types we want to plot (in desired order: vout, exc, inh, upstream)
    # vout is always present, others are optional
    activation_types = ["vout", "exc", "inh", "upstream"]
    activation_labels = {
        "exc": "Excitation",
        "inh": "Inhibition",
        "vout": "Vout",
        "upstream": "Upstream",
    }

    # Color scheme - using a different palette
    colors = {
        "vout": "#2c3e50",  # Dark blue-gray
        "exc": "#e74c3c",  # Red
        "inh": "#3498db",  # Blue
        "upstream": "#808080",  # Medium grey
    }

    # Sort layers by depth to ensure correct ordering
    sorted_layers = sorted(
        layer_statistics.items(), key=lambda x: x[1].get("depth", float("inf"))
    )

    # First, get all depths from vout (which is always present)
    plot_depths = []
    for _layer_key, layer_data in sorted_layers:
        depth = layer_data.get("depth")
        vout_mean_key = "vout_d'^2_mean"
        if depth is not None and vout_mean_key in layer_data:
            plot_depths.append(depth)

    if not plot_depths:
        logger.warning("No vout data found in layer statistics")
        return None

    # Create a mapping from depth to layer_data for quick lookup
    depth_to_layer = {}
    for _layer_key, layer_data in sorted_layers:
        depth = layer_data.get("depth")
        if depth is not None:
            depth_to_layer[depth] = layer_data

    # Collect d'^2 values and standard deviations for each activation type
    # Aligned to plot_depths (vout reference) - use None for missing values
    results = {}
    for act_type in activation_types:
        mean_key = f"{act_type}_d'^2_mean"
        std_key = f"{act_type}_d'^2_std"

        y_values = []
        yerr_values = []
        valid_mask = []  # Track which depths have data

        for depth in plot_depths:
            if depth in depth_to_layer:
                layer_data = depth_to_layer[depth]
                if mean_key in layer_data and std_key in layer_data:
                    y_values.append(layer_data[mean_key])
                    yerr_values.append(layer_data[std_key])
                    valid_mask.append(True)
                else:
                    y_values.append(None)
                    yerr_values.append(None)
                    valid_mask.append(False)
            else:
                y_values.append(None)
                yerr_values.append(None)
                valid_mask.append(False)

        # Only add if we have at least some data for this activation type
        if any(valid_mask):
            results[act_type] = {
                "depths": plot_depths,
                "d'^2": y_values,
                "std": yerr_values,
                "valid_mask": valid_mask,
            }

    if not results:
        logger.warning("No layer statistics data found for plotting")
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Count only activation types that have data
    available_types = [act_type for act_type in activation_types if act_type in results]
    n_activation_types = len(available_types)

    if n_activation_types == 0:
        logger.warning("No activation types with data to plot")
        return None

    width = 0.2

    # Plot bars for each activation type with error bars
    for i, act_type in enumerate(available_types):
        act_data = results[act_type]
        act_depths = np.array(act_data["depths"])
        valid_mask = np.array(act_data["valid_mask"])

        # Filter to only valid data points before converting to numpy
        valid_depths = act_depths[valid_mask]
        valid_y_values = np.array(
            [v for j, v in enumerate(act_data["d'^2"]) if valid_mask[j]]
        )
        valid_yerr_values = np.array(
            [v for j, v in enumerate(act_data["std"]) if valid_mask[j]]
        )

        # Calculate x positions for bars (aligned to plot_depths)
        x = valid_depths + width * (i - (n_activation_types - 1) / 2)

        ax.bar(
            x,
            valid_y_values,
            width=width,
            label=activation_labels[act_type],
            color=colors[act_type],
            yerr=valid_yerr_values,
            capsize=5,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )

    # Create depth labels - if depth=0 label as "Soma", else f"Depth {depth}"
    depth_labels = []
    for depth in plot_depths:
        if depth == 0:
            depth_labels.append("Soma")
        else:
            depth_labels.append(f"Depth {depth}")

    ax.set_xticks(plot_depths)
    ax.set_xticklabels(depth_labels, rotation=45)
    ax.set_xlabel("Branch Depth (Soma to Distal)")
    ax.set_ylabel(r"d'$^2$")
    ax.set_title(r"d'$^2$ vs. Branch Depth")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, "branch_analysis_layer_agg_final.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig


def plot_branch_analysis_branch_agg_final(branch_data: dict, save_path: str):
    """
    Plot the final branch statistics for branch analysis branch aggregation.

    Args:
        branch_data: The branch data dictionary to plot for branch analysis branch aggregation
        save_path: Path to save the plot
    """
    agg_data: dict[str, dict[str, list[float]]] = {}

    # First pass: collect all available activation types
    available_activation_types = set()

    for _layer_key, layer_data in branch_data.items():
        for branch_item in layer_data.values():
            # Check which activation types are present
            for act_type in ["exc", "inh", "vinf", "vout", "upstream"]:
                key = f"{act_type}_d'^2"
                if key in branch_item:
                    available_activation_types.add(act_type)

    # vout should always be present, but check anyway
    if "vout" not in available_activation_types:
        logger.warning("No vout data found in branch statistics")
        return

    # Second pass: aggregate data for available activation types
    for _layer_key, layer_data in branch_data.items():
        for branch_item in layer_data.values():
            depth = branch_item["depth"]
            if f"depth {depth}" not in agg_data.keys():
                agg_data[f"depth {depth}"] = {}

            # Initialize lists for available activation types
            for act_type in available_activation_types:
                if act_type not in agg_data[f"depth {depth}"].keys():
                    agg_data[f"depth {depth}"][act_type] = []

            # Append data only for activation types that exist
            for act_type in available_activation_types:
                key = f"{act_type}_d'^2"
                if key in branch_item:
                    agg_data[f"depth {depth}"][act_type].append(branch_item[key])

    # Use available activation types for plotting
    indeps = sorted(available_activation_types)
    deps = sorted(available_activation_types)

    # Build long-form DataFrame from agg_data
    rows = []
    for depth, data in agg_data.items():
        for dep in deps:  # y
            for indep in indeps:  # x
                # Only process if both activation types have data
                if (
                    indep in data
                    and dep in data
                    and len(data[indep]) > 0
                    and len(data[dep]) > 0
                ):
                    x = np.asarray(data[indep])
                    y = np.asarray(data[dep])
                    # Only add if arrays have the same length
                    if len(x) == len(y):
                        rows.append(
                            pd.DataFrame(
                                {
                                    "x": x,
                                    "y": y,
                                    "depth": depth,
                                    "row": dep,
                                    "col": indep,
                                }
                            )
                        )

    if not rows:
        logger.warning("No valid data to plot in branch statistics")
        return

    df = pd.concat(rows, ignore_index=True)

    # Labels - include all available activation types
    labels = {
        "exc": r"$d'^2_{E}$",
        "inh": r"$d'^2_{I}$",
        "vinf": r"$d'^2_{V_{\infty}}$",
        "vout": r"$d'^2_{V_{out}}$",
        "upstream": r"$d'^2_{upstream}$",
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
                ax.set_xlabel(labels.get(col_key, col_key))
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(labels.get(row_key, row_key))
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plot_path = os.path.join(save_path, "branch_analysis_branch_agg_final.png")
    g.figure.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(g.figure)


def plot_layer_analysis_layer_agg_final(
    layer_statistics: dict,
    save_path: Optional[str] = None,
    # metric: str = "d'^2"
    metric: str = "d_max",
) -> Optional[plt.Figure]:
    """
    Plot layer statistics showing d'^2 for vout, excitation, inhibition, and upstream across layers for branch analysis layer aggregation.

    This function produces a figure analogous to plot_layer_ablation but shows d'^2 values
    instead of accuracy/log-likelihood decrease.

    Args:
        layer_statistics: Dictionary containing layer statistics for branch analysis with structure:
                         {layer_key: {depth: int, exc_d'^2_mean: float,
                                     inh_d'^2_mean: float, vout_d'^2_mean: float,
                                     upstream_d'^2_mean: float, ...}}
        save_path: Path to save plot for branch analysis layer aggregation. If None, returns figure

    Returns:
        Figure if save_path is None, otherwise None
    """
    # Extract activation types we want to plot (in desired order: vout, exc, inh, upstream)
    # vout is always present, others are optional
    activation_types = ["vout", "exc", "inh", "upstream"]
    activation_labels = {
        "exc": "Excitation",
        "inh": "Inhibition",
        "vout": "Vout",
        "upstream": "Upstream",
    }

    # Color scheme - using a different palette
    colors = {
        "vout": "#2c3e50",  # Dark blue-gray
        "exc": "#e74c3c",  # Red
        "inh": "#3498db",  # Blue
        "upstream": "#808080",  # Medium grey
    }

    # Sort layers by depth to ensure correct ordering
    sorted_layers = sorted(
        layer_statistics.items(), key=lambda x: x[1].get("depth", float("inf"))
    )

    # First, get all depths from vout (which is always present)
    plot_depths = []
    for _layer_key, layer_data in sorted_layers:
        depth = layer_data.get("depth")
        vout_key = f"vout_{metric}"
        if depth is not None and vout_key in layer_data:
            plot_depths.append(depth)

    if not plot_depths:
        logger.warning("No vout data found in layer statistics")
        return None

    # Create a mapping from depth to layer_data for quick lookup
    depth_to_layer = {}
    for _layer_key, layer_data in sorted_layers:
        depth = layer_data.get("depth")
        if depth is not None:
            depth_to_layer[depth] = layer_data

    # Collect d'^2 values and standard deviations for each activation type
    # Aligned to plot_depths (vout reference) - use None for missing values
    results = {}
    for act_type in activation_types:
        act_key = f"{act_type}_{metric}"

        y_values = []
        valid_mask = []  # Track which depths have data

        for depth in plot_depths:
            if depth in depth_to_layer:
                layer_data = depth_to_layer[depth]
                if act_key in layer_data:
                    y_values.append(layer_data[act_key])
                    valid_mask.append(True)
                else:
                    y_values.append(None)
                    valid_mask.append(False)
            else:
                y_values.append(None)
                valid_mask.append(False)

        # Only add if we have at least some data for this activation type
        if any(valid_mask):
            results[act_type] = {
                "depths": plot_depths,
                metric: y_values,
                "valid_mask": valid_mask,
            }

    if not results:
        logger.warning("No layer statistics data found for plotting")
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Count only activation types that have data
    available_types = [act_type for act_type in activation_types if act_type in results]
    n_activation_types = len(available_types)

    if n_activation_types == 0:
        logger.warning("No activation types with data to plot")
        return None

    width = 0.2

    # Plot bars for each activation type with error bars
    for i, act_type in enumerate(available_types):
        act_data = results[act_type]
        act_depths = np.array(act_data["depths"])
        valid_mask = np.array(act_data["valid_mask"])

        # Filter to only valid data points before converting to numpy
        valid_depths = act_depths[valid_mask]
        valid_y_values = np.array(
            [v for j, v in enumerate(act_data[metric]) if valid_mask[j]]
        )

        # Calculate x positions for bars (aligned to plot_depths)
        x = valid_depths + width * (i - (n_activation_types - 1) / 2)

        ax.bar(
            x,
            valid_y_values,
            width=width,
            label=activation_labels[act_type],
            color=colors[act_type],
            capsize=5,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )

    # Create depth labels - if depth=0 label as "Soma", else f"Depth {depth}"
    depth_labels = []
    for depth in plot_depths:
        if depth == 0:
            depth_labels.append("Soma")
        else:
            depth_labels.append(f"Depth {depth}")

    ax.set_xticks(plot_depths)
    ax.set_xticklabels(depth_labels, rotation=45)
    ax.set_xlabel("Branch Depth (Soma to Distal)")

    ylabel = METRIC_MAPPING[metric]
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. Branch Depth")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        # plot_path = os.path.join(save_path, f"layer_analysis_layer_agg_final_{metric.replace('/', 'div')}.png")
        plot_path = os.path.join(
            save_path, f"layer_analysis_layer_agg_final_{metric}.png"
        )
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig


def plot_layer_analysis_layer_agg_final_both_metrics(
    layer_statistics: dict, save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot layer statistics with dual subplots showing different metrics.

    Left subplot: d'^2/D for vout and upstream across depths
    Right subplot: d'^2 for exc and inh across depths

    This function produces a static figure with the same layout as frames in
    plot_layer_analysis_layer_agg_vs_training_both_metrics.

    Args:
        layer_statistics: Dictionary containing layer statistics with structure:
                         {layer_key: {depth: int, exc_d'^2: float,
                                     inh_d'^2: float, vout_d'^2: float,
                                     vout_d'^2/D: float, upstream_d'^2/D: float, ...}}
        save_path: Path to save plot. If None, returns figure

    Returns:
        Figure if save_path is None, otherwise None
    """
    # Define activation types and labels for each subplot
    left_activation_types = ["vout", "upstream"]  # d'^2/D
    right_activation_types = ["exc", "inh"]  # d'^2

    activation_labels = {
        "exc": "Excitation",
        "inh": "Inhibition",
        "vout": "Vout",
        "upstream": "Upstream",
    }

    colors = {
        "vout": "#2c3e50",  # Dark blue-gray
        "exc": "#e74c3c",  # Red
        "inh": "#3498db",  # Blue
        "upstream": "#808080",  # Medium grey
    }

    # Sort layers by depth to ensure correct ordering
    sorted_layers = sorted(
        layer_statistics.items(), key=lambda x: x[1].get("depth", float("inf"))
    )

    # Get all depths from vout d'^2/D (which should be present)
    plot_depths = []
    for _layer_key, layer_data in sorted_layers:
        depth = layer_data.get("depth")
        vout_key = "vout_d'^2/D"
        if depth is not None and vout_key in layer_data:
            plot_depths.append(depth)

    if not plot_depths:
        logger.warning("No vout d'^2/D data found in layer statistics")
        return None

    # Create a mapping from depth to layer_data for quick lookup
    depth_to_layer = {}
    for _layer_key, layer_data in sorted_layers:
        depth = layer_data.get("depth")
        if depth is not None:
            depth_to_layer[depth] = layer_data

    # Collect values for left subplot (d'^2/D for vout and upstream)
    left_results = {}
    for act_type in left_activation_types:
        act_key = f"{act_type}_d'^2/D"

        y_values = []
        valid_mask = []

        for depth in plot_depths:
            if depth in depth_to_layer:
                layer_data = depth_to_layer[depth]
                if act_key in layer_data:
                    y_values.append(layer_data[act_key])
                    valid_mask.append(True)
                else:
                    y_values.append(None)
                    valid_mask.append(False)
            else:
                y_values.append(None)
                valid_mask.append(False)

        if any(valid_mask):
            left_results[act_type] = {
                "depths": plot_depths,
                "values": y_values,
                "valid_mask": valid_mask,
            }

    # Collect values for right subplot (d'^2 for exc and inh)
    right_results = {}
    for act_type in right_activation_types:
        act_key = f"{act_type}_d'^2"

        y_values = []
        valid_mask = []

        for depth in plot_depths:
            if depth in depth_to_layer:
                layer_data = depth_to_layer[depth]
                if act_key in layer_data:
                    y_values.append(layer_data[act_key])
                    valid_mask.append(True)
                else:
                    y_values.append(None)
                    valid_mask.append(False)
            else:
                y_values.append(None)
                valid_mask.append(False)

        if any(valid_mask):
            right_results[act_type] = {
                "depths": plot_depths,
                "values": y_values,
                "valid_mask": valid_mask,
            }

    if not left_results and not right_results:
        logger.warning("No layer statistics data found for plotting")
        return None

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    ax_left = axes[0]
    ax_right = axes[1]

    # Determine available activation types for each subplot
    left_available_types = [
        act_type for act_type in left_activation_types if act_type in left_results
    ]
    right_available_types = [
        act_type for act_type in right_activation_types if act_type in right_results
    ]

    if not left_available_types and not right_available_types:
        logger.warning("No activation types with data to plot")
        plt.close(fig)
        return None

    # Calculate y-axis ranges for each subplot
    left_all_values = []
    for act_type in left_available_types:
        act_data = left_results[act_type]
        valid_values = [
            v for j, v in enumerate(act_data["values"]) if act_data["valid_mask"][j]
        ]
        left_all_values.extend(valid_values)

    right_all_values = []
    for act_type in right_available_types:
        act_data = right_results[act_type]
        valid_values = [
            v for j, v in enumerate(act_data["values"]) if act_data["valid_mask"][j]
        ]
        right_all_values.extend(valid_values)

    # Set y-axis ranges
    if not left_all_values:
        left_y_min, left_y_max = 0.0, 1.0
    else:
        left_y_min = 0.0
        left_y_max = max(left_all_values)
        y_range = left_y_max - left_y_min
        if y_range == 0:
            left_y_max += 0.1
        else:
            padding = y_range * 0.05
            left_y_max = left_y_max + padding

    if not right_all_values:
        right_y_min, right_y_max = 0.0, 1.0
    else:
        right_y_min = 0.0
        right_y_max = max(right_all_values)
        y_range = right_y_max - right_y_min
        if y_range == 0:
            right_y_max += 0.1
        else:
            padding = y_range * 0.05
            right_y_max = right_y_max + padding

    width = 0.2

    # Calculate x-position mapping: groups spaced by one bar width
    # Center-to-center distance = (n_bars * width) + width = width * (n_bars + 1)
    left_group_spacing = width * (len(left_available_types) + 1)
    right_group_spacing = width * (len(right_available_types) + 1)

    # Map depth values to x-positions
    depth_to_x_left = {
        depth: i * left_group_spacing for i, depth in enumerate(plot_depths)
    }
    depth_to_x_right = {
        depth: i * right_group_spacing for i, depth in enumerate(plot_depths)
    }

    # Plot left subplot: d'^2/D for vout and upstream
    for i, act_type in enumerate(left_available_types):
        act_data = left_results[act_type]
        act_depths = np.array(act_data["depths"])
        valid_mask = np.array(act_data["valid_mask"])

        valid_depths = act_depths[valid_mask]
        valid_y_values = np.array(
            [v for j, v in enumerate(act_data["values"]) if valid_mask[j]]
        )

        # Map depths to x-positions and offset by bar position within group
        x = np.array([depth_to_x_left[depth] for depth in valid_depths]) + width * (
            i - (len(left_available_types) - 1) / 2
        )

        ax_left.bar(
            x,
            valid_y_values,
            width=width,
            label=activation_labels[act_type],
            color=colors[act_type],
            capsize=5,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )

    # Plot right subplot: d'^2 for exc and inh
    for i, act_type in enumerate(right_available_types):
        act_data = right_results[act_type]
        act_depths = np.array(act_data["depths"])
        valid_mask = np.array(act_data["valid_mask"])

        valid_depths = act_depths[valid_mask]
        valid_y_values = np.array(
            [v for j, v in enumerate(act_data["values"]) if valid_mask[j]]
        )

        # Map depths to x-positions and offset by bar position within group
        x = np.array([depth_to_x_right[depth] for depth in valid_depths]) + width * (
            i - (len(right_available_types) - 1) / 2
        )

        ax_right.bar(
            x,
            valid_y_values,
            width=width,
            label=activation_labels[act_type],
            color=colors[act_type],
            capsize=5,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )

    # Create depth labels
    depth_labels = []
    for depth in plot_depths:
        if depth == 0:
            depth_labels.append("Soma")
        else:
            depth_labels.append(f"Depth {depth}")

    # Configure left subplot - use mapped x-positions for ticks
    left_x_positions = [depth_to_x_left[depth] for depth in plot_depths]
    ax_left.set_xticks(left_x_positions)
    ax_left.set_xticklabels(depth_labels, rotation=45)
    ax_left.set_xlabel("Branch Depth (Soma to Distal)")
    ax_left.set_ylabel(r"d'$^2$/D")
    ax_left.set_ylim(left_y_min, left_y_max)
    ax_left.set_title(r"d'$^2$/D vs. Branch Depth")
    ax_left.legend()
    ax_left.grid(True, axis="y", alpha=0.3)

    # Configure right subplot - use mapped x-positions for ticks
    right_x_positions = [depth_to_x_right[depth] for depth in plot_depths]
    ax_right.set_xticks(right_x_positions)
    ax_right.set_xticklabels(depth_labels, rotation=45)
    ax_right.set_xlabel("Branch Depth (Soma to Distal)")
    ax_right.set_ylabel(r"d'$^2$")
    ax_right.set_ylim(right_y_min, right_y_max)
    ax_right.set_title(r"d'$^2$ vs. Branch Depth")
    ax_right.legend()
    ax_right.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(
            save_path, "layer_analysis_layer_agg_final_both_metrics.png"
        )
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig


def plot_layer_analysis_layer_agg_vs_training(
    layer_data: dict,
    epoch_numbers: list,
    save_path: str,
    # metric: str = "d'^2"
    metric: str = "d_max",
):
    """
    Create a video showing layer analysis statistics evolving over training epochs.

    Each frame in the video looks exactly like the figure produced by
    plot_layer_analysis_layer_agg_final, showing d'^2 values vs. branch depth.

    Args:
        layer_data: Aggregated dictionary with layer statistics from epochs_data.
                   Structure: {layer_key: {depth: int, exc_d'^2: [list],
                                          inh_d'^2: [list], vout_d'^2: [list], ...}}
        epoch_numbers: List of epoch numbers corresponding to the data
        save_path: Path to save the video
    """
    if not layer_data:
        logger.warning("No layer data provided for video creation")
        return

    # Determine number of epochs from the data structure
    # Check the length of any list value to get number of epochs
    num_epochs = None
    for _layer_key, layer_stats in layer_data.items():
        for key, value in layer_stats.items():
            if key != "depth" and isinstance(value, list):
                if num_epochs is None:
                    num_epochs = len(value)
                elif len(value) != num_epochs:
                    logger.warning(
                        f"Inconsistent epoch counts: {num_epochs} vs {len(value)}"
                    )
                    num_epochs = min(num_epochs, len(value))
                break
        if num_epochs is not None:
            break

    if num_epochs is None or num_epochs == 0:
        logger.warning("No epoch data found in layer_data")
        return

    # Calculate fps targeting ~30 seconds total
    # fps = int(num_epochs / 30.0)
    # fps = min(max(fps, 2), 10)  # Clamp between 2 and 10 fps
    fps = 10

    # Extract activation types and labels (same as plot_layer_analysis_layer_agg_final)
    activation_types = ["vout", "exc", "inh", "upstream"]
    activation_labels = {
        "exc": "Excitation",
        "inh": "Inhibition",
        "vout": "Vout",
        "upstream": "Upstream",
    }

    colors = {
        "vout": "#2c3e50",  # Dark blue-gray
        "exc": "#e74c3c",  # Red
        "inh": "#3498db",  # Blue
        "upstream": "#808080",  # Medium grey
    }

    # Sort layers by depth to ensure correct ordering
    sorted_layers = sorted(
        layer_data.items(), key=lambda x: x[1].get("depth", float("inf"))
    )

    # Get all depths from vout (which is always present)
    plot_depths = []
    for _layer_key, layer_stats in sorted_layers:
        depth = layer_stats.get("depth")
        vout_key = f"vout_{metric}"
        if depth is not None and vout_key in layer_stats:
            if (
                isinstance(layer_stats[vout_key], list)
                and len(layer_stats[vout_key]) > 0
            ):
                plot_depths.append(depth)

    if not plot_depths:
        logger.warning("No vout data found in layer statistics")
        return

    # Create a mapping from depth to layer_key for quick lookup
    depth_to_layer_key = {}
    for layer_key, layer_stats in sorted_layers:
        depth = layer_stats.get("depth")
        if depth is not None:
            depth_to_layer_key[depth] = layer_key

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine available activation types (check first epoch)
    # Maintain order from activation_types list
    available_types = []
    for act_type in activation_types:
        act_key = f"{act_type}_{metric}"
        for depth in plot_depths:
            if depth in depth_to_layer_key:
                layer_key = depth_to_layer_key[depth]
                if act_key in layer_data[layer_key]:
                    value = layer_data[layer_key][act_key]
                    if isinstance(value, list) and len(value) > 0:
                        available_types.append(act_type)
                        break
    n_activation_types = len(available_types)

    if n_activation_types == 0:
        logger.warning("No activation types with data to plot")
        plt.close(fig)
        return

    # Calculate global min and max values across all epochs for consistent y-axis
    all_values = []
    for epoch_idx in range(num_epochs):
        # Extract data at this epoch index
        for _layer_key, layer_stats in layer_data.items():
            for act_type in available_types:
                act_key = f"{act_type}_{metric}"
                if act_key in layer_stats:
                    value = layer_stats[act_key]
                    if isinstance(value, list):
                        if epoch_idx < len(value):
                            val = value[epoch_idx]
                            if val is not None:
                                all_values.append(float(val))
                    elif value is not None:
                        all_values.append(float(value))

    if not all_values:
        y_min, y_max = 0.0, 1.0
    else:
        y_min = 0.0  # Hard set to 0 as requested
        y_max = max(all_values)
        # Add small padding (5% of range)
        y_range = y_max - y_min
        if y_range == 0:
            y_max += 0.1
        else:
            padding = y_range * 0.05
            y_max = y_max + padding

    # Create evenly spaced y-ticks rounded to 1 decimal place
    num_ticks = 6  # Number of ticks desired
    y_ticks = np.linspace(y_min, y_max, num_ticks)
    y_ticks = np.round(y_ticks, 1)  # Round to 1 decimal place
    y_ticks = np.unique(y_ticks)  # Remove duplicates

    width = 0.2

    def animate(frame):
        """Animation function that updates the plot for each epoch"""
        ax.clear()

        # Extract data at this epoch index
        epoch_snapshot = {}
        for _layer_key, layer_stats in layer_data.items():
            epoch_snapshot[_layer_key] = {"depth": layer_stats.get("depth")}
            for key, value in layer_stats.items():
                if key != "depth":
                    if isinstance(value, list):
                        if frame < len(value):
                            epoch_snapshot[_layer_key][key] = value[frame]
                        else:
                            # Use last available value if frame exceeds list length
                            epoch_snapshot[_layer_key][key] = (
                                value[-1] if value else None
                            )
                    else:
                        epoch_snapshot[_layer_key][key] = value

        # Create depth_to_layer mapping for this epoch
        depth_to_layer = {}
        for _layer_key, layer_stats in epoch_snapshot.items():
            depth = layer_stats.get("depth")
            if depth is not None:
                depth_to_layer[depth] = layer_stats

        # Collect d'^2 values for each activation type
        results = {}
        for act_type in activation_types:
            act_key = f"{act_type}_{metric}"

            y_values = []
            valid_mask = []

            for depth in plot_depths:
                if depth in depth_to_layer:
                    layer_stats = depth_to_layer[depth]
                    if act_key in layer_stats and layer_stats[act_key] is not None:
                        y_values.append(layer_stats[act_key])
                        valid_mask.append(True)
                    else:
                        y_values.append(None)
                        valid_mask.append(False)
                else:
                    y_values.append(None)
                    valid_mask.append(False)

            if any(valid_mask):
                results[act_type] = {
                    "depths": plot_depths,
                    metric: y_values,
                    "valid_mask": valid_mask,
                }

        # Plot bars for each activation type
        for i, act_type in enumerate(available_types):
            if act_type not in results:
                continue

            act_data = results[act_type]
            act_depths = np.array(act_data["depths"])
            valid_mask = np.array(act_data["valid_mask"])

            # Filter to only valid data points
            valid_depths = act_depths[valid_mask]
            valid_y_values = np.array(
                [v for j, v in enumerate(act_data[metric]) if valid_mask[j]]
            )

            # Calculate x positions for bars
            x = valid_depths + width * (i - (n_activation_types - 1) / 2)

            ax.bar(
                x,
                valid_y_values,
                width=width,
                label=activation_labels[act_type],
                color=colors[act_type],
                capsize=5,
                error_kw={"elinewidth": 1.5, "capthick": 1.5},
            )

        # Create depth labels
        depth_labels = []
        for depth in plot_depths:
            if depth == 0:
                depth_labels.append("Soma")
            else:
                depth_labels.append(f"Depth {depth}")

        ax.set_xticks(plot_depths)
        ax.set_xticklabels(depth_labels, rotation=45)
        ax.set_xlabel("Branch Depth (Soma to Distal)")

        ylabel = METRIC_MAPPING[metric]
        ax.set_ylabel(ylabel)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)

        # Get epoch number for title
        epoch_num = epoch_numbers[frame] if frame < len(epoch_numbers) else frame
        ax.set_title(f"{ylabel} vs. Branch Depth (Epoch {epoch_num})")
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

        return []

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=num_epochs,
        interval=1000 / fps,  # Convert fps to interval in milliseconds
        repeat=True,
        blit=True,
    )

    # Save video
    video_filename = (
        f"layer_analysis_layer_agg_vs_training_{metric.replace('/', 'div')}"
    )
    logger.info(
        f"Creating {num_epochs}-frame video at {fps:.2f} fps (~{num_epochs/fps:.1f} seconds total)"
    )

    os.makedirs(save_path, exist_ok=True)

    if animation.writers.is_available("ffmpeg"):
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            video_path = os.path.join(save_path, f"{video_filename}.mp4")
            ani.save(video_path, writer=writer)
            logger.info(f"Video saved to {video_path}")
        except Exception as e:
            logger.warning(f"Error saving MP4 video: {e}")
            logger.info("Falling back to GIF format...")
            try:
                writer = animation.PillowWriter(fps=fps)
                video_path = os.path.join(save_path, f"{video_filename}.gif")
                ani.save(video_path, writer=writer)
                logger.info(f"GIF saved to {video_path}")
            except Exception as e2:
                logger.error(f"Error saving GIF video: {e2}")
    else:
        logger.info("ffmpeg writer not available; saving GIF format...")
        try:
            writer = animation.PillowWriter(fps=fps)
            video_path = os.path.join(save_path, f"{video_filename}.gif")
            ani.save(video_path, writer=writer)
            logger.info(f"GIF saved to {video_path}")
        except Exception as e2:
            logger.error(f"Error saving GIF video: {e2}")

    plt.close(fig)


def plot_layer_analysis_layer_agg_vs_training_both_metrics(
    layer_data: dict, epoch_numbers: list, save_path: str
):
    """
    Create a video showing layer analysis statistics evolving over training epochs with dual subplots.

    Left subplot: d'^2/D for vout and upstream across depths
    Right subplot: d'^2 for exc and inh across depths

    Args:
        layer_data: Aggregated dictionary with layer statistics from epochs_data.
                   Structure: {layer_key: {depth: int, exc_d'^2: [list],
                                          inh_d'^2: [list], vout_d'^2: [list],
                                          vout_d'^2/D: [list], upstream_d'^2/D: [list], ...}}
        epoch_numbers: List of epoch numbers corresponding to the data
        save_path: Path to save the video
    """
    if not layer_data:
        logger.warning("No layer data provided for video creation")
        return

    # Determine number of epochs from the data structure
    num_epochs = None
    for _layer_key, layer_stats in layer_data.items():
        for key, value in layer_stats.items():
            if key != "depth" and isinstance(value, list):
                if num_epochs is None:
                    num_epochs = len(value)
                elif len(value) != num_epochs:
                    logger.warning(
                        f"Inconsistent epoch counts: {num_epochs} vs {len(value)}"
                    )
                    num_epochs = min(num_epochs, len(value))
                break
        if num_epochs is not None:
            break

    if num_epochs is None or num_epochs == 0:
        logger.warning("No epoch data found in layer_data")
        return

    fps = 10

    # Define activation types and labels for each subplot
    left_activation_types = ["vout", "upstream"]  # d'^2/D
    right_activation_types = ["exc", "inh"]  # d'^2

    activation_labels = {
        "exc": "Excitation",
        "inh": "Inhibition",
        "vout": "Vout",
        "upstream": "Upstream",
    }

    colors = {
        "vout": "#2c3e50",  # Dark blue-gray
        "exc": "#e74c3c",  # Red
        "inh": "#3498db",  # Blue
        "upstream": "#808080",  # Medium grey
    }

    # Sort layers by depth to ensure correct ordering
    sorted_layers = sorted(
        layer_data.items(), key=lambda x: x[1].get("depth", float("inf"))
    )

    # Get all depths from vout (which is always present)
    plot_depths = []
    for _layer_key, layer_stats in sorted_layers:
        depth = layer_stats.get("depth")
        vout_key = "vout_d'^2/D"
        if depth is not None and vout_key in layer_stats:
            if (
                isinstance(layer_stats[vout_key], list)
                and len(layer_stats[vout_key]) > 0
            ):
                plot_depths.append(depth)

    if not plot_depths:
        logger.warning("No vout data found in layer statistics")
        return

    # Create a mapping from depth to layer_key for quick lookup
    depth_to_layer_key = {}
    for layer_key, layer_stats in sorted_layers:
        depth = layer_stats.get("depth")
        if depth is not None:
            depth_to_layer_key[depth] = layer_key

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    ax_left = axes[0]
    ax_right = axes[1]

    # Determine available activation types for each subplot
    left_available_types = []
    for act_type in left_activation_types:
        act_key = f"{act_type}_d'^2/D"
        for depth in plot_depths:
            if depth in depth_to_layer_key:
                layer_key = depth_to_layer_key[depth]
                if act_key in layer_data[layer_key]:
                    value = layer_data[layer_key][act_key]
                    if isinstance(value, list) and len(value) > 0:
                        left_available_types.append(act_type)
                        break

    right_available_types = []
    for act_type in right_activation_types:
        act_key = f"{act_type}_d'^2"
        for depth in plot_depths:
            if depth in depth_to_layer_key:
                layer_key = depth_to_layer_key[depth]
                if act_key in layer_data[layer_key]:
                    value = layer_data[layer_key][act_key]
                    if isinstance(value, list) and len(value) > 0:
                        right_available_types.append(act_type)
                        break

    if not left_available_types and not right_available_types:
        logger.warning("No activation types with data to plot")
        plt.close(fig)
        return

    # Calculate global min and max values for left subplot (d'^2/D)
    left_all_values = []
    if left_available_types:
        for epoch_idx in range(num_epochs):
            for _layer_key, layer_stats in layer_data.items():
                for act_type in left_available_types:
                    act_key = f"{act_type}_d'^2/D"
                    if act_key in layer_stats:
                        value = layer_stats[act_key]
                        if isinstance(value, list):
                            if epoch_idx < len(value):
                                val = value[epoch_idx]
                                if val is not None:
                                    left_all_values.append(float(val))
                        elif value is not None:
                            left_all_values.append(float(value))

    # Calculate global min and max values for right subplot (d'^2)
    right_all_values = []
    if right_available_types:
        for epoch_idx in range(num_epochs):
            for _layer_key, layer_stats in layer_data.items():
                for act_type in right_available_types:
                    act_key = f"{act_type}_d'^2"
                    if act_key in layer_stats:
                        value = layer_stats[act_key]
                        if isinstance(value, list):
                            if epoch_idx < len(value):
                                val = value[epoch_idx]
                                if val is not None:
                                    right_all_values.append(float(val))
                        elif value is not None:
                            right_all_values.append(float(value))

    # Set y-axis ranges
    if not left_all_values:
        left_y_min, left_y_max = 0.0, 1.0
    else:
        left_y_min = 0.0
        left_y_max = max(left_all_values)
        y_range = left_y_max - left_y_min
        if y_range == 0:
            left_y_max += 0.1
        else:
            padding = y_range * 0.05
            left_y_max = left_y_max + padding

    if not right_all_values:
        right_y_min, right_y_max = 0.0, 1.0
    else:
        right_y_min = 0.0
        right_y_max = max(right_all_values)
        y_range = right_y_max - right_y_min
        if y_range == 0:
            right_y_max += 0.1
        else:
            padding = y_range * 0.05
            right_y_max = right_y_max + padding

    # Create evenly spaced y-ticks for each subplot
    num_ticks = 6
    left_y_ticks = np.linspace(left_y_min, left_y_max, num_ticks)
    left_y_ticks = np.round(left_y_ticks, 1)
    left_y_ticks = np.unique(left_y_ticks)

    right_y_ticks = np.linspace(right_y_min, right_y_max, num_ticks)
    right_y_ticks = np.round(right_y_ticks, 1)
    right_y_ticks = np.unique(right_y_ticks)

    width = 0.2

    # Calculate x-position mapping: groups spaced by one bar width
    # Center-to-center distance = (n_bars * width) + width = width * (n_bars + 1)
    left_group_spacing = width * (len(left_available_types) + 1)
    right_group_spacing = width * (len(right_available_types) + 1)

    # Map depth values to x-positions
    depth_to_x_left = {
        depth: i * left_group_spacing for i, depth in enumerate(plot_depths)
    }
    depth_to_x_right = {
        depth: i * right_group_spacing for i, depth in enumerate(plot_depths)
    }

    def animate(frame):
        """Animation function that updates both subplots for each epoch"""
        ax_left.clear()
        ax_right.clear()

        # Extract data at this epoch index
        epoch_snapshot = {}
        for _layer_key, layer_stats in layer_data.items():
            epoch_snapshot[layer_key] = {"depth": layer_stats.get("depth")}
            for key, value in layer_stats.items():
                if key != "depth":
                    if isinstance(value, list):
                        if frame < len(value):
                            epoch_snapshot[layer_key][key] = value[frame]
                        else:
                            epoch_snapshot[layer_key][key] = (
                                value[-1] if value else None
                            )
                    else:
                        epoch_snapshot[layer_key][key] = value

        # Create depth_to_layer mapping for this epoch
        depth_to_layer = {}
        for _layer_key, layer_stats in epoch_snapshot.items():
            depth = layer_stats.get("depth")
            if depth is not None:
                depth_to_layer[depth] = layer_stats

        # LEFT SUBPLOT: d'^2/D for vout and upstream
        left_results = {}
        for act_type in left_activation_types:
            act_key = f"{act_type}_d'^2/D"

            y_values = []
            valid_mask = []

            for depth in plot_depths:
                if depth in depth_to_layer:
                    layer_stats = depth_to_layer[depth]
                    if act_key in layer_stats and layer_stats[act_key] is not None:
                        y_values.append(layer_stats[act_key])
                        valid_mask.append(True)
                    else:
                        y_values.append(None)
                        valid_mask.append(False)
                else:
                    y_values.append(None)
                    valid_mask.append(False)

            if any(valid_mask):
                left_results[act_type] = {
                    "depths": plot_depths,
                    "values": y_values,
                    "valid_mask": valid_mask,
                }

        # Plot left subplot
        for i, act_type in enumerate(left_available_types):
            if act_type not in left_results:
                continue

            act_data = left_results[act_type]
            act_depths = np.array(act_data["depths"])
            valid_mask = np.array(act_data["valid_mask"])

            valid_depths = act_depths[valid_mask]
            valid_y_values = np.array(
                [v for j, v in enumerate(act_data["values"]) if valid_mask[j]]
            )

            # Map depths to x-positions and offset by bar position within group
            x = np.array([depth_to_x_left[depth] for depth in valid_depths]) + width * (
                i - (len(left_available_types) - 1) / 2
            )

            ax_left.bar(
                x,
                valid_y_values,
                width=width,
                label=activation_labels[act_type],
                color=colors[act_type],
                capsize=5,
                error_kw={"elinewidth": 1.5, "capthick": 1.5},
            )

        # RIGHT SUBPLOT: d'^2 for exc and inh
        right_results = {}
        for act_type in right_activation_types:
            act_key = f"{act_type}_d'^2"

            y_values = []
            valid_mask = []

            for depth in plot_depths:
                if depth in depth_to_layer:
                    layer_stats = depth_to_layer[depth]
                    if act_key in layer_stats and layer_stats[act_key] is not None:
                        y_values.append(layer_stats[act_key])
                        valid_mask.append(True)
                    else:
                        y_values.append(None)
                        valid_mask.append(False)
                else:
                    y_values.append(None)
                    valid_mask.append(False)

            if any(valid_mask):
                right_results[act_type] = {
                    "depths": plot_depths,
                    "values": y_values,
                    "valid_mask": valid_mask,
                }

        # Plot right subplot
        for i, act_type in enumerate(right_available_types):
            if act_type not in right_results:
                continue

            act_data = right_results[act_type]
            act_depths = np.array(act_data["depths"])
            valid_mask = np.array(act_data["valid_mask"])

            valid_depths = act_depths[valid_mask]
            valid_y_values = np.array(
                [v for j, v in enumerate(act_data["values"]) if valid_mask[j]]
            )

            # Map depths to x-positions and offset by bar position within group
            x = np.array(
                [depth_to_x_right[depth] for depth in valid_depths]
            ) + width * (i - (len(right_available_types) - 1) / 2)

            ax_right.bar(
                x,
                valid_y_values,
                width=width,
                label=activation_labels[act_type],
                color=colors[act_type],
                capsize=5,
                error_kw={"elinewidth": 1.5, "capthick": 1.5},
            )

        # Create depth labels
        depth_labels = []
        for depth in plot_depths:
            if depth == 0:
                depth_labels.append("Soma")
            else:
                depth_labels.append(f"Depth {depth}")

        # Configure left subplot - use mapped x-positions for ticks
        left_x_positions = [depth_to_x_left[depth] for depth in plot_depths]
        ax_left.set_xticks(left_x_positions)
        ax_left.set_xticklabels(depth_labels, rotation=45)
        ax_left.set_xlabel("Branch Depth (Soma to Distal)")
        ax_left.set_ylabel(r"d'$^2$/D")
        ax_left.set_ylim(left_y_min, left_y_max)
        ax_left.set_yticks(left_y_ticks)
        ax_left.legend(loc="upper right")
        ax_left.grid(True, axis="y", alpha=0.3)

        # Configure right subplot - use mapped x-positions for ticks
        right_x_positions = [depth_to_x_right[depth] for depth in plot_depths]
        ax_right.set_xticks(right_x_positions)
        ax_right.set_xticklabels(depth_labels, rotation=45)
        ax_right.set_xlabel("Branch Depth (Soma to Distal)")
        ax_right.set_ylabel(r"d'$^2$")
        ax_right.set_ylim(right_y_min, right_y_max)
        ax_right.set_yticks(right_y_ticks)
        ax_right.legend(loc="upper right")
        ax_right.grid(True, axis="y", alpha=0.3)

        # Set overall title with epoch number
        epoch_num = epoch_numbers[frame] if frame < len(epoch_numbers) else frame
        fig.suptitle(f"Layer Analysis Across Training (Epoch {epoch_num})", fontsize=16)

        return []

    # Create animation
    ani = animation.FuncAnimation(
        fig, animate, frames=num_epochs, interval=1000 / fps, repeat=True, blit=True
    )

    # Save video
    video_filename = "layer_analysis_layer_agg_vs_training_both_metrics"
    logger.info(
        f"Creating {num_epochs}-frame video at {fps:.2f} fps (~{num_epochs/fps:.1f} seconds total)"
    )

    os.makedirs(save_path, exist_ok=True)

    if animation.writers.is_available("ffmpeg"):
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
            video_path = os.path.join(save_path, f"{video_filename}.mp4")
            ani.save(video_path, writer=writer)
            logger.info(f"Video saved to {video_path}")
        except Exception as e:
            logger.warning(f"Error saving MP4 video: {e}")
            logger.info("Falling back to GIF format...")
            try:
                writer = animation.PillowWriter(fps=fps)
                video_path = os.path.join(save_path, f"{video_filename}.gif")
                ani.save(video_path, writer=writer)
                logger.info(f"GIF saved to {video_path}")
            except Exception as e2:
                logger.error(f"Error saving GIF video: {e2}")
    else:
        logger.info("ffmpeg writer not available; saving GIF format...")
        try:
            writer = animation.PillowWriter(fps=fps)
            video_path = os.path.join(save_path, f"{video_filename}.gif")
            ani.save(video_path, writer=writer)
            logger.info(f"GIF saved to {video_path}")
        except Exception as e2:
            logger.error(f"Error saving GIF video: {e2}")

    plt.close(fig)


__all__ = ["CompartmentSNRAnalyzer"]
