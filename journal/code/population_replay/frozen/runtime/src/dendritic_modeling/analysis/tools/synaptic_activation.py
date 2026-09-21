"""
Synaptic activation analysis module.

This module contains classes for analyzing synaptic activation patterns during forward passes,
tracking excitatory and inhibitory synaptic activations with comprehensive statistical summaries.
"""

from typing import Optional

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import run_model_over_analysis_batches
from dendritic_modeling.config import SynapticActivationAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer, TopKLinear
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_pathways import (
    iter_topk_path_modules,
)
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    iter_modules_of_type,
    register_hook_groups,
    register_named_forward_hook_groups,
)


class SynapticActivationAnalyzer(ForwardHookRemovalMixin, AbstractAnalyzer):
    """
    Analyzer for tracking synaptic activation patterns during model inference.

    Uses forward hooks to collect excitatory and inhibitory synaptic activations
    and computes detailed statistical summaries at network, layer, and branch levels.
    """

    def __init__(self, params: SynapticActivationAnalysisParams):
        """
        Initialize the synaptic activation analyzer.

        Args:
            network_summary: Whether to compute network-wide activation statistics
            layer_summary: Whether to compute per-layer activation statistics
            branch_summary: Whether to compute per-branch activation statistics
            raw_summary: Whether to include raw activation data in results
            logspace: Whether to apply log transformation to activations
            downsample: Whether to downsample input data for efficiency
            samples: Number of samples to use if downsampling
        """
        super().__init__("SynapticActivationAnalyzer")
        self.network_summary = getattr(params, "network_summary", False)
        self.layer_summary = getattr(params, "layer_summary", False)
        self.branch_summary = getattr(params, "branch_summary", False)
        self.raw_summary = getattr(params, "raw_summary", False)
        self.logspace = getattr(params, "logspace", False)
        self.downsample = getattr(params, "downsample", False)
        self.samples = getattr(params, "samples", 5)

    def attach_forward_hooks(self, model: BaseModel):
        """
        Attach forward hooks to collect synaptic activations.

        Args:
            model: The model to attach hooks to

        Returns:
            List of hook handles for later removal
        """
        return register_named_forward_hook_groups(
            model,
            DendriticBranchLayer,
            self._attach_layer_hooks,
        )

    def _attach_layer_hooks(
        self, layer_name: str, module: DendriticBranchLayer
    ) -> list[torch.utils.hooks.RemovableHandle]:
        self.data_dict.setdefault(layer_name, {})

        def _register_path_hooks(path_entry):
            path, synapse_module = path_entry
            return [
                self._register_synapse_hook(
                    layer_name,
                    path.pathway,
                    synapse_module,
                )
            ]

        return register_hook_groups(
            iter_topk_path_modules(module),
            _register_path_hooks,
        )

    def _register_synapse_hook(
        self,
        layer_name: str,
        synapse_type: str,
        synapse_module: TopKLinear,
    ) -> torch.utils.hooks.RemovableHandle:
        self._tag_synapse_module(synapse_module, layer_name, synapse_type)
        return synapse_module.register_forward_hook(self.forward_hook)

    def _tag_synapse_module(
        self,
        synapse_module: TopKLinear,
        layer_name: str,
        synapse_type: str,
    ) -> None:
        synapse_module._analysis_layer_name = layer_name
        synapse_module._analysis_synapse_type = synapse_type

    def forward_hook(
        self,
        module: TopKLinear,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ):
        """
        Forward hook to collect synaptic activations.

        Args:
            module: The synaptic TopK layer.
            input: Input tuple to the synaptic layer.
            output: Branch current output.
        """
        key = module._analysis_layer_name
        synapse_type = module._analysis_synapse_type
        # Record the per-synapse contribution (input_i * pruned_weight_ij)
        # with shape (..., in_features, n_branches) so per-synapse resolution is preserved.
        # `output` is the summed branch current (== BranchActivationAnalyzer's signal), which
        # collapsed per-synapse detail and made this analyzer redundant with branch_activation.
        x = input[0]
        per_synapse = x[..., None] * module.pruned_weight().t()
        self.data_dict[key].setdefault(synapse_type, []).append(
            per_synapse.detach().cpu()
        )

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ):
        """
        Analyze synaptic activations in the model.

        Args:
            model: The model to analyze
            test_dataset: Test dataset for analysis
            device: Device to run analysis on
            save_path: Path to save results (optional)
            filename: Filename for saved results

        Returns:
            Dictionary containing activation analysis if save_path is None, otherwise None
        """
        if not self._has_dendritic_branch_layers(model):
            return

        self.data_dict = {}
        self._collect_activation_batches(
            model=model,
            test_dataset=test_dataset,
            device=device,
            runtime=runtime,
        )
        self._concatenate_activation_batches()
        results = self._build_results_from_activations()

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results

    @staticmethod
    def _has_dendritic_branch_layers(model: BaseModel) -> bool:
        core = getattr(model, "core_network", model)
        return any(iter_modules_of_type(core, DendriticBranchLayer))

    def _collect_activation_batches(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str,
        runtime: Optional[EvaluationRuntimeConfig],
    ) -> None:
        sample_cap = self.samples if self.downsample else None
        run_model_over_analysis_batches(
            model=model,
            dataset=test_dataset,
            device=device,
            runtime=runtime,
            explicit_max_samples=sample_cap,
            attach_hooks=lambda: self.attach_forward_hooks(model),
            remove_hooks=self.remove_forward_hooks,
        )

    def _concatenate_activation_batches(self) -> None:
        for module_dict in self.data_dict.values():
            for key, value in list(module_dict.items()):
                if isinstance(value, list) and value:
                    module_dict[key] = torch.cat(value, dim=0)

    def _build_results_from_activations(self) -> dict:
        network_values: dict[str, list[torch.Tensor]] = {}
        layer_summary: dict[str, dict] = {}
        branch_summary: dict[str, dict] = {}

        for name, module_dict in self.data_dict.items():
            self._add_layer_summaries(
                name=name,
                module_dict=module_dict,
                network_values=network_values,
                layer_summary=layer_summary,
                branch_summary=branch_summary,
            )

        return self._compile_results(
            network_values=network_values,
            layer_summary=layer_summary,
            branch_summary=branch_summary,
        )

    def _add_layer_summaries(
        self,
        name: str,
        module_dict: dict,
        network_values: dict[str, list[torch.Tensor]],
        layer_summary: dict[str, dict],
        branch_summary: dict[str, dict],
    ) -> None:
        if self.layer_summary:
            layer_summary[name] = {}
        if self.branch_summary:
            branch_summary[name] = {}

        for synapse_type, activation in list(module_dict.items()):
            if not isinstance(activation, torch.Tensor):
                continue
            self._add_synapse_summary(
                module_dict=module_dict,
                layer_name=name,
                synapse_type=synapse_type,
                activation=self._transform_activation(activation),
                network_values=network_values,
                layer_summary=layer_summary,
                branch_summary=branch_summary,
            )

    def _add_synapse_summary(
        self,
        module_dict: dict,
        layer_name: str,
        synapse_type: str,
        activation: torch.Tensor,
        network_values: dict[str, list[torch.Tensor]],
        layer_summary: dict[str, dict],
        branch_summary: dict[str, dict],
    ) -> None:
        if self.network_summary or self.layer_summary:
            flat = self._filtered_flat_activation(activation)

        if self.network_summary:
            network_values.setdefault(synapse_type, []).append(flat)

        if self.layer_summary:
            layer_summary[layer_name][synapse_type] = {
                "mean": flat.mean().item(),
                "std": flat.std(unbiased=False).item(),
            }

        if self.branch_summary:
            branch_summary[layer_name][synapse_type] = {
                "mean": activation.mean(dim=0).tolist(),
                "std": activation.std(dim=0, unbiased=False).tolist(),
            }

        if self.raw_summary:
            module_dict[synapse_type] = activation.tolist()

    def _transform_activation(self, activation: torch.Tensor) -> torch.Tensor:
        if self.logspace:
            return torch.log(activation + 1e-9)
        return activation

    def _filtered_flat_activation(self, activation: torch.Tensor) -> torch.Tensor:
        flat = activation.flatten()
        if self.logspace:
            flat = flat[flat > -20]
        else:
            flat = flat[flat > 0]
        if flat.numel() == 0:
            flat = torch.zeros(1)
        return flat

    def _compile_results(
        self,
        network_values: dict[str, list[torch.Tensor]],
        layer_summary: dict[str, dict],
        branch_summary: dict[str, dict],
    ) -> dict:
        results = {"logspace": self.logspace}

        if self.network_summary:
            results["network_summary"] = {
                synapse_type: {
                    "mean": torch.cat(values).mean().item(),
                    "std": torch.cat(values).std(unbiased=False).item(),
                }
                for synapse_type, values in network_values.items()
                if values
            }

        if self.layer_summary:
            results["layer_summary"] = layer_summary

        if self.branch_summary:
            results["branch_summary"] = branch_summary

        if self.raw_summary:
            results["raw_summary"] = self.data_dict

        return results
