"""
Branch activation analysis module.

This module contains classes for analyzing both synaptic and branch-level activation patterns
during forward passes, providing detailed statistical analysis at multiple levels.
"""

from typing import Optional

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.core.information_parts.information_hooks_mixin import (
    _enable_analysis_current_capture,
    _resolve_synaptic_branch_currents,
    _restore_analysis_current_capture,
)
from dendritic_modeling.analysis.utils.runtime import run_model_over_analysis_batches
from dendritic_modeling.config import BranchActivationAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    iter_modules_of_type,
    register_named_forward_hooks,
)


class BranchActivationAnalyzer(ForwardHookRemovalMixin, AbstractAnalyzer):
    """
    Analyzer for tracking both synaptic and branch activation patterns during model inference.

    Uses forward hooks to collect excitatory synaptic activations, inhibitory synaptic activations,
    and branch output activations, computing detailed statistical summaries at multiple levels.
    """

    def __init__(self, params: BranchActivationAnalysisParams):
        """
        Initialize the branch activation analyzer.

        Args:
            synapse_activations: Whether to collect excitatory and inhibitory synapse activations
            branch_activations: Whether to collect branch output activations (Vout)
            network_summary: Whether to compute network-wide activation statistics
            layer_summary: Whether to compute per-layer activation statistics
            branch_summary: Whether to compute per-branch activation statistics
            raw_summary: Whether to include raw activation data in results
            logspace: Whether to apply log transformation to activations
            downsample: Whether to downsample input data for efficiency
            samples: Number of samples to use if downsampling
        """
        super().__init__("BranchActivationAnalyzer")
        self.synapse_activations = getattr(params, "synapse_activations", False)
        self.branch_activations = getattr(params, "branch_activations", False)
        self.network_summary = getattr(params, "network_summary", False)
        self.layer_summary = getattr(params, "layer_summary", False)
        self.branch_summary = getattr(params, "branch_summary", False)
        self.raw_summary = getattr(params, "raw_summary", False)
        self.logspace = getattr(params, "logspace", False)
        self.downsample = getattr(params, "downsample", False)
        self.samples = getattr(params, "samples", 5)

    def attach_forward_hooks(self, model: BaseModel):
        """
        Attach forward hooks to collect branch activations.

        Args:
            model: The model to attach hooks to

        Returns:
            List of hook handles for later removal
        """
        self._analysis_hook_modules = []
        try:
            return register_named_forward_hooks(
                model,
                DendriticBranchLayer,
                self.forward_hook,
                predicate=self._should_hook_branch_layer,
                prepare=self._prepare_branch_layer_hook,
                with_kwargs=True,
            )
        except Exception:
            self._restore_current_capture()
            raise

    def _should_hook_branch_layer(
        self,
        layer_name: str,
        _module: DendriticBranchLayer,
    ) -> bool:
        return layer_name not in self.data_dict

    def _prepare_branch_layer_hook(
        self,
        layer_name: str,
        module: DendriticBranchLayer,
    ) -> None:
        self.data_dict[layer_name] = {}
        module._name = layer_name
        previous_store = _enable_analysis_current_capture(module)
        self._analysis_hook_modules.append((module, previous_store))

    def _restore_current_capture(self) -> None:
        for module, previous_store in getattr(self, "_analysis_hook_modules", []):
            _restore_analysis_current_capture(module, previous_store)
        self._analysis_hook_modules = []

    def remove_forward_hooks(self, handles) -> None:
        """Remove hooks and restore each branch layer's capture state."""
        super().remove_forward_hooks(handles)
        self._restore_current_capture()

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input_args: tuple[torch.Tensor, ...],
        input_kwargs: dict,
        output: torch.Tensor,
    ):
        """
        Forward hook to collect branch activations.

        Args:
            module: The dendritic branch layer
            input: Input tuple (excitatory, inhibitory, optional)
            output: Module output
        """
        excitatory_input = input_kwargs.get(
            "x", input_args[0] if len(input_args) > 0 else None
        )
        inhibitory_input = input_kwargs.get(
            "inhibitory_input", input_args[1] if len(input_args) > 1 else None
        )
        key = module._name
        excitation, inhibition, _has_no_synapses = _resolve_synaptic_branch_currents(
            module,
            excitatory_input,
            inhibitory_input,
            output,
            getattr(module, "_last_analysis_currents", None),
        )

        # Collect excitatory synapse activations
        if self.synapse_activations and module.branch_excitation is not None:
            self.data_dict[key].setdefault("exc", []).append(excitation.detach().cpu())

        # Collect inhibitory synapse activations
        if (
            self.synapse_activations
            and module.branch_inhibition is not None
            and inhibitory_input is not None
        ):
            self.data_dict[key].setdefault("inh", []).append(inhibition.detach().cpu())

        # Collect branch output activations
        if self.branch_activations:
            self.data_dict[key].setdefault("Vout", []).append(output.detach().cpu())

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
        Analyze branch activations in the model.

        Args:
            model: The model to analyze
            test_dataset: Test dataset for analysis
            device: Device to run analysis on
            save_path: Path to save results (optional)
            filename: Filename for saved results

        Returns:
            Dictionary containing branch activation analysis if save_path is None, otherwise None
        """
        core = getattr(model, "core_network", model)
        if not any(iter_modules_of_type(core, DendriticBranchLayer)):
            return

        self.data_dict = {}

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

        for module_dict in self.data_dict.values():
            for key, value in list(module_dict.items()):
                if isinstance(value, list) and value:
                    module_dict[key] = torch.cat(value, dim=0)

        # Initialize summary structures
        if self.network_summary:
            if self.synapse_activations:
                exc_net = []
                inh_net = []
            if self.branch_activations:
                Vout_net = []

        if self.layer_summary:
            layer_summary = {}

        if self.branch_summary:
            branch_summary = {}

        # Process collected activation data with detailed debugging
        for name, module_dict in self.data_dict.items():
            module_dict: dict
            if self.layer_summary:
                layer_summary[name] = {}

            if self.branch_summary:
                branch_summary[name] = {}

            # Process excitatory synapse activations
            if "exc" in module_dict.keys():
                exc: torch.Tensor = module_dict["exc"]

                if self.logspace:
                    exc = torch.log(exc)

                if self.network_summary:
                    exc_net.append(exc.flatten())

                if self.layer_summary:
                    layer_summary[name]["exc"] = {}
                    layer_summary[name]["exc"]["mean"] = torch.mean(exc).item()
                    layer_summary[name]["exc"]["std"] = torch.std(exc).item()

                if self.branch_summary:
                    branch_summary[name]["exc"] = {}
                    branch_summary[name]["exc"]["mean"] = torch.mean(
                        exc, dim=0
                    ).tolist()
                    branch_summary[name]["exc"]["std"] = torch.std(exc, dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]["exc"] = exc.tolist()

            # Process inhibitory synapse activations
            if "inh" in module_dict.keys():
                inh: torch.Tensor = module_dict["inh"]

                if self.logspace:
                    inh = torch.log(inh)

                if self.network_summary:
                    inh_net.append(inh.flatten())

                if self.layer_summary:
                    layer_summary[name]["inh"] = {}
                    layer_summary[name]["inh"]["mean"] = torch.mean(inh).item()
                    layer_summary[name]["inh"]["std"] = torch.std(inh).item()

                if self.branch_summary:
                    branch_summary[name]["inh"] = {}
                    branch_summary[name]["inh"]["mean"] = torch.mean(
                        inh, dim=0
                    ).tolist()
                    branch_summary[name]["inh"]["std"] = torch.std(inh, dim=0).tolist()

                if self.raw_summary:
                    self.data_dict[name]["inh"] = inh.tolist()

            # Process branch output activations
            if "Vout" in module_dict.keys():
                Vout: torch.Tensor = module_dict["Vout"]

                if self.logspace:
                    Vout = torch.log(Vout)

                if self.network_summary:
                    Vout_net.append(Vout.flatten())

                if self.layer_summary:
                    layer_summary[name]["Vout"] = {}
                    layer_summary[name]["Vout"]["mean"] = torch.mean(Vout).item()
                    layer_summary[name]["Vout"]["std"] = torch.std(Vout).item()

                if self.branch_summary:
                    branch_summary[name]["Vout"] = {}
                    branch_summary[name]["Vout"]["mean"] = torch.mean(
                        Vout, dim=0
                    ).tolist()
                    branch_summary[name]["Vout"]["std"] = torch.std(
                        Vout, dim=0
                    ).tolist()

                if self.raw_summary:
                    self.data_dict[name]["Vout"] = Vout.tolist()

        # Compile results
        results = {"logspace": self.logspace}

        if self.network_summary:
            network_summary = {}

            if self.synapse_activations and exc_net:
                exc_net = torch.cat(exc_net, dim=0)
                network_summary["exc"] = {}
                network_summary["exc"]["mean"] = torch.mean(exc_net).item()
                network_summary["exc"]["std"] = torch.std(exc_net).item()

            if self.synapse_activations and inh_net:
                inh_net = torch.cat(inh_net, dim=0)
                network_summary["inh"] = {}
                network_summary["inh"]["mean"] = torch.mean(inh_net).item()
                network_summary["inh"]["std"] = torch.std(inh_net).item()

            if self.branch_activations and Vout_net:
                Vout_net = torch.cat(Vout_net, dim=0)
                network_summary["Vout"] = {}
                network_summary["Vout"]["mean"] = torch.mean(Vout_net).item()
                network_summary["Vout"]["std"] = torch.std(Vout_net).item()

            results["network_summary"] = network_summary

        if self.layer_summary:
            results["layer_summary"] = layer_summary

        if self.branch_summary:
            results["branch_summary"] = branch_summary

        if self.raw_summary:
            results["raw_summary"] = self.data_dict

        if save_path is not None:
            save_dict(results, save_path, filename)
        else:
            return results
