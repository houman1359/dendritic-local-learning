"""
Hierarchical correlation analysis for dendritic networks.

This module provides correlation analysis at multiple levels of granularity:
- Synaptic level: Individual synapse correlations (using sampling)
- Branch level: Correlations between branch activations
- Layer level: Aggregated correlations within layers
- Network level: Full network correlation structure

The implementation is aligned with the information analysis module for consistency.
"""

import logging
import os
import traceback
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from tqdm import tqdm

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    materialize_dataset,
)
from dendritic_modeling.config import CorrelationAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    register_hook_groups,
    register_named_forward_hook_groups,
    register_named_forward_hooks,
    run_with_forward_hooks,
)

logger = logging.getLogger(__name__)

_CORRELATION_COMPUTE_DEFAULTS = {
    "computation_level": "layer_branch",
    "max_samples": 5000,
    "batch_process": True,
    "per_layer_analysis": True,
    "per_einet_analysis": True,
    "per_neuron_analysis": False,
}
_CORRELATION_SAMPLING_DEFAULTS = {
    "n_synapse_pairs": 10000,
    "sampling_strategy": "stratified",
    "stratified_categories": ["within_branch", "within_layer", "across_layer"],
    "max_pairs_per_category": 1000,
}


def _prefer_nested_config_value(nested_cfg, flat_cfg, name: str, default):
    """Use legacy flat values when the nested config still has its default."""
    nested_value = getattr(nested_cfg, name, default)
    flat_value = getattr(flat_cfg, name, default)
    if flat_value != default and nested_value == default:
        return flat_value
    return nested_value


class CorrelationAnalyzer(ForwardHookRemovalMixin, AbstractAnalyzer):
    """Hierarchical correlation analyzer for dendritic networks."""

    def __init__(self, params: CorrelationAnalysisParams):
        """
        Initialize correlation analyzer.

        Args:
            params: Configuration parameters for correlation analysis
        """
        super().__init__()
        self.logger = logger
        self.params = params

        def _norm(x: str) -> str:
            return str(x).strip().lower().replace("-", "_").replace(" ", "_")

        # New (preferred) grouped config blocks (optional; keep backward compatibility)
        selection_cfg = getattr(params, "selection", None)
        compute_cfg = getattr(params, "compute", None)
        sampling_cfg = getattr(params, "sampling", None)

        # Core parameters
        self.computation_level = _prefer_nested_config_value(
            compute_cfg,
            params,
            "computation_level",
            _CORRELATION_COMPUTE_DEFAULTS["computation_level"],
        )

        # ------------------------------------------------------------------
        # Legacy boolean selection (still supported).
        # ------------------------------------------------------------------
        self.compute_noise_correlations = bool(
            getattr(params, "compute_noise_correlations", True)
        )
        self.compute_signal_correlations = bool(
            getattr(params, "compute_signal_correlations", True)
        )
        self.compute_total_correlations = bool(
            getattr(params, "compute_total_correlations", True)
        )
        self.compute_tuning_curves = bool(
            getattr(params, "compute_tuning_curves", True)
        )

        self.analyze_EE = bool(getattr(params, "analyze_EE", True))
        self.analyze_II = bool(getattr(params, "analyze_II", True))
        self.analyze_EI = bool(getattr(params, "analyze_EI", True))
        self.analyze_E_output = bool(getattr(params, "analyze_E_output", True))
        self.analyze_I_output = bool(getattr(params, "analyze_I_output", True))
        self.analyze_output_output = bool(
            getattr(params, "analyze_output_output", True)
        )

        self.per_layer_analysis = bool(
            _prefer_nested_config_value(
                compute_cfg,
                params,
                "per_layer_analysis",
                _CORRELATION_COMPUTE_DEFAULTS["per_layer_analysis"],
            )
        )
        self.per_einet_analysis = bool(
            _prefer_nested_config_value(
                compute_cfg,
                params,
                "per_einet_analysis",
                _CORRELATION_COMPUTE_DEFAULTS["per_einet_analysis"],
            )
        )
        self.per_neuron_analysis = bool(
            _prefer_nested_config_value(
                compute_cfg,
                params,
                "per_neuron_analysis",
                _CORRELATION_COMPUTE_DEFAULTS["per_neuron_analysis"],
            )
        )

        # ------------------------------------------------------------------
        # Preferred list-based selectors (override booleans if non-empty).
        # ------------------------------------------------------------------
        components_raw = list(
            getattr(selection_cfg, "components", getattr(params, "components", []))
            or []
        )
        if components_raw:
            comps = {_norm(x) for x in components_raw if str(x).strip()}
            self.compute_total_correlations = bool({"total", "all"}.intersection(comps))
            self.compute_noise_correlations = bool({"noise"}.intersection(comps))
            self.compute_signal_correlations = bool({"signal"}.intersection(comps))
            self.compute_tuning_curves = bool(
                {"tuning", "tuning_curves"}.intersection(comps)
            )

        pairs_raw = list(
            getattr(selection_cfg, "pairs", getattr(params, "pairs", [])) or []
        )
        if pairs_raw:
            pairs = {_norm(x) for x in pairs_raw if str(x).strip()}
            self.analyze_EE = bool({"ee"}.intersection(pairs))
            self.analyze_II = bool({"ii"}.intersection(pairs))
            self.analyze_EI = bool({"ei"}.intersection(pairs))
            self.analyze_E_output = bool(
                {"e_output", "e_vout", "exc_output", "exc_vout"}.intersection(pairs)
            )
            self.analyze_I_output = bool(
                {"i_output", "i_vout", "inh_output", "inh_vout"}.intersection(pairs)
            )
            self.analyze_output_output = bool(
                {"output_output", "vout_vout", "out_out"}.intersection(pairs)
            )

        scopes_raw = list(
            getattr(selection_cfg, "scopes", getattr(params, "scopes", [])) or []
        )
        if scopes_raw:
            scopes = {_norm(x) for x in scopes_raw if str(x).strip()}
            self.per_layer_analysis = bool({"layer", "per_layer"}.intersection(scopes))
            self.per_einet_analysis = bool({"einet", "per_einet"}.intersection(scopes))
            self.per_neuron_analysis = bool(
                {"neuron", "per_neuron"}.intersection(scopes)
            )

        # Sampling parameters (for synaptic level) (with defaults)
        self.n_synapse_pairs = _prefer_nested_config_value(
            sampling_cfg,
            params,
            "n_synapse_pairs",
            _CORRELATION_SAMPLING_DEFAULTS["n_synapse_pairs"],
        )
        self.sampling_strategy = _prefer_nested_config_value(
            sampling_cfg,
            params,
            "sampling_strategy",
            _CORRELATION_SAMPLING_DEFAULTS["sampling_strategy"],
        )
        self.stratified_categories = _prefer_nested_config_value(
            sampling_cfg,
            params,
            "stratified_categories",
            _CORRELATION_SAMPLING_DEFAULTS["stratified_categories"],
        )
        self.max_pairs_per_category = _prefer_nested_config_value(
            sampling_cfg,
            params,
            "max_pairs_per_category",
            _CORRELATION_SAMPLING_DEFAULTS["max_pairs_per_category"],
        )

        # Performance (with defaults)
        self.max_samples = _prefer_nested_config_value(
            compute_cfg,
            params,
            "max_samples",
            _CORRELATION_COMPUTE_DEFAULTS["max_samples"],
        )
        self.batch_process = _prefer_nested_config_value(
            compute_cfg,
            params,
            "batch_process",
            _CORRELATION_COMPUTE_DEFAULTS["batch_process"],
        )

        self.logger.info(
            f"Initialized CorrelationAnalyzer with computation_level: {self.computation_level}"
        )

    def attach_forward_hooks(self, model: BaseModel) -> list:
        """Attach forward hooks to collect activations."""
        self.data_dict = {}
        if self.computation_level == "synaptic":
            handles = self._attach_synaptic_hooks(model)
        else:
            handles = self._attach_branch_level_hooks(model)

        self.logger.info(f"Attached hooks to {len(handles)} modules")
        return handles

    def _attach_synaptic_hooks(
        self, model: BaseModel
    ) -> list[torch.utils.hooks.RemovableHandle]:
        return register_named_forward_hook_groups(
            model,
            DendriticBranchLayer,
            self._attach_synaptic_layer_hooks,
            prepare=self._prepare_synaptic_layer_hook,
        )

    def _prepare_synaptic_layer_hook(
        self,
        layer_name: str,
        _module: DendriticBranchLayer,
    ) -> None:
        self.data_dict[layer_name] = {}

    def _attach_synaptic_layer_hooks(
        self, layer_name: str, module: DendriticBranchLayer
    ) -> list[torch.utils.hooks.RemovableHandle]:
        def _hook_specs():
            if module.branch_excitation is not None:
                yield (
                    module.branch_excitation,
                    self._make_synapse_hook(layer_name, "E"),
                )

            if module.branch_inhibition is not None:
                yield (
                    module.branch_inhibition,
                    self._make_synapse_hook(layer_name, "I"),
                )

            yield module, self._make_output_hook(layer_name)

        def _register_hook(spec):
            hook_module, hook = spec
            return [hook_module.register_forward_hook(hook)]

        return register_hook_groups(_hook_specs(), _register_hook)

    def _make_synapse_hook(self, layer_name: str, synapse_type: str):
        def hook(module, input, output):
            x = input[0] if isinstance(input, tuple) else input
            with torch.no_grad():
                weights = module.weight()
                active_indices = torch.topk(
                    module.pre_w,
                    module.K,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )[1]

                batch_size = x.shape[0]
                n_branches = active_indices.shape[0]
                x_expanded = x.unsqueeze(1).expand(-1, n_branches, -1)
                indices_expanded = active_indices.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
                synaptic_inputs = torch.gather(x_expanded, 2, indices_expanded)

                weights_active = torch.gather(weights, 1, active_indices)
                weights_expanded = weights_active.unsqueeze(0).expand(
                    batch_size, -1, -1
                )
                synaptic_activations = synaptic_inputs * weights_expanded

                self.data_dict[layer_name][
                    f"{synapse_type}_synaptic"
                ] = synaptic_activations
                self.data_dict[layer_name][f"{synapse_type}_indices"] = active_indices
                self.data_dict[layer_name][f"{synapse_type}_weights"] = weights_active
                self.data_dict[layer_name][f"{synapse_type}_branch_output"] = output

        return hook

    def _make_output_hook(self, layer_name: str):
        def hook(module, input, output):
            self.data_dict[layer_name]["output"] = output

        return hook

    def _attach_branch_level_hooks(
        self, model: BaseModel
    ) -> list[torch.utils.hooks.RemovableHandle]:
        return register_named_forward_hooks(
            model,
            DendriticBranchLayer,
            self.forward_hook,
            predicate=self._should_attach_branch_level_hook,
            prepare=self._prepare_branch_level_hook,
        )

    @staticmethod
    def _should_attach_branch_level_hook(
        _layer_name: str,
        module: DendriticBranchLayer,
    ) -> bool:
        return (
            module.branch_excitation is not None
            and module.branch_inhibition is not None
        )

    def _prepare_branch_level_hook(
        self,
        layer_name: str,
        module: DendriticBranchLayer,
    ) -> None:
        self.data_dict[layer_name] = {}
        module._name = layer_name

    def forward_hook(
        self, module: DendriticBranchLayer, input: tuple, output: torch.Tensor
    ):
        """Standard forward hook for branch/layer level analysis."""
        excitatory_input, inhibitory_input, _ = input

        # Get branch outputs
        excitation_per_branch = module.branch_excitation(excitatory_input)
        inhibition_per_branch = module.branch_inhibition(inhibitory_input)

        # Store data
        self.data_dict[module._name]["excitation"] = excitation_per_branch
        self.data_dict[module._name]["inhibition"] = inhibition_per_branch
        self.data_dict[module._name]["output"] = output
        self.data_dict[module._name]["excitatory_input"] = excitatory_input
        self.data_dict[module._name]["inhibitory_input"] = inhibitory_input

    def _compute_correlation_matrix(
        self, X: np.ndarray, Y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute correlation matrix between X and Y."""
        if Y is None:
            # Self-correlation
            return np.corrcoef(X.T)
        else:
            # Cross-correlation using vectorized operations
            # Standardize the data
            X_std = np.std(X, axis=0)
            Y_std = np.std(Y, axis=0)

            # Find non-zero variance columns
            valid_X = X_std > 0
            valid_Y = Y_std > 0

            # Initialize correlation matrix
            n_x, n_y = X.shape[1], Y.shape[1]
            corr_matrix = np.zeros((n_x, n_y))

            if np.any(valid_X) and np.any(valid_Y):
                # Compute correlations only for valid columns
                X_valid = X[:, valid_X]
                Y_valid = Y[:, valid_Y]

                # Center the data
                X_centered = X_valid - np.mean(X_valid, axis=0)
                Y_centered = Y_valid - np.mean(Y_valid, axis=0)

                # Normalize
                X_norm = X_centered / X_std[valid_X]
                Y_norm = Y_centered / Y_std[valid_Y]

                # Compute correlations using matrix multiplication
                valid_corr = np.dot(X_norm.T, Y_norm) / X.shape[0]

                # Fill in the valid correlations
                corr_matrix[np.ix_(valid_X, valid_Y)] = valid_corr

            return corr_matrix

    def _compute_conditional_correlations(
        self, X: np.ndarray, Y: Optional[np.ndarray], labels: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute correlations conditioned on class labels."""
        results = {}
        unique_labels = np.unique(labels)

        # Total correlation
        if self.compute_total_correlations:
            results["total"] = self._compute_correlation_matrix(X, Y)

        # Per-class and noise correlations
        if self.compute_noise_correlations:
            per_class_corrs = []
            for label in unique_labels:
                mask = labels == label
                X_class = X[mask]
                Y_class = Y[mask] if Y is not None else None

                class_corr = self._compute_correlation_matrix(X_class, Y_class)
                results[f"class_{label}"] = class_corr
                per_class_corrs.append(class_corr)

            # Noise correlation (average within-class)
            results["noise"] = np.mean(per_class_corrs, axis=0)

        # Signal correlation (between-class difference)
        if self.compute_signal_correlations and len(unique_labels) == 2:
            results["signal"] = (
                results[f"class_{unique_labels[1]}"]
                - results[f"class_{unique_labels[0]}"]
            )

        return results

    def _compute_synaptic_correlations(
        self, data: dict[str, torch.Tensor], labels: np.ndarray
    ) -> dict[str, Any]:
        """Compute correlations at synaptic level using sampling."""
        results = {}

        # Extract synaptic data
        E_syn = data.get("E_synaptic")
        I_syn = data.get("I_synaptic")
        E_indices = data.get("E_indices")
        I_indices = data.get("I_indices")

        if E_syn is None and I_syn is None:
            return results

        # Convert to numpy
        if E_syn is not None:
            E_syn_np = E_syn.cpu().numpy()
        if I_syn is not None:
            I_syn_np = I_syn.cpu().numpy()

        # Sample and compute correlations for each type
        for corr_type in ["EE", "II", "EI"]:
            if corr_type == "EE" and self.analyze_EE and E_syn is not None:
                pairs = self._sample_synapse_pairs(E_indices, corr_type)
                results[corr_type] = self._compute_sampled_correlations(
                    E_syn_np, E_syn_np, pairs, labels, corr_type
                )
            elif corr_type == "II" and self.analyze_II and I_syn is not None:
                pairs = self._sample_synapse_pairs(I_indices, corr_type)
                results[corr_type] = self._compute_sampled_correlations(
                    I_syn_np, I_syn_np, pairs, labels, corr_type
                )
            elif (
                corr_type == "EI"
                and self.analyze_EI
                and E_syn is not None
                and I_syn is not None
            ):
                pairs = self._sample_cross_synapse_pairs(E_indices, I_indices)
                results[corr_type] = self._compute_sampled_correlations(
                    E_syn_np, I_syn_np, pairs, labels, corr_type
                )

        return results

    def _sample_synapse_pairs(
        self, indices: torch.Tensor, corr_type: str
    ) -> list[tuple]:
        """Sample synapse pairs for correlation computation."""
        n_branches, n_synapses = indices.shape
        all_synapses = [(b, s) for b in range(n_branches) for s in range(n_synapses)]

        if self.sampling_strategy == "stratified":
            pairs = []

            # Within-branch pairs
            if "within_branch" in self.stratified_categories:
                for branch in range(n_branches):
                    n_pairs = min(
                        n_synapses * (n_synapses - 1) // 2,
                        self.max_pairs_per_category // n_branches,
                    )
                    for _ in range(n_pairs):
                        idx1, idx2 = np.random.choice(n_synapses, 2, replace=False)
                        pairs.append(((branch, idx1), (branch, idx2)))

            # Within-layer pairs (across branches)
            if "within_layer" in self.stratified_categories:
                n_pairs = min(
                    self.max_pairs_per_category,
                    len(all_synapses) * (len(all_synapses) - 1) // 2,
                )
                for _ in range(n_pairs):
                    syn1, syn2 = np.random.choice(len(all_synapses), 2, replace=False)
                    pairs.append((all_synapses[syn1], all_synapses[syn2]))

        else:  # Random sampling
            n_total_pairs = min(
                self.n_synapse_pairs, len(all_synapses) * (len(all_synapses) - 1) // 2
            )
            pairs = []
            for _ in range(n_total_pairs):
                syn1, syn2 = np.random.choice(len(all_synapses), 2, replace=False)
                pairs.append((all_synapses[syn1], all_synapses[syn2]))

        return pairs

    def _sample_cross_synapse_pairs(
        self, E_indices: torch.Tensor, I_indices: torch.Tensor
    ) -> list[tuple]:
        """Sample E-I synapse pairs."""
        E_branches, E_synapses = E_indices.shape
        I_branches, I_synapses = I_indices.shape

        E_all = [(b, s) for b in range(E_branches) for s in range(E_synapses)]
        I_all = [(b, s) for b in range(I_branches) for s in range(I_synapses)]

        n_sample = min(self.n_synapse_pairs, len(E_all) * len(I_all))

        pairs = []
        for _ in range(n_sample):
            e_idx = np.random.choice(len(E_all))
            i_idx = np.random.choice(len(I_all))
            pairs.append((E_all[e_idx], I_all[i_idx]))

        return pairs

    def _compute_sampled_correlations(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        pairs: list[tuple],
        labels: np.ndarray,
        corr_type: str,
    ) -> dict[str, Any]:
        """Compute correlations for sampled pairs."""
        results = {"n_pairs": len(pairs)}

        # Compute correlations for all pairs
        all_correlations = []
        for (b1, s1), (b2, s2) in pairs:
            x_series = X[:, b1, s1]
            y_series = Y[:, b2, s2]

            if np.std(x_series) > 0 and np.std(y_series) > 0:
                corr, _ = pearsonr(x_series, y_series)
                all_correlations.append(corr)

        if not all_correlations:
            return results

        all_correlations = np.array(all_correlations)

        # Total statistics
        if self.compute_total_correlations:
            results["total"] = {
                "mean": np.mean(all_correlations),
                "std": np.std(all_correlations),
                "median": np.median(all_correlations),
                "q25": np.percentile(all_correlations, 25),
                "q75": np.percentile(all_correlations, 75),
            }

        # Class-conditional correlations
        if self.compute_noise_correlations:
            unique_labels = np.unique(labels)
            class_correlations = {}

            for label in unique_labels:
                mask = labels == label
                label_corrs = []

                for (b1, s1), (b2, s2) in pairs:
                    x_series = X[mask, b1, s1]
                    y_series = Y[mask, b2, s2]

                    if np.std(x_series) > 0 and np.std(y_series) > 0:
                        corr, _ = pearsonr(x_series, y_series)
                        label_corrs.append(corr)

                if label_corrs:
                    class_correlations[label] = np.array(label_corrs)
                    results[f"class_{label}"] = {
                        "mean": np.mean(label_corrs),
                        "std": np.std(label_corrs),
                    }

            # Noise correlation
            if class_correlations:
                all_within_class = np.concatenate(list(class_correlations.values()))
                results["noise"] = {
                    "mean": np.mean(all_within_class),
                    "std": np.std(all_within_class),
                }

            # Signal correlation
            if self.compute_signal_correlations and len(unique_labels) == 2:
                results["signal"] = {
                    "mean": results[f"class_{unique_labels[1]}"]["mean"]
                    - results[f"class_{unique_labels[0]}"]["mean"]
                }

        return results

    def _compute_branch_correlations(
        self, E: np.ndarray, Inh: np.ndarray, Vout: np.ndarray, labels: np.ndarray
    ) -> dict[str, Any]:
        """Compute correlations at branch level."""
        results = {}

        # Ensure 2D arrays
        E = E.reshape(E.shape[0], -1) if E.ndim > 2 else E
        Inh = Inh.reshape(Inh.shape[0], -1) if Inh.ndim > 2 else Inh
        Vout = Vout.reshape(Vout.shape[0], -1) if Vout.ndim > 2 else Vout

        self.logger.debug(
            f"Computing correlations - E shape: {E.shape}, Inh shape: {Inh.shape}, Vout shape: {Vout.shape}"
        )

        # Compute requested correlations with progress
        correlation_tasks = []
        if self.analyze_EE:
            correlation_tasks.append(("EE", E, None))
        if self.analyze_II:
            correlation_tasks.append(("II", Inh, None))
        if self.analyze_EI:
            correlation_tasks.append(("EI", E, Inh))
        if self.analyze_E_output:
            correlation_tasks.append(("E_Vout", E, Vout))
        if self.analyze_I_output:
            correlation_tasks.append(("I_Vout", Inh, Vout))
        if self.analyze_output_output and Vout.shape[1] > 1:
            correlation_tasks.append(("Vout_Vout", Vout, None))

        for name, x, y in tqdm(
            correlation_tasks, desc="Computing correlations", leave=False
        ):
            self.logger.debug(f"Computing {name} correlations")
            results[name] = self._compute_conditional_correlations(x, y, labels)

        return results

    def _compute_tuning_curves(
        self, E: np.ndarray, Inh: np.ndarray, Vout: np.ndarray, labels: np.ndarray
    ) -> dict[str, Any]:
        """Compute tuning curves (class selectivity) for multi-class scenarios.

        For each unit, computes:
        - Mean response per class
        - Preferred class (class with highest mean response)
        - Selectivity index (normalized difference between best and second-best class)
        - For binary classification: traditional difference between classes
        """
        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)
        n_units_E = E.shape[1] if E.ndim > 1 else 1
        n_units_I = Inh.shape[1] if Inh.ndim > 1 else 1
        n_units_Vout = Vout.shape[1] if Vout.ndim > 1 else 1

        results = {
            "n_classes": n_classes,
            "class_labels": unique_labels.tolist(),
        }

        # Compute mean response per class for each variable
        E_class_means = np.zeros((n_classes, n_units_E))
        I_class_means = np.zeros((n_classes, n_units_I))
        Vout_class_means = np.zeros((n_classes, n_units_Vout))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            E_class_means[i] = np.mean(E[mask], axis=0)
            I_class_means[i] = np.mean(Inh[mask], axis=0)
            Vout_class_means[i] = np.mean(Vout[mask], axis=0)

        # Store per-class means
        results["E_class_means"] = E_class_means
        results["I_class_means"] = I_class_means
        results["Vout_class_means"] = Vout_class_means

        # Compute selectivity metrics
        results["E_selectivity"] = self._compute_selectivity_metrics(E_class_means, "E")
        results["I_selectivity"] = self._compute_selectivity_metrics(I_class_means, "I")
        results["Vout_selectivity"] = self._compute_selectivity_metrics(
            Vout_class_means, "Vout"
        )

        # For binary classification, also compute traditional tuning (difference)
        if n_classes == 2:
            results["E_tuning"] = (
                E_class_means[1] - E_class_means[0]
            )  # Class 1 - Class 0
            results["I_tuning"] = I_class_means[1] - I_class_means[0]
            results["Vout_tuning"] = Vout_class_means[1] - Vout_class_means[0]

        return results

    def _compute_selectivity_metrics(
        self, class_means: np.ndarray, var_name: str
    ) -> dict[str, np.ndarray]:
        """Compute selectivity metrics for multi-class scenarios.

        Args:
            class_means: Array of shape (n_classes, n_units) with mean response per class
            var_name: Variable name for logging

        Returns:
            Dictionary with selectivity metrics
        """
        n_classes, n_units = class_means.shape

        # Find preferred class for each unit (class with highest mean response)
        preferred_class = np.argmax(class_means, axis=0)

        # Compute selectivity index for each unit
        selectivity_index = np.zeros(n_units)
        class_selectivity = np.zeros(n_units)  # How selective is each unit

        for unit in range(n_units):
            # Sort responses for this unit
            sorted_responses = np.sort(class_means[:, unit])[::-1]  # Descending order

            if n_classes > 1 and sorted_responses[0] > 0:
                # Selectivity index: (best - second_best) / best
                # Normalized to [0, 1] where 1 means perfectly selective
                selectivity_index[unit] = (
                    sorted_responses[0] - sorted_responses[1]
                ) / sorted_responses[0]

                # Class selectivity: (best - mean_others) / (best + mean_others)
                # Alternative metric that considers all non-preferred classes
                best = sorted_responses[0]
                mean_others = np.mean(sorted_responses[1:])
                if best + mean_others > 0:
                    class_selectivity[unit] = (best - mean_others) / (
                        best + mean_others
                    )

        # Compute sparseness (how many classes each unit responds to)
        # Using lifetime sparseness metric
        sparseness = np.zeros(n_units)
        for unit in range(n_units):
            responses = class_means[:, unit]
            if np.sum(responses) > 0:
                # Lifetime sparseness: 1 - (sum(r)^2 / n) / sum(r^2)
                sum_r = np.sum(responses)
                sum_r2 = np.sum(responses**2)
                if sum_r2 > 0:
                    sparseness[unit] = 1 - (sum_r**2 / n_classes) / sum_r2

        return {
            "preferred_class": preferred_class,
            "selectivity_index": selectivity_index,
            "class_selectivity": class_selectivity,
            "sparseness": sparseness,
            "mean_selectivity": np.mean(selectivity_index),
            "mean_sparseness": np.mean(sparseness),
        }

    def _compute_summary_statistics(
        self, correlations: dict[str, Any]
    ) -> dict[str, float]:
        """Compute summary statistics from correlation results."""
        stats = {}

        for corr_type, corr_data in correlations.items():
            if not isinstance(corr_data, dict):
                continue

            # For synaptic level with sampling
            if "mean" in corr_data:
                for metric in ["total", "noise", "signal"]:
                    if metric in corr_data and isinstance(corr_data[metric], dict):
                        stats[f"{corr_type}_{metric}_mean"] = corr_data[metric].get(
                            "mean", 0
                        )
                        stats[f"{corr_type}_{metric}_std"] = corr_data[metric].get(
                            "std", 0
                        )
            else:
                # For branch/layer level with full matrices
                for cond in ["total", "noise", "signal"]:
                    if cond in corr_data:
                        matrix = corr_data[cond]
                        if corr_type in ["EE", "II", "Vout_Vout"]:
                            # Self-correlations: exclude diagonal
                            upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
                            stats[f"{corr_type}_{cond}_mean"] = np.mean(
                                np.abs(upper_tri)
                            )
                            stats[f"{corr_type}_{cond}_std"] = np.std(upper_tri)
                        else:
                            # Cross-correlations: use all values
                            stats[f"{corr_type}_{cond}_mean"] = np.mean(np.abs(matrix))
                            stats[f"{corr_type}_{cond}_std"] = np.std(matrix)

        return stats

    def _compute_per_neuron_correlation_summary(
        self, correlations: dict[str, Any]
    ) -> dict[str, Any]:
        r"""Compute per-unit correlation strength summaries.

        This is a lightweight "per-neuron" view: instead of returning full
        correlation matrices only, we also compute a per-unit summary such as
        the mean absolute correlation of each unit with others.

        Notes
        -----
        - For self-correlation matrices (EE / II / Vout_Vout), we compute for each
          unit \(i\): mean_j!=i |corr(i, j)|.
        - For cross-correlation matrices (EI / E_Vout / I_Vout), we compute mean
          absolute correlations per row (x-side) and per column (y-side).
        """

        by_type: dict[str, Any] = {}
        corr_types = ["EE", "II", "EI", "E_Vout", "I_Vout", "Vout_Vout"]
        conds = ["total", "noise", "signal"]

        for corr_type in corr_types:
            corr_data = correlations.get(corr_type)
            if not isinstance(corr_data, dict):
                continue

            by_type[corr_type] = {}
            for cond in conds:
                matrix = corr_data.get(cond)
                if not isinstance(matrix, np.ndarray):
                    continue

                abs_mat = np.abs(matrix)
                if corr_type in {"EE", "II", "Vout_Vout"}:
                    n = int(abs_mat.shape[0])
                    if n <= 1:
                        per_unit = np.zeros(n, dtype=float)
                    else:
                        per_unit = (np.sum(abs_mat, axis=1) - np.diag(abs_mat)) / (
                            n - 1
                        )

                    by_type[corr_type][cond] = {
                        "per_unit_mean_abs": per_unit,
                        "mean": float(np.mean(per_unit)) if per_unit.size else 0.0,
                        "std": float(np.std(per_unit)) if per_unit.size else 0.0,
                        "n_units": n,
                    }
                else:
                    n_x, n_y = int(abs_mat.shape[0]), int(abs_mat.shape[1])
                    per_x = np.mean(abs_mat, axis=1) if n_y > 0 else np.zeros(n_x)
                    per_y = np.mean(abs_mat, axis=0) if n_x > 0 else np.zeros(n_y)

                    by_type[corr_type][cond] = {
                        "per_x_mean_abs": per_x,
                        "per_y_mean_abs": per_y,
                        "x_mean": float(np.mean(per_x)) if per_x.size else 0.0,
                        "x_std": float(np.std(per_x)) if per_x.size else 0.0,
                        "y_mean": float(np.mean(per_y)) if per_y.size else 0.0,
                        "y_std": float(np.std(per_y)) if per_y.size else 0.0,
                        "n_x": n_x,
                        "n_y": n_y,
                    }

        return {"by_type": by_type}

    @staticmethod
    def _layer_arrays(layer_data: dict[str, torch.Tensor]) -> tuple[np.ndarray, ...]:
        """Return E, I, and output arrays for one recorded dendritic layer."""
        return (
            layer_data["excitation"].cpu().numpy(),
            layer_data["inhibition"].cpu().numpy(),
            layer_data["output"].cpu().numpy(),
        )

    @staticmethod
    def _flatten_features(values: np.ndarray) -> np.ndarray:
        """Flatten all non-batch dimensions into feature columns."""
        return values.reshape(values.shape[0], -1)

    def _network_activity_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Concatenate flattened E, I, and output activity across recorded layers."""
        all_E, all_I, all_Vout = [], [], []
        for layer_data in self.data_dict.values():
            E, Inh, Vout = self._layer_arrays(layer_data)
            all_E.append(self._flatten_features(E))
            all_I.append(self._flatten_features(Inh))
            all_Vout.append(self._flatten_features(Vout))

        return np.hstack(all_E), np.hstack(all_I), np.hstack(all_Vout)

    def _compute_branch_result(
        self, E: np.ndarray, Inh: np.ndarray, Vout: np.ndarray, labels: np.ndarray
    ) -> dict[str, Any]:
        """Compute branch correlations plus optional tuning and summary metrics."""
        result = self._compute_branch_correlations(E, Inh, Vout, labels)
        if self.compute_tuning_curves:
            result["tuning"] = self._compute_tuning_curves(E, Inh, Vout, labels)
        result["summary_stats"] = self._compute_summary_statistics(result)
        return result

    def _add_per_layer_results(
        self, results: dict[str, Any], labels: np.ndarray
    ) -> None:
        layer_results = {}
        layers = list(self.data_dict.items())
        for layer_name, layer_data in tqdm(
            layers, desc="Layer correlation", leave=False, ncols=100
        ):
            E, Inh, Vout = self._layer_arrays(layer_data)
            layer_results[layer_name] = self._compute_branch_result(
                E, Inh, Vout, labels
            )

        results["per_layer_analysis"] = layer_results

    def _compute_network_result(
        self, labels: np.ndarray
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        network_E, network_I, network_Vout = self._network_activity_arrays()
        self.logger.info(
            f"Network-wide data shapes - E: {network_E.shape}, Inh: {network_I.shape}, Vout: {network_Vout.shape}"
        )
        network_corr = self._compute_branch_result(
            network_E, network_I, network_Vout, labels
        )
        return network_corr, network_E, network_I, network_Vout

    def _add_network_optional_results(
        self,
        results: dict[str, Any],
        network_corr: dict[str, Any],
        network_E: np.ndarray,
        network_I: np.ndarray,
        network_Vout: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        if self.per_neuron_analysis:
            results["per_neuron_analysis"] = (
                self._compute_per_neuron_correlation_summary(network_corr)
            )

        if self.per_einet_analysis:
            results["per_einet_analysis"] = self._compute_per_ei_analysis(
                network_E, network_I, network_Vout, labels
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
        """Run hierarchical correlation analysis."""
        self.logger.info(
            f"Starting correlation analysis at {self.computation_level} level"
        )
        self.logger.info(
            f"Model: {type(model).__name__}, Dataset size: {len(test_dataset)}, Device: {device}"
        )

        try:
            # Validate model
            if not isinstance(model.core_network, ExcitationInhibitionNetwork):
                self.logger.warning(
                    "Model does not have ExcitationInhibitionNetwork. Skipping analysis."
                )
                return

            with analysis_device_context(model, device) as analysis_device:
                items = materialize_dataset(
                    test_dataset,
                    runtime,
                    explicit_max_samples=self.max_samples,
                    device=analysis_device,
                )
                inputs = items[0].to(analysis_device)
                labels_np = items[1].numpy()

                def _forward_pass() -> None:
                    with torch.no_grad():
                        _ = model(inputs)

                run_with_forward_hooks(
                    attach=lambda: self.attach_forward_hooks(model),
                    remove=self.remove_forward_hooks,
                    body=_forward_pass,
                )

            # Initialize results
            results = {
                "method": "hierarchical_correlation_analysis",
                "computation_level": self.computation_level,
                "n_samples": len(inputs),
                "n_layers": len(self.data_dict),
                "per_layer_analysis_enabled": self.per_layer_analysis,
                "per_einet_analysis_enabled": self.per_einet_analysis,
                "per_neuron_analysis_enabled": self.per_neuron_analysis,
                "config": {
                    "per_layer_analysis": self.per_layer_analysis,
                    "per_einet_analysis": self.per_einet_analysis,
                    "per_neuron_analysis": self.per_neuron_analysis,
                },
            }

            # Analyze based on computation level
            self.logger.info(f"Computation level: {self.computation_level}")
            if self.computation_level == "synaptic":
                self.logger.info("Calling _analyze_synaptic_level")
                self._analyze_synaptic_level(results, labels_np)
            elif self.computation_level == "single_branch":
                self.logger.info("Calling _analyze_single_branch_level")
                self._analyze_single_branch_level(results, labels_np)
            elif self.computation_level == "layer_branch":
                self.logger.info("Calling _analyze_layer_branch_level")
                self._analyze_layer_branch_level(results, labels_np)
            else:  # all_branch
                self.logger.info("Calling _analyze_all_branch_level")
                self._analyze_all_branch_level(results, labels_np)

            # Save results
            self.logger.info(f"Save path provided: {save_path}")
            if save_path is not None:
                self.logger.info(f"Creating save_path directory: {save_path}")
                os.makedirs(save_path, exist_ok=True)
                self.logger.info("Directory created, saving results...")
                save_dict(results, save_path, filename)

                # Generate plots
                self._generate_plots(results, save_path)

                # Save summary CSV
                summary = self._create_summary_dataframe(results)
                summary.to_csv(
                    os.path.join(save_path, f"{filename}_summary.csv"), index=False
                )

                self.logger.info(f"Results saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Error during correlation analysis: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _analyze_synaptic_level(self, results: dict[str, Any], labels: np.ndarray):
        """Analyze at synaptic level using sampling."""
        self.logger.info("Running synaptic-level correlation analysis...")

        # Network-level synaptic analysis
        all_results = {}
        layers = list(self.data_dict.items())
        for layer_name, layer_data in tqdm(
            layers, desc="Synaptic analysis", leave=False, ncols=100
        ):
            layer_results = self._compute_synaptic_correlations(layer_data, labels)
            all_results[layer_name] = layer_results

        results["synaptic_analysis"] = all_results

        # Compute aggregated statistics
        results["network_summary"] = self._aggregate_synaptic_stats(all_results)

    def _analyze_single_branch_level(self, results: dict[str, Any], labels: np.ndarray):
        """Analyze each branch individually."""
        self.logger.info("Running single-branch level correlation analysis...")

        layers = list(self.data_dict.items())
        for layer_name, layer_data in tqdm(
            layers, desc="Layers", leave=False, ncols=100
        ):
            E = layer_data["excitation"].cpu().numpy()
            Inh = layer_data["inhibition"].cpu().numpy()
            Vout = layer_data["output"].cpu().numpy()

            # Analyze each branch separately
            n_branches = E.shape[1] if E.ndim > 1 else 1
            branch_results = []

            short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
            for branch_idx in tqdm(
                range(n_branches),
                desc=f"  {short_name}",
                leave=False,
                ncols=100,
                mininterval=0.5,
            ):
                e_branch = E[:, branch_idx : branch_idx + 1] if E.ndim > 1 else E
                i_branch = Inh[:, branch_idx : branch_idx + 1] if Inh.ndim > 1 else Inh
                v_branch = (
                    Vout[:, branch_idx : branch_idx + 1] if Vout.ndim > 1 else Vout
                )

                branch_corr = self._compute_branch_correlations(
                    e_branch, i_branch, v_branch, labels
                )
                if self.compute_tuning_curves:
                    branch_corr["tuning"] = self._compute_tuning_curves(
                        e_branch, i_branch, v_branch, labels
                    )

                branch_results.append(branch_corr)

            results[f"{layer_name}_branches"] = branch_results

    def _analyze_layer_branch_level(self, results: dict[str, Any], labels: np.ndarray):
        """Analyze at layer level (all branches together)."""
        self.logger.info("Running layer-branch level correlation analysis...")

        # Analyze each layer
        if self.per_layer_analysis:
            self._add_per_layer_results(results, labels)

        # Network-wide analysis (aggregate all layers)
        self.logger.info("Computing network-wide correlations...")
        network_corr, network_E, network_I, network_Vout = self._compute_network_result(
            labels
        )
        results["network_level"] = network_corr

        self._add_network_optional_results(
            results,
            network_corr,
            network_E,
            network_I,
            network_Vout,
            labels,
        )

    def _analyze_all_branch_level(self, results: dict[str, Any], labels: np.ndarray):
        """Analyze with full network aggregation."""
        self.logger.info("Running all-branch level correlation analysis...")

        # This is similar to network-level in layer_branch but without per-layer
        network_corr, network_E, network_I, network_Vout = self._compute_network_result(
            labels
        )
        results["network_level"] = network_corr

        # Optional per-layer analysis (still meaningful even in all-branch mode)
        if self.per_layer_analysis:
            self._add_per_layer_results(results, labels)

        self._add_network_optional_results(
            results,
            network_corr,
            network_E,
            network_I,
            network_Vout,
            labels,
        )

    def _compute_per_ei_analysis(
        self, E: np.ndarray, Inh: np.ndarray, Vout: np.ndarray, labels: np.ndarray
    ) -> dict[str, Any]:
        """Separate E and I network analysis."""
        results = {}

        # Excitatory network
        exc_results = {}
        if self.analyze_EE:
            exc_results["EE"] = self._compute_conditional_correlations(E, None, labels)
        if self.analyze_E_output:
            exc_results["E_Vout"] = self._compute_conditional_correlations(
                E, Vout, labels
            )
        if self.compute_tuning_curves:
            tuning_results = self._compute_tuning_curves(E, Inh, Vout, labels)
            if "E_tuning" in tuning_results:
                exc_results["tuning"] = {"E_tuning": tuning_results["E_tuning"]}
            else:
                # For multi-class, use selectivity metrics
                exc_results["tuning"] = {
                    "E_selectivity": tuning_results["E_selectivity"]
                }

        results["excitatory_results"] = {
            "correlations": exc_results,
            "summary_stats": self._compute_summary_statistics(exc_results),
        }

        # Inhibitory network
        inh_results = {}
        if self.analyze_II:
            inh_results["II"] = self._compute_conditional_correlations(
                Inh, None, labels
            )
        if self.analyze_I_output:
            inh_results["I_Vout"] = self._compute_conditional_correlations(
                Inh, Vout, labels
            )
        if self.compute_tuning_curves:
            tuning_results = self._compute_tuning_curves(E, Inh, Vout, labels)
            if "I_tuning" in tuning_results:
                inh_results["tuning"] = {"I_tuning": tuning_results["I_tuning"]}
            else:
                # For multi-class, use selectivity metrics
                inh_results["tuning"] = {
                    "I_selectivity": tuning_results["I_selectivity"]
                }

        results["inhibitory_results"] = {
            "correlations": inh_results,
            "summary_stats": self._compute_summary_statistics(inh_results),
        }

        # Combined E-I
        if self.analyze_EI:
            results["combined_results"] = {
                "EI": self._compute_conditional_correlations(E, Inh, labels)
            }

        return results

    def _aggregate_synaptic_stats(
        self, synaptic_results: dict[str, Any]
    ) -> dict[str, float]:
        """Aggregate statistics across layers for synaptic analysis."""
        all_stats = {}

        for _layer_name, layer_results in synaptic_results.items():
            for corr_type, corr_data in layer_results.items():
                if isinstance(corr_data, dict) and "total" in corr_data:
                    for metric in ["total", "noise", "signal"]:
                        if metric in corr_data and "mean" in corr_data[metric]:
                            key = f"{corr_type}_{metric}_mean"
                            if key not in all_stats:
                                all_stats[key] = []
                            all_stats[key].append(corr_data[metric]["mean"])

        # Average across layers
        summary = {}
        for key, values in all_stats.items():
            summary[f"avg_{key}"] = np.mean(values)
            summary[f"std_{key}"] = np.std(values)

        return summary

    def _generate_plots(self, results: dict[str, Any], save_path: str):
        """Generate correlation plots."""
        try:
            self.logger.info(f"Generating correlation plots, save_path: {save_path}")
            self.logger.info(f"Results keys: {list(results.keys())}")

            from dendritic_modeling.plotting.visualizations.correlation_plots import (
                plot_correlation_comparisons,
                plot_correlation_matrices,
                plot_hierarchical_correlations,
                plot_tuning_curves,
            )

            # Main hierarchical plot
            self.logger.info("Calling plot_hierarchical_correlations...")
            plot_hierarchical_correlations(results, save_path=save_path)

            # Generate dedicated comparison plots
            self.logger.info("Calling plot_correlation_comparisons...")
            plot_correlation_comparisons(results, save_path=save_path)

            # Additional plots based on computation level
            if "network_level" in results:
                # Correlation matrices
                if any(k in results["network_level"] for k in ["EE", "II", "EI"]):
                    plot_correlation_matrices(
                        results["network_level"], save_path=save_path
                    )

                # Tuning curves
                if "tuning" in results["network_level"]:
                    plot_tuning_curves(
                        results["network_level"]["tuning"], save_path=save_path
                    )

        except ImportError as e:
            self.logger.warning(f"Could not import plotting functions: {e}")
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")

    def _create_summary_dataframe(self, results: dict[str, Any]) -> pd.DataFrame:
        """Create summary dataframe."""
        summary_data = {
            "method": results["method"],
            "computation_level": results["computation_level"],
            "n_samples": results["n_samples"],
            "n_layers": results["n_layers"],
        }

        # Add statistics based on computation level
        if "network_summary" in results:
            # Synaptic level
            for key, value in results["network_summary"].items():
                summary_data[key] = value
        elif "network_level" in results and "summary_stats" in results["network_level"]:
            # Branch/layer level
            for key, value in results["network_level"]["summary_stats"].items():
                summary_data[f"network_{key}"] = value

        return pd.DataFrame([summary_data])

    def analyze_hierarchical(self, *args, **kwargs):
        """Alias for backward compatibility."""
        return self.analyze(*args, **kwargs)
