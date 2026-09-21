"""
Synaptic turnover analysis module.

This module contains classes for analyzing synaptic plasticity and turnover during training,
tracking the dynamics of active, stable, and transient synapses over time.
"""

import logging
from typing import Optional

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
    plot_synapse_turnover,
)
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import iter_named_modules_of_type

logger = logging.getLogger(__name__)


def _update_turnover_counts(
    prev_mask: torch.Tensor,
    curr_mask: torch.Tensor,
    total_active: int,
    total_stable: int,
    total_transient: int,
) -> tuple[int, int, int]:
    """Update active/stable/transient counts for two consecutive masks."""
    if prev_mask.dtype != torch.bool:
        prev_mask = prev_mask.bool()
    if curr_mask.dtype != torch.bool:
        curr_mask = curr_mask.bool()

    curr_active = curr_mask.sum().item()
    curr_transient = (curr_mask & ~prev_mask).sum().item()
    curr_stable = curr_active - curr_transient

    total_active += curr_active
    total_stable += curr_stable
    total_transient += curr_transient
    return total_active, total_stable, total_transient


class SynapticTurnoverAnalyzer(AbstractAnalyzer):
    """
    Analyzer for tracking synaptic turnover and plasticity during training.

    Collects snapshots of synaptic weight masks over time and computes turnover
    statistics including active, stable, and transient synapses for both
    excitatory and inhibitory connections.
    """

    def __init__(self, model: BaseModel, params):
        """
        Initialize the synaptic turnover analyzer.

        Args:
            model: The model to analyze (must have ExcitationInhibitionNetwork)
            params: Parameters object containing snapshot_interval, synaptic_analysis, save_synaptic_report

        This automatically takes an initial snapshot of the model's synaptic masks.
        """
        super().__init__(self.__class__.__name__)
        self.data_dict_list = []
        self.epoch_counter = 0
        self.snapshot_interval = max(1, int(getattr(params, "snapshot_interval", 1)))

        # Store additional params for potential future use
        self.synaptic_analysis = getattr(params, "synaptic_analysis", True)
        self.save_synaptic_report = getattr(params, "save_synaptic_report", True)

        # Default to unsupported, then enable if model has EI core.
        self.enabled = isinstance(
            getattr(model, "core_network", None), ExcitationInhibitionNetwork
        )
        self.synapse_config = {"ee": [], "ei": [], "ie": [], "ii": []}
        if not self.enabled:
            return

        # Store synapse configuration for layer-wise analysis
        self.synapse_config = {
            "ee": getattr(model.core_network, "ee_synapses_per_branch_per_layer", []),
            "ei": getattr(model.core_network, "ei_synapses_per_branch_per_layer", []),
            "ie": getattr(model.core_network, "ie_synapses_per_branch_per_layer", []),
            "ii": getattr(model.core_network, "ii_synapses_per_branch_per_layer", []),
        }

        self.analyze(model=model, training=True)

    def _compute_layer_synapse_averages(
        self, total_exc_active: int, total_inh_active: int
    ) -> dict:
        """
        Compute average synapse counts per layer for each synapse type.

        Args:
            total_exc_active: Total active excitatory synapses
            total_inh_active: Total active inhibitory synapses

        Returns:
            Dictionary with average synapse counts for ee, ei, ie, ii
        """
        # Get configuration
        ee_config = self.synapse_config["ee"]
        ei_config = self.synapse_config["ei"]
        ie_config = self.synapse_config["ie"]
        ii_config = self.synapse_config["ii"]

        # Default to 0 if no configuration available
        num_layers = max(
            len(ee_config), len(ei_config), len(ie_config), len(ii_config), 1
        )

        # Estimate average synapses per layer for each type
        # This is an approximation since we don't have exact layer-wise breakdowns
        # from the mask data. We use the configuration ratios as a guide.
        total_ee_config = sum(ee_config) if ee_config else 0
        total_ei_config = sum(ei_config) if ei_config else 0
        total_ie_config = sum(ie_config) if ie_config else 0
        total_ii_config = sum(ii_config) if ii_config else 0

        total_config = total_ee_config + total_ie_config  # Excitatory connections
        total_inh_config = total_ei_config + total_ii_config  # Inhibitory connections

        # Compute averages per layer based on actual vs configured ratios
        if total_config > 0:
            exc_ratio = total_exc_active / max(total_config, 1)
            ee_avg = (total_ee_config * exc_ratio) / max(num_layers, 1)
            ie_avg = (total_ie_config * exc_ratio) / max(num_layers, 1)
        else:
            ee_avg = ie_avg = 0

        if total_inh_config > 0:
            inh_ratio = total_inh_active / max(total_inh_config, 1)
            ei_avg = (total_ei_config * inh_ratio) / max(num_layers, 1)
            ii_avg = (total_ii_config * inh_ratio) / max(num_layers, 1)
        else:
            ei_avg = ii_avg = 0

        return {"ee": ee_avg, "ei": ei_avg, "ie": ie_avg, "ii": ii_avg}

    def analyze(
        self,
        model: BaseModel,
        training: bool = True,
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """
        Analyze synaptic turnover by taking snapshots or computing statistics.

        Args:
            model: The model to analyze
            training: If True, takes a snapshot of current synaptic masks.
                     If False, computes turnover statistics from collected snapshots.
            save_path: Path to save results (optional)
            filename: Filename for saved results

        Returns:
            Dictionary containing turnover statistics if training=False and save_path is None
        """
        if not self._supports_model(model):
            return

        if training:
            self._record_training_snapshot(model)
            return

        turnover_dict = self._compute_turnover_statistics()
        if save_path is not None:
            save_dict(turnover_dict, save_path, filename)
            self._plot_turnover_results(turnover_dict, save_path, filename)
        else:
            return turnover_dict

    def _supports_model(self, model: BaseModel) -> bool:
        return self.enabled and isinstance(
            getattr(model, "core_network", None), ExcitationInhibitionNetwork
        )

    def _record_training_snapshot(self, model: BaseModel) -> None:
        self.epoch_counter += 1
        if (self.epoch_counter - 1) % self.snapshot_interval != 0:
            return
        self.data_dict_list.append(self._collect_synaptic_mask_snapshot(model))

    def _collect_synaptic_mask_snapshot(self, model: BaseModel) -> dict:
        data_dict = {}
        with torch.no_grad():
            for name, module in iter_named_modules_of_type(model, DendriticBranchLayer):
                data_dict[name] = self._collect_layer_masks(module)
        return data_dict

    @staticmethod
    def _collect_layer_masks(module: DendriticBranchLayer) -> dict:
        layer_masks = {}
        if module.branch_excitation is not None:
            mask = module.branch_excitation.weight_mask()
            layer_masks["exc_mask"] = mask.bool().cpu()
        if module.branch_inhibition is not None:
            mask = module.branch_inhibition.weight_mask()
            layer_masks["inh_mask"] = mask.bool().cpu()
        return layer_masks

    def _compute_turnover_statistics(self) -> dict:
        logger.info(
            f"Analyzing synaptic turnover from {len(self.data_dict_list)} snapshots"
        )
        incremental_turnover = self._empty_turnover_stats()
        cumulative_turnover = self._empty_turnover_stats()
        layer_stats = self._empty_layer_stats()
        initial_data_dict = self.data_dict_list[0] if self.data_dict_list else None

        for i in range(1, len(self.data_dict_list)):
            prev_data_dict: dict = self.data_dict_list[i - 1]
            curr_data_dict: dict = self.data_dict_list[i]
            incremental_counts, cumulative_counts = self._compare_snapshots(
                prev_data_dict=prev_data_dict,
                curr_data_dict=curr_data_dict,
                initial_data_dict=initial_data_dict,
            )
            self._append_turnover_counts(incremental_turnover, incremental_counts)
            self._append_layer_averages(
                layer_stats,
                "incremental",
                incremental_counts["exc"][0],
                incremental_counts["inh"][0],
            )
            self._append_turnover_counts(cumulative_turnover, cumulative_counts)
            self._append_layer_averages(
                layer_stats,
                "cumulative",
                cumulative_counts["exc"][0],
                cumulative_counts["inh"][0],
            )

        return {
            "incremental": incremental_turnover,
            "cumulative": cumulative_turnover,
            "layer_statistics": layer_stats,
            "n_snapshots": len(self.data_dict_list),
        }

    @staticmethod
    def _empty_turnover_stats() -> dict:
        return {
            "exc": {"active": [], "stable": [], "transient": []},
            "inh": {"active": [], "stable": [], "transient": []},
            "both": {"active": [], "stable": [], "transient": []},
        }

    @staticmethod
    def _empty_layer_stats() -> dict:
        return {
            "incremental": {"per_layer_avg": {"ee": [], "ei": [], "ie": [], "ii": []}},
            "cumulative": {"per_layer_avg": {"ee": [], "ei": [], "ie": [], "ii": []}},
        }

    def _compare_snapshots(
        self,
        prev_data_dict: dict,
        curr_data_dict: dict,
        initial_data_dict: Optional[dict],
    ) -> tuple[dict[str, tuple[int, int, int]], dict[str, tuple[int, int, int]]]:
        incremental_counts = {"exc": (0, 0, 0), "inh": (0, 0, 0)}
        cumulative_counts = {"exc": (0, 0, 0), "inh": (0, 0, 0)}

        for name in curr_data_dict.keys():
            prev_module_dict: dict = prev_data_dict[name]
            curr_module_dict: dict = curr_data_dict[name]
            initial_module_dict: dict = (
                initial_data_dict[name] if initial_data_dict else prev_module_dict
            )
            incremental_counts, cumulative_counts = self._compare_layer_masks(
                prev_module_dict=prev_module_dict,
                curr_module_dict=curr_module_dict,
                initial_module_dict=initial_module_dict,
                incremental_counts=incremental_counts,
                cumulative_counts=cumulative_counts,
            )

        return incremental_counts, cumulative_counts

    @staticmethod
    def _compare_layer_masks(
        prev_module_dict: dict,
        curr_module_dict: dict,
        initial_module_dict: dict,
        incremental_counts: dict[str, tuple[int, int, int]],
        cumulative_counts: dict[str, tuple[int, int, int]],
    ) -> tuple[dict[str, tuple[int, int, int]], dict[str, tuple[int, int, int]]]:
        for label, mask_key in (("exc", "exc_mask"), ("inh", "inh_mask")):
            if mask_key not in curr_module_dict.keys():
                continue
            incremental_counts[label] = _update_turnover_counts(
                prev_module_dict[mask_key],
                curr_module_dict[mask_key],
                *incremental_counts[label],
            )
            cumulative_counts[label] = _update_turnover_counts(
                initial_module_dict[mask_key],
                curr_module_dict[mask_key],
                *cumulative_counts[label],
            )
        return incremental_counts, cumulative_counts

    @staticmethod
    def _append_turnover_counts(
        turnover: dict,
        counts: dict[str, tuple[int, int, int]],
    ) -> None:
        exc_active, exc_stable, exc_transient = counts["exc"]
        inh_active, inh_stable, inh_transient = counts["inh"]
        for label, (active, stable, transient) in counts.items():
            turnover[label]["active"].append(active)
            turnover[label]["stable"].append(stable)
            turnover[label]["transient"].append(transient)
        turnover["both"]["active"].append(exc_active + inh_active)
        turnover["both"]["stable"].append(exc_stable + inh_stable)
        turnover["both"]["transient"].append(exc_transient + inh_transient)

    def _append_layer_averages(
        self,
        layer_stats: dict,
        mode: str,
        total_exc_active: int,
        total_inh_active: int,
    ) -> None:
        layer_avg = self._compute_layer_synapse_averages(
            total_exc_active,
            total_inh_active,
        )
        for synapse_type in ("ee", "ei", "ie", "ii"):
            layer_stats[mode]["per_layer_avg"][synapse_type].append(
                layer_avg[synapse_type]
            )

    def _plot_turnover_results(self, results: dict, save_path: str, filename: str):
        """Generate plots for synaptic turnover results."""
        try:
            # Generate turnover plots
            plot_synapse_turnover(results, save_path=save_path)
        except Exception as e:
            logger.error(f"Error generating synaptic turnover plots: {e}")
