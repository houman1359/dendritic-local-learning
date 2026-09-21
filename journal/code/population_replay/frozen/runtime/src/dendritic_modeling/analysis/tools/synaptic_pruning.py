"""
Synaptic Pruning Analysis Module
===============================

This module extends the synaptic turnover analysis to include pruning-specific
metrics and integrates with the enhanced training system to track the effects
of weight pruning on synaptic connectivity.
"""

import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from dendritic_modeling.analysis.tools.synaptic_turnover import SynapticTurnoverAnalyzer
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.plotting.visualizations import (
    plot_active_synapses_summary_by_layer,
)
from dendritic_modeling.utils.hooks import iter_modules_of_type

logger = logging.getLogger(__name__)


def _format_pruning_event_row(event: dict[str, Any]) -> str:
    epoch = event["epoch"]
    pruned = event["pruning_stats"].get("pruned_params", 0)
    ratio = event["pruning_stats"].get("pruning_ratio", 0)
    sparsity_inc = event.get("connectivity_changes", {}).get("sparsity_increase", 0)
    return f"{epoch:<8} {pruned:<12,} {ratio:<8.4f} {sparsity_inc:<12.4f}"


def _format_pruning_summary(
    pruning_events: list[dict[str, Any]],
    initial_connectivity_stats: dict[str, Any],
    data_dict_list: list[dict[str, Any]],
) -> list[str]:
    if not pruning_events:
        return ["No pruning events recorded."]

    total_events = len(pruning_events)
    total_pruned = sum(
        event["pruning_stats"].get("pruned_params", 0) for event in pruning_events
    )
    avg_pruning_ratio = np.mean(
        [event["pruning_stats"].get("pruning_ratio", 0) for event in pruning_events]
    )

    lines = [
        "\n" + "=" * 60,
        "SYNAPTIC PRUNING ANALYSIS SUMMARY",
        "=" * 60,
        f"Total pruning events: {total_events}",
        f"Total synapses pruned: {total_pruned:,}",
        f"Average pruning ratio per event: {avg_pruning_ratio:.4f}",
    ]

    if initial_connectivity_stats and data_dict_list:
        initial_active = initial_connectivity_stats["active_synapses"]
        initial_total = initial_connectivity_stats["total_synapses"]
        initial_sparsity = initial_connectivity_stats["overall_sparsity"]
        lines.extend(
            [
                "\nInitial connectivity:",
                f"  Active synapses: {initial_active:,} / {initial_total:,}",
                f"  Initial sparsity: {initial_sparsity:.4f}",
            ]
        )

    lines.extend(
        [
            "\nPruning events:",
            f"{'Epoch':<8} {'Pruned':<12} {'Ratio':<8} {'Sparsity Inc':<12}",
            "-" * 45,
        ]
    )
    lines.extend(_format_pruning_event_row(event) for event in pruning_events)
    lines.append("=" * 60)
    return lines


def _format_pruning_summary_text(
    pruning_events: list[dict[str, Any]],
    initial_connectivity_stats: dict[str, Any],
    data_dict_list: list[dict[str, Any]],
) -> str:
    """Return the full synaptic pruning summary as display-ready text."""
    return "\n".join(
        _format_pruning_summary(
            pruning_events,
            initial_connectivity_stats,
            data_dict_list,
        )
    )


def _iter_dendritic_branch_layers(model: BaseModel) -> Iterator[DendriticBranchLayer]:
    core_network = getattr(model, "core_network", None)
    if not isinstance(core_network, ExcitationInhibitionNetwork):
        return

    yield from iter_modules_of_type(core_network, DendriticBranchLayer)


def _synapse_weight_mask(synapse: Any):
    if synapse is None or not hasattr(synapse, "weight_mask"):
        return None
    return synapse.weight_mask()


def _mask_rows(mask: Any):
    if mask is None:
        return None
    if mask.ndim == 0:
        return mask.reshape(1, 1)
    if mask.ndim == 1:
        return mask.reshape(1, -1)
    return mask.reshape(mask.shape[0], -1)


def _active_mask_row_stats(mask_rows: Any, branch_idx: int) -> dict[str, int | float]:
    if mask_rows is None or branch_idx >= mask_rows.shape[0]:
        return {"total": 0, "active": 0}

    row = mask_rows[branch_idx]
    return {"total": row.numel(), "active": row.sum().item()}


class SynapticPruningAnalyzer(SynapticTurnoverAnalyzer):
    """
    Enhanced synaptic turnover analyzer that includes pruning-specific metrics.

    This class extends the base SynapticTurnoverAnalyzer to track:
    - Active synapses before and after pruning
    - Pruning-induced changes in connectivity
    - Sparsity patterns across layers and branches
    - Recovery patterns after pruning
    """

    def __init__(self, model: BaseModel, snapshot_interval: int = 1):
        """
        Initialize the synaptic pruning analyzer.

        Args:
            model: The model to analyze (must have ExcitationInhibitionNetwork)
            snapshot_interval: Collect snapshots every N epochs
        """
        snapshot_interval = max(1, int(snapshot_interval))
        params = SimpleNamespace(
            snapshot_interval=snapshot_interval,
            synaptic_analysis=True,
            save_synaptic_report=True,
        )
        super().__init__(model, params)

        # Track pruning events
        self.pruning_events = []
        self.pre_pruning_snapshots = {}
        self.post_pruning_snapshots = {}

        # Store initial connectivity statistics
        self.initial_connectivity_stats = self._compute_connectivity_stats(model)

    def record_pruning_event(
        self,
        model: BaseModel,
        epoch: int,
        pruning_stats: dict[str, Any],
        pre_pruning: bool = True,
    ):
        """
        Record a pruning event with before/after synaptic statistics.

        Args:
            model: The model being pruned
            epoch: Current training epoch
            pruning_stats: Statistics from the pruning operation
            pre_pruning: Whether this is before (True) or after (False) pruning
        """
        if pre_pruning:
            # Take snapshot before pruning
            self.pre_pruning_snapshots[epoch] = {
                "connectivity_stats": self._compute_connectivity_stats(model),
                "synaptic_snapshot": self._extract_synaptic_masks(model),
                "epoch": epoch,
            }
        else:
            # Take snapshot after pruning and record the event
            post_stats = self._compute_connectivity_stats(model)
            post_snapshot = self._extract_synaptic_masks(model)

            self.post_pruning_snapshots[epoch] = {
                "connectivity_stats": post_stats,
                "synaptic_snapshot": post_snapshot,
                "epoch": epoch,
            }

            # Calculate pruning-induced changes
            if epoch in self.pre_pruning_snapshots:
                pre_stats = self.pre_pruning_snapshots[epoch]["connectivity_stats"]
                pruning_changes = self._compute_pruning_changes(pre_stats, post_stats)
            else:
                pruning_changes = {}

            # Record the pruning event
            pruning_event = {
                "epoch": epoch,
                "pruning_stats": pruning_stats,
                "post_pruning_stats": post_stats,  # Add this for plotting functions
                "connectivity_changes": pruning_changes,
                "sparsity_analysis": self._analyze_sparsity_patterns(model),
            }

            self.pruning_events.append(pruning_event)

            logger.info(f"Recorded pruning event at epoch {epoch}")
            logger.info(
                f"  Total synapses pruned: {pruning_stats.get('pruned_params', 0):,}"
            )
            logger.info(f"  Pruning ratio: {pruning_stats.get('pruning_ratio', 0):.4f}")

    def _compute_connectivity_stats(self, model: BaseModel) -> dict[str, Any]:
        """Compute detailed connectivity statistics for the model."""
        stats = {
            "total_synapses": 0,
            "active_synapses": 0,
            "layer_stats": {},
            "branch_stats": {},
            "connection_type_stats": {
                "excitatory": {"total": 0, "active": 0},
                "inhibitory": {"total": 0, "active": 0},
            },
        }

        if not isinstance(
            getattr(model, "core_network", None), ExcitationInhibitionNetwork
        ):
            return stats

        # Analyze each dendritic branch layer
        for layer_idx, layer in enumerate(_iter_dendritic_branch_layers(model)):
            exc_rows = _mask_rows(_synapse_weight_mask(layer.branch_excitation))
            inh_rows = _mask_rows(_synapse_weight_mask(layer.branch_inhibition))
            n_branches = max(
                exc_rows.shape[0] if exc_rows is not None else 0,
                inh_rows.shape[0] if inh_rows is not None else 0,
            )

            layer_stats = {
                "total_synapses": 0,
                "active_synapses": 0,
                "branch_stats": {},
            }

            # Process each branch in the layer
            for branch_idx in range(n_branches):
                branch_name = f"layer_{layer_idx}_branch_{branch_idx}"

                branch_stats = {
                    "excitatory": _active_mask_row_stats(exc_rows, branch_idx),
                    "inhibitory": _active_mask_row_stats(inh_rows, branch_idx),
                }

                stats["connection_type_stats"]["excitatory"]["total"] += branch_stats[
                    "excitatory"
                ]["total"]
                stats["connection_type_stats"]["excitatory"]["active"] += branch_stats[
                    "excitatory"
                ]["active"]
                stats["connection_type_stats"]["inhibitory"]["total"] += branch_stats[
                    "inhibitory"
                ]["total"]
                stats["connection_type_stats"]["inhibitory"]["active"] += branch_stats[
                    "inhibitory"
                ]["active"]

                # Aggregate branch stats
                branch_total = (
                    branch_stats["excitatory"]["total"]
                    + branch_stats["inhibitory"]["total"]
                )
                branch_active = (
                    branch_stats["excitatory"]["active"]
                    + branch_stats["inhibitory"]["active"]
                )

                layer_stats["total_synapses"] += branch_total
                layer_stats["active_synapses"] += branch_active
                layer_stats["branch_stats"][branch_name] = branch_stats

                stats["branch_stats"][branch_name] = {
                    "excitatory": branch_stats["excitatory"],
                    "inhibitory": branch_stats["inhibitory"],
                    "total": branch_total,
                    "active": branch_active,
                    "sparsity": 1.0
                    - (branch_active / branch_total if branch_total > 0 else 0),
                }

            stats["layer_stats"][f"layer_{layer_idx}"] = layer_stats
            stats["total_synapses"] += layer_stats["total_synapses"]
            stats["active_synapses"] += layer_stats["active_synapses"]

        # Calculate overall sparsity
        stats["overall_sparsity"] = 1.0 - (
            stats["active_synapses"] / stats["total_synapses"]
            if stats["total_synapses"] > 0
            else 0
        )

        return stats

    def _extract_synaptic_masks(self, model: BaseModel) -> dict[str, Any]:
        """Extract synaptic masks for detailed analysis."""
        masks = {}

        for layer_idx, layer in enumerate(_iter_dendritic_branch_layers(model)):
            exc_rows = _mask_rows(_synapse_weight_mask(layer.branch_excitation))
            inh_rows = _mask_rows(_synapse_weight_mask(layer.branch_inhibition))
            n_branches = max(
                exc_rows.shape[0] if exc_rows is not None else 0,
                inh_rows.shape[0] if inh_rows is not None else 0,
            )

            for branch_idx in range(n_branches):
                branch_name = f"layer_{layer_idx}_branch_{branch_idx}"

                branch_masks = {}

                if exc_rows is not None and branch_idx < exc_rows.shape[0]:
                    branch_masks["exc_mask"] = exc_rows[branch_idx].clone()

                if inh_rows is not None and branch_idx < inh_rows.shape[0]:
                    branch_masks["inh_mask"] = inh_rows[branch_idx].clone()

                if branch_masks:
                    masks[branch_name] = branch_masks

        return masks

    def _compute_pruning_changes(
        self, pre_stats: dict, post_stats: dict
    ) -> dict[str, Any]:
        """Compute changes in connectivity due to pruning."""
        changes = {
            "total_synapses_lost": pre_stats["active_synapses"]
            - post_stats["active_synapses"],
            "sparsity_increase": post_stats["overall_sparsity"]
            - pre_stats["overall_sparsity"],
            "layer_changes": {},
            "branch_changes": {},
            "connection_type_changes": {},
        }

        # Layer-wise changes
        for layer_name in pre_stats["layer_stats"]:
            if layer_name in post_stats["layer_stats"]:
                pre_layer = pre_stats["layer_stats"][layer_name]
                post_layer = post_stats["layer_stats"][layer_name]

                changes["layer_changes"][layer_name] = {
                    "synapses_lost": pre_layer["active_synapses"]
                    - post_layer["active_synapses"],
                    "sparsity_increase": (
                        (
                            1
                            - post_layer["active_synapses"]
                            / post_layer["total_synapses"]
                        )
                        - (
                            1
                            - pre_layer["active_synapses"] / pre_layer["total_synapses"]
                        )
                        if post_layer["total_synapses"] > 0
                        else 0
                    ),
                }

        # Branch-wise changes
        for branch_name in pre_stats["branch_stats"]:
            if branch_name in post_stats["branch_stats"]:
                pre_branch = pre_stats["branch_stats"][branch_name]
                post_branch = post_stats["branch_stats"][branch_name]

                changes["branch_changes"][branch_name] = {
                    "synapses_lost": pre_branch["active"] - post_branch["active"],
                    "sparsity_increase": post_branch["sparsity"]
                    - pre_branch["sparsity"],
                }

        # Connection type changes
        for conn_type in ["excitatory", "inhibitory"]:
            if (
                conn_type in pre_stats["connection_type_stats"]
                and conn_type in post_stats["connection_type_stats"]
            ):
                pre_conn = pre_stats["connection_type_stats"][conn_type]
                post_conn = post_stats["connection_type_stats"][conn_type]

                changes["connection_type_changes"][conn_type] = {
                    "synapses_lost": pre_conn["active"] - post_conn["active"],
                    "pruning_ratio": (
                        (pre_conn["active"] - post_conn["active"]) / pre_conn["active"]
                        if pre_conn["active"] > 0
                        else 0
                    ),
                }

        return changes

    def _analyze_sparsity_patterns(self, model: BaseModel) -> dict[str, Any]:
        """Analyze sparsity patterns across the network."""
        patterns = {
            "layer_sparsities": [],
            "branch_sparsities": [],
            "sparsity_distribution": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
        }

        sparsities = []

        for _layer_idx, layer in enumerate(_iter_dendritic_branch_layers(model)):
            exc_rows = _mask_rows(_synapse_weight_mask(layer.branch_excitation))
            inh_rows = _mask_rows(_synapse_weight_mask(layer.branch_inhibition))
            n_branches = max(
                exc_rows.shape[0] if exc_rows is not None else 0,
                inh_rows.shape[0] if inh_rows is not None else 0,
            )

            layer_sparsities = []

            for branch_idx in range(n_branches):
                # Calculate branch sparsity
                exc_stats = _active_mask_row_stats(exc_rows, branch_idx)
                inh_stats = _active_mask_row_stats(inh_rows, branch_idx)
                total_synapses = exc_stats["total"] + inh_stats["total"]
                active_synapses = exc_stats["active"] + inh_stats["active"]

                branch_sparsity = 1.0 - (
                    active_synapses / total_synapses if total_synapses > 0 else 0
                )
                layer_sparsities.append(branch_sparsity)
                patterns["branch_sparsities"].append(branch_sparsity)
                sparsities.append(branch_sparsity)

            # Calculate layer average sparsity
            layer_avg_sparsity = np.mean(layer_sparsities) if layer_sparsities else 0.0
            patterns["layer_sparsities"].append(layer_avg_sparsity)

        # Calculate distribution statistics
        if sparsities:
            patterns["sparsity_distribution"] = {
                "mean": np.mean(sparsities),
                "std": np.std(sparsities),
                "min": np.min(sparsities),
                "max": np.max(sparsities),
            }

        return patterns

    def generate_pruning_report(self) -> dict[str, Any]:
        """Generate a comprehensive pruning analysis report."""
        report = {
            "summary": {
                "num_pruning_events": len(self.pruning_events),
                "total_snapshots": len(self.data_dict_list),
                "analysis_period": {
                    "start_epoch": (
                        self.data_dict_list[0].get("epoch", 0)
                        if self.data_dict_list
                        else 0
                    ),
                    "end_epoch": (
                        self.data_dict_list[-1].get("epoch", 0)
                        if self.data_dict_list
                        else 0
                    ),
                },
            },
            "initial_connectivity": self.initial_connectivity_stats,
            "pruning_events": self.pruning_events,
            "connectivity_evolution": self._analyze_connectivity_evolution(),
            "pruning_effectiveness": self._analyze_pruning_effectiveness(),
        }

        return report

    def _analyze_connectivity_evolution(self) -> dict[str, Any]:
        """Analyze how connectivity evolves over time with pruning."""
        evolution = {
            "active_synapses_over_time": [],
            "sparsity_over_time": [],
            "pruning_impact_timeline": [],
        }

        # Extract connectivity from all snapshots
        for _snapshot in self.data_dict_list:
            # Connectivity-evolution extraction depends on richer snapshot
            # metadata than some runs save today, so leave the timelines empty
            # when that structure is unavailable.
            pass

        return evolution

    def _analyze_pruning_effectiveness(self) -> dict[str, Any]:
        """Analyze the effectiveness of pruning operations."""
        effectiveness = {
            "average_pruning_ratio": 0.0,
            "synapses_per_pruning_event": [],
            "sparsity_increase_per_event": [],
            "layer_pruning_patterns": {},
            "branch_pruning_patterns": {},
        }

        if not self.pruning_events:
            return effectiveness

        # Calculate average pruning statistics
        pruning_ratios = [
            event["pruning_stats"].get("pruning_ratio", 0)
            for event in self.pruning_events
        ]
        effectiveness["average_pruning_ratio"] = (
            np.mean(pruning_ratios) if pruning_ratios else 0.0
        )

        for event in self.pruning_events:
            pruning_stats = event["pruning_stats"]
            effectiveness["synapses_per_pruning_event"].append(
                pruning_stats.get("pruned_params", 0)
            )

            # Extract sparsity changes if available
            connectivity_changes = event.get("connectivity_changes", {})
            sparsity_increase = connectivity_changes.get("sparsity_increase", 0)
            effectiveness["sparsity_increase_per_event"].append(sparsity_increase)

        return effectiveness

    def print_pruning_summary(self):
        """Print a human-readable summary of pruning analysis."""
        print(
            _format_pruning_summary_text(
                self.pruning_events,
                self.initial_connectivity_stats,
                self.data_dict_list,
            )
        )

    def plot_active_synapses_per_branch(
        self,
        save_path: Optional[str] = None,
        filename: str = "active_synapses_per_branch",
    ):
        """
        Generate plots showing active E and I synapses per branch per layer.

        Args:
            save_path: Directory to save the plots
            filename: Base filename for the plots
        """
        try:
            from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
                plot_active_synapses_per_branch,
            )

            post_stats = self._latest_post_pruning_stats()
            if post_stats is None:
                return None

            # Create the plot
            fig = plot_active_synapses_per_branch(
                post_stats, save_path=save_path, filename=filename
            )

            return fig

        except ImportError as e:
            logger.error(f"Could not import plotting function: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating active synapses plot: {e}")
            return None

    def _latest_post_pruning_stats(self) -> dict[str, Any] | None:
        """Return post-pruning stats from the latest pruning event, if available."""
        if not self.pruning_events:
            logger.warning("No pruning events recorded for plotting")
            return None

        latest_event = self.pruning_events[-1]
        if "post_pruning_stats" not in latest_event:
            logger.error("No post_pruning_stats found in latest pruning event")
            logger.error(f"Available keys: {list(latest_event.keys())}")
            return None

        return latest_event["post_pruning_stats"]

    def plot_active_synapses_summary_by_layer(
        self,
        save_path: Optional[str] = None,
        filename: str = "active_synapses_summary_by_layer",
    ):
        """
        Generate summary plots showing mean and std of active synapses per layer for each synapse type.

        This creates plots showing:
        - X-axis: Layer number (L)
        - Y-axis: Number of active synapses
        - For each layer L: mean(ee_i) and std(ee_i) where i is the branch index in layer L
        - Separate plots for excitatory (EE), inhibitory (IE), and total synapses

        Args:
            save_path: Directory to save the plots
            filename: Base filename for the plots
        """
        try:
            post_stats = self._latest_post_pruning_stats()
            if post_stats is None:
                return None

            # Create the summary plot
            fig = plot_active_synapses_summary_by_layer(
                post_stats, save_path=save_path, filename=filename
            )

            return fig

        except ImportError as e:
            logger.error(f"Could not import plotting function: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating active synapses summary plot: {e}")
            return None


def integrate_pruning_with_turnover_analysis(
    model: BaseModel, pruning_manager, analysis_manager, epoch: int
) -> None:
    """
    Integrate pruning analysis with existing turnover analysis system.

    Args:
        model: The model being trained
        pruning_manager: The pruning manager instance
        analysis_manager: The analysis manager instance
        epoch: Current training epoch
    """
    # Check if synaptic turnover analysis is enabled
    if hasattr(analysis_manager, "synaptic_turnover_analyzer"):
        analyzer = analysis_manager.synaptic_turnover_analyzer

        # If it's not already a pruning analyzer, upgrade it
        if not isinstance(analyzer, SynapticPruningAnalyzer):
            # Create new pruning analyzer with existing data
            new_analyzer = SynapticPruningAnalyzer(model, analyzer.snapshot_interval)
            new_analyzer.data_dict_list = analyzer.data_dict_list
            new_analyzer.epoch_counter = analyzer.epoch_counter

            # Replace the analyzer
            analysis_manager.synaptic_turnover_analyzer = new_analyzer
            analyzer = new_analyzer

        # Record pruning event if pruning stats are available
        if hasattr(pruning_manager, "pruning_stats") and pruning_manager.pruning_stats:
            analyzer.record_pruning_event(
                model=model,
                epoch=epoch,
                pruning_stats=pruning_manager.pruning_stats,
                pre_pruning=False,
            )

            logger.info(
                f"Integrated pruning analysis with turnover analysis at epoch {epoch}"
            )
