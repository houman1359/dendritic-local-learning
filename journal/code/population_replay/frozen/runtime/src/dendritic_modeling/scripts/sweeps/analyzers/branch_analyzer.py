"""Branch/architecture sweep analyzer."""

import logging
from pathlib import Path
from typing import ClassVar

import pandas as pd

from ..plotters import (
    AblationPlotter,
    InformationPlotter,
    NoisePlotter,
    PerformancePlotter,
    WeightPlotter,
)
from .base_analyzer import BaseSweepAnalyzer

logger = logging.getLogger(__name__)


class BranchSweepAnalyzer(BaseSweepAnalyzer):
    """Analyzer for branch/architecture sweeps."""

    ARCHITECTURE_PARAMS: ClassVar[list[str]] = [
        "excitatory_layers",
        "inhibitory_layers",
        "branch_factors",
        "nparams",
    ]

    def __init__(self):
        """Initialize branch sweep analyzer."""
        super().__init__()
        self.performance_plotter = PerformancePlotter()
        self.information_plotter = InformationPlotter()
        self.weight_plotter = WeightPlotter()
        self.ablation_plotter = AblationPlotter()
        self.noise_plotter = NoisePlotter()

    def can_handle(self, results_dir: Path) -> bool:
        """
        Detect if this is a branch/architecture sweep.

        Args:
            results_dir: Path to results directory

        Returns:
            True if this analyzer can handle the directory
        """
        from ..collectors import ConfigExtractor

        configs_dir = results_dir / "configs"
        if not configs_dir.exists():
            return False

        # Load a few configs and check for varying architecture parameters
        config_files = sorted(configs_dir.glob("*.yaml"))[:5]

        extractor = ConfigExtractor()
        arch_values = {param: set() for param in self.ARCHITECTURE_PARAMS}

        for config_file in config_files:
            params = extractor.extract_from_file(config_file)
            for param in self.ARCHITECTURE_PARAMS:
                if param in params and params[param] is not None:
                    arch_values[param].add(str(params[param]))

        # This is an architecture sweep if architecture parameters vary
        return any(len(values) > 1 for values in arch_values.values())

    def get_groupby_columns(self) -> list[str]:
        """Get columns to group by for branch sweeps."""
        return [
            "excitatory_layers",
            "inhibitory_layers",
            "branch_factors",
            "use_shunting",
            "reactivation_init_policy",
            "network_category",
            "nparams",
        ]

    def get_plot_types(self) -> list[str]:
        """Get plot types specific to branch sweeps."""
        return [
            "performance_vs_nparams",
            "information_vs_nparams",
            "architecture_comparison",
            "parameter_scaling",
        ]

    def _generate_all_plots(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all branch/architecture-specific plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        # Check if we have parameter count data
        if "nparams" not in data.columns:
            logger.warning(
                "No nparams column - cannot generate parameter scaling plots"
            )

        # Generate standard plots (these already include nparams scaling if available)
        logger.info("Generating performance plots...")
        plot_paths.extend(self.performance_plotter.generate_all(data, output_dir))

        logger.info("Generating information plots...")
        plot_paths.extend(self.information_plotter.generate_all(data, output_dir))

        logger.info("Generating weight plots...")
        plot_paths.extend(self.weight_plotter.generate_all(data, output_dir))

        # Generate ablation and noise plots if data available
        logger.info("Generating ablation plots...")
        plot_paths.extend(self.ablation_plotter.generate_all(data, output_dir))

        logger.info("Generating noise plots...")
        plot_paths.extend(self.noise_plotter.generate_all(data, output_dir))

        return plot_paths
