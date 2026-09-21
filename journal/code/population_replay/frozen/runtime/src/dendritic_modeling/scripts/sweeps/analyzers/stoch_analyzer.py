"""Stochastic dataset sweep analyzer."""

import logging
from pathlib import Path

import pandas as pd

from ..plotters import (
    AblationPlotter,
    InformationPlotter,
    MultiplicativeGainPlotter,
    NoisePlotter,
    PerformancePlotter,
    SNRPlotter,
    WeightPlotter,
)
from .base_analyzer import BaseSweepAnalyzer

logger = logging.getLogger(__name__)


class StochasticDatasetSweepAnalyzer(BaseSweepAnalyzer):
    """Analyzer for stochastic dataset sweeps."""

    def __init__(self):
        """Initialize stochastic dataset sweep analyzer."""
        super().__init__()
        self.performance_plotter = PerformancePlotter()
        self.information_plotter = InformationPlotter()
        self.weight_plotter = WeightPlotter()
        self.ablation_plotter = AblationPlotter()
        self.snr_plotter = SNRPlotter()
        self.noise_plotter = NoisePlotter()
        self.multiplicative_gain_plotter = MultiplicativeGainPlotter()

    def can_handle(self, results_dir: Path) -> bool:
        """
        Detect if this is a stochastic dataset sweep.

        Checks for:
        - Multiple stimulus_duration and max_gain_factor combinations in configs
        - Presence of stochastic-related parameters in sweep configs

        Args:
            results_dir: Path to results directory

        Returns:
            True if this analyzer can handle the directory
        """
        from ..collectors import ConfigExtractor

        configs_dir = results_dir / "configs"
        if not configs_dir.exists():
            return False

        # Load a few config files and check for stimulus_duration and max_gain_factor parameters
        config_files = sorted(configs_dir.glob("*.yaml"))[:5]  # Check first 5

        extractor = ConfigExtractor()
        sd_values = set()
        mgf_values = set()

        for config_file in config_files:
            params = extractor.extract_from_file(config_file)
            if "stimulus_duration" in params:
                sd_values.add(params["stimulus_duration"])
            if "max_gain_factor" in params:
                mgf_values.add(params["max_gain_factor"])

        # This is a stochastic dataset sweep if we have multiple stimulus_duration or max_gain_factor values
        return len(sd_values) > 1 or len(mgf_values) > 1

    def get_groupby_columns(self) -> list[str]:
        """Get columns to group by for stochastic dataset sweeps."""
        return [
            "ee_value",
            "ie_value",
            "ei_ratio",
            "use_shunting",
            "network_type",
            "network_category",
            "nparams",
            "branch_factors",
            "stimulus_duration",
            "fixed_gain_factor",
            "max_gain_factor",
            "max_gain_tau_ratio",
        ]

    def get_plot_types(self) -> list[str]:
        """Get plot types specific to stochastic dataset sweeps."""
        return [
            # "performance_vs_ei_ratio",
            # "performance_heatmaps",
            # "information_vs_ei_ratio",
            # "information_panels",
            # "weight_vs_ei_ratio",
            # "ablation_analysis",
            # "noise_robustness",
            # "shunting_comparison",
        ]

    def _generate_all_plots(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all stochastic dataset-specific plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        # Generate performance plots
        logger.info("Generating performance plots...")
        plot_paths.extend(self.performance_plotter.generate_all(data, output_dir))

        # Generate information plots
        logger.info("Generating information plots...")
        plot_paths.extend(self.information_plotter.generate_all(data, output_dir))

        # Generate weight plots
        # logger.info("Generating weight plots...")
        # plot_paths.extend(self.weight_plotter.generate_all(data, output_dir))

        # Generate ablation plots (if data available)
        logger.info("Generating ablation plots...")
        plot_paths.extend(self.ablation_plotter.generate_all(data, output_dir))

        # Generate SNR plots (if data available)
        logger.info("Generating SNR plots...")
        plot_paths.extend(self.snr_plotter.generate_all(data, output_dir))

        # Generate multiplicative gain plots (if data available)
        logger.info("Generating multiplicative gain plots...")
        plot_paths.extend(
            self.multiplicative_gain_plotter.generate_all(data, output_dir)
        )

        # # Generate noise plots (if data available)
        # logger.info("Generating noise plots...")
        # plot_paths.extend(self.noise_plotter.generate_all(data, output_dir))

        # # Generate E/I-specific plots
        # logger.info("Generating E/I-specific plots...")
        # plot_paths.extend(self._generate_shunting_comparison(data, output_dir))

        return plot_paths
