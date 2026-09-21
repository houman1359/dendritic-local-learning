"""Base class for sweep analyzers."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import pandas as pd

from ..collectors import DataCollector
from ..utils.aggregation import (
    aggregate_over_seeds,
    compute_ei_ratio,
    extract_network_category,
)

logger = logging.getLogger(__name__)


class BaseSweepAnalyzer(ABC):
    """Abstract base class for sweep-specific analyzers."""

    def __init__(self):
        """Initialize the analyzer."""
        self.sweep_type = self.__class__.__name__.replace("SweepAnalyzer", "").lower()
        self.data_collector = None
        self.raw_data = None
        self.aggregated_data = None

    @abstractmethod
    def can_handle(self, results_dir: Path) -> bool:
        """
        Check if this analyzer can handle the given results directory.

        Args:
            results_dir: Path to results directory

        Returns:
            True if this analyzer can handle the directory
        """

    @abstractmethod
    def get_groupby_columns(self) -> list[str]:
        """
        Get columns to group by for aggregation.

        Returns:
            List of column names for grouping
        """

    @abstractmethod
    def get_plot_types(self) -> list[str]:
        """
        Get plot types specific to this sweep.

        Returns:
            List of plot type names
        """

    def analyze(self, results_dir: Path, output_dir: Optional[Path] = None) -> dict:
        """
        Run complete analysis on sweep results.

        Args:
            results_dir: Path to results directory
            output_dir: Path to output directory (defaults to results_dir/analysis)

        Returns:
            Dictionary with analysis results and plot paths
        """
        logger.info(f"Running {self.sweep_type} sweep analysis...")

        # Set output directory
        if output_dir is None:
            output_dir = results_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Collect data
        logger.info("Collecting data...")
        self.data_collector = DataCollector(results_dir)
        self.raw_data = self.data_collector.collect_all()

        if self.raw_data.empty:
            logger.warning("No data collected")
            return {"status": "no_data", "plot_paths": []}

        logger.info(f"Collected {len(self.raw_data)} configurations")

        # Process data
        logger.info("Processing data...")
        processed_data = self._process_data(self.raw_data)

        # Aggregate over seeds
        logger.info("Aggregating over seeds...")
        group_cols = self.get_groupby_columns()
        self.aggregated_data = aggregate_over_seeds(processed_data, group_cols)

        logger.info(
            f"Aggregated to {len(self.aggregated_data)} unique parameter combinations"
        )

        # Save processed data
        self._save_data(processed_data, self.aggregated_data, output_dir)

        # Generate plots
        logger.info("Generating plots...")
        plot_paths = self._generate_all_plots(self.aggregated_data, output_dir)

        logger.info(f"Analysis complete! Generated {len(plot_paths)} plots")

        return {
            "status": "success",
            "sweep_type": self.sweep_type,
            "n_configs": len(self.raw_data),
            "n_aggregated": len(self.aggregated_data),
            "plot_paths": plot_paths,
            "output_dir": str(output_dir),
        }

    def _process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data (compute derived columns, clean, etc.).

        Args:
            data: Raw data DataFrame

        Returns:
            Processed DataFrame
        """
        processed = data.copy()

        # Add derived columns
        processed = compute_ei_ratio(processed)
        processed = extract_network_category(processed)

        # Call sweep-specific processing
        processed = self.process_sweep_specific(processed)

        return processed

    def process_sweep_specific(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Sweep-specific data processing (override if needed).

        Args:
            data: Processed DataFrame

        Returns:
            Further processed DataFrame
        """
        return data

    def _save_data(
        self,
        processed_data: pd.DataFrame,
        aggregated_data: pd.DataFrame,
        output_dir: Path,
    ):
        """Save processed and aggregated data."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw processed data
        processed_data.to_csv(
            output_dir / f"{self.sweep_type}_processed_data.csv", index=False
        )

        # Save aggregated data
        aggregated_data.to_csv(
            output_dir / f"{self.sweep_type}_aggregated_data.csv", index=False
        )

        logger.info(f"Saved data to {output_dir}")

    @abstractmethod
    def _generate_all_plots(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all plots for this sweep type.

        Args:
            data: Aggregated data DataFrame
            output_dir: Output directory for plots

        Returns:
            List of paths to generated plots
        """
