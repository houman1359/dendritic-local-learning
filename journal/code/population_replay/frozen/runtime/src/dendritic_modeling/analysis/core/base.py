"""
Abstract base class for analysis components.

This module provides a common interface for all analyzer classes,
enabling consistent behavior and easy extensibility.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import torch

from dendritic_modeling.models.base import BaseModel


class AbstractAnalyzer(ABC):
    """
    Abstract base class for all analyzer components.

    This class provides common functionality for saving results,
    logging, and standardized interfaces for analysis.
    """

    def __init__(self, logger_name: Optional[str] = None):
        """Initialize the analyzer with optional custom logger name."""
        self.logger = logging.getLogger(logger_name or self.__class__.__name__)

    @abstractmethod
    def analyze(
        self,
        model: BaseModel,
        data: torch.utils.data.Dataset,
        device: str = "cpu",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Perform analysis on the given model and data.

        Args:
            model: The model to analyze
            data: Dataset to use for analysis
            device: Device to run analysis on
            **kwargs: Additional analysis-specific parameters

        Returns:
            Dictionary containing analysis results
        """

    def save_results(self, results: dict[str, Any], path: str) -> None:
        """
        Save analysis results to a JSON file.

        Args:
            results: Dictionary containing analysis results
            path: Path to save the results file
        """
        try:
            # Convert path to Path object for easier handling
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert any torch tensors to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)

            with open(save_path, "w") as f:
                json.dump(serializable_results, f, indent=4)

            self.logger.info(f"Results saved to {save_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results to {path}: {e}")
            raise

    def _make_json_serializable(self, obj: Any) -> Any:
        """
        Convert objects to JSON-serializable format.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, "__dict__"):
            # Handle custom objects by converting to dict
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj

    def log_analysis_start(self, analysis_type: str, **kwargs) -> None:
        """Log the start of an analysis with relevant parameters."""
        self.logger.info(f"Starting {analysis_type} analysis...")
        if kwargs:
            param_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            self.logger.info(f"Analysis parameters: {param_str}")

    def log_analysis_end(
        self, analysis_type: str, num_results: Optional[int] = None
    ) -> None:
        """Log the completion of an analysis."""
        end_msg = f"Completed {analysis_type} analysis"
        if num_results is not None:
            end_msg += f" with {num_results} results"
        self.logger.info(end_msg)


__all__ = ["AbstractAnalyzer"]
