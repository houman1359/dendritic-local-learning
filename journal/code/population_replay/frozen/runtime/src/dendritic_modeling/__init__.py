"""
Dendritic Modeling Package
==========================

A comprehensive library for dendritic neural network modeling and analysis.

Package Structure:
- analysis/: Analysis tools for network behavior and information flow
- config/: Configuration management and dataclasses
- datasets/: Synthetic and real dataset implementations
- models/: Model wrappers (classification, regression, baselines)
- networks/: Network architectures, layers, and activations
- plotting/: Visualization and plotting utilities
- scripts/: Main training and experiment scripts
- training/: Training strategies and optimizers
- utils/: General utility functions

For detailed documentation on each module, see the respective module docstrings.
"""

# NOTE:
# -----
# Keep this module *lightweight* so users can import analysis/plotting utilities
# without pulling in optional training/script dependencies (e.g., wandb/torchinfo).

from __future__ import annotations

# Set up the logging environment
import importlib
from typing import Any

from dendritic_modeling.utils.logging_config import LoggerManager

logger_manager = LoggerManager()
logger = logger_manager.get_logger()

# Version information
__version__ = "0.1.0"

_LAZY_SUBMODULES: set[str] = {
    "analysis",
    "config",
    "datasets",
    "models",
    "networks",
    "plotting",
    "scripts",
    "training",
    "utils",
}


def __getattr__(name: str) -> Any:
    """Lazily import top-level subpackages on first access."""
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_SUBMODULES))


__all__ = [
    "__version__",
    "analysis",
    "config",
    "datasets",
    "logger",
    "logger_manager",
    "models",
    "networks",
    "plotting",
    "scripts",
    "training",
    "utils",
]

# from dendritic_modeling import pid_utils
# from dendritic_modeling.information import (
#     build_sr_matrix,
#     compute_ei_mi,
#     compute_mi_cc,
#     compute_mi_cd,
#     compute_pairwise_sr,
#     compute_synaptic_information,
# )

# Import utility functions
# from dendritic_modeling.utils import (
#     # Utility functions would be imported here if needed
# )
