"""
Training optimizer implementations.

This module contains custom optimizer wrappers and implementations
for dendritic network training.
"""

from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer
from dendritic_modeling.training.optimizers.factory import (
    create_optimizer,
    get_available_optimizers,
    register_optimizer,
    unregister_optimizer,
    validate_optimizer_config,
)

# Importing registers "muonh" in the optimizer registry (opt-in via
# training.main.optimizer.name; the default AdamW path is untouched).
from dendritic_modeling.training.optimizers.muonh import MuonH

__all__ = [
    "CustomWeightDecayOptimizer",
    "MuonH",
    "create_optimizer",
    "get_available_optimizers",
    "register_optimizer",
    "unregister_optimizer",
    "validate_optimizer_config",
]
