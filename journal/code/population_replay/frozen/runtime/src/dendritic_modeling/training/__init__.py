"""
Training package for dendritic models.

This package provides various training strategies, optimizers, and utilities.
"""

from dendritic_modeling.networks import GradientScaler
from dendritic_modeling.training.factory import (
    extract_weights_from_simple_mlp,
    get_available_trainer_strategies,
    get_trainer,
    register_trainer_strategy,
    transfer_mlp_weights_to_input_net,
)
from dendritic_modeling.training.optimizers import CustomWeightDecayOptimizer
from dendritic_modeling.training.strategies import Trainer
from dendritic_modeling.training.transformer_replacement import (
    TransformerReplacementBenchmarkResult,
    TransformerReplacementTrainingResult,
    benchmark_saved_transformer_replacement,
    run_transformer_replacement_training,
)
from dendritic_modeling.training.vision_replacement import (
    VisionReplacementTrainingResult,
    load_vision_replacement_core_checkpoint,
    run_vision_replacement_training,
)

__all__ = [
    "CustomWeightDecayOptimizer",
    "GradientScaler",
    "Trainer",
    "TransformerReplacementBenchmarkResult",
    "TransformerReplacementTrainingResult",
    "VisionReplacementTrainingResult",
    "benchmark_saved_transformer_replacement",
    "extract_weights_from_simple_mlp",
    "get_available_trainer_strategies",
    "get_trainer",
    "load_vision_replacement_core_checkpoint",
    "register_trainer_strategy",
    "run_transformer_replacement_training",
    "run_vision_replacement_training",
    "transfer_mlp_weights_to_input_net",
]
