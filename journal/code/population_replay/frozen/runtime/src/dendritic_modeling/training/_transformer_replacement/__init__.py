"""Implementation modules for transformer replacement training."""

from dendritic_modeling.training._transformer_replacement.common import (
    DistillationUnit,
    SyntheticTransformerMLP,
    TransformerReplacementBenchmarkResult,
    TransformerReplacementTrainingResult,
)
from dendritic_modeling.training._transformer_replacement.evaluation import (
    benchmark_saved_transformer_replacement,
)
from dendritic_modeling.training._transformer_replacement.loops import (
    run_transformer_replacement_training,
)

__all__ = [
    "DistillationUnit",
    "SyntheticTransformerMLP",
    "TransformerReplacementBenchmarkResult",
    "TransformerReplacementTrainingResult",
    "benchmark_saved_transformer_replacement",
    "run_transformer_replacement_training",
]
