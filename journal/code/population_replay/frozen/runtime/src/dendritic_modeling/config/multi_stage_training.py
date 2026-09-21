"""
Multi-stage training configuration.

This module defines configuration structures for training networks
in multiple sequential stages with different learning strategies.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from dendritic_modeling.config.base import BaseConfig


@dataclass
class TrainingStageConfig(BaseConfig):
    """Configuration for a single training stage."""

    # Learning strategy for this stage (REQUIRED)
    learning_strategy: str = "standard"

    # Optional stage name for logging/identification (auto-generated if not provided)
    stage_name: Optional[str] = None

    # Number of epochs for this stage (can be overridden)
    epochs: Optional[int] = None

    # Trainer-specific configurations for this stage
    trainer_config: dict[str, Any] = field(default_factory=dict)

    # Optional: override batch size for this stage
    batch_size: Optional[int] = None

    # Optional: override param_groups for this stage
    param_groups: Optional[dict[str, Any]] = None

    # Optional: whether to reset optimizer state before this stage
    reset_optimizer: bool = False

    # Optional: whether to reset model weights before this stage (for ablation studies)
    reset_model: bool = False

    # Optional: checkpoint to load before starting this stage
    load_checkpoint: Optional[str] = None

    # Optional: whether to save checkpoint after this stage
    save_checkpoint: bool = True

    # Local learning rule specific config (if using local_ca strategy)
    local_rule_config: Optional[dict[str, Any]] = None


@dataclass
class MultiStageTrainingConfig(BaseConfig):
    """Configuration for multi-stage training."""

    # List of training stages to execute sequentially
    stages: list[TrainingStageConfig] = field(default_factory=list)

    # Whether to continue to next stage if current stage fails
    continue_on_failure: bool = False

    # Whether to run analysis after each stage
    analyze_between_stages: bool = True

    # Global settings that apply to all stages unless overridden
    global_batch_size: Optional[int] = None
    global_shuffle: bool = True
    global_grad_clip_value: float = 5.0

    # Whether to use the best checkpoint from each stage
    use_best_from_each_stage: bool = True
