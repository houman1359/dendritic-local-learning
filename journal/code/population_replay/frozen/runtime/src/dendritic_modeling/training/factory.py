"""
Training Factory and Utilities

Factory functions for creating training strategies and utility functions
for MLP training and weight transfer.
"""

import logging
from typing import Any, Optional

import torch

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.multi_stage_training import (
    MultiStageTrainingConfig,
    TrainingStageConfig,
)
from dendritic_modeling.config.training import (
    OptimizerConfig,
    ParamGroupsConfig,
    PruningConfig,
    RegularizationConfig,
    ReportingConfig,
)
from dendritic_modeling.training.strategies import (
    FeedbackAlignmentTrainer,
    HomeostaticControlTrainer,
    LayerWiseTrainer,
    LocalCreditAssignment,
    MultiStageTrainer,
    RecurrentTrainer,
    ShuntingFeedbackAlignmentTrainer,
    SomaDFATrainer,
    Trainer,
    TwoStepTrainer,
    TwoStepTrainerWithKL,
    VisionDistillationTrainer,
    VoltageStabilizationTrainer,
)
from dendritic_modeling.utils.hooks import iter_child_modules_of_type

logger = logging.getLogger(__name__)

# These classes are resolved by name through ``globals()`` below, and tests
# patch them through this module path. Keep them materialized in module globals.
_TRAINER_CLASS_GLOBALS = (
    FeedbackAlignmentTrainer,
    HomeostaticControlTrainer,
    LayerWiseTrainer,
    LocalCreditAssignment,
    MultiStageTrainer,
    RecurrentTrainer,
    ShuntingFeedbackAlignmentTrainer,
    SomaDFATrainer,
    Trainer,
    TwoStepTrainer,
    TwoStepTrainerWithKL,
    VisionDistillationTrainer,
    VoltageStabilizationTrainer,
)

TRAINER_STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "standard": {"class": "Trainer"},
    "freeze_layers": {"class": "LayerWiseTrainer"},
    "freeze_branches": {"class": "TwoStepTrainer"},
    "fa": {"class": "FeedbackAlignmentTrainer", "fixed_kwargs": {"mode": "fa"}},
    "dfa": {"class": "FeedbackAlignmentTrainer", "fixed_kwargs": {"mode": "dfa"}},
    "soma_dfa": {"class": "SomaDFATrainer"},
    "local_ca": {"class": "LocalCreditAssignment"},
    "freeze_branch_kl": {"class": "TwoStepTrainerWithKL"},
    "voltage_stabilization": {"class": "VoltageStabilizationTrainer"},
    "homeostatic_control": {"class": "HomeostaticControlTrainer"},
    "vision_distillation": {"class": "VisionDistillationTrainer"},
}

SHUNTING_FEEDBACK_ALIGNMENT_MODES = {
    "shunting_fa": "fa",
    "shunting_dfa": "dfa",
}
SPECIAL_TRAINER_STRATEGIES = [
    *SHUNTING_FEEDBACK_ALIGNMENT_MODES.keys(),
    "recurrent",
    "multi_stage",
]


def register_trainer_strategy(
    name: str,
    trainer_class: type[Trainer] | str,
    *,
    fixed_kwargs: dict[str, Any] | None = None,
) -> None:
    """Register a trainer strategy without editing ``get_trainer``."""
    if not name:
        raise ValueError("trainer strategy name must be non-empty")
    TRAINER_STRATEGY_REGISTRY[str(name).lower()] = {
        "class": trainer_class,
        "fixed_kwargs": dict(fixed_kwargs or {}),
    }


def get_available_trainer_strategies() -> list[str]:
    """Return supported trainer strategy names."""
    return list(TRAINER_STRATEGY_REGISTRY.keys()) + SPECIAL_TRAINER_STRATEGIES


def _resolve_trainer_class(class_ref: type[Trainer] | str) -> type[Trainer]:
    if isinstance(class_ref, str):
        return globals()[class_ref]
    return class_ref


def _build_multi_stage_config(raw_config: Any) -> MultiStageTrainingConfig:
    """Normalize multi-stage config into a typed MultiStageTrainingConfig."""
    if isinstance(raw_config, MultiStageTrainingConfig):
        return raw_config

    if raw_config is None:
        raise ValueError(
            "strategy='multi_stage' requires 'multi_stage_config' in trainer config."
        )

    if not isinstance(raw_config, dict):
        raw_config = to_plain_dict(raw_config)

    if not isinstance(raw_config, dict):
        raise ValueError(
            "multi_stage_config must be a dict or MultiStageTrainingConfig."
        )

    config_dict = dict(raw_config)
    stages_raw = config_dict.get("stages", [])
    if not stages_raw:
        raise ValueError("multi_stage_config must contain at least one stage.")

    stage_configs: list[TrainingStageConfig] = []
    for idx, stage in enumerate(stages_raw):
        if isinstance(stage, TrainingStageConfig):
            stage_configs.append(stage)
            continue
        if not isinstance(stage, dict):
            stage = to_plain_dict(stage)
        if not isinstance(stage, dict):
            raise ValueError(
                f"Stage {idx} in multi_stage_config must be a dict or TrainingStageConfig."
            )
        stage_configs.append(TrainingStageConfig(**stage))

    config_dict["stages"] = stage_configs
    return MultiStageTrainingConfig(**config_dict)


_FLATTENED_STRATEGY_CONFIG_KEYS = (
    "voltage_stabilization_config",
    "homeostatic_control_config",
    "freeze_layers_config",
    "freeze_branches_config",
    "fa_config",
    "freeze_branch_kl_config",
)


def _pop_flat_strategy_config(
    enhanced_configs: dict[str, Any],
    trainer_config_dict: dict[str, Any],
    key: str,
) -> None:
    """Pop a nested strategy config and flatten dict values into trainer kwargs."""
    if key not in trainer_config_dict:
        return
    nested_config = trainer_config_dict.pop(key)
    if isinstance(nested_config, dict):
        enhanced_configs.update(nested_config)


def _extract_enhanced_configs(trainer_config_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Extract enhanced configuration objects from trainer config dict.

    Args:
        trainer_config_dict: Configuration dictionary

    Returns:
        Dictionary containing enhanced config objects
    """
    enhanced_configs = {}

    # Extract optimizer config
    if "optimizer" in trainer_config_dict and isinstance(
        trainer_config_dict["optimizer"], dict
    ):
        enhanced_configs["optimizer_config"] = OptimizerConfig(
            **trainer_config_dict["optimizer"]
        )
        # Don't pass the dict version to avoid conflicts
        trainer_config_dict.pop("optimizer", None)

    # Extract regularization config
    if "regularization" in trainer_config_dict:
        reg_config = trainer_config_dict.pop("regularization")
        if isinstance(reg_config, dict):
            # Handle nested SelectiveConfig properly
            if "selective" in reg_config and isinstance(reg_config["selective"], dict):
                from dendritic_modeling.config.regularization import SelectiveConfig

                reg_config["selective"] = SelectiveConfig(**reg_config["selective"])
            enhanced_configs["regularization_config"] = RegularizationConfig(
                **reg_config
            )
        else:
            enhanced_configs["regularization_config"] = reg_config
    else:
        # Create default regularization config and check for E/I parameters
        reg_config_dict = {}

        # Check for E/I regularization parameters in trainer config
        ei_params = [
            "enforce_ei_weight_ratio",
            "target_ei_weight_ratio",
            "ei_ratio_loss_weight",
            "ei_ratio_scope",
            "ei_ratio_metric",
        ]
        for param in ei_params:
            if param in trainer_config_dict:
                reg_config_dict[param] = trainer_config_dict.pop(param)

        # Create regularization config with parameters if found
        if reg_config_dict:
            enhanced_configs["regularization_config"] = RegularizationConfig(
                **reg_config_dict
            )

    # Extract pruning config
    if "pruning" in trainer_config_dict:
        prune_config = trainer_config_dict.pop("pruning")
        if isinstance(prune_config, dict):
            # Handle nested SelectiveConfig properly
            if "selective" in prune_config and isinstance(
                prune_config["selective"], dict
            ):
                from dendritic_modeling.config.regularization import SelectiveConfig

                prune_config["selective"] = SelectiveConfig(**prune_config["selective"])
            enhanced_configs["pruning_config"] = PruningConfig(**prune_config)
        else:
            enhanced_configs["pruning_config"] = prune_config

    # Extract reporting config
    if "reporting" in trainer_config_dict:
        report_config = trainer_config_dict.pop("reporting")
        if isinstance(report_config, dict):
            enhanced_configs["reporting_config"] = ReportingConfig(**report_config)
        else:
            enhanced_configs["reporting_config"] = report_config

    # Extract param_groups config (preserve original parameter grouping behavior)
    if "param_groups" in trainer_config_dict:
        param_groups_config = trainer_config_dict.pop("param_groups")
        if isinstance(param_groups_config, dict):
            allowed_fields = set(ParamGroupsConfig.__dataclass_fields__.keys())
            sanitized = {
                key: value
                for key, value in param_groups_config.items()
                if key in allowed_fields
            }
            param_groups_config = ParamGroupsConfig(**sanitized)
        enhanced_configs["param_groups_config"] = param_groups_config

    # Extract strategy-specific nested configs and flatten into kwargs
    # Local credit assignment
    if "local_rule_config" in trainer_config_dict:
        enhanced_configs["local_rule_config"] = trainer_config_dict.pop(
            "local_rule_config"
        )

    for key in _FLATTENED_STRATEGY_CONFIG_KEYS:
        _pop_flat_strategy_config(enhanced_configs, trainer_config_dict, key)

    return enhanced_configs


def _apply_enhanced_configs(trainer_config_dict: dict[str, Any]) -> None:
    trainer_config_dict.update(_extract_enhanced_configs(trainer_config_dict))


def _build_shunting_feedback_alignment_trainer(
    mode: str,
    trainer_config_dict: dict[str, Any],
) -> ShuntingFeedbackAlignmentTrainer:
    _apply_enhanced_configs(trainer_config_dict)
    trainer_config_dict["mode"] = mode
    e_rev = trainer_config_dict.pop("e_rev_exc", 1.0)
    return ShuntingFeedbackAlignmentTrainer(**trainer_config_dict, e_rev_exc=e_rev)


def _build_recurrent_trainer(trainer_config_dict: dict[str, Any]) -> RecurrentTrainer:
    """Build a recurrent trainer while preserving legacy kwarg precedence."""
    # Strategy-specific recurrent config takes precedence over generic fields.
    recurrent_cfg = trainer_config_dict.pop("recurrent_config", None)
    if recurrent_cfg is not None and not isinstance(recurrent_cfg, dict):
        recurrent_cfg = to_plain_dict(recurrent_cfg)
    if isinstance(recurrent_cfg, dict):
        # recurrent_cfg should only contain fields the user explicitly set
        # (default-valued fields are stripped by config_utils).  Explicit
        # strategy overrides take precedence over common settings.
        trainer_config_dict.update(recurrent_cfg)

    task = trainer_config_dict.pop("task", "classification")
    grad_clip = trainer_config_dict.pop(
        "grad_clip", trainer_config_dict.pop("grad_clip_value", 1.0)
    )
    use_amp = trainer_config_dict.pop("use_amp", False)
    lr_schedule = trainer_config_dict.pop("lr_schedule", "none")
    lr_warmup_epochs = trainer_config_dict.pop("lr_warmup_epochs", 0)
    _apply_enhanced_configs(trainer_config_dict)
    return RecurrentTrainer(
        task=task,
        grad_clip=grad_clip,
        use_amp=use_amp,
        lr_schedule=lr_schedule,
        lr_warmup_epochs=lr_warmup_epochs,
        **trainer_config_dict,
    )


def _build_multi_stage_trainer(
    trainer_config_dict: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer,
    analysis_manager: Optional[object],
) -> MultiStageTrainer:
    """Build a multi-stage trainer from normalized factory kwargs."""
    raw_multi_stage_config = trainer_config_dict.pop("multi_stage_config", None)
    multi_stage_config = _build_multi_stage_config(raw_multi_stage_config)

    _apply_enhanced_configs(trainer_config_dict)

    base_trainer_config = dict(trainer_config_dict)
    return MultiStageTrainer(
        multi_stage_config=multi_stage_config,
        base_optimizer=optimizer,
        base_trainer_config=base_trainer_config,
        analysis_manager=analysis_manager,
    )


def get_trainer(
    strategy: str,
    optimizer: torch.optim.Optimizer,
    trainer_config_dict: dict[str, Any],
    analysis_manager: Optional[object] = None,
) -> Trainer:
    """
    Factory function to create training strategies.

    Args:
        strategy: String specifying the training strategy
        optimizer: The optimizer to use
        trainer_config_dict: The trainer configuration dict
        analysis_manager: Optional analysis manager

    Returns:
        An instance of the requested training strategy
    """
    strategy = strategy.lower()
    trainer_config_dict["analysis_manager"] = analysis_manager
    trainer_config_dict["optimizer"] = optimizer

    if strategy in TRAINER_STRATEGY_REGISTRY:
        _apply_enhanced_configs(trainer_config_dict)
        spec = TRAINER_STRATEGY_REGISTRY[strategy]
        trainer_config_dict.update(spec.get("fixed_kwargs", {}))
        trainer_cls = _resolve_trainer_class(spec["class"])
        return trainer_cls(**trainer_config_dict)

    if strategy in SHUNTING_FEEDBACK_ALIGNMENT_MODES:
        return _build_shunting_feedback_alignment_trainer(
            SHUNTING_FEEDBACK_ALIGNMENT_MODES[strategy],
            trainer_config_dict,
        )
    elif strategy == "recurrent":
        return _build_recurrent_trainer(trainer_config_dict)
    elif strategy == "multi_stage":
        return _build_multi_stage_trainer(
            trainer_config_dict,
            optimizer=optimizer,
            analysis_manager=analysis_manager,
        )
    else:
        supported = get_available_trainer_strategies()
        raise ValueError(
            f"Unknown training strategy '{strategy}'. Supported strategies: {supported}"
        )


def transfer_mlp_weights_to_input_net(mlp_net, input_net, head_type="excitatory"):
    """
    Transfer weights from a trained MLP to the corresponding part of an InputNetTransform.

    Args:
        mlp_net: The trained MLP network
        input_net: The InputNetTransform to transfer weights to
        head_type: Either "excitatory" or "inhibitory" to specify which head to update
    """
    logger.info(
        f"Transferring weights from MLP to {head_type} head of InputNetTransform"
    )

    # Extract the MLP layers
    if hasattr(mlp_net, "layers"):
        mlp_layers = mlp_net.layers
    elif hasattr(mlp_net, "net") and hasattr(mlp_net.net, "layers"):
        mlp_layers = mlp_net.net.layers
    else:
        logger.warning(
            f"Could not extract layers from MLP model. Using random initialization for {head_type} head."
        )
        return

    # Get the shared backbone and target head of the InputNetTransform
    shared_layers = list(iter_child_modules_of_type(input_net.shared, torch.nn.Linear))

    target_head = (
        input_net.head_exc if head_type == "excitatory" else input_net.head_inh
    )
    head_layers = list(iter_child_modules_of_type(target_head, torch.nn.Linear))

    # Transfer weights for the shared backbone (only once, typically from excitatory head)
    if head_type == "excitatory":
        for i, shared_layer in enumerate(shared_layers):
            if (
                i < len(mlp_layers) - 1
            ):  # All but the last MLP layer go to shared backbone
                mlp_layer = mlp_layers[i]
                if hasattr(mlp_layer, "excit_pre_w"):
                    # MLPExcInhLayer has a different structure
                    shared_layer.weight.data = mlp_layer.excit_pre_w.exp().t()
                    shared_layer.bias.data = mlp_layer.bias.data
                else:
                    logger.warning(
                        f"Unexpected MLP layer structure. Using random initialization for shared layer {i}."
                    )

    # Transfer weights for the head-specific part
    if len(head_layers) > 0 and len(mlp_layers) > 0:
        # The last layer of MLP goes to the head
        last_mlp_layer = mlp_layers[-1]
        if hasattr(last_mlp_layer, "excit_pre_w"):
            head_layers[0].weight.data = last_mlp_layer.excit_pre_w.exp().t()
            head_layers[0].bias.data = last_mlp_layer.bias.data
        else:
            logger.warning(
                f"Unexpected MLP layer structure. Using random initialization for {head_type} head."
            )

    logger.info(f"Weight transfer completed for {head_type} head")


def extract_weights_from_simple_mlp(simple_mlp, input_net_transform):
    """
    Extract weights from a SimpleMLP model and transfer them to an InputNetTransform.

    Args:
        simple_mlp: The trained SimpleMLP model
        input_net_transform: The InputNetTransform to transfer weights to
    """
    logger.info("Transferring weights from SimpleMLP to InputNetTransform")

    # Get network modules
    mlp_linear_modules = list(
        iter_child_modules_of_type(simple_mlp.net.network, torch.nn.Linear)
    )

    # Get shared backbone modules of InputNetTransform
    shared_modules = list(
        iter_child_modules_of_type(input_net_transform.shared, torch.nn.Linear)
    )

    # Get head modules
    exc_head_modules = list(
        iter_child_modules_of_type(input_net_transform.head_exc, torch.nn.Linear)
    )

    inh_head_modules = list(
        iter_child_modules_of_type(input_net_transform.head_inh, torch.nn.Linear)
    )

    # Transfer weights for the shared backbone (from early layers of MLP)
    for i, shared_module in enumerate(shared_modules):
        if i < len(mlp_linear_modules) - 1:  # All but the last MLP layer
            mlp_layer = mlp_linear_modules[i]
            shared_module.weight.data = mlp_layer.weight.data.clone()
            shared_module.bias.data = mlp_layer.bias.data.clone()
            logger.info(f"Transferred weights for shared layer {i}")

    # Transfer weights for heads (from last layer of MLP)
    if len(mlp_linear_modules) > 0 and len(exc_head_modules) > 0:
        # Use the second-to-last layer for the heads (since the last layer is the classifier)
        if len(mlp_linear_modules) >= 2:
            source_layer = mlp_linear_modules[-2]
        else:
            source_layer = mlp_linear_modules[0]

        # Transfer to excitatory head
        exc_head_modules[0].weight.data = source_layer.weight.data[
            : exc_head_modules[0].weight.shape[0], :
        ].clone()
        exc_head_modules[0].bias.data = source_layer.bias.data[
            : exc_head_modules[0].bias.shape[0]
        ].clone()
        logger.info("Transferred weights to excitatory head")

        # Transfer to inhibitory head
        inh_head_modules[0].weight.data = source_layer.weight.data[
            : inh_head_modules[0].weight.shape[0], :
        ].clone()
        inh_head_modules[0].bias.data = source_layer.bias.data[
            : inh_head_modules[0].bias.shape[0]
        ].clone()
        logger.info("Transferred weights to inhibitory head")

    logger.info("Weight transfer from SimpleMLP to InputNetTransform completed")
