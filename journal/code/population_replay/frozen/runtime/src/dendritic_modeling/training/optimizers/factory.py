"""
Optimizer Factory
================

Factory functions for creating optimizers with flexible configuration.
"""

from collections.abc import Callable, Iterator
from typing import Any, Optional

import torch
import torch.optim as optim

from dendritic_modeling.config.training import OptimizerConfig

OptimizerBuilder = Callable[
    [Iterator[torch.nn.Parameter], OptimizerConfig], torch.optim.Optimizer
]

OPTIMIZER_REGISTRY: dict[str, OptimizerBuilder] = {}
_PARAMETER_GROUP_ORDER = ("default", "topk", "blocklinear", "reactivation", "decoder")
_SPECIAL_PARAMETER_GROUP_PATTERNS = (
    ("topk", "topk"),
    ("blocklinear", "blocklinear"),
    ("reactivation", "reactivation"),
    ("decoder", "decoder"),
)
_PARAMETER_GROUP_LR_KEYS = {
    "topk": "topk_lr",
    "blocklinear": "blocklinear_lr",
    "reactivation": "reactivation_lr",
    "decoder": "decoder_lr",
}


def register_optimizer(
    name: str,
    builder: OptimizerBuilder,
    *,
    aliases: tuple[str, ...] | list[str] = (),
    allow_override: bool = False,
) -> None:
    """Register an optimizer builder.

    Builders are called as ``builder(model_parameters, config)``.
    """
    names = [name, *aliases]
    if not names or any(not str(item).strip() for item in names):
        raise ValueError("optimizer name and aliases must be non-empty")
    for raw_name in names:
        key = str(raw_name).lower()
        if key in OPTIMIZER_REGISTRY and not allow_override:
            raise ValueError(f"Optimizer '{key}' is already registered")
        OPTIMIZER_REGISTRY[key] = builder


def unregister_optimizer(name: str) -> None:
    """Remove an optimizer builder if present."""
    OPTIMIZER_REGISTRY.pop(str(name).lower(), None)


def get_available_optimizers() -> list[str]:
    """Return registered optimizer names."""
    return sorted(OPTIMIZER_REGISTRY)


def _base_optimizer_params(config: OptimizerConfig) -> dict[str, Any]:
    return {"lr": config.lr, "weight_decay": config.weight_decay}


def _parameter_group_name(parameter_name: str) -> str:
    normalized_name = parameter_name.lower()
    for group_name, pattern in _SPECIAL_PARAMETER_GROUP_PATTERNS:
        if pattern in normalized_name:
            return group_name
    return "default"


def _split_named_parameters(
    model: torch.nn.Module,
) -> dict[str, list[torch.nn.Parameter]]:
    grouped_params: dict[str, list[torch.nn.Parameter]] = {
        group_name: [] for group_name in _PARAMETER_GROUP_ORDER
    }
    for name, param in model.named_parameters():
        grouped_params[_parameter_group_name(name)].append(param)
    return grouped_params


def _parameter_group_lr(
    group_name: str,
    config: OptimizerConfig,
    param_groups_config: dict[str, Any],
) -> float:
    if group_name == "default":
        return config.lr
    lr_key = _PARAMETER_GROUP_LR_KEYS[group_name]
    return param_groups_config.get(lr_key, config.lr)


def _build_adam(model_parameters, config: OptimizerConfig) -> torch.optim.Optimizer:
    params = _base_optimizer_params(config)
    params.update({"betas": config.betas, "eps": config.eps})
    if config.foreach is not None:
        params["foreach"] = config.foreach
    return optim.Adam(model_parameters, **params)


def _build_adamw(model_parameters, config: OptimizerConfig) -> torch.optim.Optimizer:
    params = _base_optimizer_params(config)
    params.update({"betas": config.betas, "eps": config.eps})
    if config.foreach is not None:
        params["foreach"] = config.foreach
    return optim.AdamW(model_parameters, **params)


def _build_sgd(model_parameters, config: OptimizerConfig) -> torch.optim.Optimizer:
    params = _base_optimizer_params(config)
    params.update({"momentum": config.momentum})
    return optim.SGD(model_parameters, **params)


def _build_rmsprop(model_parameters, config: OptimizerConfig) -> torch.optim.Optimizer:
    params = _base_optimizer_params(config)
    params.update({"eps": config.eps, "momentum": config.momentum})
    return optim.RMSprop(model_parameters, **params)


register_optimizer("adam", _build_adam)
register_optimizer("adamw", _build_adamw)
register_optimizer("sgd", _build_sgd)
register_optimizer("rmsprop", _build_rmsprop)


def create_optimizer(
    model_parameters: Iterator[torch.nn.Parameter], config: OptimizerConfig
) -> torch.optim.Optimizer:
    """
    Create an optimizer based on configuration.

    Args:
        model_parameters: Iterator over model parameters to optimize
        config: Optimizer configuration

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If optimizer name is not supported
    """
    optimizer_name = config.name.lower()
    builder = OPTIMIZER_REGISTRY.get(optimizer_name)
    if builder is None:
        valid_optimizers = get_available_optimizers()
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. "
            f"Choose from {valid_optimizers}"
        )
    return builder(model_parameters, config)


def create_parameter_groups(
    model: torch.nn.Module,
    config: OptimizerConfig,
    param_groups_config: Optional[dict[str, Any]] = None,
) -> list:
    """
    Create parameter groups for differential learning rates.

    Args:
        model: The model to create parameter groups for
        config: Optimizer configuration
        param_groups_config: Optional configuration for parameter groups

    Returns:
        List of parameter groups
    """
    if param_groups_config is None or not param_groups_config.get(
        "split_params", False
    ):
        # Return all parameters with default config
        return [{"params": model.parameters(), "lr": config.lr}]

    grouped_params = _split_named_parameters(model)
    param_groups = []
    for group_name in _PARAMETER_GROUP_ORDER:
        params = grouped_params[group_name]
        if params:
            param_groups.append(
                {
                    "params": params,
                    "lr": _parameter_group_lr(group_name, config, param_groups_config),
                }
            )
    return param_groups


def validate_optimizer_config(config: OptimizerConfig) -> None:
    """
    Validate optimizer configuration.

    Args:
        config: Optimizer configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    valid_optimizers = get_available_optimizers()
    if config.name.lower() not in valid_optimizers:
        raise ValueError(
            f"Invalid optimizer: {config.name}. " f"Choose from {valid_optimizers}"
        )

    if config.lr <= 0:
        raise ValueError(f"Learning rate must be positive, got {config.lr}")

    if config.weight_decay < 0:
        raise ValueError(
            f"Weight decay must be non-negative, got {config.weight_decay}"
        )

    if config.name.lower() in ["adam", "adamw"]:
        if config.foreach is not None and not isinstance(config.foreach, bool):
            raise ValueError(
                "Adam/AdamW foreach must be boolean or None, " f"got {config.foreach!r}"
            )
        if len(config.betas) != 2 or not all(0 <= b < 1 for b in config.betas):
            raise ValueError(f"Invalid betas for {config.name}: {config.betas}")

        if config.eps <= 0:
            raise ValueError(f"Epsilon must be positive, got {config.eps}")

    if config.name.lower() in ["sgd", "rmsprop"]:
        if config.momentum < 0:
            raise ValueError(f"Momentum must be non-negative, got {config.momentum}")
