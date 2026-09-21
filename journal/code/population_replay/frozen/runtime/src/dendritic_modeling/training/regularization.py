"""
Regularization and Pruning Utilities
====================================

This module provides utilities for L1/L2 regularization and weight pruning
with selective application based on layer, branch, and weight type.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from dendritic_modeling.config.training import PruningConfig, RegularizationConfig
from dendritic_modeling.training.loss.functions import EIWeightRatioLoss

_PARAM_GROUP_FALLBACK_NAMES = (
    "topk",
    "blocklinear",
    "reactivation",
    "mlp_inh_net",
    "decoder",
    "encoder",
    "catch_all",
)


def _config_value(config: Any, key: str, default: float) -> float:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _regularization_weights(
    group_config: Any,
    *,
    default_l1: float,
    default_l2: float,
) -> tuple[float, float]:
    return (
        _config_value(group_config, "l1_weight", default_l1),
        _config_value(group_config, "l2_weight", default_l2),
    )


def _format_pruning_stats_section(
    title: str,
    rows: dict[str, dict[str, Any]],
    key_width: int,
    separator_width: int,
) -> list[str]:
    if not rows:
        return []

    lines = [
        f"\n{title:<{key_width}} {'Before':<10} {'After':<10} {'Pruned':<10} {'Ratio':<10}",
        "-" * separator_width,
    ]
    for name, row_stats in rows.items():
        ratio = (
            row_stats["pruned"] / row_stats["total_before"]
            if row_stats["total_before"] > 0
            else 0
        )
        lines.append(
            f"{name:<{key_width}} {row_stats['total_before']:<10,} "
            f"{row_stats['remaining']:<10,} {row_stats['pruned']:<10,} "
            f"{ratio:<10.4f}"
        )
    return lines


def _format_pruning_report(stats: dict[str, Any]) -> list[str]:
    lines = [
        "\n" + "=" * 50,
        "WEIGHT PRUNING REPORT",
        "=" * 50,
        f"Total parameters before pruning: {stats['total_params_before']:,}",
        f"Total parameters after pruning:  {stats['total_params_after']:,}",
        f"Pruned parameters:               {stats['pruned_params']:,}",
        f"Pruning ratio:                   {stats['pruning_ratio']:.4f}",
    ]

    lines.extend(
        _format_pruning_stats_section(
            "Layer",
            stats["layer_stats"],
            key_width=15,
            separator_width=60,
        )
    )
    lines.extend(
        _format_pruning_stats_section(
            "Branch",
            stats["branch_stats"],
            key_width=20,
            separator_width=65,
        )
    )
    lines.extend(
        _format_pruning_stats_section(
            "Weight Type",
            stats["weight_type_stats"],
            key_width=15,
            separator_width=60,
        )
    )
    lines.append("=" * 50)
    return lines


def _format_pruning_report_text(stats: dict[str, Any]) -> str:
    """Return the full pruning report as display-ready text."""
    return "\n".join(_format_pruning_report(stats))


class RegularizationManager:
    """Manages L1/L2 regularization with selective application."""

    def __init__(self, config: RegularizationConfig):
        self.config = config
        self.l1_weight = config.l1_weight
        self.l2_weight = config.l2_weight
        self.selective = config.selective
        self.split_params = config.split_params
        self.param_group_weights = config.param_group_weights

        # E/I Weight Ratio Regularization
        self.enforce_ei_weight_ratio = getattr(config, "enforce_ei_weight_ratio", False)
        if self.enforce_ei_weight_ratio:
            self.ei_ratio_loss = EIWeightRatioLoss(
                target_ratio=getattr(config, "target_ei_weight_ratio", 1.5),
                loss_weight=getattr(config, "ei_ratio_loss_weight", 0.1),
                scope=getattr(config, "ei_ratio_scope", "per_branch"),
                metric=getattr(config, "ei_ratio_metric", "mean"),
            )
        else:
            self.ei_ratio_loss = None

        # Cache whether any regularization is enabled for performance
        self._has_regularization = self._check_has_regularization()

    def _check_has_regularization(self) -> bool:
        """Check if any form of regularization is enabled."""
        # Check L1/L2 regularization
        if self.l1_weight > 0 or self.l2_weight > 0:
            return True

        # Check parameter group regularization
        if self.split_params and self.param_group_weights:
            for group_weights in self.param_group_weights.values():
                l1_weight, l2_weight = _regularization_weights(
                    group_weights,
                    default_l1=0,
                    default_l2=0,
                )
                if l1_weight > 0 or l2_weight > 0:
                    return True

        # Check E/I ratio regularization
        if self.enforce_ei_weight_ratio and self.ei_ratio_loss is not None:
            return True

        return False

    def has_regularization(self) -> bool:
        """Return whether any regularization is enabled."""
        return self._has_regularization

    def compute_regularization_loss(
        self,
        model: nn.Module,
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        param_groups: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Compute all regularization losses based on configuration.

        Args:
            model: The model to regularize
            x: Input tensor (needed for E/I weight ratio regularization)
            y: Target tensor (needed for E/I weight ratio regularization)
            param_groups: Optional parameter groups with their parameters

        Returns:
            Total regularization loss tensor
        """
        device = next(model.parameters()).device
        total_reg_loss = torch.tensor(0.0, device=device)

        # Compute L1/L2 regularization
        l1_l2_loss = self._compute_l1_l2_regularization(model, param_groups, device)
        total_reg_loss += l1_l2_loss

        # Compute E/I weight ratio regularization
        if self.enforce_ei_weight_ratio and self.ei_ratio_loss is not None:
            ei_ratio_loss = self.ei_ratio_loss(model, x, y)
            total_reg_loss += ei_ratio_loss

        return total_reg_loss

    def _compute_l1_l2_regularization(
        self,
        model: nn.Module,
        param_groups: Optional[list] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Compute L1/L2 regularization loss.

        Args:
            model: The model to regularize
            param_groups: Optional parameter groups with their parameters
            device: Device for tensor operations

        Returns:
            L1/L2 regularization loss tensor
        """
        if device is None:
            device = next(model.parameters()).device

        l1_l2_loss = torch.tensor(0.0, device=device)

        # If split_params is enabled and parameter group weights are specified and param_groups are provided
        if self.split_params and self.param_group_weights and param_groups:
            return self._compute_param_group_regularization(param_groups, device)

        # Otherwise use global regularization weights
        if self.l1_weight <= 0 and self.l2_weight <= 0:
            return l1_l2_loss

        l1_reg = torch.tensor(0.0, device=device)
        l2_reg = torch.tensor(0.0, device=device)

        for name, param in model.named_parameters():
            if not param.requires_grad or "weight" not in name:
                continue

            if self._should_regularize(name, model):
                if self.l1_weight > 0:
                    l1_reg += torch.sum(torch.abs(param))
                if self.l2_weight > 0:
                    l2_reg += torch.sum(param**2)

        return self.l1_weight * l1_reg + self.l2_weight * l2_reg

    def _compute_param_group_regularization(
        self, param_groups: list, device
    ) -> torch.Tensor:
        """Compute regularization loss for parameter groups with different weights."""
        total_reg_loss = torch.tensor(0.0, device=device)

        for group_idx, group in enumerate(param_groups):
            # Determine group type based on learning rate or parameter names
            group_type = self._identify_group_type(group, group_idx)

            # Get regularization weights for this group
            group_reg_config = self.param_group_weights.get(group_type, {})
            l1_weight, l2_weight = _regularization_weights(
                group_reg_config,
                default_l1=self.l1_weight,
                default_l2=self.l2_weight,
            )

            if l1_weight <= 0 and l2_weight <= 0:
                continue

            # Apply regularization to parameters in this group
            l1_reg = torch.tensor(0.0, device=device)
            l2_reg = torch.tensor(0.0, device=device)

            for param in group["params"]:
                if param.requires_grad:
                    if l1_weight > 0:
                        l1_reg += torch.sum(torch.abs(param))
                    if l2_weight > 0:
                        l2_reg += torch.sum(param**2)

            total_reg_loss += l1_weight * l1_reg + l2_weight * l2_reg

        return total_reg_loss

    def _identify_group_type(self, group: dict, group_idx: int) -> str:
        """Identify optimizer group type for per-group regularization weights."""
        group_name = group.get("name")
        if group_name:
            return str(group_name)

        # Backward-compatible fallback for externally supplied param groups
        # that predate the explicit BaseModel group names.
        if group_idx < len(_PARAM_GROUP_FALLBACK_NAMES):
            return _PARAM_GROUP_FALLBACK_NAMES[group_idx]

        # Fallback to generic naming for additional groups
        return f"group_{group_idx}"

    def _should_regularize(self, param_name: str, model: nn.Module) -> bool:
        """Check if parameter should be regularized based on selective criteria."""
        # Extract layer index from parameter name
        layer_idx = self._extract_layer_index(param_name)

        # Check layer selection
        if self.selective.layers and layer_idx is not None:
            if layer_idx not in self.selective.layers:
                return False

        # Check branch selection
        if self.selective.branches:
            branch_found = any(
                branch in param_name for branch in self.selective.branches
            )
            if not branch_found:
                return False

        # Check weight type selection
        if self.selective.weight_types and "all" not in self.selective.weight_types:
            weight_type_found = any(
                wt in param_name for wt in self.selective.weight_types
            )
            if not weight_type_found:
                return False

        return True

    def _extract_layer_index(self, param_name: str) -> Optional[int]:
        """Extract layer index from parameter name."""
        parts = param_name.split(".")
        for part in parts:
            if part.isdigit():
                return int(part)
        return None


class PruningManager:
    """Manages weight pruning with selective application and reporting."""

    def __init__(self, config: PruningConfig):
        self.config = config
        self.threshold = config.threshold
        self.selective = config.selective
        self.pruning_stats = {}

    def prune_model(self, model: nn.Module) -> dict[str, Any]:
        """
        Prune model weights based on configuration.

        Args:
            model: The model to prune

        Returns:
            Dictionary containing pruning statistics
        """
        if not self.config.enabled:
            return {}

        pruning_stats = {
            "total_params_before": 0,
            "total_params_after": 0,
            "pruned_params": 0,
            "layer_stats": {},
            "branch_stats": {},
            "weight_type_stats": {},
        }

        for name, param in model.named_parameters():
            if "weight" not in name:
                continue

            original_count = param.numel()
            pruning_stats["total_params_before"] += original_count

            if self._should_prune(name, model):
                # Apply pruning mask
                mask = torch.abs(param.data) >= self.threshold
                param.data[~mask] = 0.0

                remaining_count = mask.sum().item()
                pruned_count = original_count - remaining_count

                # Update statistics
                layer_idx = self._extract_layer_index(name)
                layer_key = f"layer_{layer_idx}" if layer_idx is not None else "unknown"

                if layer_key not in pruning_stats["layer_stats"]:
                    pruning_stats["layer_stats"][layer_key] = {
                        "total_before": 0,
                        "remaining": 0,
                        "pruned": 0,
                    }

                pruning_stats["layer_stats"][layer_key][
                    "total_before"
                ] += original_count
                pruning_stats["layer_stats"][layer_key]["remaining"] += remaining_count
                pruning_stats["layer_stats"][layer_key]["pruned"] += pruned_count

                # Branch statistics
                branch_name = self._extract_branch_name(name)
                if branch_name:
                    if branch_name not in pruning_stats["branch_stats"]:
                        pruning_stats["branch_stats"][branch_name] = {
                            "total_before": 0,
                            "remaining": 0,
                            "pruned": 0,
                        }

                    pruning_stats["branch_stats"][branch_name][
                        "total_before"
                    ] += original_count
                    pruning_stats["branch_stats"][branch_name][
                        "remaining"
                    ] += remaining_count
                    pruning_stats["branch_stats"][branch_name]["pruned"] += pruned_count

                # Weight type statistics
                weight_type = self._extract_weight_type(name)
                if weight_type not in pruning_stats["weight_type_stats"]:
                    pruning_stats["weight_type_stats"][weight_type] = {
                        "total_before": 0,
                        "remaining": 0,
                        "pruned": 0,
                    }

                pruning_stats["weight_type_stats"][weight_type][
                    "total_before"
                ] += original_count
                pruning_stats["weight_type_stats"][weight_type][
                    "remaining"
                ] += remaining_count
                pruning_stats["weight_type_stats"][weight_type][
                    "pruned"
                ] += pruned_count

                pruning_stats["total_params_after"] += remaining_count
                pruning_stats["pruned_params"] += pruned_count
            else:
                pruning_stats["total_params_after"] += original_count

        # Calculate pruning ratio
        if pruning_stats["total_params_before"] > 0:
            pruning_stats["pruning_ratio"] = (
                pruning_stats["pruned_params"] / pruning_stats["total_params_before"]
            )
        else:
            pruning_stats["pruning_ratio"] = 0.0

        self.pruning_stats = pruning_stats
        return pruning_stats

    def _should_prune(self, param_name: str, model: nn.Module) -> bool:
        """Check if parameter should be pruned based on selective criteria."""
        # Extract layer index from parameter name
        layer_idx = self._extract_layer_index(param_name)

        # Check layer selection
        if self.selective.layers and layer_idx is not None:
            if layer_idx not in self.selective.layers:
                return False

        # Check branch selection
        if self.selective.branches:
            branch_found = any(
                branch in param_name for branch in self.selective.branches
            )
            if not branch_found:
                return False

        # Check weight type selection
        if self.selective.weight_types and "all" not in self.selective.weight_types:
            weight_type_found = any(
                wt in param_name for wt in self.selective.weight_types
            )
            if not weight_type_found:
                return False

        return True

    def _extract_layer_index(self, param_name: str) -> Optional[int]:
        """Extract layer index from parameter name."""
        parts = param_name.split(".")
        for part in parts:
            if part.isdigit():
                return int(part)
        return None

    def _extract_branch_name(self, param_name: str) -> Optional[str]:
        """Extract branch name from parameter name."""
        if "branch" in param_name:
            parts = param_name.split(".")
            for _i, part in enumerate(parts):
                if "branch" in part:
                    return part
        return None

    def _extract_weight_type(self, param_name: str) -> str:
        """Extract weight type from parameter name."""
        if "recurrent" in param_name:
            return "recurrent"
        elif "input" in param_name:
            return "input"
        elif "output" in param_name:
            return "output"
        elif "linear" in param_name:
            return "linear"
        else:
            return "other"

    def print_pruning_report(self) -> None:
        """Print detailed pruning report."""
        if not self.pruning_stats:
            print("No pruning statistics available.")
            return

        print(_format_pruning_report_text(self.pruning_stats))

    def save_pruning_stats(self, save_path: str) -> None:
        """Save pruning statistics to file."""
        if not self.pruning_stats:
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(self.pruning_stats, f, indent=2)

        print(f"Pruning statistics saved to: {save_path}")


def create_sparsity_mask(
    model: nn.Module, sparsity_ratio: float
) -> dict[str, torch.Tensor]:
    """
    Create sparsity masks for magnitude-based pruning.

    Args:
        model: The model to create masks for
        sparsity_ratio: Fraction of weights to prune (0.0 to 1.0)

    Returns:
        Dictionary mapping parameter names to binary masks
    """
    masks = {}

    # Collect all weights and their magnitudes
    all_weights = []
    weight_names = []

    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            all_weights.append(param.data.abs().flatten())
            weight_names.append(name)

    if not all_weights:
        return masks

    # Calculate global threshold
    all_weights_concat = torch.cat(all_weights)
    threshold = torch.quantile(all_weights_concat, sparsity_ratio)

    # Create masks
    for name, param in model.named_parameters():
        if "weight" in name and param.requires_grad:
            masks[name] = param.data.abs() >= threshold

    return masks


def apply_sparsity_masks(model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
    """
    Apply sparsity masks to model parameters.

    Args:
        model: The model to apply masks to
        masks: Dictionary mapping parameter names to binary masks
    """
    for name, param in model.named_parameters():
        if name in masks:
            param.data *= masks[name].float()
