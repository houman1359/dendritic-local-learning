"""
Multi-stage training strategy.

This module implements a trainer that can execute multiple training stages
sequentially, each with different learning strategies and configurations.
"""

import json
import logging
import os
import shutil
from collections.abc import Iterator
from typing import Any, Optional

import torch
import torch.nn as nn

from dendritic_modeling.config import ParamGroupsConfig
from dendritic_modeling.config.multi_stage_training import (
    MultiStageTrainingConfig,
    TrainingStageConfig,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.utils.epoch_files import epoch_files_by_number
from dendritic_modeling.utils.hooks import iter_modules_matching

logger = logging.getLogger(__name__)


def _remove_epoch_files(epochs_dir: str) -> None:
    existing_files = [f for f in os.listdir(epochs_dir) if f.startswith("epoch")]
    for filename in existing_files:
        os.remove(os.path.join(epochs_dir, filename))


def _epoch_files_by_local_epoch(epochs_dir: str) -> list[tuple[int, str]]:
    return epoch_files_by_number(epochs_dir, require_json=False)


def _json_file_has_payload(path: str) -> bool:
    try:
        with open(path) as f:
            data = json.load(f)
        return bool(data)
    except Exception:
        return False


def _copy_valid_epoch_file(src_path: str, dest_path: str) -> bool:
    try:
        if not os.path.exists(src_path):
            return False
        if not _json_file_has_payload(src_path):
            return False
        shutil.copy(src_path, dest_path)
        return True
    except Exception:
        return False


def _is_resettable_module(module: nn.Module) -> bool:
    """Return whether a module participates in between-stage weight resets."""
    return hasattr(module, "reset_parameters") or isinstance(
        module, (nn.Linear, nn.Conv2d)
    )


def _reset_module_weights(module: nn.Module) -> None:
    """Reset one module using the historical multi-stage reset rule."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    elif isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _iter_resettable_modules(model: nn.Module) -> Iterator[nn.Module]:
    """Yield resettable modules in module traversal order."""
    yield from iter_modules_matching(model, _is_resettable_module)


def _concatenate_stage_performance_epochs(
    stages: list[TrainingStageConfig],
    original_save_root: str,
) -> None:
    main_epochs_dir = os.path.join(original_save_root, "performance", "epochs")
    os.makedirs(main_epochs_dir, exist_ok=True)
    _remove_epoch_files(main_epochs_dir)

    global_epoch = 1
    for stage_idx, _stage_config in enumerate(stages):
        stage_name = f"stage_{stage_idx + 1}"
        stage_dir = os.path.join(original_save_root, stage_name)
        stage_epochs_dir = os.path.join(stage_dir, "performance", "epochs")

        for _local_epoch, epoch_file in _epoch_files_by_local_epoch(stage_epochs_dir):
            src_path = os.path.join(stage_epochs_dir, epoch_file)
            dest_file = f"epoch{global_epoch}.json"
            dest_path = os.path.join(main_epochs_dir, dest_file)

            if _copy_valid_epoch_file(src_path, dest_path):
                global_epoch += 1


def _prepare_stage_analysis_root(
    analysis_manager,
    original_save_root: str,
    stage_name: str,
) -> None:
    if analysis_manager is None or original_save_root is None:
        return

    stage_save_root = os.path.join(original_save_root, stage_name)
    os.makedirs(stage_save_root, exist_ok=True)
    analysis_manager.save_root = stage_save_root

    stage_epochs_dir = os.path.join(stage_save_root, "performance", "epochs")
    if os.path.exists(stage_epochs_dir):
        _remove_epoch_files(stage_epochs_dir)


def _run_between_stage_analysis(
    analysis_manager,
    stage_name: str,
    original_save_root: str,
) -> None:
    logger.info(f"Running analysis after stage {stage_name}")
    if original_save_root is not None:
        stage_save_root = os.path.join(original_save_root, stage_name)
        analysis_manager.save_root = stage_save_root

    try:
        analysis_manager.run_analysis(filename=f"{stage_name}", training=False)
    finally:
        if original_save_root is not None:
            analysis_manager.save_root = original_save_root


def _build_stage_trainer_config(
    base_trainer_config: dict[str, Any],
    multi_stage_config: MultiStageTrainingConfig,
    stage_config: TrainingStageConfig,
) -> dict[str, Any]:
    stage_trainer_config = base_trainer_config.copy()
    stage_trainer_config.update(stage_config.trainer_config)

    if stage_config.epochs is not None:
        stage_trainer_config["epochs"] = stage_config.epochs
    if stage_config.batch_size is not None:
        stage_trainer_config["batch_size"] = stage_config.batch_size
    elif multi_stage_config.global_batch_size is not None:
        stage_trainer_config["batch_size"] = multi_stage_config.global_batch_size

    if stage_config.param_groups is not None:
        stage_trainer_config["param_groups"] = stage_config.param_groups

    stage_trainer_config["shuffle"] = stage_trainer_config.get(
        "shuffle", multi_stage_config.global_shuffle
    )
    stage_trainer_config["grad_clip_value"] = stage_trainer_config.get(
        "grad_clip_value", multi_stage_config.global_grad_clip_value
    )

    if stage_config.local_rule_config is not None:
        stage_trainer_config["local_rule_config"] = stage_config.local_rule_config

    return stage_trainer_config


def _optimizer_class_and_weight_decay(base_optimizer):
    if hasattr(base_optimizer, "optimizer"):
        actual_optimizer = base_optimizer.optimizer
        optimizer_class = type(actual_optimizer)
        weight_decay = base_optimizer.weight_decay
    else:
        optimizer_class = type(base_optimizer)
        weight_decay = 0.0
    return optimizer_class, weight_decay


def _wrap_with_custom_weight_decay(model, optimizer, weight_decay: float):
    if weight_decay > 0:
        return CustomWeightDecayOptimizer(
            model=model,
            optimizer=optimizer,
            weight_decay=weight_decay,
        )
    return optimizer


def _build_stage_optimizer(
    model: BaseModel,
    base_optimizer,
    stage_config: TrainingStageConfig,
    stage_idx: int,
):
    if stage_config.param_groups is not None:
        param_groups_config = ParamGroupsConfig(**stage_config.param_groups)
        optimizer_class, weight_decay = _optimizer_class_and_weight_decay(
            base_optimizer
        )
        param_groups = model.get_param_groups(param_groups_config)
        new_optimizer = optimizer_class(param_groups)
        return _wrap_with_custom_weight_decay(model, new_optimizer, weight_decay)

    if stage_config.reset_optimizer or stage_idx == 0:
        optimizer_class, weight_decay = _optimizer_class_and_weight_decay(
            base_optimizer
        )
        new_optimizer = optimizer_class(model.parameters(), lr=0.001)
        return _wrap_with_custom_weight_decay(model, new_optimizer, weight_decay)

    return base_optimizer


def _load_stage_checkpoint(model: BaseModel, checkpoint_path: str) -> None:
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def _build_stage_trainer(
    stage_config: TrainingStageConfig,
    stage_optimizer,
    stage_trainer_config: dict[str, Any],
    analysis_manager,
):
    from dendritic_modeling.training.factory import get_trainer

    return get_trainer(
        strategy=stage_config.learning_strategy,
        optimizer=stage_optimizer,
        trainer_config_dict=stage_trainer_config,
        analysis_manager=analysis_manager,
    )


def _stage_trainer_save_path(
    save_path: str,
    stage_idx: int,
    stage_name: str,
) -> str:
    return f"{save_path}/stage_{stage_idx}_{stage_name}"


def _stage_checkpoint_path(
    save_path: str,
    stage_idx: int,
    stage_name: str,
) -> str:
    return f"{save_path}/checkpoint_stage_{stage_idx}_{stage_name}.pt"


def _configure_stage_trainer_save_path(
    stage_trainer,
    save_path: str,
    stage_config: TrainingStageConfig,
    stage_idx: int,
    stage_name: str,
) -> None:
    if save_path and stage_config.save_checkpoint:
        stage_trainer.save_path = _stage_trainer_save_path(
            save_path,
            stage_idx,
            stage_name,
        )


def _optimizer_state_dict(optimizer):
    if optimizer and hasattr(optimizer, "optimizer"):
        return optimizer.optimizer.state_dict()
    if optimizer:
        return optimizer.state_dict()
    return None


def _stage_checkpoint_optimizer_state(stage_trainer, stage_optimizer):
    if hasattr(stage_trainer, "optimizer"):
        return _optimizer_state_dict(stage_trainer.optimizer)
    return _optimizer_state_dict(stage_optimizer)


def _save_stage_checkpoint(
    model,
    stage_config: TrainingStageConfig,
    stage_results: dict[str, Any],
    stage_trainer,
    stage_optimizer,
    checkpoint_path: str,
) -> None:
    optimizer_state = _stage_checkpoint_optimizer_state(stage_trainer, stage_optimizer)
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "stage_config": stage_config,
        "stage_results": stage_results,
    }
    if optimizer_state is not None:
        checkpoint_data["optimizer_state_dict"] = optimizer_state

    torch.save(checkpoint_data, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def _initial_multi_stage_results() -> dict[str, Any]:
    return {
        "stages": [],
        "final_state_dict": None,
        "best_stage": None,
        "best_loss": float("inf"),
    }


def _append_stage_result(
    all_results: dict[str, Any],
    stage_name: str,
    stage_idx: int,
    stage_results: dict[str, Any],
) -> None:
    all_results["stages"].append(
        {
            "stage_name": stage_name,
            "stage_idx": stage_idx,
            "results": stage_results,
        }
    )


def _update_best_stage(
    stage_results: dict[str, Any],
    best_overall_loss: float,
    best_overall_state_dict,
    best_stage_name: Optional[str],
    stage_name: str,
) -> tuple[float, Any, Optional[str]]:
    if stage_results.get("best_loss", float("inf")) < best_overall_loss:
        return (
            stage_results["best_loss"],
            stage_results.get("best_state_dict"),
            stage_name,
        )

    return best_overall_loss, best_overall_state_dict, best_stage_name


def _finalize_multi_stage_results(
    all_results: dict[str, Any],
    model: BaseModel,
    best_stage_name: Optional[str],
    best_overall_loss: float,
) -> None:
    all_results["final_state_dict"] = model.state_dict()
    all_results["best_stage"] = best_stage_name
    all_results["best_loss"] = best_overall_loss


class MultiStageTrainer(Trainer):
    """
    Multi-stage training strategy that executes multiple training phases sequentially.

    This trainer allows you to:
    - Train with standard backprop first, then switch to local learning
    - Use different hyperparameters for each stage
    - Save/load checkpoints between stages
    - Run analysis between stages
    """

    def __init__(
        self,
        multi_stage_config: MultiStageTrainingConfig,
        base_optimizer: torch.optim.Optimizer,
        base_trainer_config: dict[str, Any],
        analysis_manager: Optional[object] = None,
        **kwargs,
    ):
        """
        Initialize multi-stage trainer.

        Args:
            multi_stage_config: Configuration for multi-stage training
            base_optimizer: Base optimizer (can be recreated for each stage)
            base_trainer_config: Base trainer configuration (can be overridden per stage)
            analysis_manager: Optional analysis manager
            **kwargs: Additional arguments passed to base Trainer
        """
        # Remove optimizer from base_trainer_config if present to avoid duplicate
        base_trainer_config_copy = base_trainer_config.copy()
        base_trainer_config_copy.pop("optimizer", None)

        # Initialize with base config for compatibility
        super().__init__(optimizer=base_optimizer, **base_trainer_config_copy, **kwargs)

        self.multi_stage_config = multi_stage_config
        self.base_optimizer = base_optimizer
        self.base_trainer_config = base_trainer_config
        self.analysis_manager = analysis_manager
        self.stage_results = []

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict[str, Any]:
        """
        Execute multi-stage training.

        Args:
            model: Model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing results from all stages
        """
        if not self.multi_stage_config.stages:
            raise ValueError("No training stages defined in multi_stage_config")

        logger.info(
            f"Starting multi-stage training with {len(self.multi_stage_config.stages)} stages"
        )

        all_results = _initial_multi_stage_results()

        # Keep track of the best model across all stages
        best_overall_loss = float("inf")
        best_overall_state_dict = None
        best_stage_name = None

        original_save_root = (
            self.analysis_manager.save_root if self.analysis_manager else None
        )

        for stage_idx, stage_config in enumerate(self.multi_stage_config.stages):
            stage_name = f"stage_{stage_idx + 1}"

            logger.info(f"\n{'='*60}")
            logger.info(
                f"Starting Stage {stage_idx + 1}/{len(self.multi_stage_config.stages)}: {stage_name}"
            )
            logger.info(f"Learning strategy: {stage_config.learning_strategy}")
            logger.info(f"{'='*60}\n")

            try:
                stage_results = self._execute_stage(
                    model,
                    train_data,
                    valid_data,
                    stage_config,
                    stage_idx,
                    stage_name,
                    original_save_root,
                )

                _append_stage_result(
                    all_results,
                    stage_name,
                    stage_idx,
                    stage_results,
                )

                (
                    best_overall_loss,
                    best_overall_state_dict,
                    best_stage_name,
                ) = _update_best_stage(
                    stage_results,
                    best_overall_loss,
                    best_overall_state_dict,
                    best_stage_name,
                    stage_name,
                )

                if (
                    self.multi_stage_config.analyze_between_stages
                    and self.analysis_manager is not None
                ):
                    _run_between_stage_analysis(
                        self.analysis_manager,
                        stage_name,
                        original_save_root,
                    )

            except Exception as e:
                logger.error(f"Error in stage {stage_name}: {e!s}")
                if not self.multi_stage_config.continue_on_failure:
                    raise
                logger.warning("Continuing to next stage despite failure")

        _finalize_multi_stage_results(
            all_results,
            model,
            best_stage_name,
            best_overall_loss,
        )

        # Load best model if requested
        if (
            self.multi_stage_config.use_best_from_each_stage
            and best_overall_state_dict is not None
        ):
            logger.info(
                f"Loading best model from stage: {best_stage_name} (loss: {best_overall_loss:.4f})"
            )
            model.load_state_dict(best_overall_state_dict)

        if (
            self.multi_stage_config.analyze_between_stages
            and self.analysis_manager is not None
            and original_save_root is not None
        ):
            logger.info("Concatenating performance data across stages")
            _concatenate_stage_performance_epochs(
                self.multi_stage_config.stages,
                original_save_root,
            )

        return all_results

    def _execute_stage(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
        stage_config: TrainingStageConfig,
        stage_idx: int,
        stage_name: str,
        original_save_root: str,
    ) -> dict[str, Any]:
        """Execute a single training stage."""

        _prepare_stage_analysis_root(
            self.analysis_manager,
            original_save_root,
            stage_name,
        )

        try:
            stage_trainer_config = _build_stage_trainer_config(
                self.base_trainer_config,
                self.multi_stage_config,
                stage_config,
            )

            if stage_config.load_checkpoint is not None:
                _load_stage_checkpoint(model, stage_config.load_checkpoint)

            if stage_config.reset_model:
                logger.warning(f"Resetting model weights for stage {stage_name}")
                self._reset_model_weights(model)

            stage_optimizer = _build_stage_optimizer(
                model,
                self.base_optimizer,
                stage_config,
                stage_idx,
            )

            stage_trainer = _build_stage_trainer(
                stage_config,
                stage_optimizer,
                stage_trainer_config,
                self.analysis_manager,
            )

            _configure_stage_trainer_save_path(
                stage_trainer,
                self.save_path,
                stage_config,
                stage_idx,
                stage_name,
            )

            logger.info(
                f"Training with {stage_config.learning_strategy} for {stage_trainer_config.get('epochs', 'default')} epochs"
            )
            stage_results = stage_trainer.train(model, train_data, valid_data)

            if hasattr(stage_trainer, "optimizer"):
                self.base_optimizer = stage_trainer.optimizer

            if stage_config.save_checkpoint and self.save_path:
                checkpoint_path = _stage_checkpoint_path(
                    self.save_path,
                    stage_idx,
                    stage_name,
                )

                _save_stage_checkpoint(
                    model,
                    stage_config,
                    stage_results,
                    stage_trainer,
                    stage_optimizer,
                    checkpoint_path,
                )

            return stage_results
        finally:
            if self.analysis_manager is not None and original_save_root is not None:
                self.analysis_manager.save_root = original_save_root

    def _reset_model_weights(self, model: nn.Module):
        """Reset model weights to random initialization."""
        for module in _iter_resettable_modules(model):
            _reset_module_weights(module)
