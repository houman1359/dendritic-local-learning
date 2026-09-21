"""
Specialized training strategies.

This module implements specialized training approaches including voltage stabilization
and other advanced training techniques for dendritic networks.
"""

import logging
from functools import partial

import torch

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.activations import ParametricActivation
from dendritic_modeling.training.loss.functions import (
    ExcitationInhibitionBalanceLoss,
    HomeostaticControlLoss,
    VoltageStabilizationLoss,
)
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.utils.hooks import iter_modules_of_type, run_with_forward_hooks

logger = logging.getLogger(__name__)


def _set_parametric_activation_trainability(model, requires_grad: bool) -> None:
    for module in iter_modules_of_type(model, ParametricActivation):
        for param in module.parameters():
            param.requires_grad = requires_grad


def _start_warmup_phase(trainer, warmup_loss, filename_prefix, logger_info_prefix):
    default_loss_fn = trainer.loss_function
    default_epochs = trainer.epochs

    trainer.loss_function = warmup_loss
    trainer.epochs = max(trainer.pretrain_epochs, 1)
    trainer.filename_prefix = filename_prefix
    trainer.logger_info_prefix = logger_info_prefix

    return default_loss_fn, default_epochs


def _start_standard_phase(trainer, default_loss_fn, default_epochs):
    trainer.loss_function = default_loss_fn
    trainer.epochs = default_epochs
    trainer.filename_prefix = "standard_"
    trainer.logger_info_prefix = "[Standard] "
    trainer.best_loss = float("inf")


def _merge_warmup_results(pretrain_results, final_results):
    final_results["train_losses"] = (
        pretrain_results["train_losses"] + final_results["train_losses"]
    )
    final_results["valid_losses"] = (
        pretrain_results["valid_losses"] + final_results["valid_losses"]
    )
    return final_results


def _run_warmup_then_standard_training(
    trainer,
    train_fn,
    model,
    train_data,
    valid_data,
    warmup_loss,
    *,
    filename_prefix,
    logger_info_prefix,
    warmup_label,
    standard_label,
):
    handles = warmup_loss.attach_forward_hooks(model)

    default_loss_fn, default_epochs = _start_warmup_phase(
        trainer,
        warmup_loss,
        filename_prefix,
        logger_info_prefix,
    )

    if trainer.freeze_reactivation:
        _set_parametric_activation_trainability(model, False)

    logger.info(
        f"=== {warmup_label}: "
        f"Starting pretraining phase for {trainer.pretrain_epochs} epochs ==="
    )
    try:
        pretrain_results = run_with_forward_hooks(
            attach=lambda: handles,
            remove=warmup_loss.remove_forward_hooks,
            body=partial(train_fn, model, train_data, valid_data),
        )
    finally:
        if trainer.freeze_reactivation:
            _set_parametric_activation_trainability(model, True)

        _start_standard_phase(trainer, default_loss_fn, default_epochs)

    logger.info(
        f"=== {standard_label}: "
        f"Starting normal training for {trainer.epochs} epochs ==="
    )
    final_results = train_fn(model, train_data, valid_data)
    return _merge_warmup_results(pretrain_results, final_results)


class VoltageStabilizationTrainer(Trainer):
    """
    Voltage Stabilization Training Strategy.

    This trainer includes a warm-up phase that stabilizes dendritic voltages
    to target values before proceeding with normal training. This can help
    with training stability in dendritic networks.
    """

    pretrain_epochs: int = 1
    stabilize_mode: str = "vinf"
    freeze_reactivation: bool = False
    target_voltage: float = 0.5
    target_saturation: float = 0.2
    loss_metric: str = "mse"

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train the model with voltage stabilization warm-up.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """
        voltage_stabilization_loss = VoltageStabilizationLoss(
            stabilize_mode=self.stabilize_mode,
            target_voltage=self.target_voltage,
            target_saturation=self.target_saturation,
            loss_metric=self.loss_metric,
            reduction=self.loss_function.reduction,
        )

        return _run_warmup_then_standard_training(
            self,
            super().train,
            model,
            train_data,
            valid_data,
            voltage_stabilization_loss,
            filename_prefix="voltage_stabilization_",
            logger_info_prefix="[Voltage Stabilization] ",
            warmup_label="VoltageStabilization",
            standard_label="Voltage Stabilization",
        )


class ExcitationInhibitionEquilibriumTrainer(Trainer):
    """
    Excitation-Inhibition Equilibrium Training Strategy.

    This trainer includes a warm-up phase that stabilizes dendritic voltages
    to target values before proceeding with normal training. This can help
    with training stability in dendritic networks.
    """

    pretrain_epochs: int = 1
    freeze_reactivation: bool = False
    loss_metric: str = "mse"

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train the model with excitation-inhibition equilibrium warm-up.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """
        ei_equilibrium_loss = ExcitationInhibitionBalanceLoss(
            loss_metric=self.loss_metric
        )

        return _run_warmup_then_standard_training(
            self,
            super().train,
            model,
            train_data,
            valid_data,
            ei_equilibrium_loss,
            filename_prefix="ei_equilibrium_",
            logger_info_prefix="[Excitation-Inhibition Equilibrium] ",
            warmup_label="Excitation-Inhibition Equilibrium",
            standard_label="Excitation-Inhibition Equilibrium",
        )


class HomeostaticControlTrainer(Trainer):
    """
    Homeostatic Control Training Strategy.

    This trainer includes a warm-up phase that stabilizes dendritic voltages
    and excitatory-inhibitory balance to target values before proceeding with normal training.
    This can help with training stability in dendritic networks.
    """

    pretrain_epochs: int = 1
    stabilize_mode: str = "vinf"
    freeze_reactivation: bool = False
    target_voltage: float = 0.5
    target_saturation: float = 0.2
    loss_metric: str = "mse"

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train the model with homeostatic control warm-up.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """
        homeostatic_control_loss = HomeostaticControlLoss(
            stabilize_mode=self.stabilize_mode,
            target_voltage=self.target_voltage,
            target_saturation=self.target_saturation,
            loss_metric=self.loss_metric,
            reduction=self.loss_function.reduction,
        )

        return _run_warmup_then_standard_training(
            self,
            super().train,
            model,
            train_data,
            valid_data,
            homeostatic_control_loss,
            filename_prefix="homeostatic_control_",
            logger_info_prefix="[Homeostatic Control] ",
            warmup_label="Homeostatic Control",
            standard_label="HomeostaticControl",
        )


class TrainOnlyMReactivation(Trainer):
    """
    Training strategy that focuses only on M-reactivation parameters.

    This trainer freezes log_b parameters and trains all other parameters,
    which can be useful for specific dendritic network configurations.
    """

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train the model with M-reactivation focus.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """

        logger.info(
            "TrainOnlyMReactivation: "
            "Freezing log_b and training all other parameters."
        )

        # Freeze log_b parameters, unfreeze all others
        for name, param in model.named_parameters():
            if "log_b" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.filename_prefix = "train_only_m_reactivation_"
        results = super().train(model, train_data, valid_data)
        return results
