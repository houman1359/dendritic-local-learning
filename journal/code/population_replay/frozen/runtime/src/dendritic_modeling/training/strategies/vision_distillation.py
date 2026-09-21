"""Online boundary distillation for AlexNet-shaped dendritic students."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model

from dendritic_modeling.training.loss.functions import LossFunction
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)

_BOUNDARY_NAMES = (
    "conv1",
    "conv2",
    "conv3",
    "conv4",
    "conv5",
    "fc6",
    "fc7",
    "fc8",
)
_TEACHER_FEATURE_INDICES = (1, 4, 7, 9, 11)
_TEACHER_FEATURE_INPUT_INDICES = (0, 3, 6, 8, 10)
_TEACHER_CLASSIFIER_INDICES = (2, 5, 6)
_TEACHER_CLASSIFIER_INPUT_INDICES = (1, 4, 6)


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "asdict") and callable(value.asdict):
        return dict(value.asdict())
    if hasattr(value, "__dict__"):
        return {
            key: item for key, item in vars(value).items() if not key.startswith("_")
        }
    raise TypeError("vision_distillation_config must be a mapping or config object")


def _student_core(model: nn.Module) -> nn.Module:
    unwrapped = model.module if hasattr(model, "module") else model
    core = getattr(unwrapped, "core_network", None)
    if core is None:
        raise TypeError("vision distillation requires a model with core_network")
    spatial = getattr(core, "spatial_blocks", None)
    feedforward = getattr(core, "feedforward_blocks", None)
    if spatial is None or len(spatial) != 5:
        raise TypeError("vision distillation requires five student spatial blocks")
    if feedforward is None or len(feedforward) < 3:
        raise TypeError("vision distillation requires fc6, fc7, and output blocks")
    return core


def _register_output_capture(
    modules: list[nn.Module],
    names: tuple[str, ...],
    destination: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in zip(names, modules, strict=True):
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, key=name: destination.__setitem__(
                    key, output
                )
            )
        )
    return handles


def _register_input_capture(
    modules: list[nn.Module],
    names: tuple[str, ...],
    destination: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, module in zip(names, modules, strict=True):
        handles.append(
            module.register_forward_pre_hook(
                lambda _module, inputs, key=name: destination.__setitem__(
                    key, inputs[0]
                )
            )
        )
    return handles


class AlexNetBoundaryDistillationLoss(LossFunction):
    """Combine matched-boundary feature loss, logit KL, and label loss."""

    def __init__(self, config: Mapping[str, Any] | Any):
        super().__init__(reduction="mean")
        cfg = _plain_mapping(config)
        self._loss_name = "AlexNet Boundary Distillation"
        self.teacher_backbone = str(cfg.get("teacher_backbone", "alexnet"))
        self.teacher_weights = str(cfg.get("teacher_weights", "IMAGENET1K_V1"))
        raw_weights = _plain_mapping(cfg.get("boundary_weights", {}))
        unknown = sorted(set(raw_weights) - set(_BOUNDARY_NAMES))
        if unknown:
            raise ValueError(f"Unknown vision-distillation boundaries: {unknown}")
        self.boundary_weights = {
            name: float(weight)
            for name, weight in raw_weights.items()
            if float(weight) > 0
        }
        self.boundary_input_mode = (
            str(cfg.get("boundary_input_mode", "composed")).strip().lower()
        )
        if self.boundary_input_mode not in {"composed", "teacher_forced", "mixed"}:
            raise ValueError(
                "vision distillation boundary_input_mode must be composed, "
                "teacher_forced, or mixed, "
                f"got {self.boundary_input_mode!r}"
            )
        self.teacher_forcing_ratio = float(cfg.get("teacher_forcing_ratio", 0.5))
        if not 0 <= self.teacher_forcing_ratio <= 1:
            raise ValueError("teacher_forcing_ratio must be between zero and one")
        if self.boundary_input_mode == "mixed" and self.teacher_forcing_ratio in {
            0.0,
            1.0,
        }:
            raise ValueError(
                "mixed boundary inputs require teacher_forcing_ratio strictly "
                "between zero and one"
            )
        self.feature_weight = float(cfg.get("feature_weight", 1.0))
        self.logit_kl_weight = float(cfg.get("logit_kl_weight", 1.0))
        self.supervised_weight = float(cfg.get("supervised_weight", 1.0))
        self.temperature = float(cfg.get("temperature", 2.0))
        self.relative_mse_epsilon = float(cfg.get("relative_mse_epsilon", 1e-6))
        self.class_output_target = (
            str(cfg.get("class_output_target", "raw")).strip().lower()
        )
        if self.class_output_target not in {
            "raw",
            "softmax_equivalent_nonnegative",
        }:
            raise ValueError(
                "class_output_target must be raw or " "softmax_equivalent_nonnegative"
            )
        if self.temperature <= 0:
            raise ValueError("vision distillation temperature must be positive")
        if self.relative_mse_epsilon <= 0:
            raise ValueError("relative_mse_epsilon must be positive")
        if (
            min(
                self.feature_weight,
                self.logit_kl_weight,
                self.supervised_weight,
            )
            < 0
        ):
            raise ValueError("vision distillation loss weights must be non-negative")
        if not (
            self.feature_weight > 0
            or self.logit_kl_weight > 0
            or self.supervised_weight > 0
        ):
            raise ValueError("At least one vision distillation loss must be enabled")
        self.teacher: nn.Module | None = None

    def _teacher_for(self, x: torch.Tensor) -> nn.Module:
        if self.teacher is None:
            teacher = get_model(self.teacher_backbone, weights=self.teacher_weights)
            teacher.eval()
            teacher.requires_grad_(False)
            self.teacher = teacher.to(device=x.device)
            logger.info(
                "Loaded frozen %s teacher (%s) for online boundary distillation",
                self.teacher_backbone,
                self.teacher_weights,
            )
        else:
            parameter = next(self.teacher.parameters(), None)
            if parameter is not None and parameter.device != x.device:
                self.teacher.to(device=x.device)
        return self.teacher

    @staticmethod
    def _teacher_boundary_modules(teacher: nn.Module) -> list[nn.Module]:
        features = getattr(teacher, "features", None)
        classifier = getattr(teacher, "classifier", None)
        if features is None or classifier is None:
            raise TypeError("AlexNet teacher must expose features and classifier")
        return [
            *(features[index] for index in _TEACHER_FEATURE_INDICES),
            *(classifier[index] for index in _TEACHER_CLASSIFIER_INDICES),
        ]

    @staticmethod
    def _teacher_boundary_input_modules(teacher: nn.Module) -> list[nn.Module]:
        features = getattr(teacher, "features", None)
        classifier = getattr(teacher, "classifier", None)
        if features is None or classifier is None:
            raise TypeError("AlexNet teacher must expose features and classifier")
        return [
            *(features[index] for index in _TEACHER_FEATURE_INPUT_INDICES),
            *(classifier[index] for index in _TEACHER_CLASSIFIER_INPUT_INDICES),
        ]

    @staticmethod
    def _student_boundary_modules(core: nn.Module) -> list[nn.Module]:
        spatial = getattr(core, "spatial_output_adapters", core.spatial_blocks)
        feedforward = getattr(
            core,
            "feedforward_output_adapters",
            core.feedforward_blocks,
        )
        return [*list(spatial), *list(feedforward)[:3]]

    @staticmethod
    def _student_boundary_input_modules(core: nn.Module) -> list[nn.Module]:
        return [*list(core.spatial_blocks), *list(core.feedforward_blocks)[:3]]

    @staticmethod
    def _conditioned_student_outputs(
        core: nn.Module,
        teacher_inputs: dict[str, torch.Tensor],
        *,
        teacher_forcing_ratio: float,
        composed_inputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        spatial_adapters = getattr(
            core,
            "spatial_output_adapters",
            [nn.Identity() for _ in range(5)],
        )
        spatial_reducers = getattr(
            core,
            "spatial_population_reducers",
            [nn.Identity() for _ in range(5)],
        )
        feedforward_adapters = getattr(
            core,
            "feedforward_output_adapters",
            [nn.Identity() for _ in range(3)],
        )
        feedforward_reducers = getattr(
            core,
            "feedforward_population_reducers",
            [nn.Identity() for _ in range(3)],
        )
        input_adapter = getattr(core, "input_adapter", nn.Identity())
        outputs: dict[str, torch.Tensor] = {}
        for index, name in enumerate(_BOUNDARY_NAMES[:5]):
            teacher_value = teacher_inputs[name]
            if index == 0:
                teacher_value = input_adapter(teacher_value)
            value = teacher_value
            if composed_inputs is not None:
                composed_value = composed_inputs[name]
                teacher_value = teacher_value.to(dtype=composed_value.dtype)
                if composed_value.shape != teacher_value.shape:
                    raise ValueError(
                        f"Boundary {name} input shape mismatch: composed "
                        f"{list(composed_value.shape)} versus teacher "
                        f"{list(teacher_value.shape)}"
                    )
                value = torch.lerp(
                    composed_value,
                    teacher_value,
                    teacher_forcing_ratio,
                )
            value = core.spatial_blocks[index](value)
            value = spatial_reducers[index](value)
            outputs[name] = spatial_adapters[index](value)
        for index, name in enumerate(_BOUNDARY_NAMES[5:]):
            teacher_value = teacher_inputs[name]
            value = teacher_value
            if composed_inputs is not None:
                composed_value = composed_inputs[name]
                teacher_value = teacher_value.to(dtype=composed_value.dtype)
                if composed_value.shape != teacher_value.shape:
                    raise ValueError(
                        f"Boundary {name} input shape mismatch: composed "
                        f"{list(composed_value.shape)} versus teacher "
                        f"{list(teacher_value.shape)}"
                    )
                value = torch.lerp(
                    composed_value,
                    teacher_value,
                    teacher_forcing_ratio,
                )
            value = core.feedforward_blocks[index](value)
            value = feedforward_reducers[index](value)
            outputs[name] = feedforward_adapters[index](value)
        return outputs

    def _feature_loss(
        self,
        student: dict[str, torch.Tensor],
        teacher: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        terms: list[torch.Tensor] = []
        weights: list[float] = []
        for name, weight in self.boundary_weights.items():
            student_value = student[name]
            teacher_value = teacher[name]
            if (
                name == "fc8"
                and self.class_output_target == "softmax_equivalent_nonnegative"
            ):
                teacher_value = teacher_value - teacher_value.amin(
                    dim=-1,
                    keepdim=True,
                )
            if student_value.shape != teacher_value.shape:
                raise ValueError(
                    f"Boundary {name} shape mismatch: student "
                    f"{list(student_value.shape)} versus teacher "
                    f"{list(teacher_value.shape)}"
                )
            scale = (
                teacher_value.detach()
                .square()
                .mean()
                .clamp_min(self.relative_mse_epsilon)
            )
            terms.append(F.mse_loss(student_value, teacher_value) / scale)
            weights.append(weight)
        if not terms:
            reference = next(iter(student.values()))
            return reference.new_zeros(())
        weighted = sum(
            weight * term for weight, term in zip(weights, terms, strict=True)
        )
        return weighted / sum(weights)

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is None:
            raise ValueError("vision distillation requires class labels")
        core = _student_core(model)
        teacher = self._teacher_for(x)
        teacher.eval()

        teacher_outputs: dict[str, torch.Tensor] = {}
        teacher_inputs: dict[str, torch.Tensor] = {}
        student_outputs: dict[str, torch.Tensor] = {}
        student_inputs: dict[str, torch.Tensor] = {}
        teacher_handles = _register_output_capture(
            self._teacher_boundary_modules(teacher),
            _BOUNDARY_NAMES,
            teacher_outputs,
        )
        teacher_input_handles: list[torch.utils.hooks.RemovableHandle] = []
        student_handles: list[torch.utils.hooks.RemovableHandle] = []
        if self.boundary_input_mode in {"teacher_forced", "mixed"}:
            teacher_input_handles = _register_input_capture(
                self._teacher_boundary_input_modules(teacher),
                _BOUNDARY_NAMES,
                teacher_inputs,
            )
            if self.boundary_input_mode == "mixed":
                student_handles = _register_input_capture(
                    self._student_boundary_input_modules(core),
                    _BOUNDARY_NAMES,
                    student_inputs,
                )
        else:
            student_handles = _register_output_capture(
                self._student_boundary_modules(core),
                _BOUNDARY_NAMES,
                student_outputs,
            )
        try:
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = model(x)
            if self.boundary_input_mode in {"teacher_forced", "mixed"}:
                student_outputs = self._conditioned_student_outputs(
                    core,
                    teacher_inputs,
                    teacher_forcing_ratio=(
                        1.0
                        if self.boundary_input_mode == "teacher_forced"
                        else self.teacher_forcing_ratio
                    ),
                    composed_inputs=(
                        student_inputs if self.boundary_input_mode == "mixed" else None
                    ),
                )
        finally:
            for handle in (
                *teacher_handles,
                *teacher_input_handles,
                *student_handles,
            ):
                handle.remove()

        total = student_logits.new_zeros(())
        if self.feature_weight > 0 and self.boundary_weights:
            total = total + self.feature_weight * self._feature_loss(
                student_outputs,
                teacher_outputs,
            )
        if self.logit_kl_weight > 0:
            temperature = self.temperature
            kl = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature**2)
            total = total + self.logit_kl_weight * kl
        if self.supervised_weight > 0:
            total = total + self.supervised_weight * F.cross_entropy(student_logits, y)
        if self.logit_kl_weight == 0 and self.supervised_weight == 0:
            # Teacher-forced single-boundary audits deliberately leave most
            # student blocks outside the feature objective. Keep every block
            # in the zero-valued autograd graph so ordinary DDP can complete
            # its reductions without changing the objective.
            total = total + student_logits.sum() * 0.0
        return total


class VisionDistillationTrainer(Trainer):
    """Standard DDP trainer with online AlexNet boundary supervision."""

    def __init__(self, vision_distillation_config: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.loss_function = AlexNetBoundaryDistillationLoss(
            vision_distillation_config or {}
        )
        self.filename_prefix = "vision_distillation_"


__all__ = [
    "AlexNetBoundaryDistillationLoss",
    "VisionDistillationTrainer",
]
