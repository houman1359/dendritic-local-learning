"""Isolated tensor-boundary primitives for gradual replacement training.

These primitives are not connected to a trainer, patcher, or deployment loader.
The caller must install both branches at the same mathematical boundary. In
particular, a collapsed span needs all original residual contributions and
post-FFN normalizations handled explicitly; wrapping only its exit is not enough.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn


def _require_step(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class InsertionSchedule:
    """Replacement fraction at an explicitly supplied optimizer-step boundary.

    ``alpha(start_step) == 0`` and ``alpha(end_step) == 1``. The caller owns step
    advancement; forward calls, evaluation, and gradient accumulation do not
    advance this schedule. At alpha zero, the replacement receives no gradient.
    """

    start_step: int
    end_step: int
    curve: str = "linear"

    def __post_init__(self) -> None:
        _require_step(self.start_step, name="start_step")
        _require_step(self.end_step, name="end_step")
        if self.end_step <= self.start_step:
            raise ValueError("end_step must be greater than start_step")
        if self.curve not in {"linear", "cosine"}:
            raise ValueError("curve must be 'linear' or 'cosine'")

    def alpha(self, step: int) -> float:
        step = _require_step(step, name="step")
        if step <= self.start_step:
            return 0.0
        if step >= self.end_step:
            return 1.0
        fraction = (step - self.start_step) / (self.end_step - self.start_step)
        if self.curve == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * fraction))
        return fraction


def _tensor_ownership(module: nn.Module) -> tuple[set[int], set[tuple[Any, ...]]]:
    identities: set[int] = set()
    storages: set[tuple[Any, ...]] = set()
    for tensor in (*module.parameters(), *module.buffers()):
        identities.add(id(tensor))
        if tensor.device.type != "meta" and tensor.numel():
            storages.add((tensor.device, tensor.untyped_storage().data_ptr()))
    return identities, storages


class GradualReplacement(nn.Module):
    """Blend a frozen teacher and trainable replacement at one tensor boundary.

    This retains both branches during training and is not a compressed export.
    Construct it before the optimizer. The teacher stays in evaluation mode and
    has no parameter gradients, but its input Jacobian remains in autograd so
    earlier trainable replacements receive the correct end-to-end gradient.

    Branches must be disjoint, nonmutating tensor maps. Their floating outputs
    must already match the input device and dtype; this wrapper never casts
    values or master weights. Output shapes must match each other when blended.
    At alpha zero/one only the active branch executes, preserving its exact
    output and avoiding unnecessary work and inactive-branch NaNs.
    """

    def __init__(
        self,
        teacher: nn.Module,
        replacement: nn.Module,
        *,
        schedule: InsertionSchedule,
        step: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(teacher, nn.Module) or not isinstance(replacement, nn.Module):
            raise TypeError("teacher and replacement must be nn.Module instances")
        if not isinstance(schedule, InsertionSchedule):
            raise TypeError("schedule must be an InsertionSchedule")
        _require_step(step, name="step")
        teacher_ids, teacher_storages = _tensor_ownership(teacher)
        replacement_ids, replacement_storages = _tensor_ownership(replacement)
        shared_modules = {id(module) for module in teacher.modules()} & {
            id(module) for module in replacement.modules()
        }
        if (
            shared_modules
            or teacher_ids & replacement_ids
            or teacher_storages & replacement_storages
        ):
            raise ValueError(
                "teacher and replacement must not share modules or tensors"
            )
        self.teacher = teacher
        self.replacement = replacement
        self._schedule = schedule
        self._step = step
        self._finalized = False
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.teacher.eval()

    @property
    def schedule(self) -> InsertionSchedule:
        return self._schedule

    @property
    def step(self) -> int:
        return self._step

    @property
    def alpha(self) -> float:
        return self.schedule.alpha(self.step)

    def set_step(self, step: int) -> None:
        """Set the global optimizer step explicitly, including on restart."""
        self._require_active()
        self._step = _require_step(step, name="step")

    def _require_active(self) -> None:
        if self._finalized:
            raise RuntimeError("gradual replacement has been finalized and released")

    def train(self, mode: bool = True) -> GradualReplacement:
        self._require_active()
        super().train(mode)
        self.teacher.eval()
        return self

    def get_extra_state(self) -> dict[str, Any]:
        self._require_active()
        return {
            "schema_version": 1,
            "schedule": asdict(self.schedule),
            "step": self.step,
        }

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self._require_active()
        if (
            not isinstance(state, dict)
            or set(state) != {"schema_version", "schedule", "step"}
            or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or not isinstance(state["schedule"], dict)
            or set(state["schedule"]) != {"start_step", "end_step", "curve"}
        ):
            raise ValueError("checkpoint gradual-insertion schedule does not match")
        restored_schedule = InsertionSchedule(**state["schedule"])
        if restored_schedule != self.schedule:
            raise ValueError("checkpoint gradual-insertion schedule does not match")
        self.set_step(state["step"])

    @staticmethod
    def _check_output(
        output: torch.Tensor, inputs: torch.Tensor, *, branch: str
    ) -> None:
        if not torch.is_tensor(output) or not output.is_floating_point():
            raise TypeError(f"{branch} must return a floating tensor")
        if output.device != inputs.device or output.dtype != inputs.dtype:
            raise ValueError(f"{branch} output must match the input device and dtype")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self._require_active()
        if not torch.is_tensor(inputs) or not inputs.is_floating_point():
            raise TypeError("gradual replacement requires a floating input tensor")
        alpha = self.alpha
        if alpha == 0.0:
            output = self.teacher(inputs)
            self._check_output(output, inputs, branch="teacher")
            return output
        if alpha == 1.0:
            output = self.replacement(inputs)
            self._check_output(output, inputs, branch="replacement")
            return output
        teacher_output = self.teacher(inputs)
        replacement_output = self.replacement(inputs)
        self._check_output(teacher_output, inputs, branch="teacher")
        self._check_output(replacement_output, inputs, branch="replacement")
        if teacher_output.shape != replacement_output.shape:
            raise ValueError("teacher and replacement output shapes must match")
        return (1.0 - alpha) * teacher_output + alpha * replacement_output

    def finalize(self) -> nn.Module:
        """Release the replacement only after alpha reaches one.

        The caller must replace this wrapper in its parent graph with the
        returned module and release any other teacher/optimizer references.
        This retired wrapper cannot be executed or checkpointed afterward.
        """
        self._require_active()
        if self.alpha != 1.0:
            raise RuntimeError("cannot finalize before replacement alpha reaches one")
        replacement = self.replacement
        del self.teacher
        del self.replacement
        self._finalized = True
        return replacement
