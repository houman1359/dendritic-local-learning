"""Capture and profile real teacher-module boundaries for FMI.

The profiler operates on two-dimensional examples. This module supplies the
model-facing bridge: it samples aligned module inputs and outputs from ordinary
teacher forwards, uniformly over every observed token or spatial position, and
retains the differentiable teacher module needed by gradient-based estimators.
No replacement implementation lives here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing_extensions import Self

from dendritic_modeling.analysis.fmi.profiler import (
    FMIProfilerConfig,
    profile_teacher_component,
)


def _extract_tensor(value: Any, *, name: str) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"{name} must be a Tensor or start with a Tensor")


def _conv_patch_rows(
    module: nn.Conv2d,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if module.groups != 1:
        raise ValueError("FMI Conv2d capture currently requires groups=1")
    patches = F.unfold(
        inputs,
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )
    input_rows = patches.transpose(1, 2).reshape(-1, patches.shape[1])
    output_rows = outputs.flatten(2).transpose(1, 2).reshape(-1, outputs.shape[1])
    if input_rows.shape[0] != output_rows.shape[0]:
        raise RuntimeError("Conv2d patch and output positions do not align")
    return input_rows, output_rows


def flatten_module_examples(
    module: nn.Module,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert aligned module I/O into ``[examples, features]`` rows."""

    if isinstance(module, nn.Conv2d):
        return _conv_patch_rows(module, inputs, outputs)
    if inputs.ndim < 2 or outputs.ndim < 2:
        raise ValueError("teacher boundary tensors must have a feature dimension")
    input_rows = inputs.reshape(-1, inputs.shape[-1])
    output_rows = outputs.reshape(-1, outputs.shape[-1])
    if input_rows.shape[0] != output_rows.shape[0]:
        raise ValueError(
            "generic boundary input/output leading dimensions do not align; "
            "target a complete module with one output per input position"
        )
    return input_rows, output_rows


@dataclass(frozen=True)
class TeacherBoundaryTarget:
    """One named, differentiable teacher component to fingerprint."""

    target_id: str
    module: nn.Module
    module_path: str
    layer_index: int | None = None
    input_module: nn.Module | None = None
    output_module: nn.Module | None = None

    def capture_input_module(self) -> nn.Module:
        return self.module if self.input_module is None else self.input_module

    def capture_output_module(self) -> nn.Module:
        return self.module if self.output_module is None else self.output_module


@dataclass(frozen=True)
class CapturedTeacherBoundary:
    """Bounded uniform sample of one teacher boundary."""

    target: TeacherBoundaryTarget
    inputs: torch.Tensor
    outputs: torch.Tensor
    rows_seen: int
    row_group_ids: torch.Tensor | None = None

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "target_id": self.target.target_id,
            "module_path": self.target.module_path,
            "layer_index": self.target.layer_index,
            "module_class": self.target.module.__class__.__qualname__,
            "rows_seen": int(self.rows_seen),
            "rows_profiled": int(self.inputs.shape[0]),
            "input_dim": int(self.inputs.shape[1]),
            "output_dim": int(self.outputs.shape[1]),
        }
        if self.row_group_ids is None:
            metadata.update(
                {
                    "row_group_ids_present": False,
                    "row_group_semantics": None,
                    "row_group_count": None,
                }
            )
        else:
            if self.row_group_ids.ndim != 1 or self.row_group_ids.shape[0] != len(
                self.inputs
            ):
                raise ValueError(
                    "row_group_ids must align one-to-one with captured rows"
                )
            metadata.update(
                {
                    "row_group_ids_present": True,
                    "row_group_semantics": "teacher_forward_batch",
                    "row_group_count": int(self.row_group_ids.unique().numel()),
                }
            )
        return metadata


class ViTFFNTeacher(nn.Module):
    """Differentiable whole-FFN view of a split Hugging Face ViT block."""

    def __init__(self, intermediate: nn.Module, output_dense: nn.Module) -> None:
        super().__init__()
        self.intermediate = intermediate
        self.output_dense = output_dense

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output_dense(self.intermediate(inputs))


def transformer_boundary_targets(
    model: nn.Module,
    layer_indices: Iterable[int],
    *,
    layers_attr: str | None,
    target_module: str,
    model_family: str = "causal_lm",
) -> list[TeacherBoundaryTarget]:
    """Resolve causal-LM, DINO, or split-ViT FFN targets without patching."""

    from dendritic_modeling.networks.architectures.transformer import (
        resolve_transformer_layers,
    )
    from dendritic_modeling.networks.architectures.transformer.utils import (
        _get_attr_path,
    )

    layers = resolve_transformer_layers(model, layers_attr=layers_attr)
    family = str(model_family).strip().lower()
    targets = []
    for layer_index_raw in layer_indices:
        layer_index = int(layer_index_raw)
        layer = layers[layer_index]
        if family == "vit":
            intermediate = getattr(layer, "intermediate", None)
            output = getattr(layer, "output", None)
            output_dense = getattr(output, "dense", None)
            if not isinstance(intermediate, nn.Module) or not isinstance(
                output_dense, nn.Module
            ):
                raise ValueError(
                    f"ViT layer {layer_index} lacks intermediate/output.dense"
                )
            module = ViTFFNTeacher(intermediate, output_dense)
            targets.append(
                TeacherBoundaryTarget(
                    target_id=f"teacher_L{layer_index}",
                    module=module,
                    module_path=f"layer[{layer_index}].intermediate+output.dense",
                    layer_index=layer_index,
                    input_module=intermediate,
                    output_module=output_dense,
                )
            )
            continue
        module = _get_attr_path(layer, target_module)
        if not isinstance(module, nn.Module):
            raise TypeError(
                f"layer {layer_index} target {target_module!r} is not an nn.Module"
            )
        targets.append(
            TeacherBoundaryTarget(
                target_id=f"teacher_L{layer_index}",
                module=module,
                module_path=f"layer[{layer_index}].{target_module}",
                layer_index=layer_index,
            )
        )
    return targets


class TeacherBoundaryCollector:
    """Uniformly sample aligned boundary rows through forward hooks."""

    def __init__(
        self,
        targets: Iterable[TeacherBoundaryTarget],
        *,
        max_examples: int,
        seed: int = 0,
    ) -> None:
        self.targets = list(targets)
        if not self.targets:
            raise ValueError("at least one teacher boundary target is required")
        if len({target.target_id for target in self.targets}) != len(self.targets):
            raise ValueError("teacher boundary target ids must be unique")
        if int(max_examples) < 4:
            raise ValueError("max_examples must be at least four")
        self.max_examples = int(max_examples)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self._pending: dict[str, torch.Tensor] = {}
        self._keys: dict[str, torch.Tensor] = {}
        self._inputs: dict[str, torch.Tensor] = {}
        self._outputs: dict[str, torch.Tensor] = {}
        self._group_ids: dict[str, torch.Tensor] = {}
        self._rows_seen = {target.target_id: 0 for target in self.targets}
        self._implicit_group_ids = {target.target_id: 0 for target in self.targets}
        self._active_group_id: int | None = None
        self._handles: list[Any] = []

    def __enter__(self) -> Self:
        for target in self.targets:
            self._handles.append(
                target.capture_input_module().register_forward_pre_hook(
                    self._pre_hook(target)
                )
            )
            self._handles.append(
                target.capture_output_module().register_forward_hook(
                    self._forward_hook(target)
                )
            )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending.clear()

    def begin_source_group(self, group_id: int) -> None:
        """Label subsequent captured rows with one source-forward identity.

        Group labels make it possible to keep every token or spatial position
        from a calibration forward wholly on one side of a later validation
        split. The integer is provenance only; it is never an input feature.
        """

        self._active_group_id = int(group_id)

    def _pre_hook(self, target: TeacherBoundaryTarget):
        def hook(module: nn.Module, args: tuple[Any, ...]) -> None:
            del module
            if not args:
                raise ValueError(f"target {target.target_id!r} received no input")
            self._pending[target.target_id] = _extract_tensor(
                args[0], name=f"input of {target.target_id}"
            ).detach()

        return hook

    def _forward_hook(self, target: TeacherBoundaryTarget):
        def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            del args
            if target.target_id not in self._pending:
                raise RuntimeError(f"target {target.target_id!r} has no captured input")
            inputs = self._pending.pop(target.target_id)
            outputs = _extract_tensor(output, name=f"output of {target.target_id}")
            del module
            input_rows, output_rows = flatten_module_examples(
                target.module, inputs, outputs
            )
            self._update(target.target_id, input_rows, output_rows)

        return hook

    def _update(
        self,
        target_id: str,
        input_rows: torch.Tensor,
        output_rows: torch.Tensor,
    ) -> None:
        input_rows = input_rows.detach().float().cpu()
        output_rows = output_rows.detach().float().cpu()
        count = int(input_rows.shape[0])
        self._rows_seen[target_id] += count
        if self._active_group_id is None:
            group_id = self._implicit_group_ids[target_id]
            self._implicit_group_ids[target_id] += 1
        else:
            group_id = self._active_group_id
        group_ids = torch.full((count,), group_id, dtype=torch.int64)
        keys = torch.rand(count, generator=self.generator)
        if target_id in self._keys:
            keys = torch.cat([self._keys[target_id], keys])
            input_rows = torch.cat([self._inputs[target_id], input_rows])
            output_rows = torch.cat([self._outputs[target_id], output_rows])
            group_ids = torch.cat([self._group_ids[target_id], group_ids])
        keep = min(self.max_examples, int(keys.numel()))
        selected = torch.topk(keys, keep, largest=False).indices
        self._keys[target_id] = keys[selected]
        self._inputs[target_id] = input_rows[selected]
        self._outputs[target_id] = output_rows[selected]
        self._group_ids[target_id] = group_ids[selected]

    def captured(self) -> dict[str, CapturedTeacherBoundary]:
        missing = [
            target.target_id
            for target in self.targets
            if target.target_id not in self._inputs
        ]
        if missing:
            raise RuntimeError(f"teacher forwards never reached targets {missing}")
        return {
            target.target_id: CapturedTeacherBoundary(
                target=target,
                inputs=self._inputs[target.target_id],
                outputs=self._outputs[target.target_id],
                rows_seen=self._rows_seen[target.target_id],
                row_group_ids=self._group_ids[target.target_id],
            )
            for target in self.targets
        }


def capture_teacher_boundaries(
    model: nn.Module,
    targets: Iterable[TeacherBoundaryTarget],
    batches: Iterable[Any],
    *,
    forward_fn: Callable[[nn.Module, Any], Any] | None = None,
    max_examples: int = 512,
    seed: int = 0,
) -> dict[str, CapturedTeacherBoundary]:
    """Run the frozen teacher and sample every declared component boundary."""

    runner = forward_fn or (lambda current_model, batch: current_model(batch))
    model.eval()
    with TeacherBoundaryCollector(
        targets, max_examples=max_examples, seed=seed
    ) as collector:
        with torch.no_grad():
            for group_id, batch in enumerate(batches):
                collector.begin_source_group(group_id)
                runner(model, batch)
    return collector.captured()


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


def differentiable_boundary_function(
    module: nn.Module,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Adapt a captured module to the profiler's two-dimensional contract."""

    device, dtype = _module_device_dtype(module)
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            raise ValueError("FMI Conv2d profiling currently requires groups=1")

        def conv_function(rows: torch.Tensor) -> torch.Tensor:
            weights = module.weight.reshape(module.out_channels, -1)
            return F.linear(
                rows.to(device=device, dtype=dtype), weights, module.bias
            ).float()

        return conv_function

    def module_function(rows: torch.Tensor) -> torch.Tensor:
        output = module(rows.to(device=device, dtype=dtype))
        return _extract_tensor(output, name="teacher function output").float()

    return module_function


def profile_captured_boundaries(
    captured: Mapping[str, CapturedTeacherBoundary],
    *,
    config: FMIProfilerConfig | None = None,
    task_weights: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, dict[str, Any]]:
    """Profile captured boundaries and return artifact-ready target entries."""

    entries: dict[str, dict[str, Any]] = {}
    for target_id, boundary in captured.items():
        device, _ = _module_device_dtype(boundary.target.module)
        fingerprint = profile_teacher_component(
            differentiable_boundary_function(boundary.target.module),
            boundary.inputs.to(device=device),
            outputs=boundary.outputs.to(device=device),
            task_weight=(
                None
                if task_weights is None or target_id not in task_weights
                else task_weights[target_id].to(device=device)
            ),
            config=config,
        )
        entries[target_id] = {
            "boundary": boundary.metadata(),
            "fingerprint": fingerprint,
        }
    return entries


def profiler_config_dict(config: FMIProfilerConfig) -> dict[str, Any]:
    """Return the exact JSON-safe profiler configuration."""

    return asdict(config)


__all__ = [
    "CapturedTeacherBoundary",
    "TeacherBoundaryCollector",
    "TeacherBoundaryTarget",
    "capture_teacher_boundaries",
    "differentiable_boundary_function",
    "flatten_module_examples",
    "profile_captured_boundaries",
    "profiler_config_dict",
    "transformer_boundary_targets",
]
