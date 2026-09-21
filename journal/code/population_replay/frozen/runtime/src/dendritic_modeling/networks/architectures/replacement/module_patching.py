"""Model-agnostic placement of compiled PopulationNetwork replacements.

This path covers vector-valued ``nn.Linear`` boundaries, 1x1 convolutional
channel maps, and general KxK convolutions inside graphs that cannot be
flattened into a simple encoder/core/decoder sequence (notably ResNet
residual blocks and classical CNN trunks).  A KxK convolution is exactly a
vector map on unfolded patches -- its weight ``(out, in, kh, kw)`` reshaped
to ``(out, in*kh*kw)`` is the dense teacher matrix of that map -- so the same
compiled cell, teacher-conditioned initialization, and export contract apply
per spatial location without any convolution-specific cell family.  It contains
no model-family-specific training logic: callers name module paths, and the
same FMI/manual selector and compiled boundary cell used by transformers and
classical vision models is installed at each path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.networks.architectures.replacement.cells import (
    RUNTIME_TENSOR_CONTRACT_SCHEMA,
    preserve_runtime_tensor_contract,
    require_runtime_tensor_contract,
)


def _get_module_path(root: nn.Module, path: str) -> nn.Module:
    current: Any = root
    for part in str(path).split("."):
        if isinstance(current, (nn.Sequential, nn.ModuleList)) and part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise TypeError(f"module path {path!r} did not resolve to nn.Module")
    return current


def _set_module_path(root: nn.Module, path: str, value: nn.Module) -> None:
    parts = str(path).split(".")
    parent: Any = root
    for part in parts[:-1]:
        if isinstance(parent, (nn.Sequential, nn.ModuleList)) and part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    leaf = parts[-1]
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and leaf.isdigit():
        parent[int(leaf)] = value
    else:
        setattr(parent, leaf, value)


class ChannelMapReplacement(nn.Module):
    """Apply a vector replacement independently at every spatial location."""

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(
        self,
        replacement: nn.Module,
        *,
        input_channels: int,
        output_channels: int,
        stride: tuple[int, int] = (1, 1),
    ):
        super().__init__()
        self.replacement = replacement
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.stride = (int(stride[0]), int(stride[1]))
        if min(*self.stride, self.input_channels, self.output_channels) < 1:
            raise ValueError("channel-map dimensions and stride must be positive")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.input_channels:
            raise ValueError(
                "ChannelMapReplacement expects BCHW input with "
                f"C={self.input_channels}, got {tuple(inputs.shape)}"
            )
        if self.stride != (1, 1):
            inputs = inputs[:, :, :: self.stride[0], :: self.stride[1]]
        batch, channels, height, width = inputs.shape
        flat = inputs.permute(0, 2, 3, 1).reshape(-1, channels)
        output = self.replacement(flat)
        result = (
            output.reshape(batch, height, width, self.output_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        return preserve_runtime_tensor_contract(
            result,
            inputs,
            boundary=type(self).__name__,
        )

    def apply_rewiring(self) -> None:
        rewire = getattr(self.replacement, "apply_rewiring", None)
        if callable(rewire):
            rewire()

    def parameter_estimate(self) -> dict[str, int]:
        estimate = getattr(self.replacement, "parameter_estimate", None)
        if callable(estimate):
            return estimate()
        stored = sum(parameter.numel() for parameter in self.parameters())
        return {"stored_total": int(stored), "active_total": int(stored)}


class PatchMapReplacement(nn.Module):
    """Apply a vector replacement to every unfolded KxK convolution patch.

    ``F.unfold`` materializes the ``in_channels * kh * kw`` patch vector at
    each output location; the compiled cell maps it to ``out_channels``; the
    result folds back to BCHW with the original convolution's output
    geometry.  The teacher's bias, when present, is carried as a trainable
    per-channel parameter initialized from the original convolution rather
    than being silently dropped.
    """

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(
        self,
        replacement: nn.Module,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        bias: torch.Tensor | None = None,
    ):
        super().__init__()
        self.replacement = replacement
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        self.stride = (int(stride[0]), int(stride[1]))
        self.padding = (int(padding[0]), int(padding[1]))
        self.dilation = (int(dilation[0]), int(dilation[1]))
        if (
            min(
                *self.kernel_size,
                *self.stride,
                *self.dilation,
                self.in_channels,
                self.out_channels,
            )
            < 1
            or min(self.padding) < 0
        ):
            raise ValueError("patch-map geometry must be positive")
        self.patch_dim = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        if bias is not None:
            if bias.shape != (self.out_channels,):
                raise ValueError(
                    f"bias must have shape ({self.out_channels},), "
                    f"got {tuple(bias.shape)}"
                )
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.bias = None

    def _output_extent(self, size: int, axis: int) -> int:
        effective = self.dilation[axis] * (self.kernel_size[axis] - 1) + 1
        return (size + 2 * self.padding[axis] - effective) // self.stride[axis] + 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(
                "PatchMapReplacement expects BCHW input with "
                f"C={self.in_channels}, got {tuple(inputs.shape)}"
            )
        batch, _, height, width = inputs.shape
        out_height = self._output_extent(height, 0)
        out_width = self._output_extent(width, 1)
        if min(out_height, out_width) < 1:
            raise ValueError(
                f"input {height}x{width} is smaller than one "
                f"{self.kernel_size} patch at padding {self.padding}"
            )
        patches = F.unfold(
            inputs,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride,
        )
        locations = patches.shape[-1]
        flat = patches.transpose(1, 2).reshape(-1, self.patch_dim)
        output = self.replacement(flat)
        output = output.reshape(batch, locations, self.out_channels)
        if self.bias is not None:
            output = output + self.bias
        result = (
            output.transpose(1, 2)
            .reshape(batch, self.out_channels, out_height, out_width)
            .contiguous()
        )
        return preserve_runtime_tensor_contract(
            result,
            inputs,
            boundary=type(self).__name__,
        )

    def apply_rewiring(self) -> None:
        rewire = getattr(self.replacement, "apply_rewiring", None)
        if callable(rewire):
            rewire()

    def parameter_estimate(self) -> dict[str, int]:
        estimate = getattr(self.replacement, "parameter_estimate", None)
        bias_count = 0 if self.bias is None else int(self.bias.numel())
        if callable(estimate):
            inner = dict(estimate())
            inner["stored_total"] = int(inner.get("stored_total", 0)) + bias_count
            inner["active_total"] = int(inner.get("active_total", 0)) + bias_count
            return inner
        stored = sum(parameter.numel() for parameter in self.parameters())
        return {"stored_total": int(stored), "active_total": int(stored)}


@dataclass(frozen=True)
class SelectedModuleReplacementRecord:
    """One atomic module-path replacement and its compiled provenance."""

    path: str
    original: nn.Module
    replacement: nn.Module
    selection_manifest: dict[str, Any]
    compiled_plan: dict[str, Any]


def _module_boundary(module: nn.Module) -> tuple[int, int, str]:
    if isinstance(module, nn.Linear):
        return int(module.in_features), int(module.out_features), "linear"
    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            raise ValueError(
                "convolution replacement requires groups=1; a grouped or "
                "depthwise convolution is not one dense channel map"
            )
        if module.padding_mode != "zeros":
            raise ValueError(
                "convolution replacement requires padding_mode='zeros'; "
                f"got {module.padding_mode!r}"
            )
        if isinstance(module.padding, str):
            raise ValueError(
                "convolution replacement requires explicit integer padding; "
                "'same'/'valid' string padding is not supported"
            )
        if (
            module.kernel_size == (1, 1)
            and module.padding == (0, 0)
            and module.dilation == (1, 1)
        ):
            return int(module.in_channels), int(module.out_channels), "conv1x1"
        kh, kw = (int(module.kernel_size[0]), int(module.kernel_size[1]))
        return (
            int(module.in_channels) * kh * kw,
            int(module.out_channels),
            ("conv_patch"),
        )
    raise TypeError(
        "selected module replacement supports nn.Linear and nn.Conv2d, "
        f"got {type(module).__name__}"
    )


def replace_modules_with_selected_population_networks(
    model: nn.Module,
    targets: Sequence[str | Mapping[str, Any]],
    *,
    selection: Mapping[str, Any],
    preserve_device_dtype: bool = True,
) -> list[SelectedModuleReplacementRecord]:
    """Compile and atomically replace named linear/channel-map modules.

    A target mapping may override ``selection`` and may provide
    ``layer_index`` for ``{layer}`` fingerprint keys. Every cell is built and
    validated before the model is mutated.
    """

    from dendritic_modeling.networks.architectures.replacement.selection import (
        resolve_replacement_selection,
    )
    from dendritic_modeling.networks.architectures.transformer.patching import (
        build_compiled_population_replacement,
    )

    prepared = []
    seen: set[str] = set()
    for position, target in enumerate(targets):
        entry = {"path": target} if isinstance(target, str) else dict(target)
        path = str(entry.get("path", ""))
        if not path or path in seen:
            raise ValueError("replacement target paths must be non-empty and unique")
        seen.add(path)
        original = _get_module_path(model, path)
        input_dim, output_dim, kind = _module_boundary(original)
        target_selection = dict(entry.get("selection", selection))
        layer_index = int(entry.get("layer_index", position))
        resolved = resolve_replacement_selection(
            target_selection,
            hidden_size=input_dim,
            teacher_intermediate_size=output_dim,
            layer_index=layer_index,
            module_path=path,
            boundary_output_dim=output_dim,
        )
        replacement = build_compiled_population_replacement(
            resolved,
            hidden_size=input_dim,
            teacher_intermediate_size=output_dim,
            output_size=output_dim,
            transformer_replacement={"selection": target_selection},
        )
        if kind == "conv1x1":
            assert isinstance(original, nn.Conv2d)
            if original.bias is not None:
                # A 1x1 map with bias is still a KxK patch map with a
                # kernel of one: route it through the bias-carrying wrapper
                # instead of silently dropping the teacher's bias.
                replacement = PatchMapReplacement(
                    replacement,
                    in_channels=input_dim,
                    out_channels=output_dim,
                    kernel_size=(1, 1),
                    stride=original.stride,
                    bias=original.bias,
                )
            else:
                replacement = ChannelMapReplacement(
                    replacement,
                    input_channels=input_dim,
                    output_channels=output_dim,
                    stride=original.stride,
                )
        elif kind == "conv_patch":
            assert isinstance(original, nn.Conv2d)
            replacement = PatchMapReplacement(
                replacement,
                in_channels=int(original.in_channels),
                out_channels=output_dim,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                dilation=original.dilation,
                bias=original.bias,
            )
        if preserve_device_dtype:
            parameter = next(original.parameters(), None)
            if parameter is not None:
                dtype = parameter.dtype if parameter.dtype.is_floating_point else None
                replacement = replacement.to(device=parameter.device, dtype=dtype)
        require_runtime_tensor_contract(
            replacement,
            boundary=f"selected module {path!r}",
        )
        prepared.append((path, original, replacement, resolved))

    records = []
    for path, original, replacement, resolved in prepared:
        _set_module_path(model, path, replacement)
        records.append(
            SelectedModuleReplacementRecord(
                path=path,
                original=original,
                replacement=replacement,
                selection_manifest=resolved.manifest,
                compiled_plan=resolved.plan.as_dict(),
            )
        )
    return records


def rebuild_replacement_from_export(
    original: nn.Module,
    payload: Mapping[str, Any],
    *,
    layer_index: int = 0,
) -> nn.Module:
    """Rebuild a driver-exported trained cell against its original module.

    The screen drivers export every replacement as ``{state_dict, target,
    compiled_plan}``; frozen re-evaluation and composed-model admissibility
    gates must reconstruct the exact trained module from that file alone.
    The architecture is recompiled from the saved plan's own axes, the
    freshly realized plan must equal the saved plan verbatim (fail closed on
    any compiler drift), and the state dict is loaded strictly -- topology
    indices are persisted buffers, so the trained wiring is restored
    bitwise regardless of construction seeds.
    """

    import json as _json

    plan = dict(payload["compiled_plan"])
    family = str(plan["family"])
    manual_axes = {
        "biological_neuron": "positive" in family,
        "explicit_ei": "_ei" in family,
        "gated": family.startswith("gated_"),
        "density": float(plan["density"]),
        "population_width": int(plan["population_width"]),
        "branch_factors": list(plan["branch_factors"]),
        "integration_rule": (
            "shunting" if family.endswith("shunting") else "raw_additive"
        ),
    }
    selection = {
        "enabled": True,
        "mode": "manual",
        "allow_non_biological": not manual_axes["biological_neuron"],
        "manual": manual_axes,
    }
    host = nn.Module()
    host.add_module("site", original)
    records = replace_modules_with_selected_population_networks(
        host,
        [{"path": "site", "layer_index": int(layer_index)}],
        selection=selection,
    )
    record = records[0]

    def _normalize(value: Any) -> Any:
        return _json.loads(_json.dumps(value, default=str, sort_keys=True))

    realized = _normalize(record.compiled_plan)
    saved = _normalize(plan)
    for volatile in ("requested_candidate",):
        realized.pop(volatile, None)
        saved.pop(volatile, None)
    if realized != saved:
        drifted = [
            key
            for key in set(realized) | set(saved)
            if realized.get(key) != saved.get(key)
        ]
        raise ValueError(
            "exported cell rebuild drifted from the saved compiled plan at "
            f"{payload.get('target', '?')}: differing keys {sorted(drifted)}"
        )
    record.replacement.load_state_dict(payload["state_dict"], strict=True)
    record.replacement.eval()
    for parameter in record.replacement.parameters():
        parameter.requires_grad_(False)
    return record.replacement


def restore_selected_module_replacements(
    model: nn.Module,
    records: Sequence[SelectedModuleReplacementRecord],
) -> None:
    """Reinstall every record's original module at its path.

    Fails closed: each path must currently hold the exact replacement object
    the record installed.  A mismatch means the model was mutated again after
    patching (a second replacement, a wrapper, or an already-run restore), and
    silently overwriting whatever sits there would corrupt that later state.
    """

    for record in records:
        current = _get_module_path(model, record.path)
        if current is not record.replacement:
            raise RuntimeError(
                f"cannot restore {record.path!r}: the module currently "
                f"installed ({type(current).__name__}) is not this record's "
                "replacement -- the model was mutated after patching or this "
                "record was already restored"
            )
    for record in records:
        _set_module_path(model, record.path, record.original)


__all__ = [
    "ChannelMapReplacement",
    "PatchMapReplacement",
    "SelectedModuleReplacementRecord",
    "rebuild_replacement_from_export",
    "replace_modules_with_selected_population_networks",
    "restore_selected_module_replacements",
]
