"""Frozen, teacher-shaped signed SwiGLU with compact resident sparse weights.

This explicit deployment format retains the teacher's neurons, selected signed
weights, and SiLU gate. It is the signed flat corner, not an E/I construction or
a converter for arbitrary compiled PopulationNetwork cells. Bitmasks remain
resident, as do BF16 values or packed unsigned INT4/INT8 affine codes. Forward
decodes a transient dense BF16 matrix; this is not a sparse-kernel speed claim.

The group-128 affine convention matches the measured physical frontier codec:
scales and offsets are BF16, and multiply and add each round to BF16. Groups
refer to original input columns, never to the compressed active-value stream.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from dendritic_modeling.networks.architectures.replacement.teacher_init import (
    copy_topk_weight_,
)

PROJECTION_FORMAT = "teacher_sparse_original_group128_bf16_affine/v1"
CELL_FORMAT = "teacher_shaped_signed_swiglu/v1"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _tensor_sha(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    header = json.dumps([str(tensor.dtype), list(tensor.shape)]).encode()
    return hashlib.sha256(
        header + tensor.view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def _json_copy(value: Any) -> Any:
    """Copy provenance without dropping unknown keys or normalizing tuples."""
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("metadata keys must be strings")
        for child in value.values():
            _json_copy(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _json_copy(child)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("metadata must contain only JSON-compatible provenance")
    json.dumps(value, allow_nan=False)
    return copy.deepcopy(value)


def _pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    if type(bits) is not int or bits not in (1, 4, 8):
        raise ValueError("only bitmask, INT4, and INT8 packing are supported")
    if codes.dtype != torch.uint8 or (codes.numel() and int(codes.max()) >= 1 << bits):
        raise ValueError("unsigned codes exceed their declared width")
    codes = codes.flatten()
    if bits == 8:
        return codes.clone()
    padded = F.pad(codes, (0, (-codes.numel()) % 8))
    shifts = torch.arange(8, device=codes.device, dtype=torch.int64) * bits
    words = (padded.reshape(-1, 8).long() << shifts).sum(-1)
    byte_shifts = torch.arange(bits, device=codes.device, dtype=torch.int64) * 8
    return ((words[:, None] >> byte_shifts) & 255).to(torch.uint8).flatten()


def _unpack(packed: torch.Tensor, bits: int, count: int) -> torch.Tensor:
    if type(bits) is not int or bits not in (1, 4, 8):
        raise ValueError("invalid packed width")
    if type(count) is not int or count < 0:
        raise ValueError("invalid code count")
    expected = count if bits == 8 else math.ceil(count / 8) * bits
    if packed.dtype != torch.uint8 or packed.ndim != 1 or packed.numel() != expected:
        raise ValueError("packed dtype or length differs from declared format")
    if bits == 8:
        return packed.clone()
    shifts = torch.arange(bits, device=packed.device, dtype=torch.int64) * 8
    words = (packed.reshape(-1, bits).long() << shifts).sum(-1)
    code_shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * bits
    result = ((words[:, None] >> code_shifts) & ((1 << bits) - 1)).byte().flatten()
    if result.numel() > count and bool(torch.count_nonzero(result[count:])):
        raise ValueError("nonzero padding is noncanonical")
    return result[:count]


class FrozenTeacherSparseLinear(nn.Module):
    """A frozen signed teacher projection, with no resident index expansion."""

    def __init__(
        self,
        shape,
        bits,
        active_count,
        packed_mask,
        values,
        scales=None,
        offsets=None,
    ):
        super().__init__()
        if (
            len(shape) != 2
            or any(type(n) is not int or n < 1 for n in shape)
            or shape[1] % 128
        ):
            raise ValueError(
                "positive dimensions and complete original 128-column groups required"
            )
        if bits is not None and (type(bits) is not int or bits not in (4, 8)):
            raise ValueError("BF16 (bits=None), INT4, or INT8 required")
        total = math.prod(shape)
        if type(active_count) is not int or not 0 <= active_count <= total:
            raise ValueError("invalid active count")
        mask = _unpack(packed_mask, 1, total)
        if int(mask.sum()) != active_count:
            raise ValueError("mask population differs from active count")
        if bits is None:
            if (
                values.dtype != torch.bfloat16
                or values.shape != (active_count,)
                or not bool(torch.isfinite(values).all())
                or scales is not None
                or offsets is not None
            ):
                raise ValueError(
                    "unquantized format requires finite active BF16 values only"
                )
        else:
            _unpack(values, bits, active_count)
            for field in (scales, offsets):
                if (
                    field is None
                    or field.dtype != torch.bfloat16
                    or field.shape != (total // 128,)
                    or not bool(torch.isfinite(field).all())
                ):
                    raise ValueError(
                        "finite BF16 scale and offset per original group required"
                    )
            if not bool((scales > 0).all()):
                raise ValueError("positive group scales required")
        tensors = (packed_mask, values, scales, offsets)
        if len({value.device for value in tensors if value is not None}) != 1:
            raise ValueError("all resident buffers must share one device")
        self.shape = tuple(shape)
        self.out_features, self.in_features = self.shape
        self.bits = bits
        self.active_count = active_count
        for name, value in zip(("packed_mask", "values", "scales", "offsets"), tensors):
            self.register_buffer(
                name, None if value is None else value.detach().clone()
            )
        super().train(False)

    def train(self, mode=True):
        if mode:
            raise RuntimeError("teacher sparse deployment is frozen inference only")
        return super().train(False)

    def support(self):
        return (
            _unpack(self.packed_mask, 1, math.prod(self.shape))
            .bool()
            .reshape(self.shape)
        )

    def dequantize(self):
        """Decode transient BF16 weights; pruned coordinates are exactly zero."""
        mask = self.support().flatten()
        total = math.prod(self.shape)
        if self.bits is None:
            if self.values.dtype != torch.bfloat16:
                raise RuntimeError("BF16 resident value dtype drifted")
            dense = torch.zeros(total, dtype=torch.bfloat16, device=self.values.device)
            dense[mask] = self.values
        else:
            if (
                self.scales.dtype != torch.bfloat16
                or self.offsets.dtype != torch.bfloat16
            ):
                raise RuntimeError("BF16 affine metadata dtype drifted")
            codes = torch.zeros(total, dtype=torch.uint8, device=self.values.device)
            codes[mask] = _unpack(self.values, self.bits, self.active_count)
            dense = (
                codes.reshape(-1, 128).to(torch.bfloat16) * self.scales[:, None]
                + self.offsets[:, None]
            ).flatten()
            dense.masked_fill_(~mask, 0)
        return dense.reshape(self.shape)

    def forward(self, inputs):
        if inputs.dtype != torch.bfloat16 or inputs.requires_grad:
            raise ValueError("frozen BF16 inputs without gradients required")
        if inputs.device != self.values.device or inputs.shape[-1] != self.in_features:
            raise ValueError("input device or final dimension differs")
        return F.linear(inputs, self.dequantize())

    @classmethod
    def from_weight(cls, weight, mask):
        """Copy exact teacher values on explicit support before quantization."""
        if (
            weight.ndim != 2
            or weight.dtype != torch.bfloat16
            or not bool(torch.isfinite(weight).all())
            or mask.dtype != torch.bool
            or mask.shape != weight.shape
            or mask.device != weight.device
        ):
            raise ValueError(
                "finite BF16 teacher matrix and matching boolean mask required"
            )
        retained = weight.detach()[mask]
        result = cls(
            list(weight.shape), None, retained.numel(), _pack(mask.byte(), 1), retained
        )
        if not torch.equal(result.dequantize(), weight.detach().masked_fill(~mask, 0)):
            raise RuntimeError("teacher support/value correspondence was not preserved")
        return result

    def quantized(self, bits):
        """Min/max RTN baseline, never a claim of calibrated quantization."""
        if self.bits is not None:
            raise ValueError(
                "quantize the original BF16 teacher-derived projection only"
            )
        if type(bits) is not int or bits not in (4, 8):
            raise ValueError("INT4 or INT8 required")
        grouped = self.dequantize().reshape(-1, 128).float()
        offsets = grouped.amin(-1).to(torch.bfloat16)
        scales = ((grouped.amax(-1) - grouped.amin(-1)) / ((1 << bits) - 1)).to(
            torch.bfloat16
        )
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        codes = (
            ((grouped - offsets.float()[:, None]) / scales.float()[:, None])
            .round()
            .clamp(0, (1 << bits) - 1)
            .byte()
            .flatten()
        )
        retained = codes[self.support().flatten()]
        return type(self)(
            list(self.shape),
            bits,
            self.active_count,
            self.packed_mask,
            _pack(retained, bits),
            scales,
            offsets,
        )

    def payload(self):
        return {
            "format": PROJECTION_FORMAT,
            "shape": list(self.shape),
            "bits": self.bits,
            "active_count": self.active_count,
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
        }

    @classmethod
    def from_payload(cls, payload):
        if (
            set(payload) != {"format", "shape", "bits", "active_count", "state_dict"}
            or payload["format"] != PROJECTION_FORMAT
        ):
            raise ValueError("unknown or incomplete projection payload")
        expected = {"packed_mask", "values"}
        if payload["bits"] is not None:
            expected |= {"scales", "offsets"}
        if set(payload["state_dict"]) != expected:
            raise ValueError("projection state key set differs")
        return cls(
            payload["shape"],
            payload["bits"],
            payload["active_count"],
            **payload["state_dict"],
        )

    def accounting(self):
        sizes = {
            name + "_bytes": value.numel() * value.element_size()
            for name, value in self.named_buffers()
        }
        return {
            **sizes,
            "registered_bytes": sum(sizes.values()),
            "logical_weights": math.prod(self.shape),
            "active_values": self.active_count,
            "transient_dense_weight_bytes": math.prod(self.shape) * 2,
            "workspace_scope": "Dense decoded weights only; excludes masks, codes, affine temporaries, GEMM workspace, activations, allocator reservation, and serialization overhead.",
        }


def _teacher_topk_mask(weight, fan_in, input_scales):
    if type(fan_in) is not int or not 1 <= fan_in <= weight.shape[1]:
        raise ValueError("fan-in must be an integer within the teacher input width")
    if input_scales is not None and (
        input_scales.ndim != 1
        or input_scales.numel() != weight.shape[1]
        or not bool(torch.isfinite(input_scales).all())
        or bool((input_scales < 0).any())
    ):
        raise ValueError(
            "activation scales must be finite, nonnegative, and input-shaped"
        )
    target = SimpleNamespace(
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        K=fan_in,
        weight_transform="identity",
        connection_indices=torch.empty(
            (weight.shape[0], fan_in), dtype=torch.int64, device=weight.device
        ),
        pre_w=torch.empty(
            (weight.shape[0], fan_in), dtype=weight.dtype, device=weight.device
        ),
    )
    copy_topk_weight_(weight, target, input_scales=input_scales)
    mask = torch.zeros_like(weight, dtype=torch.bool)
    mask.scatter_(1, target.connection_indices, True)
    if not torch.equal(weight.gather(1, target.connection_indices), target.pre_w):
        raise RuntimeError("existing TopK initializer changed teacher values")
    return mask


def _is_silu(activation):
    if type(activation) is nn.SiLU:
        return not activation.inplace
    if activation is F.silu:
        return True
    # Transformers is optional for the package. Only import its activation
    # class when the supplied teacher already uses that installed module.
    if type(activation).__module__ == "transformers.activations":
        from transformers.activations import SiLUActivation

        return type(activation) is SiLUActivation
    return False


class TeacherSparseSwiGLU(nn.Module):
    """Teacher-shaped signed gated cell; explicit format and explicit reload."""

    def __init__(self, gate_proj, up_proj, down_proj, *, metadata, initialization):
        super().__init__()
        projections = (gate_proj, up_proj, down_proj)
        if any(type(layer) is not FrozenTeacherSparseLinear for layer in projections):
            raise TypeError("frozen teacher sparse projections required")
        if gate_proj.shape != up_proj.shape or down_proj.shape != tuple(
            reversed(gate_proj.shape)
        ):
            raise ValueError("teacher SwiGLU projection dimensions do not compose")
        if len({layer.bits for layer in projections}) != 1:
            raise ValueError("one declared precision per cell required")
        if len({layer.values.device for layer in projections}) != 1:
            raise ValueError("cell projections must share one device")
        metadata = _json_copy(dict(metadata))
        if (
            "compiled_plan" in metadata
            and "compiled_replacement_plan" in metadata
            and metadata["compiled_plan"] != metadata["compiled_replacement_plan"]
        ):
            raise ValueError("conflicting compiled-plan aliases in provenance")
        for name, projection in zip(PROJECTIONS, projections):
            self.add_module(name, projection)
        self.metadata = metadata
        self.initialization = _json_copy(initialization)
        super().train(False)

    def train(self, mode=True):
        if mode:
            raise RuntimeError("teacher sparse deployment has no training path")
        return super().train(False)

    def forward(self, inputs):
        return self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))

    @classmethod
    @torch.no_grad()
    def from_teacher(
        cls,
        teacher,
        probe_inputs,
        *,
        fan_ins=None,
        masks=None,
        input_scales=None,
        bits=None,
        metadata=None,
    ):
        """Assert full-support parity first, then select exact teacher weights.

        Explicit masks permit an externally calibrated selection policy. TopK
        uses the existing initializer, including its device-dependent tie choice;
        the realized mask hash is recorded. Probe parity is scoped to supplied
        inputs and is not a replacement for a full-model deployment canary.
        """
        if teacher.training:
            raise ValueError("teacher must already be in eval mode")
        if fan_ins is not None and masks is not None:
            raise ValueError("provide fan-ins or explicit masks, not both")
        if masks is not None and input_scales is not None:
            raise ValueError("explicit masks already determine support")
        for supplied in (fan_ins, masks, input_scales):
            if supplied is not None and set(supplied) != set(PROJECTIONS):
                raise ValueError(
                    "provide exactly gate_proj, up_proj, and down_proj entries"
                )
        activation = getattr(teacher, "act_fn", None)
        if not _is_silu(activation):
            raise ValueError("explicit ordinary SiLU teacher activation required")
        weights = {}
        for name in PROJECTIONS:
            layer = getattr(teacher, name, None)
            if type(layer) is not nn.Linear or layer.bias is not None:
                raise ValueError(
                    "teacher must have three ordinary bias-free linear projections"
                )
            weights[name] = layer.weight.detach()
        if (
            probe_inputs.dtype != torch.bfloat16
            or probe_inputs.requires_grad
            or not probe_inputs.numel()
            or not bool(torch.isfinite(probe_inputs).all())
        ):
            raise ValueError("nonempty finite BF16 probe without gradients required")
        full = {
            name: FrozenTeacherSparseLinear.from_weight(
                weight, torch.ones_like(weight, dtype=torch.bool)
            )
            for name, weight in weights.items()
        }
        exact = cls(**full, metadata={}, initialization={})
        reference = teacher(probe_inputs)
        reproduced = exact(probe_inputs)
        if (
            reference.dtype != reproduced.dtype
            or reference.device != reproduced.device
            or not bool(torch.isfinite(reference).all())
            or not torch.equal(reference, reproduced)
        ):
            raise ValueError(
                "full-support teacher FFN identity failed before sparsification"
            )
        identity = {
            "full_support_weight_identity": True,
            "full_support_probe_identity": True,
            "probe_inputs_sha256": _tensor_sha(probe_inputs),
            "teacher_probe_output_sha256": _tensor_sha(reference),
            "teacher_weight_sha256": {
                name: _tensor_sha(weight) for name, weight in weights.items()
            },
            "training_updates": 0,
            "representation": "teacher-shaped signed flat SwiGLU; no E/I or branched claim",
            "probe_scope": "Exact supplied FFN probes before quantization; full-model reload qualification remains separate.",
        }
        selected = {}
        support_receipt = {}
        for name, weight in weights.items():
            if masks is not None:
                mask = masks[name]
                policy = "caller_supplied_support"
            else:
                fan_in = weight.shape[1] if fan_ins is None else fan_ins[name]
                scales = None if input_scales is None else input_scales[name]
                mask = _teacher_topk_mask(weight, fan_in, scales)
                policy = (
                    "teacher_absolute_topk"
                    if scales is None
                    else "teacher_absolute_times_input_rms_topk"
                )
            selected[name] = FrozenTeacherSparseLinear.from_weight(weight, mask)
            support_receipt[name] = {
                "policy": policy,
                "mask_sha256": _tensor_sha(mask),
                "active_teacher_values_sha256": _tensor_sha(weight[mask]),
                "active_count": int(mask.sum()),
                "input_scales_sha256": (
                    None
                    if input_scales is None or input_scales[name] is None
                    else _tensor_sha(input_scales[name])
                ),
            }
        identity["selection"] = support_receipt
        identity["quantization"] = "none"
        result = cls(
            **selected,
            metadata={} if metadata is None else metadata,
            initialization=identity,
        )
        return result if bits is None else result.quantized(bits)

    def quantized(self, bits):
        initialization = copy.deepcopy(self.initialization)
        initialization["quantization"] = (
            "original_group128_minmax_rtn_bf16_scale_offset"
        )
        return type(self)(
            **{name: getattr(self, name).quantized(bits) for name in PROJECTIONS},
            metadata=self.metadata,
            initialization=initialization,
        )

    def payload(self):
        payload = {
            "format": CELL_FORMAT,
            "projections": {
                name: getattr(self, name).payload() for name in PROJECTIONS
            },
            "metadata": _json_copy(self.metadata),
            "initialization": _json_copy(self.initialization),
        }
        payload["manifest_sha256"] = _payload_sha(payload)
        return payload

    @classmethod
    def from_payload(cls, payload, *, expected_metadata: Mapping[str, Any]):
        """Strict reload; caller must bind the complete expected provenance.

        The new format preserves both compiled-plan aliases and unknown metadata.
        Those plans are provenance, not authority to use the legacy compiled-cell
        loader for this distinct teacher-shaped module.
        """
        if (
            set(payload)
            != {
                "format",
                "projections",
                "metadata",
                "initialization",
                "manifest_sha256",
            }
            or payload["format"] != CELL_FORMAT
        ):
            raise ValueError("unknown or incomplete teacher sparse cell payload")
        if set(payload["projections"]) != set(PROJECTIONS):
            raise ValueError("projection names differ")
        if payload["manifest_sha256"] != _payload_sha(payload):
            raise ValueError("cell metadata or tensor integrity differs")
        if payload["metadata"] != _json_copy(dict(expected_metadata)):
            raise ValueError(
                "expected export metadata, including compiled plans, differs"
            )
        return cls(
            **{
                name: FrozenTeacherSparseLinear.from_payload(
                    payload["projections"][name]
                )
                for name in PROJECTIONS
            },
            metadata=payload["metadata"],
            initialization=payload["initialization"],
        )

    def accounting(self):
        projections = {name: getattr(self, name).accounting() for name in PROJECTIONS}
        return {
            "projections": projections,
            "registered_bytes": sum(
                item["registered_bytes"] for item in projections.values()
            ),
            "file_bytes_scope": "Use the actual serialized file size; registered tensors exclude manifest and container overhead.",
            "runtime": "Frozen eager BF16 dense dequantization, with no persistent dense weight cache or speed claim.",
        }


def _payload_sha(payload):
    manifest = {
        key: value
        for key, value in payload.items()
        if key not in {"projections", "manifest_sha256"}
    }
    manifest["projections"] = {
        name: {
            **{key: value for key, value in projection.items() if key != "state_dict"},
            "state_dict": {
                key: _tensor_sha(value)
                for key, value in projection["state_dict"].items()
            },
        }
        for name, projection in payload["projections"].items()
    }
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
