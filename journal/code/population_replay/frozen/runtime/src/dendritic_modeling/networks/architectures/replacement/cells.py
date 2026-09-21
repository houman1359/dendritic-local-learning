"""Reusable replacement-cell contracts."""

from __future__ import annotations

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.replacement.adapters import (
    NonNegativeInputAdapter,
    validate_input_transform,
)

RUNTIME_TENSOR_CONTRACT_SCHEMA = "dendritic_replacement_runtime_tensor_contract/v1"


def preserve_runtime_tensor_contract(
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    boundary: str,
) -> torch.Tensor:
    """Return a replacement output in its surrounding tensor contract.

    Replacement parameters may deliberately remain FP32 master values inside a
    BF16/FP16 model.  Internal E/I integration can therefore promote an
    activation even when an indexed projection happens to preserve its input
    dtype.  A replacement boundary is nevertheless substituting for the dense
    module it removed, so its floating output must re-enter the surrounding
    graph on the input activation's device and in its runtime dtype.

    The post-conversion checks are intentional fail-closed assertions: callers
    cannot silently pass a non-floating tensor or an unfulfilled device/dtype
    contract downstream to attention, normalization, or a residual add.
    ``Tensor.to`` remains in the autograd graph, so FP32 master gradients are
    retained through the lower-precision boundary.
    """

    if not torch.is_tensor(reference) or not reference.is_floating_point():
        raise TypeError(f"{boundary} requires a floating reference tensor")
    if not torch.is_tensor(output) or not output.is_floating_point():
        raise TypeError(f"{boundary} must return a floating tensor")
    if output.device != reference.device or output.dtype != reference.dtype:
        output = output.to(device=reference.device, dtype=reference.dtype)
    if output.device != reference.device or output.dtype != reference.dtype:
        raise RuntimeError(
            f"{boundary} violated {RUNTIME_TENSOR_CONTRACT_SCHEMA}: "
            f"output=({output.device}, {output.dtype}) "
            f"reference=({reference.device}, {reference.dtype})"
        )
    return output


def require_runtime_tensor_contract(module: nn.Module, *, boundary: str) -> None:
    """Fail unless an installed replacement declares the current contract."""

    observed = getattr(module, "runtime_tensor_contract_schema", None)
    if observed != RUNTIME_TENSOR_CONTRACT_SCHEMA:
        raise RuntimeError(
            f"{boundary} lacks the required replacement runtime tensor contract: "
            f"observed={observed!r}, required={RUNTIME_TENSOR_CONTRACT_SCHEMA!r}"
        )


def make_linear_output_projection(
    input_dim: int,
    output_dim: int,
    *,
    bias: bool = False,
    init_std: float = 0.02,
) -> nn.Linear:
    """Create the signed output projection used after dendritic activity."""
    projection = nn.Linear(int(input_dim), int(output_dim), bias=bool(bias))
    nn.init.normal_(projection.weight, mean=0.0, std=float(init_std))
    if projection.bias is not None:
        nn.init.zeros_(projection.bias)
    return projection


class TokenReplacementCell(nn.Module):
    """Single-site replacement cell with an adapter/core/projection contract.

    The module keeps the conventional attribute names ``pre_norm``, ``core``,
    and ``output_projection`` so existing transformer replacement checkpoints
    do not need state-dict key migration.
    """

    runtime_tensor_contract_schema = RUNTIME_TENSOR_CONTRACT_SCHEMA

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        input_transform: str,
        core: nn.Module,
        core_output_dim: int,
        output_bias: bool = False,
        output_init_std: float = 0.02,
        output_scale: float = 1.0,
        pre_norm: bool | nn.Module = False,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.input_transform = validate_input_transform(input_transform)
        self.output_scale = float(output_scale)
        self.pre_norm = self._build_pre_norm(pre_norm)
        self.input_adapter = NonNegativeInputAdapter(
            self.input_dim,
            self.input_transform,
        )
        self.core = core
        self.output_projection = make_linear_output_projection(
            core_output_dim,
            self.output_dim,
            bias=output_bias,
            init_std=output_init_std,
        )

    def _build_pre_norm(self, pre_norm: bool | nn.Module) -> nn.Module:
        if isinstance(pre_norm, nn.Module):
            return pre_norm
        return nn.LayerNorm(self.input_dim) if pre_norm else nn.Identity()

    def _check_input_shape(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected hidden size {self.input_dim}, got {x.shape[-1]}"
            )

    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the input-side adapter before the dendritic core."""
        self._check_input_shape(x)
        return self.input_adapter(self.pre_norm(x))

    def run_core(self, adapted: torch.Tensor) -> torch.Tensor:
        """Run the replacement core. Subclasses may override this."""
        return self.core(adapted)

    def decode_output(self, core_state: torch.Tensor) -> torch.Tensor:
        """Project dendritic activity back to the target module contract."""
        return self.output_projection(core_state) * self.output_scale

    def forward(self, x: torch.Tensor, *args: object, **kwargs: object):
        del args, kwargs
        output = self.decode_output(self.run_core(self.encode_input(x)))
        return preserve_runtime_tensor_contract(
            output,
            x,
            boundary=type(self).__name__,
        )


class LayerwiseTokenReplacementStack(nn.Module):
    """Shared adapter/output contract for one replacement cell per layer."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        input_transform: str,
        output_scale: float = 1.0,
        pre_norm: bool = False,
    ):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.input_transform = validate_input_transform(input_transform)
        self.output_scale = float(output_scale)
        self.pre_norm = (
            nn.ModuleList(
                [nn.LayerNorm(self.input_dim) for _ in range(self.num_layers)]
            )
            if pre_norm
            else nn.ModuleList([nn.Identity() for _ in range(self.num_layers)])
        )
        self.output_projections = nn.ModuleList()
        self.input_adapters = nn.ModuleList(
            [
                NonNegativeInputAdapter(self.input_dim, self.input_transform)
                for _ in range(self.num_layers)
            ]
        )

    @property
    def adapted_input_dim(self) -> int:
        return self.input_adapters[0].output_dim

    def _validate_layer_index(self, layer_index: int) -> int:
        idx = int(layer_index)
        if idx < 0 or idx >= self.num_layers:
            raise IndexError(
                f"replacement layer {idx} out of range for {self.num_layers}"
            )
        return idx

    def _check_input_shape(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected hidden size {self.input_dim}, got {x.shape[-1]}"
            )

    def encode_layer_input(self, layer_index: int, x: torch.Tensor) -> torch.Tensor:
        idx = self._validate_layer_index(layer_index)
        self._check_input_shape(x)
        return self.input_adapters[idx](self.pre_norm[idx](x))

    def decode_layer_output(
        self,
        layer_index: int,
        core_state: torch.Tensor,
    ) -> torch.Tensor:
        idx = self._validate_layer_index(layer_index)
        return self.output_projections[idx](core_state) * self.output_scale
