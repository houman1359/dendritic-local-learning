"""Fixed Legendre Delay Network memory with sparse learned interfaces.

This module implements the linear memory cell of the Legendre Delay Network
(LDN).  It is deliberately not a full Legendre Memory Unit (LMU): the memory
write signal has no learned recurrent controller, and the state transition,
input injection, and delay horizon are fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.degree_grouped_indexed import (
    DegreeGroupedIndexedLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
    normalize_indexed_projection_backend,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    apply_recurrent_weight_cache,
)
from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.utils.stable_hash import stable_seed_offset

_BUFFER_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive and finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be positive and finite") from None
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return parsed


def legendre_delay_continuous_matrices(
    memory_order: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the primary LDN continuous-time ``A`` and ``B`` in float64.

    Indices are zero based.  The returned matrices implement

    ``A[i,j] = (2i+1)/theta * (-1 if i<j else (-1)**(i-j+1))`` and
    ``B[i] = (2i+1)/theta * (-1)**i``.
    """

    q = _positive_integer(memory_order, name="memory_order")
    horizon = _positive_finite(theta, name="theta")
    A = torch.empty((q, q), dtype=torch.float64)
    B = torch.empty(q, dtype=torch.float64)
    for i in range(q):
        scale = float(2 * i + 1) / horizon
        B[i] = scale * (-1.0 if i % 2 else 1.0)
        for j in range(q):
            if i < j:
                sign = -1.0
            else:
                sign = -1.0 if (i - j + 1) % 2 else 1.0
            A[i, j] = scale * sign
    return A, B


def zero_order_hold_discretization(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    dt: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretize ``dm/dt = A m + B u`` by an augmented matrix exponential.

    Calculation is always performed in float64.  The top-right block of the
    augmented exponential is exactly the zero-order-hold integral, including
    when ``A`` is singular, so this implementation never forms ``A^{-1}``.
    """

    step = _positive_finite(dt, name="dt")
    if not isinstance(A, torch.Tensor) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square rank-two tensor")
    q = int(A.shape[0])
    if not isinstance(B, torch.Tensor) or tuple(B.shape) not in {(q,), (q, 1)}:
        raise ValueError(f"B must have shape ({q},) or ({q}, 1)")
    A64 = A.detach().to(device="cpu", dtype=torch.float64)
    B64 = B.detach().reshape(q).to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(A64).all()) or not bool(torch.isfinite(B64).all()):
        raise ValueError("A and B must contain only finite values")

    augmented = torch.zeros((q + 1, q + 1), dtype=torch.float64)
    augmented[:q, :q] = A64
    augmented[:q, q] = B64
    discrete = torch.linalg.matrix_exp(augmented * step)
    A_bar = discrete[:q, :q].contiguous()
    B_bar = discrete[:q, q].contiguous()
    if not bool(torch.isfinite(A_bar).all()) or not bool(torch.isfinite(B_bar).all()):
        raise ValueError("zero-order-hold discretization produced non-finite values")
    return A_bar, B_bar


@dataclass
class FixedLegendreMemoryConfig:
    """Configuration for a fixed LDN memory and sparse learned interfaces."""

    input_dim: int = 1
    output_dim: int = 1
    memory_order: int = 6
    n_memory_channels: int = 1
    theta: float = 1.0
    dt: float = 1.0
    input_contacts: int = 1
    readout_contacts: int = 6
    topology_seed: int = 0
    initialization_seed: int = 0
    output_mode: str = "last"
    transition_buffer_dtype: str = "float32"
    trainable_transition: bool = False
    trainable_theta: bool = False
    indexed_output_chunk_size: int = 2048
    indexed_workspace_mb: float | None = None
    indexed_cache_transformed_weights: bool = False
    indexed_recompute_backward: bool = False
    indexed_persistent_indices: bool = True
    indexed_projection_backend: str = "eager"

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "output_dim",
            "memory_order",
            "n_memory_channels",
            "input_contacts",
            "readout_contacts",
            "indexed_output_chunk_size",
        ):
            _positive_integer(getattr(self, name), name=name)
        for name in ("topology_seed", "initialization_seed"):
            _non_negative_integer(getattr(self, name), name=name)
        _positive_finite(self.theta, name="theta")
        _positive_finite(self.dt, name="dt")

        if self.output_mode not in {"all", "last", "mean"}:
            raise ValueError("output_mode must be 'all', 'last', or 'mean'")
        if self.transition_buffer_dtype not in _BUFFER_DTYPES:
            choices = ", ".join(sorted(_BUFFER_DTYPES))
            raise ValueError(f"transition_buffer_dtype must be one of: {choices}")
        if not isinstance(self.trainable_transition, bool):
            raise TypeError("trainable_transition must be boolean")
        if self.trainable_transition:
            raise ValueError("the LDN transition is fixed and cannot be trainable")
        if not isinstance(self.trainable_theta, bool):
            raise TypeError("trainable_theta must be boolean")
        if self.trainable_theta:
            raise ValueError("the LDN theta horizon is fixed and cannot be trainable")
        if self.indexed_workspace_mb is not None:
            _positive_finite(self.indexed_workspace_mb, name="indexed_workspace_mb")
        self.indexed_projection_backend = normalize_indexed_projection_backend(
            self.indexed_projection_backend
        )
        if self.indexed_recompute_backward and self.indexed_projection_backend not in {
            "eager",
            "recompute",
        }:
            raise ValueError(
                "indexed_recompute_backward cannot be combined with "
                f"indexed_projection_backend={self.indexed_projection_backend!r}"
            )

        self._validate_contacts(
            name="input_contacts",
            contacts=self.input_contacts,
            input_width=self.input_dim,
            output_width=self.n_memory_channels,
        )
        self._validate_contacts(
            name="readout_contacts",
            contacts=self.readout_contacts,
            input_width=self.state_dim,
            output_width=self.output_dim,
        )

    @property
    def state_dim(self) -> int:
        return self.memory_order * self.n_memory_channels

    @staticmethod
    def _validate_contacts(
        *,
        name: str,
        contacts: int,
        input_width: int,
        output_width: int,
    ) -> None:
        minimum = max(input_width, output_width)
        maximum = input_width * output_width
        if not minimum <= contacts <= maximum:
            raise ValueError(
                f"{name} must be in [{minimum}, {maximum}] so every source and "
                f"output is covered, got {contacts}"
            )


@dataclass
class LegendreMemoryState:
    """Carried LDN coefficients with shape ``[batch, channels, order]``."""

    memory: torch.Tensor

    def detach(self) -> LegendreMemoryState:
        return LegendreMemoryState(memory=self.memory.detach())

    def clone(self) -> LegendreMemoryState:
        return LegendreMemoryState(memory=self.memory.clone())

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
    ) -> LegendreMemoryState:
        return LegendreMemoryState(memory=self.memory.to(device=device, dtype=dtype))


def _seed(base: int, *parts: object) -> int:
    return (int(base) + stable_seed_offset("fixed_legendre_memory", *parts)) % (
        (1 << 63) - 1
    )


def _balanced_degrees(
    *,
    total_contacts: int,
    input_width: int,
    output_width: int,
    seed: int,
) -> torch.Tensor:
    """Allocate an exact contact budget over outputs with degree spread <= 1."""

    base, remainder = divmod(total_contacts, output_width)
    if base > input_width or (base == input_width and remainder):
        raise ValueError("contact budget exceeds the available unique supports")
    degrees = torch.full((output_width,), base, dtype=torch.long)
    if remainder:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        rows = torch.randperm(output_width, generator=generator)[:remainder]
        degrees[rows] += 1
    if int(degrees.sum().item()) != total_contacts:
        raise RuntimeError("degree allocation changed the contact budget")
    if bool((degrees < 1).any()):
        raise RuntimeError("degree allocation failed to cover every output")
    return degrees


class FixedLegendreMemory(BaseNetwork):
    """Fixed linear LDN memory with learned sparse input and state maps."""

    def __init__(self, config: FixedLegendreMemoryConfig):
        super().__init__()
        if not isinstance(config, FixedLegendreMemoryConfig):
            raise TypeError("config must be FixedLegendreMemoryConfig")
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self._output_mode = config.output_mode

        A64, B64 = legendre_delay_continuous_matrices(
            config.memory_order,
            config.theta,
        )
        A_bar64, B_bar64 = zero_order_hold_discretization(A64, B64, dt=config.dt)
        buffer_dtype = _BUFFER_DTYPES[config.transition_buffer_dtype]
        self.register_buffer(
            "theta",
            torch.tensor(float(config.theta), dtype=buffer_dtype),
        )
        self.register_buffer("A", A64.to(dtype=buffer_dtype))
        self.register_buffer("B", B64.to(dtype=buffer_dtype))
        self.register_buffer("A_bar", A_bar64.to(dtype=buffer_dtype))
        self.register_buffer("B_bar", B_bar64.to(dtype=buffer_dtype))

        sparse_options = {
            "weight_transform": "identity",
            "output_chunk_size": config.indexed_output_chunk_size,
            "workspace_mb": config.indexed_workspace_mb,
            "cache_transformed_weights": config.indexed_cache_transformed_weights,
            "recompute_backward": config.indexed_recompute_backward,
            "projection_backend": config.indexed_projection_backend,
            "persistent_indices": config.indexed_persistent_indices,
        }
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed(config.initialization_seed, "input_mixer"))
            self.input_mixer = DegreeGroupedIndexedLinear(
                config.input_dim,
                config.n_memory_channels,
                _balanced_degrees(
                    total_contacts=config.input_contacts,
                    input_width=config.input_dim,
                    output_width=config.n_memory_channels,
                    seed=_seed(config.topology_seed, "input_degrees"),
                ),
                seed=_seed(config.topology_seed, "input_support"),
                require_source_coverage=True,
                **sparse_options,
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed(config.initialization_seed, "state_readout"))
            self.state_readout = DegreeGroupedIndexedLinear(
                config.state_dim,
                config.output_dim,
                _balanced_degrees(
                    total_contacts=config.readout_contacts,
                    input_width=config.state_dim,
                    output_width=config.output_dim,
                    seed=_seed(config.topology_seed, "readout_degrees"),
                ),
                seed=_seed(config.topology_seed, "readout_support"),
                require_source_coverage=True,
                **sparse_options,
            )

    @property
    def is_recurrent(self) -> bool:
        return True

    @property
    def memory_order(self) -> int:
        return self.config.memory_order

    @property
    def n_memory_channels(self) -> int:
        return self.config.n_memory_channels

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> LegendreMemoryState:
        _positive_integer(batch_size, name="batch_size")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch dtype")
        return LegendreMemoryState(
            memory=torch.zeros(
                batch_size,
                self.config.n_memory_channels,
                self.config.memory_order,
                device=device,
                dtype=dtype,
            )
        )

    def _validate_step_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("x must have shape [batch, input_dim]")
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"x must have input dimension {self.config.input_dim}, got {x.shape[-1]}"
            )
        if not x.dtype.is_floating_point:
            raise TypeError("x must have a floating-point dtype")

    def _validate_state(self, state: LegendreMemoryState, x: torch.Tensor) -> None:
        if not isinstance(state, LegendreMemoryState):
            raise TypeError(
                f"state must be LegendreMemoryState, got {type(state).__name__}"
            )
        expected = (
            x.shape[0],
            self.config.n_memory_channels,
            self.config.memory_order,
        )
        if tuple(state.memory.shape) != expected:
            raise ValueError(
                f"state.memory must have shape {expected}, got "
                f"{tuple(state.memory.shape)}"
            )
        if state.memory.device != x.device or state.memory.dtype != x.dtype:
            raise ValueError("input and recurrent state must share device and dtype")

    def step(
        self,
        x: torch.Tensor,
        state: LegendreMemoryState | None = None,
    ) -> tuple[torch.Tensor, LegendreMemoryState]:
        """Advance the fixed LDN state by one zero-order-held input step."""

        self._validate_step_input(x)
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        self._validate_state(state, x)

        drive = self.input_mixer(x)
        A_bar = self.A_bar.to(device=x.device, dtype=x.dtype)
        B_bar = self.B_bar.to(device=x.device, dtype=x.dtype)
        memory = torch.matmul(state.memory, A_bar.transpose(0, 1))
        memory = memory + drive.unsqueeze(-1) * B_bar
        output = self.state_readout(memory.flatten(start_dim=1))
        return output, LegendreMemoryState(memory=memory)

    def _reduce_outputs(
        self,
        outputs: torch.Tensor,
        seq_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        if seq_lengths is not None:
            if not isinstance(seq_lengths, torch.Tensor) or tuple(
                seq_lengths.shape
            ) != (outputs.shape[0],):
                raise ValueError("seq_lengths must have shape [batch]")
            integer_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if seq_lengths.dtype not in integer_dtypes:
                raise TypeError("seq_lengths must have an integer dtype")
            lengths = seq_lengths.to(device=outputs.device, dtype=torch.long)
            if bool((lengths < 0).any()) or bool((lengths > outputs.shape[1]).any()):
                raise ValueError("seq_lengths entries must be in [0, time]")
        else:
            lengths = None

        if self._output_mode == "last":
            if lengths is None:
                return outputs[:, -1]
            indices = (lengths - 1).clamp(0, outputs.shape[1] - 1)
            selected = outputs[
                torch.arange(outputs.shape[0], device=outputs.device),
                indices,
            ]
            return torch.where(
                lengths[:, None] > 0, selected, torch.zeros_like(selected)
            )
        if self._output_mode == "mean":
            if lengths is None:
                return outputs.mean(dim=1)
            mask = torch.arange(outputs.shape[1], device=outputs.device)[None, :]
            mask = mask < lengths[:, None]
            denominator = lengths.clamp_min(1).to(outputs.dtype)[:, None]
            return (outputs * mask[..., None].to(outputs.dtype)).sum(
                dim=1
            ) / denominator
        if self._output_mode == "all":
            if lengths is None:
                return outputs
            mask = torch.arange(outputs.shape[1], device=outputs.device)[None, :]
            return outputs * (mask < lengths[:, None])[..., None].to(outputs.dtype)
        raise RuntimeError(f"unsupported output mode {self._output_mode!r}")

    def forward(
        self,
        x: torch.Tensor,
        hidden: LegendreMemoryState | None = None,
        return_hidden: bool = False,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, LegendreMemoryState]:
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must have shape [batch, time, input_dim]")
        if x.shape[1] < 1:
            raise ValueError("x must contain at least one timestep")
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"x must have input dimension {self.config.input_dim}, got {x.shape[-1]}"
            )
        if not x.dtype.is_floating_point:
            raise TypeError("x must have a floating-point dtype")

        state = hidden
        sequence_outputs = []
        apply_recurrent_weight_cache(self, True)
        try:
            for timestep in range(x.shape[1]):
                output, state = self.step(x[:, timestep], state)
                sequence_outputs.append(output)
        finally:
            apply_recurrent_weight_cache(self, False)
        assert state is not None
        reduced = self._reduce_outputs(
            torch.stack(sequence_outputs, dim=1), seq_lengths
        )
        if return_hidden:
            return reduced, state
        return reduced

    def connectivity_resource_records(self) -> list[dict[str, Any]]:
        """Return exact learned-map records for generic resource accounting."""

        modules = (
            ("input_mixer", "input_mixer", self.input_mixer),
            ("state_readout", "state_readout", self.state_readout),
        )
        records = []
        for name, pathway, module in modules:
            counts = module.connectivity_resource_counts()
            records.append(
                {
                    "module": name,
                    "pathway": pathway,
                    "soma_relative_depth": -1,
                    "out_features": int(counts["out_features"]),
                    "in_features": int(counts["in_features"]),
                    "candidate_parameters": int(counts["candidate_parameters"]),
                    "candidate_slots": int(counts["candidate_slots"]),
                    "active_synapses": int(counts["active_synapses"]),
                    "realized_k_min": int(counts["realized_k_min"]),
                    "realized_k_max": int(counts["realized_k_max"]),
                    "realized_k_mean": float(counts["realized_k_mean"]),
                    "selection_policy": str(counts["selection_policy"]),
                    "mask_source": str(counts["mask_source"]),
                }
            )
        return records

    def resource_ledger(self) -> dict[str, Any]:
        """Separate carried state, learned contacts, and fixed LDN couplings."""

        q = self.config.memory_order
        channels = self.config.n_memory_channels
        fixed_transition_declared = channels * q * q
        fixed_injection_declared = channels * q
        fixed_transition_nonzero = channels * int(torch.count_nonzero(self.A_bar))
        fixed_injection_nonzero = channels * int(torch.count_nonzero(self.B_bar))
        return {
            "memory_state_scalars_per_sample": self.config.state_dim,
            "reachable_carried_state_scalars_per_sample": self.config.state_dim,
            "memory_order": q,
            "memory_channels": channels,
            "theta": float(self.config.theta),
            "dt": float(self.config.dt),
            "learned_input_mixer_contacts": self.config.input_contacts,
            "learned_state_readout_contacts": self.config.readout_contacts,
            "active_learned_contacts": (
                self.config.input_contacts + self.config.readout_contacts
            ),
            "fixed_transition_coefficients_declared": fixed_transition_declared,
            "fixed_input_injection_coefficients_declared": fixed_injection_declared,
            "fixed_couplings_declared": (
                fixed_transition_declared + fixed_injection_declared
            ),
            "fixed_transition_coefficients_nonzero": fixed_transition_nonzero,
            "fixed_input_injection_coefficients_nonzero": fixed_injection_nonzero,
            "shared_fixed_discrete_operator_scalars": q * q + q,
            "registered_fixed_buffer_scalars": 2 * q * q + 2 * q + 1,
            "input_source_coverage": bool(
                (self.input_mixer.source_outdegrees() > 0).all()
            ),
            "state_source_coverage": bool(
                (self.state_readout.source_outdegrees() > 0).all()
            ),
            "transition_trainable": False,
            "theta_trainable": False,
        }


__all__ = [
    "FixedLegendreMemory",
    "FixedLegendreMemoryConfig",
    "LegendreMemoryState",
    "legendre_delay_continuous_matrices",
    "zero_order_hold_discretization",
]
