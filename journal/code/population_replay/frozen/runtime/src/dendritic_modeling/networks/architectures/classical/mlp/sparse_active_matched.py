"""All-output-active fixed-index sparse affine point control.

This module provides a point-neuron comparison that matches a structured
dendritic core's number of learned scalars without introducing a bottleneck,
zero-padded channels, branching, E/I pathways, or shunting.  Connectivity is
fixed, label independent, and stored as explicit integer source indices.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork,
)


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    normalized = int(value)
    if normalized < 1 or normalized != value:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _fixed_support_generator(seed: int) -> torch.Generator:
    if isinstance(seed, bool) or int(seed) != seed:
        raise ValueError("indexed_seed must be an integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    return generator


def _balanced_fixed_support(
    *,
    input_dim: int,
    output_dim: int,
    weight_parameters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Build balanced, unique, seed-keyed fixed source indices.

    The equal-fan-in part walks a seed-permuted input ring.  Consequently, its
    total input degree differs by at most one even when the dimensions are not
    equal.  Remainder edges are assigned to seed-selected output rows and then
    greedily placed on minimum-degree inputs that are not already used by that
    output.  No dense candidate mask is materialized or retained.
    """

    base_fan_in, higher_fan_in_rows = divmod(weight_parameters, output_dim)
    if base_fan_in < 1:
        raise ValueError(
            "The active-parameter target must provide at least one weight per "
            "output in addition to its bias"
        )
    if base_fan_in + bool(higher_fan_in_rows) > input_dim:
        raise ValueError(
            "The active-parameter target requires more unique inputs per output "
            f"than are available: maximum fan-in={base_fan_in + 1}, "
            f"input_dim={input_dim}"
        )

    generator = _fixed_support_generator(seed)
    source_permutation = torch.randperm(input_dim, generator=generator)
    row_starts = torch.arange(output_dim, dtype=torch.long)[:, None] * base_fan_in
    within_row = torch.arange(base_fan_in, dtype=torch.long)[None, :]
    ring_positions = (row_starts + within_row).remainder(input_dim)
    base_connection_indices = source_permutation[ring_positions].contiguous()

    if higher_fan_in_rows == 0:
        return (
            base_connection_indices,
            torch.empty(0, dtype=torch.long),
            (),
        )

    high_output_rows_tensor = torch.randperm(output_dim, generator=generator)[
        :higher_fan_in_rows
    ]
    high_output_rows = tuple(int(row) for row in high_output_rows_tensor.tolist())

    input_degree = torch.bincount(
        base_connection_indices.reshape(-1), minlength=input_dim
    ).tolist()
    candidate_priority = torch.randperm(input_dim, generator=generator).tolist()
    extra_sources: list[int] = []
    for output_row in high_output_rows:
        existing = {int(value) for value in base_connection_indices[output_row]}
        minimum_allowed_degree = min(
            input_degree[source]
            for source in candidate_priority
            if source not in existing
        )
        selected_source = next(
            source
            for source in candidate_priority
            if source not in existing and input_degree[source] == minimum_allowed_degree
        )
        extra_sources.append(int(selected_source))
        input_degree[selected_source] += 1

    return (
        base_connection_indices,
        torch.tensor(extra_sources, dtype=torch.long),
        high_output_rows,
    )


class SparseActiveMatchedPointAffine(nn.Module):
    """Fixed-index sparse affine point layer followed by ReLU.

    The number of biases is the output width.  All remaining learned scalars
    are signed affine weights distributed as evenly as possible across output
    rows.  ``target_active_parameters`` is an explicit testing/diagnostic
    override; normal structured configs derive the target from the referenced
    :class:`ExcitationInhibitionNetwork`.
    """

    def __init__(
        self,
        input_dim: int,
        excitatory_layer_sizes: list[int],
        inhibitory_layer_sizes,
        excitatory_branch_factors,
        inhibitory_branch_factors,
        ee_synapses_per_branch_per_layer,
        ei_synapses_per_branch_per_layer,
        ie_synapses_per_branch_per_layer,
        ii_synapses_per_branch_per_layer=None,
        reactivate: bool = False,
        somatic_synapses: bool = True,
        activation: str = "relu",
        target_active_parameters: int | None = None,
        indexed_seed: int | None = None,
        indexed_output_chunk_size: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        if not excitatory_layer_sizes:
            raise ValueError("excitatory_layer_sizes cannot be empty")
        if str(activation).lower() != "relu":
            raise ValueError(
                "SparseActiveMatchedPointAffine has a fixed ReLU activation"
            )
        if indexed_seed is None:
            raise ValueError(
                "sparse_active_matched_point requires sparsity.indexed.seed so "
                "its fixed support is reproducible"
            )
        if isinstance(indexed_output_chunk_size, bool):
            raise ValueError("indexed_output_chunk_size must be a positive integer")

        self.input_dim = _positive_integer(input_dim, name="input_dim")
        self.output_dim = _positive_integer(
            excitatory_layer_sizes[-1], name="output_dim"
        )
        self.support_seed = int(indexed_seed)
        self.output_chunk_size = _positive_integer(
            indexed_output_chunk_size,
            name="indexed_output_chunk_size",
        )

        transfer_params = dict(kwargs.get("transfer_params") or {})
        output_activation = transfer_params.get("output_activation")
        normalized_output_activation = (
            "none" if output_activation is None else str(output_activation).lower()
        )
        if normalized_output_activation not in {"none", "relu"}:
            raise ValueError(
                "SparseActiveMatchedPointAffine supports only the fixed ReLU interface"
            )

        if target_active_parameters is None:
            if ii_synapses_per_branch_per_layer is None:
                ii_synapses_per_branch_per_layer = []
            reference_kwargs = dict(kwargs)
            reference_kwargs["transfer_params"] = {
                "input_mode": 0,
                "independent_pathways": False,
                "excitatory_dim": [],
                "inhibitory_dim": [],
                "output_activation": "none",
                **transfer_params,
            }
            # Counting the reference must not perturb this point control's
            # parameter-initialization stream.
            with torch.random.fork_rng(devices=[]):
                reference = ExcitationInhibitionNetwork(
                    input_dim=self.input_dim,
                    excitatory_layer_sizes=excitatory_layer_sizes,
                    inhibitory_layer_sizes=inhibitory_layer_sizes,
                    excitatory_branch_factors=excitatory_branch_factors,
                    inhibitory_branch_factors=inhibitory_branch_factors,
                    ee_synapses_per_branch_per_layer=(ee_synapses_per_branch_per_layer),
                    ei_synapses_per_branch_per_layer=(ei_synapses_per_branch_per_layer),
                    ie_synapses_per_branch_per_layer=(ie_synapses_per_branch_per_layer),
                    ii_synapses_per_branch_per_layer=(ii_synapses_per_branch_per_layer),
                    reactivate=reactivate,
                    somatic_synapses=somatic_synapses,
                    **reference_kwargs,
                )
                target_active_parameters = int(reference.get_effective_params())
                del reference

        target = _positive_integer(
            target_active_parameters,
            name="target_active_parameters",
        )
        if target <= self.output_dim:
            raise ValueError(
                "target_active_parameters must exceed output_dim so every output "
                "has a bias and at least one fixed input"
            )

        weight_parameters = target - self.output_dim
        base_indices, extra_indices, high_output_rows = _balanced_fixed_support(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            weight_parameters=weight_parameters,
            seed=self.support_seed,
        )
        self.register_buffer("base_connection_indices", base_indices, persistent=True)
        self.register_buffer("extra_connection_indices", extra_indices, persistent=True)
        self.register_buffer(
            "extra_output_connection_indices",
            torch.tensor(high_output_rows, dtype=torch.long),
            persistent=True,
        )

        self.base_fan_in = int(base_indices.shape[1])
        self.higher_fan_in_row_count = int(extra_indices.numel())
        self.maximum_fan_in = self.base_fan_in + bool(self.higher_fan_in_row_count)
        self.weight_parameters = int(weight_parameters)
        self.bias_parameters = self.output_dim
        self.target_active_parameters = target
        self.actual_active_parameters = target
        self.actual_stored_parameters = target
        self.weight_parameterization = "signed_affine"
        self.all_outputs_active = True

        self.base_weight = nn.Parameter(torch.empty(self.output_dim, self.base_fan_in))
        if self.higher_fan_in_row_count:
            self.extra_weight = nn.Parameter(torch.empty(self.higher_fan_in_row_count))
        else:
            # An empty Parameter is never used by ``forward`` and is therefore
            # reported as unfinished by DistributedDataParallel.  Registering
            # ``None`` preserves the exact zero-scalar accounting without
            # advertising a trainable parameter that cannot receive a gradient.
            self.register_parameter("extra_weight", None)
        self.bias = nn.Parameter(torch.empty(self.output_dim))
        self.activation = nn.ReLU()
        self._initialize_parameters()

    @property
    def high_fan_in_output_rows(self) -> tuple[int, ...]:
        """Return persisted destination rows as an immutable Python view."""

        return tuple(
            int(row)
            for row in self.extra_output_connection_indices.detach().cpu().tolist()
        )

    def _initialize_parameters(self) -> None:
        """Apply row-wise Kaiming fan-in initialization and zero bias."""

        nn.init.kaiming_normal_(
            self.base_weight,
            mode="fan_in",
            nonlinearity="relu",
        )
        if self.higher_fan_in_row_count:
            assert self.extra_weight is not None
            high_rows = self.extra_output_connection_indices.to(self.base_weight.device)
            with torch.no_grad():
                scaled_high_rows = self.base_weight.index_select(
                    0, high_rows
                ) * math.sqrt(self.base_fan_in / self.maximum_fan_in)
                self.base_weight.index_copy_(
                    0,
                    high_rows,
                    scaled_high_rows,
                )
                self.extra_weight.normal_(
                    mean=0.0,
                    std=math.sqrt(2.0 / self.maximum_fan_in),
                )
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the signed sparse affine projection and fixed ReLU."""

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")
        squeezed = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeezed = True
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Input feature dimension mismatch: got {x.shape[-1]}, "
                f"expected {self.input_dim}"
            )

        original_shape = x.shape
        flat_x = x.reshape(-1, self.input_dim)
        affine = flat_x.new_empty(flat_x.shape[0], self.output_dim)
        base_indices = self.base_connection_indices.to(flat_x.device)
        for start in range(0, self.output_dim, self.output_chunk_size):
            end = min(start + self.output_chunk_size, self.output_dim)
            index_chunk = base_indices[start:end]
            selected = flat_x[:, index_chunk.reshape(-1)].reshape(
                flat_x.shape[0], end - start, self.base_fan_in
            )
            affine[:, start:end] = (
                selected * self.base_weight[start:end].unsqueeze(0)
            ).sum(dim=-1)

        if self.higher_fan_in_row_count:
            assert self.extra_weight is not None
            extra_inputs = flat_x.index_select(
                1, self.extra_connection_indices.to(flat_x.device)
            )
            # Mixed precision promotes ``reduced input * FP32 parameter`` to
            # FP32, while the chunked base projection has already been stored
            # in ``affine`` at the input activation dtype.  ``index_add``
            # requires exact dtype equality, so round the extra contribution
            # at the same accumulator boundary as every base-row sum.
            extra_values = (extra_inputs * self.extra_weight.unsqueeze(0)).to(
                dtype=affine.dtype
            )
            affine = affine.index_add(
                1,
                self.extra_output_connection_indices.to(flat_x.device),
                extra_values,
            )
        affine = affine + self.bias
        output = self.activation(affine).reshape(*original_shape[:-1], self.output_dim)
        return output.squeeze(0) if squeezed else output

    def get_effective_params(self) -> int:
        """Return the exact number of learned scalars used in every forward."""

        return self.actual_active_parameters

    def connectivity_resource_records(self) -> list[dict[str, Any]]:
        """Expose the fixed sparse affine support to standard resource tools."""

        return [
            {
                "module": "sparse_point_affine",
                "pathway": "point_affine",
                "soma_relative_depth": 0,
                "out_features": self.output_dim,
                "in_features": self.input_dim,
                "candidate_parameters": self.weight_parameters,
                "candidate_slots": self.output_dim * self.input_dim,
                "active_synapses": self.weight_parameters,
                "realized_k_min": self.base_fan_in,
                "realized_k_max": self.maximum_fan_in,
                "realized_k_mean": self.weight_parameters / self.output_dim,
                "selection_policy": "fixed_index",
                "mask_source": "connection_indices",
            }
        ]

    def connectivity_support_sha256(self) -> str:
        """Return a canonical hash of the exact output-to-input support."""

        digest = hashlib.sha256()
        digest.update(b"SparseActiveMatchedPointAffine.support.v2\0")
        digest.update(
            struct.pack(
                "<5q",
                self.input_dim,
                self.output_dim,
                self.base_fan_in,
                self.higher_fan_in_row_count,
                self.weight_parameters,
            )
        )
        digest.update(
            self.base_connection_indices.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("<i8", copy=False)
            .tobytes()
        )
        digest.update(
            self.extra_connection_indices.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("<i8", copy=False)
            .tobytes()
        )
        digest.update(
            self.extra_output_connection_indices.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype("<i8", copy=False)
            .tobytes()
        )
        return digest.hexdigest()

    def resource_accounting(self) -> dict[str, int | bool | str]:
        """Expose exact active, stored, support, and buffer accounting."""

        stored_parameters = int(
            sum(parameter.numel() for parameter in self.parameters())
        )
        persistent_buffer_bytes = int(
            sum(buffer.numel() * buffer.element_size() for buffer in self.buffers())
        )
        source_index_entries = int(
            self.base_connection_indices.numel() + self.extra_connection_indices.numel()
        )
        source_index_buffer_bytes = int(
            self.base_connection_indices.numel()
            * self.base_connection_indices.element_size()
            + self.extra_connection_indices.numel()
            * self.extra_connection_indices.element_size()
        )
        destination_index_entries = int(self.extra_output_connection_indices.numel())
        destination_index_buffer_bytes = int(
            destination_index_entries
            * self.extra_output_connection_indices.element_size()
        )
        connection_index_entries = source_index_entries + destination_index_entries
        connection_index_buffer_bytes = (
            source_index_buffer_bytes + destination_index_buffer_bytes
        )
        return {
            "target_active_parameters": self.target_active_parameters,
            "active_parameters": self.actual_active_parameters,
            "stored_parameters": stored_parameters,
            "weight_parameters": self.weight_parameters,
            "bias_parameters": self.bias_parameters,
            "source_index_entries": source_index_entries,
            "source_index_buffer_bytes": source_index_buffer_bytes,
            "destination_index_entries": destination_index_entries,
            "destination_index_buffer_bytes": destination_index_buffer_bytes,
            "connection_index_entries": connection_index_entries,
            "connection_index_buffer_bytes": connection_index_buffer_bytes,
            "persistent_buffer_bytes": persistent_buffer_bytes,
            "other_buffer_bytes": (
                persistent_buffer_bytes - connection_index_buffer_bytes
            ),
            "all_outputs_active": self.all_outputs_active,
            "output_fan_in_min": self.base_fan_in,
            "output_fan_in_max": self.maximum_fan_in,
            "higher_fan_in_output_count": self.higher_fan_in_row_count,
            "weight_parameterization": self.weight_parameterization,
            "support_sha256": self.connectivity_support_sha256(),
        }


__all__ = ["SparseActiveMatchedPointAffine"]
