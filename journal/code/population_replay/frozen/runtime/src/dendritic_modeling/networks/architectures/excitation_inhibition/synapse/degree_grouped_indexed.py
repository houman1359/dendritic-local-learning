"""Fixed-index sparse projections with an exact per-output degree vector."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
    normalize_indexed_projection_options,
)
from dendritic_modeling.networks.utils.weight_transforms import (
    WeightTransformType,
    inverse_weight_transform,
)


class _IndexedDegreeGroup(nn.Module):
    """One uniform-degree subset used by :class:`DegreeGroupedIndexedLinear`."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_indices: torch.Tensor,
        connection_indices: torch.Tensor,
        degree: int,
        weight_transform: WeightTransformType,
        row_sum_init: float | None,
        output_chunk_size: int,
        index_dtype: str | torch.dtype,
        workspace_mb: float | None,
        cache_transformed_weights: bool,
        recompute_backward: bool,
        projection_backend: str,
        persistent_indices: bool,
    ) -> None:
        super().__init__()
        self.register_buffer("output_indices", output_indices.to(torch.long))
        self.projection = IndexedSparseLinear(
            in_features=input_dim,
            out_features=int(output_indices.numel()),
            K=degree,
            connection_indices=connection_indices,
            init_method="xavier_normal",
            weight_transform=weight_transform,
            output_chunk_size=output_chunk_size,
            index_dtype=index_dtype,
            workspace_mb=workspace_mb,
            cache_transformed_weights=cache_transformed_weights,
            recompute_backward=recompute_backward,
            projection_backend=projection_backend,
            persistent_indices=persistent_indices,
        )
        if row_sum_init is not None:
            per_contact = float(row_sum_init) / float(degree)
            if per_contact <= 0.0 and weight_transform != "identity":
                raise ValueError(
                    "row_sum_init must be positive for a positive weight transform"
                )
            raw_value = inverse_weight_transform(
                torch.tensor(
                    per_contact,
                    dtype=self.projection.pre_w.dtype,
                    device=self.projection.pre_w.device,
                ),
                weight_transform,
            )
            with torch.no_grad():
                self.projection.pre_w.fill_(float(raw_value.item()))


def _validate_degree_vector(
    degrees: Sequence[int] | torch.Tensor,
    *,
    input_dim: int,
    output_dim: int,
) -> torch.Tensor:
    try:
        raw_degrees = torch.as_tensor(degrees)
    except (TypeError, ValueError) as exc:
        raise TypeError("degrees must contain integer values") from exc
    if raw_degrees.ndim != 1 or raw_degrees.numel() != output_dim:
        raise ValueError(
            f"degrees must have shape ({output_dim},), got {tuple(raw_degrees.shape)}"
        )
    if isinstance(degrees, torch.Tensor):
        integral_dtype = degrees.dtype != torch.bool and not (
            degrees.is_floating_point() or degrees.is_complex()
        )
        if not integral_dtype:
            raise TypeError("degrees must contain integer values")
    elif any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in degrees
    ):
        raise TypeError("degrees must contain integer values")
    degree_tensor = raw_degrees.to(dtype=torch.long)
    if bool((degree_tensor < 0).any()):
        raise ValueError("degrees must be non-negative")
    if bool((degree_tensor > input_dim).any()):
        raise ValueError(f"degrees cannot exceed the input dimension ({input_dim})")
    return degree_tensor.contiguous()


def _balanced_connection_rows(
    degrees: torch.Tensor,
    *,
    input_dim: int,
    seed: int,
) -> list[torch.Tensor]:
    """Build unique row supports with optimally balanced source usage.

    A seeded source permutation is traversed cyclically across all requested
    slots.  Each row contains at most ``input_dim`` consecutive permutation
    entries and therefore has no duplicate.  Globally, source outdegrees differ
    by at most one; all sources are covered whenever the contact budget permits.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    source_order = torch.randperm(input_dim, generator=generator)
    start = int(torch.randint(input_dim, (1,), generator=generator).item())
    cursor = start
    rows: list[torch.Tensor] = []
    for degree in degrees.tolist():
        if degree == 0:
            rows.append(torch.empty(0, dtype=torch.long))
            continue
        positions = (torch.arange(degree, dtype=torch.long) + cursor) % input_dim
        rows.append(source_order[positions])
        cursor += degree
    return rows


def _validate_connection_mask(
    connection_mask: torch.Tensor,
    *,
    degrees: torch.Tensor,
    input_dim: int,
    output_dim: int,
) -> torch.Tensor:
    """Validate a hard allowed-source mask against heterogeneous row degrees."""

    mask = torch.as_tensor(connection_mask, dtype=torch.bool, device="cpu")
    expected = (output_dim, input_dim)
    if tuple(mask.shape) != expected:
        raise ValueError(
            f"connection_mask must have shape {expected}, got {tuple(mask.shape)}"
        )
    allowed_counts = mask.sum(dim=1)
    invalid = allowed_counts < degrees
    if bool(invalid.any()):
        row = int(torch.nonzero(invalid, as_tuple=False)[0].item())
        raise ValueError(
            "connection_mask must allow at least the requested degree in every "
            f"row; row {row} allows {int(allowed_counts[row].item())} sources "
            f"for degree {int(degrees[row].item())}"
        )
    return mask.contiguous()


def _balanced_masked_connection_rows(
    degrees: torch.Tensor,
    *,
    connection_mask: torch.Tensor,
    seed: int,
) -> list[torch.Tensor]:
    """Build deterministic supports balanced within equal-eligibility classes.

    Rows with the same degree and allowed-source set form one equivalence class.
    A seeded cyclic assignment then balances source usage within that class while
    keeping every row unique.  This is the constrained analogue of
    :func:`_balanced_connection_rows`; asymmetric eligibility intentionally need
    not produce globally equal source outdegrees.
    """

    rows = [torch.empty(0, dtype=torch.long) for _ in range(degrees.numel())]
    classes: dict[tuple[int, tuple[int, ...]], list[int]] = {}
    for row, degree in enumerate(degrees.tolist()):
        if degree == 0:
            continue
        allowed = tuple(
            torch.nonzero(connection_mask[row], as_tuple=False).flatten().tolist()
        )
        classes.setdefault((degree, allowed), []).append(row)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for (degree, allowed_tuple), class_rows in classes.items():
        allowed = torch.tensor(allowed_tuple, dtype=torch.long)
        source_order = allowed[torch.randperm(allowed.numel(), generator=generator)]
        row_order = torch.tensor(class_rows, dtype=torch.long)
        row_order = row_order[torch.randperm(row_order.numel(), generator=generator)]
        cursor = int(torch.randint(allowed.numel(), (1,), generator=generator).item())
        for row in row_order.tolist():
            positions = (
                torch.arange(degree, dtype=torch.long) + cursor
            ) % allowed.numel()
            rows[row] = source_order[positions]
            cursor += degree
    return rows


class DegreeGroupedIndexedLinear(nn.Module):
    """Sparse linear map with exact, potentially heterogeneous row degrees.

    The implementation composes the existing fixed-index sparse kernel by
    grouping output rows that share a degree.  It supports zero-degree rows,
    exact total contact budgets, and deterministic topology.  The unrestricted
    construction guarantees source coverage whenever the total number of
    contacts is at least ``input_dim``.  A supplied hard mask constrains every
    selected index; requested source coverage is verified and fails closed.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        degrees: Sequence[int] | torch.Tensor,
        *,
        seed: int,
        require_source_coverage: bool = False,
        connection_mask: torch.Tensor | None = None,
        weight_transform: WeightTransformType = "exp",
        row_sum_init: float | None = None,
        output_chunk_size: int = 2048,
        index_dtype: str | torch.dtype = "int64",
        workspace_mb: float | None = None,
        cache_transformed_weights: bool = False,
        recompute_backward: bool = False,
        projection_backend: str = "eager",
        persistent_indices: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(input_dim, bool) or int(input_dim) != input_dim or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if (
            isinstance(output_dim, bool)
            or int(output_dim) != output_dim
            or output_dim < 1
        ):
            raise ValueError("output_dim must be a positive integer")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight_transform = weight_transform
        self.projection_backend = normalize_indexed_projection_options(
            projection_backend,
            recompute_backward=bool(recompute_backward),
        )
        degree_tensor = _validate_degree_vector(
            degrees,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
        )
        total_contacts = int(degree_tensor.sum().item())
        if require_source_coverage and total_contacts < self.input_dim:
            raise ValueError(
                "source coverage requires at least input_dim total contacts; "
                f"got {total_contacts} < {self.input_dim}"
            )
        self.require_source_coverage = bool(require_source_coverage)
        self.register_buffer("degrees", degree_tensor)

        if connection_mask is None:
            normalized_mask = torch.empty(0, dtype=torch.bool)
            candidate_slots = self.output_dim * self.input_dim
            mask_source = "seeded_balanced_indices"
            connection_rows = _balanced_connection_rows(
                degree_tensor,
                input_dim=self.input_dim,
                seed=int(seed),
            )
        else:
            normalized_mask = _validate_connection_mask(
                connection_mask,
                degrees=degree_tensor,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
            )
            if self.require_source_coverage and bool(
                (~normalized_mask.any(dim=0)).any()
            ):
                missing = torch.nonzero(
                    ~normalized_mask.any(dim=0), as_tuple=False
                ).flatten()
                raise ValueError(
                    "connection_mask excludes sources required for source coverage: "
                    f"{missing.tolist()}"
                )
            candidate_slots = int(normalized_mask.sum().item())
            mask_source = "configured_mask_seeded_balanced_indices"
            connection_rows = _balanced_masked_connection_rows(
                degree_tensor,
                connection_mask=normalized_mask,
                seed=int(seed),
            )
        self.register_buffer(
            "_allowed_connection_mask",
            normalized_mask,
            persistent=False,
        )
        self._candidate_slots = int(candidate_slots)
        self._mask_source = mask_source
        self.groups = nn.ModuleList()
        for degree in sorted(set(degree_tensor.tolist())):
            if degree == 0:
                continue
            output_indices = torch.nonzero(
                degree_tensor == degree,
                as_tuple=False,
            ).flatten()
            connection_indices = torch.stack(
                [connection_rows[index] for index in output_indices.tolist()]
            )
            self.groups.append(
                _IndexedDegreeGroup(
                    input_dim=self.input_dim,
                    output_indices=output_indices,
                    connection_indices=connection_indices,
                    degree=degree,
                    weight_transform=weight_transform,
                    row_sum_init=row_sum_init,
                    output_chunk_size=output_chunk_size,
                    index_dtype=index_dtype,
                    workspace_mb=workspace_mb,
                    cache_transformed_weights=cache_transformed_weights,
                    recompute_backward=recompute_backward,
                    projection_backend=self.projection_backend,
                    persistent_indices=persistent_indices,
                )
            )

        if self.require_source_coverage and bool((self.source_outdegrees() == 0).any()):
            raise RuntimeError("failed to construct the required source coverage")

    @property
    def total_contacts(self) -> int:
        return int(self.degrees.sum().item())

    def connection_indices_padded(self) -> torch.Tensor:
        """Return ``[output_dim, max_degree]`` indices with ``-1`` padding."""

        max_degree = int(self.degrees.max().item())
        padded = torch.full(
            (self.output_dim, max_degree),
            -1,
            dtype=torch.long,
            device=self.degrees.device,
        )
        for group in self.groups:
            degree = group.projection.K
            padded[group.output_indices, :degree] = (
                group.projection.connection_indices.to(padded.device)
            )
        return padded

    def source_outdegrees(self) -> torch.Tensor:
        """Count how often each source coordinate appears in the fixed support."""

        counts = torch.zeros(
            self.input_dim,
            dtype=torch.long,
            device=self.degrees.device,
        )
        for group in self.groups:
            indices = group.projection.connection_indices.to(counts.device).flatten()
            counts = counts + torch.bincount(indices, minlength=self.input_dim)
        return counts

    def connectivity_resource_counts(self) -> dict[str, int | float | str]:
        """Return exact compact connectivity counts for resource ledgers."""

        return {
            "out_features": self.output_dim,
            "in_features": self.input_dim,
            "candidate_parameters": self.total_contacts,
            "candidate_slots": self._candidate_slots,
            "active_synapses": self.total_contacts,
            "realized_k_min": int(self.degrees.min().item()),
            "realized_k_max": int(self.degrees.max().item()),
            "realized_k_mean": float(self.degrees.float().mean().item()),
            "selection_policy": "indexed_fixed_degree_grouped",
            "mask_source": self._mask_source,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")
        if x.ndim < 1 or x.shape[-1] != self.input_dim:
            raise ValueError(
                f"input must end in dimension {self.input_dim}, got {tuple(x.shape)}"
            )
        output = x.new_zeros(*x.shape[:-1], self.output_dim)
        for group in self.groups:
            values = group.projection(x)
            output = torch.index_copy(
                output,
                -1,
                group.output_indices.to(output.device),
                values,
            )
        return output


__all__ = ["DegreeGroupedIndexedLinear"]
