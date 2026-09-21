"""Named, matrix-free response operators for recurrent state coordinates.

The full recurrent state can be very large.  These helpers address one declared
population/trace/level block at a time and compose the existing matrix-free
state-transition JVPs.  They therefore quantify a factorization-relative
intervention without materializing the full state Jacobian.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from dendritic_modeling.analysis.utils.state_transition import (
    BlockSubspaceSingularValueEstimate,
    HeterogeneousLeakStateCodec,
    PopulationStateCodec,
    StateTensorSpec,
    StateTransitionLinearization,
    StateTransitionProduct,
    leading_matrix_free_singular_value_block_subspace,
)

StateCodec = PopulationStateCodec | HeterogeneousLeakStateCodec


@dataclass(frozen=True)
class StateCoordinateSelector:
    """Select one carried state tensor by its declared model coordinates.

    Dendritic ``level_index`` follows the model's distal-to-soma convention.
    ``level_from_soma=0`` names the soma level, ``1`` its immediate children,
    and so on.  Exactly one convention must be supplied for a level-indexed
    trace.  Non-level state such as ``outputs`` uses neither.
    """

    layer_index: int
    population_name: str
    field: str
    level_index: int | None = None
    level_from_soma: int | None = None


@dataclass(frozen=True)
class ResolvedStateCoordinate:
    """One exact tensor slice in a canonical recurrent-state vector."""

    spec: StateTensorSpec
    full_dimension: int
    device: torch.device
    dtype: torch.dtype

    @property
    def dimension(self) -> int:
        return self.spec.numel

    def embed(self, values: torch.Tensor) -> torch.Tensor:
        """Embed a coordinate vector into an otherwise-zero full state."""

        self._validate_coordinate_vector(values, "values")
        full = torch.zeros(
            self.full_dimension,
            device=self.device,
            dtype=self.dtype,
        )
        return full.index_copy(
            0,
            torch.arange(self.spec.start, self.spec.stop, device=self.device),
            values,
        )

    def project(self, state_vector: torch.Tensor) -> torch.Tensor:
        """Project a full state vector onto this coordinate block."""

        if not isinstance(state_vector, torch.Tensor):
            raise TypeError("state_vector must be a tensor")
        if state_vector.shape != (self.full_dimension,):
            raise ValueError(
                f"state_vector must have shape ({self.full_dimension},), got "
                f"{tuple(state_vector.shape)}"
            )
        if state_vector.device != self.device or state_vector.dtype != self.dtype:
            raise ValueError("state_vector device/dtype does not match the coordinate")
        return state_vector[self.spec.start : self.spec.stop]

    def metadata(self) -> dict[str, object]:
        return {
            "path": self.spec.path,
            "layer_index": self.spec.layer_index,
            "layer_name": self.spec.layer_name,
            "population_name": self.spec.population_name,
            "field": self.spec.field,
            "level_index_distal_to_soma": self.spec.level_index,
            "shape": list(self.spec.shape),
            "dimension": self.dimension,
        }

    def _validate_coordinate_vector(self, vector: torch.Tensor, name: str) -> None:
        if not isinstance(vector, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"{name} must have shape ({self.dimension},), got {tuple(vector.shape)}"
            )
        if vector.device != self.device or vector.dtype != self.dtype:
            raise ValueError(f"{name} device/dtype does not match the coordinate")


@dataclass(frozen=True)
class InputFeatureCoordinate:
    """A fixed subset of input features, repeated across the reference batch."""

    feature_indices: tuple[int, ...]

    def resolve(self, linearization: StateTransitionLinearization) -> ResolvedInput:
        if not self.feature_indices:
            raise ValueError("feature_indices must not be empty")
        indices: list[int] = []
        for index in self.feature_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("feature_indices must contain integers")
            if index < 0 or index >= linearization.input_t.shape[1]:
                raise ValueError(
                    f"input feature index {index} is outside "
                    f"[0, {linearization.input_t.shape[1]})"
                )
            indices.append(index)
        if len(set(indices)) != len(indices):
            raise ValueError("feature_indices must be unique")
        return ResolvedInput(
            feature_indices=tuple(indices),
            batch_size=int(linearization.input_t.shape[0]),
            input_dim=int(linearization.input_t.shape[1]),
            device=linearization.device,
            dtype=linearization.dtype,
        )


@dataclass(frozen=True)
class ResolvedInput:
    """Resolved input-feature subspace for an input-to-state JVP."""

    feature_indices: tuple[int, ...]
    batch_size: int
    input_dim: int
    device: torch.device
    dtype: torch.dtype

    @property
    def dimension(self) -> int:
        return self.batch_size * len(self.feature_indices)

    def embed(self, values: torch.Tensor) -> torch.Tensor:
        self._validate_coordinate_vector(values, "values")
        matrix = torch.zeros(
            (self.batch_size, self.input_dim),
            device=self.device,
            dtype=self.dtype,
        )
        return matrix.index_copy(
            1,
            torch.tensor(self.feature_indices, device=self.device),
            values.view(self.batch_size, len(self.feature_indices)),
        )

    def project(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("input_tensor must be a tensor")
        if input_tensor.shape != (self.batch_size, self.input_dim):
            raise ValueError(
                "input_tensor must have shape "
                f"({self.batch_size}, {self.input_dim}), got "
                f"{tuple(input_tensor.shape)}"
            )
        if input_tensor.device != self.device or input_tensor.dtype != self.dtype:
            raise ValueError("input_tensor device/dtype does not match the coordinate")
        return input_tensor[:, self.feature_indices].reshape(-1)

    def metadata(self) -> dict[str, object]:
        return {
            "feature_indices": list(self.feature_indices),
            "batch_size": self.batch_size,
            "input_dim": self.input_dim,
            "dimension": self.dimension,
        }

    def _validate_coordinate_vector(self, vector: torch.Tensor, name: str) -> None:
        if not isinstance(vector, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"{name} must have shape ({self.dimension},), got {tuple(vector.shape)}"
            )
        if vector.device != self.device or vector.dtype != self.dtype:
            raise ValueError(f"{name} device/dtype does not match the coordinate")


@dataclass(frozen=True)
class NamedCoordinateGainEstimate:
    """Leading finite-horizon gain between two declared coordinate blocks."""

    source_metadata: dict[str, object]
    target_metadata: dict[str, object]
    horizon_steps: int
    estimate: BlockSubspaceSingularValueEstimate

    def metadata(self) -> dict[str, object]:
        return {
            "source": self.source_metadata,
            "target": self.target_metadata,
            "horizon_steps": self.horizon_steps,
            "estimate": self.estimate.metadata(),
        }


def resolve_state_coordinate(
    codec: StateCodec,
    selector: StateCoordinateSelector,
) -> ResolvedStateCoordinate:
    """Resolve a selector to exactly one canonical state tensor or fail closed."""

    if isinstance(selector.layer_index, bool) or not isinstance(
        selector.layer_index, int
    ):
        raise TypeError("layer_index must be an integer")
    if not selector.population_name or not selector.field:
        raise ValueError("population_name and field must be non-empty")
    candidates = [
        spec
        for spec in codec.specs
        if spec.layer_index == selector.layer_index
        and spec.population_name == selector.population_name
        and spec.field == selector.field
    ]
    if not candidates:
        raise ValueError(
            "state coordinate does not exist: "
            f"layer={selector.layer_index}, population={selector.population_name!r}, "
            f"field={selector.field!r}"
        )

    level_index = selector.level_index
    if level_index is not None:
        if isinstance(level_index, bool) or not isinstance(level_index, int):
            raise TypeError("level_index must be an integer")
        if level_index < 0:
            raise ValueError("level_index must be non-negative")
    if selector.level_from_soma is not None:
        if level_index is not None:
            raise ValueError("specify only one of level_index and level_from_soma")
        if isinstance(selector.level_from_soma, bool) or not isinstance(
            selector.level_from_soma, int
        ):
            raise TypeError("level_from_soma must be an integer")
        if selector.level_from_soma < 0:
            raise ValueError("level_from_soma must be non-negative")
        indexed = [spec for spec in candidates if spec.level_index is not None]
        if not indexed:
            raise ValueError(f"field {selector.field!r} is not level-indexed")
        soma_level_index = max(int(spec.level_index) for spec in indexed)
        level_index = soma_level_index - selector.level_from_soma

    level_indexed = any(spec.level_index is not None for spec in candidates)
    if level_indexed and level_index is None:
        raise ValueError(
            f"field {selector.field!r} is level-indexed; specify level_index or "
            "level_from_soma"
        )
    if not level_indexed and level_index is not None:
        raise ValueError(f"field {selector.field!r} is not level-indexed")
    matches = [spec for spec in candidates if spec.level_index == level_index]
    if len(matches) != 1:
        raise ValueError(
            f"state coordinate resolved to {len(matches)} tensors; requested "
            f"level_index={level_index}"
        )
    return ResolvedStateCoordinate(
        spec=matches[0],
        full_dimension=codec.dimension,
        device=codec.device,
        dtype=codec.dtype,
    )


def leading_named_state_gain(
    product: StateTransitionProduct,
    *,
    source: StateCoordinateSelector,
    target: StateCoordinateSelector,
    block_size: int | None = None,
    max_iterations: int | None = None,
    relative_residual_tolerance: float = 1e-3,
    absolute_residual_tolerance: float = 1e-7,
    seed: int = 0,
) -> NamedCoordinateGainEstimate:
    """Estimate the largest gain from one named state block to another."""

    source_coordinate = resolve_state_coordinate(
        product.linearizations[0].codec, source
    )
    target_coordinate = resolve_state_coordinate(
        product.linearizations[-1].codec, target
    )
    estimate = leading_matrix_free_singular_value_block_subspace(
        input_dimension=source_coordinate.dimension,
        output_dimension=target_coordinate.dimension,
        device=product.device,
        dtype=product.dtype,
        matvec=lambda vector: target_coordinate.project(
            product.jvp(source_coordinate.embed(vector))
        ),
        rmatvec=lambda cotangent: source_coordinate.project(
            product.vjp(target_coordinate.embed(cotangent))
        ),
        block_size=block_size,
        max_iterations=max_iterations,
        relative_residual_tolerance=relative_residual_tolerance,
        absolute_residual_tolerance=absolute_residual_tolerance,
        seed=seed,
    )
    return NamedCoordinateGainEstimate(
        source_metadata=source_coordinate.metadata(),
        target_metadata=target_coordinate.metadata(),
        horizon_steps=len(product.linearizations),
        estimate=estimate,
    )


def leading_input_to_named_state_gain(
    linearizations: Sequence[StateTransitionLinearization],
    *,
    source: InputFeatureCoordinate,
    target: StateCoordinateSelector,
    block_size: int | None = None,
    max_iterations: int | None = None,
    relative_residual_tolerance: float = 1e-3,
    absolute_residual_tolerance: float = 1e-7,
    seed: int = 0,
) -> NamedCoordinateGainEstimate:
    """Estimate gain from selected input features at step one to a later state."""

    product = StateTransitionProduct(linearizations)
    first = product.linearizations[0]
    source_coordinate = source.resolve(first)
    target_coordinate = resolve_state_coordinate(
        product.linearizations[-1].codec, target
    )

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        propagated = first.input_jvp(source_coordinate.embed(vector))
        for linearization in product.linearizations[1:]:
            propagated = linearization.jvp(propagated)
        return target_coordinate.project(propagated)

    def rmatvec(cotangent: torch.Tensor) -> torch.Tensor:
        propagated = target_coordinate.embed(cotangent)
        for linearization in reversed(product.linearizations[1:]):
            propagated = linearization.vjp(propagated)
        return source_coordinate.project(first.input_vjp(propagated))

    estimate = leading_matrix_free_singular_value_block_subspace(
        input_dimension=source_coordinate.dimension,
        output_dimension=target_coordinate.dimension,
        device=product.device,
        dtype=product.dtype,
        matvec=matvec,
        rmatvec=rmatvec,
        block_size=block_size,
        max_iterations=max_iterations,
        relative_residual_tolerance=relative_residual_tolerance,
        absolute_residual_tolerance=absolute_residual_tolerance,
        seed=seed,
    )
    return NamedCoordinateGainEstimate(
        source_metadata={"kind": "input_features", **source_coordinate.metadata()},
        target_metadata=target_coordinate.metadata(),
        horizon_steps=len(product.linearizations),
        estimate=estimate,
    )


__all__ = [
    "InputFeatureCoordinate",
    "NamedCoordinateGainEstimate",
    "ResolvedInput",
    "ResolvedStateCoordinate",
    "StateCoordinateSelector",
    "leading_input_to_named_state_gain",
    "leading_named_state_gain",
    "resolve_state_coordinate",
]
