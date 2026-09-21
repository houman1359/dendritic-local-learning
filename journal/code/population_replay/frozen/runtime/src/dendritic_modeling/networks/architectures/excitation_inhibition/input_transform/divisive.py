"""Fixed global or grouped divisive input-normalization controls."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

import torch
from torch import nn


class DivisiveInputNormalizer(nn.Module):
    """Normalize input magnitude globally or within a declared feature partition.

    This module is a non-learned comparison control.  ``global_l1`` removes one
    common magnitude direction.  ``group_l1`` applies the same operation to
    each configured feature range and can therefore remove several independent
    group magnitudes.  The default ``none`` mode is exactly the identity.
    """

    MODES = frozenset({"none", "global_l1", "group_l1"})

    def __init__(
        self,
        input_dim: int,
        *,
        mode: str = "none",
        group_ranges: list[list[int]] | tuple[tuple[int, int], ...] | None = None,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if (
            isinstance(input_dim, bool)
            or not isinstance(input_dim, int)
            or input_dim <= 0
        ):
            raise ValueError("input_dim must be a positive integer")
        self.input_dim = input_dim
        self.mode = str(mode).strip().lower()
        if self.mode not in self.MODES:
            choices = ", ".join(sorted(self.MODES))
            raise ValueError(f"input normalization mode must be one of: {choices}")
        if isinstance(epsilon, bool) or not isinstance(epsilon, Real):
            raise TypeError("normalization epsilon must be a real number")
        self.epsilon = float(epsilon)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("normalization epsilon must be finite and positive")

        resolved_ranges = self._resolve_ranges(group_ranges)
        self._group_ranges = resolved_ranges
        starts = torch.tensor(
            [start for start, _stop in resolved_ranges], dtype=torch.long
        )
        stops = torch.tensor(
            [stop for _start, stop in resolved_ranges], dtype=torch.long
        )
        self.group_starts: torch.Tensor
        self.group_stops: torch.Tensor
        # The ranges are derived entirely from configuration.  Keeping these
        # buffers non-persistent preserves strict loading of checkpoints saved
        # before input normalization was introduced while still moving them
        # with the module across devices.
        self.register_buffer("group_starts", starts, persistent=False)
        self.register_buffer("group_stops", stops, persistent=False)

    @classmethod
    def from_config(
        cls,
        input_dim: int,
        config: dict[str, Any] | None,
    ) -> DivisiveInputNormalizer:
        values = dict(config or {})
        allowed = {"mode", "group_ranges", "epsilon"}
        unexpected = set(values).difference(allowed)
        if unexpected:
            raise ValueError(
                "unsupported input normalization option(s): "
                + ", ".join(sorted(unexpected))
            )
        return cls(input_dim, **values)

    def _resolve_ranges(
        self,
        group_ranges: list[list[int]] | tuple[tuple[int, int], ...] | None,
    ) -> tuple[tuple[int, int], ...]:
        if self.mode != "group_l1":
            if group_ranges not in (None, [], ()):
                raise ValueError("group_ranges are only valid for mode='group_l1'")
            return ()
        if not isinstance(group_ranges, (list, tuple)) or not group_ranges:
            raise ValueError("mode='group_l1' requires non-empty group_ranges")
        resolved = []
        coverage = torch.zeros(self.input_dim, dtype=torch.int64)
        for bounds in group_ranges:
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ValueError("each group range must be a [start, stop] pair")
            start, stop = bounds
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in bounds
            ):
                raise TypeError("group range bounds must be integers")
            if start < 0 or stop > self.input_dim or stop <= start:
                raise ValueError(
                    "group ranges must satisfy 0 <= start < stop <= input_dim"
                )
            coverage[start:stop] += 1
            resolved.append((start, stop))
        if not torch.all(coverage == 1):
            raise ValueError(
                "group_ranges must form an exact non-overlapping partition"
            )
        return tuple(resolved)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError("input value must be a tensor")
        if value.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected last input dimension {self.input_dim}, got {value.shape[-1]}"
            )
        if not value.is_floating_point():
            raise TypeError("input value must be floating point")
        if self.mode == "none":
            return value
        if self.mode == "global_l1":
            denominator = value.abs().sum(dim=-1, keepdim=True)
            return value / (denominator + self.epsilon)

        output = torch.empty_like(value)
        for start_index, stop_index in self._group_ranges:
            group = value[..., start_index:stop_index]
            denominator = group.abs().sum(dim=-1, keepdim=True)
            output[..., start_index:stop_index] = group / (denominator + self.epsilon)
        return output


__all__ = ["DivisiveInputNormalizer"]
