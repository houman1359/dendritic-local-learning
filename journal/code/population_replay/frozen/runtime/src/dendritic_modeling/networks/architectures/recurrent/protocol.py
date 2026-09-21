"""Shared protocol for recurrent cores used by trainers and analysis tools."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class RecurrentCore(Protocol):
    """Minimal sequence-core interface used by recurrent trainers.

    The protocol is intentionally small: trainers need a sequence ``forward``
    method, an ``output_dim`` for decoder construction, and an ``is_recurrent``
    flag. More specialized analysis tools can inspect optional implementation
    details such as ``layers`` or ``populations`` through duck-typed helpers.
    """

    @property
    def output_dim(self) -> int:
        """Feature dimension returned at each recurrent step."""

    @property
    def is_recurrent(self) -> bool:
        """Whether this core consumes temporal state."""

    def forward(
        self,
        x: torch.Tensor,
        hidden: Any | None = None,
        return_hidden: bool = False,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Run a complete input sequence."""


@runtime_checkable
class SteppableRecurrentCore(Protocol):
    """Lower-level one-step interface for recurrent populations."""

    @property
    def output_dim(self) -> int:
        """Feature dimension returned at each recurrent step."""

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Any:
        """Create a zero recurrent state for a batch."""

    def step(
        self,
        x: torch.Tensor,
        state: Any | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Any]:
        """Advance the recurrent state by one input step."""


__all__ = ["RecurrentCore", "SteppableRecurrentCore"]
