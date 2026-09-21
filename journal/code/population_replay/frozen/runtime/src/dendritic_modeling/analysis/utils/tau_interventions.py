"""Same-checkpoint interventions on StatefulDendriNet time constants.

The context manager in this module changes only the tensor from which a
``StatefulDendriNet`` computes its per-level decay factors.  It supports both
fixed taus (stored as ``decays``) and learned taus (stored as ``log_tau``), and
restores the original tensor exactly when the context exits.

This is an inference-time causal intervention.  It does not edit configuration
files, checkpoint files, weights, connectivity, or optimizer state.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)


@dataclass(frozen=True)
class TauOverrideAudit:
    """Immutable description of one temporary per-level tau intervention."""

    module_type: str
    state_field: str
    learnable_tau: bool
    dt: float
    original_taus: tuple[float, ...]
    override_taus: tuple[float, ...]
    state_tensor_noop: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe intervention metadata."""

        return {
            "module_type": self.module_type,
            "state_field": self.state_field,
            "learnable_tau": self.learnable_tau,
            "dt": self.dt,
            "original_taus": list(self.original_taus),
            "override_taus": list(self.override_taus),
            "state_tensor_noop": self.state_tensor_noop,
        }


def _validated_taus(
    module: StatefulDendriNet,
    level_taus: Sequence[float],
) -> tuple[float, ...]:
    if isinstance(level_taus, (str, bytes)):
        raise TypeError("level_taus must be a sequence of positive finite numbers")
    values = tuple(float(value) for value in level_taus)
    if len(values) != int(module.n_levels):
        raise ValueError(
            f"Expected {module.n_levels} level taus, received {len(values)}"
        )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("level_taus must contain only positive finite values")
    if not math.isfinite(float(module.dt)) or float(module.dt) <= 0.0:
        raise ValueError("StatefulDendriNet.dt must be positive and finite")
    return values


def _tau_state_tensor(module: StatefulDendriNet) -> tuple[str, torch.Tensor]:
    if bool(module.learnable_tau):
        if not isinstance(module.log_tau, torch.Tensor):
            raise TypeError("learnable StatefulDendriNet has no log_tau tensor")
        return "log_tau", module.log_tau
    decays = getattr(module, "decays", None)
    if not isinstance(decays, torch.Tensor):
        raise TypeError("fixed-tau StatefulDendriNet has no decays tensor")
    return "decays", decays


def _replacement_state(
    module: StatefulDendriNet,
    level_taus: tuple[float, ...],
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    if bool(module.learnable_tau):
        values = [math.log(tau) for tau in level_taus]
    else:
        values = [math.exp(-float(module.dt) / tau) for tau in level_taus]
    return torch.tensor(values, device=reference.device, dtype=reference.dtype)


@contextmanager
def temporary_level_tau_override(
    module: StatefulDendriNet,
    level_taus: Sequence[float],
) -> Iterator[TauOverrideAudit]:
    """Temporarily replace one population's per-level time constants.

    The original state tensor is cloned before mutation and copied back in a
    ``finally`` block.  Supplying the native configured taus is a byte-level
    no-op when those values reproduce the stored tensor, which is exposed by
    ``TauOverrideAudit.state_tensor_noop`` for an explicit equivalence audit.

    Args:
        module: Exact ``StatefulDendriNet`` population to intervene on.
        level_taus: Positive per-level time constants in the module's native
            distal-to-somatic order.

    Yields:
        A JSON-serializable audit record describing the transient change.
    """

    if not isinstance(module, StatefulDendriNet):
        raise TypeError(
            f"tau intervention requires StatefulDendriNet, got {type(module).__name__}"
        )
    taus = _validated_taus(module, level_taus)
    state_field, state_tensor = _tau_state_tensor(module)
    original_state = state_tensor.detach().clone()
    original_taus = tuple(
        float(value) for value in module.current_taus.detach().to(device="cpu").tolist()
    )
    replacement = _replacement_state(module, taus, reference=state_tensor)
    if replacement.shape != state_tensor.shape:
        raise RuntimeError("tau replacement tensor shape changed unexpectedly")
    audit = TauOverrideAudit(
        module_type=type(module).__name__,
        state_field=state_field,
        learnable_tau=bool(module.learnable_tau),
        dt=float(module.dt),
        original_taus=original_taus,
        override_taus=taus,
        state_tensor_noop=torch.equal(original_state, replacement),
    )
    try:
        with torch.no_grad():
            state_tensor.copy_(replacement)
        yield audit
    finally:
        with torch.no_grad():
            state_tensor.copy_(original_state)


__all__ = ["TauOverrideAudit", "temporary_level_tau_override"]
