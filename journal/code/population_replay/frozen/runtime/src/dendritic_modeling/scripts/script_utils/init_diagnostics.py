"""Gate diagnostics for models with learnable reactivation gates.

Captures a summary of reactivation gate parameters (m, b) for each dendritic
layer immediately after the model is built and before training begins. The
default output is written to ``<run_dir>/init_gate_stats.json`` immediately
after model construction.  Data-driven training paths additionally write
``post_calibration_gate_stats.json`` after calibration, so the requested policy
can be audited from the executed state rather than inferred from configuration.

This is intentionally lightweight: it does not depend on experiment-specific
task logic, and it only inspects modules that already exist in the built model.
The main goal is to catch suspicious gate states early, before a long training
run silently fails because the reactivation gates started saturated or clipped.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.utils.hooks import iter_named_modules_matching

logger = logging.getLogger(__name__)


def _extract_m(module: nn.Module) -> torch.Tensor | None:
    """Return a flattened tensor of m values, or None if not available."""
    if hasattr(module, "log_m") and torch.is_tensor(module.log_m):
        return module.log_m.detach().exp().flatten().cpu()
    if hasattr(module, "m") and torch.is_tensor(module.m):
        return module.m.detach().flatten().cpu()
    return None


def _extract_b(module: nn.Module) -> torch.Tensor | None:
    """Return a flattened tensor of b values, or None if not available."""
    if hasattr(module, "b") and torch.is_tensor(module.b):
        return module.b.detach().flatten().cpu()
    if hasattr(module, "fixed_b"):
        val = float(module.fixed_b)
        return torch.tensor([val])
    return None


def _tensor_stats(values: torch.Tensor, tol: float = 1e-6) -> dict[str, Any]:
    """Summarise a tensor for diagnostic output (min / max / mean / n_unique)."""
    if values.numel() == 0:
        return {"n": 0}
    v = values.double()
    # Count unique values (with tolerance) — the key regression signal.
    # Rounded to avoid spurious differences from float noise.
    rounded = torch.round(v / tol) * tol
    n_unique = int(torch.unique(rounded).numel())
    return {
        "n": int(v.numel()),
        "n_unique": n_unique,
        "min": float(v.min().item()),
        "max": float(v.max().item()),
        "mean": float(v.mean().item()),
        "std": float(v.std(unbiased=False).item()) if v.numel() > 1 else 0.0,
    }


def collect_reactivation_stats(model: nn.Module) -> dict[str, Any]:
    """Walk the model and collect (m, b) stats for every reactivation module.

    Returns a dict keyed by module qualified name.
    """
    stats: dict[str, Any] = {}
    for name, module in iter_named_modules_matching(
        model,
        lambda module_name, _module: module_name.endswith("reactivation"),
    ):
        # Heuristic: any module named ".reactivation" with an m or b attribute.
        m_vals = _extract_m(module)
        b_vals = _extract_b(module)
        if m_vals is None and b_vals is None:
            continue
        entry: dict[str, Any] = {"class": module.__class__.__name__}
        if m_vals is not None:
            entry["m"] = _tensor_stats(m_vals)
        if b_vals is not None:
            entry["b"] = _tensor_stats(b_vals)
        stats[name] = entry
    return stats


def _flag_suspicious(stats: dict[str, Any], m_clip_warn: float = 40.0) -> list[str]:
    """Return a list of human-readable warnings about suspicious init patterns.

    Warnings are intentionally conservative. Identical values alone are not
    automatically bad, because some fixed or analytical init rules legitimately
    start every neuron in a layer from the same gate state. We only flag them
    when they coincide with clear saturation or clipping signals.
    """
    warnings: list[str] = []
    for name, entry in stats.items():
        m = entry.get("m") or {}
        b = entry.get("b") or {}
        m_n = m.get("n", 0)
        b_n = b.get("n", 0)
        m_unique = m.get("n_unique", 0)
        b_unique = b.get("n_unique", 0)
        m_max = float(m.get("max", 0.0))
        m_mean = float(m.get("mean", 0.0))
        bmean = float(b.get("mean", 0.0))

        identical_m = m_n > 1 and m_unique == 1
        identical_b = b_n > 1 and b_unique == 1
        m_clipped = m_max >= m_clip_warn
        # Upper-edge (b >= 0.95) saturates regardless of m, because shunting
        # voltages are bounded in (0,1). Lower-edge (b <= 0.05) is only a
        # saturation risk with a very sharp slope. Moderate fixed gates
        # centered at zero are healthy for normalized additive recurrent
        # voltages, so do not warn on m ~= 1, b = 0.
        b_saturated = bmean >= 0.95 or (bmean <= 0.05 and m_mean >= 5.0)

        # Always flag a clipped slope regardless of variance. Even when a run
        # recovers later, starting from a clipped gate is usually worth review.
        if m_clipped:
            warnings.append(
                f"{name}: m.max={m_max:.2f} is near or at the calibration "
                f"clip ({m_clip_warn})"
            )

        # Flag identical-value init only if combined with saturation or clip,
        # since identical values alone can be a healthy fixed-init baseline.
        if (identical_m or identical_b) and (m_clipped or b_saturated):
            warnings.append(
                f"{name}: identical (m, b) across {max(m_n, b_n)} entries "
                f"combined with m_max={m_max:.2f}, b.mean={bmean:.3f}"
            )

        # Flag bare saturation.
        if b_saturated and b_n > 0 and not (identical_b and not m_clipped):
            warnings.append(f"{name}: b.mean={bmean:.3f} is near saturation edge")

    return warnings


def dump_init_gate_stats(
    model: nn.Module, run_dir: str | Path, name: str = "init_gate_stats.json"
) -> dict[str, Any]:
    """Collect reactivation gate stats and write them to the run directory.

    Returns the payload that was written (also returned for logging convenience).
    """
    stats = collect_reactivation_stats(model)
    warnings = _flag_suspicious(stats)
    payload = {
        "reactivation_modules": stats,
        "warnings": warnings,
        "n_reactivation_modules": len(stats),
    }

    path = Path(run_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    stage = "Post-calibration" if "post_calibration" in name else "Post-build"
    if warnings:
        logger.warning(
            "%s gate diagnostic flagged %d suspicious patterns (see %s):",
            stage,
            len(warnings),
            path,
        )
        for warning in warnings:
            logger.warning("  %s", warning)
    else:
        logger.info(
            "%s gate diagnostic clean (%d reactivation modules, written to %s).",
            stage,
            len(stats),
            path,
        )

    return payload
