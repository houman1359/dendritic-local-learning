"""Opt-in NVTX ranges for Nsight Systems timelines.

Enabled only when the environment variable ``DENDRITIC_NVTX=1`` is set and
CUDA is available, so instrumented code paths carry zero overhead by
default. Use coarse ranges (region / stage granularity), never inside hot
per-chunk loops.
"""

from __future__ import annotations

import contextlib
import os

import torch

_ENABLED = os.environ.get("DENDRITIC_NVTX", "0") == "1"

__all__ = ["nvtx_enabled", "nvtx_range"]


def nvtx_enabled() -> bool:
    return _ENABLED and torch.cuda.is_available()


def nvtx_range(name: str):
    """Context manager emitting an NVTX range when profiling is enabled."""
    if nvtx_enabled():
        return torch.cuda.nvtx.range(name)
    return contextlib.nullcontext()
