"""Backend selection for fixed indexed sparse projections.

The portable eager and recompute implementations live in ``indexed_common``.
This module owns selection only; the optional Triton dependency stays lazy so
importing or constructing the default layer never compiles a kernel.
"""

from __future__ import annotations

from typing import Literal

import torch

IndexedProjectionBackend = Literal[
    "eager",
    "recompute",
    "triton_transposed",
    "triton_fused",
    "triton_ell",
    "auto",
]

INDEXED_PROJECTION_BACKENDS = frozenset(
    {
        "auto",
        "eager",
        "recompute",
        "triton_ell",
        "triton_fused",
        "triton_transposed",
    }
)

#: ``auto`` prefers the fused chunked kernel at and above this input width on a
#: Triton-capable CUDA device (kernel-search round 1, 2026-08-18: 3.4x H100 /
#: 8.5x Blackwell over chunked recompute at giant-IN M=4 recurrent shapes).
_AUTO_FUSED_MIN_IN_FEATURES = 100_000


def normalize_indexed_projection_backend(value: object) -> str:
    """Return a validated indexed-projection backend name."""
    normalized = str(value).strip().lower()
    if normalized == "triton":
        # Retired 2026-08-17: the batch-tiled Triton gather lost to the
        # transposed-coalesced kernel at every measured shape (2.4-21x,
        # H100 and Blackwell; docs/kernel_provenance.rst). Old configs
        # keep working and get the faster kernel.
        import warnings

        warnings.warn(
            "indexed_projection_backend='triton' is retired; resolving to "
            "'triton_transposed'",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized = "triton_transposed"
    if normalized == "cuda":
        # Retired 2026-08-17: beaten by the transposed Triton kernel at
        # every measured feedforward shape, and the JIT extension no
        # longer builds on current cluster nodes (H100 + Blackwell). The
        # sources remain in git history and the curated kernel lineage
        # (docs/kernel_provenance.rst) if an architecture where it wins
        # returns. 'auto' picks the best available backend instead.
        import warnings

        warnings.warn(
            "indexed_projection_backend='cuda' is retired; resolving to "
            "'auto'",
            DeprecationWarning,
            stacklevel=2,
        )
        normalized = "auto"
    if normalized not in INDEXED_PROJECTION_BACKENDS:
        choices = ", ".join(sorted(INDEXED_PROJECTION_BACKENDS))
        raise ValueError(
            f"indexed_projection_backend must be one of {{{choices}}}, "
            f"got {value!r}"
        )
    return normalized


def normalize_indexed_projection_options(
    backend: object,
    *,
    recompute_backward: bool,
) -> str:
    """Resolve the legacy recompute flag into the single backend selector."""
    normalized = normalize_indexed_projection_backend(backend)
    if recompute_backward:
        if normalized not in {"eager", "recompute"}:
            raise ValueError(
                "indexed_recompute_backward cannot be combined with "
                f"indexed_projection_backend={normalized!r}; use "
                "projection_backend='recompute' instead"
            )
        return "recompute"
    return normalized


def indexed_projection_backend_available(
    backend: object,
    *,
    device: torch.device | str | None = None,
) -> bool:
    """Return whether a backend is usable for the requested device."""
    normalized = normalize_indexed_projection_backend(backend)
    if normalized in {"auto", "eager", "recompute"}:
        # ``auto`` always has the portable recompute implementation available,
        # even when neither optional accelerator backend can run.
        return True

    resolved_device = torch.device(device) if device is not None else None
    if resolved_device is not None and resolved_device.type != "cuda":
        return False
    if not torch.cuda.is_available():
        return False

    if normalized == "triton_transposed":
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_indexed_gather_transposed import (
            triton_gather_available,
        )

        return triton_gather_available()
    if normalized == "triton_fused":
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_fused_chunked import (
            triton_fused_available,
        )

        return triton_fused_available()
    if normalized == "triton_ell":
        from dendritic_modeling.kernels.triton_ell import triton_available

        return triton_available()
    raise AssertionError(f"unhandled indexed projection backend: {normalized}")


def resolve_indexed_projection_backend(
    backend: object,
    *,
    device: torch.device | str,
    in_features: int | None = None,
) -> str:
    """Resolve an indexed backend without silently demoting explicit choices.

    ``in_features`` lets ``auto`` pick a shape-dependent kernel per layer;
    ``None`` preserves the shape-agnostic policy (the transposed kernel on a
    Triton-capable CUDA device, otherwise portable recompute).
    """
    normalized = normalize_indexed_projection_backend(backend)
    resolved_device = torch.device(device)
    if normalized in {"eager", "recompute"}:
        return normalized

    if normalized == "auto":
        # Policy from the 2026-08-17 backend decision matrix, amended by
        # kernel-search round 1 (2026-08-18): the transposed-coalesced
        # Triton kernel wins every measured feedforward/transformer shape
        # on H100 AND Blackwell (2.4-21x over the other sparse backends),
        # so auto prefers it wherever Triton runs. At giant input widths
        # (in_features >= 100k; the tiny-batch recurrent regime where
        # chunked recompute used to be the pinned recommendation), the
        # fused chunked kernel measured 3.4x (H100) / 8.5x (Blackwell)
        # over recompute, so auto resolves there when the layer's width
        # is known.
        if resolved_device.type == "cuda":
            if (
                in_features is not None
                and in_features >= _AUTO_FUSED_MIN_IN_FEATURES
                and indexed_projection_backend_available(
                    "triton_fused",
                    device=resolved_device,
                )
            ):
                return "triton_fused"
            if indexed_projection_backend_available(
                "triton_transposed",
                device=resolved_device,
            ):
                return "triton_transposed"
        return "recompute"

    if indexed_projection_backend_available(normalized, device=resolved_device):
        return normalized

    reason = "Triton and a CUDA device are required"
    raise RuntimeError(
        f"indexed_projection_backend={normalized!r} is unavailable on "
        f"device {resolved_device}: {reason}. Use 'auto', 'recompute', or 'eager' "
        "for a portable fallback."
    )


__all__ = [
    "INDEXED_PROJECTION_BACKENDS",
    "IndexedProjectionBackend",
    "indexed_projection_backend_available",
    "normalize_indexed_projection_backend",
    "normalize_indexed_projection_options",
    "resolve_indexed_projection_backend",
]
