"""Fused chunked Triton kernel for giant-input indexed sparse projections.

Computes, for a fixed sparse fan-in::

    out[m, o] = sum_k  x[m, connection_indices[o, k]] * weight[o, k]

with ONE kernel body that can also produce ``grad_w`` and ``grad_x`` in the
same launch. Drop-in autograd alternative to :func:`triton_sparse_gather`
(same public signature and dtype contract).

Design provenance
-----------------
Agentic kernel search, round 1 (2026-08-18); winning variant
``recompute_fused`` (outputs/kernel_search/variants/round1/recompute_fused.py).
This file is maintained work porting that tournament artifact. Measured at the
giant-IN M=4 recurrent shapes of record (IN=1M -> O=200k K=1000 and
IN=1M -> O=800k K=300): beats the chunked-recompute path -- the previous
recommendation for those shapes -- by 3.4x on H100 and 8.5x on Blackwell, and
the maintained transposed kernels by more. Correctness gates passed, including
non-divisible O/K/M edge tiles and duplicate indices.

The kernel keeps the recompute chunk loop's winning memory pattern -- per
output-tile processing, no ``[M, chunk, K]`` intermediates -- but collapses
the python chunk loop into the Triton grid and fuses the passes, so each
``idx``/``w`` tile is loaded once per launch and reused for every product:

* grid = (o-tiles, m-blocks); at the M~4 regime of record there is exactly
  one m-block (``B_M = next_pow2(M) <= 16``), so every batch lane is live --
  unlike the transposed kernels, whose swept ``B_M`` of 128 (fwd) / 32
  (grad_x) masks off ~97% / ~88% of their lanes at M=4.
* per k-tile, ``idx``/``w`` tiles ``[B_O, B_K]`` are loaded with coalesced
  row-vector loads and reused from registers:
    fwd    : ``acc[o, m] += sum_k xT[idx, m] * w`` (register accumulator)
    grad_w : ``gw[o, k] = sum_m xT[idx, m] * go[o, m]`` (direct coalesced
             store when one m-block covers M -- no atomics)
    grad_x : ``gxT[idx, m] += w * go[o, m]`` (coalesced fp32 atomics on the
             already-formed gather addresses)
* ``x`` is staged once as ``xT [IN, M]`` fp32 (16 MB at IN=1M, M=4): gather
  addresses ``idx*M + m`` are contiguous over the batch lanes and the table
  is L2-resident on H100/Blackwell, so gathers are cache traffic, not DRAM.

Autograd wiring (maintained; differs from the tournament harness): the
tournament measured ``train_step(x, idx, w, go)`` as a single launch, but
autograd only provides ``grad_output`` after the forward, so
:class:`_TritonFusedChunkedFunction` launches the same kernel twice -- a
forward-only pass (out) and a fused backward pass (grad_w + grad_x sharing
one ``idx``/``w`` read and one gather). :func:`fused_train_step` preserves
the tournament's measured one-launch entry for benchmarks and non-autograd
callers.

Numerics: all staging and accumulation is fp32 (bf16/fp16 inputs upcast once
on the host). Output is returned in ``x``'s dtype, ``grad_x`` in ``x``'s
dtype, ``grad_w`` in ``weight``'s dtype -- the shared indexed-backend
contract. ``grad_x`` uses atomics and is therefore run-to-run
nondeterministic at the last-ulp level, like the recompute path it replaces
(``grad_w`` too when M spans several m-blocks, i.e. M > 16 -- correct, merely
not the tuned path). No divisibility assumptions on O, K, or M (masked edge
tiles). Requires ``in_features * M < 2**31`` (int32 gather addresses); the
public entry raises past that bound rather than degrading.

Triton is an OPTIONAL dependency: importing this module never fails. Query
:func:`triton_fused_available` before selecting this path.
"""

from __future__ import annotations

import torch

try:  # Triton is an optional dependency -- never hard-fail on import
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    HAS_TRITON = False

# (B_O, B_K): output-tile and k-tile sizes, swept on Blackwell RTX 6000 at the
# shapes of record. B_K=128 gives 512B coalesced idx/w row segments; the sweep
# is flat within ~15% across (2-8, 32-128), so the choice is robust.
_TILES = (2, 128)
_NUM_WARPS = 8
_B_M_CAP = 16  # m-block cap for the general-M fallback path


def triton_fused_available() -> bool:
    """Whether the fused chunked Triton path can be used on this install."""
    return HAS_TRITON and torch.cuda.is_available()


if HAS_TRITON:

    @triton.jit
    def _fused_chunked_kernel(
        xT_ptr,  # fp32 [IN, M]
        goT_ptr,  # fp32 [O, M]   (read only when LOAD_GO)
        idx_ptr,  # int32 [O, K]
        w_ptr,  # fp32 [O, K]
        out_ptr,  # fp32 [M, O]   (written only when COMPUTE_OUT)
        gw_ptr,  # fp32 [O, K]   (COMPUTE_GW; zero-init unless GW_DIRECT)
        gxT_ptr,  # fp32 [IN, M]  (COMPUTE_GX; zero-init atomic target)
        M,
        O,
        K,
        B_O: tl.constexpr,
        B_K: tl.constexpr,
        B_M: tl.constexpr,
        GW_DIRECT: tl.constexpr,
        COMPUTE_OUT: tl.constexpr,
        COMPUTE_GW: tl.constexpr,
        COMPUTE_GX: tl.constexpr,
        LOAD_GO: tl.constexpr,  # == COMPUTE_GW or COMPUTE_GX (host-computed)
        LOAD_X: tl.constexpr,  # == COMPUTE_OUT or COMPUTE_GW (host-computed)
    ):
        """Program (ob, mb): requested products for its [B_O] outputs and
        [B_M] batch rows in one pass over the k axis. idx/w/go tiles are each
        loaded once; the gather tile xv and its addresses feed every product.
        GW_DIRECT (single m-block) stores grad_w without atomics. The
        COMPUTE_* flags are compile-time, so unused loads/stores are pruned;
        the tournament train_step is the all-flags-on specialization."""
        ob = tl.program_id(0)
        mb = tl.program_id(1)
        o_off = ob * B_O + tl.arange(0, B_O)
        m_off = mb * B_M + tl.arange(0, B_M)
        o_mask = o_off < O
        m_mask = m_off < M
        o64 = o_off.to(tl.int64)
        om_go = o_mask[:, None] & m_mask[None, :]
        go = tl.zeros([B_O, B_M], dtype=tl.float32)
        if LOAD_GO:
            go = tl.load(
                goT_ptr + o64[:, None] * M + m_off[None, :], mask=om_go, other=0.0
            )  # [B_O, B_M]
        acc = tl.zeros([B_O, B_M], dtype=tl.float32)
        for kb in range(tl.cdiv(K, B_K)):
            k_off = kb * B_K + tl.arange(0, B_K)
            k_mask = k_off < K
            ok = o64[:, None] * K + k_off[None, :]  # int64: O*K may pass 2^31
            om = o_mask[:, None] & k_mask[None, :]
            i2 = tl.load(idx_ptr + ok, mask=om, other=0)
            w2 = tl.load(w_ptr + ok, mask=om, other=0.0)
            # int32 gather addresses: IN*M < 2^31 (module docstring)
            xaddr = i2[:, :, None] * M + m_off[None, None, :]  # [B_O, B_K, B_M]
            xm = om[:, :, None] & m_mask[None, None, :]
            xv = tl.zeros([B_O, B_K, B_M], dtype=tl.float32)
            if LOAD_X:
                xv = tl.load(xT_ptr + xaddr, mask=xm, other=0.0)
            if COMPUTE_OUT:
                # forward: out[m, o] = sum_k x * w
                acc += tl.sum(xv * w2[:, :, None], axis=1)
            if COMPUTE_GW:
                # grad_w[o, k] = sum_m x * go -- coalesced store along k
                gw_tile = tl.sum(xv * go[:, None, :], axis=2)
                if GW_DIRECT:
                    tl.store(gw_ptr + ok, gw_tile, mask=om)
                else:
                    tl.atomic_add(gw_ptr + ok, gw_tile, mask=om)
            if COMPUTE_GX:
                # grad_x: gxT[idx, m] += w * go -- same addresses as the gather
                tl.atomic_add(gxT_ptr + xaddr, w2[:, :, None] * go[:, None, :], mask=xm)
        if COMPUTE_OUT:
            oo = m_off[None, :].to(tl.int64) * O + o_off[:, None]  # [B_O, B_M]
            tl.store(out_ptr + oo, acc, mask=om_go)


def _m_blocking(M: int) -> tuple[int, int, bool]:
    """Return (B_M, n_mblocks, gw_direct) for a batch of ``M`` rows."""
    B_M = min(triton.next_power_of_2(max(M, 1)), _B_M_CAP)
    n_mb = triton.cdiv(M, B_M)
    return B_M, n_mb, n_mb == 1


def _stage_inputs(
    x: torch.Tensor, connection_indices: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stage (xT fp32 [IN, M], idx int32 [O, K], w fp32 [O, K]) for launch."""
    xT = x.t().contiguous().to(torch.float32)  # [IN, M] -- L2-resident table
    wf = weight.contiguous()
    if wf.dtype != torch.float32:
        wf = wf.float()
    idxc = connection_indices.contiguous()
    if idxc.dtype != torch.int32:
        idxc = idxc.to(torch.int32)
    return xT, idxc, wf


def fused_train_step(
    x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, go: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tournament one-launch entry: full fwd+bwd in a single kernel launch.

    Returns ``(out [M, O] fp32, grad_w [O, K] fp32, grad_x [M, IN] fp32)``
    with ``grad_w[o, k] = sum_m go[m, o] * x[m, idx[o, k]]`` and
    ``grad_x[m, i] = sum_{(o, k): idx[o, k] == i} go[m, o] * w[o, k]``.

    This is the exact specialization the kernel-search round 1 numbers were
    measured on; the autograd path launches the same kernel as separate
    forward-only and grads-only passes. Not an autograd entry -- use
    :func:`triton_fused_sparse_gather` inside training graphs.
    """
    if not (triton_fused_available() and x.is_cuda):
        raise RuntimeError("fused_train_step requires CUDA + Triton")
    M, IN = x.shape
    O, K = idx.shape
    if w.shape != (O, K):
        raise ValueError(f"w shape {tuple(w.shape)} != idx shape {(O, K)}")
    if go.shape != (M, O):
        raise ValueError(f"go shape {tuple(go.shape)} != {(M, O)}")
    if IN * M >= 2**31:
        raise ValueError("fused_train_step assumes IN*M < 2^31 (int32 addresses)")
    dev = x.device
    xT, idxc, wf = _stage_inputs(x, idx, w)
    goT = go.t().contiguous().to(torch.float32)  # [O, M]
    B_O, B_K = _TILES
    B_M, n_mb, gw_direct = _m_blocking(M)
    out = torch.empty(M, O, device=dev, dtype=torch.float32)
    if gw_direct:
        grad_w = torch.empty(O, K, device=dev, dtype=torch.float32)
    else:
        grad_w = torch.zeros(O, K, device=dev, dtype=torch.float32)
    gxT = torch.zeros(IN, M, device=dev, dtype=torch.float32)
    # Launch on the tensors' device (Triton uses the CURRENT
    # device/stream; see triton_indexed_gather_transposed 2026-08-20
    # cross-device race note).
    with torch.cuda.device(xT.device):
        _fused_chunked_kernel[(triton.cdiv(O, B_O), n_mb)](
            xT,
            goT,
            idxc,
            wf,
            out,
            grad_w,
            gxT,
            M,
            O,
            K,
            B_O=B_O,
            B_K=B_K,
            B_M=B_M,
            GW_DIRECT=gw_direct,
            COMPUTE_OUT=True,
            COMPUTE_GW=True,
            COMPUTE_GX=True,
            LOAD_GO=True,
            LOAD_X=True,
            num_warps=_NUM_WARPS,
        )
    return out, grad_w, gxT.t().contiguous()


class _TritonFusedChunkedFunction(torch.autograd.Function):
    """Autograd wiring of the fused chunked kernel (see module docstring)."""

    @staticmethod
    def forward(ctx, x, idx, weight):
        # x: [M, IN] (any float dtype)   idx: [O, K] int32/int64   weight: [O, K]
        in_dtype = x.dtype
        w_dtype = weight.dtype
        M, IN = x.shape
        O, K = idx.shape
        xT, idxc, wf = _stage_inputs(x, idx, weight)
        out = torch.empty(M, O, device=x.device, dtype=torch.float32)
        if min(M, O, K) > 0:
            B_O, B_K = _TILES
            B_M, n_mb, _ = _m_blocking(M)
            # Launch on the tensors' device (Triton uses the CURRENT
            # device/stream; see triton_indexed_gather_transposed 2026-08-20
            # cross-device race note).
            with torch.cuda.device(xT.device):
                _fused_chunked_kernel[(triton.cdiv(O, B_O), n_mb)](
                    xT,
                    xT,  # goT unused in forward mode (LOAD_GO=False prunes it)
                    idxc,
                    wf,
                    out,
                    out,  # gw unused (COMPUTE_GW=False)
                    out,  # gxT unused (COMPUTE_GX=False)
                    M,
                    O,
                    K,
                    B_O=B_O,
                    B_K=B_K,
                    B_M=B_M,
                    GW_DIRECT=True,
                    COMPUTE_OUT=True,
                    COMPUTE_GW=False,
                    COMPUTE_GX=False,
                    LOAD_GO=False,
                    LOAD_X=True,
                    num_warps=_NUM_WARPS,
                )
        elif K == 0:
            out.zero_()
        # xT is exactly the staged table backward needs; saving it (fp32)
        # avoids re-staging at the cost of fp32 residency, matching the
        # tournament kernel's host contract.
        ctx.save_for_backward(xT, idxc, wf)
        ctx.meta = (M, O, IN, K, in_dtype, w_dtype)
        return out.to(in_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        xT, idxc, wf = ctx.saved_tensors
        M, O, IN, K, in_dtype, w_dtype = ctx.meta
        needs_x, _, needs_w = ctx.needs_input_grad[:3]
        if not (needs_x or needs_w):
            return None, None, None
        dev = xT.device
        goT = grad_out.contiguous().to(torch.float32).t().contiguous()  # [O, M]
        B_O, B_K = _TILES
        B_M, n_mb, gw_direct = _m_blocking(M)
        grad_w = grad_x = None
        if needs_w:
            if gw_direct:
                grad_w = torch.empty(O, K, device=dev, dtype=torch.float32)
            else:
                grad_w = torch.zeros(O, K, device=dev, dtype=torch.float32)
        gxT = torch.zeros(IN, M, device=dev, dtype=torch.float32) if needs_x else None
        if min(M, O, K) > 0:
            # Launch on the tensors' device (Triton uses the CURRENT
            # device/stream; see triton_indexed_gather_transposed 2026-08-20
            # cross-device race note).
            with torch.cuda.device(xT.device):
                _fused_chunked_kernel[(triton.cdiv(O, B_O), n_mb)](
                    xT,
                    goT,
                    idxc,
                    wf,
                    goT,  # out unused in backward mode (COMPUTE_OUT=False)
                    grad_w if grad_w is not None else goT,
                    gxT if gxT is not None else goT,
                    M,
                    O,
                    K,
                    B_O=B_O,
                    B_K=B_K,
                    B_M=B_M,
                    GW_DIRECT=gw_direct,
                    COMPUTE_OUT=False,
                    COMPUTE_GW=needs_w,
                    COMPUTE_GX=needs_x,
                    LOAD_GO=True,
                    LOAD_X=needs_w,
                    num_warps=_NUM_WARPS,
                )
        elif needs_w and K > 0:
            grad_w.zero_()
        if gxT is not None:
            grad_x = gxT.t().contiguous()  # [M, IN]
        return (
            grad_x.to(in_dtype) if grad_x is not None else None,
            None,
            grad_w.to(w_dtype) if grad_w is not None else None,
        )


def triton_fused_sparse_gather(
    x: torch.Tensor,
    connection_indices: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 2048,  # accepted for signature parity; unused
) -> torch.Tensor:
    """Return ``out[m, o] = sum_k x[m, connection_indices[o, k]] * weight[o, k]``.

    Drop-in for :func:`triton_sparse_gather` / :func:`fused_sparse_gather`:
    ``x`` is ``[M, in_features]``; ``connection_indices`` and ``weight`` are
    ``[out_features, K]``. Differentiable w.r.t. ``x`` and ``weight``;
    indices are fixed. Requires a CUDA tensor, a working Triton install (see
    :func:`triton_fused_available`), and ``in_features * M < 2**31``.
    ``chunk_size`` is ignored (tiles are fixed, swept-in).
    """
    if not x.is_cuda:
        raise RuntimeError("triton_fused_sparse_gather requires a CUDA tensor")
    if x.ndim != 2:
        raise ValueError(f"x must be two-dimensional, got shape {tuple(x.shape)}")
    if connection_indices.ndim != 2 or weight.ndim != 2:
        raise ValueError("connection_indices and weight must be two-dimensional")
    if connection_indices.shape != weight.shape:
        raise ValueError(
            "connection_indices and weight must have identical shapes, got "
            f"{tuple(connection_indices.shape)} and {tuple(weight.shape)}"
        )
    if connection_indices.dtype not in {torch.int32, torch.int64}:
        raise TypeError("connection_indices must use int32 or int64")
    if x.device != connection_indices.device or x.device != weight.device:
        raise ValueError("x, connection_indices, and weight must share one device")
    if not x.is_floating_point() or not weight.is_floating_point():
        raise TypeError("x and weight must use floating-point dtypes")
    supported_dtypes = {torch.bfloat16, torch.float16, torch.float32}
    if x.dtype not in supported_dtypes or weight.dtype not in supported_dtypes:
        raise TypeError(
            "triton_fused_sparse_gather supports float16, bfloat16, and float32 "
            f"inputs and weights; got {x.dtype} and {weight.dtype}"
        )
    if x.shape[0] * x.shape[1] >= 2**31:
        raise ValueError(
            "triton_fused_sparse_gather requires in_features * batch < 2^31 "
            f"(int32 gather addresses), got {x.shape[1]} * {x.shape[0]}; use "
            "'triton_transposed' or 'recompute' for this shape"
        )
    if not triton_fused_available():
        raise RuntimeError(
            "triton_fused_sparse_gather requires Triton and a CUDA device. "
            "Use projection_backend='auto' or 'recompute' for a portable path."
        )
    return _TritonFusedChunkedFunction.apply(x, connection_indices, weight)


__all__ = [
    "HAS_TRITON",
    "fused_train_step",
    "triton_fused_available",
    "triton_fused_sparse_gather",
]
