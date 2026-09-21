"""Triton fused sparse gather-multiply-sum: forward + backward in one Function.

Drop-in alternative to :func:`fused_sparse_gather` (same signature) that computes,
for a fixed sparse fan-in::

    out[m, o] = sum_k  x[m, connection_indices[o, k]] * weight[o, k]

TRANSPOSED-COALESCED design (2026-08-07; historically "v3" on the development
branch), replacing the original batch-tiled
kernels after a measured 4-6x per-op win (bench_triton_v2.py, Blackwell b512 K40):

  * forward  : gathers read from ``xT [IN, M]`` so the address ``idx*M + m`` is
    CONTIGUOUS over the batch -- every gather is a coalesced row segment and the
    kernel runs at the DRAM roofline for its traffic (1.55 ms at 16k->144k vs
    9.38 ms for the CUDA picker's kernel; the one-off transpose is ~0.05 ms).
  * backward : SPLIT. grad_w via a transposed reduction over m (``goT``/``xT``
    rows, coalesced, NO atomics); grad_x deterministic by default (below), with
    the original coalesced ``tl.atomic_add`` scatter into ``gxT [IN, M]``
    retained as an explicit opt-out. fp32 accumulation throughout.
  * indices  : int32 end-to-end (the model's buffer dtype) -- the old kernels
    upcast to int64 EVERY call (an [O,K] alloc + copy per layer per step).

Measured per-op vs the old Triton kernels (Blackwell, batch 512, K=40):
    forward 3.7-14.6x faster, backward 2.8-5.8x faster; also beats the CUDA
    extension's picker at every swept FF shape (its staged design is occupancy-
    capped on sm120; brute coalesced traffic wins at this bandwidth). See
    profilings/dendrinet_harness/rnn_prof/bench_triton_v2.py and the ledger.

Correctness (fp64-reference, batch 512, all swept shapes): forward <= 3.1e-7,
grad_w <= 2.2e-5, grad_x <= 4.6e-6 -- fp32 reassociation only.

Deterministic grad_x (the DEFAULT since 2026-08-18): the atomic grad_x scatter
is replaced by a two-stage sorted-CSR gather with no atomics anywhere. A
one-time precompute over the static connection indices stable-sorts the
flattened index buffer into per-input CSR segments (``perm`` / ``o_sorted`` /
``row_ptr``); each backward then gathers ``w.flat[perm]`` (one coalesced
``index_select``) and stage 1 gives every program EXCLUSIVE ownership of one
(input, split, m-tile) row of the workspace ``ws [NSPLIT, IN, M]`` fp32 --
written exactly once, accumulated in registers in a fixed order, so there are
no read-modify-write hazards -- while stage 2 reduces the splits in a fixed
block order. grad_x is therefore bit-identical run-to-run BY CONSTRUCTION,
and the path measured ~2x FASTER than the atomic scatter on both H100 and
Blackwell at the OLMo FFN-replacement shapes (agentic kernel search round 1,
2026-08-18; bit-stability verified 3x by the tournament harness). The CSR
setup is cached per source index tensor and invalidated automatically when
rewiring changes the indices: in-place writes (``apply_rewiring`` /
``apply_indexed_rewiring_after_step``) bump the tensor version, and a replaced
buffer changes its data pointer -- either produces a fresh cache entry. The
cache is a bounded LRU (hard entry cap AND hard byte cap, env
``DENDRITIC_GX_SETUP_CACHE_MAX`` / ``DENDRITIC_GX_SETUP_CACHE_MAX_BYTES``), so
neither many cells nor FSDP unshard/reshard churn (fresh index tensor objects
every step) can grow it past the caps; :func:`gx_setup_cache_stats` reports
occupancy. Setups are pure deterministic functions of the index content, so
eviction/recompute never changes grad_x by a single bit.

Set env ``DENDRITIC_ATOMIC_GRADX=1`` (or the module flag
:data:`ATOMIC_GRADX`) to opt back into the original atomic scatter (correct,
but run-to-run nondeterministic at the last-ulp level, exactly like the
previous kernels and the CUDA path). Shapes whose single workspace block
``[IN, M]`` fp32 exceeds the 2 GiB cap, or whose ``O*K >= 2^31`` (int32
sorted positions), fall back to the atomic path with a one-time
RuntimeWarning.
Self-test: ``python triton_indexed_gather_transposed.py`` (needs CUDA+Triton).

AMP-safe: inputs are upcast to fp32 for the kernels (atomics stay fp32; bf16
atomics are unreliable across archs) and outputs/grads are cast back to the
caller's dtypes, so switching this path on is numerically equivalent under
autocast.

Triton is an OPTIONAL dependency: importing this module never fails. Query
:data:`HAS_TRITON` (or call :func:`triton_gather_available`) before selecting
this path; ``triton_sparse_gather`` raises a clear error if Triton is missing.

Source provenance: Ryan Whalen, ``origin/ryanconfig`` commit ``c1973844c``.
The maintained adapter keeps this backend explicit and out of ``auto`` until
current-model and current-hardware crossover benchmarks are complete.
Deterministic backward from agentic kernel search round 1 (2026-08-18),
maintained work; it supersedes the retired S4b workspace-scatter attempt
(2026-08-17). Whalen's forward/grad_w kernels are unchanged, and his atomic
grad_x kernel remains verbatim as the ``DENDRITIC_ATOMIC_GRADX`` opt-out.
"""

from __future__ import annotations

import os
import warnings

import torch

try:  # Triton is an optional dependency -- never hard-fail on import
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:
    triton = None
    tl = None
    HAS_TRITON = False

# Swept on Blackwell (bench_triton_v2.py): uniform winners across the FF grid.
# (B_M, B_O, B_K) for the forward and grad_x scatter; (B_O, B_M) for grad_w.
_FWD_TILES = (128, 16, 8)
_GX_TILES = (32, 32, 8)
_GW_TILES = (64, 64)

# --- Deterministic grad_x (default; agentic kernel search round 1, 2026-08-18).
_ENV_FALSY = {"", "0", "false", "off", "no"}


def _env_atomic_gradx() -> bool:
    return (
        os.environ.get("DENDRITIC_ATOMIC_GRADX", "").strip().lower() not in _ENV_FALSY
    )


#: Opt-out mirroring env DENDRITIC_ATOMIC_GRADX; either restores the original
#: atomic grad_x scatter. Read at each backward, so tests may toggle it.
ATOMIC_GRADX = _env_atomic_gradx()

#: Hard cap on the deterministic stage-1 workspace [nsplit, IN, M] fp32. The
#: split count shrinks to fit; shapes whose SINGLE block [IN, M] exceeds the
#: cap fall back to atomics (warned once).
_DET_GX_WORKSPACE_CAP_BYTES = 2 * 1024**3

_GX_DET_TILES = (16, 128)  # (B_P sorted positions, B_M batch) for stage 1
_GX_DET_NSPLIT = 4  # workspace splits per input (load balance / occupancy)
_GX_DET_REDUCE_BLOCK = 1024  # stage-2 elementwise-reduction tile

#: CSR setup cache: (data_ptr, shape, device, IN) -> (pinned idx_src, version,
#: (perm, o_sorted, row_ptr), entry_nbytes). Insertion-ordered dict used as an
#: LRU. Each entry pins its source index tensor, so the caching allocator
#: cannot hand that data_ptr to a different tensor while the entry lives; an
#: in-place rewire bumps the source tensor's ``_version`` and forces a
#: recompute.
_GX_SETUP_CACHE: dict = {}
#: Bounds on the cache: a hard ENTRY cap and a hard BYTE cap; eviction is LRU
#: until both hold. Each entry holds perm (int64, 2x the index bytes) +
#: o_sorted (int32, 1x) + row_ptr, i.e. ~3x the index bytes, MEASURED at 3.00x
#: on 72B cell geometry, plus the pinned source index tensor (free when it is
#: the module's resident buffer, 1x more when it is a churned temporary --
#: both are counted so the byte cap is a true upper bound in either regime).
#:
#: Why both caps (2026-08-23 measurements, see the fsdp-72b-gx-cache-blocker
#: record): the cache is a per-rank FLOOR that no data parallelism reduces --
#: it is built from the replicated index buffers, so 64 resident entries on
#: 72B cell geometry carry 43.4 GiB per rank and OOM'd three 8-GPU recovery
#: runs. It is also pure waste there: every model in this project has more
#: cells than the entry cap (96 at v2, 192 at 32B, 240 at 72B) and the
#: backward touches each cell once per step in the same order, so the LRU
#: evicts every entry before reuse -- the measured hit rate is 0%, with 240
#: misses per step at both bound 64 and bound 8. Bounding the BYTES is
#: therefore provably free at scale (the miss count is unchanged) and returns
#: the memory: correctness needs at most ONE live entry per module (its
#: current index tensor), never a deep pool of stale ones. The same byte cap
#: also bounds the FSDP unshard/reshard pattern, where fresh tensor objects
#: with unchanged content arrive every step (new data_ptr -> new key) and
#: would otherwise pile up dead pinned entries until the entry cap.
#:
#: A single entry larger than the byte cap is still cached ALONE (the module
#: needs its current setup; everything else is evicted first), so the true
#: bound is max(byte cap, largest single entry). Entry cap default 64 keeps
#: hit behaviour unchanged for small models; the byte cap default of 8 GiB
#: reproduces the measured "bound 8, fits with 32.3 GiB margin" budget at 72B
#: geometry while leaving every sub-8-GiB working set fully cached. Override
#: with DENDRITIC_GX_SETUP_CACHE_MAX / DENDRITIC_GX_SETUP_CACHE_MAX_BYTES.
_GX_SETUP_CACHE_MAX = int(os.environ.get("DENDRITIC_GX_SETUP_CACHE_MAX", "64"))
assert _GX_SETUP_CACHE_MAX >= 1, (
    "DENDRITIC_GX_SETUP_CACHE_MAX must be >= 1; a zero-size cache would make "
    "the eviction loop discard the entry it just computed"
)
_GX_SETUP_CACHE_MAX_BYTES = int(
    os.environ.get("DENDRITIC_GX_SETUP_CACHE_MAX_BYTES", str(8 * 1024**3))
)
assert _GX_SETUP_CACHE_MAX_BYTES >= 1, (
    "DENDRITIC_GX_SETUP_CACHE_MAX_BYTES must be >= 1 (an over-cap entry is "
    "still cached alone, so 1 byte means 'keep exactly one entry')"
)
#: Tracked bytes of all live cache entries (setup tensors + pinned sources).
_gx_setup_cache_nbytes = 0

#: Diagnostic counter: number of CSR setups computed (cache misses). Tests use
#: it to assert rewire invalidation without timing anything.
_gx_setup_misses = 0

_det_gx_warned_fallback = False


def _gradx_deterministic_enabled() -> bool:
    """Deterministic grad_x is the default; atomic is the explicit opt-out."""
    return not (ATOMIC_GRADX or _env_atomic_gradx())


def _warn_gradx_fallback(reason: str) -> None:
    global _det_gx_warned_fallback
    if not _det_gx_warned_fallback:
        _det_gx_warned_fallback = True
        warnings.warn(
            "deterministic grad_x: " + reason + "; falling back to the "
            "atomic grad_x path (correct, but run-to-run nondeterministic).",
            RuntimeWarning,
            stacklevel=3,
        )


def _gx_entry_nbytes(idx_src: torch.Tensor, num_inputs: int) -> int:
    """Exact bytes a cache entry will hold, computable BEFORE the sort.

    perm (int64) + o_sorted (int32) + row_ptr (int32, IN+1) plus the pinned
    source index tensor (its true cost when it is a churned temporary; an
    overcount by its own size when it is the module's resident buffer, which
    keeps the byte cap a valid upper bound in both regimes).
    """
    nnz = idx_src.numel()
    return 12 * nnz + 4 * (num_inputs + 1) + nnz * idx_src.element_size()


def gx_setup_cache_stats() -> dict:
    """Introspection hook: current CSR setup cache occupancy and caps.

    ``entries``/``bytes`` are the live totals, ``entry_cap``/``byte_cap`` the
    configured bounds, ``misses`` the process-lifetime setup computations.
    Tests and driver diagnostics read this instead of the private globals.
    """
    return {
        "entries": len(_GX_SETUP_CACHE),
        "bytes": _gx_setup_cache_nbytes,
        "entry_cap": _GX_SETUP_CACHE_MAX,
        "byte_cap": _GX_SETUP_CACHE_MAX_BYTES,
        "misses": _gx_setup_misses,
    }


def _gradx_setup(idx_src: torch.Tensor, idx: torch.Tensor, num_inputs: int):
    """(perm int64 [NNZ], o_sorted int32 [NNZ], row_ptr int32 [IN+1]), cached.

    ``idx_src`` is the caller's index tensor (normally the module's int32
    buffer, in which case ``idx is idx_src``); ``idx`` is the contiguous int32
    view the kernels use. The cache key is derived from ``idx_src`` so an
    int64 buffer that is converted per call still hits, and rewire
    invalidation rides the source tensor: ``apply_rewiring`` writes
    ``connection_indices[rows, slots] = ...`` in place, which bumps
    ``idx_src._version``; replacing the buffer changes ``data_ptr``. Either
    way the stale CSR entry is recomputed on the next backward. The entry
    pins ``idx_src`` while cached, so a data_ptr collision with a different
    tensor is impossible by construction.

    The cache is evict-on-miss LRU under two hard caps (entries and bytes,
    see the cap comments above): eviction runs BEFORE the sort so the resident
    peak never holds both a full cache and the incoming entry. The setup is a
    pure, deterministic function of the index content (stable sort), so an
    eviction can never change grad_x -- a re-computed setup is bit-identical.
    """
    global _gx_setup_misses, _gx_setup_cache_nbytes
    key = (
        idx_src.data_ptr(),
        tuple(idx_src.shape),
        str(idx_src.device),
        num_inputs,
    )
    version = idx_src._version
    hit = _GX_SETUP_CACHE.get(key)
    if hit is not None and hit[1] == version:
        _GX_SETUP_CACHE[key] = _GX_SETUP_CACHE.pop(key)  # LRU refresh
        return hit[2]
    _gx_setup_misses += 1
    nbytes = _gx_entry_nbytes(idx_src, num_inputs)
    stale = _GX_SETUP_CACHE.pop(key, None)  # replace a stale-version entry
    if stale is not None:
        _gx_setup_cache_nbytes -= stale[3]
    # Evict least-recent until BOTH caps hold with the incoming entry counted.
    # An entry larger than the byte cap empties the cache and is kept alone.
    while _GX_SETUP_CACHE and (
        len(_GX_SETUP_CACHE) >= _GX_SETUP_CACHE_MAX
        or _gx_setup_cache_nbytes + nbytes > _GX_SETUP_CACHE_MAX_BYTES
    ):
        evicted = _GX_SETUP_CACHE.pop(next(iter(_GX_SETUP_CACHE)))
        _gx_setup_cache_nbytes -= evicted[3]
    _, k = idx.shape
    flat = idx.reshape(-1).long()
    sorted_i, perm = torch.sort(flat, stable=True)  # stable -> deterministic perm
    row_ptr = torch.searchsorted(
        sorted_i,
        torch.arange(num_inputs + 1, device=idx.device, dtype=torch.int64),
    ).to(torch.int32)
    o_sorted = torch.div(perm, k, rounding_mode="floor").to(torch.int32)
    setup = (perm, o_sorted.contiguous(), row_ptr.contiguous())
    _GX_SETUP_CACHE[key] = (idx_src, version, setup, nbytes)
    _gx_setup_cache_nbytes += nbytes
    return setup


def triton_gather_available() -> bool:
    """Whether the Triton fused-gather path can be used on this install."""
    return HAS_TRITON and torch.cuda.is_available()


if HAS_TRITON:

    @triton.jit
    def _sparse_gather_fwd_t(
        xT_ptr,
        idx_ptr,
        w_ptr,
        out_ptr,
        M,
        NUM_OUTPUTS,
        IN,
        K,
        B_M: tl.constexpr,
        B_O: tl.constexpr,
        B_K: tl.constexpr,
    ):
        """out[m, o] = sum_k xT[idx[o, k], m] * w[o, k] -- coalesced over m.
        Output tiles ride grid axis 0 (2^31 limit): 20M-output RNN modules
        overflow the 65,535 cap of axis 1."""
        ob = tl.program_id(0)
        mb = tl.program_id(1)
        m_off = mb * B_M + tl.arange(0, B_M)
        o_off = ob * B_O + tl.arange(0, B_O)
        m_mask = m_off < M
        o_mask = o_off < NUM_OUTPUTS
        acc = tl.zeros([B_O, B_M], dtype=tl.float32)
        for kb in range(tl.cdiv(K, B_K)):
            k_off = kb * B_K + tl.arange(0, B_K)
            k_mask = k_off < K
            ok = o_off[:, None] * K + k_off[None, :]
            om = o_mask[:, None] & k_mask[None, :]
            i2 = tl.load(idx_ptr + ok, mask=om, other=0)
            w2 = tl.load(w_ptr + ok, mask=om, other=0.0)
            xo = i2[:, :, None] * M + m_off[None, None, :]  # [B_O, B_K, B_M]
            xm = om[:, :, None] & m_mask[None, None, :]
            xv = tl.load(xT_ptr + xo, mask=xm, other=0.0).to(tl.float32)
            acc += tl.sum(xv * w2[:, :, None].to(tl.float32), axis=1)
        oo = m_off[None, :] * NUM_OUTPUTS + o_off[:, None]
        tl.store(out_ptr + oo, acc, mask=(m_mask[None, :] & o_mask[:, None]))

    @triton.jit
    def _sparse_gather_gw_t(
        goT_ptr,
        xT_ptr,
        idx_ptr,
        gw_ptr,
        M,
        NUM_OUTPUTS,
        IN,
        K,
        B_O: tl.constexpr,
        B_M: tl.constexpr,
    ):
        """grad_w[o, k] = sum_m go[m, o] * x[m, idx[o, k]] -- transposed,
        coalesced over m, no atomics. Grid: (O tiles, K)."""
        ob = tl.program_id(0)
        k = tl.program_id(1)
        o_off = ob * B_O + tl.arange(0, B_O)
        o_mask = o_off < NUM_OUTPUTS
        idx = tl.load(idx_ptr + o_off * K + k, mask=o_mask, other=0)
        acc = tl.zeros([B_O], dtype=tl.float32)
        for mb in range(tl.cdiv(M, B_M)):
            m_off = mb * B_M + tl.arange(0, B_M)
            m_mask = m_off < M
            mm = o_mask[:, None] & m_mask[None, :]
            go = tl.load(
                goT_ptr + o_off[:, None] * M + m_off[None, :], mask=mm, other=0.0
            ).to(tl.float32)
            xv = tl.load(
                xT_ptr + idx[:, None] * M + m_off[None, :], mask=mm, other=0.0
            ).to(tl.float32)
            acc += tl.sum(go * xv, axis=1)
        tl.store(gw_ptr + o_off * K + k, acc, mask=o_mask)

    @triton.jit
    def _sparse_gather_gx_t(
        goT_ptr,
        idx_ptr,
        w_ptr,
        gxT_ptr,
        M,
        NUM_OUTPUTS,
        IN,
        K,
        B_M: tl.constexpr,
        B_O: tl.constexpr,
        B_K: tl.constexpr,
    ):
        """gxT[idx[o, k], m] += go[m, o] * w[o, k] -- coalesced fp32 atomics.
        Output tiles on grid axis 0 (2^31 limit), as in the forward.
        The DENDRITIC_ATOMIC_GRADX opt-out; deterministic stage-1/2 kernels
        below are the default."""
        ob = tl.program_id(0)
        mb = tl.program_id(1)
        m_off = mb * B_M + tl.arange(0, B_M)
        o_off = ob * B_O + tl.arange(0, B_O)
        m_mask = m_off < M
        o_mask = o_off < NUM_OUTPUTS
        om_go = o_mask[:, None] & m_mask[None, :]
        go = tl.load(
            goT_ptr + o_off[:, None] * M + m_off[None, :], mask=om_go, other=0.0
        )
        for kb in range(tl.cdiv(K, B_K)):
            k_off = kb * B_K + tl.arange(0, B_K)
            k_mask = k_off < K
            ok = o_off[:, None] * K + k_off[None, :]
            om = o_mask[:, None] & k_mask[None, :]
            i2 = tl.load(idx_ptr + ok, mask=om, other=0)
            w2 = tl.load(w_ptr + ok, mask=om, other=0.0)
            xo = i2[:, :, None] * M + m_off[None, None, :]  # [B_O, B_K, B_M]
            xm = om[:, :, None] & m_mask[None, None, :]
            tl.atomic_add(
                gxT_ptr + xo,
                w2[:, :, None].to(tl.float32) * go[:, None, :].to(tl.float32),
                mask=xm,
            )

    @triton.jit
    def _sparse_gather_gx_det_stage1(
        o_sorted_ptr,  # int32 [NNZ]  output row of each sorted position
        w_sorted_ptr,  # fp    [NNZ]  w.flat[perm]
        row_ptr_ptr,  # int32 [IN+1] CSR bounds over the input axis
        goT_ptr,  # fp    [O, M]
        ws_ptr,  # fp32  [NSPLIT, IN, M]  written exactly once per element
        M,
        IN,
        NSPLIT,
        B_P: tl.constexpr,
        B_M: tl.constexpr,
    ):
        """Deterministic grad_x stage 1: program (i, s, mb) reduces the s-th
        slice of input i's sorted CSR run into registers and stores it to its
        EXCLUSIVE workspace row ws[s, i, m-tile]. Sequential fixed-order loop,
        no atomics, no read-modify-write -- bitwise deterministic."""
        pid0 = tl.program_id(0)  # = i * NSPLIT + s
        mb = tl.program_id(1)
        i = pid0 // NSPLIT
        s = pid0 % NSPLIT
        m_off = mb * B_M + tl.arange(0, B_M)
        m_mask = m_off < M
        start = tl.load(row_ptr_ptr + i)
        end = tl.load(row_ptr_ptr + i + 1)
        seg = end - start
        chunk = (seg + NSPLIT - 1) // NSPLIT
        lo = start + s * chunk
        hi = tl.minimum(lo + chunk, end)
        n = hi - lo  # may be 0 -> store zeros (keeps ws fully covered)
        acc = tl.zeros([B_M], dtype=tl.float32)
        for pb in range(tl.cdiv(n, B_P)):
            p_off = lo + pb * B_P + tl.arange(0, B_P)
            p_mask = p_off < hi
            o = tl.load(o_sorted_ptr + p_off, mask=p_mask, other=0)
            wv = tl.load(w_sorted_ptr + p_off, mask=p_mask, other=0.0).to(tl.float32)
            g = tl.load(
                goT_ptr + o.to(tl.int64)[:, None] * M + m_off[None, :],
                mask=p_mask[:, None] & m_mask[None, :],
                other=0.0,
            ).to(tl.float32)  # [B_P, B_M], coalesced over m
            acc += tl.sum(g * wv[:, None], axis=0)
        base = (s.to(tl.int64) * IN + i) * M
        tl.store(ws_ptr + base + m_off, acc, mask=m_mask)

    @triton.jit
    def _gx_ws_reduce(
        ws_ptr,
        out_ptr,
        NUMEL,
        N_BLOCKS,
        B: tl.constexpr,
    ):
        """Deterministic grad_x stage 2: out[e] = sum_b ws[b, e] in fixed
        block order (no atomics, deterministic)."""
        pid = tl.program_id(0)
        off = pid * B + tl.arange(0, B)
        mask = off < NUMEL
        acc = tl.zeros([B], dtype=tl.float32)
        p = ws_ptr + off
        for _ in range(N_BLOCKS):
            acc += tl.load(p, mask=mask, other=0.0)
            p += NUMEL
        tl.store(out_ptr + off, acc, mask=mask)


def _deterministic_gradx(goT, idx, wf, M, num_outputs, IN, K, idx_src):
    """Two-stage sorted-CSR deterministic grad_x. Returns ``gxT [IN, M]``
    fp32, or ``None`` (with a one-time warning) when the shape is outside the
    scheme's limits, in which case the caller runs the original atomic path.
    """
    slice_bytes = 4 * IN * M
    if slice_bytes > _DET_GX_WORKSPACE_CAP_BYTES:
        _warn_gradx_fallback(
            f"workspace block [IN={IN}, M={M}] fp32 needs "
            f"{slice_bytes / 2**30:.2f} GiB, over the "
            f"{_DET_GX_WORKSPACE_CAP_BYTES / 2**30:.0f} GiB cap"
        )
        return None
    if num_outputs * K >= 2**31:
        _warn_gradx_fallback(
            f"O*K = {num_outputs * K} >= 2^31 exceeds int32 sorted positions"
        )
        return None
    perm, o_sorted, row_ptr = _gradx_setup(idx_src, idx, IN)
    w_sorted = wf.reshape(-1).index_select(0, perm)  # coalesced, deterministic
    nsplit = max(1, min(_GX_DET_NSPLIT, _DET_GX_WORKSPACE_CAP_BYTES // slice_bytes))
    # written exactly once per element by stage 1 -> no zero-init needed
    ws = torch.empty(nsplit, IN, M, device=goT.device, dtype=torch.float32)
    B_P, B_M = _GX_DET_TILES
    _sparse_gather_gx_det_stage1[(IN * nsplit, triton.cdiv(M, B_M))](
        o_sorted, w_sorted, row_ptr, goT, ws, M, IN, nsplit, B_P=B_P, B_M=B_M
    )
    if nsplit == 1:
        return ws[0]
    gxT = torch.empty(IN, M, device=goT.device, dtype=torch.float32)
    numel = IN * M
    _gx_ws_reduce[(triton.cdiv(numel, _GX_DET_REDUCE_BLOCK),)](
        ws, gxT, numel, nsplit, _GX_DET_REDUCE_BLOCK
    )
    return gxT


class _TritonSparseGatherTransposed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, weight):
        # x: [M, IN] (any float dtype)   idx: [O, K] int32/int64   weight: [O, K]
        in_dtype = x.dtype
        w_dtype = weight.dtype
        # bf16 inputs pass through at half the bytes -- every kernel casts its
        # loads to fp32 in registers, so ALL accumulation stays fp32 (the sm_80
        # atomics + precision lesson). fp32 inputs are unchanged.
        compute_dtype = torch.bfloat16 if x.dtype == torch.bfloat16 else torch.float32
        xf = x.contiguous().to(compute_dtype)
        wf = weight.contiguous().to(compute_dtype)
        idx_src = idx  # the module's buffer: keys the CSR setup cache
        idx = idx.contiguous().to(torch.int32)  # no-op for the int32 buffer
        M, IN = xf.shape
        num_outputs, K = idx.shape
        xT = xf.t().contiguous()
        out = torch.empty(M, num_outputs, device=xf.device, dtype=torch.float32)
        B_M, B_O, B_K = _FWD_TILES
        grid = (triton.cdiv(num_outputs, B_O), triton.cdiv(M, B_M))
        # Triton launches on the process's CURRENT device/stream, not the
        # tensors' device. Without this guard a module on cuda:N (N>0) is
        # enqueued on cuda:0's stream with pointers into cuda:N memory —
        # legal via UVA, silently racing every cuda:N consumer (2026-08-20:
        # deterministic stale-read step-0 losses + nondeterministic training
        # divergence on device-straddled spans). The guard also selects
        # PyTorch's current stream ON that device, which is what the
        # surrounding ATen ops use.
        with torch.cuda.device(xf.device):
            _sparse_gather_fwd_t[grid](
                xT, idx, wf, out, M, num_outputs, IN, K, B_M, B_O, B_K
            )
        # save the untransposed xf: backward re-transposes (cheap) rather than
        # doubling the saved-activation footprint.
        ctx.save_for_backward(xf, idx, wf, idx_src)
        ctx.meta = (M, num_outputs, IN, K, in_dtype, w_dtype)
        return out.to(in_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        xf, idx, wf, idx_src = ctx.saved_tensors
        M, num_outputs, IN, K, in_dtype, w_dtype = ctx.meta
        needs_x, _, needs_w = ctx.needs_input_grad[:3]
        goT = grad_out.contiguous().to(xf.dtype).t().contiguous()
        grad_x = grad_w = None
        # Same current-device guard as forward: autograd's per-device worker
        # usually has the right current device, but hooks/backward on graphs
        # that span devices do not guarantee it — make the launch device
        # explicit everywhere.
        with torch.cuda.device(xf.device):
            if needs_w:
                xT = xf.t().contiguous()
                grad_w = torch.empty(
                    num_outputs, K, device=xf.device, dtype=torch.float32
                )
                B_O, B_M = _GW_TILES
                _sparse_gather_gw_t[(triton.cdiv(num_outputs, B_O), K)](
                    goT, xT, idx, grad_w, M, num_outputs, IN, K, B_O, B_M
                )
            if needs_x:
                gxT = None
                if _gradx_deterministic_enabled():
                    # Default: two-stage sorted-CSR deterministic grad_x.
                    # Returns None past the shape limits -> original atomic
                    # path.
                    gxT = _deterministic_gradx(
                        goT, idx, wf, M, num_outputs, IN, K, idx_src
                    )
                if gxT is None:
                    gxT = torch.zeros(IN, M, device=xf.device, dtype=torch.float32)
                    B_M2, B_O2, B_K2 = _GX_TILES
                    grid = (triton.cdiv(num_outputs, B_O2), triton.cdiv(M, B_M2))
                    _sparse_gather_gx_t[grid](
                        goT,
                        idx,
                        wf,
                        gxT,
                        M,
                        num_outputs,
                        IN,
                        K,
                        B_M2,
                        B_O2,
                        B_K2,
                    )
                grad_x = gxT.t().contiguous()
        return (
            grad_x.to(in_dtype) if grad_x is not None else None,
            None,
            grad_w.to(w_dtype) if grad_w is not None else None,
        )


def triton_sparse_gather(
    x: torch.Tensor,
    connection_indices: torch.Tensor,
    weight: torch.Tensor,
    chunk_size: int = 2048,  # accepted for signature parity; unused
    *,
    B_M: int | None = None,  # accepted for back-compat; tiles are fixed
    B_O: int | None = None,
) -> torch.Tensor:
    """Return ``out[m, o] = sum_k x[m, connection_indices[o, k]] * weight[o, k]``.

    Drop-in for :func:`fused_sparse_gather`: ``x`` is ``[M, in_features]``;
    ``connection_indices`` and ``weight`` are ``[out_features, K]``.
    Differentiable w.r.t. ``x`` and ``weight``; indices are fixed.

    Requires a CUDA tensor and a working Triton install (see :data:`HAS_TRITON`).
    ``chunk_size``/``B_M``/``B_O`` are ignored (tiles are fixed, swept-in).
    """
    if not x.is_cuda:
        raise RuntimeError("triton_sparse_gather requires a CUDA tensor")
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
            "triton_sparse_gather supports float16, bfloat16, and float32 inputs "
            f"and weights; got {x.dtype} and {weight.dtype}"
        )
    if not triton_gather_available():
        raise RuntimeError(
            "triton_sparse_gather requires Triton, which is not available. "
            "Use fused_sparse_gather (use_fused_gather=True) instead, or install triton."
        )
    return _TritonSparseGatherTransposed.apply(x, connection_indices, weight)


def _selftest_deterministic_gradx() -> None:
    """Self-test: deterministic (default) vs atomic (opt-out) grad_x.

    Checks (per shape): deterministic result bit-identical across 3 repeats;
    deterministic and atomic each within fp32-reassociation tolerance of an
    fp64 dense reference (computed on CPU, so itself deterministic); the two
    paths within tolerance of each other. Also exercises the workspace-cap
    fallback (one-time RuntimeWarning + atomic result) and in-place rewire
    invalidation of the CSR setup cache. Needs CUDA + Triton.
    """
    global ATOMIC_GRADX, _DET_GX_WORKSPACE_CAP_BYTES, _det_gx_warned_fallback
    if not triton_gather_available():
        print("SKIP: self-test needs CUDA + Triton", flush=True)
        return
    os.environ.pop("DENDRITIC_ATOMIC_GRADX", None)  # module flag drives
    dev = "cuda"
    torch.manual_seed(0)

    def run_gradx(x, idx, w, go):
        y = triton_sparse_gather(x, idx, w)
        (gx,) = torch.autograd.grad(y, x, go)
        return gx

    def dense_ref(idx, w, go, M, IN):
        ref = torch.zeros(M, IN, dtype=torch.float64)
        vals = go.double().cpu().unsqueeze(2) * w.double().cpu().unsqueeze(0)
        ref.index_add_(1, idx.reshape(-1).long().cpu(), vals.reshape(M, idx.numel()))
        return ref

    # (M, IN, O, K, dtype, force_in_row_duplicates)
    cases = [
        (64, 256, 512, 16, torch.float32, False),
        (33, 130, 1000, 33, torch.float32, True),
        (128, 1024, 2048, 64, torch.float32, False),
        (64, 512, 1024, 32, torch.bfloat16, False),
    ]
    for M, IN, O, K, dtype, force_dups in cases:
        x = torch.randn(M, IN, device=dev, dtype=dtype, requires_grad=True)
        w = 0.1 * torch.randn(O, K, device=dev, dtype=dtype)
        idx = torch.randint(0, IN, (O, K), device=dev, dtype=torch.int32)
        if force_dups:
            idx[:, -1] = idx[:, 0]  # guaranteed duplicates within each row
            idx[0] = 0  # one massively hot input
        go = torch.randn(M, O, device=dev, dtype=dtype)

        ATOMIC_GRADX = True
        gx_atomic = run_gradx(x, idx, w, go)
        ATOMIC_GRADX = False
        runs = [run_gradx(x, idx, w, go) for _ in range(3)]
        assert torch.equal(runs[0], runs[1]) and torch.equal(runs[0], runs[2]), (
            f"deterministic grad_x not bit-identical across repeats at "
            f"{(M, IN, O, K, dtype)}"
        )

        ref = dense_ref(idx, w, go, M, IN)
        tol = (5e-5 if dtype == torch.float32 else 5e-2) * (
            1.0 + ref.abs().max().item()
        )
        err_det = (runs[0].double().cpu() - ref).abs().max().item()
        err_atomic = (gx_atomic.double().cpu() - ref).abs().max().item()
        gap = (runs[0].double() - gx_atomic.double()).abs().max().item()
        assert err_det <= tol, f"det vs fp64 ref: {err_det:.3e} > {tol:.3e}"
        assert err_atomic <= tol, f"atomic vs fp64 ref: {err_atomic:.3e} > {tol:.3e}"
        assert gap <= 2 * tol, f"det vs atomic: {gap:.3e} > {2 * tol:.3e}"
        print(
            f"PASS  M={M} IN={IN} O={O} K={K} {str(dtype).split('.')[-1]:8s}"
            f" dups={force_dups}  det_err={err_det:.3e}"
            f" atomic_err={err_atomic:.3e} det_vs_atomic={gap:.3e}",
            flush=True,
        )

    # in-place rewire invalidation: mutating idx must bump its version, force
    # a fresh CSR setup, and yield the gradient of the NEW topology.
    M, IN, O, K = 32, 128, 256, 8
    x = torch.randn(M, IN, device=dev, requires_grad=True)
    w = 0.1 * torch.randn(O, K, device=dev)
    idx = torch.randint(0, IN, (O, K), device=dev, dtype=torch.int32)
    go = torch.randn(M, O, device=dev)
    run_gradx(x, idx, w, go)  # populate the cache
    misses_before = _gx_setup_misses
    idx[0, 0] = (int(idx[0, 0].item()) + 1) % IN  # in-place "rewire"
    gx_new = run_gradx(x, idx, w, go)
    assert _gx_setup_misses == misses_before + 1, "rewire did not invalidate"
    ref = dense_ref(idx, w, go, M, IN)
    err = (gx_new.double().cpu() - ref).abs().max().item()
    assert err <= 5e-5 * (1.0 + ref.abs().max().item()), err
    print(f"PASS  rewire invalidation (err={err:.3e})", flush=True)

    # workspace-cap fallback: shrink the cap below one [IN, M] block -> the
    # deterministic default must warn once and produce the atomic result.
    old_cap, old_warned = _DET_GX_WORKSPACE_CAP_BYTES, _det_gx_warned_fallback
    ATOMIC_GRADX = True
    ref_atomic = run_gradx(x, idx, w, go)
    ATOMIC_GRADX = False
    _DET_GX_WORKSPACE_CAP_BYTES = 4 * IN * M - 1
    _det_gx_warned_fallback = False
    try:
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            gx_fb = run_gradx(x, idx, w, go)
            run_gradx(x, idx, w, go)  # second call: warning must NOT repeat
        hits = [wi for wi in wlist if "falling back" in str(wi.message)]
        assert len(hits) == 1, f"expected exactly one fallback warning, got {len(hits)}"
        gap = (gx_fb - ref_atomic).abs().max().item()
        assert gap <= 1e-4 * (1.0 + ref_atomic.abs().max().item()), gap
        print(f"PASS  workspace-cap fallback (one warning, gap={gap:.3e})", flush=True)
    finally:
        _DET_GX_WORKSPACE_CAP_BYTES = old_cap
        _det_gx_warned_fallback = old_warned
        ATOMIC_GRADX = False
    print("deterministic grad_x self-test: all checks passed", flush=True)


__all__ = [
    "ATOMIC_GRADX",
    "HAS_TRITON",
    "gx_setup_cache_stats",
    "triton_gather_available",
    "triton_sparse_gather",
]


if __name__ == "__main__":
    _selftest_deterministic_gradx()
