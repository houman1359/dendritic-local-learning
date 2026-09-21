"""Fused ELL gather-GEMM kernels for indexed dendritic synapses.

Development provenance: Houman Safaai developed this later fixed-fan-in ELL,
feature-major, shunting, and gated-GLU family. It was informed by the earlier
native gather performance line developed by Ryan Whalen, but defines a separate
layout and fused-operator implementation boundary.

Computes, without materializing any gathered intermediate:

- ``ell_gather_gemm``:  ``out[m, o] = sum_k T(w[o, k]) * x[m, idx[o, k]]``
  with the weight transform ``T`` (identity / softplus / exp) fused into the
  weight load — one memory pass over ``x`` gathers instead of the eager
  path's gather-materialize-reduce.
- ``ell_shunting_forward``: the dendritic branch op in ONE kernel —
  excitatory and inhibitory banks gathered and reduced together, then
  ``V = (E + C) / (1 + E + I + G)`` applied in registers. This is the
  op-level fusion no generic framework kernel provides.

v0 scope: forward/inference only (deployment path; training backward is a
follow-up), fp32/bf16 inputs, int32/int64/uint16-widened indices. Layout
note: ``x`` may have arbitrary strides; a ``[D, M]``-contiguous activation
layout (pass ``x.t().contiguous().t()``) makes the gather coalesced across
the batch dimension and is exposed to benchmarks via strides alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:  # Triton ships with CUDA builds of torch; degrade gracefully elsewhere.
    import triton
    import triton.language as tl

    _TRITON = True
except Exception:  # pragma: no cover - CPU-only environments
    _TRITON = False

TRANSFORMS = {"identity": 0, "softplus": 1, "exp": 2}

__all__ = [
    "ell_gather_gemm",
    "ell_gather_gemm_train",
    "ell_shunting_forward",
    "reference_gather_gemm",
    "reference_shunting_forward",
    "triton_available",
]


def triton_available() -> bool:
    return _TRITON and torch.cuda.is_available()


def _apply_transform(weights: torch.Tensor, transform: str) -> torch.Tensor:
    if transform == "softplus":
        return F.softplus(weights)
    if transform == "exp":
        return torch.exp(weights)
    return weights


def reference_gather_gemm(
    x: torch.Tensor,
    indices: torch.Tensor,
    pre_w: torch.Tensor,
    transform: str = "softplus",
) -> torch.Tensor:
    """Eager reference: gather-materialize-reduce (the path being replaced)."""
    weights = _apply_transform(pre_w, transform)
    gathered = x[:, indices.long()]  # [M, O, K]
    return torch.einsum("mok,ok->mo", gathered, weights)


def reference_shunting_forward(
    x: torch.Tensor,
    idx_exc: torch.Tensor,
    w_exc: torch.Tensor,
    idx_inh: torch.Tensor,
    w_inh: torch.Tensor,
    somatic: float = 0.0,
    conductance: float = 0.0,
    transform: str = "softplus",
) -> torch.Tensor:
    excitation = reference_gather_gemm(x, idx_exc, w_exc, transform)
    inhibition = reference_gather_gemm(x, idx_inh, w_inh, transform)
    return (excitation + somatic) / (1.0 + excitation + inhibition + conductance)


if _TRITON:

    @triton.jit
    def _transform_weights(w, TRANSFORM: tl.constexpr):
        if TRANSFORM == 1:  # numerically stable softplus
            w = tl.where(w > 20.0, w, tl.log(1.0 + tl.exp(w)))
        elif TRANSFORM == 2:
            w = tl.exp(w)
        return w

    @triton.jit
    def _ell_gather_gemm_kernel(
        x_ptr,
        idx_ptr,
        w_ptr,
        out_ptr,
        M,
        K,
        stride_xm,
        stride_xd,
        stride_io,
        stride_ik,
        stride_wo,
        stride_wk,
        stride_om,
        stride_oo,
        TRANSFORM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        o = tl.program_id(1)
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < M
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K
            idx = tl.load(
                idx_ptr + o * stride_io + k_offs * stride_ik, mask=k_mask, other=0
            ).to(tl.int64)
            w = tl.load(
                w_ptr + o * stride_wo + k_offs * stride_wk, mask=k_mask, other=0.0
            ).to(tl.float32)
            w = _transform_weights(w, TRANSFORM)
            w = tl.where(k_mask, w, 0.0)
            gathered = tl.load(
                x_ptr + m_offs[:, None] * stride_xm + idx[None, :] * stride_xd,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(gathered * w[None, :], axis=1)
        tl.store(out_ptr + m_offs * stride_om + o * stride_oo, acc, mask=m_mask)

    @triton.jit
    def _ell_shunting_kernel(
        x_ptr,
        idx_e_ptr,
        w_e_ptr,
        idx_i_ptr,
        w_i_ptr,
        out_ptr,
        M,
        KE,
        KI,
        somatic,
        conductance,
        stride_xm,
        stride_xd,
        stride_ieo,
        stride_iek,
        stride_weo,
        stride_wek,
        stride_iio,
        stride_iik,
        stride_wio,
        stride_wik,
        stride_om,
        stride_oo,
        TRANSFORM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        o = tl.program_id(1)
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < M
        excitation = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, KE, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offs < KE
            idx = tl.load(
                idx_e_ptr + o * stride_ieo + k_offs * stride_iek, mask=k_mask, other=0
            ).to(tl.int64)
            w = tl.load(
                w_e_ptr + o * stride_weo + k_offs * stride_wek, mask=k_mask, other=0.0
            ).to(tl.float32)
            w = tl.where(k_mask, _transform_weights(w, TRANSFORM), 0.0)
            gathered = tl.load(
                x_ptr + m_offs[:, None] * stride_xm + idx[None, :] * stride_xd,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            excitation += tl.sum(gathered * w[None, :], axis=1)
        inhibition = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, KI, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offs < KI
            idx = tl.load(
                idx_i_ptr + o * stride_iio + k_offs * stride_iik, mask=k_mask, other=0
            ).to(tl.int64)
            w = tl.load(
                w_i_ptr + o * stride_wio + k_offs * stride_wik, mask=k_mask, other=0.0
            ).to(tl.float32)
            w = tl.where(k_mask, _transform_weights(w, TRANSFORM), 0.0)
            gathered = tl.load(
                x_ptr + m_offs[:, None] * stride_xm + idx[None, :] * stride_xd,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            inhibition += tl.sum(gathered * w[None, :], axis=1)
        voltage = (excitation + somatic) / (1.0 + excitation + inhibition + conductance)
        tl.store(out_ptr + m_offs * stride_om + o * stride_oo, voltage, mask=m_mask)

    @triton.jit
    def _ell_glu_kernel(
        x_ptr,
        idx_g_ptr,
        w_g_ptr,
        idx_u_ptr,
        w_u_ptr,
        out_ptr,
        M,
        KG,
        KU,
        stride_xm,
        stride_xd,
        stride_igo,
        stride_igk,
        stride_wgo,
        stride_wgk,
        stride_iuo,
        stride_iuk,
        stride_wuo,
        stride_wuk,
        stride_om,
        stride_oo,
        TRANSFORM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        o = tl.program_id(1)
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < M
        gate = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, KG, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offs < KG
            idx = tl.load(
                idx_g_ptr + o * stride_igo + k_offs * stride_igk, mask=k_mask, other=0
            ).to(tl.int64)
            w = tl.load(
                w_g_ptr + o * stride_wgo + k_offs * stride_wgk, mask=k_mask, other=0.0
            ).to(tl.float32)
            w = tl.where(k_mask, _transform_weights(w, TRANSFORM), 0.0)
            gathered = tl.load(
                x_ptr + m_offs[:, None] * stride_xm + idx[None, :] * stride_xd,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            gate += tl.sum(gathered * w[None, :], axis=1)
        up = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, KU, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offs < KU
            idx = tl.load(
                idx_u_ptr + o * stride_iuo + k_offs * stride_iuk, mask=k_mask, other=0
            ).to(tl.int64)
            w = tl.load(
                w_u_ptr + o * stride_wuo + k_offs * stride_wuk, mask=k_mask, other=0.0
            ).to(tl.float32)
            w = tl.where(k_mask, _transform_weights(w, TRANSFORM), 0.0)
            gathered = tl.load(
                x_ptr + m_offs[:, None] * stride_xm + idx[None, :] * stride_xd,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            up += tl.sum(gathered * w[None, :], axis=1)
        fused = (gate / (1.0 + tl.exp(-gate))) * up
        tl.store(out_ptr + m_offs * stride_om + o * stride_oo, fused, mask=m_mask)

    @triton.jit
    def _transform_grad(w, TRANSFORM: tl.constexpr):
        """dT/dw for the fused transforms."""
        if TRANSFORM == 1:  # softplus' = sigmoid
            return tl.sigmoid(w)
        elif TRANSFORM == 2:  # exp' = exp
            return tl.exp(w)
        return w * 0.0 + 1.0  # identity

    @triton.jit
    def _ell_grad_w_kernel(
        x_ptr,
        idx_ptr,
        w_ptr,
        grad_out_ptr,
        grad_w_ptr,
        M,
        K,
        stride_xm,
        stride_xd,
        stride_io,
        stride_ik,
        stride_wo,
        stride_wk,
        stride_gm,
        stride_go,
        TRANSFORM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        o = tl.program_id(0)
        kb = tl.program_id(1)
        k_offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        idx = tl.load(
            idx_ptr + o * stride_io + k_offs * stride_ik, mask=k_mask, other=0
        ).to(tl.int64)
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for m0 in range(0, M, BLOCK_M):
            m_offs = m0 + tl.arange(0, BLOCK_M)
            m_mask = m_offs < M
            grad = tl.load(
                grad_out_ptr + m_offs * stride_gm + o * stride_go,
                mask=m_mask,
                other=0.0,
            ).to(tl.float32)
            gathered = tl.load(
                x_ptr + m_offs[:, None] * stride_xm + idx[None, :] * stride_xd,
                mask=m_mask[:, None] & k_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(gathered * grad[:, None], axis=0)
        w = tl.load(
            w_ptr + o * stride_wo + k_offs * stride_wk, mask=k_mask, other=0.0
        ).to(tl.float32)
        acc = acc * _transform_grad(w, TRANSFORM)
        tl.store(grad_w_ptr + o * stride_wo + k_offs * stride_wk, acc, mask=k_mask)

    @triton.jit
    def _ell_grad_x_kernel(
        idx_ptr,
        w_ptr,
        grad_out_ptr,
        grad_x_ptr,
        M,
        K,
        stride_io,
        stride_ik,
        stride_wo,
        stride_wk,
        stride_gm,
        stride_go,
        stride_xm,
        stride_xd,
        TRANSFORM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        o = tl.program_id(1)
        m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = m_offs < M
        grad = tl.load(
            grad_out_ptr + m_offs * stride_gm + o * stride_go, mask=m_mask, other=0.0
        ).to(tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = k_offs < K
            idx = tl.load(
                idx_ptr + o * stride_io + k_offs * stride_ik, mask=k_mask, other=0
            ).to(tl.int64)
            w = tl.load(
                w_ptr + o * stride_wo + k_offs * stride_wk, mask=k_mask, other=0.0
            ).to(tl.float32)
            w = tl.where(k_mask, _transform_weights(w, TRANSFORM), 0.0)
            tl.atomic_add(
                grad_x_ptr + m_offs[:, None] * stride_xm + idx[None, :] * stride_xd,
                grad[:, None] * w[None, :],
                mask=m_mask[:, None] & k_mask[None, :],
            )


def ell_gather_gemm(
    x: torch.Tensor,
    indices: torch.Tensor,
    pre_w: torch.Tensor,
    transform: str = "softplus",
    block_m: int = 64,
    block_k: int = 64,
    num_warps: int = 2,
) -> torch.Tensor:
    """Fused indexed forward. ``x`` [M, D] (any strides), returns [M, O] fp32."""
    if not triton_available() or not x.is_cuda:
        return reference_gather_gemm(x, indices, pre_w, transform)
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {sorted(TRANSFORMS)}")
    m, _ = x.shape
    out_features, k = indices.shape
    out = torch.empty((m, out_features), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), out_features)
    # Launch on the tensors' device, not the process-current one
    # (Triton cross-device race, 2026-08-20).
    with torch.cuda.device(x.device):
        _ell_gather_gemm_kernel[grid](
            x,
            indices,
            pre_w,
            out,
            m,
            k,
            x.stride(0),
            x.stride(1),
            indices.stride(0),
            indices.stride(1),
            pre_w.stride(0),
            pre_w.stride(1),
            out.stride(0),
            out.stride(1),
            TRANSFORM=TRANSFORMS[transform],
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
    return out


def ell_shunting_forward(
    x: torch.Tensor,
    idx_exc: torch.Tensor,
    w_exc: torch.Tensor,
    idx_inh: torch.Tensor,
    w_inh: torch.Tensor,
    somatic: float = 0.0,
    conductance: float = 0.0,
    transform: str = "softplus",
    block_m: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Fused E/I shunting branch forward: both banks + denominator, one pass."""
    if not triton_available() or not x.is_cuda:
        return reference_shunting_forward(
            x, idx_exc, w_exc, idx_inh, w_inh, somatic, conductance, transform
        )
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {sorted(TRANSFORMS)}")
    m, _ = x.shape
    out_features, k_exc = idx_exc.shape
    _, k_inh = idx_inh.shape
    out = torch.empty((m, out_features), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), out_features)
    # Launch on the tensors' device, not the process-current one
    # (Triton cross-device race, 2026-08-20).
    with torch.cuda.device(x.device):
        _ell_shunting_kernel[grid](
            x,
            idx_exc,
            w_exc,
            idx_inh,
            w_inh,
            out,
            m,
            k_exc,
            k_inh,
            somatic,
            conductance,
            x.stride(0),
            x.stride(1),
            idx_exc.stride(0),
            idx_exc.stride(1),
            w_exc.stride(0),
            w_exc.stride(1),
            idx_inh.stride(0),
            idx_inh.stride(1),
            w_inh.stride(0),
            w_inh.stride(1),
            out.stride(0),
            out.stride(1),
            TRANSFORM=TRANSFORMS[transform],
            BLOCK_M=block_m,
            BLOCK_K=block_k,
        )
    return out


def reference_glu_forward(
    x: torch.Tensor,
    idx_gate: torch.Tensor,
    w_gate: torch.Tensor,
    idx_up: torch.Tensor,
    w_up: torch.Tensor,
    transform: str = "identity",
) -> torch.Tensor:
    """Eager reference for the fused gated (SwiGLU-style) indexed FFN front."""
    gate = reference_gather_gemm(x, idx_gate, w_gate, transform)
    up = reference_gather_gemm(x, idx_up, w_up, transform)
    return torch.nn.functional.silu(gate) * up


def ell_glu_forward(
    x: torch.Tensor,
    idx_gate: torch.Tensor,
    w_gate: torch.Tensor,
    idx_up: torch.Tensor,
    w_up: torch.Tensor,
    transform: str = "identity",
    block_m: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Fused gated front: gate and value gathers + silu*mul in one launch.

    The two [M, O] intermediates (gate and value pre-activations) never
    round-trip HBM — each program accumulates both banks in registers and
    stores only the fused product. This is the gated-cell counterpart of
    the fused E/I shunting operator.
    """
    if not triton_available() or not x.is_cuda:
        return reference_glu_forward(x, idx_gate, w_gate, idx_up, w_up, transform)
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {sorted(TRANSFORMS)}")
    m, _ = x.shape
    out_features, k_gate = idx_gate.shape
    _, k_up = idx_up.shape
    out = torch.empty((m, out_features), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(m, block_m), out_features)
    # Launch on the tensors' device, not the process-current one
    # (Triton cross-device race, 2026-08-20).
    with torch.cuda.device(x.device):
        _ell_glu_kernel[grid](
            x,
            idx_gate,
            w_gate,
            idx_up,
            w_up,
            out,
            m,
            k_gate,
            k_up,
            x.stride(0),
            x.stride(1),
            idx_gate.stride(0),
            idx_gate.stride(1),
            w_gate.stride(0),
            w_gate.stride(1),
            idx_up.stride(0),
            idx_up.stride(1),
            w_up.stride(0),
            w_up.stride(1),
            out.stride(0),
            out.stride(1),
            TRANSFORM=TRANSFORMS[transform],
            BLOCK_M=block_m,
            BLOCK_K=block_k,
        )
    return out


class _EllGluFunction(torch.autograd.Function):
    """Training-capable fused gated front (recompute-style backward).

    Forward stores only the fused product; backward recomputes the gate and
    value pre-activations with the plain fused gather, applies the silu*mul
    chain rule elementwise, and routes each bank through the existing
    deterministic grad_w / atomic grad_x kernels.
    """

    @staticmethod
    def forward(ctx, x, idx_gate, w_gate, idx_up, w_up, transform, block_m, block_k):
        ctx.save_for_backward(x, idx_gate, w_gate, idx_up, w_up)
        ctx.transform = transform
        ctx.blocks = (block_m, block_k)
        return ell_glu_forward(
            x, idx_gate, w_gate, idx_up, w_up, transform, block_m, block_k
        )

    @staticmethod
    def backward(ctx, grad_out):
        x, idx_gate, w_gate, idx_up, w_up = ctx.saved_tensors
        transform = ctx.transform
        block_m, block_k = ctx.blocks
        grad_out = grad_out.contiguous().float()
        gate = ell_gather_gemm(x, idx_gate, w_gate, transform, block_m, block_k)
        up = ell_gather_gemm(x, idx_up, w_up, transform, block_m, block_k)
        sig = torch.sigmoid(gate)
        grad_gate = (grad_out * up * (sig * (1.0 + gate * (1.0 - sig)))).contiguous()
        grad_up = (grad_out * gate * sig).contiguous()

        def _bank_grads(indices, pre_w, upstream, need_w):
            grad_w = None
            m = x.shape[0]
            out_features, k = indices.shape
            if need_w:
                grad_w = torch.empty_like(pre_w, dtype=torch.float32)
                grid = (out_features, triton.cdiv(k, block_k))
                # Launch on the tensors' device, not the process-current one
                # (Triton cross-device race, 2026-08-20).
                with torch.cuda.device(x.device):
                    _ell_grad_w_kernel[grid](
                        x,
                        indices,
                        pre_w,
                        upstream,
                        grad_w,
                        m,
                        k,
                        x.stride(0),
                        x.stride(1),
                        indices.stride(0),
                        indices.stride(1),
                        pre_w.stride(0),
                        pre_w.stride(1),
                        upstream.stride(0),
                        upstream.stride(1),
                        TRANSFORM=TRANSFORMS[transform],
                        BLOCK_M=block_m,
                        BLOCK_K=block_k,
                    )
                grad_w = grad_w.to(pre_w.dtype)
            return grad_w

        grad_w_gate = _bank_grads(idx_gate, w_gate, grad_gate, ctx.needs_input_grad[2])
        grad_w_up = _bank_grads(idx_up, w_up, grad_up, ctx.needs_input_grad[4])

        grad_x = None
        if ctx.needs_input_grad[0]:
            m = x.shape[0]
            grad_x = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
            for indices, pre_w, upstream in (
                (idx_gate, w_gate, grad_gate),
                (idx_up, w_up, grad_up),
            ):
                out_features, k = indices.shape
                grid = (triton.cdiv(m, block_m), out_features)
                # Launch on the tensors' device, not the process-current one
                # (Triton cross-device race, 2026-08-20).
                with torch.cuda.device(indices.device):
                    _ell_grad_x_kernel[grid](
                        indices,
                        pre_w,
                        upstream,
                        grad_x,
                        m,
                        k,
                        indices.stride(0),
                        indices.stride(1),
                        pre_w.stride(0),
                        pre_w.stride(1),
                        upstream.stride(0),
                        upstream.stride(1),
                        grad_x.stride(0),
                        grad_x.stride(1),
                        TRANSFORM=TRANSFORMS[transform],
                        BLOCK_M=block_m,
                        BLOCK_K=block_k,
                    )
            grad_x = grad_x.to(x.dtype)
        return grad_x, None, grad_w_gate, None, grad_w_up, None, None, None


def ell_glu_train(
    x: torch.Tensor,
    idx_gate: torch.Tensor,
    w_gate: torch.Tensor,
    idx_up: torch.Tensor,
    w_up: torch.Tensor,
    transform: str = "identity",
    block_m: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Differentiable fused gated front (falls back to eager off-GPU)."""
    if not triton_available() or not x.is_cuda:
        gate = reference_gather_gemm(x, idx_gate, w_gate, transform)
        up = reference_gather_gemm(x, idx_up, w_up, transform)
        return torch.nn.functional.silu(gate) * up
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {sorted(TRANSFORMS)}")
    return _EllGluFunction.apply(
        x, idx_gate, w_gate, idx_up, w_up, transform, block_m, block_k
    )


class _EllGatherGemmFunction(torch.autograd.Function):
    """Training-capable fused indexed forward.

    Saves only ``(x, indices, pre_w)`` — the recompute strategy: backward
    re-reads the small operands and never materializes the eager path's
    ``[M, O, K]`` intermediate.
    """

    @staticmethod
    def forward(ctx, x, indices, pre_w, transform, block_m, block_k):
        ctx.save_for_backward(x, indices, pre_w)
        ctx.transform = transform
        ctx.blocks = (block_m, block_k)
        return ell_gather_gemm(x, indices, pre_w, transform, block_m, block_k)

    @staticmethod
    def backward(ctx, grad_out):
        x, indices, pre_w = ctx.saved_tensors
        transform = ctx.transform
        block_m, block_k = ctx.blocks
        grad_out = grad_out.contiguous()
        m = x.shape[0]
        out_features, k = indices.shape
        grad_x = grad_w = None
        if ctx.needs_input_grad[2]:
            grad_w = torch.empty_like(pre_w, dtype=torch.float32)
            grid = (out_features, triton.cdiv(k, block_k))
            # Launch on the tensors' device, not the process-current one
            # (Triton cross-device race, 2026-08-20).
            with torch.cuda.device(x.device):
                _ell_grad_w_kernel[grid](
                    x,
                    indices,
                    pre_w,
                    grad_out,
                    grad_w,
                    m,
                    k,
                    x.stride(0),
                    x.stride(1),
                    indices.stride(0),
                    indices.stride(1),
                    pre_w.stride(0),
                    pre_w.stride(1),
                    grad_out.stride(0),
                    grad_out.stride(1),
                    TRANSFORM=TRANSFORMS[transform],
                    BLOCK_M=block_m,
                    BLOCK_K=block_k,
                )
            grad_w = grad_w.to(pre_w.dtype)
        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
            grid = (triton.cdiv(m, block_m), out_features)
            # Launch on the tensors' device, not the process-current one
            # (Triton cross-device race, 2026-08-20).
            with torch.cuda.device(indices.device):
                _ell_grad_x_kernel[grid](
                    indices,
                    pre_w,
                    grad_out,
                    grad_x,
                    m,
                    k,
                    indices.stride(0),
                    indices.stride(1),
                    pre_w.stride(0),
                    pre_w.stride(1),
                    grad_out.stride(0),
                    grad_out.stride(1),
                    grad_x.stride(0),
                    grad_x.stride(1),
                    TRANSFORM=TRANSFORMS[transform],
                    BLOCK_M=block_m,
                    BLOCK_K=block_k,
                )
            grad_x = grad_x.to(x.dtype)
        return grad_x, None, grad_w, None, None, None


def ell_gather_gemm_train(
    x: torch.Tensor,
    indices: torch.Tensor,
    pre_w: torch.Tensor,
    transform: str = "softplus",
    block_m: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Differentiable fused indexed forward (falls back to eager off-GPU)."""
    if not triton_available() or not x.is_cuda:
        return reference_gather_gemm(x, indices, pre_w, transform)
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {sorted(TRANSFORMS)}")
    return _EllGatherGemmFunction.apply(x, indices, pre_w, transform, block_m, block_k)
