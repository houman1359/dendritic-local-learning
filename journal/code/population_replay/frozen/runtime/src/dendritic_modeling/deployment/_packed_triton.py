"""Opt-in frozen BF16 indexed inference with compact resident connectivity.

The gather/reduce layout follows the native ``triton_fused_chunked`` forward:
small output and batch tiles reuse each fan-in tile. This inference-only kernel
reads BF16 values and compact indices directly; it never expands or caches a
weight/index matrix. Only an input activation transpose and the output tensor
are allocated. FP32 reduction order can differ from the generic reference.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Optional backend; default packed inference stays usable.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _block_reduce_bf16(
        X,
        W,
        OUT,
        TOTAL,
        OUTPUTS,
        K,
        SX0: tl.constexpr,
        SX1: tl.constexpr,
        BO: tl.constexpr,
        BK: tl.constexpr,
    ):
        # Flatten (batch, output) so both axes have masked tails without a
        # Python output-chunk loop. Connectivity is implicit and never stored.
        item = tl.program_id(0).to(tl.int64) * BO + tl.arange(0, BO)
        output = item % OUTPUTS
        batch = item // OUTPUTS
        k = tl.arange(0, BK)
        mask = (item[:, None] < TOTAL) & (k[None, :] < K)
        weight = tl.load(W + output[:, None] * K + k[None, :], mask=mask, other=0).to(
            tl.float32
        )
        x = tl.load(
            X + batch[:, None] * SX0 + (output[:, None] * K + k[None, :]) * SX1,
            mask=mask,
            other=0,
        ).to(tl.float32)
        result = tl.sum(x * weight, axis=1)
        tl.store(OUT + item, result, mask=item < TOTAL)

    @triton.jit
    def _gather_bf16(
        XT,
        W,
        IDX,
        OUT,
        M,
        OUTPUTS,
        K,
        PERIOD: tl.constexpr,
        BO: tl.constexpr,
        BK: tl.constexpr,
        BM: tl.constexpr,
    ):
        o = tl.program_id(0) * BO + tl.arange(0, BO)
        m = tl.program_id(1) * BM + tl.arange(0, BM)
        o64 = o.to(tl.int64)
        ir = o64
        if PERIOD > 0:
            ir = o64 % PERIOD
        acc = tl.zeros((BO, BM), tl.float32)
        for begin in range(tl.cdiv(K, BK)):
            k = begin * BK + tl.arange(0, BK)
            live = (o[:, None] < OUTPUTS) & (k[None, :] < K)
            # Native uint16 loads zero-extend, including indices above 32767.
            idx = tl.load(IDX + ir[:, None] * K + k[None, :], mask=live, other=0).to(
                tl.int64
            )
            weight = tl.load(W + o64[:, None] * K + k[None, :], mask=live, other=0).to(
                tl.float32
            )
            x = tl.load(
                XT + idx[:, :, None] * M + m[None, None, :],
                mask=live[:, :, None] & (m[None, None, :] < M),
                other=0,
            ).to(tl.float32)
            acc += tl.sum(x * weight[:, :, None], axis=1)
        tl.store(
            OUT + m[None, :].to(tl.int64) * OUTPUTS + o64[:, None],
            acc,
            mask=(o[:, None] < OUTPUTS) & (m[None, :] < M),
        )


def packed_bf16_gather(inputs, values, indices, period=None):
    """Run a validated packed bank; topology validity is owned by its module.

    Inputs must be CUDA FP32/FP16/BF16 and values actual BF16. Index tensors
    remain int16, uint16 or int32 throughout. Unsupported explicit opt-ins
    raise rather than silently switching backends.
    """
    if triton is None:
        raise RuntimeError("triton_bf16 requires the optional Triton package")
    if (
        inputs.device.type != "cuda"
        or inputs.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or values.dtype != torch.bfloat16
        or indices.dtype not in (torch.int16, torch.uint16, torch.int32)
        or inputs.ndim != 2
        or values.ndim != 2
        or indices.ndim != 2
        or inputs.device != values.device
        or inputs.device != indices.device
        or not values.is_contiguous()
        or not indices.is_contiguous()
    ):
        raise ValueError("Invalid CUDA input, BF16 values or compact topology")
    if inputs.requires_grad or values.requires_grad:
        raise RuntimeError("Packed Triton inference does not accept gradients")
    outputs, contacts = values.shape
    if period is not None and (type(period) is not int or not 1 <= period < outputs):
        raise ValueError("Invalid compact row period")
    if (
        outputs < 1
        or contacts < 1
        or tuple(indices.shape) != (period or outputs, contacts)
    ):
        raise ValueError("Compact topology shape changed")
    rows = inputs.shape[0]
    result = inputs.new_empty((rows, outputs))
    if rows:
        transposed = inputs.t().contiguous()
        bm = min(16, triton.next_power_of_2(rows))
        _gather_bf16[(triton.cdiv(outputs, 2), triton.cdiv(rows, bm))](
            transposed,
            values,
            indices,
            result,
            rows,
            outputs,
            contacts,
            PERIOD=period or 0,
            BO=2,
            BK=128,
            BM=bm,
            num_warps=8,
            enable_fp_fusion=False,
        )
    return result


def packed_bf16_block_reduce(inputs, values):
    """Implicit contiguous block sums, with FP32-promoted reference outputs.

    Values already contain the effective native transform. No transform is
    applied again. Conductance APIs remain on the owning packed module.
    """
    if triton is None:
        raise RuntimeError("triton_bf16_block requires the optional Triton package")
    if (
        inputs.device.type != "cuda"
        or inputs.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or values.dtype != torch.bfloat16
        or inputs.ndim != 2
        or values.ndim != 2
        or inputs.device != values.device
        or not values.is_contiguous()
    ):
        raise ValueError("Invalid CUDA input or effective BF16 block values")
    if inputs.requires_grad or values.requires_grad:
        raise RuntimeError("Packed Triton inference does not accept gradients")
    outputs, contacts = values.shape
    if outputs < 1 or contacts < 1 or inputs.shape[1] != outputs * contacts:
        raise ValueError("Implicit block dimensions changed")
    if contacts > 4096:
        raise ValueError("Fused BF16 block fan-in exceeds the supported 4096")
    rows = inputs.shape[0]
    result = inputs.new_empty((rows, outputs), dtype=torch.float32)
    if rows:
        bk = triton.next_power_of_2(contacts)
        bo = max(1, min(256, 1024 // bk))
        _block_reduce_bf16[(triton.cdiv(rows * outputs, bo),)](
            inputs,
            values,
            result,
            rows * outputs,
            outputs,
            contacts,
            SX0=inputs.stride(0),
            SX1=inputs.stride(1),
            BO=bo,
            BK=bk,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return result
