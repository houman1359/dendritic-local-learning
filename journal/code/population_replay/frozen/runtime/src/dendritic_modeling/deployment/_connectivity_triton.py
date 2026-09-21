"""Inference-only BF16 gathers directly from bit-packed contact indices."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather(XT, W, IDX, OUT, M, OUTPUTS, K,
            BITS: tl.constexpr, ROW_BYTES: tl.constexpr,
            PERIOD: tl.constexpr, GROUP: tl.constexpr,
            BO: tl.constexpr, BK: tl.constexpr, BM: tl.constexpr):
    o = tl.program_id(0) * BO + tl.arange(0, BO)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    o64 = o.to(tl.int64)
    ir = o64 // GROUP
    if PERIOD > 0:
        ir = ir % PERIOD
    acc = tl.zeros((BO, BM), tl.float32)
    for begin in range(tl.cdiv(K, BK)):
        k = begin * BK + tl.arange(0, BK)
        live = (o[:, None] < OUTPUTS) & (k[None, :] < K)
        bit = k * BITS
        byte = bit // 8
        shift = bit % 8
        encoded = tl.full((BO, BK), 0, tl.uint64)
        for j in tl.static_range((BITS + 14) // 8):
            fragment = tl.load(IDX + ir[:, None] * ROW_BYTES + byte[None, :] + j,
                mask=live & (byte[None, :] + j < ROW_BYTES), other=0).to(tl.uint64)
            encoded = encoded | (fragment << (8 * j))
        idx = ((encoded >> shift[None, :]) & ((1 << BITS) - 1)).to(tl.int64)
        weight = tl.load(W + o64[:, None] * K + k[None, :], mask=live, other=0).to(tl.float32)
        x = tl.load(XT + idx[:, :, None] * M + m[None, None, :],
            mask=live[:, :, None] & (m[None, None, :] < M), other=0).to(tl.float32)
        acc += tl.sum(x * weight[:, :, None], axis=1)
    tl.store(OUT + m[None, :].to(tl.int64) * OUTPUTS + o64[:, None], acc,
        mask=(o[:, None] < OUTPUTS) & (m[None, :] < M))


def bitpacked_bf16_gather(inputs, values, packed, spec, *, period=None, group=1):
    """Caller owns validation of decoded indices; no expanded GPU index cache.

    ``group`` consecutive output rows may share support, with independent
    values. ``period`` applies to the group index. The canonical encoder and
    artifact loader must prove that mapping exactly before this entry point.
    """
    from dendritic_modeling.deployment.connectivity_codec import validate_descriptor
    validate_descriptor(spec)
    if (inputs.ndim != 2 or values.ndim != 2 or inputs.device.type != "cuda"
        or values.device != inputs.device or packed.device != inputs.device
        or inputs.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or values.dtype != torch.bfloat16 or packed.dtype != torch.uint8
        or not values.is_contiguous() or not packed.is_contiguous()
        or values.requires_grad or inputs.requires_grad
        or type(group) is not int or group < 1):
        raise ValueError("Invalid inference input/value/topology state")
    outputs, contacts = values.shape
    expected_rows = (outputs + group - 1) // group
    if period is not None:
        if type(period) is not int or not 1 <= period < expected_rows:
            raise ValueError("Invalid support period")
        expected_rows = period
    if (contacts != spec['contacts'] or expected_rows != spec['rows']
        or tuple(packed.shape) != (expected_rows, spec['row_bytes'])
        or inputs.shape[1] != spec['domain']):
        raise ValueError("Connectivity dimensions disagree")
    rows = inputs.shape[0]
    result = inputs.new_empty((rows, outputs))
    if rows:
        transposed = inputs.t().contiguous()
        bm = min(16, triton.next_power_of_2(rows))
        _gather[(triton.cdiv(outputs, 2), triton.cdiv(rows, bm))](
            transposed, values, packed, result, rows, outputs, contacts,
            BITS=spec['bits'], ROW_BYTES=spec['row_bytes'], PERIOD=period or 0,
            GROUP=group, BO=2, BK=128, BM=bm, num_warps=8, enable_fp_fusion=False)
    return result
