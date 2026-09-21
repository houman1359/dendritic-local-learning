"""Inference prototypes: low-bit values with packed or procedural contacts.

No resident expanded value/index cache. Procedural mode is a NEW versioned
support and cannot silently replace historical trained connectivity.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _permutation(x, key, MASK: tl.constexpr, SHIFT: tl.constexpr):
    x = (((x ^ (x >> SHIFT)) * 0x7FEB352D) + key) & MASK
    x = (((x ^ (x >> SHIFT)) * 0x846CA68B) + (key >> 11)) & MASK
    return (x ^ (x >> SHIFT)) & MASK


@triton.jit
def _forward(XT, W, S, IDX, OUT, M, O, K,
             DOMAIN: tl.constexpr, LAYOUT: tl.constexpr, VALUE_BITS: tl.constexpr,
             ZERO: tl.constexpr, GROUP_SIZE: tl.constexpr, SCALE_COLUMNS: tl.constexpr,
             INDEX_BITS: tl.constexpr, ROW_BYTES: tl.constexpr,
             PERIOD: tl.constexpr, SHARE: tl.constexpr,
             SEED: tl.constexpr, BANK: tl.constexpr, PROC_BITS: tl.constexpr,
             BM: tl.constexpr, BO: tl.constexpr, BK: tl.constexpr):
    o = tl.program_id(0) * BO + tl.arange(0, BO)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    ir = o.to(tl.int64) // SHARE
    if PERIOD > 0:
        ir = ir % PERIOD
    if LAYOUT == 1:
        key = ir.to(tl.uint32) ^ SEED ^ BANK
        key = (key ^ (key >> 16)) * 0x7FEB352D
        key = (key ^ (key >> 15)) * 0x846CA68B
        key = key ^ (key >> 16)
    acc = tl.zeros((BO, BM), tl.float32)
    for begin in range(tl.cdiv(K, BK)):
        k = begin * BK + tl.arange(0, BK)
        live = (o[:, None] < O) & (k[None, :] < K)
        if LAYOUT == 0:
            bit = k * INDEX_BITS
            encoded = tl.full((BO, BK), 0, tl.uint64)
            for j in tl.static_range((INDEX_BITS + 14) // 8):
                frag = tl.load(IDX + ir[:, None] * ROW_BYTES + (bit // 8)[None, :] + j,
                    mask=live & ((bit // 8)[None, :] + j < ROW_BYTES), other=0).to(tl.uint64)
                encoded = encoded | (frag << (8 * j))
            idx = ((encoded >> (bit % 8)[None, :]) & ((1 << INDEX_BITS) - 1)).to(tl.int64)
        elif LAYOUT == 1:
            idx = tl.broadcast_to(k[None, :], (BO, BK)).to(tl.uint32)
            idx = _permutation(idx, key[:, None], (1 << PROC_BITS) - 1, max(1, PROC_BITS // 2))
            invalid = live & (idx >= DOMAIN)
            while tl.sum(tl.sum(invalid.to(tl.int32), axis=1), axis=0) > 0:
                candidate = _permutation(idx, key[:, None], (1 << PROC_BITS) - 1, max(1, PROC_BITS // 2))
                idx = tl.where(invalid, candidate, idx)
                invalid = live & (idx >= DOMAIN)
            idx = idx.to(tl.int64)
        else:
            idx = o[:, None].to(tl.int64) * K + k[None, :]
        if VALUE_BITS == 16:
            w = tl.load(W + o[:, None].to(tl.int64) * K + k[None, :], live, 0).to(tl.float32)
        else:
            if VALUE_BITS == 4:
                raw = tl.load(W + o[:, None].to(tl.int64) * tl.cdiv(K, 2) + (k // 2)[None, :], live, 0)
                code = ((raw >> ((k % 2) * 4)[None, :]) & 15).to(tl.float32)
            else:
                code = tl.load(W + o[:, None].to(tl.int64) * K + k[None, :], live, 0).to(tl.float32)
            scale = tl.load(S + o[:, None].to(tl.int64) * SCALE_COLUMNS + (k // GROUP_SIZE)[None, :], live, 0)
            w = (code - ZERO) * scale
        x = tl.load(XT + idx[:, :, None] * M + m[None, None, :],
                    live[:, :, None] & (m[None, None, :] < M), 0).to(tl.float32)
        acc += tl.sum(x * w[:, :, None], axis=1)
    tl.store(OUT + m[None, :].to(tl.int64) * O + o[:, None].to(tl.int64), acc,
             (o[:, None] < O) & (m[None, :] < M))


def gather(inputs, matrix, packed=None, spec=None, *, period=None, share=1, procedural=None, block=False):
    """Validated matrix from PackedValueMatrix; caller validates decoded indices."""
    if (inputs.ndim != 2 or inputs.device.type != 'cuda' or inputs.requires_grad
            or inputs.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or matrix.values.device != inputs.device or matrix.values.requires_grad
            or matrix.compute_dtype != torch.float32 or type(share) is not int or share < 1):
        raise ValueError('Invalid inference inputs or packed matrix')
    outputs, contacts = matrix.rows, matrix.columns
    bits = {'bf16': 16, 'int8': 8, 'int4': 4}[matrix.format]
    if matrix.values.dtype != (torch.bfloat16 if bits == 16 else torch.uint8):
        raise ValueError('Stored value format mismatch')
    if bits != 16 and (matrix.scales.dtype != torch.float32 or matrix.scales.device != inputs.device):
        raise ValueError('Stored quantizer scales mismatch')
    expected_rows = (outputs + share - 1) // share
    if period is not None:
        if type(period) is not int or not 1 <= period < expected_rows:
            raise ValueError('Invalid index period')
        expected_rows = period
    if block:
        if any(x is not None for x in (packed, spec, procedural, period)) or share != 1 or inputs.shape[1] != outputs * contacts:
            raise ValueError('Invalid implicit block')
        layout = 2
    elif procedural is not None:
        from dendritic_modeling.deployment.procedural_connectivity import validate
        validate(procedural)
        if packed is not None or spec is not None:
            raise ValueError('Ambiguous stored/procedural topology')
        if (procedural['rows'], procedural['contacts'], procedural['domain']) != (expected_rows, contacts, inputs.shape[1]):
            raise ValueError('Procedural dimensions mismatch')
        layout = 1
    else:
        from dendritic_modeling.deployment.connectivity_codec import validate_descriptor
        validate_descriptor(spec)
        if (spec['rows'], spec['contacts'], spec['domain']) != (expected_rows, contacts, inputs.shape[1]):
            raise ValueError('Packed dimensions mismatch')
        if (packed.dtype != torch.uint8 or packed.device != inputs.device or not packed.is_contiguous()
                or tuple(packed.shape) != (expected_rows, spec['row_bytes'])):
            raise ValueError('Invalid packed topology')
        layout = 0
    result = inputs.new_empty((inputs.shape[0], outputs), dtype=torch.float32 if block else inputs.dtype)
    if inputs.shape[0]:
        transposed = inputs.t().contiguous()
        bm = min(16, triton.next_power_of_2(inputs.shape[0]))
        p = procedural or {}
        s = spec or {}
        _forward[(triton.cdiv(outputs, 2), triton.cdiv(inputs.shape[0], bm))](
            transposed, matrix.values, matrix.scales if bits != 16 else matrix.values,
            packed if packed is not None else matrix.values, result, inputs.shape[0], outputs, contacts,
            DOMAIN=inputs.shape[1], LAYOUT=layout, VALUE_BITS=bits,
            ZERO=0 if matrix.nonnegative or bits == 16 else 1 << (bits - 1),
            GROUP_SIZE=matrix.group_size, SCALE_COLUMNS=triton.cdiv(contacts, matrix.group_size),
            INDEX_BITS=s.get('bits', 1), ROW_BYTES=s.get('row_bytes', 1),
            PERIOD=period or 0, SHARE=share, SEED=p.get('seed', 0), BANK=p.get('bank', 0),
            PROC_BITS=p.get('bits', 1), BM=bm, BO=2, BK=128, num_warps=8, enable_fp_fusion=False)
    return result
