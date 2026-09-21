"""Lossless, row-aligned connectivity coding, independent of weight precision.

This format preserves contact order, multiplicity and selected zero weights.
Packing is an offline CPU operation. The optional Triton consumer reads the
byte stream directly; ``unpack_indices`` is a CPU reference/loader operation.
No implicit pruning, sorting or approximation is performed.
"""

from __future__ import annotations

SCHEMA = "dendrinet_connectivity_lsb_rows/v1"


def descriptor(rows, contacts, domain):
    if any(type(x) is not int or x < 1 for x in (rows, contacts, domain)):
        raise ValueError("Positive integer dimensions required")
    if domain > 2**31:
        raise ValueError("Input domain exceeds supported 31-bit indices")
    bits = max(1, (domain - 1).bit_length())
    return dict(schema=SCHEMA, rows=rows, contacts=contacts, domain=domain,
                bits=bits, row_bytes=(contacts * bits + 7) // 8)


def validate_descriptor(spec):
    if not isinstance(spec, dict) or spec != descriptor(
        spec.get("rows"), spec.get("contacts"), spec.get("domain")
    ):
        raise ValueError("Unknown or noncanonical connectivity descriptor")


def pack_indices(indices, domain, *, chunk_rows=256):
    import numpy as np
    import torch

    if (not torch.is_tensor(indices) or indices.device.type != "cpu"
        or indices.ndim != 2 or indices.dtype not in
        (torch.int16, torch.uint16, torch.int32, torch.int64)):
        raise ValueError("CPU integer matrix required")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("Positive chunk_rows required")
    rows, contacts = indices.shape
    spec = descriptor(rows, contacts, domain)
    bits, rb = spec["bits"], spec["row_bytes"]
    result = np.zeros((rows, rb), dtype=np.uint8)
    positions = np.arange(rb, dtype=np.int64) * 8
    first, shift = positions // bits, positions % bits
    # A byte touches at most ceil(7/bits)+1 adjacent indices. Compute the
    # little-endian byte directly, avoiding a contacts-by-bits unpacked array.
    for start in range(0, rows, chunk_rows):
        values = indices[start:start + chunk_rows].numpy().astype(np.uint64)
        if np.any(values >= domain):
            raise ValueError("Out-of-domain or negative connection")
        acc = values[:, first] >> shift.astype(np.uint64)
        for j in range(1, (7 + bits - 1) // bits + 1):
            next_index = first + j
            left = j * bits - shift
            live = (next_index < contacts) & (left < 8)
            safe = np.minimum(next_index, contacts - 1)
            term = values[:, safe] << np.clip(left, 0, 63).astype(np.uint64)
            acc |= np.where(live, term, 0)
        result[start:start + len(values)] = (acc & 255).astype(np.uint8)
    return torch.from_numpy(result), spec


def unpack_indices(packed, spec, *, chunk_rows=256):
    import numpy as np
    import torch

    validate_descriptor(spec)
    if (not torch.is_tensor(packed) or packed.device.type != "cpu"
        or packed.dtype != torch.uint8 or not packed.is_contiguous()
        or tuple(packed.shape) != (spec["rows"], spec["row_bytes"])):
        raise ValueError("Canonical CPU byte matrix required")
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("Positive chunk_rows required")
    bits, contacts = spec["bits"], spec["contacts"]
    position = np.arange(contacts, dtype=np.int64) * bits
    byte, shift = position // 8, position % 8
    raw = packed.numpy()
    unused = spec["row_bytes"] * 8 - contacts * bits
    if unused and np.any(raw[:, -1] >> (8 - unused)):
        raise ValueError("Nonzero padding bits")
    result = np.empty((spec["rows"], contacts), dtype=np.int64)
    for start in range(0, len(raw), chunk_rows):
        block = raw[start:start + chunk_rows]
        acc = np.zeros((len(block), contacts), dtype=np.uint64)
        for j in range((bits + 14) // 8):
            source = byte + j
            live = source < spec["row_bytes"]
            piece = block[:, np.minimum(source, spec["row_bytes"] - 1)].astype(np.uint64)
            acc |= np.where(live, piece, 0) << np.uint64(8 * j)
        values = (acc >> shift.astype(np.uint64)) & np.uint64((1 << bits) - 1)
        if np.any(values >= spec["domain"]):
            raise ValueError("Encoded index exceeds declared domain")
        result[start:start + len(block)] = values
    return torch.from_numpy(result)


def pack_reference(rows, domain):
    """Independent Python integer oracle for byte-order/boundary tests."""
    spec = descriptor(len(rows), len(rows[0]), domain)
    result = []
    for row in rows:
        if len(row) != spec["contacts"]:
            raise ValueError("Ragged contacts")
        word = 0
        for slot, value in enumerate(row):
            if type(value) is not int or not 0 <= value < domain:
                raise ValueError("Invalid index")
            word |= value << (slot * spec["bits"])
        result.append(word.to_bytes(spec["row_bytes"], "little"))
    return b"".join(result)
