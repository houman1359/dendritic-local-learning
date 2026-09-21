"""Versioned, addressable topology for NEW fixed-support native cells.

This is not a decoder for historical torch.randint/randperm supports. Each
row uses an invertible integer permutation followed by cycle walking into
the input domain, giving distinct contacts without storing indices. Learned
values and any profiled input ordering still require storage. This initial
permutation family is not claimed to sample all possible supports uniformly.
"""

SCHEMA = "dendrinet_cyclewalk_mix32/v1"
MASK32 = (1 << 32) - 1


def descriptor(rows, contacts, domain, seed, bank):
    if any(type(x) is not int or x < 1 for x in (rows, contacts, domain)):
        raise ValueError("Positive integer dimensions required")
    if contacts > domain or domain > (1 << 20):
        raise ValueError("Unique contacts and at most 20-bit domain required")
    if any(type(x) is not int or not 0 <= x <= MASK32 for x in (seed, bank)):
        raise ValueError("Unsigned 32-bit seed and bank id required")
    return dict(schema=SCHEMA, rows=rows, contacts=contacts, domain=domain,
                seed=seed, bank=bank, bits=max(1, (domain - 1).bit_length()))


def validate(spec):
    if spec != descriptor(*(spec.get(k) for k in
                            ("rows", "contacts", "domain", "seed", "bank"))):
        raise ValueError("Noncanonical procedural descriptor")


def index_reference(row, contact, spec):
    """Scalar integer oracle, independent of NumPy/Triton vectorization."""
    validate(spec)
    if not 0 <= row < spec['rows'] or not 0 <= contact < spec['contacts']:
        raise ValueError("Contact outside declared bank")
    mask = (1 << spec['bits']) - 1
    shift = max(1, spec['bits'] // 2)
    key = (row ^ spec['seed'] ^ spec['bank']) & MASK32
    key = ((key ^ (key >> 16)) * 0x7FEB352D) & MASK32
    key = ((key ^ (key >> 15)) * 0x846CA68B) & MASK32
    key ^= key >> 16
    x = contact
    for count in range(1, mask + 2):
        x = (((x ^ (x >> shift)) * 0x7FEB352D) + key) & mask
        x = (((x ^ (x >> shift)) * 0x846CA68B) + (key >> 11)) & mask
        x = (x ^ (x >> shift)) & mask
        if x < spec['domain']:
            return x, count
    raise AssertionError("Cycle walking a finite permutation must terminate")


def materialize(spec, start=0, end=None):
    """CPU construction/reference; training may materialize this NEW support."""
    import numpy as np
    import torch

    validate(spec)
    end = spec['rows'] if end is None else end
    if not 0 <= start <= end <= spec['rows']:
        raise ValueError("Invalid row range")
    mask = np.uint64((1 << spec['bits']) - 1)
    shift = np.uint64(max(1, spec['bits'] // 2))
    row = np.arange(start, end, dtype=np.uint64)[:, None]
    key = row ^ np.uint64(spec['seed']) ^ np.uint64(spec['bank'])
    key = ((key ^ (key >> np.uint64(16))) * np.uint64(0x7FEB352D)) & np.uint64(MASK32)
    key = ((key ^ (key >> np.uint64(15))) * np.uint64(0x846CA68B)) & np.uint64(MASK32)
    key ^= key >> np.uint64(16)
    x = np.broadcast_to(np.arange(spec['contacts'], dtype=np.uint64),
                        (end - start, spec['contacts'])).copy()
    active = np.ones_like(x, dtype=bool)
    maximum_steps = 0
    while active.any():
        y = (((x ^ (x >> shift)) * np.uint64(0x7FEB352D)) + key) & mask
        y = (((y ^ (y >> shift)) * np.uint64(0x846CA68B)) + (key >> np.uint64(11))) & mask
        y = (y ^ (y >> shift)) & mask
        x = np.where(active, y, x)
        active = x >= spec['domain']
        maximum_steps += 1
        if maximum_steps > int(mask) + 1:
            raise AssertionError("Permutation cycle-walk bound violated")
    return torch.from_numpy(x.astype(np.int32)), maximum_steps
