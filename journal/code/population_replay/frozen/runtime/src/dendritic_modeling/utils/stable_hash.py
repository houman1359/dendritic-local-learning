"""Stable hashing helpers for reproducible seeded model construction."""

from __future__ import annotations

import hashlib


def stable_seed_offset(*parts: object) -> int:
    """Return a deterministic 64-bit seed offset for structured RNG streams."""

    text = "|".join(str(part) for part in parts)
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16)


__all__ = ["stable_seed_offset"]
