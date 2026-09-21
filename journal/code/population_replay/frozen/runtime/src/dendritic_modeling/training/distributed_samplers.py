"""Distributed samplers shared across training strategies."""

from __future__ import annotations

import math

from torch.utils.data import Dataset, Sampler


class PaddedDistributedEvalSampler(Sampler[int]):
    """Distributed eval sampler with explicit padding sentinels.

    Each rank receives ``ceil(N / world_size)`` indices. Real indices are
    interleaved across ranks and any remainder is padded with ``-1`` so every
    rank yields the same number of batches. Callers must exclude the padding
    entries from aggregated metrics.
    """

    def __init__(self, dataset: Dataset, num_replicas: int, rank: int):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        n = len(dataset)
        self.samples_per_rank = math.ceil(n / num_replicas)
        self.real_indices = list(range(rank, n, num_replicas))
        self.num_padding = self.samples_per_rank - len(self.real_indices)
        self._indices = self.real_indices + ([-1] * self.num_padding)

    def __iter__(self):
        return iter(self._indices)

    def __len__(self) -> int:
        return self.samples_per_rank


__all__ = ["PaddedDistributedEvalSampler"]
