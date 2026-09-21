"""Low-rank broadcast helpers for local credit assignment."""

from __future__ import annotations

import torch


class LocalLearningLowRankBroadcastMixin:
    """Cached low-rank random broadcast projections."""

    def _get_low_rank_broadcast_factors(
        self,
        layer_idx: int,
        in_features: int,
        out_features: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached low-rank random broadcast factors for a layer."""

        rank = max(
            1,
            min(
                int(getattr(self.local_cfg, "broadcast_rank", 4)),
                int(in_features),
                int(out_features),
            ),
        )

        cache_key = (
            "low_rank_broadcast",
            layer_idx,
            in_features,
            out_features,
            rank,
            str(device),
            str(dtype),
        )

        cached = self._broadcast_cache.get(cache_key)

        if cached is not None:

            return cached

        generator = torch.Generator()

        seed = (
            15485863
            + 7919 * int(layer_idx)
            + 104729 * int(in_features)
            + 53 * int(out_features)
        )

        generator.manual_seed(seed)

        left = torch.randn(in_features, rank, generator=generator, dtype=dtype)

        right = torch.randn(rank, out_features, generator=generator, dtype=dtype)

        left = left / left.norm(dim=0, keepdim=True).clamp_min(1e-6)

        right = right / right.norm(dim=1, keepdim=True).clamp_min(1e-6)

        left = left.to(device=device)

        right = right.to(device=device)

        self._broadcast_cache[cache_key] = (left, right)

        return left, right

    def _compute_low_rank_broadcast(
        self,
        delta: torch.Tensor,
        layer_idx: int,
        out_features: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Project soma error into a layer-specific vector broadcast."""

        if delta.dim() == 1:

            delta = delta.unsqueeze(-1)

        elif delta.dim() != 2:

            delta = delta.view(delta.size(0), -1)

        in_features = delta.size(1)

        if in_features <= 1:

            return delta.mean(dim=1, keepdim=True).expand(-1, out_features).to(dtype)

        left, right = self._get_low_rank_broadcast_factors(
            layer_idx=layer_idx,
            in_features=in_features,
            out_features=out_features,
            device=device,
            dtype=dtype,
        )

        scale = float(getattr(self.local_cfg, "broadcast_init_scale", 1.0)) / (
            float(left.size(1)) ** 0.5
        )

        return (delta.to(dtype=dtype, device=device) @ left @ right) * scale


__all__ = ["LocalLearningLowRankBroadcastMixin"]
