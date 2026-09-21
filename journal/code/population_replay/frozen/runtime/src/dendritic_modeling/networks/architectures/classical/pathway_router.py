"""
Pathway router encoder.

This module implements a small encoder that exposes cue groups as explicit
latent pathways before the dendritic core. It supports:

- fixed routing: hard-coded feature groups per pathway
- learned routing: soft feature-to-pathway assignments initialized from groups

The router is intentionally lightweight so the downstream dendritic network
still carries the main computational burden.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.base import BaseNetwork


def _normalize_indices(indices: Sequence[int] | None) -> list[int]:
    if indices is None:
        return []
    return [int(idx) for idx in indices]


class PathwayRouter(BaseNetwork):
    """Route input features into explicit latent pathways."""

    def __init__(
        self,
        input_dim: int,
        router_mode: str = "fixed",
        pathway_groups: list[list[int]] | None = None,
        shared_indices: list[int] | None = None,
        n_pathways: int = 2,
        pathway_dim: int | None = None,
        pathway_dims: list[int] | None = None,
        router_activation: str = "none",
        learned_router_temperature: float = 1.0,
        learned_router_init_scale: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("PathwayRouter requires a positive input_dim")

        self.input_dim = int(input_dim)
        self.router_mode = str(router_mode).lower()
        if self.router_mode not in {"fixed", "learned"}:
            raise ValueError(
                "router_mode must be one of ('fixed', 'learned'), got "
                f"{router_mode!r}"
            )

        self.shared_indices = _normalize_indices(shared_indices)
        self._validate_indices(self.shared_indices, "shared_indices")
        self._validate_bounds(self.shared_indices, "shared_indices")
        shared_set = set(self.shared_indices)

        group_list = pathway_groups or []
        if not group_list:
            routed_pool = [
                idx for idx in range(self.input_dim) if idx not in shared_set
            ]
            if not routed_pool:
                raise ValueError("PathwayRouter requires at least one routed feature")
            n_pathways = max(1, int(n_pathways))
            step = max(1, len(routed_pool) // n_pathways)
            group_list = []
            start = 0
            for pathway_idx in range(n_pathways):
                end = (
                    len(routed_pool) if pathway_idx == n_pathways - 1 else start + step
                )
                group_list.append(routed_pool[start:end])
                start = end

        self.pathway_groups = [list(map(int, group)) for group in group_list]
        if not self.pathway_groups:
            raise ValueError("PathwayRouter requires at least one pathway group")

        all_grouped: list[int] = []
        for pathway_idx, group in enumerate(self.pathway_groups):
            self._validate_indices(group, f"pathway_groups[{pathway_idx}]")
            self._validate_bounds(group, f"pathway_groups[{pathway_idx}]")
            overlap = shared_set.intersection(group)
            if overlap:
                raise ValueError(
                    "shared_indices and pathway_groups must be disjoint, found overlap "
                    f"{sorted(overlap)}"
                )
            all_grouped.extend(group)

        if len(set(all_grouped)) != len(all_grouped):
            raise ValueError("pathway_groups must not reuse the same routed feature")

        self.num_pathways = len(self.pathway_groups)
        self.routed_indices = sorted(all_grouped)
        if not self.routed_indices:
            raise ValueError("PathwayRouter requires non-shared routed features")

        inferred_dims = [
            len(self.shared_indices) + len(group) for group in self.pathway_groups
        ]
        if pathway_dims:
            if len(pathway_dims) != self.num_pathways:
                raise ValueError(
                    "pathway_dims must match the number of pathway groups: "
                    f"{len(pathway_dims)} vs {self.num_pathways}"
                )
            self.pathway_dims = [int(dim) for dim in pathway_dims]
        elif pathway_dim is not None:
            self.pathway_dims = [int(pathway_dim)] * self.num_pathways
        else:
            self.pathway_dims = inferred_dims

        if any(dim <= 0 for dim in self.pathway_dims):
            raise ValueError("All pathway_dims must be positive")

        self.learned_router_temperature = max(float(learned_router_temperature), 1e-6)
        self.output_dim = int(sum(self.pathway_dims))

        activation_factory = ActivationFactory()
        self.pathway_projections = nn.ModuleList()
        self.pathway_activations = nn.ModuleList()

        shared_tensor = torch.tensor(self.shared_indices, dtype=torch.long)
        routed_tensor = torch.tensor(self.routed_indices, dtype=torch.long)
        self.register_buffer("_shared_indices", shared_tensor, persistent=False)
        self.register_buffer("_routed_indices", routed_tensor, persistent=False)

        if self.router_mode == "fixed":
            for pathway_idx, (group, out_dim) in enumerate(
                zip(self.pathway_groups, self.pathway_dims, strict=True)
            ):
                indices = torch.tensor(self.shared_indices + group, dtype=torch.long)
                self.register_buffer(
                    f"_pathway_indices_{pathway_idx}", indices, persistent=False
                )
                layer = nn.Linear(indices.numel(), out_dim)
                self._init_projection(layer, prefer_identity=True)
                self.pathway_projections.append(layer)
                self.pathway_activations.append(
                    activation_factory.create(router_activation, output_dim=out_dim)
                )
            self.assignment_logits = None
        else:
            in_dim = len(self.shared_indices) + len(self.routed_indices)
            if in_dim <= 0:
                raise ValueError("Learned routing requires at least one input feature")

            assignment_logits = torch.zeros(
                len(self.routed_indices), self.num_pathways, dtype=torch.float32
            )
            routed_lookup = {
                feature_idx: routed_idx
                for routed_idx, feature_idx in enumerate(self.routed_indices)
            }
            init_scale = float(learned_router_init_scale)
            for pathway_idx, group in enumerate(self.pathway_groups):
                for feature_idx in group:
                    routed_idx = routed_lookup[feature_idx]
                    assignment_logits[routed_idx].fill_(-init_scale)
                    assignment_logits[routed_idx, pathway_idx] = init_scale
            self.assignment_logits = nn.Parameter(assignment_logits)

            for out_dim in self.pathway_dims:
                layer = nn.Linear(in_dim, out_dim)
                self._init_projection(layer, prefer_identity=False)
                self.pathway_projections.append(layer)
                self.pathway_activations.append(
                    activation_factory.create(router_activation, output_dim=out_dim)
                )

    @staticmethod
    def _validate_indices(indices: list[int], name: str) -> None:
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} contains duplicate indices")
        for idx in indices:
            if idx < 0:
                raise ValueError(f"{name} must contain non-negative indices")

    def _validate_bounds(self, indices: list[int], name: str) -> None:
        for idx in indices:
            if idx >= self.input_dim:
                raise ValueError(
                    f"{name} contains index {idx}, but input_dim={self.input_dim}"
                )

    @staticmethod
    def _init_projection(layer: nn.Linear, prefer_identity: bool) -> None:
        nn.init.zeros_(layer.bias)
        if prefer_identity and layer.in_features == layer.out_features:
            with torch.no_grad():
                layer.weight.zero_()
                layer.weight.copy_(torch.eye(layer.out_features, layer.in_features))
            return
        nn.init.xavier_uniform_(layer.weight)

    def _select(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == 0:
            return x.new_zeros(x.size(0), 0)
        return x.index_select(dim=-1, index=indices)

    def _learned_assignments(self) -> torch.Tensor:
        if self.assignment_logits is None:
            raise RuntimeError("Learned assignments requested in fixed router mode")
        return torch.softmax(
            self.assignment_logits / self.learned_router_temperature, dim=-1
        )

    def get_assignment_matrix(self) -> torch.Tensor:
        """Return a feature-by-pathway assignment matrix."""
        full = torch.zeros(
            self.input_dim,
            self.num_pathways,
            device=self._shared_indices.device,
            dtype=torch.float32,
        )
        if self._shared_indices.numel() > 0:
            full.index_fill_(0, self._shared_indices, 1.0)
        if self.router_mode == "fixed":
            for pathway_idx, group in enumerate(self.pathway_groups):
                if not group:
                    continue
                group_tensor = torch.tensor(group, device=full.device, dtype=torch.long)
                full[group_tensor, pathway_idx] = 1.0
            return full

        full[self._routed_indices] = self._learned_assignments()
        return full

    def get_output_slices(self) -> list[slice]:
        """Return output slices for each latent pathway chunk."""
        slices: list[slice] = []
        start = 0
        for pathway_dim in self.pathway_dims:
            stop = start + int(pathway_dim)
            slices.append(slice(start, stop))
            start = stop
        return slices

    def get_output_role_matrix(self) -> torch.Tensor:
        """Return an output-dimension-by-pathway ownership matrix."""
        role = torch.zeros(
            self.output_dim,
            self.num_pathways,
            device=self._shared_indices.device,
            dtype=torch.float32,
        )
        for pathway_idx, output_slice in enumerate(self.get_output_slices()):
            role[output_slice, pathway_idx] = 1.0
        return role

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected final input dimension {self.input_dim}, got {x.size(-1)}"
            )

        batch_shape = x.shape[:-1]
        flat_x = x.reshape(-1, self.input_dim)
        outputs: list[torch.Tensor] = []

        shared_x = self._select(flat_x, self._shared_indices)

        if self.router_mode == "fixed":
            for pathway_idx, (projection, activation) in enumerate(
                zip(self.pathway_projections, self.pathway_activations, strict=True)
            ):
                indices = getattr(self, f"_pathway_indices_{pathway_idx}")
                pathway_x = self._select(flat_x, indices)
                outputs.append(activation(projection(pathway_x)))
        else:
            routed_x = self._select(flat_x, self._routed_indices)
            assignments = self._learned_assignments()
            for pathway_idx, (projection, activation) in enumerate(
                zip(self.pathway_projections, self.pathway_activations, strict=True)
            ):
                pathway_routed = routed_x * assignments[:, pathway_idx].unsqueeze(0)
                pathway_x = torch.cat([shared_x, pathway_routed], dim=-1)
                outputs.append(activation(projection(pathway_x)))

        routed = torch.cat(outputs, dim=-1)
        return routed.reshape(*batch_shape, self.output_dim)


__all__ = ["PathwayRouter"]
