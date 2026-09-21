"""Morphology-aware helper methods for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_hsic import (
    compute_dendritic_normalization,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerStats,
)


def _normalized_inverse_depth_factor(
    depths: torch.Tensor,
    offset: float,
) -> torch.Tensor:
    """Return inverse-depth factors normalized to unit mean."""

    length_factor = 1.0 / (depths + offset)
    return length_factor / (length_factor.mean() + 1e-8)


def _role_specialization_scale(
    selectivity: torch.Tensor,
    *,
    power: float,
    specialized_scale: float,
    mixed_scale: float,
) -> torch.Tensor:
    """Interpolate between mixed and specialized branch scales."""

    return mixed_scale + selectivity.pow(power) * (specialized_scale - mixed_scale)


def _centered_alignment_gain(
    alignment: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Convert raw role alignment to nonnegative multiplicative gains."""

    centered_alignment = alignment - alignment.mean(dim=1, keepdim=True)
    return torch.clamp(1.0 + weight * centered_alignment, min=0.0)


def _apply_branch_length_factor(
    scale: torch.Tensor | float,
    length_factor: torch.Tensor,
) -> torch.Tensor:
    """Apply branch-length factors to scalar or per-branch scales."""

    if isinstance(scale, torch.Tensor):
        return scale * length_factor
    return float(scale) * length_factor


def _apply_dendritic_length_factor(
    scale: torch.Tensor,
    length_factor: torch.Tensor,
) -> torch.Tensor:
    """Apply branch-length factors to branch or block-level dendritic scales."""

    if scale.dim() == 2:
        return scale * length_factor.unsqueeze(-1)
    return scale * length_factor


def _default_branch_depths(
    out_features: int,
    layer_idx: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return legacy uniform branch depths for a recorded layer."""

    base_depth = float(layer_idx + 1)
    return torch.ones(out_features, device=device, dtype=dtype) * base_depth


def _default_branch_types(
    out_features: int,
    layer_idx: int,
    n_layers: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return legacy basal/apical heuristic branch types for a recorded layer."""

    branch_types = torch.zeros(out_features, device=device, dtype=dtype)
    if layer_idx >= n_layers // 2:
        branch_types[:] = 1.0
    return branch_types


def _reset_branch_role_metadata(stats: _LayerStats) -> None:
    """Clear inferred role metadata while preserving morphology annotations."""

    stats.branch_role_profile = None
    stats.branch_role_selectivity = None
    stats.input_role_profile = None
    stats.block_input_role_profile = None
    stats.branch_block_roles = None
    stats.branch_block_selectivity = None


def _sum_role_mass_terms(role_mass_terms: list[torch.Tensor]) -> torch.Tensor:
    """Combine role-mass terms with the original left-to-right tensor addition."""

    total_role_mass = role_mass_terms[0]
    for extra_mass in role_mass_terms[1:]:
        total_role_mass = total_role_mass + extra_mass
    return total_role_mass


class LocalLearningMorphologyMixin:
    """Morphology metadata, branch-role, and branch-scale helpers."""

    def _compute_path_propagation_factor(
        self, layer_idx: int, rec: dict[str, Any], layer_records: list[dict[str, Any]]
    ) -> torch.Tensor:
        """Backward-compatible wrapper around the precomputed path factors."""

        path_factors = self._precompute_path_propagation_factors(layer_records)

        path_factor = path_factors[layer_idx]

        if isinstance(path_factor, torch.Tensor):

            return path_factor

        v_n = rec.get("v_n")

        if isinstance(v_n, torch.Tensor):

            return torch.ones(v_n.size(0), 1, device=v_n.device, dtype=v_n.dtype)

        return torch.ones(1, 1)

    def _compute_branch_depth_modulator(
        self, rec: dict[str, Any], rho_base: float
    ) -> torch.Tensor:
        """Compute branch-specific modulator based on depth.



        Modulates rho by branch depth: rho_j = rho_base / (depth_j + offset)



        Args:

            rec: Layer record

            rho_base: Base rho value to modulate



        Returns:

            Per-branch modulator tensor or scalar

        """

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.get(layer_id)

        if stats is None or stats.branch_depths is None:

            # No depth info: return uniform modulator

            return rho_base

        depths = stats.branch_depths  # [out_features]

        offset = self.local_cfg.morphology_aware.morphology_depth_offset

        # Depth-based scaling: deeper branches get smaller factors

        modulator = rho_base / (depths + offset)

        return modulator

    def _compute_branch_centrality_modulator(
        self, rec: dict[str, Any], rho_base: float
    ) -> torch.Tensor:
        """Compute branch-specific modulator based on centrality proxy.



        Without explicit tree metadata, we use depth-derived centrality:

        proximal branches (lower depth) receive higher centrality.

        """

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.get(layer_id)

        if stats is None or stats.branch_depths is None:

            return rho_base

        depths = stats.branch_depths

        offset = self.local_cfg.morphology_aware.morphology_depth_offset

        return rho_base * _normalized_inverse_depth_factor(depths, offset)

    def _compute_dendritic_normalization(
        self, rec: dict[str, Any], grad_g_den: torch.Tensor
    ) -> torch.Tensor:
        """Apply dendritic normalization to branch conductance updates."""

        return compute_dendritic_normalization(rec, grad_g_den)

    def _get_branch_type_scale(self, rec: dict[str, Any]) -> torch.Tensor | float:
        """Get scaling factors for apical vs basal branches.



        Args:

            rec: Layer record



        Returns:

            Scale factor per branch [out_features] or scalar 1.0

        """

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.get(layer_id)

        if stats is None:

            return 1.0

        scale: Any = 1.0

        if stats.branch_types is not None:

            branch_types = stats.branch_types  # [out_features], 1=apical, 0=basal

            apical_scale = self.local_cfg.morphology_aware.apical_branch_scale

            basal_scale = self.local_cfg.morphology_aware.basal_branch_scale

            # Linear interpolation based on branch type

            scale = basal_scale + branch_types * (apical_scale - basal_scale)

        if (
            self.local_cfg.morphology_aware.use_branch_length_modulation
            and stats.branch_depths is not None
        ):

            depth_offset = self.local_cfg.morphology_aware.morphology_depth_offset

            length_factor = _normalized_inverse_depth_factor(
                stats.branch_depths, depth_offset
            )

            scale = _apply_branch_length_factor(scale, length_factor)

        return scale

    def _initialize_branch_morphology(
        self, model: BaseModel, layer_records: list[dict[str, Any]]
    ) -> None:
        """Initialize morphology metadata and infer pathway roles from connectivity."""

        base_input_roles = self._resolve_router_output_roles(model)

        prior_output_roles: list[torch.Tensor] = []

        for layer_idx, rec in enumerate(layer_records):

            layer_id = id(rec.get("layer"))

            stats = self._layer_stats.setdefault(layer_id, _LayerStats())

            v_n = rec.get("v_n")

            if v_n is None:

                continue

            out_features = v_n.size(1)

            _reset_branch_role_metadata(stats)

            # Heuristic depth: based on layer index (deeper layers = higher depth)

            # In future, this should come from actual dendritic tree graph

            if stats.branch_depths is None:

                # Assign uniform depth based on layer position

                stats.branch_depths = _default_branch_depths(
                    out_features,
                    layer_idx,
                    device=v_n.device,
                    dtype=v_n.dtype,
                )

            # Heuristic branch types: assume first half are basal, second half apical

            # In future, this should come from actual branch annotations

            if stats.branch_types is None:

                stats.branch_types = _default_branch_types(
                    out_features,
                    layer_idx,
                    len(layer_records),
                    device=v_n.device,
                    dtype=v_n.dtype,
                )

            role_mass_terms: list[torch.Tensor] = []

            exc_layer: TopKLinear | None = rec.get("exc_module")

            x_exc = rec.get("x_exc")

            if exc_layer is not None and isinstance(x_exc, torch.Tensor):

                exc_input_roles = self._resolve_recent_role_profile(
                    x_exc.size(1), base_input_roles, prior_output_roles
                )

                if exc_input_roles is not None:

                    stats.input_role_profile = exc_input_roles.to(
                        device=v_n.device, dtype=v_n.dtype
                    )

                    exc_weight = self._effective_topk_weight(
                        exc_layer, rec.get("exc_mask")
                    )

                    exc_role_mass = self._role_mass_from_weight_matrix(
                        exc_weight, stats.input_role_profile
                    )

                    if exc_role_mass is not None:

                        role_mass_terms.append(
                            exc_role_mass.to(device=v_n.device, dtype=v_n.dtype)
                        )

            inh_layer: TopKLinear | None = rec.get("inh_module")

            x_inh = rec.get("x_inh")

            if inh_layer is not None and isinstance(x_inh, torch.Tensor):

                inh_input_roles = self._resolve_recent_role_profile(
                    x_inh.size(1), base_input_roles, prior_output_roles
                )

                if inh_input_roles is not None:

                    if stats.input_role_profile is None:

                        stats.input_role_profile = inh_input_roles.to(
                            device=v_n.device, dtype=v_n.dtype
                        )

                    inh_weight = self._effective_topk_weight(
                        inh_layer, rec.get("inh_mask")
                    )

                    inh_role_mass = self._role_mass_from_weight_matrix(
                        inh_weight, inh_input_roles
                    )

                    if inh_role_mass is not None:

                        role_mass_terms.append(
                            inh_role_mass.to(device=v_n.device, dtype=v_n.dtype)
                        )

            blk_layer = rec.get("blk_module")

            x_blk_raw = rec.get("x_blk_raw")

            if blk_layer is not None and isinstance(x_blk_raw, torch.Tensor):

                block_input_roles = self._resolve_recent_role_profile(
                    x_blk_raw.size(1), base_input_roles, prior_output_roles
                )

                if block_input_roles is not None:

                    stats.block_input_role_profile = block_input_roles.to(
                        device=v_n.device, dtype=v_n.dtype
                    )

                    block_size = int(getattr(blk_layer, "block_size", 1))

                    n_roles = stats.block_input_role_profile.size(1)

                    expected_in = out_features * block_size

                    if stats.block_input_role_profile.size(0) == expected_in:

                        block_roles = stats.block_input_role_profile.reshape(
                            out_features, block_size, n_roles
                        )

                        stats.branch_block_roles = block_roles

                        block_selectivity = self._compute_role_selectivity(
                            block_roles.reshape(-1, n_roles)
                        )

                        stats.branch_block_selectivity = block_selectivity.reshape(
                            out_features, block_size
                        )

                        blk_weight = (
                            blk_layer.weight()
                            .detach()
                            .to(device=v_n.device, dtype=v_n.dtype)
                        )

                        block_role_mass = (
                            blk_weight.unsqueeze(-1)
                            * block_roles.to(device=v_n.device, dtype=v_n.dtype)
                        ).sum(dim=1)

                        role_mass_terms.append(block_role_mass)

            if role_mass_terms:

                total_role_mass = _sum_role_mass_terms(role_mass_terms)

                role_profile = self._normalize_role_mass(total_role_mass)

                stats.branch_role_profile = role_profile

                stats.branch_role_selectivity = self._compute_role_selectivity(
                    role_profile
                )

                prior_output_roles.append(role_profile.detach())

    def _get_branch_role_scales(
        self, rec: dict[str, Any]
    ) -> tuple[torch.Tensor | float, torch.Tensor | float]:
        """Get plasticity scales from inferred pathway specialization."""

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.get(layer_id)

        if (
            stats is None
            or not isinstance(stats.branch_role_selectivity, torch.Tensor)
            or stats.branch_role_selectivity.numel() == 0
        ):

            return 1.0, 1.0

        cfg = self.local_cfg.morphology_aware

        power = max(float(getattr(cfg, "branch_role_power", 1.0)), 1e-6)

        specialized_scale = float(getattr(cfg, "specialized_branch_scale", 1.0))

        mixed_scale = float(getattr(cfg, "mixed_branch_scale", 1.0))

        neuron_selectivity = stats.branch_role_selectivity.to(
            dtype=stats.branch_role_selectivity.dtype
        )

        branch_scale = _role_specialization_scale(
            neuron_selectivity,
            power=power,
            specialized_scale=specialized_scale,
            mixed_scale=mixed_scale,
        )

        dendritic_scale: torch.Tensor | float = branch_scale

        if isinstance(stats.branch_block_selectivity, torch.Tensor):

            block_selectivity = stats.branch_block_selectivity.to(
                dtype=branch_scale.dtype, device=branch_scale.device
            )

            dendritic_scale = _role_specialization_scale(
                block_selectivity,
                power=power,
                specialized_scale=specialized_scale,
                mixed_scale=mixed_scale,
            )

        if cfg.use_branch_length_modulation and stats.branch_depths is not None:

            depth_offset = cfg.morphology_depth_offset

            length_factor = _normalized_inverse_depth_factor(
                stats.branch_depths, depth_offset
            )

            branch_scale = branch_scale * length_factor

            if isinstance(dendritic_scale, torch.Tensor):

                dendritic_scale = _apply_dendritic_length_factor(
                    dendritic_scale, length_factor
                )

        return branch_scale, dendritic_scale

    def _get_branch_role_alignment_factors(
        self, rec: dict[str, Any]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get synapse- and branch-specific alignment gains from inferred roles."""

        layer_id = id(rec.get("layer"))

        stats = self._layer_stats.get(layer_id)

        if (
            stats is None
            or not isinstance(stats.branch_role_profile, torch.Tensor)
            or stats.branch_role_profile.numel() == 0
        ):

            return None, None

        weight = float(
            getattr(
                self.local_cfg.morphology_aware, "branch_role_alignment_weight", 0.0
            )
        )

        if weight <= 0.0:

            return None, None

        role_profile = stats.branch_role_profile

        syn_factor: torch.Tensor | None = None

        if isinstance(stats.input_role_profile, torch.Tensor):

            alignment = role_profile @ stats.input_role_profile.t().to(
                device=role_profile.device, dtype=role_profile.dtype
            )

            syn_factor = _centered_alignment_gain(alignment, weight)

        block_factor: torch.Tensor | None = None

        if isinstance(stats.branch_block_roles, torch.Tensor):

            block_roles = stats.branch_block_roles.to(
                device=role_profile.device, dtype=role_profile.dtype
            )

            alignment = (block_roles * role_profile.unsqueeze(1)).sum(dim=-1)

            block_factor = _centered_alignment_gain(alignment, weight)

        return syn_factor, block_factor


__all__ = ["LocalLearningMorphologyMixin"]
