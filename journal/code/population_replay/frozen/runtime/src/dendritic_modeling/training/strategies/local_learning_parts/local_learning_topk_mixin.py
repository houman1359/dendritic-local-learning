"""TopK synapse-gradient helpers for local credit assignment."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_gradient_utils import (
    _apply_optional_alignment,
    _factor_synapse_weight_grad,
    _topk_pre_weight_grad,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_pathways import (
    EXCITATORY_TOPK_PATHS,
    INHIBITORY_TOPK_PATHS,
    TopKGradientPath,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerModulators,
    _PostFactors,
)

T = TypeVar("T")


def _record_structural_credit(layer: Any, gradient: torch.Tensor) -> None:
    """Offer the LocalCA conductance gradient to an optional structural rule."""
    record = getattr(layer, "record_local_credit", None)
    if callable(record):
        record(gradient.detach())


def _apply_output_modulator(
    factor: torch.Tensor,
    modulator: torch.Tensor | float,
) -> torch.Tensor:
    """Apply a neuron-wise or scalar output modulator."""
    if isinstance(modulator, torch.Tensor):
        return factor * modulator.unsqueeze(0)
    return factor * float(modulator)


def _apply_topk_rho_modulator(
    factor: torch.Tensor,
    rho: torch.Tensor | float,
    *,
    rule_variant: str,
) -> torch.Tensor:
    """Apply the historical rho rule for inhibitory TopK factors."""
    if isinstance(rho, torch.Tensor):
        return factor * rho.unsqueeze(0)
    if rule_variant in {"4f", "5f"}:
        return factor * float(rho)
    return factor


def _apply_topk_phi_modulator(
    factor: torch.Tensor,
    phi: float,
    *,
    rule_variant: str,
) -> torch.Tensor:
    """Apply the historical phi rule for inhibitory TopK factors."""
    if rule_variant == "5f":
        return factor * float(phi)
    return factor


def _apply_inhibitory_topk_modulators(
    factor: torch.Tensor,
    *,
    rho: torch.Tensor | float,
    phi: float,
    branch_scale: torch.Tensor | float,
    rule_variant: str,
) -> torch.Tensor:
    """Apply inhibitory TopK modulators in the legacy order."""
    factor = _apply_topk_rho_modulator(factor, rho, rule_variant=rule_variant)
    factor = _apply_topk_phi_modulator(factor, phi, rule_variant=rule_variant)
    return _apply_output_modulator(factor, branch_scale)


def _conductance_inhibitory_topk_factor(
    *,
    e_v: torch.Tensor,
    v_n: torch.Tensor,
    r_tot: torch.Tensor | float,
    e_rev_inh: float,
    theta: float,
    use_driving_force: bool,
) -> torch.Tensor:
    """Compute the inhibitory TopK factor for conductance dynamics."""
    if use_driving_force:
        inh_post = e_rev_inh - v_n
    else:
        inh_post = theta - v_n
    return e_v * r_tot * inh_post


def _additive_inhibitory_topk_factor(
    *,
    e_v: torch.Tensor,
    v_n: torch.Tensor,
    post_factor_raw: torch.Tensor,
    additive_mode: str,
    theta: float,
    pseudo_R: torch.Tensor | None,
) -> torch.Tensor:
    """Compute the inhibitory TopK factor for additive dynamics."""
    if additive_mode == "input_dependent":
        if pseudo_R is None:
            raise ValueError("pseudo_R is required for input_dependent additive mode.")
        inh_post = theta - v_n
        return e_v * pseudo_R * inh_post
    if additive_mode in ("learned_gain", "running_stats"):
        return -post_factor_raw
    return -e_v


def _dispatch_topk_gradient_paths(
    paths: Iterable[TopKGradientPath],
    apply_path: Callable[[TopKGradientPath], T | None],
) -> T | None:
    """Apply a TopK local-gradient callback to each pathway in order."""
    first_layer: T | None = None
    for path_idx, path in enumerate(paths):
        layer = apply_path(path)
        if path_idx == 0:
            first_layer = layer
    return first_layer


class LocalLearningTopKGradientMixin:
    """Gradient writeback for TopK feedforward and recurrent pathways."""

    def _apply_excitatory_topk_pathways(
        self,
        *,
        rec: dict[str, Any],
        v_n: torch.Tensor,
        r_tot: torch.Tensor | float,
        layer_dynamics_mode: str,
        batch_size: int,
        modulators: _LayerModulators,
        post_factors: _PostFactors,
    ) -> TopKLinear | None:
        """Apply local gradients to feedforward and recurrent excitatory paths."""

        def apply_path(path: TopKGradientPath) -> TopKLinear | None:
            return self._apply_excitatory_topk_local_grad(
                rec=rec,
                module_key=path.module_key,
                input_key=path.input_key,
                mask_key=path.mask_key,
                post_factor=post_factors.post_factor,
                v_n=v_n,
                r_tot=r_tot,
                layer_dynamics_mode=layer_dynamics_mode,
                batch_size=batch_size,
                role_synaptic_alignment=modulators.role_synaptic_alignment,
            )

        return _dispatch_topk_gradient_paths(EXCITATORY_TOPK_PATHS, apply_path)

    def _apply_inhibitory_topk_pathways(
        self,
        *,
        rec: dict[str, Any],
        v_n: torch.Tensor,
        r_tot: torch.Tensor | float,
        layer_dynamics_mode: str,
        batch_size: int,
        modulators: _LayerModulators,
        post_factors: _PostFactors,
    ) -> None:
        """Apply local gradients to feedforward and recurrent inhibitory paths."""

        def apply_path(path: TopKGradientPath) -> TopKLinear | None:
            return self._apply_inhibitory_topk_local_grad(
                rec=rec,
                module_key=path.module_key,
                input_key=path.input_key,
                mask_key=path.mask_key,
                e_v=post_factors.e_v,
                post_factor_raw=post_factors.post_factor_raw,
                v_n=v_n,
                r_tot=r_tot,
                rho=modulators.rho,
                phi=modulators.phi,
                branch_scale=modulators.branch_scale,
                layer_dynamics_mode=layer_dynamics_mode,
                batch_size=batch_size,
                role_synaptic_alignment=modulators.role_synaptic_alignment,
            )

        _dispatch_topk_gradient_paths(INHIBITORY_TOPK_PATHS, apply_path)

    def _apply_excitatory_topk_local_grad(
        self,
        *,
        rec: dict[str, Any],
        module_key: str,
        input_key: str,
        mask_key: str,
        post_factor: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        layer_dynamics_mode: str,
        batch_size: int,
        role_synaptic_alignment: torch.Tensor | None,
    ) -> TopKLinear | None:
        layer: TopKLinear | None = rec.get(module_key)
        if layer is None or rec.get(input_key) is None:
            return layer

        x_pre: torch.Tensor = rec[input_key]
        grad_g = _factor_synapse_weight_grad(
            layer,
            post_factor,
            x_pre,
            normalize_by_batch=self.local_cfg.normalize_by_batch,
            batch_size=batch_size,
        )
        _record_structural_credit(
            layer,
            _apply_optional_alignment(grad_g, role_synaptic_alignment),
        )

        grad_homeo = self._compute_topk_voltage_homeostasis_grad(
            layer=layer,
            v_n=v_n,
            x_pre=x_pre,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            batch_size=batch_size,
            conductance_reversal=self.local_cfg.three_factor.e_rev_exc,
            additive_sign=1.0,
        )
        if grad_homeo is not None:
            grad_g = grad_g + grad_homeo

        grad_g = _apply_optional_alignment(grad_g, role_synaptic_alignment)

        self._add_topk_transformed_grad(layer, grad_g, rec.get(mask_key))
        return layer

    def _apply_inhibitory_topk_local_grad(
        self,
        *,
        rec: dict[str, Any],
        module_key: str,
        input_key: str,
        mask_key: str,
        e_v: torch.Tensor,
        post_factor_raw: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        rho: Any,
        phi: float,
        branch_scale: Any,
        layer_dynamics_mode: str,
        batch_size: int,
        role_synaptic_alignment: torch.Tensor | None,
    ) -> TopKLinear | None:
        layer: TopKLinear | None = rec.get(module_key)
        if layer is None or rec.get(input_key) is None:
            return layer

        x_pre: torch.Tensor = rec[input_key]
        inh_factor = self._compute_inhibitory_topk_factor(
            rec=rec,
            e_v=e_v,
            post_factor_raw=post_factor_raw,
            v_n=v_n,
            r_tot=r_tot,
            rho=rho,
            phi=phi,
            branch_scale=branch_scale,
            layer_dynamics_mode=layer_dynamics_mode,
        )

        grad_g = _factor_synapse_weight_grad(
            layer,
            inh_factor,
            x_pre,
            normalize_by_batch=self.local_cfg.normalize_by_batch,
            batch_size=batch_size,
        )

        grad_g = _apply_optional_alignment(grad_g, role_synaptic_alignment)
        _record_structural_credit(layer, grad_g)

        grad_homeo = self._compute_topk_voltage_homeostasis_grad(
            layer=layer,
            v_n=v_n,
            x_pre=x_pre,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            batch_size=batch_size,
            conductance_reversal=self.local_cfg.three_factor.e_rev_inh,
            additive_sign=-1.0,
        )
        if grad_homeo is not None:
            grad_g = grad_g + grad_homeo

        inhibitory_homeo_factor = self._compute_inhibitory_homeostasis_factor(
            rec=rec, R_tot=r_tot, v_n=v_n
        )
        if inhibitory_homeo_factor is not None:
            grad_homeo = _factor_synapse_weight_grad(
                layer,
                inhibitory_homeo_factor,
                x_pre,
                normalize_by_batch=self.local_cfg.normalize_by_batch,
                batch_size=batch_size,
            )
            grad_g = grad_g + grad_homeo

        self._add_topk_transformed_grad(layer, grad_g, rec.get(mask_key))
        return layer

    def _compute_topk_voltage_homeostasis_grad(
        self,
        *,
        v_n: torch.Tensor,
        x_pre: torch.Tensor,
        r_tot: Any,
        layer_dynamics_mode: str,
        batch_size: int,
        conductance_reversal: float,
        additive_sign: float,
        layer: TopKLinear | None = None,
    ) -> torch.Tensor | None:
        voltage_homeo_error = self._compute_voltage_homeostasis_error(v_n)
        if voltage_homeo_error is None:
            return None

        if layer_dynamics_mode == "conductance":
            homeo_factor = voltage_homeo_error * r_tot * (conductance_reversal - v_n)
        else:
            homeo_factor = voltage_homeo_error * float(additive_sign)

        return _factor_synapse_weight_grad(
            layer,
            homeo_factor,
            x_pre,
            normalize_by_batch=self.local_cfg.normalize_by_batch,
            batch_size=batch_size,
        )

    def _compute_inhibitory_topk_factor(
        self,
        *,
        rec: dict[str, Any],
        e_v: torch.Tensor,
        post_factor_raw: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        rho: Any,
        phi: float,
        branch_scale: Any,
        layer_dynamics_mode: str,
    ) -> torch.Tensor:
        inh_factor = self._compute_base_inhibitory_topk_factor(
            rec=rec,
            e_v=e_v,
            post_factor_raw=post_factor_raw,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
        )
        return _apply_inhibitory_topk_modulators(
            inh_factor,
            rho=rho,
            phi=phi,
            branch_scale=branch_scale,
            rule_variant=self.local_cfg.rule_variant,
        )

    def _compute_base_inhibitory_topk_factor(
        self,
        *,
        rec: dict[str, Any],
        e_v: torch.Tensor,
        post_factor_raw: torch.Tensor,
        v_n: torch.Tensor,
        r_tot: Any,
        layer_dynamics_mode: str,
    ) -> torch.Tensor:
        """Compute the inhibitory TopK factor before rho/phi/branch modulation."""
        if layer_dynamics_mode == "conductance":
            return _conductance_inhibitory_topk_factor(
                e_v=e_v,
                v_n=v_n,
                r_tot=r_tot,
                e_rev_inh=self.local_cfg.three_factor.e_rev_inh,
                theta=self.local_cfg.three_factor.theta,
                use_driving_force=self.local_cfg.three_factor.use_driving_force,
            )

        additive_mode = getattr(
            self.local_cfg.three_factor, "additive_gain_mode", "none"
        )
        pseudo_R = None
        if additive_mode == "input_dependent":
            pseudo_R, _ = self._compute_additive_pseudo_signals(rec, v_n)
        return _additive_inhibitory_topk_factor(
            e_v=e_v,
            v_n=v_n,
            post_factor_raw=post_factor_raw,
            additive_mode=additive_mode,
            theta=self.local_cfg.three_factor.theta,
            pseudo_R=pseudo_R,
        )

    def _add_topk_transformed_grad(
        self,
        layer: TopKLinear,
        local_grad: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        grad_param = _topk_pre_weight_grad(
            local_grad,
            pre_weight=layer.pre_w,
            weight_transform=getattr(layer, "weight_transform", "exp"),
            derivative_fn=self._weight_transform_derivative,
        )
        self._add_topk_grad(layer, grad_param, mask)


__all__ = ["LocalLearningTopKGradientMixin"]
