"""Conductance-dynamic gradient scale helpers for dendritic branch layers."""

from __future__ import annotations

from typing import Any

import torch


def _set_topk_param_scale(layer: Any, param_scale, *, use_forward_hooks: bool) -> None:
    """Apply a TopK parameter scale through hooks or direct compatibility state."""
    if use_forward_hooks and hasattr(layer, "register_forward_param_gradient_scale"):
        layer.register_forward_param_gradient_scale(param_scale)
    else:
        layer.param_scale_vec = param_scale


def compute_branch_grad_scales(
    owner: Any,
    g_total: torch.Tensor | None = None,
    *,
    use_forward_hooks: bool = False,
) -> None:
    """Update conductance-dynamic gradient scales for active branch sublayers."""
    with torch.no_grad():
        if owner.use_shunting:
            if g_total.dim() > 1:
                g_total = g_total.mean(dim=tuple(range(g_total.dim() - 1)))

            if (
                hasattr(owner, "branches_to_output")
                and owner.blocklinear_strategy == "conductance_dynamic"
            ):
                weight = owner.branches_to_output.weight().clamp_min(owner.epsilon)
                input_scale = (g_total[:, None] / weight).flatten()
                param_scale = g_total[:, None]
                if use_forward_hooks:
                    owner.branches_to_output.register_forward_gradient_scales(
                        param_scale=param_scale,
                        input_scale=input_scale,
                    )
                else:
                    owner.branches_to_output.input_scale_vec = input_scale
                    owner.branches_to_output.param_scale_vec = param_scale

            if (
                owner.branch_excitation is not None
                and owner.topk_strategy == "conductance_dynamic"
            ):
                _set_topk_param_scale(
                    owner.branch_excitation,
                    g_total[:, None],
                    use_forward_hooks=use_forward_hooks,
                )

            if (
                owner.branch_inhibition is not None
                and owner.topk_strategy == "conductance_dynamic"
            ):
                _set_topk_param_scale(
                    owner.branch_inhibition,
                    g_total[:, None],
                    use_forward_hooks=use_forward_hooks,
                )

            if (
                owner.branch_recurrent is not None
                and owner.topk_strategy == "conductance_dynamic"
            ):
                _set_topk_param_scale(
                    owner.branch_recurrent,
                    g_total[:, None],
                    use_forward_hooks=use_forward_hooks,
                )

            if (
                owner.branch_rec_inhibition is not None
                and owner.topk_strategy == "conductance_dynamic"
            ):
                _set_topk_param_scale(
                    owner.branch_rec_inhibition,
                    g_total[:, None],
                    use_forward_hooks=use_forward_hooks,
                )

        else:
            if (
                hasattr(owner, "branches_to_output")
                and owner.blocklinear_strategy == "conductance_dynamic"
            ):
                w = owner.branches_to_output.weight()

                if (owner.weight_transform or "").lower() == "identity":
                    w = w.abs() + owner.epsilon

                input_scale = torch.exp(-1 * w.log()).flatten()
                if use_forward_hooks:
                    owner.branches_to_output.register_forward_gradient_scales(
                        input_scale=input_scale
                    )
                else:
                    owner.branches_to_output.input_scale_vec = input_scale


__all__ = ["compute_branch_grad_scales"]
