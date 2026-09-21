"""STDP gradient helpers for local credit assignment."""

from __future__ import annotations

import logging
from typing import Any

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_pathways import (
    STDP_TOPK_PATHS,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_stdp import (
    stdp_activity,
    stdp_apply_error_modulation,
    stdp_apply_pathway_sign,
    stdp_clamp_update,
    stdp_decay,
    stdp_enabled,
    stdp_next_trace,
    stdp_pathway_enabled,
    stdp_trace_like,
    stdp_weight_delta,
    weight_transform_derivative,
)

logger = logging.getLogger(__name__)


class LocalLearningSTDPMixin:
    @staticmethod
    def _weight_transform_derivative(
        raw_param: torch.Tensor, transform_type: str
    ) -> torch.Tensor:
        """Derivative of transformed conductance w.r.t. raw parameter."""
        return weight_transform_derivative(
            raw_param,
            transform_type,
            warn_unknown=lambda unknown: logger.warning(
                "Unknown weight transform '%s', falling back to exp derivative.",
                unknown,
            ),
        )

    def _stdp_enabled(self) -> bool:
        """Return whether trace-based STDP updates should be applied."""

        return stdp_enabled(self.local_cfg)

    def _stdp_pathway_enabled(self, pathway: str) -> bool:
        """Return whether STDP is configured for a recorded synaptic pathway."""

        return stdp_pathway_enabled(self.local_cfg, pathway)

    def _stdp_activity(
        self, values: torch.Tensor, threshold: float, mode: str
    ) -> torch.Tensor:
        """Convert raw local values to non-backpropagating STDP activities."""

        return stdp_activity(values, threshold, mode)

    @staticmethod
    def _stdp_decay(tau: float) -> float:
        """Single-step exponential decay for an STDP eligibility trace."""

        return stdp_decay(tau)

    def _add_topk_grad(
        self,
        layer: TopKLinear,
        grad_param: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        """Accumulate a raw-parameter gradient while respecting connectivity masks."""

        if hasattr(layer, "_connection_mask_like"):

            grad_param = grad_param * layer._connection_mask_like(grad_param)

        if mask is not None and not self.local_cfg.update_inactive_weights:

            grad_param = grad_param * mask.to(
                device=grad_param.device, dtype=grad_param.dtype
            )

        if layer.pre_w.grad is None:

            layer.pre_w.grad = grad_param.to(layer.pre_w.dtype)

        else:

            layer.pre_w.grad = layer.pre_w.grad + grad_param.to(layer.pre_w.dtype)

    def _apply_stdp_gradient(
        self,
        layer: TopKLinear,
        pre: torch.Tensor,
        post: torch.Tensor,
        mask: torch.Tensor | None,
        pathway: str,
        error_signal: torch.Tensor | None = None,
    ) -> None:
        """Apply one trace-based STDP gradient to a TopK synapse module.

        Positive STDP potentiation is represented as a negative optimizer
        gradient because PyTorch optimizers perform ``param -= lr * grad``.
        """
        if not self._stdp_pathway_enabled(pathway):
            return

        stdp_cfg = self.local_cfg.stdp
        if not hasattr(self, "_stdp_traces"):
            self._stdp_traces = {}

        pre_activity = self._stdp_activity(
            pre,
            threshold=float(getattr(stdp_cfg, "pre_threshold", 0.0)),
            mode=str(getattr(stdp_cfg, "activity_mode", "relu")),
        )
        post_activity = self._stdp_activity(
            post,
            threshold=float(getattr(stdp_cfg, "post_threshold", 0.0)),
            mode=str(getattr(stdp_cfg, "activity_mode", "relu")),
        )

        if (
            pre_activity.size(1) != layer.in_features
            or post_activity.size(1) != layer.out_features
        ):
            logger.debug(
                "Skipping STDP for %s: activity shape %s -> %s does not match %s -> %s.",
                pathway,
                tuple(pre_activity.shape),
                tuple(post_activity.shape),
                layer.in_features,
                layer.out_features,
            )
            return

        trace_state = self._stdp_traces.setdefault(id(layer), {})
        prev_pre = stdp_trace_like(
            trace_state.get("pre"),
            features=layer.in_features,
            activity=pre_activity,
        )
        prev_post = stdp_trace_like(
            trace_state.get("post"),
            features=layer.out_features,
            activity=post_activity,
        )

        pre_batch = pre_activity.mean(dim=0)
        post_batch = post_activity.mean(dim=0)

        delta_w = stdp_weight_delta(
            pre_batch=pre_batch,
            post_batch=post_batch,
            prev_pre=prev_pre,
            prev_post=prev_post,
            a_plus=float(getattr(stdp_cfg, "a_plus", 1.0)),
            a_minus=float(getattr(stdp_cfg, "a_minus", 0.5)),
        )
        delta_w = stdp_apply_pathway_sign(
            delta_w,
            pathway=pathway,
            inhibitory_update_sign=float(
                getattr(stdp_cfg, "inhibitory_update_sign", -1.0)
            ),
        )

        if bool(getattr(stdp_cfg, "use_error_modulation", False)):
            delta_w = stdp_apply_error_modulation(
                delta_w,
                error_signal=error_signal,
                mode=str(
                    getattr(stdp_cfg, "error_modulation_mode", "scalar_abs")
                ).lower(),
            )
        delta_w = stdp_clamp_update(
            delta_w,
            float(getattr(stdp_cfg, "clamp_update", 0.0)),
        )

        chain = self._weight_transform_derivative(
            layer.pre_w.detach(),
            getattr(layer, "weight_transform", "exp"),
        )
        grad_param = -float(getattr(stdp_cfg, "learning_rate_scale", 1.0)) * delta_w
        grad_param = grad_param * chain
        self._add_topk_grad(layer, grad_param, mask)

        detach_traces = bool(getattr(stdp_cfg, "detach_traces", True))
        trace_state["pre"] = stdp_next_trace(
            prev_pre,
            pre_batch,
            tau=float(getattr(stdp_cfg, "tau_pre", 20.0)),
            detach=detach_traces,
        )
        trace_state["post"] = stdp_next_trace(
            prev_post,
            post_batch,
            tau=float(getattr(stdp_cfg, "tau_post", 20.0)),
            detach=detach_traces,
        )

    def _apply_stdp_pathways(
        self,
        rec: dict[str, Any],
        post_signal: torch.Tensor,
        error_signal: torch.Tensor | None = None,
    ) -> None:
        """Apply configured STDP updates to all recorded TopK pathways."""

        for path in STDP_TOPK_PATHS:

            layer = rec.get(path.module_key)

            pre = rec.get(path.input_key)

            if isinstance(layer, TopKLinear) and isinstance(pre, torch.Tensor):

                self._apply_stdp_gradient(
                    layer=layer,
                    pre=pre,
                    post=post_signal,
                    mask=rec.get(path.mask_key),
                    pathway=path.pathway,
                    error_signal=error_signal,
                )


__all__ = ["LocalLearningSTDPMixin"]
