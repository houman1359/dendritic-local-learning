"""Local-rule gradient orchestration for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_epoch import (
    _apply_path_propagation_factor,
    _resolve_stdp_error_signals,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LocalRuleRecordState,
)


class LocalLearningRuleMixin:
    """Batch and record-level orchestration for local-rule gradient writeback."""

    def _apply_local_rule_gradients(
        self,
        model: BaseModel,
        layer_records: list[dict[str, Any]],
        v0: torch.Tensor,
        delta: torch.Tensor,
        y_target: torch.Tensor | None = None,
        update_reactivation: bool = True,
    ) -> None:
        batch_size = v0.size(0)
        delta_scalar: torch.Tensor = self._reduce_error_to_scalar(delta)  # [B, 1]
        broadcast_state = self._prepare_local_rule_broadcast_state(
            model=model,
            layer_records=layer_records,
            delta=delta,
            delta_scalar=delta_scalar,
        )

        for layer_idx, rec in enumerate(layer_records):
            self._apply_local_rule_record_gradients(
                rec=rec,
                layer_idx=layer_idx,
                num_layers=len(layer_records),
                v0=v0,
                delta=delta,
                delta_scalar=delta_scalar,
                batch_size=batch_size,
                broadcast_state=broadcast_state,
                y_target=y_target,
                update_reactivation=update_reactivation,
            )

    def _apply_local_rule_record_gradients(
        self,
        *,
        rec: dict[str, Any],
        layer_idx: int,
        num_layers: int,
        v0: torch.Tensor,
        delta: torch.Tensor,
        delta_scalar: torch.Tensor,
        batch_size: int,
        broadcast_state: Any,
        y_target: torch.Tensor | None,
        update_reactivation: bool,
    ) -> None:
        """Prepare one recorded layer and apply its local-rule gradients."""
        if not self._should_update_local_record(rec):
            # In freeze mode, explicit inhibitory-cell dendrites are held fixed.
            # E-cell I-to-E synapses still learn through excitatory records.
            return

        v_n: torch.Tensor | None = rec.get("v_n")
        if v_n is None:
            return

        record_state = self._prepare_local_rule_record_state(
            rec=rec,
            layer_idx=layer_idx,
            v0=v0,
            delta=delta,
            delta_scalar=delta_scalar,
            v_n=v_n,
            broadcast_state=broadcast_state,
        )
        self._apply_layer_local_gradients(
            rec=rec,
            layer_idx=layer_idx,
            num_layers=num_layers,
            y_target=y_target,
            batch_size=batch_size,
            e_n=record_state.e_n,
            stdp_error_signal=record_state.stdp_error_signal,
            v_n=record_state.v_n,
            r_tot=record_state.r_tot,
            layer_dynamics_mode=record_state.layer_dynamics_mode,
            modulators=record_state.modulators,
            post_factors=record_state.post_factors,
            update_reactivation=update_reactivation,
        )

    def _prepare_local_rule_record_state(
        self,
        *,
        rec: dict[str, Any],
        layer_idx: int,
        v0: torch.Tensor,
        delta: torch.Tensor,
        delta_scalar: torch.Tensor,
        v_n: torch.Tensor,
        broadcast_state: Any,
    ) -> _LocalRuleRecordState:
        """Prepare local-rule factors for one recorded layer."""
        layer_depth = layer_idx + 1
        e_n = self._compute_local_broadcast_error(
            rec=rec,
            layer_idx=layer_idx,
            out_features=v_n.size(1),
            v_n=v_n,
            delta=delta,
            delta_scalar=delta_scalar,
            broadcast_state=broadcast_state,
        )
        e_n, stdp_error_signal = _resolve_stdp_error_signals(
            self.local_cfg,
            e_n,
        )

        layer_dynamics_mode = self._resolve_layer_dynamics_mode(rec)
        r_tot = self._compute_local_input_resistance(
            rec=rec,
            v_n=v_n,
            layer_dynamics_mode=layer_dynamics_mode,
        )
        e_n = _apply_path_propagation_factor(
            self.local_cfg,
            broadcast_state,
            rec,
            e_n,
        )

        modulators = self._compute_local_layer_modulators(
            rec=rec,
            v0=v0,
            layer_depth=layer_depth,
        )
        post_factors = self._compute_local_post_factors(
            rec=rec,
            e_n=e_n,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            layer_idx=layer_idx,
            modulators=modulators,
        )
        return _LocalRuleRecordState(
            v_n=v_n,
            e_n=e_n,
            stdp_error_signal=stdp_error_signal,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            modulators=modulators,
            post_factors=post_factors,
        )


__all__ = ["LocalLearningRuleMixin"]
