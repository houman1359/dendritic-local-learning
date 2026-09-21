"""Signal construction helpers for local credit assignment."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_activation_derivatives import (
    _parametric_tanh_derivative as _parametric_tanh_derivative,
    compute_reactivation_derivative,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_hsic import (
    compute_additive_pseudo_signals,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerStats,
)
from dendritic_modeling.utils.hooks import iter_modules_of_type

logger = logging.getLogger(__name__)


def _is_class_index_target(y: torch.Tensor) -> bool:
    """Return whether a target tensor stores class indices."""

    return y.dim() == 1 or (y.dim() == 2 and y.size(1) == 1)


def _prepare_ce_target(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Prepare cross-entropy targets with legacy one-hot handling."""

    if _is_class_index_target(y):
        num_classes = y_hat.size(-1)
        return functional.one_hot(y.view(-1), num_classes=num_classes).to(y_hat.dtype)
    return y


def _prepare_bce_prediction(y_hat: torch.Tensor) -> torch.Tensor:
    """Prepare BCE predictions from logits or probabilities."""

    y_hat_detached = y_hat.detach()
    if y_hat_detached.min() < 0 or y_hat_detached.max() > 1:
        return torch.sigmoid(y_hat)
    return torch.clamp(y_hat, min=1e-6, max=1.0 - 1e-6)


def _error_mode_from_loss_name(loss_name: str) -> str | None:
    """Infer soma-error mode from a normalized loss name."""

    if (
        "cross entropy" in loss_name
        or "negative log likelihood" in loss_name
        or "nll" in loss_name
        or "categorical" in loss_name
    ):
        return "ce"

    if "binary cross entropy" in loss_name:
        return "bce"

    return None


def _looks_like_class_index_logits(y_hat: torch.Tensor, y: torch.Tensor) -> bool:
    """Return whether tensors match the legacy class-index/logits heuristic."""

    return (
        y_hat.dim() >= 2
        and _is_class_index_target(y)
        and y.dtype in (torch.long, torch.int64)
    )


def _resolve_soma_error_mode(
    mode: str,
    loss_name: str,
    y_hat: torch.Tensor,
    y: torch.Tensor,
) -> str:
    """Resolve the local soma-error mode without computing the tensor error."""

    mode = mode.lower()
    if mode in {"mse", "bce", "ce"}:
        return mode

    loss_mode = _error_mode_from_loss_name(loss_name)
    if loss_mode is not None:
        return loss_mode

    if _looks_like_class_index_logits(y_hat, y):
        return "ce"

    return "mse"


def _dynamics_mode_from_config(cfg_mode: str) -> str | None:
    """Return an explicit dynamics mode from config, if one was requested."""

    if cfg_mode in {"conductance", "legacy"}:
        return "conductance"

    if cfg_mode == "additive":
        return "additive"

    return None


def _dynamics_mode_from_layer(layer: Any) -> str:
    """Infer dynamics mode from layer shunting metadata."""

    uses_shunting = bool(getattr(layer, "use_shunting", True))
    return "conductance" if uses_shunting else "additive"


def _decoder_linear_layers(decoder_network: nn.Module) -> list[nn.Linear]:
    """Return decoder linear layers, preserving legacy traversal failure handling."""

    try:
        return list(iter_modules_of_type(decoder_network, nn.Linear))
    except Exception:
        return []


def _has_single_linear_decoder(decoder_network: nn.Module) -> bool:
    """Return whether the decoder is a single linear readout."""

    return len(_decoder_linear_layers(decoder_network)) == 1


def _linear_decoder_shapes_compatible(
    *,
    delta_out: torch.Tensor,
    h_in: torch.Tensor,
    lin_mod: nn.Linear,
) -> bool:
    """Return whether cached tensors match a single linear decoder mapping."""

    return (
        delta_out.dim() == 2
        and h_in.dim() == 2
        and delta_out.size(0) == h_in.size(0)
        and delta_out.size(1) == lin_mod.weight.size(0)
        and h_in.size(1) == lin_mod.weight.size(1)
    )


def _map_decoder_input_error(
    *,
    y_hat: torch.Tensor,
    decoder_input: Any,
    delta_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Map output error through a cached decoder input when autograd can resolve it."""

    if not isinstance(decoder_input, torch.Tensor):
        return None

    try:
        delta_core = torch.autograd.grad(
            outputs=y_hat,
            inputs=decoder_input,
            grad_outputs=delta_out,
            retain_graph=True,
            allow_unused=True,
        )[0]
    except RuntimeError:
        delta_core = None

    if isinstance(delta_core, torch.Tensor):
        return decoder_input.detach(), delta_core.detach()
    return None


class LocalLearningSignalMixin:
    def _compute_logging_loss(
        self,
        model: BaseModel,
        x: torch.Tensor,
        y: torch.Tensor,
        predictions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the configured scalar loss, reusing predictions when supported."""

        if predictions is not None:
            try:
                return self.loss_function.from_predictions(predictions, y)
            except NotImplementedError:
                pass
        return self.loss_function(model, x, y)

    def _warn_decoder_soma_fallback_once(self, message: str) -> None:
        """Log a decoder soma-mapping fallback once per trainer instance."""

        if self._warned_decoder_soma_fallback:
            return
        logger.info(message)
        self._warned_decoder_soma_fallback = True

    def _compute_soma_error(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute output-space error signal: dL/dy_hat per sample.



        Supports common losses. Falls back to MSE-like residual if unknown.

        Returns tensor of shape [batch, output_dim].

        """

        loss_name = getattr(self.loss_function, "_loss_name", "").lower()
        mode = _resolve_soma_error_mode(
            self.local_cfg.error_mode,
            loss_name,
            y_hat,
            y,
        )
        return self._error_by_mode(y_hat, y, mode)

    def _resolve_local_soma_signals(
        self,
        model: BaseModel,
        y_hat: torch.Tensor,
        delta_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve soma-space signals for local rules.



        Local dendritic rules ideally use error at the core soma output (pre-decoder).

        When the decoder input is cached with graph information intact, we can map

        output-space error back to core space through the exact decoder Jacobian:



            delta_core = J_decoder(h_core)^T delta_out



        This keeps per-soma and path-transport modes meaningful even when the

        decoder is nonlinear. If that mapping is unavailable, we fall back to the

        legacy output-space behavior for backward compatibility.

        With ``soma_error_source: dfa`` the decoder Jacobian is replaced by a

        fixed random feedback matrix (the SomaDFA construction), so the

        between-neuron production of the per-soma error can be varied

        independently of the within-arbor broadcast mode.

        """

        local_cfg = getattr(self, "local_cfg", None)
        if str(getattr(local_cfg, "soma_error_source", "decoder")).lower() == "dfa":
            return self._resolve_dfa_soma_signals(delta_out=delta_out)

        # Backward-compatible fallback

        v0_local = y_hat.detach()

        delta_local = delta_out.detach()

        component_model = self._unwrap_model(model)
        if not hasattr(component_model, "decoder_network"):

            return v0_local, delta_local

        decoder_input = self._decoder_cache.get("decoder_input")

        mapped_decoder_input = _map_decoder_input_error(
            y_hat=y_hat,
            decoder_input=decoder_input,
            delta_out=delta_out,
        )
        if mapped_decoder_input is not None:
            return mapped_decoder_input

        # We only apply explicit mapping when decoder is a single linear readout.

        if not _has_single_linear_decoder(component_model.decoder_network):

            self._warn_decoder_soma_fallback_once(
                "Local soma-error mapping fallback: decoder is not a single linear layer."
            )

            return v0_local, delta_local

        lin_mod = self._decoder_cache.get("module")

        h_in = self._decoder_cache.get("input")

        if not (isinstance(lin_mod, nn.Linear) and isinstance(h_in, torch.Tensor)):

            self._warn_decoder_soma_fallback_once(
                "Local soma-error mapping fallback: missing cached decoder input."
            )

            return v0_local, delta_local

        # Ensure dimensions are aligned before mapping.

        if not _linear_decoder_shapes_compatible(
            delta_out=delta_out,
            h_in=h_in,
            lin_mod=lin_mod,
        ):

            self._warn_decoder_soma_fallback_once(
                "Local soma-error mapping fallback: incompatible decoder shapes."
            )

            return v0_local, delta_local

        delta_core = delta_out @ lin_mod.weight.detach()

        return h_in.detach(), delta_core.detach()

    def _dfa_feedback_for(
        self,
        *,
        soma_dim: int,
        output_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return the cached fixed random feedback matrix, building it once.

        Shape, seeding and scaling follow ``SomaDFATrainer._feedback_for`` so a
        matched seed yields the same between-neuron feedback column across
        within-neuron broadcast modes.
        """

        expected_shape = (soma_dim, output_dim)
        cached = getattr(self, "_dfa_feedback_matrix", None)
        if cached is not None:
            if tuple(cached.shape) != expected_shape:
                raise RuntimeError(
                    "DFA soma-error geometry changed after feedback initialization: "
                    f"{tuple(cached.shape)} vs {expected_shape}"
                )
            return cached.to(device=device, dtype=dtype)

        cfg_seed = getattr(self.local_cfg, "dfa_feedback_seed", None)
        seed = int(getattr(self, "seed", 0) if cfg_seed is None else cfg_seed)
        scale = float(getattr(self.local_cfg, "dfa_feedback_scale", 1.0))
        generator = torch.Generator()
        generator.manual_seed(seed)
        feedback = torch.randn(expected_shape, generator=generator)
        feedback.mul_(scale / float(output_dim) ** 0.5)
        self._dfa_feedback_matrix = feedback.to(device=device, dtype=dtype)
        logger.info(
            "Initialized local-CA DFA feedback matrix shape=%s seed=%s scale=%s",
            expected_shape,
            seed,
            scale,
        )
        return self._dfa_feedback_matrix

    def _resolve_dfa_soma_signals(
        self,
        *,
        delta_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Produce per-soma error through a fixed random feedback matrix.

        This is an explicit experimental mode: unlike the decoder mapping it
        raises instead of silently falling back, so a between-neuron DFA cell
        cannot degrade into the exact-readout cell unnoticed.
        """

        h_in = self._decoder_cache.get("input")
        if not isinstance(h_in, torch.Tensor):
            decoder_input = self._decoder_cache.get("decoder_input")
            if isinstance(decoder_input, torch.Tensor):
                h_in = decoder_input.detach()
        if not isinstance(h_in, torch.Tensor) or h_in.dim() != 2:
            raise RuntimeError(
                "soma_error_source='dfa' requires a cached two-dimensional "
                "decoder input"
            )
        if delta_out.dim() != 2 or delta_out.size(0) != h_in.size(0):
            raise RuntimeError(
                "soma_error_source='dfa' received incompatible error shapes: "
                f"{tuple(delta_out.shape)} vs decoder input {tuple(h_in.shape)}"
            )

        feedback = self._dfa_feedback_for(
            soma_dim=int(h_in.size(1)),
            output_dim=int(delta_out.size(1)),
            device=h_in.device,
            dtype=h_in.dtype,
        )
        delta_core = delta_out.to(dtype=h_in.dtype) @ feedback.t()
        return h_in.detach(), delta_core.detach()

    def _compute_reactivation_derivative(
        self,
        react_module: nn.Module | None,
        v_n: torch.Tensor,
        v_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the elementwise local slope d a / d V for a branch reactivation."""

        return compute_reactivation_derivative(
            react_module,
            v_n,
            v_out,
            logger=logger,
        )

    def _get_layer_activation_derivative(
        self,
        rec: dict[str, Any],
        v_n: torch.Tensor | None = None,
        v_out: torch.Tensor | None = None,
    ) -> torch.Tensor | None:

        cached = rec.get("_activation_derivative")

        if isinstance(cached, torch.Tensor):

            return cached

        if v_n is None:

            v_n = rec.get("v_n")

        if not isinstance(v_n, torch.Tensor):

            return None

        if v_out is None:

            v_out = rec.get("v_out")

        react_module = getattr(rec.get("layer"), "reactivation", None)

        deriv = self._compute_reactivation_derivative(
            react_module=react_module, v_n=v_n, v_out=v_out
        )

        rec["_activation_derivative"] = deriv

        return deriv

    def _resolve_layer_dynamics_mode(self, rec: dict[str, Any]) -> str:
        """Resolve local update dynamics for the current layer.



        Returns:

            "conductance" for shunting-style updates or "additive" for

            additive-consistent updates.

        """

        cfg_mode = str(
            getattr(self.local_cfg.three_factor, "dynamics_mode", "auto")
        ).lower()

        resolved_mode = _dynamics_mode_from_config(cfg_mode)
        if resolved_mode is not None:
            return resolved_mode

        if cfg_mode != "auto":

            logger.warning(
                f"Unknown three_factor.dynamics_mode='{cfg_mode}', falling back to auto."
            )

        return _dynamics_mode_from_layer(rec.get("layer"))

    @staticmethod
    def _error_by_mode(y_hat: torch.Tensor, y: torch.Tensor, mode: str) -> torch.Tensor:

        if mode == "mse":

            # d/dy_hat (1/2 ||y_hat - y||^2) = (y_hat - y)

            return y_hat - y

        elif mode == "bce":

            # Support both logits (BCEWithLogits-style) and probability outputs (BCELoss).

            pred = _prepare_bce_prediction(y_hat)

            return pred - y

        elif mode == "ce":

            # Gradient of CrossEntropy w.r.t logits: softmax(y_hat) - one_hot(y)

            y_onehot = _prepare_ce_target(y_hat, y)

            probs = functional.softmax(y_hat, dim=-1)

            return probs - y_onehot

        else:

            # Fallback

            return y_hat - y

    def _ensure_additive_gain_params(
        self, out_features: int, device: torch.device
    ) -> torch.Tensor:
        """Get or create per-neuron learnable gain params for additive mode.



        Uses ``_additive_gain_cache`` (persists across steps) rather than

        ``_decoder_cache`` (cleared every forward pass).

        """

        key = f"additive_gain_{out_features}"

        if key not in self._additive_gain_cache:

            # Initialize so sigmoid(param) ≈ 0.5

            param = torch.nn.Parameter(torch.zeros(out_features, device=device))

            self.optimizer.add_param_group(
                {"params": [param], "lr": self.optimizer.param_groups[0]["lr"]}
            )

            self._additive_gain_cache[key] = param

        return self._additive_gain_cache[key]

    def _compute_additive_pseudo_signals(
        self, rec: dict[str, Any], v_n: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pseudo-R_tot and pseudo-driving-force for additive networks."""

        return compute_additive_pseudo_signals(rec, v_n)

    def _get_additive_running_inv_std(
        self, layer_idx: int, v_n: torch.Tensor
    ) -> torch.Tensor:
        """Return inverse running std for additive normalization scaling."""

        stats = self._layer_stats.setdefault(layer_idx, _LayerStats())

        alpha = getattr(self.local_cfg.three_factor, "additive_stats_ema_alpha", 0.01)

        with torch.no_grad():

            batch_var = v_n.var(dim=0)  # [out]

            if stats.var_ema is None:

                stats.var_ema = batch_var.detach()

            else:

                stats.var_ema = (1 - alpha) * stats.var_ema + alpha * batch_var.detach()

        inv_std = 1.0 / (stats.var_ema.sqrt() + 1e-6)  # [out]

        return inv_std.unsqueeze(0)  # [1, out]


__all__ = ["LocalLearningSignalMixin"]
