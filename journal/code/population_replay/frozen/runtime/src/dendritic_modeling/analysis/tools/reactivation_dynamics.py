"""
Reactivation dynamics analyzer.

Tracks per-epoch trajectories of every parametric reactivation's gate
parameters ``(log_m, b)`` alongside statistics of the pre-reactivation
voltage distribution ``V`` and the post-reactivation output distribution
``r = (tanh(m*(V-b)) + 1) / 2``.

Purpose
-------
Diagnose whether the reactivation gate drifts during training (e.g., in
the "empirical calibration over-sharpens shunting" hypothesis): a
progressively steeper ``log_m`` paired with a shift in ``V`` quantiles
toward the saturation regions ``r < 0.05`` or ``r > 0.95`` indicates the
gate is memorizing the training signal by rail-saturating its output.

Each call emits one JSON snapshot per reactivation layer for that epoch;
a post-hoc tool (or a notebook) can concatenate all epoch files under
``<save_root>/reactivation_dynamics/epochs/epoch*.json`` into full
trajectories.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import DataLoader

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import analysis_device_context
from dendritic_modeling.config.analysis import (
    EvaluationRuntimeConfig,
    ReactivationDynamicsAnalysisParams,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.activations.parametric import ParametricTanh
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    iter_named_modules_of_type,
    register_hook_groups,
)


def _quantile_stats(v: torch.Tensor) -> dict[str, float]:
    """Compute the diagnostic statistics for a 1-D voltage tensor ``v``.

    Returns both moment-based (mean/std) and robust (median/MAD) estimators
    so we can cross-check them — when the distribution is approximately
    symmetric the two agree, and when they diverge the robust estimators
    are the ones actually used by ``calibrate_reactivation_empirical``.
    """
    v = v.reshape(-1).float()
    v = v[torch.isfinite(v)]
    if v.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "mad_std": 0.0,
            "q01": 0.0,
            "q10": 0.0,
            "q50": 0.0,
            "q90": 0.0,
            "q99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
        }
    mean = float(v.mean().item())
    std = float(v.std(unbiased=False).item())
    median = float(torch.median(v).item())
    mad = float(torch.median(torch.abs(v - median)).item())
    mad_std = 1.4826 * mad
    # Percentiles used by the quantile-occupancy calibration proposal.
    q = torch.quantile(
        v, torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99], dtype=v.dtype, device=v.device)
    )
    return {
        "mean": mean,
        "std": std,
        "median": median,
        "mad_std": mad_std,
        "q01": float(q[0].item()),
        "q10": float(q[1].item()),
        "q50": float(q[2].item()),
        "q90": float(q[3].item()),
        "q99": float(q[4].item()),
        "min": float(v.min().item()),
        "max": float(v.max().item()),
        "n": int(v.numel()),
    }


def _reactivation_output_stats(
    reactivation: ParametricTanh, v: torch.Tensor
) -> dict[str, float]:
    """Return summary stats of ``r = reactivation(v)`` for the given voltage.

    Captures the fraction of outputs piled at the rails (``< 0.05`` or
    ``> 0.95``) as a direct saturation indicator, plus simple moments
    for sanity-checking against ``r ∈ [0, 1]``.
    """
    with torch.no_grad():
        r = reactivation(v)
    r_flat = r.reshape(-1).float()
    r_flat = r_flat[torch.isfinite(r_flat)]
    if r_flat.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "sat_lo": 0.0,
            "sat_hi": 0.0,
            "active_frac": 0.0,
        }
    sat_lo = float((r_flat < 0.05).float().mean().item())
    sat_hi = float((r_flat > 0.95).float().mean().item())
    return {
        "mean": float(r_flat.mean().item()),
        "std": float(r_flat.std(unbiased=False).item()),
        "sat_lo": sat_lo,
        "sat_hi": sat_hi,
        "active_frac": 1.0 - sat_lo - sat_hi,
    }


def _param_tanh_params(reactivation: ParametricTanh) -> dict[str, float]:
    """Snapshot the learnable ``(log_m, b)`` parameters of a ParametricTanh.

    ``log_m`` is stored as a per-unit vector; we report mean and std across
    units so the trajectory compresses to two scalars per layer. ``m`` is
    derived from ``exp(log_m)`` for easier reading in plots.
    """
    with torch.no_grad():
        log_m = reactivation.log_m.detach()
        b = reactivation.b.detach() if torch.is_tensor(reactivation.b) else None
    m = torch.exp(log_m)
    out = {
        "log_m_mean": float(log_m.mean().item()),
        "log_m_std": float(log_m.std(unbiased=False).item()),
        "m_mean": float(m.mean().item()),
        "m_std": float(m.std(unbiased=False).item()),
        "m_min": float(m.min().item()),
        "m_max": float(m.max().item()),
    }
    if b is not None:
        out["b_mean"] = float(b.mean().item())
        out["b_std"] = float(b.std(unbiased=False).item())
        out["b_min"] = float(b.min().item())
        out["b_max"] = float(b.max().item())
    else:
        out["b_mean"] = 0.0
        out["b_std"] = 0.0
        out["b_min"] = 0.0
        out["b_max"] = 0.0
    return out


class ReactivationDynamicsAnalyzer(ForwardHookRemovalMixin, AbstractAnalyzer):
    """Per-epoch trajectory tracker for parametric reactivation gates.

    For every ``ParametricTanh`` in the model, on each ``analyze()`` call
    the analyzer:

    1. Runs a fixed-size sample batch through the model with forward
       pre-hooks on the reactivation modules, capturing the
       pre-reactivation voltage ``V``.
    2. Computes moment-based and robust-statistic summaries of ``V``
       (``mean/std`` and ``median/MAD*1.4826``), quantiles
       (``q01, q10, q50, q90, q99``), and extremes.
    3. Snapshots the current learnable ``(log_m, b)`` parameter values.
    4. Applies the reactivation to ``V`` once to measure the post-gate
       output's mean, std, and the saturation fractions
       ``sat_lo = P(r < 0.05)``, ``sat_hi = P(r > 0.95)``.
    5. Saves one JSON file per epoch containing one entry per layer.

    The analyzer has no internal epoch counter; the trainer passes
    ``filename=f"epoch{N}"`` into ``analyze()`` on each call. After
    training, the set of files under
    ``<save_root>/reactivation_dynamics/epochs/`` concatenates into
    per-layer trajectories.
    """

    def __init__(self, params: ReactivationDynamicsAnalysisParams):
        super().__init__("ReactivationDynamicsAnalyzer")
        self.params = params
        self.max_samples = int(getattr(params, "max_samples", 1024))

    def _list_parametric_tanh_layers(
        self, model: torch.nn.Module
    ) -> list[tuple[str, ParametricTanh]]:
        """Return ``(dotted_name, module)`` pairs for every ParametricTanh in the model."""
        return list(iter_named_modules_of_type(model, ParametricTanh))

    def _sample_inputs(
        self,
        data: torch.utils.data.Dataset,
        device: torch.device,
    ) -> torch.Tensor | None:
        """Pull up to ``self.max_samples`` inputs from the dataset as a single tensor."""
        if data is None:
            return None
        try:
            loader = DataLoader(
                data,
                batch_size=min(self.max_samples, max(1, len(data))),
                shuffle=False,
            )
            batch = next(iter(loader))
        except Exception as e:
            self.logger.warning(
                "ReactivationDynamicsAnalyzer could not draw a sample batch: %s", e
            )
            return None
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if x is None or not torch.is_tensor(x):
            return None
        return x[: self.max_samples].to(device)

    def analyze(
        self,
        model: BaseModel,
        train_ds: torch.utils.data.Dataset | None = None,
        valid_ds: torch.utils.data.Dataset | None = None,
        test_ds: torch.utils.data.Dataset | None = None,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "final",
        training: bool = False,
        runtime: EvaluationRuntimeConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Capture reactivation dynamics for one epoch snapshot.

        The canonical training-time call path from
        ``AnalysisManager.run_analysis`` provides ``filename=f"epoch{N}"``
        and ``training=True`` each epoch. On the final call
        (``training=False``), we use the test split so the last snapshot
        reflects held-out voltages.
        """
        layers = self._list_parametric_tanh_layers(model)
        if not layers:
            self.logger.debug("No ParametricTanh reactivations found; skipping")
            return {}

        dataset = self._select_dataset(training, train_ds, valid_ds, test_ds)
        if dataset is None:
            self.logger.debug("No dataset available; skipping dynamics snapshot")
            return {}

        with analysis_device_context(model, device) as current_device:
            x = self._sample_inputs(dataset, current_device)
            if x is None:
                return {}

            x = self._prepare_inputs(model, x)
            captured = self._capture_reactivation_inputs(model, layers, x)
            if captured is None:
                return {}

            snapshot = self._build_snapshot(layers, captured, filename, training)

        if save_path is not None:
            self._save_snapshot(snapshot, save_path, filename)

        return snapshot

    def _select_dataset(
        self,
        training: bool,
        train_ds: torch.utils.data.Dataset | None,
        valid_ds: torch.utils.data.Dataset | None,
        test_ds: torch.utils.data.Dataset | None,
    ) -> torch.utils.data.Dataset | None:
        dataset = train_ds if training else test_ds
        if dataset is None:
            dataset = valid_ds or test_ds
        return dataset

    def _prepare_inputs(
        self,
        model: BaseModel,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(model, "_is_recurrent_core", False) and x.dim() == 2:
            return x.unsqueeze(1)
        return x

    def _capture_reactivation_inputs(
        self,
        model: BaseModel,
        layers: list[tuple[str, ParametricTanh]],
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor] | None:
        captured: dict[str, torch.Tensor] = {}
        handles = self._attach_capture_hooks(layers, captured)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                try:
                    _ = model(x)
                except Exception as e:
                    self.logger.warning(
                        "ReactivationDynamicsAnalyzer forward pass failed: %s", e
                    )
                    return None
        finally:
            self.remove_forward_hooks(handles)
            if was_training:
                model.train()
        return captured

    def _attach_capture_hooks(
        self,
        layers: list[tuple[str, ParametricTanh]],
        captured: dict[str, torch.Tensor],
    ) -> list[torch.utils.hooks.RemovableHandle]:
        return register_hook_groups(
            layers,
            lambda layer_entry: [self._register_capture_hook(layer_entry, captured)],
        )

    def _register_capture_hook(
        self,
        layer_entry: tuple[str, ParametricTanh],
        captured: dict[str, torch.Tensor],
    ) -> torch.utils.hooks.RemovableHandle:
        name, layer = layer_entry
        return layer.register_forward_pre_hook(self._make_hook(name, captured))

    @staticmethod
    def _make_hook(layer_name: str, captured: dict[str, torch.Tensor]):
        def hook(_mod, inputs):
            if layer_name in captured:
                return
            captured[layer_name] = inputs[0].detach()

        return hook

    def _build_snapshot(
        self,
        layers: list[tuple[str, ParametricTanh]],
        captured: dict[str, torch.Tensor],
        filename: str,
        training: bool,
    ) -> dict[str, Any]:
        per_layer: dict[str, dict[str, Any]] = {}
        for name, layer in layers:
            v = captured.get(name)
            if v is None:
                per_layer[name] = {"skipped": True}
                continue
            per_layer[name] = {
                "V": _quantile_stats(v),
                "params": _param_tanh_params(layer),
                "reactivation_output": _reactivation_output_stats(layer, v),
            }

        return {
            "filename": filename,
            "training": bool(training),
            "n_layers": len(per_layer),
            "layers": per_layer,
        }

    def _save_snapshot(
        self,
        snapshot: dict[str, Any],
        save_path: str,
        filename: str,
    ) -> None:
        epochs_dir = os.path.join(save_path, "epochs")
        os.makedirs(epochs_dir, exist_ok=True)
        save_dict(snapshot, epochs_dir, f"{filename}.json")


__all__ = ["ReactivationDynamicsAnalyzer"]
