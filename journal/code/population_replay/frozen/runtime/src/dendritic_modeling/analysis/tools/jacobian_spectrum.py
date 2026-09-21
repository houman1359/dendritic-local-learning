"""
Per-level effective recurrent-weight singular-value analyzer.

For every dendritic branch layer of the excitatory population, this
analyzer reads the realized sparse recurrent excitation and inhibition
operators, computes their
leading singular values, and reports the top singular value at each level.
When the dendritic population also exposes a per-level time constant
``log_tau``, these operator-norm summaries are tabulated against τ so the
gradient flow through long-τ compartments can be read off at a glance.

This is a static operator summary, not the Jacobian of the recurrent state
transition. It returns an empty dict for baseline RNN cores or layers without
recurrent compartments.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.analysis.utils.recurrent_introspection import (
    iter_recurrent_populations,
)
from dendritic_modeling.config.analysis import JacobianSpectrumAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


def _singular_values(w: torch.Tensor, top_k: int) -> list[float]:
    """Return the top-``top_k`` singular values of ``w`` as a list of floats."""
    if w.numel() == 0:
        return []
    # torch.linalg.svdvals is cheaper than full SVD when we only want
    # singular values.
    with torch.no_grad():
        s = torch.linalg.svdvals(w.float().detach().cpu())
    return s[:top_k].tolist()


def _spectra_for_population(
    e_population: torch.nn.Module, top_k: int
) -> dict[str, Any] | None:
    """Collect per-level ``{sigma_rec, sigma_inh}`` and τ from a population."""
    branch_layers = getattr(e_population, "branch_layers", None)
    if branch_layers is None or len(branch_layers) == 0:
        return None

    tau_per_level: list[float] | None = None
    if hasattr(e_population, "current_taus"):
        with torch.no_grad():
            tau_per_level = e_population.current_taus.detach().cpu().tolist()
    else:
        log_tau = getattr(e_population, "log_tau", None)
        if log_tau is not None:
            with torch.no_grad():
                tau_per_level = torch.exp(log_tau.detach().cpu()).tolist()

    levels: list[dict[str, Any]] = []
    saw_any = False
    for l_idx, lyr in enumerate(branch_layers):
        entry: dict[str, Any] = {"level": int(l_idx)}
        if tau_per_level is not None and l_idx < len(tau_per_level):
            entry["tau"] = float(tau_per_level[l_idx])

        rec = getattr(lyr, "branch_recurrent", None)
        inh = getattr(lyr, "branch_rec_inhibition", None)

        if rec is not None:
            with torch.no_grad():
                rec_snapshot = effective_synapse_snapshot(rec)
                w_rec = rec_snapshot.effective_weight.cpu()
            sigma_rec = _singular_values(w_rec, top_k)
            entry["sigma_rec"] = sigma_rec
            top_rec = float(sigma_rec[0]) if sigma_rec else 0.0
            entry["top_singular_value_rec"] = top_rec
            # Backward-compatible alias for existing consumers/tests.
            entry["spectral_radius_rec"] = top_rec
            entry["realized_k_rec"] = rec_snapshot.realized_k.cpu().tolist()
            entry["mask_source_rec"] = rec_snapshot.mask_source
            saw_any = True
        if inh is not None:
            with torch.no_grad():
                inh_snapshot = effective_synapse_snapshot(inh)
                w_inh = inh_snapshot.effective_weight.cpu()
            sigma_inh = _singular_values(w_inh, top_k)
            entry["sigma_inh"] = sigma_inh
            top_inh = float(sigma_inh[0]) if sigma_inh else 0.0
            entry["top_singular_value_inh"] = top_inh
            # Backward-compatible alias for existing consumers/tests.
            entry["spectral_radius_inh"] = top_inh
            entry["realized_k_inh"] = inh_snapshot.realized_k.cpu().tolist()
            entry["mask_source_inh"] = inh_snapshot.mask_source
            saw_any = True

        levels.append(entry)

    if not saw_any:
        return None

    return {"n_levels": len(levels), "levels": levels}


class JacobianSpectrumAnalyzer(AbstractAnalyzer):
    """Per-level singular-value snapshot of a recurrent model.

    Each trained E-I layer produces one entry in the output JSON: a list
    of per-level records carrying τ, the top-``top_k`` singular values of
    ``W_rec_l`` and ``W_inh_l``. The optional PDF plots the top-k
    singular-value curves and a bar chart of top singular values by level
    for quick visual inspection.
    """

    def __init__(self, params: JacobianSpectrumAnalysisParams | None = None):
        super().__init__("JacobianSpectrumAnalyzer")
        if params is None:
            params = JacobianSpectrumAnalysisParams()
        self.params = params

    def analyze(
        self,
        model: BaseModel,
        data: torch.utils.data.Dataset | None = None,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "jacobian_spectrum",
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.log_analysis_start("jacobian_spectrum")

        core = getattr(model, "core_network", model)
        top_k = int(self.params.top_k)
        per_layer = self._collect_layer_spectra(core, top_k)

        if not per_layer:
            self.logger.info(
                "Skipping jacobian spectrum: no recurrent dendritic layer found"
            )
            return {}

        results = self._build_results(top_k, per_layer)

        if save_path is not None:
            self._save_results(results, save_path, filename)

        self.log_analysis_end("jacobian_spectrum", num_results=len(per_layer))
        return results

    def _collect_layer_spectra(
        self, core: torch.nn.Module, top_k: int
    ) -> dict[str, dict[str, Any]]:
        per_layer: dict[str, dict[str, Any]] = {}
        for record in iter_recurrent_populations(core):
            if record.polarity != "excitatory":
                continue
            entry = _spectra_for_population(record.population, top_k=top_k)
            if entry is None:
                continue
            per_layer[self._entry_key(record)] = entry
        return per_layer

    @staticmethod
    def _entry_key(record) -> str:
        if record.layer_index is not None and record.population_name == "excitatory":
            return f"layer_{record.layer_index}"
        return record.key

    @staticmethod
    def _build_results(
        top_k: int, per_layer: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "analysis_type": "effective_recurrent_weight_spectrum",
            "is_state_transition_jacobian": False,
            "top_k": top_k,
            "layers": per_layer,
        }

    def _save_results(
        self, results: dict[str, Any], save_path: str, filename: str
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        save_dict(results, save_path, f"{filename}.json")
        if self.params.save_figures:
            try:
                self._save_figures(results, save_path, filename)
            except Exception as e:  # pragma: no cover - plotting is best-effort
                self.logger.warning(
                    "Figure generation failed for jacobian spectrum: %s", e
                )

    def _save_figures(
        self, results: dict[str, Any], save_path: str, filename: str
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        layers = results["layers"]
        n_layers = len(layers)
        fig, axes = plt.subplots(
            n_layers,
            2,
            figsize=(7.0, 2.7 * max(n_layers, 1)),
            gridspec_kw={"wspace": 0.35, "hspace": 0.55},
            squeeze=False,
        )

        for li, (layer_key, entry) in enumerate(layers.items()):
            ax_curve, ax_bar = axes[li]
            n_levels = entry["n_levels"]
            cmap = plt.get_cmap("viridis")
            colors = [cmap(j / max(1, n_levels - 1)) for j in range(n_levels)]

            for j, lvl in enumerate(entry["levels"]):
                sigma_rec = lvl.get("sigma_rec", [])
                tau = lvl.get("tau", None)
                label = f"L{lvl['level']}"
                if tau is not None:
                    label += f" (τ={tau:.1f})"
                if sigma_rec:
                    ax_curve.plot(
                        range(len(sigma_rec)),
                        sigma_rec,
                        marker="o",
                        ms=3.0,
                        lw=1.0,
                        color=colors[j],
                        label=label,
                    )
            ax_curve.set_xlabel("singular-value rank")
            ax_curve.set_ylabel(r"$\sigma_k(W^{\mathrm{rec}}_l)$")
            ax_curve.set_title(f"{layer_key}: rec spectrum", fontsize=8)
            ax_curve.legend(fontsize=6, loc="upper right", framealpha=0.9)

            bar_x = np.arange(n_levels)
            top_rec = [
                lvl.get("top_singular_value_rec", 0.0) for lvl in entry["levels"]
            ]
            top_inh = [
                lvl.get("top_singular_value_inh", 0.0) for lvl in entry["levels"]
            ]
            ax_bar.bar(
                bar_x - 0.18,
                top_rec,
                0.32,
                color=colors,
                edgecolor="white",
                linewidth=0.4,
                label=r"$\sigma_1(W^{\mathrm{rec}})$",
                alpha=0.9,
            )
            ax_bar.bar(
                bar_x + 0.18,
                top_inh,
                0.32,
                color=colors,
                edgecolor="white",
                linewidth=0.4,
                label=r"$\sigma_1(W^{\mathrm{inh}})$",
                alpha=0.55,
                hatch="//",
            )
            tick_labels = []
            for lvl in entry["levels"]:
                t = f"L{lvl['level']}"
                if "tau" in lvl:
                    t += f"\nτ={lvl['tau']:.1f}"
                tick_labels.append(t)
            ax_bar.set_xticks(bar_x)
            ax_bar.set_xticklabels(tick_labels, fontsize=6.5)
            ax_bar.set_ylabel("top singular value")
            ax_bar.set_title(f"{layer_key}: top singular values", fontsize=8)
            ax_bar.legend(fontsize=6, loc="upper right", framealpha=0.9)

        fig.suptitle("Per-level recurrent spectrum", fontsize=9, y=0.995)
        fig.savefig(os.path.join(save_path, f"{filename}.pdf"), bbox_inches="tight")
        plt.close(fig)


__all__ = ["JacobianSpectrumAnalyzer"]
