"""
Dendritic time-trace simulator for recurrent E-I dendritic models.

Given a trained per-level time constant vector ``log_tau`` and the
parametric-reactivation parameters ``(log_m, b)`` at each branch layer,
this analyzer simulates the idealized cascade dynamics of a single E
population on a configurable synthetic input trial::

    V_L(t)   = rho_L * V_L(t-1)   + (1 - rho_L) * u(t)
    V_l(t)   = rho_l * V_l(t-1)   + (1 - rho_l) * V_{l+1}(t)    for l < L
    hat V_l  = 0.5 * (1 + tanh(m_l * (V_l - b_l)))

where ``rho_l = exp(-1 / tau_l)`` and ``m_l``, ``b_l`` are the mean of
the per-unit reactivation parameters at level ``l``. The output is the
per-level ``V_l(t)`` and ``hat V_l(t)`` under the specified stimulus
protocol (default: a Romo-style two-pulse trial), suitable for figures
that illustrate how the learned dendritic cascade shapes temporal
dynamics across compartments.

The analyzer is a static readout of the trained parameters — it does
not consume any dataset — and returns an empty dict when the model has
no dendritic population or no ``log_tau`` (e.g. baseline RNN).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.recurrent_introspection import (
    iter_recurrent_populations,
)
from dendritic_modeling.config.analysis import DendriticTimetracesAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


def _build_input(params: DendriticTimetracesAnalysisParams) -> list[float]:
    """Return the synthetic input signal ``u(t)`` for ``t = 0 .. T-1``."""
    T = int(params.n_timesteps)
    u = [0.0] * T
    if params.stim1_window is not None:
        s, e = params.stim1_window
        for t in range(max(0, s), min(T, e)):
            u[t] += float(params.stim1_amp)
    if params.stim2_window is not None:
        s, e = params.stim2_window
        for t in range(max(0, s), min(T, e)):
            u[t] += float(params.stim2_amp)
    if params.noise_std > 0:
        g = torch.Generator()
        g.manual_seed(int(params.seed))
        noise = torch.randn(T, generator=g) * float(params.noise_std)
        for t in range(T):
            u[t] = float(u[t] + noise[t].item())
    return u


def _simulate_cascade(
    u: list[float],
    tau: list[float],
    m: list[float],
    b: list[float],
) -> tuple[list[list[float]], list[list[float]]]:
    """Simulate the per-level cascade. Returns ``(V, V_hat)`` of shape ``(L, T)``."""
    T = len(u)
    L = len(tau)
    rho = [math.exp(-1.0 / max(tau_l, 1e-6)) for tau_l in tau]
    V = [[0.0] * T for _ in range(L)]
    # Convention: level 0 = most-distal, level L-1 = soma. StatefulDendriNet
    # integrates distal compartments first, then aggregates toward the soma.
    for t in range(1, T):
        V[0][t] = rho[0] * V[0][t - 1] + (1.0 - rho[0]) * u[t]
        for level_idx in range(1, L):
            V[level_idx][t] = (
                rho[level_idx] * V[level_idx][t - 1]
                + (1.0 - rho[level_idx]) * V[level_idx - 1][t]
            )
    V_hat = [
        [
            0.5 * (1.0 + math.tanh(m[level_idx] * (V[level_idx][t] - b[level_idx])))
            for t in range(T)
        ]
        for level_idx in range(L)
    ]
    return V, V_hat


def _timetraces_for_population(
    e_population: torch.nn.Module,
    params: DendriticTimetracesAnalysisParams,
) -> dict[str, Any] | None:
    with torch.no_grad():
        if hasattr(e_population, "current_taus"):
            tau = e_population.current_taus.detach().cpu().tolist()
        else:
            log_tau = getattr(e_population, "log_tau", None)
            if log_tau is None:
                return None
            tau = torch.exp(log_tau.detach().cpu()).tolist()

    branch_layers = getattr(e_population, "branch_layers", None)
    if branch_layers is None or len(branch_layers) != len(tau):
        return None

    m: list[float] = []
    b: list[float] = []
    for lyr in branch_layers:
        reactivation = getattr(lyr, "reactivation", None)
        if reactivation is None:
            return None
        log_m = getattr(reactivation, "log_m", None)
        bias = getattr(reactivation, "b", None)
        if log_m is None or bias is None:
            return None
        with torch.no_grad():
            m.append(float(torch.exp(log_m.detach().cpu()).mean().item()))
            b.append(float(bias.detach().cpu().mean().item()))

    u = _build_input(params)
    V, V_hat = _simulate_cascade(u, tau, m, b)

    return {
        "tau": tau,
        "m": m,
        "b": b,
        "n_levels": len(tau),
        "n_timesteps": len(u),
        "u": u,
        "V": V,
        "V_hat": V_hat,
        "stim1_window": list(params.stim1_window) if params.stim1_window else None,
        "stim2_window": list(params.stim2_window) if params.stim2_window else None,
    }


class DendriticTimetracesAnalyzer(AbstractAnalyzer):
    """Simulate the per-level V and hat-V time traces of a trained model.

    For each E-I layer with a dendritic excitatory population, writes a
    JSON payload with the trained τ/m/b, the stimulus protocol used,
    and the simulated ``V_l(t)`` and ``\\hat V_l(t)`` arrays; plus a
    multi-row PDF showing input, per-level V and hat V overlaid.
    """

    def __init__(self, params: DendriticTimetracesAnalysisParams | None = None):
        super().__init__("DendriticTimetracesAnalyzer")
        if params is None:
            params = DendriticTimetracesAnalysisParams()
        self.params = params

    def analyze(
        self,
        model: BaseModel,
        data: torch.utils.data.Dataset | None = None,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "dendritic_timetraces",
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.log_analysis_start("dendritic_timetraces")

        core = getattr(model, "core_network", model)
        per_layer = self._collect_layer_timetraces(core)

        if not per_layer:
            self.logger.info(
                "Skipping dendritic timetraces: no excitatory recurrent dendritic "
                "population with tau/readout parameters"
            )
            return {}

        results = self._build_results(per_layer)

        if save_path is not None:
            self._save_results(results, save_path, filename)

        self.log_analysis_end("dendritic_timetraces", num_results=len(per_layer))
        return results

    def _collect_layer_timetraces(self, core: torch.nn.Module) -> dict[str, Any]:
        per_layer: dict[str, dict[str, Any]] = {}
        for record in iter_recurrent_populations(core):
            if record.polarity != "excitatory":
                continue
            entry = _timetraces_for_population(record.population, self.params)
            if entry is None:
                continue
            entry["population"] = record.population_name
            entry["polarity"] = record.polarity
            per_layer[self._entry_key(record)] = entry
        return per_layer

    @staticmethod
    def _entry_key(record) -> str:
        if record.layer_index is not None and record.population_name == "excitatory":
            return f"layer_{record.layer_index}"
        return record.key

    def _build_results(self, per_layer: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            "stim1_window": (
                list(self.params.stim1_window) if self.params.stim1_window else None
            ),
            "stim2_window": (
                list(self.params.stim2_window) if self.params.stim2_window else None
            ),
            "response_window": (
                list(self.params.response_window)
                if self.params.response_window
                else None
            ),
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
                    "Figure generation failed for dendritic timetraces: %s", e
                )

    def _save_figures(
        self, results: dict[str, Any], save_path: str, filename: str
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        layers = results["layers"]
        for layer_key, entry in layers.items():
            L = entry["n_levels"]
            cmap = plt.get_cmap("viridis")
            colors = [cmap(j / max(1, L - 1)) for j in range(L)]

            fig, axes = plt.subplots(
                L + 1,
                1,
                figsize=(7.0, 1.2 * (L + 1)),
                sharex=True,
                gridspec_kw={"hspace": 0.25},
            )

            ax_in = axes[0]
            ax_in.plot(entry["u"], color="#333", lw=1.0, label="u(t)")
            for win_key, color in (
                ("stim1_window", "#1C7C54"),
                ("stim2_window", "#7A8352"),
            ):
                if entry.get(win_key):
                    s, e = entry[win_key]
                    ax_in.axvspan(s, e, color=color, alpha=0.12, lw=0)
            if results.get("response_window"):
                s, e = results["response_window"]
                ax_in.axvspan(s, e, color="#A64D79", alpha=0.10, lw=0)
            ax_in.set_ylabel("input")
            ax_in.set_title(
                f"{layer_key}: V_l(t) and hat V_l(t) on synthetic trial",
                fontsize=8,
            )
            ax_in.legend(fontsize=5.5, loc="upper right", framealpha=0.9)

            for level_idx in range(L):
                ax = axes[level_idx + 1]
                tau = entry["tau"][level_idx]
                ax.plot(
                    entry["V"][level_idx],
                    color=colors[level_idx],
                    lw=1.0,
                    alpha=0.5,
                    label="V",
                )
                ax.plot(
                    entry["V_hat"][level_idx],
                    color=colors[level_idx],
                    lw=1.4,
                    label=r"$\hat V$",
                )
                ax.set_ylabel(f"L{level_idx}\nτ={tau:.1f}", fontsize=7)
                ax.legend(fontsize=5.5, loc="upper right", framealpha=0.9)

            axes[-1].set_xlabel("time step t")
            fig.savefig(
                os.path.join(save_path, f"{filename}_{layer_key}.pdf"),
                bbox_inches="tight",
            )
            plt.close(fig)


__all__ = ["DendriticTimetracesAnalyzer"]
