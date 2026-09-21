"""Gaussian/Fisher proxy metrics for information analysis."""

from __future__ import annotations

import numpy as np

from dendritic_modeling.analysis.core.information_parts.information_helpers import (
    fisher_dprime2_binary,
    fisher_dprime2_pairwise_avg,
    fisher_mi_from_f,
    gaussian_fisher_proxy,
)

FisherVariableSpec = tuple[str, np.ndarray]


def _gaussian_fisher_variable_specs(
    E: np.ndarray,
    I_var: np.ndarray,
    Vout: np.ndarray,
    Vinf: np.ndarray | None,
    Vb: np.ndarray | None,
    *,
    include_vinf: bool,
) -> list[FisherVariableSpec]:
    specs = [
        ("E", E),
        ("I", I_var),
    ]
    if Vb is not None:
        specs.append(("Vb", Vb))
    specs.append(("Vout", Vout))
    if Vinf is not None and include_vinf:
        specs.append(("Vinf", Vinf))
    return specs


def _store_gaussian_fisher_proxy(
    metrics: dict[str, float],
    name: str,
    F: float,
    I_proxy: float,
) -> None:
    # Use an I(.;.) prefix so LDA/shuffled variants get token-renamed.
    metrics[f"I({name};C)_fisher_F"] = float(F)
    metrics[f"I({name};C)_fisher"] = float(I_proxy)


class InformationFisherMetricsMixin:
    """Computes Gaussian/Fisher proxy metrics and flow increments."""

    def _gaussian_fisher_proxy(
        self, x: np.ndarray, labels: np.ndarray
    ) -> tuple[float, float]:
        """Return (F, I_proxy) for I(X;C) under a Gaussian/Fisher proxy."""
        return gaussian_fisher_proxy(
            x,
            labels,
            eps=float(getattr(self, "gaussian_fisher_eps", 1e-8)),
            aggregation=str(getattr(self, "gaussian_fisher_aggregation", "sum")),
            mi_transform=str(
                getattr(self, "gaussian_fisher_mi_transform", "half_log1p")
            ),
        )

    def _fisher_mi_from_f(self, f: float) -> float:
        """Map a Fisher/discriminability-like scalar to an MI-like proxy."""
        return fisher_mi_from_f(
            f,
            mi_transform=str(
                getattr(self, "gaussian_fisher_mi_transform", "half_log1p")
            ),
        )

    def _fisher_dprime2_binary(self, x: np.ndarray, labels: np.ndarray) -> float:
        """Binary-class Fisher discriminability: d'^2 = Delta mu^T Sigma^-1 Delta mu."""
        return fisher_dprime2_binary(
            x, labels, eps=float(getattr(self, "gaussian_fisher_eps", 1e-8))
        )

    def _fisher_dprime2_pairwise_avg(self, x: np.ndarray, labels: np.ndarray) -> float:
        """Multi-class surrogate: average binary d'^2 over all class pairs."""
        return fisher_dprime2_pairwise_avg(
            x, labels, eps=float(getattr(self, "gaussian_fisher_eps", 1e-8))
        )

    def _compute_gaussian_fisher(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: np.ndarray | None = None,
        Vb: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute Gaussian/Fisher proxy metrics for available variables."""
        metrics: dict[str, float] = {}

        for name, X in _gaussian_fisher_variable_specs(
            E,
            I_var,
            Vout,
            Vinf,
            Vb,
            include_vinf=self.include_vinf,
        ):
            F, I_proxy = self._gaussian_fisher_proxy(X, C)
            _store_gaussian_fisher_proxy(metrics, name, F, I_proxy)

        # Information-flow-style increment from upstream branches (Vb) beyond local synapses (E,I).
        # This mirrors the Schur-complement increment in the theory draft (Sec. 'Multi-branch Trees
        # and Information Flow') but uses a Fisher/geometry surrogate (d'^2) computed from moments.
        if Vb is not None:
            try:
                EI = np.hstack([E, I_var])
                EIVb = np.hstack([E, I_var, Vb])
                d2_full = self._fisher_dprime2_pairwise_avg(EIVb, C)
                d2_base = self._fisher_dprime2_pairwise_avg(EI, C)
                inc = float(max(0.0, d2_full - d2_base))
                metrics["I(Vb;C|E,I)_fisher_F"] = inc
                metrics["I(Vb;C|E,I)_fisher"] = float(self._fisher_mi_from_f(inc))
            except Exception as e:
                self.logger.warning(
                    f"Failed to compute Fisher flow increment I(Vb;C|E,I)_fisher: {e}"
                )
                metrics["I(Vb;C|E,I)_fisher_F"] = 0.0
                metrics["I(Vb;C|E,I)_fisher"] = 0.0

        return metrics


__all__ = ["InformationFisherMetricsMixin"]
