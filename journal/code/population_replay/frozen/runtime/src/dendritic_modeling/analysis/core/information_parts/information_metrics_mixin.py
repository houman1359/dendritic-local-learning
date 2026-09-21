"""Metric orchestration helpers for information analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from dendritic_modeling.analysis.core.information_parts.information_mutual_information_mixin import (
    InformationMutualInformationMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_null_baseline_mixin import (
    _metric_variable_for_null_baseline,
    _parse_mi_key_for_null_baseline,
)


class InformationMetricsMixin(InformationMutualInformationMixin):
    """Orchestrates requested information metrics for one activation set."""

    def _metric_compute_jobs(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: np.ndarray | None,
        Vb: np.ndarray | None,
        S: np.ndarray | None,
    ) -> tuple[tuple[bool, str, str, Callable[[], Any]], ...]:
        return (
            (
                self.compute_basic_mi,
                "basic_mi",
                "Computing basic MI metrics...",
                lambda: self._compute_basic_mi(E, I_var, Vout, C, Vinf, Vb),
            ),
            (
                self.compute_pairwise_mi,
                "pairwise_mi",
                "Computing pairwise MI metrics...",
                lambda: self._compute_pairwise_mi(E, I_var, Vout, Vinf, Vb),
            ),
            (
                self.compute_conditional_mi,
                "conditional_mi",
                "Computing conditional MI metrics...",
                lambda: self._compute_conditional_mi(E, I_var, Vout, C, Vinf, Vb),
            ),
            (
                self.compute_gaussian_fisher,
                "gaussian_fisher",
                "Computing Gaussian/Fisher proxy metrics...",
                lambda: self._compute_gaussian_fisher(E, I_var, Vout, C, Vinf, Vb),
            ),
            (
                self.compute_pid and self.pid_available,
                "pid",
                "Computing PID metrics...",
                lambda: self._compute_pid(E, I_var, Vout, C),
            ),
            (
                getattr(self, "compute_soma_coupling_mi", False) and S is not None,
                "soma_coupling_mi",
                "Computing branch-to-parent-soma MI metrics...",
                lambda: self._compute_soma_coupling_mi(E, I_var, Vout, C, S),
            ),
        )

    def compute_metrics(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: np.ndarray | None = None,
        Vb: np.ndarray | None = None,
        S: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute all requested information metrics.

        Parameters
        ----------
        E : np.ndarray
            Excitatory inputs, shape (n_samples,) or (n_samples, n_features)
        I_var : np.ndarray
            Inhibitory inputs, shape (n_samples,) or (n_samples, n_features)
        Vout : np.ndarray
            Branch outputs, shape (n_samples,) or (n_samples, n_features)
        C : np.ndarray
            Class labels, shape (n_samples,)
        Vinf : np.ndarray, optional
            Soma activation (Vinf), shape (n_samples,) or (n_samples, n_features)
        Vb : np.ndarray, optional
            Branch input (upstream dendritic input), shape (n_samples,) or (n_samples, n_features)
        S : np.ndarray, optional
            Parent soma output corresponding to the analyzed branch.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all computed metrics
        """
        # Validate inputs
        self._validate_inputs(E, I_var, Vout, C, Vinf)

        # Ensure all arrays are properly shaped
        E = self._ensure_2d(E)
        I_var = self._ensure_2d(I_var)
        Vout = self._ensure_2d(Vout)
        C = C.ravel()  # Labels are always 1D

        if Vinf is not None:
            Vinf = self._ensure_2d(Vinf)

        if Vb is not None:
            Vb = self._ensure_2d(Vb)
        if S is not None:
            S = self._ensure_2d(S)
            if len(S) != len(E):
                raise ValueError(
                    f"S must have same number of samples as E: {len(S)} vs {len(E)}"
                )

        results = {
            "method": self.method,
            "n_samples": len(E),
        }

        for enabled, result_key, log_message, compute_fn in self._metric_compute_jobs(
            E, I_var, Vout, C, Vinf, Vb, S
        ):
            if enabled:
                self.logger.info(log_message)
                results[result_key] = compute_fn()

        return results


__all__ = [
    "InformationMetricsMixin",
    "_metric_variable_for_null_baseline",
    "_parse_mi_key_for_null_baseline",
]
