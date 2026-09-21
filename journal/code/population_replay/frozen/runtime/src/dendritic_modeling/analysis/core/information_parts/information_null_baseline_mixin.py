"""Label-shuffle null baselines for information metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from dendritic_modeling.analysis.core.information_parts.information_pid_mixin import (
    InformationPidMetricsMixin,
)

ParsedMiKey = tuple[list[str], str, list[str]]


def _parse_mi_key_for_null_baseline(mi_key: Any) -> ParsedMiKey | None:
    if not isinstance(mi_key, str):
        return None
    key = mi_key.strip().replace(" ", "")
    if not (key.startswith("I(") and key.endswith(")")):
        return None
    inside = key[2:-1]
    if ";" not in inside:
        return None
    left, right = inside.split(";", 1)
    if "|" in right:
        y, z = right.split("|", 1)
        z_vars = [t for t in z.split(",") if t]
    else:
        y = right
        z_vars = []
    x_vars = [t for t in left.split(",") if t]
    return x_vars, y, z_vars


def _metric_variable_for_null_baseline(
    token: str,
    *,
    E: np.ndarray | None,
    I_var: np.ndarray | None,
    Vout: np.ndarray | None,
    C_col: np.ndarray,
    Vinf: np.ndarray | None,
    Vb: np.ndarray | None,
    S: np.ndarray | None = None,
    has_exc_synapses: bool,
    has_inh_synapses: bool,
    has_branch_input: bool,
) -> np.ndarray | None:
    base = token.split("_", 1)[0]
    if base == "C":
        return C_col
    if base == "E":
        return E if has_exc_synapses else None
    if base == "I":
        return I_var if has_inh_synapses else None
    if base == "Vb":
        return Vb if has_branch_input else None
    if base == "Vout":
        return Vout
    if base == "Vinf":
        return Vinf
    if base == "S":
        return S
    return None


class InformationNullBaselineMixin(InformationPidMetricsMixin):
    """Adds permutation floors and signed excesses for selected MI/CMI metrics."""

    def _add_label_shuffle_null_baselines_in_place(
        self,
        result: dict[str, Any],
        *,
        E: np.ndarray | None,
        I_var: np.ndarray | None,
        Vout: np.ndarray | None,
        C: np.ndarray,
        Vinf: np.ndarray | None = None,
        Vb: np.ndarray | None = None,
        S: np.ndarray | None = None,
        has_exc_synapses: bool = True,
        has_inh_synapses: bool = True,
        has_branch_input: bool = True,
        seed_offset: int = 0,
    ) -> None:
        r"""Add permutation floors and excess MI for selected metrics (in-place).

        This is a finite-sample permutation-floor diagnostic: it permutes *one
        of the variables* across samples and recomputes MI/CMI using the *same*
        estimator selected by ``estimator.method``.  For every computed null,
        ``*_excess`` is the signed observed estimate minus the mean permutation
        floor.  This estimator-specific floor subtraction is not a general
        analytical bias correction; for example, it remains distinct from the
        Miller--Madow correction used by the binned estimator.

        Notes
        -----
        - For ``I(X;Y)`` we use ``estimator.mutual_information_null(X, Y)`` which
          permutes ``Y`` across samples.
        - For ``I(X;Y|Z)`` we permute ``Y`` and recompute
          ``estimator.conditional_mutual_information(X, Y_perm, Z)``.
          If the conditioning variable is exactly ``Z=C`` (discrete labels),
          we permute ``Y`` \emph{within each class} to preserve the marginal
          distribution of ``Y|C`` while destroying within-class dependence.
        """
        if int(getattr(self, "mi_null_shuffles", 0) or 0) <= 0:
            return

        null_components = getattr(self, "mi_null_components", {"basic"})
        want_all = "all" in null_components
        want_basic = want_all or ("basic" in null_components)
        want_pairwise = want_all or ("pairwise" in null_components)
        want_conditional = want_all or bool(
            {"conditional", "ablation_aligned", "upstream_unique_cmi"}.intersection(
                null_components
            )
        )
        want_soma_coupling = want_all or bool(
            {
                "soma_coupling",
                "soma_coupling_mi",
                "branch_soma",
                "parent_soma",
            }.intersection(null_components)
        )
        categories: list[str] = []
        if want_basic:
            categories.append("basic_mi")
        if want_pairwise:
            categories.append("pairwise_mi")
        if want_conditional:
            categories.append("conditional_mi")
        if want_soma_coupling:
            categories.append("soma_coupling_mi")
        if not categories:
            return

        estimator = self.estimators[self.method]
        n_shuf = int(self.mi_null_shuffles)
        seed_base = int(getattr(self, "mi_null_seed", 0) or 0) + int(seed_offset)
        C_vec = np.asarray(C).ravel()
        C_col = C_vec.reshape(-1, 1)

        def _var_data(token: str) -> np.ndarray | None:
            return _metric_variable_for_null_baseline(
                token,
                E=E,
                I_var=I_var,
                Vout=Vout,
                C_col=C_col,
                Vinf=Vinf,
                Vb=Vb,
                S=S,
                has_exc_synapses=has_exc_synapses,
                has_inh_synapses=has_inh_synapses,
                has_branch_input=has_branch_input,
            )

        perms: list[np.ndarray] | None = None
        seed_off = 0
        for category in categories:
            if category not in result or not isinstance(result[category], dict):
                continue
            metrics_dict: dict[str, Any] = result[category]
            for metric_key in list(metrics_dict.keys()):
                if not isinstance(metric_key, str):
                    continue
                if metric_key.endswith(("_null", "_excess", "_std", "_normalized")):
                    continue
                if f"{metric_key}_null" in metrics_dict:
                    continue

                parsed = _parse_mi_key_for_null_baseline(metric_key)
                if parsed is None:
                    continue
                x_vars, y_var, z_vars = parsed

                x_parts = [_var_data(t) for t in x_vars]
                if any(p is None for p in x_parts):
                    continue
                x = np.hstack([p for p in x_parts if p is not None])

                y_arr = _var_data(y_var)
                if y_arr is None:
                    continue

                # Unconditional: I(X;Y)
                if not z_vars:
                    null_mean, null_std = estimator.mutual_information_null(
                        x,
                        y_arr,
                        n_shuffles=n_shuf,
                        seed=seed_base + seed_off,
                    )
                    metrics_dict[f"{metric_key}_null"] = float(null_mean)
                    metrics_dict[f"{metric_key}_null_std"] = float(null_std)
                    observed = metrics_dict.get(metric_key)
                    if isinstance(observed, (int, float, np.number)):
                        metrics_dict[f"{metric_key}_excess"] = float(
                            observed - null_mean
                        )
                    seed_off += 1
                    continue

                # Conditional: I(X;Y|Z)
                z_parts = [_var_data(t) for t in z_vars]
                if any(p is None for p in z_parts):
                    continue
                z = np.hstack([p for p in z_parts if p is not None])

                stratify_by_c = len(z_vars) == 1 and z_vars[0].split("_", 1)[0] == "C"
                if perms is None:
                    rng = np.random.default_rng(seed_base + 10_000)
                    perms = [rng.permutation(C_vec.shape[0]) for _ in range(n_shuf)]

                vals: list[float] = []
                for perm in perms:
                    if stratify_by_c:
                        # Permute Y within each class label to preserve Y|C.
                        y_perm = np.asarray(y_arr).copy()
                        for c_val in np.unique(C_vec):
                            idx = np.flatnonzero(C_vec == c_val)
                            if idx.size <= 1:
                                continue
                            idx_perm = rng.permutation(idx)
                            y_perm[idx] = y_arr[idx_perm]
                    else:
                        y_perm = y_arr[perm]
                    vals.append(
                        float(estimator.conditional_mutual_information(x, y_perm, z))
                    )
                metrics_dict[f"{metric_key}_null"] = float(np.mean(vals))
                metrics_dict[f"{metric_key}_null_std"] = float(np.std(vals))
                observed = metrics_dict.get(metric_key)
                if isinstance(observed, (int, float, np.number)):
                    metrics_dict[f"{metric_key}_excess"] = float(
                        observed - np.mean(vals)
                    )


__all__ = [
    "InformationNullBaselineMixin",
    "_metric_variable_for_null_baseline",
    "_parse_mi_key_for_null_baseline",
]
