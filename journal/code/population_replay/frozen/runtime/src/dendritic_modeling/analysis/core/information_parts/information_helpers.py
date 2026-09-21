"""Pure helper functions for information analysis."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D."""

    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def is_discrete_class_labels(labels: np.ndarray) -> bool:
    """Heuristic: decide whether labels are discrete class IDs."""

    labels = np.asarray(labels).ravel()
    if labels.size == 0:
        return False
    unique = np.unique(labels)
    # Typical classification: small number of unique values
    if unique.size <= 50:
        return True
    # If labels are integer-like and the unique-label count is still smallish.
    if unique.size <= 200 and np.allclose(unique, np.round(unique), atol=1e-6):
        return True
    return False


def discrete_entropy_nats(labels: np.ndarray) -> float:
    """Discrete entropy H(C) in natural units (nats)."""

    labels = np.asarray(labels).ravel()
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts.astype(float) / float(counts.sum())
    # Natural log: matches the Kraskov MI units (nats)
    return float(-np.sum(p * np.log(p + 1e-12)))


def metric_targets_class(metric_key: str) -> bool:
    """Return True if the metric is of the form I(X;C) or I(X;C|...)."""

    if (
        not isinstance(metric_key, str)
        or "I(" not in metric_key
        or ")" not in metric_key
    ):
        return False
    lpar = metric_key.find("I(")
    lpar = metric_key.find("(", lpar)
    rpar = metric_key.find(")", lpar)
    if lpar == -1 or rpar == -1 or rpar <= lpar + 1:
        return False
    inside = metric_key[lpar + 1 : rpar]
    main = inside.split("|", 1)[0]
    parts = [p.strip() for p in main.split(";")]
    if len(parts) < 2:
        return False
    return parts[1] == "C"


def extract_metric_tokens(metric_key: str) -> list[str]:
    """Extract variable tokens from metric keys like 'I(E;Vb|C)'."""

    if not isinstance(metric_key, str):
        return []
    if "I(" not in metric_key or ")" not in metric_key:
        return []

    lpar = metric_key.find("I(")
    lpar = metric_key.find("(", lpar)
    rpar = metric_key.find(")", lpar)
    if lpar == -1 or rpar == -1 or rpar <= lpar + 1:
        return []

    inside = metric_key[lpar + 1 : rpar]
    inside = inside.replace("|", ";").replace(",", ";")
    return [t.strip() for t in inside.split(";") if t.strip()]


def rename_metric_key_tokens(metric_key: str, token_map: dict[str, str]) -> str:
    """Rename variable tokens inside MI keys while preserving separators."""

    if not isinstance(metric_key, str):
        return metric_key
    if "I(" not in metric_key:
        return metric_key

    renamed = metric_key
    for src, dst in token_map.items():
        # Replace full-token occurrences inside the I( ... ) block. Tokens are
        # delimited by one of: (, ;, ,, |, )
        renamed = re.sub(
            rf"(?<=[(;,|]){re.escape(src)}(?=[,;|)])",
            dst,
            renamed,
        )
    return renamed


def keep_metric_key_given_availability(
    metric_key: str,
    *,
    has_exc_synapses: bool,
    has_inh_synapses: bool,
    has_branch_input: bool,
) -> bool:
    """Return True if the metric is meaningful given available variables."""

    tokens = extract_metric_tokens(metric_key)
    if not tokens:
        return True

    if (not has_exc_synapses) and any(t.startswith("E") for t in tokens):
        return False
    if (not has_inh_synapses) and any(t.startswith("I") for t in tokens):
        return False
    if (not has_branch_input) and any(t.startswith("Vb") for t in tokens):
        return False

    return True


def should_convert_information_key_to_bits(key: Any) -> bool:
    """Heuristic: return True if `key` is an MI/CMI-like metric name in nats."""

    if not isinstance(key, str):
        return False
    if "I(" not in key:
        return False
    # Dimensionless ratios; do not convert.
    if "_normalized" in key:
        return False
    # Fisher discriminability proxies (d'^2-like); not information units.
    if "_fisher" in key:
        return False
    # Counts used for proxy bookkeeping.
    if key.endswith("_n"):
        return False
    return True


def convert_information_units_in_place(obj: Any, *, output_units: str) -> None:
    """Convert MI/CMI-like quantities in-place from nats to bits."""

    if output_units != "bits":
        return

    nat_to_bit = 1.0 / float(np.log(2.0))
    nat_to_bit_sq = nat_to_bit * nat_to_bit

    def _convert_any(x: Any) -> None:
        if isinstance(x, dict):
            for k in list(x.keys()):
                v = x.get(k)
                if isinstance(v, (dict, list)):
                    _convert_any(v)
                elif isinstance(v, (int, float, np.floating)):
                    if not should_convert_information_key_to_bits(k):
                        continue
                    scale = (
                        nat_to_bit_sq
                        if isinstance(k, str) and k.endswith(("_var", "_variance"))
                        else nat_to_bit
                    )
                    x[k] = float(v) * scale
        elif isinstance(x, list):
            for item in x:
                _convert_any(item)

    _convert_any(obj)


def gaussian_fisher_proxy(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    eps: float = 1e-8,
    aggregation: str = "sum",
    mi_transform: str = "half_log1p",
) -> tuple[float, float]:
    """Return ``(F, I_proxy)`` for I(X;C) under a Gaussian/Fisher proxy."""
    agg = str(aggregation).strip().lower()
    mi_transform = str(mi_transform).strip().lower()

    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else:
        x = x.reshape(x.shape[0], -1)

    labels = np.asarray(labels).reshape(-1)
    uniq = np.unique(labels)
    if x.shape[0] <= 2 or uniq.size <= 1:
        return 0.0, 0.0

    if uniq.size == 2:
        c0, c1 = uniq[0], uniq[1]
        m0 = labels == c0
        m1 = labels == c1
        if np.sum(m0) <= 1 or np.sum(m1) <= 1:
            return 0.0, 0.0
        mu0 = np.mean(x[m0], axis=0)
        mu1 = np.mean(x[m1], axis=0)
        var0 = np.var(x[m0], axis=0)
        var1 = np.var(x[m1], axis=0)
        pooled_var = 0.5 * (var0 + var1)
        fisher = (mu1 - mu0) ** 2 / (pooled_var + eps)
    else:
        means = []
        variances = []
        counts = []
        for c in uniq:
            mc = labels == c
            n_c = int(np.sum(mc))
            if n_c <= 1:
                continue
            means.append(np.mean(x[mc], axis=0))
            variances.append(np.var(x[mc], axis=0))
            counts.append(n_c)
        if len(means) <= 1:
            return 0.0, 0.0
        means = np.stack(means, axis=0)
        variances = np.stack(variances, axis=0)
        weights = np.asarray(counts, dtype=float)
        weights /= max(float(weights.sum()), eps)
        grand_mean = np.average(means, axis=0, weights=weights)
        between_var = np.average((means - grand_mean) ** 2, axis=0, weights=weights)
        within_var = np.average(variances, axis=0, weights=weights)
        fisher = between_var / (within_var + eps)

    fisher = np.maximum(0.0, fisher)

    if agg == "mean":
        F = float(np.mean(fisher))
    elif agg == "max":
        F = float(np.max(fisher))
    else:
        F = float(np.sum(fisher))

    if mi_transform == "none":
        I_proxy = F
    elif mi_transform == "log1p":
        I_proxy = float(np.log1p(F))
    else:
        I_proxy = float(0.5 * np.log1p(F))

    return F, I_proxy


def fisher_mi_from_f(f: float, *, mi_transform: str = "half_log1p") -> float:
    """Map a Fisher/discriminability-like scalar to an MI-like proxy."""
    mi_transform = str(mi_transform).strip().lower()

    f = float(f)
    if not np.isfinite(f) or f <= 0.0:
        return 0.0

    if mi_transform == "none":
        return f
    if mi_transform == "log1p":
        return float(np.log1p(f))
    return float(0.5 * np.log1p(f))


def fisher_dprime2_binary(
    x: np.ndarray, labels: np.ndarray, *, eps: float = 1e-8
) -> float:
    """Binary-class Fisher discriminability d'^2 = Δμᵀ Σ^{-1} Δμ (ridge-regularized)."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else:
        x = x.reshape(x.shape[0], -1)

    labels = np.asarray(labels).reshape(-1)
    uniq = np.unique(labels)
    if uniq.size != 2 or x.shape[0] <= 2:
        return 0.0

    c0, c1 = uniq[0], uniq[1]
    m0 = labels == c0
    m1 = labels == c1
    if np.sum(m0) <= 1 or np.sum(m1) <= 1:
        return 0.0

    x0 = x[m0]
    x1 = x[m1]
    mu0 = np.mean(x0, axis=0)
    mu1 = np.mean(x1, axis=0)
    dmu = mu1 - mu0

    cov0 = np.atleast_2d(np.cov(x0, rowvar=False, bias=True))
    cov1 = np.atleast_2d(np.cov(x1, rowvar=False, bias=True))
    pooled = 0.5 * (cov0 + cov1)
    pooled = pooled + eps * np.eye(pooled.shape[0])

    try:
        sol = np.linalg.solve(pooled, dmu)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(pooled, dmu, rcond=None)[0]

    d2 = float(dmu @ sol)
    if not np.isfinite(d2):
        return 0.0
    return float(max(0.0, d2))


def fisher_dprime2_pairwise_avg(
    x: np.ndarray, labels: np.ndarray, *, eps: float = 1e-8
) -> float:
    """Multi-class surrogate: average binary d'^2 over all class pairs."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    else:
        x = x.reshape(x.shape[0], -1)

    labels = np.asarray(labels).reshape(-1)
    uniq = np.unique(labels)
    if uniq.size <= 1 or x.shape[0] <= 2:
        return 0.0
    if uniq.size == 2:
        return fisher_dprime2_binary(x, labels, eps=eps)

    d2s: list[float] = []
    for i in range(uniq.size):
        for j in range(i + 1, uniq.size):
            a, b = uniq[i], uniq[j]
            m = (labels == a) | (labels == b)
            if np.sum(m) <= 2:
                continue
            d2 = fisher_dprime2_binary(x[m], labels[m], eps=eps)
            if np.isfinite(d2):
                d2s.append(float(d2))

    if not d2s:
        return 0.0
    return float(np.mean(d2s))


__all__ = [
    "convert_information_units_in_place",
    "discrete_entropy_nats",
    "ensure_2d",
    "extract_metric_tokens",
    "fisher_dprime2_binary",
    "fisher_dprime2_pairwise_avg",
    "fisher_mi_from_f",
    "gaussian_fisher_proxy",
    "is_discrete_class_labels",
    "keep_metric_key_given_availability",
    "metric_targets_class",
    "rename_metric_key_tokens",
    "should_convert_information_key_to_bits",
]
