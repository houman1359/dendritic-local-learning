"""Boolean teachers and diagnostics; no production architecture claims.

Teacher signs/supports are paired across rho. Model contacts use a separate seed.
Two-panel projection statistics require independent evaluation panels, sampled
after freezing the predictor. Their standard errors condition on that predictor.
"""
import math

import numpy as np


def target_packet(d=64, rho=0.5, seed=61001, k_min=2, terms_per_order=8):
    if not (1 <= k_min <= d <= 32767 and 0 < rho < 1 and terms_per_order >= 1):
        raise ValueError("Invalid dimension, spectrum, or term count")
    masks, coefficients, orders, energy = [], [], [], []
    for k in range(k_min, d + 1):
        e = (1 - rho) * rho ** (k - k_min) / (1 - rho ** (d - k_min + 1))
        rng = np.random.default_rng(np.random.SeedSequence([seed, d, k, 381]))
        m = min(terms_per_order, math.comb(d, k))
        supports = set()
        while len(supports) < m:
            supports.add(tuple(sorted(rng.choice(d, k, replace=False).tolist())))
        for subset in sorted(supports):
            mask = np.zeros(d, dtype=np.int16)
            mask[list(subset)] = 1
            masks.append(mask)
            coefficients.append(float(rng.choice([-1, 1])) * math.sqrt(e / m))
            orders.append(k)
        energy.append(e)
    return {"masks": np.stack(masks), "coefficients": np.array(coefficients),
            "orders": np.array(orders), "order_energy": np.array(energy),
            "d": d, "rho": rho, "seed": seed, "k_min": k_min}


def characters(x, packet):
    x = np.asarray(x)
    if x.ndim != 2 or x.shape[1] != packet["d"] or not np.isin(x, [-1, 1]).all():
        raise ValueError("Expected a matrix of raw Rademacher coordinates")
    if packet["d"] > 32767:
        raise ValueError("Parity accumulator width exceeded")
    parity = ((1 - x).astype(np.int16) // 2) @ packet["masks"].T
    return (1 - 2 * (parity % 2)).astype(np.float64)


def labels(x, packet):
    return characters(x, packet) @ packet["coefficients"]


def sample_raw_supports(d, s, count, seed):
    """IID s-subsets; increasing count preserves the same branch prefix."""
    if not (1 <= s <= d and count >= 0):
        raise ValueError("Invalid raw support dimensions")
    rng = np.random.default_rng(np.random.SeedSequence([seed, d, s, 991]))
    return np.array([np.sort(rng.choice(d, s, replace=False)) for _ in range(count)],
                    dtype=np.int64).reshape(count, s)


def sign_split_contacts(raw, d):
    """Both sign channels per raw coordinate; not an E/I bank allocation policy."""
    raw = np.asarray(raw)
    if (raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer)
            or np.any(raw < 0) or np.any(raw >= d)
            or any(len(set(row)) != len(row) for row in raw)):
        raise ValueError("Expected distinct valid raw coordinates in every support")
    return np.concatenate([raw, raw + d], axis=1)


def coverage_probability(d, s, k):
    if not (0 <= s <= d and 0 <= k <= d):
        raise ValueError("Invalid coverage dimensions")
    return math.comb(s, k) / math.comb(d, k) if k <= s else 0.0


def uncovered_probability(q, count):
    """P(no hit), including zero features and certain coverage without NaNs."""
    q, count = np.broadcast_arrays(np.asarray(q, float), np.asarray(count, float))
    if np.any((q < 0) | (q > 1)) or np.any(count < 0) or np.any(count != np.floor(count)):
        raise ValueError("Probabilities and integer counts required")
    out = np.ones(q.shape)
    active = count > 0
    with np.errstate(divide="ignore"):
        out[active] = np.exp(count[active] * np.log1p(-q[active]))
    return out


def coverage_oracle(branches, packet, s):
    q = np.array([coverage_probability(packet["d"], s, k)
                  for k in range(packet["k_min"], packet["d"] + 1)])
    return uncovered_probability(q, np.asarray(branches)[..., None]) @ packet["order_energy"]


def realized_coverage_residual(raw_supports, packet):
    """Unrestricted additive-local oracle for these actual sampled supports."""
    supports = [set(row) for row in np.asarray(raw_supports)]
    uncovered = [not any(set(np.flatnonzero(mask)) <= r for r in supports)
                 for mask in packet["masks"]]
    return float(np.square(packet["coefficients"])[uncovered].sum())


def schedule_multiplier(step, horizon, warmup_fraction=0.05, min_ratio=0.05):
    """Fresh-run schedule, one-indexed updates, warmup then cosine to min_ratio."""
    if not (horizon >= 2 and 1 <= step <= horizon and 0 < warmup_fraction < 1
            and 0 <= min_ratio <= 1):
        raise ValueError("Invalid schedule")
    warmup = min(horizon - 1, math.ceil(warmup_fraction * horizon))
    if step <= warmup:
        return step / warmup
    phase = (step - warmup) / (horizon - warmup)
    return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * phase)) / 2


def order_diagnostics(packet, x_a, pred_a, x_b, pred_b, blocks=16):
    """Unbiased products from independent panels, with paired-block MC errors.

    Teacher-aligned energy is not total degree energy. Recovery means reduction
    in target-coefficient squared error. Do not clip negative estimates. Blocks
    must contain independent draws; exhaustive-cube tests are deterministic.
    """
    phi_a, phi_b = characters(x_a, packet), characters(x_b, packet)
    pa, pb = np.asarray(pred_a, float), np.asarray(pred_b, float)
    n = len(phi_a)
    if (blocks < 2 or n < blocks or n % blocks or len(phi_b) != n
            or pa.shape != (n,) or pb.shape != (n,)
            or not np.isfinite(pa).all() or not np.isfinite(pb).all()):
        raise ValueError("Equal finite panels divisible into at least two blocks required")
    theta = packet["coefficients"]
    a = (pa[:, None] * phi_a).reshape(blocks, n // blocks, -1).mean(axis=1)
    b = (pb[:, None] * phi_b).reshape(blocks, n // blocks, -1).mean(axis=1)
    err, energy = (a - theta) * (b - theta), a * b

    def estimate(values):
        return {"estimate": float(np.mean(values)),
                "mc_se": float(np.std(values, ddof=1) / math.sqrt(blocks))}

    def panel_mean(va, vb):
        return (va.reshape(blocks, -1).mean(axis=1)
                + vb.reshape(blocks, -1).mean(axis=1)) / 2

    ya, yb = phi_a @ theta, phi_b @ theta
    risk = panel_mean((pa - ya) ** 2, (pb - yb) ** 2)
    nuisance = panel_mean(pa ** 2, pb ** 2) - energy.sum(axis=1)
    per_order = []
    for k in np.unique(packet["orders"]):
        mask = packet["orders"] == k
        e = float(np.square(theta[mask]).sum())
        per_order.append({"order": int(k), "target_energy": e,
                          "coefficient_error": estimate(err[:, mask].sum(axis=1)),
                          "recovered_error_reduction": estimate(e - err[:, mask].sum(axis=1)),
                          "predicted_teacher_span_energy": estimate(energy[:, mask].sum(axis=1)),
                          "target_alignment": estimate(((a[:, mask] + b[:, mask]) * theta[mask] / 2).sum(axis=1))})
    return {"scope": "Teacher-aligned subspace; MC uncertainty conditional on a fixed predictor",
            "draws_per_panel": n, "blocks": blocks, "orders": per_order,
            "coefficients": [dict(index=j, order=int(packet["orders"][j]),
                                  target=float(theta[j]), **estimate((a[:, j] + b[:, j]) / 2))
                             for j in range(len(theta))],
            "mse": estimate(risk), "off_teacher_span_energy": estimate(nuisance),
            "decomposition_residual": estimate(risk - err.sum(axis=1) - nuisance)}
