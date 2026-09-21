"""Shared reactivation initialization policy helpers."""

from __future__ import annotations

import warnings

# Canonical default reactivation-init policy for additive cores that omit an
# explicit ``reactivation.init_policy``. Additive voltage is fragile under the
# analytical/fixed gate (depth- and normalization-dependent), so the data-driven
# occupancy_quantile calibration is the safe default. Route additive-default
# assignments through this constant rather than hardcoding the string.
DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY = "occupancy_quantile"

REACTIVATION_INIT_POLICY_ALIASES = {
    "analytical": "analytical",
    "auto": "analytical",
    "fixed": "fixed",
    "manual": "fixed",
    "empirical": "empirical",
    # Backward-compatible alias. The canonical public name is
    # occupancy_quantile because the rule maps voltage quantiles to target
    # reactivation occupancies.
    "quantile": "occupancy_quantile",
    "occupancy_quantile": "occupancy_quantile",
    # Deterministic center/slope decomposition of the analytical and occupancy
    # policies. These use no fitted mixing coefficient.
    "analytical_slope_occupancy_center": "analytical_slope_occupancy_center",
    "occupancy_slope_analytical_center": "occupancy_slope_analytical_center",
}


def normalize_reactivation_init_policy(policy: str | None) -> str:
    """Return the canonical reactivation-init policy name."""

    key = "analytical" if policy is None else str(policy).strip().lower()
    if key == "quantile":
        warnings.warn(
            "reactivation init policy 'quantile' is deprecated; use "
            "'occupancy_quantile' for the canonical occupancy-calibration rule.",
            DeprecationWarning,
            stacklevel=2,
        )
    return REACTIVATION_INIT_POLICY_ALIASES.get(key, key)


def reactivation_policy_to_calibration_mode(policy: str | None) -> str | None:
    """Map a canonical init policy to its data-driven calibration mode."""

    canonical = normalize_reactivation_init_policy(policy)
    return {
        "empirical": "median_mad",
        "occupancy_quantile": "occupancy_quantile",
        "analytical_slope_occupancy_center": ("analytical_slope_occupancy_center"),
        "occupancy_slope_analytical_center": ("occupancy_slope_analytical_center"),
    }.get(canonical)


def is_data_driven_reactivation_policy(policy: str | None) -> bool:
    """Whether this init policy requires a post-build calibration pass."""

    return reactivation_policy_to_calibration_mode(policy) is not None


def warn_untrainable_init_integration_pairing(
    *,
    weight_transform: str | None,
    use_shunting: bool,
    additive_mode: str | None,
    reactivation_init_policy: str | None,
    reactivate: bool = True,
) -> str | None:
    """Warn when a config pairs an init policy with an integration rule that
    was measured NOT to train.

    Frozen-architecture matrix, Pythia-70M L2 budget, 2026-08-24: strict
    positive-weight cells under additive integration (raw or
    conductance-normalized) reached fit ONLY with the occupancy_quantile
    policy (0.62 valid vs ~0.99 no-fit for analytical and both hybrid
    policies); shunting cells trained identically under every policy. This
    guard encodes that measurement: it stays silent for shunting, signed
    weights, or the quantile pairing, and returns the warning text (also
    emitted via ``warnings.warn``) for the measured-untrainable pairings so
    deliberate off-recipe runs remain possible but loudly labeled.
    """

    positive_transform = str(weight_transform or "").lower() in ("softplus", "exp")
    if use_shunting and not positive_transform:
        # Hybrid feature matrix v1 (2026-08-24): all three signed shunting
        # arms peaked at step 50 and then diverged (best 0.71-0.97, final up
        # to 1.27) — divisive drive requires strictly positive conductances.
        message = (
            "Measured-unstable pairing: shunting integration on signed "
            "(identity-transform) weights diverged in every measured arm "
            "(2026-08-24 hybrid feature matrix); use raw additive "
            "integration for signed families, or a strict-positive family "
            "for shunting."
        )
        warnings.warn(message, UserWarning, stacklevel=2)
        return message
    if use_shunting or not reactivate:
        return None
    if not positive_transform:
        return None
    if str(additive_mode or "raw").lower() not in ("raw", "conductance_normalized"):
        return None
    policy = normalize_reactivation_init_policy(reactivation_init_policy)
    if policy == DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY:
        return None
    message = (
        "Measured-untrainable pairing: positive-weight additive/conductance "
        f"cells with reactivation_init_policy={policy!r} did not fit at the "
        "frozen Pythia-L2 budget (2026-08-24 init-integration matrix); use "
        f"{DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY!r} or shunting "
        "integration unless this run deliberately probes the pairing."
    )
    warnings.warn(message, UserWarning, stacklevel=2)
    return message


def collect_model_data_driven_reactivation_policies(model) -> tuple[set[str], bool]:
    """Return data-driven module policies and whether any module policy exists."""

    requested_policies: set[str] = set()
    saw_layer_policy = False
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return requested_policies, saw_layer_policy

    for module in modules():
        layer_policy = getattr(module, "reactivation_init_policy", None)
        if layer_policy is None:
            continue
        saw_layer_policy = True
        layer_policy = normalize_reactivation_init_policy(layer_policy)
        if is_data_driven_reactivation_policy(layer_policy):
            requested_policies.add(layer_policy)

    return requested_policies, saw_layer_policy


__all__ = [
    "DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY",
    "REACTIVATION_INIT_POLICY_ALIASES",
    "collect_model_data_driven_reactivation_policies",
    "is_data_driven_reactivation_policy",
    "normalize_reactivation_init_policy",
    "reactivation_policy_to_calibration_mode",
    "warn_untrainable_init_integration_pairing",
]
