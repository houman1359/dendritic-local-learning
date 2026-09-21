"""Compatibility exports for dendritic branch-layer initialization.

Implementation lives in purpose-specific modules:
``initialize_expectations``, ``initialize_reactivation``, ``initialize_scaling``,
and ``initialize_methods``. This facade preserves the historical import path.
"""

from dendritic_modeling.config.reactivation import (
    collect_model_data_driven_reactivation_policies,
    is_data_driven_reactivation_policy,
    normalize_reactivation_init_policy,
    reactivation_policy_to_calibration_mode,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    initialize_expectations as _expectations_impl,
    initialize_reactivation as _reactivation_impl,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_expectations import (
    ADAPTIVE_INITIALIZATION_POLICY_ALIASES,
    MAX_CALIBRATION_QUANTILE_SAMPLES,
    _aggregate_quantiles_from_chunks,
    _compute_expectation_truncated_relu,
    _compute_expectation_truncated_softplus,
    _expected_weight_for_transform,
    _mean_for_expected_weight,
    _safe_quantile,
    _solve_first_crossing_expected_weight_mean,
    _truncated_normal_moments,
    compute_expectation_truncated_inverse_softplus_normal,
    compute_expectation_truncated_log_normal,
    normalize_adaptive_initialization_policy,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_methods import (
    analytical_expectation_dbl_init,
    default_dbl_init,
    ei_equivalence_dbl_init,
    identity_weighttransform_dbl_init,
    mechanism_neutral_dbl_init,
    naive_dbl_init,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_reactivation import (
    _apply_reactivation_calibration,
    _apply_reactivation_init,
    _current_reactivation_state,
    _reactivation_calibration_safeguards,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_scaling import (
    _center_preserving_shunting_conductances,
    compute_network_scale_factor,
)


def calibrate_reactivation_from_data(*args, **kwargs):
    """Compatibility wrapper for data-driven reactivation calibration.

    Historically this function and its helper constants lived in this module,
    so tests and downstream code may monkeypatch ``initialize.MAX_*`` or
    ``initialize._aggregate_quantiles_from_chunks``. Propagate those facade
    values before dispatching to the implementation module.
    """

    _expectations_impl.MAX_CALIBRATION_QUANTILE_SAMPLES = (
        MAX_CALIBRATION_QUANTILE_SAMPLES
    )
    _reactivation_impl._aggregate_quantiles_from_chunks = (
        _aggregate_quantiles_from_chunks
    )
    return _reactivation_impl.calibrate_reactivation_from_data(*args, **kwargs)


def calibrate_reactivation_empirical(*args, **kwargs):
    """Backward-compatible alias for :func:`calibrate_reactivation_from_data`."""

    return calibrate_reactivation_from_data(*args, **kwargs)


__all__ = [
    "ADAPTIVE_INITIALIZATION_POLICY_ALIASES",
    "MAX_CALIBRATION_QUANTILE_SAMPLES",
    "_aggregate_quantiles_from_chunks",
    "_apply_reactivation_calibration",
    "_apply_reactivation_init",
    "_center_preserving_shunting_conductances",
    "_compute_expectation_truncated_relu",
    "_compute_expectation_truncated_softplus",
    "_current_reactivation_state",
    "_expected_weight_for_transform",
    "_mean_for_expected_weight",
    "_reactivation_calibration_safeguards",
    "_safe_quantile",
    "_solve_first_crossing_expected_weight_mean",
    "_truncated_normal_moments",
    "analytical_expectation_dbl_init",
    "calibrate_reactivation_empirical",
    "calibrate_reactivation_from_data",
    "collect_model_data_driven_reactivation_policies",
    "compute_expectation_truncated_inverse_softplus_normal",
    "compute_expectation_truncated_log_normal",
    "compute_network_scale_factor",
    "default_dbl_init",
    "ei_equivalence_dbl_init",
    "identity_weighttransform_dbl_init",
    "is_data_driven_reactivation_policy",
    "mechanism_neutral_dbl_init",
    "naive_dbl_init",
    "normalize_adaptive_initialization_policy",
    "normalize_reactivation_init_policy",
    "reactivation_policy_to_calibration_mode",
]
