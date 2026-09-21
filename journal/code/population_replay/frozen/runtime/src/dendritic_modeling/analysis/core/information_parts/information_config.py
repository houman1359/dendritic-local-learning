"""Configuration parsing helpers for information analysis."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _InformationComponentSelection:
    """Boolean analysis-component switches parsed from selection.components."""

    compute_basic_mi: bool
    compute_pairwise_mi: bool
    compute_conditional_mi: bool
    compute_pid: bool
    compute_gaussian_fisher: bool
    compute_ablation_aligned_mi: bool
    compute_layer_total_proxies: bool
    compute_upstream_unique_cmi: bool
    compute_soma_coupling_mi: bool


@dataclass(frozen=True)
class _InformationScopeSelection:
    """Boolean analysis-scope switches parsed from selection.scopes/views."""

    per_neuron_analysis: bool
    per_einet_analysis: bool
    per_layer_analysis: bool


@dataclass(frozen=True)
class _InformationSignalVariantSelection:
    """LDA and random-weight signal settings parsed from grouped/legacy config."""

    compute_lda_weights: bool
    lda_n_shuffles: int
    random_weights_seed: int


@dataclass(frozen=True)
class _InformationLabelShuffleNullConfig:
    """Label-shuffle null baseline settings parsed from grouped/legacy config."""

    mi_null_shuffles: int
    mi_null_seed: int
    mi_null_components: set[str]


@dataclass(frozen=True)
class _InformationPidConfig:
    """PID preprocessing settings parsed from grouped/legacy config."""

    pid_binarize: bool
    pid_binarize_method: str
    pid_binarize_threshold: float


@dataclass(frozen=True)
class _InformationGaussianFisherConfig:
    """Gaussian/Fisher proxy settings parsed from grouped config."""

    gaussian_fisher_aggregation: str
    gaussian_fisher_mi_transform: str
    gaussian_fisher_eps: float


@dataclass(frozen=True)
class _InformationAblationAlignedConfig:
    """Ablation-aligned metric defaults parsed from legacy/grouped config."""

    compute_ablation_aligned_mi: bool
    compute_upstream_unique_cmi: bool
    compute_layer_total_proxies: bool
    layer_total_topk: int


@dataclass(frozen=True)
class _InformationRuntimeConfig:
    """Runtime/reporting settings parsed from grouped/legacy config."""

    include_vinf: bool
    normalize_mi: bool
    output_units: str
    analysis_split: str
    verbose: bool
    invalid_output_units: str | None


@dataclass(frozen=True)
class _InformationGranularityConfig:
    """Granularity and enhanced-view settings parsed from grouped/legacy config."""

    computation_level: str
    branch_aggregation: str
    branch_sample_count_per_layer: int | None
    branch_sample_seed: int
    branch_sample_strategy: str
    per_neuron_analysis: bool
    per_einet_analysis: bool
    per_layer_analysis: bool


@dataclass(frozen=True)
class _InformationEstimatorCoreConfig:
    """Estimator core settings parsed from grouped/legacy config."""

    method: str
    max_samples: Any
    n_neighbors: int
    n_bins: int
    copula_type: str


def _get_config_value(obj: Any, key: str, default: Any = None) -> Any:
    """Get a config value via attribute or mapping access."""
    if obj is None:
        return default
    try:
        val = getattr(obj, key)
    except Exception:
        try:
            val = obj.get(key, default)  # type: ignore[attr-defined]
        except Exception:
            return default
    return default if val is None else val


def _get_config_path(
    obj: Any,
    path: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Get a nested config value with the same None-as-missing semantics."""
    cur = obj
    for key in path:
        cur = _get_config_value(cur, key, None)
        if cur is None:
            return default
    return cur


def _get_grouped_or_legacy_value(
    *,
    grouped_cfg: Any,
    grouped_key: str,
    params: Any,
    legacy_key: str,
    default: Any = None,
) -> Any:
    """Get a grouped config value, falling back to its legacy flat key."""
    return _get_config_value(
        grouped_cfg,
        grouped_key,
        _get_config_value(params, legacy_key, default),
    )


def _get_grouped_path_or_legacy_value(
    *,
    grouped_cfg: Any,
    grouped_path: tuple[str, ...],
    params: Any,
    legacy_key: str,
    default: Any = None,
) -> Any:
    """Get a nested grouped config value, falling back to its legacy flat key."""
    return _get_config_path(
        grouped_cfg,
        grouped_path,
        _get_config_value(params, legacy_key, default),
    )


def _get_lower_config_value(obj: Any, key: str, default: Any) -> str:
    """Get a config value using legacy stripped-lowercase normalization."""
    return str(_get_config_value(obj, key, default)).strip().lower()


def _normalize_config_token(value: Any) -> str:
    """Normalize config selector tokens for legacy and grouped configs."""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_config_tokens(values: Sequence[Any]) -> set[str]:
    """Normalize non-blank config selector tokens into a set."""
    return {_normalize_config_token(value) for value in values if str(value).strip()}


def _has_any_config_token(tokens: set[str], aliases: set[str]) -> bool:
    """Return whether normalized tokens include any accepted aliases."""
    return bool(aliases.intersection(tokens))


def _parse_information_component_selection(
    components_raw: Sequence[Any],
) -> _InformationComponentSelection | None:
    """Parse selection.components into analysis component switches."""
    if not components_raw:
        return None

    components = _normalize_config_tokens(components_raw)
    return _InformationComponentSelection(
        compute_basic_mi=_has_any_config_token(components, {"basic", "basic_mi"}),
        compute_pairwise_mi=_has_any_config_token(
            components, {"pairwise", "pairwise_mi"}
        ),
        compute_conditional_mi=_has_any_config_token(
            components, {"conditional", "conditional_mi"}
        ),
        compute_pid="pid" in components,
        compute_gaussian_fisher=_has_any_config_token(
            components, {"gaussian_fisher", "gaussianfisher", "fisher"}
        ),
        compute_ablation_aligned_mi=_has_any_config_token(
            components, {"ablation_aligned", "ablation"}
        ),
        compute_layer_total_proxies=_has_any_config_token(
            components,
            {"layer_proxies", "layer_total_proxies", "layer_total"},
        ),
        compute_upstream_unique_cmi=_has_any_config_token(
            components, {"upstream_unique_cmi", "upstream_unique"}
        ),
        compute_soma_coupling_mi=_has_any_config_token(
            components,
            {
                "soma_coupling",
                "soma_coupling_mi",
                "branch_soma",
                "parent_soma",
            },
        ),
    )


def _parse_information_scope_selection(
    scopes_raw: Sequence[Any],
) -> _InformationScopeSelection | None:
    """Parse selection.scopes/views into enhanced analysis switches."""
    if not scopes_raw:
        return None

    scopes = _normalize_config_tokens(scopes_raw)
    return _InformationScopeSelection(
        per_neuron_analysis=_has_any_config_token(scopes, {"neuron", "per_neuron"}),
        per_einet_analysis=_has_any_config_token(scopes, {"einet", "per_einet"}),
        per_layer_analysis=_has_any_config_token(scopes, {"layer", "per_layer"}),
    )


def _build_information_signal_variant_selection(
    *,
    params: Any,
    selection_cfg: Any,
    compute_cfg: Any,
    baselines_cfg: Any,
) -> _InformationSignalVariantSelection:
    """Build LDA/random-weight signal settings from grouped or legacy config."""
    compute_lda_weights = bool(
        _get_grouped_or_legacy_value(
            grouped_cfg=compute_cfg,
            grouped_key="compute_lda_weights",
            params=params,
            legacy_key="compute_lda_weights",
            default=False,
        )
    )
    lda_n_shuffles = int(
        _get_config_path(
            baselines_cfg,
            ("random_weights", "n_shuffles"),
            _get_config_path(
                baselines_cfg,
                ("signal_shuffle", "n_shuffles"),
                _get_config_value(params, "lda_n_shuffles", 0),
            ),
        )
        or 0
    )
    random_weights_seed = int(
        _get_config_path(
            baselines_cfg,
            ("random_weights", "seed"),
            _get_config_path(
                baselines_cfg,
                ("signal_shuffle", "seed"),
                _get_config_value(params, "random_weights_seed", 0),
            ),
        )
        or 0
    )

    variants_raw = list(
        _get_grouped_or_legacy_value(
            grouped_cfg=selection_cfg,
            grouped_key="signal_variants",
            params=params,
            legacy_key="signal_variants",
            default=[],
        )
        or []
    )
    if variants_raw:
        variants = _normalize_config_tokens(variants_raw)
        want_lda = "lda" in variants
        want_random = _has_any_config_token(variants, {"random", "shuffle", "shuffled"})
        compute_lda_weights = bool(want_lda or want_random)
        lda_n_shuffles = lda_n_shuffles if want_random else 0

    return _InformationSignalVariantSelection(
        compute_lda_weights=compute_lda_weights,
        lda_n_shuffles=lda_n_shuffles,
        random_weights_seed=random_weights_seed,
    )


def _build_information_label_shuffle_null_config(
    *,
    params: Any,
    baselines_cfg: Any,
) -> _InformationLabelShuffleNullConfig:
    """Build label-shuffle null baseline settings from grouped or legacy config."""
    mi_null_shuffles = int(
        _get_grouped_path_or_legacy_value(
            grouped_cfg=baselines_cfg,
            grouped_path=("label_shuffle_null", "n_shuffles"),
            params=params,
            legacy_key="mi_null_shuffles",
            default=0,
        )
        or 0
    )
    mi_null_seed = int(
        _get_grouped_path_or_legacy_value(
            grouped_cfg=baselines_cfg,
            grouped_path=("label_shuffle_null", "seed"),
            params=params,
            legacy_key="mi_null_seed",
            default=0,
        )
        or 0
    )
    null_components_raw = list(
        _get_grouped_path_or_legacy_value(
            grouped_cfg=baselines_cfg,
            grouped_path=("label_shuffle_null", "components"),
            params=params,
            legacy_key="mi_null_components",
            default=["basic"],
        )
        or ["basic"]
    )
    mi_null_components = _normalize_config_tokens(null_components_raw) or {"basic"}

    return _InformationLabelShuffleNullConfig(
        mi_null_shuffles=mi_null_shuffles,
        mi_null_seed=mi_null_seed,
        mi_null_components=mi_null_components,
    )


def _build_information_pid_config(
    *, params: Any, pid_cfg: Any
) -> _InformationPidConfig:
    """Build PID preprocessing settings from grouped or legacy config."""
    return _InformationPidConfig(
        pid_binarize=bool(
            _get_grouped_or_legacy_value(
                grouped_cfg=pid_cfg,
                grouped_key="binarize",
                params=params,
                legacy_key="pid_binarize",
                default=False,
            )
        ),
        pid_binarize_method=str(
            _get_grouped_or_legacy_value(
                grouped_cfg=pid_cfg,
                grouped_key="binarize_method",
                params=params,
                legacy_key="pid_binarize_method",
                default="median",
            )
        ),
        pid_binarize_threshold=float(
            _get_grouped_or_legacy_value(
                grouped_cfg=pid_cfg,
                grouped_key="binarize_threshold",
                params=params,
                legacy_key="pid_binarize_threshold",
                default=0.5,
            )
        ),
    )


def _build_information_gaussian_fisher_config(
    *, gaussian_fisher_cfg: Any
) -> _InformationGaussianFisherConfig:
    """Build Gaussian/Fisher proxy settings from grouped config."""
    return _InformationGaussianFisherConfig(
        gaussian_fisher_aggregation=_get_lower_config_value(
            gaussian_fisher_cfg, "aggregation", "sum"
        ),
        gaussian_fisher_mi_transform=_get_lower_config_value(
            gaussian_fisher_cfg, "mi_transform", "half_log1p"
        ),
        gaussian_fisher_eps=float(
            _get_config_value(
                gaussian_fisher_cfg,
                "ridge",
                _get_config_value(gaussian_fisher_cfg, "eps", 1e-8),
            )
        ),
    )


def _build_information_ablation_aligned_config(
    *,
    params: Any,
    compute_cfg: Any,
) -> _InformationAblationAlignedConfig:
    """Build ablation-aligned metric defaults from legacy/grouped config."""
    return _InformationAblationAlignedConfig(
        compute_ablation_aligned_mi=bool(
            getattr(params, "compute_ablation_aligned_mi", True)
        ),
        compute_upstream_unique_cmi=bool(
            getattr(params, "compute_upstream_unique_cmi", False)
        ),
        compute_layer_total_proxies=bool(
            getattr(params, "compute_layer_total_proxies", True)
        ),
        layer_total_topk=int(
            _get_config_value(
                compute_cfg,
                "layer_total_topk",
                _get_config_value(params, "layer_total_topk", 10),
            )
        ),
    )


def _build_information_runtime_config(
    *,
    params: Any,
    compute_cfg: Any,
) -> _InformationRuntimeConfig:
    """Build runtime/reporting settings from grouped or legacy config."""
    output_units = _normalize_config_token(
        _get_grouped_or_legacy_value(
            grouped_cfg=compute_cfg,
            grouped_key="output_units",
            params=params,
            legacy_key="output_units",
            default="bits",
        )
    )
    invalid_output_units = None
    if output_units not in {"bits", "bit", "nats", "nat"}:
        invalid_output_units = output_units
        output_units = "bits"
    output_units = "bits" if output_units in {"bits", "bit"} else "nats"

    analysis_split = _normalize_config_token(
        _get_grouped_or_legacy_value(
            grouped_cfg=compute_cfg,
            grouped_key="analysis_split",
            params=params,
            legacy_key="analysis_split",
            default="test",
        )
    )
    if analysis_split == "valid":
        analysis_split = "validation"
    if analysis_split not in {"validation", "test"}:
        raise ValueError("information analysis_split must be 'validation' or 'test'")

    return _InformationRuntimeConfig(
        include_vinf=bool(
            _get_grouped_or_legacy_value(
                grouped_cfg=compute_cfg,
                grouped_key="include_vinf",
                params=params,
                legacy_key="include_vinf",
                default=False,
            )
        ),
        normalize_mi=bool(
            _get_grouped_or_legacy_value(
                grouped_cfg=compute_cfg,
                grouped_key="normalize_mi",
                params=params,
                legacy_key="normalize_mi",
                default=False,
            )
        ),
        output_units=output_units,
        analysis_split=analysis_split,
        verbose=bool(
            _get_grouped_or_legacy_value(
                grouped_cfg=compute_cfg,
                grouped_key="verbose",
                params=params,
                legacy_key="verbose",
                default=False,
            )
        ),
        invalid_output_units=invalid_output_units,
    )


def _build_information_granularity_config(
    *,
    params: Any,
    compute_cfg: Any,
) -> _InformationGranularityConfig:
    """Build granularity and enhanced-view settings from grouped/legacy config."""
    return _InformationGranularityConfig(
        computation_level=str(
            _get_grouped_or_legacy_value(
                grouped_cfg=compute_cfg,
                grouped_key="computation_level",
                params=params,
                legacy_key="computation_level",
                default="single_branch",
            )
        ),
        branch_aggregation=str(
            _get_grouped_or_legacy_value(
                grouped_cfg=compute_cfg,
                grouped_key="branch_aggregation",
                params=params,
                legacy_key="branch_aggregation",
                default="mean",
            )
        ),
        branch_sample_count_per_layer=_get_grouped_or_legacy_value(
            grouped_cfg=compute_cfg,
            grouped_key="branch_sample_count_per_layer",
            params=params,
            legacy_key="branch_sample_count_per_layer",
            default=None,
        ),
        branch_sample_seed=int(
            _get_grouped_or_legacy_value(
                grouped_cfg=compute_cfg,
                grouped_key="branch_sample_seed",
                params=params,
                legacy_key="branch_sample_seed",
                default=0,
            )
            or 0
        ),
        branch_sample_strategy=str(
            _get_grouped_or_legacy_value(
                grouped_cfg=compute_cfg,
                grouped_key="branch_sample_strategy",
                params=params,
                legacy_key="branch_sample_strategy",
                default="uniform",
            )
        ),
        per_neuron_analysis=bool(
            _get_grouped_or_legacy_value(
                grouped_cfg=compute_cfg,
                grouped_key="per_neuron_analysis",
                params=params,
                legacy_key="per_neuron_analysis",
                default=False,
            )
        ),
        per_einet_analysis=bool(
            _get_grouped_or_legacy_value(
                grouped_cfg=compute_cfg,
                grouped_key="per_einet_analysis",
                params=params,
                legacy_key="per_einet_analysis",
                default=False,
            )
        ),
        per_layer_analysis=bool(
            _get_grouped_or_legacy_value(
                grouped_cfg=compute_cfg,
                grouped_key="per_layer_analysis",
                params=params,
                legacy_key="per_layer_analysis",
                default=False,
            )
        ),
    )


def _build_information_estimator_core_config(
    *,
    params: Any,
    estimator_cfg: Any,
    compute_cfg: Any,
) -> _InformationEstimatorCoreConfig:
    """Build estimator identity/sample/bin settings from grouped or legacy config."""
    return _InformationEstimatorCoreConfig(
        method=str(
            _get_grouped_or_legacy_value(
                grouped_cfg=estimator_cfg,
                grouped_key="method",
                params=params,
                legacy_key="method",
                default="kraskov",
            )
        ),
        max_samples=_get_grouped_or_legacy_value(
            grouped_cfg=compute_cfg,
            grouped_key="max_samples",
            params=params,
            legacy_key="max_samples",
            default=None,
        ),
        n_neighbors=int(
            _get_grouped_path_or_legacy_value(
                grouped_cfg=estimator_cfg,
                grouped_path=("kraskov", "n_neighbors"),
                params=params,
                legacy_key="n_neighbors",
                default=10,
            )
        ),
        n_bins=int(
            _get_grouped_path_or_legacy_value(
                grouped_cfg=estimator_cfg,
                grouped_path=("binned", "n_bins"),
                params=params,
                legacy_key="n_bins",
                default=20,
            )
        ),
        copula_type=str(
            _get_grouped_path_or_legacy_value(
                grouped_cfg=estimator_cfg,
                grouped_path=("copula", "type"),
                params=params,
                legacy_key="copula_type",
                default="gaussian",
            )
        ),
    )


def _build_information_estimator_method_params(
    *,
    params: Any,
    estimator_cfg: Any,
    method: str,
) -> dict[str, Any]:
    """Build estimator-specific method parameters from grouped or legacy config."""
    decoder_cfg = _get_config_value(estimator_cfg, "decoder", None)
    decoder_cont_cfg = _get_config_value(decoder_cfg, "continuous", None)
    binned_cfg = _get_config_value(estimator_cfg, "binned", None)
    normalized_method = _normalize_config_token(method)

    if normalized_method == "decoder":
        return {
            "cv_folds": int(
                _get_grouped_or_legacy_value(
                    grouped_cfg=decoder_cfg,
                    grouped_key="cv_folds",
                    params=params,
                    legacy_key="decoder_cv_folds",
                    default=5,
                )
            ),
            "C": float(
                _get_grouped_or_legacy_value(
                    grouped_cfg=decoder_cfg,
                    grouped_key="C",
                    params=params,
                    legacy_key="decoder_C",
                    default=1.0,
                )
            ),
            "standardize": bool(
                _get_grouped_or_legacy_value(
                    grouped_cfg=decoder_cfg,
                    grouped_key="standardize",
                    params=params,
                    legacy_key="decoder_standardize",
                    default=True,
                )
            ),
            "seed": int(
                _get_grouped_or_legacy_value(
                    grouped_cfg=decoder_cfg,
                    grouped_key="seed",
                    params=params,
                    legacy_key="decoder_seed",
                    default=0,
                )
            ),
            "continuous_strategy": str(
                _get_grouped_or_legacy_value(
                    grouped_cfg=decoder_cont_cfg,
                    grouped_key="strategy",
                    params=params,
                    legacy_key="decoder_continuous_strategy",
                    default="none",
                )
            ),
            "continuous_max_dim": int(
                _get_grouped_or_legacy_value(
                    grouped_cfg=decoder_cont_cfg,
                    grouped_key="max_total_dim",
                    params=params,
                    legacy_key="decoder_continuous_max_total_dim",
                    default=2,
                )
            ),
            "gaussian_ridge": float(
                _get_grouped_or_legacy_value(
                    grouped_cfg=decoder_cont_cfg,
                    grouped_key="gaussian_ridge",
                    params=params,
                    legacy_key="decoder_gaussian_ridge",
                    default=1e-6,
                )
            ),
            "binning_strategy": str(
                _get_grouped_or_legacy_value(
                    grouped_cfg=binned_cfg,
                    grouped_key="binning_strategy",
                    params=params,
                    legacy_key="binning_strategy",
                    default="quantile",
                )
            ),
            "bias_correction": str(
                _get_grouped_or_legacy_value(
                    grouped_cfg=binned_cfg,
                    grouped_key="bias_correction",
                    params=params,
                    legacy_key="binning_bias_correction",
                    default="none",
                )
            ),
        }

    if normalized_method == "binned":
        return {
            "binning_strategy": str(
                _get_grouped_or_legacy_value(
                    grouped_cfg=binned_cfg,
                    grouped_key="binning_strategy",
                    params=params,
                    legacy_key="binning_strategy",
                    default="quantile",
                )
            ),
            "bias_correction": str(
                _get_grouped_or_legacy_value(
                    grouped_cfg=binned_cfg,
                    grouped_key="bias_correction",
                    params=params,
                    legacy_key="binning_bias_correction",
                    default="none",
                )
            ),
        }

    return {}
