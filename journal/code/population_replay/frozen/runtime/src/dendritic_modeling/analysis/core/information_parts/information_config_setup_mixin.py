"""Analyzer configuration setup helpers for information analysis."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.analysis.core.information_parts.information_config import (
    _build_information_ablation_aligned_config,
    _build_information_estimator_core_config,
    _build_information_gaussian_fisher_config,
    _build_information_granularity_config,
    _build_information_label_shuffle_null_config,
    _build_information_pid_config,
    _build_information_runtime_config,
    _build_information_signal_variant_selection,
    _get_config_value,
    _parse_information_component_selection,
    _parse_information_scope_selection,
)


class InformationConfigSetupMixin:
    """Assigns parsed information-analysis configuration groups."""

    def _configure_estimator_core(
        self,
        *,
        params: Any,
        estimator_cfg: Any,
        compute_cfg: Any,
    ) -> None:
        """Assign estimator method and core sampling parameters."""
        estimator_core_config = _build_information_estimator_core_config(
            params=params,
            estimator_cfg=estimator_cfg,
            compute_cfg=compute_cfg,
        )
        self.method = estimator_core_config.method
        self.max_samples = estimator_core_config.max_samples
        self.n_neighbors = estimator_core_config.n_neighbors
        self.n_bins = estimator_core_config.n_bins
        self.copula_type = estimator_core_config.copula_type

    def _configure_label_shuffle_null(
        self,
        *,
        params: Any,
        baselines_cfg: Any,
    ) -> None:
        """Assign label-shuffle null-baseline parameters."""
        label_shuffle_null = _build_information_label_shuffle_null_config(
            params=params,
            baselines_cfg=baselines_cfg,
        )
        self.mi_null_shuffles = label_shuffle_null.mi_null_shuffles
        self.mi_null_seed = label_shuffle_null.mi_null_seed
        self.mi_null_components = label_shuffle_null.mi_null_components

    def _configure_legacy_component_selection(self, *, params: Any) -> None:
        """Assign legacy flat compute_* component flags."""
        self.compute_basic_mi = bool(getattr(params, "compute_basic_mi", False))
        self.compute_pairwise_mi = bool(getattr(params, "compute_pairwise_mi", False))
        self.compute_conditional_mi = bool(
            getattr(params, "compute_conditional_mi", False)
        )
        self.compute_pid = bool(getattr(params, "compute_pid", False))
        self.compute_gaussian_fisher = False
        self.compute_soma_coupling_mi = False

    def _configure_pid_preprocessing(self, *, params: Any, pid_cfg: Any) -> None:
        """Assign PID preprocessing parameters."""
        pid_config = _build_information_pid_config(params=params, pid_cfg=pid_cfg)
        self.pid_binarize = pid_config.pid_binarize
        self.pid_binarize_method = pid_config.pid_binarize_method
        self.pid_binarize_threshold = pid_config.pid_binarize_threshold

    def _configure_ablation_aligned_metrics(
        self,
        *,
        params: Any,
        compute_cfg: Any,
    ) -> None:
        """Assign ablation-aligned metric parameters."""
        ablation_aligned_config = _build_information_ablation_aligned_config(
            params=params,
            compute_cfg=compute_cfg,
        )
        self.compute_ablation_aligned_mi = (
            ablation_aligned_config.compute_ablation_aligned_mi
        )
        self.compute_upstream_unique_cmi = (
            ablation_aligned_config.compute_upstream_unique_cmi
        )
        self.compute_layer_total_proxies = (
            ablation_aligned_config.compute_layer_total_proxies
        )
        self.layer_total_topk = ablation_aligned_config.layer_total_topk

    def _configure_component_selection_override(
        self,
        *,
        params: Any,
        selection_cfg: Any,
    ) -> None:
        """Apply selection.components overrides to legacy component flags."""
        components_raw = list(
            _get_config_value(
                selection_cfg,
                "components",
                _get_config_value(params, "components", []),
            )
            or []
        )
        component_selection = _parse_information_component_selection(components_raw)
        if component_selection is None:
            return

        self.compute_basic_mi = component_selection.compute_basic_mi
        self.compute_pairwise_mi = component_selection.compute_pairwise_mi
        self.compute_conditional_mi = component_selection.compute_conditional_mi
        self.compute_pid = component_selection.compute_pid
        self.compute_gaussian_fisher = component_selection.compute_gaussian_fisher
        self.compute_ablation_aligned_mi = (
            component_selection.compute_ablation_aligned_mi
        )
        self.compute_layer_total_proxies = (
            component_selection.compute_layer_total_proxies
        )
        self.compute_upstream_unique_cmi = (
            component_selection.compute_upstream_unique_cmi
        )
        self.compute_soma_coupling_mi = component_selection.compute_soma_coupling_mi

    def _disable_pid_if_requested(self) -> None:
        """Apply the current policy that PID requests are acknowledged but skipped."""
        if not self.compute_pid:
            return

        self.logger.warning(
            "PID requested via selection.components, but PID is currently disabled "
            "and will be skipped."
        )
        self.compute_pid = False

    def _configure_gaussian_fisher_metrics(self, *, gaussian_fisher_cfg: Any) -> None:
        """Assign Gaussian/Fisher proxy parameters."""
        gaussian_fisher_config = _build_information_gaussian_fisher_config(
            gaussian_fisher_cfg=gaussian_fisher_cfg
        )
        self.gaussian_fisher_aggregation = (
            gaussian_fisher_config.gaussian_fisher_aggregation
        )
        self.gaussian_fisher_mi_transform = (
            gaussian_fisher_config.gaussian_fisher_mi_transform
        )
        self.gaussian_fisher_eps = gaussian_fisher_config.gaussian_fisher_eps

    def _configure_runtime_options(self, *, params: Any, compute_cfg: Any) -> None:
        """Assign runtime/reporting options."""
        runtime_config = _build_information_runtime_config(
            params=params,
            compute_cfg=compute_cfg,
        )
        self.include_vinf = runtime_config.include_vinf
        self.normalize_mi = runtime_config.normalize_mi
        self.network_population_information = bool(
            _get_config_value(
                compute_cfg,
                "network_population_information",
                False,
            )
        )
        if runtime_config.invalid_output_units is not None:
            self.logger.warning(
                "Unknown output_units=%r; falling back to 'bits'.",
                runtime_config.invalid_output_units,
            )
        self.output_units = runtime_config.output_units
        self.analysis_split = runtime_config.analysis_split
        self.verbose = runtime_config.verbose

    def _configure_granularity_defaults(
        self,
        *,
        params: Any,
        compute_cfg: Any,
    ) -> None:
        """Assign granularity and enhanced-view defaults."""
        granularity_config = _build_information_granularity_config(
            params=params,
            compute_cfg=compute_cfg,
        )
        self.computation_level = granularity_config.computation_level
        self.branch_aggregation = granularity_config.branch_aggregation
        sample_count = granularity_config.branch_sample_count_per_layer
        if sample_count is not None:
            sample_count = int(sample_count)
            if sample_count < 1:
                raise ValueError("branch_sample_count_per_layer must be at least 1")
        self.branch_sample_count_per_layer = sample_count
        self.branch_sample_seed = granularity_config.branch_sample_seed
        sample_strategy = (
            granularity_config.branch_sample_strategy.strip().lower().replace("-", "_")
        )
        if sample_strategy not in {"uniform", "parent_soma_balanced"}:
            raise ValueError(
                "branch_sample_strategy must be 'uniform' or " "'parent_soma_balanced'"
            )
        self.branch_sample_strategy = sample_strategy
        self.per_neuron_analysis = granularity_config.per_neuron_analysis
        self.per_einet_analysis = granularity_config.per_einet_analysis
        self.per_layer_analysis = granularity_config.per_layer_analysis

    def _configure_scope_selection_override(
        self,
        *,
        params: Any,
        selection_cfg: Any,
    ) -> None:
        """Apply selection.scopes/views overrides to per-view flags."""
        scopes_raw = list(
            _get_config_value(
                selection_cfg,
                "scopes",
                _get_config_value(params, "scopes", []),
            )
            or []
        )
        if not scopes_raw:
            scopes_raw = list(
                _get_config_value(
                    selection_cfg,
                    "views",
                    _get_config_value(params, "views", []),
                )
                or []
            )

        scope_selection = _parse_information_scope_selection(scopes_raw)
        if scope_selection is None:
            return

        self.per_neuron_analysis = scope_selection.per_neuron_analysis
        self.per_einet_analysis = scope_selection.per_einet_analysis
        self.per_layer_analysis = scope_selection.per_layer_analysis

    def _configure_signal_variants(
        self,
        *,
        params: Any,
        selection_cfg: Any,
        compute_cfg: Any,
        baselines_cfg: Any,
    ) -> None:
        """Assign LDA and random-weight signal variant settings."""
        signal_variant_selection = _build_information_signal_variant_selection(
            params=params,
            selection_cfg=selection_cfg,
            compute_cfg=compute_cfg,
            baselines_cfg=baselines_cfg,
        )
        self.compute_lda_weights = signal_variant_selection.compute_lda_weights
        self.lda_n_shuffles = signal_variant_selection.lda_n_shuffles
        self.random_weights_seed = signal_variant_selection.random_weights_seed


__all__ = ["InformationConfigSetupMixin"]
