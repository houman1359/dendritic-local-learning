"""Integrated information analyzer for dendritic networks.

This module provides a unified interface for computing various information-theoretic
metrics using the multivariate methods from information.py. It works directly
with the main configuration system.
"""

import logging
from typing import Optional

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.core.information_parts.information_all_branch_mixin import (
    InformationAllBranchMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_branch_mixin import (
    InformationBranchAnalysisMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_config import (
    _build_information_ablation_aligned_config as _build_information_ablation_aligned_config,
    _build_information_estimator_core_config as _build_information_estimator_core_config,
    _build_information_estimator_method_params as _build_information_estimator_method_params,
    _build_information_gaussian_fisher_config as _build_information_gaussian_fisher_config,
    _build_information_granularity_config as _build_information_granularity_config,
    _build_information_label_shuffle_null_config as _build_information_label_shuffle_null_config,
    _build_information_pid_config as _build_information_pid_config,
    _build_information_runtime_config as _build_information_runtime_config,
    _build_information_signal_variant_selection as _build_information_signal_variant_selection,
    _get_config_path as _get_config_path,
    _get_config_value as _get_config_value,
    _get_grouped_or_legacy_value as _get_grouped_or_legacy_value,
    _get_grouped_path_or_legacy_value as _get_grouped_path_or_legacy_value,
    _get_lower_config_value as _get_lower_config_value,
    _InformationAblationAlignedConfig as _InformationAblationAlignedConfig,
    _InformationComponentSelection as _InformationComponentSelection,
    _InformationEstimatorCoreConfig as _InformationEstimatorCoreConfig,
    _InformationGaussianFisherConfig as _InformationGaussianFisherConfig,
    _InformationGranularityConfig as _InformationGranularityConfig,
    _InformationLabelShuffleNullConfig as _InformationLabelShuffleNullConfig,
    _InformationPidConfig as _InformationPidConfig,
    _InformationRuntimeConfig as _InformationRuntimeConfig,
    _InformationScopeSelection as _InformationScopeSelection,
    _InformationSignalVariantSelection as _InformationSignalVariantSelection,
    _normalize_config_token as _normalize_config_token,
    _normalize_config_tokens as _normalize_config_tokens,
    _parse_information_component_selection as _parse_information_component_selection,
    _parse_information_scope_selection as _parse_information_scope_selection,
)
from dendritic_modeling.analysis.core.information_parts.information_config_setup_mixin import (
    InformationConfigSetupMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_data_mixin import (
    InformationDataMixin,
    _InformationAnalysisData as _InformationAnalysisData,
)
from dendritic_modeling.analysis.core.information_parts.information_dispatch_mixin import (
    InformationDispatchMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_enhanced_mixin import (
    InformationEnhancedAnalysisMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_estimator_setup import (
    _build_information_estimators,
)
from dendritic_modeling.analysis.core.information_parts.information_finalization_mixin import (
    InformationFinalizationMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_hooks_mixin import (
    InformationHooksMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_lda_mixin import (
    InformationLdaComparisonMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_lifecycle_mixin import (
    InformationLifecycleMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_metrics_mixin import (
    InformationMetricsMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_pipeline_mixin import (
    InformationPipelineMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_plotting_mixin import (
    InformationPlottingMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_signal_mixin import (
    InformationSignalAggregationMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_single_branch_mixin import (
    InformationSingleBranchMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_synthetic_soma_mixin import (
    InformationSyntheticSomaMixin,
)
from dendritic_modeling.config import InformationAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel


class InformationAnalyzer(
    InformationMetricsMixin,
    InformationConfigSetupMixin,
    InformationHooksMixin,
    InformationDataMixin,
    InformationDispatchMixin,
    InformationPipelineMixin,
    InformationSyntheticSomaMixin,
    InformationSignalAggregationMixin,
    InformationLifecycleMixin,
    InformationLdaComparisonMixin,
    InformationPlottingMixin,
    InformationFinalizationMixin,
    InformationEnhancedAnalysisMixin,
    InformationBranchAnalysisMixin,
    InformationAllBranchMixin,
    InformationSingleBranchMixin,
    AbstractAnalyzer,
):
    """Unified analyzer for computing information-theoretic metrics in dendritic networks.

    This analyzer uses the multivariate-aware methods from information.py to compute
    mutual information between excitatory inputs, inhibitory inputs, branch outputs,
    and class labels. It supports continuous multivariate data natively.
    """

    def __init__(self, params: InformationAnalysisParams):
        """Initialize the analyzer with configuration from main YAML config.

        Parameters
        ----------
        config : InformationAnalysisConfig or dict
            Configuration from the main config system (analysis.information)
        """
        super().__init__("InformationAnalyzer")
        get_config = _get_config_value

        # New (preferred) grouped configs (all optional for backward compatibility)
        selection_cfg = get_config(params, "selection", None)
        compute_cfg = get_config(params, "compute", None)
        estimator_cfg = get_config(params, "estimator", None)
        baselines_cfg = get_config(params, "baselines", None)
        pid_cfg = get_config(params, "pid", None)
        gaussian_fisher_cfg = get_config(compute_cfg, "gaussian_fisher", None)

        # ------------------------------------------------------------------
        # Extract configuration parameters (support both new nested and legacy flat keys).
        # ------------------------------------------------------------------
        self._configure_estimator_core(
            params=params,
            estimator_cfg=estimator_cfg,
            compute_cfg=compute_cfg,
        )

        self._configure_label_shuffle_null(
            params=params,
            baselines_cfg=baselines_cfg,
        )

        # ------------------------------------------------------------------
        # Legacy boolean selection (still supported).
        # ------------------------------------------------------------------
        self._configure_legacy_component_selection(params=params)

        self._configure_pid_preprocessing(params=params, pid_cfg=pid_cfg)

        self._configure_ablation_aligned_metrics(
            params=params,
            compute_cfg=compute_cfg,
        )

        # ------------------------------------------------------------------
        # Preferred list-based selection of components.
        #
        # If provided, overrides the individual compute_* booleans above.
        # ------------------------------------------------------------------
        self._configure_component_selection_override(
            params=params,
            selection_cfg=selection_cfg,
        )

        self._disable_pid_if_requested()

        self._configure_gaussian_fisher_metrics(gaussian_fisher_cfg=gaussian_fisher_cfg)

        self._configure_runtime_options(
            params=params,
            compute_cfg=compute_cfg,
        )

        self._configure_granularity_defaults(
            params=params,
            compute_cfg=compute_cfg,
        )

        # Preferred list-based selection of scopes (overrides per_* flags if provided).
        #
        # `views` is a backward-compatible alias for `scopes`.
        self._configure_scope_selection_override(
            params=params,
            selection_cfg=selection_cfg,
        )

        self._configure_signal_variants(
            params=params,
            selection_cfg=selection_cfg,
            compute_cfg=compute_cfg,
            baselines_cfg=baselines_cfg,
        )

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        # Create estimators for each method
        self.estimators = _build_information_estimators(
            params=params,
            estimator_cfg=estimator_cfg,
            method=self.method,
            n_neighbors=self.n_neighbors,
            n_bins=self.n_bins,
            copula_type=self.copula_type,
            verbose=self.verbose,
        )

        self._check_pid_available()

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ):
        """Run integrated information analysis on the model.

        Parameters
        ----------
        model : BaseModel
            The trained model to analyze
        test_dataset : torch.utils.data.Dataset
            Test dataset for analysis
        device : str, optional
            Device to run analysis on, by default "cpu"
        save_path : Optional[str], optional
            Path to save results, by default None
        filename : str, optional
            Filename for saved results, by default "final"
        """
        self._log_analysis_start(
            model=model,
            test_dataset=test_dataset,
            device=device,
        )

        lifecycle_state = self._capture_model_lifecycle_state(model)
        try:
            # Check if model has ExcitationInhibitionNetwork
            if not self._model_supports_information_analysis(model):
                return

            analysis_start_time = self._mark_analysis_validation_passed()

            # Prepare data
            model = self._prepare_model_for_analysis(model, device)
            lifecycle_state = self._mark_model_moved(lifecycle_state)

            return self._run_analysis_pipeline(
                model=model,
                test_dataset=test_dataset,
                device=device,
                save_path=save_path,
                filename=filename,
                runtime=runtime,
                analysis_start_time=analysis_start_time,
            )

        except Exception as e:
            self._handle_analysis_exception(e)
        finally:
            self._restore_model_after_analysis(
                model=model,
                lifecycle_state=lifecycle_state,
            )
