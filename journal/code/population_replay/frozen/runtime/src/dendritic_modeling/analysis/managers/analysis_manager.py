"""
Analysis manager module.

This module contains the AnalysisManager class which coordinates and orchestrates
all the different analysis tools in the dendritic modeling package.
"""

import logging
import os
from collections.abc import Callable
from typing import Optional

import torch

from dendritic_modeling.analysis.managers.analysis_plan import (
    _ANALYZER_INITIALIZERS,
    _DENDRITIC_TIMETRACES_ANALYSIS,
    _INHIBITORY_SPECIALIZATION_ANALYSIS,
    _JACOBIAN_SPECTRUM_ANALYSIS,
    _MODEL_ONLY_ANALYSES,
    _ORDERED_ANALYSIS_SEQUENCE,
    _POST_REACTIVATION_TEST_DATASET_ANALYSES,
    _PRE_WEIGHT_TEST_DATASET_ANALYSES,
    _ModelOnlyAnalysisRunner,
    _registered_analyzer_spec,
    _TestDatasetAnalysisRunner,
)
from dendritic_modeling.config import AnalysisConfig
from dendritic_modeling.models import BaseModel, Regressor
from dendritic_modeling.utils.reproducibility import isolated_random_seed

__all__ = [
    "_DENDRITIC_TIMETRACES_ANALYSIS",
    "_INHIBITORY_SPECIALIZATION_ANALYSIS",
    "_JACOBIAN_SPECTRUM_ANALYSIS",
    "_MODEL_ONLY_ANALYSES",
    "_POST_REACTIVATION_TEST_DATASET_ANALYSES",
    "_PRE_WEIGHT_TEST_DATASET_ANALYSES",
    "AnalysisManager",
]


class AnalysisManager:
    """
    Coordinates and orchestrates all analysis tools for dendritic modeling.

    This class manages the initialization and execution of various analysis tools
    based on configuration settings, providing a unified interface for running
    comprehensive model analysis.
    """

    def __init__(
        self,
        model: BaseModel,
        data: dict[str, torch.utils.data.Dataset],
        analysis_config: AnalysisConfig,
        save_root: Optional[str] = None,
        device: str = "cpu",
        evaluation_seed: int | None = None,
        probe_seed: int | None = None,
    ):
        self.model = model
        self.data = data
        self.config = analysis_config
        self.save_root = save_root
        self.device = device
        self.evaluation_seed = None if evaluation_seed is None else int(evaluation_seed)
        self.probe_seed = None if probe_seed is None else int(probe_seed)
        os.makedirs(self.save_root, exist_ok=True)
        # Logging should be configured externally at the application startup
        self.logger = logging.getLogger(__name__)

        self._apply_reproducibility_seeds()
        self._sanitize_performance_metrics_for_task()
        self._initialize_configured_analyzers()

    def _apply_reproducibility_seeds(self) -> None:
        """Route experiment evaluation/probe seeds to their concrete consumers."""
        if self.evaluation_seed is not None:
            runtime = getattr(self.config, "runtime", None)
            for phase in ("training", "final"):
                profile = getattr(runtime, phase, None) if runtime is not None else None
                if profile is not None:
                    profile.seed = self.evaluation_seed

        if self.probe_seed is None:
            return
        representation = getattr(
            self.config,
            "representation_accessibility_analysis",
            None,
        )
        representation_params = getattr(representation, "params", None)
        if representation_params is not None and hasattr(representation_params, "seed"):
            representation_params.seed = self.probe_seed
        information = getattr(self.config, "information_analysis", None)
        params = getattr(information, "params", None)
        if params is None:
            return
        # Keep the legacy and grouped decoder representations synchronized;
        # grouped config takes precedence when the analyzer normalizes options.
        if hasattr(params, "decoder_seed"):
            params.decoder_seed = self.probe_seed
        estimator = getattr(params, "estimator", None)
        decoder = getattr(estimator, "decoder", None) if estimator is not None else None
        if decoder is not None:
            decoder.seed = self.probe_seed

    def _initialize_configured_analyzers(self) -> None:
        """Instantiate enabled analyzers from registry-backed metadata."""
        for initializer in _ANALYZER_INITIALIZERS:
            cfg = getattr(self.config, initializer.config_field, None)
            if cfg is None or not getattr(cfg, "enabled", False):
                continue
            if (
                getattr(cfg, "training", False)
                and not initializer.spec.supports_training
            ):
                raise ValueError(
                    f"Analyzer {initializer.registry_name!r} is final-only and "
                    "cannot set training=true"
                )
            params = getattr(cfg, "params", None)
            setattr(
                self,
                initializer.attribute_name,
                initializer.build(model=self.model, params=params),
            )
        self.correlation_analysis_enabled = hasattr(self, "correlation_analyzer")

    def _sanitize_performance_metrics_for_task(self) -> None:
        """Disable incompatible default metrics for regression models."""
        perf_cfg = getattr(self.config, "performance_analysis", None)
        params = getattr(perf_cfg, "params", None) if perf_cfg is not None else None
        if params is None:
            return

        is_regression = (
            isinstance(self.model, Regressor)
            or getattr(self.model, "task", None) == "regression"
        )
        if is_regression:
            changed = []
            for name in ("accuracy", "auc", "categorical_loglikelihood"):
                if getattr(params, name, False):
                    setattr(params, name, False)
                    changed.append(name)
            if changed:
                for name in ("mse", "cosine_similarity"):
                    if not getattr(params, name, False):
                        setattr(params, name, True)
                self.logger.info(
                    "Regression model detected; disabled incompatible performance "
                    "metrics %s and enabled mse/cosine_similarity",
                    changed,
                )

    @staticmethod
    def _should_run_config(cfg: object | None, training: bool) -> bool:
        """Return whether an analysis config is enabled for this phase."""
        if cfg is None or not getattr(cfg, "enabled", False):
            return False
        return (getattr(cfg, "training", False) and training) or not training

    def _save_path(self, *parts: str, create: bool = False) -> str:
        """Build an analysis output path and optionally create it."""
        path = os.path.join(self.save_root, *parts)
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    def _analysis_config(self, registry_name: str) -> object | None:
        """Return the config object for a registered analyzer."""
        spec = _registered_analyzer_spec(registry_name)
        return getattr(self.config, spec.config_field, None)

    def _analysis_analyzer(self, registry_name: str) -> object:
        """Return the analyzer instance for a registered analyzer."""
        spec = _registered_analyzer_spec(registry_name)
        return getattr(self, spec.attribute_name)

    def _dataset_for_analysis_split(self, split: str) -> torch.utils.data.Dataset:
        """Resolve a canonical analysis split name to its dataset object."""
        canonical = str(split).lower()
        dataset_key = "valid" if canonical == "validation" else canonical
        if dataset_key not in self.data or canonical not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(f"Unknown analysis dataset split: {split!r}")
        return self.data[dataset_key]

    def _call_analysis_analyzer(self, registry_name: str, **kwargs) -> None:
        """Invoke a registered analyzer with precomputed keyword arguments."""
        self._analysis_analyzer(registry_name).analyze(**kwargs)

    def _has_analysis_analyzer(self, registry_name: str) -> bool:
        """Return whether the registered analyzer has been initialized."""
        spec = _registered_analyzer_spec(registry_name)
        return hasattr(self, spec.attribute_name)

    def _analysis_save_path(
        self, registry_name: str, *parts: str, create: bool = False
    ) -> str:
        """Build an output path from registry-declared analyzer metadata."""
        spec = _registered_analyzer_spec(registry_name)
        return self._save_path(spec.save_subdir, *parts, create=create)

    def _run_logged_action(
        self,
        *,
        start_message: str,
        complete_message: str,
        action: Callable[[], None],
    ) -> None:
        """Log a start/success pair around an analysis action."""
        self.logger.info(start_message)
        action()
        self.logger.info(complete_message)

    def _should_run_registered_analysis(
        self,
        registry_name: str,
        *,
        training: bool,
        require_analyzer: bool = False,
    ) -> bool:
        """Return whether a registered analyzer should run in this phase."""
        if not self._should_run_config(self._analysis_config(registry_name), training):
            return False
        if require_analyzer and not self._has_analysis_analyzer(registry_name):
            return False
        return True

    def _run_test_dataset_analysis(
        self,
        *,
        registry_name: str,
        filename: str,
        training: bool,
        runtime: object | None,
        pass_training: bool = False,
        require_analyzer: bool = False,
        dataset_split: str = "test",
    ) -> None:
        """Run an analyzer with the common test-dataset call signature."""
        if not self._should_run_registered_analysis(
            registry_name,
            training=training,
            require_analyzer=require_analyzer,
        ):
            return

        dataset_key = "valid" if dataset_split == "validation" else dataset_split
        if dataset_key not in self.data:
            raise ValueError(f"Unknown analysis dataset split: {dataset_split!r}")

        kwargs = {
            "model": self.model,
            "test_dataset": self.data[dataset_key],
            "device": self.device,
            "save_path": self._analysis_save_path(registry_name),
            "filename": filename,
            "runtime": runtime,
        }
        if pass_training:
            kwargs["training"] = training
        if registry_name in {"ablation", "path_matched_intervention"}:
            # Fixed-reference interventions are calibrated once on training
            # data, then held fixed while evaluating the test split.
            kwargs["reference_dataset"] = self.data.get("train", self.data["test"])

        self._call_analysis_analyzer(registry_name, **kwargs)

    def _run_test_dataset_analysis_runner(
        self,
        runner: _TestDatasetAnalysisRunner,
        *,
        filename: str,
        training: bool,
        runtime: object | None,
    ) -> None:
        """Run a table-declared test-dataset analyzer."""
        runner.run(
            self,
            filename=filename,
            training=training,
            runtime=runtime,
        )

    def _run_test_dataset_analysis_runners(
        self,
        runners: tuple[_TestDatasetAnalysisRunner, ...],
        *,
        filename: str,
        training: bool,
        runtime: object | None,
    ) -> None:
        """Run a table of test-dataset analyzers in declaration order."""
        for runner in runners:
            self._run_test_dataset_analysis_runner(
                runner,
                filename=filename,
                training=training,
                runtime=runtime,
            )

    def _run_logged_test_dataset_analysis(
        self,
        *,
        registry_name: str,
        filename: str,
        training: bool,
        runtime: object | None,
        start_message: str,
        complete_message: str,
        require_analyzer: bool = False,
        dataset_split: str = "test",
    ) -> None:
        """Run a common test-dataset analyzer with start/complete logging."""
        if not self._should_run_registered_analysis(
            registry_name,
            training=training,
            require_analyzer=require_analyzer,
        ):
            return

        self._run_logged_action(
            start_message=start_message,
            complete_message=complete_message,
            action=lambda: self._run_test_dataset_analysis(
                registry_name=registry_name,
                filename=filename,
                training=training,
                runtime=runtime,
                require_analyzer=require_analyzer,
                dataset_split=dataset_split,
            ),
        )

    def _run_logged_data_analysis(
        self,
        *,
        registry_name: str,
        filename: str,
        training: bool,
        start_message: str,
        complete_message: str,
        runtime: object | None = None,
        pass_runtime: bool = False,
        require_analyzer: bool = True,
    ) -> None:
        """Run a logged analyzer with the common data= test-set signature."""
        if not self._should_run_registered_analysis(
            registry_name,
            training=training,
            require_analyzer=require_analyzer,
        ):
            return

        kwargs = {
            "model": self.model,
            "data": self.data["test"],
            "device": self.device,
            "save_path": self._analysis_save_path(registry_name),
            "filename": filename,
        }
        if pass_runtime:
            kwargs["runtime"] = runtime

        self._run_logged_action(
            start_message=start_message,
            complete_message=complete_message,
            action=lambda: self._call_analysis_analyzer(registry_name, **kwargs),
        )

    def set_pruning_results(self, pruning_results: dict):
        """Set pruning results for inclusion in performance plots."""
        self.pruning_results = pruning_results

    def _analysis_runtime(self, training: bool) -> object | None:
        """Return the runtime profile for the current analysis phase."""
        return (
            getattr(self.config.runtime, "training", None)
            if training
            else getattr(self.config.runtime, "final", None)
        )

    def _run_performance_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run performance analysis with split-specific options."""
        perf_cfg = self._analysis_config("performance")
        if not self._should_run_config(perf_cfg, training):
            return

        self._call_analysis_analyzer(
            "performance",
            model=self.model,
            train_ds=self.data["train"],
            valid_ds=self.data["valid"],
            test_ds=self.data["test"],
            device=self.device,
            save_path=self._performance_save_path(training=training),
            filename=filename,
            training=training,
            runtime=runtime,
            splits=self._performance_splits(perf_cfg, training=training),
        )

    def _performance_save_path(self, *, training: bool) -> str:
        """Return the performance output directory for the current phase."""
        if training:
            return self._analysis_save_path("performance", "epochs", create=True)
        return self._analysis_save_path("performance")

    @staticmethod
    def _performance_splits(perf_cfg: object, *, training: bool) -> object | None:
        """Return configured dataset splits for the current performance phase."""
        if training:
            return getattr(perf_cfg, "training_splits", None)
        return getattr(perf_cfg, "final_splits", None)

    def _run_information_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run information analysis."""

        if not self._should_run_registered_analysis(
            "information", training=training, require_analyzer=True
        ):
            return

        def run() -> None:
            analysis_split = str(
                getattr(self.information_analyzer, "analysis_split", "test")
            )
            self._run_logged_test_dataset_analysis(
                registry_name="information",
                filename=filename,
                training=training,
                runtime=runtime,
                start_message="Starting information analysis...",
                complete_message="Information analysis completed successfully",
                dataset_split=analysis_split,
            )

        if self.probe_seed is None:
            run()
        else:
            with isolated_random_seed(self.probe_seed):
                run()

    def _run_path_matched_information_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run matched-width dendritic information on validation data only."""
        if not self._should_run_registered_analysis(
            "path_matched_information",
            training=training,
            require_analyzer=True,
        ):
            return
        config = self._analysis_config("path_matched_information")
        params = getattr(config, "params", None)
        add_probe_offset = bool(getattr(params, "add_probe_seed_offset", False))
        seed_offset = (
            int(self.probe_seed)
            if add_probe_offset and self.probe_seed is not None
            else 0
        )
        self._call_analysis_analyzer(
            "path_matched_information",
            model=self.model,
            validation_dataset=self.data["valid"],
            device=self.device,
            save_path=self._analysis_save_path("path_matched_information"),
            filename=filename,
            training=training,
            runtime=runtime,
            seed_offset=seed_offset,
        )

    def _run_synaptic_turnover_analysis(self, *, filename: str, training: bool) -> None:
        """Run synaptic turnover analysis."""
        self._run_training_flag_model_analysis(
            registry_name="synapse_turnover",
            filename=filename,
            training=training,
        )

    def _run_weight_analysis(
        self,
        *,
        filename: str,
        training: bool,
        runtime: object | None,
    ) -> None:
        """Run weight analysis with its configured structural reference split."""
        if not self._should_run_registered_analysis("weight", training=training):
            return
        analyzer_kwargs = {
            "model": self.model,
            "save_path": self._analysis_save_path("weight"),
            "filename": filename,
            "training": training,
            "runtime": runtime,
        }
        config = self._analysis_config("weight")
        params = getattr(config, "params", None)
        if bool(getattr(params, "compute_source_region_alignment", False)):
            reference_split = str(
                getattr(params, "source_region_reference_split", "validation")
            ).lower()
            dataset_key = (
                "valid" if reference_split == "validation" else reference_split
            )
            if dataset_key not in self.data:
                raise ValueError(
                    "weight-analysis reference split is unavailable: "
                    f"{reference_split!r}"
                )
            analyzer_kwargs.update(
                reference_dataset=self.data[dataset_key],
                reference_split=reference_split,
            )
        self._call_analysis_analyzer("weight", **analyzer_kwargs)

    def _run_training_flag_model_analysis(
        self,
        *,
        registry_name: str,
        filename: str,
        training: bool,
    ) -> None:
        """Run an analyzer with the model/save-path/filename/training signature."""
        if not self._should_run_registered_analysis(
            registry_name,
            training=training,
        ):
            return

        self._call_analysis_analyzer(
            registry_name,
            model=self.model,
            save_path=self._analysis_save_path(registry_name),
            filename=filename,
            training=training,
        )

    def _run_reactivation_dynamics_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run reactivation dynamics analysis."""
        self._run_split_dataset_analysis(
            registry_name="reactivation_dynamics",
            filename=filename,
            training=training,
            runtime=runtime,
        )

    def _run_representation_accessibility_analysis(
        self,
        *,
        filename: str,
        training: bool,
        runtime: object | None,
    ) -> None:
        """Run validation-fit representation probes and information metrics."""
        self._run_split_dataset_analysis(
            registry_name="representation_accessibility",
            filename=filename,
            training=training,
            runtime=runtime,
            require_analyzer=True,
        )

    def _run_source_tuning_support_analysis(
        self,
        *,
        filename: str,
        training: bool,
        runtime: object | None,
    ) -> None:
        """Run split-safe source tuning with independent preference selection."""
        if not self._should_run_registered_analysis(
            "source_tuning_support",
            training=training,
            require_analyzer=True,
        ):
            return
        config = self._analysis_config("source_tuning_support")
        params = getattr(config, "params", None)
        reference_split = str(getattr(params, "reference_split", "train"))
        evaluation_split = str(getattr(params, "evaluation_split", "validation"))
        self._call_analysis_analyzer(
            "source_tuning_support",
            model=self.model,
            data=self._dataset_for_analysis_split(evaluation_split),
            reference_data=self._dataset_for_analysis_split(reference_split),
            reference_split_name=reference_split,
            evaluation_split_name=evaluation_split,
            device=self.device,
            save_path=self._analysis_save_path("source_tuning_support"),
            filename=filename,
            runtime=runtime,
        )

    def _run_input_region_intervention_analysis(
        self,
        *,
        filename: str,
        training: bool,
        runtime: object | None,
    ) -> None:
        """Run paired input-region interventions on the configured split."""
        if not self._should_run_registered_analysis(
            "input_region_intervention",
            training=training,
            require_analyzer=True,
        ):
            return
        config = self._analysis_config("input_region_intervention")
        params = getattr(config, "params", None)
        reference_split = str(getattr(params, "reference_split", "train"))
        evaluation_split = str(getattr(params, "evaluation_split", "test"))
        self._call_analysis_analyzer(
            "input_region_intervention",
            model=self.model,
            test_dataset=self._dataset_for_analysis_split(evaluation_split),
            reference_dataset=self._dataset_for_analysis_split(reference_split),
            runtime=runtime,
            reference_runtime=runtime,
            device=self.device,
            save_path=self._analysis_save_path("input_region_intervention"),
            filename=filename,
        )

    def _run_local_rule_component_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run local-rule component analysis."""
        self._run_split_dataset_analysis(
            registry_name="local_rule_components",
            filename=filename,
            training=training,
            runtime=runtime,
            test_dataset_kwarg="test_dataset",
            require_analyzer=True,
        )

    def _run_split_dataset_analysis(
        self,
        *,
        registry_name: str,
        filename: str,
        training: bool,
        runtime: object | None,
        test_dataset_kwarg: str = "test_ds",
        require_analyzer: bool = False,
    ) -> None:
        """Run an analyzer with train/valid/test split datasets."""
        if not self._should_run_registered_analysis(
            registry_name,
            training=training,
            require_analyzer=require_analyzer,
        ):
            return

        kwargs = {
            "model": self.model,
            "train_ds": self.data.get("train"),
            "valid_ds": self.data.get("valid"),
            test_dataset_kwarg: self.data.get("test"),
            "device": self.device,
            "save_path": self._analysis_save_path(registry_name),
            "filename": filename,
            "training": training,
            "runtime": runtime,
        }
        self._call_analysis_analyzer(registry_name, **kwargs)

    def _run_correlation_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run correlation analysis."""
        self._run_logged_test_dataset_analysis(
            registry_name="correlation",
            filename=filename,
            training=training,
            runtime=runtime,
            start_message="Starting correlation analysis...",
            complete_message="Correlation analysis completed successfully",
            require_analyzer=True,
        )

    def _run_routing_analysis(self, *, filename: str, training: bool) -> None:
        """Run recurrent routing analysis."""
        self._run_logged_data_analysis(
            registry_name="routing",
            filename=filename,
            training=training,
            start_message="Starting routing analysis...",
            complete_message="Routing analysis completed successfully",
        )

    def _run_model_only_analysis(
        self,
        *,
        registry_name: str,
        filename: str,
        training: bool,
        start_message: str,
        complete_message: str,
    ) -> None:
        """Run an analyzer with the common model/device/save-path signature."""
        if not self._should_run_registered_analysis(
            registry_name,
            training=training,
            require_analyzer=True,
        ):
            return

        self._run_logged_action(
            start_message=start_message,
            complete_message=complete_message,
            action=lambda: self._call_analysis_analyzer(
                registry_name,
                model=self.model,
                device=self.device,
                save_path=self._analysis_save_path(registry_name),
                filename=filename,
            ),
        )

    def _run_model_only_analysis_runner(
        self,
        runner: _ModelOnlyAnalysisRunner,
        *,
        filename: str,
        training: bool,
    ) -> None:
        """Run a table-declared model-only analyzer."""
        runner.run(
            self,
            filename=filename,
            training=training,
        )

    def _run_model_only_analysis_runners(
        self,
        runners: tuple[_ModelOnlyAnalysisRunner, ...],
        *,
        filename: str,
        training: bool,
    ) -> None:
        """Run a table of model-only analyzers in declaration order."""
        for runner in runners:
            self._run_model_only_analysis_runner(
                runner,
                filename=filename,
                training=training,
            )

    def _run_inhibitory_specialization_analysis(
        self, *, filename: str, training: bool
    ) -> None:
        """Run inhibitory specialization analysis."""
        self._run_model_only_analysis_runner(
            _INHIBITORY_SPECIALIZATION_ANALYSIS,
            filename=filename,
            training=training,
        )

    def _run_jacobian_spectrum_analysis(self, *, filename: str, training: bool) -> None:
        """Run jacobian spectrum analysis."""
        self._run_model_only_analysis_runner(
            _JACOBIAN_SPECTRUM_ANALYSIS,
            filename=filename,
            training=training,
        )

    def _run_dendritic_timetraces_analysis(
        self, *, filename: str, training: bool
    ) -> None:
        """Run dendritic timetraces analysis."""
        self._run_model_only_analysis_runner(
            _DENDRITIC_TIMETRACES_ANALYSIS,
            filename=filename,
            training=training,
        )

    def _run_spike_train_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run spike-train analysis."""
        self._run_logged_data_analysis(
            registry_name="spike_train",
            filename=filename,
            training=training,
            start_message="Starting spike train analysis...",
            complete_message="Spike train analysis completed successfully",
            runtime=runtime,
            pass_runtime=True,
        )

    def _run_adversarial_robustness_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run adversarial robustness analysis."""
        self._run_logged_test_dataset_analysis(
            registry_name="adversarial_robustness",
            filename=filename,
            training=training,
            runtime=runtime,
            start_message="Starting adversarial robustness analysis...",
            complete_message="Adversarial robustness analysis completed successfully",
            require_analyzer=True,
        )

    def _run_ordered_analysis_sequence(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run analyzer wrappers in the historically defined order."""
        for step in _ORDERED_ANALYSIS_SEQUENCE:
            step.run(
                self,
                filename=filename,
                training=training,
                runtime=runtime,
            )

    def run_analysis(self, filename: str = "final", training: bool = False):
        """
        Run all enabled analysis tools.

        Args:
            filename: Base filename for saving analysis results
            training: Whether this is being called during training (affects which analyses run)
        """
        runtime = self._analysis_runtime(training)
        if self.evaluation_seed is None:
            self._run_ordered_analysis_sequence(
                filename=filename,
                training=training,
                runtime=runtime,
            )
        else:
            with isolated_random_seed(self.evaluation_seed):
                self._run_ordered_analysis_sequence(
                    filename=filename,
                    training=training,
                    runtime=runtime,
                )
