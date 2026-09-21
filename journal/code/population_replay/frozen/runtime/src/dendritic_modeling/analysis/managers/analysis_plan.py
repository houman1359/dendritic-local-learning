"""Registry-backed analyzer construction and execution metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from dendritic_modeling.analysis.registry import (
    AnalyzerSpec,
    get_analyzer_spec,
    get_builtin_analyzer_names,
    load_analyzer_class,
)

if TYPE_CHECKING:
    from dendritic_modeling.analysis.managers.analysis_manager import AnalysisManager


def _registered_analyzer_spec(registry_name: str) -> AnalyzerSpec:
    """Return registered analyzer metadata or fail loudly on registry drift."""
    spec = get_analyzer_spec(registry_name)
    if spec is None:
        raise KeyError(f"Analyzer {registry_name!r} is not registered")
    return spec


@dataclass(frozen=True)
class _RegisteredAnalysisMetadata:
    """Base metadata for entries backed by the analyzer registry."""

    registry_name: str

    @property
    def spec(self) -> AnalyzerSpec:
        return _registered_analyzer_spec(self.registry_name)

    @property
    def config_field(self) -> str:
        return self.spec.config_field

    @property
    def attribute_name(self) -> str:
        return self.spec.attribute_name

    @property
    def analyzer_attr(self) -> str:
        return self.spec.attribute_name

    @property
    def save_subdir(self) -> str:
        return self.spec.save_subdir


@dataclass(frozen=True)
class _AnalyzerInitializer(_RegisteredAnalysisMetadata):
    """Construction metadata used by AnalysisManager."""

    @property
    def analyzer_cls(self) -> type:
        return load_analyzer_class(self.registry_name)

    @property
    def constructor_style(self) -> str:
        return self.spec.constructor_style

    def build(self, *, model: object, params: object | None) -> object:
        """Instantiate this analyzer using its registered constructor style."""
        analyzer_cls = self.analyzer_cls
        constructor_style = self.constructor_style
        if constructor_style == "none":
            return analyzer_cls()
        if constructor_style == "model_params":
            return analyzer_cls(model=model, params=params)
        if constructor_style == "params":
            params_builder = getattr(params, "to_analyzer_params", None)
            if callable(params_builder):
                params = params_builder()
            return analyzer_cls(params)
        raise ValueError(
            "Unsupported analyzer constructor style for "
            f"{self.registry_name}: {constructor_style}"
        )


@dataclass(frozen=True)
class _TestDatasetAnalysisRunner(_RegisteredAnalysisMetadata):
    """Execution metadata for analyzers sharing the test-dataset signature."""

    pass_training: bool = False
    require_analyzer: bool = False

    def run(
        self,
        manager: AnalysisManager,
        *,
        filename: str,
        training: bool,
        runtime: object | None,
    ) -> None:
        """Run this analyzer through the manager's shared test-dataset path."""
        manager._run_test_dataset_analysis(
            registry_name=self.registry_name,
            filename=filename,
            training=training,
            runtime=runtime,
            pass_training=self.pass_training,
            require_analyzer=self.require_analyzer,
        )


@dataclass(frozen=True)
class _ModelOnlyAnalysisRunner(_RegisteredAnalysisMetadata):
    """Execution metadata for analyzers sharing the model-only signature."""

    start_message: str
    complete_message: str

    def run(
        self,
        manager: AnalysisManager,
        *,
        filename: str,
        training: bool,
    ) -> None:
        """Run this analyzer through the manager's shared model-only path."""
        manager._run_model_only_analysis(
            registry_name=self.registry_name,
            filename=filename,
            training=training,
            start_message=self.start_message,
            complete_message=self.complete_message,
        )


@dataclass(frozen=True)
class _AnalysisSequenceStep:
    """One entry in the top-level analysis execution order."""

    method_name: str
    pass_runtime: bool = False
    runners: object | None = None

    def run(
        self,
        manager: AnalysisManager,
        *,
        filename: str,
        training: bool,
        runtime: object | None,
    ) -> None:
        """Call the manager method declared by this step."""
        kwargs = {"filename": filename, "training": training}
        if self.runners is not None:
            kwargs["runners"] = self.runners
        if self.pass_runtime:
            kwargs["runtime"] = runtime
        getattr(manager, self.method_name)(**kwargs)


_ANALYZER_INITIALIZER_NAMES = get_builtin_analyzer_names()
_ANALYZER_INITIALIZERS = tuple(
    _AnalyzerInitializer(name) for name in _ANALYZER_INITIALIZER_NAMES
)


_INHIBITORY_SPECIALIZATION_ANALYSIS = _ModelOnlyAnalysisRunner(
    "inhibitory_specialization",
    start_message="Starting inhibitory specialization analysis...",
    complete_message="Inhibitory specialization analysis completed successfully",
)
_JACOBIAN_SPECTRUM_ANALYSIS = _ModelOnlyAnalysisRunner(
    "jacobian_spectrum",
    start_message="Starting jacobian spectrum analysis...",
    complete_message="Jacobian spectrum analysis completed successfully",
)
_DENDRITIC_TIMETRACES_ANALYSIS = _ModelOnlyAnalysisRunner(
    "dendritic_timetraces",
    start_message="Starting dendritic timetraces analysis...",
    complete_message="Dendritic timetraces analysis completed successfully",
)
_MODEL_ONLY_ANALYSES = (
    _INHIBITORY_SPECIALIZATION_ANALYSIS,
    _JACOBIAN_SPECTRUM_ANALYSIS,
    _DENDRITIC_TIMETRACES_ANALYSIS,
)


_PRE_WEIGHT_TEST_DATASET_ANALYSES = (
    _TestDatasetAnalysisRunner(
        "vision_boundary_diagnostics",
        require_analyzer=True,
    ),
    _TestDatasetAnalysisRunner("noise_perturbation"),
    _TestDatasetAnalysisRunner("single_layer_contribution"),
    _TestDatasetAnalysisRunner(
        "ablation",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner(
        "path_matched_intervention",
        pass_training=True,
        require_analyzer=True,
    ),
    _TestDatasetAnalysisRunner(
        "local_inhibitory_population_intervention",
        pass_training=True,
        require_analyzer=True,
    ),
)


_POST_REACTIVATION_TEST_DATASET_ANALYSES = (
    _TestDatasetAnalysisRunner(
        "branch_local_sensitivity",
        pass_training=True,
        require_analyzer=True,
    ),
    _TestDatasetAnalysisRunner(
        "compartment_statistics",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner(
        "compartment_snr",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner(
        "multiplicative_gain",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner(
        "gain_load_perturbation",
        pass_training=True,
        require_analyzer=True,
    ),
    _TestDatasetAnalysisRunner(
        "receptive_field",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner("synaptic_activation"),
    _TestDatasetAnalysisRunner("branch_activation"),
)


_ORDERED_ANALYSIS_SEQUENCE = (
    _AnalysisSequenceStep("_run_performance_analysis", pass_runtime=True),
    _AnalysisSequenceStep("_run_information_analysis", pass_runtime=True),
    _AnalysisSequenceStep(
        "_run_path_matched_information_analysis",
        pass_runtime=True,
    ),
    _AnalysisSequenceStep(
        "_run_representation_accessibility_analysis",
        pass_runtime=True,
    ),
    _AnalysisSequenceStep(
        "_run_source_tuning_support_analysis",
        pass_runtime=True,
    ),
    _AnalysisSequenceStep("_run_synaptic_turnover_analysis"),
    _AnalysisSequenceStep(
        "_run_test_dataset_analysis_runners",
        pass_runtime=True,
        runners=_PRE_WEIGHT_TEST_DATASET_ANALYSES,
    ),
    _AnalysisSequenceStep(
        "_run_input_region_intervention_analysis",
        pass_runtime=True,
    ),
    _AnalysisSequenceStep("_run_weight_analysis", pass_runtime=True),
    _AnalysisSequenceStep("_run_reactivation_dynamics_analysis", pass_runtime=True),
    _AnalysisSequenceStep(
        "_run_test_dataset_analysis_runners",
        pass_runtime=True,
        runners=_POST_REACTIVATION_TEST_DATASET_ANALYSES,
    ),
    _AnalysisSequenceStep("_run_local_rule_component_analysis", pass_runtime=True),
    _AnalysisSequenceStep("_run_correlation_analysis", pass_runtime=True),
    _AnalysisSequenceStep("_run_routing_analysis"),
    _AnalysisSequenceStep(
        "_run_model_only_analysis_runners",
        runners=_MODEL_ONLY_ANALYSES,
    ),
    _AnalysisSequenceStep("_run_spike_train_analysis", pass_runtime=True),
    _AnalysisSequenceStep("_run_adversarial_robustness_analysis", pass_runtime=True),
)
