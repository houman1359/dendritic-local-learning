import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Optional

from dendritic_modeling.analysis.registry import get_registered_analyzers
from dendritic_modeling.config.base import BaseConfig


@dataclass
class EvaluationRuntimeConfig(BaseConfig):
    """Controls how analysis iterates over a dataset.

    mode:
        "auto"        - materialize if estimated size < *materialize_threshold_mb*,
                         otherwise stream via DataLoader.
        "materialize" - always load the full dataset into memory (old behaviour).
        "stream"      - always iterate batch-by-batch through a DataLoader.

    max_batches:
        If set, cap evaluation to this many batches per split (useful for
        cheap approximate metrics during training).

    max_samples:
        If set, cap the number of examples exposed to the analysis.
        This is useful for expensive hook-based analyses.

    materialize_threshold_mb:
        Dataset size threshold (MB) for the ``"auto"`` mode switch.
        Default 1024 MB keeps CIFAR/MNIST fully materialized.
    """

    mode: str = "auto"
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = 2
    in_order: Optional[bool] = None
    max_batches: Optional[int] = None
    max_samples: Optional[int] = None
    materialize_threshold_mb: int = 1024
    seed: int = 0


@dataclass
class AnalysisRuntimeConfig(BaseConfig):
    """Shared runtime profiles for analysis during training and final evaluation."""

    training: EvaluationRuntimeConfig = field(default_factory=EvaluationRuntimeConfig)
    final: EvaluationRuntimeConfig = field(default_factory=EvaluationRuntimeConfig)


@dataclass
class PerformanceAnalysisParams(BaseConfig):
    """Parameters for performance evaluation metrics.

    Each boolean toggles whether that metric is computed/reported:
    - accuracy
    - auc
    - categorical_loglikelihood
    - mse
    - cosine_similarity
    """

    accuracy: bool = True
    auc: bool = True
    categorical_loglikelihood: bool = True
    mse: bool = False
    cosine_similarity: bool = False


@dataclass
class PerformanceAnalysisConfig(BaseConfig):
    """Enable/disable performance analysis.

    - enabled: turns the analysis module on/off.
    - training: if true, run during training.
    - training_splits / final_splits: which dataset splits are evaluated
      during training-time vs. final evaluation.
    """

    enabled: bool = False
    training: bool = False
    params: PerformanceAnalysisParams = field(default_factory=PerformanceAnalysisParams)
    training_splits: list[str] = field(
        default_factory=lambda: ["train", "valid", "test"]
    )
    final_splits: list[str] = field(default_factory=lambda: ["train", "valid", "test"])


@dataclass
class InformationSelectionParams(BaseConfig):
    """Selectors for what to compute and how to break it down.

    - components: metric families to compute
      - "basic": I(E;C), I(I;C), I(Vb;C), I(Vout;C), I(E,I;C), ...
      - "pairwise": I(E;I), I(E;Vout), I(Vb;Vout), ...
      - "conditional": conditional MI terms like I(E;Vout|C)
      - "gaussian_fisher": Gaussian/Fisher MI proxy metrics (fast; theory-inspired)
      - "pid": partial information decomposition (currently disabled; reserved)
      - "ablation_aligned": extra CMI terms designed to mirror ablations,
        such as I(Vout;C|E), I(Vout;C|I), or I(Vout;C|E,I)
      - "layer_proxies": layer-total proxies derived from single-branch metrics
      - "upstream_unique_cmi": adds I(Vb;C|E,I) (can be expensive)
      - "soma_coupling": branch-current and branch-output information about the
        corresponding soma output
    - scopes: extra summaries/breakdowns to run
      - "layer": per-layer breakdown
      - "einet": separate excitatory vs inhibitory pathway summaries
      - "neuron": per-neuron breakdown (expensive)
    - signal_variants: which signal variants to compute/plot
      - "network": DendriNet signals (always included)
      - "lda": *_lin variants using LDA projections
      - "random": *_sh random-weight baseline for LDA (requires baselines.random_weights.n_shuffles > 0)
    """

    components: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    signal_variants: list[str] = field(default_factory=list)
    # Backward-compatible alias (deprecated): prefer `scopes`.
    views: list[str] = field(default_factory=list)


@dataclass
class InformationGaussianFisherParams(BaseConfig):
    """Gaussian/Fisher proxy parameters (enabled via selection.components)."""

    aggregation: str = "sum"  # "sum"|"mean"|"max"
    mi_transform: str = "half_log1p"  # "half_log1p"|"log1p"|"none"
    eps: float = 1e-8
    # Optional alias for `eps` (if provided, prefer ridge over eps).
    ridge: Optional[float] = None


@dataclass
class InformationComputeParams(BaseConfig):
    """Runtime/aggregation controls (method-agnostic)."""

    max_samples: Optional[int] = None
    include_vinf: bool = True  # Include soma activation (Vinf)
    normalize_mi: bool = False  # Normalize by H(C)
    # Raw high-dimensional soma-population MI is opt-in. Prefer the dedicated
    # representation accessibility analyzer for matched projections and nulls.
    network_population_information: bool = False
    # Output units for MI/CMI-like quantities reported by the analyzer/plots.
    # Estimators operate in nats internally; the analyzer can convert to bits for reporting.
    output_units: str = "bits"  # "bits"|"nats"
    # Use validation while selecting estimators or quantities; the default keeps
    # backward compatibility for existing final-analysis configurations.
    analysis_split: str = "test"  # "validation"|"test"
    verbose: bool = False

    computation_level: str = (
        "single_branch"  # "single_branch", "layer_branch", "all_branch"
    )
    branch_aggregation: str = "mean"  # "mean", "multivariate", "sample"
    # Optional, deterministic compute cap for ``single_branch`` analyses.
    # Sampling is independent of activations and labels and is applied separately
    # within each captured layer. ``None`` preserves the exhaustive legacy behavior.
    branch_sample_count_per_layer: Optional[int] = None
    branch_sample_seed: int = 0
    # ``parent_soma_balanced`` cycles across parent somas before selecting a
    # second branch from any soma. ``uniform`` preserves legacy behavior.
    branch_sample_strategy: str = "uniform"

    # Layer-total proxies derived from single-branch metrics
    layer_total_topk: int = 10  # K used for *_topK_sum metrics (int >= 1)

    # Method-specific compute configs (enabled via selection.components)
    gaussian_fisher: InformationGaussianFisherParams = field(
        default_factory=InformationGaussianFisherParams
    )

    # Legacy (deprecated) scope toggles (use selection.scopes)
    per_neuron_analysis: bool = False
    per_einet_analysis: bool = False
    per_layer_analysis: bool = False

    # Legacy (deprecated) signal-variant toggles (use selection.signal_variants / baselines)
    compute_lda_weights: bool = False


@dataclass
class InformationPIDParams(BaseConfig):
    """PID-specific options (currently disabled; reserved for future use).

    Notes
    -----
    PID is currently disabled:
    - Synergy/redundancy are computed via an optional external PID backend.
      If that backend is not installed, PID metrics are skipped.
    - The reported "unique_*" terms are computed as:
      ``I(source;target) - redundancy`` using the *same* MI estimator selected by
      ``estimator.method``.

    Because typical PID backends assume **discrete** variables, we default to
    binarization. Treat PID numbers as exploratory diagnostics, not a primary
    metric.
    """

    binarize: bool = True
    binarize_method: str = "median"  # "median", "mean", "quantile"
    binarize_threshold: float = 0.5


@dataclass
class InformationEstimatorKraskovParams(BaseConfig):
    """Kraskov/KSG (kNN) estimator hyperparameters."""

    n_neighbors: int = 10


@dataclass
class InformationEstimatorBinnedParams(BaseConfig):
    """Histogram/binning plug-in estimator hyperparameters."""

    n_bins: int = 20
    binning_strategy: str = "quantile"  # "quantile", "uniform"
    bias_correction: str = "none"  # "none", "miller_madow"


@dataclass
class InformationEstimatorDecoderContinuousParams(BaseConfig):
    """Continuous fallback options for the decoder estimator."""

    strategy: str = "none"  # "none", "binned", "gaussian"
    max_total_dim: int = 2  # max total dim (x+y [+z]) for continuous MI/CMI
    gaussian_ridge: float = 1e-6


@dataclass
class InformationEstimatorDecoderParams(BaseConfig):
    """Decoder-based estimator hyperparameters (discrete target C)."""

    cv_folds: int = 5
    C: float = 1.0
    standardize: bool = True
    seed: int = 0
    continuous: InformationEstimatorDecoderContinuousParams = field(
        default_factory=InformationEstimatorDecoderContinuousParams
    )


@dataclass
class InformationEstimatorCopulaParams(BaseConfig):
    """Copula-based estimator hyperparameters (experimental).

    - type="gaussian": Gaussian copula (closed-form in the univariate case).
    - type="kernel": nonparametric KDE on the copula density (sample-hungry).
    - type="dvc": placeholder for a TF/DVC implementation; currently falls back
      to the simple copula path in this repository.

    Note: for multivariate inputs, the current implementation falls back to a
    kNN/KSG estimator on rank-transformed (uniform-marginal) data.
    """

    type: str = "gaussian"  # "gaussian", "kernel", "dvc"


@dataclass
class InformationEstimatorSklearnParams(BaseConfig):
    """Sklearn MI estimator hyperparameters (histogram/KDE depending on dimensionality)."""

    n_bins: int = 20
    bandwidth: Optional[float] = None


@dataclass
class InformationEstimatorParams(BaseConfig):
    """Estimator selection and method-specific hyperparameters.

    - method: "kraskov", "decoder", "binned", "sklearn", "copula", "auto"
    """

    method: str = "kraskov"
    kraskov: InformationEstimatorKraskovParams = field(
        default_factory=InformationEstimatorKraskovParams
    )
    binned: InformationEstimatorBinnedParams = field(
        default_factory=InformationEstimatorBinnedParams
    )
    decoder: InformationEstimatorDecoderParams = field(
        default_factory=InformationEstimatorDecoderParams
    )
    copula: InformationEstimatorCopulaParams = field(
        default_factory=InformationEstimatorCopulaParams
    )
    sklearn: InformationEstimatorSklearnParams = field(
        default_factory=InformationEstimatorSklearnParams
    )


@dataclass
class InformationBaselinesRandomWeightsParams(BaseConfig):
    """Random-weight baseline for LDA aggregation (produces *_sh metrics).

    This baseline generates random non-negative weights (with the same
    normalization/constraints as LDA weights) and recomputes the aggregated
    signals before running MI. It is **not** a permutation/null baseline.
    """

    n_shuffles: int = 0  # int >= 0; 0 disables
    seed: int = 0


@dataclass
class InformationBaselinesLabelShuffleNullParams(BaseConfig):
    """Permutation null baseline for MI/CMI terms (produces *_null metrics).

    This is a finite-sample bias / null diagnostic: it permutes the label vector
    (or, more generally, permutes one variable across samples) and recomputes MI/CMI
    using the same estimator selected by ``estimator.method``.

    You can control which metric groups receive a *_null counterpart via
    ``components``.

    - components=["basic"]: add *_null for basic MI terms (typically I(*;C))
    - components=["pairwise"]: add *_null for pairwise MI terms like I(E;I), I(E;Vout)
    - components=["conditional"]: add *_null for conditional MI terms like I(E;Vout|C)
    - components=["all"]: apply to all available groups
    """

    n_shuffles: int = 0  # int >= 0; 0 disables
    seed: int = 0
    components: list[str] = field(
        default_factory=lambda: ["basic"]
    )  # "basic", "pairwise", "conditional", "ablation_aligned", "layer_proxies", "upstream_unique_cmi", "all"


@dataclass
class InformationBaselinesParams(BaseConfig):
    """Baseline configuration for information analysis.

    All baselines use the same estimator selected by ``estimator.method``. They
    differ only in how surrogate data are generated:
    - random_weights: random-weight aggregation baseline (*_sh)
    - label_shuffle_null: label-permutation null (*_null)
    """

    random_weights: InformationBaselinesRandomWeightsParams = field(
        default_factory=InformationBaselinesRandomWeightsParams
    )
    label_shuffle_null: InformationBaselinesLabelShuffleNullParams = field(
        default_factory=InformationBaselinesLabelShuffleNullParams
    )


@dataclass
class InformationAnalysisParams(BaseConfig):
    """Parameters for information-theoretic analysis.

    Core string options (new grouped structure):
    - estimator.method: "kraskov", "decoder", "binned", "sklearn", "copula", "auto"
    - estimator.copula.type (method="copula"): "gaussian", "kernel", "dvc"
      (experimental; "dvc" is a placeholder and currently falls back to the simple copula path)
    - compute.computation_level: "single_branch", "layer_branch", "all_branch"
    - compute.branch_aggregation (layer_branch/all_branch): "mean", "multivariate", "sample"
    - pid.binarize_method: "median", "mean", "quantile" (PID is currently disabled)

    Baselines are computed with the same ``estimator.method``; baseline settings
    only control how surrogate data are constructed and how many repeats to run.

    Preferred (list-based) selectors:
    - selection.components: which families of metrics to compute
      - "basic": I(E;C), I(I;C), I(Vout;C)
      - "pairwise": I(E;I), I(E;Vout), I(I;Vout)
      - "conditional": conditional MI terms like I(E;C|I,Vb)
      - "pid": partial information decomposition (if available)
      - "ablation_aligned": extra conditional terms designed to mirror ablations,
        such as I(Vout;C|E), I(Vout;C|I), or I(Vout;C|E,I)
      - "layer_proxies": layer-total proxies derived from single-branch metrics
      - "upstream_unique_cmi": adds I(Vb;C|E,I) (can be expensive)
      - "soma_coupling": branch-current and branch-output information about the
        corresponding soma output
    - selection.scopes: which additional breakdowns to run
      - "layer": per-layer breakdown
      - "einet": separate excitatory vs inhibitory pathway summaries
      - "neuron": per-neuron breakdown (expensive)
    - selection.signal_variants: which signal variants to compute/plot
      - "network": DendriNet signals
      - "lda": *_lin variants using LDA projections
      - "random": *_sh random-weight baseline (requires baselines.random_weights.n_shuffles > 0)

    If any list-based selector is non-empty, it overrides the corresponding legacy
    boolean flags.

    Decoder-specific options (method="decoder"):
    - decoder_continuous_strategy: "none" (skip), "binned" (discretize+count), "gaussian" (Gaussian MI)
    - decoder_continuous_max_total_dim: maximum total dimension for continuous MI/CMI under decoder
    - decoder_gaussian_ridge: small diagonal ridge for covariance stability (gaussian strategy)

    Binned-specific options (method="binned"):
    - binning_strategy: "quantile" or "uniform"
    - binning_bias_correction: "none" or "miller_madow"
    """

    # ------------------------------------------------------------------
    # New (preferred) grouped structure
    # ------------------------------------------------------------------
    selection: InformationSelectionParams = field(
        default_factory=InformationSelectionParams
    )
    compute: InformationComputeParams = field(default_factory=InformationComputeParams)
    estimator: InformationEstimatorParams = field(
        default_factory=InformationEstimatorParams
    )
    baselines: InformationBaselinesParams = field(
        default_factory=InformationBaselinesParams
    )
    pid: InformationPIDParams = field(default_factory=InformationPIDParams)

    # ------------------------------------------------------------------
    # Legacy flat keys (DEPRECATED; still supported for backward compatibility).
    #
    # MIGRATION GUIDE -- use the grouped sub-dataclasses above instead:
    #   method, n_neighbors, n_bins, copula_type, decoder_*, binning_*
    #       -> estimator.method, estimator.kraskov.n_neighbors, etc.
    #   max_samples, include_vinf, normalize_mi, verbose, computation_level,
    #   branch_aggregation, per_neuron_analysis, per_einet_analysis,
    #   per_layer_analysis, compute_lda_weights
    #       -> compute.<field>
    #   components, scopes, views, signal_variants
    #       -> selection.<field>
    #   pid_binarize, pid_binarize_method, pid_binarize_threshold
    #       -> pid.<field>
    #   mi_null_shuffles, mi_null_seed, lda_n_shuffles
    #       -> baselines.<field>
    # ------------------------------------------------------------------

    # Estimation method: "kraskov", "decoder", "binned", "sklearn", "copula", "auto"
    method: str = "kraskov"

    # Optional: limit the number of samples to analyze
    max_samples: Optional[int] = None

    # Method-specific parameters
    n_neighbors: int = 10  # For Kraskov
    n_bins: int = 20  # For binned estimator + sklearn histogram
    copula_type: str = "gaussian"  # For copula: "gaussian", "kernel", "dvc"

    # Decoder (method="decoder") hyperparameters
    decoder_cv_folds: int = 5
    decoder_C: float = 1.0
    decoder_standardize: bool = True
    decoder_seed: int = 0
    decoder_continuous_strategy: str = "none"  # "none", "binned", "gaussian"
    decoder_continuous_max_total_dim: int = (
        2  # only used if decoder_continuous_strategy != "none"
    )
    decoder_gaussian_ridge: float = 1e-6  # covariance ridge for gaussian strategy

    # Binned (method="binned") hyperparameters
    binning_strategy: str = "quantile"  # "quantile" or "uniform"
    binning_bias_correction: str = "none"  # "none" or "miller_madow"

    # Optional: estimate finite-sample (estimator) bias via a shuffled-null baseline.
    #
    # If > 0, the analyzer will compute MI again after permuting the target across
    # samples (e.g. shuffling class labels) and report the null mean/std as a
    # diagnostic. This is separate from `lda_n_shuffles`, which creates random
    # *signal variants* (random-weight baselines).
    mi_null_shuffles: int = 0  # int >= 0; 0 disables
    mi_null_seed: int = 0  # RNG seed for the null permutations

    # Which metrics to compute
    compute_basic_mi: bool = True  # I(E;C), I(I;C), I(Vout;C)
    compute_pairwise_mi: bool = True  # I(E;I), I(E;Vout), I(I;Vout)
    compute_conditional_mi: bool = True  # I(E;I|C), I(E;Vout|C), etc.

    # New (preferred): list-based selection of analysis components.
    #
    # If non-empty, this overrides the individual compute_* boolean flags above.
    # Supported entries (synonyms are accepted by the analyzer):
    # - "basic" / "basic_mi"
    # - "pairwise" / "pairwise_mi"
    # - "conditional" / "conditional_mi"
    # - "pid"
    # - "ablation_aligned"
    # - "layer_proxies" / "layer_total_proxies"
    # - "upstream_unique_cmi"
    # - "soma_coupling" / "branch_soma" / "parent_soma"
    components: list[str] = field(default_factory=list)

    # Ablation-aligned information components (cheap extra CMI terms)
    # These are designed to better track "remove E/I/Vb/all" style ablations.
    compute_ablation_aligned_mi: bool = True

    # Optional extra upstream conditionals (can be expensive for high-dimensional synapse sets)
    # - Adds I(Vb;C|E,I) when Vb exists.
    compute_upstream_unique_cmi: bool = False

    # Layer-total proxies derived from single-branch MI (no high-dim multivariate MI)
    # Only used in computation_level="single_branch".
    # Produces additional per-layer metrics like:
    #   I(E;C|I,Vb)_sum, I(E;C|I,Vb)_union, I(E;C|I,Vb)_topK_sum, ...
    compute_layer_total_proxies: bool = True
    layer_total_topk: int = 10  # K used for *_topK_sum metrics

    # New (preferred): list-based selection of analysis "scopes" (breakdowns).
    #
    # If non-empty, this overrides the per_* boolean flags below.
    # Supported entries:
    # - "neuron" / "per_neuron"
    # - "einet" / "per_einet"
    # - "layer" / "per_layer"
    scopes: list[str] = field(default_factory=list)

    # Backward-compatible alias (deprecated): prefer `scopes`.
    views: list[str] = field(default_factory=list)

    # PID settings
    pid_binarize: bool = True
    pid_binarize_method: str = "median"  # "median", "mean", "quantile"
    pid_binarize_threshold: float = 0.5

    # Additional options
    include_vinf: bool = True  # Include soma activation
    normalize_mi: bool = False  # Normalize by entropy
    verbose: bool = False

    # Computation Level Control - determines the granularity of information analysis
    computation_level: str = (
        "single_branch"  # Options: "single_branch", "layer_branch", "all_branch"
    )
    # "single_branch": Individual branches (most granular and accurate)
    # "layer_branch": Layer-level aggregation (treats each layer as a unit)
    # "all_branch": Full network aggregation (most efficient)

    # Branch aggregation method - only used when computation_level is "layer_branch" or "all_branch"
    branch_aggregation: str = "mean"  # Options: "mean", "multivariate", "sample"

    # Per-neuron analysis - return information values per individual neuron
    per_neuron_analysis: bool = False  # If True, compute MI for each neuron separately

    # Per-EI-network analysis - analyze excitatory and inhibitory networks separately
    per_einet_analysis: bool = False  # If True, compute separate E and I network MI

    # Per-layer analysis - compute enhanced analysis for each layer separately
    per_layer_analysis: bool = False  # If True, compute per-layer enhanced analysis

    # Compute LDA-based versions of E, I, Vb
    compute_lda_weights: bool = False  # If True, compute E_lin, I_lin, Vb_lin using LDA

    # New (preferred): choose which "signal variants" to compute/plot.
    #
    # If non-empty, this overrides compute_lda_weights / lda_n_shuffles behavior:
    # - Always include "network" (DendriNet) metrics.
    # - Add "lda" to compute *_lin variants.
    # - Add "random" (or "shuffle") to compute *_sh variants (requires lda_n_shuffles > 0).
    signal_variants: list[str] = field(default_factory=list)

    # Number of random weight shuffles for LDA baseline comparison
    # If > 0, computes MI with random weights (same normalization as LDA) as a baseline
    lda_n_shuffles: int = 0


@dataclass
class InformationAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: InformationAnalysisParams = field(default_factory=InformationAnalysisParams)


@dataclass
class RepresentationProjectionConfig(BaseConfig):
    """Label-independent feature projection for population analyses."""

    name: str = "native"
    type: str = "identity"  # identity | balanced_countsketch
    output_dim: Optional[int] = None
    draws: int = 1

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.type = str(self.type).lower()
        if not self.name:
            raise ValueError("representation projection name cannot be empty")
        if self.type not in {"identity", "balanced_countsketch"}:
            raise ValueError(f"unsupported representation projection: {self.type}")
        if self.type == "balanced_countsketch":
            if self.output_dim is None or int(self.output_dim) < 1:
                raise ValueError("balanced_countsketch requires a positive output_dim")
            self.output_dim = int(self.output_dim)
        if int(self.draws) < 1:
            raise ValueError("representation projection draws must be positive")
        self.draws = int(self.draws)


@dataclass
class RepresentationProbeConfig(BaseConfig):
    """One fixed-capacity held-out representation probe."""

    name: str = "linear"
    type: str = "ridge"  # ridge | mlp
    draws: int = 1
    max_projection_draws: Optional[int] = None
    n_label_shuffles: int = 20
    ridge_alpha: float = 1.0
    hidden_dims: list[int] = field(default_factory=lambda: [32, 16])
    activation: str = "relu"
    alpha: float = 1e-4
    learning_rate_init: float = 1e-3
    batch_size: int = 256
    max_iter: int = 300
    early_stopping: bool = True
    validation_fraction: float = 0.1
    n_iter_no_change: int = 20

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.type = str(self.type).lower()
        if self.type == "linear":
            self.type = "ridge"
        if not self.name or self.type not in {"ridge", "mlp"}:
            raise ValueError(f"unsupported representation probe: {self.type}")
        if int(self.draws) < 1 or int(self.n_label_shuffles) < 0:
            raise ValueError("probe draws must be positive and shuffles non-negative")
        self.draws = int(self.draws)
        self.n_label_shuffles = int(self.n_label_shuffles)
        if self.max_projection_draws is not None:
            if int(self.max_projection_draws) < 1:
                raise ValueError("max_projection_draws must be positive")
            self.max_projection_draws = int(self.max_projection_draws)
        self.hidden_dims = [int(width) for width in self.hidden_dims]


@dataclass
class RepresentationInformationConfig(BaseConfig):
    """Joint and per-soma class-information sensitivity analyses."""

    enabled: bool = False
    compute_joint: bool = True
    compute_per_soma: bool = False
    projection_names: list[str] = field(default_factory=lambda: ["matched16"])
    per_soma_representations: list[str] = field(
        default_factory=lambda: ["excitatory", "inhibitory"]
    )
    n_neighbors: int = 5
    n_label_shuffles: int = 20
    standardize: bool = True

    def __post_init__(self) -> None:
        if int(self.n_neighbors) < 1 or int(self.n_label_shuffles) < 1:
            raise ValueError(
                "information neighbors and label shuffles must be positive"
            )
        self.n_neighbors = int(self.n_neighbors)
        self.n_label_shuffles = int(self.n_label_shuffles)
        self.projection_names = [str(name) for name in self.projection_names]
        self.per_soma_representations = [
            str(name) for name in self.per_soma_representations
        ]


@dataclass
class RepresentationSubsetProbeConfig(BaseConfig):
    """Label-independent population-coordinate subset decoding.

    Each draw samples one permutation of the native representation coordinates
    without
    replacement.  Requested subset sizes are nested prefixes of that
    permutation, which pairs the capacity curve within a checkpoint while
    keeping coordinate selection independent of labels.  Coordinates are
    individual somas for a single E or I population; concatenated joint
    representations contain coordinates from both populations.
    """

    enabled: bool = False
    representations: list[str] = field(default_factory=lambda: ["excitatory"])
    subset_sizes: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    draws: int = 10
    probe: RepresentationProbeConfig = field(
        default_factory=lambda: RepresentationProbeConfig(
            name="subset_linear",
            type="ridge",
            n_label_shuffles=5,
        )
    )

    def __post_init__(self) -> None:
        self.representations = [str(name) for name in self.representations]
        if not self.representations:
            raise ValueError("subset decoding requires at least one representation")
        sizes = sorted({int(size) for size in self.subset_sizes})
        if not sizes or sizes[0] < 1:
            raise ValueError("subset sizes must contain positive integers")
        self.subset_sizes = sizes
        if int(self.draws) < 1:
            raise ValueError("subset decoding draws must be positive")
        self.draws = int(self.draws)
        if isinstance(self.probe, Mapping):
            self.probe = RepresentationProbeConfig(**dict(self.probe))
        elif not isinstance(self.probe, RepresentationProbeConfig):
            raise TypeError("subset probe must be a mapping")
        if self.probe.type != "ridge":
            raise ValueError("subset decoding currently requires a ridge probe")
        if self.probe.draws != 1:
            raise ValueError("subset ridge decoding requires exactly one probe draw")


@dataclass
class RepresentationGeometryConfig(BaseConfig):
    """Unsupervised population dimensionality diagnostics.

    The analyzer reports participation-ratio and entropy effective ranks for
    both the centered covariance spectrum and a correlation spectrum obtained
    after per-coordinate standardization.  Geometry is measured on validation
    representations only and never uses class labels.
    """

    enabled: bool = False
    representations: list[str] = field(default_factory=lambda: ["excitatory"])
    variance_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        self.representations = [str(name) for name in self.representations]
        if not self.representations:
            raise ValueError("population geometry requires a representation")
        self.variance_epsilon = float(self.variance_epsilon)
        if self.variance_epsilon <= 0:
            raise ValueError("population geometry variance_epsilon must be positive")


@dataclass
class RepresentationAccessibilityAnalysisParams(BaseConfig):
    """Config-driven analysis of stacked soma-population representations."""

    representations: list[str] = field(
        default_factory=lambda: ["excitatory", "inhibitory", "joint"]
    )
    projections: list[RepresentationProjectionConfig] = field(
        default_factory=lambda: [RepresentationProjectionConfig()]
    )
    probes: list[RepresentationProbeConfig] = field(
        default_factory=lambda: [RepresentationProbeConfig()]
    )
    information: RepresentationInformationConfig = field(
        default_factory=RepresentationInformationConfig
    )
    subset_decoding: RepresentationSubsetProbeConfig = field(
        default_factory=RepresentationSubsetProbeConfig
    )
    geometry: RepresentationGeometryConfig = field(
        default_factory=RepresentationGeometryConfig
    )
    max_samples: Optional[int] = None
    seed: int = 0

    def __post_init__(self) -> None:
        self.representations = [str(name) for name in self.representations]
        normalized_projections = []
        for value in self.projections:
            if isinstance(value, RepresentationProjectionConfig):
                normalized_projections.append(value)
            elif isinstance(value, Mapping):
                normalized_projections.append(
                    RepresentationProjectionConfig(**dict(value))
                )
            else:
                raise TypeError("representation projections must be mappings")
        self.projections = normalized_projections
        normalized_probes = []
        for value in self.probes:
            if isinstance(value, RepresentationProbeConfig):
                normalized_probes.append(value)
            elif isinstance(value, Mapping):
                normalized_probes.append(RepresentationProbeConfig(**dict(value)))
            else:
                raise TypeError("representation probes must be mappings")
        self.probes = normalized_probes
        if isinstance(self.information, Mapping):
            self.information = RepresentationInformationConfig(**dict(self.information))
        elif not isinstance(self.information, RepresentationInformationConfig):
            raise TypeError("representation information must be a mapping")
        if isinstance(self.subset_decoding, Mapping):
            self.subset_decoding = RepresentationSubsetProbeConfig(
                **dict(self.subset_decoding)
            )
        elif not isinstance(self.subset_decoding, RepresentationSubsetProbeConfig):
            raise TypeError("representation subset decoding must be a mapping")
        if isinstance(self.geometry, Mapping):
            self.geometry = RepresentationGeometryConfig(**dict(self.geometry))
        elif not isinstance(self.geometry, RepresentationGeometryConfig):
            raise TypeError("representation geometry must be a mapping")
        if not self.representations or not self.projections or not self.probes:
            raise ValueError(
                "representation analysis requires representations, projections, and probes"
            )
        if self.max_samples is not None and int(self.max_samples) < 1:
            raise ValueError("representation max_samples must be positive")
        self.seed = int(self.seed)


@dataclass
class RepresentationAccessibilityAnalysisConfig(BaseConfig):
    """Enable held-out linear/nonlinear and population-information analyses."""

    enabled: bool = False
    training: bool = False
    params: RepresentationAccessibilityAnalysisParams = field(
        default_factory=RepresentationAccessibilityAnalysisParams
    )


@dataclass
class VisionBoundaryDiagnosticsAnalysisParams(BaseConfig):
    """Matched diagnostics for a vision replacement and its exact boundary."""

    backbone: str = "alexnet"
    weights: str = "IMAGENET1K_V1"
    projection_dim: int = 128
    projection_seed: int = 0
    max_samples: Optional[int] = 1000
    zero_tolerance: float = 1e-8

    def __post_init__(self) -> None:
        self.backbone = str(self.backbone)
        self.weights = str(self.weights)
        self.projection_dim = int(self.projection_dim)
        self.projection_seed = int(self.projection_seed)
        if self.projection_dim < 1:
            raise ValueError("vision boundary projection_dim must be positive")
        if self.max_samples is not None:
            self.max_samples = int(self.max_samples)
            if self.max_samples < 2:
                raise ValueError("vision boundary max_samples must be at least two")
        self.zero_tolerance = float(self.zero_tolerance)
        if self.zero_tolerance < 0:
            raise ValueError("vision boundary zero_tolerance must be non-negative")


@dataclass
class VisionBoundaryDiagnosticsAnalysisConfig(BaseConfig):
    """Enable final-only matched vision-boundary diagnostics."""

    enabled: bool = False
    training: bool = False
    params: VisionBoundaryDiagnosticsAnalysisParams = field(
        default_factory=VisionBoundaryDiagnosticsAnalysisParams
    )


@dataclass
class SourceTuningSupportAnalysisParams(BaseConfig):
    """Split-safe source tuning aligned with exact sparse synaptic support."""

    pathways: list[str] = field(
        default_factory=lambda: ["ff_excitatory", "ff_inhibitory"]
    )
    target_polarities: list[str] = field(
        default_factory=lambda: ["excitatory", "inhibitory"]
    )
    image_shape: Optional[tuple[int, ...]] = None
    image_source_layers: list[int] = field(default_factory=list)
    image_source_names: list[str] = field(
        default_factory=lambda: ["input", "input_e", "input_i"]
    )
    image_binarization_threshold: float = 0.5
    continuous_activity_threshold: float = 0.0
    input_batch_index: int = 0
    label_batch_index: int = 1
    max_reference_samples: Optional[int] = None
    max_evaluation_samples: Optional[int] = None
    require_independent_splits: bool = True
    retain_coordinate_profiles: bool = False
    epsilon: float = 1e-12
    reference_split: str = "train"
    evaluation_split: str = "validation"

    def __post_init__(self) -> None:
        self.pathways = [str(value) for value in self.pathways]
        self.target_polarities = [str(value) for value in self.target_polarities]
        self.image_source_layers = [int(value) for value in self.image_source_layers]
        self.image_source_names = [str(value) for value in self.image_source_names]
        if self.image_shape is not None:
            self.image_shape = tuple(int(value) for value in self.image_shape)
        self.input_batch_index = int(self.input_batch_index)
        self.label_batch_index = int(self.label_batch_index)
        if self.input_batch_index < 0 or self.label_batch_index < 0:
            raise ValueError("source-tuning batch indices must be non-negative")
        self.reference_split = str(self.reference_split).lower()
        self.evaluation_split = str(self.evaluation_split).lower()
        allowed_splits = {"train", "validation", "test"}
        if self.reference_split not in allowed_splits:
            raise ValueError(
                "source-tuning reference_split must be train, validation, or test"
            )
        if self.evaluation_split not in allowed_splits:
            raise ValueError(
                "source-tuning evaluation_split must be train, validation, or test"
            )
        if self.require_independent_splits and (
            self.reference_split == self.evaluation_split
        ):
            raise ValueError(
                "source-tuning independent reference and evaluation splits must differ"
            )

    def to_analyzer_params(self) -> object:
        """Build the analyzer's immutable package-native options object."""
        from dendritic_modeling.analysis.tools.source_tuning import SourceTuningOptions

        return SourceTuningOptions(
            pathways=tuple(self.pathways),
            target_polarities=tuple(self.target_polarities),
            image_shape=self.image_shape,
            image_source_layers=tuple(self.image_source_layers),
            image_source_names=tuple(self.image_source_names),
            image_binarization_threshold=float(self.image_binarization_threshold),
            continuous_activity_threshold=float(self.continuous_activity_threshold),
            input_batch_index=self.input_batch_index,
            label_batch_index=self.label_batch_index,
            max_reference_samples=self.max_reference_samples,
            max_evaluation_samples=self.max_evaluation_samples,
            require_independent_splits=bool(self.require_independent_splits),
            retain_coordinate_profiles=bool(self.retain_coordinate_profiles),
            epsilon=float(self.epsilon),
        )


@dataclass
class SourceTuningSupportAnalysisConfig(BaseConfig):
    """Enable final-only source-tuning/support analysis."""

    enabled: bool = False
    training: bool = False
    params: SourceTuningSupportAnalysisParams = field(
        default_factory=SourceTuningSupportAnalysisParams
    )


@dataclass
class InputImageRegionConfig(BaseConfig):
    """One typed image-region definition for a paired input intervention."""

    name: str
    kind: str
    center_fraction: float = 0.5
    foreground_threshold: float = 0.5

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.kind = str(self.kind).lower()
        if not self.name:
            raise ValueError("input-region names must be non-empty")
        if self.kind not in {
            "central_square",
            "surround",
            "foreground",
            "background",
        }:
            raise ValueError(f"unknown input-region kind: {self.kind!r}")
        self.center_fraction = float(self.center_fraction)
        self.foreground_threshold = float(self.foreground_threshold)

    def to_image_region_spec(self) -> object:
        """Build the analyzer's immutable region specification."""
        from dendritic_modeling.analysis.tools.input_region_intervention import (
            ImageRegionSpec,
        )

        return ImageRegionSpec(
            name=self.name,
            kind=self.kind,
            center_fraction=self.center_fraction,
            foreground_threshold=self.foreground_threshold,
        )


@dataclass
class InputRegionInterventionAnalysisParams(BaseConfig):
    """Configuration for paired frozen-checkpoint image-region interventions.

    ``target_population_polarities`` selects the target polarity of dendritic
    branch-response records. Soma-population responses remain available for all
    populations so E/I propagation can be compared without conflating source
    current polarity with target-cell polarity. Records from E- and I-target
    trees are distinct estimands and must not be pooled.
    """

    image_shape: Optional[tuple[int, ...]] = None
    regions: list[InputImageRegionConfig] = field(default_factory=list)
    operations: list[str] = field(
        default_factory=lambda: ["remove_only", "retain_only"]
    )
    replacement_methods: list[str] = field(
        default_factory=lambda: ["zero", "reference_mean"]
    )
    reference_split: str = "train"
    evaluation_split: str = "test"
    max_reference_samples: Optional[int] = None
    max_evaluation_samples: Optional[int] = None
    target_population_polarities: list[str] = field(
        default_factory=lambda: ["excitatory"]
    )
    identity_soma_class_mapping: Optional[list[int]] = None
    identity_soma_population_name: Optional[str] = None
    identity_soma_network_layer_index: Optional[int] = None

    def __post_init__(self) -> None:
        if self.image_shape is not None:
            self.image_shape = tuple(int(value) for value in self.image_shape)
        normalized_regions = []
        for value in self.regions:
            if isinstance(value, InputImageRegionConfig):
                normalized_regions.append(value)
            elif isinstance(value, Mapping):
                normalized_regions.append(InputImageRegionConfig(**dict(value)))
            else:
                raise TypeError("input-region definitions must be mappings")
        self.regions = normalized_regions
        self.operations = [str(value).lower() for value in self.operations]
        self.replacement_methods = [
            str(value).lower() for value in self.replacement_methods
        ]
        self.target_population_polarities = [
            str(value).lower() for value in self.target_population_polarities
        ]
        if self.identity_soma_class_mapping is not None:
            self.identity_soma_class_mapping = [
                int(value) for value in self.identity_soma_class_mapping
            ]
        if self.identity_soma_population_name is not None:
            self.identity_soma_population_name = str(
                self.identity_soma_population_name
            ).strip()
        if self.identity_soma_network_layer_index is not None:
            self.identity_soma_network_layer_index = int(
                self.identity_soma_network_layer_index
            )
        self.reference_split = str(self.reference_split).lower()
        self.evaluation_split = str(self.evaluation_split).lower()
        allowed_splits = {"train", "validation", "test"}
        if self.reference_split not in allowed_splits:
            raise ValueError(
                "input-region reference_split must be train, validation, or test"
            )
        if self.evaluation_split not in allowed_splits:
            raise ValueError(
                "input-region evaluation_split must be train, validation, or test"
            )

    def to_analyzer_params(self) -> object:
        """Build immutable package-native settings and nested region specs."""
        from dendritic_modeling.analysis.tools.input_region_intervention import (
            InputRegionInterventionSettings,
        )

        if self.image_shape is None:
            raise ValueError("input-region intervention requires image_shape")
        if not self.regions:
            raise ValueError("input-region intervention requires at least one region")
        return InputRegionInterventionSettings(
            image_shape=self.image_shape,
            regions=tuple(region.to_image_region_spec() for region in self.regions),
            operations=tuple(self.operations),
            replacement_methods=tuple(self.replacement_methods),
            reference_split=self.reference_split,
            evaluation_split=self.evaluation_split,
            max_reference_samples=self.max_reference_samples,
            max_evaluation_samples=self.max_evaluation_samples,
            target_population_polarities=tuple(self.target_population_polarities),
            identity_soma_class_mapping=(
                None
                if self.identity_soma_class_mapping is None
                else tuple(self.identity_soma_class_mapping)
            ),
            identity_soma_population_name=self.identity_soma_population_name,
            identity_soma_network_layer_index=(self.identity_soma_network_layer_index),
        )


@dataclass
class InputRegionInterventionAnalysisConfig(BaseConfig):
    """Enable final-only paired image-region intervention analysis."""

    enabled: bool = False
    training: bool = False
    params: InputRegionInterventionAnalysisParams = field(
        default_factory=InputRegionInterventionAnalysisParams
    )


@dataclass
class SynapseTurnoverAnalysisParams(BaseConfig):
    """Parameters for synapse turnover analysis.

    - snapshot_interval: how often to snapshot connectivity (in epochs).
    - synaptic_analysis: whether to compute connectivity summaries.
    - save_synaptic_report: whether to write a detailed report to disk.
    """

    snapshot_interval: int = 1  # Snapshot frequency in epochs
    synaptic_analysis: bool = True  # Enable synaptic connectivity analysis
    save_synaptic_report: bool = True  # Save synaptic analysis report


@dataclass
class SynapseTurnoverAnalysisConfig(BaseConfig):
    """Enable/disable synapse turnover analysis."""

    enabled: bool = False
    training: bool = False
    params: SynapseTurnoverAnalysisParams = field(
        default_factory=SynapseTurnoverAnalysisParams
    )


@dataclass
class PoissonExposureTransferParams(BaseConfig):
    """Reproducible parameters for posthoc Poisson exposure transfer.

    This standalone parameter block is intentionally not an ``AnalysisConfig``
    field. Exposure transfer rebuilds multiple stochastic test datasets and
    coordinates draws across independently trained checkpoints, while
    ``AnalysisManager`` analyzes one fixed test split within one run. The
    posthoc analyzer accepts this dataclass directly and it can be saved to or
    loaded from YAML through :class:`BaseConfig`.
    """

    test_durations: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.3162, 1.0, 3.1623]
    )
    draws_per_duration: int = 5
    base_draw_seed: int = 73000
    batch_size: int = 256
    max_samples: Optional[int] = None
    fingerprint_realized_datasets: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.test_durations, (str, bytes)):
            raise TypeError("test_durations must be a sequence of positive values")
        self.test_durations = [float(value) for value in self.test_durations]
        if not self.test_durations:
            raise ValueError("At least one test duration is required")
        keys = []
        for duration in self.test_durations:
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    f"Test durations must be finite and positive, got {duration!r}"
                )
            key = round(duration * 1_000_000)
            if key <= 0:
                raise ValueError(
                    f"Test duration is below micro-duration resolution: {duration}"
                )
            keys.append(key)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Test durations collide at micro-duration precision: "
                f"{tuple(self.test_durations)}"
            )

        for name in ("draws_per_duration", "base_draw_seed", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"{name} must be an integer, got {value!r}")
            setattr(self, name, int(value))
        if not 1 <= self.draws_per_duration <= 100:
            raise ValueError("draws_per_duration must be in [1, 100]")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_samples is not None:
            if (
                isinstance(self.max_samples, bool)
                or int(self.max_samples) != self.max_samples
            ):
                raise ValueError(
                    f"max_samples must be an integer, got {self.max_samples!r}"
                )
            self.max_samples = int(self.max_samples)
            if self.max_samples <= 0:
                raise ValueError("max_samples must be positive")
        if not isinstance(self.fingerprint_realized_datasets, bool):
            raise TypeError("fingerprint_realized_datasets must be a bool")


@dataclass
class NoisePerturbationAnalysisParams(BaseConfig):
    """Parameters for noise perturbation analysis.

    Noise settings:
    - uniform_noise: enable uniform noise sweeps.
    - uniform_magnitudes: list of magnitudes for uniform noise.
    - gaussian_noise: enable Gaussian noise sweeps.
    - gaussian_sdevs: list of standard deviations for Gaussian noise.
    - n_samples: number of noisy replicates per magnitude/sdev.
    - clamp_range: optional ``[min, max]`` output clamp. Set to ``None`` for
      unnormalized neural activity where noise should not be clipped to an
      image-like range.

    Metric toggles (same semantics as PerformanceAnalysisParams):
    - accuracy, auc, categorical_loglikelihood, mse, cosine_similarity
    """

    uniform_noise: bool = True
    uniform_magnitudes: list[float] = field(default_factory=lambda: [0.1, 0.2])
    gaussian_noise: bool = True
    gaussian_sdevs: list[float] = field(default_factory=lambda: [0.1, 0.2])
    n_samples: int = 10
    clamp_range: Optional[list[float]] = field(default_factory=lambda: [0.0, 1.0])
    accuracy: bool = True
    auc: bool = True
    categorical_loglikelihood: bool = True
    mse: bool = True
    cosine_similarity: bool = True
    seed: int = 0


@dataclass
class NoisePerturbationAnalysisConfig(BaseConfig):
    """Enable/disable noise perturbation analysis."""

    enabled: bool = False
    training: bool = False
    params: NoisePerturbationAnalysisParams = field(
        default_factory=NoisePerturbationAnalysisParams
    )


@dataclass
class AdversarialRobustnessAnalysisParams(BaseConfig):
    """Parameters for adversarial robustness (FGSM) evaluation."""

    epsilons: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    )
    batch_size: int = 256


@dataclass
class AdversarialRobustnessAnalysisConfig(BaseConfig):
    """Enable/disable adversarial robustness analysis."""

    enabled: bool = False
    training: bool = False
    params: AdversarialRobustnessAnalysisParams = field(
        default_factory=AdversarialRobustnessAnalysisParams
    )


@dataclass
class AblationSelectionParams(BaseConfig):
    """Preferred (grouped) selectors for ablation analysis."""

    levels: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)


@dataclass
class AblationMethodsParams(BaseConfig):
    """Preferred (grouped) method settings for ablation analysis."""

    ablation_methods: list[str] = field(default_factory=lambda: ["lesion"])
    shuffle_seed: int = 0
    clamp_statistic: str = "mean"  # "mean" (supported), "median" (future)


@dataclass
class AblationAnalysisParams(BaseConfig):
    """Configuration for ablation / intervention analysis.

    Notes
    -----
    We support multiple ablation *methods*:
    - "lesion": structural ablation by zeroing the relevant connection weights
      (current behavior; can change baseline/gain and induce saturation/vanishing).
    - "shuffle": signal intervention that permutes the targeted signal across samples
      (destroys label-alignment while preserving marginal distribution).
    - "mean_clamp": signal intervention that replaces the targeted signal with its
      fixed training-dataset mean (preserves a fixed operating-point reference while
      removing trial-to-trial variability). Direct low-level hook use retains the
      legacy batch-mean fallback for compatibility.

    Core string options:
    - clamp_statistic: "mean" (supported), "median" (reserved)

    Preferred (list-based) selectors:
    - levels: what granularity to ablate
      - "layer": ablate whole dendritic layers
      - "compartment": ablate specific compartments (lesion-only)
    - targets: what signal/connection class to ablate
      - "all_synapses": excitatory + inhibitory together
      - "excitation": excitatory synapses only
      - "inhibition": inhibitory synapses only
      - "upstream": upstream branch input / Vb pathway
    - metrics: what to report
      - "accuracy", "auc", "categorical_loglikelihood", "mse", "cosine_similarity"
      - "pred_label_mi_bits": I(C; C_hat) in bits for predicted labels C_hat

    All ablation configuration is specified through the nested selection and methods
    fields below.
    """

    selection: AblationSelectionParams = field(default_factory=AblationSelectionParams)
    methods: AblationMethodsParams = field(default_factory=AblationMethodsParams)


@dataclass
class AblationAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: AblationAnalysisParams = field(default_factory=AblationAnalysisParams)


@dataclass
class PathMatchedInterventionAnalysisParams(BaseConfig):
    """Equal-coordinate causal interventions across dendritic depth.

    One complete soma-to-distal path is selected per soma and technical draw.
    Its prefixes give a fixed coordinate count at every depth.  Supports are
    selected without data or labels and shared across intervention targets.

    Current targets act before the compartment transfer/reactivation.  The
    optional ``post_gate_output`` target acts on the branch-layer output after
    that transformation and before it enters the next more-proximal layer.
    Depths are soma-relative and non-somatic: 1 is proximal, increasing values
    are more distal, and 0 is intentionally rejected.

    ``module_name_prefix`` selects one dendritic tree in a multi-layer model;
    leaving it unset is valid only when exactly one branch module exists at
    each soma-relative depth.  Optional expected morphology fields make
    publication analyses fail closed if checkpoint architecture drifts.
    """

    targets: list[str] = field(
        default_factory=lambda: [
            "excitation_current",
            "inhibition_current",
            "joint_EI_currents",
        ]
    )
    methods: list[str] = field(
        default_factory=lambda: [
            "zero",
            "fixed_train_mean",
            "global_test_shuffle",
        ]
    )
    depths: list[int] = field(default_factory=list)
    paths_per_soma: list[int] = field(default_factory=list)
    support_draws: int = 5
    support_seed: int = 271828
    support_namespace: str = "path-matched-depth-intervention-v1"
    shuffle_seed: int = 314159
    module_name_prefix: Optional[str] = None
    expected_n_somas: Optional[int] = None
    expected_branch_factors: list[int] = field(default_factory=list)
    chance_accuracy: Optional[float] = None
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        allowed_targets = {
            "excitation_current",
            "inhibition_current",
            "joint_EI_currents",
            "post_gate_output",
        }
        allowed_methods = {"zero", "fixed_train_mean", "global_test_shuffle"}
        self.targets = [str(value) for value in self.targets]
        self.methods = [str(value) for value in self.methods]
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("path-matched intervention targets must be unique")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("path-matched intervention methods must be unique")
        unknown_targets = sorted(set(self.targets) - allowed_targets)
        unknown_methods = sorted(set(self.methods) - allowed_methods)
        if unknown_targets:
            raise ValueError(f"unknown path-matched targets: {unknown_targets}")
        if unknown_methods:
            raise ValueError(f"unknown path-matched methods: {unknown_methods}")
        self.depths = sorted({int(value) for value in self.depths})
        if any(value < 1 for value in self.depths):
            raise ValueError("path-matched intervention depths must be dendritic (>=1)")
        self.paths_per_soma = sorted({int(value) for value in self.paths_per_soma})
        if any(value < 1 for value in self.paths_per_soma):
            raise ValueError("paths_per_soma must contain positive integers")
        self.support_draws = int(self.support_draws)
        if self.support_draws < 1:
            raise ValueError("support_draws must be positive")
        self.support_seed = int(self.support_seed)
        self.shuffle_seed = int(self.shuffle_seed)
        self.support_namespace = str(self.support_namespace)
        if not self.support_namespace:
            raise ValueError("support_namespace must be non-empty")
        if self.module_name_prefix is not None:
            self.module_name_prefix = str(self.module_name_prefix)
            if not self.module_name_prefix:
                self.module_name_prefix = None
        if self.expected_n_somas is not None:
            self.expected_n_somas = int(self.expected_n_somas)
            if self.expected_n_somas < 1:
                raise ValueError("expected_n_somas must be positive")
        self.expected_branch_factors = [
            int(value) for value in self.expected_branch_factors
        ]
        if any(value < 1 for value in self.expected_branch_factors):
            raise ValueError("expected_branch_factors must be positive")
        if self.chance_accuracy is not None:
            self.chance_accuracy = float(self.chance_accuracy)
            if not 0.0 <= self.chance_accuracy < 1.0:
                raise ValueError("chance_accuracy must lie in [0, 1)")
        if self.max_samples is not None:
            self.max_samples = int(self.max_samples)
            if self.max_samples < 1:
                raise ValueError("max_samples must be positive")


@dataclass
class PathMatchedInterventionAnalysisConfig(BaseConfig):
    """Enable the final-only path-matched dendritic intervention analyzer."""

    enabled: bool = False
    training: bool = False
    params: PathMatchedInterventionAnalysisParams = field(
        default_factory=PathMatchedInterventionAnalysisParams
    )


@dataclass
class BranchLocalSensitivityAnalysisParams(BaseConfig):
    """Observed local branch sensitivities including fitted reactivation gain.

    ``module_name_prefix`` identifies exactly one dendritic tree.  Depths use
    the soma-relative convention: zero is the soma, one is proximal, and
    larger values are more distal. The soma is excluded unless
    ``include_soma`` is explicitly enabled.
    """

    depths: list[int] = field(default_factory=list)
    module_name_prefix: Optional[str] = None
    expected_n_somas: Optional[int] = None
    expected_branch_factors: list[int] = field(default_factory=list)
    max_samples: Optional[int] = None
    include_soma: bool = False

    def __post_init__(self) -> None:
        self.depths = sorted({int(value) for value in self.depths})
        minimum_depth = 0 if self.include_soma else 1
        if any(value < minimum_depth for value in self.depths):
            qualifier = (
                "soma-relative (>=0)" if self.include_soma else "dendritic (>=1)"
            )
            raise ValueError(f"branch local sensitivity depths must be {qualifier}")
        if self.module_name_prefix is not None:
            self.module_name_prefix = str(self.module_name_prefix) or None
        if self.expected_n_somas is not None:
            self.expected_n_somas = int(self.expected_n_somas)
            if self.expected_n_somas < 1:
                raise ValueError("expected_n_somas must be positive")
        self.expected_branch_factors = [
            int(value) for value in self.expected_branch_factors
        ]
        if any(value < 1 for value in self.expected_branch_factors):
            raise ValueError("expected_branch_factors must be positive")
        if self.max_samples is not None:
            self.max_samples = int(self.max_samples)
            if self.max_samples < 1:
                raise ValueError("max_samples must be positive")


@dataclass
class BranchLocalSensitivityAnalysisConfig(BaseConfig):
    """Enable the final-only branch local-sensitivity analyzer."""

    enabled: bool = False
    training: bool = False
    params: BranchLocalSensitivityAnalysisParams = field(
        default_factory=BranchLocalSensitivityAnalysisParams
    )


@dataclass
class LocalInhibitoryPopulationInterventionAnalysisParams(BaseConfig):
    """Whole-population local-I interventions across feedforward layers.

    Each analyzed layer must contain exactly one inhibitory population whose
    only outgoing route is one same-step ``ff_inhibitory`` projection to a
    local excitatory population. Replacing that population output therefore
    has the unambiguous semantics of replacing the complete local I-to-E
    signal, rather than a compartment current or a recurrent/lateral route.
    """

    methods: list[str] = field(
        default_factory=lambda: [
            "global_test_shuffle",
            "within_class_test_shuffle",
            "zero",
        ]
    )
    permutation_draws: int = 3
    shuffle_seed: int = 20260712
    expected_network_layers: Optional[int] = None
    expected_inhibitory_width: Optional[int] = None
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        allowed = {
            "global_test_shuffle",
            "within_class_test_shuffle",
            "zero",
        }
        self.methods = [str(value) for value in self.methods]
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("local-I intervention methods must be unique")
        unknown = sorted(set(self.methods) - allowed)
        if unknown:
            raise ValueError(f"unknown local-I intervention methods: {unknown}")
        self.permutation_draws = int(self.permutation_draws)
        if self.permutation_draws < 1:
            raise ValueError("permutation_draws must be positive")
        self.shuffle_seed = int(self.shuffle_seed)
        if self.expected_network_layers is not None:
            self.expected_network_layers = int(self.expected_network_layers)
            if self.expected_network_layers < 1:
                raise ValueError("expected_network_layers must be positive")
        if self.expected_inhibitory_width is not None:
            self.expected_inhibitory_width = int(self.expected_inhibitory_width)
            if self.expected_inhibitory_width < 1:
                raise ValueError("expected_inhibitory_width must be positive")
        if self.max_samples is not None:
            self.max_samples = int(self.max_samples)
            if self.max_samples < 1:
                raise ValueError("max_samples must be positive")


@dataclass
class LocalInhibitoryPopulationInterventionAnalysisConfig(BaseConfig):
    """Enable the final-only feedforward local-I population analyzer."""

    enabled: bool = False
    training: bool = False
    params: LocalInhibitoryPopulationInterventionAnalysisParams = field(
        default_factory=LocalInhibitoryPopulationInterventionAnalysisParams
    )


@dataclass
class PathMatchedInformationAnalysisParams(BaseConfig):
    """Matched-width class information along one selected dendritic tree.

    Complete soma-to-distal paths are selected without inspecting examples,
    labels, activations, mechanisms, or checkpoints.  Prefixes of each path
    provide the same number of compartment coordinates at every non-somatic
    depth.  The soma has one full-population support; non-somatic path draws
    are technical support-sensitivity draws and are never model replicates.

    The analyzer estimates joint-vector continuous--discrete mutual
    information after label-free, per-coordinate standardization.  It is a
    final-only validation-split analysis; test examples are not accepted by
    its manager route.
    """

    signals: list[str] = field(
        default_factory=lambda: [
            "excitation_current",
            "inhibition_current",
            "joint_EI_currents",
            "upstream_child_current",
            "pre_gate_voltage",
            "post_gate_output",
        ]
    )
    depths: list[int] = field(default_factory=list)
    support_draws: int = 5
    support_seed: int = 271828
    support_namespace: str = "path-matched-depth-intervention-v1"
    information_seed: int = 314159
    add_probe_seed_offset: bool = True
    n_neighbors: int = 10
    n_label_shuffles: int = 20
    module_name_prefix: Optional[str] = None
    expected_n_somas: Optional[int] = None
    expected_branch_factors: list[int] = field(default_factory=list)
    max_samples: Optional[int] = None

    def __post_init__(self) -> None:
        allowed_signals = {
            "excitation_current",
            "inhibition_current",
            "joint_EI_currents",
            "upstream_child_current",
            "pre_gate_voltage",
            "post_gate_output",
        }
        self.signals = [str(value) for value in self.signals]
        if not self.signals or len(set(self.signals)) != len(self.signals):
            raise ValueError("path-matched information signals must be unique")
        unknown_signals = sorted(set(self.signals) - allowed_signals)
        if unknown_signals:
            raise ValueError(
                f"unknown path-matched information signals: {unknown_signals}"
            )
        self.depths = sorted({int(value) for value in self.depths})
        if any(value < 0 for value in self.depths):
            raise ValueError("path-matched information depths must be non-negative")
        self.support_draws = int(self.support_draws)
        if self.support_draws < 1:
            raise ValueError("support_draws must be positive")
        self.support_seed = int(self.support_seed)
        self.information_seed = int(self.information_seed)
        self.add_probe_seed_offset = bool(self.add_probe_seed_offset)
        self.support_namespace = str(self.support_namespace)
        if not self.support_namespace:
            raise ValueError("support_namespace must be non-empty")
        self.n_neighbors = int(self.n_neighbors)
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        self.n_label_shuffles = int(self.n_label_shuffles)
        if self.n_label_shuffles < 1:
            raise ValueError("n_label_shuffles must be positive")
        if self.module_name_prefix is not None:
            self.module_name_prefix = str(self.module_name_prefix)
            if not self.module_name_prefix:
                self.module_name_prefix = None
        if self.expected_n_somas is not None:
            self.expected_n_somas = int(self.expected_n_somas)
            if self.expected_n_somas < 1:
                raise ValueError("expected_n_somas must be positive")
        self.expected_branch_factors = [
            int(value) for value in self.expected_branch_factors
        ]
        if any(value < 1 for value in self.expected_branch_factors):
            raise ValueError("expected_branch_factors must be positive")
        if self.max_samples is not None:
            self.max_samples = int(self.max_samples)
            if self.max_samples < 1:
                raise ValueError("max_samples must be positive")


@dataclass
class PathMatchedInformationAnalysisConfig(BaseConfig):
    """Enable final-only, validation-set path-matched class information."""

    enabled: bool = False
    training: bool = False
    params: PathMatchedInformationAnalysisParams = field(
        default_factory=PathMatchedInformationAnalysisParams
    )


@dataclass
class WeightAnalysisParams(BaseConfig):
    """Parameters for weight analysis.

    Core string options:
    - computation_level: "single_branch", "layer_branch", "all_branch"
    - branch_aggregation (layer_branch/all_branch): "mean", "multivariate", "sample"

    Scope toggles:
    - per_layer_analysis, per_einet_analysis, per_neuron_analysis

    Content toggles:
    - synapse_weights, branch_weights
    - analyze_excitatory, analyze_inhibitory, analyze_branch_output

    Summary statistic toggles:
    - compute_mean, compute_variance, compute_percentiles, compute_min_max,
      compute_dendritic_strength

    Source-support analysis:
    - compute_source_support: emit exact sparse contact-frequency and effective
      conductance profiles by network layer, target population, soma, dendritic
      depth, and synaptic pathway.
    - source_support_target_polarities / source_support_pathways: restrict the
      target populations and pathways included in that analysis.
    - source_support_image_shape: optional image-space shape used for the
      package-native audit plot and source-region alignment; saved support
      profiles remain one-dimensional.
    - source_support_plot_soma_indices: optional soma subset for that plot.

    Source-region alignment:
    - compute_source_region_alignment: align exact image-space source-support
      profiles with class-conditional activity in the configured reference split.
    - source_region_soma_to_class: explicit class label for every target soma.
    - source_region_reference_split: ``train``, ``validation``, or ``test``;
      defaults to validation so final checkpoint analysis does not fit its
      publication template on the test set.
    - source_region_image_network_layers: network-layer indices whose source
      coordinates are explicitly declared to be image pixels. Later layers are
      skipped even if their width happens to equal the image pixel count.
    - source_region_foreground_thresholds: pixel thresholds used to define
      sample-dependent foreground (``x > threshold``) and background.
    - source_region_center_bounds: optional half-open ``[r0, r1, c0, c1]``
      center rectangle. When omitted, the centered half-height/half-width
      rectangle is used.
    - source_region_max_reference_samples: optional deterministic cap on the
      reference examples used to estimate class-conditional pixel occupancy.
    """

    synapse_weights: bool = True  # Analyze excitatory and inhibitory synapse weights
    branch_weights: bool = True  # Analyze branch-to-output connection weights

    computation_level: str = (
        "single_branch"  # Options: "single_branch", "layer_branch", "all_branch"
    )
    branch_aggregation: str = "mean"  # Options: "mean", "multivariate", "sample"

    # Enhanced analysis modes
    per_layer_analysis: bool = True  # Analyze each layer separately (soma to distal)
    per_einet_analysis: bool = (
        True  # Separate excitatory and inhibitory network analysis
    )
    per_neuron_analysis: bool = False  # Analyze individual neurons (expensive)

    analyze_excitatory: bool = True  # Analyze excitatory synaptic weights
    analyze_inhibitory: bool = True  # Analyze inhibitory synaptic weights
    analyze_branch_output: bool = True  # Analyze branch-to-output weights

    compute_mean: bool = True  # Compute mean weights per branch
    compute_variance: bool = True  # Compute variance of weights per branch
    compute_percentiles: bool = True  # Compute percentiles (median, quartiles)
    compute_min_max: bool = True  # Compute min/max per layer
    compute_dendritic_strength: bool = (
        True  # Compute total dendritic strength per layer
    )

    weight_threshold: float = 1e-6  # Minimum weight to consider as "active"
    compute_source_support: bool = False
    source_support_target_polarities: list[str] = field(
        default_factory=lambda: ["excitatory"]
    )
    source_support_pathways: list[str] = field(
        default_factory=lambda: ["ff_excitatory", "ff_inhibitory"]
    )
    source_support_image_shape: Optional[tuple[int, int]] = None
    source_support_plot_soma_indices: list[int] = field(default_factory=list)
    source_support_fig_save_format: str = "pdf"
    compute_source_region_alignment: bool = False
    source_region_soma_to_class: list[int] = field(default_factory=list)
    source_region_reference_split: str = "validation"
    source_region_image_network_layers: list[int] = field(default_factory=lambda: [0])
    source_region_foreground_thresholds: list[float] = field(
        default_factory=lambda: [0.0, 0.5]
    )
    source_region_center_bounds: Optional[tuple[int, int, int, int]] = None
    source_region_max_reference_samples: Optional[int] = None

    def __post_init__(self) -> None:
        allowed_polarities = {"excitatory", "inhibitory"}
        allowed_pathways = {
            "ff_excitatory",
            "ff_inhibitory",
            "rec_excitatory",
            "rec_inhibitory",
        }
        self.source_support_target_polarities = [
            str(value).lower() for value in self.source_support_target_polarities
        ]
        self.source_support_pathways = [
            str(value).lower() for value in self.source_support_pathways
        ]
        unknown_polarities = sorted(
            set(self.source_support_target_polarities) - allowed_polarities
        )
        if unknown_polarities:
            raise ValueError(
                f"unknown source-support target polarities: {unknown_polarities}"
            )
        unknown_pathways = sorted(set(self.source_support_pathways) - allowed_pathways)
        if unknown_pathways:
            raise ValueError(f"unknown source-support pathways: {unknown_pathways}")
        if len(set(self.source_support_pathways)) != len(self.source_support_pathways):
            raise ValueError("source_support_pathways must be unique")
        self.source_support_plot_soma_indices = [
            int(value) for value in self.source_support_plot_soma_indices
        ]
        if any(value < 0 for value in self.source_support_plot_soma_indices):
            raise ValueError("source_support_plot_soma_indices must be non-negative")
        if self.source_support_image_shape is not None:
            self.source_support_image_shape = tuple(
                int(value) for value in self.source_support_image_shape
            )
            if len(self.source_support_image_shape) != 2 or any(
                value < 1 for value in self.source_support_image_shape
            ):
                raise ValueError(
                    "source_support_image_shape must contain two positive integers"
                )
        self.source_support_fig_save_format = str(
            self.source_support_fig_save_format
        ).lstrip(".")
        if not self.source_support_fig_save_format:
            raise ValueError("source_support_fig_save_format cannot be empty")
        self.compute_source_region_alignment = bool(
            self.compute_source_region_alignment
        )
        self.source_region_soma_to_class = [
            int(value) for value in self.source_region_soma_to_class
        ]
        if any(value < 0 for value in self.source_region_soma_to_class):
            raise ValueError("source_region_soma_to_class must be non-negative")
        self.source_region_reference_split = str(
            self.source_region_reference_split
        ).lower()
        if self.source_region_reference_split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError(
                "source_region_reference_split must be train, validation, or test"
            )
        self.source_region_image_network_layers = [
            int(value) for value in self.source_region_image_network_layers
        ]
        if not self.source_region_image_network_layers:
            raise ValueError("source_region_image_network_layers cannot be empty")
        if any(value < -1 for value in self.source_region_image_network_layers):
            raise ValueError(
                "source_region_image_network_layers must contain indices >= -1"
            )
        if len(set(self.source_region_image_network_layers)) != len(
            self.source_region_image_network_layers
        ):
            raise ValueError("source_region_image_network_layers must be unique")
        self.source_region_foreground_thresholds = [
            float(value) for value in self.source_region_foreground_thresholds
        ]
        if not self.source_region_foreground_thresholds:
            raise ValueError("source_region_foreground_thresholds cannot be empty")
        if not all(
            math.isfinite(value) for value in self.source_region_foreground_thresholds
        ):
            raise ValueError(
                "source_region_foreground_thresholds must contain finite values"
            )
        if len(set(self.source_region_foreground_thresholds)) != len(
            self.source_region_foreground_thresholds
        ):
            raise ValueError("source_region_foreground_thresholds must be unique")
        if self.source_region_center_bounds is not None:
            self.source_region_center_bounds = tuple(
                int(value) for value in self.source_region_center_bounds
            )
            if len(self.source_region_center_bounds) != 4:
                raise ValueError(
                    "source_region_center_bounds must contain four integers"
                )
            if self.source_support_image_shape is None:
                raise ValueError(
                    "source_region_center_bounds requires source_support_image_shape"
                )
            r0, r1, c0, c1 = self.source_region_center_bounds
            height, width = self.source_support_image_shape
            if not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width):
                raise ValueError(
                    "source_region_center_bounds must lie inside the image shape"
                )
        if self.source_region_max_reference_samples is not None:
            self.source_region_max_reference_samples = int(
                self.source_region_max_reference_samples
            )
            if self.source_region_max_reference_samples < 1:
                raise ValueError("source_region_max_reference_samples must be positive")
        if self.compute_source_region_alignment:
            if not self.compute_source_support:
                raise ValueError(
                    "compute_source_region_alignment requires compute_source_support"
                )
            if self.source_support_image_shape is None:
                raise ValueError(
                    "compute_source_region_alignment requires "
                    "source_support_image_shape"
                )
            if not self.source_region_soma_to_class:
                raise ValueError(
                    "compute_source_region_alignment requires an explicit "
                    "source_region_soma_to_class mapping"
                )


@dataclass
class WeightAnalysisConfig(BaseConfig):
    """Enable/disable weight analysis."""

    enabled: bool = False
    training: bool = False
    params: WeightAnalysisParams = field(default_factory=WeightAnalysisParams)


@dataclass
class CompartmentStatisticsAnalysisParams(BaseConfig):
    """Parameters for compartment statistics analysis.

    Toggles:
    - synapse_weights / branch_weights: include weight statistics.
    - inputs / activations: include signal statistics.
    - global_analysis / layer_analysis / branch_analysis: which aggregation levels.
    - compute_mean / compute_variance / compute_percentiles / compute_min_max /
      compute_entropy: which summary statistics.
    - per_class_analysis: split statistics by class label.

    Sampling:
    - n_samples: optional limit on number of samples processed.
    """

    synapse_weights: bool = True  # Analyze excitatory and inhibitory synapse weights
    branch_weights: bool = True  # Analyze branch-to-output connection weights

    inputs: bool = True  # Analyze inputs
    activations: bool = True  # Analyze activations

    global_analysis: bool = True  # Analyze global statistics
    layer_analysis: bool = True  # Analyze layer statistics
    branch_analysis: bool = True  # Analyze branch statistics

    compute_mean: bool = True  # Compute mean weights per branch
    compute_variance: bool = True  # Compute variance of weights per branch
    compute_percentiles: bool = True  # Compute percentiles (median, quartiles)
    compute_min_max: bool = True  # Compute min/max per layer
    compute_entropy: bool = True  # Compute entropy per layer

    per_class_analysis: bool = False  # Analyze statistics separately for each class
    n_samples: Optional[int] = None  # Number of samples to analyze
    seed: int = 0


@dataclass
class CompartmentStatisticsAnalysisConfig(BaseConfig):
    """Enable/disable compartment statistics analysis."""

    enabled: bool = False
    training: bool = False
    params: CompartmentStatisticsAnalysisParams = field(
        default_factory=CompartmentStatisticsAnalysisParams
    )


@dataclass
class CompartmentSNRAnalysisParams(BaseConfig):
    """Parameters for compartment SNR analysis.

    Toggles which aggregation levels to compute:
    - input_analysis
    - global_analysis
    - layer_analysis
    - branch_analysis
    """

    input_analysis: bool = True  # Analyze input-level statistics
    input_capture_mode: str = "branch"  # "branch" or "network"
    global_analysis: bool = True  # Analyze global statistics
    layer_analysis: bool = True  # Analyze layer statistics
    branch_analysis: bool = True  # Analyze branch statistics

    global_aggregation: bool = True  # Aggregate global statistics
    layer_aggregation: bool = True  # Aggregate layer statistics
    branch_aggregation: bool = True  # Aggregate branch statistics

    epsilon: float = 1e-6


@dataclass
class CompartmentSNRAnalysisConfig(BaseConfig):
    """Enable/disable compartment SNR analysis."""

    enabled: bool = False
    training: bool = False
    params: CompartmentSNRAnalysisParams = field(
        default_factory=CompartmentSNRAnalysisParams
    )


@dataclass
class MultiplicativeGainAnalysisParams(BaseConfig):
    uniform_gain: bool = True
    gain_min: float = 0.0
    gain_max: float = 10.0
    n_steps: int = 10
    logspace: bool = False
    input_preprocess: str = (
        "none"  # "none", "relu", "softplus", "shift_min", "global_shift_min"
    )
    noise_model: str = "none"  # "none", "poisson"
    poisson_eps: float = 1e-6
    accuracy: bool = False
    auc: bool = False
    categorical_loglikelihood: bool = False
    mse: bool = False
    cosine_similarity: bool = False


@dataclass
class MultiplicativeGainAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: MultiplicativeGainAnalysisParams = field(
        default_factory=MultiplicativeGainAnalysisParams
    )


@dataclass
class GainLoadPerturbationAnalysisParams(BaseConfig):
    """Parameters for frozen-checkpoint shared-gain/load factorials.

    ``gaussian_input`` treats ``load_levels`` as independent additive-input
    standard deviations. ``positive_inhibitory`` treats them as the mean of a
    positive lognormal background added to an ordered inhibitory input stream.
    Neither mode should be described as denominator-only conductance load.
    """

    gain_log_sds: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.5, 0.8])
    load_levels: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.5])
    load_mode: str = "gaussian_input"
    load_log_sd: float = 0.2
    inhibitory_start: Optional[int] = None
    draws_per_condition: int = 5
    base_draw_seed: int = 94103
    max_samples: Optional[int] = None
    clamp_min: Optional[float] = 0.0
    clamp_max: Optional[float] = None
    accuracy: bool = True
    categorical_loglikelihood: bool = True
    record_input_statistics: bool = False
    record_branch_local_sensitivity: bool = False
    branch_sensitivity_module_name_prefix: Optional[str] = None
    branch_sensitivity_expected_n_somas: Optional[int] = None
    branch_sensitivity_include_soma: bool = False
    branch_sensitivity_max_samples: Optional[int] = None
    branch_sensitivity_draws: list[int] = field(default_factory=lambda: [0])


@dataclass
class GainLoadPerturbationAnalysisConfig(BaseConfig):
    """Enable frozen-checkpoint shared-gain/load factorial analysis."""

    enabled: bool = False
    training: bool = False
    params: GainLoadPerturbationAnalysisParams = field(
        default_factory=GainLoadPerturbationAnalysisParams
    )


@dataclass
class SynapticActivationAnalysisParams(BaseConfig):
    """Parameters for synaptic activation analysis.

    Toggles:
    - network_summary / layer_summary / branch_summary / raw_summary: what to report.
    - logspace: whether to plot/report in log-space.
    - downsample: whether to subsample activations for speed/memory.

    - samples: number of examples to keep when downsampling/raw summaries are enabled.
    """

    network_summary: bool = True
    layer_summary: bool = True
    branch_summary: bool = True
    raw_summary: bool = True
    logspace: bool = False
    downsample: bool = True
    samples: int = 5


@dataclass
class SynapticActivationAnalysisConfig(BaseConfig):
    """Enable/disable synaptic activation analysis."""

    enabled: bool = False
    training: bool = False
    params: SynapticActivationAnalysisParams = field(
        default_factory=SynapticActivationAnalysisParams
    )


@dataclass
class BranchActivationAnalysisParams(BaseConfig):
    """Parameters for branch activation analysis.

    Toggles:
    - synapse_activations / branch_activations: what signals to log.
    - network_summary / layer_summary / branch_summary / raw_summary: what to report.
    - logspace: whether to plot/report in log-space.
    - downsample: whether to subsample for speed/memory.

    - samples: number of examples to keep when downsampling/raw summaries are enabled.
    """

    synapse_activations: bool = True
    branch_activations: bool = True
    network_summary: bool = True
    layer_summary: bool = True
    branch_summary: bool = True
    raw_summary: bool = True
    logspace: bool = False
    downsample: bool = True
    samples: int = 5


@dataclass
class BranchActivationAnalysisConfig(BaseConfig):
    """Enable/disable branch activation analysis."""

    enabled: bool = False
    training: bool = False
    params: BranchActivationAnalysisParams = field(
        default_factory=BranchActivationAnalysisParams
    )


@dataclass
class LocalRuleComponentAnalysisParams(BaseConfig):
    """Parameters for LocalCA component trajectory analysis.

    The analyzer records the fast factors in the local rule by branch depth and
    synapse type: branch voltage, input resistance, E/I driving force, active
    presynaptic drive, conductance-form eligibility, rule-correct eligibility,
    optional scalar/per-soma broadcast error, and the resulting local update
    factor before optimizer-specific parameterization.

    - max_samples: deterministic cap on examples used per snapshot.
    - include_dendritic: also record branch-to-parent conductance terms.
    - include_broadcast: compute an approximate LocalCA broadcast from labels.
    - broadcast_mode: "scalar" or "per_soma" for the recorded error field.
    - error_mode: "auto", "ce", "mse", or "none".
    """

    max_samples: Optional[int] = 1024
    include_dendritic: bool = True
    include_broadcast: bool = True
    broadcast_mode: str = "scalar"
    error_mode: str = "auto"
    e_rev_exc: float = 1.0
    e_rev_inh: float = 0.0
    save_json: bool = True
    save_csv: bool = True
    append_training_csv: bool = True


@dataclass
class LocalRuleComponentAnalysisConfig(BaseConfig):
    """Enable/disable LocalCA local-rule component analysis."""

    enabled: bool = False
    training: bool = False
    params: LocalRuleComponentAnalysisParams = field(
        default_factory=LocalRuleComponentAnalysisParams
    )


@dataclass
class SingleLayerContributionAnalysisParams(BaseConfig):
    """Parameters for single-layer contribution analysis.

    Contribution modes:
    - excitation_contribution: effect of excitation alone
    - inhibition_contribution: effect of inhibition alone
    - both_contribution: combined E+I effect

    Metric toggles (same semantics as PerformanceAnalysisParams):
    - accuracy, auc, categorical_loglikelihood, mse, cosine_similarity
    """

    excitation_contribution: bool = True
    inhibition_contribution: bool = True
    both_contribution: bool = True
    accuracy: bool = True
    auc: bool = True
    categorical_loglikelihood: bool = True
    mse: bool = True
    cosine_similarity: bool = True


@dataclass
class SingleLayerContributionAnalysisConfig(BaseConfig):
    """Enable/disable single-layer contribution analysis."""

    enabled: bool = False
    training: bool = False
    params: SingleLayerContributionAnalysisParams = field(
        default_factory=SingleLayerContributionAnalysisParams
    )


@dataclass
class ReceptiveFieldAnalysisParams(BaseConfig):
    """Parameters for receptive field visualization.

    - rf_shape: optional (H, W) receptive field shape if known.
    - exc_rf_shape / inh_rf_shape: optional synapse-specific receptive-field
      shapes; fall back to rf_shape when omitted.
    - compute_activation_tuning: also summarize class tuning of branch signals.
    - activation_sample_signals / activation_sample_depths: optionally retain a
      deterministic, class-balanced sample of scalar branch signals for
      distribution plots. In addition to E, I, and post-gate output, ``upstream``
      records the aggregated child-branch current and ``depolarizing_drive``
      records E plus that upstream current. ``activation_sample_branch_indices``
      can restrict records globally, while
      ``activation_sample_branch_indices_by_depth`` supports a predeclared
      depth-specific selection. Empty selections retain all branches.
    - compute_component_rfs: run compact E/I mean and conditioned component RF
      maps, matching the empirical paper's soma-level RF summary.
    - compute_rf_tuning: run the aggregate RF tuning,
      discriminability, entropy, and compartment activation analysis.
    """

    n_rows: int = 2
    n_cols: int = 2
    rf_shape: Optional[tuple[int, int]] = None
    exc_rf_shape: Optional[tuple[int, int]] = None
    inh_rf_shape: Optional[tuple[int, int]] = None
    fig_save_format: str = "png"
    compute_activation_tuning: bool = False
    activation_sample_signals: list[str] = field(default_factory=list)
    activation_sample_depths: list[int] = field(default_factory=list)
    activation_sample_branch_indices: list[int] = field(default_factory=list)
    activation_sample_branch_indices_by_depth: dict[int, list[int]] = field(
        default_factory=dict
    )
    activation_samples_per_class: int = 0
    activation_sample_seed: int = 0
    compute_component_rfs: bool = False
    plot_component_rfs: bool = True
    compute_rf_tuning: bool = False
    plot_rf_tuning: bool = True
    rf_tuning_layer_index: int = 0
    n_class_examples: int = 5
    save_raw_data: bool = True
    load_raw_data: bool = False
    raw_filename: Optional[str] = None
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        allowed_signals = {
            "exc",
            "inh",
            "upstream",
            "depolarizing_drive",
            "vout",
            "ei_ratio",
        }
        self.activation_sample_signals = [
            str(value) for value in self.activation_sample_signals
        ]
        self.activation_sample_depths = [
            int(value) for value in self.activation_sample_depths
        ]
        self.activation_sample_branch_indices = [
            int(value) for value in self.activation_sample_branch_indices
        ]
        self.activation_sample_branch_indices_by_depth = {
            int(depth): [int(value) for value in indices]
            for depth, indices in self.activation_sample_branch_indices_by_depth.items()
        }
        if len(set(self.activation_sample_signals)) != len(
            self.activation_sample_signals
        ):
            raise ValueError("activation_sample_signals must be unique")
        if len(set(self.activation_sample_depths)) != len(
            self.activation_sample_depths
        ):
            raise ValueError("activation_sample_depths must be unique")
        if len(set(self.activation_sample_branch_indices)) != len(
            self.activation_sample_branch_indices
        ):
            raise ValueError("activation_sample_branch_indices must be unique")
        unknown = sorted(set(self.activation_sample_signals) - allowed_signals)
        if unknown:
            raise ValueError(f"unknown activation sample signals: {unknown}")
        if any(depth < 0 for depth in self.activation_sample_depths):
            raise ValueError("activation sample depths must be non-negative")
        if any(index < 0 for index in self.activation_sample_branch_indices):
            raise ValueError("activation sample branch indices must be non-negative")
        for depth, indices in self.activation_sample_branch_indices_by_depth.items():
            if depth < 0:
                raise ValueError("activation sample branch depths must be non-negative")
            if len(indices) != len(set(indices)):
                raise ValueError(
                    "activation sample branch indices must be unique within depth"
                )
            if any(index < 0 for index in indices):
                raise ValueError(
                    "activation sample branch indices must be non-negative"
                )
        self.activation_samples_per_class = int(self.activation_samples_per_class)
        self.activation_sample_seed = int(self.activation_sample_seed)
        if self.activation_samples_per_class < 0:
            raise ValueError("activation_samples_per_class must be non-negative")
        sampling_requested = bool(
            self.activation_sample_signals or self.activation_sample_depths
        )
        if sampling_requested and (
            not self.activation_sample_signals
            or not self.activation_sample_depths
            or self.activation_samples_per_class < 1
        ):
            raise ValueError(
                "activation sampling requires signals, depths, and a positive "
                "activation_samples_per_class"
            )


@dataclass
class ReceptiveFieldAnalysisConfig(BaseConfig):
    """Enable/disable receptive field visualization."""

    enabled: bool = False
    training: bool = False
    params: ReceptiveFieldAnalysisParams = field(
        default_factory=ReceptiveFieldAnalysisParams
    )


@dataclass
class ReactivationDynamicsAnalysisParams(BaseConfig):
    """Parameters for the reactivation-dynamics analyzer.

    The analyzer snapshots each ``ParametricTanh`` layer per epoch,
    capturing the learnable ``(log_m, b)`` parameters together with
    statistics of the pre-reactivation voltage ``V`` and the post-gate
    output ``r``. ``max_samples`` caps the batch size drawn from the
    training dataset to keep the analyzer lightweight.
    """

    max_samples: int = 1024


@dataclass
class ReactivationDynamicsAnalysisConfig(BaseConfig):
    """Enable/disable reactivation-dynamics analysis.

    When ``enabled=True`` and ``training=True``, the analyzer fires each
    epoch (one JSON file per epoch under
    ``<save_root>/reactivation_dynamics/epochs/``). This provides a direct
    record of gate parameters, voltage statistics, and output occupancy
    throughout training.
    """

    enabled: bool = False
    training: bool = True
    params: ReactivationDynamicsAnalysisParams = field(
        default_factory=ReactivationDynamicsAnalysisParams
    )


@dataclass
class CorrelationSelectionParams(BaseConfig):
    """Preferred (grouped) selectors for correlation analysis."""

    components: list[str] = field(default_factory=list)
    pairs: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)


@dataclass
class CorrelationComputeParams(BaseConfig):
    """Runtime/aggregation controls for correlation analysis."""

    computation_level: str = (
        "layer_branch"  # "synaptic", "single_branch", "layer_branch", "all_branch"
    )
    max_samples: Optional[int] = 5000
    batch_process: bool = True
    # Breakdown/visualization level (overridden by selection.scopes if provided)
    per_layer_analysis: bool = True
    per_einet_analysis: bool = True
    per_neuron_analysis: bool = False


@dataclass
class CorrelationSamplingParams(BaseConfig):
    """Sampling controls (used only when computation_level="synaptic")."""

    n_synapse_pairs: int = 10000
    sampling_strategy: str = "stratified"  # "random", "stratified", "importance"
    stratified_categories: list[str] = field(
        default_factory=lambda: ["within_branch", "within_layer", "across_layer"]
    )
    max_pairs_per_category: int = 1000


@dataclass
class CorrelationAnalysisParams(BaseConfig):
    """Parameters for correlation analysis.

    Core string options:
    - computation_level:
      - "synaptic": individual synapse correlations (uses sampling)
      - "single_branch": branch-by-branch correlations
      - "layer_branch": all branches in a layer together
      - "all_branch": full network aggregation
    - sampling_strategy (synaptic only): "random", "stratified", "importance"

    Preferred (list-based) selectors:
    - components: which correlation components to compute
      - "total": correlations across all samples
      - "noise": within-class (stimulus-independent) correlations
      - "signal": between-class difference (binary-only currently)
      - "tuning": tuning/selectivity summary across classes
    - pairs: which variable pairs to analyze
      - "EE", "II", "EI"
      - "E_output", "I_output", "output_output"
    - scopes: which breakdowns to include in outputs
      - "layer", "einet", "neuron"

    All ablation configuration is specified through the nested selection and methods
    fields below.
    """

    # ------------------------------------------------------------------
    # New (preferred) grouped structure
    # ------------------------------------------------------------------
    selection: CorrelationSelectionParams = field(
        default_factory=CorrelationSelectionParams
    )
    compute: CorrelationComputeParams = field(default_factory=CorrelationComputeParams)
    sampling: CorrelationSamplingParams = field(
        default_factory=CorrelationSamplingParams
    )

    # ------------------------------------------------------------------
    # Legacy flat keys (deprecated; still supported for backward compatibility)
    # ------------------------------------------------------------------

    # Computation level (aligned with information analysis)
    # "synaptic": Individual synapse correlations (most detailed, uses sampling)
    # "single_branch": Branch-by-branch analysis (detailed)
    # "layer_branch": All branches in a layer together (balanced)
    # "all_branch": Full network aggregation (most efficient)
    computation_level: str = "layer_branch"

    # New (preferred): list-based selection of analysis options.
    # If non-empty, these override the boolean flags below.
    components: list[str] = field(default_factory=list)
    pairs: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)

    # Correlation types to compute
    compute_noise_correlations: bool = (
        True  # Within-class (stimulus-independent) correlations
    )
    compute_signal_correlations: bool = (
        True  # Between-class (stimulus-driven) correlations
    )
    compute_total_correlations: bool = True  # Across all data
    compute_tuning_curves: bool = (
        True  # Class selectivity (mean difference between classes)
    )

    # Which correlation pairs to analyze
    analyze_EE: bool = True  # Excitatory-Excitatory correlations
    analyze_II: bool = True  # Inhibitory-Inhibitory correlations
    analyze_EI: bool = True  # Excitatory-Inhibitory cross-correlations
    analyze_E_output: bool = True  # Excitatory-Output (Vout) correlations
    analyze_I_output: bool = True  # Inhibitory-Output (Vout) correlations
    analyze_output_output: bool = True  # Output-Output (Vout-Vout) correlations

    # Hierarchical analysis options (similar to information analysis)
    per_layer_analysis: bool = True  # Analyze each layer separately
    per_einet_analysis: bool = True  # Separate E and I network analysis
    per_neuron_analysis: bool = False  # Analyze individual neurons (expensive)

    # Sampling parameters (only used when computation_level="synaptic")
    n_synapse_pairs: int = 10000  # Number of synapse pairs to sample
    sampling_strategy: str = "stratified"  # "random", "stratified", "importance"
    stratified_categories: list[str] = field(
        default_factory=lambda: ["within_branch", "within_layer", "across_layer"]
    )
    max_pairs_per_category: int = (
        1000  # Maximum pairs per category in stratified sampling
    )

    # Performance optimization
    max_samples: Optional[int] = 5000  # Limit data samples for faster computation
    batch_process: bool = True  # Process correlations in batches for memory efficiency


@dataclass
class CorrelationAnalysisConfig(BaseConfig):
    """Enable/disable correlation analysis."""

    enabled: bool = False
    training: bool = False
    params: CorrelationAnalysisParams = field(default_factory=CorrelationAnalysisParams)


@dataclass
class RoutingAnalysisConfig(BaseConfig):
    """Configuration for branch routing (alpha) analysis in recurrent models.

    Only applicable to recurrent EINetwork cores with store_routing=True.
    """

    enabled: bool = False
    training: bool = False


@dataclass
class InhibitorySpecializationAnalysisParams(BaseConfig):
    """Parameters for inhibitory-specialization (gate-index) analysis.

    - gate_threshold: absolute mean-G cutoff separating gate/context-specialists
      from diffuse neurons (per-I view) and soma/distal-dominant from balanced
      pairs (per-pair view).
    - std_threshold: std-G cutoff above which a neuron is classified
      mixed-conditional rather than a consistent specialist.
    - save_figures: whether to write a PDF alongside the JSON results.
    """

    gate_threshold: float = 0.1
    std_threshold: float = 0.15
    save_figures: bool = True


@dataclass
class InhibitorySpecializationAnalysisConfig(BaseConfig):
    """Enable/disable inhibitory-specialization analysis.

    Applicable only to recurrent E-I dendritic cores that carry at least one
    layer with a local inhibitory pool wired through ``branch_rec_inhibition``.
    Silently skipped for feedforward-only or baseline RNN cores.
    """

    enabled: bool = False
    training: bool = False
    params: InhibitorySpecializationAnalysisParams = field(
        default_factory=InhibitorySpecializationAnalysisParams
    )


@dataclass
class JacobianSpectrumAnalysisParams(BaseConfig):
    """Parameters for per-level Jacobian / recurrent-spectrum analysis.

    - top_k: number of leading singular values to keep per branch layer.
    - save_figures: whether to write a PDF alongside the JSON results.
    """

    top_k: int = 16
    save_figures: bool = True


@dataclass
class JacobianSpectrumAnalysisConfig(BaseConfig):
    """Enable/disable per-level Jacobian / recurrent-spectrum analysis.

    Reads the effective ``branch_recurrent`` / ``branch_rec_inhibition``
    weights of the excitatory population at each branch level and reports
    their top singular values. Skipped on baseline RNN cores and on layers
    without recurrent compartments.
    """

    enabled: bool = False
    training: bool = False
    params: JacobianSpectrumAnalysisParams = field(
        default_factory=JacobianSpectrumAnalysisParams
    )


@dataclass
class DendriticTimetracesAnalysisParams(BaseConfig):
    """Parameters for dendritic-time-trace simulation.

    The analyzer simulates ``V_l(t)`` and ``hat V_l(t)`` for the trained
    τ and reactivation parameters on a synthetic two-pulse stimulus.

    - n_timesteps: trial length.
    - stim1_window / stim2_window: ``(start, end)`` index tuples in [0, T).
      Set either to ``None`` to disable the corresponding pulse.
    - stim1_amp / stim2_amp: amplitudes of the two pulses.
    - response_window: optional shaded region marking the response period.
    - noise_std: white noise added to the input; 0 disables.
    - seed: RNG seed for the noise tensor (deterministic).
    - save_figures: whether to write a PDF per E-I layer.
    """

    n_timesteps: int = 90
    stim1_window: Optional[tuple[int, int]] = (5, 25)
    stim2_window: Optional[tuple[int, int]] = (55, 75)
    response_window: Optional[tuple[int, int]] = (75, 90)
    stim1_amp: float = 0.8
    stim2_amp: float = 0.6
    noise_std: float = 0.05
    seed: int = 0
    save_figures: bool = True


@dataclass
class DendriticTimetracesAnalysisConfig(BaseConfig):
    """Enable/disable dendritic-time-trace simulation.

    Simulates the learned per-level cascade dynamics (τ, m, b) on a
    synthetic two-pulse trial. Skipped on baseline RNN cores or on
    dendritic populations without ``log_tau`` exposed.
    """

    enabled: bool = False
    training: bool = False
    params: DendriticTimetracesAnalysisParams = field(
        default_factory=DendriticTimetracesAnalysisParams
    )


@dataclass
class SpikeTrainAnalysisParams(BaseConfig):
    """Parameters for spike-train analysis of LIF recurrent populations.

    - dt: seconds or arbitrary time units per recurrent step; rates are
      reported as spikes per ``dt``.
    - threshold: spike binarization threshold for recorded LIF outputs.
    - max_samples: optional cap on dataset examples for analysis.
    - include_raster: include compact event lists for the first
      ``max_raster_samples`` examples.
    - include_per_timestep_rate: include a population firing-rate trace.
    """

    dt: float = 1.0
    threshold: float = 0.5
    max_samples: Optional[int] = 512
    include_raster: bool = True
    max_raster_samples: int = 8
    include_per_timestep_rate: bool = True


@dataclass
class SpikeTrainAnalysisConfig(BaseConfig):
    """Enable/disable spike-train analysis for opt-in LIF recurrent cores."""

    enabled: bool = False
    training: bool = False
    params: SpikeTrainAnalysisParams = field(default_factory=SpikeTrainAnalysisParams)


@dataclass
class AnalysisConfig(BaseConfig):
    """Top-level analysis configuration container.

    Each field corresponds to an analysis module under `analysis.*` in YAML.
    """

    runtime: AnalysisRuntimeConfig = field(default_factory=AnalysisRuntimeConfig)

    performance_analysis: PerformanceAnalysisConfig = field(
        default_factory=PerformanceAnalysisConfig
    )
    synapse_turnover_analysis: SynapseTurnoverAnalysisConfig = field(
        default_factory=SynapseTurnoverAnalysisConfig
    )
    noise_perturbation_analysis: NoisePerturbationAnalysisConfig = field(
        default_factory=NoisePerturbationAnalysisConfig
    )
    single_layer_contribution: SingleLayerContributionAnalysisConfig = field(
        default_factory=SingleLayerContributionAnalysisConfig
    )
    ablation_analysis: AblationAnalysisConfig = field(
        default_factory=AblationAnalysisConfig
    )
    path_matched_intervention_analysis: PathMatchedInterventionAnalysisConfig = field(
        default_factory=PathMatchedInterventionAnalysisConfig
    )
    branch_local_sensitivity_analysis: BranchLocalSensitivityAnalysisConfig = field(
        default_factory=BranchLocalSensitivityAnalysisConfig
    )
    local_inhibitory_population_intervention_analysis: (
        LocalInhibitoryPopulationInterventionAnalysisConfig
    ) = field(default_factory=LocalInhibitoryPopulationInterventionAnalysisConfig)
    path_matched_information_analysis: PathMatchedInformationAnalysisConfig = field(
        default_factory=PathMatchedInformationAnalysisConfig
    )
    weight_analysis: WeightAnalysisConfig = field(default_factory=WeightAnalysisConfig)

    compartment_statistics_analysis: CompartmentStatisticsAnalysisConfig = field(
        default_factory=CompartmentStatisticsAnalysisConfig
    )

    compartment_snr_analysis: CompartmentSNRAnalysisConfig = field(
        default_factory=CompartmentSNRAnalysisConfig
    )
    multiplicative_gain_analysis: MultiplicativeGainAnalysisConfig = field(
        default_factory=MultiplicativeGainAnalysisConfig
    )
    gain_load_perturbation_analysis: GainLoadPerturbationAnalysisConfig = field(
        default_factory=GainLoadPerturbationAnalysisConfig
    )
    synaptic_activation_analysis: SynapticActivationAnalysisConfig = field(
        default_factory=SynapticActivationAnalysisConfig
    )
    branch_activation_analysis: BranchActivationAnalysisConfig = field(
        default_factory=BranchActivationAnalysisConfig
    )
    local_rule_component_analysis: LocalRuleComponentAnalysisConfig = field(
        default_factory=LocalRuleComponentAnalysisConfig
    )
    information_analysis: InformationAnalysisConfig = field(
        default_factory=InformationAnalysisConfig
    )
    representation_accessibility_analysis: RepresentationAccessibilityAnalysisConfig = (
        field(default_factory=RepresentationAccessibilityAnalysisConfig)
    )
    vision_boundary_diagnostics_analysis: VisionBoundaryDiagnosticsAnalysisConfig = (
        field(default_factory=VisionBoundaryDiagnosticsAnalysisConfig)
    )
    source_tuning_support_analysis: SourceTuningSupportAnalysisConfig = field(
        default_factory=SourceTuningSupportAnalysisConfig
    )
    input_region_intervention_analysis: InputRegionInterventionAnalysisConfig = field(
        default_factory=InputRegionInterventionAnalysisConfig
    )
    receptive_field_analysis: ReceptiveFieldAnalysisConfig = field(
        default_factory=ReceptiveFieldAnalysisConfig
    )
    reactivation_dynamics_analysis: ReactivationDynamicsAnalysisConfig = field(
        default_factory=ReactivationDynamicsAnalysisConfig
    )
    correlation_analysis: CorrelationAnalysisConfig = field(
        default_factory=CorrelationAnalysisConfig
    )
    routing_analysis: RoutingAnalysisConfig = field(
        default_factory=RoutingAnalysisConfig
    )
    inhibitory_specialization_analysis: InhibitorySpecializationAnalysisConfig = field(
        default_factory=InhibitorySpecializationAnalysisConfig
    )
    jacobian_spectrum_analysis: JacobianSpectrumAnalysisConfig = field(
        default_factory=JacobianSpectrumAnalysisConfig
    )
    dendritic_timetraces_analysis: DendriticTimetracesAnalysisConfig = field(
        default_factory=DendriticTimetracesAnalysisConfig
    )
    spike_train_analysis: SpikeTrainAnalysisConfig = field(
        default_factory=SpikeTrainAnalysisConfig
    )
    adversarial_robustness_analysis: AdversarialRobustnessAnalysisConfig = field(
        default_factory=AdversarialRobustnessAnalysisConfig
    )

    def any_enabled(self) -> bool:
        """Return whether the final training analysis gate should run."""
        return any(
            bool(getattr(getattr(self, spec.config_field, None), "enabled", False))
            for spec in get_registered_analyzers().values()
        )
