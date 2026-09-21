from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.regularization import (
    PruningConfig,
    RegularizationConfig,
    SelectiveConfig as _SelectiveConfig,
)

# Backward-compatible re-export: downstream code imports SelectiveConfig
# from config.training in several places.
SelectiveConfig = _SelectiveConfig


@dataclass
class ParamGroupsConfig(BaseConfig):
    """Parameter groups with different learning rates."""

    lr: float = 0.01
    split_params: bool = True
    topk_lr: float = 0.001
    blocklinear_lr: float = 0.0001
    reactivation_lr: float = 0.0001
    decoder_lr: float = 0.001
    # Optional LR for only the first decoder Linear weight. This supports
    # width-scaled updates without slowing decoder biases or the output head.
    decoder_input_lr: Optional[float] = None
    encoder_lr: Optional[float] = None  # None = frozen / use default LR


@dataclass
class CommonTrainerConfig(BaseConfig):
    """Common training parameters shared across strategies."""

    epochs: int = 1000
    batch_size: int = 512
    shuffle: bool = True
    grad_clip_value: float = 2.0
    loss_function: str = "ce"
    early_stopping: bool = True
    patience: int = 500
    load_best_state_dict: bool = True
    plot_losses: bool = True
    suppress_prints: bool = False
    print_every: int = 10
    # DataLoader controls. None keeps the trainer's automatic behavior.
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    persistent_workers: Optional[bool] = None
    prefetch_factor: Optional[int] = None
    in_order: Optional[bool] = None
    use_amp: bool = False
    lr_schedule: str = "none"
    lr_warmup_epochs: int = 0
    enable_nan_checking: bool = False
    enable_adaptive_clipping: bool = False
    profile_dataloader: bool = False
    # Record compact decoder-input, logit, and gradient statistics for the
    # first FSDP optimizer step. Disabled by default to avoid extra reductions.
    log_first_step_diagnostics: bool = False
    # Data-driven reactivation initialization can retain intermediate voltage
    # tensors for every dendritic layer. These controls bound that one-time
    # calibration for deep or spatial models without changing its estimator.
    reactivation_initialization_batch_size: int = 256
    reactivation_initialization_num_batches: int = 3
    # None iterates to convergence with the package default cap.
    reactivation_initialization_max_iterations: Optional[int] = None
    weight_decay_rate: float = 0.01
    weight_boosting: bool = False
    # Periodic reactivation recalibration during training.
    # 0 = disabled (default). N > 0 = recompute (m, b) from the current
    # voltage distribution every N epochs using the same calibration rule
    # as the configured reactivation init policy.
    recalibrate_reactivation_every: int = 0
    # How reactivation parameters are updated during standard training.
    # "backprop" = train via the optimizer (default, historical behavior).
    # "quantile" = freeze BP updates and refresh (m, b) with the data-driven
    # quantile calibration rule during training.
    reactivation_update_mode: str = "backprop"
    # Calibration rule used by periodic reactivation refreshes.
    # "from_init_policy" preserves historical behavior. When
    # reactivation_update_mode="quantile", this defaults to the occupancy
    # quantile rule unless explicitly overridden.
    reactivation_recalibration_mode: str = "from_init_policy"
    # First epoch at which periodic reactivation refreshes are allowed.
    # This lets experiments preserve the chosen initialization through epoch 1
    # before switching to quantile-rule updates.
    reactivation_recalibration_start_epoch: int = 1
    # Number of training batches used to estimate the live voltage
    # distribution for a periodic (m, b) refresh.
    reactivation_recalibration_num_batches: int = 3
    # EMA-style blend factor for periodic reactivation refreshes.
    # 1.0 = hard overwrite with the freshly calibrated (m, b).
    # 0 < alpha < 1 = blend the current parameters toward the calibrated target.
    reactivation_recalibration_ema_alpha: float = 1.0

    param_groups: ParamGroupsConfig = field(default_factory=ParamGroupsConfig)


@dataclass
class StandardStrategyConfig(BaseConfig):
    """Standard backpropagation strategy configuration."""

    # Uses common config only


@dataclass
class MultiStageStrategyConfig(BaseConfig):
    """Multi-stage training strategy configuration."""

    analyze_between_stages: bool = True
    use_best_from_each_stage: bool = True
    continue_on_failure: bool = False

    stages: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "learning_strategy": "standard",
                "epochs": 100,
                "batch_size": 512,
                "save_checkpoint": False,
                "param_groups": {
                    "lr": 0.01,
                    "split_params": True,
                    "topk_lr": 0.001,
                    "blocklinear_lr": 0.0001,
                    "reactivation_lr": 0.0001,
                    "decoder_lr": 0.001,
                },
                "trainer_config": {
                    "loss_function": "ce",
                    "early_stopping": True,
                    "patience": 500,
                },
            },
            {
                "learning_strategy": "local_ca",
                "epochs": 100,
                "batch_size": 512,
                "reset_optimizer": False,
                "save_checkpoint": False,
                "param_groups": {
                    "lr": 0.0001,
                    "split_params": True,
                    "topk_lr": 0.0001,
                    "blocklinear_lr": 0.00001,
                    "reactivation_lr": 0.00001,
                    "decoder_lr": 0.001,
                },
                "trainer_config": {"loss_function": "ce", "early_stopping": False},
            },
        ]
    )


@dataclass
class VoltageStabilizationStrategyConfig(BaseConfig):
    """Voltage stabilization strategy configuration."""

    pretrain_epochs: int = 100
    stabilize_mode: str = "vout"
    freeze_reactivation: bool = True
    target_voltage: float = 0.5
    target_saturation: float = 0.2
    loss_metric: str = "mse"


@dataclass
class HomeostaticControlStrategyConfig(BaseConfig):
    """Homeostatic control strategy configuration."""

    pretrain_epochs: int = 100
    stabilize_mode: str = "vinf"
    freeze_reactivation: bool = False
    target_voltage: float = 0.5
    target_saturation: float = 0.2
    loss_metric: str = "mse"


@dataclass
class RecurrentStrategyConfig(BaseConfig):
    """Recurrent trainer-specific overrides."""

    grad_clip: float = 1.0
    use_amp: bool = False
    lr_schedule: str = "none"
    lr_warmup_epochs: int = 0


@dataclass
class FreezeLayersStrategyConfig(BaseConfig):
    """Freeze layers strategy configuration."""

    pretrain_epochs: int = 10
    reverse_training: bool = False
    epochs_per_layer: int = 50
    final_tune_epochs: int = 10


@dataclass
class FreezeBranchesStrategyConfig(BaseConfig):
    """Freeze branches strategy configuration."""

    pretrain_epochs: int = 10
    reverse_training: bool = False
    epochs_per_branch: int = 50
    final_tune_epochs: int = 10


@dataclass
class FreezeBranchKLStrategyConfig(BaseConfig):
    """Freeze branch KL strategy configuration."""

    pretrain_epochs: int = 10
    reverse_training: bool = False
    epochs_per_branch: int = 50
    final_tune_epochs: int = 10
    kl_weight: float = 0.1


@dataclass
class VisionDistillationStrategyConfig(BaseConfig):
    """Online feature and logit distillation for AlexNet-shaped students.

    Boundary losses are a weighted mean, so enabling more boundaries does not
    implicitly increase the overall feature-loss scale. ``mixed`` inputs blend
    teacher and composed-student activations using ``teacher_forcing_ratio``.
    An empty mapping keeps intermediate feature supervision disabled.
    """

    teacher_backbone: str = "alexnet"
    teacher_weights: str = "IMAGENET1K_V1"
    boundary_weights: dict[str, float] = field(default_factory=dict)
    boundary_input_mode: str = "composed"
    teacher_forcing_ratio: float = 0.5
    feature_weight: float = 1.0
    logit_kl_weight: float = 1.0
    supervised_weight: float = 1.0
    temperature: float = 2.0
    relative_mse_epsilon: float = 1e-6
    class_output_target: str = "raw"


@dataclass
class TrainingStrategiesConfig(BaseConfig):
    """Training strategies configuration."""

    standard: StandardStrategyConfig = field(default_factory=StandardStrategyConfig)
    multi_stage: MultiStageStrategyConfig = field(
        default_factory=MultiStageStrategyConfig
    )
    recurrent: RecurrentStrategyConfig = field(default_factory=RecurrentStrategyConfig)
    voltage_stabilization: VoltageStabilizationStrategyConfig = field(
        default_factory=VoltageStabilizationStrategyConfig
    )
    homeostatic_control: HomeostaticControlStrategyConfig = field(
        default_factory=HomeostaticControlStrategyConfig
    )
    freeze_layers: FreezeLayersStrategyConfig = field(
        default_factory=FreezeLayersStrategyConfig
    )
    freeze_branches: FreezeBranchesStrategyConfig = field(
        default_factory=FreezeBranchesStrategyConfig
    )
    freeze_branch_kl: FreezeBranchKLStrategyConfig = field(
        default_factory=FreezeBranchKLStrategyConfig
    )
    vision_distillation: VisionDistillationStrategyConfig = field(
        default_factory=VisionDistillationStrategyConfig
    )


@dataclass
class EncoderTrainingConfig(BaseConfig):
    """Encoder training configuration."""

    strategy: str = "standard"

    common: CommonTrainerConfig = field(
        default_factory=lambda: CommonTrainerConfig(
            loss_function="log_mse",
            epochs=500,
            early_stopping=True,
            patience=100,
            batch_size=512,
            shuffle=True,
            grad_clip_value=5.0,
            load_best_state_dict=True,
            plot_losses=False,
            suppress_prints=False,
            print_every=10,
            use_amp=False,
            enable_nan_checking=False,
            enable_adaptive_clipping=False,
        )
    )

    strategies: TrainingStrategiesConfig = field(
        default_factory=TrainingStrategiesConfig
    )
    learning_strategy_config: Optional["LocalRuleConfig"] = field(
        default_factory=lambda: None
    )


@dataclass
class OptimizerConfig(BaseConfig):
    """Configuration for optimizer selection and parameters."""

    name: str = "adam"  # Options: "adam", "sgd", "rmsprop", "adamw", "muonh"
    lr: float = 0.001
    weight_decay: float = 0.0  # Built-in L2 regularization (global)
    momentum: float = 0.9  # For SGD
    betas: tuple = (0.9, 0.999)  # For Adam/AdamW
    eps: float = 1e-8  # For Adam/AdamW
    # ``None`` preserves PyTorch's automatic implementation choice.  An
    # explicit boolean is useful for prospectively matched experiments whose
    # optimizer execution contract requires the foreach path to be fixed.
    foreach: bool | None = None
    # MuonH options (Puro-2B, arXiv 2608.27370): Muon (momentum +
    # Newton-Schulz orthogonalized updates for 2D matrices) with a hyperball
    # projection back to each matrix's initial Frobenius norm, and an AdamW
    # fallback for non-2D / gate-like parameters.  Read ONLY when
    # ``name == "muonh"``; every other optimizer ignores these fields, so
    # existing configs are behaviorally untouched.
    muon_lr_multiplier: float = 10.0  # effective Muon LR = lr * multiplier
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_hyperball: bool = True
    # 2D matrices with min(shape) < muon_min_dim (e.g. [N, 1] single-contact
    # columns) fall back to AdamW alongside vectors and scalars.
    muon_min_dim: int = 2
    # First N distributed steps assert bitwise-identical Muon parameters
    # across ranks (Muon must run after the DDP gradient all-reduce).
    muon_rank_consistency_check_steps: int = 1


@dataclass
class MainTrainingConfig(BaseConfig):
    """Main network training configuration."""

    strategy: str = "standard"

    common: CommonTrainerConfig = field(default_factory=CommonTrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    strategies: TrainingStrategiesConfig = field(
        default_factory=TrainingStrategiesConfig
    )
    learning_strategy_config: Optional["LocalRuleConfig"] = field(
        default_factory=lambda: None
    )


@dataclass
class TransformerReplacementTrainingConfig(BaseConfig):
    """Training options for transformer MLP/FFN dendritic replacements.

    This block controls *how* the transformer replacement is trained.  The
    target model/layers still live in ``model.transformer_replacement`` and the
    dendritic replacement architecture still lives in ``model.core``.

    Supported ``teacher_source`` values:

    - ``"synthetic_mlp"``: train against a small generated teacher MLP.  This is
      the fast smoke-test path and does not require Hugging Face dependencies.
    - ``"hidden_cache"``: train from saved hidden-state/target tensors.
    - ``"hf_text"``: run a Hugging Face causal LM on raw text or a Hugging Face
      dataset and distill the selected target modules from captured hidden
      activations.

    Supported ``mode`` values:

    - ``"layerwise_distillation"``: train replacement modules against captured
      target-module outputs.  ``train_target`` must be ``"replacement_only"``.
    - ``"joint_lm_distillation"``: patch a causal LM and train through the
      language-modeling objective.  ``train_target`` may be
      ``"replacement_only"`` or ``"full_model"``.
    """

    enabled: bool = False
    # Versioned comparison/training protocol selected by a generator.  The
    # embedded manifest records its evidence scope and prevents a model-specific
    # measured prior from being mistaken for a universal optimum.
    training_recipe: str = ""
    training_recipe_manifest: dict[str, Any] = field(default_factory=dict)
    mode: str = "layerwise_distillation"
    teacher_source: str = "synthetic_mlp"
    train_target: str = "replacement_only"
    # Joint-LM teacher role. ``student_base`` preserves the historical path:
    # teacher and student are loaded from the same model declaration. The
    # ``model_name_pretrained`` role loads global supervision from
    # ``model.transformer_replacement.model_name`` while the student may use a
    # heterogeneous external source. This separates compressed-source local
    # statistics from pristine end-to-end supervision in cascaded recovery.
    joint_global_teacher_role: str = "student_base"
    joint_global_teacher_model_kwargs: dict[str, Any] = field(default_factory=dict)
    # Opt-in fixed-topology, single-process joint recovery policy. Missing
    # layer keys use the base LR; zero freezes that replacement completely.
    # Keys are canonical decimal transformer layer indices (e.g. "7").
    joint_layer_lr_multipliers: dict[str, float] = field(default_factory=dict)
    # Training-only retention relative to the post-warm-start parameters:
    # mean over tensors of mean((p-p0)^2) / max(mean(p0^2), epsilon).
    # Validation/checkpoint selection never includes this regularizer.
    joint_proximal_weight: float = 0.0
    joint_proximal_layers: list[int] = field(default_factory=list)
    joint_proximal_epsilon: float = 1.0e-8
    # Joint-LM student initialization. ``pretrained`` preserves the current
    # distillation/fine-tuning path; ``from_config`` constructs random model
    # weights from the Hugging Face architecture config before patching.
    student_initialization: str = "pretrained"
    layers: list[int] = field(default_factory=list)
    device: str = "auto"
    dtype: str = "float32"
    seed: int | None = None
    # ``ddp`` is the replicated-model torchrun path used by layerwise and
    # replacement-only joint distillation. Full-model 7B+ training still
    # requires the planned FSDP integration.
    distributed_mode: str = "none"
    distributed_backend: str = ""
    ddp_find_unused_parameters: bool = False
    # Optional exact cross-rank gate over replacement parameters and persistent
    # topology/state buffers. Zero disables it; one verifies every optimizer
    # and sparse-topology update. Intended for short systems qualification,
    # since cryptographic tensor hashing adds synchronization overhead.
    ddp_state_digest_every: int = 0
    # Optional prefix length for a two-phase systems qualification: hash at the
    # requested cadence only through this many optimizer steps, then leave the
    # remaining timed phase untouched until the mandatory post-restore digest.
    # Zero retains cadence checks throughout training.
    ddp_state_digest_qualification_steps: int = 0
    # Optional exact optimizer-step digest schedule.  This composes with the
    # periodic prefix above, allowing dynamic-topology campaigns to qualify at
    # a tight early cadence and still gate every later validation rung without
    # paying cryptographic hashing cost on all intervening updates.
    ddp_state_digest_steps: list[int] = field(default_factory=list)
    data_parallel: bool = False
    data_parallel_devices: list[int] = field(default_factory=list)

    max_steps: int = 200
    eval_every: int = 50
    log_every: int = 25
    batch_size: int = 32
    # DataLoader controls for layerwise tensor-cache distillation. None keeps
    # the current auto behavior: in-memory tensor caches stay single-process.
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    persistent_workers: Optional[bool] = None
    prefetch_factor: Optional[int] = None
    in_order: Optional[bool] = None
    sequence_length: int = 8
    train_samples: int = 2048
    valid_samples: int = 512
    # Optional global validation-window contract for distributed runs.  When
    # set, this count must be divisible by ``batch_size * world_size`` and is
    # used instead of interpreting ``valid_samples`` independently per rank.
    # ``valid_samples`` must then equal the exact per-rank share so old configs
    # cannot silently change meaning.
    validation_global_samples: int | None = None
    # Formal frozen-window campaigns can prohibit modulo reuse during
    # validation.  Historical runs retain wrapping unless they opt out.
    validation_allow_window_wrap: bool = True

    hidden_size: int | None = None
    intermediate_size: int | None = None
    synthetic_teacher: dict[str, Any] = field(
        default_factory=lambda: {
            "kind": "gated_mlp",
            "activation": "silu",
            "bias": False,
            "input_std": 1.0,
        }
    )
    hidden_cache: dict[str, Any] = field(
        default_factory=lambda: {
            "train_inputs": "",
            "train_targets": "",
            "valid_inputs": "",
            "valid_targets": "",
        }
    )
    text: dict[str, Any] = field(
        default_factory=lambda: {
            "text_path": "",
            "dataset_name": "",
            "dataset_config": "",
            # ``data_files`` supports datasets-compatible paths/globs, including
            # local JSON/JSONL gzip shards such as the shared Dolma corpus.
            "data_files": None,
            "split": "train",
            "text_field": "text",
            "max_documents": None,
            "streaming": False,
            # Optional weighted document-level mixture of auxiliary corpora
            # into the streaming training stream (default off).  Each entry
            # is a mapping with ``dataset_name`` (or ``data_files``),
            # ``weight`` in (0, 1), and optional ``dataset_config``,
            # ``split``, ``text_field`` or ``text_fields`` (joined by
            # newlines), ``streaming`` (default true), and
            # ``max_documents``.  The primary corpus keeps probability
            # ``1 - sum(weights)``.  Auxiliary sources are reshuffled and
            # cycled when exhausted; the primary stream still governs epoch
            # boundaries, and the validation stream is never mixed.
            # Requires ``streaming: true`` and no ``text_path``/frozen
            # manifests (fail-closed otherwise).
            "mixture": None,
            # Optional teacher-loss curriculum ordering of the streaming
            # training corpus (default off).  Adapted from Puro-2B
            # (arXiv 2608.27370) low-to-high quality data ordering; under a
            # fixed KD teacher the per-window quality analog is the teacher
            # loss.  Points at a sealed window-order manifest produced by
            # scripts.text.score_recovery_windows over the SAME corpus,
            # tokenizer, and sequence length as this config.  When set, the
            # training stream replays the manifest's windows in score order
            # -- deterministic and epoch-stable, with strided DDP sharding
            # -- instead of shuffled packed documents; the validation stream
            # is unchanged.  Fail-closed on any seal/digest/coverage or
            # corpus-identity mismatch, and mutually exclusive with
            # ``mixture``, ``text_path``, and frozen manifests (requires
            # ``streaming: true``).
            "curriculum_order_manifest": "",
            # ``low_to_high`` streams easy (low teacher loss) windows first;
            # ``high_to_low`` reverses the order.  Ties break on window id.
            "curriculum_direction": "low_to_high",
            "shuffle_seed": None,
            "shuffle_buffer_size": 10_000,
            "validation_data_files": None,
            "validation_split": "",
            "validation_documents": 64,
            "validation_max_tokens": None,
            # Evaluation can reserve a non-overlapping contiguous token range.
            # Positive offsets are paired with fail-closed, non-wrapping windows
            # by the disjoint-evaluation entry point.
            "validation_token_offset": 0,
            "validation_allow_window_wrap": True,
            # Content-addressed training/validation tensors created by
            # scripts.text.frozen_text_windows.  Both manifests are required
            # together and are mutually exclusive with raw/Hugging Face text
            # selectors.  Formal runs fail closed on exploratory manifests.
            "frozen_training_manifest": "",
            "frozen_validation_manifest": "",
            "frozen_require_evidentiary": True,
            "frozen_primary_tokenizer_artifact_set_sha256": "",
            "frozen_primary_tokenizer_model_root": "",
            "frozen_shuffle_seed": None,
            # Process-local token installed only after a campaign-bound,
            # all-rank live exact-SHA verification. Config files cannot create
            # this token; absent registration retains the complete semantic
            # frozen-window verifier.
            "frozen_execution_binding_token": "",
            # Optional exact packed-stream state. Model/optimizer checkpointing
            # must be coordinated at the same step by the caller.
            "resume_stream_state": "",
            "tokenizer_name": "",
            "tokenizer_kwargs": {},
            "max_tokens": None,
            "max_tokens_per_document": None,
        }
    )

    loss: str = "mse"
    cosine_weight: float = 0.0
    lm_loss_weight: float = 1.0
    kl_loss_weight: float = 0.0
    kl_temperature: float = 1.0
    # Optional capability-sensitive distillation term.  At every next-token
    # position, select the teacher's top-k candidates and preserve each
    # candidate's logit gap from the teacher-preferred token.  Unlike ordinary
    # token-mean CE, this directly supervises relative alternatives; unlike
    # full-vocabulary KL, negligible-probability vocabulary entries cannot
    # dominate the averaging convention.  The term is experimental and is
    # disabled by default until a held-out recovery comparison supports it.
    teacher_topk_margin_weight: float = 0.0
    teacher_topk_margin_k: int = 32
    teacher_topk_margin_epsilon: float = 1.0e-6
    hidden_loss_weight: float = 0.0
    hidden_loss_layers: list[int] = field(default_factory=list)
    # ``mse`` preserves the historical objective. Scale-invariant alternatives
    # are useful when anchors at different transformer depths have very
    # different activation energies.
    hidden_loss_type: str = "mse"
    # Empty means equal weight over ``hidden_loss_layers``. Non-empty weights
    # are normalized by the trainer, so adding anchors does not silently
    # multiply the effective hidden-loss coefficient.
    hidden_loss_layer_weights: list[float] = field(default_factory=list)
    hidden_loss_epsilon: float = 1.0e-6
    # Optional upper-tail emphasis for joint LM recovery.  With weight zero,
    # the Hugging Face token-mean LM loss is retained exactly.  Positive
    # weights blend it with the mean of the worst ``sequence_risk_fraction``
    # sequences in each batch.  This turns the diverse-recovery observation
    # that rare registers/windows can fail catastrophically into an explicit,
    # auditable objective rather than a one-off sampling script.
    sequence_risk_weight: float = 0.0
    sequence_risk_fraction: float = 0.25
    # Checkpoint selection can independently emphasize the upper tail across
    # deterministic validation batches.  A value of 1 selects by validation
    # CVaR; zero preserves mean-objective selection.
    validation_tail_weight: float = 0.0
    validation_tail_fraction: float = 0.20
    # Joint recovery historically selected the minimum total distillation
    # objective.  Matched external controls may prospectively select pure LM
    # loss while retaining KL/hidden terms in the training objective.
    checkpoint_selection_metric: str = "objective"  # objective | lm | kl
    # Weights-only initialization from exported
    # ``layer_<index>_replacement.pt`` artifacts. Unlike
    # ``resume_checkpoint``, this deliberately starts a new optimizer,
    # validation history, and token stream. This is the canonical equivalent
    # of the output-pilot prune/recover and staged-recovery workflows. Learned
    # sparse topology, affine paths, and reactivation parameters are preserved;
    # teacher initialization/calibration is therefore not rerun.
    warm_start_dir: str = ""
    gradient_checkpointing: bool = False
    max_grad_norm: float | None = None
    # Layerwise distillation target. ``teacher_function`` (default) matches
    # the dense layer's output on the captured inputs. ``trajectory_anchored``
    # targets the CLEAN teacher trajectory instead: with earlier layers
    # pre-patched, layer k trains on the composed student's drifted inputs but
    # its target is clean_mlp_out + clean_layer_in - drifted_layer_in, so each
    # replacement compensates accumulated upstream error (first-order in the
    # block's post-norm). Reduces to teacher_function when nothing is
    # pre-patched.
    distillation_target: str = "teacher_function"
    # Explicit opt-in for zero-gradient-step (compile/calibrate-only)
    # runs; max_steps == 0 without this flag is now an error.
    zero_shot_compile: bool = False
    # Parameter dtype of the trainable replacements. FP32 master weights are
    # required for optimization: BF16 parameters round nearly all small
    # updates to zero (audit 2026-08-16: 234 of 33.8M values changed in 300
    # steps). The surrounding model may remain BF16.
    replacement_dtype: str = "float32"
    # Data-driven reactivation policies such as occupancy_quantile require
    # actual replacement inputs. The generic trainer already calibrates these;
    # these controls provide the equivalent transformer path.
    reactivation_calibration_enabled: bool = True
    reactivation_calibration_batches: int = 3
    reactivation_calibration_iterations: int = 3
    # Empty means respect each dendritic layer's configured init policy.
    reactivation_calibration_mode: str = ""
    # Restartable layerwise distillation checkpoint. Each DDP rank also writes
    # its own RNG/stream progress at the same barriered step.
    checkpoint_every: int = 0
    resume_checkpoint: str = ""
    # Optional immutable joint-LM rung schedule.  The list must be sorted,
    # unique, begin at zero, and end at max_steps.  It is mutually exclusive
    # with checkpoint_every so a formal campaign has one unambiguous schedule.
    joint_checkpoint_steps: list[int] = field(default_factory=list)
    # Canonical campaign-specific execution identity carried into every exact
    # joint restart checkpoint.  Formal wrappers use this to bind immutable
    # data, architecture, optimizer, seed, and token-stream semantics that the
    # generic trainer cannot reconstruct from ``train_cfg`` alone.
    joint_restart_execution_contract: dict[str, Any] = field(default_factory=dict)
    # Layerwise training restores this validation-best state before exporting
    # replacement checkpoints. This includes step 0, which matters for
    # teacher-pruned initializations that fine-tuning can degrade.
    restore_best_replacement: bool = True
    # Final per-layer export only. Training/restart checkpoints retain slot-wise
    # indices so optimizer state remains aligned. ``bitmask`` reorders each
    # sparse row and stores its fixed topology as one bit per possible edge.
    replacement_checkpoint_encoding: str = "auto"  # auto | uint | bitmask
    # Convert train-time TopK, dense-to-sparse, dynamic-indexed, and rewiring
    # synapses into exact fixed-contact IndexedSparseLinear modules in the
    # exported per-layer checkpoints. Dense-to-sparse export fails closed until
    # every pruning schedule has reached its configured final K.
    freeze_sparse_topology_on_export: bool = False
    # Stochastic TopK has no single inference graph. Keep "reject" for manual
    # configs unless the experiment explicitly chooses deterministic learned
    # scores or one reproducible sampled graph.
    stochastic_topology_freeze_policy: str = "reject"
    stochastic_topology_freeze_seed: int = 0
    ragged_topology_format: str = "csr"
    # Optional post-training structural prune -> recovery ladder. Each rung
    # resolves exact contact counts before mutation and may set a global
    # ``default_density``, path-specific density/contact selectors, and
    # ``per_unit`` overrides (for example layer_0 vs layer_31). Optimizers are
    # rebuilt after every rung because fixed-contact pruning replaces modules.
    pruning_ladder: dict[str, Any] = field(
        default_factory=lambda: {"enabled": False, "rungs": []}
    )
    # Cache the teacher's layer inputs/targets once over an ALIGNED
    # non-overlapping window grid of the training pool, then serve training
    # batches from that cache instead of running the frozen teacher every
    # step. This changes the sampling protocol (aligned grid instead of
    # random offsets) and must therefore be held constant within a campaign;
    # evaluation stays live. Layerwise hf_text only; the canonical analog of
    # the span campaign's cached-entry optimization, and the difference
    # between minutes and hours per local fit once the teacher is large.
    # Re-initialize softplus pathways that start in the transform's dead
    # zone (measured: strict-positive input paths get ~300x less gradient
    # than signed ones and cannot traverse to the live region within screen
    # budgets). Init-only and opt-in: enabling it changes the config hash, so
    # frozen campaigns stay exactly reproducible.
    positive_pathway_rebalance: bool = False
    teacher_pair_cache: bool = False
    teacher_pair_cache_max_windows: int = 256
    save_dir: str = "results/transformer_replacement_distillation"
    save_replacements: bool = True
    # Opt-in because a full 7B+ state dict is a large artifact. The saved state
    # is reconstructed by applying the same YAML replacement config first.
    save_student_model: bool = False
    use_amp: bool = False

    def __post_init__(self) -> None:
        ladder = self.pruning_ladder
        if isinstance(ladder, Mapping) and bool(ladder.get("enabled", False)):
            # The ladder is consumed by the layerwise hf_text loop
            # (training/_transformer_replacement/loops.py), which prunes,
            # rebuilds the optimizer, resets the best-checkpoint tracker to
            # the post-prune state, and recovers per rung. Every other mode
            # would still train, export unpruned, and report compression from
            # never-reduced contact counts -- a silently dead scientific
            # setting -- so those continue to fail closed until wired.
            supported = (
                str(self.mode) == "layerwise_distillation"
                and str(self.teacher_source) == "hf_text"
            )
            if not supported:
                raise NotImplementedError(
                    "training.pruning_ladder is only consumed by "
                    "mode='layerwise_distillation' with "
                    "teacher_source='hf_text'; with "
                    f"mode={self.mode!r} and "
                    f"teacher_source={self.teacher_source!r} the run would "
                    "silently skip every rung. Disable the ladder or use "
                    "scripts/deployment/prune_checkpoint.py."
                )


@dataclass
class VisionReplacementTrainingConfig(BaseConfig):
    """Training options for vision-backbone replacement distillation.

    Standard supervised training of a vision replacement uses
    ``training.main`` after ``model.vision_replacement.enabled=true`` builds the
    encoder/core/decoder model.  This block covers block-output distillation,
    where cached inputs to the replaced span and cached outputs from the
    original span are used to train the replacement core directly.  Image-level
    cache mode can also train the replacement plus decoder or the full student.
    """

    enabled: bool = False
    mode: str = "blockwise_distillation"
    teacher_source: str = "tensor_cache"
    train_target: str = "replacement_only"
    device: str = "auto"
    dtype: str = "float32"
    seed: int | None = None

    max_steps: int = 200
    eval_every: int = 50
    log_every: int = 25
    batch_size: int = 32
    # DataLoader controls for tensor/image-cache distillation. None uses
    # conservative auto-selection: in-memory tensor caches stay single-process,
    # while non-tensor datasets can use workers.
    num_workers: Optional[int] = None
    pin_memory: Optional[bool] = None
    persistent_workers: Optional[bool] = None
    prefetch_factor: Optional[int] = None
    in_order: Optional[bool] = None

    tensor_cache: dict[str, Any] = field(
        default_factory=lambda: {
            "train_inputs": "",
            "train_targets": "",
            "valid_inputs": "",
            "valid_targets": "",
        }
    )

    loss: str = "mse"
    cosine_weight: float = 0.0
    max_grad_norm: float | None = None
    save_dir: str = "results/vision_replacement_distillation"
    save_replacement: bool = True
    # Opt-in final deployment conversion. Training continues to use the
    # configured dynamic/rewiring/dense-to-sparse modules; export stores the
    # exact retained contacts as fixed IndexedSparseLinear modules.
    freeze_sparse_topology_on_export: bool = False
    stochastic_topology_freeze_policy: str = "reject"
    stochastic_topology_freeze_seed: int = 0
    ragged_topology_format: str = "csr"
    replacement_checkpoint_encoding: str = "auto"  # auto | raw | uint | bitmask
    use_amp: bool = False


@dataclass
class ThreeFactorConfig(BaseConfig):
    """Parameters for 3-factor (3F) base learning rule.

    The 3F rule uses: presynaptic * postsynaptic * error
    """

    # Compartmental dynamics
    # "auto": conductance-aware updates for shunting layers, additive-consistent
    # updates for additive layers.
    # "conductance": force conductance-aware updates (legacy behavior).
    # "additive": force additive-consistent updates.
    dynamics_mode: str = "auto"  # "auto", "conductance", "additive"
    use_conductance_scaling: bool = True  # Scale by R_tot = 1/g_tot
    use_driving_force: bool = True  # Use (E_rev - V_n) driving force
    theta: float = 0.0  # Leak reversal potential
    e_rev_exc: float = 1.0  # Excitatory reversal potential
    e_rev_inh: float = 0.0  # Inhibitory reversal (shunting)

    # Additive gain control mode (only used when dynamics_mode is "additive").
    # "none"            : post_factor = e_n  (current default, no local modulation)
    # "input_dependent" : post_factor = e_n * pseudo_R * pseudo_drive
    #                     pseudo_R = 1/(1 + |activations|), pseudo_drive = (v_mean - v_n + 1)
    # "running_stats"   : post_factor = e_n * (1 / running_std_n)
    # "learned_gain"    : post_factor = e_n * sigmoid(learnable_param_n)
    additive_gain_mode: str = "none"
    additive_stats_ema_alpha: float = 0.01  # EMA rate for running_stats mode


@dataclass
class FourFactorConfig(BaseConfig):
    """Parameters for 4-factor (4F) morphology correlation.

    Adds morphology factor rho to 3F rule.
    """

    # Morphology correlation mode
    rho_mode: str = "pearson"  # "pearson", "dot", "none"
    # Estimator for single-sample scenarios
    rho_estimator: str = "ema"  # "ema", "augment", "ema+augment"
    # EMA smoothing rate for rho statistics
    ema_alpha: float = 0.05
    # Layer-wise scaling: rho / sqrt(layer_depth)
    layer_wise_rho_scale: float = 1.0
    # Augmentation parameters (for rho_estimator="augment")
    augment_k: int = 8  # Number of augmented samples
    augment_noise_sigma: float = 0.01  # Noise level


@dataclass
class FiveFactorConfig(BaseConfig):
    """Parameters for 5-factor (5F) conditional information proxy.

    Adds information factor φ to 4F rule.
    """

    # Information proxy mode
    phi_mode: str = "conditional"  # "conditional", "variance"
    # Estimator for single-sample scenarios
    phi_estimator: str = "conditional_ema"  # "conditional_ema", "variance_ema"
    # Ridge regularization for conditional regression
    phi_ridge_lambda: float = 1e-3
    # Layer-wise scaling: φ / sqrt(layer_depth)
    layer_wise_phi_scale: float = 1.0
    # RLS forgetting factor (if using RLS-based estimators)
    rls_forgetting: float = 0.99
    # Stability clamp for the phi confidence factor
    phi_clamp_min: float = 0.25
    phi_clamp_max: float = 4.0


@dataclass
class MorphologyAwareConfig(BaseConfig):
    """Parameters for morphology-aware extensions.

    These extensions use dendritic tree topology explicitly.
    """

    # Path-integrated propagation: approximate ∏ R_k g_k attenuation
    use_path_propagation: bool = False
    path_factor_mode: str = "per_branch"  # "per_branch", "scalar_mean"

    # Branch-specific depth modulation: rho_j = rho_base / (depth_j + alpha)
    morphology_modulator_mode: str = "none"  # "none", "depth", "centrality"
    morphology_depth_offset: float = 1.0  # alpha parameter
    morphology_centrality_metric: str = "betweenness"  # For centrality mode

    # Dendritic normalization: Δg_j ← Δg_j / (Σ_k g_k + ε)
    use_dendritic_normalization: bool = False

    # Apical vs basal branch differentiation
    use_branch_type_rules: bool = False
    apical_branch_scale: float = 1.0  # Scale for apical (feedback) branches
    basal_branch_scale: float = 1.0  # Scale for basal (feedforward) branches
    use_branch_length_modulation: bool = False

    # Router/pathway-derived branch roles (preferred over heuristic branch types)
    use_branch_role_rules: bool = False
    branch_role_source: str = "router_pathways"  # "router_pathways"
    specialized_branch_scale: float = 1.0
    mixed_branch_scale: float = 1.0
    branch_role_power: float = 1.0
    branch_role_alignment_weight: float = 0.0


@dataclass
class HSICConfig(BaseConfig):
    """Parameters for HSIC auxiliary loss."""

    enabled: bool = False
    weight: float = 0.0
    self_weight: float = 1.0  # Self-decorrelation
    target_weight: float = 1.0  # Target-correlation
    target_source: str = "labels"  # "labels" or "logits"
    # Kernel configuration
    kernel: str = "linear"  # "linear", "rbf", "polynomial"
    sigma: float = 1.0  # RBF bandwidth
    degree: int = 2  # Polynomial degree
    coef0: float = 1.0  # Polynomial coefficient
    # Stability
    grad_clip_value: float = 0.0
    warmup_epochs: int = 0
    apply_last_layer_only: bool = False


@dataclass
class STDPConfig(BaseConfig):
    """Parameters for spike-timing / trace-based local plasticity.

    The rule is implemented at the same TopK synapse sites used by LocalCA. It
    can run alone via ``rule_variant: "stdp"`` or as an auxiliary update on top
    of 3F/4F/5F by setting ``stdp.enabled: true``.
    """

    enabled: bool = False
    # Pathways are TopK inputs recorded on each DendriticBranchLayer:
    # "exc", "inh", "rec_exc", "rec_inh".
    apply_to: list[str] = field(default_factory=lambda: ["exc", "rec_exc"])
    activity_mode: str = "relu"  # "relu", "identity", "binary", "abs"
    pre_threshold: float = 0.0
    post_threshold: float = 0.0
    tau_pre: float = 20.0
    tau_post: float = 20.0
    a_plus: float = 1.0
    a_minus: float = 0.5
    learning_rate_scale: float = 1.0
    inhibitory_update_sign: float = -1.0
    use_error_modulation: bool = False
    error_modulation_mode: str = "scalar_abs"  # "scalar_abs", "scalar_signed"
    clamp_update: float = 0.0
    detach_traces: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.apply_to, str):
            self.apply_to = [self.apply_to]
        elif self.apply_to is None:
            self.apply_to = ["exc", "rec_exc"]
        self.apply_to = [
            str(pathway).strip().lower().replace("-", "_") for pathway in self.apply_to
        ]
        self.activity_mode = str(self.activity_mode).strip().lower()
        self.error_modulation_mode = (
            str(self.error_modulation_mode).strip().lower().replace("-", "_")
        )


@dataclass
class InhibitoryHomeostasisConfig(BaseConfig):
    """Parameters for local inhibitory homeostasis updates.

    These auxiliary updates are applied only to inhibitory synapses and are
    designed to steer branches toward stable shunting regimes using local state.
    """

    enabled: bool = False
    mode: str = "r_tot"  # "r_tot", "voltage"
    weight: float = 0.0
    target_r_tot: float = 0.35
    target_voltage: float = 0.15


@dataclass
class LocalVoltageHomeostasisConfig(BaseConfig):
    """Parameters for strictly local voltage-centering updates.

    These auxiliary updates use only branch-local voltages and the local
    sensitivity terms already available to LocalCA. They are intended as a
    local counterpart to the separate voltage-stabilization pretraining
    strategy, not as a replacement for the task-driven credit signal.
    """

    enabled: bool = False
    weight: float = 0.0
    target_voltage: float = 0.5


@dataclass
class LocalGateHomeostasisConfig(BaseConfig):
    """Parameters for strictly local gate-occupancy updates.

    These updates act only on reactivation parameters and use only each
    layer's own gate outputs plus local activation derivatives. This is the
    local-learning counterpart to the global gate-occupancy warmup strategy.
    """

    enabled: bool = False
    center_weight: float = 0.0
    saturation_weight: float = 0.0
    target_mean: float = 0.5
    target_saturation: float = 0.2
    low_threshold: float = 0.1
    high_threshold: float = 0.9
    saturation_slope: float = 20.0


@dataclass
class LocalRuleConfig(BaseConfig):
    """Configuration for local learning rules.

    Organizes parameters by learning rule method for clarity.
    Each rule (3F/4F/5F/STDP) has its own nested configuration.
    """

    # ========== CORE CONFIGURATION ==========
    rule_variant: str = "3f"  # "3f", "4f", "5f", "stdp" (also accepts *_vh aliases)

    # ========== ERROR SIGNAL ==========
    error_mode: str = "auto"  # "auto", "mse", "ce", "bce"
    # Between-neuron production of the per-soma error, orthogonal to
    # error_broadcast_mode (which distributes it within each arbor).
    # "decoder": exact readout Jacobian (delta_core = J^T delta_out).
    # "dfa": fixed random feedback matrix in place of the decoder Jacobian,
    #        matching the SomaDFA construction (seed/scale conventions).
    soma_error_source: str = "decoder"
    dfa_feedback_seed: Optional[int] = None  # None -> trainer seed
    dfa_feedback_scale: float = 1.0
    error_broadcast_mode: str = (
        "scalar"  # "scalar", legacy "per_soma", "per_soma_shared", "per_soma_shuffled", "local_mismatch", "low_rank", "path_transport", "pathway_vector"
    )
    broadcast_seed: int = 0  # Fixed routing seed for randomized feedback controls
    error_noise_sigma: float = (
        0.0  # Additive Gaussian noise on broadcast error (0 = off)
    )
    broadcast_rank: int = 4  # Rank for low-rank vectorized broadcast
    broadcast_init_scale: float = 1.0  # Scale for low-rank random broadcast
    pathway_broadcast_residual: float = 0.25  # Blend with direct local error
    pathway_activity_gate_strength: float = 1.0  # Modulate by local pathway activity

    # ========== BROADCAST BANDWIDTH CONTROL ==========
    # Reduce broadcast precision to test low-bandwidth biological plausibility
    broadcast_bandwidth: str = "full"  # "full", "sign_only", "quantized", "sparse_topk"
    broadcast_bits: int = (
        8  # Quantization bits (2, 4, 8); only used when bandwidth="quantized"
    )
    broadcast_topk_fraction: float = (
        0.3  # Fraction of neurons to keep; only for "sparse_topk"
    )

    # ========== RULE-SPECIFIC PARAMETERS ==========
    # 3-factor rule parameters
    three_factor: ThreeFactorConfig = field(default_factory=ThreeFactorConfig)

    # 4-factor rule parameters (includes 3F)
    four_factor: FourFactorConfig = field(default_factory=FourFactorConfig)

    # 5-factor rule parameters (includes 4F)
    five_factor: FiveFactorConfig = field(default_factory=FiveFactorConfig)

    # Morphology-aware extensions
    morphology_aware: MorphologyAwareConfig = field(
        default_factory=MorphologyAwareConfig
    )

    # HSIC auxiliary loss
    hsic: HSICConfig = field(default_factory=HSICConfig)

    # Trace-based spike-timing plasticity rule
    stdp: STDPConfig = field(default_factory=STDPConfig)

    # Local inhibitory homeostasis
    inhibitory_homeostasis: InhibitoryHomeostasisConfig = field(
        default_factory=InhibitoryHomeostasisConfig
    )

    # Strictly local voltage-centering auxiliary
    voltage_homeostasis: LocalVoltageHomeostasisConfig = field(
        default_factory=LocalVoltageHomeostasisConfig
    )

    # Strictly local gate-occupancy auxiliary
    gate_homeostasis: LocalGateHomeostasisConfig = field(
        default_factory=LocalGateHomeostasisConfig
    )

    # ========== PARAMETER UPDATE CONTROL ==========
    update_inactive_weights: bool = False
    update_reactivation: bool = True
    # LocalCA-only policy for explicit inhibitory DendriNet populations.
    # "local_ca": update I-cell dendrites with the same local rule machinery
    #             used for E-cell dendrites. This is the matched/comparable mode.
    # "freeze": skip LocalCA updates for explicit I-cell dendrites while still
    #           updating I-to-E synapses on excitatory dendrites.
    explicit_inhibitory_update_mode: str = "local_ca"
    # Deprecated compatibility shim. If set by old YAMLs, it is converted to
    # explicit_inhibitory_update_mode in __post_init__.
    update_explicit_inhibitory_cells: Optional[bool] = None
    encoder_update_mode: str = "none"  # "backprop", "none"
    decoder_update_mode: str = "backprop"  # "backprop", "local", "none"

    # ========== TRAINING SCHEDULE ==========
    freeze_encoder_epochs: int = 0
    freeze_reactivation_epochs: int = 0
    freeze_decoder_epochs: int = 0
    training_schedule: Optional[list[dict[str, Any]]] = None

    # ========== OPTIMIZATION ==========
    clip_grad_value: float = 5.0
    normalize_by_batch: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.stdp, dict):
            self.stdp = STDPConfig(**self.stdp)

        normalized_source = str(self.soma_error_source).strip().lower()
        if normalized_source not in {"decoder", "dfa"}:
            raise ValueError(
                "soma_error_source must be 'decoder' or 'dfa', got "
                f"{self.soma_error_source!r}"
            )
        self.soma_error_source = normalized_source
        if self.dfa_feedback_scale <= 0:
            raise ValueError("dfa_feedback_scale must be positive")

        normalized_rule = str(self.rule_variant).strip().lower().replace("-", "_")
        stdp_aliases = {
            "stdp",
            "spike_timing",
            "spike_timing_dependent_plasticity",
            "trace_stdp",
        }
        if normalized_rule in stdp_aliases:
            self.rule_variant = "stdp"
            self.stdp.enabled = True
        else:
            self.rule_variant = normalized_rule

        mode = self._normalize_explicit_inhibitory_update_mode(
            self.explicit_inhibitory_update_mode
        )
        legacy_update = self.update_explicit_inhibitory_cells
        if legacy_update is not None:
            legacy_mode = "local_ca" if bool(legacy_update) else "freeze"
            if mode != "local_ca" and mode != legacy_mode:
                raise ValueError(
                    "Conflicting explicit inhibitory update settings: "
                    f"explicit_inhibitory_update_mode={mode!r} but "
                    f"update_explicit_inhibitory_cells={legacy_update!r}"
                )
            mode = legacy_mode

        self.explicit_inhibitory_update_mode = mode
        self.update_explicit_inhibitory_cells = mode == "local_ca"

    @staticmethod
    def _normalize_explicit_inhibitory_update_mode(mode: Any) -> str:
        if isinstance(mode, bool):
            return "local_ca" if mode else "freeze"
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "local_ca": "local_ca",
            "same_as_excitatory": "local_ca",
            "same_local_rule": "local_ca",
            "update": "local_ca",
            "train": "local_ca",
            "true": "local_ca",
            "freeze": "freeze",
            "none": "freeze",
            "off": "freeze",
            "false": "freeze",
        }
        if normalized not in aliases:
            allowed = "'local_ca' or 'freeze'"
            raise ValueError(
                f"Invalid explicit_inhibitory_update_mode={mode!r}; expected {allowed}"
            )
        return aliases[normalized]


@dataclass
class ReportingConfig(BaseConfig):
    """Configuration for reporting metrics."""

    report_non_pruned: bool = True  # Report number of non-pruned synapses
    save_pruning_stats: bool = True  # Save pruning statistics to file
    detailed_branch_report: bool = True  # Detailed per-branch reporting
    redo_analysis_after_pruning: bool = (
        True  # Redo all analyses after pruning for comparison
    )


@dataclass
class TrainingConfig(BaseConfig):
    """Complete training configuration."""

    encoder: EncoderTrainingConfig = field(default_factory=EncoderTrainingConfig)
    main: MainTrainingConfig = field(default_factory=MainTrainingConfig)
    transformer_replacement: TransformerReplacementTrainingConfig = field(
        default_factory=TransformerReplacementTrainingConfig
    )
    vision_replacement: VisionReplacementTrainingConfig = field(
        default_factory=VisionReplacementTrainingConfig
    )
