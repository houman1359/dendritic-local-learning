"""Versioned, evidence-scoped recipes for dendritic replacement training.

The recipes in this module are not universal hyperparameter optima.  They are
either fixed comparison protocols or measured priors from a named experiment
domain.  Keeping that distinction in executable metadata prevents a useful
campaign observation (for example, the OLMo-3-7B learning-rate screen) from
silently becoming a claim about every teacher, morphology, or scale.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

REPLACEMENT_TRAINING_RECIPE_SCHEMA = "dendritic_replacement_training_recipe/v1"


@dataclass(frozen=True)
class ReplacementTrainingRecipe:
    """One auditable training prior or fixed comparison protocol."""

    recipe_id: str
    status: str
    scope: str
    optimizer_overrides: dict[str, Any]
    training_overrides: dict[str, Any]
    orchestration: dict[str, Any]
    evidence: tuple[str, ...]
    cautions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = REPLACEMENT_TRAINING_RECIPE_SCHEMA
        return payload


_CAMPAIGN_ROOT = "drafts/dendriLLM/experiments/stage2_olmo3_7b_depth_density"


REPLACEMENT_TRAINING_RECIPES: dict[str, ReplacementTrainingRecipe] = {
    "fmi_reference_grid_v1": ReplacementTrainingRecipe(
        recipe_id="fmi_reference_grid_v1",
        status="fixed_comparison_protocol",
        scope=(
            "Small planted-mechanism PopulationNetwork reference grid; identical "
            "training budget for every candidate."
        ),
        optimizer_overrides={
            "name": "adamw",
            "lr": 0.005,
            "weight_decay": 0.0001,
        },
        training_overrides={
            "mode": "layerwise_distillation",
            "train_target": "replacement_only",
            "distillation_target": "teacher_function",
            "replacement_dtype": "float32",
            "sequence_length": 4,
            "train_samples": 512,
            "valid_samples": 128,
            "batch_size": 32,
            "max_steps": 120,
            "eval_every": 20,
            "log_every": 20,
            "loss": "mse",
            "restore_best_replacement": True,
        },
        orchestration={
            "context": "teacher",
            "grouping": "single planted component",
            "checkpoint_selection": "lowest validation loss including step 0",
        },
        evidence=(
            "configs/examples/population_ei_fmi_hybrid_synthetic.yaml",
            "tests/test_fmi_v2_reference_grid_scripts.py",
        ),
        cautions=(
            "This is a fairness protocol, not evidence that its learning rate or "
            "batch size is optimal for a real model.",
            "The smoke profiler budget is an integration test and is not admissible "
            "selection evidence.",
        ),
    ),
    "olmo3_7b_staged_layerwise_v1": ReplacementTrainingRecipe(
        recipe_id="olmo3_7b_staged_layerwise_v1",
        status="measured_prior",
        scope=(
            "OLMo-3-7B SwiGLU FFN replacements at density 0.25, trained in "
            "depth-ordered groups of four on student-context activations."
        ),
        optimizer_overrides={
            "name": "adamw",
            "lr": 3.0e-4,
            "weight_decay": 0.0,
        },
        training_overrides={
            "mode": "layerwise_distillation",
            "train_target": "replacement_only",
            "distillation_target": "teacher_function",
            "replacement_dtype": "float32",
            "sequence_length": 64,
            "train_samples": 64,
            "valid_samples": 16,
            "batch_size": 1,
            "max_steps": 300,
            "eval_every": 25,
            "log_every": 1,
            "loss": "relative_mse",
            "cosine_weight": 0.1,
            "restore_best_replacement": True,
            "freeze_sparse_topology_on_export": True,
            "replacement_checkpoint_encoding": "auto",
        },
        orchestration={
            "context": "student",
            "group_size": 4,
            "group_order": "increasing_depth",
            "target": "teacher module output on the composed-student input",
            "pre_patch_completed_groups": True,
            "joint_recovery": "optional_after_all_groups",
        },
        evidence=(
            f"{_CAMPAIGN_ROOT}/fp32ctrl_lr1e4.yaml",
            f"{_CAMPAIGN_ROOT}/fp32ctrl_lr3e4.yaml",
            f"{_CAMPAIGN_ROOT}/fp32ctrl_lr1e3.yaml",
            f"{_CAMPAIGN_ROOT}/staged32fn4_stage0.yaml",
            f"{_CAMPAIGN_ROOT}/RESULTS.md#fp32-control-verdict-the-architecture-was-never-the-problem-2026-08-16",
            f"{_CAMPAIGN_ROOT}/RESULTS.md#the-trained-composition-finale-2026-08-17",
        ),
        cautions=(
            "The 3e-4 result is a measured OLMo-3-7B prior, not a universal "
            "learning-rate optimum.",
            "Function matching beat first-order trajectory anchoring at 32 layers, "
            "although trajectory anchoring won at shorter residual depth.",
            "The full-stack held-out result still required composition-aware recovery.",
        ),
    ),
    "olmo3_7b_span_exit_v1": ReplacementTrainingRecipe(
        recipe_id="olmo3_7b_span_exit_v1",
        status="measured_prior",
        scope=(
            "OLMo-3-7B depth-ordered groups trained jointly against the clean "
            "teacher hidden state at each group exit."
        ),
        optimizer_overrides={
            "name": "adamw",
            "lr": 3.0e-4,
            "weight_decay": 0.0,
        },
        training_overrides={
            "mode": "joint_lm_distillation",
            "train_target": "replacement_only",
            "replacement_dtype": "float32",
            "sequence_length": 512,
            "batch_size": 1,
            "max_steps": 400,
            "eval_every": 25,
            "log_every": 10,
            "lm_loss_weight": 0.0,
            "kl_loss_weight": 0.0,
            "hidden_loss_weight": 1.0,
            "restore_best_replacement": True,
            "freeze_sparse_topology_on_export": True,
            "replacement_checkpoint_encoding": "auto",
        },
        orchestration={
            "context": "student",
            "group_size": 8,
            "group_order": "increasing_depth",
            "target": "clean teacher hidden state at current group exit",
            "pre_patch_completed_groups": True,
            "compare_group_sizes": [4, 8],
            "checkpoint_selection": "frozen held-out objective ladder",
        },
        evidence=(
            "src/dendritic_modeling/scripts/transformer_replacement/span_replace.py",
            "src/dendritic_modeling/scripts/transformer_replacement/span_replace_sharded.py",
            f"{_CAMPAIGN_ROOT}/RESULTS.md#span--mixed-recovery-new-best-generalizing-all-32--overtraining-knee-2026-08-18-job-40033257",
        ),
        cautions=(
            "The historical OLMo result used groups of eight; MoE measurements "
            "favored more frequent recapture with groups of four, so group size "
            "must remain a sweep axis.",
            "This recipe transfers the training objective to PopulationNetwork "
            "replacements; it does not relabel the historical SparseGLU result as "
            "an E/I or dendritic-morphology result.",
        ),
    ),
    "mixed_joint_recovery_v1": ReplacementTrainingRecipe(
        recipe_id="mixed_joint_recovery_v1",
        status="measured_prior",
        scope=(
            "Warm-started, replacement-only global recovery of a composed causal "
            "LM on a mixed-register text stream."
        ),
        optimizer_overrides={
            "name": "adamw",
            "lr": 1.0e-4,
            "weight_decay": 0.0,
        },
        training_overrides={
            "mode": "joint_lm_distillation",
            "train_target": "replacement_only",
            "replacement_dtype": "float32",
            "sequence_length": 1024,
            "batch_size": 1,
            "max_steps": 1000,
            "eval_every": 125,
            "log_every": 25,
            "lm_loss_weight": 1.0,
            "kl_loss_weight": 1.0,
            "kl_temperature": 1.0,
            "hidden_loss_weight": 0.0,
            "restore_best_replacement": True,
        },
        orchestration={
            "warm_start_required": True,
            "training_distribution": "mixed registers",
            "first_checkpoints": [125, 250],
            "continue_if_held_out_improves": True,
            "selection": "frozen held-out rung; capability battery on selected rung",
            "matched_dense_control": True,
            "batch_size_sweep": [1, 2],
        },
        evidence=(
            "src/dendritic_modeling/scripts/transformer_replacement/span_recover.py",
            "src/dendritic_modeling/scripts/transformer_replacement/span_recover_fsdp.py",
            f"{_CAMPAIGN_ROOT}/CAMPAIGN_REPORT_20260822.md#42-recovery-budget-was-the-largest-missed-lever",
            f"{_CAMPAIGN_ROOT}/RESULTS.md#2026-08-23--recovery-budget-saturates-by-250-steps-and-then-regresses",
        ),
        cautions=(
            "About 125--250 steps is an efficient first rung, not a universal stop; "
            "some high-compression arms improved much later.",
            "The apparent 6.5% effective-batch gain came from a throughput smoke run "
            "and requires a controlled batch-size sweep before becoming a default.",
            "Selection and final reporting must use separate frozen data, and every "
            "quality claim needs a matched dense no-compression control.",
        ),
    ),
}


def get_replacement_training_recipe(recipe_id: str) -> ReplacementTrainingRecipe:
    """Return a registered recipe or fail with the available identifiers."""

    normalized = str(recipe_id).strip()
    try:
        return REPLACEMENT_TRAINING_RECIPES[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(REPLACEMENT_TRAINING_RECIPES))
        raise ValueError(
            f"Unknown replacement training recipe {recipe_id!r}; available: {available}"
        ) from exc


def apply_replacement_training_recipe(
    config: dict[str, Any], recipe_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply a deliberately selected recipe and return config plus provenance.

    Recipe-controlled values overwrite the base config.  This is intentional:
    selecting a named recipe means selecting its comparison protocol.  Callers
    that need a different value should create/version a new recipe or apply an
    explicitly recorded post-recipe override in their experiment generator.
    """

    recipe = get_replacement_training_recipe(recipe_id)
    result = deepcopy(config)
    training = result.setdefault("training", {})
    optimizer = training.setdefault("main", {}).setdefault("optimizer", {})
    optimizer.update(deepcopy(recipe.optimizer_overrides))
    replacement = training.setdefault("transformer_replacement", {})
    replacement.update(deepcopy(recipe.training_overrides))
    manifest = recipe.as_dict()
    replacement["training_recipe"] = recipe.recipe_id
    replacement["training_recipe_manifest"] = deepcopy(manifest)
    return result, manifest


__all__ = [
    "REPLACEMENT_TRAINING_RECIPES",
    "REPLACEMENT_TRAINING_RECIPE_SCHEMA",
    "ReplacementTrainingRecipe",
    "apply_replacement_training_recipe",
    "get_replacement_training_recipe",
]
