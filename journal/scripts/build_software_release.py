#!/usr/bin/env python3
"""Build and validate the reviewer software archive.

The release has three deliberately distinct source trees:

* ``dendritic_modeling/`` is exported from the implementation repository's committed HEAD,
  including its installable src-layout package.
  Uncommitted working-tree changes never enter this snapshot.
* ``journal_package/`` separately exports the clean paper repository, including
  manuscript sources and historical vector figure components.
* ``article_analysis/`` contains an explicit allow-list from this journal
  package. It includes analysis code, frozen configurations and three worker
  records required to authenticate the new protocols. Raw data, checkpoints,
  other scheduler scripts, logs and caches are excluded.

The exported files are deterministic for fixed implementation/paper commits.
Archive metadata also records the implementation working-tree state at build time.
Machine-specific paths in historical, committed auxiliary configurations are
replaced with documented placeholders in the release copy only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import time
import zipfile
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
PAPER_REPOSITORY_ROOT = JOURNAL_ROOT.parent
SUBMISSION_ROOT = JOURNAL_ROOT / "submission"
STAGE_NAME = "software_release"
ARCHIVE_NAME = "Dendritic_credit_assignment_software.zip"
PHYSICAL_RUNTIME_COMMIT = "a99c3a777f99913e13dfe673a3f3a28bfe3566af"
PHYSICAL_RUNTIME_DIRECTORY = "historical_runtimes/physical_depth_a99c3a7"
IMAGE_RUNTIME_COMMIT = "6c1aaa25abd056c417842e1c46378b65d036f6a7"
IMAGE_RUNTIME_DIRECTORY = "historical_runtimes/image_ladder_6c1aaa2"

# Only the explicitly selected committed package and article sources are retained.  These replacements affect only
# historical machine-local defaults; each changed file is listed in
# PORTABILITY_PATCHES.tsv.  Tokens are intentionally conspicuous so that a
# reviewer cannot mistake them for working paths.
PORTABILITY_REPLACEMENTS: tuple[tuple[str, str, str], ...] = (
    ("/n/home13/hsafaai/.cache/torch", "${TORCH_HOME}",
     "replace optional pretrained-feature cache root in historical runtime"),
    (
        "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
        "journal_extension_20260828",
        "${DENDRITIC_RUNS_ROOT}",
        "replace frozen path-demand execution root",
    ),
    (
        "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
        "journal_extension_20260820",
        "${DENDRITIC_RUNS_ROOT}",
        "replace frozen project-B execution root",
    ),
    (
        "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING",
        "${DENDRITIC_RUNS_ROOT}",
        "replace separately archived population/review execution root",
    ),
    (
        "/n/holylabs/kempner_dev/Users/hsafaai/Code/"
        ".dendritic-modeling-journal-runtimes",
        "${DENDRITIC_RUNS_ROOT}",
        "replace clean-source execution root",
    ),
    (
        "/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling",
        "${DENDRITIC_MODELING_ROOT}",
        "replace archived figure-generator repository root",
    ),
    (
        "/n/holylabs/LABS/kempner_dev/Users/hsafaai/conda_envs/"
        "vdc_paper_v2/bin/python",
        "python",
        "replace historical environment interpreter with PATH lookup",
    ),
    (
        "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling",
        "${DENDRITIC_MODELING_ROOT}",
        "replace historical repository root",
    ),
    (
        "/n/holylabs/kempner_dev/Users/hsafaai/dendrinet_revision_2026",
        "${DENDRITIC_EXTERNAL_ARTIFACTS}",
        "replace external-artifact root not distributed with the article",
    ),
    (
        "/n/holylabs/kempner_dev/Users/hsafaai/dendrinet_rnn_runs_protected",
        "${DENDRITIC_RUNS_ROOT}",
        "replace historical run root not distributed with the article",
    ),
    (
        "/n/holylabs/kempner_dev/Users/hsafaai",
        "${DENDRITIC_LOCAL_WORKSPACE}",
        "replace auxiliary production workspace, environment and cached-input roots",
    ),
    (
        "/n/holylabs/...",
        "${DENDRITIC_LOCAL_WORKSPACE}",
        "replace illustrative cluster-root placeholder in production help text",
    ),
)

JOURNAL_DIRECTORIES = (
    "code", "configs", "tests", "reproducibility",
    "scripts/morphology_structure", "scripts/morphology_dynamics",
    "scripts/morphology_credit", "scripts/morphology_calibration",
    "scripts/morphology_conductance",
    "scripts/boolean_morphology", "scripts/boolean_theory",
    "scripts/credit_rule_bridge", "scripts/credit_resolution_bridge",
    "scripts/anatomy_commonmode", "scripts/physical_depth_budget",
    "scripts/shunt_ancestry_gain", "scripts/credit_first_figures",
    "scripts/conductance_credit_demand", "scripts/image_ladder_controls",
    "scripts/conductance_local_gate", "scripts/credit_rule_extension",
    "scripts/conductance_gate_generalization", "scripts/inhibitory_credit_transfer",
    "scripts/inhibitory_selection", "scripts/inhibitory_rescue",
    "scripts/review_completion", "scripts/submission_revision",
    "scripts/measured_alignment_power", "scripts/physical_depth_followup",
    "scripts/supplement_consolidation", "scripts/inherited_neurips",
)
JOURNAL_ANALYSIS_RECORDS = (
    "ANIMAL_DATA_CONTRACT.md",
    "CIFAR10_ADDITIVE_FEEDBACK_LADDER_CONFIRMATORY_20260828.md",
    "CREDIT_PHASE_THEORY_EXPERIMENT_CONTRACT_20260811.md",
    "EXPERIMENT_CONTRACT.md",
    "MICRONS_FUNCTIONAL_INHIBITORY_CONTRACT.md",
    "NONLINEAR_PHYSICAL_DEPTH_CONFIRMATORY_CONTRACT_20260812.md",
    "PINKY_V185_SECOND_ANIMAL_CONTRACT_20260820.md",
    "PATH_NECESSITY_CREDIT_CONFLICT_CONTRACT_20260828.md",
    "POSITIVE_CONDUCTANCE_STEP_CONSISTENT_CONTRACT_20260811.md",
    "SAME_SPAN_COEFFICIENT_LEARNING_CONTRACT_20260811.md",
    "TASK_FAMILY_ALIGNMENT_CONTRACT_20260820.md",
    "TRAINED_SUBTREE_ADDRESS_EXPERIMENT_CONTRACT.md",
)
JOURNAL_SCRIPTS = (
    'build_supplementary_figure_boolean_capacity_native.py',
    'build_supplementary_figure_boolean_learning_native.py',
    'build_supplementary_figure_conductance_grouping_native.py',
    'build_supplementary_figure_conductance_optimization_native.py',
    'build_supplementary_figure_conductance_precision_native.py',
    'build_supplementary_figure_credit_optimizer_controls_native.py',
    'build_supplementary_figure_error_field_geometry_native.py',
    'build_supplementary_figure_finite_horizon_native.py',
    'build_supplementary_figure_fixed_profile_budget_native.py',
    'build_supplementary_figure_local_gate_controls_native.py',
    'build_supplementary_figure_measured_transfer_geometry_native.py',
    'build_supplementary_figure_mnist_dictionary_geometry_native.py',
    'build_supplementary_figure_morphology_estimation_native.py',
    'build_supplementary_figure_oracle_profile_credit_native.py',
    'build_supplementary_figure_physical_architecture_native.py',
    'build_supplementary_figure_scalar_tree_capacity_native.py',
    'build_supplementary_figure_shunt_replication_native.py',
    'build_supplementary_figure_shunt_sensitivity_native.py',
    'build_supplementary_figure_utility_signal_noise_native.py',

    "passive_field_diagnostics.py",
    "aggregate_all_scan_functional_topology.py",
    "analyze_bandwidth_matched_routing.py",
    "analyze_branch_credit_interference.py",
    "analyze_credit_phase_existing.py",
    "analyze_review_evidence.py",
    "audit_review_curve_lineage.py",
    "analyze_fulltree_within_span_oracle.py",
    "rebuild_review_visual_corrections.py",
    "analyze_credit_phase_spectral_bound.py",
    "analyze_focal_gradient_decomposition.py",
    "analyze_focal_gradient_shapley.py",
    "analyze_francioni_signed_credit.py",
    "analyze_microns_inhibitory_routes.py",
    "analyze_nonlinear_physical_depth_confirmatory.py",
    "analyze_path_necessity_boundary.py",
    "analyze_pinky_v185_replication.py",
    "analyze_remaining_physical_experiments.py",
    "analyze_physical_cable_sensitivity.py",
    "analyze_prospective_followup_results.py",
    "analyze_prospective_learning_results.py",
    "analyze_positive_conductance_reliability.py",
    "analyze_reciprocal_routing_controls.py",
    "analyze_spatial_topology_audit.py",
    "analyze_same_span_coefficient_learning.py",
    "analyze_task_family_alignment_factorial.py",
    "audit_prospective_learning_runs.py",
    "audit_figure_style_lineage.py",
    "audit_nature_communications_format.py",
    "audit_submission.py",
    "build_journal_figures.py",
    "build_new_confirmatory_figures.py",
    "build_prospective_input_validity_audit.py",
    "build_alignment_animal_figure.py",
    "build_credit_phase_figure.py",
    "build_microns_v661_replication_figure.py",
    "build_nature_source_data.py",
    "build_path_necessity_fashion_figure.py",
    "build_positive_conductance_reliability_figure.py",
    "build_same_span_coefficient_figure.py",
    "build_focal_selectivity_figure.py",
    "build_regular_tree_regime_figure.py",
    "build_trained_subtree_address_figure.py",
    "build_static_microns_replication.py",
    "collect_feedback_gradient_rerun.py",
    "collect_feedback_rerun.py",
    "collect_clean_exact_bp_rerun.py",
    "export_figure2d_seed_data.py",
    "export_regular_tree_source_data.py",
    "fetch_expanded_microns_cohort.py",
    "figure1_neurips_components.py",
    "freeze_pinky_v185_cohort.py",
    "diagnose_nonlinear_physical_depth.py",
    "generate_nonlinear_physical_depth_confirmatory.py",
    "generate_nonlinear_physical_depth_sweeps.py",
    "generate_remaining_physical_experiments.py",
    "generate_task_family_alignment_factorial.py",
    "journal_style.py",
    "neurips_style.py",
    "prepare_pinky_v185_replication.py",
    "run_alignment_controlled_learning.py",
    "render_nonlinear_physical_depth_calibration.py",
    "run_credit_phase_theory_experiment.py",
    "run_focal_selectivity_active_ensemble.py",
    "run_focal_selectivity_phase1.py",
    "run_positive_conductance_reliability.py",
    "run_path_necessity_fashion.py",
    "run_reconstructed_tree_task_learning.py",
    "run_same_span_coefficient_learning.py",
    "run_trained_subtree_address_full_factorial.py",
    "run_trained_subtree_address_phase1.py",
    "summarize_static_microns_replication.py",
    "update_provenance_hashes.py",
)
# Current native figure builders and their local Python dependencies.
JOURNAL_SCRIPTS += (
    'current_source_data_inventory.py',
    'release_version.py',
    'tex_sources.py',
    'build_utility_supplement.py',
    'analyze_physical_depth_clean_source_replication.py',
    'assemble_compact_main_figures.py',
    'build_interior_optimum_figure.py',
    'build_main_figure_01.py',
    'build_main_figure_02.py',
    'build_main_figure_03.py',
    'build_main_figure_04.py',
    'build_main_figure_05.py',
    'build_main_figure_06.py',
    'build_main_figure_07.py',
    'build_main_figure_08.py',
    'build_main_figure_09.py',
    'build_main_figure_10.py',
    'build_supplementary_figure_s01_native.py',
    'build_supplementary_figure_s02_native.py',
    'build_supplementary_figure_s03_native.py',
    'build_supplementary_figure_s04_native.py',
    'build_supplementary_figure_s30_native.py',
    'build_trained_partition_residual_figure.py',
    'collect_prospective_mechanism_diagnostics.py',
    'credit_tree_schematics.py',
    'figure_canvas.py',
    'native_schematics.py',
    'routing_figure_panels.py',
)

# Completed prospective and review follow-ups, with local import dependencies.
JOURNAL_SCRIPTS += (
    'analyze_review_morphology_uncertainty.py',
    'analyze_review_response_baselines.py',
    'audit_review_response_baselines.py',
    'audit_prospective_morphology_selection.py',
    'build_main_figure_11.py',
    'rebuild_final_publication_figures.py',
    'build_adaptive_conductance_reliability_figure.py',
    'build_irregular_tree_wavelet_figure.py',
    'build_review_completion_figures.py',
    'build_review_response_baselines_figure.py',
    'build_supplementary_figure_s09_native.py',
    'build_supplementary_figure_s27_native.py',
    'build_supplementary_figure_s21_native.py',
    'build_supplementary_figure_s28_native.py',
    'fetch_review_fashion_cache.py',
    'fill_machine_learning_checklist_draft.py',
    'run_prospective_morphology_selection.py',
    'run_review_branch_trajectories.py',
    'run_review_coefficient_encoder.py',
    'run_review_coefficient_hard_readout.py',
)

# Historical generators still referenced by the canonical provenance ledger.
# Keep their original script paths so the restored package passes the same audit.
JOURNAL_SCRIPTS += (
    'analyze_cifar10_additive_feedback_ladder_confirmatory.py',
    'analyze_fashion_feedback_ladder.py',
    'analyze_fig2_path_gain_dispersion.py',
    'analyze_operator_argmax.py',
    'analyze_physical_alignment_dose.py',
    'analyze_point_dendrite_credit_controls.py',
    'analyze_route_dictionary_atlas.py',
    'build_capture_per_wire_figure.py',
    'collect_mnist_between_within_factorial.py',
    'collect_mnist_feedback_ladder.py',
)

# Release builders, manuscript auditors and local dependencies of released tests.
JOURNAL_SCRIPTS += (
    'audit_citations.py',
    'audit_latex_layout.py',
    'audit_neurips_text_overlap.py',
    'build_overleaf_bundle.py',
    'build_software_release.py',
    'build_submission_bundle.py',
    'combine_manuscript_pdfs.py',
    'build_main_figure_12.py',
    'build_boolean_morphology_figures.py',
    'build_morphology_followup_figures.py',
    'build_morphology_bridge_figures.py',
    'build_morphology_credit_figure_tables.py',
    'build_morphology_calibration_figure_tables.py',
    'build_supplementary_figure_s35_native.py',
    'build_supplementary_figures_s17_s20_native.py',
    'validate_supplementary_s17_s20_replay.py',
    'export_morphology_investigation_sources.py',
)

# Frozen protocols authenticate these execution records during portable replay.
# Retain exactly these three workers, with declared release-only path changes;
# they document execution and are not generic reviewer scheduler recipes.
FROZEN_WORKER_RECORDS = tuple(
    f"scripts/{study}/worker.sh"
    for study in (
        "conductance_local_gate", "credit_rule_extension", "measured_alignment_power"
    )
)
JOURNAL_SCRIPTS += tuple(name.removeprefix("scripts/") for name in FROZEN_WORKER_RECORDS)

ARCHIVED_ANALYSIS_SCRIPTS = (
    (
        Path("neurips/scripts/summarize_init_policy_factorial.py"),
        "Figure 2 architecture-by-initialization-policy factorial summarizer",
    ),
    (
        Path("neurips/scripts/measure_layer_soma_factorial.py"),
        "Figure 2 layer/soma feedback and backward-only gradient diagnostic",
    ),
    (
        Path("neurips/scripts/measure_theory_diagnostics.py"),
        "Figure 2 exact-gradient and compartment-error diagnostic dependency",
    ),
)
# No additional archived-only analysis is required. The complete-tree runner
# and its bounded common-checkpoint oracle replay are explicitly released above;
# raw response caches remain subject to upstream access and redistribution terms.
OPTIONAL_JOURNAL_ARCHIVED_SCRIPTS = ()

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "data",
    "external_data",
    "logs",
    "reproduced_results",
    "reproduction_runs",
    "results",
}
EXCLUDED_SUFFIXES = {
    ".sbatch",
    ".sh",
    ".ppt",
    ".pptx",
    ".odp",
    ".ckpt",
    ".err",
    ".log",
    ".out",
    ".pem",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
}
EXCLUDED_FILE_NAMES = {
    ".env",
    "credentials.json",
    "id_rsa",
    "secrets.json",
    # Its corresponding exploratory analysis is intentionally not released.
    "test_reconstructed_tree_task_learning.py",
}

PRIVATE_PATH_PATTERNS = (
    re.compile(rb"/n/(?:home[^/]*|holylabs)/"),
    re.compile(rb"/home/[A-Za-z0-9._-]+/"),
    re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
)
SECRET_PATTERNS = (
    ("private key", re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("AWS access key", re.compile(rb"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("OpenAI-style token", re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("Slack token", re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
)


def run_git(*args: str, repository_root: Path, text: bool = True) -> str | bytes:
    """Run Git against the source repository and return stdout."""

    completed = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=True,
        capture_output=True,
        text=text,
    )
    return completed.stdout


def run_journal_git(*args: str) -> str:
    """Run Git against the nested journal repository."""

    completed = subprocess.run(
        ["git", "-C", str(JOURNAL_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def discover_implementation_root(start: Path, explicit: Path | None = None) -> Path:
    """Find the actual installable package, never mistake the paper for it."""

    candidates = [explicit.resolve()] if explicit is not None else [start.resolve(), *start.resolve().parents]
    for candidate in candidates:
        if not (candidate / "pyproject.toml").is_file():
            continue
        if not (candidate / "src" / "dendritic_modeling").is_dir():
            continue
        top = Path(str(run_git("rev-parse", "--show-toplevel", repository_root=candidate)).strip()).resolve()
        if top != candidate:
            continue
        return candidate
    raise RuntimeError(
        "Cannot locate the production repository (pyproject.toml and "
        "src/dendritic_modeling). Pass --implementation-root when building "
        "from an isolated paper snapshot."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_submission_target(path: Path) -> None:
    """Reject output paths outside the journal submission directory."""

    submission = SUBMISSION_ROOT.resolve()
    resolved = path.resolve()
    if resolved == submission or submission not in resolved.parents:
        raise RuntimeError(f"Refusing unsafe output target: {resolved}")


def remove_generated_tree(path: Path) -> None:
    """Remove one validated generated tree, tolerating brief NFS latency."""

    ensure_submission_target(path)
    if not path.exists():
        return
    last_error: OSError | None = None
    for _ in range(8):
        try:
            shutil.rmtree(path)
            return
        except OSError as error:
            last_error = error
            time.sleep(0.25)
    assert last_error is not None
    raise last_error


# Frozen execution records of completed cluster runs (see
# reproducibility/audit_reproducibility.py FROZEN_EXECUTION_RECORDS): the
# confirmatory YAML is byte-pinned by its analyzer and all five carry
# site-specific paths that document what actually ran.  They are provenance
# records, not portable recipes, so the release omits them; their summaries,
# hashes and outcomes ship in source_data instead.
FROZEN_EXECUTION_RECORD_FILES = {
    "cifar10_additive_feedback_ladder_confirmatory.yaml",
    "cifar10_additive_operator_compatibility.yaml",
    "cifar10_bp_recipe_init_screen.yaml",
    "cifar10_credit_ladder_pilot.yaml",
    "cifar10_historical_bp_reproduction.yaml",
}


def excluded(relative: Path) -> bool:
    parts = set(relative.parts)
    if parts & EXCLUDED_DIRECTORY_NAMES:
        return True
    if relative.name in EXCLUDED_FILE_NAMES:
        return True
    if relative.name in FROZEN_EXECUTION_RECORD_FILES:
        return True
    if "/".join(relative.parts[-3:]) in FROZEN_WORKER_RECORDS:
        return False
    # The documented reviewer helper is portable and has no embedded data paths.
    # Match its directory as well as its name, including code-directory copying.
    if relative.parts[-2:] == ("release_noise", "cleanroom_worker.sh"):
        return False
    if relative.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    if relative.name.startswith("slurm_"):
        return True
    return False


def repository_file_allowed(relative: Path, scope: str | None) -> bool:
    """Select article sources before archive extraction, using explicit roots.

    Generic library modules are retained because the public network/config API
    imports them. Unrelated experiment drivers, configs, docs and tests are not.
    """
    if excluded(relative):
        return False
    name = relative.as_posix()
    if scope is None:
        return True
    if scope == "implementation":
        if name in {"pyproject.toml", "LICENSE", "README.md"}:
            return True
        prefix = "src/dendritic_modeling/"
        if not name.startswith(prefix):
            return False
        package_name = name[len(prefix):]
        if not package_name.startswith("scripts/"):
            return relative.suffix in {".py", ".json", ".yaml", ".yml", ".toml"}
        if package_name == "scripts/__init__.py":
            return True
        if package_name.startswith(("scripts/script_utils/", "scripts/sweeps/")):
            return relative.suffix == ".py"
        return package_name in {
            # The shared training API eagerly imports these artifact readers.
            # Keep only this dependency closure, not the text experiment suite.
            "scripts/text/__init__.py",
            "scripts/text/frozen_text_windows.py",
            "scripts/text/model_artifact_identity.py",
            "scripts/training/train_experiments.py",
            "scripts/training/train_experiments_fsdp.py",
            "scripts/training/train_encoder_network.py",
            "scripts/training/model_preverification.py",
        }
    if scope != "paper":
        raise ValueError(f"Unknown release scope: {scope}")
    if name == "LICENSE" or relative in {path for path, _ in ARCHIVED_ANALYSIS_SCRIPTS}:
        return True
    if name in {
        "journal/main.tex", "journal/references.bib", "journal/Makefile",
        "journal/pytest.ini", "journal/RELEASE_WORKFLOW.md",
        "journal/README.md", "journal/OVERLEAF_README.md",
        "journal/figures/README.md",
                "journal/source_data/README.md", "journal/source_data/provenance_manifest.tsv",
    }:
        return True
    if name == "journal/configs/supplement_consolidation/contact_sheet.pdf":
        return False
    if name.startswith("journal/figures/"):
        if name.startswith(("journal/figures/provenance/structure_restoration_20260908/",
                            "journal/figures/provenance/credit_clarity_20260908/")):
            return relative.suffix.lower() in {".json", ".csv"}
        if name == "journal/figures/provenance/publication_render_environment.json":
            return True
        canonical = re.fullmatch(r"journal/figures/main/figure_(\d+)\.pdf", name)
        if canonical:
            from build_submission_bundle import MAIN_FIGURES
            if name.removeprefix("journal/figures/") not in MAIN_FIGURES:
                return False
        return relative.suffix.lower() in {".pdf", ".png", ".svg", ".jpg", ".jpeg", ".eps"}
    if name.startswith("journal/supplementary/"):
        return name == "journal/supplementary/supplementary.tex" or (name.startswith("journal/supplementary/curated/") and relative.suffix == ".tex")
    if name.startswith("journal/source_data/release_task_identity/"):
        return relative.suffix in {".json", ".csv", ".md"}
    if name.startswith("journal/scripts/inherited_neurips/"):
        return relative.suffix == ".py"
    if name in {"journal/scripts/" + item for item in JOURNAL_SCRIPTS}:
        return True
    if name in {"journal/analysis/" + item for item in JOURNAL_ANALYSIS_RECORDS}:
        return True
    return any(name.startswith("journal/" + item + "/") for item in JOURNAL_DIRECTORIES)


def verify_reachable_commit(repository_root: Path, commit: str) -> dict[str, object]:
    """Require a real commit reachable from a named repository reference."""
    resolved = str(run_git("rev-parse", "--verify", f"{commit}^{{commit}}", repository_root=repository_root)).strip()
    if resolved != commit:
        raise RuntimeError("Release provenance must use a full commit identifier")
    refs = str(run_git("for-each-ref", "--contains", commit, "--format=%(refname)",
                       "refs/heads", "refs/tags", "refs/remotes", repository_root=repository_root)).splitlines()
    if not refs:
        raise RuntimeError(f"Release commit {commit} is not reachable from a named branch, tag or remote reference")
    return {
        "commit": resolved,
        "tree": str(run_git("rev-parse", f"{commit}^{{tree}}", repository_root=repository_root)).strip(),
        "reachable_from_refs": sorted(refs),
        "object_verified": True,
    }


def required_article_input_paths(source_root: Path) -> set[str]:
    """Name all live allowlisted inputs, including Git-ignored YAML recipes."""
    selected = set()
    for directory in JOURNAL_DIRECTORIES:
        for path in (source_root / directory).rglob("*"):
            if path.is_file() and not path.is_symlink() and not excluded(path.relative_to(source_root / directory)):
                relative = Path("journal") / path.relative_to(source_root)
                if repository_file_allowed(relative, "paper"):
                    selected.add(relative.as_posix())
    selected.update("journal/scripts/" + name for name in JOURNAL_SCRIPTS)
    selected.update("journal/analysis/" + name for name in JOURNAL_ANALYSIS_RECORDS)
    selected.update(path.as_posix() for path, _ in ARCHIVED_ANALYSIS_SCRIPTS)
    selected.update("journal/source_data/release_task_identity/" + name for name in
                    ("README.md", "task_identity.json", "task_identity.csv", "clean_noise_input_dimension_check.csv"))
    for path in (source_root / "source_data/release_task_identity").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".csv", ".md"}:
            selected.add("journal/" + path.relative_to(source_root).as_posix())
    return selected


def assert_registered_sources_allowlisted(source_root: Path) -> None:
    """Refuse a release that drops registered generators or scientific protocols."""
    manifest = source_root / "source_data/provenance_manifest.tsv"
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    omitted = set()
    prefix = Path("drafts/dendritic-local-learning")
    for row in rows:
        if row.get("status") != "ready":
            continue
        selected = [row.get("generator_path", "").strip()]
        source = row.get("source_path", "").strip()
        if "/journal/analysis/" in source:
            selected.append(source)
        for raw in filter(None, selected):
            try:
                relative = Path(raw).relative_to(prefix)
            except ValueError:
                omitted.add(raw)
                continue
            if not repository_file_allowed(relative, "paper"):
                omitted.add(raw)
    if omitted:
        raise RuntimeError("Canonical provenance references omitted software inputs:\n- "
                           + "\n- ".join(sorted(omitted)))


def assert_article_inputs_committed(source_root: Path, paper_root: Path, commit: str) -> None:
    """Refuse a clean-looking checkout that silently omits ignored inputs."""
    tracked = set(str(run_git("ls-tree", "-r", "--name-only", commit, repository_root=paper_root)).splitlines())
    missing = sorted(required_article_input_paths(source_root) - tracked)
    if missing:
        raise RuntimeError("Allowlisted article inputs are absent from the recorded paper commit. "
                           "Review and commit them (Git-ignored recipes require git add -f):\n- "
                           + "\n- ".join(missing))


def prune_release_entrypoints(root: Path, release_prefix: str = "dendritic_modeling") -> list[dict[str, str | int]]:
    """Remove command registrations for drivers outside the allowlist."""
    path = root / "pyproject.toml"
    original = path.read_bytes()
    lines = original.decode().splitlines(keepends=True)
    in_scripts = False
    kept, removed = [], []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_scripts = stripped == "[project.scripts]"
        match = re.match(r'[^=]+\s*=\s*"([A-Za-z0-9_.]+):[^\"]+"', stripped) if in_scripts else None
        if match:
            module = root / "src" / Path(*match.group(1).split("."))
            if not module.with_suffix(".py").is_file() and not (module / "__init__.py").is_file():
                removed.append(stripped.split("=", 1)[0].strip())
                continue
        kept.append(line)
    if not removed:
        return []
    path.write_text("".join(kept))
    return [{"path": release_prefix + "/pyproject.toml",
             "origin_sha256": hashlib.sha256(original).hexdigest(), "release_sha256": sha256(path),
             "replacement_count": len(removed),
             "reason": "remove entrypoints for excluded unrelated drivers: " + ", ".join(removed)}]


def extract_git_head(destination: Path, commit: str, *, repository_root: Path, scope: str | None = None) -> None:
    """Safely extract the release-eligible files from a committed Git tree."""

    paths = str(run_git("ls-tree", "-r", "--name-only", commit, repository_root=repository_root)).splitlines()
    paths = [name for name in paths if repository_file_allowed(Path(name), scope)]
    if not paths:
        raise RuntimeError(f"No release-eligible files in {scope or 'repository'} snapshot")
    # Pass only allowlisted members to git archive: excluded LFS presentation
    # objects are never downloaded/materialized merely to discard them later.
    payload = run_git("archive", "--format=tar", commit, "--", *paths, repository_root=repository_root, text=False)
    assert isinstance(payload, bytes)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe Git archive member: {member.name}")
            release_relative = Path(*relative.parts)
            if excluded(release_relative):
                continue
            target = destination / release_relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise RuntimeError(
                    f"Unsupported non-regular Git member: {member.name}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot extract Git member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            target.chmod(0o755 if member.mode & stat.S_IXUSR else 0o644)


def export_repository_snapshots(
    destination: Path,
    implementation_root: Path,
    implementation_commit: str,
    paper_commit: str,
    *,
    paper_root: Path | None = None,
) -> dict[str, int]:
    """Export both committed trees and assert the essential package boundary."""

    paper_root = PAPER_REPOSITORY_ROOT if paper_root is None else paper_root
    implementation = destination / "dendritic_modeling"
    paper = destination / "journal_package"
    extract_git_head(implementation, implementation_commit, repository_root=implementation_root, scope="implementation")
    extract_git_head(paper, paper_commit, repository_root=paper_root, scope="paper")
    required = (
        implementation / "pyproject.toml",
        implementation / "LICENSE",
        implementation / "src/dendritic_modeling/__init__.py",
        implementation / "src/dendritic_modeling/networks/architectures/excitation_inhibition/dendritic/branch_dynamics.py",
        paper / "journal/main.tex",
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Missing essential software release input: {path.relative_to(destination)}")
    return {"implementation": len(list_files(implementation)), "paper": len(list_files(paper))}


def export_physical_runtime(root: Path, implementation_root: Path) -> dict[str, object]:
    """Export the exact historical source used by the depth budget extension."""
    return export_historical_runtime(root, implementation_root, PHYSICAL_RUNTIME_COMMIT,
                                     PHYSICAL_RUNTIME_DIRECTORY, "Physical-depth budget extension runtime")


def export_historical_runtime(root: Path, implementation_root: Path, commit: str,
                              directory: str, role: str) -> dict[str, object]:
    """Export one reachable cohort-specific runtime through the package allowlist."""
    provenance = verify_reachable_commit(implementation_root, commit)
    destination = root / directory
    extract_git_head(destination, commit,
                     repository_root=implementation_root, scope="implementation")
    original_files = list_files(destination)
    with (destination / "RUNTIME_ORIGINS.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("path", "original_sha256", "commit"))
        writer.writerows((p.relative_to(destination).as_posix(), sha256(p), commit)
                         for p in original_files)
    metadata = dict(provenance, role=role,
                    source_files=len(original_files), export_directory=directory,
                    git_metadata_included=False,
                    identity_check="Original source digests plus declared released-byte transformations; no fabricated Git checkout identity")
    (destination / "RUNTIME_PROVENANCE.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def copy_tree_allowlisted(source: Path, destination: Path) -> int:
    """Copy regular files while applying the release exclusion policy."""

    copied = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(source)
        if excluded(relative):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        target.chmod(0o755 if os.access(path, os.X_OK) else 0o644)
        copied += 1
    return copied


def copy_journal_material(destination: Path, *, source_root: Path | None = None) -> dict[str, int]:
    source_root = JOURNAL_ROOT if source_root is None else source_root
    counts: dict[str, int] = {}
    for name in JOURNAL_DIRECTORIES:
        source = source_root / name
        if not source.is_dir():
            raise RuntimeError(f"Missing journal release input: {source}")
        counts[name] = copy_tree_allowlisted(source, destination / name)

    scripts_destination = destination / "scripts"
    scripts_destination.mkdir(parents=True, exist_ok=True)
    copied_scripts = 0
    for name in JOURNAL_SCRIPTS:
        source = source_root / "scripts" / name
        if not source.is_file():
            raise RuntimeError(f"Missing journal script: {source}")
        target = scripts_destination / name
        shutil.copyfile(source, target)
        target.chmod(0o755 if os.access(source, os.X_OK) else 0o644)
        copied_scripts += 1
    counts["scripts"] = copied_scripts
    inherited_source = source_root / "scripts" / "inherited_neurips"
    if not inherited_source.is_dir():
        raise RuntimeError(f"Missing inherited figure generators: {inherited_source}")
    counts["inherited_figure_generators"] = copy_tree_allowlisted(
        inherited_source, scripts_destination / "inherited_neurips"
    )

    # Source Data are distributed as a separate archive.  These two compact
    # files retain the interpretation and panel-level provenance contract.
    source_data_destination = destination / "source_data_metadata"
    source_data_destination.mkdir(parents=True, exist_ok=True)
    for relative in (Path("source_data/README.md"), Path("source_data/provenance_manifest.tsv")):
        source = source_root / relative
        if not source.is_file():
            raise RuntimeError(f"Missing source-data metadata: {source}")
        target = source_data_destination / source.name
        shutil.copyfile(source, target)
        target.chmod(0o644)
    identity = source_root / "source_data/release_task_identity"
    counts["source_data_metadata"] = 2
    if not identity.is_dir():
        raise RuntimeError("Missing cohort-to-generator identity metadata in the committed paper")
    counts["task_identity"] = copy_tree_allowlisted(identity, source_data_destination / "release_task_identity")

    analysis_destination = destination / "analysis_records"
    analysis_destination.mkdir(parents=True, exist_ok=True)
    for name in JOURNAL_ANALYSIS_RECORDS:
        source = source_root / "analysis" / name
        if not source.is_file():
            raise RuntimeError(f"Missing journal analysis record: {source}")
        target = analysis_destination / name
        shutil.copyfile(source, target)
        target.chmod(0o644)
    counts["analysis_records"] = len(JOURNAL_ANALYSIS_RECORDS)
    return counts


def git_tracked(relative: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(PAPER_REPOSITORY_ROOT), "ls-files", "--error-unmatch", str(relative)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def copy_archived_analysis_scripts(destination: Path, *, paper_root: Path | None = None) -> dict[str, object]:
    """Copy explicitly selected historical diagnostics from the paper repository."""

    paper_root = PAPER_REPOSITORY_ROOT if paper_root is None else paper_root
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    requested = list(ARCHIVED_ANALYSIS_SCRIPTS)
    for relative, role in OPTIONAL_JOURNAL_ARCHIVED_SCRIPTS:
        source = JOURNAL_ROOT / relative
        if source.is_file():
            requested.append((source.relative_to(PAPER_REPOSITORY_ROOT), role))

    for relative, role in requested:
        source = paper_root / relative
        if not source.is_file():
            raise RuntimeError(f"Missing explicitly archived analysis script: {source}")
        target = destination / source.name
        if target.exists():
            raise RuntimeError(f"Duplicate archived script name: {target.name}")
        shutil.copyfile(source, target)
        target.chmod(0o755 if os.access(source, os.X_OK) else 0o644)
        source_digest = sha256(source)
        copy_digest = sha256(target)
        if source_digest != copy_digest:
            raise RuntimeError(f"Archived script copy differs: {relative}")
        records.append(
            {
                "filename": target.name,
                "origin": relative.as_posix(),
                "origin_sha256": source_digest,
                "copy_sha256": copy_digest,
                "bytes": target.stat().st_size,
                "git_tracked_at_release_head": True,
                "copy_status": "byte-identical",
                "role": role,
            }
        )

    origins = destination / "ORIGINS.tsv"
    with origins.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "filename",
                "origin",
                "origin_sha256",
                "copy_sha256",
                "bytes",
                "git_tracked_at_release_head",
                "copy_status",
                "role",
            ),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    origins.chmod(0o644)

    script_lines = "\n".join(
        f"- `{record['filename']}`: {record['role']}; SHA-256 "
        f"`{record['copy_sha256']}`."
        for record in records
    )
    (destination / "README.md").write_text(
        "# Archived analysis scripts\n\n"
        "These diagnostic scripts were copied byte-identically from their "
        "explicit paper-repository origins before any declared release-only "
        "portability substitutions. Their repository-relative origins, source and copy hashes, "
        "sizes, and Git-tracking state are recorded in `ORIGINS.tsv`.\n\n"
        f"{script_lines}\n\n"
        "The two Figure 2 diagnostics operate on trained checkpoint directories, "
        "which are not redistributed. They can be inspected directly and run "
        "against reviewer-generated checkpoints after installing the included "
        "`dendritic_modeling` package. Their original relative repository-root "
        "discovery remains unchanged; an installed package on `PYTHONPATH` is "
        "sufficient when they are run from this archive.\n",
        encoding="utf-8",
    )
    (destination / "README.md").chmod(0o644)
    return {
        "script_count": len(records),
        "records": records,
        "optional_reconstructed_tree_script_included": any(
            record["filename"] == "run_reconstructed_tree_task_learning.py"
            for record in records
        ),
    }


def sanitize_git_snapshot(root: Path) -> list[dict[str, str | int]]:
    """Replace machine-local paths in the release copy and record each edit."""

    changes: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        payload = path.read_bytes()
        if b"\x00" in payload:
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        original = text
        file_reasons: list[str] = []
        replacement_count = 0
        for old, new, reason in PORTABILITY_REPLACEMENTS:
            count = text.count(old)
            if count:
                text = text.replace(old, new)
                replacement_count += count
                file_reasons.append(f"{reason} ({count})")
        if text != original:
            path.write_text(text, encoding="utf-8")
            changes.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "origin_sha256": hashlib.sha256(payload).hexdigest(),
                    "release_sha256": sha256(path),
                    "replacement_count": replacement_count,
                    "reason": "; ".join(file_reasons),
                }
            )
    return changes


def relocate_image_invariant_test(root: Path) -> list[dict[str, str | int]]:
    """Read restored MNIST configs without changing the frozen scientific test."""
    original = "cfg=yaml.safe_load(Path(record['config']).read_text())"
    replacement = (
        "cfg=yaml.safe_load((run.OUT/'configs'/'development'/"
        "f\"condition_{record['index']:03d}.yaml\").read_text())"
    )
    changes = []
    for prefix in ("article_analysis", "journal_package/journal"):
        relative = f"{prefix}/scripts/image_ladder_controls/test_delivery.py"
        path = root / relative
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        if text.count(original) != 1:
            raise RuntimeError(f"Unexpected frozen MNIST test layout: {relative}")
        path.write_text(text.replace(original, replacement), encoding="utf-8")
        changes.append({
            "path": relative,
            "origin_sha256": hashlib.sha256(payload).hexdigest(),
            "release_sha256": sha256(path),
            "replacement_count": 1,
            "reason": "resolve frozen MNIST development configs from restored study root and unchanged condition index",
        })
    return changes


def write_portability_manifest(
    path: Path, changes: Iterable[dict[str, str | int]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "origin_sha256", "release_sha256", "replacement_count", "reason"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(changes)


def capture_release_origins(root: Path, implementation_commit: str, paper_commit: str) -> dict[str, dict[str, str]]:
    """Freeze original bytes before any declared release-only transformation."""
    records = {}
    historical = {path.name: path.as_posix() for path, _ in ARCHIVED_ANALYSIS_SCRIPTS}
    for path in list_files(root):
        relative = path.relative_to(root).as_posix()
        repository, commit, source = "generated", "", relative
        if relative.startswith("dendritic_modeling/"):
            repository, commit, source = "implementation", implementation_commit, relative.split("/", 1)[1]
        elif relative.startswith(PHYSICAL_RUNTIME_DIRECTORY + "/"):
            repository, commit = "implementation", PHYSICAL_RUNTIME_COMMIT
            source = relative[len(PHYSICAL_RUNTIME_DIRECTORY) + 1:]
            if source in {"RUNTIME_ORIGINS.tsv", "RUNTIME_PROVENANCE.json"}:
                repository, commit = "generated", ""
        elif relative.startswith(IMAGE_RUNTIME_DIRECTORY + "/"):
            repository, commit = "implementation", IMAGE_RUNTIME_COMMIT
            source = relative[len(IMAGE_RUNTIME_DIRECTORY) + 1:]
            if source in {"RUNTIME_ORIGINS.tsv", "RUNTIME_PROVENANCE.json"}:
                repository, commit = "generated", ""
        elif relative.startswith("journal_package/"):
            repository, commit, source = "paper", paper_commit, relative.split("/", 1)[1]
        elif relative.startswith("article_analysis/"):
            local = relative.split("/", 1)[1]
            repository, commit = "paper", paper_commit
            if local.startswith("archived_analysis_scripts/") and path.name in historical:
                source = historical[path.name]
            elif local.startswith("source_data_metadata/"):
                source = "journal/source_data/" + local.split("/", 1)[1]
            elif local.startswith("analysis_records/"):
                source = "journal/analysis/" + local.split("/", 1)[1]
            elif local.startswith("archived_analysis_scripts/"):
                repository, commit, source = "generated", "", relative
            else:
                source = "journal/" + local
        records[relative] = {"origin_sha256": sha256(path), "origin_repository": repository,
                             "origin_commit": commit, "origin_path": source}
    return records


def write_released_source_hashes(root: Path, origins: dict[str, dict[str, str]]) -> int:
    """Link unchanged canonical identities to verified portable software bytes."""
    provenance = root / "PORTABILITY_PATCHES.tsv"
    with provenance.open(newline="") as handle:
        changes = list(csv.DictReader(handle, delimiter="\t"))
    provenance_digest = sha256(provenance)
    rows = []
    for relative, origin in sorted(origins.items()):
        path = root / relative
        actual = sha256(path)
        chain = [change for change in changes if change["path"] == relative]
        current = origin["origin_sha256"]
        for change in chain:
            if change["origin_sha256"] != current:
                raise RuntimeError(f"Broken release transformation chain: {relative}")
            current = change["release_sha256"]
        if current != actual:
            raise RuntimeError(f"Undeclared change to release source: {relative}")
        rows.append(dict(path=relative, kind="software", **origin, release_sha256=actual,
                         transformation="; ".join(change["reason"] for change in chain) or "byte-identical",
                         provenance_file=provenance.name, provenance_sha256=provenance_digest))
    fields = ("path", "kind", "origin_sha256", "release_sha256", "transformation",
              "provenance_file", "provenance_sha256", "origin_repository", "origin_commit", "origin_path")
    with (root / "RELEASED_SOURCE_HASHES.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return len(rows)


def release_readme(commit: str, journal_commit: str) -> str:
    return f"""# Dendritic credit-assignment software

This reviewer archive accompanies *Dendritic morphology as a dictionary for local
credit assignment*. It contains the release-eligible committed Dendritic
Modeling implementation at Git commit `{commit}`, a separate committed paper
snapshot at `{journal_commit}`, and allowlisted article-specific analysis,
configuration, validation, and provenance code from that paper snapshot.

## Layout

- `dendritic_modeling/`: an export of the production repository Git HEAD, including
  the installable package, training and local-learning implementations,
  and the small set of training/sweep drivers used by this article. Generic
  modules imported by the public package API remain; unrelated transformer,
  text and vision experiment drivers/configs/tests and presentation files are
  excluded. Uncommitted files are excluded.
- `journal_package/`: a separate export of the clean paper repository Git HEAD.
  `journal_package/journal/main.tex` is the article source, and
  `journal_package/journal/figures/` contains the current figures and authenticated
  vector inputs needed to rebuild them. Only the current supplementary TeX
  modules are included; retired drafts and review contact sheets are excluded.
  This directory is a paper package, not the installable Python implementation.
- `article_analysis/code/`: standalone regular-tree checks, reconstructed-tree
  analyses, and the portable CAVE/DANDI measured-response pipeline.
- `article_analysis/configs/`: frozen and portable experiment specifications.
  The five CIFAR-10 launch records with site-specific paths are intentionally
  omitted; their frozen summaries and hashes ship in the source-data package.
- `article_analysis/scripts/`: figure, source-data, cohort, perturbation,
  rerun-validation, and controlled-learning scripts.
- `article_analysis/archived_analysis_scripts/`: explicitly selected historical
  Figure 2 diagnostics, with paper-repository origins, Git-tracking state and
  pre-portability-copy SHA-256 records. Any subsequent substitutions are listed
  in the release-root portability manifest.
- `article_analysis/tests/`: article-level validation tests.
- `article_analysis/reproducibility/`: source hashes, cohort manifests,
  archived hardware accounting, and archive boundaries.
- `article_analysis/analysis_records/`: the frozen experiment contract and
  experiment protocols. Internal review/revision logs are excluded.
- `historical_runtimes/physical_depth_a99c3a7/`: selected source files from
  the exact reachable implementation commit used for the physical-depth
  extension, with per-file original hashes and commit/ref provenance.
  Its Git metadata are not fabricated or included. The portable launcher
  verifies the exported bytes directly.
- `historical_runtimes/image_ladder_6c1aaa2/`: the exact historical source
  for the new MNIST dictionary, learning-rate and decoder controls. Original
  source identities are checked separately from any declared portability edits.
  Scientific fits recorded NumPy 2.2.6 and PyTorch 2.9.1; separate CPU checks
  with NumPy 1.26.4 establish portability, not identical numerical outcomes.
- `PORTABILITY_PATCHES.tsv`: machine-local defaults changed in the release
  copies, with paths relative to this release root. Scientific parameters are
  not modified. Frozen source hashes describe the original source bytes; this
  manifest documents the release-only portability transformations.
- `RELEASED_SOURCE_HASHES.tsv`: unchanged original source hashes linked to released
  bytes, their repository/commit origins and the declared transformation chain.
  Canonical protocol manifests retain original hashes. The release helper
  `article_analysis/code/release_noise/release_hashes.py` verifies both identities.
- `SHA256SUMS.tsv`: SHA-256 digest and size of every other released file.

The three new study directories retain their protocol-authenticated `worker.sh`
execution records. Reviewer replay uses each study's portable Python launcher;
the worker records document the original cluster execution and its environment.

Numerical panel data are distributed separately in `Source_Data.zip`; its
`manifest.tsv` maps each released display file to its original article-relative
source path. Rebuilding figures requires those inputs at the recorded
`source_data/` paths, together with the retained vector components in
`journal_package/journal/figures/`. That directory includes the historical
full-panel sources used by the final renderer. The extra script copies in
`article_analysis/scripts/` do not by themselves provide a complete figure
working directory. To preserve the original repository-relative source discovery,
restore the paper snapshot under the implementation repository before running
source-dependent checks or figure builders:

```bash
mkdir -p dendritic_modeling/drafts/dendritic-local-learning
cp -R journal_package/. dendritic_modeling/drafts/dendritic-local-learning/
python article_analysis/code/release_noise/release_hashes.py \\
  --remap-paper --release-root . \\
  --journal-root dendritic_modeling/drafts/dendritic-local-learning/journal
```

The remapping step verifies every copied paper file against the software archive
and preserves the original archive paths and transformation chain in explicit
local sidecars. Subsequent Source Data restoration retains these software links.

Extract `Source_Data.zip` so its `manifest.tsv` is inside a `Source_Data/`
directory. Restore by the manifest's `original_source` field; display homes such
as `Methods/retained_evidence/` are not analysis paths:

```bash
python article_analysis/code/release_noise/restore_source_data.py \\
  --source-data-root Source_Data \\
  --journal-root dendritic_modeling/drafts/dendritic-local-learning/journal \\
  --dry-run
python article_analysis/code/release_noise/restore_source_data.py \\
  --source-data-root Source_Data \\
  --journal-root dendritic_modeling/drafts/dendritic-local-learning/journal
```

The helper checks released SHA-256 digests, chooses complete copies over
explicit display-specific subsets, and refuses conflicting copies or differing
existing destinations before writing. Portable released bytes need not equal
the original research-file hash. If only filtered copies exist, the dry run
identifies the missing complete source instead of silently restoring a truncated
table. Restoration writes `RELEASED_SOURCE_HASHES.tsv` and an unchanged copy of
the package manifest, `RELEASED_SOURCE_MANIFEST.tsv`, beside the restored journal.
Audits retain canonical original hashes and verify the separate released-byte
link through `release_hashes.py`. Removed private run-directory columns are not
reconstructed; checkpoint-level reanalysis requires separately supplied or
reviewer-generated run records. Use the current manifest with `original_source` and `transformation`
columns; old archives do not establish current package identity.

The depth extension requires the historical runtime; the current installed core
is not a substitute for its frozen source identity. After restoring Source Data,
verify one condition with the portable launcher (run from the release root):

```bash
python -B article_analysis/code/release_noise/physical_depth_launcher.py \\
  --source-root dendritic_modeling/drafts/dendritic-local-learning/journal/source_data/physical_depth_budget/canonical \\
  --journal-root dendritic_modeling/drafts/dendritic-local-learning/journal \\
  --runtime-root historical_runtimes/physical_depth_a99c3a7 \\
  --condition 30 --verify-only
```

For a full rerun, replace `--verify-only` with `--output-root NEW_OUTPUT_DIR`.
The original 600-epoch cap, validation-selection rule, patience, seed and rate
remain frozen. Conditions 0–59 are listed in `extension_protocol.json`.
An optional `--smoke-epochs 1` changes the budget explicitly and labels the output
as an excluded smoke check. The original `extend.py` is preserved; the launcher
uses its verified passive observation helpers and records all source identities
and library versions. It imports the verified historical runtime in a fresh
process rather than invoking its machine-specific Git checkout check. This
establishes source identity, not bitwise equivalence across devices and library
versions. The synthetic hierarchical task requires no external dataset.

The new conductance-credit studies are standalone directed E/I reference
models under `article_analysis/scripts/conductance_credit_demand/`. Their
original 16-conductance construction and the separately frozen 24-conductance
opponent-tuning construction retain all development, fresh, continuation and
parameter-bound outcomes. The Source Data protocol and methods distinguish
these families, and oracle projection coefficients are explicit in both code
and manuscript. They do not require a current production-package import.
After restoring the journal and Source Data, run from the restored journal directory:

```bash
python scripts/conductance_credit_demand/portable_run.py --study-root source_data/conductance_credit_demand --family opponent --phase fresh --seed 2101 --verify-only
```

Replace `--verify-only` with `--output-root NEW_DIRECTORY` for a full replay.
Its `PORTABLE_README.md` documents the study-root argument, excluded smoke
mode, continuation semantics and the distinction between source verification
and cross-environment numerical identity.

For the MNIST studies, the restored `scripts/image_ladder_controls/README.md`
provides the complete sequence: verify the upstream data, generate original-to-
released runtime/adapter links with `prepare_release_links.py`, then replay a
frozen condition with `portable_run.py`. The optional forty-state checkpoint
archive enables `portable_capture.py` without repeating model training. The
new image and conductance replays must write to fresh output directories.

The structure and finite-horizon investigations retain their original analysis
paths in their frozen runners. Their released numerical copies are under
`source_data/morphology_structure/` and
`source_data/morphology_finite_horizon/`. After restoring Source Data to the
article-relative paths recorded in `Source_Data/manifest.tsv`, use each
investigation's `export_manifest.json` to restore these additional analysis
paths. From the extracted software-release root, after copying the paper into
the implementation tree as above:

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import json
import shutil

journal = Path("dendritic_modeling/drafts/dendritic-local-learning/journal").resolve()
import sys
sys.path.insert(0, str(Path("article_analysis/code/release_noise").resolve()))
from restore_source_data import restoration_plan
plan, issues = restoration_plan(Path("Source_Data").resolve(), journal)
if issues:
    raise RuntimeError("Resolve incomplete Source Data before restoring investigation paths")
released_records = dict((row["original_source"], row) for source, target, row in plan)
restored = 0
for folder in ("morphology_structure", "morphology_finite_horizon"):
    manifest_path = journal / "source_data" / folder / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["files"]:
        released = (journal / record["destination"]).resolve()
        original = (journal / record["source"]).resolve()
        if not released.is_relative_to(journal) or not original.is_relative_to(journal):
            raise RuntimeError("Manifest path leaves the restored journal tree")
        if not record["source"].startswith("analysis/morphology_investigation_"):
            raise RuntimeError("Unexpected legacy investigation destination")
        provenance = released_records[record["destination"]]
        if record["sha256"] != provenance["original_sha256"]:
            raise RuntimeError("Investigation and package origin hashes disagree")
        expected = provenance["sha256"]
        if hashlib.sha256(released.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Released source digest differs: " + str(released))
        if original.exists():
            if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Refusing to overwrite a differing file: " + str(original))
        else:
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(released, original)
        if hashlib.sha256(original.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Restored source digest differs: " + str(original))
        restored += 1
print("Verified/restored", restored, "investigation files")
PY
```

This copies the verified released bytes without modifying them or the frozen
runner paths. The original export hashes are checked against `original_sha256`;
restored portable copies are checked against the current released `sha256`. It is required before running the analysis commands in the
finite-horizon investigation README. The structure mapping also restores its
separately exported design-certificate files. Completed training outcomes remain
preserved; use a separate output tree for new training runs.

Run `rebuild_final_publication_figures.py` from that restored journal layout;
it only redraws frozen results. Raw
MICRONS/CAVE and DANDI/NWB assets are not redistributed. Public identifiers,
asset paths, access requirements, and derived-data provenance are documented
under `article_analysis/reproducibility/` and in the Source Data archive.

## Environment

Keep an untouched copy of the ZIP and its checksummed contents. Run the following
installation, restoration and replay commands from a separate extracted working
copy: historical runtime imports can create log files beside their sources,
even when Python bytecode is disabled.

Python 3.10 or 3.11 is recommended. From the extracted release root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
REVIEWER_BUILD=$(mktemp -d)
cp -R dendritic_modeling/. "$REVIEWER_BUILD/"
python -m pip install -c article_analysis/code/release_noise/constraints.txt "$REVIEWER_BUILD[test]"
```

Figure reconstruction and PDF assembly also require PyMuPDF, which provides the
`fitz` module and is not a dependency of the core modeling package:

```bash
python -m pip install -c article_analysis/code/release_noise/constraints.txt PyMuPDF
```

The CPU replay constraints pin the validated PyMuPDF 1.28.2. The exact package
versions and font hashes used to render the retained publication figures are
recorded separately in
`journal_package/journal/figures/provenance/publication_render_environment.json`;
that rendering used PyMuPDF 1.28.0. These are distinct environment records.
`article_analysis/code/release_noise/ENVIRONMENTS.md` maps the scientific
cohorts to their original source/runtime records and supported replay paths.
It also documents the fully resolved CPU environment and required external
fonts. The NumPy 1.26.4 replay environment is distinct from the NumPy 2.2.6
scientific executions; compatibility checks do not claim identical trajectories.

The core package dependencies are declared in
`dendritic_modeling/pyproject.toml`. CAVE and NWB retrieval additionally
require service-specific packages and, for restricted queries, user-provided
credentials. No credentials are included here.

## Fast validation

```bash
python article_analysis/code/regular_tree/test_additive_reference.py
python article_analysis/code/reconstructed_tree/verify_credit_capture_bound.py \\
  --help
pytest -q article_analysis/tests/test_alignment_controlled_learning.py
```

The other article tests document full-package checks that require archived run
directories or the complete working-paper source-data tree and are therefore
not standalone in this software-only archive. Only article tests are distributed; full production integration tests and
unrelated project tests are outside this release scope.

## Reproducing analyses

Read the README in each analysis directory before execution. The regular-tree
training sweeps use the installed `dendritic_modeling` package and the frozen
YAML files under `article_analysis/configs/`. For example, a portable sweep can
be materialized without scheduler submission with:

```bash
python dendritic_modeling/src/dendritic_modeling/scripts/sweeps/sweep_manager.py \\
  --config article_analysis/configs/reruns/feedback_definition_shunting_15seed.yaml \\
  --generate-only
```

Cluster account, partition, data-root, and output-root fields must be replaced
for the reviewer's environment. Historical auxiliary configurations that
referenced undistributed local artifacts contain explicit placeholders such
as `${{DENDRITIC_EXTERNAL_ARTIFACTS}}`; they are listed in
`PORTABILITY_PATCHES.tsv` and are not needed to execute the article's primary
portable analyses.

## Scope

This archive contains source code and lightweight configuration/provenance
records, not trained checkpoints, raw datasets, scheduler logs, or data
caches. Both Git snapshots are taken from their recorded commits, so unrelated
dirty implementation working-tree changes cannot enter the release. The paper
repository must be clean at build time; article-specific copies are tied to
that paper commit and individually checksummed. The implementation working-tree
status is checked and its change count recorded without including those changes.

Historical noise tasks are explicitly separated in
`article_analysis/source_data_metadata/release_task_identity/task_identity.json`.
The frozen generator bodies and their source hashes are in
`article_analysis/code/release_noise/`. Add that directory to `PYTHONPATH` to use
the installed core's optional dataset hook. Choose `legacy_noisy_lines` or
`projected_noise_mnist` explicitly; the ambiguous historical key is rejected.
The former is recovered from the clean source commit; the latter is a newly
frozen reference for the intended protocol, not proof of older executed bytes.
MNIST images require the upstream torchvision download/cache; the download-free
smoke substitutes tiny image tensors only to verify the noise transformation.

From the extracted archive root, run:

```bash
python -I -B article_analysis/code/release_noise/cleanroom_smoke.py --release-root .
```

The software is released under the MIT license included at
`dendritic_modeling/LICENSE`.
"""


def scan_release(root: Path) -> list[str]:
    """Return high-confidence private-path, credential, and policy findings."""

    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if excluded(relative):
            findings.append(f"excluded artifact present: {relative.as_posix()}")
            continue
        payload = path.read_bytes()
        # These two packaging utilities contain the literal private-path
        # regular expressions used to detect and sanitize release content.
        # Their own detector patterns are not filesystem references.
        if relative.name not in {
            "build_nature_source_data.py",
            "build_software_release.py",
        }:
            for pattern in PRIVATE_PATH_PATTERNS:
                if pattern.search(payload):
                    findings.append(f"private absolute path: {relative.as_posix()}")
                    break
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(payload):
                findings.append(f"possible {label}: {relative.as_posix()}")
    return findings


def list_files(root: Path, *, omit: set[str] | None = None) -> list[Path]:
    omitted = omit or set()
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(root).as_posix() not in omitted
    ]


def write_checksums(root: Path) -> int:
    manifest = root / "SHA256SUMS.tsv"
    files = list_files(root, omit={manifest.name})
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("sha256", "bytes", "path"))
        for path in files:
            writer.writerow(
                (
                    sha256(path),
                    path.stat().st_size,
                    path.relative_to(root).as_posix(),
                )
            )
    manifest.chmod(0o644)
    return len(files)


def validate_checksums(root: Path) -> int:
    manifest = root / "SHA256SUMS.tsv"
    expected_paths: set[str] = set()
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["sha256", "bytes", "path"]:
            raise RuntimeError("Malformed SHA256SUMS.tsv header")
        for row in reader:
            relative = row["path"]
            if relative in expected_paths:
                raise RuntimeError(f"Duplicate checksum path: {relative}")
            expected_paths.add(relative)
            path = root / relative
            if not path.is_file():
                raise RuntimeError(f"Missing checksummed file: {relative}")
            if path.stat().st_size != int(row["bytes"]):
                raise RuntimeError(f"Size mismatch: {relative}")
            if sha256(path) != row["sha256"]:
                raise RuntimeError(f"Digest mismatch: {relative}")
    observed = {
        path.relative_to(root).as_posix()
        for path in list_files(root, omit={manifest.name})
    }
    if observed != expected_paths:
        missing = sorted(observed - expected_paths)
        extra = sorted(expected_paths - observed)
        raise RuntimeError(
            f"Checksum coverage mismatch; unlisted={missing}, absent={extra}"
        )
    return len(expected_paths)


def zip_datetime(epoch: int) -> tuple[int, int, int, int, int, int]:
    dt = datetime.fromtimestamp(max(epoch, 315532800), tz=timezone.utc)
    # ZIP stores timestamps at two-second resolution.
    return (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second // 2 * 2)


def write_deterministic_zip(root: Path, archive_path: Path, epoch: int) -> None:
    timestamp = zip_datetime(epoch)
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for path in list_files(root):
            relative = Path(STAGE_NAME) / path.relative_to(root)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=timestamp)
            mode = 0o755 if os.access(path, os.X_OK) else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as handle:
                archive.writestr(info, handle.read(), compresslevel=9)


def validate_zip(archive_path: Path, release_root: Path) -> int:
    disk_paths = {
        (Path(STAGE_NAME) / path.relative_to(release_root)).as_posix()
        for path in list_files(release_root)
    }
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("ZIP contains duplicate member names")
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe ZIP member: {name}")
        if set(names) != disk_paths:
            raise RuntimeError("ZIP membership differs from staged release")
    return len(disk_paths)


def build(force: bool, implementation_root: Path | None = None) -> dict[str, object]:
    implementation_root = discover_implementation_root(JOURNAL_ROOT, implementation_root)
    SUBMISSION_ROOT.mkdir(parents=True, exist_ok=True)
    stage = SUBMISSION_ROOT / STAGE_NAME
    archive = SUBMISSION_ROOT / ARCHIVE_NAME
    digest_file = SUBMISSION_ROOT / f"{ARCHIVE_NAME}.sha256"
    for target in (stage, archive, digest_file):
        ensure_submission_target(target)
        if target.exists() and not force:
            raise RuntimeError(
                f"Output already exists: {target}. Re-run with --force to replace "
                "only these generated release targets."
            )

    commit = str(run_git("rev-parse", "HEAD", repository_root=implementation_root)).strip()
    implementation_provenance = verify_reachable_commit(implementation_root, commit)
    commit_epoch = int(str(run_git("show", "-s", "--format=%ct", commit, repository_root=implementation_root)).strip())
    commit_utc = datetime.fromtimestamp(commit_epoch, tz=timezone.utc).isoformat()
    subject = str(run_git("show", "-s", "--format=%s", commit, repository_root=implementation_root)).strip()
    status_lines = [
        line
        for line in str(
            run_git("status", "--porcelain=v1", "--untracked-files=all", repository_root=implementation_root)
        ).splitlines()
        if line.strip()
    ]
    journal_commit = run_journal_git("rev-parse", "HEAD").strip()
    paper_provenance = verify_reachable_commit(PAPER_REPOSITORY_ROOT, journal_commit)
    journal_epoch = int(run_journal_git("show", "-s", "--format=%ct", journal_commit).strip())
    archive_epoch = max(commit_epoch, journal_epoch)
    journal_status_lines = [
        line
        for line in run_journal_git(
            "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line.strip()
    ]
    if journal_status_lines:
        raise RuntimeError(
            "The journal repository must be clean before building the software "
            "release so every article-specific file is tied to a commit."
        )
    assert_article_inputs_committed(JOURNAL_ROOT, PAPER_REPOSITORY_ROOT, journal_commit)
    assert_registered_sources_allowlisted(JOURNAL_ROOT)

    temporary_parent = SUBMISSION_ROOT / ".software_release_build"
    ensure_submission_target(temporary_parent)
    if temporary_parent.exists():
        if not force:
            raise RuntimeError(f"Stale temporary directory: {temporary_parent}")
        remove_generated_tree(temporary_parent)
    temporary_parent.mkdir(parents=True)

    try:
        temporary_stage = temporary_parent / STAGE_NAME
        repository_destination = temporary_stage / "dendritic_modeling"
        article_destination = temporary_stage / "article_analysis"
        repository_destination.mkdir(parents=True)
        article_destination.mkdir(parents=True)

        snapshot_counts = export_repository_snapshots(
            temporary_stage, implementation_root, commit, journal_commit
        )
        physical_runtime = export_physical_runtime(temporary_stage, implementation_root)
        image_runtime = export_historical_runtime(
            temporary_stage, implementation_root, IMAGE_RUNTIME_COMMIT,
            IMAGE_RUNTIME_DIRECTORY, "MNIST dictionary, learning-rate and decoder-control runtime")
        git_file_count = snapshot_counts["implementation"]
        portability_changes = []
        committed_paper = temporary_stage / "journal_package"
        journal_counts = copy_journal_material(article_destination, source_root=committed_paper / "journal")
        archived_scripts = copy_archived_analysis_scripts(
            article_destination / "archived_analysis_scripts", paper_root=committed_paper
        )
        release_origins = capture_release_origins(temporary_stage, commit, journal_commit)
        portability_changes.extend(prune_release_entrypoints(repository_destination))
        portability_changes.extend(prune_release_entrypoints(
            temporary_stage / PHYSICAL_RUNTIME_DIRECTORY, PHYSICAL_RUNTIME_DIRECTORY))
        portability_changes.extend(prune_release_entrypoints(
            temporary_stage / IMAGE_RUNTIME_DIRECTORY, IMAGE_RUNTIME_DIRECTORY))
        for snapshot_name in ("dendritic_modeling", "journal_package", "article_analysis", PHYSICAL_RUNTIME_DIRECTORY, IMAGE_RUNTIME_DIRECTORY):
            for change in sanitize_git_snapshot(temporary_stage / snapshot_name):
                change["path"] = f"{snapshot_name}/{change['path']}"
                portability_changes.append(change)
        portability_changes.extend(relocate_image_invariant_test(temporary_stage))

        (temporary_stage / "README.md").write_text(
            release_readme(commit, journal_commit), encoding="utf-8"
        )
        (temporary_stage / "README.md").chmod(0o644)
        shutil.copyfile(repository_destination / "LICENSE", temporary_stage / "LICENSE")
        (temporary_stage / "LICENSE").chmod(0o644)
        write_portability_manifest(
            temporary_stage / "PORTABILITY_PATCHES.tsv", portability_changes
        )

        released_source_hash_count = write_released_source_hashes(temporary_stage, release_origins)
        metadata = {
            "released_source_hash_count": released_source_hash_count,
            "physical_depth_historical_runtime": physical_runtime,
            "image_ladder_historical_runtime": image_runtime,
            "release": "Dendritic credit-assignment software",
            "release_schema": 3,
            "implementation_repository_commit": commit,
            "implementation_repository_export": "dendritic_modeling/",
            "paper_repository_export": "journal_package/",
            "paper_snapshot_file_count": snapshot_counts["paper"],
            "repository_commit": commit,
            "implementation_provenance": implementation_provenance,
            "paper_provenance": paper_provenance,
            "repository_commit_utc": commit_utc,
            "repository_commit_subject": subject,
            "repository_snapshot_method": "git archive recorded commit with explicit file allowlist",
            "journal_repository_commit": journal_commit,
            "journal_repository_commit_utc": datetime.fromtimestamp(journal_epoch, tz=timezone.utc).isoformat(),
            "journal_repository_snapshot_method": "git archive recorded commit with explicit file allowlist",
            "journal_repository_clean_at_build": True,
            "working_tree_dirty_at_build": bool(status_lines),
            "working_tree_change_count": len(status_lines),
            "working_tree_material_included": False,
            "journal_material_source": "allowlisted files copied only from the committed paper export",
            "git_snapshot_file_count": git_file_count,
            "journal_file_counts": journal_counts,
            "archived_analysis_scripts": archived_scripts,
            "portability_patch_file_count": len(portability_changes),
            "portability_replacement_count": sum(
                int(change["replacement_count"]) for change in portability_changes
            ),
            "excluded_classes": [
                "raw data and data caches",
                "model checkpoints",
                "scheduler scripts and logs except three protocol-authenticated worker records",
                "temporary results",
                "Python and test caches",
                "credential files",
                "presentations",
                "internal assistant and revision logs",
                "unrelated transformer/text/vision project runners, configs and tests",
            ],
            "raw_microns_or_dandi_data_included": False,
            "trained_checkpoints_included": False,
            "credentials_included": False,
            "source_data_distribution": "separate Source_Data.zip",
            "deterministic_timestamp_epoch": archive_epoch,
        }
        (temporary_stage / "METADATA.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary_stage / "METADATA.json").chmod(0o644)

        findings = scan_release(temporary_stage)
        if findings:
            raise RuntimeError(
                "Release policy scan failed:\n- " + "\n- ".join(findings)
            )

        checksummed = write_checksums(temporary_stage)
        validated = validate_checksums(temporary_stage)
        if validated != checksummed:
            raise RuntimeError("Internal checksum-count mismatch")

        temporary_archive = temporary_parent / ARCHIVE_NAME
        write_deterministic_zip(temporary_stage, temporary_archive, archive_epoch)
        zip_members = validate_zip(temporary_archive, temporary_stage)

        if stage.exists():
            remove_generated_tree(stage)
        for target in (archive, digest_file):
            if target.exists():
                target.unlink()
        temporary_stage.replace(stage)
        temporary_archive.replace(archive)
        digest_file.write_text(
            f"{sha256(archive)}  {archive.name}\n", encoding="utf-8"
        )
        digest_file.chmod(0o644)

        # Revalidate after the atomic moves.
        validate_checksums(stage)
        validate_zip(archive, stage)
        post_findings = scan_release(stage)
        if post_findings:
            raise RuntimeError(
                "Post-move release scan failed:\n- " + "\n- ".join(post_findings)
            )
        expected_digest = digest_file.read_text(encoding="utf-8").split()[0]
        if sha256(archive) != expected_digest:
            raise RuntimeError("Published ZIP checksum mismatch")

        return {
            "status": "ok",
            "stage": str(stage.relative_to(JOURNAL_ROOT)),
            "archive": str(archive.relative_to(JOURNAL_ROOT)),
            "archive_sha256": expected_digest,
            "archive_bytes": archive.stat().st_size,
            "zip_members": zip_members,
            "checksummed_files": validated,
            "git_commit": commit,
            "journal_git_commit": journal_commit,
            "git_snapshot_files": git_file_count,
            "journal_file_counts": journal_counts,
            "portability_patched_files": len(portability_changes),
            "portability_replacements": sum(
                int(change["replacement_count"]) for change in portability_changes
            ),
            "policy_scan_findings": 0,
        }
    finally:
        if temporary_parent.exists():
            remove_generated_tree(temporary_parent)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the generated software-release stage and archive",
    )
    parser.add_argument(
        "--implementation-root", type=Path,
        help="Production Git repository with pyproject.toml and src/dendritic_modeling; required for isolated paper snapshots without that ancestor.",
    )
    args = parser.parse_args()
    print(json.dumps(build(force=args.force, implementation_root=args.implementation_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
