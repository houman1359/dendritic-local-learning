#!/usr/bin/env python3
"""Generate the frozen positive-rate nonlinear physical-depth canary.

The generator copies the complete production population-network recipe into
the journal directory, then varies only prespecified task/routing fields.  The
resulting YAML files are self-contained; running them does not depend on the
sibling gain--load manuscript tree.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = JOURNAL_ROOT.parents[2]
SOURCE = (
    REPO_ROOT
    / "drafts"
    / "gain-load-journal"
    / "neurips"
    / "configs"
    / "sweeps"
    / "neurips_hierarchical_gain_inventory_pilot_revision.yaml"
)
OUTPUT = JOURNAL_ROOT / "configs" / "nonlinear_physical_depth"
RUNS = JOURNAL_ROOT / "nonlinear_physical_depth_runs"

MORPHOLOGIES = [[8], [2, 3], [2, 1, 2]]
CANARY_SEEDS = [10100, 10101]
ACCESSIBILITY_SEEDS = [10110, 10111]
ACCESSIBILITY_TEST_GAINS = [0.25, 0.5, 0.8]
COUPLING_SEEDS = [10120, 10121]
COUPLING_LADDER = [1.0, 4.0, 16.0, 64.0]
SIGNAL_SEEDS = [10130, 10131]
SIGNAL_LADDER = [0.24, 0.36, 0.48, 0.72]
BOUNDARY_SEEDS = [10140, 10141]
FEEDBACK_MODES = ["per_soma_shared", "path_transport"]

REGIMES: dict[str, dict[str, Any]] = {
    "aligned": {
        "sensor_alignment_alpha": 1.0,
        "sensor_support_mode": "matched",
        "feature_ranges": [[0, 21], [21, 43], [43, 64]],
    },
    "zero_alignment": {
        "sensor_alignment_alpha": 0.0,
        "sensor_support_mode": "matched",
        "feature_ranges": [[0, 21], [21, 43], [43, 64]],
    },
    "sensor_shuffled": {
        "sensor_alignment_alpha": 1.0,
        "sensor_support_mode": "shuffled",
        "feature_ranges": [[0, 21], [21, 43], [43, 64]],
    },
    "rewired_tree": {
        "sensor_alignment_alpha": 1.0,
        "sensor_support_mode": "matched",
        # Reverse fine and global placement while retaining the middle tier.
        # The blocks have widths 21/22/21, so a three-cycle would assign the
        # 22-wide block to a four-branch inventory and change candidate slots.
        # This reversal is the non-overlapping placement control that exactly
        # preserves every tier width and resource count.
        "feature_ranges": [[43, 64], [21, 43], [0, 21]],
    },
}


def _plain_config() -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.load(SOURCE), resolve=True)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {SOURCE}")
    return value


def _base(
    *,
    strategy: str,
    regime: str,
    seeds: list[int] | None = None,
    test_gain_sigma: float = 1.2,
) -> dict[str, Any]:
    config = copy.deepcopy(_plain_config())
    seeds = CANARY_SEEDS if seeds is None else seeds
    if strategy not in {"standard", "local_ca"}:
        raise ValueError(strategy)
    if regime not in REGIMES:
        raise ValueError(regime)

    config["output_dir"] = str(RUNS)
    config["slurm_config"].update(
        {
            "account": "kempner_dev",
            "partition": "kempner_requeue",
            "time": "02:00:00",
            "cpus_per_task": 4,
            "mem": "32GB",
            "max_concurrent_jobs": 6,
            "training_script": (
                "src/dendritic_modeling/scripts/training/train_experiments.py"
            ),
        }
    )
    config["experiment_settings"] = {
        "seeds_per_condition": len(seeds),
        "base_seed": seeds[0],
        "use_array_jobs": True,
    }

    base = config["base_config"]
    dataset = base["data"]["dataset_params"]["hierarchical_gain_load"]
    dataset.update(
        {
            "n_samples": 10000,
            "n_levels": 3,
            "gain_structure": "hierarchical",
            "e_signal_delta": 0.24,
            "train_gain_sigma": 0.25,
            "test_gain_sigma": test_gain_sigma,
            "private_gain_sigma": 0.0,
            "load_mean": 0.0,
            "load_noise_sigma": 0.0,
            "valid_split_mode": "train",
            "sensor_alignment_alpha": REGIMES[regime]["sensor_alignment_alpha"],
            "sensor_support_mode": REGIMES[regime]["sensor_support_mode"],
        }
    )

    layer = base["model"]["core"]["population_network"]["layers"][0]
    defaults = layer["population_defaults"]
    defaults.update(
        {
            "use_shunting": True,
            "reactivate": False,
            "reactivation_type": "identity",
            "dbl_init_method": "mechanism_neutral",
            "adaptive_initialization": False,
            "initial_child_conductance": 16.0,
        }
    )
    structured = defaults["structured_connectivity"]["pathways"]
    for pathway in ("ee", "ie"):
        structured[pathway]["feature_ranges"] = copy.deepcopy(
            REGIMES[regime]["feature_ranges"]
        )

    training = base["training"]["main"]
    training["strategy"] = strategy
    common = training["common"]
    common.update(
        {
            "epochs": 180,
            "early_stopping": True,
            "patience": 30,
            "checkpointing": True,
            "checkpoint_interval": 25,
            "load_best_state_dict": True,
            "enable_nan_checking": True,
            "param_groups": {
                "lr": 0.003 if strategy == "standard" else 0.0008,
                "split_params": strategy == "local_ca",
                "topk_lr": 0.0015,
                "blocklinear_lr": 0.0007,
                "reactivation_lr": 0.0,
                "decoder_lr": 0.0015,
            },
        }
    )
    training["optimizer"] = {"name": "adam", "lr": common["param_groups"]["lr"]}
    training["learning_strategy_config"] = {
        "rule_variant": "3f",
        "error_mode": "auto",
        "error_broadcast_mode": "per_soma_shared",
        "broadcast_bandwidth": "full",
        "error_noise_sigma": 0.0,
        "decoder_update_mode": "local",
        "encoder_update_mode": "none",
        "update_reactivation": False,
        "update_inactive_weights": False,
        "normalize_by_batch": True,
        "clip_grad_value": 5.0,
        "three_factor": {
            "dynamics_mode": "auto",
            "use_conductance_scaling": True,
            "use_driving_force": True,
            "theta": 0.0,
            "e_rev_exc": 1.0,
            "e_rev_inh": 0.0,
        },
        "morphology_aware": {
            "use_path_propagation": False,
            "morphology_modulator_mode": "none",
            "use_dendritic_normalization": False,
        },
        "hsic": {"enabled": False, "weight": 0.0},
    }

    base["analysis"] = {
        "performance": {
            "enabled": True,
            "training": True,
            "params": {
                "accuracy": True,
                "auc": True,
                "categorical_loglikelihood": True,
            },
        },
        "information": {"enabled": False},
        # The generic compartment-statistics renderer assumes the legacy
        # single-cell weight schema (``exc_weights_mean``).  PopulationNetwork
        # checkpoints expose indexed sparse synapses instead, so enabling that
        # renderer produces a post-training KeyError even though training and
        # performance export complete.  Physical-depth mechanism endpoints are
        # collected by a dedicated, schema-aware diagnostic rather than by the
        # incompatible legacy renderer.
        "compartment_statistics": {"enabled": False, "training": False},
    }
    label = "bp" if strategy == "standard" else "local3f"
    run_name = f"journal_canary_physical_depth_{regime}_{label}"
    base["outputs"]["run_name"] = run_name
    base["outputs"]["results_dir"] = "outputs"
    config["outputs"] = {"run_name": run_name}
    return config


def _render_one(*, strategy: str, regime: str) -> tuple[Path, str]:
    config = _base(strategy=strategy, regime=regime)
    sweep: dict[str, Any] = {
        "model.core.population_network.layers.0.populations.0.branch_factors": (
            MORPHOLOGIES
        ),
    }
    conditions = len(MORPHOLOGIES)
    if strategy == "local_ca":
        sweep["training.main.learning_strategy_config.error_broadcast_mode"] = (
            FEEDBACK_MODES
        )
        conditions *= len(FEEDBACK_MODES)
    config["sweep_config"] = sweep
    config["sweep_contract"] = {
        "expected_config_count": conditions * len(CANARY_SEEDS)
    }
    label = "bp" if strategy == "standard" else "local3f"
    path = OUTPUT / f"canary_{regime}_{label}.yaml"
    return path, OmegaConf.to_yaml(OmegaConf.create(config))


def render() -> dict[Path, str]:
    return dict(
        _render_one(strategy=strategy, regime=regime)
        for regime in REGIMES
        for strategy in ("standard", "local_ca")
    )


def render_accessibility() -> dict[Path, str]:
    """Render the post-canary, explicitly exploratory gain-shift ladder."""
    config = _base(
        strategy="standard",
        regime="aligned",
        seeds=ACCESSIBILITY_SEEDS,
        test_gain_sigma=ACCESSIBILITY_TEST_GAINS[0],
    )
    run_name = "journal_exploratory_physical_depth_accessibility_bp"
    config["base_config"]["outputs"]["run_name"] = run_name
    config["outputs"]["run_name"] = run_name
    config["sweep_config"] = {
        "model.core.population_network.layers.0.populations.0.branch_factors": (
            MORPHOLOGIES
        ),
        "data.dataset_params.hierarchical_gain_load.test_gain_sigma": (
            ACCESSIBILITY_TEST_GAINS
        ),
    }
    config["sweep_contract"] = {
        "expected_config_count": (
            len(MORPHOLOGIES)
            * len(ACCESSIBILITY_TEST_GAINS)
            * len(ACCESSIBILITY_SEEDS)
        ),
        "status": "exploratory_post_canary",
    }
    path = OUTPUT / "exploratory_accessibility_aligned_bp.yaml"
    return {path: OmegaConf.to_yaml(OmegaConf.create(config))}


def render_coupling_accessibility() -> dict[Path, str]:
    """Render the post-accessibility, exploratory axial-coupling ladder."""
    config = _base(
        strategy="standard",
        regime="aligned",
        seeds=COUPLING_SEEDS,
        test_gain_sigma=0.25,
    )
    run_name = "journal_exploratory_physical_depth_coupling_bp"
    config["base_config"]["outputs"]["run_name"] = run_name
    config["outputs"]["run_name"] = run_name
    config["sweep_config"] = {
        "model.core.population_network.layers.0.populations.0.branch_factors": (
            MORPHOLOGIES
        ),
        (
            "model.core.population_network.layers.0.population_defaults."
            "initial_child_conductance"
        ): COUPLING_LADDER,
    }
    config["sweep_contract"] = {
        "expected_config_count": (
            len(MORPHOLOGIES) * len(COUPLING_LADDER) * len(COUPLING_SEEDS)
        ),
        "status": "exploratory_post_accessibility",
    }
    path = OUTPUT / "exploratory_coupling_depth_aligned_bp.yaml"
    return {path: OmegaConf.to_yaml(OmegaConf.create(config))}


def render_signal_accessibility() -> dict[Path, str]:
    """Render the final exploratory signal-accessibility calibration."""
    config = _base(
        strategy="standard",
        regime="aligned",
        seeds=SIGNAL_SEEDS,
        test_gain_sigma=0.25,
    )
    run_name = "journal_exploratory_physical_depth_signal_accessibility_bp"
    config["base_config"]["outputs"]["run_name"] = run_name
    config["outputs"]["run_name"] = run_name
    config["sweep_config"] = {
        "model.core.population_network.layers.0.populations.0.branch_factors": (
            MORPHOLOGIES
        ),
        "data.dataset_params.hierarchical_gain_load.e_signal_delta": SIGNAL_LADDER,
    }
    config["sweep_contract"] = {
        "expected_config_count": (
            len(MORPHOLOGIES) * len(SIGNAL_LADDER) * len(SIGNAL_SEEDS)
        ),
        "status": "exploratory_post_coupling",
        "fixed_initial_child_conductance": 16.0,
    }
    path = OUTPUT / "exploratory_signal_accessibility_aligned_bp.yaml"
    return {path: OmegaConf.to_yaml(OmegaConf.create(config))}


def render_boundary_resolution() -> dict[Path, str]:
    """Render the single-point post-ladder accessibility boundary check."""
    config = _base(
        strategy="standard",
        regime="aligned",
        seeds=BOUNDARY_SEEDS,
        test_gain_sigma=0.25,
    )
    run_name = "journal_exploratory_physical_depth_boundary_bp"
    config["base_config"]["outputs"]["run_name"] = run_name
    config["outputs"]["run_name"] = run_name
    dataset = config["base_config"]["data"]["dataset_params"][
        "hierarchical_gain_load"
    ]
    dataset["e_signal_delta"] = 0.80
    config["sweep_config"] = {
        "model.core.population_network.layers.0.populations.0.branch_factors": (
            MORPHOLOGIES
        )
    }
    config["sweep_contract"] = {
        "expected_config_count": len(MORPHOLOGIES) * len(BOUNDARY_SEEDS),
        "status": "exploratory_boundary_resolution",
        "fixed_signal_delta": 0.80,
        "no_further_signal_search": True,
    }
    path = OUTPUT / "exploratory_boundary_resolution_aligned_bp.yaml"
    return {path: OmegaConf.to_yaml(OmegaConf.create(config))}


def generate(*, include_accessibility: bool = True) -> list[Path]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rendered = render()
    if include_accessibility:
        rendered.update(render_accessibility())
        rendered.update(render_coupling_accessibility())
        rendered.update(render_signal_accessibility())
        rendered.update(render_boundary_resolution())
    for path, content in rendered.items():
        path.write_text(content)
    return list(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--canary-only",
        action="store_true",
        help="generate/check only the original frozen canary files",
    )
    args = parser.parse_args()
    expected = render()
    if not args.canary_only:
        expected.update(render_accessibility())
        expected.update(render_coupling_accessibility())
        expected.update(render_signal_accessibility())
        expected.update(render_boundary_resolution())
    if args.check:
        mismatches = [
            path
            for path, content in expected.items()
            if not path.exists() or path.read_text() != content
        ]
        if mismatches:
            raise SystemExit("Generated config mismatch: " + ", ".join(map(str, mismatches)))
        return
    for path in generate(include_accessibility=not args.canary_only):
        print(path.relative_to(JOURNAL_ROOT))


if __name__ == "__main__":
    main()
