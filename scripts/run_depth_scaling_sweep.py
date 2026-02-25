#!/usr/bin/env python3
"""Generate and launch depth-scaling sweep (Bartunov-style).

Tests how gradient fidelity and accuracy scale with dendritic depth.
Varies branch_factors depth while keeping total parameter budget roughly constant.

Generates configs for:
  - depth = 1, 2, 3, 4 dendritic layers
  - core_type = shunting, additive
  - strategy = standard (backprop), local_ca (5f per_soma)
  - seeds = 42, 43, 44
Total: 4 depths × 2 cores × 2 strategies × 3 seeds = 48 configs
"""
from __future__ import annotations

import argparse
import os
import stat
import yaml
from pathlib import Path
from copy import deepcopy


# Base config template
BASE_CONFIG = {
    "experiment": {
        "seed": 42,
        "enable_amp": False,
    },
    "data": {
        "dataset_name": "mnist",
        "base_dir": "",
        "processing": "flatten",
    },
    "model": {
        "task": "classification",
        "encoder": {"type": "identity"},
        "core": {
            "type": "dendritic_shunting",
            "architecture": {
                "excitatory_layer_sizes": [20],
                "inhibitory_layer_sizes": [],
                "excitatory_branch_factors": [3, 3],
                "inhibitory_branch_factors": [1],
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [40],
                "ei_synapses_per_branch_per_layer": [0],
                "ie_synapses_per_branch_per_layer": [10],
                "ii_synapses_per_branch_per_layer": [0],
            },
            "transfer": {
                "input_mode": 1,
                "independent_pathways": False,
                "output_activation": None,
            },
            "morphology": {
                "somatic_synapses": False,
                "weight_transform": "softplus",
            },
            "reactivation": {
                "enabled": True,
                "type": "param_tanh",
                "init_m": 1.5,
                "init_b": 0.5,
            },
        },
    },
    "training": {
        "main": {
            "strategy": "standard",
            "common": {
                "epochs": 50,
                "batch_size": 256,
                "shuffle": True,
                "loss_function": "cat_nll",
                "early_stopping": False,
                "checkpointing": True,
                "checkpoint_interval": 5,
                "param_groups": {
                    "lr": 0.002,
                    "split_params": True,
                    "topk_lr": 0.002,
                    "blocklinear_lr": 0.001,
                    "reactivation_lr": 0.001,
                    "decoder_lr": 0.002,
                },
            },
        }
    },
    "analysis": {
        "performance_analysis": True,
    },
}

LOCAL_CA_OVERRIDE = {
    "strategy": "local_ca",
    "learning_strategy_config": {
        "rule_variant": "5f",
        "error_broadcast_mode": "per_soma",
        "error_mode": "auto",
        "three_factor": {
            "dynamics_mode": "auto",
            "use_conductance_scaling": True,
            "use_driving_force": True,
            "theta": 0.0,
        },
        "four_factor": {
            "rho_mode": "correlation",
            "ema_alpha": 0.1,
        },
        "five_factor": {
            "phi_mode": "conditional_variance",
            "phi_estimator": "ridge",
            "phi_ridge_lambda": 0.01,
            "phi_clamp_min": 0.25,
            "phi_clamp_max": 4.0,
        },
        "morphology_aware": {
            "use_path_propagation": False,
        },
        "hsic": {"enabled": False},
        "decoder_update_mode": "backprop",
    },
}


# Depth configurations: vary branch_factors while keeping ~similar capacity
# More layers = smaller per-layer branching to keep params comparable
DEPTH_CONFIGS = {
    1: {"branch_factors": [9], "n_soma": 20, "ee_syn": [40], "ie_syn": [10]},
    2: {"branch_factors": [3, 3], "n_soma": 20, "ee_syn": [40], "ie_syn": [10]},
    3: {"branch_factors": [3, 3, 3], "n_soma": 15, "ee_syn": [30], "ie_syn": [8]},
    4: {"branch_factors": [2, 2, 2, 2], "n_soma": 15, "ee_syn": [25], "ie_syn": [6]},
}


def make_config(depth: int, core_type: str, strategy: str, seed: int) -> dict:
    """Build a single experiment config."""
    cfg = deepcopy(BASE_CONFIG)
    cfg["experiment"]["seed"] = seed

    # Core type
    cfg["model"]["core"]["type"] = core_type

    # Depth
    dc = DEPTH_CONFIGS[depth]
    cfg["model"]["core"]["architecture"]["excitatory_branch_factors"] = dc["branch_factors"]
    cfg["model"]["core"]["architecture"]["excitatory_layer_sizes"] = [dc["n_soma"]]
    cfg["model"]["core"]["connectivity"]["ee_synapses_per_branch_per_layer"] = dc["ee_syn"]
    cfg["model"]["core"]["connectivity"]["ie_synapses_per_branch_per_layer"] = dc["ie_syn"]

    # Strategy
    if strategy == "local_ca":
        cfg["training"]["main"].update(deepcopy(LOCAL_CA_OVERRIDE))
    else:
        cfg["training"]["main"]["strategy"] = "standard"

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/depth_scaling_sweep"),
    )
    parser.add_argument("--submit", action="store_true", help="Submit SLURM jobs")
    parser.add_argument(
        "--partition", default="kempner_h100", help="SLURM partition"
    )
    parser.add_argument("--time", default="01:00:00", help="SLURM time limit")
    parser.add_argument(
        "--sweep-output-dir",
        type=Path,
        default=Path(
            "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for depth in [1, 2, 3, 4]:
        for core_type in ["dendritic_shunting", "dendritic_additive"]:
            for strategy in ["standard", "local_ca"]:
                for seed in [42, 43, 44]:
                    cfg = make_config(depth, core_type, strategy, seed)
                    name = f"depth{depth}_{core_type.split('_')[1]}_{strategy}_s{seed}"
                    cfg_path = args.output_dir / f"{name}.yaml"
                    with open(cfg_path, "w") as f:
                        yaml.dump(cfg, f, default_flow_style=False)
                    configs.append((name, cfg_path))

    print(f"Generated {len(configs)} configs in {args.output_dir}")

    if args.submit:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        sweep_name = f"sweep_neurips_depth_scaling_{timestamp}"
        sweep_dir = args.sweep_output_dir / sweep_name
        sweep_dir.mkdir(parents=True, exist_ok=True)
        (sweep_dir / "configs").mkdir(exist_ok=True)
        (sweep_dir / "results").mkdir(exist_ok=True)
        (sweep_dir / "jobs").mkdir(exist_ok=True)

        src_root = Path(__file__).resolve().parents[2] / "src"
        train_script = src_root / "dendritic_modeling" / "scripts" / "training" / "train_experiments.py"

        submit_script = sweep_dir / "submit_all_jobs.sh"
        lines = ["#!/bin/bash", f"# Depth scaling sweep: {sweep_name}", ""]

        for i, (name, cfg_path) in enumerate(configs):
            # Copy config
            import shutil
            dest_cfg = sweep_dir / "configs" / f"unified_config_{i}.yaml"
            shutil.copy2(cfg_path, dest_cfg)

            result_dir = sweep_dir / "results" / f"config_{i}"
            result_dir.mkdir(exist_ok=True)

            job_script = sweep_dir / "jobs" / f"job_{i}.sh"
            with open(job_script, "w") as f:
                f.write(f"""#!/bin/bash
#SBATCH --job-name=depth_{name}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output={sweep_dir}/jobs/job_{i}.out
#SBATCH --error={sweep_dir}/jobs/job_{i}.err

export PYTHONPATH={src_root}:$PYTHONPATH
python {train_script} {dest_cfg} \\
    --output_dir {result_dir} \\
    --run_name {name}
""")
            os.chmod(job_script, os.stat(job_script).st_mode | stat.S_IEXEC)
            lines.append(f"sbatch {job_script}")

        with open(submit_script, "w") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(submit_script, os.stat(submit_script).st_mode | stat.S_IEXEC)
        print(f"\nSweep directory: {sweep_dir}")
        print(f"Submit all with: bash {submit_script}")
    else:
        print("Use --submit to generate and submit SLURM jobs")


if __name__ == "__main__":
    main()
