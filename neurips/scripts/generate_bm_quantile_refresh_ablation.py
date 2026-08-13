#!/usr/bin/env python
from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
SWEEP_DIR = REPO_ROOT / "drafts" / "dendritic-local-learning" / "configs" / "sweeps"
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"

BRANCH_FACTORS = {
    "bf22": [2, 2],
    "bf333": [3, 3, 3],
}
INIT_POLICIES = {
    "previnit": "analytical",
    "quantinit": "quantile",
}
EMA_VARIANTS = {
    "hard": 1.0,
    "ema050": 0.50,
    "ema025": 0.25,
}
QUANTILE_REFRESH = {
    "reactivation_update_mode": "quantile",
    "recalibrate_reactivation_every": 1,
    "reactivation_recalibration_mode": "occupancy_quantile",
    "reactivation_recalibration_start_epoch": 2,
    "reactivation_recalibration_num_batches": 8,
}

DATASET_SPECS = {
    "mnist": {
        "run_name": "sweep_bm_quantile_refresh_ablation_mnist_classification",
        "slurm": {
            "account": "kempner_dev",
            "partition": "kempner_eng",
            "time": "08:00:00",
            "nodes": 1,
            "ntasks_per_node": 1,
            "gpus_per_node": 1,
            "cpus_per_task": 8,
            "mem": "48GB",
            "max_concurrent_jobs": 8,
            "modules_to_load": [],
            "conda_env_path": "",
            "training_script": "src/dendritic_modeling/scripts/training/train_experiments.py",
        },
        "seeds_per_condition": 1,
        "base_seed": 42,
        "shared_overrides": {
            "data": {
                "dataset_name": "mnist",
                "base_dir": str(DATA_DIR),
                "processing": {"flatten": True, "normalize": False},
            },
            "model": {
                "task": "classification",
                "encoder": {"type": "identity", "params": {"input_dim": None}},
                "core": {
                    "architecture": {
                        "excitatory_layer_sizes": [128],
                        "inhibitory_layer_sizes": [],
                        "inhibitory_branch_factors": [1],
                    },
                    "connectivity": {
                        "ee_synapses_per_branch_per_layer": [40],
                        "ei_synapses_per_branch_per_layer": [0],
                        "ie_synapses_per_branch_per_layer": [20],
                        "ii_synapses_per_branch_per_layer": [0],
                    },
                    "transfer": {
                        "input_mode": 1,
                        "independent_pathways": False,
                        "output_activation": "relu",
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
                "decoder": {
                    "type": "MLP",
                    "params": {"hidden_dims": [], "activation": "relu", "output_dim": 10},
                },
            },
        },
        "base_paths": {
            ("standard", "shunting"): "drafts/dendritic-local-learning/neurips/configs/fig1_mnist_shunting_bp_fixed100.yaml",
            ("standard", "additive"): "drafts/dendritic-local-learning/neurips/configs/fig1_mnist_shunting_bp_fixed100.yaml",
            ("local_ca", "shunting"): "drafts/dendritic-local-learning/neurips/configs/fig1_mnist_shunting_localca_fixed100.yaml",
            ("local_ca", "additive"): "drafts/dendritic-local-learning/neurips/configs/fig1_mnist_additive_localca_fixed100.yaml",
        },
    },
    "cifar10": {
        "run_name": "sweep_bm_quantile_refresh_ablation_cifar10_classification",
        "slurm": {
            "account": "kempner_dev",
            "partition": "kempner_eng",
            "time": "14:00:00",
            "nodes": 1,
            "ntasks_per_node": 1,
            "gpus_per_node": 1,
            "cpus_per_task": 8,
            "mem": "64GB",
            "max_concurrent_jobs": 8,
            "modules_to_load": [],
            "conda_env_path": "",
            "training_script": "src/dendritic_modeling/scripts/training/train_experiments.py",
        },
        "seeds_per_condition": 1,
        "base_seed": 42,
        "shared_overrides": {
            "data": {
                "dataset_name": "cifar10",
                "base_dir": str(DATA_DIR),
                "processing": {"flatten": True, "normalize": False},
            },
            "model": {
                "task": "classification",
                "encoder": {"type": "identity", "params": {"input_dim": None}},
                "core": {
                    "architecture": {
                        "excitatory_layer_sizes": [256],
                        "inhibitory_layer_sizes": [],
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
                "decoder": {
                    "type": "MLP",
                    "params": {"hidden_dims": [], "activation": "relu", "output_dim": 10},
                },
            },
        },
        "base_paths": {
            ("standard", "shunting"): "drafts/dendritic-local-learning/neurips/configs/cifar10_shunting_deep333_ie10_standard_strong.yaml",
            ("standard", "additive"): "drafts/dendritic-local-learning/neurips/configs/cifar10_additive_deep333_ie10_standard_strong.yaml",
            ("local_ca", "shunting"): "drafts/dendritic-local-learning/neurips/configs/cifar10_shunting_deep333_ie10_localca_strong.yaml",
            ("local_ca", "additive"): "drafts/dendritic-local-learning/neurips/configs/cifar10_additive_deep333_ie10_localca_strong.yaml",
        },
    },
}


def deep_merge(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


for dataset_name, spec in DATASET_SPECS.items():
    sweep = {
        "output_dir": OUTPUT_DIR,
        "outputs": {"run_name": spec["run_name"]},
        "slurm_config": spec["slurm"],
        "experiment_settings": {
            "seeds_per_condition": spec["seeds_per_condition"],
            "base_seed": spec["base_seed"],
            "use_array_jobs": True,
        },
        "configs": [],
    }

    for strategy, core, branch_key, init_key, ema_key in product(
        ("standard", "local_ca"),
        ("additive", "shunting"),
        BRANCH_FACTORS,
        INIT_POLICIES,
        EMA_VARIANTS,
    ):
        branch_factors = BRANCH_FACTORS[branch_key]
        init_policy = INIT_POLICIES[init_key]
        ema_alpha = EMA_VARIANTS[ema_key]
        run_name = (
            f"{dataset_name}_{core}_{strategy}_quantilebm_{init_key}_{branch_key}_"
            f"e1_nb8_{ema_key}"
        )
        overrides = deep_merge(
            spec["shared_overrides"],
            {
                "model": {
                    "core": {
                        "type": f"dendritic_{core}",
                        "architecture": {"excitatory_branch_factors": branch_factors},
                        "reactivation": {"init_policy": init_policy},
                    },
                },
                "training": {
                    "main": {
                        "strategy": strategy,
                        "common": {
                            **QUANTILE_REFRESH,
                            "reactivation_recalibration_ema_alpha": ema_alpha,
                        },
                    }
                },
                "outputs": {"run_name": run_name},
            },
        )
        if strategy == "local_ca":
            overrides = deep_merge(
                overrides,
                {
                    "training": {
                        "main": {
                            "learning_strategy_config": {
                                "update_reactivation": False,
                            }
                        }
                    }
                },
            )

        sweep["configs"].append(
            {
                "name": run_name,
                "path": spec["base_paths"][(strategy, core)],
                "run_name": run_name,
                "overrides": overrides,
            }
        )

    out_path = SWEEP_DIR / f"{spec['run_name']}.yaml"
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(sweep, handle, sort_keys=False)
    print(f"wrote {out_path}")
