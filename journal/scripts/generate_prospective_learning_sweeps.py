#!/usr/bin/env python3
"""Generate the frozen depth x feedback x core experiment suite.

The script deliberately writes separate arrays for each task/core pair. This
keeps architecture-specific initialization explicit and prevents backprop
controls from being duplicated across feedback modes that they do not use.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


JOURNAL_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = JOURNAL_ROOT / "configs" / "prospective_learning_benefits"
BASE_CONFIGS = {
    "shunting": JOURNAL_ROOT
    / "configs"
    / "reruns"
    / "feedback_definition_shunting_15seed.yaml",
    "additive": JOURNAL_ROOT
    / "configs"
    / "reruns"
    / "feedback_definition_additive_15seed.yaml",
}

DEPTHS = [[2], [2, 2], [2, 2, 2], [2, 2, 2, 2]]
CANARY_DEPTHS = [DEPTHS[0], DEPTHS[-1]]
FEEDBACK_MODES = ["per_soma", "per_soma_shared", "path_transport"]


def _load_base(core: str) -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.load(BASE_CONFIGS[core]), resolve=True)
    assert isinstance(value, dict)
    return value


def _configure_task(config: dict[str, Any], task: str) -> None:
    data = config["base_config"]["data"]
    data["dataset_name"] = "mnist" if task == "mnist" else "noise_resilience"
    data["base_dir"] = "data"
    data["processing"] = {"flatten": True, "normalize": False}
    if task == "noise":
        data["dataset_params"] = {
            "noise_resilience": {
                "sigma_task": 1.5,
                "noise_latent_dim": 50,
                "projection_seed": 0,
                "train_noise_seed": 1,
                "valid_noise_seed": 2,
                "test_noise_seed": 3,
                "clamp_inputs": True,
            }
        }
    else:
        data.pop("dataset_params", None)


def _common_config(
    *, core: str, task: str, phase: str, strategy: str
) -> dict[str, Any]:
    config = copy.deepcopy(_load_base(core))
    confirmatory = phase == "confirmatory"
    seeds = 10 if confirmatory else 1
    depths = DEPTHS if confirmatory else CANARY_DEPTHS
    feedback = strategy == "local_ca"

    config["output_dir"] = "drafts/dendritic-local-learning/journal/prospective_runs"
    config["slurm_config"].update(
        {
            "account": "kempner_dev",
            "partition": "kempner_requeue",
            "time": "04:00:00" if confirmatory else "01:00:00",
            "cpus_per_task": 4,
            "mem": "48GB",
            "max_concurrent_jobs": 2 if confirmatory and feedback else 1,
        }
    )
    config["experiment_settings"] = {
        "seeds_per_condition": seeds,
        "base_seed": 42,
        "use_array_jobs": True,
    }

    sweep = {
        "model.core.architecture.excitatory_branch_factors": depths,
    }
    if feedback:
        sweep["training.main.learning_strategy_config.error_broadcast_mode"] = (
            FEEDBACK_MODES
        )
    config["sweep_config"] = sweep
    config["sweep_contract"] = {
        "expected_config_count": len(depths)
        * seeds
        * (len(FEEDBACK_MODES) if feedback else 1)
    }

    _configure_task(config, task)
    base = config["base_config"]
    base["model"]["core"]["architecture"]["excitatory_layer_sizes"] = [128]
    base["model"]["core"]["architecture"]["inhibitory_layer_sizes"] = []
    base["model"]["core"]["connectivity"].update(
        {
            "ee_synapses_per_branch_per_layer": [40],
            "ei_synapses_per_branch_per_layer": [0],
            "ie_synapses_per_branch_per_layer": [20],
            "ii_synapses_per_branch_per_layer": [0],
        }
    )
    base["model"]["core"]["transfer"]["output_activation"] = None
    base["training"]["main"]["strategy"] = (
        "local_ca" if feedback else "standard"
    )
    base["training"]["main"]["common"].update(
        {
            "epochs": 180,
            "early_stopping": False,
            "load_best_state_dict": True,
            "print_every": 10,
            "checkpointing": False,
        }
    )
    local = base["training"]["main"]["learning_strategy_config"]
    local["rule_variant"] = "3f"
    local["decoder_update_mode"] = "local"
    local["morphology_aware"]["use_path_propagation"] = False
    base["analysis"]["performance_analysis"]["training"] = True

    strategy_label = "local3f" if feedback else "backprop"
    run_name = f"journal_{phase}_{task}_{core}_{strategy_label}_depth_feedback"
    base["outputs"]["run_name"] = run_name
    return config


def render() -> dict[Path, str]:
    rendered: dict[Path, str] = {}
    for phase in ("canary", "confirmatory"):
        for task in ("mnist", "noise"):
            for core in ("shunting", "additive"):
                for strategy in ("local_ca", "backprop"):
                    config = _common_config(
                        core=core,
                        task=task,
                        phase=phase,
                        strategy=strategy,
                    )
                    label = "local3f" if strategy == "local_ca" else "backprop"
                    path = CONFIG_DIR / f"{phase}_{task}_{core}_{label}.yaml"
                    rendered[path] = OmegaConf.to_yaml(OmegaConf.create(config))
    return rendered


def generate() -> list[Path]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    rendered = render()
    for path, text in rendered.items():
        path.write_text(text)
    return list(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that generated files already match the frozen generator",
    )
    args = parser.parse_args()

    if args.check:
        expected = render()
        mismatches = [
            path
            for path, text in expected.items()
            if not path.exists() or path.read_text() != text
        ]
        if mismatches:
            raise SystemExit("Generated config mismatch: " + ", ".join(map(str, mismatches)))
    else:
        for path in generate():
            print(path.relative_to(JOURNAL_ROOT))


if __name__ == "__main__":
    main()
