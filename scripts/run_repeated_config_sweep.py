#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import subprocess
from datetime import datetime
from pathlib import Path

import yaml
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]


def deep_merge(base, update):
    if isinstance(base, dict) and isinstance(update, dict):
        merged = copy.deepcopy(base)
        for key, value in update.items():
            if key in merged:
                merged[key] = deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged
    return copy.deepcopy(update)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def dump_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def build_configs(manifest: dict, sweep_root: Path) -> list[Path]:
    configs_dir = sweep_root / "configs"
    results_dir = sweep_root / "results"
    configs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    exp_settings = manifest.get("experiment_settings", {})
    seeds_per_condition = int(exp_settings.get("seeds_per_condition", 1))
    base_seed = int(exp_settings.get("base_seed", 42))

    saved_paths: list[Path] = []
    config_index = 0
    for entry in manifest.get("configs", []):
        base_path = REPO_ROOT / entry["path"]
        base_cfg = load_yaml(base_path)
        overrides = entry.get("overrides", {})
        run_name = str(entry.get("run_name") or entry.get("name"))

        for seed_offset in range(seeds_per_condition):
            seed = base_seed + seed_offset
            cfg = deep_merge(base_cfg, overrides)
            cfg.setdefault("experiment", {})["seed"] = seed
            cfg.setdefault("outputs", {})
            seeded_name = run_name if seeds_per_condition == 1 else f"{run_name}_s{seed}"
            cfg["outputs"]["run_name"] = seeded_name
            cfg["outputs"]["results_dir"] = str(results_dir / f"config_{config_index}")
            out_path = configs_dir / f"{seeded_name}.yaml"
            dump_yaml(out_path, cfg)
            saved_paths.append(out_path)
            config_index += 1

    return saved_paths


def write_array_script(manifest: dict, sweep_root: Path, config_paths: list[Path]) -> Path:
    jobs_dir = sweep_root / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    slurm = manifest["slurm_config"]
    run_name = manifest["outputs"]["run_name"]
    array_max = len(config_paths) - 1
    max_concurrent = slurm.get("max_concurrent_jobs")
    array_spec = f"0-{array_max}"
    if max_concurrent:
        array_spec += f"%{int(max_concurrent)}"

    config_entries = "\n".join(f'    "{path}"' for path in config_paths)
    script = f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --account={slurm['account']}
#SBATCH --partition={slurm['partition']}
#SBATCH --chdir={REPO_ROOT}
#SBATCH --time={slurm['time']}
#SBATCH --nodes={slurm['nodes']}
#SBATCH --ntasks-per-node={slurm['ntasks_per_node']}
#SBATCH --gpus-per-node={slurm['gpus_per_node']}
#SBATCH --cpus-per-task={slurm['cpus_per_task']}
#SBATCH --mem={slurm['mem']}
#SBATCH --array={array_spec}
#SBATCH --output={sweep_root}/jobs/%A_%a/output_%A_%a.out
#SBATCH --error={sweep_root}/jobs/%A_%a/error_%A_%a.err

set -euo pipefail

module purge
PYTHON=\"python\"
export PYTHONPATH=\"$(pwd)/src:${{PYTHONPATH:-}}\"
OUTPUT_DIR=\"{sweep_root}\"
CONFIG_FILES=(
{config_entries}
)
CONFIG=\"${{CONFIG_FILES[$SLURM_ARRAY_TASK_ID]}}\"
OUTPUT_FOLDER_PATH=\"${{OUTPUT_DIR}}/results/config_${{SLURM_ARRAY_TASK_ID}}\"
mkdir -p \"${{OUTPUT_FOLDER_PATH}}\"
mkdir -p \"{sweep_root}/jobs/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}\"
echo \"Starting training job at $(date)\"
echo \"Array task $SLURM_ARRAY_TASK_ID processing: $CONFIG\"
start_time=$(date +%s)
srun --cpus-per-task=${{SLURM_CPUS_PER_TASK}} --kill-on-bad-exit $PYTHON -u $(pwd)/{slurm['training_script']} \"$CONFIG\" --output_dir \"${{OUTPUT_FOLDER_PATH}}\"
end_time=$(date +%s)
echo \"Done with training job at $(date)\"
echo \"Total duration: $((end_time - start_time)) seconds.\" 
"""
    path = jobs_dir / "run_array_sweep.sh"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/submit repeated-config sweep")
    parser.add_argument("--config", required=True, help="Manifest path")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.config)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    manifest = load_yaml(manifest_path)

    output_dir = Path(manifest["output_dir"])
    run_name = manifest["outputs"]["run_name"]
    sweep_root = output_dir / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    dump_yaml(sweep_root / "original_config.yaml", manifest)

    config_paths = build_configs(manifest, sweep_root)
    array_script = write_array_script(manifest, sweep_root, config_paths)

    print(f"Created sweep directory: {sweep_root}")
    print(f"Saved {len(config_paths)} configs")
    print(f"Array script: {array_script}")

    if args.generate_only:
        return

    result = subprocess.run(["sbatch", str(array_script)], capture_output=True, text=True, check=True)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
