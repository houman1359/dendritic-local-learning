#!/usr/bin/env python3
"""Expand a list of base configs across seeds and submit them as one sweep."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dendritic_modeling.scripts.sweeps.unified.base_sweep import SweepJobManager  # noqa: E402


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _make_sweep_dir(base_dir: str, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_dir = Path(base_dir) / f"{prefix}_{timestamp}"
    for subdir in ("configs", "results", "jobs", "plots"):
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir


def _clone_config(config: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


def _apply_overrides(config: DictConfig, overrides: Any) -> DictConfig:
    if not overrides:
        return config
    override_cfg = OmegaConf.create(overrides)
    return OmegaConf.merge(config, override_cfg)


def _set_seed(config: DictConfig, seed: int) -> None:
    if "experiment" not in config or config.experiment is None:
        config["experiment"] = {}
    config.experiment.seed = int(seed)


def _set_run_outputs(
    config: DictConfig,
    *,
    run_name: str,
    results_dir: Path,
    group_name: str,
    base_config_path: str,
) -> None:
    if "outputs" not in config or config.outputs is None:
        config["outputs"] = {}
    config.outputs.run_name = run_name
    config.outputs.results_dir = str(results_dir)
    config["_seed_repeat_group"] = group_name
    config["_seed_repeat_base_config"] = base_config_path


def _save_manifest_copy(manifest: DictConfig, sweep_dir: Path) -> None:
    OmegaConf.save(manifest, sweep_dir / "original_manifest.yaml")


def _generate_configs(manifest: DictConfig, sweep_dir: Path) -> list[str]:
    seeds_per_condition = int(
        manifest.get("experiment_settings", {}).get("seeds_per_condition", 1)
    )
    base_seed = int(manifest.get("experiment_settings", {}).get("base_seed", 42))
    config_specs = manifest.get("configs", [])
    if not config_specs:
        raise ValueError("Manifest must define at least one entry in 'configs'.")

    config_paths: list[str] = []
    config_index = 0
    for spec in config_specs:
        spec_path = _resolve_path(str(spec["path"]))
        if not spec_path.exists():
            raise FileNotFoundError(f"Base config not found: {spec_path}")
        base_config = OmegaConf.load(spec_path)
        group_name = str(spec.get("name") or spec_path.stem)
        run_prefix = str(spec.get("run_name") or group_name)
        overrides = spec.get("overrides")

        for seed_offset in range(seeds_per_condition):
            seed = base_seed + seed_offset
            config = _clone_config(base_config)
            config = _apply_overrides(config, overrides)
            _set_seed(config, seed)
            _set_run_outputs(
                config,
                run_name=f"{run_prefix}_s{seed}",
                results_dir=sweep_dir / "results" / f"config_{config_index}",
                group_name=group_name,
                base_config_path=str(spec_path),
            )
            config["_sweep_config_id"] = f"config_{config_index}"

            config_path = sweep_dir / "configs" / f"{group_name}_s{seed}.yaml"
            OmegaConf.save(config, config_path)
            config_paths.append(str(config_path))
            config_index += 1

    metadata: dict[str, Any] = {
        "num_configs": len(config_paths),
        "seeds_per_condition": seeds_per_condition,
        "base_seed": base_seed,
        "config_groups": [
            {
                "name": str(spec.get("name") or Path(str(spec["path"])).stem),
                "path": str(_resolve_path(str(spec["path"]))),
                "overrides": OmegaConf.to_container(
                    spec.get("overrides"), resolve=True
                )
                if spec.get("overrides")
                else None,
            }
            for spec in config_specs
        ],
    }
    OmegaConf.save(OmegaConf.create(metadata), sweep_dir / "configs" / "metadata.yaml")
    return config_paths


def _submit(job_scripts: list[str], *, dry_run: bool) -> int:
    main_script = next(
        (path for path in job_scripts if path.endswith("submit_all_jobs.sh")), None
    )
    if main_script is None:
        raise RuntimeError("Main submission script was not generated.")

    if dry_run:
        print(f"DRY RUN: bash {main_script}")
        return 0

    result = subprocess.run(
        ["bash", main_script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = OmegaConf.load(args.manifest)
    output_dir = str(manifest.get("output_dir") or "output")
    run_name = str(
        manifest.get("outputs", {}).get("run_name")
        or manifest.get("run_name")
        or "config_repeat_sweep"
    )
    sweep_dir = _make_sweep_dir(output_dir, run_name)
    _save_manifest_copy(manifest, sweep_dir)
    config_paths = _generate_configs(manifest, sweep_dir)

    slurm_config = OmegaConf.create(
        OmegaConf.to_container(manifest.get("slurm_config", {}), resolve=True)
    )
    slurm_config["run_name"] = run_name
    job_manager = SweepJobManager(slurm_config)
    use_array_jobs = bool(
        manifest.get("experiment_settings", {}).get("use_array_jobs", True)
    )
    split_jobs = manifest.get("experiment_settings", {}).get("split_jobs")
    job_scripts = job_manager.create_job_scripts(
        config_paths=config_paths,
        output_dir=str(sweep_dir),
        split_jobs=split_jobs,
        use_array_jobs=use_array_jobs,
    )

    print(f"Created sweep directory: {sweep_dir}")
    print(f"Generated {len(config_paths)} seeded configs")
    print(f"Job scripts: {job_scripts}")

    if args.generate_only:
        return 0
    return _submit(job_scripts, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
