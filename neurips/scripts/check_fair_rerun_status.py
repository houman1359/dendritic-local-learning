#!/usr/bin/env python3
"""Report progress for fair LocalCA rerun sweeps."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SWEEP_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
)

PREFIXES = [
    "sweep_neurips_additive_dynamics_fairness_audit",
    "sweep_neurips_localca_core_fair_tuning",
    "sweep_neurips_claimA_ei_grid_pilot",
    "sweep_neurips_claimA_ei_grid_info_shunting_pilot",
    "sweep_neurips_claimB_morphology_ei_pilot",
]


def _latest_dir(root: Path, prefix: str) -> Path | None:
    matches = sorted(root.glob(f"{prefix}_*"), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def _progress(sweep_dir: Path) -> tuple[int, int, int]:
    total = len(list((sweep_dir / "configs").glob("unified_config_*.yaml")))
    started = 0
    done = 0
    for idx in range(total):
        run_dir = sweep_dir / "results" / f"config_{idx}"
        if (run_dir / "config.json").exists():
            started += 1
        if (run_dir / "performance" / "final.json").exists():
            done += 1
    return total, started, done


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    args = parser.parse_args()

    print(f"Sweep root: {args.sweep_root}")
    print("")

    for prefix in PREFIXES:
        latest = _latest_dir(args.sweep_root, prefix)
        if latest is None:
            print(f"{prefix}: not found")
            continue

        total, started, done = _progress(latest)
        print(f"{prefix}")
        print(f"  dir: {latest}")
        print(f"  progress: done={done}/{total}, started={started}/{total}")
        print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
