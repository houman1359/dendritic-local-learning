"""Execute the prospectively fixed study with sequential data-release barriers."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "source_manifest.json").read_text())
    for row in manifest["files"]:
        if hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Changed bound source {row['path']}")
    driver = root / "source/rank_study_campaign.py"
    logdir = root / "logs"
    logdir.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        DENDRITIC_LOG_DIR=str(logdir),
    )
    stages = []

    def run(stage, architecture=None, family=None):
        name = "_".join(filter(None, [stage, architecture, family]))
        command = [
            sys.executable,
            "-B",
            str(driver),
            "--root",
            str(root),
            "--stage",
            stage,
        ]
        if architecture:
            command += ["--architecture", architecture, "--family", family]
        start = time.time()
        with (logdir / (name + ".log")).open("x") as stream:
            result = subprocess.run(
                command, env=env, stdout=stream, stderr=subprocess.STDOUT
            )
        row = {
            "stage": stage,
            "architecture": architecture,
            "family": family,
            "start_unix": start,
            "elapsed_seconds": time.time() - start,
            "returncode": result.returncode,
            "command": command,
        }
        write(logdir / (name + ".json"), row)
        if result.returncode:
            raise RuntimeError(f"Stage failed: {name}; see its preserved log")
        return row

    start = time.time()
    try:
        stages.append(run("initialize"))
        config = json.loads((root / "config.json").read_text())
        for stage in [
            "development",
            "select",
            "bridge",
            "freeze",
            "confirmation",
            "complete",
        ]:
            if stage in ["development", "bridge", "confirmation"]:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [
                        pool.submit(run, stage, a, f)
                        for a, f in itertools.product(
                            config["architectures"], config["families"]
                        )
                    ]
                    stages.extend(future.result() for future in futures)
            else:
                stages.append(run(stage))
        status = "complete"
    except Exception:
        write(
            root / "orchestration_failure.json",
            {"traceback": traceback.format_exc(), "time_unix": time.time()},
        )
        raise
    finally:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        write(
            root / "orchestration_receipt.json",
            {
                "status": locals().get("status", "failed"),
                "start_unix": start,
                "elapsed_seconds": time.time() - start,
                "child_user_seconds": usage.ru_utime,
                "child_system_seconds": usage.ru_stime,
                "workers": args.workers,
                "stages": stages,
            },
        )


if __name__ == "__main__":
    main()
