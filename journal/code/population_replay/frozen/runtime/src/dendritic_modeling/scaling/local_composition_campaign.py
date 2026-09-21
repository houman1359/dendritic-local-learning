"""Frozen development-only local-composition qualification and calibration."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from dendritic_modeling.scaling import local_composition as lc
from dendritic_modeling.scaling.diagnostics import TrainingDiagnostics
from dendritic_modeling.scaling.production_continuum import dump, sha


def task_grid(plan, stage):
    if stage not in {"profile", "calibration"}:
        raise ValueError(
            "Only qualified profile and optimizer calibration are executable"
        )
    tasks = []
    seeds = [701] if stage == "profile" else plan["development_model_seeds"]
    rates = [0.001] if stage == "profile" else plan["training"]["learning_rates"]
    for arm in lc.catalog():
        for budget in [8192, 131072]:
            for condition in plan["task_conditions"]:
                for seed in seeds:
                    for rate in rates:
                        tasks.append(
                            {
                                "index": len(tasks),
                                "arm": arm,
                                "budget": budget,
                                "condition": condition,
                                "target_seed": 51001,
                                "model_seed": seed,
                                "learning_rate": rate,
                                "steps": 40 if stage == "profile" else 1000,
                            }
                        )
    assert len(tasks) == (120 if stage == "profile" else 720)
    return tasks


def prepare(plan_path, output, stage, qualification, frozen_data=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if stage == "calibration" and frozen_data is None:
        raise ValueError("Calibration requires the byte-exact exported profile data")
    plan = json.loads(Path(plan_path).read_text())
    expected = {a["id"] for a in plan["architectures"]}
    assert (
        expected == {a["id"] for a in lc.catalog()} and plan["arms_per_condition"] == 20
    )
    q = json.loads(Path(qualification).read_text())
    if q["status"] != "passed" or q["skipped"] != 0:
        raise ValueError("CPU and allocated-CUDA qualification must pass without skips")
    tasks = task_grid(plan, stage)
    output.mkdir(parents=True)
    for d in ("runs", "logs", "qualification"):
        (output / d).mkdir()
    shutil.copy2(plan_path, output / "plan.json")
    shutil.copy2(qualification, output / "qualification/receipt.json")
    source = Path(__file__).resolve().parents[1]
    dest = output / "source/src/dendritic_modeling"
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    sources = [
        {"path": str(p.relative_to(output)), "sha256": sha(p)}
        for p in sorted(dest.rglob("*"))
        if p.is_file()
    ]
    # Bind qualification to the exact implementation copied into the campaign.
    for name, digest in q["qualified_source_sha256"].items():
        if sha(output / "source" / name) != digest:
            raise ValueError("Qualified source changed before preparation")
    manifest = {
        "schema": "local_composition_campaign_v1",
        "stage": stage,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": tasks,
        "plan_sha256": sha(output / "plan.json"),
        "source_files": sources,
        "qualification_sha256": sha(output / "qualification/receipt.json"),
        "test_authorized": False,
        "test_materialized": False,
        "training": plan["training"],
        "data": plan["data"],
        "execution_amendments": [
            "Paired graph initialization uses actual flat14 production factory parameters mapped by canonical node to both trees, preserving every parameter scalar. This fixes previously unspecified cross-graph initialization.",
            "Profile uses fixed LR0.001, one model/target seed and40 steps. No optimization or scaling claim.",
            "Independent unrolled arithmetic may differ at floating-point rounding; qualify outputs and every gradient before fitting, do not require chaotic optimization trajectories to be bitwise equal.",
        ],
    }
    if frozen_data is not None:
        frozen_data = Path(frozen_data).resolve()
        export = json.loads((frozen_data / "manifest.json").read_text())
        assert export["status"] == "passed"
        shutil.copytree(frozen_data, output / "datasets")
        manifest["dataset_export_sha256"] = sha(output / "datasets/manifest.json")
        manifest["datasets"] = export["datasets"]
        for row in manifest["datasets"]:
            for name in ("arrays", "receipt"):
                assert sha(output / "datasets" / row[name]) == row[name + "_sha256"]
            original = lc.load_frozen_data(
                output / "datasets" / row["arrays"],
                output / "datasets" / row["receipt"],
            )
            assert all(
                sha(Path(r["path"])) == r["sha256"]
                for r in row["matched_profile_receipts"]
            )
            assert all(
                json.loads(Path(r["path"]).read_text()) == original.receipt()
                for r in row["matched_profile_receipts"]
            )
            assert original.identity == {
                "condition": row["condition"],
                "target_seed": row["target_seed"],
                "data_seed": plan["data"]["development_data_seed"],
            }
            assert (
                len(original.train_x) == plan["data"]["train_size"]
                and len(original.validation_x) == plan["data"]["validation_size"]
            )
        assert {(r["condition"], r["target_seed"]) for r in manifest["datasets"]} == {
            (t["condition"], t["target_seed"]) for t in tasks
        }
        manifest["execution_amendments"].append(
            "Calibration loads byte-exact profile TRAIN/VAL arrays verified against all120 original receipts. This prevents cross-host BLAS rounding from changing labels or normalization; original functions, splits and data exposure are preserved."
        )
    dump(output / "manifest.json", manifest)
    return {
        "campaign": str(output),
        "manifest_sha256": sha(output / "manifest.json"),
        "tasks": len(tasks),
    }


def verify_campaign(campaign):
    manifest = json.loads((campaign / "manifest.json").read_text())
    assert sha(campaign / "plan.json") == manifest["plan_sha256"]
    assert (
        sha(campaign / "qualification/receipt.json") == manifest["qualification_sha256"]
    )
    assert not manifest["test_authorized"] and not manifest["test_materialized"]
    for row in manifest["source_files"]:
        assert sha(campaign / row["path"]) == row["sha256"], row["path"]
    if "datasets" in manifest:
        assert (
            sha(campaign / "datasets/manifest.json")
            == manifest["dataset_export_sha256"]
        )
        for row in manifest["datasets"]:
            for name in ("arrays", "receipt"):
                assert sha(campaign / "datasets" / row[name]) == row[name + "_sha256"]
    return manifest


def campaign_data(campaign, manifest, task):
    if "datasets" in manifest:
        row = next(
            r
            for r in manifest["datasets"]
            if r["condition"] == task["condition"]
            and r["target_seed"] == task["target_seed"]
        )
        return lc.load_frozen_data(
            campaign / "datasets" / row["arrays"],
            campaign / "datasets" / row["receipt"],
        )
    return lc.make_data(
        task["condition"],
        task["target_seed"],
        manifest["data"]["development_data_seed"],
        manifest["data"]["train_size"],
        manifest["data"]["validation_size"],
    )


@torch.no_grad()
def evaluate(model, x, y, std, batch_size=128):
    predictions = torch.cat(
        [model(chunk).detach().cpu() for chunk in x.split(batch_size)]
    )
    mse = float((predictions.double() - y.detach().cpu().double()).square().mean())
    if not np.isfinite(mse):
        raise FloatingPointError("Nonfinite development evaluation")
    return {"normalized_mse": mse, "raw_mse": mse * std**2}, predictions


def fit(task, data, settings, output, *, device="cuda"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, inventory = lc.construct(task["arm"], task["budget"], task["model_seed"])
    dump(output / "inventory.json", inventory)
    dump(output / "data.json", data.receipt())
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=task["learning_rate"], weight_decay=0
    )
    x, y, vx, vy = [
        v.to(device=device, dtype=torch.float32)
        for v in (data.train_x, data.train_y, data.validation_x, data.validation_y)
    ]
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(initial, output / "initial.pt")
    diagnostic = TrainingDiagnostics(
        model,
        output / "diagnostics.jsonl",
        every=settings["diagnostics_every"],
        steps=task["steps"],
    )
    generator = torch.Generator().manual_seed(task["model_seed"] + 200003)
    batch_size = settings["batch_size"]
    permutation, cursor, exposure, step = None, 0, 0, 0
    seen = torch.zeros(len(x), dtype=torch.bool)
    log = []
    checkpoint = output / "checkpoint.pt"
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.monotonic()
    receipt = {
        "task": task,
        "test_materialized": False,
        "data_sha256": sha(output / "data.json"),
        "inventory_sha256": sha(output / "inventory.json"),
        "initial_sha256": sha(output / "initial.pt"),
    }
    try:
        metrics, _ = evaluate(model, vx, vy, data.std)
        log.append({"step": 0, "validation": metrics})
        for step in range(1, task["steps"] + 1):
            if permutation is None or cursor == len(x):
                permutation, cursor = torch.randperm(len(x), generator=generator), 0
            indices = permutation[cursor : cursor + batch_size]
            cursor += len(indices)
            seen[indices] = True
            exposure += len(indices)
            diagnostic.begin_step(step)
            optimizer.zero_grad(set_to_none=True)
            loss = (
                (model(x[indices.to(device)]) - y[indices.to(device)]).square().mean()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite TRAIN objective")
            loss.backward()
            # Infinity leaves gradients unscaled; it only checks the norm is finite.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf"), error_if_nonfinite=True
            )
            diagnostic.finish_step()
            optimizer.step()
            if step % settings["eval_every"] == 0 or step == task["steps"]:
                metrics, _ = evaluate(model, vx, vy, data.std)
                log.append(
                    {
                        "step": step,
                        "train_batch_nmse": float(loss.detach()),
                        "validation": metrics,
                    }
                )
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite terminal parameters")
        metrics, predictions = evaluate(model, vx, vy, data.std)
        # Training data are generated in float64 and cast only at model input.
        np.save(output / "validation_predictions.npy", predictions.numpy())
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "step": step,
                "batch_rng": generator.get_state(),
                "permutation": permutation,
                "cursor": cursor,
            },
            checkpoint,
        )
        receipt.update(
            status="completed",
            validation=metrics,
            checkpoint_sha256=sha(checkpoint),
            prediction_sha256=sha(output / "validation_predictions.npy"),
        )
    except Exception as error:
        diagnostic.record_failure(error, step)
        receipt.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        torch.save(
            {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "step": step,
            },
            output / "failed_state.pt",
        )
    finally:
        diagnostic.close()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    receipt.update(
        completed_steps=step,
        fit_seconds=time.monotonic() - start,
        train_exposure=exposure,
        unique_train_examples=int(seen.sum()),
        device=device,
        peak_allocated_bytes=torch.cuda.max_memory_allocated()
        if device.startswith("cuda")
        else None,
        diagnostic_sha256=sha(output / "diagnostics.jsonl"),
    )
    dump(output / "metrics.json", log)
    receipt["metrics_sha256"] = sha(output / "metrics.json")
    dump(output / "receipt.json", receipt)
    return receipt


def run(campaign, shard, shards):
    campaign = Path(campaign).resolve()
    manifest = verify_campaign(campaign)
    frozen_root = campaign / "source/src"
    if not Path(__file__).resolve().is_relative_to(frozen_root):
        raise ValueError("Workers must execute the frozen source snapshot")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("This campaign requires one allocated CUDA GPU per worker")
    environment = {
        "python": sys.executable,
        "torch_version": torch.__version__,
        "torch_path": torch.__file__,
        "numpy_version": np.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "manifest_sha256": sha(campaign / "manifest.json"),
        "shard": shard,
        "shards": shards,
    }
    dump(campaign / f"worker_{shard:02d}.json", environment)
    data_cache = {}
    for task in manifest["tasks"]:
        if task["index"] % shards != shard:
            continue
        destination = campaign / "runs" / f"{task['index']:04d}"
        if destination.exists():
            raise FileExistsError(
                "Worker refuses to overwrite or silently retry a prior fit"
            )
        key = (task["condition"], task["target_seed"])
        if key not in data_cache:
            data_cache[key] = campaign_data(campaign, manifest, task)
        row = fit(task, data_cache[key], manifest["training"], destination)
        print(
            json.dumps(
                {
                    "index": task["index"],
                    "status": row["status"],
                    "seconds": row["fit_seconds"],
                }
            ),
            flush=True,
        )


def audit(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = verify_campaign(campaign)
    rows, errors = [], []
    for task in manifest["tasks"]:
        directory = campaign / "runs" / f"{task['index']:04d}"
        path = directory / "receipt.json"
        if not path.exists():
            rows.append({"index": task["index"], "status": "missing"})
            continue
        r = json.loads(path.read_text())
        try:
            assert r["task"] == task and not r["test_materialized"]
            for name, field in [
                ("data.json", "data_sha256"),
                ("inventory.json", "inventory_sha256"),
                ("initial.pt", "initial_sha256"),
                ("metrics.json", "metrics_sha256"),
                ("diagnostics.jsonl", "diagnostic_sha256"),
            ]:
                assert sha(directory / name) == r[field]
            if r["status"] == "completed":
                assert r["completed_steps"] == task["steps"]
                assert (
                    r["train_exposure"]
                    == task["steps"] * manifest["training"]["batch_size"]
                )
                assert sha(directory / "checkpoint.pt") == r["checkpoint_sha256"]
                assert (
                    sha(directory / "validation_predictions.npy")
                    == r["prediction_sha256"]
                )
                data = campaign_data(campaign, manifest, task)
                assert data.receipt() == json.loads(
                    (directory / "data.json").read_text()
                )
                predictions = np.load(directory / "validation_predictions.npy")
                mse = float(
                    np.mean(
                        (
                            predictions.astype(np.float64)
                            - data.validation_y.float().numpy().astype(np.float64)
                        )
                        ** 2
                    )
                )
                assert np.isclose(
                    mse, r["validation"]["normalized_mse"], rtol=2e-14, atol=1e-15
                )
                inventory = json.loads((directory / "inventory.json").read_text())
                assert inventory["actual_parameters"] == lc.parameter_count(
                    task["arm"], inventory["width"]
                )
                assert (
                    abs(inventory["actual_parameters"] - task["budget"])
                    / task["budget"]
                    <= 0.02
                )
                diagnostics = [
                    json.loads(line)
                    for line in (directory / "diagnostics.jsonl")
                    .read_text()
                    .splitlines()
                ]
                cadence = sorted(
                    {
                        1,
                        task["steps"],
                        *range(
                            manifest["training"]["diagnostics_every"],
                            task["steps"] + 1,
                            manifest["training"]["diagnostics_every"],
                        ),
                    }
                )
                assert [d["step"] for d in diagnostics] == cadence
                assert all(
                    g["nonfinite_gradient_elements"]
                    == g["nonfinite_parameter_elements"]
                    == 0
                    for d in diagnostics
                    for g in d["parameter_groups"].values()
                )
                r["actual_parameters"] = inventory["actual_parameters"]
                r["width"] = inventory["width"]
        except Exception as error:
            errors.append({"index": task["index"], "error": repr(error)})
        rows.append(r)
    counts = dict(Counter(r["status"] for r in rows))
    report = {
        "status": "passed"
        if not errors and counts == {"completed": len(manifest["tasks"])}
        else "incomplete_or_failed",
        "scope": "Source, artifacts, data and saved validation predictions reconciled; no checkpoint forward replay or TEST.",
        "manifest_sha256": sha(campaign / "manifest.json"),
        "errors": errors,
        "status_counts": counts,
        "gpu_fit_hours": sum(r.get("fit_seconds", 0) for r in rows) / 3600,
        "rows": rows,
    }
    output.mkdir(parents=True)
    dump(output / "audit.json", report)
    return {k: v for k, v in report.items() if k != "rows"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--plan", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage", choices=["profile", "calibration"], required=True)
    p.add_argument("--qualification", required=True)
    p.add_argument("--frozen-data")
    p = commands.add_parser("run")
    p.add_argument("--campaign", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p = commands.add_parser("audit")
    p.add_argument("--campaign", required=True)
    p.add_argument("--output", required=True)
    a = parser.parse_args()
    if a.command == "prepare":
        result = prepare(a.plan, a.output, a.stage, a.qualification, a.frozen_data)
    elif a.command == "run":
        result = run(a.campaign, a.shard, a.shards)
    else:
        result = audit(a.campaign, a.output)
    if result is not None:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
