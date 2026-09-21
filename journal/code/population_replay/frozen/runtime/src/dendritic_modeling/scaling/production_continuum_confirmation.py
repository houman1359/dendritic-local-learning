"""Frozen fresh-seed confirmation with lazy TEST and independent numerical checks.

The qualified development trainer is reused unchanged. TEST is evaluated only
after its terminal checkpoint exists, with no outcome-dependent fit selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback

import mpmath as mp
import numpy as np
import torch

from . import production_continuum as pc


def read_freeze(path, expected_sha256):
    if pc.sha(path) != expected_sha256:
        raise ValueError("Confirmation freeze hash differs")
    freeze = json.loads(Path(path).read_text())
    if (
        not freeze.get("selection_frozen")
        or freeze.get("test_materialized") is not False
    ):
        raise ValueError("Require selection and forecasts frozen before TEST")
    if freeze.get("schema") != "production_continuum_confirmation_freeze_v1":
        raise ValueError("Unknown confirmation freeze")
    return freeze


def test_grid(points):
    """Called lazily after freeze verification and a saved terminal fit."""
    if points < 2:
        raise ValueError("At least two midpoint points")
    return torch.cat(
        (
            torch.zeros(1, dtype=torch.float64),
            (torch.arange(points, dtype=torch.float64) + 0.5) / points,
            torch.ones(1, dtype=torch.float64),
        )
    )[:, None]


def numpy_target(x, task):
    if task == "relu_spline_mismatch":
        return abs(x - 0.2) - 1.5 * abs(x - 0.5) + 0.75 * abs(x - 0.8)
    if task not in pc.TASKS:
        raise ValueError("Unknown target")
    a, b = (1.0, 2.0) if task == "continuum_narrow" else (0.01, 4.0)
    return x * np.log1p((b - a) / (x + a)) / (b - a)


def numpy_prediction(model, family, x):
    """Independent scalar circuit; also propagate a conservative rounding scale.

    The envelope is a diagnostic, not a rigorous interval certificate. It scales
    with cancellation and depth rather than hiding discrepancies behind MSE.
    """
    eps = np.finfo(np.float64).eps

    def to_np(t):
        return t.detach().cpu().numpy()

    if family in pc.FAMILIES[:2]:
        pre, head, bias = [to_np(p) for p in pc.mapped_parameters(model, family)]
        weight = np.where(pre > 20, pre, np.logaddexp(0, pre)).T
        current = x * weight
        feature = current / ((1 + current) + 1e-8)
        error = 16 * eps * (1 + np.abs(feature))
    else:
        feature, error = x, np.zeros_like(x)
        for layer in model.hidden:
            weight, bias = to_np(layer[0].weight), to_np(layer[0].bias)
            error = error @ abs(weight).T + 16 * eps * (weight.shape[1] + 2) * (
                abs(feature) @ abs(weight).T + abs(bias) + 1
            )
            feature = feature @ weight.T + bias
            feature = (
                np.tanh(feature)
                if family == "biased_tanh_shallow"
                else np.maximum(feature, 0)
            )
        head, bias = to_np(model.readout.weight), to_np(model.readout.bias)
    scale = abs(feature) @ abs(head).T + abs(bias) + 1
    error = error @ abs(head).T + 16 * eps * (head.shape[1] + 2) * scale
    return feature @ head.T + bias, error


def high_precision_check(model, family, task, x, dps=70, truth_evaluator=None):
    """Replay 33 declared test locations in high precision; no dense-grid certificate."""

    def scalar(v):
        return mp.mpf(float(v))

    with mp.workdps(dps):
        if family in pc.FAMILIES[:2]:
            pre, head, bias = [
                p.detach().flatten().tolist()
                for p in pc.mapped_parameters(model, family)
            ]
            gains = [scalar(v) if v > 20 else mp.log1p(mp.exp(scalar(v))) for v in pre]
            head = [scalar(v) for v in head]
            bias = scalar(bias[0])
        else:
            layers = []
            for layer in model.hidden:
                layers.append(
                    (
                        [
                            [scalar(v) for v in row]
                            for row in layer[0].weight.detach().tolist()
                        ],
                        [scalar(v) for v in layer[0].bias.detach().tolist()],
                    )
                )
            head = [scalar(v) for v in model.readout.weight.detach().flatten().tolist()]
            bias = scalar(model.readout.bias.detach()[0])
        errors, predictions = [], []
        for value in x.flatten().tolist():
            z = scalar(value)
            if family in pc.FAMILIES[:2]:
                feature = [(z * gain) / (1 + z * gain + scalar(1e-8)) for gain in gains]
            else:
                feature = [z]
                for weights, biases in layers:
                    preact = [
                        mp.fsum(w * f for w, f in zip(row, feature)) + b
                        for row, b in zip(weights, biases)
                    ]
                    feature = [
                        mp.tanh(v) if family == "biased_tanh_shallow" else max(v, 0)
                        for v in preact
                    ]
            prediction = mp.fsum(w * f for w, f in zip(head, feature)) + bias
            if truth_evaluator is not None:
                truth = truth_evaluator(z)
            elif task == "relu_spline_mismatch":
                truth = (
                    abs(z - scalar(0.2))
                    - scalar(1.5) * abs(z - scalar(0.5))
                    + scalar(0.75) * abs(z - scalar(0.8))
                )
            else:
                a, b = (
                    (scalar(1), scalar(2))
                    if task == "continuum_narrow"
                    else (scalar(0.01), scalar(4))
                )
                truth = z * mp.log((z + b) / (z + a)) / (b - a)
            predictions.append(float(prediction))
            errors.append(prediction - truth)
        return {
            "dps": dps,
            "points": len(errors),
            "mse_at_checked_points": float(
                mp.fsum(e * e for e in errors) / len(errors)
            ),
            "max_absolute_error_at_checked_points": float(max(abs(e) for e in errors)),
            "predictions": predictions,
            "scope": "High-precision finite-parameter circuit at33 fixed test locations; not a quadrature or interval bound on population error.",
        }


def evaluate_saved_fit(directory, task_row, freeze_path, freeze_sha256, density=None):
    directory = Path(directory)
    freeze = read_freeze(freeze_path, freeze_sha256)
    if task_row not in freeze["tasks"]:
        raise ValueError("Fit is not in frozen confirmation grid")
    if "density" in task_row:
        if density is None or density.receipt() != task_row["density"]:
            raise ValueError("Density evaluator differs from the frozen target")
    elif density is not None:
        raise ValueError("Unexpected density override for a fixed uniform target")
    destination = directory / "evaluation.json"
    if destination.exists():
        raise FileExistsError("Do not reevaluate or overwrite an existing TEST result")
    receipt = json.loads((directory / "receipt.json").read_text())
    spec = task_row["spec"]
    if density is not None and receipt["data"].get("density") != density.receipt():
        raise ValueError("Training target differs from the frozen evaluation density")
    if receipt["status"] != "completed" or receipt["spec"] != spec:
        raise ValueError("Completed frozen fit required before TEST")
    if receipt["state_sha256"] != pc.sha(directory / "states.pt"):
        raise ValueError("Checkpoint hash differs")
    result = {
        "status": "running",
        "task": task_row,
        "freeze_sha256": freeze_sha256,
        "training_receipt_sha256": pc.sha(directory / "receipt.json"),
        "checkpoint_sha256": receipt["state_sha256"],
        "test_materialized": False,
    }
    pc.dump(destination, result)
    started = time.perf_counter()
    try:
        model = pc.construct(spec["family"], spec["budget"], spec["seed"])
        states = torch.load(directory / "states.pt", weights_only=True)
        model.load_state_dict(states["terminal"])
        model.eval()
        x = test_grid(freeze["evaluation_rules"]["test_points"])
        data = pc.make_data(spec["task"])
        assert (
            torch.unique(torch.cat((x, data.train_x, data.validation_x))).numel()
            == len(x) + 4102
        )
        y = density.labels(x) if density is not None else pc.target(x, spec["task"])
        with torch.no_grad():
            prediction = torch.cat([model(chunk) for chunk in x.split(4096)])
        independent_y = (
            density.numpy_quadrature(x.numpy())
            if density is not None
            else numpy_target(x.numpy(), spec["task"])
        )
        independent, envelope = numpy_prediction(model, spec["family"], x.numpy())
        mse = float((prediction - y).square().mean())
        numpy_mse = float(np.mean((independent - independent_y) ** 2))
        if not all(
            np.isfinite(v).all()
            for v in (prediction.numpy(), independent, independent_y, envelope)
        ):
            raise FloatingPointError(
                "Nonfinite TEST circuit output or rounding envelope"
            )
        delta = np.abs(prediction.numpy() - independent)
        passed = bool(
            np.isfinite(independent).all()
            and np.isfinite(prediction.numpy()).all()
            and np.all(delta <= envelope)
            and np.allclose(y.numpy(), independent_y, rtol=5e-15, atol=3e-16)
        )
        result.update(
            test_materialized=True,
            test_points=len(x),
            test_mse=mse,
            test_max_absolute_error=float((prediction - y).abs().max()),
            numpy_test_mse=numpy_mse,
            numpy_prediction_max_difference=float(delta.max()),
            numpy_max_envelope_ratio=float(np.max(delta / envelope)),
            numpy_prediction_envelope_passed=passed,
            test_x_sha256=pc.tensor_hash(x),
            test_y_sha256=pc.tensor_hash(y),
            prediction_sha256=pc.tensor_hash(prediction),
            censored_at_interpretation_floor=mse
            <= freeze["evaluation_rules"]["interpretation_floor"],
        )
        if min(mse, numpy_mse) <= freeze["evaluation_rules"]["interpretation_floor"]:
            indices = np.linspace(0, len(x) - 1, 33, dtype=int)
            high = high_precision_check(
                model,
                spec["family"],
                spec["task"],
                x[indices],
                truth_evaluator=density.value_mp if density is not None else None,
            )
            high_delta = abs(
                np.array(high["predictions"])[:, None] - prediction.numpy()[indices]
            )
            high["indices"] = indices.tolist()
            high["torch_prediction_max_difference"] = float(high_delta.max())
            high["rounding_envelope_passed"] = bool(
                np.all(high_delta <= envelope[indices])
            )
            result["high_precision"] = high
            passed = passed and high["rounding_envelope_passed"]
        result["status"] = "completed" if passed else "numerical_review_required"
        # Preserve pointwise outputs so subsequent analyses never have to reopen TEST.
        np.savez_compressed(
            directory / "test_predictions.npz",
            x=x.numpy(),
            target=y.numpy(),
            torch_prediction=prediction.numpy(),
            numpy_prediction=independent,
            numpy_target=independent_y,
            rounding_envelope=envelope,
        )
        result["predictions_sha256"] = pc.sha(directory / "test_predictions.npz")
    except Exception as exc:
        result.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    result["elapsed_seconds"] = time.perf_counter() - started
    pc.dump(destination, result)
    return result


def prepare(freeze_directory, destination):
    freeze_directory, destination = Path(freeze_directory), Path(destination)
    freeze_path = freeze_directory / "freeze.json"
    frozen = read_freeze(freeze_path, pc.sha(freeze_path))
    parent = Path(frozen["development_campaign"])
    parent_manifest = json.loads((parent / "manifest.json").read_text())
    assert pc.sha(parent / "manifest.json") == frozen["development_manifest_sha256"]
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        parent / "source",
        destination / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # New evaluation code is added; every previously qualified training source stays identical.
    relative = (
        "source/src/dendritic_modeling/scaling/production_continuum_confirmation.py"
    )
    shutil.copy2(__file__, destination / relative)
    for source in parent_manifest["source_files"]:
        assert pc.sha(destination / source["path"]) == source["sha256"]
    shutil.copy2(freeze_path, destination / "freeze.json")
    shutil.copy2(freeze_directory / "base_plan.json", destination / "plan.json")
    manifest = {
        "schema": "production_continuum_confirmation_campaign_v1",
        "stage": "confirmation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_sha256": pc.sha(destination / "freeze.json"),
        "plan_sha256": pc.sha(destination / "plan.json"),
        "test_evaluation_authorized": True,
        "test_materialized_at_prepare": False,
        "source_files": parent_manifest["source_files"]
        + [{"path": relative, "sha256": pc.sha(__file__)}],
        "tasks": frozen["tasks"],
    }
    pc.dump(destination / "manifest.json", manifest)
    return manifest


def run(campaign, shard_index, shards):
    campaign = Path(campaign).resolve()
    if (
        Path(__file__).resolve()
        != campaign
        / "source/src/dendritic_modeling/scaling/production_continuum_confirmation.py"
    ):
        raise ValueError("Use the frozen campaign PYTHONPATH")
    manifest = json.loads((campaign / "manifest.json").read_text())
    frozen = read_freeze(campaign / "freeze.json", manifest["freeze_sha256"])
    assert manifest["tasks"] == frozen["tasks"] and 0 <= shard_index < shards
    assert pc.sha(campaign / "plan.json") == manifest["plan_sha256"]
    for source in manifest["source_files"]:
        assert pc.sha(campaign / source["path"]) == source["sha256"]
    plan = json.loads((campaign / "plan.json").read_text())
    pc.validate_plan(plan)
    torch.set_num_threads(1)
    results = []
    for task in manifest["tasks"]:
        if task["index"] % shards != shard_index:
            continue
        spec = pc.FitSpec(**task["spec"])
        data = pc.make_data(spec.task)
        directory = campaign / f"runs/{task['index']:04d}"
        receipt = pc.fit(spec, data, plan["training"], directory)
        evaluation = None
        if receipt["status"] == "completed":
            evaluation = evaluate_saved_fit(
                directory, task, campaign / "freeze.json", manifest["freeze_sha256"]
            )
        row = {
            "index": task["index"],
            "training_status": receipt["status"],
            "evaluation_status": evaluation["status"]
            if evaluation
            else "not_evaluated",
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    pc.dump(
        campaign / f"shard_{shard_index:02d}.json",
        {"manifest_sha256": pc.sha(campaign / "manifest.json"), "results": results},
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--freeze-directory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--campaign", type=Path, required=True)
    r.add_argument("--shard-index", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.freeze_directory, args.output_dir)
        print(
            json.dumps(
                {"tasks": len(result["tasks"]), "sources": len(result["source_files"])}
            )
        )
    else:
        results = run(args.campaign, args.shard_index, args.shards)
        if any(
            r["training_status"] != "completed" or r["evaluation_status"] != "completed"
            for r in results
        ):
            raise SystemExit(1)
