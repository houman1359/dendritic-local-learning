"""Audit frozen confirmation using saved predictions; never reevaluate TEST."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from . import production_continuum as pc
from .production_continuum_confirmation import read_freeze


def analyze(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((campaign / "manifest.json").read_text())
    density_campaign = manifest["stage"] == "fresh_density_confirmation"
    expected_count = 450 if density_campaign else 576
    frozen = read_freeze(campaign / "freeze.json", manifest["freeze_sha256"])
    assert (
        manifest["tasks"] == frozen["tasks"]
        and len(manifest["tasks"]) == expected_count
    )
    assert pc.sha(campaign / "plan.json") == manifest["plan_sha256"]
    plan = json.loads((campaign / "plan.json").read_text())
    for source in manifest["source_files"]:
        assert pc.sha(campaign / source["path"]) == source["sha256"]
    rows, bindings = [], {}
    for task in manifest["tasks"]:
        spec = task["spec"]
        directory = campaign / f"runs/{task['index']:04d}"
        row = {
            "index": task["index"],
            **spec,
            "status": "missing",
            "density_seed": task.get("density_seed"),
        }
        receipt_path, evaluation_path = (
            directory / "receipt.json",
            directory / "evaluation.json",
        )
        if not receipt_path.exists():
            rows.append(row)
            continue
        receipt = json.loads(receipt_path.read_text())
        bindings[str(receipt_path)] = pc.sha(receipt_path)
        assert receipt["spec"] == spec and receipt["settings"] == plan["training"]
        assert receipt["data"]["test_materialized"] is False
        if density_campaign:
            assert receipt["data"]["density"] == task["density"]
        if receipt["status"] != "completed":
            row.update(status="training_" + receipt["status"])
            rows.append(row)
            continue
        assert spec["steps"] == receipt["trace"][-1]["step"] == 3200
        assert (
            receipt["model_report"]["total_parameters"]
            == receipt["terminal_model_report"]["total_parameters"]
            == spec["budget"]
        )
        assert (
            receipt["model_report"]["topology_sha256"]
            == receipt["terminal_model_report"]["topology_sha256"]
        )
        assert (
            receipt["exposures"]["optimization_train_examples"]
            == 2049 * receipt["closure_calls"]
        )
        assert receipt["state_sha256"] == pc.sha(directory / "states.pt")
        row.update(
            validation_mse=receipt["terminal_validation_mse"],
            training_seconds=receipt["elapsed_seconds"],
            hidden_parameter_movement=receipt["hidden_parameter_movement"],
            closure_calls=receipt["closure_calls"],
            readout_max_abs=receipt.get("least_squares", {}).get("readout_max_abs"),
        )
        if not evaluation_path.exists():
            row.update(status="evaluation_missing")
            rows.append(row)
            continue
        evaluation = json.loads(evaluation_path.read_text())
        bindings[str(evaluation_path)] = pc.sha(evaluation_path)
        assert (
            evaluation["task"] == task
            and evaluation["freeze_sha256"] == manifest["freeze_sha256"]
        )
        assert evaluation["training_receipt_sha256"] == pc.sha(receipt_path)
        assert evaluation["checkpoint_sha256"] == receipt["state_sha256"]
        if evaluation["status"] not in ["completed", "numerical_review_required"]:
            row.update(status="evaluation_" + evaluation["status"])
            rows.append(row)
            continue
        assert (
            evaluation["test_materialized"] is True
            and evaluation["test_points"] == 32770
        )
        predictions_path = directory / "test_predictions.npz"
        assert pc.sha(predictions_path) == evaluation["predictions_sha256"]
        with np.load(predictions_path) as predictions:
            for array in predictions.values():
                assert array.shape == (32770, 1) and np.isfinite(array).all()
            actual = float(
                np.mean((predictions["torch_prediction"] - predictions["target"]) ** 2)
            )
            independent = float(
                np.mean(
                    (predictions["numpy_prediction"] - predictions["numpy_target"]) ** 2
                )
            )
            assert math.isclose(
                actual, evaluation["test_mse"], rel_tol=2e-15, abs_tol=0
            )
            assert independent == evaluation["numpy_test_mse"]
            assert (
                float(
                    np.max(abs(predictions["torch_prediction"] - predictions["target"]))
                )
                == evaluation["test_max_absolute_error"]
            )
            assert (
                pc.tensor_hash(torch.from_numpy(predictions["x"]))
                == evaluation["test_x_sha256"]
            )
        if evaluation["censored_at_interpretation_floor"]:
            assert evaluation["high_precision"]["points"] == 33
        row.update(
            status=evaluation["status"],
            test_mse=actual,
            numpy_test_mse=independent,
            test_max_absolute_error=evaluation["test_max_absolute_error"],
            prediction_sha256=evaluation["prediction_sha256"],
            censored_at_interpretation_floor=evaluation[
                "censored_at_interpretation_floor"
            ],
            evaluation_seconds=evaluation["elapsed_seconds"],
            high_precision=evaluation.get("high_precision"),
            numpy_max_envelope_ratio=evaluation["numpy_max_envelope_ratio"],
        )
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"], row["family"], row["budget"], row["density_seed"]].append(
            row
        )
    means = []
    for (task, family, budget, density_seed), values in sorted(grouped.items()):
        assert sorted(r["seed"] for r in values) == [7207, 7211, 7213]
        complete = all(r["status"] == "completed" for r in values)
        means.append(
            {
                "task": task,
                "family": family,
                "budget": budget,
                "density_seed": density_seed,
                "complete_three_seeds": complete,
                "status_counts": dict(Counter(r["status"] for r in values)),
                "test_mse": statistics.mean(r["test_mse"] for r in values)
                if complete
                else None,
                "individual_mse": {str(r["seed"]): r.get("test_mse") for r in values},
                "indices": [r["index"] for r in values],
                "withheld_size": budget in [481, 673],
            }
        )
    assert len(means) == (150 if density_campaign else 192)
    pairs = []
    by = {
        (r["task"], r["family"], r["budget"], r["seed"], r["density_seed"]): r
        for r in rows
    }
    for r in rows:
        if r["family"] == "production_shunt":
            other = by[
                r["task"],
                "ordinary_rational",
                r["budget"],
                r["seed"],
                r["density_seed"],
            ]
            pairs.append(
                {
                    "indices": [r["index"], other["index"]],
                    "both_complete": r["status"] == other["status"] == "completed",
                    "predictions_identical": r.get("prediction_sha256") is not None
                    and r.get("prediction_sha256") == other.get("prediction_sha256"),
                    "closure_counts_identical": r.get("closure_calls")
                    == other.get("closure_calls"),
                }
            )
    assert len(pairs) == (90 if density_campaign else 126)
    by_mean = {
        (r["task"], r["family"], r["budget"], r["density_seed"]): r for r in means
    }
    scored = []
    floor_predictions = []
    for forecast in frozen.get("size_forecasts", []):
        for p in [481, 673]:
            actual = by_mean[forecast["task"], forecast["family"], p, None]["test_mse"]
            if forecast["floor_category_prediction"] != "no_floor_prediction":
                floor_predictions.append(
                    {
                        "task": forecast["task"],
                        "family": forecast["family"],
                        "budget": p,
                        "test_mse": actual,
                        "predicted_censored": True,
                        "prediction_met": actual is not None and actual <= 1e-24,
                    }
                )
            for model in forecast["models"]:
                prediction = model["predictions"][str(p)]
                scored.append(
                    {
                        "task": forecast["task"],
                        "family": forecast["family"],
                        "budget": p,
                        "model": model["name"],
                        "prediction": prediction,
                        "test_mse": actual,
                        "absolute_log10_error": abs(
                            math.log10(prediction) - math.log10(actual)
                        )
                        if actual is not None and actual > 1e-24 and prediction > 0
                        else None,
                    }
                )
    counts = dict(Counter(r["status"] for r in rows))
    result = {
        "status": "passed"
        if counts == {"completed": expected_count}
        and all(
            p["predictions_identical"] and p["closure_counts_identical"] for p in pairs
        )
        else "review_required",
        "scope": "Three reserved positive-density functions, frozen optimizer choices and fresh model seeds; saved TEST predictions reconciled without reevaluation. Generalization beyond this density family and convergence remain unestablished."
        if density_campaign
        else "Frozen same-target fresh-seed/withheld-size confirmation; saved TEST predictions independently reconciled, no TEST reevaluation. Target-function generalization and converged learning remain unestablished.",
        "manifest_sha256": pc.sha(campaign / "manifest.json"),
        "freeze_sha256": manifest["freeze_sha256"],
        "analysis_source_sha256": pc.sha(__file__),
        "input_sha256": bindings,
        "status_counts": counts,
        "rows": rows,
        "means": means,
        "production_ordinary_pairs": pairs,
        "forecast_scores": scored,
        "floor_predictions": floor_predictions,
        "cpu_fit_hours": sum(r.get("training_seconds", 0) for r in rows) / 3600,
        "cpu_evaluation_hours": sum(r.get("evaluation_seconds", 0) for r in rows)
        / 3600,
    }
    output.mkdir(parents=True)
    pc.dump(output / "audit.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "status_counts",
                    "cpu_fit_hours",
                    "cpu_evaluation_hours",
                ]
            }
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.campaign, args.output_dir)
