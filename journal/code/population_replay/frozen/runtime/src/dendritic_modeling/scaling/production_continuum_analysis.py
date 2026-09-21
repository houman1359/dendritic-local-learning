"""Audit complete production continuum development and freeze horizon choices.

Only the already used TRAIN/VAL splits are reconstructed. No TEST exists in
this analysis. Selection is development-only, including the declared floor.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics

import torch

from . import production_continuum as pc


def analyze(campaign, output):
    campaign = Path(campaign).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    manifest = json.loads((campaign / "manifest.json").read_text())
    plan = json.loads((campaign / "plan.json").read_text())
    assert manifest["stage"] in {"development", "extended_development"}
    extended = manifest["stage"] == "extended_development"
    horizon = 3200 if extended else 400
    if extended:
        assert (
            pc.sha(campaign / "extension_plan.json")
            == manifest["extension_plan_sha256"]
        )
        parent = Path(manifest["parent_development"])
        assert pc.sha(parent / "manifest.json") == manifest["parent_manifest_sha256"]
    assert pc.sha(campaign / "plan.json") == manifest["plan_sha256"]
    assert [r["spec"] for r in manifest["tasks"]] == [
        asdict(s) | {"steps": horizon} for s in pc.development_specs(plan)
    ]
    for source in manifest["source_files"]:
        assert pc.sha(campaign / source["path"]) == source["sha256"]
    assert pc.sha(Path(pc.__file__)) == pc.sha(
        campaign / "source/src/dendritic_modeling/scaling/production_continuum.py"
    )
    data = {task: pc.make_data(task) for task in pc.TASKS}
    models, rows, bindings = {}, [], {}
    for task in manifest["tasks"]:
        index, spec = task["index"], task["spec"]
        directory = campaign / f"runs/{index:04d}"
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["status"] == "completed" and receipt["spec"] == spec
        assert receipt["settings"] == plan["training"]
        assert receipt["data"] == data[spec["task"]].receipt()
        assert receipt["state_sha256"] == pc.sha(directory / "states.pt")
        assert (
            receipt["model_report"]["total_parameters"]
            == receipt["terminal_model_report"]["trainable_parameters"]
            == spec["budget"]
        )
        assert (
            receipt["model_report"]["topology_sha256"]
            == receipt["terminal_model_report"]["topology_sha256"]
        )
        expected_steps = [-1, 0, 50, 100, 200, 400] + ([3200] if extended else [])
        assert [r["step"] for r in receipt["trace"]] == expected_steps
        prefix_exact = True
        if extended:
            previous_path = parent / f"runs/{index:04d}/receipt.json"
            previous = json.loads(previous_path.read_text())
            assert previous["spec"] == spec | {"steps": 400}
            prefix_exact = receipt["trace"][:-1] == previous["trace"]
            bindings[str(previous_path)] = pc.sha(previous_path)
        assert (
            receipt["closure_calls"] * 2049
            == receipt["exposures"]["optimization_train_examples"]
        )
        assert (
            receipt["exposures"]["measurement_train_examples"]
            == len(expected_steps) * 2049
        )
        assert receipt["exposures"]["readout_initialization_train_examples"] == (
            2049 if spec["recipe"] == pc.RECIPES[1] else 0
        )
        key = (spec["family"], spec["budget"])
        if key not in models:
            models[key] = pc.construct(*key, spec["seed"])
        model = models[key]
        states = torch.load(directory / "states.pt", weights_only=True)
        model.load_state_dict(states["terminal"])
        assert all(
            torch.isfinite(v).all()
            for v in states["terminal"].values()
            if v.is_floating_point()
        )
        with torch.no_grad():
            residual = (
                model(data[spec["task"]].validation_x) - data[spec["task"]].validation_y
            )
            replay = float(residual.square().mean())
        recorded = receipt["terminal_validation_mse"]
        bindings[str(directory / "receipt.json")] = pc.sha(directory / "receipt.json")
        rows.append(
            {
                "index": index,
                **spec,
                "validation_mse": recorded,
                "replay_validation_mse": replay,
                "replay_absolute_difference": abs(recorded - replay),
                "replay_exact": replay == recorded,
                "replay_within_tolerance": math.isclose(
                    replay, recorded, rel_tol=1e-8, abs_tol=1e-26
                ),
                "initial_train_mse": receipt["trace"][0]["train_mse"],
                "post_readout_train_mse": receipt["trace"][1]["train_mse"],
                "terminal_train_mse": receipt["terminal_train_mse"],
                "hidden_parameter_movement": receipt["hidden_parameter_movement"],
                "readout_max_abs": receipt.get("least_squares", {}).get(
                    "readout_max_abs"
                ),
                "seconds": receipt["elapsed_seconds"],
                "closure_calls": receipt["closure_calls"],
                "parent_prefix_exact": prefix_exact,
            }
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            row["task"], row["family"], row["budget"], row["recipe"], row["lr"]
        ].append(row)
    means = []
    for (task, family, budget, recipe, lr), values in sorted(grouped.items()):
        assert sorted(v["seed"] for v in values) == [7103, 7109]
        means.append(
            {
                "task": task,
                "family": family,
                "budget": budget,
                "recipe": recipe,
                "lr": lr,
                "mean_validation_mse": statistics.mean(
                    r["validation_mse"] for r in values
                ),
                "individual_mse": [r["validation_mse"] for r in values],
                "indices": [r["index"] for r in values],
                "zero_hidden_movement_seeds": sum(
                    r["hidden_parameter_movement"] == 0 for r in values
                ),
            }
        )
    cells = sorted({(r["task"], r["family"], r["budget"]) for r in means})
    selected = []
    for key in cells:
        choices = [r for r in means if (r["task"], r["family"], r["budget"]) == key]
        assert len(choices) == 4
        selected.append(
            min(
                choices,
                key=lambda r: (
                    max(r["mean_validation_mse"], 1e-24),
                    pc.RECIPES.index(r["recipe"]),
                    r["lr"],
                ),
            )
        )
    by = {
        (r["task"], r["family"], r["budget"], r["seed"], r["recipe"], r["lr"]): r
        for r in rows
    }
    pairs = []
    for r in rows:
        if r["family"] == "production_shunt":
            other = by[
                r["task"],
                "ordinary_rational",
                r["budget"],
                r["seed"],
                r["recipe"],
                r["lr"],
            ]
            pairs.append(
                {
                    "production_index": r["index"],
                    "ordinary_index": other["index"],
                    "endpoint_exact": r["validation_mse"] == other["validation_mse"],
                    "closure_count_exact": r["closure_calls"] == other["closure_calls"],
                }
            )
    assert (
        len(rows) == 1296
        and len(means) == 648
        and len(selected) == 162
        and len(pairs) == 288
    )
    result = {
        "status": "passed"
        if all(r["replay_within_tolerance"] and r["parent_prefix_exact"] for r in rows)
        else "replay_review_required",
        "scope": "All development records and CPU checkpoint predictions; reused validation selects and reports. No TEST or independent confirmation, no exponent estimate.",
        "training_steps": horizon,
        "exact_parent_prefix_count": sum(r["parent_prefix_exact"] for r in rows)
        if extended
        else None,
        "manifest_sha256": pc.sha(campaign / "manifest.json"),
        "analysis_source_sha256": pc.sha(__file__),
        "training_source_sha256": pc.sha(pc.__file__),
        "source_files_verified": len(manifest["source_files"]),
        "input_sha256": bindings,
        "rows": rows,
        "candidate_means": means,
        "selected": selected,
        "production_ordinary_pairs": pairs,
        "replay_tolerance": {
            "relative": 1e-8,
            "absolute_mse": 1e-26,
            "interpretation_floor": 1e-24,
        },
        "replay_exact_count": sum(r["replay_exact"] for r in rows),
        "replay_within_tolerance_count": sum(
            r["replay_within_tolerance"] for r in rows
        ),
        "maximum_replay_mse_difference": max(
            r["replay_absolute_difference"] for r in rows
        ),
        "cpu_fit_hours": sum(r["seconds"] for r in rows) / 3600,
    }
    output.mkdir(parents=True)
    pc.dump(output / "audit.json", result)
    pc.dump(
        output
        / (
            "selected_for_confirmation.json"
            if extended
            else "selected_for_horizon.json"
        ),
        {
            "scope": "Choices for the fixed3200-step confirmation; predictions and qualified lazy-TEST evaluation still required before release."
            if extended
            else "Choices frozen for the prescribed 800-step horizon check; not a final confirmation release.",
            "development_manifest_sha256": result["manifest_sha256"],
            "audit_sha256": pc.sha(output / "audit.json"),
            "selected": selected,
        },
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "replay_exact_count",
                    "replay_within_tolerance_count",
                    "maximum_replay_mse_difference",
                    "cpu_fit_hours",
                )
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
