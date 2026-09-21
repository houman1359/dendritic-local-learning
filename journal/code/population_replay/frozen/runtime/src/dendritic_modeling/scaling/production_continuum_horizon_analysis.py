"""Check all prescribed continuum horizon fits, checkpoints and update prefixes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import torch

from . import production_continuum as pc


def analyze(campaign, output):
    campaign, output = Path(campaign).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(1)
    manifest = json.loads((campaign / "manifest.json").read_text())
    assert manifest["stage"] == "horizon" and len(manifest["tasks"]) == 162
    development = Path(manifest["development_campaign"])
    assert (
        pc.sha(development / "manifest.json") == manifest["development_manifest_sha256"]
    )
    plan = json.loads((campaign / "plan.json").read_text())
    assert pc.sha(campaign / "plan.json") == manifest["plan_sha256"]
    assert (
        pc.sha(campaign / "selection_audit.json") == manifest["selection_audit_sha256"]
    )
    for source in manifest["source_files"]:
        assert pc.sha(campaign / source["path"]) == source["sha256"]
    assert pc.sha(pc.__file__) == pc.sha(
        campaign / "source/src/dendritic_modeling/scaling/production_continuum.py"
    )
    datasets = {task: pc.make_data(task) for task in pc.TASKS}
    models, rows, bindings = {}, [], {}
    for task in manifest["tasks"]:
        spec = task["spec"]
        assert spec["steps"] == 800
        directory = campaign / f"runs/{task['index']:04d}"
        prior_path = development / f"runs/{task['development_index']:04d}/receipt.json"
        receipt_path = directory / "receipt.json"
        prior = json.loads(prior_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "completed" and receipt["spec"] == spec
        assert prior["spec"] == spec | {"steps": 400}
        assert receipt["settings"] == prior["settings"] == plan["training"]
        assert receipt["data"] == prior["data"] == datasets[spec["task"]].receipt()
        assert receipt["data"]["test_materialized"] is False
        assert receipt["state_sha256"] == pc.sha(directory / "states.pt")
        assert (
            receipt["model_report"]["total_parameters"]
            == receipt["terminal_model_report"]["total_parameters"]
            == spec["budget"]
        )
        assert (
            receipt["model_report"]["topology_sha256"]
            == receipt["terminal_model_report"]["topology_sha256"]
        )
        assert [r["step"] for r in receipt["trace"]] == [-1, 0, 50, 100, 200, 400, 800]
        prefix = receipt["trace"][:-1] == prior["trace"]
        assert (
            receipt["closure_calls"] * 2049
            == receipt["exposures"]["optimization_train_examples"]
        )
        assert receipt["exposures"]["measurement_train_examples"] == 7 * 2049
        key = (spec["family"], spec["budget"])
        if key not in models:
            models[key] = pc.construct(*key, spec["seed"])
        model = models[key]
        states = torch.load(directory / "states.pt", weights_only=True)
        model.load_state_dict(states["terminal"])
        with torch.no_grad():
            data = datasets[spec["task"]]
            replay = float(
                (model(data.validation_x) - data.validation_y).square().mean()
            )
        for path in [receipt_path, prior_path]:
            bindings[str(path)] = pc.sha(path)
        rows.append(
            {
                "index": task["index"],
                **spec,
                "prefix_exact": prefix,
                "mse400": prior["terminal_validation_mse"],
                "mse800": receipt["terminal_validation_mse"],
                "replay_mse800": replay,
                "replay_exact": replay == receipt["terminal_validation_mse"],
                "seconds": receipt["elapsed_seconds"],
            }
        )
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["task"], r["family"], r["budget"]].append(r)
    means = []
    for (task, family, budget), values in sorted(grouped.items()):
        assert sorted(r["seed"] for r in values) == [7103, 7109]
        before = statistics.mean(r["mse400"] for r in values)
        after = statistics.mean(r["mse800"] for r in values)
        means.append(
            {
                "task": task,
                "family": family,
                "budget": budget,
                "mse400": before,
                "mse800": after,
                "ratio800_over400": after / before,
                "above_floor_halving": before > 1e-24 and after <= before / 2,
                "individual_mse800": [r["mse800"] for r in values],
            }
        )
    ranking_changes = []
    for task in pc.TASKS:
        for budget in [193, 397]:
            cell = [r for r in means if r["task"] == task and r["budget"] == budget]
            # Pairwise strict ordering after floor censoring avoids arbitrary tie ranks.
            for i, a in enumerate(cell):
                for b in cell[i + 1 :]:
                    d400 = max(a["mse400"], 1e-24) - max(b["mse400"], 1e-24)
                    d800 = max(a["mse800"], 1e-24) - max(b["mse800"], 1e-24)
                    if (d400 > 0) - (d400 < 0) != (d800 > 0) - (d800 < 0):
                        ranking_changes.append(
                            {
                                "task": task,
                                "budget": budget,
                                "families": [a["family"], b["family"]],
                            }
                        )
    assert len(means) == 81
    result = {
        "status": "passed"
        if all(r["prefix_exact"] and r["replay_exact"] for r in rows)
        else "review_required",
        "scope": "Complete development horizon and exact checkpoint/prefix review; no TEST. Adequacy decision is separate from audit success.",
        "manifest_sha256": pc.sha(campaign / "manifest.json"),
        "analysis_source_sha256": pc.sha(__file__),
        "input_sha256": bindings,
        "rows": rows,
        "means": means,
        "ranking_changes": ranking_changes,
        "above_floor_halving_cells": sum(r["above_floor_halving"] for r in means),
        "optimization_bracketed": not ranking_changes
        and not any(r["above_floor_halving"] for r in means),
        "exact_prefix_count": sum(r["prefix_exact"] for r in rows),
        "exact_checkpoint_count": sum(r["replay_exact"] for r in rows),
        "cpu_fit_hours": sum(r["seconds"] for r in rows) / 3600,
    }
    output.mkdir(parents=True)
    pc.dump(output / "audit.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "exact_prefix_count",
                    "exact_checkpoint_count",
                    "above_floor_halving_cells",
                    "optimization_bracketed",
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
