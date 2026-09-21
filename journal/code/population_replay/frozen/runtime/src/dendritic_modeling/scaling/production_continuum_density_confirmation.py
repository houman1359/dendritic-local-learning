"""Execute the 450 reserved-density fits without retuning uniform-target choices."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

from . import production_continuum as pc
from . import production_continuum_confirmation as cf
from . import production_continuum_density as targets


class DensityEvaluator:
    def __init__(self, density):
        self.density = density
        self.cache = {}

    def receipt(self):
        return self.density.receipt()

    def labels(self, x):
        key = (tuple(x.shape), pc.tensor_hash(x))
        if key not in self.cache:
            self.cache[key] = self.density.labels(x)
        return self.cache[key]

    def numpy_quadrature(self, x):
        return self.density.numpy_quadrature(x)

    def value_mp(self, x):
        return self.density.value_mp(x)


@dataclass(frozen=True)
class DensityTrainingData(pc.TrainingData):
    density: dict

    def receipt(self):
        return super().receipt() | {"density": self.density}


def training_data(base_task, evaluator):
    original = pc.make_data(base_task)
    return DensityTrainingData(
        original.train_x,
        evaluator.labels(original.train_x),
        original.validation_x,
        evaluator.labels(original.validation_x),
        evaluator.receipt(),
    )


def task_grid(parent_freeze):
    plan = parent_freeze["fresh_density_confirmation_plan"]
    choices = {
        (r["task"], r["family"], r["budget"]): r for r in parent_freeze["selected"]
    }
    rows = []
    for family in pc.FAMILIES:
        cells = (
            plan["deep_relu_cells"]
            if family == "biased_relu_depth2"
            else plan["primary_family_cells"]
        )
        for density_seed in plan["density_seeds"]:
            for cell in cells:
                lower, upper = cell["range"]
                base_task = "continuum_narrow" if lower == 1 else "continuum_broad"
                density = targets.PositiveDensity.from_seed(lower, upper, density_seed)
                source_budget = min(cell["P"], 397)
                choice = choices[base_task, family, source_budget]
                for seed in plan["model_seeds"]:
                    spec = pc.FitSpec(
                        family,
                        cell["P"],
                        base_task,
                        seed,
                        choice["recipe"],
                        choice["lr"],
                        3200,
                    )
                    spec.validate()
                    rows.append(
                        {
                            "index": len(rows),
                            "spec": asdict(spec),
                            "density_seed": density_seed,
                            "density": density.receipt(),
                            "selection_source_budget": source_budget,
                        }
                    )
    assert len(rows) == 450
    assert (
        len(
            {
                json.dumps({k: v for k, v in r.items() if k != "index"}, sort_keys=True)
                for r in rows
            }
        )
        == 450
    )
    return rows


def prepare(parent_freeze_directory, destination):
    parent_freeze_directory, destination = (
        Path(parent_freeze_directory),
        Path(destination),
    )
    path = parent_freeze_directory / "freeze.json"
    parent_freeze = cf.read_freeze(path, pc.sha(path))
    parent = Path(parent_freeze["development_campaign"])
    original = json.loads((parent / "manifest.json").read_text())
    assert (
        pc.sha(parent / "manifest.json") == parent_freeze["development_manifest_sha256"]
    )
    rows = task_grid(parent_freeze)
    destination.mkdir(parents=True, exist_ok=False)
    frozen = {
        "schema": "production_continuum_confirmation_freeze_v1",
        "selection_frozen": True,
        "test_materialized": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parent_scientific_freeze_sha256": pc.sha(path),
        "selection_policy": "Every recipe/LR copied from the uniform-target3200-step development selection frozen before any same-target TEST. No fresh-density retuning.",
        "scope": "Three reserved positive densities across narrow/broad intervals; not a new target-family or input-dimension claim.",
        "execution_amendment": "Original plan specified high-precision quadrature labels. Use equivalent70-digit analytic integration for full-grid labels, qualified against independent70-digit adaptive quadrature; independent256-node positive quadrature evaluates every TEST point. Functions, seeds, data coordinates and model choices are unchanged.",
        "evaluation_rules": parent_freeze["evaluation_rules"],
        "tasks": rows,
        "ordinal_predictions": [
            "Production shunt and matched ordinary division have identical predictions at identical states.",
            "Report shunt/shallow-ReLU and shunt/tanh error ratios for every declared cell and each density, retaining reversals.",
            "Do not infer a learned exponent when fewer than4 uncensored sizes over factor8 remain.",
        ],
    }
    shutil.copytree(
        parent / "source",
        destination / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for source in original["source_files"]:
        assert pc.sha(destination / source["path"]) == source["sha256"]
    sources = list(original["source_files"])
    for source_path in [Path(cf.__file__), Path(targets.__file__), Path(__file__)]:
        relative = "source/src/dendritic_modeling/scaling/" + source_path.name
        assert not any(r["path"] == relative for r in sources)
        shutil.copy2(source_path, destination / relative)
        sources.append({"path": relative, "sha256": pc.sha(source_path)})
    shutil.copy2(parent_freeze_directory / "base_plan.json", destination / "plan.json")
    pc.dump(destination / "freeze.json", frozen)
    manifest = {
        "schema": "production_continuum_confirmation_campaign_v1",
        "stage": "fresh_density_confirmation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": sources,
        "tasks": rows,
        "freeze_sha256": pc.sha(destination / "freeze.json"),
        "plan_sha256": pc.sha(destination / "plan.json"),
        "test_evaluation_authorized": True,
        "test_materialized_at_prepare": False,
    }
    pc.dump(destination / "manifest.json", manifest)
    return manifest


def run(campaign, shard_index, shards):
    campaign = Path(campaign).resolve()
    assert (
        Path(__file__).resolve()
        == campaign
        / "source/src/dendritic_modeling/scaling/production_continuum_density_confirmation.py"
    )
    manifest = json.loads((campaign / "manifest.json").read_text())
    frozen = cf.read_freeze(campaign / "freeze.json", manifest["freeze_sha256"])
    assert frozen["tasks"] == manifest["tasks"] and 0 <= shard_index < shards
    assert pc.sha(campaign / "plan.json") == manifest["plan_sha256"]
    for source in manifest["source_files"]:
        assert pc.sha(campaign / source["path"]) == source["sha256"]
    plan = json.loads((campaign / "plan.json").read_text())
    pc.validate_plan(plan)
    torch.set_num_threads(1)
    cache, results = {}, []
    for task in manifest["tasks"]:
        if task["index"] % shards != shard_index:
            continue
        spec = pc.FitSpec(**task["spec"])
        key = (spec.task, task["density_seed"])
        if key not in cache:
            lower, upper = task["density"]["interval"]
            evaluator = DensityEvaluator(
                targets.PositiveDensity.from_seed(lower, upper, task["density_seed"])
            )
            cache[key] = evaluator, training_data(spec.task, evaluator)
        evaluator, data = cache[key]
        assert evaluator.receipt() == task["density"]
        directory = campaign / f"runs/{task['index']:04d}"
        receipt = pc.fit(spec, data, plan["training"], directory)
        evaluation = None
        if receipt["status"] == "completed":
            evaluation = cf.evaluate_saved_fit(
                directory,
                task,
                campaign / "freeze.json",
                manifest["freeze_sha256"],
                density=evaluator,
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
    p.add_argument("--parent-freeze-directory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--campaign", type=Path, required=True)
    r.add_argument("--shard-index", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        m = prepare(args.parent_freeze_directory, args.output_dir)
        print(
            json.dumps(
                {"tasks": len(m["tasks"]), "source_files": len(m["source_files"])}
            )
        )
    else:
        results = run(args.campaign, args.shard_index, args.shards)
        if any(
            r["training_status"] != "completed" or r["evaluation_status"] != "completed"
            for r in results
        ):
            raise SystemExit(1)
