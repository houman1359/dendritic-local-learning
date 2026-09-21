"""Freeze the prescribed development horizon refits using the qualified trainer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil

from . import production_continuum as pc


def prepare(development: Path, audit_path: Path, destination: Path):
    development = development.resolve()
    manifest = json.loads((development / "manifest.json").read_text())
    plan = json.loads((development / "plan.json").read_text())
    audit = json.loads(audit_path.read_text())
    assert manifest["stage"] == "development"
    assert audit["status"] == "passed" and audit["replay_exact_count"] == 1296
    assert audit["manifest_sha256"] == pc.sha(development / "manifest.json")
    assert manifest["plan_sha256"] == pc.sha(development / "plan.json")
    pc.validate_plan(plan)
    assert [r["spec"] for r in manifest["tasks"]] == [
        asdict(s) for s in pc.development_specs(plan)
    ]
    for source in manifest["source_files"]:
        assert pc.sha(development / source["path"]) == source["sha256"]
    for path, digest in audit["input_sha256"].items():
        assert Path(path).resolve().is_relative_to(development)
        assert pc.sha(path) == digest
    selected = {(r["task"], r["family"], r["budget"]): r for r in audit["selected"]}
    assert len(selected) == 162
    cells = set(itertools.product(pc.TASKS, pc.FAMILIES, [13, 193, 397]))
    cells |= set(
        itertools.product(
            pc.TASKS, plan["primary_mechanism_curve"]["families"], [7, 25, 73]
        )
    )
    # Use explicit field order; JSON writers may sort object keys.
    original = {
        tuple(r["spec"][k] for k in pc.FitSpec.__dataclass_fields__): r["index"]
        for r in manifest["tasks"]
    }
    tasks = []
    for task, family, budget in sorted(cells):
        choice = selected[task, family, budget]
        for seed in plan["development_seeds"]:
            spec = pc.FitSpec(
                family, budget, task, seed, choice["recipe"], choice["lr"], 800
            )
            old = asdict(spec) | {"steps": 400}
            index = original[tuple(old[k] for k in pc.FitSpec.__dataclass_fields__)]
            assert index in choice["indices"]
            tasks.append(
                {"index": len(tasks), "spec": asdict(spec), "development_index": index}
            )
    assert len(tasks) == 162
    assert len({tuple(r["spec"].values()) for r in tasks}) == 162
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        development / "source",
        destination / "source",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copy2(development / "plan.json", destination / "plan.json")
    shutil.copy2(audit_path, destination / "selection_audit.json")
    shutil.copy2(__file__, destination / "publisher.py")
    for source in manifest["source_files"]:
        assert pc.sha(destination / source["path"]) == source["sha256"]
    result = {
        "schema": manifest["schema"],
        "stage": "horizon",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": manifest["plan_sha256"],
        "test_materialized": False,
        "source_files": manifest["source_files"],
        "tasks": tasks,
        "development_campaign": str(development),
        "development_manifest_sha256": pc.sha(development / "manifest.json"),
        "selection_audit_sha256": pc.sha(audit_path),
        "publisher_sha256": pc.sha(__file__),
        "horizon_contract": "Each FitSpec.steps=800 overrides the original plan's 400-step development default. Same source, data, seed, recipe, LR and optimizer; cold refits. No TEST. Compare shared 400-step traces before interpreting extension.",
    }
    pc.dump(destination / "manifest.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.development, args.audit, args.output_dir)
    print(json.dumps({"tasks": len(result["tasks"]), "stage": result["stage"]}))
