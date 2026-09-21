"""Audit both frozen gate-horizon campaigns and every shared training prefix."""

from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import statistics

from .analyze import collect_campaign
from .dendrinet_intervention_audit import (
    _config_contract,
    _identity,
    _no_test,
    _readout_receipt,
    _source_receipt,
    _successful_streams,
    _terminal_inventory,
)
from .production_continuum import dump, sha


def audit(campaigns, selection):
    selected = json.loads(selection.read_text())
    rates = {(r["base"], r["gate"], r["budget"]): r["lr"] for r in selected["selected"]}
    bindings, rows = {}, []

    def read(path):
        path = Path(path).resolve()
        bindings[str(path)] = sha(path)
        return path.read_bytes()

    for source in [
        Path(__file__),
        Path(__file__).with_name("dendrinet_intervention_audit.py"),
        Path(__file__).with_name("analyze.py"),
        selection,
    ]:
        read(source)
    for directory in campaigns:
        directory = directory.resolve()

        def inside(relative, directory=directory):
            path = (directory / relative).resolve()
            assert path.is_relative_to(directory)
            return path

        manifest = json.loads(read(directory / "manifest.json"))
        spec = json.loads(read(directory / "spec.json"))
        assert sha(directory / "spec.json") == manifest["spec_file_sha256"]
        assert spec["study_design"]["stage"] == "gate_horizon"
        assert spec["study_design"]["parent_selection_audit_sha256"] == sha(selection)
        assert spec["seeds"] == [431, 433] and len(manifest["tasks"]) == 108
        sources = {s["path"]: s["sha256"] for s in manifest["source_files"]}
        assert len(sources) == 938
        for path, digest in sources.items():
            read(inside(path))
            assert sha(inside(path)) == digest
        collected = {r["id"]: r for r in collect_campaign(directory, "validation")}
        assert len(collected) == 108 and all(
            r["status"] == "completed" for r in collected.values()
        )
        for task in manifest["tasks"]:
            config = json.loads(read(inside(task["config"])))
            assert sha(inside(task["config"])) == task["config_file_sha256"]
            _config_contract(config, spec)
            assert config["evaluate_test"] is False
            run = inside(task["output"])
            receipt = json.loads(read(run / "receipt.json"))
            assert receipt["config"] == json.loads(read(run / "config.json")) == config
            assert (
                receipt["config_sha256"] == _identity(config)
                and receipt["status"] == "completed"
            )
            _no_test(receipt)
            _source_receipt(receipt, sources, read, inside)
            _readout_receipt(receipt["readout_initialization"], {"mode": "preserve"})
            _terminal_inventory(receipt, config, task["total_parameters"])
            streams = {
                name: [
                    json.loads(line)
                    for line in read(run / name).splitlines()
                    if line.strip()
                ]
                for name in ["metrics.jsonl", "diagnostics.jsonl"]
            }
            training = config["training"]
            assert training["steps"] in [500, 1000, 2000]
            assert training["eval_every"] == training["diagnostics_every"] == 250
            _successful_streams(
                streams["metrics.jsonl"],
                streams["diagnostics.jsonl"],
                receipt,
                training,
            )
            assert receipt["training"]["parameters_missing_gradients_max"] == 0
            axes = spec["study_design"]["architecture_axes"][config["architecture_id"]]
            key = (
                axes["base_architecture_id"],
                config["model"]["activation"],
                config["target_parameters"],
            )
            assert training["lr"] == rates[key] == axes["selected_lr"]
            trajectory = [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ["horizon_steps", "horizon_fraction", "elapsed_seconds"]
                }
                for r in streams["metrics.jsonl"]
            ]
            rows.append(
                {
                    "base": key[0],
                    "gate": key[1],
                    "budget": key[2],
                    "seed": config["seed"],
                    "steps": training["steps"],
                    "lr": training["lr"],
                    "actual_parameters": task["total_parameters"],
                    "validation_ce": receipt["metrics"]["validation_loss"],
                    "accuracy": receipt["metrics"]["validation_accuracy"],
                    "trajectory": trajectory,
                }
            )
    keys = [(r["base"], r["gate"], r["budget"], r["steps"], r["seed"]) for r in rows]
    assert len(keys) == len(set(keys)) == 216
    assert set(keys) == set(
        itertools.product(
            sorted({r["base"] for r in rows}),
            ["relu", "param_relu", "param_tanh"],
            [8192, 131072],
            [500, 1000, 2000],
            [431, 433],
        )
    )
    by_key = dict(zip(keys, rows))
    prefixes = []
    for r in rows:
        if r["steps"] == 2000:
            continue
        long = by_key[r["base"], r["gate"], r["budget"], 2000, r["seed"]]
        prefix = [x for x in long["trajectory"] if x["step"] <= r["steps"]]
        prefixes.append(
            {
                "base": r["base"],
                "gate": r["gate"],
                "budget": r["budget"],
                "seed": r["seed"],
                "steps": r["steps"],
                "exact": r["trajectory"] == prefix,
            }
        )
    groups = defaultdict(list)
    for r in rows:
        groups[r["base"], r["gate"], r["budget"], r["steps"]].append(r)
    means = [
        {
            "base": b,
            "gate": g,
            "budget": p,
            "steps": h,
            "ce": statistics.mean(r["validation_ce"] for r in values),
            "individual_ce": [r["validation_ce"] for r in values],
            "accuracy": statistics.mean(r["accuracy"] for r in values),
        }
        for (b, g, p, h), values in sorted(groups.items())
    ]
    chosen = [
        min(
            [r for r in means if (r["base"], r["gate"], r["budget"]) == key],
            key=lambda r: (r["ce"], r["steps"]),
        )
        for key in sorted(rates)
    ]
    assert len(means) == 108 and len(chosen) == 36 and len(prefixes) == 144
    return {
        "status": "passed"
        if all(p["exact"] for p in prefixes)
        else "prefix_review_required",
        "scope": "Source/config/inventory/log audit with exact shared-prefix comparison; fresh initialization/routing seeds on existing target/data. No checkpoint replay, TEST or exponent estimate.",
        "rows": rows,
        "horizon_means": means,
        "selected": chosen,
        "prefixes": prefixes,
        "exact_prefix_count": sum(p["exact"] for p in prefixes),
        "input_sha256": bindings,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaigns", type=Path, nargs=2, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaigns, args.selection)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dump(args.output_dir / "audit.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "fits": len(result["rows"]),
                "exact_prefix_count": result["exact_prefix_count"],
            }
        )
    )
