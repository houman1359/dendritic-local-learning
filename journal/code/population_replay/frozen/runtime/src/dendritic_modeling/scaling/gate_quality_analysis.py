"""Audit and summarize the frozen 216-fit gate calibration without TEST access."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
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


def audit(directory: Path) -> dict:
    directory = directory.resolve()
    bindings = {}

    def read(path):
        raw = Path(path).read_bytes()
        bindings[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
        return raw

    def inside(relative):
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Escaping campaign path")
        return path

    manifest = json.loads(read(directory / "manifest.json"))
    spec = json.loads(read(directory / "spec.json"))
    assert (
        hashlib.sha256((directory / "spec.json").read_bytes()).hexdigest()
        == manifest["spec_file_sha256"]
    )
    assert spec["study_design"]["stage"] == "gate_quality_calibration"
    assert spec["seeds"] == [421, 423]
    assert len(manifest["tasks"]) == 216
    sources = {s["path"]: s["sha256"] for s in manifest["source_files"]}
    assert len(sources) == 938
    for path, digest in sources.items():
        assert hashlib.sha256(read(inside(path))).hexdigest() == digest
    for path in (
        Path(__file__),
        Path(__file__).with_name("dendrinet_intervention_audit.py"),
        Path(__file__).with_name("analyze.py"),
    ):
        read(path)
    collected = {r["id"]: r for r in collect_campaign(directory, "validation")}
    assert len(collected) == 216 and all(
        r["status"] == "completed" for r in collected.values()
    ), "Require all completed, verified outcomes"
    rows, seen = [], set()
    for task in manifest["tasks"]:
        raw = read(inside(task["config"]))
        assert hashlib.sha256(raw).hexdigest() == task["config_file_sha256"]
        config = json.loads(raw)
        _config_contract(config, spec)
        assert not config.get("evaluate_test", False)
        run = inside(task["output"])
        receipt = json.loads(read(run / "receipt.json"))
        assert receipt["config"] == config and receipt["config_sha256"] == _identity(
            config
        )
        assert json.loads(read(run / "config.json")) == config
        assert receipt["status"] == "completed"
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
            for name in ("metrics.jsonl", "diagnostics.jsonl")
        }
        training = config["training"]
        assert (
            training["steps"] == 1000
            and training["eval_every"] == training["diagnostics_every"] == 250
        )
        _successful_streams(
            streams["metrics.jsonl"], streams["diagnostics.jsonl"], receipt, training
        )
        assert receipt["training"]["parameters_missing_gradients_max"] == 0
        assert collected[task["id"]]["loss"] == receipt["metrics"]["validation_loss"]
        axes = spec["study_design"]["architecture_axes"][config["architecture_id"]]
        key = (
            axes["base_architecture_id"],
            config["model"]["activation"],
            config["target_parameters"],
            training["lr"],
            config["seed"],
        )
        assert key not in seen
        seen.add(key)
        trajectory = streams["metrics.jsonl"]
        rows.append(
            {
                "id": task["id"],
                "base": key[0],
                "gate": key[1],
                "budget": key[2],
                "lr": key[3],
                "seed": key[4],
                "actual_parameters": task["total_parameters"],
                "validation_ce": receipt["metrics"]["validation_loss"],
                "validation_accuracy": receipt["metrics"]["validation_accuracy"],
                "trajectory": [
                    {
                        "step": r["step"],
                        "ce": r["validation_loss"],
                        "accuracy": r["validation_accuracy"],
                    }
                    for r in trajectory
                ],
                "terminal_minus_best_logged_ce": trajectory[-1]["validation_loss"]
                - min(r["validation_loss"] for r in trajectory),
            }
        )
    bases = sorted({r["base"] for r in rows})
    assert len(bases) == 6
    assert seen == set(
        itertools.product(
            bases,
            ["relu", "param_relu", "param_tanh"],
            [8192, 131072],
            [0.0003, 0.001, 0.003],
            [421, 423],
        )
    )
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["base"], r["gate"], r["budget"], r["lr"]].append(r)
    means = []
    for (base, gate, budget, lr), records in sorted(grouped.items()):
        assert sorted(r["seed"] for r in records) == [421, 423]
        means.append(
            {
                "base": base,
                "gate": gate,
                "budget": budget,
                "lr": lr,
                "ce": statistics.mean(r["validation_ce"] for r in records),
                "accuracy": statistics.mean(r["validation_accuracy"] for r in records),
                "actual_parameters": records[0]["actual_parameters"],
                "individual_ce": {str(r["seed"]): r["validation_ce"] for r in records},
            }
        )
    selected = [
        min(
            (r for r in means if (r["base"], r["gate"], r["budget"]) == key),
            key=lambda r: (r["ce"], r["lr"]),
        )
        for key in itertools.product(
            bases, ["relu", "param_relu", "param_tanh"], [8192, 131072]
        )
    ]
    assert len(means) == 108 and len(selected) == 36
    contrasts = []
    for r in selected:
        if r["gate"] == "relu":
            continue
        baseline = next(
            b
            for b in selected
            if (b["base"], b["budget"], b["gate"]) == (r["base"], r["budget"], "relu")
        )
        contrasts.append(
            {
                **r,
                "baseline_ce": baseline["ce"],
                "baseline_lr": baseline["lr"],
                "difference_vs_relu": r["ce"] - baseline["ce"],
            }
        )
    return {
        "status": "passed",
        "scope": "Complete recorded-source/configuration/inventory/log audit and development aggregation; no checkpoint replay, no TEST, no scaling-law claim.",
        "rows": rows,
        "lr_means": means,
        "selected": selected,
        "gate_contrasts": contrasts,
        "terminal_over_best_by_005": sum(
            r["terminal_minus_best_logged_ce"] > 0.05 for r in rows
        ),
        "source_files_verified": len(sources),
        "input_sha256": bindings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.campaign)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    lines = [
        "# Gate quality calibration",
        "",
        "All 216 fits completed and passed source/configuration/inventory/log checks. No TEST data or checkpoints were opened. The table selects LR using the same development validation set on which it reports the two-seed mean; these are calibration outcomes, not independent confirmation.",
        "",
        "Learned gates rematch soma width and can change support. Each gate receives the same three-rate search; this is an equal-budget family comparison, not fixed-anatomy gate causality.",
        "",
        "| Base | Budget | ReLU CE (LR) | Learned ReLU CE (LR) | Learned tanh CE (LR) |",
        "|---|---:|---:|---:|---:|",
    ]
    for base, budget in sorted({(r["base"], r["budget"]) for r in result["selected"]}):
        cells = [
            next(
                r
                for r in result["selected"]
                if (r["base"], r["budget"], r["gate"]) == (base, budget, gate)
            )
            for gate in ("relu", "param_relu", "param_tanh")
        ]
        lines.append(
            f"| {base} | {budget} | "
            + " | ".join(f"{r['ce']:.6f} ({r['lr']:g})" for r in cells)
            + " |"
        )
    lines += [
        "",
        f"Terminal CE exceeds an earlier logged minimum by more than 0.05 in {result['terminal_over_best_by_005']}/216 fits. The complete audit retains all 108 LR means, 36 selections, individual seed results, and all logged trajectories.",
        "",
        "The planned horizon comparison must use cold refits at 500/1000/2000 steps; intermediate validation logs are not saved checkpoints. Deeper ordinary controls and new task/sample draws remain necessary before a scaling claim.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed": len(result["rows"]),
                "selected_cells": len(result["selected"]),
                "terminal_over_best_by_005": result["terminal_over_best_by_005"],
            }
        )
    )


if __name__ == "__main__":
    main()
