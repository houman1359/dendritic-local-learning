#!/usr/bin/env python3
"""Verify the frozen H=4 D1 serial/grouped-point initialization gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment


SERIAL_STEM = "journal_h4_aligned_serial_shunting_bp"
GROUPED_STEM = "journal_h4_aligned_grouped_point_shunting_bp"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_config(runs_root: Path, stem: str, index: int) -> Path:
    matches = sorted(runs_root.glob(f"{stem}_*"))
    if not matches:
        raise FileNotFoundError(f"No frozen run directory for {stem}")
    path = matches[-1] / "configs" / f"unified_config_{index}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build(path: Path):
    raw = yaml.safe_load(path.read_text())
    factors = raw["model"]["core"]["population_network"]["layers"][0][
        "populations"
    ][0]["branch_factors"]
    if factors != [8]:
        raise ValueError(f"Expected D1=[8] in config 0, found {factors}")
    config = load_config(str(path))
    _, train, _, _, model, _ = setup_environment(
        config.model,
        config.training,
        config.data,
        config.wandb,
        config.outputs,
        config.experiment,
        is_main=False,
    )
    model.eval()
    return raw, train, model


def logits(model, values: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        output = model(values)
    if isinstance(output, tuple):
        output = output[0]
    return output.detach().cpu()


def source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    serial_path = latest_config(args.runs_root.resolve(), SERIAL_STEM, args.config_index)
    grouped_path = latest_config(args.runs_root.resolve(), GROUPED_STEM, args.config_index)
    serial_raw, serial_data, serial_model = build(serial_path)
    grouped_raw, grouped_data, grouped_model = build(grouped_path)

    serial_seed = int(serial_raw["experiment"]["seed"])
    grouped_seed = int(grouped_raw["experiment"]["seed"])
    if serial_seed != grouped_seed:
        raise ValueError(f"Unpaired seeds: {serial_seed} and {grouped_seed}")
    if len(serial_data) != len(grouped_data):
        raise ValueError("D1 controls resolved different training-set sizes")

    serial_values = torch.stack(
        [serial_data[index][0] for index in range(args.batch_size)]
    )
    grouped_values = torch.stack(
        [grouped_data[index][0] for index in range(args.batch_size)]
    )
    if not torch.equal(serial_values, grouped_values):
        raise ValueError("D1 controls resolved different paired input batches")

    serial_logits = logits(serial_model, serial_values)
    grouped_logits = logits(grouped_model, grouped_values)
    max_abs = float((serial_logits - grouped_logits).abs().max())
    record = {
        "status": "pass" if torch.equal(serial_logits, grouped_logits) else "fail",
        "source_commit": source_commit(),
        "config_index": args.config_index,
        "seed": serial_seed,
        "batch_size": args.batch_size,
        "serial_config": str(serial_path.relative_to(args.runs_root.resolve())),
        "serial_config_sha256": sha256(serial_path),
        "grouped_config": str(grouped_path.relative_to(args.runs_root.resolve())),
        "grouped_config_sha256": sha256(grouped_path),
        "input_shape": list(serial_values.shape),
        "logit_shape": list(serial_logits.shape),
        "inputs_bitwise_equal": True,
        "logits_bitwise_equal": bool(torch.equal(serial_logits, grouped_logits)),
        "maximum_absolute_logit_difference": max_abs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if record["status"] != "pass":
        raise SystemExit("D1 initialization equivalence gate failed")


if __name__ == "__main__":
    main()
