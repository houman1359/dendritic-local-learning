"""Frozen-feature linear/deep companions; reuse the immutable main fitter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

MAIN_ROOT = "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/outputs/parameter_scaling/20260913/practical_transfer_v1/campaign_v2"
MAIN_SOURCE_SHA = "3bb1c19d86df365dfee0732dd886d1071919272d46eb90601e35e37badee42eb"
MAIN_MANIFEST_SHA = "def93528c13f19aa12db7014c5a838127a9bdf9398631935363d00a8823fc89d"
MAIN_PROTOCOL_SHA = "d6ca417d5240d5927f50a58fb9f8d08161f85d6f4db108c8117a74759b97d362"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def default_config():
    return {
        "schema": "practical_feature_companion_v1",
        "main_root": MAIN_ROOT,
        "main_source_sha256": MAIN_SOURCE_SHA,
        "main_manifest_sha256": MAIN_MANIFEST_SHA,
        "main_protocol_sha256": MAIN_PROTOCOL_SHA,
        "families": ["linear", "deep_relu"],
        "input_dim": 256,
        "classes": 1000,
        "linear_parameters": 257000,
        "deep_development_budgets": [250000, 500000, 1000000, 2000000],
        "heldout_budget": 4000000,
        "development_seeds": [73101],
        "confirmation_seeds": [74101, 74107, 74113],
        "learning_rates": [0.0003, 0.001],
        "epochs": 50,
        "batch_size": 512,
        "weight_decay": 0.0001,
        "optimizer": "AdamW",
        "selection": "Per-family/budget endpoint validation CE; deep4M inherits2M LR",
        "forecast": "Deep only: log selected development validationCE on log actualP; freeze4M before confirmation",
        "fits": {"development": 10, "confirmation": 18, "total": 28},
        "gpu_hour_ceiling_total": 1,
        "scope": "Additional ordinary references frozen before main quality outcomes; unchanged main primary hypotheses",
    }


def budgets(config, family, confirmation=False):
    if family == "linear":
        return [config["linear_parameters"]]
    if family != "deep_relu":
        raise ValueError(family)
    return config["deep_development_budgets"] + (
        [config["heldout_budget"]] if confirmation else []
    )


class ControlHead(nn.Module):
    def __init__(self, family, budget, *, seed=0, input_dim=256, classes=1000):
        super().__init__()
        self.family = family
        if family == "linear":
            self.width = None
            expected = (input_dim + 1) * classes
            if budget != expected:
                raise ValueError("The linear reference is one exact-count point")
        elif family == "deep_relu":
            coefficient = input_dim + classes + 2
            self.width = (
                math.isqrt(coefficient**2 + 4 * (budget - classes)) - coefficient
            ) // 2
            if self.width < 1:
                raise ValueError("Budget cannot fit two hidden layers")
            expected = self.width**2 + coefficient * self.width + classes
            assert (self.width + 1) ** 2 + coefficient * (
                self.width + 1
            ) + classes > budget
        else:
            raise ValueError(family)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if family == "linear":
                self.layers = nn.Linear(input_dim, classes)
            else:
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, self.width),
                    nn.ReLU(),
                    nn.Linear(self.width, self.width),
                    nn.ReLU(),
                    nn.Linear(self.width, classes),
                )
        self.actual_parameters = sum(p.numel() for p in self.parameters())
        assert self.actual_parameters == expected <= budget

    def forward(self, x):
        return self.layers(x)


def verify_protocol(root):
    root = Path(root)
    frozen = root / "frozen"
    manifest = json.loads((frozen / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha256(frozen / name) == digest
    assert sha256(__file__) == manifest["files"]["real_feature_controls.py"]
    config = json.loads((frozen / "protocol.json").read_text())
    assert config == default_config()
    main_frozen = Path(config["main_root"]) / "frozen"
    assert sha256(main_frozen / "manifest.json") == config["main_manifest_sha256"]
    assert sha256(main_frozen / "protocol.json") == config["main_protocol_sha256"]
    main_manifest = json.loads((main_frozen / "manifest.json").read_text())
    for name, digest in main_manifest["files"].items():
        assert sha256(main_frozen / name) == digest
    assert sha256(main_frozen / "real_feature_heads.py") == config["main_source_sha256"]
    return config


def initialize(root):
    root = Path(root)
    config = verify_protocol(root)
    # Only immutable configuration/source files are read, never head outcomes.
    write_json(
        root / "initialized.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha256(__file__),
            "protocol_sha256": sha256(root / "frozen/protocol.json"),
            "manifest_sha256": sha256(root / "frozen/manifest.json"),
            "main_manifest_sha256": config["main_manifest_sha256"],
            "quality_data_access": False,
        },
    )


def verify_initialized(root):
    root = Path(root)
    config = verify_protocol(root)
    receipt = json.loads((root / "initialized.json").read_text())
    assert receipt["source_sha256"] == sha256(__file__)
    assert receipt["protocol_sha256"] == sha256(root / "frozen/protocol.json")
    assert receipt["manifest_sha256"] == sha256(root / "frozen/manifest.json")
    return config


def import_main(config):
    path = Path(config["main_root"]) / "frozen/real_feature_heads.py"
    assert sha256(path) == config["main_source_sha256"]
    spec = importlib.util.spec_from_file_location("frozen_main_feature_fitter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_bindings(main_root):
    main_root = Path(main_root)
    return {
        name: sha256(main_root / name)
        for name in (
            "index_receipt.json",
            "train_extraction_receipt.json",
            "validation_extraction_receipt.json",
        )
    }


def cell_path(root, stage, family, budget, seed, lr):
    return Path(root) / stage / family / f"p{budget}_seed{seed}_lr{lr:.8g}"


def read_complete(root, path, config):
    root, path = Path(root), Path(path)
    receipt = json.loads((path / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    for name, digest in receipt["files"].items():
        assert sha256(path / name) == digest
    request = json.loads((path / "request.json").read_text())
    assert request["source_sha256"] == sha256(__file__)
    assert request["protocol_sha256"] == sha256(root / "frozen/protocol.json")
    assert request["main_source_sha256"] == config["main_source_sha256"]
    assert request["cache_receipts"] == cache_bindings(config["main_root"])
    result = json.loads((path / "result.json").read_text())
    assert result["traces"][-1]["epoch"] == config["epochs"]
    return request, result


def load_selection(root):
    root = Path(root)
    config = verify_initialized(root)
    receipt = json.loads((root / "selection_complete.json").read_text())
    assert receipt["selection_sha256"] == sha256(root / "selection.json")
    assert receipt["source_sha256"] == sha256(__file__)
    assert receipt["protocol_sha256"] == sha256(root / "frozen/protocol.json")
    selection = json.loads((root / "selection.json").read_text())
    expected = {
        str(
            cell_path(root, "development", f, b, s, lr).relative_to(root)
            / "receipt.json"
        )
        for f in config["families"]
        for b in budgets(config, f)
        for s in config["development_seeds"]
        for lr in config["learning_rates"]
    }
    assert set(selection["development_receipts"]) == expected
    for name, digest in selection["development_receipts"].items():
        assert sha256(root / name) == digest
        read_complete(root, (root / name).parent, config)
    return selection


def run_fit(root, stage, family, budget, seed, lr=None):
    root = Path(root)
    config = verify_initialized(root)
    if family not in config["families"]:
        raise ValueError(family)
    if stage == "development":
        assert budget in budgets(config, family)
        assert seed in config["development_seeds"] and lr in config["learning_rates"]
        selection_sha = None
    elif stage == "confirmation":
        assert budget in budgets(config, family, True)
        assert seed in config["confirmation_seeds"]
        selection = load_selection(root)
        chosen = selection["choices"][family][str(budget)]["lr"]
        assert lr is None or lr == chosen
        lr, selection_sha = chosen, sha256(root / "selection.json")
    else:
        raise ValueError(stage)
    main = import_main(config)
    main_root = Path(config["main_root"])
    main.verify_bundle(main_root, ("train", "validation"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {
        s: torch.load(
            main_root / f"{s}_features.pt", map_location=device, weights_only=True
        )
        for s in ("train", "validation")
    }
    rms = main.feature_rms(loaded["train"]["features"])
    model = ControlHead(family, budget, seed=seed).to(device)
    destination = cell_path(root, stage, family, budget, seed, lr)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(
        destination / "request.json",
        {
            "family": family,
            "budget": budget,
            "seed": seed,
            "lr": lr,
            "stage": stage,
            "epochs": config["epochs"],
            "actual_parameters": model.actual_parameters,
            "hidden_width": model.width,
            "frozen_backbone_parameters": 2469696,
            "fixed_rms_buffer_scalars": 256,
            "source_sha256": sha256(__file__),
            "protocol_sha256": sha256(root / "frozen/protocol.json"),
            "main_source_sha256": config["main_source_sha256"],
            "cache_receipts": cache_bindings(main_root),
            "selection_sha256": selection_sha,
        },
    )
    torch.save(
        {"state": model.state_dict(), "rms": rms.cpu()},
        destination / "initial_state.pt",
    )
    progress_stream = (destination / "progress.jsonl").open("x")

    def progress(row):
        progress_stream.write(json.dumps(row, allow_nan=False) + "\n")
        if "epoch_metrics" in row:
            progress_stream.flush()

    try:
        result, initial = main.fit_model(
            model,
            loaded["train"]["features"] / rms,
            loaded["train"]["labels"],
            loaded["validation"]["features"] / rms,
            loaded["validation"]["labels"],
            lr=lr,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            weight_decay=config["weight_decay"],
            seed=seed,
            progress=progress,
        )
        torch.save(
            {"initial": initial, "terminal": model.state_dict(), "rms": rms.cpu()},
            destination / "states.pt",
        )
        write_json(destination / "result.json", result)
    except Exception as exc:
        torch.save(model.state_dict(), destination / "failed_state.pt")
        write_json(
            destination / "failure.json",
            {"type": type(exc).__name__, "error": str(exc)},
        )
        raise
    finally:
        progress_stream.close()
    write_json(
        destination / "receipt.json",
        {
            "status": "complete",
            "actual_parameters": model.actual_parameters,
            "files": {
                name: sha256(destination / name)
                for name in (
                    "request.json",
                    "initial_state.pt",
                    "progress.jsonl",
                    "states.pt",
                    "result.json",
                )
            },
        },
    )


def select_recipes(root):
    root = Path(root)
    config = verify_initialized(root)
    main = import_main(config)
    main.verify_bundle(config["main_root"], ("train", "validation"))
    choices, bindings = {}, {}
    for family in config["families"]:
        choices[family] = {}
        for budget in budgets(config, family):
            trials = []
            for lr in config["learning_rates"]:
                losses = []
                for seed in config["development_seeds"]:
                    path = cell_path(root, "development", family, budget, seed, lr)
                    request, result = read_complete(root, path, config)
                    assert (
                        request["family"],
                        request["budget"],
                        request["seed"],
                        request["lr"],
                        request["stage"],
                    ) == (family, budget, seed, lr, "development")
                    losses.append(result["traces"][-1]["validation"]["cross_entropy"])
                    bindings[str(path.relative_to(root) / "receipt.json")] = sha256(
                        path / "receipt.json"
                    )
                trials.append({"lr": lr, "mean_validation_ce": float(np.mean(losses))})
            chosen = min(trials, key=lambda row: (row["mean_validation_ce"], row["lr"]))
            choices[family][str(budget)] = {"lr": chosen["lr"], "trials": trials}
    parent = max(config["deep_development_budgets"])
    choices["deep_relu"][str(config["heldout_budget"])] = {
        "lr": choices["deep_relu"][str(parent)]["lr"],
        "inherited_from_budget": parent,
    }
    xs, ys = [], []
    for budget in config["deep_development_budgets"]:
        choice = choices["deep_relu"][str(budget)]
        risk = next(
            row["mean_validation_ce"]
            for row in choice["trials"]
            if row["lr"] == choice["lr"]
        )
        xs.append(math.log(ControlHead("deep_relu", budget).actual_parameters))
        ys.append(math.log(risk))
    slope, intercept = np.linalg.lstsq(
        np.column_stack([xs, np.ones(len(xs))]), ys, rcond=None
    )[0]
    held_actual = ControlHead("deep_relu", config["heldout_budget"]).actual_parameters
    write_json(
        root / "selection.json",
        {
            "choices": choices,
            "development_receipts": bindings,
            "deep_forecast": {
                "heldout_actual_parameters": held_actual,
                "slope": float(slope),
                "intercept": float(intercept),
                "predicted_validation_cross_entropy": math.exp(
                    float(intercept + slope * math.log(held_actual))
                ),
                "scope": "Selection-optimistic descriptive4M validationCE point forecast, no asymptotic claim",
            },
            "linear_forecast": None,
            "test_access": False,
        },
    )
    write_json(
        root / "selection_complete.json",
        {
            "source_sha256": sha256(__file__),
            "protocol_sha256": sha256(root / "frozen/protocol.json"),
            "selection_sha256": sha256(root / "selection.json"),
        },
    )


def confirmation_barrier(root, config, main):
    """No test tensor loading until all180 main and18 companion cells pass."""
    root = Path(root)
    selected = load_selection(root)
    main_root = Path(config["main_root"])
    main_config = main.verify_bundle(main_root, ("train", "validation"))
    main_selected = main.load_selection(main_root)
    main_bindings = {}
    for family in main_config["families"]:
        for budget in main_config["development_budgets"] + [
            main_config["heldout_budget"]
        ]:
            for seed in main_config["confirmation_seeds"]:
                lr = main_selected["choices"][family][str(budget)]["lr"]
                path = main.cell_path(
                    main_root, "confirmation", family, budget, seed, lr
                )
                request, result = main.read_complete_cell(
                    path, sha256(main_root / "config.json")
                )
                assert request["selection_sha256"] == sha256(
                    main_root / "selection.json"
                )
                assert (
                    request["family"],
                    request["budget"],
                    request["seed"],
                    request["lr"],
                    request["stage"],
                ) == (family, budget, seed, lr, "confirmation")
                assert result["traces"][-1]["epoch"] == config["epochs"]
                main_bindings[str(path.relative_to(main_root) / "receipt.json")] = (
                    sha256(path / "receipt.json")
                )
    cells = []
    for family in config["families"]:
        for budget in budgets(config, family, True):
            for seed in config["confirmation_seeds"]:
                lr = selected["choices"][family][str(budget)]["lr"]
                path = cell_path(root, "confirmation", family, budget, seed, lr)
                request, _ = read_complete(root, path, config)
                assert request["selection_sha256"] == sha256(root / "selection.json")
                assert (
                    request["family"],
                    request["budget"],
                    request["seed"],
                    request["lr"],
                    request["stage"],
                ) == (family, budget, seed, lr, "confirmation")
                cells.append((path, request))
    assert len(main_bindings) == 180 and len(cells) == 18
    return cells, main_bindings


def score_confirmation(root, device):
    root = Path(root)
    config = verify_initialized(root)
    main = import_main(config)
    cells, main_bindings = confirmation_barrier(root, config, main)
    main_root = Path(config["main_root"])
    main.verify_bundle(main_root, ("train", "validation", "test"))
    test = torch.load(
        main_root / "test_features.pt", map_location=device, weights_only=True
    )
    for path, request in cells:
        states = torch.load(path / "states.pt", map_location=device, weights_only=True)
        model = ControlHead(
            request["family"], request["budget"], seed=request["seed"]
        ).to(device)
        model.load_state_dict(states["terminal"], strict=True)
        score = main.metrics(
            model, test["features"] / states["rms"].to(device), test["labels"]
        )
        write_json(
            path / "test_metrics.json",
            {
                "metrics": score,
                "states_sha256": sha256(path / "states.pt"),
                "source_sha256": sha256(__file__),
                "selection_sha256": sha256(root / "selection.json"),
                "test_extraction_receipt_sha256": sha256(
                    main_root / "test_extraction_receipt.json"
                ),
            },
        )
    write_json(
        root / "final_test_receipt.json",
        {
            "status": "complete",
            "cells": len(cells),
            "all_main_confirmation_receipts": main_bindings,
            "test_scores": {
                str(path.relative_to(root) / "test_metrics.json"): sha256(
                    path / "test_metrics.json"
                )
                for path, _ in cells
            },
        },
    )


def run_training(root):
    """One GPU allocation, with no waiting on main confirmation or test data."""
    root = Path(root)
    config = verify_initialized(root)
    started = time.perf_counter()
    cpu_started = time.process_time()
    write_json(
        root / "training_started.json",
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": sha256(__file__),
            "protocol_sha256": sha256(root / "frozen/protocol.json"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "gpu_hour_ceiling_including_later_scoring": 1,
        },
    )
    try:
        for family in config["families"]:
            for budget in budgets(config, family):
                for seed in config["development_seeds"]:
                    for lr in config["learning_rates"]:
                        run_fit(root, "development", family, budget, seed, lr)
                        print(
                            json.dumps(
                                {
                                    "complete": True,
                                    "stage": "development",
                                    "family": family,
                                    "budget": budget,
                                    "seed": seed,
                                    "lr": lr,
                                }
                            ),
                            flush=True,
                        )
        select_recipes(root)
        for family in config["families"]:
            for budget in budgets(config, family, True):
                for seed in config["confirmation_seeds"]:
                    run_fit(root, "confirmation", family, budget, seed)
                    print(
                        json.dumps(
                            {
                                "complete": True,
                                "stage": "confirmation",
                                "family": family,
                                "budget": budget,
                                "seed": seed,
                            }
                        ),
                        flush=True,
                    )
        paths = sorted(root.glob("development/*/*/receipt.json")) + sorted(
            root.glob("confirmation/*/*/receipt.json")
        )
        assert len(paths) == config["fits"]["total"] == 28
        assert not list(root.glob("**/failure.json"))
        for path in paths:
            read_complete(root, path.parent, config)
        write_json(
            root / "training_complete.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "status": "complete",
                "fits": len(paths),
                "wall_seconds": time.perf_counter() - started,
                "process_seconds": time.process_time() - cpu_started,
                "selection_sha256": sha256(root / "selection.json"),
                "receipts": {
                    str(path.relative_to(root)): sha256(path) for path in paths
                },
                "test_access": False,
            },
        )
    except Exception as exc:
        write_json(
            root / "training_failure.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "type": type(exc).__name__,
                "error": str(exc),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "initialize",
            "development",
            "select",
            "confirmation",
            "train",
            "score",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--family", choices=["linear", "deep_relu"], default="deep_relu"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.stage == "initialize":
        initialize(args.root)
    elif args.stage == "train":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The bounded companion training release requires one GPU"
            )
        run_training(args.root)
    elif args.stage in ("development", "confirmation"):
        config = verify_initialized(args.root)
        confirmation = args.stage == "confirmation"
        for budget in budgets(config, args.family, confirmation):
            for seed in config[
                "confirmation_seeds" if confirmation else "development_seeds"
            ]:
                for lr in [None] if confirmation else config["learning_rates"]:
                    run_fit(args.root, args.stage, args.family, budget, seed, lr)
    elif args.stage == "select":
        select_recipes(args.root)
    else:
        score_confirmation(args.root, "cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
