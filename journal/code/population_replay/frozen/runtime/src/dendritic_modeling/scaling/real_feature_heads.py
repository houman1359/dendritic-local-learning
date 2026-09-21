"""Real-label ImageNet heads on a frozen local AlexNet representation.

Standalone commands are intentionally separate: index, extract, profile, fit.
The module never downloads weights/data and never selects using final test data.
"""

from __future__ import annotations

import os
import sys

# This directory also contains select.py; direct execution must not shadow the
# standard-library extension while NumPy/Torch import subprocess.
if not __package__:
    _script_directory = os.path.dirname(os.path.abspath(__file__))
    sys.path = [
        entry
        for entry in sys.path
        if os.path.abspath(entry or ".") != _script_directory
    ]

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = (
    "dense_relu",
    "dense_tanh",
    "dense_shunt",
    *(
        f"{geometry}_{activation}"
        for geometry in ("rank1", "rank2", "full")
        for activation in ("relu", "tanh", "shunt")
    ),
)
DATA_ROOT = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k"
CHECKPOINT = "/n/home13/hsafaai/.cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
CHECKPOINT_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def default_config():
    return {
        "schema": "real_feature_heads_v1",
        "families": list(FAMILIES),
        "input_dim": 256,
        "classes": 1000,
        "branches": 4,
        "development_budgets": [250000, 500000, 1000000, 2000000],
        "heldout_budget": 4000000,
        "development_seeds": [73101],
        "confirmation_seeds": [74101, 74107, 74113],
        "learning_rates": [0.0003, 0.001],
        "epochs": 50,
        "batch_size": 512,
        "optimizer": "AdamW",
        "weight_decay": 0.0001,
        "train_per_class": 64,
        "validation_per_class": 16,
        "split_seed": 72101,
        "data_root": DATA_ROOT,
        "checkpoint": CHECKPOINT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "representation": "frozen AlexNet features, adaptive_avg_pool2d(output,1), flatten to256",
        "normalization": "train-only per-feature RMS; no centering; stored256scalar buffer",
        "selection": "mean validation cross entropy at fixed50epoch endpoint; per family/budget LR; heldout4M inherits2M LR",
        "forecast": "OLS log(selected development validationCE) on log(actualP), all4 development sizes; predict4M validationCE before confirmation; descriptive finite-range extrapolation, no universal power-law claim",
        "scope": "supervised head training on pretrained frozen representation; no teacher targets and no from-scratch whole-network claim",
    }


def verify_bundle(output_dir, splits=()):
    """Check source/config/index/cache lineage before any optimization or score."""
    root = Path(output_dir)
    seal = json.loads((root / "index_receipt.json").read_text())
    assert sha256(root / "config.json") == seal["config_sha256"]
    assert sha256(__file__) == seal["source_sha256"]
    config = json.loads((root / "config.json").read_text())
    assert (config["input_dim"], config["classes"], config["branches"]) == (
        256,
        1000,
        4,
    )
    assert config["families"] == list(FAMILIES)
    for split in splits:
        index = root / f"{split}_index.json"
        assert sha256(index) == seal["files"][index.name]
        receipt_path = root / f"{split}_extraction_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["source_sha256"] == seal["source_sha256"]
        assert receipt["config_sha256"] == seal["config_sha256"]
        assert receipt["index_sha256"] == sha256(index)
        assert receipt["checkpoint_sha256"] == config["checkpoint_sha256"]
        assert receipt["feature_sha256"] == sha256(root / f"{split}_features.pt")
        assert receipt["pixel_manifest_sha256"] == sha256(
            root / f"{split}_pixel_hashes.json"
        )
    return config


def cell_path(root, stage, family, budget, seed, lr):
    return Path(root) / stage / family / f"p{budget}_seed{seed}_lr{lr:.8g}"


def read_complete_cell(path, config_hash):
    path = Path(path)
    receipt = json.loads((path / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert sha256(path / "result.json") == receipt["result_sha256"]
    assert sha256(path / "states.pt") == receipt["states_sha256"]
    assert sha256(path / "request.json") == receipt["request_sha256"]
    assert sha256(path / "initial_state.pt") == receipt["initial_state_sha256"]
    assert sha256(path / "progress.jsonl") == receipt["progress_sha256"]
    request = json.loads((path / "request.json").read_text())
    assert request["config_sha256"] == config_hash
    assert request["source_sha256"] == sha256(__file__)
    result = json.loads((path / "result.json").read_text())
    return request, result


def select_recipes(root):
    root = Path(root)
    config = verify_bundle(root, ("train", "validation"))
    selections, bindings, forecasts = {}, {}, {}
    for family in config["families"]:
        selections[family] = {}
        for budget in config["development_budgets"]:
            trials = []
            for lr in config["learning_rates"]:
                losses = []
                for seed in config["development_seeds"]:
                    path = cell_path(root, "development", family, budget, seed, lr)
                    request, result = read_complete_cell(
                        path, sha256(root / "config.json")
                    )
                    assert (
                        request["family"],
                        request["budget"],
                        request["seed"],
                        request["lr"],
                        request["epochs"],
                    ) == (family, budget, seed, lr, config["epochs"])
                    assert result["traces"][-1]["epoch"] == config["epochs"]
                    losses.append(result["traces"][-1]["validation"]["cross_entropy"])
                    bindings[str(path.relative_to(root) / "receipt.json")] = sha256(
                        path / "receipt.json"
                    )
                trials.append(
                    {
                        "lr": lr,
                        "mean_validation_cross_entropy": sum(losses) / len(losses),
                    }
                )
            choice = min(
                trials,
                key=lambda row: (row["mean_validation_cross_entropy"], row["lr"]),
            )
            selections[family][str(budget)] = {"lr": choice["lr"], "trials": trials}
        parent = max(config["development_budgets"])
        selections[family][str(config["heldout_budget"])] = {
            "lr": selections[family][str(parent)]["lr"],
            "inherited_from_budget": parent,
        }
        xs, ys = [], []
        for budget in config["development_budgets"]:
            choice = selections[family][str(budget)]
            score = next(
                t["mean_validation_cross_entropy"]
                for t in choice["trials"]
                if t["lr"] == choice["lr"]
            )
            actual = config["classes"] + (
                (budget - config["classes"]) // node_cost(family)
            ) * node_cost(family)
            xs.append(math.log(actual))
            ys.append(math.log(score))
        slope, intercept = np.linalg.lstsq(
            np.column_stack([xs, np.ones(len(xs))]), ys, rcond=None
        )[0]
        held_actual = config["classes"] + (
            (config["heldout_budget"] - config["classes"]) // node_cost(family)
        ) * node_cost(family)
        forecasts[family] = {
            "heldout_actual_parameters": held_actual,
            "slope": float(slope),
            "intercept": float(intercept),
            "predicted_validation_cross_entropy": math.exp(
                float(intercept + slope * math.log(held_actual))
            ),
            "scope": "confirmation-seed mean validationCE at4M; point forecast without acceptance band or asymptotic claim",
        }
    write_json(
        root / "selection.json",
        {
            "source_sha256": sha256(__file__),
            "config_sha256": sha256(root / "config.json"),
            "all_development_receipts": bindings,
            "choices": selections,
            "forecasts": forecasts,
            "test_access": False,
        },
    )
    write_json(
        root / "selection_complete.json",
        {
            "status": "complete",
            "selection_sha256": sha256(root / "selection.json"),
            "source_sha256": sha256(__file__),
            "config_sha256": sha256(root / "config.json"),
        },
    )


def load_selection(root):
    root = Path(root)
    seal = json.loads((root / "selection_complete.json").read_text())
    assert seal["status"] == "complete"
    assert seal["selection_sha256"] == sha256(root / "selection.json")
    assert seal["source_sha256"] == sha256(__file__)
    assert seal["config_sha256"] == sha256(root / "config.json")
    selected = json.loads((root / "selection.json").read_text())
    assert selected["source_sha256"] == sha256(__file__)
    assert selected["config_sha256"] == sha256(root / "config.json")
    config = json.loads((root / "config.json").read_text())
    expected = {
        str(
            cell_path(root, "development", f, b, s, lr).relative_to(root)
            / "receipt.json"
        )
        for f in config["families"]
        for b in config["development_budgets"]
        for s in config["development_seeds"]
        for lr in config["learning_rates"]
    }
    assert set(selected["all_development_receipts"]) == expected
    assert set(selected["forecasts"]) == set(config["families"])
    for path, digest in selected["all_development_receipts"].items():
        assert sha256(root / path) == digest
    return selected


def node_cost(family, input_dim=256, classes=1000, branches=4):
    geometry = family.split("_")[0]
    if geometry == "dense":
        return input_dim + 1 + classes
    if geometry == "full":
        return branches * (input_dim + 2) + classes
    rank = int(geometry.removeprefix("rank"))
    return input_dim * rank + branches * (rank + 2) + classes


class FeatureHead(nn.Module):
    def __init__(
        self, family, budget, *, input_dim=256, classes=1000, branches=4, seed=0
    ):
        super().__init__()
        if family not in FAMILIES and family != "rank1_generic":
            raise ValueError(family)
        self.family = family
        self.geometry, self.activation = family.split("_")
        self.input_dim, self.classes, self.branches = input_dim, classes, branches
        self.width = (budget - classes) // node_cost(
            family, input_dim, classes, branches
        )
        if self.width < 1:
            raise ValueError("Budget cannot fit a complete unit and output bias")
        # Preserve callers' RNG; all random trainable layers are seed-bound.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            if self.geometry == "dense":
                self.body = nn.Linear(input_dim, self.width)
            elif self.geometry == "full":
                self.direction = nn.Parameter(
                    torch.randn(self.width, branches, input_dim) / math.sqrt(input_dim)
                )
                self.threshold = nn.Parameter(torch.zeros(self.width, branches))
                self.branch_readout = nn.Parameter(
                    torch.randn(self.width, branches) / math.sqrt(branches)
                )
            else:
                rank = int(self.geometry.removeprefix("rank"))
                self.direction = nn.Parameter(
                    torch.randn(self.width, rank, input_dim) / math.sqrt(input_dim)
                )
                self.mixing = nn.Parameter(
                    torch.randn(self.width, branches, rank) / math.sqrt(rank)
                )
                self.threshold = nn.Parameter(torch.zeros(self.width, branches))
                self.branch_readout = nn.Parameter(
                    torch.randn(self.width, branches) / math.sqrt(branches)
                )
            self.output = nn.Linear(self.width, classes)
        self.actual_parameters = sum(p.numel() for p in self.parameters())
        assert (
            self.actual_parameters
            == self.width * node_cost(family, input_dim, classes, branches) + classes
        )
        assert self.actual_parameters <= budget

    def nonlinearity(self, x):
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "tanh":
            return x.tanh()
        positive = F.relu(x)
        if self.activation == "generic":
            return positive / (1 + positive)
        return positive / (1 + positive)

    def forward(self, x):
        if self.geometry == "dense":
            nodes = self.nonlinearity(self.body(x))
        else:
            projected = F.linear(x, self.direction.flatten(0, 1)).view(
                x.shape[0], self.width, -1
            )
            if self.geometry != "full":
                projected = torch.einsum("bnr,nsr->bns", projected, self.mixing)
            branches = self.nonlinearity(projected + self.threshold)
            nodes = (branches * self.branch_readout).sum(-1)
        return self.output(nodes)


def split_class_files(paths, *, class_index, seed, n_train, n_valid):
    """Path-only seeded partition; labels, pixels and outcomes do not rank rows."""
    paths = sorted(str(p) for p in paths)
    if len(paths) < n_train + n_valid:
        raise ValueError("Insufficient distinct training images")
    rng = np.random.default_rng(np.random.SeedSequence([seed, class_index]))
    order = rng.permutation(len(paths))[: n_train + n_valid]
    return ([paths[i] for i in order[:n_train]], [paths[i] for i in order[n_train:]])


def build_indexes(output_dir, config):
    output_dir = Path(output_dir)
    if (output_dir / "index_receipt.json").exists():
        raise FileExistsError(output_dir)
    root = Path(config["data_root"])
    classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    assert len(classes) == config["classes"]
    assert classes == sorted(p.name for p in (root / "val").iterdir() if p.is_dir())
    rows = {"train": [], "validation": [], "test": []}
    for label, cls in enumerate(classes):
        files = [
            p
            for p in (root / "train" / cls).iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        train, validation = split_class_files(
            files,
            class_index=label,
            seed=config["split_seed"],
            n_train=config["train_per_class"],
            n_valid=config["validation_per_class"],
        )
        for split, paths in [("train", train), ("validation", validation)]:
            rows[split].extend({"path": p, "label": label, "wnid": cls} for p in paths)
        tests = sorted(
            p
            for p in (root / "val" / cls).iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        assert len(tests) == 50
        rows["test"].extend(
            {"path": str(p), "label": label, "wnid": cls} for p in tests
        )
    for split in rows:
        assert len({r["path"] for r in rows[split]}) == len(rows[split])
        write_json(output_dir / f"{split}_index.json", rows[split])
    assert not (
        {r["path"] for r in rows["train"]} & {r["path"] for r in rows["validation"]}
    )
    write_json(output_dir / "config.json", config)
    write_json(
        output_dir / "index_receipt.json",
        {
            "split_sizes": {s: len(r) for s, r in rows.items()},
            "files": {
                f"{s}_index.json": sha256(output_dir / f"{s}_index.json") for s in rows
            },
            "config_sha256": sha256(output_dir / "config.json"),
            "source_sha256": sha256(__file__),
            "pixel_hash_scope": "pixels hashed during extraction",
            "data_selection": "train-only path permutation; all official validation is final test",
        },
    )


class ImageRows(torch.utils.data.Dataset):
    def __init__(self, rows):
        from torchvision import transforms

        self.rows = rows
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image

        row = self.rows[i]
        with Image.open(row["path"]) as im:
            value = self.transform(im.convert("RGB"))
        return value, int(row["label"]), sha256(row["path"])


def load_encoder(config, device):
    from torchvision import models

    if sha256(config["checkpoint"]) != config["checkpoint_sha256"]:
        raise ValueError("Local pretrained checkpoint checksum mismatch")
    model = models.alexnet(weights=None)
    model.load_state_dict(
        torch.load(config["checkpoint"], map_location="cpu", weights_only=True),
        strict=True,
    )
    encoder = model.features.eval().requires_grad_(False).to(device)
    assert sum(p.numel() for p in encoder.parameters()) == 2469696
    return encoder


def extract(output_dir, split, device, workers=4):
    output_dir = Path(output_dir)
    destination = output_dir / f"{split}_features.pt"
    if destination.exists():
        raise FileExistsError(destination)
    if split == "test":
        # Pixel access requires all development choices and4M predictions frozen.
        load_selection(output_dir)
    config = json.loads((output_dir / "config.json").read_text())
    seal = json.loads((output_dir / "index_receipt.json").read_text())
    assert sha256(output_dir / "config.json") == seal["config_sha256"]
    index_path = output_dir / f"{split}_index.json"
    assert sha256(index_path) == seal["files"][index_path.name]
    rows = json.loads(index_path.read_text())
    loader = torch.utils.data.DataLoader(
        ImageRows(rows),
        batch_size=128,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
    )
    encoder = load_encoder(config, device)
    xs, ys, hashes = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels, image_hashes in loader:
            features = F.adaptive_avg_pool2d(encoder(images.to(device)), 1).flatten(1)
            assert features.shape[1] == 256 and torch.isfinite(features).all()
            xs.append(features.cpu())
            ys.append(labels)
            hashes.extend(image_hashes)
    torch.save({"features": torch.cat(xs), "labels": torch.cat(ys)}, destination)
    write_json(output_dir / f"{split}_pixel_hashes.json", hashes)
    write_json(
        output_dir / f"{split}_extraction_receipt.json",
        {
            "count": len(rows),
            "shape": [len(rows), 256],
            "dtype": "float32",
            "device": device,
            "wall_seconds": time.perf_counter() - started,
            "checkpoint_sha256": config["checkpoint_sha256"],
            "index_sha256": sha256(index_path),
            "source_sha256": sha256(__file__),
            "config_sha256": sha256(output_dir / "config.json"),
            "feature_sha256": sha256(destination),
            "pixel_manifest_sha256": sha256(output_dir / f"{split}_pixel_hashes.json"),
            "test_access": (
                "frozen encoder only; no head scoring or selection"
                if split == "test"
                else None
            ),
        },
    )


def feature_rms(train):
    return train.square().mean(0).sqrt().clamp_min(1e-6)


@torch.no_grad()
def metrics(model, x, y, batch_size=1024):
    loss, correct = 0.0, 0
    for start in range(0, len(x), batch_size):
        logits = model(x[start : start + batch_size])
        labels = y[start : start + batch_size]
        loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += int((logits.argmax(1) == labels).sum())
    return {"cross_entropy": loss / len(x), "accuracy": correct / len(x), "n": len(x)}


def fit_model(
    model,
    train_x,
    train_y,
    valid_x,
    valid_y,
    *,
    lr,
    epochs,
    batch_size,
    weight_decay,
    seed,
    progress=None,
):
    """Only TRAIN and validation accepted; final endpoint, no checkpoint selection."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    traces = [
        {
            "epoch": 0,
            "train": metrics(model, train_x, train_y),
            "validation": metrics(model, valid_x, valid_y),
        }
    ]
    if progress is not None:
        progress({"epoch_metrics": traces[0]})
    steps = []
    initial = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train_x), generator=generator)
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size].to(train_x.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(train_x[indices]), train_y[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            gradnorm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float("inf"), error_if_nonfinite=True
            )
            optimizer.step()
            steps.append(
                {
                    "step": len(steps) + 1,
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradnorm),
                }
            )
            if progress is not None:
                progress({"step_metrics": steps[-1]})
        model.eval()
        traces.append(
            {
                "epoch": epoch,
                "train": metrics(model, train_x, train_y),
                "validation": metrics(model, valid_x, valid_y),
            }
        )
        if progress is not None:
            progress({"epoch_metrics": traces[-1]})
    movement = {
        name: float((value.detach().cpu() - initial[name]).norm())
        for name, value in model.state_dict().items()
    }
    return {
        "traces": traces,
        "steps": steps,
        "parameter_movement": movement,
        "wall_seconds": time.perf_counter() - started,
    }, initial


def run_fit(
    output_dir, destination, family, budget, seed, lr, epochs=None, stage="development"
):
    output_dir = Path(output_dir)
    config = verify_bundle(output_dir, ("train", "validation"))
    if family not in config["families"]:
        raise ValueError("Unregistered family")
    if epochs is not None and epochs != config["epochs"]:
        raise ValueError("Scientific fits must retain frozen exposure")
    if stage == "development":
        assert (
            budget in config["development_budgets"]
            and seed in config["development_seeds"]
        )
        assert lr in config["learning_rates"]
    elif stage == "confirmation":
        assert budget in config["development_budgets"] + [config["heldout_budget"]]
        assert seed in config["confirmation_seeds"]
        selected = load_selection(output_dir)
        selected_lr = selected["choices"][family][str(budget)]["lr"]
        if lr is not None and lr != selected_lr:
            raise ValueError("Confirmation must use development-selected learning rate")
        lr = selected_lr
    else:
        raise ValueError(stage)
    canonical = cell_path(output_dir, stage, family, budget, seed, lr)
    if destination is not None and Path(destination).resolve() != canonical.resolve():
        raise ValueError("Scientific cell output must use its canonical immutable path")
    destination = canonical
    if destination.exists():
        raise FileExistsError(destination)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {}
    for split in ["train", "validation"]:
        p = output_dir / f"{split}_features.pt"
        seal = json.loads((output_dir / f"{split}_extraction_receipt.json").read_text())
        assert sha256(p) == seal["feature_sha256"]
        loaded[split] = torch.load(p, map_location=device, weights_only=True)
    rms = feature_rms(loaded["train"]["features"])
    model = FeatureHead(family, budget, seed=seed).to(device)
    destination.mkdir(parents=True)
    write_json(
        destination / "request.json",
        {
            "family": family,
            "budget": budget,
            "seed": seed,
            "lr": lr,
            "epochs": epochs or config["epochs"],
            "source_sha256": sha256(__file__),
            "config_sha256": sha256(output_dir / "config.json"),
            "actual_head_parameters": model.actual_parameters,
            "frozen_backbone_parameters": 2469696,
            "fixed_rms_buffer_scalars": 256,
            "stage": stage,
            "index_receipt_sha256": sha256(output_dir / "index_receipt.json"),
            "train_extraction_receipt_sha256": sha256(
                output_dir / "train_extraction_receipt.json"
            ),
            "validation_extraction_receipt_sha256": sha256(
                output_dir / "validation_extraction_receipt.json"
            ),
            "selection_sha256": (
                sha256(output_dir / "selection.json")
                if stage == "confirmation"
                else None
            ),
        },
    )
    torch.save(
        {"state": model.state_dict(), "rms": rms.cpu()},
        destination / "initial_state.pt",
    )
    progress_stream = (destination / "progress.jsonl").open("w")

    def progress(row):
        progress_stream.write(json.dumps(row, allow_nan=False) + "\n")
        if "epoch_metrics" in row:
            progress_stream.flush()

    try:
        result, initial = fit_model(
            model,
            loaded["train"]["features"] / rms,
            loaded["train"]["labels"],
            loaded["validation"]["features"] / rms,
            loaded["validation"]["labels"],
            lr=lr,
            epochs=epochs or config["epochs"],
            batch_size=config["batch_size"],
            weight_decay=config["weight_decay"],
            seed=seed,
            progress=progress,
        )
    except Exception as exc:
        torch.save(model.state_dict(), destination / "failed_state.pt")
        write_json(
            destination / "failure.json",
            {"type": type(exc).__name__, "error": str(exc)},
        )
        raise
    finally:
        progress_stream.close()
    torch.save(
        {"initial": initial, "terminal": model.state_dict(), "rms": rms.cpu()},
        destination / "states.pt",
    )
    write_json(destination / "result.json", result)
    write_json(
        destination / "receipt.json",
        {
            "status": "complete",
            "states_sha256": sha256(destination / "states.pt"),
            "result_sha256": sha256(destination / "result.json"),
            "actual_parameters": model.actual_parameters,
            "request_sha256": sha256(destination / "request.json"),
            "initial_state_sha256": sha256(destination / "initial_state.pt"),
            "progress_sha256": sha256(destination / "progress.jsonl"),
        },
    )


def score_confirmation(root, device):
    """Global confirmation completion barrier precedes the first final test score."""
    root = Path(root)
    config = verify_bundle(root, ("train", "validation", "test"))
    selected = load_selection(root)
    cells = []
    for family in config["families"]:
        for budget in config["development_budgets"] + [config["heldout_budget"]]:
            for seed in config["confirmation_seeds"]:
                lr = selected["choices"][family][str(budget)]["lr"]
                path = cell_path(root, "confirmation", family, budget, seed, lr)
                request, result = read_complete_cell(path, sha256(root / "config.json"))
                assert request["selection_sha256"] == sha256(root / "selection.json")
                assert (
                    request["family"],
                    request["budget"],
                    request["seed"],
                    request["lr"],
                    request["stage"],
                ) == (family, budget, seed, lr, "confirmation")
                assert result["traces"][-1]["epoch"] == config["epochs"]
                cells.append((path, request))
    # Test tensors are first loaded only after every confirmation endpoint passes.
    test = torch.load(root / "test_features.pt", map_location=device, weights_only=True)
    for path, request in cells:
        destination = path / "test_metrics.json"
        if destination.exists():
            raise FileExistsError(destination)
        states = torch.load(path / "states.pt", map_location=device, weights_only=True)
        model = FeatureHead(
            request["family"], request["budget"], seed=request["seed"]
        ).to(device)
        model.load_state_dict(states["terminal"], strict=True)
        score = metrics(
            model, test["features"] / states["rms"].to(device), test["labels"]
        )
        write_json(
            destination,
            {
                "metrics": score,
                "source_sha256": sha256(__file__),
                "states_sha256": sha256(path / "states.pt"),
                "selection_sha256": sha256(root / "selection.json"),
                "test_extraction_receipt_sha256": sha256(
                    root / "test_extraction_receipt.json"
                ),
            },
        )
    write_json(
        root / "final_test_receipt.json",
        {
            "status": "complete",
            "cells": len(cells),
            "test_scores": {
                str(p.relative_to(root) / "test_metrics.json"): sha256(
                    p / "test_metrics.json"
                )
                for p, _ in cells
            },
        },
    )


def profile(output_dir, device, steps=20):
    """Batch512 profile uses only predeclared TRAIN images and2M dev P."""
    output_dir = Path(output_dir)
    if (output_dir / "gpu_profile_receipt.json").exists():
        raise FileExistsError(output_dir)
    config = default_config()
    root = Path(config["data_root"])
    classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    rows = []
    for label, cls in enumerate(classes[:8]):
        files = [
            p
            for p in (root / "train" / cls).iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        train, _ = split_class_files(
            files, class_index=label, seed=config["split_seed"], n_train=64, n_valid=16
        )
        rows.extend({"path": p, "label": label, "wnid": cls} for p in train)
    loader = torch.utils.data.DataLoader(
        ImageRows(rows), batch_size=64, num_workers=4, shuffle=False
    )
    encoder = load_encoder(config, device)
    started = time.perf_counter()
    xs, ys = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            xs.append(F.adaptive_avg_pool2d(encoder(images.to(device)), 1).flatten(1))
            ys.append(labels.to(device))
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    extraction_seconds = time.perf_counter() - started
    x, y = torch.cat(xs), torch.cat(ys)
    x = x / feature_rms(x)
    records = []
    for family in FAMILIES:
        model = FeatureHead(family, 2000000, seed=73101).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        losses = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            assert torch.isfinite(loss)
            losses.append(float(loss.detach()))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        records.append(
            {
                "family": family,
                "budget": 2000000,
                "actual_parameters": model.actual_parameters,
                "width": model.width,
                "steps": steps,
                "batch_size": len(x),
                "wall_seconds": time.perf_counter() - started,
                "losses": losses,
                "gpu_peak_bytes": (
                    torch.cuda.max_memory_allocated()
                    if device.startswith("cuda")
                    else 0
                ),
            }
        )
    write_json(
        output_dir / "gpu_profile_receipt.json",
        {
            "status": "passed",
            "source_sha256": sha256(__file__),
            "device": device,
            "device_name": (
                torch.cuda.get_device_name() if device.startswith("cuda") else "cpu"
            ),
            "torch_version": torch.__version__,
            "extraction_seconds": extraction_seconds,
            "extraction_images": len(x),
            "image_scope": "all64 predefinedTRAIN images in first8classes; batch512; no validation or test access; no model selection",
            "records": records,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "scope": "engineering throughput/finite-gradient canary; train-loss decreases are not generalization evidence",
        },
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=["index", "extract", "profile", "fit", "config", "select", "score"],
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], default="train")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--destination")
    p.add_argument("--family", choices=FAMILIES, default="dense_relu")
    p.add_argument("--budget", type=int, default=250000)
    p.add_argument("--seed", type=int, default=73101)
    p.add_argument("--lr", type=float)
    p.add_argument(
        "--stage", choices=["development", "confirmation"], default="development"
    )
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.command == "config":
        write_json(Path(args.output_dir) / "proposed_config.json", default_config())
    elif args.command == "index":
        build_indexes(args.output_dir, default_config())
    elif args.command == "extract":
        extract(args.output_dir, args.split, args.device, args.workers)
    elif args.command == "profile":
        profile(args.output_dir, args.device)
    elif args.command == "select":
        select_recipes(args.output_dir)
    elif args.command == "score":
        score_confirmation(args.output_dir, args.device)
    else:
        lr = 0.001 if args.lr is None and args.stage == "development" else args.lr
        run_fit(
            args.output_dir,
            args.destination,
            args.family,
            args.budget,
            args.seed,
            lr,
            stage=args.stage,
        )


if __name__ == "__main__":
    main()
