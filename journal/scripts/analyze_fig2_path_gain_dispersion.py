#!/usr/bin/env python3
"""Within-neuron path-gain dispersion at the Figure 2 ladder checkpoints.

Figure 2E originally reused the NeurIPS-era cohort behind Supplementary
Fig. S1A, whose additive condition is the *normalized*-additive model, while
every other panel of Figure 2 shows the raw-additive model.  This analyzer
computes the same within-neuron dispersion on Figure 2's OWN trained models:
the exact-path condition of the MNIST feedback ladder, fifteen paired seeds
per architecture, the very checkpoints whose accuracies stand in panel B.

Definition (stated in Methods): at the final checkpoint, on one fixed
held-out batch (the first 2048 MNIST test images), the loss gradient
``g_n = dL/dV_n`` is taken at every dendritic stage output and at the soma
by automatic differentiation.  The transported magnitude of compartment
``n`` is the batch root-mean-square of ``g_n`` divided by the batch
root-mean-square of its own soma's ``g_0`` -- the empirical counterpart of
the path gain in the manuscript's transport factorization, with no
model-specific terms, so both architectures are measured identically.  The
statistic is the coefficient of variation of that magnitude across the
neuron's dendritic compartments, averaged over neurons.

Stage tensors are soma-major: each stage of width ``n_soma * b`` chunks
into ``n_soma`` contiguous blocks of ``b`` branches (verified against the
weight layout used by ``DendriNet.sum_weights`` and empirically by the
cross-stage gradient correlation, 0.42 soma-major vs 0.03 branch-major).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(REPO / "src"))

from dendritic_modeling.config import load_config  # noqa: E402
from dendritic_modeling.scripts.script_utils.setup_utils import (  # noqa: E402
    initialize_model,
)

LADDER = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "journal_extension_20260820/sweep_runs/mnist_feedback_ladder"
)
SWEEPS = {
    "additive": "journal_mnist_feedback_ladder_exact_path_additive_15seed_20260825173519",
    "shunting": "journal_mnist_feedback_ladder_exact_path_shunting_15seed_20260825173517",
}
MNIST_RAW = REPO / "data" / "mnist" / "MNIST" / "raw"
BATCH = 2048
OUT = ROOT / "source_data" / "figure2" / "path_gain_dispersion_ladder_runs.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_batch() -> tuple[torch.Tensor, torch.Tensor]:
    raw = np.fromfile(MNIST_RAW / "t10k-images-idx3-ubyte", dtype=np.uint8)
    lab = np.fromfile(MNIST_RAW / "t10k-labels-idx1-ubyte", dtype=np.uint8)
    x = torch.tensor(raw[16:].reshape(-1, 784)[:BATCH], dtype=torch.float32) / 255.0
    y = torch.tensor(lab[8:][:BATCH], dtype=torch.long)
    return x, y


def run_cv(run_dir: Path, x: torch.Tensor, y: torch.Tensor) -> dict:
    config = load_config(str(run_dir / "config.json"))
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(int(config.experiment.model_seed))
        model, _ = initialize_model(config.model)
    state = torch.load(run_dir / "final_model.pt", map_location="cpu",
                       weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    dn = model.core_network.layers[0].excitatory_cells
    n_soma, n_stage = int(dn.n_soma), int(dn.n_branch_layers) + 1
    caught: dict[int, torch.Tensor] = {}

    def make_hook(i):
        def hook(_mod, _inp, out):
            t = out if torch.is_tensor(out) else out[0]
            t.retain_grad()
            caught[i] = t
        return hook

    handles = [dn.branch_layers[i].register_forward_hook(make_hook(i))
               for i in range(n_stage)]
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    for handle in handles:
        handle.remove()
    accuracy = float((logits.argmax(1) == y).float().mean())
    if accuracy < 0.90:
        raise RuntimeError(f"{run_dir}: probe accuracy {accuracy:.3f}")

    soma_rms = caught[n_stage - 1].grad.pow(2).mean(0).sqrt()   # [n_soma]
    if not torch.isfinite(soma_rms).all() or (soma_rms == 0).any():
        raise RuntimeError(f"{run_dir}: degenerate somatic gradient")
    ratios = []                                # [n_soma, total branches]
    for i in range(n_stage - 1):
        g = caught[i].grad
        if not torch.isfinite(g).all():
            raise RuntimeError(f"{run_dir}: non-finite stage-{i} gradient")
        rms = g.pow(2).mean(0).sqrt().reshape(n_soma, -1)       # soma-major
        ratios.append(rms / soma_rms[:, None])
    profile = torch.cat(ratios, dim=1)
    cv = (profile.std(dim=1, unbiased=True)
          / profile.mean(dim=1)).numpy()
    return {
        "seed": int(config.experiment.seed),
        "path_gain_cv_mean": float(np.mean(cv)),
        "path_gain_cv_median": float(np.median(cv)),
        "compartments_per_neuron": int(profile.shape[1]),
        "n_soma": n_soma,
        "probe_accuracy": accuracy,
        "config_sha256": sha256(run_dir / "config.json"),
        "checkpoint_sha256": sha256(run_dir / "final_model.pt"),
        # Relative to the lab run base, as the release path audit requires
        # of every packaged table.
        "run_dir": str(run_dir).split("/LOCAL_LEARNING/", 1)[1],
    }


def main() -> None:
    torch.set_grad_enabled(True)
    x, y = probe_batch()
    ladder = pd.read_csv(ROOT / "source_data" / "mnist_feedback_ladder"
                         / "seed_outcomes.csv")
    exact = ladder[ladder.feedback.eq("exact path")]
    rows = []
    for arch, sweep in SWEEPS.items():
        results = LADDER / sweep / "results"
        for cfg in sorted(results.iterdir(),
                          key=lambda p: int(p.name.split("_")[1])):
            row = run_cv(cfg, x, y)
            row["architecture"] = arch
            claim = exact[exact.architecture.eq(arch)
                          & exact.seed.eq(row["seed"])]
            if len(claim) != 1 or claim.iloc[0].checkpoint_sha256 != \
                    row["checkpoint_sha256"]:
                raise RuntimeError(
                    f"{cfg}: checkpoint is not the Figure 2 ladder checkpoint")
            rows.append(row)
            print(f"  {arch} seed {row['seed']}: "
                  f"CV {row['path_gain_cv_mean']:.3f} "
                  f"(acc {row['probe_accuracy']:.3f})")
    frame = pd.DataFrame(rows)
    frame = frame[["architecture", "seed", "path_gain_cv_mean",
                   "path_gain_cv_median", "compartments_per_neuron",
                   "n_soma", "probe_accuracy", "config_sha256",
                   "checkpoint_sha256", "run_dir"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(frame)} rows)")
    print(frame.groupby("architecture").path_gain_cv_mean
          .agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()
