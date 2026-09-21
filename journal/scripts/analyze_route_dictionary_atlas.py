#!/usr/bin/env python3
"""Route-dictionary capture atlas at the Figure 2 exact-path checkpoints.

On the paper's own trained MNIST ``[3, 3]`` exact-path checkpoints (the
fifteen paired seeds per dynamics whose accuracies stand in Figure 2B), this
analyzer asks how much of the per-example within-neuron compartment-error
field a fixed route dictionary can carry.  For one example and one neuron the
field ``d`` is the 12-vector of loss gradients ``dL/dV`` over that neuron's
routed compartments (3 proximal + 9 distal in the fixed ``[3, 3]`` ladder;
the soma is the reference point, not a routed compartment, and is excluded).
For an address matrix ``A`` whose columns are the dictionary's routes, the
captured fraction is ``||P_A d||^2 / ||d||^2`` with ``P_A`` the orthogonal
projector onto ``col(A)``.  Three nested dictionaries are scored:

* ``broadcast_k1`` -- one all-ones column over the 12 routed compartments,
  the single shared scalar of a global broadcast;
* ``subtrees_k3`` -- one indicator column per proximal compartment, covering
  itself and its 3 distal children (the neuron's three anatomical subtrees);
* ``exact_k12`` -- the identity, whose capture is 1 by definition and is
  asserted to numerical tolerance as a self-check of the projector code.

The spans are nested (the broadcast column is the sum of the three subtree
columns), so ``subtrees_k3 >= broadcast_k1`` must hold field-by-field and is
asserted per seed.  Zero-norm fields are dropped before averaging; each
seed's statistic is the mean captured fraction over its remaining
example-neuron fields, and the training seed remains the inferential unit.

Outputs (all under ``source_data/route_dictionary_atlas/``):
``capture_by_seed.csv`` holds one row per checkpoint -- the seed-level mean
captures, probe accuracy, field counts, per-slot ``|dL/dV|`` sums and
checkpoint hashes; ``capture_summary.csv`` holds the per-dynamics seed means
with their Student t intervals; ``example_field.csv`` holds the
population-mean per-compartment ``|dL/dV|`` profile.  After writing, the
script re-reads ``capture_by_seed.csv`` and asserts that both derived tables
are reproduced from it to ``1e-12``, so the displayed summary is never
detached from its seed-level source.

Compartment ordering of the 12-vector (documented here, used everywhere in
this script): slots 0-2 are the three proximal compartments (stage 1) in
soma-major order, and slots ``3 + 3*s + c`` for ``s, c in {0, 1, 2}`` are the
distal compartments (stage 0), where ``s`` is the parent proximal index and
``c`` the child within that subtree.  The parentage follows ``BlockLinear``:
its ``branches_to_output`` maps contiguous input chunks of ``block_size`` to
one output branch, so with the soma-major stage layout verified by
``analyze_fig2_path_gain_dispersion.py`` the distal branch with local index
``j`` (0-8) within a neuron feeds proximal branch ``j // 3`` of that neuron.

Checkpoint loading, the fixed 2048-example held-out probe batch, the
pre-reactivation voltage capture and the ladder-membership audit all mirror
``analyze_fig2_path_gain_dispersion.py`` exactly (its ``probe_batch`` and
``sha256`` helpers are imported; the model loading and gradient capture are
copied from its ``run_cv`` with a comment naming the origin).  Everything is
deterministic on CPU: the probe batch is fixed, gradients are exact, and the
95% interval is a Student t interval over per-seed means, so no resampling
randomness enters the frozen outputs.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "src"))

from analyze_fig2_path_gain_dispersion import (  # noqa: E402
    BATCH,
    LADDER,
    SWEEPS,
    probe_batch,
    sha256,
)
from dendritic_modeling.config import load_config  # noqa: E402
from dendritic_modeling.scripts.script_utils.setup_utils import (  # noqa: E402
    initialize_model,
)

N_PROXIMAL = 3          # branches at stage 1 (depth 1) per neuron
N_CHILDREN = 3          # distal children per proximal branch (stage 0)
N_ROUTED = N_PROXIMAL * (1 + N_CHILDREN)   # 12 routed compartments
OUT_DIR = ROOT / "source_data" / "route_dictionary_atlas"
BASES = ("broadcast_k1", "subtrees_k3", "exact_k12")
# Leading columns of capture_by_seed.csv; every other scalar the per-seed
# frame holds follows in frame order, then the 12 per-slot |dL/dV| sums.
PER_SEED_LEADING_COLUMNS = (
    "dynamics", "seed", "checkpoint_sha256", "probe_accuracy", "n_fields",
    "capture_broadcast_k1", "capture_subtrees_k3", "capture_exact_k12",
)
SLOT_SUM_COLUMNS = tuple(f"abs_error_sum_slot_{slot:02d}"
                         for slot in range(N_ROUTED))
REPRODUCTION_TOLERANCE = 1e-12


def route_dictionaries() -> dict[str, torch.Tensor]:
    """Address matrices over the 12 routed compartments, in the slot order
    documented in the module docstring (0-2 proximal, 3+3s+c distal)."""
    broadcast = torch.ones(N_ROUTED, 1, dtype=torch.float64)
    subtrees = torch.zeros(N_ROUTED, N_PROXIMAL, dtype=torch.float64)
    for s in range(N_PROXIMAL):
        subtrees[s, s] = 1.0
        for c in range(N_CHILDREN):
            subtrees[N_PROXIMAL + N_CHILDREN * s + c, s] = 1.0
    # Nested spans: the broadcast column is the sum of the subtree columns.
    assert torch.equal(subtrees.sum(dim=1, keepdim=True), broadcast)
    exact = torch.eye(N_ROUTED, dtype=torch.float64)
    return {"broadcast_k1": broadcast, "subtrees_k3": subtrees,
            "exact_k12": exact}


def capture_fractions(fields: torch.Tensor, address: torch.Tensor
                      ) -> torch.Tensor:
    """``||P_A d||^2 / ||d||^2`` per row of ``fields`` (rows are fields d)."""
    q, _ = torch.linalg.qr(address)            # orthonormal basis of col(A)
    captured = (fields @ q).square().sum(dim=1)
    return captured / fields.square().sum(dim=1)


def run_atlas(run_dir: Path, x: torch.Tensor, y: torch.Tensor) -> dict:
    # --- Model loading and pre-reactivation gradient capture copied from
    # run_cv in analyze_fig2_path_gain_dispersion.py (the origin of every
    # convention in this block); only the per-stage statistics differ. ---
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
    layers = list(dn.branch_layers[:n_stage])
    caught: dict[int, torch.Tensor] = {}

    # A branch layer returns a(V), not V: enable the analysis-current capture
    # so each stage retains the pre-reactivation voltage in the forward graph,
    # and verify the diagnostic path leaves the logits unchanged.
    with torch.no_grad():
        reference_logits = model(x)
    previous_capture = [
        bool(getattr(layer, "_store_analysis_currents", False))
        for layer in layers
    ]
    for layer in layers:
        layer._store_analysis_currents = True
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        forward_max_abs_difference = float(
            (logits.detach() - reference_logits).abs().max()
        )
        if forward_max_abs_difference > 1e-5:
            raise RuntimeError(
                f"{run_dir}: analysis capture changed logits by "
                f"{forward_max_abs_difference:.3g}"
            )
        for i, layer in enumerate(layers):
            currents = getattr(layer, "_last_analysis_currents", None)
            voltage = (
                currents.get("pre_gate_voltage")
                if isinstance(currents, dict)
                else None
            )
            if not torch.is_tensor(voltage) or not voltage.requires_grad:
                raise RuntimeError(
                    f"{run_dir}: stage-{i} pre-reactivation voltage was not captured"
                )
            voltage.retain_grad()
            caught[i] = voltage
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
    finally:
        for layer, previous in zip(layers, previous_capture, strict=True):
            layer._store_analysis_currents = previous
            if hasattr(layer, "_last_analysis_currents"):
                delattr(layer, "_last_analysis_currents")

    accuracy = float((logits.argmax(1) == y).float().mean())
    if accuracy < 0.90:
        raise RuntimeError(f"{run_dir}: probe accuracy {accuracy:.3f}")
    # --- End of the block copied from analyze_fig2_path_gain_dispersion. ---

    if n_stage != 3:
        raise RuntimeError(f"{run_dir}: expected a [3, 3] ladder, "
                           f"got {n_stage - 1} dendritic stages")
    for i in (0, 1):
        if not torch.isfinite(caught[i].grad).all():
            raise RuntimeError(f"{run_dir}: non-finite stage-{i} gradient")
    # Stage tensors are soma-major (verified in the origin script): stage 1
    # is proximal (n_soma blocks of 3), stage 0 distal (n_soma blocks of 9),
    # and distal local index j feeds proximal local index j // 3.
    g_prox = caught[1].grad.double().reshape(BATCH, n_soma, -1)
    g_dist = caught[0].grad.double().reshape(BATCH, n_soma, -1)
    if g_prox.shape[2] != N_PROXIMAL or \
            g_dist.shape[2] != N_PROXIMAL * N_CHILDREN:
        raise RuntimeError(
            f"{run_dir}: unexpected stage widths "
            f"{g_prox.shape[2]}/{g_dist.shape[2]} per neuron")
    fields = torch.cat([g_prox, g_dist], dim=2).reshape(-1, N_ROUTED)

    norms_sq = fields.square().sum(dim=1)
    kept = norms_sq > 0
    n_dropped = int((~kept).sum())
    fields = fields[kept]
    if fields.shape[0] == 0:
        raise RuntimeError(f"{run_dir}: every compartment-error field is zero")

    captures = {name: capture_fractions(fields, address)
                for name, address in route_dictionaries().items()}
    exact_error = float((captures["exact_k12"] - 1.0).abs().max())
    if exact_error > 1e-9:
        raise RuntimeError(
            f"{run_dir}: exact_k12 capture deviates from 1 by {exact_error:.3g}")
    nesting_slack = float(
        (captures["subtrees_k3"] - captures["broadcast_k1"]).min())
    if nesting_slack < -1e-12:
        raise RuntimeError(
            f"{run_dir}: nested-span violation, "
            f"subtrees_k3 - broadcast_k1 >= {nesting_slack:.3g}")

    row = {
        "seed": int(config.experiment.seed),
        "n_fields": int(fields.shape[0]),
        "n_dropped_zero_norm_fields": n_dropped,
        "n_soma": n_soma,
        "probe_accuracy": accuracy,
        "capture_forward_max_abs_difference": forward_max_abs_difference,
        "exact_k12_max_abs_deviation_from_1": exact_error,
        "config_sha256": sha256(run_dir / "config.json"),
        "checkpoint_sha256": sha256(run_dir / "final_model.pt"),
        # Relative to the lab run base, as the release path audit requires.
        "run_dir": str(run_dir).split("/LOCAL_LEARNING/", 1)[1],
    }
    for name, values in captures.items():
        row[f"capture_{name}"] = float(values.mean())
    if row["capture_subtrees_k3"] < row["capture_broadcast_k1"] - 1e-12:
        raise RuntimeError(f"{run_dir}: per-seed nested-span violation")
    # Population sum of |dL/dV| per compartment slot over every example and
    # neuron (zero fields included), for the example-field table.
    row["_abs_sum_per_slot"] = torch.cat([g_prox, g_dist], dim=2) \
        .abs().sum(dim=(0, 1)).numpy()
    row["_n_fields_total"] = BATCH * n_soma
    return row


def t_interval(values: np.ndarray) -> tuple[float, float]:
    """Two-sided 95% Student t interval for the mean of per-seed values."""
    n = len(values)
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))
    return mean - half, mean + half


def per_seed_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One public row per checkpoint: the seed-level source of both derived
    tables.  Every scalar of the atlas frame is kept, and the private
    per-slot |dL/dV| sums are expanded into ``abs_error_sum_slot_*`` columns
    so the example-field profile is recomputable from this file as well."""
    table = frame.drop(columns=["_abs_sum_per_slot", "_n_fields_total"]).copy()
    table["n_fields_total"] = frame["_n_fields_total"].astype(int)
    slot_sums = np.stack(frame["_abs_sum_per_slot"].to_list())
    for slot, column in enumerate(SLOT_SUM_COLUMNS):
        table[column] = slot_sums[:, slot]
    trailing = [c for c in table.columns if c not in PER_SEED_LEADING_COLUMNS]
    table = table[list(PER_SEED_LEADING_COLUMNS) + trailing]
    dynamics_order = {name: i for i, name in enumerate(SWEEPS)}
    return table.sort_values(
        ["dynamics", "seed"], kind="stable",
        key=lambda col: col.map(dynamics_order) if col.name == "dynamics"
        else col,
    ).reset_index(drop=True)


def verify_reproducible_from_seeds(per_seed_path: Path, summary_path: Path,
                                   field_path: Path) -> float:
    """Re-derive the summary and example-field tables from the written
    per-seed file and require agreement to ``REPRODUCTION_TOLERANCE``.
    Returns the largest absolute discrepancy found."""
    read = {"float_precision": "round_trip"}
    per_seed = pd.read_csv(per_seed_path, **read)
    summary = pd.read_csv(summary_path, **read)
    field = pd.read_csv(field_path, **read)
    worst = 0.0
    for row in summary.itertuples(index=False):
        values = per_seed.loc[per_seed.dynamics.eq(row.dynamics),
                              f"capture_{row.basis}"].to_numpy()
        if len(values) != int(row.n_seeds):
            raise RuntimeError(
                f"{per_seed_path}: {len(values)} {row.dynamics} seeds, "
                f"summary claims {row.n_seeds}")
        low, high = t_interval(values)
        for observed, expected in ((float(values.mean()), row.mean_capture),
                                   (low, row.ci95_low_capture),
                                   (high, row.ci95_high_capture)):
            worst = max(worst, abs(observed - expected))
    for dynamics, block in field.groupby("dynamics", sort=False):
        seeds = per_seed[per_seed.dynamics.eq(dynamics)]
        raw = seeds[list(SLOT_SUM_COLUMNS)].to_numpy().sum(axis=0) \
            / float(seeds.n_fields_total.sum())
        normalized = raw / raw.max()
        block = block.sort_values("compartment_index")
        if block.compartment_index.to_list() != list(range(N_ROUTED)):
            raise RuntimeError(f"{field_path}: {dynamics} slots incomplete")
        worst = max(
            worst,
            float(np.abs(block.mean_abs_error_raw.to_numpy() - raw).max()),
            float(np.abs(block.mean_abs_error.to_numpy() - normalized).max()),
        )
    if worst > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"{summary_path} / {field_path} are not reproduced from "
            f"{per_seed_path}: max |difference| {worst:.3g} > "
            f"{REPRODUCTION_TOLERANCE:g}")
    return worst


def main() -> None:
    torch.set_grad_enabled(True)
    x, y = probe_batch()
    ladder = pd.read_csv(ROOT / "source_data" / "mnist_feedback_ladder"
                         / "seed_outcomes.csv")
    exact = ladder[ladder.feedback.eq("exact path")]
    rows = []
    for dynamics, sweep in SWEEPS.items():
        results = LADDER / sweep / "results"
        for cfg in sorted(results.iterdir(),
                          key=lambda p: int(p.name.split("_")[1])):
            row = run_atlas(cfg, x, y)
            row["dynamics"] = dynamics
            claim = exact[exact.architecture.eq(dynamics)
                          & exact.seed.eq(row["seed"])]
            if len(claim) != 1 or claim.iloc[0].checkpoint_sha256 != \
                    row["checkpoint_sha256"]:
                raise RuntimeError(
                    f"{cfg}: checkpoint is not the Figure 2 ladder checkpoint")
            rows.append(row)
            print(f"  {dynamics} seed {row['seed']}: "
                  f"broadcast {row['capture_broadcast_k1']:.3f} "
                  f"subtrees {row['capture_subtrees_k3']:.3f} "
                  f"(acc {row['probe_accuracy']:.3f}, "
                  f"dropped {row['n_dropped_zero_norm_fields']})")

    frame = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_seed = per_seed_table(frame)   # `per_seed` is a loop slice below
    by_seed.to_csv(OUT_DIR / "capture_by_seed.csv", index=False)

    summary_rows = []
    for dynamics in SWEEPS:
        per_seed = frame[frame.dynamics.eq(dynamics)]
        for basis in BASES:
            values = per_seed[f"capture_{basis}"].to_numpy()
            low, high = t_interval(values)
            summary_rows.append({
                "dynamics": dynamics,
                "basis": basis,
                "n_seeds": len(values),
                "mean_capture": float(values.mean()),
                "ci95_low_capture": low,
                "ci95_high_capture": high,
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "capture_summary.csv", index=False)

    field_rows = []
    for dynamics in SWEEPS:
        per_seed = frame[frame.dynamics.eq(dynamics)]
        abs_sum = np.sum(np.stack(per_seed["_abs_sum_per_slot"].to_list()),
                         axis=0)
        raw = abs_sum / float(per_seed["_n_fields_total"].sum())
        normalized = raw / raw.max()
        for slot in range(N_ROUTED):
            depth = 1 if slot < N_PROXIMAL else 2
            subtree = slot if slot < N_PROXIMAL else \
                (slot - N_PROXIMAL) // N_CHILDREN
            field_rows.append({
                "dynamics": dynamics,
                "compartment_index": slot,
                "depth": depth,
                "subtree_index": subtree,
                "mean_abs_error": float(normalized[slot]),
                "mean_abs_error_raw": float(raw[slot]),
            })
    pd.DataFrame(field_rows).to_csv(OUT_DIR / "example_field.csv", index=False)

    # The seed-level file is the source of record: both derived tables must
    # recompute from it exactly (Student t interval included).
    reproduction_error = verify_reproducible_from_seeds(
        OUT_DIR / "capture_by_seed.csv",
        OUT_DIR / "capture_summary.csv",
        OUT_DIR / "example_field.csv",
    )

    git_head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    manifest = {
        "description": (
            "Fraction of per-example within-neuron compartment-error field "
            "energy captured by fixed route dictionaries at the Figure 2 "
            "MNIST [3,3] exact-path checkpoints. For each held-out example "
            "and neuron, d is the 12-vector of dL/dV over the neuron's "
            "routed compartments (3 proximal + 9 distal; soma excluded), "
            "taken by autograd on the pre-reactivation voltages at the "
            "final checkpoint; capture for address matrix A is "
            "||P_A d||^2 / ||d||^2 with P_A the orthogonal projector onto "
            "col(A). Zero-norm fields are dropped; each seed contributes "
            "the mean capture over its remaining example-neuron fields."
        ),
        "bases": {
            "broadcast_k1": "one all-ones column over the 12 routed compartments",
            "subtrees_k3": "one indicator column per proximal compartment, "
                           "covering itself and its 3 distal children",
            "exact_k12": "identity; capture 1 by definition, asserted to "
                         "numerical tolerance",
        },
        "compartment_ordering": (
            "slots 0-2: proximal compartments (stage 1, soma-major); slots "
            "3+3s+c (s,c in 0..2): distal compartment c of subtree s (stage "
            "0, soma-major); distal local index j feeds proximal j//3 via "
            "the contiguous BlockLinear branch aggregation"
        ),
        "interval_method": (
            "two-sided 95% Student t interval over the per-seed mean "
            "captures (mean +/- t_{0.975, n_seeds-1} * sd / sqrt(n_seeds)); "
            "no bootstrap, fully deterministic"
        ),
        "n_examples": BATCH,
        "n_seeds": {dynamics: int(frame.dynamics.eq(dynamics).sum())
                    for dynamics in SWEEPS},
        "per_seed_file": (
            "capture_by_seed.csv: one row per checkpoint with the seed-level "
            "mean captures, probe accuracy, field counts, per-slot |dL/dV| "
            "sums and checkpoint hashes; capture_summary.csv and "
            "example_field.csv are re-derived from it after writing and "
            f"must agree to {REPRODUCTION_TOLERANCE:g}"
        ),
        "probe_batch": "first 2048 MNIST test images (identical to "
                       "analyze_fig2_path_gain_dispersion.py)",
        # Relative to the lab run base, as the release path audit requires.
        "run_dirs": {
            dynamics: sorted(
                frame[frame.dynamics.eq(dynamics)].run_dir.to_list(),
                key=lambda p: int(p.rsplit("_", 1)[1]))
            for dynamics in SWEEPS
        },
        "git_head": git_head,
        "script": "scripts/analyze_route_dictionary_atlas.py",
    }
    with open(OUT_DIR / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")

    print(f"wrote {OUT_DIR / 'capture_by_seed.csv'} "
          f"({len(by_seed)} rows x {by_seed.shape[1]} columns)")
    print(f"wrote {OUT_DIR / 'capture_summary.csv'} ({len(summary)} rows)")
    print(f"wrote {OUT_DIR / 'example_field.csv'} ({len(field_rows)} rows)")
    print(f"wrote {OUT_DIR / 'manifest.json'}")
    print(f"per-seed reproduction check: max |difference| "
          f"{reproduction_error:.3g} <= {REPRODUCTION_TOLERANCE:g}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
