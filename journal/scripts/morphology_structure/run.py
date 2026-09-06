#!/usr/bin/env python3
"""Freeze, run and summarize an exhaustive synthetic structural diagnostic."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from model import candidates, tasks, domain, fourier_design, input_gradient_covariance, cut_scores, fit

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis/morphology_investigation_20260905/structure"
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(status="exhaustive_mechanistic_diagnostic_not_deployable_prospective_validation",
        model="8 scalar input leaves;7 bilinear internal units;28 trainable coefficients;14 edges;root readout only",
        tasks="All105 perfect matchings and35 unordered4+4 partitions;coefficient0.5 per interaction",
        samples="All256 equally weighted Rademacher inputs;exact population fitting and oracle target Fourier information",
        covariance="Every task has input-gradient covariance I/4, rank8, effective rank8; output variance differs by family",
        candidates=[dict(name=t.name, shape=t.shape, permutation=[int(i) for i in t.permutation]) for t in candidates()],
        sweeps=32, restarts=4, checkpoints=[0, 2, 8, 32], seed_base=4_609_050,
        optimizer="Exact coordinate least squares, alternating postorder/preorder;mean/variance gauge normalization preserves output",
        selector="Minimum max centered-cut tail (valid normalized population-MSE bound), sum tail (heuristic), full-cut tail;lexicographic deterministic ties",
        scope="Representation mechanism under local scalar nonlinear bottlenecks;no physical morphology optimum or local plasticity claim",
        inference="Finite exhaustive function families;no sampling CI across all enumerated tasks;optimizer restarts are not biological replicates",
        outcome_policy="Retain all tasks,candidates,restarts;report lower bounds and optimization gaps;no favorable task filtering",
        source_hashes={p.name: sha(p) for p in (HERE / "model.py", HERE / "run.py")})
    target = OUT / "protocol.json"
    if target.exists():
        assert json.loads(target.read_text()) == protocol, "Frozen protocol mismatch"
    else:
        dump(target, protocol)
        dump(OUT / "freeze.json", dict(protocol_sha256=sha(target), utc=pd.Timestamp.now(tz="UTC").isoformat(),
                                      numpy=np.__version__, python=platform.python_version()))
    return protocol


def run_shard(shard, nshards):
    cfg = freeze()
    rows, details = [], []
    x = domain()
    design = fourier_design(x)
    start = time.perf_counter()
    for task_index, task in enumerate(tasks()):
        if task_index % nshards != shard:
            continue
        coeff = task["coefficients"]
        cov = input_gradient_covariance(coeff)
        np.testing.assert_allclose(cov, np.eye(8) / 4, atol=1e-14)
        y = design @ coeff
        for tree_index, tree in enumerate(candidates()):
            scores = cut_scores(coeff, tree)
            details.append(dict(task_id=task["task_id"], candidate_id=tree.name, **scores))
            for restart in range(cfg["restarts"]):
                tick = time.perf_counter()
                # One seed per task/restart, shared across tree candidates.
                seed = cfg["seed_base"] + 100 * task_index + restart
                weights, curve, diagnostics = fit(x, y, tree, seed, cfg["sweeps"], tuple(cfg["checkpoints"]), diagnostics=True)
                for step, error in curve.items():
                    assert error + 1e-7 >= scores["centered_cut_bound"], "Fitted error violated analytical lower bound"
                    rows.append(dict(task_id=task["task_id"], task_index=task_index, family=task["family"],
                        candidate_id=tree.name, shape=tree.shape, permutation_index=tree_index % 4,
                        restart=restart, seed=seed, sweep=step, normalized_mse=error,
                        mse=error * float(y @ y / len(y)), variance=float(y @ y / len(y)),
                        parameter_count=28, edges=14, credit_rank=8, credit_effective_rank=8,
                        **diagnostics,
                        **{k:v for k,v in scores.items() if k != "cut_details"},
                        fit_seconds=time.perf_counter()-tick))
            print(task["task_id"], tree.name, "done", flush=True)
    dest = OUT / "shards"
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / f"shard_{shard:02d}.csv", index=False)
    dump(dest / f"shard_{shard:02d}_cuts.json", details)
    dump(dest / f"shard_{shard:02d}_audit.json", dict(shard=shard, nshards=nshards, seconds=time.perf_counter()-start,
        protocol_sha256=sha(OUT/"protocol.json"), rows=len(rows), all_outcomes_retained=True))


def summarize(nshards):
    freeze()
    files = [OUT / "shards" / f"shard_{s:02d}.csv" for s in range(nshards)]
    assert all(p.exists() for p in files)
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    assert len(df) == 140 * 12 * 4 * 4
    df.to_csv(OUT / "all_trajectories.csv", index=False)
    end = df[df.sweep == 32].groupby(["task_id", "family", "candidate_id", "shape"], as_index=False).agg(
        normalized_mse=("normalized_mse", "min"), mean_restart_error=("normalized_mse", "mean"),
        restart_std=("normalized_mse", "std"), centered_cut_bound=("centered_cut_bound", "first"),
        centered_cut_sum=("centered_cut_sum", "first"), full_cut_bound=("full_cut_bound", "first"))
    pilot = df[df.sweep == 2].groupby(["task_id", "candidate_id"], as_index=False).normalized_mse.min()
    end = end.merge(pilot.rename(columns={"normalized_mse":"pilot_error"}), on=["task_id", "candidate_id"])
    end["optimization_gap_above_bound"] = end.normalized_mse - end.centered_cut_bound
    end.to_csv(OUT / "candidate_outcomes.csv", index=False)
    policies = []
    for family, family_data in end.groupby("family"):
        fixed = family_data.groupby("candidate_id").normalized_mse.mean().idxmin()
        for task, group in family_data.groupby("task_id"):
            group = group.sort_values("candidate_id")
            best = float(group.normalized_mse.min())
            for policy, score in [("centered_cut_bound", "centered_cut_bound"), ("centered_cut_sum", "centered_cut_sum"),
                                  ("full_cut_bound", "full_cut_bound"), ("two_sweep_pilot", "pilot_error")]:
                selected = group.loc[group[score].idxmin()]
                policies.append(dict(task_id=task, family=family, policy=policy, candidate_id=selected.candidate_id,
                    normalized_mse=float(selected.normalized_mse), regret=float(selected.normalized_mse)-best))
            for policy, candidate in [("fixed_balanced_p0", "balanced_p0"), ("best_fixed_in_hindsight", fixed)]:
                selected = group[group.candidate_id == candidate].iloc[0]
                policies.append(dict(task_id=task, family=family, policy=policy, candidate_id=candidate,
                    normalized_mse=float(selected.normalized_mse), regret=float(selected.normalized_mse)-best))
            policies.append(dict(task_id=task, family=family, policy="uniform_random_expectation", candidate_id="average",
                normalized_mse=float(group.normalized_mse.mean()), regret=float(group.normalized_mse.mean())-best))
    pd.DataFrame(policies).to_csv(OUT / "policy_outcomes.csv", index=False)
    summary = pd.DataFrame(policies).groupby(["family", "policy"], as_index=False).agg(
        mean_nmse=("normalized_mse", "mean"), mean_regret=("regret", "mean"), max_regret=("regret", "max"))
    summary.to_csv(OUT / "policy_summary.csv", index=False)
    end.groupby(["family", "shape"], as_index=False).agg(mean_nmse=("normalized_mse", "mean"),
        mean_bound=("centered_cut_bound", "mean"), mean_optimization_gap=("optimization_gap_above_bound", "mean")).to_csv(
            OUT / "shape_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("Gap", end.optimization_gap_above_bound.describe().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "run", "summarize"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=14)
    args = parser.parse_args()
    if args.action == "freeze": freeze()
    elif args.action == "run": run_shard(args.shard, args.nshards)
    else: summarize(args.nshards)
