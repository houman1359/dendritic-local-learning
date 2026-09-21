#!/usr/bin/env python3
"""Frozen common-broadcast comparison using existing anatomical derivatives.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1. No network or model fitting.
All outputs stay under source_data/anatomy_commonmode.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))
import analyze_reciprocal_routing_controls as cable
from analyze_microns_morphology_credit import parent_map

OUT = JOURNAL / "source_data" / "anatomy_commonmode"
METHODS = ["common + ancestry", "common + random routes", "common + depth bins",
           "common + shuffled routes", "common + surrogate ancestry", "common-constrained SVD"]
METRICS = ["total_capture", "residual_capture", "dictionary_rank", "nonzero_coefficients",
           "wiring_density", "coverage", "incremental_total_capture"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def weighted_basis(dictionary, sqrt_weight):
    matrix = np.asarray(dictionary, float) * sqrt_weight[:, None]
    if not matrix.size or not np.any(matrix):
        return np.zeros((len(sqrt_weight), 0))
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular[0]
    return u[:, singular > tolerance]


def evaluate(dictionary, weighted_response, sqrt_weight):
    """Measure total and common-removed energy in the same weighted geometry."""
    q0 = sqrt_weight / np.linalg.norm(sqrt_weight)
    common = q0[:, None] * (q0 @ weighted_response)[None, :]
    residual = weighted_response - common
    total_energy = float(np.sum(weighted_response ** 2))
    residual_energy = float(np.sum(residual ** 2))
    basis = weighted_basis(dictionary, sqrt_weight)
    captured = float(np.sum((basis.T @ weighted_response) ** 2))
    residual_captured = float(np.sum((basis.T @ residual) ** 2))
    baseline = float(np.sum(common ** 2)) / total_energy
    total_capture = captured / total_energy
    residual_capture = residual_captured / residual_energy if residual_energy > total_energy * 1e-25 else 0.0
    assert abs(total_capture - baseline - residual_capture * (1 - baseline)) < 1e-10
    assert -1e-12 <= total_capture <= 1 + 1e-12
    assert -1e-12 <= residual_capture <= 1 + 1e-12
    return dict(total_capture=float(np.clip(total_capture, 0, 1)),
                residual_capture=float(np.clip(residual_capture, 0, 1)),
                dictionary_rank=int(basis.shape[1]),
                nonzero_coefficients=int(np.count_nonzero(dictionary)),
                wiring_density=float(np.count_nonzero(dictionary) / dictionary.size),
                coverage=float(np.any(dictionary != 0, axis=1).mean()),
                incremental_total_capture=total_capture - baseline)


def make_depth_with_common(depth, count):
    if count == 1:
        return np.ones((len(depth), 1))
    bins = cable.depth_dictionary(depth, count)
    return np.column_stack([np.ones(len(depth)), bins[:, :count - 1]])


def freeze(protocol):
    OUT.mkdir(parents=True, exist_ok=True)
    dependencies = [HERE / "protocol.json", Path(__file__), Path(cable.__file__),
                    JOURNAL / "code/reconstructed_tree/run_focal_shunting_credit_perturbation.py",
                    JOURNAL / "code/reconstructed_tree/analyze_microns_morphology_credit.py"]
    dependencies += [JOURNAL / "source_data" / value for value in protocol["cohort_inputs"].values()]
    hashes = {str(p.relative_to(JOURNAL)): digest(p) for p in dependencies}
    path = OUT / "protocol_freeze.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["input_sha256"] != hashes:
            raise RuntimeError("Frozen inputs changed; preserve this experiment and start a new version.")
        return previous
    if any((OUT / name / "cell_method_summary.csv").exists() for name in protocol["cohort_inputs"]):
        raise RuntimeError("Cannot freeze after outcome computation.")
    payload = dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
                   protocol=protocol, input_sha256=hashes,
                   outcome_status="Only original8 published/reviewer augmentation outcomes were inspected; no new v661/Pinky cable-control outcomes computed.")
    write_json(path, payload)
    return payload


def cell_records(root_id, segments, cohort, protocol, cache_dir):
    start = time.monotonic()
    response, weights, depth, beta, e_sites, i_sites, parents = cable.exact_reciprocal_response(
        segments, protocol["e_scale"], protocol["i_scale"], protocol["derivative_dose"],
        protocol["excitatory_reversal"], protocol["inhibitory_reversal"])
    weighted, sqrt_weight = cable.weighted_operator(response, weights)
    direct, ancestry = cable.direct_dictionary(e_sites, i_sites, parents, beta)
    order = np.argsort(-(beta * (ancestry.T @ weights) / weights.sum()))
    q0 = sqrt_weight / np.linalg.norm(sqrt_weight)
    residual = weighted - q0[:, None] * (q0 @ weighted)[None, :]
    u_residual, singular_residual, _ = np.linalg.svd(residual, full_matrices=False)
    singular = np.linalg.svd(weighted, compute_uv=False)
    total_energy = float(np.sum(weighted ** 2))
    residual_energy = float(np.sum(residual ** 2))
    baseline = 1 - residual_energy / total_energy
    budgets = [k for k in protocol["channels"] if k <= len(e_sites) and k - 1 <= len(i_sites)]
    np.savez_compressed(cache_dir / f"operator_{root_id}.npz", response=response,
                        weights=weights, sqrt_weight=sqrt_weight, weighted_response=weighted,
                        e_sites=e_sites, i_sites=i_sites, beta=beta, depth=depth,
                        singular_values=singular, residual_singular_values=singular_residual)
    rows = []

    def add(method, k, replicate, dictionary):
        rows.append(dict(cohort=cohort, root_id=int(root_id), channels=k,
                         method=method, control_replicate=replicate,
                         **evaluate(dictionary, weighted, sqrt_weight)))

    # Generate each topology surrogate once, then use nested leverage prefixes
    # at the different budgets. Random choices and row shuffles use per-K streams.
    surrogate_rng = cable.stable_rng(protocol["seed"], root_id, 300)
    surrogate_dictionaries = []
    for _ in range(protocol["n_controls"]):
        surrogate = cable.degree_depth_matched_surrogate(segments, surrogate_rng)
        _, surrogate_parents, _ = parent_map(surrogate)
        candidate, surrogate_ancestry = cable.direct_dictionary(e_sites, i_sites, surrogate_parents, beta)
        surrogate_order = np.argsort(-(beta * (surrogate_ancestry.T @ weights) / weights.sum()))
        surrogate_dictionaries.append(candidate[:, surrogate_order[:max(budgets) - 1]])

    constant = np.ones((len(e_sites), 1))
    old_curves = None
    if cohort == "original8":
        old_curves = pd.read_csv(JOURNAL / "source_data/reciprocal_routing/cell_method_capture.csv")
    for k in budgets:
        selected = direct[:, order[:k - 1]]
        augmented = np.column_stack([constant, selected])
        add(METHODS[0], k, -1, augmented)
        depth_dictionary = make_depth_with_common(depth, k)
        assert abs(cable.capture(depth_dictionary, weighted, sqrt_weight) -
                   cable.capture(cable.depth_dictionary(depth, k), weighted, sqrt_weight)) < 1e-10
        add(METHODS[2], k, -1, depth_dictionary)
        oracle = np.column_stack([constant, u_residual[:, :k - 1] / sqrt_weight[:, None]])
        add(METHODS[5], k, -1, oracle)
        if old_curves is not None:
            old = old_curves[old_curves.root_id.eq(root_id) & old_curves.channels.eq(k)
                             & old_curves.method.eq("morphology paths")]
            if len(old):
                assert abs(cable.capture(direct[:, order[:k]], weighted, sqrt_weight) - old.iloc[0].capture) < 1e-10
        if k == 1:
            for method in [METHODS[1], METHODS[3], METHODS[4]]:
                add(method, k, -1, constant)
            continue
        random_rng = cable.stable_rng(protocol["seed"], root_id, 1000 + k)
        shuffle_rng = cable.stable_rng(protocol["seed"], root_id, 2000 + k)
        for replicate in range(protocol["n_controls"]):
            choice = random_rng.choice(len(i_sites), size=k - 1, replace=False)
            add(METHODS[1], k, replicate, np.column_stack([constant, direct[:, choice]]))
            shuffled = np.column_stack([shuffle_rng.permutation(selected[:, j]) for j in range(k - 1)])
            add(METHODS[3], k, replicate, np.column_stack([constant, shuffled]))
            add(METHODS[4], k, replicate, np.column_stack([constant, surrogate_dictionaries[replicate][:, :k - 1]]))
    audit = dict(cohort=cohort, root_id=int(root_id), n_e_sites=len(e_sites), n_i_sites=len(i_sites),
                 budgets=budgets, total_weighted_energy=total_energy,
                 residual_weighted_energy=residual_energy,
                 common_capture=baseline, rank_one_capture=float(singular[0] ** 2 / total_energy),
                 response_rank=int(np.linalg.matrix_rank(weighted)),
                 residual_rank=int(np.linalg.matrix_rank(residual)),
                 elapsed_seconds=time.monotonic() - start)
    return pd.DataFrame(rows), audit


def bootstrap(values, rng, draws):
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    largest = 0.0
    for i, idx in enumerate(order):
        largest = max(largest, (len(pvalues) - i) * pvalues[idx])
        adjusted[idx] = min(1.0, largest)
    return adjusted


def summarize(cohort, protocol):
    directory = OUT / cohort
    raw = pd.concat([pd.read_csv(p) for p in sorted((directory / "cells").glob("rows_*.csv.gz"))], ignore_index=True)
    summary = raw.groupby(["cohort", "root_id", "channels", "method"], sort=True)[METRICS].mean().reset_index()
    summary.to_csv(directory / "cell_method_summary.csv", index=False)
    group = summary.groupby(["cohort", "channels", "method"])[METRICS].agg(["mean", "std", "count"])
    group.columns = ["_".join(x) for x in group.columns]
    group.reset_index().to_csv(directory / "cohort_method_summary.csv", index=False)
    audits = [json.loads(p.read_text()) for p in sorted((directory / "cells").glob("audit_*.json"))]
    audit_frame = pd.DataFrame(audits)
    audit_frame.to_csv(directory / "operator_audit.csv", index=False)
    focus = summary[summary.channels.eq(protocol["focus_channels"])]
    rng = np.random.default_rng(protocol["seed"] + ["original8", "v661", "pinky"].index(cohort))
    comparisons = []
    for metric in ["total_capture", "residual_capture"]:
        pivot = focus.pivot(index="root_id", columns="method", values=metric)
        for control in METHODS[1:5]:
            difference = (pivot[METHODS[0]] - pivot[control]).to_numpy()
            p = float(stats.wilcoxon(difference).pvalue) if np.any(difference) else 1.0
            comparisons.append(dict(metric=metric, control=control, n_cells=len(difference),
                                    mean_difference=float(difference.mean()),
                                    ci95=bootstrap(difference, rng, protocol["bootstrap_samples"]),
                                    cells_positive=int((difference > 1e-12).sum()),
                                    cells_negative=int((difference < -1e-12).sum()),
                                    wilcoxon_p=p))
        selected = comparisons[-4:]
        for item, p in zip(selected, holm([x["wilcoxon_p"] for x in selected])):
            item["wilcoxon_holm_p"] = float(p)
    report = dict(status="complete", cohort=cohort, n_analyzed_cells=len(audits),
                  focus_channels=protocol["focus_channels"], n_focus_cells=focus.root_id.nunique(),
                  focus_means=focus.groupby("method")[METRICS].mean().to_dict("index"),
                  common_capture_mean=float(audit_frame.common_capture.mean()),
                  rank_one_capture_mean=float(audit_frame.rank_one_capture.mean()),
                  comparisons=comparisons,
                  scope=protocol["scope"], protocol_sha256=digest(HERE / "protocol.json"))
    write_json(directory / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["original8", "v661", "pinky"])
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    frozen = freeze(protocol)
    print("Protocol frozen", frozen["frozen_utc"], flush=True)
    if args.freeze_only:
        return
    if not args.cohort:
        parser.error("--cohort is required unless --freeze-only")
    cohort = args.cohort
    segments = pd.read_csv(JOURNAL / "source_data" / protocol["cohort_inputs"][cohort])
    directory = OUT / cohort
    cache_dir = directory / "cells"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for root_id, group in segments.groupby("root_id", sort=True):
        included = "qc_included" not in group or bool(group.qc_included.all())
        ne, ni = int((group.E_size > 0).sum()), int((group.I_size > 0).sum())
        manifest.append(dict(root_id=int(root_id), inherited_qc_included=included, n_e_sites=ne, n_i_sites=ni,
                             focus_budget_eligible=included and ne >= protocol["focus_channels"] and ni >= protocol["focus_channels"] - 1))
        if not included:
            continue
        path = cache_dir / f"rows_{root_id}.csv.gz"
        if path.exists() and (cache_dir / f"audit_{root_id}.json").exists():
            print(cohort, root_id, "cached", flush=True)
            continue
        frame, audit = cell_records(int(root_id), group.copy(), cohort, protocol, cache_dir)
        frame.to_csv(path, index=False)
        write_json(cache_dir / f"audit_{root_id}.json", audit)
        print(cohort, root_id, f"{audit['elapsed_seconds']:.2f}s", flush=True)
    pd.DataFrame(manifest).to_csv(directory / "cohort_inclusion.csv", index=False)
    report = summarize(cohort, protocol)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
