#!/usr/bin/env python3
"""Independent saved-endpoint replay and analytic-capacity linkage.

This script does not import any learning-experiment module. It uses saved
parameters/children, independent theory truth tables, and a recursive forward
evaluator. It does not rerun training or claim trajectory-state replay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ("and4", "or4", "parity4", "or_of_ands", "xor_of_ands", "and_of_xors", "nested")
TREE_NAMES = {"balanced_ab_cd": "((a,b),(c,d))", "balanced_ac_bd": "((a,c),(b,d))",
              "balanced_ad_bc": "((a,d),(b,c))", "comb_a_b_cd": "(a,(b,(c,d)))"}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(x, weights, children, node=6):
    if node < 4:
        return x[:, node]
    left, right = map(int, children[node-4])
    assert 0 <= left < node and 0 <= right < node
    l = predict(x, weights, children, left)
    r = predict(x, weights, children, right)
    a, b, c, d = weights[node-4]
    return a + b*l + c*r + d*l*r


def semantic_tree(children, permutation, node=6):
    if node < 4:
        return "abcd"[int(np.flatnonzero(permutation == node)[0])]
    l, r = (semantic_tree(children, permutation, int(child)) for child in children[node-4])
    if min(c for c in r if c.isalpha()) < min(c for c in l if c.isalpha()):
        l, r = r, l
    return "(" + l + "," + r + ")"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning", type=Path, default=ROOT/"source_data/boolean_morphology")
    parser.add_argument("--out", type=Path, default=ROOT/"source_data/boolean_theory")
    args = parser.parse_args()
    start = time.perf_counter()
    protocol = json.loads((args.learning/"protocol.json").read_text())
    assert tuple(protocol["families"]) == FAMILIES
    assert protocol["steps"] == 2048 and protocol["expected_fresh_fits"] == 6720
    theory_rows = read_csv(args.out/"truth_tables.csv")
    truth = {family: np.array([float(row["target_raw"]) for row in theory_rows if row["family"] == family]) for family in FAMILIES}
    norm = {family: (v-v.mean())/v.std() for family,v in truth.items()}
    capacity = {(r["family"],r["tree"]):float(r["normalized_mse_lower_bound"])
                for r in read_csv(args.out/"tree_capacity.csv")}
    metrics = ("population_nmse", "test_noisy_nmse", "accuracy", "balanced_accuracy", "max_abs_parameter", "parameter_norm")
    max_error = dict.fromkeys(metrics,0.0)
    row_errors, endpoints, manifests = [], [], []
    minimum_margin = float("inf")
    all_checkpoints = 0
    incompatible_endpoints = 0
    failed_fits = 0
    max_target_error = 0.0
    initial_error = 0.0
    for split in ("development", "fresh"):
        for seed in protocol[split+"_seeds"]:
            base = args.learning/"runs"/split/("seed_"+str(seed))
            csv_path, npz_path = base.with_suffix(".csv"), base.with_suffix(".npz")
            rows = read_csv(csv_path)
            arrays = np.load(npz_path, allow_pickle=False)
            permutation = arrays["permutation"]
            assert sorted(permutation.tolist()) == list(range(4))
            assert len(rows) == 336*len(protocol["checkpoints"])
            assert arrays["final_weights"].shape == (336,3,4)
            px = arrays["population_x"]
            assert set(map(tuple,px)) == set(__import__("itertools").product((-1.0,1.0),repeat=4))
            pattern = (((px[:,permutation]+1)/2).astype(int) * (1 << np.arange(4))).sum(axis=1)
            for fi, family in enumerate(FAMILIES):
                assert np.array_equal(arrays["raw_population_y"][fi],truth[family][pattern])
                err = np.max(np.abs(arrays["population_y"][fi]-norm[family][pattern]))
                max_target_error = max(max_target_error,float(err))
                assert err < 1e-14
            grouped = {i:[] for i in range(336)}
            for row in rows:
                grouped[int(row["condition_id"])].append(row)
                bound = capacity[row["family"],TREE_NAMES[row["tree"]]]
                assert float(row["population_nmse"]) + 1e-11 >= bound
                all_checkpoints += 1
            for ci, trajectory in grouped.items():
                assert sorted(int(r["step"]) for r in trajectory) == protocol["checkpoints"]
                row = next(r for r in trajectory if int(r["step"]) == 2048)
                first = next(r for r in trajectory if int(r["step"]) == 0)
                family = row["family"]
                fi = FAMILIES.index(family)
                assert int(arrays["family_index"][ci]) == fi
                children, weights = arrays["children"][ci], arrays["final_weights"][ci]
                tree = semantic_tree(children,permutation)
                assert tree == TREE_NAMES[row["tree"]]
                assert sorted(c for c in tree if c.isalpha()) == list("abcd")
                assert np.isfinite(weights).all() and np.max(np.abs(weights)) <= 2
                y = norm[family][pattern]
                pred = predict(px,weights,children)
                mse = float(np.mean((pred-y)**2))
                test_pred = predict(arrays["test_x"][fi],weights,children)
                test_mse = float(np.mean((test_pred-arrays["test_y"][fi])**2))
                threshold = (.5-truth[family].mean())/truth[family].std()
                decision = pred >= threshold
                actual = truth[family][pattern].astype(bool)
                accuracy = float(np.mean(decision == actual))
                balanced = float((np.mean(decision[actual])+np.mean(~decision[~actual]))/2)
                bound = capacity[family,tree]
                assert mse + 1e-11 >= bound
                if bound > 0:
                    minimum_margin = min(minimum_margin,mse-bound)
                    incompatible_endpoints += 1
                failed = int(arrays["failure_step"][ci]) >= 0
                assert failed == (row["failed"] == "True")
                failed_fits += failed
                recomputed = dict(population_nmse=1e6 if failed else mse,
                    test_noisy_nmse=1e6 if failed else test_mse,
                    accuracy=0.0 if failed else accuracy,balanced_accuracy=0.0 if failed else balanced,
                    max_abs_parameter=float(np.max(np.abs(weights))),parameter_norm=float(np.linalg.norm(weights)))
                errors = {key:abs(recomputed[key]-float(row[key])) for key in metrics}
                for key,error in errors.items():
                    max_error[key] = max(max_error[key],error)
                    assert error < 1e-10,(split,seed,ci,key,error)
                pred0 = predict(px,arrays["initial_weights"][fi],children)
                error0 = abs(float(np.mean((pred0-y)**2))-float(first["population_nmse"]))
                initial_error = max(initial_error,error0)
                assert error0 < 1e-10
                row_errors.append(dict(seed=seed,split=split,condition_id=ci,family=family,
                    tree=row["tree"],optimizer=row["optimizer"],rule=row["rule"],rate=float(row["rate"]),
                    actual_parameter_mse=mse,analytic_nmse_lower_bound=bound,
                    bound_margin=mse-bound,maximum_endpoint_metric_error=max(errors.values()),failed=failed))
                endpoints.append(row)
            manifests.append(dict(seed=seed,split=split,csv_sha256=sha(csv_path),npz_sha256=sha(npz_path)))
            arrays.close()
    # Recompute development rate selection from raw endpoint rows only.
    selected = json.loads((args.learning/"selected_rates.json").read_text())
    for optimizer in ("sgd","adam"):
        for rule in ("exact","broadcast"):
            scores = {}
            for rate in protocol["rates"]:
                values = [float(r["population_nmse"]) for r in endpoints if r["split"] == "development"
                          and r["optimizer"] == optimizer and r["rule"] == rule and float(r["rate"]) == rate]
                assert len(values) == 5*7*4
                scores[rate] = np.mean(values)
            chosen = min(rate for rate,score in scores.items() if score <= min(scores.values())+1e-12)
            assert chosen == selected[optimizer][rule]
    # Recompute the two predeclared paired contrasts and their declared bootstrap.
    primary = []
    fresh = {(int(r["seed"]),r["tree"],r["rule"]):float(r["population_nmse"])
             for r in endpoints if r["split"] == "fresh" and r["family"] == "xor_of_ands"
             and r["optimizer"] == "adam" and float(r["rate"]) == selected["adam"][r["rule"]]}
    indices = np.random.default_rng(202609052).integers(20,size=(10000,20))
    expected = {r["contrast"]:r for r in read_csv(args.learning/"primary_contrasts.csv")}
    for contrast in ("broadcast_minus_exact_compatible","crossed_minus_compatible_exact"):
        values = []
        for seed in protocol["fresh_seeds"]:
            exact = fresh[seed,"balanced_ab_cd","exact"]
            control = (fresh[seed,"balanced_ab_cd","broadcast"] if contrast.startswith("broadcast")
                       else (fresh[seed,"balanced_ac_bd","exact"]+fresh[seed,"balanced_ad_bc","exact"])/2)
            values.append(control-exact)
        values = np.array(values)
        ci = np.quantile(values[indices].mean(axis=1),[.025,.975,.0125,.9875])
        stats = dict(mean_difference=float(values.mean()),ci95_low=float(ci[0]),ci95_high=float(ci[1]),
                     ci975_low=float(ci[2]),ci975_high=float(ci[3]))
        for key,value in stats.items():
            assert abs(value-float(expected[contrast][key])) < 1e-12
        passes = stats["ci975_low"] > 0 and stats["mean_difference"] >= .01
        assert passes == (expected[contrast]["passes_adjusted_interval_and_mean_margin"] == "True")
        primary.append(dict(contrast=contrast,**stats,passes_predeclared_criterion=passes))
    with (args.out/"learning_endpoint_validation.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(row_errors[0]));writer.writeheader();writer.writerows(row_errors)
    result = dict(status="passed",development_seeds=5,fresh_seeds=20,
        endpoints_replayed=len(row_errors),fresh_endpoints_replayed=sum(r["split"]=="fresh" for r in row_errors),
        saved_initial_states_replayed=len(row_errors),recorded_checkpoint_bounds_checked=all_checkpoints,
        incompatible_endpoints_checked=incompatible_endpoints,minimum_incompatible_bound_margin=minimum_margin,
        failed_fits_retained=failed_fits,maximum_raw_normalization_error=max_target_error,
        maximum_initial_population_mse_error=initial_error,maximum_endpoint_errors=max_error,
        all_semantic_target_permutations_and_tree_labels_match=True,
        development_only_rate_selection_recomputed=True,primary_contrasts=primary,
        scope="Saved endpoint and initial-state replay, analytic capacity linkage, rate/primary-statistic recomputation; no full training replay",
        runtime_seconds=time.perf_counter()-start,validator_sha256=sha(Path(__file__)),
        theory_truth_table_sha256=sha(args.out/"truth_tables.csv"),theory_capacity_sha256=sha(args.out/"tree_capacity.csv"),
        learning_protocol_sha256=sha(args.learning/"protocol.json"),input_manifests=manifests)
    (args.out/"learning_validation.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="input_manifests"},indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
