"""Freeze development-selected rates, validate fresh seeds, and report all tasks."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

import experiment as ex


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def freeze(root):
    choices = {}
    development_files = []
    for task in ex.TASKS:
        records = []
        for seed in range(2026091700, 2026091703):
            path = root / "results" / f"seed_{seed}_{task}.json"
            development_files.append(path)
            records.extend(read(path)["endpoints"])
        choices[task] = {}
        for rule in ex.RULES:
            values = {rate: np.mean([r["validation_nmse"] for r in records
                                    if r["rule"] == rule and r["rate"] == rate])
                      for rate in ex.RATES}
            choices[task][rule] = min(values, key=values.get)
    fresh = list(range(2026091800, 2026091820))
    assert not any((root / "results" / f"seed_{seed}_{task}.json").exists()
                   for seed, task in itertools.product(fresh, ex.TASKS))
    sources = [Path(ex.__file__), ex.MODEL_PATH, Path(__file__)]
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    status="Internally frozen after three development seeds; not externally preregistered",
                    fresh_seeds=fresh, development_seeds=list(range(2026091700, 2026091703)),
                    fixed_rates=choices, steps=4096, train_size=2048,
                    source_sha256={str(p): ex.digest(p) for p in sources},
                    development_sha256={str(p): ex.digest(p) for p in development_files},
                    endpoint="Test NMSE at minimum-validation state within 4096 updates, fixed development-selected rate",
                    primary="Paired unit-broadcast minus resistance-gate NMSE, separately for all three tasks",
                    secondary="Exact reference, proportional gate, wrong-branch control; all rates retained",
                    inference="Descriptive 95% whole-seed percentile bootstrap intervals; no significance or equivalence claim",
                    interpretation="External targets need not be representable; inspect absolute exact-path error, bound contacts and all failures; no claiming spiking or reconstructed-neuron learning")
    write_new(root / "fresh_protocol.json", protocol)
    print(json.dumps(choices, indent=2))


def validate(root, seed):
    protocol = read(root / "fresh_protocol.json")
    assert seed in protocol["fresh_seeds"]
    for path, expected in protocol["source_sha256"].items():
        assert ex.digest(path) == expected, path
    return protocol


def fresh(root, seed):
    protocol = validate(root, seed)
    for task in ex.TASKS:
        result, parameters = ex.run_task(seed, task, protocol["steps"], protocol["train_size"])
        # The generic runner's per-seed rate selection is not used here.
        result.pop("validation_selected_rates")
        result["exploratory"] = False
        result["phase"] = "fresh-seed validation of development-selected design"
        result["fixed_rate_endpoints"] = [r for r in result["endpoints"]
                                          if r["rate"] == protocol["fixed_rates"][task][r["rule"]]]
        result["protocol_sha256"] = ex.digest(root / "fresh_protocol.json")
        path = root / "results" / f"seed_{seed}_{task}.json"
        write_new(path, result)
        np.savez_compressed(path.with_suffix(".npz"), selected_parameters=parameters)
        print(seed, task, result["elapsed_seconds"], flush=True)


def report(root):
    protocol = read(root / "fresh_protocol.json")
    rng = np.random.default_rng(170918)
    resamples = rng.integers(0, 20, size=(20000, 20))
    rows, contrasts = [], []
    for task in ex.TASKS:
        by_rule = {r: [] for r in ex.RULES}
        hits = {r: [] for r in ex.RULES}
        for seed in protocol["fresh_seeds"]:
            data = read(root / "results" / f"seed_{seed}_{task}.json")
            assert data["protocol_sha256"] == ex.digest(root / "fresh_protocol.json")
            for r in data["fixed_rate_endpoints"]:
                by_rule[r["rule"]].append(r["test_nmse"])
                hits[r["rule"]].append(r["bound_steps"] > 0)
        for rule, values in by_rule.items():
            values = np.array(values)
            lo, hi = np.quantile(values[resamples].mean(1), [.025, .975])
            rows.append(dict(task=task, rule=rule, mean=float(values.mean()),
                             ci_low=float(lo), ci_high=float(hi), n=20,
                             trajectories_hitting_bounds=sum(hits[rule]),
                             rate=protocol["fixed_rates"][task][rule]))
        for left, right in [("unit_broadcast", "resistance_gate"),
                            ("proportional_gate", "resistance_gate"),
                            ("resistance_gate", "exact")]:
            difference = np.array(by_rule[left]) - np.array(by_rule[right])
            lo, hi = np.quantile(difference[resamples].mean(1), [.025, .975])
            contrasts.append(dict(task=task, left=left, right=right,
                                  mean=float(difference.mean()), ci_low=float(lo), ci_high=float(hi),
                                  positive=int((difference > 0).sum()), n=20))
    result = dict(protocol_sha256=ex.digest(root / "fresh_protocol.json"),
                  rows=rows, paired_contrasts=contrasts)
    write_new(root / "fresh_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["freeze", "run", "report"])
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    if args.action == "run":
        fresh(args.root, args.seed)
    else:
        globals()[args.action](args.root)
