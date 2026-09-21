"""Validate the published follow-up's inferential units and numerical joins."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

J = Path(__file__).resolve().parents[1]
DATA = J / "source_data/curated_publication"


def read(name):
    with (DATA / f"gate_generalization_{name}.csv").open() as handle:
        return list(csv.DictReader(handle))


def selected():
    return [r for r in read("endpoints") if r["fixed_development_rate"] == "True"]


def test_complete_paired_design_and_fixed_rates():
    rows = read("endpoints")
    chosen = selected()
    protocol = json.loads((DATA / "gate_generalization_provenance.json").read_text())["protocol"]
    assert len(rows) == 900 and len(chosen) == 300
    assert len({(r["seed"], r["task"], r["rule"], r["rate"]) for r in rows}) == 900
    assert set(map(int, (r["seed"] for r in rows))) == set(protocol["fresh_seeds"])
    assert not set(protocol["development_seeds"]) & set(protocol["fresh_seeds"])
    for r in chosen:
        assert float(r["rate"]) == protocol["fixed_rates"][r["task"]][r["rule"]]
        assert 0 <= int(r["selected_step"]) <= 4096


def test_means_and_bound_counts_match_per_seed_outcomes():
    rows = selected()
    for mean in read("summary"):
        group = [r for r in rows if r["task"] == mean["task"] and r["rule"] == mean["rule"]]
        assert len(group) == int(mean["n"]) == 20
        np.testing.assert_allclose(np.mean([float(r["test_nmse"]) for r in group]), float(mean["mean"]), atol=1e-14)
        assert sum(int(r["bound_steps"]) > 0 for r in group) == int(mean["trajectories_hitting_bounds"])


def test_contrasts_pair_seeds_not_training_runs():
    rows = {(r["task"], r["rule"], int(r["seed"])): float(r["test_nmse"]) for r in selected()}
    for contrast in read("contrasts"):
        seeds = sorted(seed for task, rule, seed in rows
                       if task == contrast["task"] and rule == contrast["left"])
        differences = np.array([rows[contrast["task"], contrast["left"], seed] -
                                rows[contrast["task"], contrast["right"], seed] for seed in seeds])
        assert len(differences) == int(contrast["n"]) == 20
        np.testing.assert_allclose(differences.mean(), float(contrast["mean"]), atol=1e-14)
        assert int((differences > 0).sum()) == int(contrast["positive"])


def test_published_generators_match_frozen_execution_sources():
    protocol = json.loads((DATA / "gate_generalization_provenance.json").read_text())["protocol"]
    for original, expected in protocol["source_sha256"].items():
        relative = original.split("/scripts/", 1)[1]
        path = J / "scripts" / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
