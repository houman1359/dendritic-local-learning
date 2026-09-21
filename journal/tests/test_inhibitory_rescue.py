"""Integrity checks for every outcome in the parent-sensitivity rescue."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

J = Path(__file__).resolve().parents[1]
D = J / "source_data/curated_publication"


def read(name):
    return pd.read_csv(D / f"inhibitory_rescue_{name}.csv")


def test_complete_fresh_cohort_and_hashes():
    meta = json.loads((D / "inhibitory_rescue_provenance.json").read_text())
    p = meta["protocol"]
    ep = read("endpoints")
    assert len(ep) == len(p["jobs"]) == 320
    assert ep.seed.nunique() == 20
    assert not ep.duplicated(["seed", "optimizer", "bound", "rule"]).any()
    assert set(ep.seed) == set(p["fresh_seeds"])
    assert not set(ep.seed) & set(p["development_seeds"])
    prior = pd.read_csv(D / "inhibitory_selection_endpoints.csv")
    assert not set(ep.seed) & set(prior.seed)
    assert meta["maximum_replay_error"] < 1e-10
    for name, digest in meta["outputs"].items():
        assert hashlib.sha256((D / f"inhibitory_rescue_{name}.csv").read_bytes()).hexdigest() == digest
    expected = {(j["seed"], j["optimizer"], j["bound"], j["rule"]): j["rate"] for j in p["jobs"]}
    for row in ep.itertuples():
        assert row.rate == expected[row.seed, row.optimizer, row.bound, row.rule]
        assert 0 <= row.selected_step <= 4096


def test_every_summary_and_contrast_recomputes_from_seed_rows():
    ep = read("endpoints")
    for row in read("summary").itertuples():
        group = ep[ep.optimizer.eq(row.optimizer) & ep.bound.eq(row.bound) & ep.rule.eq(row.rule)]
        assert len(group) == row.n == 20
        np.testing.assert_allclose(group[row.metric].mean(), row.mean, rtol=1e-10)
        assert int(group.bounds.gt(0).sum()) == row.bound_runs
    contrast = read("contrasts")
    assert int(contrast.primary.sum()) == 2
    for row in contrast.itertuples():
        group = ep[ep.optimizer.eq(row.optimizer) & ep.bound.eq(row.bound)]
        wide = group.pivot(index="seed", columns="rule", values=row.metric)
        diff = wide[row.left] - wide[row.right]
        np.testing.assert_allclose(diff.mean(), row.mean, rtol=1e-10, atol=1e-14)
        assert int(diff.gt(0).sum()) == row.positive
        assert len(diff) == row.n == 20
    assert contrast.loc[contrast.primary, "holm_p"].notna().all()
    assert contrast.loc[~contrast.primary, "signflip_p"].isna().all()


def test_full_factor_is_an_exact_bp_diagnostic_not_an_independent_effect():
    ep = read("endpoints")
    for optimizer in ["adam", "sgd"]:
        group = ep[ep.optimizer.eq(optimizer) & ep.bound.eq(9)]
        for metric in ["test_nmse", "validation_nmse", "ood_3.0"]:
            wide = group.pivot(index="seed", columns="rule", values=metric)
            np.testing.assert_allclose(wide.full_chain, wide.exact, rtol=1e-8, atol=1e-11)


def test_common_rate_reuses_seeds_and_identical_primary_trajectories():
    meta = json.loads((D / "inhibitory_rescue_common_provenance.json").read_text())
    ep = read("common_endpoints")
    assert len(ep) == 120 and ep.seed.nunique() == 20
    assert not ep.duplicated(["seed", "rule"]).any()
    assert set(ep.rate) == {0.03}
    assert meta["maximum_replay_error"] < 1e-10
    for name, digest in meta["outputs"].items():
        assert hashlib.sha256((D / name).read_bytes()).hexdigest() == digest
    primary = read("endpoints")
    primary = primary[primary.optimizer.eq("adam") & primary.bound.eq(9) & primary.rate.eq(0.03)]
    assert set(primary.seed) == set(ep.seed)
    for row in primary.itertuples():
        reuse = ep[ep.seed.eq(row.seed) & ep.rule.eq(row.rule)].iloc[0]
        assert reuse.test_nmse == row.test_nmse
        assert reuse.selected_step == row.selected_step
    for row in read("common_contrasts").itertuples():
        wide = ep.pivot(index="seed", columns="rule", values=row.metric)
        diff = wide[row.left] - wide[row.right]
        np.testing.assert_allclose(diff.mean(), row.mean, rtol=1e-10, atol=1e-14)
        assert int(diff.gt(0).sum()) == row.positive
        assert len(diff) == row.n == 20


def test_supplement_includes_rescue_and_preserves_original_failure():
    main = (J / "main.tex").read_text()
    si = (J / "supplementary/supplementary.tex").read_text()
    assert r"\input{curated/si_05_parent_sensitivity}" in si
    assert r"\input{curated/si_parent_sensitivity_table}" in si
    assert "mean ordinary-test nmse was slightly worse for the gate than for broadcast, 0.0511 versus 0.0471" in main.casefold()
    assert "while fixing the nonlinear task and forward model" in main
    assert "Supplementary Table~S15" in main
    assert "not a biological mechanism for transmitting it" in main
