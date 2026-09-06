"""Meaningful numerical and information-contract checks before protocol freeze."""
from pathlib import Path
import inspect
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from bridge import (task,sample,arrays_hash,estimate_coefficients,select_scores,
    candidates,tree_payload,tree_from_payload,forward,domain,fourier_design,
    train_candidate,adaptive_tree)


def test_fresh_tasks_are_normalized_and_random_controls_have_no_singletons():
    for seed in (720910,820910):
        for family in range(4):
            coeff=task(seed,family)
            np.testing.assert_allclose(coeff@coeff,1,atol=1e-14)
            y=fourier_design(domain())@coeff
            np.testing.assert_allclose(y.mean(),0,atol=1e-14)
            np.testing.assert_allclose(np.mean(y*y),1,atol=1e-14)
            if family==3:
                assert len(np.flatnonzero(coeff))==4
                assert all(int(mask).bit_count()>=2 for mask in np.flatnonzero(coeff))
            assert not np.array_equal(task(seed,family),task(seed+1,family))


def test_sampling_replays_but_streams_and_repeated_label_noise_are_independent():
    coeff=task(720910,0)
    first=sample(coeff,720910,0,100,1024,.5)
    repeat=sample(coeff,720910,0,100,1024,.5)
    other=sample(coeff,720910,0,10,1024,.5)
    assert arrays_hash(*first)==arrays_hash(*repeat)
    assert arrays_hash(*first)!=arrays_hash(*other)
    patterns=first[3]
    repeated=np.flatnonzero(patterns==patterns[0])
    assert len(repeated)>1
    assert len(np.unique(first[1][repeated]))==len(repeated)


def test_estimator_only_accepts_observations_and_recovers_noiseless_sparse_signal():
    assert list(inspect.signature(estimate_coefficients).parameters)==["x","y","alpha_scale"]
    coeff=task(720910,2);x=domain();y=fourier_design(x)@coeff+2.0
    estimate,diagnostic=estimate_coefficients(x,y,1e-5)
    assert diagnostic["converged"]
    assert abs(diagnostic["intercept"]-2.0)<1e-10
    assert estimate[0]==0
    np.testing.assert_allclose(estimate,coeff,atol=1e-5)


def test_ties_use_declared_max_then_sum_then_lexicographic_rule():
    assert select_scores([("b",1.,2.),("a",1.+1e-11,2.)])=="a"
    assert select_scores([("a",1.+1e-8,0.),("b",1.,2.)])=="b"
    assert select_scores([("a",1.,2.),("b",1.,1.)])=="b"


def test_serialized_adaptive_tree_preserves_forward_map_and_budget():
    tree,_=adaptive_tree(task(720910,2),"test_adaptive")
    copied=tree_from_payload(tree_payload(tree))
    assert len(copied.children)==7 and len(copied.parent)==14
    weights=np.random.default_rng(4).normal(size=(7,4))
    np.testing.assert_array_equal(forward(domain(),tree,weights),forward(domain(),copied,weights))


def test_training_selects_restart_only_by_independent_validation():
    config=dict(training_rows=64,validation_rows=96,test_rows=128,training_noise=.3,sweeps=2,restarts=2)
    rows=train_candidate(candidates()[0],task(720910,0),720910,0,config)
    assert len(rows)==2 and all(row["status"]=="completed" for row in rows)
    selected=[row for row in rows if row["selected_by_validation"]]
    assert len(selected)==1
    assert selected[0]["validation_mse"]==min(row["validation_mse"] for row in rows)
