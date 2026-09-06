#!/usr/bin/env python3
"""Complete excluded-development random-control smoke before confirmatory retry."""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from run_end_to_end_fix import experiment,activate
from bridge import task,sample,estimate_coefficients,adaptive_tree,tree_payload,arrays_hash


def main():
    cfg=experiment.verify();activate();start=time.perf_counter()
    seed,family=721000,3
    assert seed not in cfg["seeds"]
    coeff=task(seed,family);x,y,_,_=sample(coeff,seed,family,103,256,.5)
    estimated,_=estimate_coefficients(x[:192],y[:192],cfg["alpha_scale"])
    tree,_=adaptive_tree(estimated,"estimated_dp")
    choice=dict(tree=tree_payload(tree),sample_sha256=arrays_hash(x,y))
    rows,weights,audit=experiment.train(seed,family,choice,cfg)
    destination=experiment.OUT/"excluded_smoke";destination.mkdir(parents=True,exist_ok=True)
    path=destination/"trajectories.csv";pd.DataFrame(rows).to_csv(path,index=False)
    np.savez_compressed(destination/"weights.npz",weights=weights)
    experiment.dump(destination/"audit.json",audit)
    frame=pd.read_csv(path,float_precision="round_trip")
    assert len(frame)==36 and frame.step.nunique()==6 and len(frame[frame.step==1024])==6
    assert set(frame.status)=={"completed"}
    restored=np.load(destination/"weights.npz")["weights"]
    np.testing.assert_array_equal(weights,restored)
    assert json.loads((destination/"audit.json").read_text())["initial_weights_sha256"]==audit["initial_weights_sha256"]
    assert np.isfinite(frame[["test_nmse","clean_test_nmse","population_nmse"]]).all().all()
    experiment.dump(destination/"report.json",dict(status="passed",excluded_seed=seed,
        family="random_interactions",complete_fits=6,steps_per_fit=1024,checkpoint_rows=36,
        full_initialization_training_evaluation_serialization_roundtrip=True,
        finite_endpoints=True,no_scientific_choices_changed=True,seconds=time.perf_counter()-start,
        trajectory_sha256=experiment.sha(path),weights_sha256=experiment.sha(destination/"weights.npz")))
    print((destination/"report.json").read_text())


if __name__=="__main__":main()
