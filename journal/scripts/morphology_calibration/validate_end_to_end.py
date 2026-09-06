#!/usr/bin/env python3
"""Independent replay from retained weights and sealed calibration observations."""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd

from run_end_to_end_fix import experiment,activate
from bridge import (estimate_coefficients,adaptive_tree,arrays_hash,task,sample,
    candidates,tree_payload,domain,fourier_design)
from run_serialization_fix_v2 import decode_numeric_order


def main():
    started=time.perf_counter();cfg=experiment.verify();out=experiment.OUT
    activate()
    seal=json.loads((out/"selection_seal.json").read_text())
    amendment=json.loads((out/"runtime_amendment.json").read_text())
    from pathlib import Path
    assert amendment["runtime_fix_sha256"]==experiment.sha(Path(__file__).with_name("run_end_to_end_fix.py"))
    assert amendment["protocol_sha256"]==experiment.sha(out/"protocol.json")
    assert amendment["selection_seal_sha256"]==experiment.sha(out/"selection_seal.json")
    assert experiment.sha(out/"choices.json")==seal["choices_sha256"]
    assert experiment.sha(out/"calibration_samples.npz")==seal["samples_sha256"]
    choices=json.loads((out/"choices.json").read_text());cal=np.load(out/"calibration_samples.npz")
    data=pd.read_csv(out/"endpoints.csv",float_precision="round_trip")
    assert len(data)==480
    records=[];max_error=0.;sealed_time=pd.Timestamp(seal["utc"]).timestamp()
    for seed in experiment.SEEDS:
        path=out/"runs"/f"seed_{seed}.csv";assert path.stat().st_mtime>=sealed_time
        weights=np.load(out/"runs"/f"seed_{seed}_weights.npz")
        audits=json.loads((out/"runs"/f"seed_{seed}_audit.json").read_text())
        original=pd.read_csv(path,float_precision="round_trip");original=original[original.step==1024]
        for family,name in enumerate(experiment.FAMILIES):
            choice=next(row for row in choices if row["seed"]==seed and row["family"]==name)
            key=f"{seed}_{family}";cx=cal[key+"_x"].astype(float);cy=cal[key+"_y"]
            assert arrays_hash(cx,cy)==choice["sample_sha256"]
            estimate,diagnostic=estimate_coefficients(cx[:192],cy[:192],cfg["alpha_scale"])
            assert arrays_hash(estimate)==choice["estimated_coefficients_sha256"]
            tree,_=adaptive_tree(estimate,"estimated_dp")
            assert tree_payload(tree)==choice["tree"]
            coeff=task(seed,family);oracle,_,_=experiment.adaptive_tree(coeff,"oracle_dp")
            trees=[decode_numeric_order(choice["tree"]),next(t for t in candidates() if t.name==cfg["fixed_candidate"]),oracle]
            metadata,left,right,_=experiment.pack(trees)
            tx,ty,clean,_=sample(coeff,seed,family,211,4096,.15)
            x,y,_,_=sample(coeff,seed,family,210,2048,.15)
            audit=next(row for row in audits if row["family"]==name)
            assert arrays_hash(tx,ty)==audit["test_sample_sha256"]
            assert arrays_hash(x,y)==audit["training_sample_sha256"]
            init=np.random.default_rng(np.random.SeedSequence([seed,family,212])).normal(0,.5,(7,4))
            init[:,0]=0;init=np.clip(init,-2,2)
            assert arrays_hash(init)==audit["initial_weights_sha256"]
            assert len(set(audit["source_roles"].values()))==5
            w=weights[name];assert w.shape==(6,7,4) and np.max(abs(w))<=2
            assert np.isfinite(w).all()
            predicted=experiment.credit.forward(tx,w,left,right)[:,14]
            full=experiment.credit.forward(domain(),w,left,right)[:,14]
            truth=fourier_design(domain())@coeff
            errors=[]
            for index,meta in enumerate(metadata):
                row=original[(original.family==name)&(original.structure==meta["structure"])&
                             (original.rule==meta["rule"])].iloc[0]
                metrics=dict(test_nmse=np.mean((predicted[index]-ty)**2),
                    clean_test_nmse=np.mean((predicted[index]-clean)**2),
                    population_nmse=np.mean((full[index]-truth)**2))
                error=max(abs(float(value)-float(row[key])) for key,value in metrics.items())
                assert error<1e-10
                errors.append(error);max_error=max(max_error,error)
            records.append(dict(seed=seed,family=name,calibration_selection_reproduced=True,
                weights_and_endpoints_reproduced=6,max_endpoint_error=max(errors),
                source_roles_disjoint=True,label_independent_initializer_reproduced=True))
    pd.DataFrame(records).to_csv(out/"independent_validation_rows.csv",index=False)
    experiment.dump(out/"validation_report.json",dict(status="passed",tasks=80,seeds=20,fits=480,
        frozen_sources_verified=True,runtime_amendment_verified=True,choices_and_samples_hashes_verified=True,
        all_outcome_files_after_global_selection_seal=True,all80finite_calibration_constructions_reproduced=True,
        all480retained_weights_reproduce_endpoints=True,max_endpoint_absolute_error=max_error,
        source_stream_and_initializer_checks=80,parameters_within_declared_box=True,
        prior_generic_gradient_finite_difference_tests=2,seconds=time.perf_counter()-started,
        limitation="Independent observation/source streams permit repeated input patterns; no unknown-family or biological claim"))
    print((out/"validation_report.json").read_text())


if __name__=="__main__":main()
