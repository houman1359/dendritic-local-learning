#!/usr/bin/env python3
"""Sealed finite-calibration tree choice followed by ordinary gradient learning."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from bridge import (FAMILIES,STRUCTURE,task,sample,arrays_hash,estimate_coefficients,
                    adaptive_tree,candidates,tree_payload,domain,fourier_design,cut_scores)
from run_serialization_fix_v2 import decode_numeric_order,run as calibration_run

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/"source_data/morphology_calibration/end_to_end"
CREDIT=HERE.parent/"morphology_credit/experiment.py"
spec=importlib.util.spec_from_file_location("frozen_credit_reference",CREDIT)
assert spec and spec.loader
credit=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=credit
spec.loader.exec_module(credit)
SEEDS=tuple(range(930910,930930))
STRUCTURES=("estimated_dp","development_fixed","oracle_dp")
RATES={"exact":.01,"broadcast":.003}
CHECKPOINTS=(0,1,16,64,256,1024)
sha,dump=calibration_run.sha,calibration_run.dump


def sources():
    return {str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).resolve(),
        HERE/"test_end_to_end.py",HERE/"bridge.py",HERE/"run_serialization_fix_v2.py",CREDIT,
        STRUCTURE/"model.py",STRUCTURE/"constructive_dp_v2.py",STRUCTURE/"constructive.py")}


def freeze():
    primary=calibration_run.frozen_protocol()
    rate_path=ROOT/"source_data/morphology_credit/development_fit.json"
    fitted=json.loads(rate_path.read_text())
    assert all(fitted["adam"][rule]==value for rule,value in RATES.items())
    config=dict(schema="finite_calibration_then_gradient_learning/v1",seeds=list(SEEDS),families=list(FAMILIES),
        task_generation="Same registered mixture as finite-calibration study, with20new independent seeds;random signs,magnitudes and assignments;unit target variance;not isospectral across tasks",
        calibration_rows=256,calibration_fit_rows=192,calibration_noise_sd=.5,
        alpha_scale=primary["selected_alpha_scale"],calibration_rule="Frozen generic255-monomial Lasso then audited subsetDP;no family/support/true coefficients;remaining64observations reserved,not used",
        fixed_candidate=primary["development_best_fixed"],structures=list(STRUCTURES),rates=RATES,
        optimizer="Adam beta1=.9,beta2=.999,epsilon1e-8;rates copied unchanged from prior credit development",
        steps=1024,checkpoints=list(CHECKPOINTS),batch_size=64,training_rows=2048,test_rows=4096,
        training_noise_sd=.15,test_noise_sd=.15,gradient_clip=10.,parameter_bound=2.,
        initialization="Common label-independent Normal(0,.5) coefficients,zero biases,clipped tobox[-2,2];same initialweights and sampled minibatches acrossall6conditions",
        training_objective="HalfMSE divided by true target variance1;exactpath orunitbroadcast credit with exactroot sensitivity1",
        endpoint="Independent noisy-test NMSE after1024updates;clean independent-test and exactpopulation NMSE secondary;no validation,earlystopping orrate tuning",
        streams="Independent SeedSequence(seed,family,role):calibration103,training210,test211,initialization212,minibatches213;training reuses its declared2048-row noisy cache",
        construction_access="EstimatedDP andfixedchoices sealed globally beforetraining. True-coefficient oracleDP constructed afterward as privilegedreference. No target-derived weights orconstruct_weights call.",
        primary_comparisons=["development_fixed exact minus estimated_dp exact", "estimated_dp broadcast minus estimated_dp exact"],
        inference="20paired seedblocks,4families averagedwithinseed;10000bootstrap samples;95%descriptive and97.5%Bonferroni intervals for2primary contrasts;meaningful margin.01NMSE",
        failure_policy="Retain nonfinitefailedcondition with1e6loss andstatus;continueotherconditions;recordgradientclipping andparameterprojection",
        scope="Direct composition test in bounded signed algebraictrees,notbiophysicalmorphology orunknown-familygeneralization",
        source_hashes=sources(),prior_calibration_protocol_sha256=sha(calibration_run.OUT/"protocol.json"),
        prior_credit_rate_record_sha256=sha(rate_path))
    path=OUT/"protocol.json"
    if path.exists():assert json.loads(path.read_text())==config
    else:
        dump(path,config);dump(OUT/"protocol_freeze.json",dict(protocol_sha256=sha(path),utc=pd.Timestamp.now(tz="UTC").isoformat()))
    return config


def verify():
    cfg=json.loads((OUT/"protocol.json").read_text())
    assert cfg["source_hashes"]==sources()
    assert sha(OUT/"protocol.json")==json.loads((OUT/"protocol_freeze.json").read_text())["protocol_sha256"]
    return cfg


def select():
    cfg=freeze()
    assert not (OUT/"selection_seal.json").exists()
    assert not list((OUT/"runs").glob("*.csv"))
    choices=[];data={}
    for seed in SEEDS:
        for family,name in enumerate(FAMILIES):
            coeff=task(seed,family)
            x,y,_,patterns=sample(coeff,seed,family,103,256,.5)
            estimate,diagnostic=estimate_coefficients(x[:192],y[:192],cfg["alpha_scale"])
            tree,bound=adaptive_tree(estimate,"estimated_dp")
            choices.append(dict(seed=seed,family=name,family_index=family,tree=tree_payload(tree),
                estimated_bound=bound,observed_rank995=diagnostic["observed_gradient_rank995"],
                lasso_converged=diagnostic["converged"],sample_sha256=arrays_hash(x,y),
                estimated_coefficients_sha256=arrays_hash(estimate),distinct_patterns=int(len(np.unique(patterns))),
                source_prefix=f"seed{seed}/family{family}/stream103"))
            key=f"{seed}_{family}";data[key+"_x"]=x.astype(np.int8);data[key+"_y"]=y
    dump(OUT/"choices.json",choices);np.savez_compressed(OUT/"calibration_samples.npz",**data)
    dump(OUT/"selection_seal.json",dict(utc=pd.Timestamp.now(tz="UTC").isoformat(),
        protocol_sha256=sha(OUT/"protocol.json"),choices_sha256=sha(OUT/"choices.json"),
        samples_sha256=sha(OUT/"calibration_samples.npz"),tasks=80,label_queries=20480,
        completed_training_fits=0,all_estimated_tree_choices_before_any_training=True))


def pack(trees):
    metadata=[];left=[];right=[]
    for name,tree in zip(STRUCTURES,trees):
        for rule in ("exact","broadcast"):
            metadata.append(dict(structure=name,rule=rule,optimizer="adam",rate=RATES[rule]))
            left.append([tree.children[node][0] for node in range(8,15)])
            right.append([tree.children[node][1] for node in range(8,15)])
    # Exact andbroadcast bypass proximal-zone construction entirely.
    return metadata,np.asarray(left),np.asarray(right),np.repeat(np.eye(6)[None],6,axis=0)


def train(seed,family,choice,cfg):
    coeff=task(seed,family)
    oracle,_,_=adaptive_tree(coeff,"oracle_dp")
    trees=[decode_numeric_order(choice["tree"]),next(t for t in candidates() if t.name==cfg["fixed_candidate"]),oracle]
    metadata,left,right,projectors=pack(trees)
    init=np.random.default_rng(np.random.SeedSequence([seed,family,212])).normal(0,.5,(7,4))
    init[:,0]=0.;init=np.clip(init,-2,2)
    weights=np.broadcast_to(init,(6,7,4)).copy()
    x,y,_,_=sample(coeff,seed,family,210,2048,.15)
    tx,ty,clean,_=sample(coeff,seed,family,211,4096,.15)
    px=domain();py=fourier_design(px)@coeff
    generator=np.random.default_rng(np.random.SeedSequence([seed,family,213]))
    m=np.zeros_like(weights);v=np.zeros_like(weights);rate=np.asarray([row["rate"] for row in metadata])[:,None,None]
    clip_counts=np.zeros(6,int);box_counts=np.zeros(6,int);failed=np.zeros(6,bool)
    bound=[cut_scores(coeff,tree)["centered_cut_bound"] for tree in trees]
    rows=[];tick=time.perf_counter()
    def evaluate(step):
        prediction=credit.forward(tx,weights,left,right)[:,14]
        pop=credit.forward(px,weights,left,right)[:,14]
        noisy=np.mean((prediction-ty[None])**2,axis=1)
        clean_loss=np.mean((prediction-clean[None])**2,axis=1)
        population=np.mean((pop-py[None])**2,axis=1)
        for i,meta in enumerate(metadata):
            losses=[noisy[i],clean_loss[i],population[i]]
            if not np.isfinite(losses).all():failed[i]=True
            if failed[i]:losses=[1e6]*3
            rows.append(dict(seed=seed,family=FAMILIES[family],family_index=family,**meta,step=step,
                test_nmse=float(losses[0]),clean_test_nmse=float(losses[1]),population_nmse=float(losses[2]),
                true_cut_bound=bound[i//2],status="failed_nonfinite" if failed[i] else "completed",
                gradient_clipped_steps=int(clip_counts[i]),parameter_projected_steps=int(box_counts[i]),
                max_abs_parameter=float(np.abs(weights[i]).max()),parameter_norm=float(np.linalg.norm(weights[i])),
                parameters=28,edges=14,seconds=time.perf_counter()-tick))
    evaluate(0)
    for step in range(1,1025):
        batch=generator.integers(len(x),size=64)
        g,_,_,_,_=credit.gradient(x[batch],y[batch],weights,left,right,projectors,metadata,1.)
        invalid=~np.isfinite(g).all(axis=(1,2));failed|=invalid;g[failed]=0.
        norm=np.linalg.norm(g,axis=(1,2));clip_counts+=norm>10
        g*=np.minimum(1.,10/np.maximum(norm,1e-30))[:,None,None]
        m=.9*m+.1*g;v=.999*v+.001*g*g
        update=(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
        weights-=rate*update;box_counts+=np.any(abs(weights)>2,axis=(1,2));weights=np.clip(weights,-2,2)
        if step in CHECKPOINTS:evaluate(step)
    audit=dict(seed=seed,family=FAMILIES[family],initial_weights_sha256=arrays_hash(init),
        training_sample_sha256=arrays_hash(x,y),test_sample_sha256=arrays_hash(tx,ty),
        calibration_sample_sha256=choice["sample_sha256"],
        source_roles={"calibration":103,"training":210,"test":211,"initialization":212,"minibatches":213},
        label_independent_initialization=True,common_initialization_and_batches=True,
        estimated_tree=tree_payload(trees[0]),oracle_tree=tree_payload(oracle),true_coefficients=coeff.tolist())
    return rows,weights,audit


def run_seed(index):
    cfg=verify();seed=SEEDS[index]
    seal=json.loads((OUT/"selection_seal.json").read_text())
    assert seal["protocol_sha256"]==sha(OUT/"protocol.json")
    assert seal["choices_sha256"]==sha(OUT/"choices.json")
    choices=json.loads((OUT/"choices.json").read_text());rows=[];audits=[];weights={}
    for family,name in enumerate(FAMILIES):
        choice=next(row for row in choices if row["seed"]==seed and row["family"]==name)
        result,w,audit=train(seed,family,choice,cfg);rows.extend(result);audits.append(audit);weights[name]=w
        print(seed,name,"complete",flush=True)
    target=OUT/"runs";target.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(target/f"seed_{seed}.csv",index=False)
    dump(target/f"seed_{seed}_audit.json",audits);np.savez_compressed(target/f"seed_{seed}_weights.npz",**weights)


def summarize():
    verify()
    data=pd.concat([pd.read_csv(OUT/"runs"/f"seed_{seed}.csv") for seed in SEEDS],ignore_index=True)
    assert len(data)==480*6
    data.to_csv(OUT/"trajectories.csv",index=False);end=data[data.step==1024]
    end.to_csv(OUT/"endpoints.csv",index=False)
    summaries=[];contrasts=[];paired=[]
    for label,group in [("all",end)]+list(end.groupby("family")):
        for (structure,rule),part in group.groupby(["structure","rule"]):
            units=part.groupby("seed").test_nmse.mean();ci=calibration_run.bootstrap(units)
            summaries.append(dict(family=label,structure=structure,rule=rule,mean_test_nmse=units.mean(),
                ci95_low=ci[0],ci95_high=ci[1],mean_clean_test_nmse=part.clean_test_nmse.mean(),seed_count=len(units)))
        pivot=group.groupby(["seed","structure","rule"]).test_nmse.mean().unstack(["structure","rule"])
        for name,baseline,selected in (
            ("fixed_exact_minus_estimated_exact",("development_fixed","exact"),("estimated_dp","exact")),
            ("estimated_broadcast_minus_exact",("estimated_dp","broadcast"),("estimated_dp","exact")),
            ("estimated_exact_minus_oracle_exact",("estimated_dp","exact"),("oracle_dp","exact"))):
            units=pivot[baseline]-pivot[selected];ci=calibration_run.bootstrap(units);adjusted=calibration_run.bootstrap(units,.975)
            contrasts.append(dict(family=label,contrast=name,mean_difference=units.mean(),ci95_low=ci[0],ci95_high=ci[1],
                ci975_low=adjusted[0],ci975_high=adjusted[1],seed_count=len(units),
                primary=label=="all" and name!="estimated_exact_minus_oracle_exact",
                meaningful_superiority=bool(adjusted[0]>0 and units.mean()>=.01)))
            paired.extend(dict(family=label,seed=int(seed),contrast=name,difference=float(value)) for seed,value in units.items())
    pd.DataFrame(summaries).to_csv(OUT/"summary.csv",index=False)
    pd.DataFrame(contrasts).to_csv(OUT/"contrasts.csv",index=False)
    pd.DataFrame(paired).to_csv(OUT/"paired_contrasts.csv",index=False)
    dump(OUT/"report.json",dict(status="complete",fits=480,rows=len(data),seeds=20,
        final_failures=int((end.status!="completed").sum()),max_abs_parameter=float(data.max_abs_parameter.max()),
        protocol_sha256=sha(OUT/"protocol.json"),selection_seal_sha256=sha(OUT/"selection_seal.json"),
        primary_contrasts=[row for row in contrasts if row["primary"]],
        scope="Fresh-function end-to-end composition in bounded multilinear model;primary noisy-testfloor.0225;not a conductance orunknown-familyvalidation"))
    print(pd.DataFrame(summaries).to_string(index=False));print(pd.DataFrame(contrasts).query("primary").to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["freeze","select","run","summarize"])
    parser.add_argument("--index",type=int,default=0);args=parser.parse_args()
    if args.action=="freeze":freeze()
    elif args.action=="select":select()
    elif args.action=="run":run_seed(args.index)
    else:summarize()
