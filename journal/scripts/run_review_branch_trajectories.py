#!/usr/bin/env python3
"""Re-execute the frozen Fashion task and measure gradients during learning.

At every checkpoint each delivered gradient is compared to the exact gradient
at the SAME weights. A separate common-exact-state comparison isolates routing
from diverging training trajectories. Historical endpoint drift is retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_path_necessity_fashion as base

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"source_data/review_branch_trajectories"
CHECKPOINTS=(0,1,10,50,100,250)


def geometry(exact, delivered):
    a=exact.flatten(1); b=delivered.flatten(1)
    dot=(a*b).sum(1); a2=(a*a).sum(1); b2=(b*b).sum(1)
    cosine=dot/torch.sqrt(a2*b2).clamp_min(1e-30)
    ratio=dot/a2.clamp_min(1e-30)
    return cosine.cpu().numpy(),ratio.cpu().numpy(),dot.cpu().numpy()


def run(seed,device,dataset_root=None):
    cfg=base.load_config()
    if dataset_root is not None:cfg["dataset"]["root"]=str(dataset_root)
    assert seed in cfg["confirmatory_seeds"]
    pools=base.load_feature_pools(cfg)
    rows=[]; calls=0
    def instrumented(initial, values, context, labels, routes, config):
        nonlocal calls
        alpha=config["task"]["conflict_probability"][calls % len(config["task"]["conflict_probability"])]
        branches=initial.shape[0]; calls+=1
        weights=initial[None].repeat(len(base.CONDITIONS),1,1)
        exact_routes=torch.eye(branches,device=device,dtype=weights.dtype)[None].repeat(len(weights),1,1)
        for epoch in range(config["training"]["epochs"]+1):
            delivered=base.analytic_gradients(weights,values,context,labels,routes)
            if epoch in CHECKPOINTS:
                exact=base.analytic_gradients(weights,values,context,labels,exact_routes)
                own=geometry(exact,delivered)
                common_weights=weights[1:2].repeat(len(weights),1,1)
                common_exact=base.analytic_gradients(common_weights,values,context,labels,exact_routes)
                common_route=base.analytic_gradients(common_weights,values,context,labels,routes)
                common=geometry(common_exact,common_route)
                for index,condition in enumerate(base.CONDITIONS):
                    for mode,metrics in [("own_state",own),("common_exact_state",common)]:
                        rows.append(dict(seed=seed,branches=branches,conflict_probability=alpha,
                            epoch=epoch,condition=condition,state_comparison=mode,
                            gradient_cosine=float(metrics[0][index]),signed_signal_ratio=float(metrics[1][index]),
                            gradient_inner_product=float(metrics[2][index])))
            if epoch==config["training"]["epochs"]: break
            weights-=config["training"]["learning_rate"]*delivered
        return weights
    original=base.train_models
    base.train_models=instrumented
    try:
        endpoints,audits,metadata=base.run_seed(seed,cfg,pools,device)
    finally:
        base.train_models=original
    old=pd.read_csv(ROOT/"source_data/path_necessity_fashion/seed_outcomes.csv")
    keys=["seed","branches","conflict_probability","condition"]
    # Stable dose IDs avoid round-trip representation mismatches on 4/7,2/3.
    for frame in (old,endpoints): frame["conflict_probability"]=frame.conflict_probability.round(10)
    comparison=endpoints.merge(old,on=keys,suffixes=("_new","_historical"),validate="one_to_one")
    assert len(comparison)==120
    comparison["test_accuracy_replay_error"]=(comparison.test_accuracy_new-comparison.test_accuracy_historical).abs()
    comparison["test_loss_replay_error"]=(comparison.test_loss_new-comparison.test_loss_historical).abs()
    out=OUT/"runs";out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/f"seed_{seed}.csv",index=False,float_format="%.12g")
    comparison.to_csv(out/f"endpoints_{seed}.csv",index=False,float_format="%.12g")
    metadata.update(diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoints=CHECKPOINTS,maximum_accuracy_replay_error=float(comparison.test_accuracy_replay_error.max()),
        maximum_loss_replay_error=float(comparison.test_loss_replay_error.max()))
    (out/f"seed_{seed}.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"seed {seed}: {len(rows)} diagnostic rows; max accuracy replay drift {metadata['maximum_accuracy_replay_error']:.3g}",flush=True)


def aggregate():
    cfg=base.load_config()
    frames=[pd.read_csv(OUT/"runs"/f"seed_{s}.csv") for s in cfg["confirmatory_seeds"]]
    all_rows=pd.concat(frames,ignore_index=True)
    assert len(all_rows)==20*3*8*5*6*2
    all_rows.to_csv(OUT/"gradient_trajectories.csv",index=False,float_format="%.12g")
    keys=["branches","conflict_probability","epoch","condition","state_comparison"]
    summary=all_rows.groupby(keys).agg(mean_cosine=("gradient_cosine","mean"),
        sd_cosine=("gradient_cosine","std"),mean_signed_signal=("signed_signal_ratio","mean"),
        negative_gradient_seeds=("gradient_inner_product",lambda x:int((x<0).sum())),
        n_seeds=("seed","nunique")).reset_index()
    summary.to_csv(OUT/"condition_summary.csv",index=False,float_format="%.12g")
    endpoints=pd.concat([pd.read_csv(OUT/"runs"/f"endpoints_{s}.csv") for s in cfg["confirmatory_seeds"]])
    endpoints.to_csv(OUT/"endpoint_replay.csv",index=False,float_format="%.12g")
    crossings=[]
    selected=all_rows[all_rows.condition.eq("neuron_shared_k1")]
    for key,group in selected.groupby(["seed","branches","epoch","state_comparison"]):
        group=group.sort_values("conflict_probability")
        x=group.conflict_probability.to_numpy();y=group.gradient_inner_product.to_numpy()
        idx=np.flatnonzero((y[:-1]>=0)&(y[1:]<0))
        cross=float(x[idx[0]]-y[idx[0]]*(x[idx[0]+1]-x[idx[0]])/(y[idx[0]+1]-y[idx[0]])) if len(idx) else np.nan
        crossings.append(dict(zip(["seed","branches","epoch","state_comparison"],key),
            interpolated_zero_alignment=cross,no_observed_positive_to_negative_crossing=not len(idx)))
    pd.DataFrame(crossings).to_csv(OUT/"zero_alignment_crossings.csv",index=False,float_format="%.12g")
    report={"complete":True,"diagnostic_rows":len(all_rows),"seeds":20,"checkpoints":CHECKPOINTS,
        "maximum_accuracy_replay_error":float(endpoints.test_accuracy_replay_error.max()),
        "maximum_loss_replay_error":float(endpoints.test_loss_replay_error.max()),
        "scope":"Newly re-executed trajectories of the existing frozen task; no new biological replication. No-alignment crossings are retained as censored/absent observations, never discarded. Own-state and common-state geometry are separate. A zero inner product is an instantaneous descent boundary, not a trained chance crossing."}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUT/"README.md").write_text("# Branch-gradient trajectories\n\n"+report["scope"]+"\n")
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int);parser.add_argument("--aggregate",action="store_true");parser.add_argument("--device",default="cuda")
    parser.add_argument("--dataset-root",type=Path,help="Location-only override for a hash-recorded copy of the original benchmark")
    args=parser.parse_args();torch.set_num_threads(1)
    if args.aggregate:aggregate()
    elif args.seed is not None:run(args.seed,base._torch_device(args.device),args.dataset_root)
    else:parser.error("provide --seed or --aggregate")


if __name__=="__main__":main()
