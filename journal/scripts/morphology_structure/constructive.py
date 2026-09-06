#!/usr/bin/env python3
"""Exploratory exact tree construction from target interaction spectra.

Separate from the frozen candidate-screen experiment. Full truth-table access
is an oracle assumption; this is a capacity certificate, not a cheap selector.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import time
import numpy as np
import pandas as pd
from model import (Tree, candidates, tasks, domain, fourier_design, matricize,
                   cut_scores, input_gradient_covariance, forward, fit)

OUT = Path(__file__).resolve().parents[2] / "analysis/morphology_investigation_20260905/structure/constructive"


def prefix_tasks():
    rng = np.random.default_rng(984_60905)
    result = []
    for index in range(24):
        permutation = tuple(rng.permutation(8))
        coeff = np.zeros(256)
        for k in (2,4,6,8):
            coeff[sum(1 << int(i) for i in permutation[:k])] = 0.5
        result.append(dict(task_id=f"nested_prefix_{index:03d}", family="nested_prefix_control",
                           coefficients=coeff, permutation=permutation))
    return result


def optimal_bound_tree(coeff, name):
    costs = {}
    for mask in range(1,255):
        subset = tuple(i for i in range(8) if mask & (1<<i))
        if len(subset)==1:
            costs[mask] = 0.
        else:
            singular = np.linalg.svd(matricize(coeff, subset)[1:], compute_uv=False)
            costs[mask] = float(np.sum(singular[1:]**2)/(coeff@coeff))
    costs[255] = 0.

    @lru_cache(None)
    def solve(mask):
        if mask.bit_count()==1:
            return (0.,0.,0,0), None
        first = mask & -mask
        subset = (mask-1) & mask
        best = None
        while subset:
            other = mask ^ subset
            if other and subset & first:
                left, _ = solve(subset)
                right, _ = solve(other)
                score = (round(max(costs[mask],left[0],right[0]),12),
                         round(costs[mask]+left[1]+right[1],12),
                         1+max(left[2],right[2]),
                         abs(subset.bit_count()-other.bit_count())+left[3]+right[3])
                entry = (score, (subset,other))
                if best is None or entry < best:
                    best = entry
            subset = (subset-1) & mask
        return best

    _, root_split = solve(255)
    children, descendants, parent = {}, {i:(i,) for i in range(8)}, {}

    def join(mask):
        if mask.bit_count()==1:
            return mask.bit_length()-1
        _, (left_mask,right_mask) = solve(mask)
        left,right = join(left_mask),join(right_mask)
        node = 8+len(children)
        children[node] = (left,right)
        descendants[node] = descendants[left]+descendants[right]
        parent[left],parent[right] = (node,0),(node,1)
        return node

    root = join(255)
    tree = Tree(name,"adaptive",descendants[root],children,descendants,parent,root)
    return tree, solve(255)[0]


def construct_weights(coeff, tree, x, design):
    """Recover subtree features and four local coefficients with full target access."""
    weights = np.zeros((7,4))
    local_residuals = []
    for node,(left,right) in tree.children.items():
        if node == tree.root:
            target = design @ coeff
        else:
            support = tree.descendants[node]
            matrix = matricize(coeff,support)[1:]
            u,s,_ = np.linalg.svd(matrix,full_matrices=False)
            global_coeff = np.zeros(256)
            if len(s) and s[0]>1e-12:
                for local_mask,value in enumerate(u[:,0],start=1):
                    mask = sum(((local_mask>>j)&1)<<leaf for j,leaf in enumerate(support))
                    global_coeff[mask] = value
            target = design @ global_coeff
        values = forward(x,tree,weights)
        features = np.column_stack((np.ones(len(x)),values[left],values[right],values[left]*values[right]))
        weights[node-8] = np.linalg.lstsq(features,target,rcond=1e-12)[0]
        local_residuals.append(float(np.mean((features@weights[node-8]-target)**2)))
    output = forward(x,tree,weights)[tree.root]
    return weights, float(np.mean((output-design@coeff)**2)/(coeff@coeff)), max(local_residuals)


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    sources = [Path(__file__).resolve(),Path(__file__).with_name("model.py")]
    protocol = dict(status="exploratory_construction_after_fixed_candidate_screen",
        target_access="Full256-pattern target table and exact Fourier tensor;oracle capacity study",
        task_counts=dict(quadratic_matching=105,quartic_partition=35,nested_prefix_control=24),
        nested_prefix="24 seeded permutations,coeff0.5 on prefixesoflength2,4,6,8;rank8 with fixed anisotropic gradient spectrum within family",
        optimizer="DP over allinputsubsets minimizes(max centeredcut tail,sumtail,depth,totalimbalance,lexicographic split);roundtailcoststo12decimals forties",
        construction="Dominant centered Fourier direction per subtree,then exact local least squares;report all reconstruction errors",
        control="For prefix24 only, original12 candidates,four fixed random ALSrestarts,32sweeps;alloutcomesretained",
        source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    path = OUT/"protocol.json"
    if path.exists():
        assert json.loads(path.read_text())==protocol
    else:
        path.write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n")
        (OUT/"freeze.json").write_text(json.dumps(dict(utc=pd.Timestamp.now(tz="UTC").isoformat(),
            protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),indent=2)+"\n")


def run():
    freeze()
    x=domain()
    design=fourier_design(x)
    rows,manifest,comparisons=[],[],[]
    for task_index,task in enumerate(tasks()+prefix_tasks()):
        start=time.perf_counter()
        coeff=task["coefficients"]
        tree,score=optimal_bound_tree(coeff,task["task_id"]+"_adaptive")
        weights,nmse,local_error=construct_weights(coeff,tree,x,design)
        spectrum=np.linalg.eigvalsh(input_gradient_covariance(coeff))
        rows.append(dict(task_id=task["task_id"],family=task["family"],bound=score[0],sum_bound=score[1],depth=score[2],
            normalized_mse=nmse,max_local_reconstruction_mse=local_error,rank=int(np.sum(spectrum>1e-10)),
            effective_rank=float(spectrum.sum()**2/np.sum(spectrum**2)),parameters=28,edges=14,
            seconds=time.perf_counter()-start))
        manifest.append(dict(task_id=task["task_id"],children={str(k):[int(v) for v in pair] for k,pair in tree.children.items()},
            weights=weights.tolist(),input_gradient_second_moment_spectrum=spectrum.tolist()))
        if task["family"]=="nested_prefix_control":
            y=design@coeff
            for candidate in candidates():
                bound=cut_scores(coeff,candidate)["centered_cut_bound"]
                for restart in range(4):
                    _,curve=fit(x,y,candidate,9_846_090+100*task_index+restart,sweeps=32)
                    comparisons.append(dict(task_id=task["task_id"],candidate_id=candidate.name,shape=candidate.shape,
                        restart=restart,normalized_mse=curve[32],centered_cut_bound=bound))
        print(task["task_id"],"adaptive_error",nmse,"depth",score[2],flush=True)
    pd.DataFrame(rows).to_csv(OUT/"adaptive_constructions.csv",index=False)
    pd.DataFrame(comparisons).to_csv(OUT/"prefix_candidate_fits.csv",index=False)
    (OUT/"constructed_trees.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(pd.DataFrame(rows).groupby("family").agg(tasks=("task_id","size"),max_error=("normalized_mse","max"),
        mean_depth=("depth","mean"),min_depth=("depth","min"),max_depth=("depth","max")).to_string())


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["freeze","run"])
    args=parser.parse_args()
    if args.action=="freeze":freeze()
    else:run()
