#!/usr/bin/env python3
"""Two-pass DP audit: exact minimax bound, then total bound and depth.

The initial constructive diagnostic used local lexicographic tie breaking.
That is valid for its primary minimax objective, and for minimum depth when
all cut tails vanish, but is not a generic global total-tail minimizer. This
version fixes that generic secondary-objective issue without changing any
retained training results or frozen protocol.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from model import Tree, tasks, domain, fourier_design, matricize, cut_scores
from constructive import prefix_tasks, construct_weights

OUT=Path(__file__).resolve().parents[2]/"analysis/morphology_investigation_20260905/structure/constructive_dp_v2"


def solve_costs(costs,n):
    full=(1<<n)-1

    @lru_cache(None)
    def partitions(mask):
        first=mask & -mask
        result=[]
        subset=(mask-1)&mask
        while subset:
            other=mask^subset
            if other and subset & first:
                result.append((subset,other))
            subset=(subset-1)&mask
        return tuple(result)

    @lru_cache(None)
    def minimax(mask):
        if mask.bit_count()==1:return 0.
        return max(costs[mask],min(max(minimax(a),minimax(b)) for a,b in partitions(mask)))

    threshold=minimax(full)

    @lru_cache(None)
    def feasible(mask):
        if mask.bit_count()==1:return ((0.,0),None)
        if costs[mask]>threshold+1e-12:return None
        possibilities=[]
        for a,b in partitions(mask):
            left,right=feasible(a),feasible(b)
            if left is None or right is None:continue
            score=(round(costs[mask]+left[0][0]+right[0][0],12),1+max(left[0][1],right[0][1]))
            possibilities.append((score,(a,b)))
        return min(possibilities) if possibilities else None
    assert feasible(full) is not None
    return threshold,feasible,partitions


def tree_from_coeff(coeff,name):
    costs={255:0.}
    for mask in range(1,255):
        support=tuple(i for i in range(8) if mask&(1<<i))
        singular=np.linalg.svd(matricize(coeff,support)[1:],compute_uv=False)
        costs[mask]=round(float(np.sum(singular[1:]**2)/(coeff@coeff)),12)
    threshold,choice,_=solve_costs(costs,8)
    children,descendants,parent={},{i:(i,) for i in range(8)},{}
    def join(mask):
        if mask.bit_count()==1:return mask.bit_length()-1
        _,(a,b)=choice(mask)
        left,right=join(a),join(b)
        node=8+len(children)
        children[node]=(left,right)
        descendants[node]=descendants[left]+descendants[right]
        parent[left],parent[right]=(node,0),(node,1)
        return node
    root=join(255)
    return Tree(name,"adaptive",descendants[root],children,descendants,parent,root),threshold,choice(255)[0]


def test_dp_against_all_six_input_trees():
    rng=np.random.default_rng(9946)
    for trial in range(4):
        n=6;full=(1<<n)-1
        costs={m:(0. if m.bit_count()==1 else float(rng.integers(0,8))/8) for m in range(1,full+1)}
        costs[full]=0.
        bound,choice,partitions=solve_costs(costs,n)
        @lru_cache(None)
        def exhaustive(mask):
            if mask.bit_count()==1:return ((0.,0.,0),)
            results=[]
            for a,b in partitions(mask):
                for left in exhaustive(a):
                    for right in exhaustive(b):
                        results.append((max(costs[mask],left[0],right[0]),costs[mask]+left[1]+right[1],1+max(left[2],right[2])))
            return tuple(results)
        values=exhaustive(full)
        assert len(values)==945
        expected=min(values)
        actual=(bound,)+choice(full)[0]
        assert actual==expected,(actual,expected)


def main():
    test_dp_against_all_six_input_trees()
    OUT.mkdir(parents=True,exist_ok=True)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__).resolve(),Path(__file__).with_name("model.py"),Path(__file__).with_name("constructive.py"))}
    protocol=dict(status="separate_posthoc_deterministic_algorithm_audit_no_training",source_hashes=hashes,
        objective="Exact minmax centeredtail; constrain allcuts to that threshold; minimize total tail then depth; lexicographic remainingties",
        validation="Exhaustive enumeration of945six-input binarytrees on4randomcosttables,plus all164target reconstructions",
        outcomes="All164included; initialconstructive outcomesand training untouched")
    path=OUT/"protocol.json"
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2,sort_keys=True)+"\n")
    x=domain();design=fourier_design(x)
    rows,trees=[],[]
    for task in tasks()+prefix_tasks():
        coeff=task["coefficients"]
        tree,bound,(total,depth)=tree_from_coeff(coeff,task["task_id"]+"_adaptive_v2")
        weights,error,local_error=construct_weights(coeff,tree,x,design)
        actual_bound=cut_scores(coeff,tree)["centered_cut_bound"]
        assert actual_bound<=bound+1e-10
        rows.append(dict(task_id=task["task_id"],family=task["family"],bound=bound,total_bound=total,depth=depth,
            normalized_mse=error,max_local_error=local_error,parameter_norm=float(np.linalg.norm(weights)),
            max_abs_parameter=float(abs(weights).max())))
        trees.append(dict(task_id=task["task_id"],children={str(k):list(v) for k,v in tree.children.items()},weights=weights.tolist()))
    pd.DataFrame(rows).to_csv(OUT/"adaptive_constructions.csv",index=False)
    (OUT/"constructed_trees.json").write_text(json.dumps(trees,indent=2)+"\n")
    print(pd.DataFrame(rows).groupby("family").agg(n=("task_id","size"),max_error=("normalized_mse","max"),
        depth=("depth","mean"),max_abs_weight=("max_abs_parameter","max")).to_string())


if __name__=="__main__":main()
