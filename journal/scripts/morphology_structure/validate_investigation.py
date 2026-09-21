#!/usr/bin/env python3
"""Cross-check retained experiment counts, frozen inputs and constructive claims."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from model import Tree, tasks, domain, fourier_design, forward
from constructive import prefix_tasks
from constructive_dp_v2 import test_dp_against_all_six_input_trees

JOURNAL=Path(__file__).resolve().parents[2]
BASE=JOURNAL/"analysis/morphology_investigation_20260905"


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    checks=[]
    structure=BASE/"structure"
    cfg=json.loads((structure/"protocol.json").read_text())
    for name,digest in cfg["source_hashes"].items():
        assert sha(JOURNAL/"scripts/morphology_structure"/name)==digest
    trajectories=pd.read_csv(structure/"all_trajectories.csv")
    assert len(trajectories)==140*12*4*4
    assert trajectories.task_id.nunique()==140
    assert trajectories.groupby(["task_id","candidate_id","restart"]).size().eq(4).all()
    assert np.isfinite(trajectories.normalized_mse).all()
    for i in range(14):
        audit=json.loads((structure/"shards"/f"shard_{i:02d}_audit.json").read_text())
        assert audit["protocol_sha256"]==sha(structure/"protocol.json")
    checks.append(dict(check="frozen_structure_sources_and_complete_fits",passed=True,fits=6720,checkpoint_rows=len(trajectories)))

    dynamics=BASE/"dynamics"
    protocol=json.loads((dynamics/"protocol.json").read_text())
    for relative,digest in protocol["code_sha256"].items():assert sha(JOURNAL/relative)==digest
    assert sha(JOURNAL/"scripts/run_prospective_morphology_selection.py")==protocol["original_runner_sha256"]
    counts=[]
    for seed in range(9300,9320):
        stem=dynamics/"runs/fresh"/f"seed_{seed}"
        audit=json.loads(Path(str(stem)+"_audit.json").read_text())
        seal=json.loads(Path(str(stem)+"_selection_seal.json").read_text())
        assert seal["protocol_sha256"]==sha(dynamics/"protocol.json")
        assert seal["predictions_sha256"]==sha(Path(str(stem)+"_predictions.csv"))
        assert audit["n_outcomes"]==16*2*20*2*3
        counts.append(audit["n_outcomes"])
    checks.append(dict(check="fresh_dynamics_frozen_sources_and_prediction_seals",passed=True,seeds=20,
                       fits=sum(counts)//3,checkpoint_rows=sum(counts)))

    test_dp_against_all_six_input_trees()
    constructive=structure/"constructive_dp_v2"
    cfg=json.loads((constructive/"protocol.json").read_text())
    for name,digest in cfg["source_hashes"].items():assert sha(JOURNAL/"scripts/morphology_structure"/name)==digest
    records=json.loads((constructive/"constructed_trees.json").read_text())
    target_map={t["task_id"]:t for t in tasks()+prefix_tasks()}
    x=domain();design=fourier_design(x)
    errors=[];maximum_weight=0.
    for record in records:
        children={int(k):tuple(v) for k,v in record["children"].items()}
        descendants={i:(i,) for i in range(8)};parent={}
        for node,(left,right) in children.items():
            descendants[node]=descendants[left]+descendants[right]
            parent[left],parent[right]=(node,0),(node,1)
        tree=Tree(record["task_id"],"adaptive",descendants[14],children,descendants,parent,14)
        weights=np.asarray(record["weights"])
        maximum_weight=max(maximum_weight,float(abs(weights).max()))
        coeff=target_map[record["task_id"]]["coefficients"]
        err=float(np.mean((forward(x,tree,weights)[14]-design@coeff)**2)/(coeff@coeff))
        errors.append(err)
    assert len(errors)==164 and max(errors)<1e-20 and maximum_weight<1+1e-12
    checks.append(dict(check="independent_forward_evaluation_of_saved_constructions",passed=True,targets=164,
                       maximum_nmse=max(errors),maximum_absolute_coefficient=maximum_weight))
    checks.append(dict(check="two_pass_DP_matches_exhaustive_six_input_search",passed=True,trees_per_cost_table=945,cost_tables=4))

    report=BASE/"INVESTIGATION_REPORT.md"
    missing=[]
    for target in re.findall(r"\]\(([^)]+)\)",report.read_text()):
        if target.startswith(("http:","https:")):continue
        path=Path(target) if target.startswith("/") else report.parent/target
        if not path.exists():missing.append(str(path))
    assert not missing,missing
    checks.append(dict(check="local_report_links_exist",passed=True))
    result=dict(status="passed",checks=checks,original_experiment_modified=False,
                scope="Numerical/source/provenance audit; no new held-out tuning or biological validation")
    (BASE/"VALIDATION.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
