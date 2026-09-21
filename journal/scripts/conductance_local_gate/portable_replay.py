#!/usr/bin/env python3
"""Replay a declared trajectory without Git or writes into canonical Source Data.

Use an ordinary working copy or a clean standalone reviewer package. The output
folder must be outside the checksummed canonical experiment directory.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from run import OUT,run_task,digest,write,utc
from portable_contract import load_protocol,verify

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--task',choices=['aligned_strong','opposed_strong'],required=True);p.add_argument('--rule',required=True);p.add_argument('--rate',type=float,default=.03);p.add_argument('--steps',type=int,choices=[4,4096,16384],default=4096);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg,freeze,release_checks=load_protocol();assert a.rule in cfg['rules'];assert a.rate in cfg['rates'];assert a.seed in cfg['fresh_seeds']+cfg['canary_seeds']
    if a.seed in cfg['canary_seeds']:assert a.steps==4 and a.rate==cfg['primary_rate']
    assert a.output.resolve()!=OUT.resolve() and OUT.resolve() not in a.output.resolve().parents,'Write into a separate replay folder'
    assert not a.output.exists() or not any(a.output.iterdir()),'Use a new empty output directory'
    a.output.mkdir(parents=True,exist_ok=True);task=next(t for t in cfg['tasks'] if t['name']==a.task)
    cfg=dict(cfg,steps=a.steps,rates=[a.rate],rules=[a.rule],checkpoints=([0,1,4] if a.steps==4 else [s for s in cfg['checkpoints'] if s<=a.steps]),endpoint_budgets=([4] if a.steps==4 else [s for s in cfg['endpoint_budgets'] if s<=a.steps]))
    c,e,d,arrays,meta=run_task(a.seed,task,cfg)
    np.savez_compressed(a.output/'states.npz',**arrays)
    for name,rows in [('curves',c),('endpoints',e),('diagnostics',d)]:pd.DataFrame(rows).to_csv(a.output/(name+'.csv'),index=False)
    canonical=OUT/('implementation_canary' if a.seed in cfg['canary_seeds'] else 'runs');check=None
    record=canonical/f'seed_{a.seed}_{a.task}_config.json';oldpath=canonical/f'seed_{a.seed}_{a.task}_states.npz'
    if record.exists() and oldpath.exists():
        run_audit=json.loads((canonical/f'seed_{a.seed}_audit.json').read_text())
        for f in [record,oldpath,canonical/f'seed_{a.seed}_endpoints.csv']:
            release_checks.append(verify(f,run_audit['files_sha256'][f.name]))
        original=json.loads(record.read_text());records=original['task_metadata']['records'];idx=next(i for i,r in enumerate(records) if r['rate']==a.rate and r['rule']==a.rule);old=np.load(oldpath);oldindices=[list(old['steps']).index(s) for s in arrays['steps']]
        pdiff=float(np.max(np.abs(arrays['theta'][:,0]-old['theta'][oldindices,idx])))
        reference=pd.read_csv(canonical/f'seed_{a.seed}_endpoints.csv',float_precision='round_trip');reference=reference[(reference.task==a.task)&(reference.rule==a.rule)&(reference.rate==a.rate)&reference.budget.le(a.steps)].sort_values('budget');actual=pd.DataFrame(e).sort_values('budget');ediff=float(np.max(np.abs(actual.test_nmse.to_numpy()-reference.test_nmse.to_numpy())))
        check=dict(canonical_parameters_max_abs_difference=pdiff,canonical_endpoint_nmse_max_abs_difference=ediff,within_roundoff_tolerance=bool(pdiff<1e-7 and ediff<1e-9))
    hashes={f.name:digest(f) for f in a.output.iterdir() if f.is_file()}
    report=dict(completed_utc=utc(),seed=a.seed,task=a.task,rule=a.rule,rate=a.rate,steps=a.steps,protocol_sha256=freeze['protocol_sha256'],release_verifications=release_checks,scientific_source_sha256=freeze['scientific_source_sha256'],metadata=meta,files_sha256=hashes,canonical_comparison=check,scope='Numerical reproduction only; no additional independent scientific observation.')
    write(a.output/'replay_report.json',report);print(json.dumps(check,indent=2))
    if check:assert check['within_roundoff_tolerance']
if __name__=='__main__':main()
