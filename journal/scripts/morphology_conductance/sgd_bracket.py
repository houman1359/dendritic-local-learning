#!/usr/bin/env python3
"""Additional development-only SGD bracketing; original outcomes stay unchanged."""
import argparse
import json
import time

import numpy as np
import pandas as pd

import run as study


def freeze():
    assert not list((study.OUT/'runs/fresh').glob('*'))
    study.write(study.OUT/'development_sgd_bracket_protocol.json',dict(
        created_utc=pd.Timestamp.now(tz='UTC').isoformat(),reason='Original SGD optimum was upper grid boundary .01; bracket before fresh outcome evaluation',
        phase='Additional development only',rates=[.03,.1,.3,1.],seeds=list(range(15300,15305)),
        original_development_protocol_sha256=study.sha(study.OUT/'development_protocol.json'),
        script_sha256=study.sha(__file__),all_other_training_parameters_unchanged=True,
        fresh_rate_selection='Average final validation NMSE across all five development seeds, all task/student groupings and four credit rules, original and added rates'))


def run_seed(seed):
    cfg=study.check_development();protocol=json.loads((study.OUT/'development_sgd_bracket_protocol.json').read_text())
    assert study.sha(__file__)==protocol['script_sha256']
    assert seed in protocol['seeds']
    cfg['optimizers']=['sgd'];cfg['learning_rate_grid']['sgd']=protocol['rates']
    dest=study.OUT/'runs/development_sgd_bracket';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/f'seed_{seed}_audit.json').exists()
    rows=[];diagnostics=[];snapshots={};start=time.perf_counter()
    for task_group in range(3):
        values,diag,theta,profiles=study.run_task(seed,task_group,cfg,'development')
        rows.extend(values);diagnostics.extend(diag)
        snapshots[f'task_{task_group}_final_log_conductances']=theta
        snapshots[f'task_{task_group}_initial_profiles']=profiles
    pd.DataFrame(rows).to_csv(dest/f'seed_{seed}_curves.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(dest/f'seed_{seed}_credit_diagnostics.csv',index=False)
    np.savez_compressed(dest/f'seed_{seed}_final_states.npz',**snapshots)
    study.write(dest/f'seed_{seed}_audit.json',dict(seed=seed,phase='development_sgd_bracket',
        n_fits=len(study.conditions(cfg,'development'))*3,n_curve_rows=len(rows),elapsed_seconds=time.perf_counter()-start,
        protocol_sha256=study.sha(study.OUT/'development_sgd_bracket_protocol.json'),all_outcomes_retained=True))
    print(json.dumps(dict(seed=seed,seconds=time.perf_counter()-start)),flush=True)


def select_fresh():
    cfg=study.check_development();assert not list((study.OUT/'runs/fresh').glob('*'))
    frames=[];sources=[]
    for folder in ['development','development_sgd_bracket']:
        for seed in cfg['development_seeds']:
            path=study.OUT/'runs'/folder/f'seed_{seed}_curves.csv'
            assert path.with_name(f'seed_{seed}_audit.json').exists()
            frames.append(pd.read_csv(path));sources.append(dict(path=str(path.relative_to(study.OUT)),sha256=study.sha(path)))
    data=pd.concat(frames,ignore_index=True);last=data[data.step.eq(cfg['steps'])]
    summary=last.groupby(['optimizer','learning_rate']).validation_nmse.mean().reset_index()
    summary.to_csv(study.OUT/'development_learning_rate_selection.csv',index=False)
    rates={opt:float(g.sort_values(['validation_nmse','learning_rate'],kind='stable').iloc[0].learning_rate) for opt,g in summary.groupby('optimizer')}
    study.write(study.OUT/'fresh_protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        status='Frozen before any fresh-seed calibration or training',development_protocol_sha256=study.sha(study.OUT/'development_protocol.json'),
        sgd_bracket_protocol_sha256=study.sha(study.OUT/'development_sgd_bracket_protocol.json'),
        selected_learning_rates=rates,selection_table_sha256=study.sha(study.OUT/'development_learning_rate_selection.csv'),
        development_source_hashes=sources,fresh_seeds=cfg['fresh_seeds'],n_fits=20*3*3*4*2,
        primary_contrasts=cfg['primary_contrasts'],all_other_parameters_unchanged=True,
        comparison_unit='Seed; task/input permutations are paired conditions, not independent replicates'))
    print(summary.to_string(index=False));print(json.dumps(rates,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['freeze','run','select_fresh']);p.add_argument('--seed',type=int);args=p.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='run':run_seed(args.seed)
    else:select_fresh()
