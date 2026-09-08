#!/usr/bin/env python3
"""Excluded historical-seed replay; never contributes fresh scientific endpoints."""
import json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from run import OUT,J,committed_protocol,run_task,write,digest,utc

def main():
    cfg,freeze,commit=committed_protocol();plan=json.loads((OUT/'historical_replay_plan.json').read_text());seed=plan['seed'];assert seed not in cfg['fresh_seeds']
    cfg=dict(cfg,steps=4096,endpoint_budgets=[4096],checkpoints=[0,64,256,1024,2048,4096],rates=[.03],rules=['exact','unit_broadcast','calibrated_broadcast','ancestry_three_oracle'])
    source=J/'source_data/conductance_credit_demand/opponent/runs/fresh';folder=OUT/'historical_replay';folder.mkdir(exist_ok=True)
    assert not (folder/'report.json').exists()
    oldcurves=pd.read_csv(source/f'seed_{seed}_curves.csv',float_precision='round_trip');oldends=pd.read_csv(source/f'seed_{seed}_endpoints.csv',float_precision='round_trip');rows=[];files={}
    for task in cfg['tasks']:
        c,e,d,arrays,meta=run_task(seed,task,cfg);newc=pd.DataFrame(c);newe=pd.DataFrame(e);p=folder/f'{task["name"]}_states.npz';np.savez_compressed(p,**arrays);files[p.name]=digest(p)
        legacy=np.load(source/f'seed_{seed}_{task["name"]}_states.npz')
        for i,rule in enumerate(cfg['rules']):
            nc=newc[newc.rule==rule].sort_values('step');oc=oldcurves[(oldcurves.task==task['name'])&(oldcurves.rule==rule)&(oldcurves.optimizer=='adam')].sort_values('step');oe=oldends[(oldends.task==task['name'])&(oldends.rule==rule)&(oldends.optimizer=='adam')].iloc[0];ne=newe[newe.rule==rule].iloc[0]
            state_diff=float(np.max(np.abs(arrays['theta'][:,i]-legacy['theta'][:,i])))
            m_diff=float(np.max(np.abs(arrays['final_first_moment'][i]-legacy['final_first_moment'][i])))
            v_diff=float(np.max(np.abs(arrays['final_second_moment'][i]-legacy['final_second_moment'][i])))
            output_diff=float(np.max(np.abs(nc[['validation_nmse','test_nmse']].to_numpy()-oc[['validation_nmse','test_nmse']].to_numpy())))
            endpoint_diff=float(abs(ne.test_nmse-oe.test_nmse));threshold=plan['oracle_projection_parameter_tolerance'] if rule=='ancestry_three_oracle' else plan['other_parameter_tolerance']
            passed=state_diff<=threshold and output_diff<=plan['nmse_tolerance'] and endpoint_diff<=plan['nmse_tolerance'] and int(ne.best_step)==int(oe.best_step)
            rows.append(dict(task=task['name'],rule=rule,parameter_max_abs_difference=state_diff,first_moment_max_abs_difference=m_diff,second_moment_max_abs_difference=v_diff,all_checkpoint_nmse_max_abs_difference=output_diff,selected_test_nmse_abs_difference=endpoint_diff,best_step_match=int(ne.best_step)==int(oe.best_step),passed=bool(passed)))
    pd.DataFrame(rows).to_csv(folder/'comparison.csv',index=False)
    report=dict(completed_utc=utc(),seed=seed,excluded_from_scientific_outcomes=True,protocol_sha256=freeze['protocol_sha256'],protocol_commit=commit,replay_plan_sha256=digest(OUT/'historical_replay_plan.json'),script_sha256=digest(Path(__file__)),checks=rows,status='PASS' if all(r['passed'] for r in rows) else 'FAIL',files_sha256=files,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    write(folder/'report.json',report);print(json.dumps(report,indent=2));assert report['status']=='PASS'
if __name__=='__main__':main()
