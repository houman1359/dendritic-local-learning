#!/usr/bin/env python3
"""Independent endpoint reconstruction from saved states, not summary means."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import run


def main():
 p=run.check_freeze();spec=importlib.util.spec_from_file_location('credit_extension_audit_reference',run.HERE.parent/'credit_rule_bridge/run.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
 outcomes=pd.read_csv(run.OUT/'summaries/all_budget_outcomes.csv',float_precision='round_trip')
 rows=[];maxerr=0.;selectionchecks=0
 for seed in p['seeds']:
  folder=run.OUT/'runs'/f'seed_{seed}';curve=pd.read_csv(folder/'curves.csv',float_precision='round_trip')
  for task in p['tasks']:
   metadata=json.loads((folder/f'{task}_metadata.json').read_text())
   with np.load(folder/f'{task}_states.npz') as z:
    coeff=z['coefficients'];left=z['left'];right=z['right'];variance=float(coeff@coeff)
    tx,ty=ref.algebra_data(seed,coeff,'test',4096);vx,vy=ref.algebra_data(seed,coeff,'validation',1024)
    dx=ref.models.structure.domain();dy=ref.models.structure.fourier_design(dx)@coeff
    chosen=outcomes[(outcomes.seed==seed)&(outcomes.task==task)]
    for step in sorted(chosen.chosen_step.unique()):
     ix=int(np.flatnonzero(z['steps']==step)[0]);theta=z['theta'][ix]
     values={}
     for name,x,y in [('test_nmse',tx,ty),('validation_nmse',vx,vy),('population_nmse',dx,dy)]:
      predicted=ref.models.algebra_state(theta,x,left,right)['output'];values[name]=np.mean((predicted-y[None])**2,axis=1)/variance
     for k,r in enumerate(metadata['records']):
      retained=chosen[(chosen.chosen_step==step)&(chosen.optimizer==r['optimizer'])&(chosen.rule==r['rule'])&(chosen.rate==r['rate'])]
      if not len(retained):continue
      for name,v in values.items():
       error=float(np.max(abs(retained[name].to_numpy(float)-v[k])));assert error<1e-12;maxerr=max(maxerr,error)
      rows.append({'seed':seed,'task':task,'step':int(step),'optimizer':r['optimizer'],'rule':r['rule'],'rate':r['rate'],'outcome_rows_checked':len(retained),'test_nmse':float(values['test_nmse'][k]),'validation_nmse':float(values['validation_nmse'][k]),'population_nmse':float(values['population_nmse'][k])})
    for r in chosen[chosen.endpoint=='validation_selected'].itertuples():
     c=curve[(curve.task==task)&(curve.optimizer==r.optimizer)&(curve.rule==r.rule)&(curve.rate==r.rate)&(curve.step<=r.budget)]
     expected=c.sort_values(['validation_nmse','step']).iloc[0];assert r.chosen_step==expected.step;selectionchecks+=1
 out=run.OUT/'summaries';pd.DataFrame(rows).to_csv(out/'independent_state_reconstruction.csv',index=False)
 report={'status':'passed','outcome_rows':len(outcomes),'outcome_rows_reconstructed':sum(r['outcome_rows_checked'] for r in rows),'validation_only_checkpoint_selections_checked':selectionchecks,'maximum_nmse_absolute_difference':maxerr,'test_validation_full_domain_population_predictions_recomputed':True,'script_sha256':run.sha(__file__),'protocol_sha256':run.sha(run.OUT/'protocol_freeze.json'),'created_utc':run.now()}
 assert report['outcome_rows_reconstructed']==len(outcomes)
 run.write(out/'independent_validation.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
