#!/usr/bin/env python3
"""One finite, disclosed posthoc expansion of the conductance learning-rate grid."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import pandas as pd
import run_opponent as frozen
from run import sha,write
OUT=frozen.OUT/'expanded_rates'

def freeze():
 OUT.mkdir(exist_ok=True);original=frozen.freeze();record=dict(status='Posthoc optimizer-grid boundary check; original and wider-bound outcomes already known. No original source or outcome is changed.',source_sha256=sha(Path(__file__)),original_development_freeze_sha256=sha(frozen.OUT/'development_freeze.json'),original_selection_sha256=sha(frozen.OUT/'selection_freeze.json'),development_seeds=original['development_seeds'],tasks=['aligned_strong','opposed_strong'],rules=original['rules'],rates={'adam':[.003,.01,.03,.1,.3,1.],'sgd':[.03,.1,.3,1.,3.,10.]},steps=16384,checkpoints=[0,64,256,1024,2048,4096,8192,12288,16384],log_conductance_bounds=[-7.,7.],endpoint='Minimum validation NMSE at the declared checkpoints; retain every fit.',comparison='All six original-plus-expanded rates receive the same16384-update budget for every rule, task and optimizer.',followup_trigger='Report every rate. An expanded-rate broadcast improvement is material if mean opposed-task best-validation NMSE improves at least0.01 and at least10% relative to the best original-grid rate at the same budget. Such an improvement triggers a rate selection using development validation alone, followed by a labeled reuse of the20 original fresh blocks; it is not independent confirmation.')
 path=OUT/'protocol_freeze.json'
 if path.exists():assert {k:v for k,v in json.loads(path.read_text()).items() if k!='created_utc'}==record
 else:write(path,dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),**record))
 return record,original

def run(seed):
 record,cfg=freeze();assert seed in record['development_seeds'];cfg=copy.deepcopy(cfg);cfg['rates']=record['rates'];cfg['checkpoints']=record['checkpoints'];folder=OUT/'runs';folder.mkdir(exist_ok=True);assert not (folder/f'seed_{seed}_audit.json').exists();curves=[];endpoints=[];diagnostics=[];files={}
 for task in cfg['tasks']:
  if task['name'] not in record['tasks']:continue
  c,d,e,meta,arrays=frozen.run_task(seed,task,'development',cfg,steps=record['steps']);curves+=c;endpoints+=e;diagnostics+=d
  p=folder/f'seed_{seed}_{task["name"]}_states.npz';np.savez_compressed(p,**arrays);files[p.name]=sha(p);p=folder/f'seed_{seed}_{task["name"]}_metadata.json';write(p,meta);files[p.name]=sha(p);print(seed,task['name'],'expanded-grid complete',meta['wall_seconds'],flush=True)
 for kind,rows in [('curves',curves),('endpoints',endpoints),('diagnostics',diagnostics)]:
  p=folder/f'seed_{seed}_{kind}.csv';pd.DataFrame(rows).to_csv(p,index=False);files[p.name]=sha(p)
 write(folder/f'seed_{seed}_audit.json',dict(seed=seed,files_sha256=files,freeze_sha256=sha(OUT/'protocol_freeze.json'),completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))

def report():
 record,cfg=freeze();parts=[]
 for seed in record['development_seeds']:
  folder=OUT/'runs';a=json.loads((folder/f'seed_{seed}_audit.json').read_text())
  for f,h in a['files_sha256'].items():assert sha(folder/f)==h
  parts.append(pd.read_csv(folder/f'seed_{seed}_endpoints.csv'))
 d=pd.concat(parts,ignore_index=True);d.to_csv(OUT/'all_endpoints.csv',index=False);table=d.groupby(['task','optimizer','rule','rate'],as_index=False).validation_nmse.mean();table['original_grid']=[rate in cfg['rates'][opt] for opt,rate in zip(table.optimizer,table.rate)];table.to_csv(OUT/'rate_comparison.csv',index=False);selected=table.loc[table.groupby(['task','optimizer','rule']).validation_nmse.idxmin()].sort_values(['task','optimizer','rule']);selected.to_csv(OUT/'selected_rates.csv',index=False);checks=[]
 for opt in cfg['rates']:
  for rule in ['unit_broadcast','calibrated_broadcast']:
   q=table[(table.task=='opposed_strong')&(table.optimizer==opt)&(table.rule==rule)];old=float(q[q.original_grid].validation_nmse.min());new=float(q[~q.original_grid].validation_nmse.min());absolute=old-new;relative=absolute/max(old,1e-30);checks.append(dict(optimizer=opt,rule=rule,original_min=old,expanded_min=new,absolute_improvement=absolute,relative_improvement=relative,material=absolute>=.01 and relative>=.1))
 write(OUT/'selection_review.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),n_fits=len(d),checks=checks,trigger_followup=any(c['material'] for c in checks),selection_metric='Development validation only; no test outcomes used.'))
 print(selected.to_string(index=False));print(json.dumps(checks,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','report']);p.add_argument('--seed',type=int);a=p.parse_args();freeze() if a.action=='freeze' else report() if a.action=='report' else run(a.seed)
