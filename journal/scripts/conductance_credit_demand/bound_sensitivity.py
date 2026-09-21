#!/usr/bin/env python3
"""Disclosed posthoc bound check plus exact bound-contact replay, every condition."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import pandas as pd
import run_opponent as run

OUT=run.OUT/'bound_sensitivity'

def freeze():
 OUT.mkdir(exist_ok=True);p=OUT/'protocol_freeze.json';record=dict(created_utc='2026-09-07T02:20:00Z',status='Posthoc sensitivity specified after the frozen fresh and extended results; before any wider-bound outcomes.',source_sha256=run.sha(Path(__file__)),original_selection_sha256=run.sha(run.OUT/'selection_freeze.json'),bounds=[7.,20.],steps=16384,all_original_fresh_conditions=True,all_20_original_seeds=True,rate_retuning=False,initialization='Restart from original identical initialization and minibatch stream; retain original bounds7 replay to verify identity and record first contact.',bound_contact='First update before projection for each parameter; broad log bounds20 span conductances exp(-20) to exp(20).')
 if p.exists():
  prev=json.loads(p.read_text());assert {k:v for k,v in prev.items() if k!='created_utc'}=={k:v for k,v in record.items() if k!='created_utc'}
 else:
  record['created_utc']=pd.Timestamp.now(tz='UTC').isoformat();run.write(p,record)
 return record

def execute(seed):
 freeze();cfg=run.freeze();assert seed in cfg['fresh_seeds'];cfg=copy.deepcopy(cfg);cfg['checkpoints']=[0,64,256,1024,2048,4096,8192,12288,16384];sel=json.loads((run.OUT/'selection_freeze.json').read_text());tasks=[t for t in cfg['tasks'] if t['name'] in sel['confirmatory_tasks']];folder=OUT/'runs';folder.mkdir(exist_ok=True);assert not (folder/f'seed_{seed}_audit.json').exists();allends=[];allcurves=[];contacts=[];files={};max_replay=0.
 for bound in [7.,20.]:
  cfg['log_conductance_bounds']=[-bound,bound]
  for task in tasks:
   count=0;first=None;counts=None;original_clip=np.clip
   def record_clip(a,*args,**kwargs):
    nonlocal count,first,counts
    count+=1
    if first is None:first=np.full(a.shape,-1,int);counts=np.zeros(a.shape,int)
    hits=(a < -bound)|(a > bound);counts+=hits;first[(first<0)&hits]=count;return original_clip(a,*args,**kwargs)
   try:
    np.clip=record_clip;c,d,e,meta,arrays=run.run_task(seed,task,'fresh',cfg,steps=16384)
   finally:np.clip=original_clip
   assert count==16384
   if bound==7.:
    prior=np.load(run.OUT/'extension'/f'seed_{seed}_{task["name"]}_states.npz');error=float(np.max(abs(prior['final_theta']-arrays['final_theta'])));max_replay=max(max_replay,error);assert error<3e-12
    previous=pd.read_csv(run.OUT/'extension'/f'seed_{seed}_endpoints.csv');previous=previous[previous.task==task['name']];np.testing.assert_allclose(previous.test_nmse,[r['test_nmse'] for r in e],atol=3e-12,rtol=3e-12)
   for row in c:row.update(bound=bound,phase='bound_sensitivity')
   for row in e:row.update(bound=bound,phase='bound_sensitivity')
   allcurves+=c;allends+=e
   for i,r in enumerate(meta['records']):
    for j in range(first.shape[1]):contacts.append(dict(seed=seed,task=task['name'],bound=bound,**r,parameter=j,compartment=int(run.model.PARAM_UNIT[j]),first_contact_step=int(first[i,j]),n_projected_updates=int(counts[i,j])))
   p=folder/f'seed_{seed}_{task["name"]}_bound_{int(bound)}_states.npz';np.savez_compressed(p,**arrays);files[p.name]=run.sha(p);print(seed,task['name'],bound,'done',flush=True)
 for kind,rows in [('endpoints',allends),('curves',allcurves),('contacts',contacts)]:
  p=folder/f'seed_{seed}_{kind}.csv';pd.DataFrame(rows).to_csv(p,index=False);files[p.name]=run.sha(p)
 run.write(folder/f'seed_{seed}_audit.json',dict(seed=seed,files_sha256=files,maximum_original_replay_parameter_error=max_replay,freeze_sha256=run.sha(OUT/'protocol_freeze.json'),completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--freeze',action='store_true');a=p.parse_args();freeze() if a.freeze else execute(a.seed)
