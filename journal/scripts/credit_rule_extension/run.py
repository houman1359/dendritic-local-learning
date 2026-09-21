#!/usr/bin/env python3
"""Balanced, explicitly post-review extension of every algebraic bridge arm."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
J=HERE.parents[1]
OUT=J/'source_data/credit_rule_extension'
ORIGINAL=J/'source_data/credit_rule_bridge'
sys.path.insert(0,str(HERE.parent/'credit_rule_bridge'))
import run as bridge

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def now(): return datetime.now(timezone.utc).isoformat()
def original_config():
 frozen=json.loads((ORIGINAL/'protocol_freeze.json').read_text())
 for rel,h in frozen['source_sha256'].items():
  assert sha(J/rel)==h, f'Original scientific source changed: {rel}'
 selection=json.loads((ORIGINAL/'selection_freeze.json').read_text())
 assert selection['protocol_freeze_sha256']==sha(ORIGINAL/'protocol_freeze.json')
 return frozen['protocol']

def freeze():
 assert not (OUT/'protocol_freeze.json').exists(), 'Freeze is immutable'
 cfg=original_config();records=bridge.conditions('algebraic','fresh',cfg)
 assert len(records)==12
 files=[HERE/'run.py',HERE/'worker.sh',ORIGINAL/'protocol_freeze.json',ORIGINAL/'selection_freeze.json']
 files += [J/rel for rel in json.loads((ORIGINAL/'protocol_freeze.json').read_text())['source_sha256']]
 for seed in cfg['fresh_seeds']:
  folder=ORIGINAL/'runs/fresh/algebraic';auditpath=folder/f'seed_{seed}_audit.json'
  audit=json.loads(auditpath.read_text());files.append(auditpath)
  for name,h in audit['source_files_sha256'].items():
   p=folder/name;assert sha(p)==h;files.append(p)
 inv={str(p.relative_to(J)):sha(p) for p in sorted(set(files))}
 write(OUT/'input_inventory.json',inv)
 protocol={
  'version':1,'created_utc':now(),
  'scope':'Balanced longer-budget follow-up after inspecting all released 1024-step outcomes and the September7 referee review. Same seeds are reused: not a fresh confirmatory sample.',
  'known_results':'Exact quartet outcomes include17 near-floor fits and3 stalled fits at1024; calibrated profiles fail on quartet/nested. Pairwise selected rates differ(.01 calibrated versus.003 exact); common-rate curves reduce the early speed gap. Exact pairwise mean is affected by slow seeds. No16384-step outcomes have been inspected.',
  'seeds':cfg['fresh_seeds'],'tasks':['matching','quartet','nested'],
  'records':records,'unique_trajectories':720,'seed_blocks':20,
  'max_updates':16384,
  'checkpoints':sorted(set(cfg['checkpoints']+list(range(1280,16385,256)))),
  'training':'Run original run_task with unchanged data, initialization, coefficients, frozen calibration, four rules, SGD/Adam, selected/common rate union, batch64, norm clip10 and coefficient bounds±2. Only total steps and observation checkpoints change. Replay from initialization because historical states omit optimizer/RNG state.',
  'pairing':'Matching/quartet preserve identical compatible balanced tree, initial coefficients, input/noise/minibatch streams. Nested retains its own compatible tree and is not input-isospectral. Same twenty seed blocks, both optimizers, every selected/common-rate union member and all four rules are retained.',
  'replay_gate':'Check every original state array and numerical curve/diagnostic through1024 against original per-seed artifacts. Tolerance1e-10 absolute numerical difference; metadata and coordinate arrays must match exactly. Failed replay blocks acceptance and is reported, never silently replaced.',
  'stopping':'Every declared trajectory runs exactly16384 updates. No early stopping or outcome-dependent extension. Longer training is still a finite-budget comparison and no asymptotic winner is claimed.',
  'endpoint_selection':'Primary terminal held-out test NMSE at budgets1024,4096,8192,16384 preserves original final-step estimand. Secondary states minimize validation NMSE among declared checkpoints within each budget, ties to earliest step. Test/population results never select checkpoint, rate, scope or stopping.',
  'rate_selection':'No new tuning. Retain original development-selected rates and original optimizer-specific common rates; summarize both. They are historical1024-step selections, not asserted optimal at the longer budget.',
  'primary_analysis':'Budget-indexed quartet-minus-matching interaction in calibrated-minus-exact testNMSE for Adam, separately selected/common rates and terminal/validation-selected endpoints. Retain all task/rule/optimizer contrasts and clean full-domain populationNMSE.',
  'diagnostics':'Retain all checkpoint states, original field/gradient diagnostics, every seedwise endpoint, gradient clipping and parameter-bound events. Report quartic success/stall distributions and pairwise common-rate trajectories. Plot terminal and validation-selected differences without declaring a convergence threshold.',
  'uncertainty':'Descriptive paired whole-seed bootstrap,10000 draws,seed210999,95percent intervals. Twenty previously observed independent seed blocks; no new confirmatory P values, equivalence or multiplicity claim.',
  'failure_policy':'Retain every declared arm and numerical failure. A failed job may be retried only with the same frozen inputs and an explicit attempt record. No outcome-based omission.',
  'execution':'Slurm serial_requeue,one CPU per seed,OPENBLAS/OMP/MKL threads1. Root must commit this protocol before scientific launch.',
  'input_inventory_sha256':sha(OUT/'input_inventory.json'),
  'runtime_at_freeze':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__}
 }
 write(OUT/'protocol_freeze.json',protocol)
 print(json.dumps({'protocol':str(OUT/'protocol_freeze.json'),'sha256':sha(OUT/'protocol_freeze.json'),'input_inventory_sha256':sha(OUT/'input_inventory.json'),'unique_trajectories':720,'conditions_per_seed':36},indent=2))

def check_freeze():
 p=json.loads((OUT/'protocol_freeze.json').read_text());assert sha(OUT/'input_inventory.json')==p['input_inventory_sha256']
 for rel,h in json.loads((OUT/'input_inventory.json').read_text()).items():
  assert sha(J/rel)==h,f'Frozen input changed: {rel}'
 return p

def compare_numeric(actual,expected,keys):
 a=actual.sort_values(keys).reset_index(drop=True);b=expected.sort_values(keys).reset_index(drop=True)
 assert len(a)==len(b)
 maxerr=0.;columns=[]
 for col in b:
  if col=='elapsed_seconds':continue
  assert col in a
  if pd.api.types.is_numeric_dtype(b[col]) and not pd.api.types.is_bool_dtype(b[col]):
   err=float(np.max(abs(a[col].to_numpy(float)-b[col].to_numpy(float)))) if len(a) else 0.
   assert err<=1e-10,(col,err);maxerr=max(maxerr,err)
  else: assert a[col].astype(str).tolist()==b[col].astype(str).tolist(),col
  columns.append(col)
 return {'rows':len(a),'columns_checked':columns,'maximum_absolute_difference':maxerr}

def run_seed(seed,expected_hash):
 assert sha(OUT/'protocol_freeze.json')==expected_hash
 protocol=check_freeze();assert seed in protocol['seeds']
 cfg=copy.deepcopy(original_config());cfg['checkpoints']=protocol['checkpoints']
 folder=OUT/'runs'/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
 assert not (folder/'audit.json').exists(),'Completed outcomes are immutable'
 assert not list(folder.glob('*.csv')),'Partial attempt requires explicit investigation'
 git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=J,text=True).strip()
 # Prove the launched protocol is already in the reachable paper commit.
 committed=subprocess.check_output(['git','show',f'{git_commit}:journal/source_data/credit_rule_extension/protocol_freeze.json'],cwd=J)
 assert hashlib.sha256(committed).hexdigest()==expected_hash,'Protocol is not committed at launch HEAD'
 write(folder/'attempt.json',{'seed':seed,'started_utc':now(),'protocol_sha256':expected_hash,'git_commit_at_launch':git_commit,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__})
 rows=[];diagnostics=[];checks=[];files={};start=time.perf_counter()
 original_folder=ORIGINAL/'runs/fresh/algebraic'
 for task in protocol['tasks']:
  vals,diag,arrays,metadata=bridge.run_task(seed,'algebraic',task,'fresh',cfg,steps=protocol['max_updates'])
  oldpath=original_folder/f'seed_{seed}_task_{task}_states.npz'
  with np.load(oldpath) as old:
   pos=[int(np.flatnonzero(arrays['steps']==step)[0]) for step in old['steps']]
   maxerr=0.
   for name in old.files:
    actual=arrays[name][pos] if name=='theta' else (arrays['steps'][pos] if name=='steps' else arrays[name])
    assert actual.shape==old[name].shape,(task,name)
    err=float(np.max(np.abs(actual-old[name])))
    assert err<=1e-10,(task,name,err)
    if name!='theta':assert np.array_equal(actual,old[name]),(task,name)
    maxerr=max(maxerr,err)
  oldmeta=json.loads((original_folder/f'seed_{seed}_task_{task}_metadata.json').read_text())
  assert metadata==oldmeta,'Original task metadata changed'
  checks.append({'task':task,'state_arrays_max_abs_difference':maxerr,'metadata_identical':True,'all_original_checkpoints_checked':True})
  path=folder/f'{task}_states.npz';np.savez_compressed(path,**arrays);files[path.name]=sha(path)
  path=folder/f'{task}_metadata.json';write(path,metadata);files[path.name]=sha(path)
  rows.extend(vals);diagnostics.extend(diag)
  print(seed,task,'completed through',protocol['max_updates'],'seconds',round(time.perf_counter()-start,2),flush=True)
 keys=['task','step','optimizer','rule','rate']
 for kind,values in [('curves',rows),('diagnostics',diagnostics)]:
  data=pd.DataFrame(values)
  old=pd.read_csv(original_folder/f'seed_{seed}_{kind}.csv',float_precision='round_trip')
  replay=compare_numeric(data[data.step<=1024],old,keys)
  checks.append({'kind':kind,**replay})
  path=folder/f'{kind}.csv';data.to_csv(path,index=False);files[path.name]=sha(path)
 write(folder/'replay_validation.json',{'status':'passed','checks':checks});files['replay_validation.json']=sha(folder/'replay_validation.json')
 write(folder/'audit.json',{'status':'complete','seed':seed,'unique_trajectories':36,'maximum_steps':protocol['max_updates'],'checkpoint_count':len(protocol['checkpoints']),'elapsed_seconds':time.perf_counter()-start,'completed_utc':now(),'protocol_sha256':expected_hash,'input_inventory_sha256':protocol['input_inventory_sha256'],'source_files_sha256':files,'all_conditions_retained':True,'original_states_and_metrics_reproduced':True,'slurm_job_id':os.environ.get('SLURM_JOB_ID')})
 print(json.dumps(json.loads((folder/'audit.json').read_text())),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','check','run']);p.add_argument('--seed',type=int);p.add_argument('--protocol-sha256');a=p.parse_args()
 if a.action=='freeze':freeze()
 elif a.action=='check':print(json.dumps(check_freeze(),indent=2))
 else:run_seed(a.seed,a.protocol_sha256)
