#!/usr/bin/env python3
"""Development and frozen confirmatory E/I conductance credit comparison."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
import opponent_model as model
HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/'source_data/conductance_credit_demand/opponent'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n')

def default_config():
 return dict(development_seeds=[201,202,203],fresh_seeds=list(range(2101,2121)),
  tasks=[dict(name='aligned_strong',conflict=0.,gate=10.),dict(name='opposed_ungated',conflict=1.,gate=0.),dict(name='opposed_strong',conflict=1.,gate=10.),dict(name='opposed_moderate',conflict=1.,gate=1.)],
  rules=list(model.RULES),rates={'adam':[.003,.01,.03],'sgd':[.03,.1,.3]},steps=4096,checkpoints=[0,64,256,1024,2048,4096],batch_size=128,n_train=2048,n_validation=1024,n_test=4096,n_calibration=512,n_diagnostic=1024,
  gradient_clip=10.,log_conductance_bounds=[-7.,7.],noise_sd=0.,
  task_selection=dict(primary='opposed_strong',fallback_order=['opposed_moderate'],control='aligned_strong',teacher_rank_one_capture_max=.9,exact_adam_validation_nmse_max=.02,rule='First ordered task meeting teacher geometry and exact Adam validation criteria, never based on between-rule gap. All candidates retained; no passing candidate means no confirmatory spatial-credit claim.'),
  endpoint='Checkpoint with minimum validation NMSE within 4096 updates, alongside full endpoint and curves.',learning_rate_selection='Each rule and optimizer: minimum mean development best-validation NMSE at equal three-rate budgets; common rate minimizes mean validation NMSE over all rules. Fresh conditions union of selected and common rate.',
  main_contrast='Opposed-tuning minus aligned-tuning difference in calibrated-broadcast minus exact clean test NMSE, paired by fresh teacher/initialization/data/minibatches. Report unit control, SGD, common-rate, endpoints and all seeds.',
  forward_scope='Same directed seven-compartment E/I equation and all 24 trainable log conductances (two E and two I channels per leaf) as existing conductance reference; identity reactivation, no arbitrary multiplication or quartic representability assumption.',
  ancestry_control='Three fixed initial-profile patterns: distal leaves0–1, distal leaves2–3, proximal units4–5. Per-example coefficients are least-squares projections of the current exact six-unit path field, an explicit oracle routing upper bound; no independently learned encoder claim.',
  budget_extension='After primary endpoint, extend every selected-rate fresh fit to 16384 updates with unchanged optimizer/minibatch state, irrespective of gap; report fixed endpoint and best validation. No condition-specific extension or early stopping.')

def freeze():
 p=OUT/'protocol.json'
 if not p.exists(): write(p,default_config())
 c=json.loads(p.read_text())
 record=dict(protocol_sha256=sha(p),source_sha256={str(s.relative_to(HERE.parents[1])):sha(s) for s in [HERE/'run_opponent.py',HERE/'opponent_model.py',HERE/'test_opponent.py',HERE/'model.py',model.REFERENCE]},python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
 f=OUT/'development_freeze.json'
 if f.exists(): assert json.loads(f.read_text())==record,'Frozen development code changed'
 else:
  write(f,record);write(OUT/'development_timestamp.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),status='Before development outcomes'))
 return c

def conditions(cfg,phase,task):
 rows=[]
 selected=json.loads((OUT/'selection_freeze.json').read_text()) if phase=='fresh' else None
 for optimizer,rates in cfg['rates'].items():
  for rule in cfg['rules']:
   if selected:
    selected_rate=selected['rates'][task][optimizer][rule];common_rate=selected['common_rates'][task][optimizer]
    rates=sorted({selected_rate,common_rate})
   for rate in rates: rows.append(dict(optimizer=optimizer,rule=rule,rate=rate,selected_rate=bool(selected and rate==selected_rate),common_rate=bool(selected and rate==common_rate)))
 return rows

def run_task(seed,task,phase,cfg,steps=None):
 records=conditions(cfg,phase,task['name']);n=len(records);rules=[r['rule'] for r in records];rates=np.array([r['rate'] for r in records]);adam=np.array([r['optimizer']=='adam' for r in records])
 data={kind:model.data(seed,kind,cfg['n_'+kind],task['conflict'],task['gate']) for kind in ['train','validation','test','calibration','diagnostic']}
 x,y,_=data['train'];vx,vy,_=data['validation'];tx,ty,_=data['test'];dx,dy,_=data['diagnostic'];cx=data['calibration'][0]
 variance=float(np.var(y));initrng=np.random.default_rng(np.random.SeedSequence([seed,45678]));initial=np.log(model.NOMINAL)+initrng.normal(0,.4,24);theta=np.tile(initial,(n,1))
 profiles=model.forward(theta,cx)['path'][:,:,:6].mean(axis=1)
 m=np.zeros_like(theta);v=np.zeros_like(theta);clipped=np.zeros(n,int);projected=np.zeros(n,int)
 stream=np.random.default_rng(np.random.SeedSequence([seed,56789]));steps=cfg['steps'] if steps is None else steps
 checkpoints=sorted(set([s for s in cfg['checkpoints'] if s<=steps]+[steps]));snapshots=[];curves=[];diagnostics=[];best=np.full(n,np.inf);besttheta=theta.copy();beststep=np.zeros(n,int);start=time.perf_counter()
 for step in range(steps+1):
  if step in checkpoints:
   snapshots.append(theta.copy());vp=model.forward(theta,vx)['output'];tp=model.forward(theta,tx)['output'];validation=np.mean((vp-vy[None])**2,axis=1)/np.var(vy);test=np.mean((tp-ty[None])**2,axis=1)/np.var(ty)
   improved=validation<best;best[improved]=validation[improved];besttheta[improved]=theta[improved];beststep[improved]=step
   metrics=model.metrics(theta,dx,dy,variance,profiles)
   for i,r in enumerate(records):
    common=dict(seed=seed,task=task['name'],phase=phase,step=step,**r)
    curves.append(dict(**common,validation_nmse=validation[i],test_nmse=test[i],gradient_clipped_steps=int(clipped[i]),projected_steps=int(projected[i])))
    diagnostics.append(dict(**common,**metrics[i]))
  if step==steps: break
  ix=stream.integers(len(x),size=cfg['batch_size']);gradient,_=model.gradients(theta,x[ix],y[ix],variance,profiles,rules)
  norms=np.linalg.norm(gradient,axis=1);clipped+=norms>cfg['gradient_clip'];gradient*=np.minimum(1,cfg['gradient_clip']/np.maximum(norms,1e-30))[:,None]
  m=.9*m+.1*gradient;v=.999*v+.001*gradient**2;update=gradient.copy();update[adam]=(m[adam]/(1-.9**(step+1)))/(np.sqrt(v[adam]/(1-.999**(step+1)))+1e-8)
  theta-=rates[:,None]*update;projected+=np.any((theta<cfg['log_conductance_bounds'][0])|(theta>cfg['log_conductance_bounds'][1]),axis=1);theta=np.clip(theta,*cfg['log_conductance_bounds']);assert np.isfinite(theta).all()
 bp=model.forward(besttheta,tx)['output'];besttest=np.mean((bp-ty[None])**2,axis=1)/np.var(ty);bestmetrics=model.metrics(besttheta,dx,dy,variance,profiles)
 endpoints=[dict(seed=seed,task=task['name'],phase=phase,**r,best_step=int(beststep[i]),validation_nmse=best[i],test_nmse=besttest[i],**bestmetrics[i]) for i,r in enumerate(records)]
 teacher=model.teacher(seed,bool(task['conflict']))[None];teacherprofile=model.forward(teacher,cx)['path'][:,:,:6].mean(axis=1);teachergeom=model.metrics(teacher,dx,dy,variance,teacherprofile)[0]
 metadata=dict(seed=seed,task=task,phase=phase,records=records,training_target_variance=variance,test_target_variance=float(np.var(ty)),clean_noise_floor=0.,teacher_log_conductances=teacher[0].tolist(),teacher_geometry=teachergeom,constructive_forward_nmse=float(np.mean((model.forward(teacher,tx)['output'][0]-ty)**2)/np.var(ty)),all_24_conductances_trainable=True,wall_seconds=time.perf_counter()-start,source='scripts/conductance_credit_demand/opponent_model.py',slurm_job_id=os.environ.get('SLURM_JOB_ID'))
 arrays=dict(theta=np.stack(snapshots),steps=checkpoints,best_theta=besttheta,best_step=beststep,initial_profiles=profiles,diagnostic_inputs=dx,diagnostic_targets=dy,final_theta=theta,final_first_moment=m,final_second_moment=v)
 return curves,diagnostics,endpoints,metadata,arrays

def run(seed,phase):
 cfg=freeze();assert seed in cfg[phase+'_seeds'];tasks=cfg['tasks']
 if phase=='fresh':
  sel=json.loads((OUT/'selection_freeze.json').read_text());tasks=[t for t in tasks if t['name'] in sel['confirmatory_tasks']]
 folder=OUT/'runs'/phase;folder.mkdir(parents=True,exist_ok=True);assert not (folder/f'seed_{seed}_audit.json').exists(),'Completed outcomes immutable'
 files={};curves=[];diags=[];ends=[]
 for task in tasks:
  c,d,e,meta,arrays=run_task(seed,task,phase,cfg);curves+=c;diags+=d;ends+=e
  path=folder/f'seed_{seed}_{task["name"]}_states.npz';np.savez_compressed(path,**arrays);files[path.name]=sha(path)
  path=folder/f'seed_{seed}_{task["name"]}_metadata.json';write(path,meta);files[path.name]=sha(path)
  print(seed,task['name'],'done',meta['wall_seconds'],flush=True)
 for kind,rows in [('curves',curves),('diagnostics',diags),('endpoints',ends)]:
  path=folder/f'seed_{seed}_{kind}.csv';pd.DataFrame(rows).to_csv(path,index=False);files[path.name]=sha(path)
 write(folder/f'seed_{seed}_audit.json',dict(seed=seed,phase=phase,files_sha256=files,development_freeze_sha256=sha(OUT/'development_freeze.json'),selection_freeze_sha256=sha(OUT/'selection_freeze.json') if phase=='fresh' else None,completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))

def select():
 cfg=freeze();assert not (OUT/'runs/fresh').exists();frames=[];geometry=[]
 for seed in cfg['development_seeds']:
  folder=OUT/'runs/development';audit=json.loads((folder/f'seed_{seed}_audit.json').read_text())
  for f,digest in audit['files_sha256'].items(): assert sha(folder/f)==digest
  frames.append(pd.read_csv(folder/f'seed_{seed}_endpoints.csv'))
  for task in cfg['tasks']:
   meta=json.loads((folder/f'seed_{seed}_{task["name"]}_metadata.json').read_text());geometry.append(dict(seed=seed,task=task['name'],teacher_rank_one_capture=meta['teacher_geometry']['path_rank_one_capture']))
 data=pd.concat(frames);table=data.groupby(['task','optimizer','rule','rate'],as_index=False).validation_nmse.mean();table.to_csv(OUT/'development_rate_selection.csv',index=False);geo=pd.DataFrame(geometry);geo.to_csv(OUT/'development_teacher_geometry.csv',index=False)
 rates={};common={}
 for task in cfg['tasks']:
  name=task['name'];rates[name]={};common[name]={}
  for opt in cfg['rates']:
   part=table[(table.task==name)&(table.optimizer==opt)];rates[name][opt]={r:float(part[part.rule==r].sort_values(['validation_nmse','rate']).iloc[0].rate) for r in cfg['rules']};common[name][opt]=float(part.groupby('rate',as_index=False).validation_nmse.mean().sort_values(['validation_nmse','rate']).iloc[0].rate)
 criteria=cfg['task_selection'];selected=None;checks=[]
 for name in [criteria['primary']]+criteria['fallback_order']:
  g=float(geo[geo.task==name].teacher_rank_one_capture.mean());n=float(table[(table.task==name)&(table.optimizer=='adam')&(table.rule=='exact')].validation_nmse.min());passed=g<=criteria['teacher_rank_one_capture_max'] and n<=criteria['exact_adam_validation_nmse_max'];checks.append(dict(task=name,teacher_rank_one_capture=g,exact_adam_validation_nmse=n,passed=passed))
  if selected is None and passed: selected=name
 record=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),development_freeze_sha256=sha(OUT/'development_freeze.json'),checks=checks,selected_task=selected,confirmatory_tasks=[] if selected is None else [selected,criteria['control']],rates=rates,common_rates=common,selection_did_not_use_between_rule_gap=True)
 write(OUT/'selection_freeze.json',record);print(json.dumps(record,indent=2))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','select']);parser.add_argument('--seed',type=int);parser.add_argument('--phase',choices=['development','fresh'],default='development');args=parser.parse_args()
 if args.action=='freeze': print(json.dumps(freeze(),indent=2))
 elif args.action=='select': select()
 else: run(args.seed,args.phase)
