#!/usr/bin/env python3
"""Predeclared symmetric continuation: no optimizer reset or outcome-based eligibility."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import opponent_model as model
from run_opponent import OUT,freeze,write,sha

def extend(seed):
 cfg=freeze();assert seed in cfg['fresh_seeds'];selection=json.loads((OUT/'selection_freeze.json').read_text());folder=OUT/'runs/fresh';out=OUT/'extension';out.mkdir(exist_ok=True);assert not (out/f'seed_{seed}_audit.json').exists();audit=json.loads((folder/f'seed_{seed}_audit.json').read_text());
 for f,h in audit['files_sha256'].items(): assert sha(folder/f)==h
 allcurves=[];allends=[];alldiag=[];files={}
 for name in selection['confirmatory_tasks']:
  meta=json.loads((folder/f'seed_{seed}_{name}_metadata.json').read_text());task=meta['task'];records=meta['records'];archive=np.load(folder/f'seed_{seed}_{name}_states.npz');theta=archive['final_theta'].copy();m=archive['final_first_moment'].copy();v=archive['final_second_moment'].copy();profiles=archive['initial_profiles'];besttheta=archive['best_theta'].copy();beststep=archive['best_step'].copy();n=len(theta);rates=np.array([r['rate'] for r in records]);rules=[r['rule'] for r in records];adam=np.array([r['optimizer']=='adam' for r in records]);previous=pd.read_csv(folder/f'seed_{seed}_endpoints.csv');previous=previous[previous.task==name];best=previous.validation_nmse.to_numpy().copy()
  # Extend all selected-rate AND common-rate fits, preserving matching budgets.
  data={kind:model.data(seed,kind,cfg['n_'+kind],task['conflict'],task['gate']) for kind in ['train','validation','test','diagnostic']};x,y,_=data['train'];vx,vy,_=data['validation'];tx,ty,_=data['test'];dx,dy,_=data['diagnostic'];variance=meta['training_target_variance'];stream=np.random.default_rng(np.random.SeedSequence([seed,56789]))
  for _ in range(cfg['steps']): stream.integers(len(x),size=cfg['batch_size'])
  snapshots=[];save_steps=[4096,8192,12288,16384];start=time.perf_counter();clipped=np.zeros(n,int);projected=np.zeros(n,int)
  for step in range(4096,16385):
   if step in save_steps:
    snapshots.append(theta.copy());vp=model.forward(theta,vx)['output'];tp=model.forward(theta,tx)['output'];validation=np.mean((vp-vy[None])**2,axis=1)/np.var(vy);test=np.mean((tp-ty[None])**2,axis=1)/np.var(ty);improved=validation<best;best[improved]=validation[improved];besttheta[improved]=theta[improved];beststep[improved]=step;metrics=model.metrics(theta,dx,dy,variance,profiles)
    for i,r in enumerate(records):
     common=dict(seed=seed,task=name,phase='extension',step=step,**r);allcurves.append(dict(**common,validation_nmse=validation[i],test_nmse=test[i],extension_clipped_steps=int(clipped[i]),extension_projected_steps=int(projected[i])));alldiag.append(dict(**common,**metrics[i]))
   if step==16384: break
   ix=stream.integers(len(x),size=cfg['batch_size']);gradient,_=model.gradients(theta,x[ix],y[ix],variance,profiles,rules);norms=np.linalg.norm(gradient,axis=1);clipped+=norms>cfg['gradient_clip'];gradient*=np.minimum(1,cfg['gradient_clip']/np.maximum(norms,1e-30))[:,None];m=.9*m+.1*gradient;v=.999*v+.001*gradient**2;update=gradient.copy();update[adam]=(m[adam]/(1-.9**(step+1)))/(np.sqrt(v[adam]/(1-.999**(step+1)))+1e-8);theta-=rates[:,None]*update;projected+=np.any((theta<cfg['log_conductance_bounds'][0])|(theta>cfg['log_conductance_bounds'][1]),axis=1);theta=np.clip(theta,*cfg['log_conductance_bounds']);assert np.isfinite(theta).all()
  bp=model.forward(besttheta,tx)['output'];bn=np.mean((bp-ty[None])**2,axis=1)/np.var(ty);metrics=model.metrics(besttheta,dx,dy,variance,profiles)
  for i,r in enumerate(records): allends.append(dict(seed=seed,task=name,phase='extension',**r,best_step=int(beststep[i]),validation_nmse=best[i],test_nmse=bn[i],**metrics[i]))
  p=out/f'seed_{seed}_{name}_states.npz';np.savez_compressed(p,theta=np.stack(snapshots),steps=save_steps,best_theta=besttheta,best_step=beststep,final_theta=theta,final_first_moment=m,final_second_moment=v,initial_profiles=profiles);files[p.name]=sha(p);print(seed,name,'extension done',time.perf_counter()-start,flush=True)
 for kind,rows in [('curves',allcurves),('diagnostics',alldiag),('endpoints',allends)]:
  p=out/f'seed_{seed}_{kind}.csv';pd.DataFrame(rows).to_csv(p,index=False);files[p.name]=sha(p)
 write(out/f'seed_{seed}_audit.json',dict(seed=seed,files_sha256=files,primary_audit_sha256=sha(folder/f'seed_{seed}_audit.json'),script_sha256=sha(Path(__file__)),continued_optimizer_and_minibatch_state=True,all_primary_conditions_extended=True,completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();extend(a.seed)
