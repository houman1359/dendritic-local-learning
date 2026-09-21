#!/usr/bin/env python3
"""Prospective local-gate follow-up. Workers require committed protocol provenance."""
import argparse,hashlib,json,os,platform,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import model
HERE=Path(__file__).resolve().parent;J=HERE.parents[1];OUT=J/'source_data/conductance_local_gate'
reference=model  # Independent implementation, tested against the released adapter.

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def utc():return datetime.now(timezone.utc).isoformat()
def protocol():
    cfg=json.loads((OUT/'protocol.json').read_text());freeze=json.loads((OUT/'protocol_freeze.json').read_text())
    assert digest(OUT/'protocol.json')==freeze['protocol_sha256']
    for rel,sha in freeze['scientific_source_sha256'].items():assert digest(J/rel)==sha,(rel,'source changed')
    return cfg,freeze

def committed_protocol():
    cfg,freeze=protocol();repo=J.parent;rel='journal/source_data/conductance_local_gate/protocol.json'
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=repo,text=True).strip()
    assert commit,'Protocol must be committed before scientific execution'
    content=subprocess.check_output(['git','show',commit+':'+rel],cwd=repo)
    assert hashlib.sha256(content).hexdigest()==freeze['protocol_sha256'],'Protocol commit differs'
    return cfg,freeze,commit

def run_task(seed,task,cfg):
    start=time.perf_counter();records=[dict(rule=r,rate=rate,optimizer='adam',primary_rate=rate==cfg['primary_rate']) for rate in cfg['rates'] for r in cfg['rules']]
    n=len(records);rules=[r['rule'] for r in records];rates=np.array([r['rate'] for r in records])
    samples={kind:reference.data(seed,kind,cfg['n_'+kind],task['conflict'],task['gate']) for kind in ['train','validation','test','calibration','diagnostic']}
    x,y,_=samples['train'];vx,vy,_=samples['validation'];tx,ty,_=samples['test'];cx=samples['calibration'][0];dx,dy,_=samples['diagnostic']
    rng=np.random.default_rng(np.random.SeedSequence([seed,45678]));initial=np.log(reference.NOMINAL)+rng.normal(0,.4,24);theta=np.tile(initial,(n,1))
    # Calibration belongs only to comparators; local gates ignore this profile entirely.
    p=model.exact_path(model.forward(initial[None],cx))[0,:,:6].mean(0);profiles=np.tile(p,(n,1))
    m=np.zeros_like(theta);v=np.zeros_like(theta);variance=float(np.var(y));stream=np.random.default_rng(np.random.SeedSequence([seed,56789]))
    best=np.full(n,np.inf);besttheta=theta.copy();beststep=np.zeros(n,int);clipped=np.zeros(n,int);bounded=np.zeros(n,int);firstbound=np.full(n,-1,int)
    curves=[];endpoints=[];snapshots=[];diagnostics=[];beststates=[];beststeps=[];bestwindows=[]
    for step in range(cfg['steps']+1):
        if step in cfg['checkpoints']:
            snapshots.append(theta.copy());vp=model.forward(theta,vx)['output'];tp=model.forward(theta,tx)['output']
            val=np.mean((vp-vy[None])**2,1)/np.var(vy);test=np.mean((tp-ty[None])**2,1)/np.var(ty);change=val<best
            best[change]=val[change];besttheta[change]=theta[change];beststep[change]=step
            for i,r in enumerate(records):
                curves.append(dict(seed=seed,task=task['name'],step=step,**r,validation_nmse=val[i],test_nmse=test[i],gradient_clipped_steps=int(clipped[i]),projected_steps=int(bounded[i]),first_bound_step=int(firstbound[i]),parameters_sha256=array_hash(theta[i]),g_i0=float(np.exp(theta[i,16])),g_i1=float(np.exp(theta[i,17]))))
        if step in cfg['endpoint_budgets']:
            bp=model.forward(besttheta,tx)['output'];bt=np.mean((bp-ty[None])**2,1)/np.var(ty)
            beststates.append(besttheta.copy());beststeps.append(beststep.copy());bestwindows.append(step)
            for i,r in enumerate(records):
                endpoints.append(dict(seed=seed,task=task['name'],budget=step,**r,best_step=int(beststep[i]),validation_nmse=best[i],test_nmse=bt[i],fixed_endpoint_test_nmse=test[i],parameters_sha256=array_hash(besttheta[i]),n_changed_parameters=int(np.sum(np.abs(besttheta[i]-initial)>1e-12)),g_i0_changed=bool(besttheta[i,16]!=initial[16]),g_i1_changed=bool(besttheta[i,17]!=initial[17])))
        if step==cfg['steps']:break
        ix=stream.integers(len(x),size=cfg['batch_size']);gradient,_=model.gradients(theta,x[ix],y[ix],variance,profiles,rules)
        norm=np.linalg.norm(gradient,axis=1);clipped+=norm>cfg['gradient_clip'];gradient*=np.minimum(1,cfg['gradient_clip']/np.maximum(norm,1e-30))[:,None]
        m=.9*m+.1*gradient;v=.999*v+.001*gradient**2;update=(m/(1-.9**(step+1)))/(np.sqrt(v/(1-.999**(step+1)))+1e-8)
        theta-=rates[:,None]*update;hit=np.any((theta<cfg['log_conductance_bounds'][0])|(theta>cfg['log_conductance_bounds'][1]),1);bounded+=hit;firstbound[(firstbound<0)&hit]=step+1
        theta=np.clip(theta,*cfg['log_conductance_bounds']);assert np.isfinite(theta).all()
    # Diagnostics are post-fit and never enter rule delivery or endpoint selection.
    for budget,states in zip(bestwindows,beststates):
        q=model.exact_path(model.forward(states,dx))[:,:,:6]
        for i,r in enumerate(records):
            Q=q[i];s=np.linalg.svd(Q,compute_uv=False);p=profiles[i]
            D=np.zeros((6,3));D[:2,0]=p[:2];D[2:4,1]=p[2:4];D[4:,2]=p[4:]
            projected=Q@D@np.linalg.pinv(D)
            diagnostics.append(dict(seed=seed,task=task['name'],budget=budget,**r,path_rank=int(np.linalg.matrix_rank(Q)),rank_one_capture=float(s[0]**2/np.sum(s*s)),three_profile_capture=float(np.sum(projected**2)/np.sum(Q**2))))
    arrays=dict(theta=np.stack(snapshots),steps=np.array(cfg['checkpoints']),best_theta=np.stack(beststates),best_step=np.stack(beststeps),best_windows=np.array(bestwindows),initial_theta=initial,initial_profiles=profiles,final_theta=theta,final_first_moment=m,final_second_moment=v)
    inputs={}
    for kind,(features,targets,context) in samples.items():inputs[kind]={k:array_hash(a) for k,a in [('features',features),('targets',targets),('context',context)]}
    meta=dict(seed=seed,task=task,records=records,initialization_sha256=array_hash(initial),teacher_log_conductances=reference.teacher(seed,bool(task['conflict'])).tolist(),input_array_sha256=inputs,train_target_variance=variance,test_target_variance=float(np.var(ty)),wall_seconds=time.perf_counter()-start)
    return curves,endpoints,diagnostics,arrays,meta

def run(seed,canary=False):
    cfg,freeze,commit=committed_protocol();assert seed in (cfg['canary_seeds'] if canary else cfg['fresh_seeds'])
    if canary:cfg=dict(cfg,steps=4,checkpoints=[0,1,4],endpoint_budgets=[4],rates=[cfg['primary_rate']])
    folder=OUT/('implementation_canary' if canary else 'runs');folder.mkdir(exist_ok=True)
    assert not (folder/f'seed_{seed}_audit.json').exists(),'Completed outcomes immutable'
    allc=[];alle=[];alld=[];files={};metas=[]
    for task in cfg['tasks']:
        c,e,d,arrays,meta=run_task(seed,task,cfg);allc+=c;alle+=e;alld+=d;metas.append(meta)
        p=folder/f'seed_{seed}_{task["name"]}_states.npz';np.savez_compressed(p,**arrays);files[p.name]=digest(p)
        p=folder/f'seed_{seed}_{task["name"]}_config.json';write(p,dict(protocol_sha256=freeze['protocol_sha256'],protocol_commit=commit,resolved_protocol=cfg,task_metadata=meta));files[p.name]=digest(p)
        print(seed,task['name'],'completed',meta['wall_seconds'],flush=True)
    for name,rows in [('curves',allc),('endpoints',alle),('diagnostics',alld)]:
        p=folder/f'seed_{seed}_{name}.csv';pd.DataFrame(rows).to_csv(p,index=False);files[p.name]=digest(p)
    # Data/initialization pairing across target families; labels deliberately differ.
    assert metas[0]['initialization_sha256']==metas[1]['initialization_sha256']
    for kind in metas[0]['input_array_sha256']:
        for name in ['features','context']:assert metas[0]['input_array_sha256'][kind][name]==metas[1]['input_array_sha256'][kind][name]
    write(folder/f'seed_{seed}_audit.json',dict(seed=seed,canary_excluded=canary,completed_utc=utc(),protocol_sha256=freeze['protocol_sha256'],protocol_commit=commit,scientific_source_sha256=freeze['scientific_source_sha256'],files_sha256=files,python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--canary',action='store_true');a=p.parse_args();run(a.seed,a.canary)
