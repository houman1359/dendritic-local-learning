#!/usr/bin/env python3
"""Frozen development and paired fresh runs for the calibrated-credit bridge."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import models

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
OUT = JOURNAL/'source_data/credit_rule_bridge'
CONFIG = JOURNAL/'configs/credit_rule_bridge/protocol.json'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def freeze():
    config = json.loads(CONFIG.read_text())
    inputs = [CONFIG,HERE/'run.py',HERE/'models.py',HERE/'test_bridge.py',
              HERE.parent/'morphology_credit/experiment.py',HERE.parent/'morphology_structure/model.py',
              HERE.parent/'morphology_structure/constructive_dp_v2.py',HERE.parent/'morphology_conductance/model.py']
    record = dict(protocol=config,source_sha256={str(p.relative_to(JOURNAL)):sha(p) for p in inputs},
                  python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
    path = OUT/'protocol_freeze.json'
    if path.exists(): assert json.loads(path.read_text()) == record, 'Frozen inputs changed'
    else:
        write(path,record)
        write(OUT/'freeze_timestamp.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),status='Before any development outcomes'))
    return config

def conditions(model, phase, cfg):
    fit = json.loads((OUT/'selection_freeze.json').read_text()) if phase == 'fresh' else None
    records = []
    for optimizer in cfg['optimizers']:
        for rule in models.RULES:
            rates = cfg['rates'][model][optimizer]
            selected = None; common = None
            if fit:
                selected = fit['selected_rates'][model][optimizer][rule]
                common = fit['common_rates'][model][optimizer]
                rates = sorted(set([selected, common]))
            for rate in rates:
                records.append(dict(model=model,optimizer=optimizer,rule=rule,rate=rate,
                    selected_rate=bool(fit and rate == selected),common_rate=bool(fit and rate == common)))
    return records

def algebra_data(seed, coeff, kind, count):
    offset = {'training':11,'validation':17,'test':23,'calibration':29}[kind]
    rng = np.random.default_rng(np.random.SeedSequence([seed,offset]))
    x = 2.*rng.integers(2,size=(count,8))-1.
    y = models.structure.fourier_design(x)@coeff+.15*rng.normal(size=count)
    return x,y

def cosine(a,b):
    a = a.reshape(len(a),-1); b = b.reshape(len(b),-1)
    return np.sum(a*b,axis=1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-30)

def run_task(seed, model, task, phase, cfg, steps=None):
    records = conditions(model,phase,cfg); n = len(records)
    rules = [r['rule'] for r in records]
    rates = np.array([r['rate'] for r in records])
    adam = np.array([r['optimizer']=='adam' for r in records])
    root_rng = np.random.default_rng(np.random.SeedSequence([seed,41]))
    if model == 'algebraic':
        coeff, tree = models.algebra_task(seed,task)
        left = np.tile([tree.children[k][0] for k in range(8,15)],(n,1))
        right = np.tile([tree.children[k][1] for k in range(8,15)],(n,1))
        initial = root_rng.normal(0,.5,(7,4)); initial[:,0] = 0.
        theta = np.tile(initial,(n,1,1))
        x,y = algebra_data(seed,coeff,'training',cfg['algebraic']['n_training'])
        vx,vy = algebra_data(seed,coeff,'validation',cfg['algebraic']['n_validation'])
        tx,ty = algebra_data(seed,coeff,'test',cfg['algebraic']['n_test'])
        cx,_ = algebra_data(seed,coeff,'calibration',cfg['n_calibration'])
        dx = models.structure.domain(); dy = models.structure.fourier_design(dx)@coeff
        variance = float(coeff@coeff)
        state = lambda w,xx: models.algebra_state(w,xx,left,right)
        profiles = state(theta,cx)['path'][:,:,:6].mean(axis=1)
        grad = lambda w,xx,yy,rr: models.algebra_grad(w,xx,yy,left,right,variance,profiles,rr)
        metadata = dict(coefficients=coeff.tolist(),children={str(k):list(v) for k,v in tree.children.items()},
            input_spectrum=np.linalg.eigvalsh(models.structure.input_gradient_covariance(coeff)).tolist(),
            centered_cut_bound=models.structure.cut_scores(coeff,tree)['centered_cut_bound'],same_tree_scope='matching and quartet only')
        bound = 2.
    else:
        task_group = int(task)
        groups = np.tile(models.conductance.GROUPINGS[task_group],(n,1,1))
        initial = np.log(models.conductance.NOMINAL_G)+root_rng.normal(0,.4,16)
        theta = np.tile(initial,(n,1))
        x,y = models.conductance.dataset(seed,task_group,'training',cfg['conductance']['n_training'])
        vx,vy = models.conductance.dataset(seed,task_group,'validation',cfg['conductance']['n_validation'])
        tx,ty = models.conductance.dataset(seed,task_group,'test',cfg['conductance']['n_test'])
        cx,_ = models.conductance.dataset(seed,task_group,'calibration',cfg['n_calibration'])
        dx,dy = models.conductance.dataset(seed,task_group,'spectrum',cfg['conductance']['n_diagnostic'])
        variance = float(np.var(y))
        state = lambda w,xx: models.conductance.forward(w,xx,groups)
        profiles = state(theta,cx)['path'][:,:,:6].mean(axis=1)
        grad = lambda w,xx,yy,rr: models.conductance_grad(w,xx,yy,groups,variance,profiles,rr)
        metadata = dict(groups=groups[0].tolist(),teacher_log_conductances=models.conductance.teacher_parameters(seed).tolist(),
            task_group=task_group,compatible=True,same_tree_scope='All rules and three paired compatible input permutations')
        bound = 5.
    moment = np.zeros_like(theta); second = np.zeros_like(theta)
    clipped = np.zeros(n,int); projected = np.zeros(n,int)
    stream = np.random.default_rng(np.random.SeedSequence([seed,37]))
    steps = cfg['steps'] if steps is None else steps
    checkpoints = sorted(set([s for s in cfg['checkpoints'] if s <= steps]+[steps]))
    snapshots = []; rows = []; diag = []; start = time.perf_counter()
    for step in range(steps+1):
        if step in checkpoints:
            snapshots.append(theta.copy())
            pred = state(theta,tx)['output']; vp = state(theta,vx)['output']; ds = state(theta,dx)
            test_variance = variance if model == 'algebraic' else float(np.var(ty))
            val_variance = variance if model == 'algebraic' else float(np.var(vy))
            pop_variance = variance if model == 'algebraic' else float(np.var(dy))
            tn = np.mean((pred-ty[None])**2,axis=1)/test_variance
            vn = np.mean((vp-vy[None])**2,axis=1)/val_variance
            pn = np.mean((ds['output']-dy[None])**2,axis=1)/pop_variance
            delivered,_ = grad(theta,dx,dy,rules)
            exact,_ = grad(theta,dx,dy,['exact']*n)
            error = (ds['output']-dy[None])/variance
            routed = models.deliver(ds['path'],profiles,rules)
            metrics = models.field_metrics(ds['path'],error,routed,profiles)
            gc = cosine(delivered,exact)
            if step:
                optimizer_update = delivered.copy()
                optimizer_update[adam] = (moment[adam]/(1-.9**step))/(np.sqrt(second[adam]/(1-.999**step))+1e-8)
            else: optimizer_update = delivered
            uc = cosine(optimizer_update,exact)
            assert np.isfinite(tn).all() and np.isfinite(vn).all() and np.isfinite(pn).all()
            for i,record in enumerate(records):
                common = dict(seed=seed,phase=phase,task=str(task),step=step,**record)
                rows.append(dict(**common,test_nmse=float(tn[i]),validation_nmse=float(vn[i]),population_nmse=float(pn[i]),
                    parameter_norm=float(np.linalg.norm(theta[i])),max_abs_parameter=float(abs(theta[i]).max()),
                    gradient_clipped_steps=int(clipped[i]),parameter_projected_steps=int(projected[i]),
                    elapsed_seconds=time.perf_counter()-start))
                diag.append(dict(**common,state='own_checkpoint',gradient_cosine=float(gc[i]),
                    optimizer_update_cosine=float(uc[i]),exact_gradient_norm=float(np.linalg.norm(exact[i])),
                    delivered_gradient_norm=float(np.linalg.norm(delivered[i])),**metrics[i]))
        if step == steps: break
        idx = stream.integers(len(x),size=cfg['batch_size'])
        gradient,_ = grad(theta,x[idx],y[idx],rules)
        flat = gradient.reshape(n,-1)
        norms = np.linalg.norm(flat,axis=1)
        clipped += norms > cfg['gradient_norm_clip']
        gradient *= np.minimum(1,cfg['gradient_norm_clip']/np.maximum(norms,1e-30)).reshape((n,)+(1,)*(theta.ndim-1))
        moment = .9*moment+.1*gradient; second = .999*second+.001*gradient**2
        update = gradient.copy()
        update[adam] = (moment[adam]/(1-.9**(step+1)))/(np.sqrt(second[adam]/(1-.999**(step+1)))+1e-8)
        theta -= rates.reshape((n,)+(1,)*(theta.ndim-1))*update
        projected += np.any(abs(theta.reshape(n,-1)) > bound,axis=1)
        theta = np.clip(theta,-bound,bound)
        assert np.isfinite(theta).all(), 'Retain and investigate nonfinite outcomes'
    state_arrays = dict(theta=np.stack(snapshots),steps=np.array(checkpoints),initial_profiles=profiles,
        diagnostic_inputs=dx,diagnostic_targets=dy,calibration_inputs=cx,
        **({'left':left,'right':right,'coefficients':coeff} if model=='algebraic' else {'groups':groups}))
    metadata.update(dict(records=records,variance=variance,model=model,task=str(task),seed=seed,phase=phase,
                         parameter_layout='7 by4 local coefficients' if model=='algebraic' else '16 log-conductances'))
    return rows,diag,state_arrays,metadata

def run_seed(seed, model, phase, benchmark=False):
    cfg = freeze(); assert seed in cfg[phase+'_seeds']
    if phase == 'fresh':
        selection = json.loads((OUT/'selection_freeze.json').read_text())
        assert selection['protocol_freeze_sha256'] == sha(OUT/'protocol_freeze.json')
    folder = OUT/('benchmark' if benchmark else 'runs')/phase/model
    folder.mkdir(parents=True,exist_ok=True)
    audit_path = folder/f'seed_{seed}_audit.json'
    assert not audit_path.exists(), 'Completed outcomes are immutable'
    tasks = list(models.FAMILIES) if model == 'algebraic' else ['0','1','2']
    if benchmark: tasks = tasks[:1]
    rows = []; diagnostics = []; files = {}; start = time.perf_counter()
    for task in tasks:
        values,diag,arrays,metadata = run_task(seed,model,task,phase,cfg,steps=64 if benchmark else None)
        rows.extend(values); diagnostics.extend(diag)
        p = folder/f'seed_{seed}_task_{task}_states.npz'
        np.savez_compressed(p,**arrays); files[p.name] = sha(p)
        p = folder/f'seed_{seed}_task_{task}_metadata.json'
        write(p,metadata); files[p.name] = sha(p)
        print(seed,model,task,'complete',round(time.perf_counter()-start,2),flush=True)
    for kind,values in [('curves',rows),('diagnostics',diagnostics)]:
        p = folder/f'seed_{seed}_{kind}.csv'
        pd.DataFrame(values).to_csv(p,index=False); files[p.name] = sha(p)
    write(audit_path,dict(seed=seed,model=model,phase=phase,benchmark=benchmark,
        seconds=time.perf_counter()-start,n_fits=len(conditions(model,phase,cfg))*len(tasks),
        protocol_freeze_sha256=sha(OUT/'protocol_freeze.json'),
        selection_freeze_sha256=sha(OUT/'selection_freeze.json') if phase=='fresh' else None,
        source_files_sha256=files,all_conditions_retained=True,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    print(json.dumps(json.loads(audit_path.read_text())),flush=True)

def collect(phase, kind):
    cfg = freeze(); frames = []
    for model in ('algebraic','conductance'):
        for seed in cfg[phase+'_seeds']:
            path = OUT/'runs'/phase/model
            audit = json.loads((path/f'seed_{seed}_audit.json').read_text())
            for name,digest in audit['source_files_sha256'].items(): assert sha(path/name) == digest
            frames.append(pd.read_csv(path/f'seed_{seed}_{kind}.csv'))
    return pd.concat(frames,ignore_index=True)

def select():
    cfg = freeze()
    assert not list((OUT/'runs/fresh').glob('*/*')), 'Selection must precede fresh outcomes'
    data = collect('development','curves')
    last = data[data.step == cfg['steps']]
    table = last.groupby(['model','optimizer','rule','rate'],as_index=False).validation_nmse.mean()
    table.to_csv(OUT/'development_rate_selection.csv',index=False)
    selected = {}; common = {}
    for model in ('algebraic','conductance'):
        selected[model] = {}; common[model] = {}
        for optimizer in cfg['optimizers']:
            part = table[(table.model==model)&(table.optimizer==optimizer)]
            selected[model][optimizer] = {r:float(part[part.rule==r].sort_values(['validation_nmse','rate']).iloc[0].rate) for r in models.RULES}
            common[model][optimizer] = float(part.groupby('rate',as_index=False).validation_nmse.mean().sort_values(['validation_nmse','rate']).iloc[0].rate)
    write(OUT/'selection_freeze.json',dict(selected_rates=selected,common_rates=common,
        protocol_freeze_sha256=sha(OUT/'protocol_freeze.json'),selection_table_sha256=sha(OUT/'development_rate_selection.csv'),
        created_utc=pd.Timestamp.now(tz='UTC').isoformat(),status='Frozen before fresh runs',
        selection_criterion='Final validation NMSE, averaged equally across all three tasks and development seeds, per model/optimizer/rule'))
    print(json.dumps(dict(selected=selected,common=common),indent=2),flush=True)

def bootstrap(values):
    values = np.asarray(values); rng = np.random.default_rng(210_999)
    boot = values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),
                n_seeds=len(values),positive_seeds=int((values>0).sum()))

def analyze():
    cfg = freeze(); data = collect('fresh','curves'); diag = collect('fresh','diagnostics')
    folder = OUT/'summaries'; folder.mkdir(exist_ok=True)
    data.to_csv(folder/'all_curves.csv',index=False); diag.to_csv(folder/'all_diagnostics.csv',index=False)
    endpoint = data[data.step==cfg['steps']]
    contrasts = []; seed_values = []
    for sensitivity in ['selected_rate','common_rate']:
        last = endpoint[endpoint[sensitivity]]
        summary = last.groupby(['model','task','optimizer','rule']).test_nmse.agg(['mean','std','count']).reset_index()
        summary.to_csv(folder/f'{sensitivity}_endpoints.csv',index=False)
        for (model,task,optimizer),part in last.groupby(['model','task','optimizer']):
            wide = part.pivot(index='seed',columns='rule',values='test_nmse')
            for control,reference in [('unit_broadcast','exact'),('calibrated_broadcast','exact'),('sign_broadcast','exact'),('unit_broadcast','calibrated_broadcast')]:
                values = wide[control]-wide[reference]
                meta = dict(sensitivity=sensitivity,model=model,task=str(task),optimizer=optimizer,contrast=control+' minus '+reference)
                contrasts.append(dict(**meta,**bootstrap(values)))
                seed_values.extend(dict(**meta,seed=int(seed),difference=float(value)) for seed,value in values.items())
        # The task-by-credit interaction is paired on the identical balanced tree.
        algebra = last[last.model=='algebraic']
        for optimizer,part in algebra.groupby('optimizer'):
            wide = part.pivot(index='seed',columns=['task','rule'],values='test_nmse')
            for control in models.RULES[1:]:
                values = (wide['quartet',control]-wide['quartet','exact'])-(wide['matching',control]-wide['matching','exact'])
                meta = dict(sensitivity=sensitivity,model='algebraic',task='quartet_minus_matching',optimizer=optimizer,contrast=control+' minus exact interaction')
                contrasts.append(dict(**meta,**bootstrap(values)))
                seed_values.extend(dict(**meta,seed=int(seed),difference=float(value)) for seed,value in values.items())
    pd.DataFrame(contrasts).to_csv(folder/'paired_contrasts.csv',index=False)
    pd.DataFrame(seed_values).to_csv(folder/'paired_seed_contrasts.csv',index=False)
    numeric = [c for c in diag if c.startswith(('path_','credit_')) or c in ['gradient_cosine','optimizer_update_cosine']]
    diag[diag.selected_rate].groupby(['model','task','optimizer','rule','step'])[numeric].mean().reset_index().to_csv(folder/'selected_credit_geometry.csv',index=False)
    write(folder/'report.json',dict(n_seed_blocks=len(cfg['fresh_seeds']),n_fresh_fits=len(endpoint),n_curve_rows=len(data),
        all_metrics_finite=bool(np.isfinite(data[['test_nmse','validation_nmse','population_nmse']]).all().all()),
        primary_contrast='Algebraic Adam calibrated-minus-exact gap: quartet minus matching',
        uncertainty='10000 paired whole-seed bootstrap draws; 95% intervals descriptive, no multiplicity claim',
        model_scope='Matching/quartet are isospectral and share architecture/initialization/data; nested is a separate compatible-tree replication; conductance is a different positive teacher model'))
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action',choices=['freeze','benchmark','run','select','analyze'])
    p.add_argument('--phase',choices=['development','fresh'],default='development')
    p.add_argument('--model',choices=['algebraic','conductance']); p.add_argument('--seed',type=int)
    a = p.parse_args()
    if a.action == 'freeze': freeze()
    elif a.action in ('benchmark','run'): run_seed(a.seed,a.model,a.phase,a.action=='benchmark')
    elif a.action == 'select': select()
    else: analyze()
