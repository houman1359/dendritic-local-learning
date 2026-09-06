#!/usr/bin/env python3
"""Development and sealed fresh-seed conductance-credit/input-grouping bridge."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

import model

HERE=Path(__file__).resolve().parent
JOURNAL=HERE.parents[1]
REPO=JOURNAL.parents[2]
OUT=JOURNAL/'source_data/morphology_conductance'
SOURCE=REPO/'src/dendritic_modeling/networks/architectures/excitation_inhibition/dendritic/branch_dynamics.py'
RULES=['exact_path','calibrated_broadcast','subtree_projection','broadcast_projection']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def freeze_development():
    path=OUT/'development_protocol.json'
    if path.exists():return
    write(path,dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),status='Before development training',
        model='Seven-compartment directed passive shunting tree; identity activation; soma-only output',
        compartments=7,couplings=6,excitatory_contacts=4,inhibitory_contacts=6,trainable_positive_conductances=16,
        positivity='exp(log conductance); log conductance constrained to [-5,5]; all inputs lognormal',
        matched_resources='Same compartments, contacts, couplings, trainable count and admissible conductance range; total learned conductance may differ',
        task='Planted positive conductance composition under three leaf-pair groupings; input permutations preserve task input-gradient spectrum',
        teacher='Independent lognormal perturbation SD0.25 around documented positive conductance centers',
        student='Independent initialization SD0.4 around the same centers, shared across paired conditions',
        development_seeds=list(range(15300,15305)),fresh_seeds=list(range(15400,15420)),
        rules=RULES,optimizers=['sgd','adam'],learning_rate_grid={'sgd':[.001,.003,.01],'adam':[.003,.01,.03]},
        optimizer_selection='Per optimizer, minimize final validation NMSE averaged over development seeds, all task/student groupings and all four credit rules',
        primary_contrasts=['Adam exact-path incompatible-group mean minus compatible group','Adam compatible-group calibrated-broadcast minus exact path'],
        secondary='Same SGD contrasts; two-subtree vs both calibrated and projected one-profile controls; no positive credit gap required',
        steps=1000,batch_size=128,n_training=1024,n_validation=1024,n_test=4096,n_calibration=256,
        checkpoints=[0,10,100,300,1000],label_noise_sd=0.,
        loss='Training half-MSE divided by training target variance; reported NMSE=MSE/evaluation target variance',
        delivery='Frozen initial student own-state mean path profile estimated on label-free calibration; projected coefficients use current exact field and are oracle diagnostics',
        learning='Manual local eligibility times delivered voltage error, including exp-conductance chain derivative; matched optimizer within every comparison',
        uncertainty='Twenty independent fresh paired seeds; 10000 seed-bootstrap draws; full conditions retained',
        scope='Mechanistic planted-composition positive control; fixed physical shape with permuted input grouping; no biological teaching-signal or unseen-model-class claim',
        code_sha256={str(p.relative_to(JOURNAL)):sha(p) for p in sorted(HERE.glob('*.py'))},
        production_forward_sha256=sha(SOURCE),python=platform.python_version(),numpy=np.__version__))


def check_development():
    cfg=json.loads((OUT/'development_protocol.json').read_text())
    for path,digest in cfg['code_sha256'].items():assert sha(JOURNAL/path)==digest,path
    assert sha(SOURCE)==cfg['production_forward_sha256']
    return cfg


def conditions(cfg,phase):
    selected=None
    if phase=='fresh':selected=json.loads((OUT/'fresh_protocol.json').read_text())['selected_learning_rates']
    records=[]
    for optimizer in cfg['optimizers']:
        rates=cfg['learning_rate_grid'][optimizer] if selected is None else [selected[optimizer]]
        for lr in rates:
            for grouping in range(3):
                for rule_id,rule in enumerate(RULES):
                    records.append(dict(optimizer=optimizer,learning_rate=lr,student_group=grouping,
                        student_group_name=model.GROUP_NAMES[grouping],credit_rule=rule,rule_id=rule_id))
    return records


def evaluate(theta,x,y,groups):
    state=model.forward(theta,x,groups)
    mse=np.mean((state['output']-y[None])**2,axis=1)
    return mse/np.var(y),mse,state


def run_task(seed,task_group,cfg,phase):
    records=conditions(cfg,phase);n=len(records)
    groups=model.GROUPINGS[[r['student_group'] for r in records]]
    rules=np.array([r['rule_id'] for r in records]);rates=np.array([r['learning_rate'] for r in records])
    adam=np.array([r['optimizer']=='adam' for r in records])
    rng=np.random.default_rng(seed+920_000)
    initial=np.log(model.NOMINAL_G)+rng.normal(0,.4,16)
    theta=np.tile(initial,(n,1));moment=np.zeros_like(theta);second=np.zeros_like(theta)
    x,y=model.dataset(seed,task_group,'training',cfg['n_training']);variance=float(np.var(y))
    vx,vy=model.dataset(seed,task_group,'validation',cfg['n_validation'])
    tx,ty=model.dataset(seed,task_group,'test',cfg['n_test'])
    cx,cy=model.dataset(seed,task_group,'calibration',cfg['n_calibration'])
    profiles=model.calibrate_profiles(theta,cx,groups)
    stream=np.random.default_rng(np.random.SeedSequence([seed,939]))
    rows=[];diagnostics=[];bounds=np.zeros(n,dtype=int);training_seconds=0.
    exact_lookup={(r['optimizer'],r['learning_rate'],r['student_group']):i for i,r in enumerate(records) if r['rule_id']==0}
    reference=np.array([exact_lookup[(r['optimizer'],r['learning_rate'],r['student_group'])] for r in records])
    for step in range(cfg['steps']+1):
        if step in cfg['checkpoints']:
            vn,vm,_=evaluate(theta,vx,vy,groups);tn,tm,state=evaluate(theta,tx,ty,groups)
            cn,_,_=evaluate(theta,x,y,groups)
            # Common exact-state diagnostics isolate backward delivery from its
            # accumulated effect on the forward states.
            shared=theta[reference]
            delivered,common=model.gradients(shared,cx,cy,groups,variance,profiles,rules)
            exact,_=model.gradients(shared,cx,cy,groups,variance)
            delivered_paths=model.delivery(common['path'],profiles,groups,rules)
            path_capture=1-np.sum((delivered_paths[:,:,:6]-common['path'][:,:,:6])**2,axis=(1,2))/np.sum(common['path'][:,:,:6]**2,axis=(1,2))
            for i,record in enumerate(records):
                metadata=dict(seed=seed,task_group=task_group,task_group_name=model.GROUP_NAMES[task_group],
                    compatible=record['student_group']==task_group,phase=phase,step=step,**record)
                rows.append(dict(**metadata,train_nmse=float(cn[i]),validation_nmse=float(vn[i]),test_nmse=float(tn[i]),
                    test_mse=float(tm[i]),test_target_variance=float(np.var(ty)),conductance_sum=float(np.exp(theta[i]).sum()),
                    min_conductance=float(np.exp(theta[i]).min()),max_conductance=float(np.exp(theta[i]).max()),
                    min_total_conductance=float(state['denominator'][i].min()),
                    max_voltage=float(state['voltage'][i].max()),bound_events=int(bounds[i]),training_seconds=training_seconds))
                norms=float(np.linalg.norm(exact[i])*np.linalg.norm(delivered[i]))
                cosine=float(exact[i]@delivered[i]/norms) if norms>1e-24 else None
                diagnostics.append(dict(**metadata,state='common_exact_state',gradient_cosine=cosine,
                    exact_gradient_norm=float(np.linalg.norm(exact[i])),delivered_gradient_norm=float(np.linalg.norm(delivered[i])),
                    gradient_relative_error=float(np.linalg.norm(delivered[i]-exact[i])/max(np.linalg.norm(exact[i]),1e-12)),
                    path_capture=float(path_capture[i])))
        if step==cfg['steps']:break
        before=time.perf_counter()
        idx=stream.integers(len(x),size=cfg['batch_size'])
        gradient,_=model.gradients(theta,x[idx],y[idx],groups,variance,profiles,rules)
        moment=.9*moment+.1*gradient;second=.999*second+.001*gradient**2
        update=gradient.copy()
        update[adam]=(moment[adam]/(1-.9**(step+1)))/(np.sqrt(second[adam]/(1-.999**(step+1)))+1e-8)
        theta-=rates[:,None]*update
        bounds+=np.any((theta < -5)|(theta > 5),axis=1)
        theta=np.clip(theta,-5,5)
        assert np.isfinite(theta).all(),'Nonfinite parameters: retain and diagnose failure'
        training_seconds+=time.perf_counter()-before
    return rows,diagnostics,theta,profiles


def run_seed(seed,phase):
    cfg=check_development()
    allowed=cfg['development_seeds'] if phase=='development' else cfg['fresh_seeds']
    assert seed in allowed
    if phase=='fresh':
        fresh=json.loads((OUT/'fresh_protocol.json').read_text())
        assert fresh['development_protocol_sha256']==sha(OUT/'development_protocol.json')
    dest=OUT/'runs'/phase;dest.mkdir(parents=True,exist_ok=True)
    if (dest/f'seed_{seed}_audit.json').exists():raise FileExistsError('Completed seed outcomes are retained and immutable')
    start=time.perf_counter();rows=[];diagnostics=[];snapshots={};spectra=[];task_outputs=[]
    for task_group in range(3):
        values,diag,theta,profiles=run_task(seed,task_group,cfg,phase)
        rows.extend(values);diagnostics.extend(diag)
        snapshots[f'task_{task_group}_final_log_conductances']=theta
        snapshots[f'task_{task_group}_initial_profiles']=profiles
        xx,yy=model.dataset(seed,task_group,'spectrum',2048)
        jac=model.teacher_gradient_in_task_coordinates(seed,task_group,xx)
        spectra.append(np.linalg.eigvalsh(jac.T@jac/len(jac)));task_outputs.append(yy)
    spectrum_difference=float(np.max(np.abs(np.array(spectra)-spectra[0])))
    assert spectrum_difference<1e-12
    pd.DataFrame(rows).to_csv(dest/f'seed_{seed}_curves.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(dest/f'seed_{seed}_credit_diagnostics.csv',index=False)
    np.savez_compressed(dest/f'seed_{seed}_final_states.npz',**snapshots)
    write(dest/f'seed_{seed}_audit.json',dict(seed=seed,phase=phase,n_curve_rows=len(rows),
        n_fits=len(conditions(cfg,phase))*3,elapsed_seconds=time.perf_counter()-start,
        development_protocol_sha256=sha(OUT/'development_protocol.json'),
        fresh_protocol_sha256=sha(OUT/'fresh_protocol.json') if phase=='fresh' else None,
        input_gradient_spectrum_eigenvalues=np.array(spectra).tolist(),input_gradient_spectrum_max_difference=spectrum_difference,
        paired_task_output_max_difference=float(np.max(np.abs(np.array(task_outputs)-task_outputs[0]))),
        all_conditions_retained=True,numpy=np.__version__,completed_utc=pd.Timestamp.now(tz='UTC').isoformat()))
    print(json.dumps(dict(seed=seed,phase=phase,seconds=time.perf_counter()-start)),flush=True)


def select_fresh():
    cfg=check_development()
    assert not list((OUT/'runs/fresh').glob('*')),'Fresh outcomes already exist'
    tables=[]
    for seed in cfg['development_seeds']:
        assert (OUT/'runs/development'/f'seed_{seed}_audit.json').exists()
        tables.append(pd.read_csv(OUT/'runs/development'/f'seed_{seed}_curves.csv'))
    data=pd.concat(tables,ignore_index=True);last=data[data.step.eq(cfg['steps'])]
    summary=last.groupby(['optimizer','learning_rate']).validation_nmse.mean().reset_index()
    summary.to_csv(OUT/'development_learning_rate_selection.csv',index=False)
    rates={}
    for optimizer,g in summary.groupby('optimizer'):
        rates[optimizer]=float(g.sort_values(['validation_nmse','learning_rate'],kind='stable').iloc[0].learning_rate)
    write(OUT/'fresh_protocol.json',dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        status='Frozen before any fresh-seed calibration or training',development_protocol_sha256=sha(OUT/'development_protocol.json'),
        selected_learning_rates=rates,selection_table_sha256=sha(OUT/'development_learning_rate_selection.csv'),
        fresh_seeds=cfg['fresh_seeds'],n_fits=20*3*3*4*2,
        primary_contrasts=cfg['primary_contrasts'],all_other_parameters_unchanged=True,
        comparison_unit='Seed; task/input permutations are paired conditions, not independent replicates'))
    print(json.dumps(rates,indent=2))


def boot(values):
    values=np.asarray(values,float);rng=np.random.default_rng(159_000)
    draws=values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1)
    return float(values.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))


def analyze(phase):
    cfg=check_development();dest=OUT/'summaries'/phase;dest.mkdir(parents=True,exist_ok=True)
    seeds=cfg['development_seeds'] if phase=='development' else cfg['fresh_seeds']
    rows=[];diags=[]
    for seed in seeds:
        assert (OUT/'runs'/phase/f'seed_{seed}_audit.json').exists()
        rows.append(pd.read_csv(OUT/'runs'/phase/f'seed_{seed}_curves.csv'))
        diags.append(pd.read_csv(OUT/'runs'/phase/f'seed_{seed}_credit_diagnostics.csv'))
    data=pd.concat(rows,ignore_index=True);diagnostic=pd.concat(diags,ignore_index=True)
    data.to_csv(dest/'all_learning_curves.csv',index=False);diagnostic.to_csv(dest/'all_credit_diagnostics.csv',index=False)
    summary=[]
    for group,g in data.groupby(['optimizer','learning_rate','credit_rule','compatible','step']):
        avg,lo,hi=boot(g.groupby('seed').test_nmse.mean())
        summary.append(dict(zip(['optimizer','learning_rate','credit_rule','compatible','step'],group))|dict(
            mean_test_nmse=avg,ci95_low=lo,ci95_high=hi,n_seed_blocks=g.seed.nunique(),n_fits=len(g)))
    pd.DataFrame(summary).to_csv(dest/'learning_summary.csv',index=False)
    last=data[data.step.eq(cfg['steps'])]
    contrasts=[];seed_contrasts=[]
    for (optimizer,lr),g in last.groupby(['optimizer','learning_rate']):
        compatibility=g.groupby(['seed','credit_rule','compatible']).test_nmse.mean().unstack('compatible')
        for rule in RULES:
            z=compatibility.xs(rule,level='credit_rule')
            values=z[False]-z[True];name=f'incompatible_minus_compatible__{rule}'
            avg,lo,hi=boot(values)
            contrasts.append(dict(optimizer=optimizer,learning_rate=lr,contrast=name,mean_difference=avg,ci95_low=lo,ci95_high=hi,
                positive_seeds=int((values>0).sum()),n_seed_blocks=len(values)))
            seed_contrasts.extend(dict(optimizer=optimizer,learning_rate=lr,contrast=name,seed=int(seed),difference=float(value)) for seed,value in values.items())
        for compatible in [True,False]:
            z=g[g.compatible.eq(compatible)].groupby(['seed','credit_rule']).test_nmse.mean().unstack('credit_rule')
            for reference,control in [('exact_path','calibrated_broadcast'),('exact_path','broadcast_projection'),
                                      ('exact_path','subtree_projection'),('subtree_projection','calibrated_broadcast'),
                                      ('subtree_projection','broadcast_projection')]:
                values=z[control]-z[reference];name=f'{control}_minus_{reference}__compatible_{compatible}'
                avg,lo,hi=boot(values)
                contrasts.append(dict(optimizer=optimizer,learning_rate=lr,contrast=name,mean_difference=avg,ci95_low=lo,ci95_high=hi,
                    positive_seeds=int((values>0).sum()),n_seed_blocks=len(values)))
                seed_contrasts.extend(dict(optimizer=optimizer,learning_rate=lr,contrast=name,seed=int(seed),difference=float(value)) for seed,value in values.items())
    pd.DataFrame(contrasts).to_csv(dest/'paired_contrasts.csv',index=False)
    pd.DataFrame(seed_contrasts).to_csv(dest/'paired_seed_contrasts.csv',index=False)
    diagnostic.groupby(['optimizer','learning_rate','credit_rule','compatible','step'])[['gradient_cosine','gradient_relative_error','path_capture']].mean().reset_index().to_csv(dest/'credit_geometry_summary.csv',index=False)
    final_state=last.groupby(['optimizer','learning_rate','credit_rule','compatible'])[['conductance_sum','min_conductance','max_conductance','bound_events','min_total_conductance']].agg(['mean','min','max'])
    final_state.to_csv(dest/'physical_parameter_summary.csv')
    write(dest/'report.json',dict(phase=phase,n_seed_blocks=len(seeds),n_fits=len(last),n_curve_rows=len(data),
        all_outcomes_retained=True,all_metrics_finite=bool(np.isfinite(data.test_nmse).all()),
        min_total_conductance=float(data.min_total_conductance.min()),max_voltage=float(data.max_voltage.max()),
        max_bound_events=int(data.bound_events.max()),
        exact_path_min_final_compatible_nmse=float(last[last.compatible&last.credit_rule.eq('exact_path')].test_nmse.min())))
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze_development','run','select_fresh','analyze'])
    parser.add_argument('--phase',choices=['development','fresh'],default='development');parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.command=='freeze_development':freeze_development()
    elif args.command=='run':run_seed(args.seed,args.phase)
    elif args.command=='select_fresh':select_fresh()
    else:analyze(args.phase)
