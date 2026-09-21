#!/usr/bin/env python3
"""Frozen, paired Boolean-tree learning experiment; no privileged initialization."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from boolean_morphology.model import FAMILIES, TREES, domain, raw_target, normalization, target, pack_tree, forward, gradient, classification
ROOT = HERE.parents[1]
OUT = ROOT / 'source_data/boolean_morphology'
RATES = (.003, .01, .03)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def protocol():
    return dict(version=1, development_seeds=list(range(2026110,2026115)), fresh_seeds=list(range(2026210,2026230)),
        excluded_smoke_seed=2026000, families=list(FAMILIES), trees=TREES,
        optimizers=['sgd','adam'], rules=['exact','broadcast'], rates=list(RATES),
        steps=2048, checkpoints=[0,1,16,64,256,512,1024,2048], batch_size=32,
        training_samples=256, test_samples=1024, noise_sd=.15, parameter_bound=2., gradient_clip=10.,
        initialization_sd=.5, zero_initial_bias=True, adam_betas=[.9,.999], adam_epsilon=1e-8,
        resources=dict(inputs=4, internal_units=3, parameters=12, edges=6, root_outputs=1),
        label_coordinates='bits=(physical Rademacher x[:,common_seed_permutation]+1)/2; center and population-variance-normalize each complete16-row truth table',
        permutation='One common physical leaf permutation per seed applies to all semantic task variables and all candidate tree leaves.',
        seeds='Each seed has independent per-family training/test/initialization/batch streams; all paired tree/rule/optimizer/rate conditions share these streams within family.',
        input_access='All16 input patterns have uniform probability. Training/test draws are independent with replacement and fresh normalized-label Gaussian noise. Input patterns may recur.',
        initialization='GaussianSD0.5 node coefficients; biases0; clip tobox2; no target labels/coefficients/gates supplied to initialization.',
        update='Gradient of half mean squared normalized-label error. Exact path derivative or all nonroot sensitivities1; root sensitivity1 underboth. Train every coefficient.',
        failures='Retain all outcomes. A nonfinite gradient/update stops only that condition; lastfinite weights retained and subsequent NMSE assigned1e6, accuracies0 with failure_step.',
        rate_selection='For each optimizer/rule, choose smallest minimizer within absolute1e-12 tolerance of mean final CLEAN populationNMSE over all5developmentseeds,7tasks,4trees. Keep all3rates in fresh runs. Bounded recipe comparison; no exhaustive optimizer claim.',
        primary_endpoint='Exhaustive CLEAN16-pattern NMSE in exactly centered,unitvariance target coordinates; no sampled-test or observed-class-variance denominator.',
        secondary_endpoints='Independent1024-example noisy testNMSE; exhaustive accuracy and balancedaccuracy at normalized threshold(raw0.5); gradientcosine and clip counts descriptive.',
        primary_comparisons=['Adam exact XOR-of-AND: mean balanced_ac_bd/balanced_ad_bc minus balanced_ab_cd',
                             'Adam XOR-of-AND on balanced_ab_cd: broadcast minus exact'],
        primary_rate_scope='Use frozen per-optimizer/per-rule rates; broadcast/exact comparison is a development-selected recipe contrast. Same-rate outcomes retained.',
        primary_inference='20 paired seed blocks;10000bootstrapdraws;two Bonferroni-adjusted two-sided97.5% intervals; success requires positive lower bound and meanimprovement>=0.01NMSE. Other intervals95%descriptive.',
        functional_scope='Seven FIXED logical templates. Random permutation/data/noise/initialization seeds are new, not new functional families. No general Boolean/morphology law or biophysical equivalence claimed.',
        expected_fresh_fits=6720)


def core_hashes():
    return {f'scripts/boolean_morphology/{name}':sha(HERE/name) for name in ('__init__.py','model.py','experiment.py')}


def freeze():
    if (OUT/'protocol_freeze.json').exists():
        return verify()
    assert not list((OUT/'runs').glob('*/*.csv')), 'No previous fitting before protocol freeze'
    dump(OUT/'protocol.json', protocol())
    dump(OUT/'protocol_freeze.json',dict(utc=utc(), protocol_sha256=sha(OUT/'protocol.json'), source_files=core_hashes()))
    return verify()


def verify():
    record=json.loads((OUT/'protocol_freeze.json').read_text())
    assert record['protocol_sha256']==sha(OUT/'protocol.json')
    assert record['source_files']==core_hashes(), 'Frozen scientific code changed'
    return json.loads((OUT/'protocol.json').read_text())


def seed_data(seed, cfg):
    permutation=np.random.default_rng(np.random.SeedSequence([seed,701])).permutation(4)
    px=domain(); arrays={'permutation':permutation}
    trains=[];ys=[];tests=[];tys=[];initials=[];batch_indices=[];pop=[];raw=[]
    for index,family in enumerate(FAMILIES):
        for role,n,xs,zs in ((11,cfg['training_samples'],trains,ys),(23,cfg['test_samples'],tests,tys)):
            rng=np.random.default_rng(np.random.SeedSequence([seed,index,role]))
            x=2.*rng.integers(2,size=(n,4))-1.
            xs.append(x);zs.append(target(x,family,permutation)+cfg['noise_sd']*rng.normal(size=n))
        init=np.random.default_rng(np.random.SeedSequence([seed,index,41])).normal(0,cfg['initialization_sd'],(3,4))
        init[:,0]=0.;initials.append(np.clip(init,-cfg['parameter_bound'],cfg['parameter_bound']))
        rng=np.random.default_rng(np.random.SeedSequence([seed,index,37]))
        batch_indices.append(rng.integers(cfg['training_samples'],size=(cfg['steps'],cfg['batch_size'])))
        pop.append(target(px,family,permutation));raw.append(raw_target(px,family,permutation))
    arrays.update(train_x=np.array(trains),train_y=np.array(ys),test_x=np.array(tests),test_y=np.array(tys),
        initial_weights=np.array(initials),batch_indices=np.array(batch_indices),population_x=px,
        population_y=np.array(pop),raw_population_y=np.array(raw))
    return arrays


def conditions(permutation):
    rows=[];children=[]
    for family in FAMILIES:
        for name,tree in TREES.items():
            for optimizer in ('sgd','adam'):
                for rule in ('exact','broadcast'):
                    for rate in RATES:
                        rows.append(dict(family=family,tree=name,optimizer=optimizer,rule=rule,rate=rate))
                        children.append(pack_tree(tree,permutation))
    return rows,np.array(children)


def train(seed, cfg):
    arrays=seed_data(seed,cfg);metadata,children=conditions(arrays['permutation']);count=len(metadata)
    fi=np.array([FAMILIES.index(row['family']) for row in metadata]);ci=np.arange(count)
    weights=arrays['initial_weights'][fi].copy();m=np.zeros_like(weights);v=np.zeros_like(weights)
    adam=np.array([row['optimizer']=='adam' for row in metadata]);broadcast=np.array([row['rule']=='broadcast' for row in metadata])
    rates=np.array([row['rate'] for row in metadata])[:,None,None]
    clip_counts=np.zeros(count,int);box_steps=np.zeros(count,int);box_coefficients=np.zeros(count,int);failure_step=np.full(count,-1,int)
    means=np.array([normalization(row['family'])[0] for row in metadata])[:,None]
    sds=np.array([normalization(row['family'])[1] for row in metadata])[:,None]
    rows=[];start=time.perf_counter()
    def evaluate(step):
        prediction=forward(arrays['population_x'],weights,children)[:,6]
        test_prediction=forward(arrays['test_x'][fi],weights,children)[:,6]
        loss=np.mean((prediction-arrays['population_y'][fi])**2,axis=1)
        test_loss=np.mean((test_prediction-arrays['test_y'][fi])**2,axis=1)
        acc,balanced=classification(prediction,arrays['raw_population_y'][fi],means,sds)
        delivered,exact,_,_=gradient(arrays['population_x'],arrays['population_y'][fi],weights,children,broadcast)
        dot=np.sum(delivered*exact,axis=(1,2));norm=np.linalg.norm(delivered,axis=(1,2))*np.linalg.norm(exact,axis=(1,2))
        for i,row in enumerate(metadata):
            failed=failure_step[i]>=0
            rows.append(dict(seed=seed,condition_id=i,**row,step=step,
                population_nmse=1e6 if failed else float(loss[i]),test_noisy_nmse=1e6 if failed else float(test_loss[i]),
                accuracy=0. if failed else float(acc[i]),balanced_accuracy=0. if failed else float(balanced[i]),
                gradient_cosine=float(dot[i]/max(norm[i],1e-30)),gradient_clipped_steps=int(clip_counts[i]),
                parameter_clipped_steps=int(box_steps[i]),parameter_clipped_coefficients=int(box_coefficients[i]),
                max_abs_parameter=float(abs(weights[i]).max()),parameter_norm=float(np.linalg.norm(weights[i])),
                failed=bool(failed),failure_step=int(failure_step[i]),elapsed_seconds=time.perf_counter()-start))
    evaluate(0)
    for step in range(1,cfg['steps']+1):
        batch=arrays['batch_indices'][:,step-1][fi]
        x=arrays['train_x'][fi[:,None],batch];y=arrays['train_y'][fi[:,None],batch]
        g,_,_,_=gradient(x,y,weights,children,broadcast)
        bad=~np.isfinite(g).all(axis=(1,2));failure_step[bad&(failure_step<0)]=step
        g[failure_step>=0]=0.
        norms=np.linalg.norm(g,axis=(1,2));clip_counts+=norms>cfg['gradient_clip']
        g*=np.minimum(1.,cfg['gradient_clip']/np.maximum(norms,1e-30))[:,None,None]
        m=.9*m+.1*g;v=.999*v+.001*g*g
        update=g.copy();update[adam]=(m[adam]/(1-.9**step))/(np.sqrt(v[adam]/(1-.999**step))+1e-8)
        update[failure_step>=0]=0.
        proposal=weights-rates*update
        bad=~np.isfinite(proposal).all(axis=(1,2));failure_step[bad&(failure_step<0)]=step
        proposal[failure_step>=0]=weights[failure_step>=0]
        exceeds=abs(proposal)>cfg['parameter_bound'];box_steps+=exceeds.any(axis=(1,2));box_coefficients+=exceeds.sum(axis=(1,2))
        weights=np.clip(proposal,-cfg['parameter_bound'],cfg['parameter_bound'])
        if step in cfg['checkpoints']:evaluate(step)
    arrays.update(final_weights=weights,children=children,family_index=fi,failure_step=failure_step)
    return pd.DataFrame(rows),arrays


def run(split,seed):
    cfg=verify()
    assert seed in ([cfg['excluded_smoke_seed']] if split=='excluded_smoke' else cfg[split+'_seeds'])
    if split=='fresh':
        seal=json.loads((OUT/'selection_freeze.json').read_text())
        assert seal['protocol_sha256']==sha(OUT/'protocol.json') and seal['rate_file_sha256']==sha(OUT/'selected_rates.json')
        assert seal['source_files']==core_hashes()
    dest=OUT/'runs'/split;dest.mkdir(parents=True,exist_ok=True)
    if any(dest.glob(f'seed_{seed}.*')):raise FileExistsError('Seed output already exists; preserve original record')
    started=utc();frame,arrays=train(seed,cfg);frame.insert(1,'split',split)
    csv_path=dest/f'seed_{seed}.csv';npz_path=dest/f'seed_{seed}.npz'
    frame.to_csv(csv_path,index=False);np.savez_compressed(npz_path,**arrays)
    dump(dest/f'seed_{seed}.json',dict(seed=seed,split=split,utc_started=started,utc_completed=utc(),
        rows=len(frame),fits=len(frame[frame.step==cfg['steps']]),failed_fits=int(frame[frame.step==cfg['steps']].failed.sum()),
        csv_sha256=sha(csv_path),npz_sha256=sha(npz_path),protocol_sha256=sha(OUT/'protocol.json'),source_files=core_hashes(),
        selection_sha256=sha(OUT/'selection_freeze.json') if split=='fresh' else None,
        scheduler_job_id=os.environ.get('SLURM_JOB_ID'),all_outcomes_retained=True))
    print(json.dumps(dict(seed=seed,split=split,rows=len(frame),seconds=float(frame.elapsed_seconds.max())),indent=2))


def collect(split):
    cfg=verify()
    return pd.concat([pd.read_csv(OUT/'runs'/split/f'seed_{seed}.csv') for seed in cfg[split+'_seeds']],ignore_index=True)


def select():
    cfg=verify();assert not list((OUT/'runs/fresh').glob('*.csv'))
    frame=collect('development');end=frame[frame.step==cfg['steps']]
    scores=end.groupby(['optimizer','rule','rate'],as_index=False).population_nmse.mean()
    choices={}
    for optimizer in cfg['optimizers']:
        choices[optimizer]={}
        for rule in cfg['rules']:
            z=scores[(scores.optimizer==optimizer)&(scores.rule==rule)]
            choices[optimizer][rule]=float(z[z.population_nmse<=z.population_nmse.min()+1e-12].rate.min())
    scores.to_csv(OUT/'development_rate_scores.csv',index=False)
    dump(OUT/'selected_rates.json',choices)
    dump(OUT/'selection_freeze.json',dict(utc=utc(),protocol_sha256=sha(OUT/'protocol.json'),
        source_files=core_hashes(),rate_file_sha256=sha(OUT/'selected_rates.json'),
        development_files={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'runs/development').glob('*'))}))
    print(scores.to_string(index=False));print(json.dumps(choices,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['freeze','run','select'])
    parser.add_argument('--split',choices=['development','fresh','excluded_smoke']);parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.command=='freeze':print(json.dumps(freeze(),indent=2))
    elif args.command=='run':run(args.split,args.seed)
    else:select()


if __name__=='__main__':main()
