#!/usr/bin/env python3
"""Learn task-compatible algebraic trees with exact and restricted credit.

Architecture uses oracle interaction information to isolate learnability from
structure estimation. No target-derived weights initialize any training run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'morphology_structure'))
from model import Tree,domain,fourier_design,cut_scores,input_gradient_covariance
from constructive_dp_v2 import tree_from_coeff

OUT=ROOT/'source_data/morphology_credit'
FAMILIES=('matching','quartet','nested')
RULES=('exact','broadcast','global_projection','subtree_projection','shuffled_projection')
RATES=(.003,.01,.03)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def protocol():
    return dict(version=1,development_seeds=list(range(126100,126103)),fresh_seeds=list(range(127200,127220)),
        families=list(FAMILIES),rules=list(RULES),rates=list(RATES),optimizers=['sgd','adam'],
        structures=['compatible','assignment_shuffled'],steps=1024,checkpoints=[0,1,16,64,256,1024],batch_size=64,
        training_samples=2048,test_samples=4096,training_noise_sd=.15,test_noise_sd=.15,parameter_bound=2.,
        gradient_norm_clip=10.,seed_role_offsets=dict(training=11,test=23,batches=37,initialization=41),
        task='Independent leaf permutations and random interaction signs; coefficient magnitudes0.5; same sensitivity spectrum within family',
        architecture='Oracle centered-cut DP chooses compatible tree; independent random permutation of leaf labels preserves its shape and all28coefficients/14edges',
        initialization='Label-independent normal0.5 weights,zero biases; common initial coefficients across paired assignments,rules,optimizers,rates',
        credit='Exact path derivatives; unit soma broadcast; best per-trial projections of exact path field into one global or two proximal-subtree profiles; size-matched shuffled profiles',
        root='Root sensitivity remains exactly1 under everyrule; projection concerns six nonsomatic internal units',
        coefficient_access='All projection rules use current exact field to obtain oracle coefficients; no coefficient-estimator claim',
        selection='Per-optimizer/per-rule learning rate minimizes mean development populationNMSE across allfamilies/structures; global common-rate sensitivity also frozen',
        outcome='Primary final testNMSE under the frozen per-rule learning rate; populationNMSE against clean target secondary; allrate outcomesretained',
        inference='20 independent paired seed blocks; condition averages within seed; bootstrapwhole seeds; subgroup and optimizer sensitivity retained',
        scope='Bounded multi-affine algebraic learning; architecture is oracle-informed; no biophysical or limited-data architecture-prediction claim')


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    config=protocol()
    target=OUT/'protocol.json'
    if target.exists():assert json.loads(target.read_text())==config
    else:dump(target,config)
    record=dict(protocol_sha256=sha(target),runner_sha256=sha(HERE/'experiment.py'),
                model_sha256=sha(HERE.parent/'morphology_structure/model.py'),
                constructor_sha256=sha(HERE.parent/'morphology_structure/constructive_dp_v2.py'))
    path=OUT/'protocol_freeze.json'
    if path.exists():assert json.loads(path.read_text())==record
    else:dump(path,record)
    return config


def make_task(seed,family):
    rng=np.random.default_rng(np.random.SeedSequence([seed,FAMILIES.index(family),801]))
    permutation=rng.permutation(8)
    if family=='matching':supports=[permutation[j:j+2] for j in range(0,8,2)]
    elif family=='quartet':supports=[permutation[:4],permutation[4:]]
    else:supports=[permutation[:j] for j in (2,4,6,8)]
    coeff=np.zeros(256)
    for support in supports:coeff[sum(1<<int(j) for j in support)]=.5*rng.choice([-1,1])
    return coeff


def shuffled_tree(tree,seed):
    perm=np.random.default_rng(seed+822_000).permutation(8)
    children={n:tuple(int(perm[k]) if k<8 else k for k in kids) for n,kids in tree.children.items()}
    descendants={i:(i,) for i in range(8)};parent={}
    for node,(left,right) in children.items():
        descendants[node]=descendants[left]+descendants[right]
        parent[left],parent[right]=(node,0),(node,1)
    return Tree('assignment_shuffled',tree.shape,descendants[14],children,descendants,parent,14)


def zones(tree):
    """Two proximal subtrees of nonsomatic internal nodes; root routed exactly."""
    internal=set(range(8,14))
    def below(node):
        if node<8:return set()
        result={node}
        for kid in tree.children[node]:result |= below(kid)
        return result & internal
    groups=[below(kid) for kid in tree.children[14] if kid>=8]
    assert len(groups)==2 and all(groups),'This task family must have two internal root children'
    p=np.zeros((6,6))
    for group in groups:
        ix=[n-8 for n in group]
        p[np.ix_(ix,ix)]=1/len(ix)
    assert np.linalg.matrix_rank(p)==2
    return p


def pack(trees,seed):
    metadata=[];left=[];right=[];projectors=[]
    shuffle=np.random.default_rng(seed+823_000).permutation(6)
    for structure,tree in zip(['compatible','assignment_shuffled'],trees):
        p=zones(tree)
        for optimizer in ('sgd','adam'):
            for rate in RATES:
                for rule in RULES:
                    metadata.append(dict(structure=structure,optimizer=optimizer,rate=rate,rule=rule))
                    left.append([tree.children[n][0] for n in range(8,15)])
                    right.append([tree.children[n][1] for n in range(8,15)])
                    pp=np.eye(6) if rule=='exact' else np.ones((6,6))/6 if rule in ('global_projection','broadcast') else p
                    if rule=='shuffled_projection':pp=p[np.ix_(shuffle,shuffle)]
                    projectors.append(pp)
    return metadata,np.asarray(left),np.asarray(right),np.stack(projectors)


def forward(x,weights,left,right):
    count=len(weights);ix=np.arange(count)
    value=np.zeros((count,15,len(x)));value[:,:8]=x.T[None]
    for k in range(7):
        l=value[ix,left[:,k]];r=value[ix,right[:,k]]
        a,b,c,d=weights[:,k,:].T
        value[:,k+8]=a[:,None]+b[:,None]*l+c[:,None]*r+d[:,None]*l*r
    return value


def gradient(x,y,weights,left,right,projectors,metadata,variance):
    values=forward(x,weights,left,right);count=len(weights);ix=np.arange(count)
    q=np.zeros_like(values);q[:,14]=1.
    features=np.zeros((count,7,len(x),4))
    for k in range(6,-1,-1):
        l=values[ix,left[:,k]];r=values[ix,right[:,k]]
        a,b,c,d=weights[:,k,:].T
        q[ix,left[:,k]]=q[:,k+8]*(b[:,None]+d[:,None]*r)
        q[ix,right[:,k]]=q[:,k+8]*(c[:,None]+d[:,None]*l)
        features[:,k]=np.stack([np.ones_like(l),l,r,l*r],axis=-1)
    routed=q[:,8:15].copy()
    routed[:,:6]=np.einsum('cij,cjb->cib',projectors,q[:,8:14])
    for i,meta in enumerate(metadata):
        if meta['rule']=='broadcast':routed[i,:6]=1.
    residual=(values[:,14]-y[None])/variance
    exact=np.einsum('cb,cjb,cjbf->cjf',residual,q[:,8:15],features)/len(x)
    delivered=np.einsum('cb,cjb,cjbf->cjf',residual,routed,features)/len(x)
    return delivered,exact,values[:,14],q[:,8:15],routed


def dataset(seed,family,kind,coeff,n):
    offset={'training':11,'test':23}[kind]
    rng=np.random.default_rng(np.random.SeedSequence([seed,FAMILIES.index(family),offset]))
    x=2.*rng.integers(2,size=(n,8))-1.
    y=fourier_design(x)@coeff+.15*rng.normal(size=n)
    return x,y


def train_task(seed,family,config):
    coeff=make_task(seed,family)
    compatible,bound,score=tree_from_coeff(coeff,'compatible')
    assert bound==0.
    trees=[compatible,shuffled_tree(compatible,seed)]
    structural_bounds=[cut_scores(coeff,tree)['centered_cut_bound'] for tree in trees]
    metadata,left,right,projectors=pack(trees,seed)
    count=len(metadata)
    init=np.random.default_rng(np.random.SeedSequence([seed,FAMILIES.index(family),41])).normal(0,.5,(7,4))
    init[:,0]=0.
    weights=np.broadcast_to(init,(count,7,4)).copy()
    x,y=dataset(seed,family,'training',coeff,config['training_samples'])
    tx,ty=dataset(seed,family,'test',coeff,config['test_samples'])
    px=domain();py=fourier_design(px)@coeff
    variance=float(coeff@coeff)
    rng=np.random.default_rng(np.random.SeedSequence([seed,FAMILIES.index(family),37]))
    m=np.zeros_like(weights);v=np.zeros_like(weights)
    rates=np.array([z['rate'] for z in metadata])[:,None,None]
    adam=np.array([z['optimizer']=='adam' for z in metadata])
    rows=[];start=time.perf_counter();clip_counts=np.zeros(count,int);projection_counts=np.zeros(count,int)
    def evaluate(step):
        pred=forward(tx,weights,left,right)[:,14]
        pop=forward(px,weights,left,right)[:,14]
        nmse=np.mean((pred-ty[None])**2,axis=1)/variance
        pn=np.mean((pop-py[None])**2,axis=1)/variance
        delivered,exact,_,q,routed=gradient(px,py,weights,left,right,projectors,metadata,variance)
        dot=np.sum(delivered*exact,axis=(1,2));norm=np.linalg.norm(delivered,axis=(1,2))*np.linalg.norm(exact,axis=(1,2))
        capture=np.sum((q-routed)**2,axis=(1,2))/np.maximum(np.sum(q*q,axis=(1,2)),1e-30)
        assert np.isfinite(nmse).all() and np.isfinite(pn).all(),'All failures must be retained and investigated'
        for i,meta in enumerate(metadata):
            tree=trees[0 if meta['structure']=='compatible' else 1]
            rows.append(dict(seed=seed,family=family,**meta,step=step,test_nmse=float(nmse[i]),population_nmse=float(pn[i]),
                variance=variance,gradient_cosine=float(dot[i]/max(norm[i],1e-30)),field_relative_squared_error=float(capture[i]),
                gradient_clipped_steps=int(clip_counts[i]),parameter_projected_steps=int(projection_counts[i]),
                parameter_norm=float(np.linalg.norm(weights[i])),max_abs_parameter=float(abs(weights[i]).max()),
                centered_cut_bound=structural_bounds[0 if meta['structure']=='compatible' else 1],parameters=28,edges=14,depth=score[1],
                elapsed_seconds=time.perf_counter()-start))
    evaluate(0)
    for step in range(1,config['steps']+1):
        idx=rng.integers(len(x),size=config['batch_size'])
        g,_,_,_,_=gradient(x[idx],y[idx],weights,left,right,projectors,metadata,variance)
        norms=np.linalg.norm(g,axis=(1,2));clip_counts += norms>config['gradient_norm_clip']
        g *= np.minimum(1,config['gradient_norm_clip']/np.maximum(norms,1e-30))[:,None,None]
        m=.9*m+.1*g;v=.999*v+.001*g*g
        update=g.copy()
        update[adam]=(m[adam]/(1-.9**step))/(np.sqrt(v[adam]/(1-.999**step))+1e-8)
        weights-=rates*update
        projection_counts += np.any(abs(weights)>config['parameter_bound'],axis=(1,2))
        weights=np.clip(weights,-config['parameter_bound'],config['parameter_bound'])
        if step in config['checkpoints']:evaluate(step)
    return rows


def run(split,seed):
    config=freeze()
    assert seed in config['development_seeds' if split=='development' else 'fresh_seeds']
    if split=='fresh':
        selection=json.loads((OUT/'selection_freeze.json').read_text())
        assert selection['fit_sha256']==sha(OUT/'development_fit.json')
        assert selection['protocol_sha256']==sha(OUT/'protocol.json')
    dest=OUT/'runs'/split;dest.mkdir(parents=True,exist_ok=True)
    if (dest/f'seed_{seed}_audit.json').exists():raise FileExistsError('Completed seed immutable')
    rows=[];start=time.perf_counter()
    for family in FAMILIES:
        rows.extend(train_task(seed,family,config))
        print(split,seed,family,'done',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(dest/f'seed_{seed}.csv',index=False)
    dump(dest/f'seed_{seed}_audit.json',dict(seed=seed,split=split,rows=len(frame),seconds=time.perf_counter()-start,
        protocol_sha256=sha(OUT/'protocol.json'),runner_sha256=sha(HERE/'experiment.py'),
        selection_sha256=sha(OUT/'selection_freeze.json') if split=='fresh' else None,all_outcomes_retained=True))


def collect(split):
    cfg=freeze();seeds=cfg['development_seeds' if split=='development' else 'fresh_seeds']
    return pd.concat([pd.read_csv(OUT/'runs'/split/f'seed_{seed}.csv') for seed in seeds],ignore_index=True)


def select():
    df=collect('development');end=df[df.step==1024]
    means=end.groupby(['optimizer','rule','rate']).population_nmse.mean()
    choices={}
    for optimizer in ('sgd','adam'):
        choices[optimizer]={rule:float(means.loc[optimizer,rule].idxmin()) for rule in RULES}
        choices[optimizer]['common_rate']=float(end[end.optimizer==optimizer].groupby('rate').population_nmse.mean().idxmin())
    dump(OUT/'development_fit.json',choices)
    dump(OUT/'selection_freeze.json',dict(fit_sha256=sha(OUT/'development_fit.json'),protocol_sha256=sha(OUT/'protocol.json'),
        utc=pd.Timestamp.now(tz='UTC').isoformat(),source_files={str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'runs/development').glob('*.csv'))}))
    print(json.dumps(choices,indent=2));print(means.to_string())


def summarize(split):
    df=collect(split);dest=OUT/'summaries'/split;dest.mkdir(parents=True,exist_ok=True)
    choices=json.loads((OUT/'development_fit.json').read_text())
    df['selected_rate']=[choices[o][r] for o,r in zip(df.optimizer,df.rule)]
    selected=df[np.isclose(df.rate,df.selected_rate)].copy()
    selected.to_csv(dest/'selected_trajectories.csv',index=False)
    end=selected[selected.step==1024]
    end.to_csv(dest/'selected_endpoints.csv',index=False)
    end.groupby(['family','structure','optimizer','rule'],as_index=False).agg(mean_test_nmse=('test_nmse','mean'),
        mean_population_nmse=('population_nmse','mean'),n_seeds=('seed','nunique'),mean_gradient_cosine=('gradient_cosine','mean')).to_csv(dest/'summary.csv',index=False)
    contrasts=[]
    rng=np.random.default_rng(127999)
    for family in (*FAMILIES,'all'):
        z=end if family=='all' else end[end.family==family]
        for optimizer in ('sgd','adam'):
            for structure in ('compatible','assignment_shuffled'):
                part=z[(z.optimizer==optimizer)&(z.structure==structure)]
                wide=part.groupby(['seed','rule']).test_nmse.mean().unstack()
                for baseline in RULES[1:]:
                    values=(wide[baseline]-wide['exact']).to_numpy()
                    boot=values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1)
                    contrasts.append(dict(family=family,optimizer=optimizer,structure=structure,contrast=baseline+' minus exact',
                        mean=float(values.mean()),ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),
                        n_seeds=len(values),positive_seeds=int(np.sum(values>0))))
            for rule in RULES:
                part=z[(z.optimizer==optimizer)&(z.rule==rule)]
                wide=part.groupby(['seed','structure']).test_nmse.mean().unstack()
                values=(wide.assignment_shuffled-wide.compatible).to_numpy()
                boot=values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1)
                contrasts.append(dict(family=family,optimizer=optimizer,structure='paired',contrast='shuffled minus compatible: '+rule,
                    mean=float(values.mean()),ci95_low=float(np.quantile(boot,.025)),ci95_high=float(np.quantile(boot,.975)),
                    n_seeds=len(values),positive_seeds=int(np.sum(values>0))))
    pd.DataFrame(contrasts).to_csv(dest/'paired_contrasts.csv',index=False)
    df[df.step==1024].groupby(['family','structure','optimizer','rule','rate'],as_index=False).test_nmse.mean().to_csv(dest/'all_rate_sensitivity.csv',index=False)
    print(end.groupby(['structure','optimizer','rule']).test_nmse.mean().to_string())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','select','summarize'])
    parser.add_argument('--split',choices=['development','fresh'],default='development');parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run(args.split,args.seed)
    elif args.action=='select':select()
    else:summarize(args.split)
