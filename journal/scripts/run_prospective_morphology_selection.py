#!/usr/bin/env python3
"""Prospective costed selection in a differentiable tree-constrained linear learner.

This is an imposed context-decoding task, not a conductance DendriNet or a
natural somatic readout. Calibration, development training, sealed selection,
and confirmatory training are separate, hash-gated stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/prospective_morphology_selection/protocol.json'
OUT = ROOT / 'source_data/prospective_morphology_selection'
SELF = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_): return bool(value)
    return value


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2, sort_keys=True, allow_nan=False)+'\n')


def load():
    return json.loads(CONFIG.read_text())


def haar(n=8):
    cols = [np.ones(n)/np.sqrt(n)]
    width = n
    while width > 1:
        for start in range(0, n, width):
            q = np.zeros(n); q[start:start+width//2] = 1/np.sqrt(width)
            q[start+width//2:start+width] = -1/np.sqrt(width); cols.append(q)
        width //= 2
    return np.column_stack(cols)


def tree(permutation, shape):
    """Explicit connected 15-node tree: leaves 0..7, binary internal nodes 8..14."""
    children = {}; descendants = {i:[i] for i in range(8)}
    def join(leaves):
        if len(leaves)==1: return int(leaves[0])
        cut = len(leaves)//2 if shape=='balanced' else 1
        left, right = join(leaves[:cut]), join(leaves[cut:])
        node = 8+len(children); children[node]=(left,right)
        descendants[node]=descendants[left]+descendants[right]
        return node
    root=join(list(permutation)); positions={i:np.array([float(i),0.]) for i in range(8)}
    edges=[]
    for node, kids in children.items():
        positions[node]=np.array([np.mean(descendants[node]),max(positions[k][1] for k in kids)+1.])
        for kid in kids: edges.append((node,kid,float(np.linalg.norm(positions[node]-positions[kid]))))
    lap=np.zeros((15,15))
    for left,right,length in edges:
        # Every cable has positive conductance inverse to its geometric length.
        g=1/length; lap[left,left]+=g;lap[right,right]+=g;lap[left,right]-=g;lap[right,left]-=g
    return dict(root=root,children=children,descendants=descendants,edges=edges,
                positions=positions,laplacian=lap,cable=sum(e[2] for e in edges))


def candidates(cfg):
    specs=cfg['candidate_trees']; records=[]
    maximum_cable=max(tree(s['permutation'],s['shape'])['cable'] for s in specs)
    for spec in specs:
        t=tree(spec['permutation'],spec['shape'])
        transfer=np.linalg.inv(np.eye(15)+cfg['joint_kappa']*t['laplacian'])[:8,:8]
        for k in cfg['route_budgets']:
            frontier=[t['root']]
            while len(frontier)<k:
                node=max((v for v in frontier if v in t['children']),key=lambda v:(len(t['descendants'][v]),-v))
                at=frontier.index(node);frontier[at:at+1]=t['children'][node]
            dictionary=np.zeros((8,k))
            for col,node in enumerate(frontier):
                leaves=t['descendants'][node];dictionary[leaves,col]=1/np.sqrt(len(leaves))
            projector=dictionary@dictionary.T
            records.append(dict(candidate_id=f"{spec['name']}_k{k}",tree_id=spec['name'],shape=spec['shape'],
                permutation=spec['permutation'],budget_k=k,channel_cost=k/8,
                cable_cost=t['cable']/maximum_cable,cable_length=t['cable'],
                resource_cost=k/8+cfg['cable_weight']*t['cable']/maximum_cable,
                parameter_count=64,decoder_coefficients=64,encoder_coefficients=8*k,dictionary=dictionary,projector=projector,
                transfer=transfer,edges=t['edges'],positions={str(i):x.tolist() for i,x in t['positions'].items()},
                dictionary_nonzeros=int(np.count_nonzero(dictionary))))
    return records


def tasks(cfg, split, seed):
    for rank in cfg['ranks']:
        for noise in cfg['noise_sds']:
            for angle in cfg['angles_pi'][split]:
                yield dict(task_id=f's{seed}_r{rank}_n{round(noise*1000):03d}_a{round(angle*1000):03d}',
                    split=split,seed=seed,rank=rank,noise_sd=noise,angle_pi=angle)


def task_basis(task):
    rng=np.random.default_rng(task['seed']+701_000)
    axes,_=np.linalg.qr(rng.normal(size=(8,8)))
    generator=np.zeros((8,8))
    for i in range(0,8,2): generator[i,i+1]=-1;generator[i+1,i]=1
    rotation=expm(np.pi*task['angle_pi']*(axes@generator@axes.T))
    basis=rotation@haar()
    teacher,_=np.linalg.qr(np.random.default_rng(task['seed']+702_000).normal(size=(8,8)))
    return basis, teacher


def dataset(task, kind, n):
    # Coupled x/context/noise draws across angle/noise conditions; independent
    # data kinds and task seeds. Label noise is rescaled, never resampled by angle.
    kind_offset={'calibration':11,'training':23,'test':37}[kind]
    rng=np.random.default_rng(np.random.SeedSequence([task['seed'],task['rank'],kind_offset]))
    x=rng.normal(size=(n,8));context=rng.integers(task['rank'],size=n);eps=rng.normal(size=n)
    basis,teacher=task_basis(task);a=basis[:,context].T
    y=np.einsum('bi,ij,bj->b',a,teacher,x)+task['noise_sd']*eps
    return x,a,y


def moment_score(x,a,y,h,p,eta,batch_size):
    # At W=0, exact per-trial gradient = -y (H^T a) x^T;
    # actual routed gradient = -y P(H^T a) x^T.
    b=a@h; routed=b@p
    per_exact=-y[:,None,None]*b[:,:,None]*x[:,None,:]
    per_routed=-y[:,None,None]*routed[:,:,None]*x[:,None,:]
    exact=per_exact.mean(axis=0);mean=per_routed.mean(axis=0)
    mean_square=float(np.sum(mean*mean))
    single_second=float(np.mean(np.sum(per_routed**2,axis=(1,2))))
    variance=max(single_second-mean_square,0.)/batch_size
    design=(b[:,:,None]*x[:,None,:]).reshape(len(x),64)
    curvature=float(np.linalg.eigvalsh(design.T@design/len(x))[-1])
    alignment=float(np.sum(exact*mean));second=mean_square+variance
    score=eta*alignment-.5*eta**2*curvature*second
    optimum=max(alignment,0)**2/(2*curvature*second) if second>0 else 0.
    return dict(utility_actual_step=score,utility_optimized_step=optimum,
        gradient_alignment=alignment,update_mean_square=mean_square,
        minibatch_covariance_trace=variance,single_trial_second_moment=single_second,
        actual_update_second_moment=second,global_calibration_curvature=curvature,
        initial_calibration_loss=float(np.mean(y*y)/2)),mean,per_routed


def metadata(task,arm,candidate):
    return {**task,'arm':arm,**{k:candidate[k] for k in ['candidate_id','tree_id','shape','budget_k','resource_cost','channel_cost','cable_cost','parameter_count','decoder_coefficients','encoder_coefficients']}}


def check_freeze():
    record=json.loads((OUT/'protocol_freeze.json').read_text())
    assert record['protocol_sha256']==sha(CONFIG),'Protocol changed after freeze'
    assert record['runner_sha256']==sha(SELF),'Runner changed after freeze'
    return record


def freeze():
    cfg=load();OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/'protocol_freeze.json'
    if target.exists(): check_freeze();return
    manifest=candidates(cfg)
    write_json(OUT/'candidate_manifest.json',manifest)
    write_json(OUT/'protocol.json',cfg)
    write_json(target,dict(protocol_sha256=sha(CONFIG),runner_sha256=sha(SELF),
        candidate_manifest_sha256=sha(OUT/'candidate_manifest.json'),
        utc=pd.Timestamp.now(tz='UTC').isoformat(),python=platform.python_version(),numpy=np.__version__,
        status='frozen_before_development_or_confirmatory_training'))


def calibrate(split,seed):
    check_freeze();cfg=load();rows=[];means={}
    for task in tasks(cfg,split,seed):
        x,a,y=dataset(task,'calibration',cfg['n_calibration'])
        for arm in cfg['arms']:
            for c in candidates(cfg):
                h=np.eye(8) if arm=='feedback_only' else c['transfer']
                scores,mean,_=moment_score(x,a,y,h,c['projector'],cfg['learning_rate'],cfg['batch_size'])
                q,_=task_basis(task);sigma=q[:,:task['rank']]@q[:,:task['rank']].T/task['rank']
                scores.update(reference_credit_effective_rank=task['rank'],
                    reference_capture=float(np.trace(c['projector']@sigma)),
                    transported_credit_effective_rank=float(np.trace(h@sigma@h)**2/np.trace((h@sigma@h)@(h@sigma@h))))
                rows.append({**metadata(task,arm,c),**scores})
                means[f"{task['task_id']}__{arm}__{c['candidate_id']}"]=mean
    dest=OUT/'calibration'/split;dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/f'seed_{seed}.csv',index=False)
    np.savez_compressed(dest/f'seed_{seed}_update_means.npz',**means)
    print('calibrated',split,seed,len(rows),flush=True)


def train_task(task,arm,cands,cfg):
    x,a,y=dataset(task,'training',cfg['n_training']);tx,ta,ty=dataset(task,'test',cfg['n_test'])
    hs=np.stack([np.eye(8) if arm=='feedback_only' else c['transfer'] for c in cands])
    ps=np.stack([c['projector'] for c in cands]);weights=np.zeros((len(cands),8,8))
    rng=np.random.default_rng(np.random.SeedSequence([task['seed'],task['rank'],53]))
    rows=[];eta=cfg['learning_rate'];bsize=cfg['batch_size']
    for step in range(1,cfg['training_steps']+1):
        idx=rng.integers(len(x),size=bsize);xb=x[idx];ab=a[idx];yb=y[idx]
        b=np.einsum('bi,cij->cbj',ab,hs,optimize=True)
        residual=np.einsum('cbi,cij,bj->cb',b,weights,xb,optimize=True)-yb[None,:]
        raw=np.einsum('cb,cbi,bj->cij',residual,b,xb,optimize=True)/bsize
        update=np.einsum('cij,cjk->cik',ps,raw,optimize=True)
        weights-=eta*update
        if step in cfg['checkpoints']:
            # Evaluation does not affect updates, stopping or model selection.
            bt=np.einsum('bi,cij->cbj',ta,hs,optimize=True)
            prediction=np.einsum('cbi,cij,bj->cb',bt,weights,tx,optimize=True)
            losses=np.mean((prediction-ty[None,:])**2,axis=1)/2
            assert np.isfinite(losses).all(),'Nonfinite outcomes retained as failure; do not silently exclude'
            for c,loss,w in zip(cands,losses,weights):
                rows.append({**metadata(task,arm,c),'checkpoint':step,'test_loss':float(loss),
                    'penalized_test_loss':float(loss+cfg['resource_penalty']*c['resource_cost']),
                    'weight_norm':float(np.linalg.norm(w))})
    return rows


def train(split,seed):
    check_freeze();cfg=load()
    selection_sha=None
    if split=='confirmatory':
        seal=json.loads((OUT/'selection_freeze.json').read_text())
        assert seal['protocol_sha256']==sha(CONFIG) and seal['runner_sha256']==sha(SELF)
        assert seal['selection_sha256']==sha(OUT/'sealed_confirmatory_selections.csv')
        selection_sha=seal['selection_sha256']
        for item in seal['calibration_files']:assert sha(OUT/item['path'])==item['sha256']
    start=time.monotonic();rows=[]
    for task in tasks(cfg,split,seed):
        for arm in cfg['arms']: rows.extend(train_task(task,arm,candidates(cfg),cfg))
    dest=OUT/'runs'/split;dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/f'seed_{seed}.csv',index=False)
    write_json(dest/f'seed_{seed}_audit.json',dict(seed=seed,split=split,n_rows=len(rows),
        runtime_seconds=time.monotonic()-start,selection_sha256=selection_sha,
        protocol_sha256=sha(CONFIG),runner_sha256=sha(SELF),
        utc=pd.Timestamp.now(tz='UTC').isoformat(),all_outcomes_retained=True))
    print('trained',split,seed,len(rows),'seconds',time.monotonic()-start,flush=True)


def collect(folder,split,seeds):
    files=[OUT/folder/split/f'seed_{s}.csv' for s in seeds]
    assert all(p.exists() for p in files),[str(p) for p in files if not p.exists()]
    return pd.concat([pd.read_csv(p) for p in files],ignore_index=True),files


def select():
    check_freeze();cfg=load()
    assert not list((OUT/'runs/confirmatory').glob('*.csv')),'Cannot seal after confirmatory outcomes exist'
    dev,devfiles=collect('runs','development',cfg['seeds']['development'])
    dev=dev[dev.checkpoint.eq(cfg['training_steps'])]
    devs,_=collect('calibration','development',cfg['seeds']['development'])
    tests,testfiles=collect('calibration','confirmatory',cfg['seeds']['confirmatory'])
    joined=dev.merge(devs[['task_id','arm','candidate_id','utility_actual_step']],on=['task_id','arm','candidate_id'],validate='one_to_one')
    fit={};rows=[]
    for arm in cfg['arms']:
        z=joined[joined.arm.eq(arm)].copy()
        # One nonnegative scale per arm; within-task centering removes nuisance loss levels.
        u=z.utility_actual_step-z.groupby('task_id').utility_actual_step.transform('mean')
        loss=z.test_loss-z.groupby('task_id').test_loss.transform('mean')
        scale=max(0.,min(cfg['training_steps'],float(-np.sum(u*loss)/np.sum(u*u))))
        fixed=z.groupby('candidate_id').penalized_test_loss.mean().sort_values(kind='stable').index[0]
        maximum=z[z.budget_k.eq(max(cfg['route_budgets']))].groupby('candidate_id').penalized_test_loss.mean().sort_values(kind='stable').index[0]
        rank_only={str(r):g.groupby('candidate_id').penalized_test_loss.mean().sort_values(kind='stable').index[0] for r,g in z.groupby('rank')}
        fit[arm]=dict(utility_to_loss_scale=scale,development_best=fixed,max_budget=maximum,rank_only=rank_only)
        for taskid,g in tests[tests.arm.eq(arm)].groupby('task_id',sort=True):
            g=g.sort_values('candidate_id',kind='stable').copy()
            g['predicted_penalized_loss']=g.initial_calibration_loss-scale*g.utility_actual_step+cfg['resource_penalty']*g.resource_cost
            choices={'moment_selector':g.sort_values('predicted_penalized_loss',kind='stable').iloc[0].candidate_id,
                'development_best':fixed,'maximum_budget':maximum,'rank_only':rank_only[str(int(g.iloc[0]['rank']))]}
            rng=np.random.default_rng(int(g.iloc[0].seed)+int(g.iloc[0]['rank'])*997+round(g.iloc[0].angle_pi*1000)+round(g.iloc[0].noise_sd*1000))
            choices['sampled_random']=g.iloc[rng.integers(len(g))].candidate_id
            for policy,cid in choices.items():
                row=g[g.candidate_id.eq(cid)].iloc[0].to_dict();rows.append({**row,'policy':policy,'utility_to_loss_scale':scale})
    selected=pd.DataFrame(rows);path=OUT/'sealed_confirmatory_selections.csv';selected.to_csv(path,index=False)
    write_json(OUT/'development_fit.json',fit)
    write_json(OUT/'selection_freeze.json',dict(protocol_sha256=sha(CONFIG),runner_sha256=sha(SELF),
        selection_sha256=sha(path),development_fit_sha256=sha(OUT/'development_fit.json'),
        development_files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in devfiles],
        calibration_files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in testfiles],
        utc=pd.Timestamp.now(tz='UTC').isoformat(),n_tasks=selected.task_id.nunique(),n_selections=len(selected),
        status='sealed_before_any_confirmatory_candidate_training'))
    print(json.dumps(fit,indent=2),flush=True)


def bootstrap(values,seed=8311,draws=10000):
    v=np.asarray(values,float);rng=np.random.default_rng(seed)
    m=v[rng.integers(len(v),size=(draws,len(v)))].mean(axis=1)
    return float(v.mean()),float(np.quantile(m,.025)),float(np.quantile(m,.975))


def analyze():
    cfg=load();check_freeze();out,files=collect('runs','confirmatory',cfg['seeds']['confirmatory'])
    last=out[out.checkpoint.eq(cfg['training_steps'])].copy()
    selected=pd.read_csv(OUT/'sealed_confirmatory_selections.csv')
    seal=json.loads((OUT/'selection_freeze.json').read_text());assert seal['selection_sha256']==sha(OUT/'sealed_confirmatory_selections.csv')
    assert len(last)==len(cfg['seeds']['confirmatory'])*len(cfg['angles_pi']['confirmatory'])*len(cfg['noise_sds'])*len(cfg['ranks'])*len(cfg['arms'])*len(candidates(cfg))
    key=['task_id','arm'];best=last.sort_values(['penalized_test_loss','candidate_id'],kind='stable').groupby(key).first().reset_index()
    best=best[key+['candidate_id','budget_k','tree_id','penalized_test_loss']].rename(columns={'candidate_id':'oracle_candidate_id','budget_k':'oracle_budget_k','tree_id':'oracle_tree_id','penalized_test_loss':'oracle_penalized_loss'})
    scored=selected.merge(last[key+['candidate_id','test_loss','penalized_test_loss']],on=key+['candidate_id'],validate='many_to_one').merge(best,on=key,validate='many_to_one')
    scored['regret']=scored.penalized_test_loss-scored.oracle_penalized_loss
    scored['exact_candidate_match']=scored.candidate_id.eq(scored.oracle_candidate_id)
    scored['budget_match']=scored.budget_k.eq(scored.oracle_budget_k)
    scored['tree_match']=scored.tree_id.eq(scored.oracle_tree_id)
    # The uniform random policy's expectation is evaluated over all candidates,
    # avoiding incidental Monte Carlo noise from drawing one candidate per task.
    random=last.groupby(key+['seed','rank','noise_sd','angle_pi'],as_index=False)[['test_loss','penalized_test_loss']].mean().merge(best,on=key)
    random['policy']='uniform_random_expectation';random['regret']=random.penalized_test_loss-random.oracle_penalized_loss
    scored=pd.concat([scored,random],ignore_index=True)
    scored.to_csv(OUT/'policy_outcomes.csv',index=False);last.to_csv(OUT/'candidate_outcomes.csv',index=False)
    best.to_csv(OUT/'retrospective_oracle.csv',index=False)
    summaries=[];paired=[]
    for arm,g in scored.groupby('arm'):
        for policy,v in g.groupby('policy'):
            byseed=v.groupby('seed').regret.mean();mean,lo,hi=bootstrap(byseed)
            summaries.append(dict(arm=arm,policy=policy,n_seeds=len(byseed),n_tasks=len(v),mean_regret=mean,ci95_low=lo,ci95_high=hi,mean_test_loss=v.test_loss.mean(),mean_penalized_loss=v.penalized_test_loss.mean()))
        pivot=g.pivot(index=['seed','task_id'],columns='policy',values='regret')
        for policy in pivot.columns:
            if policy=='moment_selector':continue
            diff=(pivot[policy]-pivot.moment_selector).groupby('seed').mean()
            mean,lo,hi=bootstrap(diff)
            paired.append(dict(arm=arm,comparator=policy,n_seeds=len(diff),mean_regret_reduction=mean,ci95_low=lo,ci95_high=hi,positive_seeds=int((diff>0).sum())))
    pd.DataFrame(summaries).to_csv(OUT/'policy_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_regret_summary.csv',index=False)
    seedrows=scored.groupby(['arm','policy','seed'],as_index=False)[['regret','test_loss','penalized_test_loss']].mean()
    seedrows.to_csv(OUT/'policy_seed_means.csv',index=False)
    grouped=scored.groupby(['arm','policy','rank','noise_sd','angle_pi'],as_index=False)[['regret','test_loss','penalized_test_loss']].mean()
    grouped.to_csv(OUT/'grouped_policy_summary.csv',index=False)
    correlations=[]
    cal,_=collect('calibration','confirmatory',cfg['seeds']['confirmatory'])
    merged=last.merge(cal[['task_id','arm','candidate_id','utility_actual_step','reference_capture','minibatch_covariance_trace']],on=['task_id','arm','candidate_id'])
    for (taskid,arm),g in merged.groupby(['task_id','arm']):
        correlations.append(dict(task_id=taskid,arm=arm,seed=int(g.iloc[0].seed),rank=int(g.iloc[0]['rank']),noise_sd=g.iloc[0].noise_sd,angle_pi=g.iloc[0].angle_pi,
            utility_vs_negative_loss_rho=float(spearmanr(g.utility_actual_step,-g.test_loss).statistic)))
    pd.DataFrame(correlations).to_csv(OUT/'within_task_correlations.csv',index=False)
    write_json(OUT/'report.json',dict(status='complete_prospective_candidate_selection',n_independent_task_seeds=len(cfg['seeds']['confirmatory']),n_confirmatory_tasks=last.task_id.nunique(),n_candidate_fits=len(last),
        selection_freeze_sha256=sha(OUT/'selection_freeze.json'),protocol_sha256=sha(CONFIG),runner_sha256=sha(SELF),
        seed_block_bootstrap_draws=10000,policy_summary=summaries,paired_regret_summary=paired,
        scope='Tree-constrained linear learning with imposed context decoder; fixed analytic reference credit spectra. Joint forward transfer can change parameter-gradient spectra. Finite candidate grid and declared proxy costs; not a biophysical morphology law.',
        run_files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in files]))
    print(json.dumps(clean(summaries),indent=2),flush=True)


def invariants():
    cfg=load();cs=candidates(cfg);worst={}
    for c in cs:
        p=c['projector'];d=c['dictionary'];h=c['transfer']
        assert np.allclose(p@p,p) and np.allclose(d.T@d,np.eye(c['budget_k']))
        assert np.linalg.matrix_rank(p)==c['budget_k'];assert len(c['edges'])==14
        assert np.all(np.linalg.eigvalsh(h)>0)
    for spec in cfg['candidate_trees']:
        t=tree(spec['permutation'],spec['shape']);lap=t['laplacian']
        assert np.allclose(lap.sum(axis=1),0);assert np.linalg.matrix_rank(lap)==14
        assert np.allclose(np.linalg.inv(np.eye(15)+0*lap)[:8,:8],np.eye(8))
        a=np.eye(15)+cfg['joint_kappa']*lap
        schur=a[:8,:8]-a[:8,8:]@np.linalg.solve(a[8:,8:],a[8:,:8])
        assert np.allclose(np.linalg.inv(schur),np.linalg.inv(a)[:8,:8])
    for r in cfg['ranks']:
        spectra=[]
        for angle in cfg['angles_pi']['development']+cfg['angles_pi']['confirmatory']:
            q,_=task_basis(dict(seed=9123,rank=r,angle_pi=angle));assert np.allclose(q.T@q,np.eye(8))
            spectra.append(np.linalg.eigvalsh(q[:,:r]@q[:,:r].T/r))
        assert np.allclose(spectra,spectra[0]);worst[f'rank_{r}_spectrum_error']=float(np.max(np.abs(np.array(spectra)-spectra[0])))
    task=dict(seed=9123,rank=4,noise_sd=.75,angle_pi=.125)
    x,a,y=dataset(task,'calibration',256);c=cs[6];h=c['transfer'];p=c['projector']
    score,mean,per=moment_score(x,a,y,h,p,cfg['learning_rate'],cfg['batch_size'])
    rng=np.random.default_rng(55);samples=per[rng.integers(len(x),size=(20000,cfg['batch_size']))].mean(axis=1)
    observed=np.mean(np.sum((samples-mean)**2,axis=(1,2)))
    relative=abs(observed/score['minibatch_covariance_trace']-1);assert relative<.04
    exact=np.mean(-y[:,None,None]*(a@h)[:,:,None]*x[:,None,:],axis=0)
    def loss(w):return np.mean((np.einsum('bi,ij,bj->b',a@h,w,x)-y)**2)/2
    numerical=np.zeros((8,8));eps=1e-6
    for i in range(8):
        for k in range(8):
            w=np.zeros((8,8));w[i,k]=eps;numerical[i,k]=(loss(w)-loss(-w))/(2*eps)
    assert np.max(np.abs(exact-numerical))<1e-8
    # Full-batch actual decrease versus its exact global quadratic bound.
    eta=cfg['learning_rate'];observed_drop=loss(np.zeros((8,8)))-loss(-eta*mean)
    bound=eta*score['gradient_alignment']-.5*eta**2*score['global_calibration_curvature']*score['update_mean_square']
    assert observed_drop>=bound-1e-12
    write_json(OUT/'invariant_tests.json',dict(status='passed',n_candidates=len(cs),n_tree_nodes=15,
        kappa_zero_identity=True,schur_reduction_identity=True,projector_idempotence=True,
        fixed_reference_spectrum_errors=worst,minibatch_covariance_relative_error=relative,
        maximum_gradient_finite_difference_error=float(np.max(np.abs(exact-numerical))),
        fullbatch_decrease_minus_bound=observed_drop-bound))
    print('invariants passed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','invariants','calibrate','train','select','analyze'])
    parser.add_argument('--split',choices=['development','confirmatory'],default='development')
    parser.add_argument('--seed',type=int)
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    elif args.stage=='invariants':invariants()
    elif args.stage=='select':select()
    elif args.stage=='analyze':analyze()
    else:
        assert args.seed in load()['seeds'][args.split]
        (calibrate if args.stage=='calibrate' else train)(args.split,args.seed)

if __name__=='__main__':main()
