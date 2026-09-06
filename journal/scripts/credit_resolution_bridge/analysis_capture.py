#!/usr/bin/env python3
"""Post hoc spatial-credit audit, with faithful replay of unsaved checkpoints.

No original experiment/source data is changed. The frozen algebraic cohort did
not save parameter checkpoints. Selected compatible fits are replayed here and
all available deterministic archived metrics are checked before analysis.

q_j(x)=d output/d u_j; delta*q is the loss-derived per-unit credit. Six nonroot
internal units are the routing sites. Root sensitivity is always exactly one.
Projection capture measures retained squared field energy, not learning success.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
SCRIPTS = HERE.parent
OUT = JOURNAL/'source_data/credit_resolution_bridge/capture'
sys.path.insert(0, str(SCRIPTS/'morphology_credit'))
import experiment as credit


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


conductance = load_module('capture_conductance_model', SCRIPTS/'morphology_conductance/model.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def protocol():
    originals = [SCRIPTS/'morphology_credit/experiment.py', SCRIPTS/'morphology_structure/model.py',
                 SCRIPTS/'morphology_structure/constructive_dp_v2.py',
                 JOURNAL/'source_data/morphology_credit/protocol.json',
                 JOURNAL/'source_data/morphology_credit/development_fit.json',
                 SCRIPTS/'morphology_conductance/model.py']
    return dict(status='posthoc diagnostic; not a preregistered test or a new learning cohort',
        seed_blocks=list(range(127200,127220)), families=list(credit.FAMILIES),
        checkpoints=[0,1,16,64,256,1024], reference='selected compatible fits from frozen morphology_credit cohort',
        original_checkpoint_availability='morphology_credit has CSV metrics only; end_to_end and conductance save final weights',
        replay='Frozen training/data/random streams, selected rule-specific rates; every deterministic archived metric checked',
        sites='Exactly the six nonroot internal units 8..13, not eight leaves or seven units including soma',
        evaluation='Exact 256-pattern uniform input population with clean targets',
        q_definition='Per-unit output sensitivity, independent of target at fixed tree, weights and input',
        credit_definition='Clean loss residual divided by target variance times q',
        projection_capture='1 - squared Frobenius residual/squared field norm; no centering',
        per_example_capture='Mean per-pattern normalized capture; scalar residual cancels whenever nonzero',
        fixed_delivery='Unit and initial mean profiles have fixed coefficients; their fidelity may be negative and is not projection capture',
        rank_one='Best single spatial vector fitted over all evaluation patterns; separate unweighted and residual-weighted oracle bounds',
        rank='Eigenvalues of uncentered second moment; participation ratio and rank needed for95% energy',
        update='Local feature eligibilities multiply delivered per-unit credit; compare per-example update energy and population gradient',
        common_state='All candidate deliveries evaluated on each same reference checkpoint; differences at independently trained endpoints are descriptive',
        family_pairing='Matching and quartet have identical input-gradient covariance .25 I8 and the same balanced topology up to leaf assignment; original initialization/labels/permutations differ by family. New matched bridge is analyzed separately when available.',
        nested='Nested four-prefix target is separate; it is not input-isospectral with matching/quartet and uses depth4 compatible tree',
        interpretations=['Low unit broadcast fidelity can reflect spatial gain calibration',
                         'Best rank-one capture below1 indicates varying field directions, not necessary learning failure',
                         'Error-weighted capture can differ between targets at the same state solely through residual weighting',
                         'The field at a trained checkpoint depends on the training trajectory; endpoint associations are not causal task-only laws'],
        original_source_sha256={str(p.relative_to(JOURNAL)):sha(p) for p in originals})


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    record=protocol(); path=OUT/'analysis_protocol.json'
    if path.exists():
        assert json.loads(path.read_text())==record
    else: dump(path,record)
    dump(OUT/'analysis_code_hash.json', {'analysis_capture.py':sha(__file__)})


def algebraic_fields(weights, left, right, x):
    """Independent forward/reverse calculation, returned as model,example,site."""
    values=credit.forward(x,weights,left,right)
    count=len(weights); ix=np.arange(count)
    q=np.zeros_like(values);q[:,14]=1.
    features=np.zeros((count,len(x),7,4))
    for k in range(6,-1,-1):
        l=values[ix,left[:,k]];r=values[ix,right[:,k]]
        _,b,c,d=weights[:,k,:].T
        q[ix,left[:,k]]=q[:,k+8]*(b[:,None]+d[:,None]*r)
        q[ix,right[:,k]]=q[:,k+8]*(c[:,None]+d[:,None]*l)
        features[:,:,k]=np.stack([np.ones_like(l),l,r,l*r],axis=-1)
    return values[:,14],q[:,8:15].transpose(0,2,1),features


def ancestry_bases(tree):
    """Deterministic nested ancestry spans; global profile included at every K."""
    sites=list(range(8,14));n=len(sites)
    def depth(node):
        d=0
        while node!=tree.root:node=tree.parent[node][0];d+=1
        return d
    def below(node):
        result={node} if node in sites else set()
        if node in tree.children:
            for child in tree.children[node]:result|=below(child)
        return result
    cols=[np.ones(n)]
    bases={1:np.column_stack(cols)}
    for node in sorted(sites,key=lambda z:(depth(z),z)):
        col=np.array([int(i in below(node)) for i in sites],float)
        trial=np.column_stack(cols+[col])
        if np.linalg.matrix_rank(trial)>len(cols):
            cols.append(col);bases[len(cols)]=trial
    assert len(cols)==n
    return bases


def projector(basis):
    return basis@np.linalg.pinv(basis)


def spectrum(field):
    ev=np.maximum(np.linalg.eigvalsh(field.T@field/len(field))[::-1],0)
    total=float(ev.sum())
    if total<=1e-30:return ev,dict(total_energy=total,effective_rank=np.nan,rank95=0,rank99=0,leading_fraction=np.nan)
    frac=ev/total
    return ev,dict(total_energy=total,effective_rank=float(1/(frac@frac)),
        rank95=int(np.searchsorted(np.cumsum(frac),.95)+1),
        rank99=int(np.searchsorted(np.cumsum(frac),.99)+1),leading_fraction=float(frac[0]))


def best_profile(field):
    _,v=np.linalg.eigh(field.T@field)
    p=v[:,-1]
    if p.sum()<0:p=-p
    return p


def field_metrics(q, approximation, residual, features):
    eps=1e-30
    norm=np.sum(q*q,axis=1);diff=np.sum((q-approximation)**2,axis=1)
    valid=norm>eps
    per=1-diff[valid]/norm[valid]
    weighted_norm=norm*residual**2
    credit=q*residual[:,None];routed=approximation*residual[:,None]
    # Routing excludes root, but its exact gradient is retained for full-update comparisons.
    exact=credit[:,:,None]*features[:,:6]
    delivered=routed[:,:,None]*features[:,:6]
    eg=exact.mean(axis=0).ravel();dg=delivered.mean(axis=0).ravel()
    root=(residual[:,None]*features[:,6]).mean(axis=0)
    eg_full=np.r_[eg,root];dg_full=np.r_[dg,root]
    def cosine(a,b):
        denom=np.linalg.norm(a)*np.linalg.norm(b)
        return float(a@b/denom) if denom>eps else np.nan
    return dict(field_energy_fidelity=float(1-diff.sum()/max(norm.sum(),eps)),
        mean_per_example_fidelity=float(per.mean()),
        error_weighted_energy_fidelity=float(1-np.sum(diff*residual**2)/max(weighted_norm.sum(),eps)),
        nonzero_field_examples=int(valid.sum()),
        eligibility_weighted_update_fidelity=float(1-np.sum((exact-delivered)**2)/max(np.sum(exact**2),eps)),
        population_gradient_cosine_nonroot=cosine(eg,dg),
        population_gradient_cosine_full=cosine(eg_full,dg_full),
        population_gradient_relative_error_nonroot=float(np.linalg.norm(eg-dg)/max(np.linalg.norm(eg),eps)),
        population_gradient_norm_nonroot=float(np.linalg.norm(eg)))


def analyze_algebraic_state(weights, tree, coeff, initial, metadata, initial_profile=None):
    x=credit.domain(); y=credit.fourier_design(x)@coeff;variance=float(coeff@coeff)
    left=np.array([[tree.children[n][0] for n in range(8,15)]]);right=np.array([[tree.children[n][1] for n in range(8,15)]])
    output,paths,features=algebraic_fields(weights[None],left,right,x)
    q=paths[0,:,:6]; feat=features[0];residual=(output[0]-y)/variance
    _,initial_q,_=algebraic_fields(initial[None],left,right,x)
    mean_profile=initial_q[0,:,:6].mean(axis=0) if initial_profile is None else np.asarray(initial_profile)
    rank1=best_profile(q);weighted_rank1=best_profile(q*residual[:,None])
    approximations={'unit_broadcast_fixed':(np.ones_like(q),1,False),
        'initial_mean_fixed':(np.broadcast_to(mean_profile,q.shape),1,False),
        'initial_mean_projection':(q@projector(mean_profile[:,None]),1,True),
        'best_fixed_rank1_q':(q@projector(rank1[:,None]),1,True),
        'best_fixed_rank1_error':(q@projector(weighted_rank1[:,None]),1,True)}
    for k,basis in ancestry_bases(tree).items():
        approximations['uniform_projection' if k==1 else f'ancestry_K{k}']=(q@projector(basis),k,True)
    rows=[]
    for name,(approx,k,is_projection) in approximations.items():
        rows.append(metadata|dict(dictionary=name,channels=k,is_projection=is_projection,
            population_nmse=float(np.mean((output[0]-y)**2)/variance),sites=6)|field_metrics(q,approx,residual,feat))
    spectra=[];eig=[]
    fields={'path_q':q,'loss_credit':q*residual[:,None],
        'parameter_jacobian_nonroot':(q[:,:,None]*feat[:,:6]).reshape(len(x),24),
        'parameter_gradient_nonroot':(q[:,:,None]*residual[:,None,None]*feat[:,:6]).reshape(len(x),24)}
    for kind,field in fields.items():
        ev,record=spectrum(field)
        spectra.append(metadata|dict(field=kind,dimensions=field.shape[1])|record)
        eig.extend(metadata|dict(field=kind,index=i+1,eigenvalue=float(v),fraction=float(v/max(ev.sum(),1e-30))) for i,v in enumerate(ev))
    return rows,spectra,eig


def replay_family(seed,family):
    """Replay ten independently selected compatible arms; no rate retuning."""
    cfg=json.loads((JOURNAL/'source_data/morphology_credit/protocol.json').read_text())
    fitted=json.loads((JOURNAL/'source_data/morphology_credit/development_fit.json').read_text())
    coeff=credit.make_task(seed,family);tree,bound,score=credit.tree_from_coeff(coeff,'compatible')
    assert bound==0
    metadata,left,right,pp=credit.pack([tree,credit.shuffled_tree(tree,seed)],seed)
    chosen=[i for i,m in enumerate(metadata) if m['structure']=='compatible' and m['rate']==fitted[m['optimizer']][m['rule']]]
    metadata=[metadata[i] for i in chosen];left=left[chosen];right=right[chosen];pp=pp[chosen]
    count=len(metadata)
    init=np.random.default_rng(np.random.SeedSequence([seed,credit.FAMILIES.index(family),41])).normal(0,.5,(7,4));init[:,0]=0.
    weights=np.broadcast_to(init,(count,7,4)).copy()
    x,y=credit.dataset(seed,family,'training',coeff,cfg['training_samples'])
    tx,ty=credit.dataset(seed,family,'test',coeff,cfg['test_samples'])
    px=credit.domain();py=credit.fourier_design(px)@coeff;variance=float(coeff@coeff)
    rng=np.random.default_rng(np.random.SeedSequence([seed,credit.FAMILIES.index(family),37]))
    moment=np.zeros_like(weights);second=np.zeros_like(weights)
    rates=np.array([m['rate'] for m in metadata])[:,None,None]
    adam=np.array([m['optimizer']=='adam' for m in metadata])
    clip_counts=np.zeros(count,int);box_counts=np.zeros(count,int)
    archived=pd.read_csv(JOURNAL/'source_data/morphology_credit/runs/fresh'/f'seed_{seed}.csv')
    archived=archived[archived.family.eq(family)&archived.structure.eq('compatible')]
    snapshots=[];validation=[]
    for step in range(cfg['steps']+1):
        if step in cfg['checkpoints']:
            pred=credit.forward(tx,weights,left,right)[:,14];pop=credit.forward(px,weights,left,right)[:,14]
            nmse=np.mean((pred-ty[None])**2,axis=1)/variance;pn=np.mean((pop-py[None])**2,axis=1)/variance
            delivered,exact,_,q,routed=credit.gradient(px,py,weights,left,right,pp,metadata,variance)
            cosine=np.sum(delivered*exact,axis=(1,2))/np.maximum(np.linalg.norm(delivered,axis=(1,2))*np.linalg.norm(exact,axis=(1,2)),1e-30)
            field_err=np.sum((q-routed)**2,axis=(1,2))/np.maximum(np.sum(q*q,axis=(1,2)),1e-30)
            independent_output,independent_q,_=algebraic_fields(weights,left,right,px)
            assert np.allclose(independent_output,pop,rtol=0,atol=1e-13)
            assert np.allclose(independent_q,q.transpose(0,2,1),rtol=0,atol=1e-13)
            for i,m in enumerate(metadata):
                row=archived[archived.optimizer.eq(m['optimizer'])&archived.rule.eq(m['rule'])&np.isclose(archived.rate,m['rate'])&archived.step.eq(step)].iloc[0]
                metrics=dict(test_nmse=nmse[i],population_nmse=pn[i],gradient_cosine=cosine[i],field_relative_squared_error=field_err[i],
                    gradient_clipped_steps=clip_counts[i],parameter_projected_steps=box_counts[i],parameter_norm=np.linalg.norm(weights[i]),max_abs_parameter=abs(weights[i]).max())
                for name,value in metrics.items():
                    difference=abs(float(value)-float(row[name])); tolerance=2e-9+2e-9*abs(float(row[name]))
                    validation.append(dict(seed=seed,family=family,**m,step=step,metric=name,replayed=float(value),archived=float(row[name]),absolute_difference=difference,tolerance=tolerance,passed=bool(difference<=tolerance)))
                    assert difference<=tolerance,(seed,family,m,step,name,value,row[name])
            snapshots.append(weights.copy())
        if step==cfg['steps']:break
        idx=rng.integers(len(x),size=cfg['batch_size'])
        g,*_=credit.gradient(x[idx],y[idx],weights,left,right,pp,metadata,variance)
        norms=np.linalg.norm(g,axis=(1,2));clip_counts+=norms>cfg['gradient_norm_clip']
        g*=np.minimum(1,cfg['gradient_norm_clip']/np.maximum(norms,1e-30))[:,None,None]
        moment=.9*moment+.1*g;second=.999*second+.001*g*g
        update=g.copy();update[adam]=(moment[adam]/(1-.9**(step+1)))/(np.sqrt(second[adam]/(1-.999**(step+1)))+1e-8)
        weights-=rates*update;box_counts+=np.any(abs(weights)>cfg['parameter_bound'],axis=(1,2))
        weights=np.clip(weights,-cfg['parameter_bound'],cfg['parameter_bound'])
    return tree,coeff,init,metadata,np.stack(snapshots),validation


def run_seed(seed):
    freeze();dest=OUT/'replay';dest.mkdir(exist_ok=True)
    allrows=[];allspectra=[];alleigen=[];valid=[];states={};trees=[]
    tick=time.perf_counter()
    for family in credit.FAMILIES:
        tree,coeff,initial,metadata,snapshots,audit=replay_family(seed,family)
        valid.extend(audit);states[family]=snapshots
        trees.append(dict(family=family,children={str(k):list(v) for k,v in tree.children.items()},metadata=metadata,
            coefficient=coeff.tolist(),initial=initial.tolist()))
        for j,step in enumerate([0,1,16,64,256,1024]):
            for i,m in enumerate(metadata):
                meta=dict(cohort='frozen_algebraic_replay',seed=seed,family=family,step=step,**m)
                rows,spec,eig=analyze_algebraic_state(snapshots[j,i],tree,coeff,initial,meta)
                allrows.extend(rows);allspectra.extend(spec);alleigen.extend(eig)
        print('capture replay',seed,family,'seconds',round(time.perf_counter()-tick,2),flush=True)
    for name,rows in [('capture',allrows),('spectra',allspectra),('eigenvalues',alleigen),('metric_validation',valid)]:
        pd.DataFrame(rows).to_csv(dest/f'seed_{seed}_{name}.csv',index=False)
    np.savez_compressed(dest/f'seed_{seed}_checkpoints.npz',**states)
    dump(dest/f'seed_{seed}_state_metadata.json',trees)
    dump(dest/f'seed_{seed}_audit.json',dict(seed=seed,status='passed',fits=30,states=180,metric_checks=len(valid),
        max_metric_absolute_difference=max(r['absolute_difference'] for r in valid),elapsed_seconds=time.perf_counter()-tick,
        original_csv_sha256=sha(JOURNAL/'source_data/morphology_credit/runs/fresh'/f'seed_{seed}.csv'),
        replay_checkpoint_sha256=sha(dest/f'seed_{seed}_checkpoints.npz'),script_sha256=sha(__file__)))


def inspect_saved_cohorts():
    """Inventory saved states and audit conductance rank using actual final files."""
    inventory=[];rows=[];spectra=[]
    for name,pattern in [('algebraic', 'morphology_credit/runs/fresh/*'),
                         ('end_to_end','morphology_calibration/end_to_end/runs/*weights.npz'),
                         ('conductance','morphology_conductance/runs/fresh/*final_states.npz')]:
        files=list((JOURNAL/'source_data').glob(pattern));states=[p for p in files if p.suffix=='.npz']
        inventory.append(dict(cohort=name,checkpoint_files=len(states),files=[str(p.relative_to(JOURNAL)) for p in states],
            state='metrics only; replay required' if not states else 'final states only; initial state reproducible from declared seed'))
    cfg=json.loads((JOURNAL/'source_data/morphology_conductance/development_protocol.json').read_text())
    rates=json.loads((JOURNAL/'source_data/morphology_conductance/fresh_protocol.json').read_text())['selected_learning_rates']
    records=[dict(optimizer=o,rate=rates[o],group=g,rule=r) for o in cfg['optimizers'] for g in range(3) for r in ['exact_path','calibrated_broadcast','subtree_projection','broadcast_projection']]
    for seed in cfg['fresh_seeds']:
        path=JOURNAL/'source_data/morphology_conductance/runs/fresh'/f'seed_{seed}_final_states.npz'
        states=np.load(path)
        initial=np.log(conductance.NOMINAL_G)+np.random.default_rng(seed+920_000).normal(0,.4,16)
        for task in range(3):
            x,y=conductance.dataset(seed,task,'test',cfg['n_test'])
            for i,m in enumerate(records):
                if m['group']!=task or m['rule']!='exact_path':continue
                for step,weights in [(0,initial),(1000,states[f'task_{task}_final_log_conductances'][i])]:
                    group=conductance.GROUPINGS[[m['group']]]
                    state=conductance.forward(weights[None],x,group);q=state['path'][0,:,:6]
                    residual=state['output'][0]-y
                    profile=states[f'task_{task}_initial_profiles'][i]
                    meta=dict(cohort='saved_conductance',seed=seed,task_group=task,optimizer=m['optimizer'],step=step)
                    for label,field in [('path_q',q),('loss_credit',q*residual[:,None])]:
                        _,sp=spectrum(field);spectra.append(meta|dict(field=label)|sp)
                    dictionaries={'unit_projection':np.ones((6,1)), 'initial_profile_projection':profile[:,None],
                        'best_fixed_rank1_q':best_profile(q)[:,None],
                        'best_fixed_rank1_error':best_profile(q*residual[:,None])[:,None]}
                    for name,basis in dictionaries.items():
                        approx=q@projector(basis);norm=np.sum(q*q,axis=1);error=np.sum((q-approx)**2,axis=1)
                        rows.append(meta|dict(dictionary=name,sites=6,field_energy_capture=float(1-error.sum()/norm.sum()),
                            mean_per_example_capture=float(np.mean(1-error/norm)),
                            error_weighted_capture=float(1-np.sum(error*residual**2)/np.sum(norm*residual**2)),
                            test_nmse=float(np.mean(residual**2)/np.var(y)),source_sha256=sha(path)))
    dump(OUT/'checkpoint_inventory.json',inventory)
    pd.DataFrame(rows).to_csv(OUT/'conductance_saved_state_capture.csv',index=False)
    pd.DataFrame(spectra).to_csv(OUT/'conductance_saved_state_spectra.csv',index=False)


def summarize():
    frames={}
    for kind in ['capture','spectra','eigenvalues','metric_validation']:
        paths=sorted((OUT/'replay').glob(f'seed_*_{kind}.csv'))
        frames[kind]=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
        frames[kind].to_csv(OUT/f'algebraic_{kind}.csv',index=False)
    df=frames['capture'];sp=frames['spectra']
    assert frames['metric_validation'].passed.all()
    keys=['family','optimizer','rule','step','dictionary']
    metrics=['field_energy_fidelity','mean_per_example_fidelity','error_weighted_energy_fidelity',
             'eligibility_weighted_update_fidelity','population_gradient_cosine_nonroot','population_gradient_cosine_full','population_nmse']
    summary=df.groupby(keys)[metrics].agg(['mean','std','count']);summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT/'algebraic_capture_summary.csv',index=False)
    specsum=sp.groupby(['family','optimizer','rule','step','field'])[['effective_rank','rank95','rank99','leading_fraction']].mean().reset_index()
    specsum.to_csv(OUT/'algebraic_spectrum_summary.csv',index=False)
    # Descriptive paired-seed contrasts, with whole-seed resampling; not preregistered.
    contrasts=[];rng=np.random.default_rng(20260906)
    for (optimizer,rule,step,dictionary),g in df.groupby(['optimizer','rule','step','dictionary']):
        pair=g[g.family.isin(['matching','quartet'])].pivot(index='seed',columns='family',values=metrics)
        for metric in metrics:
            values=(pair[metric]['quartet']-pair[metric]['matching']).to_numpy()
            draws=values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1)
            contrasts.append(dict(optimizer=optimizer,rule=rule,step=step,dictionary=dictionary,metric=metric,
                contrast='quartet_minus_matching',mean=float(values.mean()),ci95_low=float(np.quantile(draws,.025)),
                ci95_high=float(np.quantile(draws,.975)),n_seeds=len(values),status='posthoc descriptive; no multiplicity adjustment'))
    pd.DataFrame(contrasts).to_csv(OUT/'algebraic_paired_contrasts.csv',index=False)
    audits=[json.loads(p.read_text()) for p in sorted((OUT/'replay').glob('seed_*_audit.json'))]
    dump(OUT/'validation.json',dict(status='passed',seed_blocks=df.seed.nunique(),fits=sum(a['fits'] for a in audits),
        checkpoints=sum(a['states'] for a in audits),archived_metric_checks=len(frames['metric_validation']),
        max_archived_metric_absolute_difference=float(frames['metric_validation'].absolute_difference.max()),
        projection_values_in_unit_interval=bool(df[df.is_projection].field_energy_fidelity.between(-1e-10,1+1e-10).all()),
        analysis_protocol_sha256=sha(OUT/'analysis_protocol.json')))
    write_report(df,sp)
    plot(df,sp)


def write_report(df,sp):
    n=df.seed.nunique();end=df[df.step.eq(1024)&df.rule.eq('exact')]
    table=end.groupby(['optimizer','family','dictionary'])[['field_energy_fidelity','error_weighted_energy_fidelity','population_nmse']].mean()
    selected=['uniform_projection','initial_mean_fixed','best_fixed_rank1_q','ancestry_K2','ancestry_K4']
    lines=['# Spatial credit resolution: post hoc checkpoint analysis','',
        f'This audit replays {n} existing seed blocks, three task families and ten selected compatible-tree conditions per family. Original parameter checkpoints were not saved; every deterministic archived metric is checked before the recovered states are used.',
        '', 'The routing field has six entries: one for each nonroot internal unit. The soma has exact sensitivity one under every rule and is excluded from field capture. Leaves are inputs, not additional routed internal units.',
        '', 'At fixed weights, tree and input, the path field q does not depend on the target. Multiplication by the scalar output residual changes total/error-weighted field energy but cancels from normalized capture for each example. Thus a credit-spectrum difference at trained states can describe different learned computations; it cannot by itself establish an initialization-time task-only law.',
        '', 'A uniform broadcast is one particular spatial profile. A best fixed spatial rank-one profile is an oracle chosen over the evaluation distribution. High rank-one capture with low uniform capture points to gain/sign calibration. Both quantities are distinct from eligibility-weighted update matching and from eventual learning success.',
        '', 'Matching and quartet targets share input-gradient covariance 0.25 I8. Their original fits have the same balanced topology up to input assignment, but family-specific random initialization and input assignment. Nested targets use a deeper compatible tree and are reported separately. A fully paired same-tree learning bridge is a separate new experiment.',
        '', 'Raw singular-energy fractions and effective rank depend on the units used to express each internal state. An invertible static rescaling preserves algebraic matrix rank but can change energy capture. The companion gauge audit divides each spatial coordinate by its RMS path sensitivity and verifies invariance of that normalized spectrum to static diagonal rescaling. This is a diagnostic coordinate change, not evidence that learning is invariant to reparameterization.',
        '', '## Exact-trained checkpoints, step1024','',
        '| Optimizer | Task | Dictionary | Field energy capture/fidelity | Error-weighted energy | Population NMSE |',
        '|---|---|---|---:|---:|---:|']
    for (o,f,d),r in table.iterrows():
        if d in selected:lines.append(f'| {o} | {f} | {d} | {r.iloc[0]:.4f} | {r.iloc[1]:.4f} | {r.iloc[2]:.4f} |')
    lines+=['','The initial-mean fixed profile is a delivered field with a fixed coefficient; its column is fidelity and may be negative. Other listed rows are orthogonal projection capture with per-example coefficients computed from the exact field. They are oracle capacity diagnostics, not local learning rules.',
        '', '## Source tables','',
        '- `algebraic_capture.csv`: every checkpoint, reference learning rule and candidate spatial dictionary; all metrics use the same checkpoint within a row group.',
        '- `algebraic_spectra.csv` and `algebraic_eigenvalues.csv`: path, loss-credit, parameter Jacobian and parameter-gradient spectra; no centering.',
        '- `algebraic_metric_validation.csv`: reconstructed versus archived metrics and tolerances.',
        '- `algebraic_paired_contrasts.csv`: descriptive whole-seed bootstrap contrasts; no multiplicity adjustment.',
        '- `conductance_saved_state_capture.csv`: corresponding actual saved conductance endpoint diagnostic, including both optimizers.',
        '- `replay/`: newly recovered checkpoint arrays, tree/condition metadata and file hashes.',
        '', '## Interpretation boundaries','',
        'Error weighting can make a low-error trained model appear to need different directions by concentrating the remaining residual on a few patterns. We therefore show the unweighted path field, the mean normalized per-example measure and the error-weighted field separately. The best residual-weighted spatial profile is also fitted separately, rather than treating an unweighted PCA bound as the error-weighted optimum.',
        '', 'A spectrum with more than one direction demonstrates that one fixed profile cannot reproduce the exact field over the sampled domain. It does not prove that learning requires exact reproduction. Population-gradient cosines can differ substantially from per-example update capture because contributions cancel across examples, especially near a solution. The independent matched learning controls determine whether these geometric differences matter for the tested optimization.',
        '', 'All analyses are post hoc. Numerical precision of the exact finite-domain measurement does not remove model dependence or uncertainty across training seeds. No biological use of the computed field is established here.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))


def plot(df,sp):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(SCRIPTS))
    from journal_style import apply_neurips_style,COLORS
    apply_neurips_style()
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.8));fig.subplots_adjust(left=.11,right=.98,bottom=.12,top=.90,hspace=.48,wspace=.36)
    families=['matching','quartet','nested'];names=['Pairwise','Quartic','Nested'];cols=[COLORS['additive'],COLORS['oracle'],COLORS['shunting']]
    end=df[df.step.eq(1024)&df.rule.eq('exact')&df.optimizer.eq('adam')]
    dictionaries=['uniform_projection','best_fixed_rank1_q','ancestry_K2','ancestry_K4']
    for i,family in enumerate(families):
        for ax,metric in [(axes[0,0],'field_energy_fidelity'),(axes[0,1],'error_weighted_energy_fidelity')]:
            g=end[end.family.eq(family)].groupby('dictionary')[metric]
            means=g.mean().reindex(dictionaries);se=g.std().reindex(dictionaries)/np.sqrt(g.count().reindex(dictionaries))
            ax.errorbar(np.arange(4)+(i-1)*.14,means,yerr=se,color=cols[i],fmt='o',markersize=3,capsize=2,label=names[i])
        g=sp[(sp.family.eq(family))&sp.rule.eq('exact')&sp.optimizer.eq('adam')&sp.field.eq('path_q')].groupby('step').effective_rank
        axes[1,0].errorbar(np.arange(len(g)),g.mean(),yerr=g.std()/np.sqrt(g.count()),color=cols[i],fmt='o-',markersize=3,capsize=2,label=names[i])
        g2=end[end.family.eq(family)].groupby('dictionary').eligibility_weighted_update_fidelity
        axes[1,1].errorbar(np.arange(4)+(i-1)*.14,g2.mean().reindex(dictionaries),yerr=(g2.std()/np.sqrt(g2.count())).reindex(dictionaries),color=cols[i],fmt='o',markersize=3,capsize=2)
    for ax in [axes[0,0],axes[0,1],axes[1,1]]:
        ax.set_xticks(range(4),['Uniform','Best fixed\nrank one','Ancestry\nK=2','Ancestry\nK=4']);ax.set_ylim(-.03,1.05)
    axes[0,0].set_ylabel('Retained field energy');axes[0,1].set_ylabel('Retained residual-weighted energy')
    axes[1,0].set_xticks(range(len(g)),[str(int(v)) for v in g.mean().index]);axes[1,0].set_xlabel('Training step');axes[1,0].set_ylabel('Path-field effective rank');axes[1,0].set_ylim(.9,6.1)
    axes[1,1].set_ylabel('Eligibility-weighted update fidelity')
    for letter,ax,title in zip('ABCD',axes.flat,['Output-sensitivity field','Loss-derived credit field','Rank over learning','Local update matching']):
        ax.set_title(title,loc='left',pad=8);ax.text(-.18,1.08,letter,transform=ax.transAxes,fontweight='bold',fontsize=10.5)
        ax.spines[['top','right']].set_visible(False)
    fig.legend(handles=axes[0,0].get_legend_handles_labels()[0],labels=names,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.54,1.0))
    fig.text(.5,.025,'Post hoc audit of exact-trained compatible trees; mean ± s.e.m. over paired seed blocks',ha='center',fontsize=7)
    fig.savefig(OUT/'credit_capture_bridge.pdf');fig.savefig(OUT/'credit_capture_bridge.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run','saved','summarize']);parser.add_argument('--seed',type=int,default=127200)
    args=parser.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='run':run_seed(args.seed)
    elif args.action=='saved':freeze();inspect_saved_cohorts()
    else:summarize()
