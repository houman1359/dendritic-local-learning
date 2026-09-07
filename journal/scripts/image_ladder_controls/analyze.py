#!/usr/bin/env python3
"""Complete-cohort summaries; no selection by held-out test outcomes."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from run import OUT, RAW, ARCHITECTURES, ARMS, sha, dump

def interval(values,seed=911,draws=50000):
    x=np.asarray(values,float);rng=np.random.default_rng(seed)
    q=np.quantile(rng.choice(x,(draws,len(x)),replace=True).mean(1),[.025,.975])
    return dict(mean=float(x.mean()),ci_low=float(q[0]),ci_high=float(q[1]),n=len(x),positive=int((x>0).sum()))

def collect(stage,export=True):
    records=json.loads((OUT/f'{stage}_conditions.json').read_text());rows=[];curves=[];hashes=[]
    for rec in records:
        run=Path(rec['results_dir']);audit=json.loads((run/'run_audit.json').read_text())
        assert audit['status']=='complete' and audit['epochs']==180,(stage,rec['index'])
        assert audit['config_sha256']==sha(rec['config'])
        results=Path(audit['model_results_dir'])
        final=json.loads((results/'performance/final.json').read_text())
        progress=json.loads((run/'progress.json').read_text())
        expected=min(x['valid_loss'] for x in progress)
        params=json.loads((results/'model_resources.json').read_text())
        row=dict(stage=stage,index=rec['index'],architecture=rec['architecture'],arm=rec['arm'],multiplier=rec['multiplier'],seed=rec['seed'],
            test_accuracy=final['accuracy']['test'],test_cross_entropy=-final['categorical_loglikelihood']['test'],
            validation_cross_entropy=-final['categorical_loglikelihood']['valid'],selection_validation_loss=expected,
            best_epoch=progress[-1]['best_epoch'],final_epoch_valid_loss=progress[-1]['valid_loss'],epochs=len(progress),
            initialized_model_sha256=audit['initial']['model_sha256'],initialized_core_sha256=audit['initial']['core_sha256'],
            final_core_sha256=audit['final_core_sha256'],final_model_sha256=sha(results/'final_model.pt'),
            decoder_only_core_unchanged=audit['initial']['core_sha256']==audit['final_core_sha256'],
            total_parameters=params['total_parameters'],active_synapses=params['active_synapses'],
            runtime_commit=audit['runtime_commit'],elapsed_seconds=audit['elapsed_seconds'])
        assert row['total_parameters']==2414602 and row['active_synapses']==92160
        if rec['arm']=='decoder_only': assert row['decoder_only_core_unchanged']
        rows.append(row)
        for point in progress:curves.append(dict(stage=stage,index=rec['index'],architecture=rec['architecture'],arm=rec['arm'],multiplier=rec['multiplier'],seed=rec['seed'],**point))
        include=[run/'run_audit.json',run/'progress.json',run/'initial_state_identity.json',run/'optimizer.json',
            results/'config.json',results/'resolved_seeds.json',results/'training_summary.json',results/'performance/final.json',results/'model_resources.json']
        for p in include:
            rel=Path(stage)/f'condition_{rec["index"]:03d}'/p.relative_to(run)
            target=OUT/'canonical'/rel
            if export:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
            hashes.append(dict(stage=stage,index=rec['index'],source=str(p),canonical=str(target.relative_to(OUT)),sha256=sha(p),size=p.stat().st_size))
        for p in (run/'initial_model.pt',results/'final_model.pt'):
            hashes.append(dict(stage=stage,index=rec['index'],source=str(p),canonical='',sha256=sha(p),size=p.stat().st_size))
    frame=pd.DataFrame(rows)
    assert frame.groupby(['architecture','seed']).initialized_model_sha256.nunique().eq(1).all(),'Paired models mismatch'
    return frame,pd.DataFrame(curves),pd.DataFrame(hashes)

def summarize():
    frames=[];curves=[];hashes=[]
    for stage in ('development','fresh'):
        f,c,h=collect(stage);frames.append(f);curves.append(c);hashes.append(h)
    outcomes=pd.concat(frames);allcurves=pd.concat(curves);allhashes=pd.concat(hashes)
    out=OUT/'summaries';out.mkdir(exist_ok=True)
    outcomes.to_csv(out/'all_outcomes.csv',index=False);allcurves.to_csv(out/'all_curves.csv',index=False);allhashes.to_csv(out/'source_files.csv',index=False)
    selection=pd.DataFrame(json.loads((OUT/'selection.json').read_text())['selected'])
    fresh=outcomes[outcomes.stage.eq('fresh')]
    selected=fresh.merge(selection[['architecture','arm','multiplier']],on=['architecture','arm','multiplier'],validate='many_to_one').assign(rate_policy='selected')
    common=fresh[fresh.multiplier.eq(1)].assign(rate_policy='common_original')
    views=pd.concat([selected,common]);views.to_csv(out/'fresh_analysis_rows.csv',index=False)
    conditions=[]
    for key,g in views.groupby(['rate_policy','architecture','arm']):
        for metric in ('test_accuracy','test_cross_entropy','selection_validation_loss','best_epoch'):
            conditions.append(dict(zip(('rate_policy','architecture','arm'),key),metric=metric,**interval(g[metric])))
    pd.DataFrame(conditions).to_csv(out/'condition_summary.csv',index=False)
    contrasts=[];paired=[]
    arm_pairs=[('neuron_shared','strict_scalar'),('subtree_k3','neuron_shared'),('exact_path','subtree_k3'),('neuron_shared','decoder_only'),('exact_path','neuron_shared')]
    for policy in ('selected','common_original'):
        for architecture in ARCHITECTURES:
            group=views[views.rate_policy.eq(policy)&views.architecture.eq(architecture)]
            for metric in ('test_accuracy','test_cross_entropy'):
                pivot=group.pivot(index='seed',columns='arm',values=metric)
                assert len(pivot)==10 and not pivot.isna().any().any()
                for lhs,rhs in arm_pairs:
                    values=pivot[lhs]-pivot[rhs]
                    name=f'{lhs}_minus_{rhs}'
                    contrasts.append(dict(rate_policy=policy,architecture=architecture,metric=metric,contrast=name,**interval(values)))
                    paired.extend(dict(rate_policy=policy,architecture=architecture,metric=metric,contrast=name,seed=int(seed),difference=float(value)) for seed,value in values.items())
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_seed_contrasts.csv',index=False)
    rates=[]
    for key,g in outcomes[outcomes.stage.eq('development')].groupby(['architecture','arm','multiplier']):
        for metric in ('test_accuracy','test_cross_entropy','selection_validation_loss','best_epoch'):
            rates.append(dict(zip(('architecture','arm','multiplier'),key),metric=metric,**interval(g[metric])))
    pd.DataFrame(rates).to_csv(out/'development_rate_summary.csv',index=False)
    versions=[]
    for stage in ('development','fresh'):
        for rec in json.loads((OUT/f'{stage}_conditions.json').read_text()):
            a=json.loads((Path(rec['results_dir'])/'run_audit.json').read_text())
            versions.append(dict(stage=stage,index=rec['index'],python=a['python'],cuda=a['cuda'],device_name=a['device_name'],**a['versions']))
    pd.DataFrame(versions).to_csv(out/'software_versions.csv',index=False)
    gate=dict(complete=True,development_fits=len(frames[0]),fresh_fits=len(frames[1]),scientific_fits=len(outcomes),excluded_canaries=10,
        complete_epochs=int(outcomes.epochs.sum()),all_model_pairs_identical=True,all_decoder_floor_cores_unchanged=True,
        runtime_commit=outcomes.runtime_commit.unique().tolist(),source_file_count=len(allhashes),
        primary_metric='test_accuracy',interval='50,000pairedwhole-seedbootstrapresamples; descriptive95%CI',
        parameter_count=2414602,active_synapses=92160,selection_boundary_count=int(selection.multiplier.ne(1).sum()),
        fixed_budget='180epochs; minimumvalidationcheckpoint; no convergence or globallyoptimalrate claim',
        selection_sha256=sha(OUT/'selection.json'),protocol_sha256=sha(OUT/'protocol.json'))
    dump(out/'completeness_audit.json',gate)
    print(json.dumps(gate,indent=2));print(pd.DataFrame(conditions).query("metric=='test_accuracy' and rate_policy=='selected'").to_string(index=False))

if __name__=='__main__': summarize()
