#!/usr/bin/env python3
"""Paired seed summaries; predeclared tests and full descriptive controls."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from boolean_morphology.experiment import OUT,verify,collect,dump,sha
from boolean_morphology.model import FAMILIES,normalization

METRICS=('population_nmse','test_noisy_nmse','accuracy','balanced_accuracy','gradient_cosine')


def resampling_weights():
    idx=np.random.default_rng(202609052).integers(20,size=(10000,20))
    return np.eye(20)[idx].sum(axis=1)/20


def summarize(frame,keys,weights):
    rows=[]
    for key,z in frame.groupby(keys,sort=True):
        if not isinstance(key,tuple):key=(key,)
        by_seed=z.groupby('seed')[list(METRICS)].mean().sort_index()
        assert len(by_seed)==20
        samples=weights@by_seed.to_numpy();means=by_seed.mean()
        row=dict(zip(keys,key));row.update(n_seeds=20,n_nested_fits=len(z))
        for j,metric in enumerate(METRICS):
            lo,hi=np.quantile(samples[:,j],[.025,.975])
            row.update({f'mean_{metric}':float(means[metric]),f'{metric}_ci95_low':float(lo),f'{metric}_ci95_high':float(hi)})
        row.update(failed_fits=int(z.failed.sum()),mean_gradient_clipped_steps=float(z.gradient_clipped_steps.mean()),
                   mean_parameter_clipped_steps=float(z.parameter_clipped_steps.mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def contrast_rows(frame,optimizer,rate=None):
    z=frame[(frame.family=='xor_of_ands')&(frame.optimizer==optimizer)]
    if rate is not None:z=z[np.isclose(z.rate,rate)]
    exact=z[z.rule=='exact'].pivot(index='seed',columns='tree',values='population_nmse').sort_index()
    broadcast=z[z.rule=='broadcast'].set_index(['seed','tree']).population_nmse
    rows=[]
    for seed in exact.index:
        compatible=float(exact.loc[seed,'balanced_ab_cd'])
        crossed=float(exact.loc[seed,['balanced_ac_bd','balanced_ad_bc']].mean())
        broad=float(broadcast.loc[seed,'balanced_ab_cd'])
        for name,baseline in [('crossed_minus_compatible_exact',crossed),('broadcast_minus_exact_compatible',broad)]:
            rows.append(dict(seed=int(seed),optimizer=optimizer,rate_scope='development_selected' if rate is None else 'same_rate',
                rate=rate,contrast=name,reference_nmse=compatible,comparator_nmse=baseline,difference=baseline-compatible))
    return pd.DataFrame(rows)


def contrast_summary(pairs,weights,primary):
    rows=[]
    keys=['optimizer','rate_scope','contrast']+([] if pairs.rate.isna().all() else ['rate'])
    for key,z in pairs.groupby(keys,dropna=False):
        values=z.sort_values('seed').difference.to_numpy();assert len(values)==20
        boots=weights@values;lo,hi=np.quantile(boots,[.025,.975]);alo,ahi=np.quantile(boots,[.0125,.9875])
        row=dict(zip(keys,key));row.update(n_seeds=20,mean_difference=float(values.mean()),ci95_low=float(lo),ci95_high=float(hi),
            ci975_low=float(alo),ci975_high=float(ahi),primary=bool(primary),margin_nmse=.01,
            passes_adjusted_interval_and_mean_margin=bool(alo>0 and values.mean()>=.01) if primary else None)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    cfg=verify();rates=json.loads((OUT/'selected_rates.json').read_text());frame=collect('fresh')
    assert len(frame)==6720*len(cfg['checkpoints'])
    weights=resampling_weights();end=frame[frame.step==cfg['steps']].copy()
    selected=frame[np.array([np.isclose(r.rate,rates[r.optimizer][r.rule]) for r in frame.itertuples()])].copy()
    chosen=selected[selected.step==cfg['steps']].copy()
    frame.to_csv(OUT/'all_trajectories.csv',index=False);end.to_csv(OUT/'all_rate_endpoints.csv',index=False)
    selected.to_csv(OUT/'selected_trajectories.csv',index=False);chosen.to_csv(OUT/'selected_endpoints.csv',index=False)
    keys=['family','tree','optimizer','rule']
    summary=summarize(chosen,keys,weights);summary.to_csv(OUT/'condition_summary.csv',index=False)
    summarize(end,keys+['rate'],weights).to_csv(OUT/'all_rate_condition_summary.csv',index=False)
    summarize(selected,keys+['step'],weights).to_csv(OUT/'trajectory_summary.csv',index=False)
    pairs=contrast_rows(chosen,'adam');pairs.to_csv(OUT/'paired_primary_contrasts.csv',index=False)
    contrasts=contrast_summary(pairs,weights,True);contrasts.to_csv(OUT/'primary_contrasts.csv',index=False)
    same=pd.concat([contrast_rows(end,opt,rate) for opt in cfg['optimizers'] for rate in cfg['rates']],ignore_index=True)
    same.to_csv(OUT/'paired_same_rate_contrasts.csv',index=False)
    contrast_summary(same,weights,False).to_csv(OUT/'same_rate_contrast_summary.csv',index=False)
    refs=[]
    for (seed,family,optimizer,rule),z in chosen.groupby(['seed','family','optimizer','rule']):
        mean,_=normalization(family)
        refs.extend([dict(seed=int(seed),family=family,optimizer=optimizer,rule=rule,reference='constant_mean_predictor',
            population_nmse=1.,accuracy=max(mean,1-mean),balanced_accuracy=.5),
            dict(seed=int(seed),family=family,optimizer=optimizer,rule=rule,reference='uniform_random_candidate_expectation',
            population_nmse=float(z.population_nmse.mean()),accuracy=float(z.accuracy.mean()),balanced_accuracy=float(z.balanced_accuracy.mean()))])
    pd.DataFrame(refs).to_csv(OUT/'seed_reference_controls.csv',index=False)
    baseline=[]
    for family in FAMILIES:
        mean,sd=normalization(family);baseline.append(dict(family=family,raw_mean=mean,raw_variance=sd**2,
            positive_patterns=round(16*mean),patterns=16,normalized_threshold=(.5-mean)/sd,
            constant_mean_nmse=1.,majority_class_accuracy=max(mean,1-mean),constant_balanced_accuracy=.5))
    pd.DataFrame(baseline).to_csv(OUT/'target_normalization.csv',index=False)
    dump(OUT/'analysis_record.json',dict(protocol_sha256=sha(OUT/'protocol.json'),selection_sha256=sha(OUT/'selection_freeze.json'),
        script_sha256=sha(__file__),fresh_fits=len(end),selected_rate_fits=len(chosen),fresh_trajectory_rows=len(frame),
        failed_fits=int(end.failed.sum()),bootstrap_draws=10000,seed_blocks=20,
        primary_contrasts=contrasts.to_dict(orient='records'),
        rates=rates,scope='Seven fixed logical templates; paired seeds vary permutation, initialization and sampled noisy data. No new functional families.'))
    print(contrasts.to_string(index=False));print(summary[summary.optimizer.eq('adam')][keys+['mean_population_nmse','mean_balanced_accuracy']].to_string(index=False))


if __name__=='__main__':main()
