"""Paired stopping-extension statistics, with no filtering or outcome selection."""
import numpy as np
import pandas as pd
CONTRASTS = [
    ('depth_gain_exact_bp', ('exact_autograd_bp_recipe',3), ('exact_autograd_bp_recipe',1)),
    ('localca_path_minus_shared', ('path_transport',3), ('per_soma_shared',3)),
    ('bp_exact_minus_broadcast', ('exact_autograd_bp_recipe',3), ('broadcast_autograd_bp_recipe',3)),
    ('broadcast_bp_minus_localca_recipe', ('broadcast_autograd_bp_recipe',3), ('broadcast_autograd_localca_recipe',3)),
]

def bounds(values, weights):
    values=np.asarray(values,float)
    b=weights@values
    lo,hi=np.quantile(b,[.025,.975],axis=0)
    return values.mean(axis=0),lo,hi

def summarize(d, end, out):
    ix=np.random.default_rng(601806).integers(10,size=(10000,10))
    weights=np.array([np.bincount(i,minlength=10) for i in ix],float)/10
    summaries=[];paired=[];paired_seed=[];contrasts=[]
    for (arm,depth),group in d.groupby(['arm','depth']):
        assert set(group.seed)==set(range(10200,10210))
        for metric in ['test_accuracy','test_cross_entropy','best_validation_loss']:
            v=group.pivot(index='seed',columns='epoch',values=metric).sort_index()
            assert not v.isna().any().any()
            mean,lo,hi=bounds(v.to_numpy(),weights)
            summaries.extend(dict(arm=arm,depth=int(depth),metric=metric,epoch=int(e),mean=float(m),ci95_low=float(l),ci95_high=float(h),n_seeds=10) for e,m,l,h in zip(v.columns,mean,lo,hi))
    local=d[d.depth.eq(3)&d.arm.isin(['path_transport','per_soma_shared'])]
    for metric in ['test_accuracy','test_cross_entropy','best_validation_loss']:
        p=local.pivot(index=['seed','epoch'],columns='arm',values=metric)
        v=(p.path_transport-p.per_soma_shared).unstack('epoch').sort_index()*(100 if metric=='test_accuracy' else 1)
        mean,lo,hi=bounds(v.to_numpy(),weights)
        paired.extend(dict(metric=metric,epoch=int(e),mean=float(m),ci95_low=float(l),ci95_high=float(h),n_seeds=10,positive_seeds=int((v[e]>0).sum()),negative_seeds=int((v[e]<0).sum())) for e,m,l,h in zip(v.columns,mean,lo,hi))
        paired_seed.extend(dict(metric=metric,seed=int(seed),epoch=int(e),exact_minus_shared=float(v.loc[seed,e])) for seed in v.index for e in v.columns)
    for budget,group in end.groupby('budget'):
        for metric in ['test_accuracy','test_loss']:
            p=group.pivot(index='seed',columns=['arm','depth'],values=metric).sort_index()
            for name,left,right in CONTRASTS:
                v=(p[left]-p[right]).to_numpy()*(100 if metric=='test_accuracy' else 1)
                m,l,h=bounds(v,weights)
                contrasts.append(dict(budget=int(budget),metric=metric,contrast=name,mean=float(m),ci95_low=float(l),ci95_high=float(h),n_seeds=10,positive_seeds=int((v>0).sum()),negative_seeds=int((v<0).sum()),units='percentage points' if metric=='test_accuracy' else 'nats'))
    pd.DataFrame(summaries).to_csv(out/'condition_trajectory_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_trajectory_summary.csv',index=False)
    pd.DataFrame(paired_seed).to_csv(out/'paired_seed_trajectories.csv',index=False)
    pd.DataFrame(contrasts).to_csv(out/'paired_contrasts.csv',index=False)
    return contrasts
