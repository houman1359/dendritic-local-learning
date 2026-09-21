#!/usr/bin/env python3
"""Post-outcome numerical diagnostics; never changes frozen selection or learning.

Exact expected calibration one-step decrease tests the local quadratic bound.
The first update of the actual independent training stream is evaluated on the
held-out test stream to distinguish local calibration from generalization.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import run_prospective_morphology_selection as run

OUT=run.OUT


def per_seed(seed):
    cfg=run.load();run.check_freeze();rows=[];candidates=run.candidates(cfg)
    for task in run.tasks(cfg,'confirmatory',seed):
        x,a,y=run.dataset(task,'calibration',cfg['n_calibration'])
        trainx,traina,trainy=run.dataset(task,'training',cfg['n_training'])
        testx,testa,testy=run.dataset(task,'test',cfg['n_test'])
        rng=np.random.default_rng(np.random.SeedSequence([seed,task['rank'],53]))
        idx=rng.integers(len(trainx),size=cfg['batch_size'])
        for arm in cfg['arms']:
            for c in candidates:
                h=np.eye(8) if arm=='feedback_only' else c['transfer'];p=c['projector']
                score,mean,per=run.moment_score(x,a,y,h,p,cfg['learning_rate'],cfg['batch_size'])
                design=((a@h)[:,:,None]*x[:,None,:]).reshape(len(x),64)
                hessian=design.T@design/len(x);flat=per.reshape(len(x),64);m=mean.ravel()
                mean_curvature=float(m@hessian@m)
                variance_curvature=float(np.mean(np.einsum('bi,bi->b',flat@hessian,flat))-mean_curvature)/cfg['batch_size']
                eta=cfg['learning_rate']
                exact_expected=eta*score['gradient_alignment']-.5*eta**2*(mean_curvature+variance_curvature)
                slack=exact_expected-score['utility_actual_step'];assert slack>=-1e-12
                b=traina[idx]@h
                gradient=np.mean(-trainy[idx,None,None]*b[:,:,None]*trainx[idx,None,:],axis=0)
                w=-eta*(p@gradient)
                pred=np.einsum('bi,ij,bj->b',testa@h,w,testx)
                test_decrease=float(np.mean(testy**2-(pred-testy)**2)/2)
                pred_calib=np.einsum('bi,ij,bj->b',a@h,w,x)
                calib_decrease=float(np.mean(y*y-(pred_calib-y)**2)/2)
                rows.append({**run.metadata(task,arm,c),**score,
                    'exact_expected_calibration_decrease':exact_expected,
                    'expected_calibration_decrease_minus_bound':slack,
                    'actual_training_first_step_test_decrease':test_decrease,
                    'actual_training_first_step_calibration_decrease':calib_decrease})
    dest=OUT/'first_step_diagnostics';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/f'seed_{seed}.csv',index=False)
    print('first-step diagnostic',seed,len(rows),flush=True)


def summarize():
    cfg=run.load();run.check_freeze();sel=OUT/'sealed_confirmatory_selections.csv'
    freeze=json.loads((OUT/'selection_freeze.json').read_text());assert freeze['selection_sha256']==run.sha(sel)
    final=pd.read_csv(OUT/'candidate_outcomes.csv');policies=pd.read_csv(OUT/'policy_outcomes.csv')
    c=run.candidates(cfg)
    unique=[]
    for arm in cfg['arms']:
        seen=[]
        for candidate in c:
            p=candidate['projector'];h=np.eye(8) if arm=='feedback_only' else candidate['transfer']
            key=next((i for i,(p0,h0) in enumerate(seen) if np.allclose(p,p0,atol=1e-12) and np.allclose(h,h0,atol=1e-12)),None)
            if key is None:key=len(seen);seen.append((p,h))
            unique.append(dict(arm=arm,candidate_id=candidate['candidate_id'],equivalent_forward_and_projector_class=key+1,resource_cost=candidate['resource_cost']))
    pd.DataFrame(unique).to_csv(OUT/'candidate_equivalence_classes.csv',index=False)
    oracle=pd.read_csv(OUT/'retrospective_oracle.csv').merge(final[['task_id','arm','seed','rank','noise_sd','angle_pi']].drop_duplicates(),on=['task_id','arm'],validate='one_to_one')
    oracle.groupby(['arm','rank','angle_pi','noise_sd','oracle_budget_k','oracle_tree_id']).size().rename('n_tasks').reset_index().to_csv(OUT/'oracle_choice_distribution.csv',index=False)
    chosen=policies[policies.policy.isin(['moment_selector','rank_only','development_best','maximum_budget'])].copy()
    chosen.groupby(['arm','policy','rank','angle_pi','noise_sd','budget_k','tree_id','oracle_budget_k']).size().rename('n_tasks').reset_index().to_csv(OUT/'selected_vs_oracle_choices.csv',index=False)
    contrasts=[];seedcontrasts=[]
    for arm,g in policies.groupby('arm'):
        pivot=g.pivot(index=['seed','task_id'],columns='policy',values='regret')
        for left,right in [('rank_only','development_best'),('rank_only','maximum_budget'),('moment_selector','rank_only'),('moment_selector','development_best')]:
            byseed=(pivot[right]-pivot[left]).groupby('seed').mean();mean,lo,hi=run.bootstrap(byseed)
            contrasts.append(dict(arm=arm,left_policy=left,right_policy=right,mean_left_advantage=mean,ci95_low=lo,ci95_high=hi,n_seed_blocks=len(byseed),positive_seed_blocks=int((byseed>0).sum())))
            for seed,effect in byseed.items():seedcontrasts.append(dict(arm=arm,left_policy=left,right_policy=right,seed=seed,left_advantage=effect))
    pd.DataFrame(contrasts).to_csv(OUT/'additional_paired_policy_contrasts.csv',index=False)
    pd.DataFrame(seedcontrasts).to_csv(OUT/'paired_policy_seed_contrasts.csv',index=False)
    diagnostic=pd.concat([pd.read_csv(OUT/f'first_step_diagnostics/seed_{s}.csv') for s in cfg['seeds']['confirmatory']],ignore_index=True)
    assert len(diagnostic)==len(final)
    diagnostic.to_csv(OUT/'first_step_diagnostics.csv',index=False)
    per_task=[]
    for (task_id,arm),g in diagnostic.groupby(['task_id','arm']):
        for endpoint in ['exact_expected_calibration_decrease','actual_training_first_step_test_decrease']:
            per_task.append(dict(task_id=task_id,arm=arm,seed=int(g.iloc[0].seed),rank=int(g.iloc[0]['rank']),noise_sd=g.iloc[0].noise_sd,angle_pi=g.iloc[0].angle_pi,endpoint=endpoint,
                spearman_rho=float(spearmanr(g.utility_actual_step,g[endpoint]).statistic)))
    per_task=pd.DataFrame(per_task);per_task.to_csv(OUT/'first_step_within_task_correlations.csv',index=False)
    corr=[]
    for (arm,endpoint),g in per_task.groupby(['arm','endpoint']):
        seedvalues=g.groupby('seed').spearman_rho.mean();mean,lo,hi=run.bootstrap(seedvalues)
        d=diagnostic[diagnostic.arm.eq(arm)]
        corr.append(dict(arm=arm,endpoint=endpoint,mean_within_task_rho=mean,ci95_low=lo,ci95_high=hi,n_seed_blocks=len(seedvalues),pooled_descriptive_rho=float(spearmanr(d.utility_actual_step,d[endpoint]).statistic)))
    pd.DataFrame(corr).to_csv(OUT/'first_step_correlation_summary.csv',index=False)
    audits=[json.loads((OUT/f'runs/confirmatory/seed_{s}_audit.json').read_text()) for s in cfg['seeds']['confirmatory']]
    assert all(a['selection_sha256']==freeze['selection_sha256'] for a in audits)
    assert all(pd.Timestamp(a['utc'])>pd.Timestamp(freeze['utc']) for a in audits)
    assert len(final)==12800 and final.test_loss.notna().all()
    assert oracle.oracle_budget_k.eq(oracle['rank']).all()
    run.write_json(OUT/'posthoc_validation_report.json',dict(status='completed_posthoc_diagnostic_primary_selection_unchanged',
        diagnostic_script_sha256=run.sha(Path(__file__)),runner_sha256=run.sha(run.SELF),protocol_sha256=run.sha(run.CONFIG),
        sealed_selection_sha256=run.sha(sel),all_confirmatory_shards_checked_seal=True,
        n_confirmatory_candidates=len(final),n_first_step_conditions=len(diagnostic),
        all_expected_calibration_bounds_pass=bool((diagnostic.expected_calibration_decrease_minus_bound>=-1e-12).all()),
        minimum_expected_calibration_bound_slack=float(diagnostic.expected_calibration_decrease_minus_bound.min()),
        maximum_expected_calibration_bound_slack=float(diagnostic.expected_calibration_decrease_minus_bound.max()),
        all_oracle_budgets_equal_reference_rank=True,first_step_correlations=corr,
        rank_baseline_comparisons=contrasts,
        caveat='The global bound is global in W for finite calibration loss. Its score-to-independent-test-decrease or long-horizon-generalization relation is empirical. This diagnostic was added after confirmatory outcomes; the primary selector is unchanged.'))
    print(pd.DataFrame(corr).to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int);p.add_argument('--summarize',action='store_true');a=p.parse_args()
    if a.summarize:summarize()
    else:assert a.seed in run.load()['seeds']['confirmatory'];per_seed(a.seed)

if __name__=='__main__':main()
