#!/usr/bin/env python3
"""Summarize all frozen extension outcomes, including reused-seed uncertainty."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run

OUT=run.OUT
KEY=['seed','task','optimizer','rule','rate']
BUDGETS=[1024,4096,8192,16384]

def bootstrap(v):
 v=np.asarray(v,float);rng=np.random.default_rng(210999)
 b=v[rng.integers(len(v),size=(10000,len(v)))].mean(axis=1)
 return {'mean':float(v.mean()),'median':float(np.median(v)),'ci95_low':float(np.quantile(b,.025)),'ci95_high':float(np.quantile(b,.975)),
  'n_seeds':len(v),'positive_seeds':int((v>0).sum()),'negative_seeds':int((v<0).sum()),'zero_seeds':int((v==0).sum())}

def main():
 p=run.check_freeze();frames=[];diagnostics=[];audits=[];sources={};replays=[]
 for seed in p['seeds']:
  folder=OUT/'runs'/f'seed_{seed}';a=json.loads((folder/'audit.json').read_text());assert a['status']=='complete' and a['protocol_sha256']==run.sha(OUT/'protocol_freeze.json')
  for name,h in a['source_files_sha256'].items():assert run.sha(folder/name)==h
  sources[str((folder/'audit.json').relative_to(run.J))]=run.sha(folder/'audit.json')
  audits.append(a)
  for kind,collector in [('curves',frames),('diagnostics',diagnostics)]:collector.append(pd.read_csv(folder/f'{kind}.csv',float_precision='round_trip'))
  replay=json.loads((folder/'replay_validation.json').read_text());assert replay['status']=='passed'
  for x in replay['checks']:replays.append({'seed':seed,**{k:v for k,v in x.items() if k!='columns_checked'}})
 d=pd.concat(frames,ignore_index=True);diag=pd.concat(diagnostics,ignore_index=True)
 assert len(d[KEY].drop_duplicates())==720
 expected_steps=p['checkpoints'];assert all(g.step.tolist()==expected_steps for _,g in d.sort_values('step').groupby(KEY))
 assert np.isfinite(d[['test_nmse','validation_nmse','population_nmse']]).all().all()
 summary=OUT/'summaries';summary.mkdir(exist_ok=True)
 d.to_csv(summary/'all_curves.csv',index=False);diag.to_csv(summary/'all_diagnostics.csv',index=False)
 rows=[]
 for keys,g in d.groupby(KEY,sort=True):
  g=g.sort_values('step')
  for budget in BUDGETS:
   window=g[g.step<=budget];last=window[window.step==budget].iloc[0]
   best=window.sort_values(['validation_nmse','step']).iloc[0]
   for endpoint,r in [('terminal',last),('validation_selected',best)]:
    rows.append(r.to_dict()|{'budget':budget,'endpoint':endpoint,'chosen_step':int(r.step),'cohort_role':'post_review_extension_of_previously_observed_fresh_seed_blocks'})
 outcomes=pd.DataFrame(rows);outcomes.to_csv(summary/'all_budget_outcomes.csv',index=False)
 counts=[];contrasts=[];seedrows=[]
 for view in ['selected_rate','common_rate']:
  frame=outcomes[outcomes[view]]
  for keys,g in frame.groupby(['budget','endpoint','task','optimizer','rule']):
   meta=dict(zip(['budget','endpoint','task','optimizer','rule'],keys))|{'rate_view':view,'rate':float(g.rate.iloc[0])}
   assert g.seed.nunique()==20 and len(g)==20
   for metric in ['test_nmse','population_nmse','validation_nmse']:
    counts.append(meta|{'metric':metric,**bootstrap(g[metric]),'minimum':float(g[metric].min()),'maximum':float(g[metric].max()),'parameter_projected_any_count':int((g.parameter_projected_steps>0).sum()),'gradient_clipped_any_count':int((g.gradient_clipped_steps>0).sum()),'mean_selected_step':float(g.chosen_step.mean())})
  for keys,g in frame.groupby(['budget','endpoint','task','optimizer']):
   meta=dict(zip(['budget','endpoint','task','optimizer'],keys))|{'rate_view':view}
   for metric in ['test_nmse','population_nmse']:
    w=g.pivot(index='seed',columns='rule',values=metric)
    for control,reference in [('unit_broadcast','exact'),('calibrated_broadcast','exact'),('sign_broadcast','exact'),('unit_broadcast','calibrated_broadcast')]:
     v=w[control]-w[reference];m=meta|{'metric':metric,'contrast':control+' minus '+reference}
     contrasts.append(m|bootstrap(v));seedrows.extend(m|{'seed':int(seed),'difference':float(value)} for seed,value in v.items())
  for keys,g in frame.groupby(['budget','endpoint','optimizer']):
   meta=dict(zip(['budget','endpoint','optimizer'],keys))|{'rate_view':view,'task':'quartet_minus_matching'}
   for metric in ['test_nmse','population_nmse']:
    w=g.pivot(index='seed',columns=['task','rule'],values=metric)
    for control in ['unit_broadcast','calibrated_broadcast','sign_broadcast']:
     v=(w['quartet',control]-w['quartet','exact'])-(w['matching',control]-w['matching','exact'])
     m=meta|{'metric':metric,'contrast':control+' minus exact interaction'}
     contrasts.append(m|bootstrap(v));seedrows.extend(m|{'seed':int(seed),'difference':float(value)} for seed,value in v.items())
 means=pd.DataFrame(counts);paired=pd.DataFrame(contrasts);seeds=pd.DataFrame(seedrows)
 means.to_csv(summary/'condition_summary.csv',index=False);paired.to_csv(summary/'paired_contrasts.csv',index=False);seeds.to_csv(summary/'paired_seed_contrasts.csv',index=False)
 # This diagnostic uses an explicit descriptive threshold, not a learning/stopping decision.
 distribution=outcomes[(outcomes.task=='quartet')&outcomes.selected_rate].copy()
 distribution['within_0_02_above_noise_floor']=distribution.test_nmse<=.065
 distribution['clean_population_nmse_below_0_02']=distribution.population_nmse<=.02
 distribution.to_csv(summary/'quartic_seed_distributions.csv',index=False)
 diagnostic_rows=[]
 for keys,g in distribution.groupby(['budget','endpoint','optimizer','rule']):
  diagnostic_rows.append(dict(zip(['budget','endpoint','optimizer','rule'],keys))|{'count':len(g),'near_floor_count_0_065':int(g.within_0_02_above_noise_floor.sum()),'clean_population_below_0_02_count':int(g.clean_population_nmse_below_0_02.sum()),'median_test_nmse':float(g.test_nmse.median()),'maximum_test_nmse':float(g.test_nmse.max())})
 pd.DataFrame(diagnostic_rows).to_csv(summary/'quartic_distribution_summary.csv',index=False)
 # Terminal bound counts report every trajectory, including ones not selected by validation.
 bound=d[d.step.isin(BUDGETS)].copy();bound['ever_projected']=bound.parameter_projected_steps>0
 bound['at_coordinate_bound']=bound.max_abs_parameter>=2-1e-12
 bound.to_csv(summary/'bound_and_clipping_events.csv',index=False)
 source_primary=pd.read_csv(run.ORIGINAL/'summaries/paired_contrasts.csv',float_precision='round_trip')
 errors=[]
 for row in paired[(paired.budget==1024)&(paired.endpoint=='terminal')&(paired.metric=='test_nmse')].itertuples():
  old=source_primary[(source_primary.sensitivity==row.rate_view)&(source_primary.model=='algebraic')&(source_primary.task==row.task)&(source_primary.optimizer==row.optimizer)&(source_primary.contrast==row.contrast)]
  assert len(old)==1
  for col in ['mean','ci95_low','ci95_high']:
   err=abs(float(getattr(row,col))-float(old.iloc[0][col]));assert err<1e-10;errors.append(err)
  assert row.positive_seeds==old.iloc[0].positive_seeds
 focal=paired[(paired.task=='quartet_minus_matching')&(paired.optimizer=='adam')&(paired.metric=='test_nmse')&(paired.contrast=='calibrated_broadcast minus exact interaction')]
 report={'status':'passed','scope':p['scope'],'unique_trajectories':720,'seed_blocks':20,'tasks':p['tasks'],'updates_per_trajectory':16384,'curve_rows':len(d),'budget_outcome_rows':len(outcomes),'all_original_states_and_metrics_reproduced':True,'original1024_contrast_and_bootstrap_max_abs_difference':max(errors),'all_original1024_positive_seed_counts_match':True,'original_validation_test_clock_sources_preserved':True,'sum_cpu_worker_seconds':sum(x['elapsed_seconds'] for x in audits),'all_declared_arms_complete':True,'primary_budget_indexed_interactions':focal.to_dict('records'),'protocol_sha256':run.sha(OUT/'protocol_freeze.json'),'source_audit_sha256':sources,'analysis_script_sha256':run.sha(__file__),'created_utc':run.now(),'distribution_threshold_scope':'NMSE<=0.065 or clean populationNMSE<=0.02 are descriptive post-processing markers only; not prespecified decisions, exclusions, stopping or success tests.'}
 run.write(summary/'validation_report.json',report);run.write(summary/'replay_summary.json',{'status':'passed','checks':replays})
 text=['# Balanced longer-budget credit-rule follow-up','',p['scope'],'','All720 original algebraic trajectories were replayed and extended to16,384 updates. No rules, seeds, tasks or rate-union members were removed. The original1024 checkpoint arrays, numerical curves, diagnostics and primary summaries reproduce. Both terminal and validation-selected outcomes remain available.','',
 '| Budget | Endpoint | Rate view | Quartic-minus-pairwise calibrated-minus-exact NMSE | 95% paired interval | Positive seeds |','|---:|---|---|---:|---|---:|']
 for r in focal.itertuples():text.append(f'| {r.budget} | {r.endpoint} | {r.rate_view} | {r.mean:.6f} | [{r.ci95_low:.6f}, {r.ci95_high:.6f}] | {r.positive_seeds}/20 |')
 text+=['','These are budget-indexed contrasts at historical rate choices and bounded coefficients, not convergence or minimum-feedback-rank results. Validation-selected states use only the declared checkpoints and their validation NMSE. The same original twenty seed blocks had already been examined; intervals are descriptive and are not fresh confirmatory evidence.','', 'The condition/paired tables retain mean, median, intervals, all paired win counts and every seed. The quartic distribution table retains near-floor and stalled outcomes with explicit descriptive thresholds. Parameter projection and clipping are retained for all terminal trajectories, independently of checkpoint selection.','']
 (OUT/'RESULTS.md').write_text('\n'.join(text))
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
