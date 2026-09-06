#!/usr/bin/env python3
"""Independent tolerant-tie and paired-structure audit of frozen population fits.

This only reads original outcomes and writes audit_* tables; it never mutates
frozen protocol, source hashes, fits, or the original summary tables.
"""
from pathlib import Path
import hashlib
import json
import itertools
import numpy as np
import pandas as pd
from model import candidates,tasks,domain,fourier_design,matricize,cut_scores
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/morphology_investigation_20260905/structure'
TOL=1e-10
OUTCOME_TOL=1e-7

def dump(name,value):
 (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

def select(group,score):
 ordered=group.sort_values('candidate_id',kind='stable')
 minimum=float(ordered[score].min())
 ties=ordered[ordered[score]<=minimum+TOL]
 return ties.iloc[0],len(ties)

def main():
 end=pd.read_csv(OUT/'candidate_outcomes.csv')
 raw=pd.read_csv(OUT/'all_trajectories.csv')
 expected=pd.MultiIndex.from_product([[t['task_id'] for t in tasks()],[t.name for t in candidates()]])
 assert set(map(tuple,end[['task_id','candidate_id']].values))==set(expected)
 assert not end.duplicated(['task_id','candidate_id']).any()
 assert len(raw)==140*12*4*4
 assert not raw.duplicated(['task_id','candidate_id','restart','sweep']).any()
 end['permutation_index']=end.candidate_id.str.extract(r'_p(\d+)$').astype(int)
 end['gap_above_lower_bound']=end.normalized_mse-end.centered_cut_bound
 assert end.gap_above_lower_bound.min()>-1e-7
 policy_rows=[];assignments=[];shapes=[];paired=[]
 for family,frame in end.groupby('family'):
  fixed_table=frame.groupby('candidate_id',as_index=False).normalized_mse.mean()
  fixed,_=select(fixed_table,'normalized_mse')
  for tid,g in frame.groupby('task_id'):
   best=float(g.normalized_mse.min())
   for policy,score in [('centered_cut_bound','centered_cut_bound'),('centered_cut_sum','centered_cut_sum'),('full_cut_bound','full_cut_bound'),('two_sweep_pilot','pilot_error')]:
    chosen,ntie=select(g,score)
    policy_rows.append(dict(task_id=tid,family=family,policy=policy,candidate_id=chosen.candidate_id,shape=chosen['shape'],permutation_index=int(chosen.permutation_index),score_tie_count=ntie,nmse=float(chosen.normalized_mse),regret=max(0.,float(chosen.normalized_mse)-best),oracle_nmse=best))
   for policy,cid in [('fixed_balanced_p0','balanced_p0'),('best_fixed_in_hindsight',fixed.candidate_id)]:
    chosen=g[g.candidate_id.eq(cid)].iloc[0]
    policy_rows.append(dict(task_id=tid,family=family,policy=policy,candidate_id=cid,shape=chosen['shape'],permutation_index=int(chosen.permutation_index),score_tie_count=1,nmse=float(chosen.normalized_mse),regret=max(0.,float(chosen.normalized_mse)-best),oracle_nmse=best))
   policy_rows.append(dict(task_id=tid,family=family,policy='uniform_random_expectation',candidate_id='all12_expectation',shape='mixed_expectation',permutation_index=-1,score_tie_count=12,nmse=float(g.normalized_mse.mean()),regret=max(0.,float(g.normalized_mse.mean())-best),oracle_nmse=best))
   for shape,z in g.groupby('shape'):
    assignments.append(dict(task_id=tid,family=family,shape=shape,min_nmse=float(z.normalized_mse.min()),max_nmse=float(z.normalized_mse.max()),mean_nmse=float(z.normalized_mse.mean()),assignment_spread=float(z.normalized_mse.max()-z.normalized_mse.min()),bound_spread=float(z.centered_cut_bound.max()-z.centered_cut_bound.min())))
   shape_min=g.groupby('shape').normalized_mse.min()
   winners=shape_min[shape_min<=shape_min.min()+OUTCOME_TOL].index.tolist()
   for shape in ['balanced','comb','mixed']:
    shapes.append(dict(task_id=tid,family=family,shape=shape,best_assignment_nmse=float(shape_min[shape]),oracle_shape_tied=shape in winners,oracle_shape_weight=(1/len(winners) if shape in winners else 0),n_tied_shapes=len(winners)))
   for perm,z in g.groupby('permutation_index'):
    z=z.set_index('shape')
    for left,right in [('comb','balanced'),('mixed','balanced'),('comb','mixed')]:
     paired.append(dict(task_id=tid,family=family,permutation_index=int(perm),left_shape=left,right_shape=right,nmse_left_minus_right=float(z.loc[left,'normalized_mse']-z.loc[right,'normalized_mse']),bound_left_minus_right=float(z.loc[left,'centered_cut_bound']-z.loc[right,'centered_cut_bound'])))
 policies=pd.DataFrame(policy_rows);policies.to_csv(OUT/'audit_policy_outcomes.csv',index=False)
 summary=policies.groupby(['family','policy'],as_index=False).agg(n_tasks=('task_id','nunique'),mean_nmse=('nmse','mean'),mean_regret=('regret','mean'),max_regret=('regret','max'),mean_score_tie_count=('score_tie_count','mean'),n_positive_regret=('regret',lambda x:int((x>OUTCOME_TOL).sum())))
 summary.to_csv(OUT/'audit_policy_summary.csv',index=False)
 pd.DataFrame(assignments).to_csv(OUT/'audit_within_shape_assignments.csv',index=False)
 pd.DataFrame(assignments).groupby(['family','shape'],as_index=False).agg(mean_assignment_spread=('assignment_spread','mean'),max_assignment_spread=('assignment_spread','max'),n_assignment_sensitive=('assignment_spread',lambda x:int((x>OUTCOME_TOL).sum())),mean_bound_spread=('bound_spread','mean')).to_csv(OUT/'audit_assignment_summary.csv',index=False)
 pd.DataFrame(shapes).to_csv(OUT/'audit_oracle_shapes.csv',index=False)
 pd.DataFrame(shapes).groupby(['family','shape'],as_index=False).agg(tied_win_count=('oracle_shape_tied','sum'),fractional_win_count=('oracle_shape_weight','sum')).to_csv(OUT/'audit_oracle_shape_summary.csv',index=False)
 p=pd.DataFrame(paired);p.to_csv(OUT/'audit_matched_assignment_shape_contrasts.csv',index=False)
 p.groupby(['family','left_shape','right_shape'],as_index=False).agg(n_task_assignment_pairs=('task_id','size'),mean_difference=('nmse_left_minus_right','mean'),min_difference=('nmse_left_minus_right','min'),max_difference=('nmse_left_minus_right','max'),n_left_worse=('nmse_left_minus_right',lambda x:int((x>OUTCOME_TOL).sum())),n_tied=('nmse_left_minus_right',lambda x:int((abs(x)<=OUTCOME_TOL).sum())),n_left_better=('nmse_left_minus_right',lambda x:int((x<-OUTCOME_TOL).sum()))).to_csv(OUT/'audit_matched_shape_summary.csv',index=False)
 trajectories=[]
 for key,g in raw.groupby(['task_id','candidate_id','restart']):
  v=g.sort_values('sweep').normalized_mse.to_numpy()
  trajectories.append(max(np.diff(v)))
 final=raw[raw.sweep.eq(32)]
 # A feasible small-coefficient alternative is a certificate, not an exclusion
 # or a replacement for the frozen best-restart endpoint.
 norm_certificates=[]
 for (tid,cid),z in final.groupby(['task_id','candidate_id']):
  best=z.sort_values(['normalized_mse','restart'],kind='stable').iloc[0]
  equivalent=z[z.normalized_mse<=float(best.normalized_mse)+OUTCOME_TOL]
  feasible=equivalent.sort_values(['final_parameter_norm','restart'],kind='stable').iloc[0]
  norm_certificates.append(dict(task_id=tid,candidate_id=cid,family=best.family,
   best_restart=int(best.restart),best_nmse=float(best.normalized_mse),
   best_restart_parameter_norm=float(best.final_parameter_norm),
   minimum_norm_equivalent_restart=int(feasible.restart),
   equivalent_nmse=float(feasible.normalized_mse),
   minimum_equivalent_parameter_norm=float(feasible.final_parameter_norm),
   n_equivalent_restarts=len(equivalent)))
 certificates=pd.DataFrame(norm_certificates)
 certificates.to_csv(OUT/'audit_norm_certificates.csv',index=False)
 norm_summary={'n_final_fits':len(final),'n_candidate_endpoints':len(certificates),
  'equivalent_endpoint_tolerance':OUTCOME_TOL,'scope':'Existing feasible alternative within endpoint tolerance; all original fits and best endpoints retained'}
 for threshold in [1e4,1e8]:
  tag=f'{threshold:.0e}'
  norm_summary[f'final_fits_norm_gt_{tag}']=int((final.final_parameter_norm>threshold).sum())
  norm_summary[f'best_restarts_norm_gt_{tag}']=int((certificates.best_restart_parameter_norm>threshold).sum())
  norm_summary[f'candidate_endpoints_without_equivalent_norm_le_{tag}']=int((certificates.minimum_equivalent_parameter_norm>threshold).sum())
 norm_summary['maximum_minimum_equivalent_norm']=float(certificates.minimum_equivalent_parameter_norm.max())
 chosen=policies[policies.policy.ne('uniform_random_expectation')].merge(certificates,on=['task_id','family','candidate_id'],validate='many_to_one')
 chosen.to_csv(OUT/'audit_selected_norm_certificates.csv',index=False)
 norm_summary['selected_policies']=chosen.groupby('policy').agg(n_selected=('task_id','size'),max_equivalent_norm=('minimum_equivalent_parameter_norm','max'),no_equivalent_norm_le_1e4=('minimum_equivalent_parameter_norm',lambda x:int((x>1e4).sum())),no_equivalent_norm_le_1e8=('minimum_equivalent_parameter_norm',lambda x:int((x>1e8).sum()))).reset_index().to_dict('records')
 dump('audit_norm_summary.json',norm_summary)
 old=pd.read_csv(OUT/'policy_outcomes.csv').merge(policies,on=['task_id','family','policy'],suffixes=('_original','_audit'))
 changed=old[old.candidate_id_original.ne(old.candidate_id_audit)&old.policy.ne('uniform_random_expectation')]
 changed.to_csv(OUT/'audit_tie_changed_choices.csv',index=False)
 # Exact regression checks for the stronger centered constraint.
 matching=tasks()[0]['coefficients'];quartet=next(t['coefficients'] for t in tasks() if t['family']=='quartic_partition')
 cases={}
 for name,coeff,subset in [('matching_crossed_pair',matching,(0,2)),('matching_even_odd',matching,(0,2,4,6)),('quartet_five_leaf',quartet,(0,1,2,3,4))]:
  m=matricize(coeff,subset);s=np.linalg.svd(m,compute_uv=False);c=np.linalg.svd(m[1:],compute_uv=False)
  cases[name]={'full_rank2_tail':float(np.sum(s[2:]**2)),'centered_rank1_tail':float(np.sum(c[1:]**2))}
 np.testing.assert_allclose([v['centered_rank1_tail'] for v in cases.values()],[.25,.75,.25],atol=1e-14)
 dump('audit_report.json',dict(source_sha256={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['protocol.json','candidate_outcomes.csv','all_trajectories.csv']},tie_absolute_tolerance=TOL,outcome_comparison_tolerance=OUTCOME_TOL,tie_rule='minimum score within absolute tolerance; lexicographic candidate ID',n_candidates=len(end),n_full_fits=len(final),n_changed_deterministic_choices=len(changed),max_recorded_sweep_error_increase=float(max(trajectories)),minimum_endpoint_minus_lower_bound=float(end.gap_above_lower_bound.min()),maximum_endpoint_minus_lower_bound=float(end.gap_above_lower_bound.max()),maximum_final_parameter_norm=float(final.final_parameter_norm.max()),maximum_parameter_norm=float(final.maximum_parameter_norm.max()),fit_diagnostic_scope='final whole-fit diagnostics repeated in each checkpoint row, not checkpoint-local values',gap_scope='gap above lower bound combines possible optimization error and bound looseness',input_sensitivity_scope='I/4 describes target input gradients, not model parameter-gradient moments',exact_theory_checks=cases))
 print(summary.to_string(index=False))
 print(pd.read_csv(OUT/'audit_matched_shape_summary.csv').to_string(index=False))
 print(pd.read_csv(OUT/'audit_assignment_summary.csv').to_string(index=False))
if __name__=='__main__':main()
