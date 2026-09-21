#!/usr/bin/env python3
"""Post-review, read-only trajectory reconstruction for the depth comparison."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;J=HERE.parents[1];OUT=J/'source_data/physical_depth_followup';SRC=J/'source_data/physical_depth_budget/canonical'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def main():
 OUT.mkdir(exist_ok=True)
 h=pd.read_csv(SRC/'observed_training_histories.csv',float_precision='round_trip');m=pd.read_csv(SRC/'observed_epoch_metrics.csv',float_precision='round_trip');end=pd.read_csv(SRC/'extension_endpoints.csv',float_precision='round_trip')
 rows=[];stops=[]
 for (arm,depth,seed),g in h.groupby(['arm','depth','seed']):
  g=g.sort_values('epoch');lookup=m[(m.arm==arm)&(m.depth==depth)&(m.seed==seed)].set_index('epoch');gh=g.set_index('epoch');last=int(g.epoch.max())
  for epoch in range(1,601):
   record=gh.loc[min(epoch,last)];metric=lookup.loc[record.best_epoch]
   rows.append({'arm':arm,'depth':int(depth),'seed':int(seed),'epoch':epoch,'observed_epoch':epoch<=last,'last_training_epoch':last,'selected_epoch':int(record.best_epoch),'test_accuracy':float(metric.accuracy_test),'test_cross_entropy':float(-metric.categorical_loglikelihood_test),'best_validation_loss':float(record.best_loss),'checkpoint_selection':'Minimum recorded validation loss within budget; retain stopped run best state.'})
  stops.append({'arm':arm,'depth':int(depth),'seed':int(seed),'epochs_run':last,'stopped_before600':last<600,'reached_cap600':last==600})
 d=pd.DataFrame(rows);assert len(d)==36000 and len(stops)==60
 for r in end.itertuples():
  q=d[(d.arm==r.arm)&(d.depth==r.depth)&(d.seed==r.seed)&(d.epoch==r.budget)].iloc[0]
  assert abs(q.test_accuracy-r.test_accuracy)<1e-12
  assert abs(q.test_cross_entropy-r.test_loss)<1e-12
  assert abs(q.best_validation_loss-r.best_valid_loss)<1e-12
 d.to_csv(OUT/'validation_selected_seed_trajectories.csv',index=False);pd.DataFrame(stops).to_csv(OUT/'stopping_by_seed.csv',index=False)
 ix=np.random.default_rng(601806).integers(10,size=(10000,10));weights=np.array([np.bincount(i,minlength=10) for i in ix],float)/10
 summary=[]
 for (arm,depth),g in d.groupby(['arm','depth']):
  for metric in ['test_accuracy','test_cross_entropy','best_validation_loss']:
   w=g.pivot(index='seed',columns='epoch',values=metric).sort_index();v=w.to_numpy();b=weights@v;lo,hi=np.quantile(b,[.025,.975],axis=0)
   summary.extend({'arm':arm,'depth':int(depth),'metric':metric,'epoch':int(epoch),'mean':float(mean),'ci95_low':float(l),'ci95_high':float(u),'n_seeds':10} for epoch,mean,l,u in zip(w.columns,v.mean(axis=0),lo,hi))
 pd.DataFrame(summary).to_csv(OUT/'condition_trajectory_summary.csv',index=False)
 local=d[(d.depth==3)&d.arm.isin(['path_transport','per_soma_shared'])]
 gaps=[];gaprows=[]
 for metric in ['test_accuracy','test_cross_entropy','best_validation_loss']:
  w=local.pivot(index=['seed','epoch'],columns='arm',values=metric)
  values=(w.path_transport-w.per_soma_shared).unstack('epoch').sort_index();v=values.to_numpy()*(100 if metric=='test_accuracy' else 1)
  boot=weights@v;lo,hi=np.quantile(boot,[.025,.975],axis=0)
  gaps.extend({'metric':metric,'epoch':int(epoch),'mean':float(mean),'ci95_low':float(l),'ci95_high':float(u),'n_seeds':10,'positive_seeds':int((v[:,k]>0).sum()),'negative_seeds':int((v[:,k]<0).sum())} for k,(epoch,mean,l,u) in enumerate(zip(values.columns,v.mean(axis=0),lo,hi)))
  gaprows.extend({'metric':metric,'seed':int(seed),'epoch':int(epoch),'exact_minus_shared':float(v[i,k])} for i,seed in enumerate(values.index) for k,epoch in enumerate(values.columns))
 gaps=pd.DataFrame(gaps);gaps.to_csv(OUT/'paired_trajectory_summary.csv',index=False);pd.DataFrame(gaprows).to_csv(OUT/'paired_seed_trajectories.csv',index=False)
 def cross(series):
  s=series[series.index>=180];negative=s[s<0];nonnegative=s[s>=0]
  return {'first_negative_after180':int(negative.index[0]),'negative_through600_from':int(nonnegative.index.max()+1) if len(nonnegative) else 180}
 a=gaps[gaps.metric=='test_accuracy'].set_index('epoch');ce=gaps[gaps.metric=='test_cross_entropy'].set_index('epoch');st=pd.DataFrame(stops)
 depth=[]
 for budget in [180,600]:
  w=d[(d.arm=='exact_autograd_bp_recipe')&(d.epoch==budget)].pivot(index='seed',columns='depth',values='test_accuracy');v=100*(w[3]-w[1]).to_numpy();boot=weights@v
  depth.append({'budget':budget,'mean_pp':float(v.mean()),'ci95_low_pp':float(np.quantile(boot,.025)),'ci95_high_pp':float(np.quantile(boot,.975))})
 report={'status':'passed','scope':'Post-review existing-data analysis of all sixty released restarts; no new training and no test-based checkpoint selection. Pointwise whole-seed intervals,10000 draws,seed601806,are descriptive and not simultaneous confidence bands.','all120_released_budget_endpoints_reproduced':True,'seed_trajectory_rows':len(d),'d1_stopped_before600':int(st[(st.depth==1)].stopped_before600.sum()),'d1_n':int((st.depth==1).sum()),'d3_reached_cap600':int(st[(st.depth==3)].reached_cap600.sum()),'d3_n':int((st.depth==3).sum()),'accuracy_crossing':cross(a['mean']),'accuracy_most_negative_epoch':int(a['mean'].idxmin()),'accuracy_most_negative_mean_pp':float(a['mean'].min()),'accuracy_budgets':a.loc[[180,315,486,500,600]].reset_index().to_dict('records'),'cross_entropy_budgets':ce.loc[[180,315,486,500,600]].reset_index().to_dict('records'),'exact_lower_mean_cross_entropy_all180_to600':bool((ce.loc[180:600,'mean']<0).all()),'forward_depth_advantage':depth,'inputs_sha256':{str(p.relative_to(J)):sha(p) for p in [SRC/'observed_training_histories.csv',SRC/'observed_epoch_metrics.csv',SRC/'extension_endpoints.csv']},'analysis_script_sha256':sha(__file__)}
 oldpath=OUT/'retained_main_figure06_2a298a9.pdf'
 if not oldpath.exists():oldpath.write_bytes(subprocess.check_output(['git','show','2a298a9:journal/figures/main/figure_06.pdf'],cwd=J))
 report['retained_main_figure06_sha256']=sha(oldpath);write(OUT/'analysis_validation.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
