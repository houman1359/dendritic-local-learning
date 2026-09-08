#!/usr/bin/env python3
"""Independent arithmetic checks of retained seven-target simulation outcomes."""
from __future__ import annotations
from pathlib import Path
import itertools,json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
J=Path(__file__).resolve().parents[2];O=J/'source_data/measured_alignment_power'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=json.loads((O/'protocol_freeze.json').read_text());scan_index=pd.read_csv(O/'inputs/scan_index.csv',dtype={'target_root_id':'int64'});alltargets=np.sort(scan_index.target_root_id.unique())
 pmf=np.bincount(np.array(list(itertools.product([0,1],repeat=7)))@np.arange(1,8),minlength=29)/128;tails=np.cumsum(pmf)
 rows=[];maxagg=0.;maxp=0.;nvalues=0;ndisagreements=0;rank_ties=0
 for chunk in range(p['n_chunks']):
  f=O/'runs'/f'chunk_{chunk:02d}.npz';meta=json.loads(f.with_suffix('.json').read_text());assert sha(f)==meta['output_sha256'];q=np.load(f)
  scan=q['scan_effects'];saved=q['target_effects'];rep=q['replicate_ids'];computed=np.stack([scan[...,scan_index.target_root_id.to_numpy()==target].mean(-1) for target in alltargets],axis=-1)
  maxagg=max(maxagg,float(np.abs(computed-saved).max()));nvalues+=computed.size
  ranks=rankdata(np.abs(computed),axis=-1);rank_ties+=int(np.sum(np.any(ranks!=np.rint(ranks),axis=-1)));assert not np.any(computed==0)
  wneg=(ranks*(computed<0)).sum(-1);ww=np.minimum(wneg,28-wneg).astype(int);exact=np.minimum(1,2*tails[ww]);maxp=max(maxp,float(np.abs(exact-q['exact_two_sided_p']).max()))
  decision=(q['exact_two_sided_p']<=.05)&(computed.mean(-1)>0);simple=wneg<=2;ndisagreements+=int(np.sum(decision!=simple))
  for i,lam in enumerate(q['lambdas']):
   for j,scenario in enumerate(q['scenarios']):rows.append(dict(reliability=str(scenario),**{'lambda':float(lam)},detections=int(simple[i,j].sum()),n=len(rep),effect_sum=float(computed[i,j].mean(-1).sum())))
 t=pd.DataFrame(rows).groupby(['reliability','lambda'],as_index=False).sum(numeric_only=True);t['power']=t.detections/t.n;t['mean_effect']=t.effect_sum/t.n
 summary=pd.read_csv(O/'power_summary.csv');merged=t.merge(summary,on=['reliability','lambda'],suffixes=('_audit','_reported'),validate='one_to_one');dp=float(abs(merged.power_audit-merged.power_reported).max());de=float(abs(merged.mean_effect-merged.mean_target_effect).max())
 result=dict(status='PASS',method='Seven-target untied signed-rank P from the rank1..7subset-sum PMF; positive detection independently checked as negative-rank-sum<=2; scan means reaggregated by exact target IDs.',n_retained_target_statistics=nvalues,n_outcome_rows=len(merged),max_scan_aggregation_error=maxagg,max_exact_p_error=maxp,max_power_summary_error=dp,max_mean_effect_summary_error=de,n_decision_disagreements=ndisagreements,n_average_rank_tie_patterns=rank_ties)
 assert max(maxagg,maxp,dp,de)<1e-12 and ndisagreements==0 and rank_ties==0
 (O/'independent_arithmetic_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
