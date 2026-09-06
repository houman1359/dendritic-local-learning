#!/usr/bin/env python3
"""Seed-block descriptive intervals for the completed morphology-credit study."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'source_data/morphology_credit'

def bootstrap(x):
 x=np.asarray(x,float);rng=np.random.default_rng(202609052)
 means=x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1)
 return float(x.mean()),*map(float,np.quantile(means,[.025,.975]))

def main():
 p=SOURCE/'summaries/fresh/selected_endpoints.csv';end=pd.read_csv(p)
 q=SOURCE/'summaries/fresh/selected_trajectories.csv';traj=pd.read_csv(q)
 rows=[]
 for keys,z in end.groupby(['family','structure','optimizer','rule']):
  values=z.sort_values('seed').test_nmse
  assert len(values)==20
  mean,lo,hi=bootstrap(values)
  rows.append(dict(zip(['family','structure','optimizer','rule'],keys))|dict(mean=mean,ci95_low=lo,ci95_high=hi,n_seeds=20))
 pd.DataFrame(rows).to_csv(SOURCE/'figure_condition_summary.csv',index=False)
 rows=[]
 for keys,z in traj.groupby(['structure','optimizer','rule','step']):
  values=z.groupby('seed').gradient_cosine.mean().sort_index()
  assert len(values)==20
  mean,lo,hi=bootstrap(values)
  rows.append(dict(zip(['structure','optimizer','rule','step'],keys))|dict(mean=mean,ci95_low=lo,ci95_high=hi,n_seeds=20))
 pd.DataFrame(rows).to_csv(SOURCE/'figure_gradient_summary.csv',index=False)
 record={'script':'scripts/build_morphology_credit_figure_tables.py','inputs':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [p,q]},'intervals':'pointwise 95% bootstrap; 10000 whole-seed draws; families averaged within seed for gradient summary','gradient_scope':'cosine between aggregate clean-population gradients at each learner own state; not mean per-example cosine','n_seed_blocks':20}
 (SOURCE/'figure_table_provenance.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
