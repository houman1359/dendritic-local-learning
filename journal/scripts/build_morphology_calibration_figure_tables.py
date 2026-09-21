#!/usr/bin/env python3
"""Pointwise seed intervals for absolute calibration-bridge test error."""
from pathlib import Path
import hashlib,json
import pandas as pd
from build_morphology_credit_figure_tables import bootstrap
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'source_data/morphology_calibration'

def main():
 source=OUT/'policy_outcomes.csv';frame=pd.read_csv(source);rows=[]
 for family in ['all',*sorted(frame.family.unique())]:
  part=frame if family=='all' else frame[frame.family.eq(family)]
  for keys,z in part.groupby(['calibration_rows','calibration_noise_sd','policy']):
   means=z.groupby('seed').test_nmse.mean().sort_index();assert len(means)==20
   mean,lo,hi=bootstrap(means)
   rows.append(dict(family=family,calibration_rows=keys[0],calibration_noise_sd=keys[1],policy=keys[2],mean_test_nmse=mean,ci95_low=lo,ci95_high=hi,n_seeds=20))
 pd.DataFrame(rows).to_csv(OUT/'figure_absolute_error_summary.csv',index=False)
 (OUT/'figure_table_provenance.json').write_text(json.dumps({'source':'source_data/morphology_calibration/policy_outcomes.csv','sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'builder':'scripts/build_morphology_calibration_figure_tables.py','intervals':'pointwise95%;10000 whole-seed bootstrap draws; family means first for pooled summaries; does not replace adjusted primary contrast intervals'},indent=2)+'\n')

if __name__=='__main__':main()
