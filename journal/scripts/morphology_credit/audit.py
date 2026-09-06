#!/usr/bin/env python3
"""Validate the frozen credit experiment and export common-rate sensitivity."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import OUT,freeze,collect,sha

cfg=freeze();seal=json.loads((OUT/'selection_freeze.json').read_text())
assert sha(OUT/'development_fit.json')==seal['fit_sha256']
for name,digest in seal['source_files'].items():assert sha(OUT/name)==digest
z=collect('fresh')
assert len(z)==20*3*2*2*3*5*6
keys=['seed','family','structure','optimizer','rate','rule','step']
assert not z.duplicated(keys).any()
assert np.isfinite(z.select_dtypes('number')).all().all()
assert set(z.seed)==set(cfg['fresh_seeds'])
for seed in cfg['fresh_seeds']:
    a=json.loads((OUT/'runs/fresh'/f'seed_{seed}_audit.json').read_text())
    assert a['rows']==len(z[z.seed==seed])
    assert a['selection_sha256']==sha(OUT/'selection_freeze.json')
    assert a['runner_sha256']==sha(Path(__file__).with_name('experiment.py'))
assert z.max_abs_parameter.max()<=2.
choices=json.loads((OUT/'development_fit.json').read_text())
common=z[(z.step==1024)&np.isclose(z.rate,[choices[o]['common_rate'] for o in z.optimizer])]
common.to_csv(OUT/'summaries/fresh/common_rate_endpoints.csv',index=False)
common.groupby(['family','structure','optimizer','rule'],as_index=False).agg(mean_test_nmse=('test_nmse','mean'),n_seeds=('seed','nunique')).to_csv(OUT/'summaries/fresh/common_rate_summary.csv',index=False)
record={'status':'pass','fresh_fits':3600,'checkpoint_rows':len(z),'selected_rate_fits':1200,'seed_blocks':20,'failures':0,'all_rates_retained':True,'fresh_selections_match_frozen_development':True,'parameter_max':float(z.max_abs_parameter.max()),'inference':'Pointwise descriptive paired-seed intervals; multiple subgroup contrasts not separate confirmed discoveries','scope':'No target-derived initialization; oracle-informed architecture and projected-field coefficients are explicit privileged inputs','sources':{str(p.relative_to(OUT)):sha(p) for p in sorted((OUT/'runs/fresh').glob('*.csv'))}}
(OUT/'validation.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='sources'},indent=2))
