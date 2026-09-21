#!/usr/bin/env python3
"""Excluded development-seed execution benchmark; never enters science tables."""
import json
import importlib.util
from pathlib import Path
import platform
import sys
import time
import numpy as np
import run

spec=importlib.util.spec_from_file_location('credit_extension_benchmark_reference',run.HERE.parent/'credit_rule_bridge/run.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
p=run.check_freeze();cfg=run.original_config();start=time.perf_counter();records=[]
for task in p['tasks']:
 t=time.perf_counter();rows,diag,arrays,meta=reference.run_task(210100,'algebraic',task,'fresh',cfg,steps=256)
 assert np.isfinite(arrays['theta']).all()
 records.append({'task':task,'steps':256,'trajectories':len(meta['records']),'seconds':time.perf_counter()-t,'finite_states':True})
run.write(run.OUT/'excluded_runtime_smoke.json',{'excluded_from_science':True,'seed':210100,'reason':'Historical development seed used only for execution and runtime estimation; no choices depend on outcomes.','records':records,'total_seconds':time.perf_counter()-start,'linear_estimate_seconds_per_scientific_seed_16384':(time.perf_counter()-start)*64,'created_utc':run.now(),'python':platform.python_version(),'numpy':np.__version__,'script_sha256':run.sha(__file__),'protocol_sha256':run.sha(run.OUT/'protocol_freeze.json')})
print((run.OUT/'excluded_runtime_smoke.json').read_text(),flush=True)
