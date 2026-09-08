#!/usr/bin/env python3
"""Run a frozen response-level sensitivity simulation, after protocol approval."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import scipy
from model import load_dataset,statistic,aggregate,exact_signed_rank,exact_test_lattice
J=Path(__file__).resolve().parents[2];OUT=J/'source_data/measured_alignment_power'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def checked_protocol():
 p=OUT/'protocol_freeze.json';d=json.loads(p.read_text())
 for path,value in d['scientific_code_sha256'].items():
  assert sha(J/path)==value,f'Frozen scientific source changed: {path}'
 for row in json.loads((OUT/'input_manifest.json').read_text()):assert sha(OUT/row['released'])==row['released_sha256']
 assert sha(OUT/'input_manifest.json')==d['input_manifest_sha256'];return d

def run_chunk(chunk):
 p=checked_protocol();data=load_dataset(OUT/'inputs');n=p['replicates_per_chunk'];nchunks=p['n_chunks'];assert 0<=chunk<nchunks
 path=OUT/'runs'/f'chunk_{chunk:02d}.npz';meta=path.with_suffix('.json');path.parent.mkdir(exist_ok=True)
 if path.exists() or meta.exists():raise FileExistsError('An existing scientific chunk is never overwritten.')
 seed=p['primary_seed_base']+chunk;rng=np.random.default_rng(seed);start=time.time()
 shape=(n,len(data.hashes),len(data.roots));ind=rng.normal(size=shape);ancestry=rng.normal(size=shape)@data.factor.T
 noise=rng.normal(size=(n,len(data.hashes),len(data.recordings)))
 lambdas=np.asarray(p['ancestry_variance_fractions']);scenarios=p['reliability_scenarios'];scanvals=np.empty((len(lambdas),len(scenarios),n,len(data.scans)))
 for i,lam in enumerate(lambdas):
  latent=np.sqrt(1-lam)*ind+np.sqrt(lam)*ancestry
  for j,scenario in enumerate(scenarios):
   for k,s in enumerate(data.scans):
    clean=latent[:,s.stimuli][:,:,s.roots]
    if scenario=='perfect':y=clean
    else:
     eps=noise[:,s.stimuli][:,:,s.recordings]
     y=clean*np.sqrt(s.r)[None,None,:]+eps*np.sqrt((1-s.r)/s.h)[None,None,:]/np.sqrt(s.counts)[None,:,None]
    scanvals[i,j,:,k]=statistic(y,s)
 targets=aggregate(scanvals,data.scan_target,len(data.targets));ps=exact_signed_rank(targets)
 np.savez_compressed(path,scan_effects=scanvals,target_effects=targets,exact_two_sided_p=ps,lambdas=lambdas,scenarios=np.array(scenarios),target_root_ids=data.targets,replicate_ids=np.arange(chunk*n,(chunk+1)*n))
 record=dict(status='complete',chunk=chunk,seed=seed,started_utc=datetime.fromtimestamp(start,timezone.utc).isoformat(),finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.time()-start,slurm_job_id=os.environ.get('SLURM_JOB_ID'),slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'),protocol_sha256=sha(OUT/'protocol_freeze.json'),output_sha256=sha(path),python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,dataset=data.metadata,exact_test_lattice=exact_test_lattice())
 meta.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ['dataset','exact_test_lattice']},indent=2))

def calibration_audit():
 p=checked_protocol();data=load_dataset(OUT/'inputs');rng=np.random.default_rng(p['reliability_audit_seed']);n=p['reliability_audit_replicates'];rows=[]
 # This is a model-check of the declared Gaussian moment approximation; it never retunes r.
 for s in data.scans:
  latent=rng.normal(size=(n,len(s.counts),len(s.r)));nl=rng.normal(size=latent.shape);nr=rng.normal(size=latent.shape)
  sd=np.sqrt((1-s.r)/s.h)[None,None,:]/np.sqrt(s.counts/2)[None,:,None]
  left=latent*np.sqrt(s.r)[None,None,:]+nl*sd;right=latent*np.sqrt(s.r)[None,None,:]+nr*sd
  rl=rankdata(left,axis=1);rr=rankdata(right,axis=1);rl-=rl.mean(axis=1,keepdims=True);rr-=rr.mean(axis=1,keepdims=True)
  rho=np.sum(rl*rr,axis=1)/np.sqrt(np.sum(rl*rl,axis=1)*np.sum(rr*rr,axis=1))
  raw=np.load(OUT/'inputs'/s.name/'observed_partner_responses.npz');l=rankdata(raw['half_left'],axis=0);r=rankdata(raw['half_right'],axis=0);l-=l.mean(0);r-=r.mean(0);observed=np.sum(l*r,0)/np.sqrt(np.sum(l*l,0)*np.sum(r*r,0))
  for k in range(len(s.r)):rows.append(dict(scan=s.name,partner_index=k,measured_split_half_spearman=float(observed[k]),clipped_measured_spearman=float(max(0,observed[k])),pearson_signal_fraction=float(s.r[k]),simulated_mean_split_half_spearman=float(rho[:,k].mean()),simulated_se=float(rho[:,k].std(ddof=1)/np.sqrt(n)),n_simulated=n))
 import pandas as pd
 dest=OUT/'reliability_calibration_audit.csv'
 if dest.exists():raise FileExistsError(dest)
 pd.DataFrame(rows).to_csv(dest,index=False)
 (OUT/'reliability_calibration_audit.json').write_text(json.dumps(dict(protocol_sha256=sha(OUT/'protocol_freeze.json'),seed=p['reliability_audit_seed'],n_replicates=n,scope='Check only; no data-dependent adjustment of the declared Spearman-to-Pearson moment calibration.',output_sha256=sha(dest)),indent=2)+'\n')
 print('Reliability model-check complete; see retained values, without parameter retuning.')

def main():
 a=argparse.ArgumentParser();a.add_argument('--chunk',type=int);a.add_argument('--calibration-audit',action='store_true');args=a.parse_args()
 if args.calibration_audit:calibration_audit()
 elif args.chunk is not None:run_chunk(args.chunk)
 else:a.error('Choose --chunk or --calibration-audit')
if __name__=='__main__':main()
