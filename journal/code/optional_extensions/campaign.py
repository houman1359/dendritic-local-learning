"""Freeze, run and select bounded extension experiments without test leakage."""
from __future__ import annotations
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def key(job):return f"{job['study']}_{job['arm']}_s{job['seed']}_r{job['rate']:g}"


def freeze(root):
    import experiments as e
    # These disjoint seed sets are fixed before any development run.
    seeds={s:dict(development=list(range(2026121000+100*i,2026121003+100*i)),
                  fresh=list(range(2026125000+100*i,2026125020+100*i))) for i,s in enumerate(e.ARMS)}
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Prospective optional extensions; separate from all published cohorts',
        studies=e.ARMS,seeds=seeds,rates=[.01,.03,.1],steps=4096,
        train_size=2048,validation_size=1024,test_size=4096,batch_size=128,
        optimizer='Adam, default betas/epsilon, no decay; global clipping 10; original conductance log bounds +/-9',
        selection='Minimum mean validation NMSE over three development seeds, independently per arm; checkpoint minimum validation NMSE every 128 steps including initialization; no development test evaluation',
        proxy='Original nonlinear interaction task and forward network. Noisy tanh-input voltage SD .25/.5/1; 2 bins with |V| threshold .5 and centers .25/.75; 4 bins with thresholds .25/.5/.75 and centers .125/.375/.625/.875. Bins span the analytic reachable range |V|<1. Gains use 1-tanh(center)^2. Shuffles independently per parent within cue; context means remove variation while retaining minibatch mean.',
        routing='Original nonlinear task, but terminal excitation cue set to zero in every arm. Cue only supplies a softmax 4x4 context-to-route map. Total inhibitory activity fixed at 12. Learned router uses the same local parent error unless explicitly supplied exact task gradient. Oracle, uniform and wrong-route controls have the same forward family.',
        temporal='Eight independent recurrent leaky traces, one per original feature; alpha=sigmoid(raw), gain=exp(log), initial alpha .5/gain1. Teacher alpha .8/gain1; zero initial state, 8 uniform-input steps; multiply terminal state by3 before original sensory exp encoding. Cue at readout only. Exact online eligibility versus one-step eligibility share forward memory. No-memory arm alpha0 fixed. Additional lengths4/16 are descriptive generalization.',
        primary_policy={'proxy':'common rate 0.03','routing':'development-selected rate','temporal':'development-selected rate'},
        primary_contrasts=[['proxy','bins4','resistance'],['proxy','bins4','shuffle_bins4'],
                           ['proxy','noise05','resistance'],['proxy','noise05','shuffle_noise05'],
                           ['routing','learned_local_augmented','uniform_augmented'],
                           ['routing','learned_local_augmented','learned_local_resistance'],
                           ['temporal','augmented_trace','resistance_trace'],
                           ['temporal','augmented_trace','augmented_one_step']],
        statistics='Mean ordinary-test NMSE and paired differences; paired whole-seed bootstrap 95% intervals, 20000 draws. Exact two-sided sign-flip P on paired differences, Holm over all eight prespecified contrasts. All remaining comparisons descriptive. Same seeds reused across policies; no independent-replication claim.',
        thresholds='No outcome-triggered exclusions or extended budgets; report failed/nonfinite fits and pause; no scientific success threshold used to select results',
        frozen_runtime_identity_sha256=sha(e.FROZEN/'identity.json'),
        source_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.py'))},
        development_amendment=json.loads((root/'range_amendment.json').read_text()) if (root/'range_amendment.json').exists() else None,
        environment={'python':sys.version,'torch':e.torch.__version__,'numpy':e.np.__version__},
        jobs=[dict(study=s,arm=a,seed=n,rate=r) for s,arms in e.ARMS.items() for a,n,r in itertools.product(arms,seeds[s]['development'],[.01,.03,.1])])
    save(root/'development_protocol.json',protocol)
    print('Frozen',len(protocol['jobs']),'development fits',flush=True)


def worker(root,phase,index):
    import experiments as e
    e.torch.set_num_threads(1);e.torch.set_num_interop_threads(1)
    p=root/f'{phase}_protocol.json';proto=json.loads(p.read_text())
    for name,digest in proto['source_sha256'].items():
        if sha(HERE/name)!=digest:raise RuntimeError('Execution source drift: '+name)
    if sha(e.FROZEN/'identity.json')!=proto['frozen_runtime_identity_sha256']:raise RuntimeError('Runtime drift')
    job=proto['jobs'][index];name=key(job)
    output=root/phase/'results'/f'{name}.json';checkpoint=root/phase/'checkpoints'/f'{name}.pt'
    if output.exists() or checkpoint.exists():raise FileExistsError(name)
    result,states=e.train(**job,steps=proto['steps'],phase=phase)
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    e.torch.save(states,checkpoint)
    result.update(protocol_sha256=sha(p),checkpoint_sha256=sha(checkpoint),allocation=os.environ.get('SLURM_JOB_ID'),
                  python=sys.version,torch=e.torch.__version__,numpy=e.np.__version__)
    save(output,result)
    print(name,result['validation_nmse'],result.get('test_nmse'),result['elapsed_seconds'],flush=True)


def run_phase(root,phase,workers):
    proto=json.loads((root/f'{phase}_protocol.json').read_text());started=time.monotonic()
    def run(index):
        job=proto['jobs'][index];name=key(job);out=root/phase/'results'/f'{name}.json'
        if out.exists():
            previous=json.loads(out.read_text())
            assert previous['protocol_sha256']==sha(root/f'{phase}_protocol.json')
            assert sha(root/phase/'checkpoints'/f'{name}.pt')==previous['checkpoint_sha256']
            return index
        log=root/phase/'logs'/f'{name}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('x') as handle:
            subprocess.run([sys.executable,'-B',str(HERE/'campaign.py'),'--root',str(root),'--action','worker','--phase',phase,'--index',str(index)],stdout=handle,stderr=subprocess.STDOUT,check=True)
        return index
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(run,i) for i in range(len(proto['jobs']))]
        for done,future in enumerate(concurrent.futures.as_completed(futures),1):
            index=future.result()
            print(phase,done,'/',len(futures),key(proto['jobs'][index]),'elapsed',round(time.monotonic()-started,1),flush=True)


def select(root):
    import numpy as np
    dev=json.loads((root/'development_protocol.json').read_text());selection={};inputs={}
    for study,arms in dev['studies'].items():
        selection[study]={}
        for arm in arms:
            values={}
            for rate in dev['rates']:
                group=[]
                for seed in dev['seeds'][study]['development']:
                    p=root/'development/results'/(key(dict(study=study,arm=arm,seed=seed,rate=rate))+'.json')
                    result=json.loads(p.read_text());assert 'test_nmse' not in result
                    assert result['protocol_sha256']==sha(root/'development_protocol.json')
                    group.append(result['validation_nmse']);inputs[str(p.relative_to(root))]=sha(p)
                values[rate]=float(np.mean(group))
            chosen=min(values,key=lambda r:(values[r],r))
            selection[study][arm]={'rate':chosen,'validation_means':values}
    jobs=[]
    for study,arms in dev['studies'].items():
        for arm in arms:
            rates=sorted({.03,selection[study][arm]['rate']})
            for seed,rate in itertools.product(dev['seeds'][study]['fresh'],rates):jobs.append(dict(study=study,arm=arm,seed=seed,rate=rate))
    fresh={**dev,'created_utc':datetime.now(timezone.utc).isoformat(),'development_protocol_sha256':sha(root/'development_protocol.json'),
           'selection':selection,'development_results_sha256':inputs,'jobs':jobs}
    save(root/'fresh_protocol.json',fresh)
    print('Frozen',len(jobs),'fresh fits after development-only selection',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--action',choices=['freeze','worker','run','select','all'],required=True)
    p.add_argument('--phase',choices=['development','fresh'],default='development');p.add_argument('--index',type=int)
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if a.action=='freeze':freeze(a.root)
    elif a.action=='worker':worker(a.root,a.phase,a.index)
    elif a.action=='run':run_phase(a.root,a.phase,a.workers)
    elif a.action=='select':select(a.root)
    else:
        run_phase(a.root,'development',a.workers);select(a.root);run_phase(a.root,'fresh',a.workers)
