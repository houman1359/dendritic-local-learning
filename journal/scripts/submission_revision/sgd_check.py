"""Bounded plain-SGD development check; fresh comparison only after success.

Uses the frozen rescue implementation and no development test evaluation.
The success criterion and follow-up recipes are specified before execution.
"""
from pathlib import Path
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def prepare(root, archive):
    old=json.loads((archive/'fresh_protocol.json').read_text())
    for rel,digest in old['source_sha256'].items():
        assert sha(archive/rel)==digest,rel
    root.mkdir(parents=True,exist_ok=False)
    for name in ['study','development/results','development/states','development/logs','fresh/results','fresh/states','fresh/logs']:
        (root/name).mkdir(parents=True)
    shutil.copy2(__file__,root/'study/sgd_check.py')
    for name,target in [('rescue',archive/'study'),('selection',archive/'selection'),('base',archive/'base'),('runtime',archive/'runtime')]:
        (root/name).symlink_to(target.resolve(),target_is_directory=True)
    sources={str((archive/rel).resolve()):digest for rel,digest in old['source_sha256'].items()}
    sources[str(root/'study/sgd_check.py')]=sha(root/'study/sgd_check.py')
    seeds=old['development_seeds'];rates=[.003,.01,.03,.1,.3,1.]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Internally specified post-review optimization check; reuses original development seeds, no held-out test evaluation',
        development_seeds=seeds,fresh_seeds=list(range(2026100300,2026100320)),
        rates=rates,steps=32768,rule='exact',optimizer='plain SGD, no momentum',bound=9,
        task='Frozen nonlinear-parent interaction task and rescue forward model, beta .25',
        data='Same 2048 training,1024 validation,128-example minibatches and initialization as original rescue',
        checkpoints='Validation every128 updates including initialization; best checkpoint retained',
        success='At least one rate gives validation-selected NMSE <=0.001 in all three development seeds at32768 updates',
        decision='If success, choose successful rate minimizing mean validation NMSE, ties choose smaller rate; run four rules at that common rate/budget on20 new paired seeds. If none succeeds, stop after development and retain all failures.',
        fresh_rules=['exact','resistance','derivative','shuffled_derivative'],
        fresh_primary='Resistance-minus-augmented and shuffled-minus-augmented ordinary test NMSE;95% paired bootstrap;exact sign flips with Holm across two comparisons',
        safeguards=['No W&B','kempner_project_b only','No old checkpoint modification','No fresh test evaluation before recipe freezing','All numerical failures retained'],
        source_sha256=sources,jobs=[dict(seed=s,rate=r) for s in seeds for r in rates])
    write_new(root/'development_protocol.json',protocol)
    print(len(protocol['jobs']),'development fits')


def run_one(root,phase,index):
    import torch
    import rescue
    import numpy as np
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=json.loads((root/f'{phase}_protocol.json').read_text())
    for path,digest in p['source_sha256'].items():assert sha(path)==digest,path
    job=p['jobs'][index];key=f"s{job['seed']}_r{job['rate']:g}_{job.get('rule','exact')}"
    path=root/phase/'results'/f'{key}.json';assert not path.exists()
    result=dict(**job,phase=phase,protocol_sha256=sha(root/f'{phase}_protocol.json'),job_id=os.environ.get('SLURM_JOB_ID'))
    try:
        r,states=rescue.train(job['seed'],job.get('rule','exact'),'sgd',9,job['rate'],p['steps'],phase)
        if phase=='development':assert 'test_nmse' not in r
        statepath=root/phase/'states'/f'{key}.pt';assert not statepath.exists()
        torch.save(states,statepath)
        result.update(r,status='completed',checkpoint_sha256=sha(statepath))
    except FloatingPointError as error:
        result.update(status='numerical_failure',reason=str(error))
    write_new(path,result)
    print(key,result['status'],result.get('validation_nmse'),flush=True)


def batch(root,phase):
    p=json.loads((root/f'{phase}_protocol.json').read_text())
    def one(index):
        with (root/phase/'logs'/f'{index}.log').open('x') as handle:
            subprocess.run([sys.executable,'-B',str(root/'study/sgd_check.py'),'one','--root',str(root),'--phase',phase,'--index',str(index)],stdout=handle,stderr=subprocess.STDOUT,check=True)
        print('completed',phase,index,flush=True)
    with concurrent.futures.ThreadPoolExecutor(int(os.environ.get('SLURM_CPUS_PER_TASK','1'))) as pool:
        list(pool.map(one,range(len(p['jobs']))))


def freeze(root):
    import numpy as np
    p=json.loads((root/'development_protocol.json').read_text());rows=[];scores=[]
    for job in p['jobs']:
        path=root/'development/results'/f"s{job['seed']}_r{job['rate']:g}_exact.json"
        r=json.loads(path.read_text());assert r['protocol_sha256']==sha(root/'development_protocol.json')
        rows.append(r)
    for rate in p['rates']:
        group=[r for r in rows if r['rate']==rate]
        valid=all(r['status']=='completed' for r in group)
        scores.append(dict(rate=rate,mean_validation=float(np.mean([r['validation_nmse'] for r in group])) if valid else None,
            success=valid and all(r['validation_nmse']<=.001 for r in group),completed=sum(r['status']=='completed' for r in group)))
    successful=[r for r in scores if r['success']]
    decision=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(root/'development_protocol.json'),scores=scores,
        development_results_sha256={x.name:sha(x) for x in sorted((root/'development/results').glob('*.json'))},success=bool(successful))
    if successful:
        rate=min(successful,key=lambda r:(r['mean_validation'],r['rate']))['rate'];decision['selected_rate']=rate
        fresh=dict(p,scope='Fresh paired follow-up after successful validation-only plain-SGD check',
            jobs=[dict(seed=s,rate=rate,rule=r) for s in p['fresh_seeds'] for r in p['fresh_rules']],
            selected_rate=rate,development_results_sha256=decision['development_results_sha256'])
        write_new(root/'fresh_protocol.json',fresh)
    write_new(root/'decision.json',decision);print(json.dumps(decision,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','batch','one','freeze'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',type=Path)
    p.add_argument('--phase',choices=['development','fresh'],default='development');p.add_argument('--index',type=int)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.archive)
    elif a.action=='batch':batch(a.root,a.phase)
    elif a.action=='one':run_one(a.root,a.phase,a.index)
    else:freeze(a.root)
