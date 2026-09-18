"""Freeze the corrected selection task before fresh-seed outcomes exist."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from run import save_new, sha


def jobs(seeds):
    return [dict(seed=s, variant=v, mode=m) for s in seeds
            for v,m in [('separable','shunt'),('separable','tonic'),('separable','current'),('interaction','shunt')]]


def prepare(root,parent):
    root.mkdir(parents=True,exist_ok=False)
    (root/'study').mkdir()
    for source in Path(__file__).resolve().parent.glob('*'):
        if source.suffix in {'.py','.sh','.md'}:
            shutil.copyfile(source,root/'study'/source.name)
    (root/'runtime').symlink_to(parent/'runtime',target_is_directory=True)
    (root/'base').symlink_to(parent/'study',target_is_directory=True)
    old=json.loads((parent/'protocol.json').read_text())
    hashes={k:v for k,v in old['source_sha256'].items() if k.startswith('runtime/')}
    hashes.update({f'base/{n}':sha(root/'base'/n) for n in ['model.py','run.py']})
    hashes.update({str(p.relative_to(root)):sha(p) for p in (root/'study').glob('*') if p.suffix in {'.py','.sh'}})
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  status='Internally specified development following the original pilot; not externally preregistered',
                  original_protocol_sha256=sha(parent/'protocol.json'),source_sha256=hashes,
                  jobs=jobs(range(2026092100,2026092103)),rates=[.01,.03,.1],steps=2048,
                  fresh_seeds=list(range(2026092200,2026092220)),fresh_steps=4096,
                  primary='Paired broadcast minus resistance-gate NMSE at severity 3, separable target, shunting forward',
                  primary_control='Paired uniform-RMS minus resistance-gate NMSE under the same conditions',
                  secondary=['Wrong-branch control','Exact-BP reference','Forward shunt/tonic/current factorial',
                             'Severity 1–3 curve','Equal-norm common-state core steps','Nonlinear-parent interaction task'],
                  inference='20 fresh paired seeds; seed bootstrap intervals and two-sided paired sign-flip tests, Holm across two primary contrasts',
                  learning_rate_selection='Minimum mean development-validation NMSE per variant/forward/rule; test and OOD never select settings',
                  interpretation=['All five rules retained regardless of direction; secondary interaction study remains secondary',
                    'Input amplitude stress is synthetic; exp-encoded irrelevant features broaden beyond their training range',
                    'Nonlinear-parent sensitivity changes representation; do not attribute differences against the old pilot solely to the target',
                    'Parent relative-resistance gating omits the parent activation derivative in the nonlinear arm',
                    'Bound [-9,9], fixed 4096-step fresh budget; neither statistical equivalence nor asymptotic optimality is assumed'])
    save_new(root/'development_protocol.json',protocol)
    for phase in ['development','fresh']:
        for name in ['results','checkpoints','logs']:(root/phase/name).mkdir(parents=True)
    print(root,len(protocol['jobs']))


def freeze(root):
    p=json.loads((root/'development_protocol.json').read_text())
    assert not list((root/'fresh/results').glob('*.json'))
    choices={};hashes={}
    for variant,mode in [('separable','shunt'),('separable','tonic'),('separable','current'),('interaction','shunt')]:
        choices.setdefault(variant,{})[mode]={}
        for rule in ['exact','broadcast','resistance','swapped','uniform_rms']:
            values={}
            for rate in p['rates']:
                rows=[]
                for seed in range(2026092100,2026092103):
                    path=root/'development/results'/f's{seed}_{variant}_{mode}_{rule}_r{rate:g}.json'
                    r=json.loads(path.read_text())
                    assert r['protocol_sha256']==sha(root/'development_protocol.json')
                    rows.append(r['validation_nmse']);hashes[str(path.relative_to(root))]=sha(path)
                values[rate]=float(np.mean(rows))
            choices[variant][mode][rule]=min(values,key=values.get)
    fresh={**p,'created_utc':datetime.now(timezone.utc).isoformat(),
           'status':'Fresh-seed evaluation of internally frozen development-selected settings',
           'jobs':jobs(p['fresh_seeds']),'steps':p['fresh_steps'],'fixed_rates':choices,
           'development_result_sha256':hashes,'development_protocol_sha256':sha(root/'development_protocol.json')}
    save_new(root/'fresh_protocol.json',fresh)
    print(json.dumps(choices,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','freeze'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--parent',type=Path)
    a=p.parse_args()
    prepare(a.root,a.parent) if a.action=='prepare' else freeze(a.root)
