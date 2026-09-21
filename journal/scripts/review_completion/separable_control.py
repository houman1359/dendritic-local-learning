"""Hold nonlinear parents fixed while removing the target interaction.

Reuse the rescue seeds and common Adam rate as a paired sensitivity; these are
not new independent seed blocks. The frozen rescue trainer itself is unedited.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
import rescue
from experiment import dataset


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def separable_data(seed, split, size, variant='interaction', severity=1.):
    return dataset(seed, split, size, 'separable', severity)


def run(item):
    root, seed, rule = item
    torch.set_num_threads(1)
    p=json.loads((root/'protocol.json').read_text())
    for source, digest in p['source_sha256'].items(): assert sha(source)==digest
    # Only the target adapter changes. RescueNet still installs tanh parents.
    # Target-interaction diagnostics are inapplicable when beta=0.
    with patch.object(rescue,'dataset',separable_data), patch.object(rescue,'diagnostics',lambda *args:[]):
        r,states=rescue.train(seed,rule,'adam',9,.03,4096,'fresh')
    state_path=root/'states'/f'{seed}_{rule}.pt'
    assert not state_path.exists()
    torch.save(states,state_path)
    net=rescue.RescueNet(seed).double();net.load_state_dict(states['selected'])
    v=rescue.evaluate(net,separable_data(seed,'test',4096))
    assert abs(v-r['test_nmse'])<1e-12
    r.update(task='separable target, nonlinear parents',beta=0.,
        checkpoint_sha256=sha(state_path),protocol_sha256=sha(root/'protocol.json'),replay_error=abs(v-r['test_nmse']))
    with (root/'results'/f'{seed}_{rule}.json').open('x') as f: json.dump(r,f,indent=2)
    print(seed,rule,r['test_nmse'],flush=True)


def main(a):
    prior=json.loads((a.archive/'fresh_protocol.json').read_text())
    if a.action=='prepare':
        a.root.mkdir(exist_ok=False);(a.root/'states').mkdir();(a.root/'results').mkdir()
        sources={str(a.archive/rel):digest for rel,digest in prior['source_sha256'].items()}
        sources[str(Path(__file__).resolve())]=sha(__file__)
        p=dict(created_utc=datetime.now(timezone.utc).isoformat(),
            scope='Post-review paired task sensitivity; original rescue seed blocks reused, not independent replication',
            seeds=prior['fresh_seeds'],rules=['exact','broadcast','resistance','derivative','shuffled_derivative'],
            optimizer='Adam',rate=.03,bounds=[-9,9],steps=4096,beta=0.,parent='tanh',
            selection='Same common rate chosen before the original rescue outcomes; checkpoint selection by separable validation only',
            comparison='Separate parent nonlinearity from target interactions; no outcome-based retuning',source_sha256=sources)
        (a.root/'protocol.json').write_text(json.dumps(p,indent=2))
    elif a.action=='run':
        p=json.loads((a.root/'protocol.json').read_text())
        with ProcessPoolExecutor(a.workers) as pool: list(pool.map(run,[(a.root,s,r) for s in p['seeds'] for r in p['rules']]))
    else:
        p=json.loads((a.root/'protocol.json').read_text());rows=[]
        for s in p['seeds']:
            for rule in p['rules']:
                r=json.loads((a.root/'results'/f'{s}_{rule}.json').read_text())
                assert sha(a.root/'states'/f'{s}_{rule}.pt')==r['checkpoint_sha256']
                assert r['protocol_sha256']==sha(a.root/'protocol.json')
                rows.append({k:r[k] for k in ['seed','rule','rate','selected_step','test_nmse','bounds','selected_bound_fraction','replay_error']})
        frame=pd.DataFrame(rows);frame.to_csv(a.root/'nonlinear_separable_endpoints.csv',index=False)
        rng=np.random.default_rng(2026092102);contrasts=[]
        wide=frame.pivot(index='seed',columns='rule',values='test_nmse')
        for rule in ['resistance','shuffled_derivative','broadcast','exact']:
            d=wide[rule]-wide.derivative;bs=d.to_numpy()[rng.integers(len(d),size=(10000,len(d)))].mean(1)
            contrasts.append(dict(left=rule,right='derivative',mean=d.mean(),ci_low=np.quantile(bs,.025),ci_high=np.quantile(bs,.975),positive=int((d>0).sum()),n=len(d)))
        pd.DataFrame(contrasts).to_csv(a.root/'nonlinear_separable_contrasts.csv',index=False)
        (a.root/'nonlinear_separable_provenance.json').write_text(json.dumps(dict(protocol=p,outputs={p.name:sha(p) for p in a.root.glob('*.csv')}),indent=2))
        print(frame.groupby('rule').test_nmse.mean().to_string());print(pd.DataFrame(contrasts).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze'])
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=12)
    main(p.parse_args())
