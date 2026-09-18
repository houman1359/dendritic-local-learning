"""Secondary common-rate control; does not replace the frozen primary test."""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from experiment import train, SelectionNet, dataset
from run import sha, save_new, evaluate


def prepare(root):
    protocol=json.loads((root/'fresh_protocol.json').read_text())
    p=dict(created_utc=datetime.now(timezone.utc).isoformat(),
           scope='Secondary common-rate sensitivity specified during fresh execution; does not change primary tests or select settings',
           source_sha256=sha(__file__),fresh_protocol_sha256=sha(root/'fresh_protocol.json'),
           seeds=protocol['fresh_seeds'],rules_rates=[['resistance',.03],['broadcast',.1],['uniform_rms',.1]])
    for name in ['results','checkpoints','logs']:(root/'sensitivity'/name).mkdir(parents=True,exist_ok=True)
    save_new(root/'sensitivity/protocol.json',p)


def fit(root,index):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    p=json.loads((root/'sensitivity/protocol.json').read_text())
    assert sha(__file__)==p['source_sha256']
    seed=p['seeds'][index]
    for rule,rate in p['rules_rates']:
        key=f's{seed}_{rule}_r{rate:g}'
        out=root/'sensitivity/results'/f'{key}.json'
        checkpoint=root/'sensitivity/checkpoints'/f'{key}.pt'
        assert not out.exists() and not checkpoint.exists()
        result,state=train(seed,'separable','shunt',rule,rate,4096)
        torch.save(state,checkpoint)
        result.update(protocol_sha256=sha(root/'sensitivity/protocol.json'),checkpoint_sha256=sha(checkpoint))
        save_new(out,result)
        print(key,result['test_nmse'],result['ood_nmse']['3.0'],flush=True)


def batch(root):
    p=json.loads((root/'sensitivity/protocol.json').read_text())
    def worker(index):
        with (root/'sensitivity/logs'/f'seed_{index}.out').open('x') as handle:
            subprocess.run([sys.executable,'-B',__file__,'fit','--root',str(root),'--index',str(index)],
                           stdout=handle,stderr=subprocess.STDOUT,check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK','1'))) as pool:
        list(pool.map(worker,range(len(p['seeds']))))


def collect(root,journal):
    torch.set_num_threads(1)
    from publish import interval
    p=json.loads((root/'sensitivity/protocol.json').read_text())
    assert sha(__file__)==p['source_sha256']
    files=sorted((root/'sensitivity/results').glob('*.json'))
    assert len(files)==60
    rows=[];hashes={}
    for path in files:
        r=json.loads(path.read_text());assert r['protocol_sha256']==sha(root/'sensitivity/protocol.json')
        checkpoint=root/'sensitivity/checkpoints'/path.with_suffix('.pt').name
        assert sha(checkpoint)==r['checkpoint_sha256']
        net=SelectionNet(r['seed']).double()
        net.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
        for s,value in r['ood_nmse'].items():
            assert abs(evaluate(net,dataset(r['seed'],'ood',4096,'separable',float(s)))-value)<1e-10
        assert abs(evaluate(net,dataset(r['seed'],'test',4096))-r['test_nmse'])<1e-10
        rows.append(dict(seed=r['seed'],rule=r['rule'],rate=r['rate'],test_nmse=r['test_nmse'],
                         ood_3=r['ood_nmse']['3.0'],selected_step=r['selected_step'],bound_steps=r['bounds']))
        hashes[path.name]=sha(path)
    frame=pd.DataFrame(rows)
    assert set(map(tuple,frame[['seed','rule','rate']].itertuples(index=False,name=None)))=={
        (s,r,lr) for s in p['seeds'] for r,lr in p['rules_rates']}
    out=journal/'source_data/curated_publication'
    primary=pd.read_csv(out/'inhibitory_selection_endpoints.csv')
    primary=primary[primary.variant.eq('separable')&primary.forward.eq('shunt')&primary.rule.isin(['broadcast','uniform_rms','resistance'])]
    joined=pd.concat([frame,primary.rename(columns={'ood_3.0':'ood_3'})[frame.columns]],ignore_index=True)
    assert len(joined)==120 and not joined.duplicated(['seed','rule','rate']).any()
    draws=np.random.default_rng(9222026).integers(0,20,(20000,20));contrasts=[]
    for rate,part in joined.groupby('rate'):
        for metric in ['test_nmse','ood_3']:
            wide=part.pivot(index='seed',columns='rule',values=metric).sort_index()
            for left in ['broadcast','uniform_rms']:
                difference=wide[left]-wide.resistance
                contrasts.append(dict(rate=rate,metric=metric,left=left,right='resistance',
                                      positive=int((difference>0).sum()),**interval(difference,draws)))
    frame.to_csv(out/'inhibitory_selection_rate_sensitivity.csv',index=False,mode='x')
    pd.DataFrame(contrasts).to_csv(out/'inhibitory_selection_rate_contrasts.csv',index=False,mode='x')
    save_new(out/'inhibitory_selection_rate_provenance.json',dict(protocol=p,endpoint_sha256=hashes,
        primary_endpoint_sha256=sha(out/'inhibitory_selection_endpoints.csv'),
        checkpoint_replay='All test and severity outcomes verified to absolute 1e-10',
        contrasts=contrasts,interpretation='Secondary paired-seed sensitivity, not additional independent confirmation'))
    print(pd.DataFrame(contrasts).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','fit','batch','collect'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int)
    p.add_argument('--journal',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.root)
    elif a.action=='fit':fit(a.root,a.index)
    elif a.action=='batch':batch(a.root)
    else:collect(a.root,a.journal)
