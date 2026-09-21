"""Continue only the studies retained by the user's pre-confirmation scope amendment."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,concurrent.futures,itertools,json,os,subprocess,sys,time
import numpy as np
from campaign import key,sha,save,run_phase,HERE

def main(root,workers):
    devpath=root/'development_protocol.json';dev=json.loads(devpath.read_text())
    scope=json.loads((root/'scope_amendment.json').read_text());included=set(scope['included_studies'])
    assert included=={'proxy','routing'}
    indices=[i for i,j in enumerate(dev['jobs']) if j['study'] in included]
    assert not list((root/'development/results').glob('temporal_*.json'))
    def run(index):
        job=dev['jobs'][index];name=key(job);out=root/'development/results'/f'{name}.json'
        if out.exists():
            previous=json.loads(out.read_text())
            assert previous['protocol_sha256']==sha(devpath)
            assert sha(root/'development/checkpoints'/f'{name}.pt')==previous['checkpoint_sha256']
            return index
        log=root/'development/logs'/f'{name}.log'
        with log.open('x') as handle:
            subprocess.run([sys.executable,'-B',str(HERE/'campaign.py'),'--root',str(root),'--action','worker','--phase','development','--index',str(index)],stdout=handle,stderr=subprocess.STDOUT,check=True)
        return index
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(run,i) for i in indices]
        for done,future in enumerate(concurrent.futures.as_completed(pending),1):
            index=future.result();print('development',done,'/',len(indices),key(dev['jobs'][index]),'elapsed',round(time.monotonic()-started,1),flush=True)
    studies={s:arms for s,arms in dev['studies'].items() if s in included};selection={};inputs={}
    for study,arms in studies.items():
        selection[study]={}
        for arm in arms:
            values={}
            for rate in dev['rates']:
                group=[]
                for seed in dev['seeds'][study]['development']:
                    path=root/'development/results'/(key(dict(study=study,arm=arm,seed=seed,rate=rate))+'.json')
                    result=json.loads(path.read_text());assert 'test_nmse' not in result
                    assert result['protocol_sha256']==sha(devpath)
                    group.append(result['validation_nmse']);inputs[str(path.relative_to(root))]=sha(path)
                values[rate]=float(np.mean(group))
            selection[study][arm]={'rate':min(values,key=lambda rate:(values[rate],rate)),'validation_means':values}
    jobs=[]
    for study,arms in studies.items():
        for arm in arms:
            for seed,rate in itertools.product(dev['seeds'][study]['fresh'],sorted({.03,selection[study][arm]['rate']})):
                jobs.append(dict(study=study,arm=arm,seed=seed,rate=rate))
    fresh={**dev,'created_utc':datetime.now(timezone.utc).isoformat(),'development_protocol_sha256':sha(devpath),
        'scope_amendment':scope,'scope_amendment_sha256':sha(root/'scope_amendment.json'),
        'studies':studies,'primary_contrasts':[c for c in dev['primary_contrasts'] if c[0] in included],
        'statistics':dev['statistics'].replace('all eight prespecified contrasts','all six retained prespecified contrasts'),
        'selection':selection,'development_results_sha256':inputs,'jobs':jobs,
        'source_sha256':{**dev['source_sha256'],'scoped_campaign.py':sha(__file__)}}
    save(root/'fresh_protocol.json',fresh)
    print('Frozen',len(jobs),'fresh fits; recurrent study deferred without training',flush=True)
    run_phase(root,'fresh',workers)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    main(a.root,a.workers)
