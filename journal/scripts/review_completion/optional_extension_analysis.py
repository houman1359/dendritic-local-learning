"""Analyze every prospectively frozen optional-extension fit and paired seed."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def key(j):return f"{j['study']}_{j['arm']}_s{j['seed']}_r{j['rate']:g}"

def signflip_p(d):
    d=np.asarray(d,dtype=float);n=len(d);observed=abs(d.sum());extreme=0
    for start in range(0,2**n,65536):
        bits=(np.arange(start,min(start+65536,2**n),dtype=np.uint32)[:,None] >> np.arange(n,dtype=np.uint32))&1
        values=(2*bits.astype(float)-1)@d
        extreme+=int((np.abs(values)>=observed-1e-12*max(observed,1e-12)).sum())
    return extreme/(2**n)

def holm(p):
    p=np.asarray(p);order=np.argsort(p);answer=np.empty_like(p)
    answer[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return answer

def main(args):
    root=args.root;out=args.output;out.mkdir(parents=True,exist_ok=False)
    dev=json.loads((root/'development_protocol.json').read_text());fresh=json.loads((root/'fresh_protocol.json').read_text())
    assert fresh['development_protocol_sha256']==sha(root/'development_protocol.json')
    scope=json.loads((root/'scope_amendment.json').read_text())
    assert fresh['scope_amendment_sha256']==sha(root/'scope_amendment.json')
    assert not list((root/'development/results').glob('temporal_*.json'))
    records=[];inputs={};diagnostics=[];routing=[];memory=[];development=[];histories=[]
    for phase,protocol in [('development',dev),('fresh',fresh)]:
        jobs=[j for j in protocol['jobs'] if j['study'] in scope['included_studies']]
        files=list((root/phase/'results').glob('*.json'));assert len(files)==len(jobs)
        expected={key(j):j for j in jobs};assert set(p.stem for p in files)==set(expected)
        for path in sorted(files):
            record=json.loads(path.read_text());job=expected[path.stem]
            assert all(record[k]==v for k,v in job.items())
            assert record['protocol_sha256']==sha(root/f'{phase}_protocol.json')
            checkpoint=root/phase/'checkpoints'/(path.stem+'.pt')
            assert record['checkpoint_sha256']==sha(checkpoint)
            inputs[str(path.relative_to(root))]=sha(path);inputs[str(checkpoint.relative_to(root))]=sha(checkpoint)
            best=min(record['history'],key=lambda h:h['validation_nmse'])
            assert best['step']==record['selected_step'] and best['validation_nmse']==record['validation_nmse']
            histories.extend({**job,'phase':phase,**h} for h in record['history'])
            if phase=='development':
                assert 'test_nmse' not in record
                development.append({k:record[k] for k in ['study','arm','seed','rate','validation_nmse','selected_step','elapsed_seconds']})
                continue
            base={k:record[k] for k in ['study','arm','seed','rate','validation_nmse','test_nmse','selected_step','bounds','clips','selected_bound_fraction','elapsed_seconds']}
            if 'stress3_nmse' in record:base['stress3_nmse']=record['stress3_nmse']
            for length,value in record.get('length_nmse',{}).items():base['length_'+length+'_nmse']=value
            for field in ['routing_accuracy','correct_route_mass']:
                if field in record:base[field]=record[field]
            chosen=fresh['selection'][record['study']][record['arm']]['rate']
            for policy in ['common','selected']:
                if record['rate']!=(.03 if policy=='common' else chosen):continue
                records.append({**base,'policy':policy})
                for d in record.get('diagnostic',[]):diagnostics.append({**base,'policy':policy,**{'diagnostic_'+k:v for k,v in d.items()}})
                for c,values in enumerate(record.get('route_probabilities',[])):
                    for branch,probability in enumerate(values):routing.append({**base,'policy':policy,'context':c,'branch':branch,'probability':probability})
                for feature,(decay,gain) in enumerate(zip(record.get('memory_decay',[]),record.get('memory_gain',[]))):
                    memory.append({**base,'policy':policy,'feature':feature,'decay':0. if record['arm']=='exact_no_memory' else decay,'gain':gain,'stored_parameter_decay':decay})
    endpoints=pd.DataFrame(records)
    assert not endpoints.duplicated(['study','arm','seed','policy']).any()
    for study,arms in fresh['studies'].items():
        for arm in arms:
            for policy in ['common','selected']:
                group=endpoints[endpoints.study.eq(study)&endpoints.arm.eq(arm)&endpoints.policy.eq(policy)]
                assert set(group.seed)==set(fresh['seeds'][study]['fresh'])
                assert not set(group.seed)&set(fresh['seeds'][study]['development'])
    rng=np.random.default_rng(2026092187);ix=rng.integers(20,size=(20000,20));summary=[]
    metrics=['test_nmse','stress3_nmse','length_4_nmse','length_8_nmse','length_16_nmse','correct_route_mass','routing_accuracy']
    for (study,arm,policy),g in endpoints.groupby(['study','arm','policy']):
        g=g.sort_values('seed')
        for metric in metrics:
            values=g[metric].dropna().to_numpy() if metric in g else np.array([])
            if not len(values):continue
            assert len(values)==20
            low,high=np.quantile(values[ix].mean(1),[.025,.975])
            summary.append(dict(study=study,arm=arm,policy=policy,metric=metric,n=20,mean=values.mean(),median=np.median(values),ci95_low=low,ci95_high=high,rate=g.rate.iloc[0],bound_runs=int((g.bounds>0).sum())))
    contrasts=[]
    for study,left,right in fresh['primary_contrasts']:
        policy='common' if study=='proxy' else 'selected'
        wide=endpoints[endpoints.study.eq(study)&endpoints.policy.eq(policy)].pivot(index='seed',columns='arm',values='test_nmse').sort_index()
        d=(wide[left]-wide[right]).to_numpy();low,high=np.quantile(d[ix].mean(1),[.025,.975])
        contrasts.append(dict(study=study,policy=policy,left=left,right=right,metric='test_nmse',n=20,mean_difference=d.mean(),ci95_low=low,ci95_high=high,left_better=int((d<0).sum()),right_better=int((d>0).sum()),signflip_p=signflip_p(d)))
    adjusted=holm([r['signflip_p'] for r in contrasts])
    for row,p in zip(contrasts,adjusted):row['holm_p']=p
    for name,frame in [('development',pd.DataFrame(development)),('endpoints',endpoints),('summary',pd.DataFrame(summary)),('contrasts',pd.DataFrame(contrasts)),('diagnostics',pd.DataFrame(diagnostics)),('routing',pd.DataFrame(routing)),('memory',pd.DataFrame(memory)),('histories',pd.DataFrame(histories))]:
        if frame.empty:continue
        frame.to_csv(out/(name+'.csv'),index=False)
    for name in ['development_protocol.json','fresh_protocol.json','scope_amendment.json','range_amendment.json']:(out/name).write_bytes((root/name).read_bytes())
    provenance=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope='All frozen fits and all paired fresh seeds; no exclusions; selected/common policies reuse seeds and are not independent cohorts',
                    development_fits=sum(j['study'] in scope['included_studies'] for j in dev['jobs']),fresh_fits=len(fresh['jobs']),deferred_studies=scope['deferred_studies'],source_sha256=sha(__file__),inputs=inputs,
                    outputs={p.name:sha(p) for p in sorted(out.iterdir())},allocation='47496711',results_root=str(root))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(pd.DataFrame(summary).query("metric == 'test_nmse'").to_string(index=False))
    print(pd.DataFrame(contrasts).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
