"""Frozen paired analyses; never filters seeds by outcome."""
from pathlib import Path
import itertools,json,re
import numpy as np
import pandas as pd
from validate_controls import R,sha,validate_manifest,validate_results

def contrast(values,label,family='',primary=False):
    values=np.asarray(values,dtype=float);assert len(values)==10 and np.isfinite(values).all()
    rng=np.random.default_rng(22622)
    means=rng.choice(values,size=(50000,10),replace=True).mean(axis=1)
    null=np.asarray(list(itertools.product([-1,1],repeat=10)))@values/10
    return dict(contrast=label,family=family,primary=primary,n=10,mean_pp=float(values.mean()),
                ci_low_pp=float(np.quantile(means,.025)),ci_high_pp=float(np.quantile(means,.975)),
                positive_seeds=int((values>0).sum()),negative_seeds=int((values<0).sum()),
                p_exact=float((np.abs(null)>=abs(values.mean())-1e-12).mean()))

def main():
    m=json.loads((R/'manifest.json').read_text());out=Path(__file__).resolve().parent/'analysis';out.mkdir(exist_ok=True)
    validate_manifest(m)
    try:validate_results(m)
    except Exception as e:
        (out/'integrity_failure.json').write_text(json.dumps({'exception':repr(e)},indent=2)+'\n');raise
    rows=[]
    for job in m['jobs']:
        for rec in job['runs']:
            p=Path(rec['result_dir']);final=json.loads((p/'performance/final.json').read_text())
            train=json.loads((p/'training_summary.json').read_text());run=json.loads((p/'execution.json').read_text())
            losses=np.array(train['valid_losses']);n=len(losses)
            log=(p/'console.log').read_text(errors='replace')
            epoch_seconds=[float(x) for x in re.findall(r'Epoch\s+\d+.*?(?:Time|time):\s*([\d.]+)',log)]
            row={k:v for k,v in rec.items() if k not in ['source','original']}
            row.update(study=job['study'],source_commit=rec['source']['commit'],
                       test_accuracy=final['accuracy']['test'],valid_accuracy=final['accuracy']['valid'],
                       train_accuracy=final['accuracy']['train'],epochs=n,best_epoch=train['best_epoch'],
                       reached_budget=n==180,last10_valid_slope=float(np.polyfit(np.arange(min(10,n)),losses[-10:],1)[0]),
                       wall_seconds=run['finished_unix']-run['started_unix'],
                       median_epoch_seconds=float(np.median(epoch_seconds)) if epoch_seconds else None,
                       final_sha256=sha(p/'performance/final.json'),
                       calibration_or_fallback_messages=' | '.join(line.strip() for line in log.splitlines() if re.search(r'calibrat|fallback|non.?finite|nan detected',line,re.I))[-12000:])
            if job['study']=='physical':
                row['archived_aligned_accuracy']=rec['original']['reference_test_accuracy']
                if rec['placement']=='aligned':row['replay_difference_pp']=100*(row['test_accuracy']-row['archived_aligned_accuracy'])
            rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(out/'seed_outcomes.csv',index=False)
    cs=[];seed_contrasts=[]
    def add(series,label,family='',primary=False):
        cs.append(contrast(100*series.values,label,family,primary))
        for seed,val in series.items():seed_contrasts.append({'contrast':label,'seed':seed,'difference_pp':100*val})
    fashion=frame[frame.study.eq('fashion')]
    for arch in ['shunting','additive']:
        a=fashion[fashion.architecture.eq(arch)].pivot(index='seed',columns='rule',values='test_accuracy')
        assert set(a.index)==set(range(22600,22610)) and not a.isna().any().any()
        for left,right in [('neuron','strict_scalar'),('exact','neuron'),('scalar_fallback','strict_scalar'),('neuron','scalar_fallback')]:
            primary=(left,right)==('neuron','strict_scalar')
            add(a[left]-a[right],f'{arch}:{left}_minus_{right}','fashion' if primary else '',primary)
    physical=frame[frame.study.eq('physical')]
    for h in [3,4]:
        a=physical[physical.hierarchy.eq(h)].pivot(index='seed',columns=['depth','placement'],values='test_accuracy')
        assert set(a.index)==set(range(10200 if h==3 else 10400,10210 if h==3 else 10410)) and not a.isna().any().any()
        for d in range(1,h+1):add(a[d,'reversed']-a[d,'aligned'],f'h{h}_d{d}:reversed_minus_aligned')
        add((a[h,'aligned']-a[1,'aligned'])-(a[h,'reversed']-a[1,'reversed']),f'h{h}:depth_by_placement','physical',True)
    for family in ['fashion','physical']:
        group=sorted([x for x in cs if x['family']==family],key=lambda x:x['p_exact']);assert len(group)==2
        previous=0
        for i,c in enumerate(group):
            previous=max(previous,min(1.,(2-i)*c['p_exact']));c['p_holm']=previous
    pd.DataFrame(cs).to_csv(out/'paired_contrasts.csv',index=False)
    pd.DataFrame(seed_contrasts).to_csv(out/'paired_contrasts_by_seed.csv',index=False)
    summary={'integrity_valid':True,'manifest_sha256':sha(R/'manifest.json'),'n_runs':len(rows),
             'fashion_directional_criterion':all(c['positive_seeds']>=8 and c['ci_low_pp']>0 for c in cs if c['family']=='fashion'),
             'primary_contrasts':[c for c in cs if c['primary']],
             'budget_contact_count':int(frame.reached_budget.sum()),
             'interpretation':'Inherited finite training budgets; inspect diagnostics before manuscript integration.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
