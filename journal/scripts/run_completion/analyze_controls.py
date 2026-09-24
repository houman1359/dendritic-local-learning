"""Reproduce paired control statistics from the archived 220-fit records.

No training is invoked and no seed is filtered. The archived raw metrics,
selected epochs, source commits and receipt hashes are checked before analysis.
"""
from pathlib import Path
import argparse, hashlib, itertools, json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code/release_noise"))
from release_hashes import verify_released_file

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2] / 'source_data/additional_figure_controls')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source, args.output
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source / 'seed_outcomes.csv', float_precision='round_trip')
    assert len(frame) == 220 and frame.key.nunique() == 220
    for record in json.loads((source / 'compact_records.json').read_text()):
        verdict = verify_released_file(source / record['path'], record['sha256'])
        assert verdict['verified'], (record['path'], verdict)
    for row in frame.itertuples():
        folder = source / 'runs' / row.key
        final = json.loads((folder / 'performance/final.json').read_text())
        train = json.loads((folder / 'training_summary.json').read_text())
        receipt = json.loads((folder / 'execution.json').read_text())
        assert receipt.get('status', 'complete') == 'complete' and receipt['exit_code'] == 0
        assert np.isfinite(train['valid_losses']).all()
        assert len(train['valid_losses']) == row.epochs
        assert int(train['best_epoch']) == row.best_epoch
        assert int(np.argmin(train['valid_losses'])) + 1 == row.best_epoch
        for split in ['train', 'valid', 'test']:
            assert getattr(row, split + '_accuracy') == final['accuracy'][split]
        assert sha(folder / 'performance/final.json') == row.final_sha256
    frame.to_csv(out / 'seed_outcomes.csv', index=False)
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
    summary={'integrity_valid':True,'manifest_sha256':sha(source/'execution_manifest.json'),'n_runs':len(frame),
             'fashion_directional_criterion':all(c['positive_seeds']>=8 and c['ci_low_pp']>0 for c in cs if c['family']=='fashion'),
             'primary_contrasts':[c for c in cs if c['primary']],
             'budget_contact_count':int(frame.reached_budget.sum()),
             'interpretation':'Inherited finite training budgets; inspect diagnostics before manuscript integration.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__':
    main()
