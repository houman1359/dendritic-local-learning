"""Validate every bounded SGD development outcome and publish all settings."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import torch
import rescue


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(root,journal):
    torch.set_num_threads(1)
    protocol=json.loads((root/'development_protocol.json').read_text())
    decision=json.loads((root/'decision.json').read_text())
    assert not decision['success']
    assert not (root/'fresh_protocol.json').exists()
    assert not list((root/'fresh/results').glob('*'))
    rows=[];curves=[];inputs={}
    for job in protocol['jobs']:
        key=f"s{job['seed']}_r{job['rate']:g}_exact"
        path=root/'development/results'/f'{key}.json';r=json.loads(path.read_text())
        assert sha(path)==decision['development_results_sha256'][path.name]
        assert 'test_nmse' not in r
        assert r['protocol_sha256']==sha(root/'development_protocol.json')
        inputs[str(path)]=sha(path)
        item={k:r[k] for k in ['seed','rate','status','steps','selected_step','validation_nmse','endpoint_validation_nmse','bounds','clips','first_bound_step','selected_bound_fraction']}
        state=root/'development/states'/f'{key}.pt';assert sha(state)==r['checkpoint_sha256'];inputs[str(state)]=sha(state)
        saved=torch.load(state,map_location='cpu',weights_only=True)
        net=rescue.RescueNet(r['seed']).double();net.load_state_dict(saved['selected'])
        val=rescue.evaluate(net,rescue.dataset(r['seed'],'validation',1024,'interaction'))
        item['selected_replay_error']=abs(val-r['validation_nmse'])
        net.load_state_dict(saved['endpoint'])
        val=rescue.evaluate(net,rescue.dataset(r['seed'],'validation',1024,'interaction'))
        item['endpoint_replay_error']=abs(val-r['endpoint_validation_nmse'])
        assert max(item['selected_replay_error'],item['endpoint_replay_error'])<1e-12
        assert r['validation_nmse']==min(x['validation_nmse'] for x in r['history'])
        rows.append(item)
        curves.extend(dict(seed=r['seed'],rate=r['rate'],**x) for x in r['history'])
    frame=pd.DataFrame(rows);summary=frame.groupby('rate',as_index=False).agg(mean=('validation_nmse','mean'),minimum=('validation_nmse','min'),maximum=('validation_nmse','max'),n=('seed','size'))
    summary['success_seeds']=frame.assign(success=frame.validation_nmse.le(.001)).groupby('rate').success.sum().to_numpy()
    dest=journal/'source_data/curated_publication'
    for name,d in [('endpoints',frame),('curves',pd.DataFrame(curves)),('summary',summary)]:d.to_csv(dest/f'sgd_development_{name}.csv',index=False)
    (dest/'sgd_development_provenance.json').write_text(json.dumps(dict(protocol=protocol,decision=decision,
        inputs=inputs,analysis_sha256=sha(__file__),outputs={p.name:sha(p) for p in dest.glob('sgd_development_*.csv')}),indent=2)+'\n')
    table=r'''\begin{table}[p]
\centering
\caption{Bounded plain-SGD development check. Exact BP uses the unchanged nonlinear-parent interaction task, three original development seeds and 32,768 updates. Values are validation-selected NMSE, with ranges across three seeds rather than inferential confidence intervals. The prespecified success criterion required all three values at one rate to be at most 0.001. No rate passed; no fresh cohort was launched and no test data were evaluated.}
\label{tab:sgd_development}
\begin{tabular}{rrrrr}
\toprule
Rate & Mean & Minimum & Maximum & Successful seeds \\
\midrule
'''
    for r in summary.itertuples():table+=f'{r.rate:g} & {r.mean:.5f} & {r.minimum:.5f} & {r.maximum:.5f} & {r.success_seeds}/3 '+r'\\'+'\n'
    table+=r'''\bottomrule
\end{tabular}
\end{table}
'''
    (journal/'supplementary/curated/si_sgd_development_table.tex').write_text(table)
    print(summary.to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--journal',type=Path,required=True);a=p.parse_args();main(a.root,a.journal)
