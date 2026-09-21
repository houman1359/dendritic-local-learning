"""Verify all fresh checkpoints and export paired, seed-level source data."""
import argparse
import gc
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment import SelectionNet, dataset, validate_sources
from run import evaluate, sha


def interval(values, draws):
    values=np.asarray(values,float)
    lo,hi=np.quantile(values[draws].mean(1),[.025,.975])
    return dict(mean=float(values.mean()),ci_low=float(lo),ci_high=float(hi),n=len(values))


def exact_signflip(values):
    values=np.asarray(values,float)
    observed=abs(values.sum());count=0;n=len(values)
    for start in range(0,2**n,65536):
        bits=((np.arange(start,min(start+65536,2**n),dtype=np.uint64)[:,None]>>np.arange(n,dtype=np.uint64))&1)
        sums=(2*bits.astype(float)-1)@values
        count+=int((np.abs(sums)>=observed-1e-12*max(1.,observed)).sum())
    return count/(2**n)


def supplementary_table(summary, path):
    names = dict(exact='Exact BP', broadcast='Unit broadcast',
                 resistance='Relative-resistance gate', swapped='Wrong-branch gate',
                 uniform_rms='Uniform RMS gate')
    text = r'''\begin{table}[!htbp]
\centering
\caption{Sensory selection in sixteen-neuron DendriNet populations.
Mean NMSE over twenty fresh paired seeds, using development-selected rates and
validation-selected checkpoints. Each neuron contains a $[4,2]$ tree. The
primary target is separable and uses identity outputs; the secondary target
contains within-stream interactions and uses nonlinear proximal outputs.
All five rules train the same forward model within a condition. Strong
distractors scale irrelevant latent inputs by three; ordinary test and stress
test samples are independent. Lower is better. Section~\ref{sec:dendrinet_selection}
defines targets, gating, controls, paired inference and limitations; per-seed
outcomes, intervals and bound contacts accompany Source Data.}
\label{tab:dendrinet_selection}
\small
\begin{tabular}{llrr}
\toprule
Condition & Learning rule & Ordinary test & Strong distractors \\
\midrule
'''
    for variant, forward, label in [('separable','shunt','Separable, shunting'),
                                    ('separable','tonic','Separable, tonic'),
                                    ('separable','current','Separable, current'),
                                    ('interaction','shunt','Interaction, shunting')]:
        for k, rule in enumerate(names):
            part=summary[summary.variant.eq(variant)&summary.forward.eq(forward)&summary.rule.eq(rule)].set_index('metric')
            def number(v):
                if v>=.001:return f'{v:.4f}'
                mantissa, exponent=f'{v:.2e}'.split('e')
                return rf'${mantissa}\times10^{{{int(exponent)}}}$'
            text+=(label if k==0 else '')+' & '+names[rule]+' & '+number(part.loc['test_nmse','mean'])+' & '+number(part.loc['ood_3.0','mean'])+r' \\'+'\n'
        text+=r'\addlinespace'+'\n'
    text+=r'''\bottomrule
\end{tabular}
\end{table}
'''
    with path.open('x') as handle:handle.write(text)


def publish(root,journal):
    torch.set_num_threads(1)
    protocol=json.loads((root/'fresh_protocol.json').read_text())
    validate_sources(root,protocol)
    files=sorted((root/'fresh/results').glob('*.json'))
    assert len(files)==400, f'Incomplete fresh cohort: {len(files)}/400'
    rows,curves,diagnostics=[],[],[]
    seen=set();max_replay=0.;endpoint_hashes={}
    for path in files:
        r=json.loads(path.read_text());assert r['protocol_sha256']==sha(root/'fresh_protocol.json')
        key=(r['seed'],r['variant'],r['forward'],r['rule']);assert key not in seen;seen.add(key)
        assert r['seed'] in protocol['fresh_seeds']
        assert r['rate']==protocol['fixed_rates'][r['variant']][r['forward']][r['rule']]
        checkpoint=root/'fresh/checkpoints'/path.with_suffix('.pt').name
        assert sha(checkpoint)==r['checkpoint_sha256']
        net=SelectionNet(r['seed'],r['forward'],r['variant']).double()
        net.load_state_dict(torch.load(checkpoint,weights_only=True,map_location='cpu'))
        net.current_calibrated=True
        replay=evaluate(net,dataset(r['seed'],'test',4096,r['variant']))
        diffs=[abs(replay-r['test_nmse'])]
        for s,value in r['ood_nmse'].items():
            diffs.append(abs(evaluate(net,dataset(r['seed'],'ood',4096,r['variant'],float(s)))-value))
        max_replay=max(max_replay,max(diffs));assert max(diffs)<1e-10,(path,max(diffs))
        base=dict(seed=r['seed'],variant=r['variant'],forward=r['forward'],rule=r['rule'],rate=r['rate'])
        rows.append(dict(**base,test_nmse=r['test_nmse'],validation_nmse=r['validation_nmse'],
                         selected_step=r['selected_step'],bound_steps=r['bounds'],clip_steps=r['clips'],
                         **{f'ood_{s}':v for s,v in r['ood_nmse'].items()}))
        curves.extend(dict(**base,**h) for h in r['history'])
        for state in ['initial','selected']:
            diagnostics.extend(dict(seed=r['seed'],variant=r['variant'],forward=r['forward'],
                                    trained_rule=r['rule'],rate=r['rate'],state=state,**d)
                               for d in r[f'{state}_diagnostics'])
        endpoint_hashes[path.name]=sha(path)
        # Forward-hook closures form cycles containing saved branch diagnostics.
        # Reclaim each reconstructed network before loading the next checkpoint.
        del net
        gc.collect()
    frame=pd.DataFrame(rows).sort_values(['variant','forward','rule','seed'])
    expected={(s,j['variant'],j['mode'],r) for s in protocol['fresh_seeds'] for j in protocol['jobs']
              for r in ['exact','broadcast','resistance','swapped','uniform_rms']}
    assert seen==expected
    rng=np.random.default_rng(9222026);draws=rng.integers(0,20,(20000,20))
    summaries=[]
    for keys,part in frame.groupby(['variant','forward','rule']):
        assert len(part)==20
        for metric in ['test_nmse',*[f'ood_{s}' for s in ['1.0','1.5','2.0','2.5','3.0']]]:
            summaries.append(dict(zip(['variant','forward','rule'],keys),metric=metric,
                                  bound_runs=int((part.bound_steps>0).sum()),**interval(part[metric],draws)))
    contrasts=[]
    for keys,part in frame.groupby(['variant','forward']):
        for metric in ['test_nmse','ood_3.0']:
            wide=part.pivot(index='seed',columns='rule',values=metric).sort_index()
            for left in ['broadcast','uniform_rms','swapped','exact']:
                diff=wide[left]-wide.resistance
                primary=keys==('separable','shunt') and metric=='ood_3.0' and left in ['broadcast','uniform_rms']
                contrasts.append(dict(variant=keys[0],forward=keys[1],metric=metric,left=left,right='resistance',
                                      primary=primary,positive=int((diff>0).sum()),
                                      signflip_p=exact_signflip(diff) if primary else None,**interval(diff,draws)))
    primary=sorted([c for c in contrasts if c['primary']],key=lambda c:c['signflip_p'])
    running=0.
    for i,c in enumerate(primary):
        running=max(running,min(1.,(2-i)*c['signflip_p']));c['holm_p']=running
    # Forward contrast and difference-in-differences use the same paired seeds.
    factorial=[]
    main=frame[frame.variant.eq('separable')]
    for metric in ['test_nmse','ood_3.0']:
        wide=main.pivot(index='seed',columns=['forward','rule'],values=metric).sort_index()
        for other in ['tonic','current']:
            for rule in ['exact','broadcast','resistance']:
                diff=wide[other,rule]-wide['shunt',rule]
                factorial.append(dict(metric=metric,contrast='forward',other=other,rule=rule,
                                      positive=int((diff>0).sum()),**interval(diff,draws)))
            diff=(wide['shunt','broadcast']-wide['shunt','resistance'])-(wide[other,'broadcast']-wide[other,'resistance'])
            factorial.append(dict(metric=metric,contrast='forward_by_credit',other=other,rule='broadcast_minus_gate',
                                  positive=int((diff>0).sum()),**interval(diff,draws)))
    outputs=dict(endpoints=frame,summary=pd.DataFrame(summaries),contrasts=pd.DataFrame(contrasts),
                 factorial=pd.DataFrame(factorial),diagnostics=pd.DataFrame(diagnostics),curves=pd.DataFrame(curves))
    out=journal/'source_data/curated_publication'
    for name,table in outputs.items():
        path=out/f'inhibitory_selection_{name}.csv'
        if path.exists():raise FileExistsError(path)
        table.to_csv(path,index=False)
    provenance=dict(protocol=protocol,protocol_sha256=sha(root/'fresh_protocol.json'),
                    endpoint_sha256=endpoint_hashes,n_trajectories=len(frame),n_seeds=20,
                    maximum_test_and_ood_replay_error=max_replay,publisher_sha256=sha(__file__),
                    outputs={name:sha(out/f'inhibitory_selection_{name}.csv') for name in outputs},
                    primary_contrasts=primary,
                    interpretation='Fresh paired seeds after development; all rules/conditions retained; only two primary tests Holm-adjusted')
    path=out/'inhibitory_selection_provenance.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(provenance,indent=2,allow_nan=False)+'\n')
    supplementary_table(outputs['summary'], journal/'supplementary/curated/si_dendrinet_selection_table.tex')
    print(json.dumps(dict(n=400,replay_error=max_replay,primary=primary),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--journal',type=Path,required=True)
    a=p.parse_args();publish(a.root,a.journal)
