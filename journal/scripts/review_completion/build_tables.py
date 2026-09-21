"""Render supplementary review-control tables from complete seed outcomes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

J=Path(__file__).resolve().parents[2];D=J/'source_data/curated_publication'


def interval(v):
    v=np.asarray(v);rng=np.random.default_rng(2026092104)
    bs=v[rng.integers(len(v),size=(10000,len(v)))].mean(1)
    return v.mean(),*np.quantile(bs,[.025,.975])


def table(caption,label,columns,head,rows):
    return '\n'.join([r'\begin{table}[!htbp]',r'\centering',r'\small',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+columns+'}',
        r'\toprule',head+r' \\',r'\midrule',*[r+r' \\' for r in rows],
        r'\bottomrule',r'\end{tabular}',r'\end{table}',''])


def main():
    noise=pd.read_csv(D/'noise_controls_contrasts.csv')
    rows=[]
    for common in [False,True]:
        for name,label in [('fixed_absolute','Fixed absolute'),('noise_free','Noise free'),('relative_matched','Relative matched')]:
            r=noise[noise.common_rate.eq(common)&noise.noise.eq(name)&noise.step.eq(16384)&noise.metric.eq('population_nmse')].iloc[0]
            rows.append(f"{label} & {'Common' if common else 'Selected'} & {r['mean']:.4f} & [{r.ci_low:.4f}, {r.ci_high:.4f}] & {r.positive}/20")
    out=table('Noise controls retain the interaction-dependent fixed-profile deficit. '
        'The endpoint is quartic-minus-pairwise calibrated-minus-exact clean-domain NMSE '
        'at 16,384 updates. Twenty new paired seeds use inherited selected rates or common '
        'Adam rate 0.003; the two policies reuse the same seed blocks. Intervals are '
        'descriptive 95\\% whole-seed bootstrap intervals.', 'tab:noise_sensitivity','llrcc',
        'Noise & Rate & Difference & 95\\% interval & Positive seeds',rows)
    ep=pd.read_csv(D/'nonlinear_separable_endpoints.csv');rows=[]
    labels={'exact':'Exact BP','broadcast':'Broadcast','resistance':'Resistance',
            'derivative':"Resistance $\\times f'$",'shuffled_derivative':"Shuffled $f'$"}
    for rule,label in labels.items():
        g=ep[ep.rule.eq(rule)];m,lo,hi=interval(g.test_nmse*1e5)
        rows.append(f"{label} & {m:.3f} & [{lo:.3f}, {hi:.3f}] & {int(g.bounds.gt(0).sum())}/20")
    out+=table('Nonlinear parents with a separable target. Test NMSE is reported in units '
        'of $10^{-5}$ at validation-selected states within 4,096 updates. '
        'All rules use Adam rate 0.03 and log bounds $[-9,9]$. Twenty seed blocks are '
        'reused from the rescue; these are a paired task sensitivity, not new independent '
        'replication. Bounds count any contact during training. Intervals are descriptive '
        '95\\% seed-bootstrap intervals.', 'tab:nonlinear_separable','lrcc',
        'Rule & NMSE $\\times10^5$ & 95\\% interval & Bound contacts',rows)
    summary=pd.read_csv(D/'context_alignment_summary.csv')
    summary=summary[summary.variant.eq('separable')&summary.forward.eq('shunt')&summary.trained_rule.eq('exact')]
    labels={'exact':'Exact','broadcast':'Broadcast','resistance':'Resistance','swapped':'Wrong branch','uniform_rms':'Uniform RMS'}
    rows=[]
    for block in ['terminal','proximal','soma','readout']:
        for rule,label in labels.items():
            g=summary[summary.block.eq(block)&summary.delivered_rule.eq(rule)].set_index('relation')
            a,b=g.loc['within'],g.loc['cross']
            rows.append(f"{block.capitalize()} & {label} & {a['mean']:.4f} [{a.ci_low:.4f}, {a.ci_high:.4f}] & {b['mean']:.4f} [{b.ci_low:.4f}, {b.ci_high:.4f}]")
    out+=table('Context/update alignment at the original exact-trained separable-shunting '
        'checkpoints. Values are mean cosines with descriptive 95\\% seed-bootstrap '
        'intervals. Within and cross average four diagonal or twelve off-diagonal '
        'context pairs within each seed before inference ($n=20$). Each delivered '
        'rule is evaluated at the same weights. Positive values indicate a first-order '
        'decrease of the recipient-context loss, not the effect of an Adam step. '
        'All 400 trained states and 128,000 entries remain in Source Data.',
        'tab:context_alignment','llcc','Block & Delivery & Within context & Cross context',rows)
    target=J/'supplementary/curated/si_review_control_tables.tex';target.write_text(out)
    sources=[D/n for n in ['noise_controls_contrasts.csv','nonlinear_separable_endpoints.csv','context_alignment_summary.csv']]
    (D/'review_control_tables_provenance.json').write_text(json.dumps(dict(
        source_sha256={str(p.relative_to(J)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        builder_sha256={str(Path(__file__).relative_to(J)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        output=str(target.relative_to(J)),output_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
