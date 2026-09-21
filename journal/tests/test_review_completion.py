"""Independent seed-table, pairing and direction checks for review follow-ups."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

J=Path(__file__).resolve().parents[1];D=J/'source_data/curated_publication'


def test_noise_cohort_complete_disjoint_and_relative_floor_matched():
    p=json.loads((D/'noise_controls_provenance.json').read_text())
    d=pd.read_csv(D/'noise_controls_curves.csv')
    assert len(d)==20*2*18*10
    assert not d.duplicated(['seed','task','noise','rule','rate','step']).any()
    old=json.loads((J/'source_data/credit_rule_bridge/protocol_freeze.json').read_text())['protocol']
    assert not set(d.seed)&set(old['fresh_seeds']+old['development_seeds'])
    assert p['maximum_replay_error']<1e-10
    for name,h in p['outputs'].items():
        assert hashlib.sha256((D/name).read_bytes()).hexdigest()==h
    np.testing.assert_allclose(d[d.noise.eq('relative_matched')].noise_floor,.0225,atol=1e-16)
    assert d[d.noise.eq('noise_free')].noise_floor.eq(0).all()
    # Same variance-one target, same standardized noise and rates imply
    # bit-identical trajectories across fixed-absolute and relative conditions.
    paired=d[d.task.eq('matching')].pivot(index=['seed','rule','rate','step'],columns='noise',values='population_nmse')
    np.testing.assert_array_equal(paired.fixed_absolute,paired.relative_matched)
    initial=d[d.step.eq(0)].pivot(index=['seed','task','rule','rate'],columns='noise',values='population_nmse')
    np.testing.assert_array_equal(initial.fixed_absolute,initial.noise_free)


def test_noise_interaction_contrasts_recompute_without_pooling_tasks():
    d=pd.read_csv(D/'noise_controls_curves.csv')
    contrasts=pd.read_csv(D/'noise_controls_contrasts.csv')
    assert len(contrasts)==3*4*2*2
    for r in contrasts.itertuples():
        selected=d.common_rate if r.common_rate else d.selected_rate
        g=d[selected&d.step.eq(r.step)&d.noise.eq(r.noise)]
        w=g.pivot(index='seed',columns=['task','rule'],values=r.metric)
        v=(w['quartet','calibrated_broadcast']-w['quartet','exact'])-(w['matching','calibrated_broadcast']-w['matching','exact'])
        assert len(v)==r.n==20
        np.testing.assert_allclose(v.mean(),r.mean,atol=1e-14)
        assert int(v.gt(0).sum())==r.positive


def test_context_matrices_are_common_state_and_exact_reference_symmetric():
    d=pd.read_csv(D/'context_alignment_matrices.csv')
    assert len(d)==400*4*4*5*4
    assert d.defined.all()
    keys=['seed','variant','forward','trained_rule','block']
    exact=d[d.delivered_rule.eq('exact')]
    np.testing.assert_allclose(exact[exact.source_context.eq(exact.recipient_context)].cosine,1,atol=2e-14)
    reverse=exact.rename(columns={'source_context':'recipient_context','recipient_context':'source_context'})
    merge=exact.merge(reverse,on=keys+['source_context','recipient_context'],suffixes=('_a','_b'),validate='one_to_one')
    np.testing.assert_allclose(merge.cosine_a,merge.cosine_b,atol=2e-14)
    for block in ['soma','readout']:
        wide=d[d.block.eq(block)].pivot(index=keys+['source_context','recipient_context'],columns='delivered_rule',values='cosine')
        for rule in wide:np.testing.assert_allclose(wide[rule],wide.exact,atol=1e-12)
    assert d.n_source.gt(350).all()
    p=json.loads((D/'context_alignment_provenance.json').read_text())
    for name,h in p['outputs'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==h


def test_separable_control_is_paired_with_rescue_not_independent_cohort():
    d=pd.read_csv(D/'nonlinear_separable_endpoints.csv')
    old=pd.read_csv(D/'inhibitory_rescue_common_endpoints.csv')
    assert len(d)==100 and not d.duplicated(['seed','rule']).any()
    assert set(d.seed)==set(old.seed)
    assert d.rate.eq(.03).all() and d.replay_error.lt(1e-12).all()
    w=d.pivot(index='seed',columns='rule',values='test_nmse')
    for r in pd.read_csv(D/'nonlinear_separable_contrasts.csv').itertuples():
        v=w[r.left]-w[r.right]
        np.testing.assert_allclose(v.mean(),r.mean,atol=1e-15)
        assert len(v)==r.n==20 and int(v.gt(0).sum())==r.positive


def test_continued_figure_legend_uses_one_allowance_and_keeps_math():
    sys.path.insert(0,str(J/'scripts'))
    from audit_nature_communications_format import figure_legend_words,prose_words
    tex=r'\begin{figure}\caption{First $\kappa h f$}\end{figure}'
    tex+=r'\begin{figure}\ContinuedFloat\caption{Second part}\end{figure}'
    tex+=r'\begin{figure}\caption{Third}\end{figure}'
    assert figure_legend_words(tex)==[6,1]
    assert len(prose_words(r'$\kappa h f$'))==3
