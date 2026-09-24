"""Verify complete prospective cohorts and the published joins independently."""
from itertools import product
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd

J=Path(__file__).resolve().parents[1]
D=J/'source_data/optional_extensions'
def read(name):return pd.read_csv(D/(name+'.csv'))
def protocol():return json.loads((D/'fresh_protocol.json').read_text())


def test_complete_frozen_cohorts_and_validation_only_selection():
    p=protocol();dev=read('development');ep=read('endpoints');hist=read('histories')
    assert set(ep.study)==set(dev.study)==set(hist.study)=={'proxy','routing'}
    assert len(dev)==153 and len(ep)==680
    cols=['study','arm','seed','rate']
    actual=set(ep[cols].itertuples(index=False,name=None))
    expected={tuple(j[k] for k in cols) for j in p['jobs']}
    assert actual==expected and len(actual)==460
    assert 'test_nmse' not in dev
    for study,arms in p['studies'].items():
        for arm in arms:
            selection=dev[dev.study.eq(study)&dev.arm.eq(arm)].groupby('rate').validation_nmse.mean().idxmin()
            assert selection==p['selection'][study][arm]['rate']
            for policy,rate in [('selected',selection),('common',.03)]:
                g=ep[ep.study.eq(study)&ep.arm.eq(arm)&ep.policy.eq(policy)]
                assert set(g.seed)==set(p['seeds'][study]['fresh']) and len(g)==20
                assert set(g.rate)=={rate}
                assert not set(g.seed)&set(p['seeds'][study]['development'])
    assert not set(p['seeds']['proxy']['fresh'])&set(p['seeds']['routing']['fresh'])
    for phase,frame in [('development',dev),('fresh',ep.drop_duplicates(cols))]:
        phase_history=hist[hist.phase.eq(phase)].sort_values('step')
        minima=phase_history.loc[phase_history.groupby(cols).validation_nmse.idxmin()].set_index(cols)
        for row in frame.itertuples():
            h=minima.loc[tuple(getattr(row,k) for k in cols)]
            assert h.step==row.selected_step
            np.testing.assert_allclose(h.validation_nmse,row.validation_nmse,rtol=1e-12)


def test_reported_means_paired_intervals_and_multiplicity():
    ep=read('endpoints');summary=read('summary');contrasts=read('contrasts');p=protocol()
    assert list(contrasts[['study','left','right']].itertuples(index=False,name=None))==[tuple(x) for x in p['primary_contrasts']]
    ix=np.random.default_rng(2026092187).integers(20,size=(20000,20))
    for row in summary.itertuples():
        v=ep[ep.study.eq(row.study)&ep.arm.eq(row.arm)&ep.policy.eq(row.policy)].sort_values('seed')[row.metric].to_numpy()
        np.testing.assert_allclose([row.mean,row.median,row.ci95_low,row.ci95_high],[v.mean(),np.median(v),*np.quantile(v[ix].mean(1),[.025,.975])],rtol=1e-10,atol=1e-15)
    ps=[]
    for row in contrasts.itertuples():
        wide=ep[ep.study.eq(row.study)&ep.policy.eq(row.policy)].pivot(index='seed',columns='arm',values='test_nmse').sort_index()
        d=(wide[row.left]-wide[row.right]).to_numpy()
        np.testing.assert_allclose([row.mean_difference,row.ci95_low,row.ci95_high],[d.mean(),*np.quantile(d[ix].mean(1),[.025,.975])],rtol=1e-10)
        assert row.left_better==(d<0).sum() and row.right_better==(d>0).sum()
        # With every nonzero difference of one sign, only the two uniform
        # sign assignments are as extreme as the observed sum.
        assert (d<0).all() or (d>0).all()
        ps.append(2/2**len(d));np.testing.assert_allclose(row.signflip_p,ps[-1])
    order=np.argsort(ps);adjusted=np.empty(len(ps))
    adjusted[order]=np.minimum(1,np.maximum.accumulate(np.asarray(ps)[order]*(len(ps)-np.arange(len(ps)))))
    np.testing.assert_allclose(contrasts.holm_p,adjusted)
    spec=importlib.util.spec_from_file_location('optional_analysis',J/'scripts/review_completion/optional_extension_analysis.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    values=np.array([.5,-.1,.3,-.8,.7])
    direct=sum(abs(np.asarray(s)@values)>=abs(values.sum())-1e-12 for s in product([-1,1],repeat=5))/32
    assert module.signflip_p(values)==direct


def test_saved_figure_rows_match_all_seeds_and_routing_maps():
    ep=read('endpoints');plot=read('optional_extension_plotted');routes=read('routing')
    assert set(plot.panel)==set('ABCD')
    for key,g in plot[plot.seed.notna() & ~plot.study.eq('rescue')].groupby(['panel','study','arm','policy']):
        _,study,arm,policy=key
        expected=ep[ep.study.eq(study)&ep.arm.eq(arm)&ep.policy.eq(policy)].sort_values('seed')
        assert len(g)==20 and set(g.seed)==set(expected.seed)
        np.testing.assert_allclose(g.sort_values('seed').value,expected.test_nmse,rtol=1e-12)
    for row in plot[plot.quantity.eq('mean ordinary-test NMSE')].itertuples():
        expected=ep[ep.study.eq(row.study)&ep.arm.eq(row.arm)&ep.policy.eq(row.policy)].test_nmse.mean()
        np.testing.assert_allclose(row.value,expected)
    for row in plot[plot.panel.eq('D')].itertuples():
        g=routes[routes.arm.eq(row.arm)&routes.policy.eq(row.policy)&routes.context.eq(row.context)&routes.branch.eq(row.branch)]
        assert len(g)==20
        np.testing.assert_allclose(row.value,g.probability.mean())
    matrices=routes.groupby(['arm','policy','seed','context']).probability.sum()
    np.testing.assert_allclose(matrices,1.,atol=1e-12)
    assert set(plot[plot.panel.eq('C')].policy)=={'common'}
    assert len(plot[plot.panel.eq('B') & plot.seed.notna()])==200
    for policy,filename in [('common interaction','inhibitory_rescue_common_endpoints.csv'),('no interaction','nonlinear_separable_endpoints.csv')]:
        source=pd.read_csv(J/'source_data/curated_publication'/filename)
        for rule,g in plot[plot.panel.eq('B') & plot.policy.eq(policy) & plot.seed.notna()].groupby('rule'):
            expected=source[source.rule.eq(rule)].sort_values('seed')
            assert len(g)==20 and set(g.seed)==set(expected.seed)
            np.testing.assert_allclose(g.sort_values('seed').value,expected.test_nmse,rtol=1e-12)
    # Promoted panels retain all seeds from their own cohort and rate policy.
    main=pd.read_csv(J/'source_data/curated_publication/figure_06_plotted.csv')
    for (panel,study,arm,policy),g in main[main.record.eq('extension seed')].groupby(['panel','study','arm','policy']):
        assert (panel,study,policy) in [('E','proxy','common'),('E','routing','selected')]
        expected=ep[ep.study.eq(study)&ep.arm.eq(arm)&ep.policy.eq(policy)].sort_values('seed')
        assert len(g)==20 and set(g.seed)==set(expected.seed)
        np.testing.assert_allclose(g.sort_values('seed').value,expected.test_nmse,rtol=1e-12)
    assert len(main[main.record.eq('extension seed')])==200
    # The complete-record test above checks all eleven proxy and six routing
    # arms, including controls described in SI rather than drawn twice.
    for name in ['endpoints.csv','summary.csv','routing.csv']:
        expected=json.loads((D/'optional_extension_provenance.json').read_text())['sources'][name]
        assert hashlib.sha256((D/name).read_bytes()).hexdigest()==expected


def test_new_figure_and_complete_records_are_in_release_inventory():
    import sys
    sys.path.insert(0,str(J/'scripts'))
    import build_submission_bundle as bundle
    assert 'supplementary/curated/optional_extensions.pdf' in bundle.SUPPLEMENTARY_FIGURES
    records=json.loads((J/'configs/credit_first_provenance/panel_sources.json').read_text())['records']
    for path in D.iterdir():
        if path.is_file():
            assert any(r['figure']=='figS23' and r['path']==str(path.relative_to(J)) for r in records),path.name
    scope=json.loads((D/'scope_amendment.json').read_text())
    assert scope['deferred_studies']==['temporal']
    assert all(j['study']!='temporal' for j in protocol()['jobs'])
