"""The checkpoint display must retain cohorts and separate additive interactions."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd

J=Path(__file__).resolve().parents[1]
D=J/'source_data/checkpoint_computation'
spec=importlib.util.spec_from_file_location('checkpoint_computation',J/'scripts/review_completion/checkpoint_computation.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def test_interaction_separates_known_components_with_nonuniform_weights():
    x=np.array([-2.,-.4,.3,1.7]);w=np.array([.1,.2,.4,.3]);centered=x-w@x
    expected=2.3*np.outer(centered,centered)
    surface=1.1+3*x[:,None]-.7*x[None,:]+expected
    actual=analysis.interaction_component(surface,w)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-15)
    np.testing.assert_allclose(actual@w,0,atol=1e-15)
    np.testing.assert_allclose(w@actual,0,atol=1e-15)


def test_checkpoint_displays_retain_all_primary_seeds_and_contexts():
    single=pd.read_csv(D/'branch_tuning.csv')
    population=pd.read_csv(D/'population_surfaces.csv')
    single_seeds=json.loads((J/'source_data/conductance_local_gate/protocol.json').read_text())['fresh_seeds']
    pop_seeds=json.loads((J/'code/population_replay/frozen/fresh_protocol.json').read_text())['fresh_seeds']
    assert set(single.seed)==set(single_seeds)
    assert set(population.seed)==set(pop_seeds)
    assert single.groupby(['seed','rule','context']).size().eq(65).all()
    assert population.groupby(['seed','rule','context']).size().eq(25*25).all()
    assert set(single.context)=={0,1} and set(population.context)=={0,1,2,3}
    assert set(population.rule)=={'resistance','derivative','exact'}
    expected=.5*(np.tanh(population.z1)+np.tanh(population.z2))+.25*np.tanh(population.z1)*np.tanh(population.z2)
    np.testing.assert_allclose(population.target,expected,rtol=0,atol=5e-16)


def test_exported_component_errors_recombine_without_redefining_nmse():
    metrics=pd.read_csv(D/'component_errors.csv')
    wide=metrics.pivot(index=['seed','rule'],columns='component',values='mse')
    assert len(wide)==60
    np.testing.assert_allclose(wide['full'],wide['additive']+wide['interaction'],rtol=2e-12,atol=2e-16)
    np.testing.assert_allclose(metrics.relative_mse,metrics.mse/metrics.target_energy,rtol=2e-12,atol=2e-16)
    # The published images summarize all seed/context surfaces, not a selected fit.
    source=pd.read_csv(D/'population_surfaces.csv');plotted=pd.read_csv(D/'checkpoint_plotted.csv')
    means=source.groupby(['rule','z1','z2']).prediction.mean().rename('expected').reset_index()
    image=plotted[plotted.quantity.eq('response')&~plotted.rule.eq('target')]
    joined=image.merge(means,on=['rule','z1','z2'],validate='one_to_one')
    assert len(joined)==3*625
    np.testing.assert_allclose(joined.value,joined.expected,rtol=0,atol=5e-15)


def test_main_branch_tuning_uses_each_teachers_complete_cohort():
    source=pd.read_csv(D/'branch_tuning.csv')
    plotted=pd.read_csv(J/'source_data/curated_publication/figure_05_plotted.csv')
    tuning=plotted[plotted.record.eq('tuning mean')]
    assert len(tuning)==2*4*65 and set(tuning.panel)=={'F','G'}
    ix=np.random.default_rng(2026092121).integers(20,size=(10000,20))
    for (context,rule),g in tuning.groupby(['context','rule']):
        expected=source[source.context.eq(context)&source.rule.eq(rule)].pivot(index='seed',columns='z1',values='branch_contribution').sort_index()
        assert len(expected)==20
        values=expected.to_numpy();lo,hi=np.quantile(values[ix].mean(1),[.025,.975],axis=0)
        actual=g.sort_values('z1')
        np.testing.assert_allclose(actual[['mean','ci_low','ci_high']].to_numpy(),np.array([values.mean(0),lo,hi]).T,atol=5e-15)
        assert set(actual.panel)==({'F'} if context==0 else {'G'})


def test_main_interaction_maps_use_all_rescue_seeds_without_reselection():
    source=pd.read_csv(D/'population_surfaces.csv')
    plot=pd.read_csv(J/'source_data/curated_publication/figure_06_plotted.csv')
    primary=plot[plot.panel.eq('C') & plot.record.eq('seed outcome')]
    source_endpoints=pd.read_csv(J/'source_data/curated_publication/inhibitory_rescue_endpoints.csv').query("optimizer == 'adam' and bound == 9")
    assert len(primary)==100
    for rule,g in primary.groupby('rule'):
        expected=source_endpoints[source_endpoints.rule.eq(rule)].sort_values('seed')
        assert set(g.seed)==set(expected.seed)
        np.testing.assert_allclose(g.sort_values('seed').value,expected.test_nmse,rtol=1e-12)
    maps=plot[plot.record.eq('interaction map')]
    assert len(maps)==3*625 and set(maps.panel)=={'D'}
    grid=np.sort(source.z1.unique());w=np.ones(25);w[[0,-1]]=.5;w/=w.sum()
    for rule,g in maps.groupby('rule'):
        if rule=='target':
            x,y=np.meshgrid(grid,grid,indexing='ij');full=.5*(np.tanh(x)+np.tanh(y))+.25*np.tanh(x)*np.tanh(y)
        else:
            full=source[source.rule.eq(rule)].groupby(['z1','z2']).prediction.mean().unstack().loc[grid,grid].to_numpy()
        expected=analysis.interaction_component(full,w)
        np.testing.assert_allclose(g.pivot(index='z1',columns='z2',values='value').loc[grid,grid],expected,atol=5e-15)
    for n in [5,6]:
        name=f'figure_{n:02d}_plotted.csv'
        assert (J/'source_data/curated_publication'/name).read_bytes()==(J/'figures/provenance/credit_clarity_20260908'/name).read_bytes()


def test_relocated_population_controls_keep_separable_cohort_and_rate_policy():
    plotted=pd.read_csv(D/'checkpoint_plotted.csv')
    assert set(plotted.panel)==set('ABCDEFGH')
    assert 'interaction' not in set(plotted.quantity.dropna())
    original=pd.read_csv(J/'source_data/curated_publication/inhibitory_selection_endpoints.csv').query("variant == 'separable'")
    common=pd.read_csv(J/'source_data/curated_publication/inhibitory_selection_rate_sensitivity.csv')
    control=plotted[plotted.panel.eq('C') & plotted.record.eq('seed outcome')]
    assert len(control)==13*20
    for (condition,forward,rule,metric,policy),g in control.groupby(['condition','forward','rule','metric','rate_scope']):
        expected=original[original.forward.eq(forward)&original.rule.eq(rule)]
        if policy=='common 0.1' and rule=='broadcast':
            expected=common[common.rule.eq(rule)&common.rate.eq(.1)].rename(columns={'ood_3':'ood_3.0'})
        assert len(g)==len(expected)==20
        assert set(g.seed)==set(expected.seed)
        np.testing.assert_allclose(g.sort_values('seed').value,expected.sort_values('seed')[metric],rtol=1e-12)
    ratios=plotted[plotted.record.eq('broadcast/gate ratio of means')].set_index('condition').value
    np.testing.assert_allclose(ratios[['Ordinary','Stress 3','Stress 3 rate 0.1']],[1.7,87,6.6],rtol=.035)
