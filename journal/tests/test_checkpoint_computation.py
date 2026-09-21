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
    source=pd.read_csv(D/'population_surfaces.csv');plotted=pd.read_csv(D/'figure_S37_plotted.csv')
    means=source.groupby(['rule','z1','z2']).prediction.mean().rename('expected').reset_index()
    image=plotted[plotted.quantity.eq('response')&~plotted.rule.eq('target')]
    joined=image.merge(means,on=['rule','z1','z2'],validate='one_to_one')
    assert len(joined)==3*625
    np.testing.assert_allclose(joined.value,joined.expected,rtol=0,atol=5e-15)
