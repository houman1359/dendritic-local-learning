"""The SGD stopping decision and population display retain their actual cohorts."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

D = Path(__file__).resolve().parents[1] / 'source_data/curated_publication'


def test_sgd_validation_selection_and_stopping_decision():
    provenance = json.loads((D / 'sgd_development_provenance.json').read_text())
    assert not provenance['decision']['success']
    endpoints = pd.read_csv(D / 'sgd_development_endpoints.csv')
    curves = pd.read_csv(D / 'sgd_development_curves.csv')
    assert len(endpoints) == 18 and endpoints.status.eq('completed').all()
    assert not any('test' in name for name in endpoints.columns)
    assert set(endpoints.seed) == set(provenance['protocol']['development_seeds'])
    assert endpoints.steps.eq(32768).all()
    assert endpoints[['selected_replay_error', 'endpoint_replay_error']].max().max() < 1e-12
    best = curves.groupby(['seed', 'rate']).validation_nmse.min()
    published = endpoints.set_index(['seed', 'rate']).validation_nmse
    np.testing.assert_allclose(best.sort_index(), published.sort_index(), atol=1e-14)
    for row in pd.read_csv(D / 'sgd_development_summary.csv').itertuples():
        actual = endpoints[endpoints.rate.eq(row.rate)].validation_nmse
        assert len(actual) == row.n == 3
        np.testing.assert_allclose([actual.mean(), actual.min(), actual.max()],
                                   [row.mean, row.minimum, row.maximum], atol=1e-14)
        assert int(actual.le(.001).sum()) == row.success_seeds == 0
    for name, expected in provenance['outputs'].items():
        assert hashlib.sha256((D / name).read_bytes()).hexdigest() == expected


def test_population_separable_panel_reproduces_paired_seed_values():
    table = pd.read_csv(D / 'figure_06_plotted.csv')
    assert set(table.panel) == set('BCDEFG')
    selected = table[table.panel.eq('G') & table.seed.notna()]
    original = pd.read_csv(D / 'nonlinear_separable_endpoints.csv')
    assert len(selected) == len(original) == 100
    keys = ['seed', 'rule']
    joined = selected.merge(original, on=keys, validate='one_to_one')
    np.testing.assert_allclose(joined.value, joined.test_nmse, atol=1e-15)
    assert original.rate.eq(.03).all()
    common = pd.read_csv(D / 'inhibitory_rescue_common_endpoints.csv')
    assert set(original.seed) == set(common.seed)
