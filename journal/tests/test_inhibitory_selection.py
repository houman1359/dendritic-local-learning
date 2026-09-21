"""Seed-level checks for the independently confirmed DendriNet cohort."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

J = Path(__file__).resolve().parents[1]
D = J / 'source_data/curated_publication'


def read(name):
    return pd.read_csv(D / f'inhibitory_selection_{name}.csv')


def test_complete_disjoint_fresh_cohort_and_integrity():
    meta=json.loads((D/'inhibitory_selection_provenance.json').read_text())
    protocol=meta['protocol']
    ep=read('endpoints')
    assert len(ep)==400 and len(ep.seed.unique())==20
    assert not set(ep.seed) & {2026091900,2026091901,2026091902,2026092100,2026092101,2026092102}
    assert len(ep[['seed','variant','forward','rule']].drop_duplicates())==400
    assert meta['maximum_test_and_ood_replay_error']<1e-10
    for name, digest in meta['outputs'].items():
        assert hashlib.sha256((D/f'inhibitory_selection_{name}.csv').read_bytes()).hexdigest()==digest
    for row in ep.itertuples():
        assert row.rate==protocol['fixed_rates'][row.variant][row.forward][row.rule]
        assert 0<=row.selected_step<=4096


def test_summary_and_contrasts_reproduce_endpoints():
    ep=read('endpoints')
    for row in read('summary').itertuples():
        group=ep[ep.variant.eq(row.variant)&ep.forward.eq(row.forward)&ep.rule.eq(row.rule)]
        assert len(group)==row.n==20
        np.testing.assert_allclose(group[row.metric].mean(),row.mean,rtol=1e-12)
    contrasts=read('contrasts')
    assert int(contrasts.primary.sum())==2
    for row in contrasts.itertuples():
        group=ep[ep.variant.eq(row.variant)&ep.forward.eq(row.forward)]
        wide=group.pivot(index='seed',columns='rule',values=row.metric)
        diff=wide[row.left]-wide[row.right]
        np.testing.assert_allclose(diff.mean(),row.mean,rtol=1e-10,atol=1e-14)
        assert int((diff>0).sum())==row.positive
        assert len(diff)==row.n==20
    assert contrasts[contrasts.primary].holm_p.notna().all()
    assert contrasts[~contrasts.primary].signflip_p.isna().all()


def test_frozen_adapter_matches_published_source():
    meta=json.loads((D/'inhibitory_selection_provenance.json').read_text())
    for name, digest in meta['protocol']['source_sha256'].items():
        if name.startswith('study/'):
            local=J/'scripts/inhibitory_selection'/Path(name).name
        elif name.startswith('base/'):
            local=J/'scripts/inhibitory_credit_transfer'/Path(name).name
        else:
            continue
        assert hashlib.sha256(local.read_bytes()).hexdigest()==digest


def test_exploratory_matrix_is_retained_and_floor_is_analytic():
    frame=pd.read_csv(D/'inhibitory_transfer_pilot_endpoints.csv')
    assert len(frame)==540 and set(frame.task)=={'selection','mixture'}
    assert len(frame.seed.unique())==3
    meta=json.loads((D/'inhibitory_transfer_pilot_provenance.json').read_text())
    v=1-np.tanh(2)/2
    assert meta['conditional_additivity_floor']==v/(8+v)


def test_common_rate_sensitivity_uses_same_seeds_without_pooling():
    extra=read('rate_sensitivity')
    primary=read('endpoints')
    primary=primary[primary.variant.eq('separable')&primary.forward.eq('shunt')&primary.rule.isin(['broadcast','uniform_rms','resistance'])]
    joined=pd.concat([extra,primary.rename(columns={'ood_3.0':'ood_3'})[extra.columns]])
    assert len(extra)==60 and len(joined)==120
    assert len(joined.seed.unique())==20
    assert not joined.duplicated(['seed','rule','rate']).any()
    for row in read('rate_contrasts').itertuples():
        wide=joined[joined.rate.eq(row.rate)].pivot(index='seed',columns='rule',values=row.metric)
        diff=wide[row.left]-wide[row.right]
        np.testing.assert_allclose(diff.mean(),row.mean,rtol=1e-10,atol=1e-14)
        assert len(diff)==row.n==20 and int((diff>0).sum())==row.positive
