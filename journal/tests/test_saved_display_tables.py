"""The saved display tables must match the current panel layouts, not only in-memory rows.

2026-09-21: the Figure 4 export still carried the former panel H after the
noise panels were merged, because the in-memory tests never read the saved
CSV.  These checks read the curated export and its provenance twin directly.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / 'source_data/curated_publication'
PROVENANCE = ROOT / 'figures/provenance'


def test_figure4_saved_export_has_one_noise_panel_with_both_rate_policies():
    saved = pd.read_csv(CURATED / 'figure_04_plotted.csv')
    assert set(saved.panel) == set('BCDEFG'), sorted(set(saved.panel))
    noise = saved[saved.panel.eq('E')]
    assert set(noise.rate_policy.dropna()) == {'selected', 'common'}
    for policy, common in (('selected', False), ('common', True)):
        part = noise[noise.rate_policy.eq(policy)]
        assert len(part[part.record.eq('paired seed difference')]) == 60
        assert len(part[part.record.eq('summary')]) == 3
        assert part.common_rate.eq(common).all()
    twin = PROVENANCE / 'structure_restoration_20260908/figure_04_plotted.csv'
    assert twin.read_bytes() == (CURATED / 'figure_04_plotted.csv').read_bytes()


def test_figure6_saved_export_matches_promotions_and_relocated_controls():
    saved = pd.read_csv(CURATED / 'figure_06_plotted.csv')
    assert set(saved.panel) == set('BCDE'), sorted(set(saved.panel))
    rescue = saved[saved.panel.eq('C') & saved.record.eq('seed outcome')]
    assert set(rescue.condition) == {'selected rates'} and len(rescue)==100
    primary = saved[saved.panel.eq('C') & saved.record.eq('primary contrast')]
    assert len(primary) == 2 and primary.positive.eq(20).all()
    controls=pd.read_csv(ROOT/'source_data/checkpoint_computation/checkpoint_plotted.csv')
    alignment = controls[controls.panel.eq('D') & controls.record.eq('seed outcome')]
    assert set(alignment.relation) == {'within', 'cross'} and len(alignment) == 200
    assert len(saved[saved.record.eq('interaction map')])==1875
    assert len(saved[saved.record.eq('extension seed')])==200
    twin = PROVENANCE / 'credit_clarity_20260908/figure_06_plotted.csv'
    assert twin.read_bytes() == (CURATED / 'figure_06_plotted.csv').read_bytes()
