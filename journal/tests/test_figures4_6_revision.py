"""Current-panel scientific labels and complete display exports, without writes.

The figure constructors run with their save function intercepted: no figure,
Source Data, provenance, or release artifact is rebuilt by these tests.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / 'scripts/credit_first_figures/build_restored_main.py'
GATE = ROOT / 'scripts/conductance_local_gate/figure.py'


@pytest.fixture(scope='module')
def restored():
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(BUILDER.parent.parent))
        patch.syspath_prepend(str(BUILDER.parent))
        spec = importlib.util.spec_from_file_location('restored_display_test', BUILDER)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        yield module


@pytest.fixture(scope='module')
def displays(restored):
    result = {}

    def capture(canvas, number, sources, panels, caption, rows, extra=None,
                equalize=True):
        result[number] = (pd.DataFrame(rows), canvas)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(restored, 'save', capture)
        restored.figure4()
        restored.figure6()
    yield result
    for _, canvas in result.values():
        restored.plt.close(canvas.fig)


def test_figure4_exports_every_drawn_seed_trajectory_and_dot(displays):
    rows, _ = displays[4]
    assert rows.record.notna().all()
    traces = rows[rows.record.eq('seed trajectory')]
    assert set(traces.panel) == {'D'}
    assert len(traces) == 20 * 64
    assert not traces.duplicated(['panel', 'series', 'seed', 'x']).any()
    original = pd.read_csv(ROOT / 'source_data/credit_rule_extension/summaries/all_curves.csv')
    original = original[original.model.eq('algebraic') & original.optimizer.eq('adam')
                        & original.selected_rate & original.task.eq('quartet')
                        & original.rule.eq('exact') & original.step.ge(64)]
    joined = traces.merge(original, left_on=['seed', 'x'], right_on=['seed', 'step'],
                          validate='one_to_one')
    assert len(joined) == 20 * 64
    np.testing.assert_allclose(joined.value, joined.test_nmse, rtol=0, atol=5e-16)
    for panel in ('F',):
        dots = rows[rows.panel.eq(panel) & rows.record.eq('paired seed difference')]
        assert len(dots) == 80
        summaries = rows[rows.panel.eq(panel) & rows.record.eq('summary')]
        if panel == 'F':
            summaries = summaries[summaries.endpoint.eq('endpoint state')]
        keys = ['task', 'x'] if panel == 'F' else ['task']
        means = dots.groupby(keys, dropna=False).value.mean()
        expected = summaries.set_index(keys)['mean']
        np.testing.assert_allclose(means.sort_index(), expected.sort_index(),
                                   rtol=0, atol=1e-9)


def test_figure4_learning_curves_identify_their_tasks(displays):
    _, canvas = displays[4]
    assert {name: canvas.axes[name].get_title() for name in ('C', 'D')} == {
        'C': 'Pairwise', 'D': 'Quartic'}
    assert canvas.axes['C'].get_ylabel() == 'Held-out NMSE'


def test_figure4_noise_panel_exports_both_rate_policies(displays):
    """2026-09-21: the former E (selected rates) and H (common rate) share one panel E,
    encoded by marker shape; every paired estimand is still exported per policy."""
    rows, canvas = displays[4]
    original = pd.read_csv(ROOT / 'source_data/curated_publication/noise_controls_curves.csv')
    assert 'H' not in set(rows.panel)
    for policy, common in [('selected', False), ('common', True)]:
        dots = rows[rows.panel.eq('E') & rows.record.eq('paired seed difference')
                    & rows.rate_policy.eq(policy)]
        summaries = rows[rows.panel.eq('E') & rows.record.eq('summary')
                         & rows.rate_policy.eq(policy)]
        assert len(dots) == 60 and len(summaries) == 3
        assert set(dots.noise) == {'fixed_absolute', 'noise_free', 'relative_matched'}
        assert dots.common_rate.eq(common).all()
        source = original[(original.common_rate if common else original.selected_rate)
                          & original.step.eq(16384)]
        for noise in dots.noise.unique():
            wide = source[source.noise.eq(noise)].pivot(
                index='seed', columns=['task', 'rule'], values='population_nmse')
            expected = (wide['quartet', 'calibrated_broadcast'] - wide['quartet', 'exact']
                        - wide['matching', 'calibrated_broadcast'] + wide['matching', 'exact'])
            actual = dots[dots.noise.eq(noise)].set_index('seed').value.sort_index()
            np.testing.assert_allclose(actual, expected.sort_index(), rtol=0, atol=1e-12)
        means = dots.groupby('noise').value.mean().sort_index()
        np.testing.assert_allclose(means, summaries.set_index('noise')['mean'].sort_index(),
                                   rtol=0, atol=1e-12)
    assert canvas.axes['E'].get_ylabel() == 'Interaction deficit'
    assert canvas.axes['E'].get_legend() is None


def test_figure6_exports_all_drawn_epochs_and_stopping_markers(displays):
    rows, _ = displays[6]
    assert rows.record.notna().all()
    for panel in ('E', 'F'):
        curves = rows[rows.panel.eq(panel) & rows.record.eq('curve summary')]
        assert len(curves) == 6 * 600
        assert set(curves.epoch) == set(range(1, 601))
        assert not curves.duplicated(['arm', 'depth', 'metric', 'epoch']).any()
        scale = 100 if panel == 'E' else 1
        np.testing.assert_array_equal(curves.plotted_mean, scale * curves['mean'])
        np.testing.assert_array_equal(curves.plotted_ci95_low, scale * curves.ci95_low)
        np.testing.assert_array_equal(curves.plotted_ci95_high, scale * curves.ci95_high)
    for panel in ('G', 'H'):
        curve = rows[rows.panel.eq(panel) & rows.record.eq('paired curve summary')]
        assert len(curve) == 600
        assert set(curve.epoch) == set(range(1, 601))
    stops = rows[rows.record.eq('stopping marker')]
    assert len(stops) == 8 and stops.seed.nunique() == 8
    assert stops.epoch.min() == 103 and stops.epoch.max() == 590


def test_figure6_moves_generator_to_text_and_limits_the_loss_claim(displays):
    rows, canvas = displays[6]
    texts = [text.get_text() for ax in canvas.fig.axes for text in ax.texts]
    assert not any('max(b' in text or '0.0001' in text for text in texts)
    assert 'Class signal × nuisance gains' in texts
    assert '÷ : local E/I ratio' in texts
    source = (ROOT / 'main.tex').read_text()
    assert r'\max\{b_{\rm E}+y\Delta+\epsilon_{\rm E},10^{-4}\}' in source
    assert r'$\Delta=0.80$ and $\sigma_\ell=0.25$' in source
    assert '66,178 trainable parameters and 14,336 active synapses' in source
    # design pass 2026-09-14: data panels carry no titles; the loss claim is
    # limited by the sign key drawn on H (`shared soma ahead` at the top of a
    # loss ordinate, `exact path ahead` at the bottom) and by the two marked
    # values, not by a title
    titles = [ax.get_title() for ax in canvas.fig.axes]
    assert 'Cross-entropy ordering does not flip' not in titles
    assert not any('Cross-entropy' in title for title in titles)
    assert any(text == 'exact path ahead' for text in texts)
    assert any(text == 'shared soma ahead' for text in texts)
    assert any(text.startswith('−0.073 nats at 180') for text in texts)
    assert any(text.startswith('−0.021 nats at 600') for text in texts)
    loss = rows[rows.panel.eq('H') & rows.record.eq('paired curve summary')]
    assert (loss[loss.epoch.between(180, 600)]['mean'] < 0).all()
    assert float(loss[loss.epoch.eq(34)]['mean'].iloc[0]) > 0


def test_oracle_support_drawing_excludes_proximal_and_soma():
    """Extract the small drawing helper without importing training adapters."""
    tree = ast.parse(GATE.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == 'distal_oracle_delivery')
    namespace = {'COLORS': {'oracle': 'purple'}, 'mix': lambda *_: 'pale purple'}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(GATE), 'exec'), namespace)
    positions = {'JL': (1, 1), 'JR': (4, 1), 'S': (2.5, 0),
                 'T1': (0, 2), 'T2': (2, 2), 'T3': (3, 2), 'T4': (5, 2)}

    class Nodes(dict):
        def terminals_under(self, root):
            return ['T1', 'T2'] if root == 'JL' else ['T3', 'T4']

    collars, dots = [], []
    frame = SimpleNamespace(_draw_chains=lambda chains, *_: collars.extend(chains),
                            disc=lambda xy, *_, **kwargs: dots.append(xy))
    namespace['distal_oracle_delivery'](frame, Nodes(positions))
    expected = {positions[name] for name in ('T1', 'T2', 'T3', 'T4')}
    assert set(dots) == expected
    assert {point for collar in collars for point in collar} == expected
    source = ast.get_source_segment(GATE.read_text(), next(
        node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == 'panel_deliveries'))
    # The enlarged two-card schematic indexes the two profiles by j instead
    # of repeating two tiny formula lines; its nonlocal coefficient source
    # and unit proximal/somatic factors must remain explicit.
    assert "'distal: p'" in source and "('sub', 'j')" in source
    assert "'ĉ from exact field'" in source
    assert "'Both: proximal and soma × 1'" in source
