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
    assert [t.get_text() for t in canvas.axes['E'].get_legend().get_texts()] == ['Selected rates', 'Common rate 0.003']


def test_figure6_exports_all_drawn_epochs_and_stopping_markers(displays):
    rows, _ = displays[6]
    assert rows.record.notna().all()
    folder = ROOT / 'source_data/physical_depth_stopping_extension'
    expected = pd.read_csv(folder / 'condition_trajectory_summary.csv')
    epochs = set(expected.epoch)
    assert set(range(1, 601)) <= epochs
    for panel in ('E', 'F'):
        curves = rows[rows.panel.eq(panel) & rows.record.eq('curve summary')]
        assert len(curves) == 6 * len(epochs)
        assert set(curves.epoch) == epochs
        assert not curves.duplicated(['arm', 'depth', 'metric', 'epoch']).any()
        scale = 100 if panel == 'E' else 1
        np.testing.assert_array_equal(curves.plotted_mean, scale * curves['mean'])
        np.testing.assert_array_equal(curves.plotted_ci95_low, scale * curves.ci95_low)
        np.testing.assert_array_equal(curves.plotted_ci95_high, scale * curves.ci95_high)
    for panel in ('G', 'H'):
        curve = rows[rows.panel.eq(panel) & rows.record.eq('paired curve summary')]
        assert len(curve) == len(epochs)
        assert set(curve.epoch) == epochs
    stops = rows[rows.record.eq('stopping marker')]
    expected_stops = pd.read_csv(folder / 'stopping_by_seed.csv')
    shallow = expected_stops[expected_stops.depth.eq(1)]
    assert len(stops) == 10 and stops.seed.nunique() == 10
    assert set(zip(stops.seed, stops.epoch)) == set(zip(shallow.seed, shallow.epochs_run))
    assert max(epochs) == expected_stops.epochs_run.max()


def test_figure6_moves_generator_to_text_and_limits_the_loss_claim(displays):
    rows, canvas = displays[6]
    texts = [text.get_text() for ax in canvas.fig.axes for text in ax.texts]
    assert not any('max(b' in text or '0.0001' in text for text in texts)
    # review pass 2026-09-23: B's product header and ratio key moved to the
    # legend, which must now carry both
    assert 'Class signal × nuisance gains' not in texts
    assert '÷ : local E/I ratio' not in texts
    source = (ROOT / 'main.tex').read_text()
    assert 'Nuisance gains multiply the class signal' in source
    assert r'local ratios ($\div$)' in source
    assert r'\max\{b_{\rm E}+y\Delta+\epsilon_{\rm E},10^{-4}\}' in source
    assert r'$\Delta=0.80$ and $\sigma_\ell=0.25$' in source
    assert '66,178 trainable parameters and 14,336 active synapses' in source
    assert not any('Cross-entropy' in ax.get_title() for ax in canvas.fig.axes)
    assert ('positive accuracy and negative cross-entropy differences favour '
            'exact paths') in source
    assert 'All sixty fits reached validation-based early stopping.' in source
    # The full stopping extension replaces the obsolete 600-epoch endpoints.
    # Compare the plotted contrast with the released paired-seed analysis.
    expected = pd.read_csv(ROOT / 'source_data/physical_depth_stopping_extension/paired_trajectory_summary.csv',
                           float_precision='round_trip')
    loss = rows[rows.panel.eq('H') & rows.record.eq('paired curve summary')].sort_values('epoch')
    expected = expected[expected.metric.eq('test_cross_entropy')].sort_values('epoch')
    np.testing.assert_array_equal(loss.epoch, expected.epoch)
    for column in ['mean', 'ci95_low', 'ci95_high', 'n_seeds']:
        np.testing.assert_array_equal(loss[column], expected[column])
    assert loss.epoch.max() > 6000


def test_oracle_support_drawing_excludes_proximal_and_soma():
    """The current oracle matrix has two terminal supports, not parent sites."""
    from matplotlib.patches import Rectangle

    tree = ast.parse(GATE.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == 'panel_deliveries')
    patches, labels, profiles = [], [], []

    class Nodes(dict):
        soma = (0., 0.)

    noop = lambda *args, **kwargs: None
    frame = SimpleNamespace(
        fx=lambda x: x/200, fy=lambda y: y/100,
        task_card=lambda cell, **kwargs: cell,
        balanced_tree=lambda *args, **kwargs: Nodes(JR=(1., 1.)),
        gate=noop, error_in=noop,
        subscript=lambda xy, base, sub, **kwargs: profiles.append((base, sub)),
        text=lambda xy, label, **kwargs: labels.append(label),
        require_soma_lowest=noop, require_delta0=noop)
    namespace = dict(Frame=lambda ax: frame, Rectangle=Rectangle,
        COLORS={'oracle': 'purple', 'shunting': 'green'}, MUTE='grey',
        INK='black', LW_HAIR=.5, PT_BASE=7, chain_frame=noop,
        subtree_delivery=noop, open_head=noop, _lerp=lambda a,b,t: a)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(GATE), 'exec'), namespace)
    namespace['panel_deliveries'](SimpleNamespace(add_patch=patches.append))
    assert len(patches)==8 and profiles==[('p','1'),('p','2')]
    xs=sorted({p.get_x() for p in patches})
    ys=sorted({p.get_y() for p in patches},reverse=True)
    assert len(xs)==2 and len(ys)==4
    support=np.zeros((4,2),dtype=int)
    for p in patches:
        support[ys.index(p.get_y()),xs.index(p.get_x())]=p.get_alpha()<1
    np.testing.assert_array_equal(support,[[1,0],[1,0],[0,1],[0,1]])
    assert labels==['1','2','3','4','Proximal and soma: ungated']
    caption = (ROOT / 'main.tex').read_text().split(
        r'\label{fig:conductancecredit}')[0].rsplit(r'\caption{', 1)[1]
    assert r'$\hat c_j$ its coefficient from the exact field' in caption
    assert 'Proximal/somatic signals remain ungated in both' in caption
    assert 'support on terminals 1--4, not weight magnitude' in caption
