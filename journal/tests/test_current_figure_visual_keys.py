"""Small semantic keys remain explicit without importing training adapters."""
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def drawing_helper(relative, name, **extra):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'PT_BASE': 7.0, 'INK': '#242a32', 'GRAY': '#727981',
                 'MARKER_MS': 4.0, 'LW_HAIR': 0.55, 'np': np, **extra}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'),
         namespace)
    return namespace[name]


def test_figure5_condition_is_separate_from_rule_legend():
    label = drawing_helper('scripts/conductance_local_gate/figure.py',
                           'condition_label')
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    try:
        for name in ('Exact path', 'Local distal gate', 'Unit broadcast',
                     'Two-profile oracle'):
            ax.plot([], [], label=name)
        legend = ax.legend(loc='upper left', ncol=2, fontsize=7.0,
                           frameon=False, borderaxespad=.2)
        artist = label(ax, 'Aligned targets')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        header = artist.get_window_extent(renderer)
        key = legend.get_window_extent(renderer)
        assert header.y0 > key.y1
        assert artist.get_text() == 'Aligned targets'
        assert ax._condition_label is artist
    finally:
        plt.close(fig)


def test_figure5_population_means_are_distinct_and_preserve_values(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from inhibitory_selection import figure_panels as population

    fig, (h, i) = plt.subplots(1, 2)
    try:
        population.population_panels(h, i, lambda ax, _: ax.set_yscale('log'))
        assert h.get_legend_handles_labels()[1] == [
            'Exact BP', 'Unit broadcast', 'Relative-resistance gate']
        means = [line for line in i.lines if line.get_marker() == 'D']
        assert len(means) == 9
        _, summary = population.data()
        for line, (forward, rule) in zip(means, (
                (forward, rule) for forward in ('shunt', 'tonic', 'current')
                for rule in population.RULES)):
            assert line.get_markersize() > population.SEED_MS + 1
            assert to_rgba(line.get_markerfacecolor()) == to_rgba('white')
            assert to_rgba(line.get_markeredgecolor()) == to_rgba(
                population.COLORS_BY_RULE[rule])
            assert line.get_markeredgewidth() >= 1.0
            assert line.get_zorder() > max(c.get_zorder() for c in i.collections
                                          if hasattr(c, 'get_sizes'))
            point = summary[summary.forward.eq(forward) & summary.rule.eq(rule)
                            & summary.metric.eq('ood_3.0')].iloc[0]
            np.testing.assert_array_equal(line.get_ydata(), [point['mean']])
    finally:
        plt.close(fig)


def test_figure8_fill_key_distinguishes_spatial_support():
    key = drawing_helper('scripts/shunt_ancestry_gain/build_focused_main.py',
                         'signed_fill_key')
    fig, ax = plt.subplots()
    try:
        artists = key(ax)
        assert [text.get_text() for text in ax.texts] == ['descendants', 'off-route']
        markers = list(ax.lines)
        assert len(markers) == 2
        assert to_rgba(markers[0].get_markerfacecolor()) == to_rgba('#242a32')
        assert to_rgba(markers[1].get_markerfacecolor()) == to_rgba('white')
        assert all(1 < line.get_ydata()[0] <= 1.025 for line in markers)
        assert len(artists) == 4
    finally:
        plt.close(fig)


def test_figure9_scan_rug_is_explicitly_descriptive():
    label = drawing_helper('scripts/credit_first_figures/build_measured.py',
                           'label_scan_rug')
    fig, ax = plt.subplots()
    try:
        artist = label(ax, np.array([0.62, 0.775, 0.62]))
        assert artist.get_text() == 'All scans\n(descriptive)'
        assert artist.xy == (0.0, (0.62 + 0.775) / 2)
        assert artist.xycoords == ('axes fraction', 'data')
    finally:
        plt.close(fig)
