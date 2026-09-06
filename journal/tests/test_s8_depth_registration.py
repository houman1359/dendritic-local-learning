"""The published CSV must place each depth under its corresponding tick."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

JOURNAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(JOURNAL / 'scripts'))
from analyze_prospective_followup_results import plot_fixed_budget


def test_csv_depth_coordinates_match_labels(monkeypatch):
    source = JOURNAL / 'source_data/prospective_input_validity'
    summary = pd.read_csv(source / 'followup_publication_condition_summary.csv')
    contrasts = pd.read_csv(source / 'followup_publication_paired_contrasts.csv')
    figures = []
    monkeypatch.setattr(Figure, 'savefig', lambda self, *args, **kwargs: figures.append(self))
    plot_fixed_budget(summary, contrasts)
    fig = figures[0]
    ax = fig.axes[5]
    series = [line for line in ax.lines if line.get_marker() == 'o']
    assert len(series) == 3
    np.testing.assert_array_equal(ax.get_xticks(), [1, 2, 3, 4])
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ['D1', 'D2', 'D3', 'D4']
    expected = contrasts[(contrasts.study == 'fixed_budget')
                         & (contrasts.contrast == 'local - backprop')]
    for line in series:
        np.testing.assert_array_equal(line.get_xdata(orig=False), [1, 2, 3, 4])
    # The distinctly negative scalar series must retain the true D4 value.
    scalar = min(series, key=lambda line: np.mean(line.get_ydata()))
    expected_d4 = expected[(expected.feedback == 'scalar_fallback')
                           & (expected.depth.astype(str) == '4')]
    if expected_d4.empty:
        expected_d4 = expected[expected.depth.astype(str).eq('4')].sort_values('mean_difference').head(1)
    assert len(expected_d4) == 1
    np.testing.assert_allclose(scalar.get_ydata()[-1], 100 * expected_d4.mean_difference.iloc[0])
    plt.close(fig)
