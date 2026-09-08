"""Validate existing cohorts before exposing their diagnostics in main figures.

These helpers select archived display coordinates and count recorded training
epochs. They do not rerun a model, change an endpoint or recompute an interval.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def activation_capture(summary, raw, selected_learning_rows):
    """Return the fresh exact-rule activation-space diagnostic and its intervals."""
    bases = ('broadcast_k1', 'subtrees_k3')
    points = raw[raw.cohort.eq('fresh') & raw.coordinate.eq('activation')
                 & raw.basis.isin(bases)].copy()
    means = summary[summary.coordinate.eq('activation')
                    & summary.metric.eq('mean_capture')
                    & summary.basis.isin(bases)].copy()
    exact = selected_learning_rows[selected_learning_rows.arm.eq('exact_path')]
    assert len(points) == 80 and len(means) == 8 and len(exact) == 20
    for architecture in ('shunting', 'additive'):
        expected_seeds = set(exact[exact.architecture.eq(architecture)].seed)
        assert len(expected_seeds) == 10
        for checkpoint in ('initial', 'trained'):
            for basis in bases:
                values = points[points.architecture.eq(architecture)
                                & points.checkpoint.eq(checkpoint)
                                & points.basis.eq(basis)]
                row = means[means.architecture.eq(architecture)
                            & means.checkpoint.eq(checkpoint)
                            & means.basis.eq(basis)]
                assert len(values) == 10 and set(values.seed) == expected_seeds
                assert len(row) == 1 and int(row.iloc[0]['n']) == 10
                assert np.isclose(values.mean_capture.mean(), row.iloc[0]['mean'],
                                  rtol=0, atol=1e-12)
                assert values.n_fields.le(2048 * 128).all()
                assert values.forward_max_difference.le(1e-6).all()
    paired = points.pivot(index=['architecture', 'seed', 'checkpoint'],
                          columns='basis', values='mean_capture')
    assert (paired.subtrees_k3 >= paired.broadcast_k1 - 1e-12).all()
    return means, points


def d1_training_counts(stopping, selected_trajectories):
    """Count runs that actually observed an epoch, including their final epoch."""
    stops = stopping[stopping.depth.eq(1)].copy().sort_values('seed')
    traces = selected_trajectories[selected_trajectories.depth.eq(1)].copy()
    assert len(stops) == 10 and stops.seed.nunique() == 10
    assert set(stops.arm) == {'exact_autograd_bp_recipe'}
    assert len(traces) == 6000 and traces.groupby('seed').size().eq(600).all()
    expected = traces.seed.map(stops.set_index('seed').epochs_run)
    assert traces.last_training_epoch.eq(expected).all()
    assert traces.observed_epoch.eq(traces.epoch.le(expected)).all()
    counts = traces.groupby('epoch').observed_epoch.sum().rename('n_training').reset_index()
    counts['n_retained_after_stop'] = 10 - counts.n_training
    assert counts.n_training.is_monotonic_decreasing
    assert counts.set_index('epoch').loc[[180, 400, 600], 'n_training'].tolist() == [7, 7, 2]
    assert stopping[stopping.depth.eq(3)].epochs_run.eq(600).all()
    return stops, counts


def paired_residual_cells(table, control, archived_contrast):
    """Preserve root identities and check the archived residual-coordinate mean."""
    focus = table[table.channels.eq(8)]
    paired = focus.pivot(index='root_id', columns='method', values='residual_capture')
    values = 100 * (paired['common + ancestry'] - paired[control])
    assert len(values) == 47 and not values.isna().any()
    assert np.isclose(values.mean(), archived_contrast['mean_pp'], rtol=0, atol=1e-10)
    assert int(values.gt(0).sum()) == int(archived_contrast['positive_cells'])
    return pd.DataFrame({'root_id': [str(int(root)) for root in values.index],
                         'control': control, 'residual_difference_pp': values.to_numpy()})
