"""Scientific source checks for the completed image and placement controls."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pandas as pd

J = Path(__file__).resolve().parents[1]


def test_control_statistics_reproduce_from_all_compact_run_records(tmp_path):
    subprocess.run([sys.executable, str(J / 'scripts/run_completion/analyze_controls.py'),
                    '--output', str(tmp_path)], check=True, capture_output=True)
    source = J / 'source_data/additional_figure_controls'
    for name in ['paired_contrasts.csv', 'paired_contrasts_by_seed.csv']:
        actual = pd.read_csv(tmp_path / name, float_precision='round_trip')
        expected = pd.read_csv(source / name, float_precision='round_trip')
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    outcome = pd.read_csv(tmp_path / 'seed_outcomes.csv')
    assert outcome.groupby('study').size().to_dict() == {'fashion': 80, 'physical': 140}


def test_cifar_extension_preserves_original_histories_and_completes_stopping():
    source = J / 'source_data/cifar10_shunting_stopping_extension'
    protocol = json.loads((source / 'extension_protocol.json').read_text())
    assert len(protocol['records']) == 8
    for rec in protocol['records']:
        folder = source / 'runs' / f"config_{rec['config_index']}"
        original = json.loads((folder / 'original_training_summary.json').read_text())
        extended = json.loads((folder / 'training_summary.json').read_text())
        n = len(original['valid_losses'])
        assert extended['valid_losses'][:n] == original['valid_losses']
        assert len(extended['valid_losses']) < 1600
        assert len(extended['valid_losses']) - extended['best_epoch'] >= 49
        assert extended['best_epoch'] == original['best_epoch']


def test_physical_placement_grid_uses_all_paired_additive_cells():
    sys.path.insert(0, str(J / 'scripts'))
    import build_supplementary_figure_physical_architecture_native as builder
    blocks = builder.load_h4()
    source = pd.read_csv(J / 'source_data/additional_figure_controls/seed_outcomes.csv')
    for key, placement in [('aligned', 'aligned'), ('rewired_tree', 'reversed')]:
        means = source[source.study.eq('physical') & source.hierarchy.eq(4)
                       & source.placement.eq(placement)].groupby('depth').test_accuracy.mean()
        np.testing.assert_allclose(blocks[key][0][4], 100 * means.to_numpy(), atol=1e-12, rtol=0)
        assert np.isfinite(blocks[key][0]).all()


def test_saved_depth_display_retains_all_seeds_through_stopping():
    folder = J / 'source_data/physical_depth_stopping_extension'
    stops = pd.read_csv(folder / 'stopping_by_seed.csv')
    assert len(stops) == 60 and stops.ordinary_stopping_reached.all()
    assert (stops.epochs_run < stops.cap).all()
    assert (stops.epochs_run - stops.best_epoch >= 30).all()
    display = pd.read_csv(J / 'source_data/curated_publication/figure_07_plotted.csv',
                          float_precision='round_trip')
    expected = pd.read_csv(folder / 'condition_trajectory_summary.csv', float_precision='round_trip')
    maximum = stops.epochs_run.max()
    for panel, metric in [('E', 'test_accuracy'), ('F', 'best_validation_loss')]:
        rows = display[display.panel.eq(panel) & display.record.eq('curve summary')]
        assert rows.epoch.max() == maximum
        assert (rows.n_seeds == 10).all()
        assert rows.groupby(['arm', 'depth']).ngroups == 6
        target = expected[expected.metric.eq(metric)]
        merged = rows.merge(target, on=['arm', 'depth', 'epoch', 'metric'],
                            suffixes=('_display', '_source'), validate='one_to_one')
        assert len(merged) == len(rows) == len(target)
        for column in ['mean', 'ci95_low', 'ci95_high']:
            np.testing.assert_array_equal(merged[column+'_display'], merged[column+'_source'])
