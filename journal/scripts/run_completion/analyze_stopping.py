"""Recompute stopping-extension curves from compact training and metric records.

The full tensor/checkpoint audit is retained separately. This portable replay
checks validation-only selection, ordinary early stopping and every displayed
seed trajectory before reproducing the published paired statistics.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from stopping_statistics import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[2] / 'source_data/physical_depth_stopping_extension')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source, args.output
    out.mkdir(parents=True, exist_ok=True)
    read = lambda name: pd.read_csv(source / name, float_precision='round_trip')
    histories = read('observed_training_histories.csv')
    metrics = read('selected_metric_audit.csv')
    saved = read('validation_selected_seed_trajectories.csv')
    stops = read('stopping_by_seed.csv')
    assert len(stops) == 60 and stops['index'].nunique() == 60
    assert stops.ordinary_stopping_reached.all()
    epochs = np.sort(saved.epoch.unique())
    rows = []
    for rec in stops.to_dict('records'):
        index = rec['index']
        h = histories[histories['index'].eq(index)].sort_values('epoch')
        n = len(h)
        assert n == rec['epochs_run'] and n < rec['cap']
        np.testing.assert_array_equal(h.epoch, np.arange(1, n+1))
        loss = h.validation_loss.to_numpy()
        assert np.isfinite(loss).all()
        assert int(loss.argmin())+1 == rec['best_epoch']
        assert n - rec['best_epoch'] >= 30
        best_indices = np.zeros(n, dtype=int)
        best = 0
        for k in range(n):
            if loss[k] < loss[best]:
                best = k
            best_indices[k] = best + 1
        np.testing.assert_array_equal(h.best_epoch, best_indices)
        np.testing.assert_array_equal(h.best_loss, loss[best_indices-1])
        m = metrics[metrics['index'].eq(index)].set_index('selected_epoch')
        assert m.index.is_unique
        for e in epochs:
            selected = int(best_indices[min(e, n)-1])
            r = m.loc[selected]
            assert abs(-r.categorical_loglikelihood_valid-loss[selected-1]) < 2e-6
            rows.append(dict(index=index, arm=rec['arm'], depth=rec['depth'], seed=rec['seed'],
                epoch=int(e), selected_epoch=selected, observed_epoch=bool(e<=n),
                last_training_epoch=n, test_accuracy=float(r.accuracy_test),
                test_cross_entropy=float(-r.categorical_loglikelihood_test),
                best_validation_loss=float(loss[selected-1])))
    trajectories = pd.DataFrame(rows)
    keys = ['index', 'epoch']
    pd.testing.assert_frame_equal(trajectories.sort_values(keys).reset_index(drop=True)[saved.columns],
                                 saved.sort_values(keys).reset_index(drop=True), check_exact=True)
    endpoints = read('endpoints.csv')
    reconstructed = trajectories[trajectories.epoch.isin(endpoints.budget.unique())].copy()
    reconstructed = reconstructed.rename(columns={'epoch':'budget', 'test_cross_entropy':'test_loss', 'selected_epoch':'best_epoch'})
    check = endpoints.merge(reconstructed, on=['index', 'budget'], suffixes=('_saved', '_replayed'), validate='one_to_one')
    assert len(check) == len(endpoints) == 240
    for column in ['best_epoch', 'best_validation_loss', 'test_accuracy', 'test_loss']:
        np.testing.assert_array_equal(check[column+'_saved'], check[column+'_replayed'])
    summarize(trajectories, endpoints, out)
    for name in ['condition_trajectory_summary.csv','paired_trajectory_summary.csv','paired_seed_trajectories.csv','paired_contrasts.csv']:
        pd.testing.assert_frame_equal(pd.read_csv(out/name, float_precision='round_trip'), read(name), check_exact=True)
    (out/'validation.json').write_text(json.dumps(dict(status='passed', runs=60,
        selected_seed_rows=len(trajectories), observed_epochs=len(histories),
        all_validation_selected_states_reconstructed=True,
        statistics_reproduce_exactly=True,
        scope='Compact numerical reanalysis; tensor and RNG-state checks are documented in the production checkpoint audit.'), indent=2)+'\n')
    print('Verified 60 stopped fits and reproduced all published trajectory statistics.')


if __name__ == '__main__':
    main()
