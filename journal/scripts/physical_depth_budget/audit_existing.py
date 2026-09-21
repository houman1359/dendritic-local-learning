#!/usr/bin/env python3
"""Recover the published depth trajectories without changing original results."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

JOURNAL = Path(__file__).resolve().parents[2]
ARCHIVE = JOURNAL.parents[1] / 'dendritic-local-learning-worktree-archive-20260817/journal'
OUT = JOURNAL / 'source_data/physical_depth_budget'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, curves, manifest = [], [], []
    for cohort, filename in [('nonlinear_physical_depth_confirmatory', 'seed_outcomes.csv'),
                             ('point_dendrite_credit_controls', 'new_seed_outcomes.csv')]:
        source = JOURNAL / 'source_data' / cohort / filename
        data = pd.read_csv(source)
        manifest.append({'path': str(source), 'sha256': sha(source)})
        if cohort == 'nonlinear_physical_depth_confirmatory':
            part = data[(data.regime == 'aligned') & (data.mechanism == 'shunting')
                        & data.depth.isin([1, 3])]
        else:
            part = data[(data.regime == 'aligned')
                        & data.run_dir.str.contains('soma_broadcast')
                        & data.depth.isin([1, 3])]
        for _, row in part.iterrows():
            run = ARCHIVE / row.run_dir
            config = run / 'configs' / f'unified_config_{int(row.config_index)}.yaml'
            result = run / 'results' / f'config_{int(row.config_index)}'
            metrics = result / 'performance/final.json'
            summary_path = result / 'training_summary.json'
            if sha(config) != row.config_sha256 or sha(metrics) != row.final_sha256:
                raise ValueError(f'Archived source differs from publication record: {result}')
            summary = json.loads(summary_path.read_text())
            losses = np.asarray(summary['valid_losses'], dtype=float)
            if not np.all(np.isfinite(losses)):
                raise ValueError(f'Nonfinite validation losses: {result}')
            arm = row.get('transport', 'unspecified')
            if 'matched_optimizer' in row.run_dir:
                arm = 'broadcast_autograd_localca_recipe'
            elif 'soma_broadcast_bp' in row.run_dir:
                arm = 'broadcast_autograd_bp_recipe'
            elif row.get('method') == 'bp':
                arm = 'exact_autograd_bp_recipe'
            key = dict(cohort=cohort, arm=arm, depth=int(row.depth), seed=int(row.seed))
            record = key | dict(
                epochs_run=len(losses), best_epoch=int(summary['best_epoch']),
                at_180_cap=len(losses) == 180,
                valid_loss_final=float(losses[-1]),
                valid_loss_drop_last_10=float(losses[-11] - losses[-1]),
                valid_loss_slope_last_30=float(np.polyfit(np.arange(30), losses[-30:], 1)[0]),
                test_accuracy=float(row.test_accuracy),
                config_path=str(config), config_sha256=sha(config),
                training_summary_path=str(summary_path), training_summary_sha256=sha(summary_path),
                final_metrics_path=str(metrics), final_metrics_sha256=sha(metrics),
                final_model_path=str(result / 'final_model.pt'))
            records.append(record)
            for epoch, loss in enumerate(losses, 1):
                curves.append(key | dict(epoch=epoch, valid_loss=float(loss)))
    rows = pd.DataFrame(records)
    if rows.duplicated(['arm', 'depth', 'seed']).any():
        raise ValueError('Duplicate physical-depth trajectory')
    rows.to_csv(OUT / 'original_epoch_diagnostics.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT / 'original_validation_trajectories.csv', index=False)
    grouped = rows.groupby(['arm', 'depth']).agg(
        n=('seed', 'count'), cap_count=('at_180_cap', 'sum'),
        mean_best_epoch=('best_epoch', 'mean'),
        mean_late_drop=('valid_loss_drop_last_10', 'mean'),
        mean_late_slope=('valid_loss_slope_last_30', 'mean'),
        test_accuracy=('test_accuracy', 'mean')).reset_index()
    grouped.to_csv(OUT / 'original_epoch_summary.csv', index=False)
    report = dict(status='pass', trajectories=len(rows), source_manifest=manifest,
                  all_publication_config_and_final_metric_hashes_match=True,
                  original_source_files_unchanged=True, new_training_runs=0,
                  scope='Retrospective recovery of recorded stopping and loss curves. '
                        'A declining validation loss does not prove the test-accuracy effect '
                        'will increase or that it is a lower bound on a converged effect.')
    (OUT / 'original_epoch_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(grouped.to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
