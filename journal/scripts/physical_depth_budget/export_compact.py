#!/usr/bin/env python3
"""Export complete physical-depth histories without thousands of raw files.

The raw run trees and checkpoints remain untouched. The public export preserves
every recorded epoch metric, configuration, final metric and original-file hash.
"""
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from extend import OUT, JOURNAL, dump, sha


TABLES = [
    'extension_endpoints.csv', 'extension_validation_trajectories.csv',
    'extension_source_concordance.csv', 'extension_paired_contrasts.csv',
    'extension_paired_seed_contrasts.csv', 'extension_budget_interactions.csv',
    'extension_summary.csv', 'original_epoch_diagnostics.csv',
    'original_validation_trajectories.csv', 'original_epoch_summary.csv',
]
AUDITS = [
    'extension_protocol.json', 'extension_freeze.json', 'extension_validation.json',
    'original_epoch_audit.json', 'original_checkpoint_inventory.json',
    'benchmark_validation.json', 'environment_overlay.json', 'extension_jobs.json',
]


def main():
    validation = json.loads((OUT / 'extension_validation.json').read_text())
    assert validation['status'] == 'passed' and validation['fits'] == 60
    protocol = json.loads((OUT / 'extension_protocol.json').read_text())
    assert sha(OUT / 'extension_protocol.json') == validation['protocol_sha256']
    for path, digest in protocol['source_sha256'].items():
        assert sha(path) == digest, path

    with tempfile.TemporaryDirectory(prefix='.canonical_export_', dir=OUT) as temporary:
        stage = Path(temporary)
        for name in TABLES + AUDITS + ['EXTENSION_REPORT.md']:
            shutil.copy2(OUT / name, stage / name)
        (stage / 'configurations').mkdir()
        (stage / 'run_records').mkdir()
        history = []
        epoch_metrics = []
        raw_inventory = []
        observations = 0
        for record in protocol['conditions']:
            index = record['index']
            folder = Path(record['results_dir'])
            audit_path = folder / 'extension_audit.json'
            audit = json.loads(audit_path.read_text())
            assert audit['status'] == 'complete' and not audit['benchmark']
            assert audit['protocol_sha256'] == validation['protocol_sha256']
            model = Path(audit['model_results_dir'])
            meta = {key: record[key] for key in ('index', 'arm', 'depth', 'seed')}
            progress = json.loads((folder / 'extension_progress.json').read_text())
            summary = json.loads((model / 'training_summary.json').read_text())
            assert len(progress) == audit['epochs'] == len(summary['valid_losses'])
            for item in progress:
                n = item['epoch'] - 1
                assert item['train_loss'] == summary['train_losses'][n]
                assert item['valid_loss'] == summary['valid_losses'][n]
                history.append(meta | item)
            observations += len(progress)
            files = sorted((model / 'performance/epochs').glob('epoch*.json'),
                           key=lambda path: int(path.stem.removeprefix('epoch')))
            assert len(files) == len(progress)
            for path in files:
                metrics = json.loads(path.read_text())
                flat = {f'{metric}_{split}': value
                        for metric, splits in metrics.items()
                        for split, value in splits.items()}
                epoch_metrics.append(meta | {'epoch': int(path.stem[5:])} | flat)

            # These raw output hashes were checked by analyze_extension.py.
            # Preserve their full inventory, including unpublished checkpoints.
            for path, digest in audit['output_sha256'].items():
                raw_inventory.append(meta | dict(
                    role='extension_raw', path=str((folder / path).relative_to(JOURNAL)),
                    sha256=digest, distributed_in_compact_export=False))
            raw_inventory.append(meta | dict(
                role='extension_run_audit', path=str(audit_path.relative_to(JOURNAL)),
                sha256=sha(audit_path), distributed_in_compact_export=False))
            for role in ('original_config', 'original_training_summary',
                         'original_final_metrics', 'original_final_model'):
                path = Path(record[role])
                digest = sha(path)
                if f'{role}_sha256' in record:
                    assert digest == record[f'{role}_sha256']
                raw_inventory.append(meta | dict(
                    role=role, path=str(path), sha256=digest,
                    distributed_in_compact_export=role == 'original_config'))

            for role, source in [('original', record['original_config']),
                                 ('extension', record['config'])]:
                shutil.copy2(source, stage / 'configurations' / f'condition_{index:02d}_{role}.yaml')
            metrics = {'final': json.loads((model / 'performance/final.json').read_text())}
            metrics['budget180_best'] = (
                json.loads((model / 'performance/budget180_best.json').read_text())
                if audit['epochs'] >= 180 else metrics['final'])
            public_audit = {key: value for key, value in audit.items()
                            if key not in ('output_sha256', 'model_results_dir')}
            public_audit['raw_model_results_path'] = str(model.relative_to(JOURNAL))
            public_audit['raw_inventory'] = 'raw_source_inventory.csv'
            extras = {path.stem: json.loads(path.read_text()) for path in
                      [model / 'model_resources.json', model / 'resolved_seeds.json',
                       model / 'operator_card.json', model / 'init_gate_stats.json']
                      if path.exists()}
            dump(stage / 'run_records' / f'condition_{index:02d}.json', dict(
                condition=meta, audit=public_audit, endpoint_metrics=metrics,
                training_summary={key: value for key, value in summary.items()
                                  if key not in ('train_losses', 'valid_losses')},
                additional_metadata=extras,
                complete_history='observed_training_histories.csv',
                complete_epoch_metrics='observed_epoch_metrics.csv',
                selection='Minimum validation loss; epoch test metrics were never used for selection'))

        pd.DataFrame(history).to_csv(stage / 'observed_training_histories.csv', index=False)
        pd.DataFrame(epoch_metrics).to_csv(stage / 'observed_epoch_metrics.csv', index=False)
        pd.DataFrame(raw_inventory).to_csv(stage / 'raw_source_inventory.csv', index=False)
        assert len(history) == len(epoch_metrics) == observations
        (stage / 'README.md').write_text(
            '# Physical-depth budget sensitivity: complete compact Source Data\n\n'
            'The 60 fits restart the original configurations and seeds with the maximum '
            'budget increased from 180 to 600 epochs. The original stopping rule and '
            'validation selection are retained. These are budget controls on existing '
            'seeds, not a fresh replication or a convergence guarantee.\n\n'
            '`observed_training_histories.csv` retains every observed training and '
            'validation loss, best epoch, patience counter and elapsed time. '
            '`observed_epoch_metrics.csv` retains every metric logged at every epoch, '
            'including test metrics that were never used for selection. The separate '
            'plotting table carries stopped runs forward and marks carried rows.\n\n'
            'Each run has its exact original and extension configurations, endpoint '
            'metrics, resource counts, seed metadata and audit. `raw_source_inventory.csv` '
            'preserves the complete SHA-256 inventory of the underlying raw JSON files '
            'and checkpoints and the original source files. Raw checkpoints and per-epoch '
            'JSON trees are preserved locally; their data are consolidated here. Absolute '
            'paths in frozen protocols and configurations document the source environment.\n\n'
            'Rebuild with `python scripts/physical_depth_budget/analyze_extension.py` and '
            '`python scripts/physical_depth_budget/export_compact.py` after the frozen '
            'extension jobs complete. No original result file is replaced.\n')
        dump(stage / 'export_validation.json', dict(
            status='passed', runs=60, observed_epochs=observations,
            observed_metric_rows=len(epoch_metrics), raw_inventory_rows=len(raw_inventory),
            epoch_test_metrics_used_for_selection=False,
            configurations_copied_without_changes=120,
            underlying_raw_files_preserved=True,
            source_audit_sha256=sha(OUT / 'extension_validation.json'),
            exporter_sha256=sha(__file__),
            raw_hash_verification='All extension raw artifact hashes checked by analyze_extension.py; original source hashes checked during export'))
        manifest = {str(path.relative_to(stage)): sha(path)
                    for path in sorted(stage.rglob('*')) if path.is_file()}
        dump(stage / 'compact_manifest.json', dict(status='complete', files=manifest))
        target = OUT / 'canonical'
        if target.exists():
            shutil.rmtree(target)
        stage.rename(target)
    print(json.dumps(dict(destination=str(target), runs=60, observed_epochs=observations,
                          files=len(manifest) + 1, size_bytes=sum(path.stat().st_size for path in target.rglob('*') if path.is_file()))))


if __name__ == '__main__':
    main()
