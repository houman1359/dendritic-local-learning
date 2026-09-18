"""Publish the complete exploratory matrix, including unfavorable outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root, journal):
    protocol = json.loads((root / 'protocol.json').read_text())
    for name, digest in protocol['source_sha256'].items():
        assert sha(root / name) == digest, name
    records, hashes = [], {}
    expected = {(j['seed'], j['somata'], j['streams'], j['task'], j['mode'], r, lr)
                for j in protocol['jobs'] for r in
                ['exact', 'broadcast', 'resistance', 'swapped', 'uniform_rms']
                for lr in protocol['rates']}
    keys = ['seed', 'somata', 'streams', 'task', 'forward', 'rule', 'rate']
    for path in sorted((root / 'results').glob('*.json')):
        r = json.loads(path.read_text())
        assert r['protocol_sha256'] == sha(root / 'protocol.json')
        checkpoint = root / 'checkpoints' / path.with_suffix('.pt').name
        assert sha(checkpoint) == r['checkpoint_sha256']
        row = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        row.update({f'ood_{s}': v for s, v in r['generalization_nmse'].items()})
        records.append(row)
        hashes[path.name] = sha(path)
    frame = pd.DataFrame(records)
    assert len(frame) == len(expected) == 540
    assert set(map(tuple, frame[keys].itertuples(index=False, name=None))) == expected
    groups = keys[1:]
    rates = frame.groupby(groups, as_index=False).validation_nmse.mean()
    chosen = rates.sort_values('validation_nmse').drop_duplicates(groups[:-1])
    selected = frame.merge(chosen[groups], on=groups, validate='many_to_one')
    assert len(selected) == 270
    summary = selected.groupby(groups, as_index=False).agg(
        n=('seed', 'nunique'), test_nmse=('test_nmse', 'mean'),
        ood_1=('ood_1.0', 'mean'), ood_2=('ood_2.0', 'mean'), ood_3=('ood_3.0', 'mean'),
        bound_runs=('bound_steps', lambda x: int((x > 0).sum())),
        at_training_cap=('best_step', lambda x: int((x == 1024).sum())))
    out = journal / 'source_data/curated_publication'
    outputs = {'endpoints': frame, 'selected': selected, 'summary': summary}
    for name, table in outputs.items():
        path = out / f'inhibitory_transfer_pilot_{name}.csv'
        if path.exists():
            raise FileExistsError(path)
        table.to_csv(path, index=False)
    v = 1 - np.tanh(2) / 2
    provenance = dict(protocol=protocol, protocol_sha256=sha(root / 'protocol.json'),
        endpoint_sha256=hashes, trajectories=540, paired_seeds=3,
        interpretation='Exploratory development only; validation-selected rates; all conditions retained; no confirmatory inference',
        conditional_additivity_floor=float(v / (8 + v)),
        rate_at_grid_maximum=int((chosen.rate == .03).sum()),
        selected_runs_at_training_cap=int((selected.best_step == 1024).sum()),
        selected_runs_hitting_bounds=int((selected.bound_steps > 0).sum()),
        publisher_sha256=sha(__file__),
        outputs={name: sha(out / f'inhibitory_transfer_pilot_{name}.csv') for name in outputs})
    path = out / 'inhibitory_transfer_pilot_provenance.json'
    with path.open('x') as handle:
        json.dump(provenance, handle, indent=2, allow_nan=False)
    print(summary[(summary.somata == 16) & (summary.streams == 4)].to_string(index=False))
    print({k: v for k, v in provenance.items() if isinstance(v, (int, float, str))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--journal', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.journal)
