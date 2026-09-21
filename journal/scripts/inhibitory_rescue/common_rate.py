"""Specify missing common-rate fits without changing the primary protocol.

This addendum is made during primary fresh execution, before inspecting its
outcomes. It uses the augmented rule's development-selected Adam rate, not a
test-selected rate. Reused seeds are not an additional independent cohort.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from protocol import sha, save_new


def prepare(root, primary):
    p = json.loads((primary / 'fresh_protocol.json').read_text())
    rate = p['fixed_rates']['adam_b9_derivative']
    missing = [r for r in ['broadcast', 'resistance', 'shuffled_derivative']
               if p['fixed_rates'][f'adam_b9_{r}'] != rate]
    root.mkdir(parents=True, exist_ok=False)
    for name in ['study', 'selection', 'base', 'runtime']:
        (root / name).symlink_to((primary / name).resolve(), target_is_directory=True)
    for name in ['results', 'logs', 'checkpoints']:
        (root / 'fresh' / name).mkdir(parents=True)
    shutil.copyfile(__file__, root / 'common_rate_preparation.py')
    protocol = {**p, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'status': 'Secondary common-rate addendum specified during fresh execution before inspection of fresh outcomes',
                'primary_protocol_sha256': sha(primary / 'fresh_protocol.json'),
                'common_rate': rate, 'primary': [],
                'inference': 'Secondary paired differences on the original 20 seeds, descriptive 95% seed-bootstrap intervals; not an independent replication',
                'jobs': [dict(seed=s, rule=r, optimizer='adam', bound=9, rate=rate)
                         for s in p['fresh_seeds'] for r in missing]}
    save_new(root / 'fresh_protocol.json', protocol)
    print(root, len(protocol['jobs']), 'additional fits at rate', rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.primary)
