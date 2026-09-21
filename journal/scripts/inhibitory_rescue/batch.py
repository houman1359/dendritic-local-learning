"""Bounded CPU parallelism within one scheduler allocation."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(os.environ['RESCUE_ROOT'])
phase = os.environ['RESCUE_PHASE']
protocol = json.loads((root / f'{phase}_protocol.json').read_text())


def run(index):
    log = root / phase / 'logs' / f"job_{os.environ.get('SLURM_JOB_ID', 'local')}_{index}.out"
    with log.open('x') as handle:
        subprocess.run([sys.executable, '-B', str(root / 'study/rescue.py'),
                        '--root', str(root), '--phase', phase, '--index', str(index)],
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(f'{phase} fit {index} complete', flush=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))) as pool:
    list(pool.map(run, range(len(protocol['jobs']))))
