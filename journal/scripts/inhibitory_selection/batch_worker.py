"""Run frozen indices within one CPU allocation, without changing the study."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys


root = Path(os.environ['SELECTION_ROOT'])
phase = os.environ['SELECTION_PHASE']
protocol = json.loads((root / f'{phase}_protocol.json').read_text())
workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))


def run(index):
    path = root / phase / 'logs' / f"batch_{os.environ['SLURM_JOB_ID']}_{index}.out"
    with path.open('x') as handle:
        subprocess.run([sys.executable, '-B', str(root / 'study/experiment.py'),
                        '--root', str(root), '--phase', phase, '--index', str(index)],
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    return index


with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    for index in pool.map(run, range(len(protocol['jobs']))):
        print(f'{phase} index {index} completed', flush=True)
