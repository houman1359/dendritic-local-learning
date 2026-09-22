"""Verify completed runs use the replacement cohort's single Blackwell GPU type."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import time
from pathlib import Path

FIELDS = 'JobID,State,ExitCode,Partition,QOS,NodeList,AllocTRES,Constraints,Restarts'


def validate(text: str, array_id: str, n_tasks: int = 80) -> dict:
    errors = []
    rows = list(csv.DictReader(io.StringIO(text), delimiter='|'))
    tasks = {}
    for row in rows:
        match = re.fullmatch(re.escape(array_id) + r'_(\d+)', row.get('JobID', ''))
        if not match:
            continue
        index = int(match.group(1))
        if index in tasks:
            errors.append(f'duplicate scheduler record for task {index}')
        tasks[index] = row
        if row['State'] != 'COMPLETED' or row['ExitCode'] != '0:0':
            errors.append(f'task {index}: not successfully completed')
        partition = row['Partition']
        if partition not in {'kempner_requeue'}:
            errors.append(f'task {index}: unexpected partition {partition}')
        if partition == 'kempner_requeue' and row['Constraints'] != 'rtx6000pro':
            errors.append(f'task {index}: run lost its RTX constraint')
        tres = dict(token.split('=', 1) for token in row['AllocTRES'].split(',') if '=' in token)
        if tres.get('gres/gpu:nvidia_rtx_pro_6000_blackwell_server_edition') != '1' or tres.get('gres/gpu') != '1':
            errors.append(f'task {index}: allocation is not exactly one RTX PRO 6000 Blackwell')
        if tres.get('cpu') != '8' or tres.get('mem') != '64G' or tres.get('node') != '1':
            errors.append(f'task {index}: CPU, memory or node allocation changed')
    if set(tasks) != set(range(n_tasks)):
        errors.append(f'task inventory differs from 0..{n_tasks-1}')
    return dict(valid=not errors, errors=errors, n_tasks=len(tasks),
                array_id=array_id, task_records=tasks,
                requeued_tasks=[i for i, r in tasks.items() if int(r['Restarts']) > 0],
                partition_counts={p: sum(r['Partition'] == p for r in tasks.values())
                                  for p in ('kempner_requeue',)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--array-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accounting-file', type=Path)
    parser.add_argument('--retries', type=int, default=6)
    parser.add_argument('--retry-delay', type=float, default=10.0)
    args = parser.parse_args()
    for attempt in range(args.retries):
        raw = (args.accounting_file.read_text() if args.accounting_file else
               subprocess.check_output(['sacct', '-X', '-P', '-j', args.array_id,
                                        '--format=' + FIELDS], text=True))
        audit = validate(raw, args.array_id)
        if audit['valid'] or args.accounting_file or attempt + 1 == args.retries:
            break
        time.sleep(args.retry_delay)
    audit['accounting_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
    audit['audit_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.accounting.psv').write_text(raw)
    args.output.write_text(json.dumps(audit, indent=2) + '\n')
    if not audit['valid']:
        raise SystemExit('\n'.join(audit['errors']))
    print(f"Scheduler audit passed: {audit['n_tasks']} completed RTX runs; {audit['partition_counts']}")


if __name__ == '__main__':
    main()
