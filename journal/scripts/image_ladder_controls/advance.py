#!/usr/bin/env python3
"""Advance only after the complete frozen development cohort succeeds."""
import json,subprocess,sys
from pathlib import Path
from run import HERE,OUT,RAW,PROJECT,dump,sha

def main():
    subprocess.run([sys.executable,str(HERE/'run.py'),'select'],check=True,cwd=PROJECT)
    records=json.loads((OUT/'fresh_conditions.json').read_text())
    result=subprocess.check_output(['sbatch','--parsable',f'--array=0-{len(records)-1}%12',f'--output={RAW}/fresh_%a.out',str(HERE/'worker.sh'),'fresh'],text=True,cwd=PROJECT).strip()
    job=result.split(';')[0]
    capture=subprocess.check_output(['sbatch','--parsable',f'--dependency=afterok:{job}','--array=0-19%8',f'--output={RAW}/capture_fresh_%a.out',str(HERE/'capture_worker.sh'),'fresh'],text=True,cwd=PROJECT).strip().split(';')[0]
    dump(OUT/'fresh_scheduler.json',dict(fresh_job_id=job,fresh_count=len(records),capture_job_id=capture,selection_sha256=sha(OUT/'selection.json')))
    print(json.dumps(json.loads((OUT/'fresh_scheduler.json').read_text())),flush=True)
if __name__=='__main__':main()
