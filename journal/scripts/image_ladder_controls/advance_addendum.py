#!/usr/bin/env python3
"""Joint scheduling after108developmentfits; independent frozen validationchoices."""
import json,subprocess,sys
from run import HERE,OUT,RAW,PROJECT,dump,sha

def main():
    subprocess.run([sys.executable,str(HERE/'projected_k1.py'),'select'],check=True,cwd=PROJECT)
    subprocess.run([sys.executable,str(HERE/'advance.py')],check=True,cwd=PROJECT)
    base=json.loads((OUT/'fresh_scheduler.json').read_text())
    subprocess.run(['scontrol','update','JobId='+base['fresh_job_id'],'ArrayTaskThrottle=8'],check=True)
    records=json.loads((OUT/'projected_k1_fresh_conditions.json').read_text())
    job=subprocess.check_output(['sbatch','--parsable',f'--array=0-{len(records)-1}%4',f'--output={RAW}/projected_k1_fresh_%a.out',str(HERE/'projected_k1_worker.sh'),'fresh'],text=True,cwd=PROJECT).strip().split(';')[0]
    subprocess.run([sys.executable,str(HERE/'prepare_portable.py')],check=True,cwd=PROJECT)
    dump(OUT/'projected_k1/fresh_scheduler.json',dict(job_id=job,fresh_count=len(records),base_fresh_job_id=base['fresh_job_id'],aggregate_gpu_concurrency=12))
    print(json.dumps(json.loads((OUT/'projected_k1/fresh_scheduler.json').read_text())),flush=True)
if __name__=='__main__':main()
