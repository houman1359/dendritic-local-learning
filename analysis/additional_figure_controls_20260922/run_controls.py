"""Execute frozen matched controls using unmodified historical implementations."""
from pathlib import Path
import argparse, copy, hashlib, json, os, subprocess, sys, time, yaml

R = Path(__file__).resolve().parent
OVERLAY = '/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes/depth-budget-wandb-overlay-20260906'
PYTHON = '/n/sw/Mambaforge-23.11.0-0/bin/python'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_source(source):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=source['path'],text=True).strip()==source['commit']
    assert not subprocess.check_output(['git','diff','--binary','HEAD'],cwd=source['path'])

def run(rec, smoke=False):
    verify_source(rec['source'])
    assert sha(rec['config'])==rec['config_sha256']
    path=Path(rec['config']); out=Path(rec['result_dir'])
    if smoke:
        out=R/'smoke'/rec['key']
        cfg=yaml.safe_load(path.read_text())
        cfg['outputs']['results_dir']=str(out)
        cfg['training']['main']['common']['epochs']=2
        cfg['training']['main']['common']['early_stopping']=False
        path=R/'smoke_configs'/path.name;path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy()
    env.update(PYTHONPATH=OVERLAY+':'+rec['source']['path']+'/src',
               OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1',
               MPLBACKEND='Agg',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
    receipt={'key':rec['key'],'original_config_sha256':rec['config_sha256'],
             'executed_config_sha256':sha(path),'source':rec['source'],
             'smoke':smoke,'slurm_job_id':env.get('SLURM_JOB_ID'),
             'array_job_id':env.get('SLURM_ARRAY_JOB_ID'),
             'array_task_id':env.get('SLURM_ARRAY_TASK_ID'),
             'node':env.get('SLURMD_NODENAME'),'started_unix':time.time()}
    receipt['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],text=True).strip()
    assert 'H200' in receipt['gpu'],receipt['gpu']
    (out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    command=[PYTHON,'-m','dendritic_modeling.scripts.training.train_experiments',str(path)]
    print('Starting',rec['key'],'smoke' if smoke else 'full',flush=True)
    with (out/'console.log').open('w') as log:
        result=subprocess.run(command,cwd=rec['source']['path'],env=env,stdout=log,stderr=subprocess.STDOUT)
    receipt.update(exit_code=result.returncode,finished_unix=time.time())
    (out/'execution.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode:
        print((out/'console.log').read_text()[-6000:],flush=True)
        raise RuntimeError('Training failed: '+rec['key'])
    assert (out/'performance/final.json').is_file(),out
    print('Completed',rec['key'],round(receipt['finished_unix']-receipt['started_unix'],1),'seconds',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--index',type=int)
    args=parser.parse_args()
    subprocess.run(['sha256sum','--check','frozen.sha256'],cwd=R,stdout=subprocess.DEVNULL,check=True)
    manifest=json.loads((R/'manifest.json').read_text())
    if args.preflight:
        indices=[0,10,20,30,40,50,60,70,80]
        for i in indices:
            for rec in manifest['jobs'][i]['runs']:run(rec,smoke=True)
        from validate_controls import validate_results
        validate_results(manifest,indices,smoke=True)
        (R/'preflight_passed.json').write_text(json.dumps({'indices':indices,'runs':22,'finished_unix':time.time(),'manifest_sha256':sha(R/'manifest.json')},indent=2)+'\n')
    else:
        proof=json.loads((R/'preflight_passed.json').read_text())
        assert proof['manifest_sha256']==sha(R/'manifest.json')
        for rec in manifest['jobs'][args.index]['runs']:run(rec)
        from validate_controls import validate_results
        validate_results(manifest,[args.index],smoke=False)

if __name__=='__main__':main()
