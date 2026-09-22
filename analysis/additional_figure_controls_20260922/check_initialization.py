from pathlib import Path
import os,json,subprocess,time
R=Path(__file__).resolve().parent
m=json.loads((R/'manifest.json').read_text());results=[]
for i in [0,10,40,70]:
 rec=m['jobs'][i]['runs'][0 if i<20 else 1]
 out=R/'initialization_checks'/rec['key'];out.mkdir(parents=True,exist_ok=False)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled',MPLBACKEND='Agg',PYTHONPATH='/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes/depth-budget-wandb-overlay-20260906:'+rec['source']['path']+'/src')
 cmd=['/n/sw/Mambaforge-23.11.0-0/bin/python','-m','dendritic_modeling.scripts.training.train_experiments',rec['config'],'--validate-only','--output_dir',str(out)]
 start=time.time()
 with (out/'console.log').open('w') as log:p=subprocess.run(cmd,cwd=rec['source']['path'],env=env,stdout=log,stderr=subprocess.STDOUT)
 results.append({'key':rec['key'],'exit_code':p.returncode,'seconds':time.time()-start})
 (R/'initialization_checks.json').write_text(json.dumps(results,indent=2)+'\n')
 print(results[-1],flush=True)
 if p.returncode:print((out/'console.log').read_text()[-5000:],flush=True);raise SystemExit(p.returncode)
