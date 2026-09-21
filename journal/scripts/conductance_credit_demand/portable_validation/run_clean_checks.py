"""Excluded reviewer-environment installation checks, not paper outcomes."""
import argparse,json,os,subprocess,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];J=HERE.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--python',type=Path,required=True);args=parser.parse_args();PYTHON=args.python.absolute()
scratch=Path(tempfile.mkdtemp(prefix='conductance-portable-checks-'));env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
cases=[('first_smoke','first','fresh',1101,4),('opponent_smoke','opponent','fresh',2101,4),('opponent_full_primary','opponent','fresh',2101,None),('opponent_full_extension','opponent','extension',2101,None)]
results=[]
for name,family,phase,seed,budget in cases:
 out=scratch/name;command=[str(PYTHON),str(HERE/'portable_run.py'),'--study-root',str(J/'source_data/conductance_credit_demand'),'--journal-root',str(J),'--output-root',str(out),'--family',family,'--phase',phase,'--seed',str(seed)]
 if budget is not None:command+=['--excluded-smoke-steps',str(budget)]
 p=subprocess.run(command,env=env,text=True,capture_output=True)
 if p.returncode:
  print(p.stdout);print(p.stderr);raise RuntimeError(name+' failed')
 audit=json.loads((out/'portable_audit.json').read_text());(HERE/'portable_validation'/f'{name}_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n');results.append(dict(case=name,status=audit['status'],mode=audit['mode'],n_fits=audit['n_fits'],steps=audit['steps'],actual_environment=audit['actual_environment'],original_environment=audit['original_environment'],elapsed_seconds=audit['elapsed_seconds']));print(name,audit['status'],audit['actual_environment'],flush=True)
(HERE/'portable_validation'/'clean_environment_summary.json').write_text(json.dumps(dict(excluded_from_paper_outcomes=True,checks=results),indent=2)+'\n');print('scratch',scratch)
