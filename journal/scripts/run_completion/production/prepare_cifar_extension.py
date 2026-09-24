from pathlib import Path
from datetime import datetime, timezone
import copy, csv, hashlib, json, subprocess, yaml

R = Path(__file__).resolve().parent
J = Path('/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/journal')
B = R.parent
C = B/'cifar_shunting_revision_20260922'
S = Path(json.loads((C/'requeue_frozen_paths.json').read_text())['sweep_root'])
RUNTIME = B/'journal_extension_20260827/runtime/dendritic-modeling-cifar-e516c7'
COMMIT = 'e516c7fec3169253ff8c14bc5f4ab1325469e4f5'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, d): p.write_text(json.dumps(d, indent=2)+'\n')

assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True).strip()==COMMIT
assert not subprocess.check_output(['git','diff','--binary','HEAD'],cwd=RUNTIME)
with (J/'source_data/cifar10_shunting_feedback_ladder_confirmatory/seed_outcomes.csv').open() as f:
    outcomes=list(csv.DictReader(f))
cap_seeds=sorted({int(x['seed']) for x in outcomes if int(x['n_epochs_recorded'])==400})
assert cap_seeds==[22001,22008]
D=R/'cifar_extension';D.mkdir(exist_ok=True)
for name in ['configs','results','logs']: (D/name).mkdir(exist_ok=True)
records=[]
for row in outcomes:
    if int(row['seed']) not in cap_seeds: continue
    index=int(row['config_index']);original=S/'configs'/f'unified_config_{index}.yaml'
    cfg=yaml.safe_load(original.read_text());before=copy.deepcopy(cfg)
    assert cfg['experiment']['seed']==int(row['seed'])
    common=cfg['training']['main']['common']
    assert common['epochs']==400 and common['early_stopping'] and common['patience']==50
    assert common['lr_schedule']=='none' and common['lr_warmup_epochs']==0
    key=f'config_{index}';out=D/'results'/key
    common['epochs']=1600
    cfg['outputs']['results_dir']=str(out)
    cfg['outputs']['exact_run_dir']=True
    cfg['outputs']['run_name']='cifar_stopping_extension_'+key
    check=copy.deepcopy(cfg);check['training']['main']['common']['epochs']=400;check['outputs']=before['outputs']
    assert check==before
    path=D/'configs'/f'{key}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    records.append(dict(index=len(records),config_index=index,seed=int(row['seed']),feedback=row['feedback'],
                        config=str(path),config_sha256=sha(path),result_dir=str(out),
                        original_config=str(original),original_config_sha256=sha(original),
                        original_results=str(S/'results'/key)))
assert len(records)==8
carried=[dict(seed=int(x['seed']),feedback=x['feedback'],config_index=int(x['config_index']),
              epochs=int(x['n_epochs_recorded']),best_epoch=int(x['best_epoch']))
         for x in outcomes if int(x['seed']) not in cap_seeds]
assert len(carried)==72 and all(x['epochs']<400 and x['epochs']-x['best_epoch']>=49 for x in carried)
protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),runtime=str(RUNTIME),runtime_commit=COMMIT,
  study='Existing-seed stopping-limit amendment; original results already inspected',
  rationale='Both cap-limited seeds are selected solely by training length. Replay all four rules in each to retain matched comparisons. The other 18 seed blocks already stopped under the unchanged rule and need no additional epochs.',
  changes='Maximum 400 to 1600 epochs; output paths and names; passive progress/snapshot observer only. Same seed, data, calibration, optimizer, rates and patience 50. No test-based stopping or hyperparameter selection.',
  checkpoint_limitation='Original final_model.pt contains model weights only, so affected seed blocks restart from original seeds; not claimed as exact checkpoint continuation.',
  comparison='Compare original-length loss histories, initial/calibrated states and validation-selected endpoints. Disclose numerical replay drift. Original 400-epoch analysis is immutable; extended analysis has independent provenance.',
  stopping='Require ordinary patience-based stopping for every displayed extended-cohort run. If any run reaches 1600, retain its complete state and investigate before integration; never relabel it converged.',
  interpretation='This is a disclosed post-review budget extension, not a new prospectively unobserved confirmation cohort. Retain the original statistical contrasts and every seed regardless of direction.',
  records=records,carried_early_stopped=carried,
  original_analysis={p.name:sha(p) for p in (J/'source_data/cifar10_shunting_feedback_ladder_confirmatory').iterdir() if p.name in ['summary.json','condition_summary.csv','paired_contrasts.csv','seed_outcomes.csv']})
dump(D/'protocol.json',protocol)
print('Prepared eight matched reruns and verified 72 already-stopped records.')
