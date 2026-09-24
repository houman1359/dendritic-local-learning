"""Audit the amended stopping experiment using the original scientific tests."""
from pathlib import Path
import copy,hashlib,importlib.util,json,sys
import numpy as np
import pandas as pd
import torch

R=Path(__file__).resolve().parent;D=R/'cifar_extension';J=Path('/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/journal')
BASE=J/'source_data/cifar10_shunting_feedback_ladder_confirmatory'
FROZEN=R.parent/'cifar_shunting_revision_20260922/analyzer_requeue_frozen.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(FROZEN)=='6cb0f1b9d16602a26118714433fb3922e79027ff09053f31f7a7c7b48032bcba'
spec=importlib.util.spec_from_file_location('frozen_cifar_analysis',FROZEN)
A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)

def comparable(cfg):
    cfg=copy.deepcopy(cfg);cfg.pop('outputs',None)
    cfg['training']['main']['common']['epochs']=400
    return cfg
def states(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    for key in ['state_dict','model_state_dict']:
        if key in obj:return obj[key]
    return obj
def main():
    protocol=json.loads((D/'protocol.json').read_text())
    for name,digest in protocol['original_analysis'].items():assert sha(BASE/name)==digest,name
    frame=pd.read_csv(BASE/'seed_outcomes.csv',float_precision='round_trip')
    original_sweep=Path(json.loads((R/'baseline.json').read_text())['cifar_sweep'])
    reruns={r['config_index']:r for r in protocol['records']}
    missing=[r['result_dir'] for r in reruns.values() if not (Path(r['result_dir'])/'execution.json').is_file()]
    out=D/'analysis';out.mkdir(exist_ok=True)
    if missing:
        (out/'incomplete.json').write_text(json.dumps({'missing':missing},indent=2)+'\n')
        raise RuntimeError('Incomplete stopping-extension cohort')
    revised=[];state_checks=[];replay=[];projections={};calibrations={};flags=[]
    for _,prior in frame.iterrows():
        index=int(prior.config_index);rec=reruns.get(index)
        folder=Path(rec['result_dir']) if rec else original_sweep/'results'/f'config_{index}'
        resolved=json.loads((folder/'config.json').read_text())
        if rec:
            execution=json.loads((folder/'execution.json').read_text())
            assert execution['status']=='complete' and not execution['smoke']
            assert execution['runtime_commit']==protocol['runtime_commit'] and 'H200' in execution['gpu']
            assert execution['config_sha256']==rec['config_sha256']==sha(rec['config'])
            old=json.loads((Path(rec['original_results'])/'config.json').read_text())
            assert comparable(resolved)==comparable(old),('unexpected resolved scientific difference',index)
            replay.append({k:execution[k] for k in ['index','prefix_epochs','max_abs_validation_replay_drift']})
        cap=1600 if rec else 400
        A.EXPECTED_RESOLVED_FIELDS['training.main.common.epochs']=cap
        errors=A.audit_resolved_config(resolved,prior.condition,A.CONDITIONS[prior.condition],folder)
        assert not errors,errors
        convergence=A.convergence_record(folder,resolved,prior.condition);assert convergence['valid'],convergence
        assert convergence['n_epochs_recorded']-convergence['best_epoch']>=49 or convergence['n_epochs_recorded']==cap
        calibration=A.calibration_record(folder);assert calibration['valid'],calibration
        projections.setdefault(int(prior.seed),set()).add(A.canonical_sha256(A.pairing_projection(resolved)))
        calibrations.setdefault(int(prior.seed),set()).add(calibration['calibration_signature_sha256'])
        if convergence['right_censored'] or convergence['n_epochs_recorded']==cap:flags.append(dict(seed=int(prior.seed),condition=prior.condition,details=convergence))
        final=json.loads((folder/'performance/final.json').read_text())
        for split in ['valid','test']:assert np.isfinite(final['accuracy'][split]) and 0<=final['accuracy'][split]<=1
        checkpoint=folder/'main_network'/('standard_best_model.pt' if prior.feedback=='backpropagation' else 'local_learning_best_model.pt')
        selected=states(checkpoint);restored=states(folder/'final_model.pt')
        assert selected.keys()==restored.keys() and all(torch.equal(selected[k],restored[k]) for k in selected)
        state_checks.append(dict(seed=int(prior.seed),feedback=prior.feedback,selected_final_tensors_equal=True,source=str(folder),config_sha256=sha(folder/'config.json'),final_sha256=sha(folder/'final_model.pt')))
        row=prior.to_dict()
        row.update(validation_accuracy=final['accuracy']['valid'],test_accuracy=final['accuracy']['test'],
          n_epochs_recorded=convergence['n_epochs_recorded'],best_epoch=convergence['best_epoch'],
          best_validation_loss=convergence['best_validation_loss'],final10_validation_slope=convergence['final10_validation_slope'],
          right_censored=bool(convergence['right_censored']),resolved_config_sha256=sha(folder/'config.json'),
          generated_config_sha256=sha(rec['config']) if rec else prior.generated_config_sha256,
          result_sha256=sha(folder/'performance/final.json'),checkpoint_sha256=sha(checkpoint),
          training_summary_sha256=sha(folder/'training_summary.json'),calibration_sha256=calibration['calibration_sha256'],
          calibration_signature_sha256=calibration['calibration_signature_sha256'],
          pairing_projection_sha256=A.canonical_sha256(A.pairing_projection(resolved)),
          actual_epoch_cap=cap,extended_policy_cap=1600,source_record='matched seed-block replay' if rec else 'original ordinary early stop')
        revised.append(row)
    assert all(len(x)==1 for x in projections.values()) and all(len(x)==1 for x in calibrations.values())
    revised=pd.DataFrame(revised);assert len(revised)==80 and len(revised[['condition','seed']].drop_duplicates())==80
    audit=dict(status='complete_and_validated' if not flags else 'complete_with_convergence_flags',
      integrity_valid=True,convergence_valid=not flags,n_expected=80,n_results_complete=80,
      convergence_flags=flags,all_seeds_retained=True,matched_replays=8,original_early_stops=72)
    conditions,contrasts,decision=A.summarize(revised,audit)
    revised.to_csv(out/'seed_outcomes.csv',index=False);conditions.to_csv(out/'condition_summary.csv',index=False);contrasts.to_csv(out/'paired_contrasts.csv',index=False)
    pd.DataFrame(state_checks).to_csv(out/'checkpoint_audit.csv',index=False)
    summary=dict(audit=audit,decision=decision,extension_protocol_sha256=sha(D/'protocol.json'),
      original_analyzer_sha256=sha(FROZEN),extension_analyzer_sha256=sha(__file__),
      original_design_frozen_before_original_outcomes=True,extension_frozen_before_extension_outcomes=True,
      original_results_already_inspected=True,scope='Post-review stopping-limit extension on reused seeds; original 400-epoch records preserved',
      replay_diagnostics=replay,original_analysis_sha256=protocol['original_analysis'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(audit,indent=2))
    if flags:raise RuntimeError('Extended runs still contact the limit; investigate before integration')

if __name__=='__main__':main()
