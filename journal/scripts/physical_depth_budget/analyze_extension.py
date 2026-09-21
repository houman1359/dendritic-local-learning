#!/usr/bin/env python3
"""Retain every extended trajectory and compare its 180/600-epoch windows."""
import json
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from extend import OUT,sha,dump

CONTRASTS=[
 ('depth_gain_exact_bp',('exact_autograd_bp_recipe',3),('exact_autograd_bp_recipe',1)),
 ('localca_path_minus_shared',('path_transport',3),('per_soma_shared',3)),
 ('bp_exact_minus_broadcast',('exact_autograd_bp_recipe',3),('broadcast_autograd_bp_recipe',3)),
 ('broadcast_bp_minus_localca_recipe',('broadcast_autograd_bp_recipe',3),('broadcast_autograd_localca_recipe',3))]


def boot(values):
    v=np.asarray(values,float);rng=np.random.default_rng(601806)
    samples=v[rng.integers(len(v),size=(10000,len(v)))].mean(axis=1)
    return dict(mean=float(v.mean()),ci95_low=float(np.quantile(samples,.025)),ci95_high=float(np.quantile(samples,.975)),n_seeds=len(v),positive_seeds=int((v>0).sum()))


def main():
    import torch
    protocol=json.loads((OUT/'extension_protocol.json').read_text());rows=[];curves=[];replay=[];state_checks=[]
    for record in protocol['conditions']:
        folder=Path(record['results_dir']);audit=json.loads((folder/'extension_audit.json').read_text())
        assert audit['status']=='complete' and not audit['benchmark']
        for file,digest in audit['output_sha256'].items():assert sha(folder/file)==digest
        original_config=yaml.safe_load(Path(record['original_config']).read_text())
        extension_config=yaml.safe_load(Path(record['config']).read_text())
        restored=copy.deepcopy(extension_config)
        restored['training']['main']['common']['epochs']=original_config['training']['main']['common']['epochs']
        for key in ['results_dir','run_name']:restored['outputs'][key]=original_config['outputs'][key]
        assert restored==original_config
        assert extension_config['training']['main']['common']['epochs']==600
        meta={k:record[k] for k in ['index','arm','depth','seed']}
        model_folder=Path(audit['model_results_dir'])
        resources=json.loads((model_folder/'model_resources.json').read_text())
        assert resources['trainable_parameters']==66178
        assert resources['active_synapses']==14336
        summary=json.loads((model_folder/'training_summary.json').read_text())
        losses=np.array(summary['valid_losses'],float);assert np.isfinite(losses).all()
        snapshot_path=folder/f'extension_state_{len(losses)}.pt'
        snapshot=torch.load(snapshot_path,map_location='cpu',weights_only=False)
        final_state=torch.load(model_folder/'final_model.pt',map_location='cpu',weights_only=True)
        assert set(snapshot['best_model'])==set(final_state)
        assert all(torch.equal(snapshot['best_model'][key],value) for key,value in final_state.items())
        assert snapshot['valid_losses']==summary['valid_losses']
        assert snapshot['optimizer']['state']['state']
        assert set(snapshot['rng'])=={'python','numpy','torch','cuda'}
        state_checks.append(meta|dict(config_changes_limited_to_epoch_cap_and_output_names=True,
            validation_selected_final_model_exactly_matches_saved_best_state=True,
            optimizer_and_global_random_states_available=True,
            snapshot_epoch=len(losses),snapshot_sha256=sha(snapshot_path),
            final_model_sha256=sha(model_folder/'final_model.pt')))
        best=np.minimum.accumulate(losses)
        final=json.loads((model_folder/'performance/final.json').read_text())
        if len(losses)>=180:
            original_window=json.loads((model_folder/'performance/budget180_best.json').read_text())
        else:original_window=final
        for budget,metrics in [(180,original_window),(600,final)]:
            n=min(budget,len(losses));window=losses[:n]
            assert abs(-metrics['categorical_loglikelihood']['valid']-window.min())<2e-6
            rows.append(meta|dict(budget=budget,epochs_run=n,full_run_epochs=len(losses),
                best_valid_loss=float(window.min()),best_epoch_one_based=int(np.argmin(window)+1),
                test_accuracy=float(metrics['accuracy']['test']),valid_accuracy=float(metrics['accuracy']['valid']),
                test_loss=float(-metrics['categorical_loglikelihood']['test']),
                at_budget_cap=n==budget,late_validation_loss_slope=float(np.polyfit(np.arange(min(30,n)),window[-30:],1)[0]),
                late_best_loss_drop=float(np.minimum.accumulate(window)[max(0,n-31)]-window.min()),
                checkpoint_selection='Minimum validation loss within budget; original patience30 retained'))
        # Carry a stopped run's best validation loss forward. Every plotted point has ten seed blocks.
        for epoch in range(1,601):
            curves.append(meta|dict(epoch=epoch,best_validation_loss=float(best[min(epoch,len(best))-1]),
                observed_epoch=epoch<=len(best),recorded_validation_loss=float(losses[min(epoch,len(losses))-1])))
        historical=json.loads(Path(record['original_final_metrics']).read_text())
        replay.append(meta|{k:audit[k] for k in ['first_window_epochs','archived_validation_max_abs_difference','archived_validation_rms_difference','elapsed_seconds','device']}|dict(historical_test_accuracy=float(historical['accuracy']['test']),extension180_test_accuracy=float(original_window['accuracy']['test']),test_accuracy_difference_pp=100*float(original_window['accuracy']['test']-historical['accuracy']['test'])))
    df=pd.DataFrame(rows);df.to_csv(OUT/'extension_endpoints.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'extension_validation_trajectories.csv',index=False)
    pd.DataFrame(replay).to_csv(OUT/'extension_source_concordance.csv',index=False)
    pd.DataFrame(state_checks).to_csv(OUT/'extension_checkpoint_validation.csv',index=False)
    contrasts=[];seedrows=[]
    for budget,group in df.groupby('budget'):
        wide=group.pivot(index='seed',columns=['arm','depth'],values='test_accuracy')
        for name,left,right in CONTRASTS:
            values=100*(wide[left]-wide[right])
            contrasts.append(dict(budget=int(budget),contrast=name,**boot(values)))
            seedrows.extend(dict(budget=int(budget),contrast=name,seed=int(seed),difference_pp=float(v)) for seed,v in values.items())
    contrastdf=pd.DataFrame(contrasts);contrastdf.to_csv(OUT/'extension_paired_contrasts.csv',index=False)
    seeddf=pd.DataFrame(seedrows);seeddf.to_csv(OUT/'extension_paired_seed_contrasts.csv',index=False)
    # Retain both logged loss metrics alongside accuracy; these descriptive
    # secondary contrasts clarify metric dependence, with no new primary test.
    losscontrasts=[];lossseed=[]
    for budget,group in df.groupby('budget'):
        for metric in ['test_loss','best_valid_loss']:
            wide=group.pivot(index='seed',columns=['arm','depth'],values=metric)
            for name,left,right in CONTRASTS:
                values=wide[left]-wide[right]
                losscontrasts.append(dict(budget=int(budget),metric=metric,contrast=name,
                    scope='descriptive secondary loss contrast; same validation-selected states',**boot(values)))
                lossseed.extend(dict(budget=int(budget),metric=metric,contrast=name,
                    seed=int(seed),difference=float(v)) for seed,v in values.items())
    pd.DataFrame(losscontrasts).to_csv(OUT/'extension_loss_contrasts.csv',index=False)
    pd.DataFrame(lossseed).to_csv(OUT/'extension_loss_seed_contrasts.csv',index=False)
    changes=[]
    for label,g in seeddf.groupby('contrast'):
        wide=g.pivot(index='seed',columns='budget',values='difference_pp')
        changes.append(dict(contrast=label,quantity='600_minus180_contrast',**boot(wide[600]-wide[180])))
    pd.DataFrame(changes).to_csv(OUT/'extension_budget_interactions.csv',index=False)
    summary=df.groupby(['arm','depth','budget']).agg(mean_test_accuracy=('test_accuracy','mean'),n=('seed','count'),
        mean_best_epoch=('best_epoch_one_based','mean'),cap_count=('at_budget_cap','sum'),
        mean_best_valid_loss=('best_valid_loss','mean'),mean_late_slope=('late_validation_loss_slope','mean')).reset_index()
    summary.to_csv(OUT/'extension_summary.csv',index=False)
    audit=dict(status='passed',fits=60,seed_blocks=10,budgets=[180,600],conditions_complete=True,
        original_180_results_preserved=True,training_changes='Epoch cap only; original early stopping and optimizer recipes retained',
        source_concordance='See per-run validation-curve and 180-epoch test-accuracy discrepancies; source-consistent restarts, not claimed bitwise continuation',
        max_abs_historical180_accuracy_difference_pp=max(abs(r['test_accuracy_difference_pp']) for r in replay),
        median_abs_historical180_accuracy_difference_pp=float(np.median([abs(r['test_accuracy_difference_pp']) for r in replay])),
        at_600_cap=int(df[df.budget==600].at_budget_cap.sum()),
        resource_checks='Every fit has 66178 trainable parameters and 14336 active synapses',
        configuration_and_state_checks='All 60 configurations differ only in epoch cap and output naming; saved final model tensors exactly equal validation-selected best-state tensors',
        projection='Best-validation curves carried forward after stopping, no survivor averaging',
        protocol_sha256=sha(OUT/'extension_protocol.json'),script_sha256=sha(__file__))
    dump(OUT/'extension_validation.json',audit)
    lines=['# Physical depth under a longer training budget','',
        'All 60 planned restarts completed from the pinned clean implementation. Each run reused its original task, seed, model and optimizer recipe, with the epoch cap raised from 180 to 600 and the original patience 30 retained. Endpoints are selected by validation loss; the test set is used only for reporting.',
        '', 'The 180 and 600 comparisons below come from the same new trajectories. Original outcomes remain in their existing files, and full first-window validation discrepancies from the historical runs are reported separately. A larger maximum budget is not a claim that every run converged.',
        '', '| Contrast | Budget | Mean difference (pp) | 95% paired-seed interval |',
        '|---|---:|---:|---:|']
    for r in contrasts:lines.append(f'| {r["contrast"]} | {r["budget"]} | {r["mean"]:.2f} | {r["ci95_low"]:.2f} to {r["ci95_high"]:.2f} |')
    lines+=['',f'{audit["at_600_cap"]} of 60 runs reached the 600-epoch cap. Their late validation slopes and best epochs are retained in extension_endpoints.csv. The finite-budget qualification remains wherever those curves are still declining.',
        '', 'Intervals are descriptive 95% whole-seed bootstrap intervals with 10 paired seeds. There is no new confirmatory multiplicity claim. All null, small, reversed and positive results are retained.','']
    (OUT/'EXTENSION_REPORT.md').write_text('\n'.join(lines))
    print(summary.to_string(index=False));print(contrastdf.to_string(index=False))


if __name__=='__main__':main()
