#!/usr/bin/env python3
"""Joint six-rule summaries with separate frozen addendum provenance."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
import analyze
from analyze import interval,collect
from run import OUT,ARCHITECTURES,sha,dump

def main():
    analyze.summarize()
    folder=OUT/'summaries'
    allout=[pd.read_csv(folder/'all_outcomes.csv',float_precision='round_trip')];allcurves=[pd.read_csv(folder/'all_curves.csv')];hashes=[pd.read_csv(folder/'source_files.csv')]
    for phase in ('development','fresh'):
        stage='projected_k1_'+phase
        f,c,h=collect(stage)
        f['execution_stage']=stage;f['stage']=phase;c['execution_stage']=stage;c['stage']=phase
        allout.append(f);allcurves.append(c);hashes.append(h)
    outcomes=pd.concat(allout,ignore_index=True);curves=pd.concat(allcurves,ignore_index=True)
    assert outcomes.groupby(['stage','architecture','seed']).initialized_model_sha256.nunique().eq(1).all(),'Addendum lost modelpairing'
    assert np.isfinite(outcomes[['test_accuracy','test_cross_entropy','selection_validation_loss','best_epoch']].to_numpy(float)).all(),'Nonfiniteoutcome'
    assert outcomes.test_accuracy.between(0,1).all() and outcomes.test_cross_entropy.ge(0).all(),'Invalidendpoint'
    assert outcomes.best_epoch.between(0,180).all(),'Invalidselectedepoch'
    outcomes.to_csv(folder/'all_outcomes_six_rules.csv',index=False);curves.to_csv(folder/'all_curves_six_rules.csv',index=False)
    pd.concat(hashes).to_csv(folder/'source_files_six_rules.csv',index=False)
    selection=pd.DataFrame(json.loads((OUT/'selection.json').read_text())['selected']+json.loads((OUT/'projected_k1/selection.json').read_text())['selected'])
    fresh=outcomes[outcomes.stage.eq('fresh')]
    selected=fresh.merge(selection[['architecture','arm','multiplier']],on=['architecture','arm','multiplier'],validate='many_to_one').assign(rate_policy='selected')
    common=fresh[fresh.multiplier.eq(1)].assign(rate_policy='common_original')
    views=pd.concat([selected,common]);assert len(views)==240
    views.to_csv(folder/'fresh_analysis_rows_six_rules.csv',index=False)
    rows=[]
    for key,g in views.groupby(['rate_policy','architecture','arm']):
        assert len(g)==10
        for metric in ('test_accuracy','test_cross_entropy','selection_validation_loss','best_epoch'):
            rows.append(dict(zip(('rate_policy','architecture','arm'),key),metric=metric,**interval(g[metric])))
    summary=pd.DataFrame(rows);summary.to_csv(folder/'condition_summary_six_rules.csv',index=False)
    pairs=[('neuron_shared','strict_scalar'),('projected_k1','neuron_shared'),('subtree_k3','projected_k1'),('exact_path','subtree_k3'),('neuron_shared','decoder_only'),('exact_path','neuron_shared'),('subtree_k3','neuron_shared')]
    contrasts=[];paired=[]
    for policy in ('selected','common_original'):
        for architecture in ARCHITECTURES:
            group=views[views.rate_policy.eq(policy)&views.architecture.eq(architecture)]
            for metric in ('test_accuracy','test_cross_entropy'):
                pivot=group.pivot(index='seed',columns='arm',values=metric);assert pivot.shape==(10,6) and not pivot.isna().any().any()
                for lhs,rhs in pairs:
                    values=pivot[lhs]-pivot[rhs];name=f'{lhs}_minus_{rhs}'
                    contrasts.append(dict(rate_policy=policy,architecture=architecture,metric=metric,contrast=name,**interval(values)))
                    paired.extend(dict(rate_policy=policy,architecture=architecture,metric=metric,contrast=name,seed=int(seed),difference=float(v)) for seed,v in values.items())
    pd.DataFrame(contrasts).to_csv(folder/'paired_contrasts_six_rules.csv',index=False);pd.DataFrame(paired).to_csv(folder/'paired_seed_contrasts_six_rules.csv',index=False)
    rates=[]
    for key,g in outcomes[outcomes.stage.eq('development')].groupby(['architecture','arm','multiplier']):
        assert len(g)==3
        for metric in ('test_accuracy','test_cross_entropy','selection_validation_loss','best_epoch'):
            rates.append(dict(zip(('architecture','arm','multiplier'),key),metric=metric,**interval(g[metric])))
    pd.DataFrame(rates).to_csv(folder/'development_rate_summary_six_rules.csv',index=False)
    versions=[pd.read_csv(folder/'software_versions.csv')]
    added=[]
    for phase in ('development','fresh'):
        stage='projected_k1_'+phase
        for rec in json.loads((OUT/f'{stage}_conditions.json').read_text()):
            a=json.loads((Path(rec['results_dir'])/'run_audit.json').read_text());added.append(dict(stage=stage,index=rec['index'],python=a['python'],cuda=a['cuda'],device_name=a['device_name'],**a['versions']))
    versions.append(pd.DataFrame(added));pd.concat(versions).to_csv(folder/'software_versions_six_rules.csv',index=False)
    gate=dict(complete=True,base_development_fits=90,projected_k1_development_fits=18,total_development_fits=108,
        base_fresh_fits=int((fresh.arm!='projected_k1').sum()),projected_k1_fresh_fits=int((fresh.arm=='projected_k1').sum()),total_fresh_fits=len(fresh),
        total_scientific_fits=len(outcomes),excluded_canaries=12,all_epochs180=bool(outcomes.epochs.eq(180).all()),all_initial_model_pairs_identical=True,
        decoder_floor_core_unchanged=bool(outcomes[outcomes.arm.eq('decoder_only')].decoder_only_core_unchanged.all()),
        primary_spatial_contrast='subtree_k3 minus projected_k1; identical oracle coefficient acquisition apart from fixedprojector',
        original_protocol_sha256=sha(OUT/'protocol.json'),addendum_protocol_sha256=sha(OUT/'projected_k1/protocol.json'),
        selected_rate_policies='Each of sixrules separately tuned on threecommon rate multipliers and3developmentseeds; testoutcomes never select',
        fixed_budget='All180epochsretained; validation-based beststate; no convergence or globaloptimalityclaim')
    dump(folder/'completeness_audit_six_rules.json',gate)
    print(json.dumps(gate,indent=2));print(summary.query("metric=='test_accuracy' and rate_policy=='selected'").to_string(index=False))
    print(pd.DataFrame(contrasts).query("metric=='test_accuracy' and contrast=='subtree_k3_minus_projected_k1'").to_string(index=False))
if __name__=='__main__':main()
