#!/usr/bin/env python3
"""Read retained conductance outcomes and verify seals, cohorts and statistics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run as study


def main():
    cfg=study.check_development();fresh=json.loads((study.OUT/'fresh_protocol.json').read_text())
    assert fresh['development_protocol_sha256']==study.sha(study.OUT/'development_protocol.json')
    assert fresh['sgd_bracket_protocol_sha256']==study.sha(study.OUT/'development_sgd_bracket_protocol.json')
    assert fresh['selection_table_sha256']==study.sha(study.OUT/'development_learning_rate_selection.csv')
    for record in fresh['development_source_hashes']:assert study.sha(study.OUT/record['path'])==record['sha256']
    sources=[];frames=[];geometry=[];spectral_error=0.
    for seed in cfg['fresh_seeds']:
        dest=study.OUT/'runs/fresh';audit=json.loads((dest/f'seed_{seed}_audit.json').read_text())
        assert audit['fresh_protocol_sha256']==study.sha(study.OUT/'fresh_protocol.json')
        assert audit['n_fits']==72 and audit['n_curve_rows']==360
        path=dest/f'seed_{seed}_curves.csv';data=pd.read_csv(path)
        diag=pd.read_csv(dest/f'seed_{seed}_credit_diagnostics.csv')
        assert len(data)==len(diag)==360
        assert np.isfinite(data.select_dtypes('number')).all().all()
        assert set(data.credit_rule)==set(study.RULES)
        assert data.step.nunique()==5 and data.task_group.nunique()==3 and data.student_group.nunique()==3
        assert (data.compatible==(data.task_group==data.student_group)).all()
        for optimizer,rows in data.groupby('optimizer'):
            assert np.all(rows.learning_rate==fresh['selected_learning_rates'][optimizer])
        with np.load(dest/f'seed_{seed}_final_states.npz') as saved:
            for group in range(3):
                weights=saved[f'task_{group}_final_log_conductances']
                profiles=saved[f'task_{group}_initial_profiles']
                assert weights.shape==(24,16) and profiles.shape==(24,6)
                assert np.isfinite(weights).all() and np.all((weights>=-5)&(weights<=5))
                assert np.all(profiles>0)
        spectral_error=max(spectral_error,audit['input_gradient_spectrum_max_difference'])
        assert audit['paired_task_output_max_difference']==0.
        frames.append(data);geometry.append(diag)
        sources.append(dict(path=str(path.relative_to(study.OUT)),sha256=study.sha(path)))
    frame=pd.concat(frames,ignore_index=True);diag=pd.concat(geometry,ignore_index=True)
    keys=['seed','task_group','student_group','optimizer','credit_rule','step']
    assert len(frame)==7200 and not frame.duplicated(keys).any()
    endpoint=frame[frame.step.eq(1000)];assert len(endpoint)==1440
    # Recompute every saved contrast independently from the raw rows.
    saved=pd.read_csv(study.OUT/'summaries/fresh/paired_contrasts.csv')
    differences=[]
    for _,row in saved.iterrows():
        z=endpoint[endpoint.optimizer.eq(row.optimizer)]
        if row.contrast.startswith('incompatible_minus_compatible__'):
            rule=row.contrast.split('__')[1]
            means=z[z.credit_rule.eq(rule)].groupby(['seed','compatible']).test_nmse.mean().unstack()
            values=means[False]-means[True]
        else:
            comparison,compatibility=row.contrast.split('__compatible_')
            control,reference=comparison.split('_minus_')
            means=z[z.compatible.eq(compatibility=='True')].groupby(['seed','credit_rule']).test_nmse.mean().unstack()
            values=means[control]-means[reference]
        mean,lo,hi=study.boot(values)
        differences.extend([abs(mean-row.mean_difference),abs(lo-row.ci95_low),abs(hi-row.ci95_high)])
    assert max(differences)<1e-12
    # Capture and cosine are common-exact-state diagnostics, and capture excludes soma.
    exact_diag=diag[diag.credit_rule.eq('exact_path')]
    assert np.max(abs(exact_diag.path_capture-1))<1e-12
    assert np.nanmax(abs(exact_diag.gradient_cosine-1))<1e-12
    bound=pd.read_csv(study.OUT/'interaction_bound/population_bounds.csv')
    bound_seed=bound[~bound.compatible].groupby('seed').additive_subtree_nmse_lower_bound.mean()
    mean,lo,hi=study.boot(bound_seed)
    study.write(study.OUT/'interaction_bound/seed_summary.json',dict(mean_incompatible_population_nmse_lower_bound=mean,
        ci95_low=lo,ci95_high=hi,n_seed_blocks=20,unit='Population NMSE lower bound; converged quadrature, not an observed test error'))
    study.write(study.OUT/'validation.json',dict(status='pass',n_seed_blocks=20,n_fits=1440,n_checkpoint_rows=7200,
        all_four_rules_retained=True,all_two_optimizers_retained=True,all_three_task_and_student_groupings_retained=True,
        all_metrics_finite=True,all_source_and_selection_hashes_verified=True,
        maximum_contrast_recomputation_error=max(differences),max_input_gradient_spectrum_difference=spectral_error,
        max_conductance_bound_events=int(frame.bound_events.max()),min_total_conductance=float(frame.min_total_conductance.min()),
        max_voltage=float(frame.max_voltage.max()),undefined_gradient_cosines=int(diag.gradient_cosine.isna().sum()),
        inference='Pointwise descriptive seed-bootstrap intervals; no separate confirmed discovery claimed for every subgroup contrast',
        pipeline_scope='Planted composition; label-free initial-profile calibration; per-trial projection coefficients use exact current student path fields',
        source_hashes=sources))
    print(json.dumps({k:v for k,v in json.loads((study.OUT/'validation.json').read_text()).items() if k!='source_hashes'},indent=2))


if __name__=='__main__':main()
