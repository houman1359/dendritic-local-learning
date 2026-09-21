#!/usr/bin/env python3
"""Explicit post-primary-freeze diagnostics, without changing sealed predictors.

The observed-context baseline and rank-restricted predictor were requested after
the primary freeze/development run. Their definitions are separately sealed
before fresh outcomes, and are labeled secondary added-after-primary-freeze.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd

import investigate as study


SELF = Path(__file__).resolve()


def freeze():
    path = study.OUTPUT/'supplemental_baseline_protocol.json'
    if path.exists():
        return
    assert not list((study.OUTPUT/'runs/fresh').glob('*')), 'Must seal secondary baseline before fresh runs'
    study.save_json(path, dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        script_sha256=study.sha(SELF), primary_protocol_sha256=study.sha(study.OUTPUT/'protocol.json'),
        status='Secondary baseline added after primary freeze and development inspection, before fresh outcomes',
        observed_count_cheapest='Count distinct calibrated context vectors; choose smallest allowed K >= count, then cheapest resource cost',
        observed_count_conditioning='Restrict to that same observed-count K; choose smallest sealed Gaussian plug-in SGD predicted final objective',
        claims='Observed discrete-context count is not a general inferred credit-rank estimator; this task exposes its context structure directly'))


def analyze(cohort):
    baseline_protocol = json.loads((study.OUTPUT/'supplemental_baseline_protocol.json').read_text())
    assert baseline_protocol['script_sha256'] == study.sha(SELF)
    source = study.OUTPUT/'runs'/cohort
    destination = study.OUTPUT/'summaries'/cohort
    cfg = study.original.load(); candidates = study.original.candidates(cfg)
    audits = [json.loads(p.read_text()) for p in sorted(source.glob('*_audit.json'))]
    seeds = [a['seed'] for a in audits]
    outcomes = pd.concat([pd.read_csv(source/f'seed_{s}_outcomes.csv') for s in seeds], ignore_index=True)
    predicted = pd.concat([pd.read_csv(source/f'seed_{s}_predictions.csv') for s in seeds], ignore_index=True)
    endpoint = outcomes[outcomes.checkpoint.eq(256)].copy()
    predicted = predicted[predicted.checkpoint.eq(256)&predicted.method.eq('gaussian_plugin_sgd')].copy()
    endpoint['observed_objective'] = endpoint.test_loss+.08*endpoint.resource_cost
    endpoint['population_objective'] = endpoint.population_loss+.08*endpoint.resource_cost
    predicted['predicted_objective'] = predicted.predicted_loss+.08*predicted.resource_cost
    count_rows, choices, time_rows = [], [], []
    for seed in seeds:
        split = 'development' if cohort=='development' else 'confirmatory'
        for task in study.original.tasks(cfg, split, seed):
            before = time.perf_counter()
            _, a, _ = study.original.dataset(task, 'calibration', cfg['n_calibration'])
            preparation_seconds = time.perf_counter()-before
            before = time.perf_counter()
            count = len(np.unique(np.round(a, 12), axis=0))
            budget = min(k for k in cfg['route_budgets'] if k >= count)
            cheapest = min((c for c in candidates if c['budget_k']==budget), key=lambda c:(c['resource_cost'], c['candidate_id']))
            seconds = time.perf_counter()-before
            count_rows.append(dict(task_id=task['task_id'], seed=seed, observed_context_count=count,
                                   selected_k=budget, generating_rank_for_audit_only=task['rank']))
            time_rows.append(dict(task_id=task['task_id'], method='observed_count_cheapest', seconds=seconds,
                                  calibration_preparation_seconds=preparation_seconds))
            for arm in cfg['arms']:
                p = predicted[predicted.task_id.eq(task['task_id'])&predicted.arm.eq(arm)&predicted.budget_k.eq(budget)]
                conditional_choice = p.sort_values(['predicted_objective','candidate_id'], kind='stable').iloc[0].candidate_id
                for regime in ['fixed_cache', 'fresh_iid']:
                    z = endpoint[endpoint.task_id.eq(task['task_id'])&endpoint.arm.eq(arm)&endpoint.regime.eq(regime)]
                    for method, candidate_id in [('observed_count_cheapest', cheapest['candidate_id']),
                                                 ('observed_count_conditioning', conditional_choice)]:
                        row = z[z.candidate_id.eq(candidate_id)].iloc[0].to_dict()
                        row.update(method=method, oracle_objective=z.observed_objective.min(),
                                   oracle_population_objective=z.population_objective.min())
                        row['regret'] = row['observed_objective']-row['oracle_objective']
                        row['population_regret'] = row['population_objective']-row['oracle_population_objective']
                        choices.append(row)
    existing = pd.read_csv(destination/'selected_policies.csv')
    combined = pd.concat([existing[existing.checkpoint.eq(256)], pd.DataFrame(choices)], ignore_index=True)
    combined.to_csv(destination/'endpoint_policies_with_secondary_baselines.csv', index=False)
    pd.DataFrame(count_rows).to_csv(destination/'observed_context_counts.csv', index=False)
    pd.DataFrame(time_rows).to_csv(destination/'observed_count_timings.csv', index=False)
    summary, contrasts = [], []
    for (arm, regime, method), g in combined.groupby(['arm','regime','method']):
        for rank in ['all', 1, 2, 4, 8]:
            z = g if rank=='all' else g[g['rank'].eq(rank)]
            mean, lo, hi = study.bootstrap(z.groupby('seed').regret.mean())
            summary.append(dict(arm=arm, regime=regime, method=method, rank=rank, mean_regret=mean,
                ci95_low=lo, ci95_high=hi, n_seed_blocks=z.seed.nunique(), n_tasks=len(z),
                mean_selected_k=z.budget_k.mean(), mean_population_regret=z.population_regret.mean()))
    pd.DataFrame(summary).to_csv(destination/'endpoint_summary_with_secondary_baselines.csv', index=False)
    for (arm, regime), g in combined.groupby(['arm','regime']):
        p = g.pivot(index=['task_id','seed'], columns='method', values='regret')
        for reference in ['gaussian_plugin_sgd','observed_count_cheapest']:
            for method in p:
                if method == reference:
                    continue
                differences = (p[method]-p[reference]).groupby('seed').mean()
                mean, lo, hi = study.bootstrap(differences)
                contrasts.append(dict(arm=arm, regime=regime, reference=reference, comparator=method,
                    reference_regret_reduction=mean, ci95_low=lo, ci95_high=hi,
                    positive_seeds=int((differences>0).sum()), n_seed_blocks=len(differences)))
    pd.DataFrame(contrasts).to_csv(destination/'paired_contrasts_with_secondary_baselines.csv', index=False)
    combined.groupby(['arm','regime','method','rank','budget_k','tree_id']).size().reset_index(name='n_tasks').to_csv(
        destination/'selection_budget_tree_counts.csv', index=False)
    endpoint.groupby(['arm','regime','task_id']).apply(
        lambda z:z.sort_values(['observed_objective','candidate_id'],kind='stable').iloc[0], include_groups=False).reset_index().to_csv(
        destination/'endpoint_oracle_choices.csv', index=False)
    associations = pd.read_csv(destination/'within_task_predictions.csv')
    correlations = []
    for (arm, regime, checkpoint, method), g in associations.groupby(['arm','regime','checkpoint','method']):
        for rank in ['all',1,2,4,8]:
            z = g if rank=='all' else g[g['rank'].eq(rank)]
            correlations.append(dict(arm=arm, regime=regime, checkpoint=checkpoint, method=method, rank=rank,
                n_tasks=len(z), n_defined_loss_correlations=int(z.loss_spearman.notna().sum()),
                mean_within_task_loss_rho=z.loss_spearman.mean(), mean_within_task_objective_rho=z.objective_spearman.mean(),
                rmse=float(np.sqrt(z.mse.mean())), population_rmse=float(np.sqrt(z.population_mse.mean()))))
    pd.DataFrame(correlations).to_csv(destination/'prediction_summary_by_rank_with_defined_counts.csv', index=False)
    timings = pd.read_csv(destination/'timings.csv')
    cost_rows = []
    for method, g in timings.groupby('method'):
        seconds = g.seconds.copy()
        if method in ['gaussian_plugin_sgd','gaussian_plugin_fullbatch']:
            seconds += g.calibration_fitting_shared_seconds
        cost_rows.append(dict(method=method, median_seconds_per_20_candidates=seconds.median(),
                             mean_seconds_per_20_candidates=seconds.mean(), n_measurements=len(g)))
        if method=='candidate_training':
            pilot_seconds = g.pilot_update_seconds+g.pilot_evaluation_seconds
            cost_rows.append(dict(method='pilot16', median_seconds_per_20_candidates=pilot_seconds.median(),
                                 mean_seconds_per_20_candidates=pilot_seconds.mean(), n_measurements=len(g)))
    observed = pd.DataFrame(time_rows)
    cost_rows.append(dict(method='observed_count_cheapest', median_seconds_per_20_candidates=observed.seconds.median(),
                         mean_seconds_per_20_candidates=observed.seconds.mean(), n_measurements=len(observed)))
    pd.DataFrame(cost_rows).to_csv(destination/'comparative_computational_cost.csv', index=False)
    study.save_json(destination/'supplemental_report.json', dict(cohort=cohort, n_seed_blocks=len(seeds),
        supplemental_protocol_sha256=study.sha(study.OUTPUT/'supplemental_baseline_protocol.json'),
        added_after_primary_freeze=True, added_before_fresh_outcomes=True,
        context_counts_equal_generating_rank=all(r['observed_context_count']==r['generating_rank_for_audit_only'] for r in count_rows),
        undefined_loss_correlations=int(associations.loss_spearman.isna().sum()),
        undefined_correlation_policy='Retained as NA; defined counts reported; no task or outcome exclusions'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze','analyze'])
    parser.add_argument('--cohort', choices=['development','diagnostic','fresh'], default='fresh')
    args = parser.parse_args()
    if args.command=='freeze':
        freeze()
    else:
        analyze(args.cohort)
