#!/usr/bin/env python3
"""Diagnostic and freshly frozen finite-horizon morphology prediction study.

Original experiment files are read-only. Outputs stay in the investigation
directory. Predictions are committed to disk before corresponding outcome
training; old development and confirmation cohorts are explicitly diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from operators import empirical_fullbatch_spectral, fit_gaussian_context_model, gaussian_fullbatch_spectral, gaussian_moments

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
OUTPUT = JOURNAL/'analysis/morphology_investigation_20260905/dynamics'
SOURCE = JOURNAL/'scripts/run_prospective_morphology_selection.py'
spec = importlib.util.spec_from_file_location('original_prospective', SOURCE)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def freeze():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/'protocol.json'
    if path.exists():
        return check_freeze()
    protocol = dict(
        version=1, title='Finite-horizon predictors for a frozen contextual linear learner',
        created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        diagnostic_development_seeds=list(range(6100, 6105)),
        diagnostic_old_confirmation_seeds=list(range(7200, 7220)),
        fresh_confirmation_seeds=list(range(9300, 9320)),
        fresh_angles_pi=[.125, .375], ranks=[1, 2, 4, 8], noise_sds=[0., .75],
        checkpoints=[16, 64, 256], primary_checkpoint=256, pilot_steps=16,
        n_calibration=1024, empirical_calibration_fit=512, empirical_calibration_evaluation=512,
        n_training_cache=2048, n_test=4096, batch_size=32, eta=.35,
        cost='half-MSE + 0.08 * (K/8 + 0.1 * normalized cable length)',
        primary_selector='gaussian_plugin_sgd',
        secondary_selectors=['gaussian_plugin_fullbatch', 'gaussian_oracle_sgd', 'gaussian_oracle_fullbatch',
                             'empirical_split_fullbatch', 'pilot16', 'original_scalar', 'rank_only', 'development_best', 'maximum_budget'],
        assumptions=dict(gaussian='x ~ N(0,I); observed discrete context; conditionally linear target; independent label noise',
            plugin='Context vectors/frequencies, conditional target coefficients and unbiased noise variances estimated using calibration only',
            oracle='True generating context vectors, uniform probabilities, target coefficients and label-noise variance; privileged diagnostic',
            empirical='Exact GD on first 512 calibration rows; risk on remaining 512; surrogate for 2048-cache SGD',
            fresh_sgd='Exact population moment recurrence for fresh iid minibatches; finite-cache use is model misspecification',
            pilot='Sixteen actual training steps on every candidate, evaluated on calibration observations; extra candidate-training cost is measured'),
        population_comparison='Both finite-cache and fresh-example SGD; common held-out test and paired candidate sampling',
        inference='20 independent seed blocks, conditions averaged within seed; 10000 whole-seed bootstrap resamples',
        outcomes='Retain all candidates, ranks, noises, angles, arms and training regimes; no endpoint-dependent exclusions',
        tuning='No new fitted selector hyperparameters; original scalar/rank/fixed choices read unchanged from original development fit',
        status='Frozen before fresh calibration or training outcomes',
        code_sha256={str(p.relative_to(JOURNAL)):sha(p) for p in sorted(HERE.glob('*.py'))},
        original_runner_sha256=sha(SOURCE), original_protocol_sha256=sha(original.CONFIG),
        original_development_fit_sha256=sha(original.OUT/'development_fit.json'),
        python=platform.python_version(), numpy=np.__version__,
    )
    save_json(path, protocol)
    return protocol


def check_freeze():
    protocol = json.loads((OUTPUT/'protocol.json').read_text())
    for path, digest in protocol['code_sha256'].items():
        assert sha(JOURNAL/path) == digest, f'Frozen code changed: {path}'
    assert sha(SOURCE) == protocol['original_runner_sha256']
    assert sha(original.CONFIG) == protocol['original_protocol_sha256']
    assert sha(original.OUT/'development_fit.json') == protocol['original_development_fit_sha256']
    return protocol


def padded_dictionaries(candidates):
    dictionaries = np.zeros((len(candidates), 8, 8))
    for i, candidate in enumerate(candidates):
        dictionaries[i, :, :candidate['budget_k']] = candidate['dictionary']
    return dictionaries


def predictions(task, arm, candidates, cfg):
    """Only calibration data and explicit oracle metadata enter this function."""
    start = time.perf_counter()
    x, a, y = original.dataset(task, 'calibration', cfg['n_calibration'])
    hs = np.stack([np.eye(8) if arm == 'feedback_only' else c['transfer'] for c in candidates])
    dictionaries = padded_dictionaries(candidates)
    rows, timings = [], []
    plugin = fit_gaussian_context_model(x, a, y)
    q, teacher = original.task_basis(task)
    oracle = dict(contexts=q[:, :task['rank']].T, targets=q[:, :task['rank']].T @ teacher,
                  probabilities=np.ones(task['rank'])/task['rank'], noise_variances=np.full(task['rank'], task['noise_sd']**2))
    fitting_seconds = time.perf_counter()-start
    for model_name, model in [('plugin', plugin), ('oracle', oracle)]:
        features = np.einsum('ri,cij,cjk->crk', model['contexts'], hs, dictionaries)
        for update_name, function in [('fullbatch', gaussian_fullbatch_spectral), ('sgd', gaussian_moments)]:
            before = time.perf_counter()
            kwargs = dict(context_features=features, targets=model['targets'], probabilities=model['probabilities'],
                          noise_variances=model['noise_variances'], eta=cfg['learning_rate'], checkpoints=cfg['checkpoints'])
            if update_name == 'sgd':
                kwargs['batch_size'] = cfg['batch_size']
            result = function(**kwargs)
            elapsed = time.perf_counter()-before
            method = f'gaussian_{model_name}_{update_name}'
            timings.append(dict(task_id=task['task_id'], arm=arm, method=method, seconds=elapsed,
                                calibration_fitting_shared_seconds=fitting_seconds if model_name=='plugin' else 0.))
            for step, values in result.items():
                for candidate, loss in zip(candidates, values['loss']):
                    rows.append({**original.metadata(task, arm, candidate), 'method':method, 'checkpoint':step,
                                 'predicted_loss':float(loss)})
    before = time.perf_counter()
    for i, candidate in enumerate(candidates):
        reduced_a = a @ hs[i] @ candidate['dictionary']
        z = (reduced_a[:, :, None]*x[:, None, :]).reshape(len(x), -1)
        estimates = empirical_fullbatch_spectral(z[:512], y[:512], z[512:], y[512:], cfg['learning_rate'], cfg['checkpoints'])
        for step, values in estimates.items():
            rows.append({**original.metadata(task, arm, candidate), 'method':'empirical_split_fullbatch',
                         'checkpoint':step, 'predicted_loss':values['loss']})
    timings.append(dict(task_id=task['task_id'], arm=arm, method='empirical_split_fullbatch', seconds=time.perf_counter()-before))
    before = time.perf_counter()
    fitted = json.loads((original.OUT/'development_fit.json').read_text())[arm]
    for i, candidate in enumerate(candidates):
        score, _, _ = original.moment_score(x, a, y, hs[i], candidate['projector'], cfg['learning_rate'], cfg['batch_size'])
        # Original scalar is a final-horizon prediction, not a predictor of all checkpoints.
        rows.append({**original.metadata(task, arm, candidate), 'method':'original_scalar', 'checkpoint':cfg['training_steps'],
                     'predicted_loss':score['initial_calibration_loss']-fitted['utility_to_loss_scale']*score['utility_actual_step']})
    timings.append(dict(task_id=task['task_id'], arm=arm, method='original_scalar', seconds=time.perf_counter()-before))
    return rows, timings


def train(task, arm, candidates, cfg, regime):
    """Original fixed-cache updates or explicitly different fresh-example updates."""
    x, a, y = original.dataset(task, 'training', cfg['n_training'])
    tx, ta, ty = original.dataset(task, 'test', cfg['n_test'])
    cx, ca, cy = original.dataset(task, 'calibration', cfg['n_calibration'])
    hs = np.stack([np.eye(8) if arm=='feedback_only' else c['transfer'] for c in candidates])
    ps = np.stack([c['projector'] for c in candidates])
    weights = np.zeros((len(candidates), 8, 8))
    rng = np.random.default_rng(np.random.SeedSequence([task['seed'], task['rank'], 53 if regime=='fixed_cache' else 151]))
    q, teacher = original.task_basis(task)
    true_contexts = q[:, :task['rank']].T
    true_target = true_contexts @ teacher
    rows, pilot = [], []
    elapsed_updates = 0.; pilot_update_seconds = None; pilot_evaluation_seconds = None
    for step in range(1, cfg['training_steps']+1):
        before = time.perf_counter()
        if regime == 'fixed_cache':
            idx = rng.integers(len(x), size=cfg['batch_size'])
            xb, ab, yb = x[idx], a[idx], y[idx]
        else:
            xb = rng.normal(size=(cfg['batch_size'], 8))
            context = rng.integers(task['rank'], size=cfg['batch_size'])
            eps = rng.normal(size=cfg['batch_size'])
            ab = q[:, context].T
            yb = np.einsum('bi,ij,bj->b', ab, teacher, xb)+task['noise_sd']*eps
        b = np.einsum('bi,cij->cbj', ab, hs, optimize=True)
        residual = np.einsum('cbi,cij,bj->cb', b, weights, xb, optimize=True)-yb[None]
        raw = np.einsum('cb,cbi,bj->cij', residual, b, xb, optimize=True)/cfg['batch_size']
        weights -= cfg['learning_rate']*np.einsum('cij,cjk->cik', ps, raw, optimize=True)
        elapsed_updates += time.perf_counter()-before
        if step == 16:
            pilot_update_seconds = elapsed_updates
            before = time.perf_counter()
            cb = np.einsum('bi,cij->cbj', ca, hs, optimize=True)
            pred = np.einsum('cbi,cij,bj->cb', cb, weights, cx, optimize=True)
            losses = .5*np.mean((pred-cy[None])**2, axis=1)
            for candidate, loss in zip(candidates, losses):
                pilot.append({**original.metadata(task, arm, candidate), 'regime':regime,
                              'method':'pilot16', 'checkpoint':cfg['training_steps'], 'predicted_loss':float(loss)})
            pilot_evaluation_seconds = time.perf_counter()-before
        if step in cfg['checkpoints']:
            tb = np.einsum('bi,cij->cbj', ta, hs, optimize=True)
            pred = np.einsum('cbi,cij,bj->cb', tb, weights, tx, optimize=True)
            losses = .5*np.mean((pred-ty[None])**2, axis=1)
            response = np.einsum('ri,cij,cjk->crk', true_contexts, hs, weights)
            population_losses = .5*(np.mean(np.sum((response-true_target[None])**2, axis=-1), axis=-1)+task['noise_sd']**2)
            if not np.isfinite(losses).all():
                raise FloatingPointError('Nonfinite outcomes; investigate and retain rather than exclude')
            for candidate, loss, ploss in zip(candidates, losses, population_losses):
                rows.append({**original.metadata(task, arm, candidate), 'regime':regime, 'checkpoint':step,
                             'test_loss':float(loss), 'population_loss':float(ploss)})
    return rows, pilot, dict(task_id=task['task_id'], arm=arm, regime=regime, method='candidate_training',
                            seconds=elapsed_updates, pilot_update_seconds=pilot_update_seconds,
                            pilot_evaluation_seconds=pilot_evaluation_seconds)


def run_seed(cohort, seed):
    protocol = check_freeze(); cfg = original.load()
    permitted = {'development':protocol['diagnostic_development_seeds'],
                 'diagnostic':protocol['diagnostic_old_confirmation_seeds'],
                 'fresh':protocol['fresh_confirmation_seeds']}
    if seed not in permitted[cohort]:
        raise ValueError('Seed is not in the frozen cohort')
    split = 'development' if cohort=='development' else 'confirmatory'
    candidates = original.candidates(cfg)
    destination = OUTPUT/'runs'/cohort
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/f'seed_{seed}_audit.json').exists():
        raise FileExistsError('Completed outcomes are immutable; choose another output directory for reruns')
    start = time.perf_counter()
    predicted, timings = [], []
    for task in original.tasks(cfg, split, seed):
        for arm in cfg['arms']:
            values, timing = predictions(task, arm, candidates, cfg)
            predicted.extend(values); timings.extend(timing)
    prediction_path = destination/f'seed_{seed}_predictions.csv'
    pd.DataFrame(predicted).to_csv(prediction_path, index=False)
    # All non-pilot endpoint choices are sealed before this seed's outcome training.
    seal = dict(cohort=cohort, seed=seed, predictions_sha256=sha(prediction_path), protocol_sha256=sha(OUTPUT/'protocol.json'),
                sealed_utc=pd.Timestamp.now(tz='UTC').isoformat(), pilot='Budgeted 16-step selector; evaluated before endpoint outcomes')
    save_json(destination/f'seed_{seed}_selection_seal.json', seal)
    outcomes, pilot = [], []
    for task in original.tasks(cfg, split, seed):
        for arm in cfg['arms']:
            for regime in ['fixed_cache', 'fresh_iid']:
                result, pilot_rows, timing = train(task, arm, candidates, cfg, regime)
                outcomes.extend(result); pilot.extend(pilot_rows); timings.append(timing)
    assert sha(prediction_path) == seal['predictions_sha256']
    pd.DataFrame(outcomes).to_csv(destination/f'seed_{seed}_outcomes.csv', index=False)
    pd.DataFrame(pilot).to_csv(destination/f'seed_{seed}_pilot.csv', index=False)
    pd.DataFrame(timings).to_csv(destination/f'seed_{seed}_timings.csv', index=False)
    replay_max_error = None
    if cohort in ['development', 'diagnostic']:
        retained = pd.read_csv(original.OUT/'runs'/split/f'seed_{seed}.csv')
        rerun = pd.DataFrame(outcomes).query("regime == 'fixed_cache'")
        joined = rerun.merge(retained, on=['task_id', 'arm', 'candidate_id', 'checkpoint'], validate='one_to_one', suffixes=('_new', '_old'))
        replay_max_error = float(abs(joined.test_loss_new-joined.test_loss_old).max())
        if replay_max_error > 1e-11:
            raise AssertionError(f'Original replay mismatch: {replay_max_error}')
    save_json(destination/f'seed_{seed}_audit.json', dict(cohort=cohort, seed=seed, n_predictions=len(predicted),
        n_outcomes=len(outcomes), n_pilot=len(pilot), elapsed_seconds=time.perf_counter()-start,
        original_replay_max_loss_difference=replay_max_error, prediction_seal=seal,
        completed_utc=pd.Timestamp.now(tz='UTC').isoformat(), original_runner_sha256=sha(SOURCE),
        numpy=np.__version__, host=platform.node(), thread_limit=os.environ.get('OPENBLAS_NUM_THREADS')))
    print(json.dumps(dict(cohort=cohort, seed=seed, seconds=time.perf_counter()-start), sort_keys=True), flush=True)


def bootstrap(values, draws=10000):
    values = np.asarray(values, float)
    rng = np.random.default_rng(991000)
    sample = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(sample, .025)), float(np.quantile(sample, .975))


def analyze(cohort):
    check_freeze()
    directory = OUTPUT/'runs'/cohort
    audits = sorted(directory.glob('seed_*_audit.json'))
    if not audits:
        raise FileNotFoundError('No completed seed blocks')
    seeds = [json.loads(p.read_text())['seed'] for p in audits]
    def collect(suffix):
        return pd.concat([pd.read_csv(directory/f'seed_{seed}_{suffix}.csv') for seed in seeds], ignore_index=True)
    predicted, outcomes, pilot, times = [collect(s) for s in ['predictions', 'outcomes', 'pilot', 'timings']]
    key = ['task_id', 'arm', 'candidate_id', 'checkpoint']
    scored = outcomes.merge(predicted[key+['method', 'predicted_loss']], on=key, validate='many_to_many')
    pscored = outcomes.merge(pilot[key+['regime', 'method', 'predicted_loss']], on=key+['regime'], validate='one_to_one')
    scored = pd.concat([scored, pscored], ignore_index=True)
    scored['predicted_objective'] = scored.predicted_loss+.08*scored.resource_cost
    scored['observed_objective'] = scored.test_loss+.08*scored.resource_cost
    scored['population_objective'] = scored.population_loss+.08*scored.resource_cost
    groups = ['task_id', 'arm', 'regime', 'checkpoint']
    oracle = scored.groupby(groups).agg(oracle_objective=('observed_objective', 'min'),
                                       oracle_population_objective=('population_objective', 'min')).reset_index()
    selected = scored.sort_values(['predicted_objective', 'candidate_id'], kind='stable').groupby(groups+['method'], as_index=False).first()
    # Original development choices are unchanged and never refit on this cohort.
    baseline_rows = []
    fit = json.loads((original.OUT/'development_fit.json').read_text())
    for _, g in outcomes[outcomes.checkpoint.eq(256)].groupby(['task_id', 'arm', 'regime']):
        arm = g.iloc[0].arm
        choices = {'rank_only':fit[arm]['rank_only'][str(int(g.iloc[0]['rank']))],
                   'development_best':fit[arm]['development_best'], 'maximum_budget':fit[arm]['max_budget']}
        for method, candidate_id in choices.items():
            row = g[g.candidate_id.eq(candidate_id)].iloc[0].to_dict()
            row.update(method=method, observed_objective=row['test_loss']+.08*row['resource_cost'],
                       population_objective=row['population_loss']+.08*row['resource_cost'])
            baseline_rows.append(row)
    selected = pd.concat([selected, pd.DataFrame(baseline_rows)], ignore_index=True).merge(oracle, on=groups, validate='many_to_one')
    selected['regret'] = selected.observed_objective-selected.oracle_objective
    selected['population_regret'] = selected.population_objective-selected.oracle_population_objective
    destination = OUTPUT/'summaries'/cohort; destination.mkdir(parents=True, exist_ok=True)
    selected.to_csv(destination/'selected_policies.csv', index=False)
    policy_rows = []
    for subgroup, g in selected.groupby(['arm', 'regime', 'checkpoint', 'method']):
        for rank in ['all', 1, 2, 4, 8]:
            z = g if rank=='all' else g[g['rank'].eq(rank)]
            mean, low, high = bootstrap(z.groupby('seed').regret.mean())
            policy_rows.append(dict(zip(['arm', 'regime', 'checkpoint', 'method'], subgroup)) | dict(
                rank=rank, n_seed_blocks=z.seed.nunique(), n_tasks=len(z), mean_regret=mean, ci95_low=low,
                ci95_high=high, mean_test_loss=z.test_loss.mean(), mean_selected_k=z.budget_k.mean(),
                mean_population_regret=z.population_regret.mean()))
    pd.DataFrame(policy_rows).to_csv(destination/'policy_summary.csv', index=False)
    associations = []
    for subgroup, g in scored.groupby(['task_id', 'arm', 'regime', 'checkpoint', 'method']):
        row = dict(zip(['task_id', 'arm', 'regime', 'checkpoint', 'method'], subgroup))
        row.update(seed=int(g.iloc[0].seed), rank=int(g.iloc[0]['rank']),
            loss_spearman=float(spearmanr(g.predicted_loss, g.test_loss).statistic),
            objective_spearman=float(spearmanr(g.predicted_objective, g.observed_objective).statistic),
            mse=float(np.mean((g.predicted_loss-g.test_loss)**2)),
            population_mse=float(np.mean((g.predicted_loss-g.population_loss)**2)),
            mean_error=float(np.mean(g.predicted_loss-g.test_loss)))
        associations.append(row)
    association = pd.DataFrame(associations)
    association.to_csv(destination/'within_task_predictions.csv', index=False)
    summary = association.groupby(['arm', 'regime', 'checkpoint', 'method']).agg(
        mean_within_task_loss_rho=('loss_spearman', 'mean'), mean_within_task_objective_rho=('objective_spearman', 'mean'),
        mse=('mse', 'mean'), population_mse=('population_mse', 'mean'), mean_error=('mean_error', 'mean')).reset_index()
    summary['rmse'] = np.sqrt(summary.mse)
    summary['population_rmse'] = np.sqrt(summary.population_mse)
    summary.to_csv(destination/'prediction_summary.csv', index=False)
    contrast_rows = []
    last = selected[selected.checkpoint.eq(256)]
    for (arm, regime), g in last.groupby(['arm', 'regime']):
        pivot = g.pivot(index=['task_id', 'seed'], columns='method', values='regret')
        for method in pivot:
            if method == 'gaussian_plugin_sgd':
                continue
            differences = (pivot[method]-pivot.gaussian_plugin_sgd).groupby('seed').mean()
            mean, low, high = bootstrap(differences)
            contrast_rows.append(dict(arm=arm, regime=regime, comparator=method, n_seed_blocks=len(differences),
                plugin_sgd_regret_reduction=mean, ci95_low=low, ci95_high=high, positive_seeds=int((differences>0).sum())))
    pd.DataFrame(contrast_rows).to_csv(destination/'paired_primary_contrasts.csv', index=False)
    times.to_csv(destination/'timings.csv', index=False)
    times.groupby('method').seconds.agg(['count', 'mean', 'median', 'sum']).to_csv(destination/'timing_summary.csv')
    save_json(destination/'report.json', dict(cohort=cohort, seeds=seeds, n_seed_blocks=len(seeds),
        n_prediction_rows=len(predicted), n_outcome_rows=len(outcomes), n_policy_rows=len(selected),
        protocol_sha256=sha(OUTPUT/'protocol.json'), all_outcomes_retained=True,
        original_replay_max_error=max([json.loads(p.read_text())['original_replay_max_loss_difference'] or 0. for p in audits]),
        inference='Seed-block averages; original confirmation cohort is diagnostic for new predictors',
        no_test_data_in_predictions='Predictions use calibration only except labeled population-oracle metadata; pilot uses training and calibration only'))
    print(summary[summary.checkpoint.eq(256)].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'run', 'analyze'])
    parser.add_argument('--cohort', choices=['development', 'diagnostic', 'fresh'], default='development')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.command == 'freeze':
        print(json.dumps(freeze(), indent=2))
    elif args.command == 'run':
        run_seed(args.cohort, args.seed)
    else:
        analyze(args.cohort)


if __name__ == '__main__':
    main()
