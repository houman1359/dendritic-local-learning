"""Paired label-noise sensitivity using the unchanged algebraic forward model.

Rates are inherited from the original development split, not retuned using
these outcomes. All three noise conditions share inputs, noise draws, initial
weights and minibatches. Primary reporting uses the clean complete domain.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

J = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J / 'scripts/credit_rule_bridge'))
import models


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_new(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    (root / 'results').mkdir()
    (root / 'states').mkdir()
    old = J / 'source_data/credit_rule_bridge'
    selection = json.loads((old / 'selection_freeze.json').read_text())
    inherited = json.loads((old / 'protocol_freeze.json').read_text())
    for rel, h in inherited['source_sha256'].items():
        assert sha(J / rel) == h, rel
    records = []
    for noise in ['fixed_absolute', 'noise_free', 'relative_matched']:
        for rule in models.RULES:
            selected = selection['selected_rates']['algebraic']['adam'][rule]
            common = selection['common_rates']['algebraic']['adam']
            for rate in sorted({selected, common}):
                records.append(dict(noise=noise, rule=rule, rate=rate,
                                    selected_rate=rate == selected, common_rate=rate == common))
    dependencies = [Path(__file__), old / 'selection_freeze.json', old / 'protocol_freeze.json']
    dependencies += [J / p for p in inherited['source_sha256']]
    dependencies += list((J / 'scripts/morphology_structure').glob('constructive*.py'))
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Internally specified post-review sensitivity; new paired seeds, inherited rates; not externally preregistered',
        seeds=list(range(2026092100, 2026092120)), tasks=['matching', 'quartet'],
        records=records, steps=16384, checkpoints=[0, 1, 16, 64, 256, 512, 1024, 4096, 8192, 16384],
        training_size=2048, validation_size=1024, test_size=4096, calibration_size=256,
        batch_size=64, optimizer='Adam', clip=10., parameter_bound=2.,
        noise='fixed_absolute SD=.15; noise_free SD=0; relative_matched SD=.15*sqrt(clean target variance)',
        primary='For each noise condition, (calibrated-exact clean full-domain NMSE on quartet) minus (calibrated-exact on matching), at 16384 updates and inherited selected rates',
        secondary='1024,4096,8192 checkpoints, common rates, noisy test outcomes, all other rules; descriptive paired intervals',
        pairing='Inputs, standardized noise, calibration, initial coefficients, minibatch indices shared across both tasks and all conditions within a seed',
        selection='No fresh test, validation or diagnostic outcome determines hyperparameters; endpoints at fixed budgets',
        source_sha256={str(p):sha(p) for p in set(dependencies)})
    write_new(root / 'protocol.json', protocol)
    return protocol


def data(seed, coeff, split, size):
    offset = dict(training=11, validation=17, test=23, calibration=29)[split]
    rng = np.random.default_rng(np.random.SeedSequence([seed, offset]))
    x = 2. * rng.integers(2, size=(size, 8)) - 1.
    return x, models.structure.fourier_design(x) @ coeff, rng.normal(size=size)


def run_seed(args):
    root, seed = args
    protocol = json.loads((root / 'protocol.json').read_text())
    assert seed in protocol['seeds']
    for rel, h in protocol['source_sha256'].items():
        assert sha(rel) == h, rel
    started = time.monotonic()
    records = protocol['records']; n = len(records)
    rates = np.array([r['rate'] for r in records])[:, None, None]
    rows, inventory = [], {}
    for task in protocol['tasks']:
        coeff, tree = models.algebra_task(seed, task)
        variance = float(coeff @ coeff)
        sigma = np.array([0. if r['noise'] == 'noise_free' else
                          .15 * np.sqrt(variance) if r['noise'] == 'relative_matched' else .15
                          for r in records])
        left = np.tile([tree.children[k][0] for k in range(8, 15)], (n, 1))
        right = np.tile([tree.children[k][1] for k in range(8, 15)], (n, 1))
        rng = np.random.default_rng(np.random.SeedSequence([seed, 41]))
        initial = rng.normal(0, .5, (7, 4)); initial[:, 0] = 0.
        theta = np.tile(initial, (n, 1, 1))
        x, clean, noise = data(seed, coeff, 'training', 2048)
        y = clean[None] + sigma[:, None] * noise[None]
        vx, vclean, vnoise = data(seed, coeff, 'validation', 1024)
        tx, tclean, tnoise = data(seed, coeff, 'test', 4096)
        cx, _, _ = data(seed, coeff, 'calibration', 256)
        dx = models.structure.domain(); dy = models.structure.fourier_design(dx) @ coeff
        profiles = models.algebra_state(theta, cx, left, right)['path'][:, :, :6].mean(1)
        moment = np.zeros_like(theta); second = np.zeros_like(theta)
        clips = np.zeros(n, int); bounds = np.zeros(n, int)
        stream = np.random.default_rng(np.random.SeedSequence([seed, 37]))
        states = []
        for step in range(protocol['steps'] + 1):
            if step in protocol['checkpoints']:
                states.append(theta.copy())
                pop = models.algebra_state(theta, dx, left, right)['output']
                val = models.algebra_state(theta, vx, left, right)['output']
                test = models.algebra_state(theta, tx, left, right)['output']
                metrics = dict(population_nmse=np.mean((pop - dy[None])**2, axis=1)/variance,
                    validation_nmse=np.mean((val - vclean[None] - sigma[:, None]*vnoise)**2, axis=1)/variance,
                    test_nmse=np.mean((test - tclean[None] - sigma[:, None]*tnoise)**2, axis=1)/variance)
                for i, record in enumerate(records):
                    rows.append(dict(seed=seed, task=task, step=step, **record,
                        sigma=float(sigma[i]), variance=variance, noise_floor=float(sigma[i]**2/variance),
                        clips=int(clips[i]), bounds=int(bounds[i]), **{k:float(v[i]) for k,v in metrics.items()}))
            if step == protocol['steps']:
                break
            ix = stream.integers(len(x), size=64)
            state = models.algebra_state(theta, x[ix], left, right)
            delivered = models.deliver(state['path'], profiles, [r['rule'] for r in records])
            residual = (state['output'] - y[:, ix]) / variance
            gradient = np.einsum('cb,cbj,cjbf->cjf', residual, delivered, state['features']) / len(ix)
            norm = np.linalg.norm(gradient.reshape(n, -1), axis=1)
            clips += norm > 10
            gradient *= np.minimum(1., 10./np.maximum(norm, 1e-30))[:, None, None]
            moment = .9*moment + .1*gradient; second = .999*second + .001*gradient**2
            theta -= rates * (moment/(1-.9**(step+1))) / (np.sqrt(second/(1-.999**(step+1)))+1e-8)
            bounds += np.any(abs(theta.reshape(n, -1)) > 2., axis=1)
            theta = np.clip(theta, -2., 2.)
            assert np.isfinite(theta).all()
        path = root / 'states' / f'{seed}_{task}.npz'
        assert not path.exists()
        np.savez_compressed(path, theta=np.stack(states), steps=protocol['checkpoints'],
                            coefficients=coeff, left=left, right=right, profiles=profiles)
        inventory[str(path.relative_to(root))] = sha(path)
    path = root / 'results' / f'{seed}.csv'
    assert not path.exists()
    pd.DataFrame(rows).to_csv(path, index=False)
    inventory[str(path.relative_to(root))] = sha(path)
    write_new(root / 'results' / f'{seed}.json', dict(seed=seed, seconds=time.monotonic()-started,
              protocol_sha256=sha(root/'protocol.json'), outputs=inventory, job_id=os.getenv('SLURM_JOB_ID')))
    print(seed, round(time.monotonic()-started, 1), flush=True)


def analyze(root):
    p = json.loads((root/'protocol.json').read_text())
    parts = []
    replay_error = 0.
    for seed in p['seeds']:
        audit = json.loads((root/'results'/f'{seed}.json').read_text())
        assert audit['protocol_sha256'] == sha(root/'protocol.json')
        for rel, h in audit['outputs'].items():
            assert sha(root/rel) == h
        rows = pd.read_csv(root/'results'/f'{seed}.csv'); parts.append(rows)
        # Independent endpoint replay on all 256 inputs; every condition retained.
        for task in p['tasks']:
            z = np.load(root/'states'/f'{seed}_{task}.npz')
            x = models.structure.domain(); target = models.structure.fourier_design(x)@z['coefficients']
            for step in [1024, 16384]:
                w = z['theta'][list(z['steps']).index(step)]
                pred = models.algebra.forward(x, w, z['left'], z['right'])[:, 14]
                values = np.mean((pred-target[None])**2, 1)/(z['coefficients']@z['coefficients'])
                reported = rows[rows.task.eq(task)&rows.step.eq(step)].population_nmse.to_numpy()
                replay_error=max(replay_error,float(np.max(abs(values-reported))))
    assert replay_error < 1e-10, replay_error
    allrows = pd.concat(parts, ignore_index=True)
    rng = np.random.default_rng(20260921)
    contrast = []
    for common in [False, True]:
        subset=allrows[allrows.common_rate if common else allrows.selected_rate]
        for (noise, step), g in subset.groupby(['noise','step']):
            if step not in [1024,4096,8192,16384]: continue
            for metric in ['population_nmse','test_nmse']:
                wide=g.pivot(index='seed', columns=['task','rule'], values=metric)
                d=(wide['quartet','calibrated_broadcast']-wide['quartet','exact'])-(wide['matching','calibrated_broadcast']-wide['matching','exact'])
                bs=d.to_numpy()[rng.integers(len(d),size=(10000,len(d)))].mean(1)
                contrast.append(dict(noise=noise,step=step,common_rate=common,metric=metric,
                    mean=d.mean(),ci_low=np.quantile(bs,.025),ci_high=np.quantile(bs,.975),positive=int((d>0).sum()),n=len(d)))
    out=root/'analysis';out.mkdir(exist_ok=False)
    allrows.to_csv(out/'noise_controls_curves.csv',index=False)
    pd.DataFrame(contrast).to_csv(out/'noise_controls_contrasts.csv',index=False)
    write_new(out/'noise_controls_provenance.json',dict(protocol=p,maximum_replay_error=replay_error,
        outputs={x.name:sha(x) for x in out.glob('*.csv')},analysis_sha256=sha(__file__)))
    print(pd.DataFrame(contrast).query('step==16384 and metric=="population_nmse"').to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','analyze'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.root)
    elif args.action=='analyze': analyze(args.root)
    else:
        p=json.loads((args.root/'protocol.json').read_text())
        with ProcessPoolExecutor(args.workers) as pool: list(pool.map(run_seed,[(args.root,s) for s in p['seeds']]))
