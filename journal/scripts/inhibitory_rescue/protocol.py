"""Freeze a targeted rescue protocol before evaluating fresh seeds."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


RULES = ['exact', 'broadcast', 'resistance', 'derivative', 'shuffled_derivative', 'full_chain']
DEVELOPMENT_SEEDS = list(range(2026100100, 2026100103))
FRESH_SEEDS = list(range(2026100200, 2026100220))
RATES = {'adam': [.01, .03, .1], 'sgd': [.01, .1, 1.]}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, payload):
    with Path(path).open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def conditions():
    rows = [dict(optimizer=optimizer, bound=9, rule=rule)
            for optimizer in ['adam', 'sgd'] for rule in RULES]
    rows += [dict(optimizer='adam', bound=12, rule=rule)
             for rule in ['exact', 'resistance', 'derivative', 'shuffled_derivative']]
    return rows


def condition_key(row):
    return f"{row['optimizer']}_b{row['bound']}_{row['rule']}"


def prepare(root, parent):
    old = json.loads((parent / 'fresh_protocol.json').read_text())
    for relative, expected in old['source_sha256'].items():
        if sha(parent / relative) != expected:
            raise RuntimeError(f'Parent source drift: {relative}')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'study').mkdir()
    for source in Path(__file__).parent.iterdir():
        if source.suffix in {'.py', '.sh', '.md'}:
            shutil.copyfile(source, root / 'study' / source.name)
    for name, target in [('selection', parent / 'study'), ('base', parent / 'base'),
                         ('runtime', parent / 'runtime')]:
        (root / name).symlink_to(target.resolve(), target_is_directory=True)
    hashes = {(relative.replace('study/', 'selection/', 1)
               if relative.startswith('study/') else relative): digest
              for relative, digest in old['source_sha256'].items()}
    hashes.update({str(p.relative_to(root)): sha(p) for p in (root / 'study').iterdir()})
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    status='Internally specified prospective follow-up informed by earlier results; not externally preregistered',
                    parent_protocol_sha256=sha(parent / 'fresh_protocol.json'),
                    source_sha256=hashes, development_seeds=DEVELOPMENT_SEEDS,
                    fresh_seeds=FRESH_SEEDS, conditions=conditions(), rates=RATES,
                    steps=2048, fresh_steps=4096,
                    jobs=[dict(seed=seed, **condition, rate=rate)
                          for seed in DEVELOPMENT_SEEDS for condition in conditions()
                          for rate in RATES[condition['optimizer']]],
                    task='Unchanged nonlinear-parent interaction target, beta=0.25, shunting forward, 16 [4,2] DendriNet neurons',
                    parameterization='Exponential conductances; main log bounds [-9,9], predefined sensitivity [-12,12]; norm-10 clipping',
                    tuning='Equal three-rate budget per rule/optimizer/bound; minimize mean development validation NMSE; no development test evaluation',
                    selection='Validation checkpoint selection every 128 updates including initialization; same samples and initialization per seed',
                    primary=[dict(left='resistance', right='derivative', optimizer='adam', bound=9, metric='test_nmse'),
                             dict(left='shuffled_derivative', right='derivative', optimizer='adam', bound=9, metric='test_nmse')],
                    inference='20 fresh paired seeds; mean paired differences, 95% seed-bootstrap intervals, two-sided sign-flip tests, Holm correction over two primary comparisons',
                    secondary=['SGD to expose scale dependence', 'Predefined wider-bound Adam sensitivity',
                               'Common-state terminal target-interaction gradient projection and cosine',
                               'Severity 1,2,3 stress; first bound contact and fraction at selected state'],
                    safeguards=['Context-stratified per-parent example shuffling uses an independent RNG',
                                'Full-chain arm is a diagnostic algebraic BP implementation, not a new independent method',
                                'Development and fresh outcomes are retained irrespective of direction',
                                'No learned context, endogenous somatic error, reciprocal cable, or spiking claim',
                                'No W&B; outputs and checkpoints stay on kempner_project_b'])
    save_new(root / 'development_protocol.json', protocol)
    for phase in ['development', 'fresh']:
        for name in ['results', 'checkpoints', 'logs']:
            (root / phase / name).mkdir(parents=True)
    print(f"Prepared {root}: {len(protocol['jobs'])} development fits; {len(conditions()) * len(FRESH_SEEDS)} fresh fits")


def freeze(root):
    protocol = json.loads((root / 'development_protocol.json').read_text())
    if list((root / 'fresh' / 'results').glob('*')):
        raise RuntimeError('Fresh outcomes already exist')
    rows, hashes = [], {}
    for job in protocol['jobs']:
        key = f"s{job['seed']}_{condition_key(job)}_r{job['rate']:g}"
        path = root / 'development' / 'results' / f'{key}.json'
        result = json.loads(path.read_text())
        if result['protocol_sha256'] != sha(root / 'development_protocol.json'):
            raise RuntimeError(f'Protocol mismatch: {key}')
        if any(result[k] != job[k] for k in job):
            raise RuntimeError(f'Job metadata mismatch: {key}')
        if not 0 <= result['validation_nmse'] < float('inf'):
            raise RuntimeError(f'Invalid validation loss: {key}')
        if sha(root / 'development' / 'checkpoints' / f'{key}.pt') != result['checkpoint_sha256']:
            raise RuntimeError(f'Checkpoint mismatch: {key}')
        rows.append(result)
        hashes[str(path.relative_to(root))] = sha(path)
    choices, scores = {}, {}
    for condition in protocol['conditions']:
        key = condition_key(condition)
        scores[key] = {rate: sum(r['validation_nmse'] for r in rows
                                if condition_key(r) == key and r['rate'] == rate) / len(DEVELOPMENT_SEEDS)
                       for rate in RATES[condition['optimizer']]}
        choices[key] = min(scores[key], key=scores[key].get)
    fresh = {**protocol, 'created_utc': datetime.now(timezone.utc).isoformat(),
             'status': 'Fresh-seed evaluation of validation-selected settings; internally frozen',
             'development_protocol_sha256': sha(root / 'development_protocol.json'),
             'development_result_sha256': hashes, 'validation_rate_scores': scores,
             'fixed_rates': choices, 'steps': protocol['fresh_steps'],
             'jobs': [dict(seed=seed, **condition, rate=choices[condition_key(condition)])
                      for seed in FRESH_SEEDS for condition in conditions()]}
    save_new(root / 'fresh_protocol.json', fresh)
    print(json.dumps(choices, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'freeze'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    args = parser.parse_args()
    prepare(args.root, args.parent) if args.action == 'prepare' else freeze(args.root)
