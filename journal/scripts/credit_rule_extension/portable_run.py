#!/usr/bin/env python3
"""Verify and replay the published extension without Git or canonical writes.

This is a post-experiment release wrapper. It calls the unchanged scientific
run_task after authenticating the frozen protocol and every input. Release-only
path transformations require the existing, explicit provenance hash chain.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
PROTOCOL_SHA256 = '17db44ba893bb2662fa39d2cb185df374e5cefd50f15626aac310ab534d5eacd'
EXTENSION = Path('source_data/credit_rule_extension')
ORIGINAL = Path('source_data/credit_rule_bridge')
# Transitive import omitted by the historical top-level freeze; recorded now
# for portable loading, and verified unchanged from the actual launch commit.
SUPPORTING_SOURCE_SHA256 = {'scripts/morphology_structure/constructive.py': '8259af41912cee139998b84dffef9827d44fe8a175a89e6eb91bb1f534303a46'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def safe_child(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe frozen input path')
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Frozen input escapes journal root')
    return result


def verify_inputs(journal):
    helper = load('_credit_extension_release_hashes', journal / 'code/release_noise/release_hashes.py')
    verified = []

    def verify(relative, original_digest):
        path = safe_child(journal, relative)
        verdict = helper.verify_released_file(path, original_digest, journal_root=journal)
        if not verdict['verified']:
            raise ValueError(f'Unverified frozen input {relative}: {verdict["reason"]}')
        verified.append(dict(path=Path(relative).as_posix(), original_sha256=original_digest,
                             released_sha256=verdict['released_sha256'], reason=verdict['reason']))
        return path

    protocol = json.loads(verify(EXTENSION / 'protocol_freeze.json', PROTOCOL_SHA256).read_text())
    inventory = json.loads(verify(EXTENSION / 'input_inventory.json', protocol['input_inventory_sha256']).read_text())
    for relative, expected in inventory.items():
        verify(relative, expected)
    for relative, expected in SUPPORTING_SOURCE_SHA256.items():
        verify(relative, expected)
    original = json.loads((journal / ORIGINAL / 'protocol_freeze.json').read_text())
    for relative, expected in original['source_sha256'].items():
        if inventory.get(relative) != expected:
            raise ValueError(f'Original and extended source identities disagree: {relative}')
    selection = json.loads((journal / ORIGINAL / 'selection_freeze.json').read_text())
    if selection['protocol_freeze_sha256'] != inventory[(ORIGINAL / 'protocol_freeze.json').as_posix()]:
        raise ValueError('Original selection does not identify its verified protocol')
    if protocol['max_updates'] != 16384 or protocol['unique_trajectories'] != 720:
        raise ValueError('Unexpected scientific budget or cohort')
    return protocol, original, verified


def load_scientific(journal):
    expected = {'models': journal / 'scripts/credit_rule_bridge/models.py',
                'model': journal / 'scripts/morphology_structure/model.py',
                'constructive_dp_v2': journal / 'scripts/morphology_structure/constructive_dp_v2.py',
                'constructive': journal / 'scripts/morphology_structure/constructive.py'}
    for name, path in expected.items():
        if name in sys.modules and Path(getattr(sys.modules[name], '__file__', '')).resolve() != path:
            raise ValueError(f'Conflicting preimported scientific module: {name}')
    sys.path.insert(0, str(journal / 'scripts/credit_rule_bridge'))
    runner = load('_credit_extension_original_runner', journal / 'scripts/credit_rule_bridge/run.py')
    for name, path in expected.items():
        if Path(sys.modules[name].__file__).resolve() != path:
            raise ValueError(f'Scientific import came from an unexpected file: {name}')
    return runner


def compare_historical(journal, seed, task, arrays, metadata, curves, diagnostics):
    import numpy as np
    import pandas as pd
    olddir = journal / ORIGINAL / 'runs/fresh/algebraic'
    maximum = 0.0
    with np.load(olddir / f'seed_{seed}_task_{task}_states.npz') as original:
        steps = np.intersect1d(original['steps'], arrays['steps'])
        ai = [int(np.flatnonzero(arrays['steps'] == step)[0]) for step in steps]
        oi = [int(np.flatnonzero(original['steps'] == step)[0]) for step in steps]
        for name in original.files:
            actual = arrays[name][ai] if name in ('theta', 'steps') else arrays[name]
            expected = original[name][oi] if name in ('theta', 'steps') else original[name]
            if actual.shape != expected.shape:
                raise ValueError(f'Replay shape mismatch: {task}/{name}')
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
            if name != 'theta' and not np.array_equal(actual, expected):
                raise ValueError(f'Replay coordinate mismatch: {task}/{name}')
    oldmeta = json.loads((olddir / f'seed_{seed}_task_{task}_metadata.json').read_text())
    if metadata != oldmeta:
        raise ValueError('Replay task metadata changed')
    keys = ['task', 'step', 'optimizer', 'rule', 'rate']
    for kind, records in [('curves', curves), ('diagnostics', diagnostics)]:
        old = pd.read_csv(olddir / f'seed_{seed}_{kind}.csv', float_precision='round_trip')
        old = old[(old.task == task) & old.step.isin(steps)].sort_values(keys).reset_index(drop=True)
        actual = pd.DataFrame(records)
        actual = actual[actual.step.isin(steps)].sort_values(keys).reset_index(drop=True)
        if len(actual) != len(old):
            raise ValueError('Replay row count changed')
        for column in old:
            if column == 'elapsed_seconds':
                continue
            if pd.api.types.is_numeric_dtype(old[column]) and not pd.api.types.is_bool_dtype(old[column]):
                maximum = max(maximum, float(np.max(np.abs(actual[column].to_numpy(float) - old[column].to_numpy(float)))))
            elif actual[column].astype(str).tolist() != old[column].astype(str).tolist():
                raise ValueError(f'Replay field changed: {kind}/{column}')
    if maximum > 1e-10:
        raise ValueError(f'Historical numerical replay differs by {maximum:g}; no outcomes are accepted')
    return dict(task=task, historical_checkpoints_checked=steps.tolist(), maximum_absolute_difference=maximum)


def execute(args):
    journal = args.journal_root.resolve()
    protocol, original, verified = verify_inputs(journal)
    if args.seed not in protocol['seeds']:
        raise ValueError('Seed is not in the frozen cohort')
    if args.excluded_smoke_steps is not None and args.excluded_smoke_steps not in (1, 16, 64):
        raise ValueError('Excluded smoke must end at original checkpoint 1, 16 or 64')
    cfg = copy.deepcopy(original['protocol'])
    cfg['checkpoints'] = protocol['checkpoints']
    runner = load_scientific(journal)
    if runner.conditions('algebraic', 'fresh', cfg) != protocol['records']:
        raise ValueError('Loaded conditions differ from the frozen extension')
    import numpy as np
    import pandas as pd
    identity = dict(protocol_original_sha256=PROTOCOL_SHA256, verified_inputs=verified,
                    original_environment=protocol['runtime_at_freeze'],
                    actual_environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                    portable_launcher_sha256=sha(__file__), scientific_sources_modified=False,
                    historical_git_guard='Replaced only for archive replay by frozen-protocol and release-chain verification; no Git provenance is invented.')
    if args.verify_only:
        result = dict(status='passed', mode='verify_only', **identity)
        print(json.dumps(result, indent=2)); return result
    if args.output_root is None:
        raise ValueError('Replay requires a new --output-root')
    output = args.output_root.resolve()
    if output.exists() or output.is_relative_to(journal / 'source_data'):
        raise ValueError('Replay output must be new and outside canonical source_data')
    output.mkdir(parents=True)
    smoke = args.excluded_smoke_steps is not None
    steps = args.excluded_smoke_steps if smoke else protocol['max_updates']
    result = dict(status='running', mode='excluded_smoke' if smoke else 'portable_existing_seed_replay',
                  excluded_from_scientific_results=True, seed=args.seed, steps=steps, unique_trajectories=36,
                  scientific_results_unchanged=True, **identity)
    dump(output / 'portable_audit.json', result)
    start = time.perf_counter(); checks = []; curves = []; diagnostics = []; files = {}
    try:
        for task in protocol['tasks']:
            c, d, arrays, metadata = runner.run_task(args.seed, 'algebraic', task, 'fresh', cfg, steps=steps)
            checks.append(compare_historical(journal, args.seed, task, arrays, metadata, c, d))
            curves.extend(c); diagnostics.extend(d)
            path = output / f'{task}_states.npz'; np.savez_compressed(path, **arrays); files[path.name] = sha(path)
            path = output / f'{task}_metadata.json'; dump(path, metadata); files[path.name] = sha(path)
        for name, records in [('curves', curves), ('diagnostics', diagnostics)]:
            path = output / f'{name}.csv'; pd.DataFrame(records).to_csv(path, index=False); files[path.name] = sha(path)
        result.update(status='passed', replay_checks=checks, output_sha256=files, elapsed_seconds=time.perf_counter() - start)
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - start)
        dump(output / 'portable_audit.json', result)
        raise
    dump(output / 'portable_audit.json', result)
    print(json.dumps(result, indent=2)); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal-root', type=Path, default=HERE.parents[1])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--excluded-smoke-steps', type=int)
    execute(parser.parse_args())
