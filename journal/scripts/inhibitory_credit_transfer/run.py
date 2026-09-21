"""Development and fresh-seed training with common-state mechanistic diagnostics."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from model import CreditNet, FORWARDS, RULES, dataset


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def evaluate(net, data):
    x, i, y = data
    net.local = False
    prediction = net(x, i)
    return float((prediction-y).square().mean()/y.var(unbiased=False))


def diagnose(net, data):
    """All rules at one state; norm-matched small SGD steps, not Adam steps.

    This diagnostic may evaluate exact gradients. Training local rules cannot.
    Only core directions change in the intervention, holding the readout fixed.
    """
    x, i, y = data
    variance = y.var(unbiased=False)
    original = copy.deepcopy(net.state_dict())
    parameters = list(net.core.parameters())
    vectors = {}
    for rule in RULES:
        net.gradients(x, i, y, variance, rule)
        vectors[rule] = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten()
                                    for p in parameters]).detach().clone()
    exact = vectors['exact']
    baseline = evaluate(net, data)
    results = []
    for rule, vector in vectors.items():
        norm = vector.norm()
        cosine = float(torch.dot(vector, exact)/(norm*exact.norm()).clamp_min(1e-30))
        # Same Euclidean displacement in log-conductance space, all rules.
        displacement = 1e-3 * vector / norm.clamp_min(1e-30)
        offset = 0
        with torch.no_grad():
            for p in parameters:
                p.sub_(displacement[offset:offset+p.numel()].view_as(p))
                offset += p.numel()
        after = evaluate(net, data)
        net.load_state_dict(original)
        results.append(dict(rule=rule, core_gradient_norm=float(norm),
                            exact_core_cosine=cosine, same_norm_nmse_decrease=baseline-after,
                            displacement_norm=1e-3, baseline_nmse=baseline))
    return results


def train_one(seed, somata, streams, task, mode, rule, rate, steps, size=2048):
    started = time.monotonic()
    train = dataset(seed, 'train', size, streams, task)
    validation = dataset(seed, 'validation', 1024, streams, task)
    test = dataset(seed, 'test', 4096, streams, task)
    diagnostic = dataset(seed, 'diagnostic', 128, streams, task)
    net = CreditNet(seed, somata, streams, True, mode).double()
    net.calibrate(*train[:2])
    initial = diagnose(net, diagnostic)
    optimizer = torch.optim.Adam(net.parameters(), lr=rate)
    generator = torch.Generator().manual_seed(seed + 100)
    variance = train[2].var(unbiased=False)
    best = evaluate(net, validation)
    best_state = copy.deepcopy(net.state_dict())
    best_step = 0
    history = [dict(step=0, validation_nmse=best)]
    bound_steps = 0
    clip_steps = 0
    masks = {}
    for layer in net.core.branch_layers:
        for name in ('branch_excitation', 'branch_inhibition'):
            synapse = getattr(layer, name, None)
            if synapse is not None:
                masks[id(synapse.pre_w)] = synapse.connection_mask
    for step in range(1, steps+1):
        ix = torch.randint(size, (128,), generator=generator)
        net.gradients(train[0][ix], train[1][ix], train[2][ix], variance, rule)
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        if not torch.isfinite(norm):
            raise FloatingPointError(f'Nonfinite gradient: {seed, task, mode, rule, rate, step}')
        clip_steps += int(norm > 10)
        optimizer.step()
        hit = False
        with torch.no_grad():
            for p in net.core.parameters():
                active = p[masks[id(p)]] if id(p) in masks else p
                hit |= bool(((active < -7) | (active > 7)).any())
                p.clamp_(-7, 7)
        bound_steps += int(hit)
        if step % 128 == 0 or step == steps:
            value = evaluate(net, validation)
            if not np.isfinite(value):
                raise FloatingPointError('Nonfinite validation loss')
            history.append(dict(step=step, validation_nmse=value))
            if value < best:
                best, best_step = value, step
                best_state = copy.deepcopy(net.state_dict())
    final_validation = evaluate(net, validation)
    net.load_state_dict(best_state)
    endpoint = dict(seed=seed, somata=somata, streams=streams, branching=[streams, 2],
                    active_parameters=net.active_parameters(), stored_parameters=sum(p.numel() for p in net.parameters()),
                    task=task, forward=mode, rule=rule, rate=rate, steps=steps,
                    best_step=best_step, validation_nmse=best, final_validation_nmse=final_validation,
                    test_nmse=evaluate(net, test), bound_steps=bound_steps, clip_steps=clip_steps,
                    elapsed_seconds=time.monotonic()-started,
                    generalization_nmse={str(s): evaluate(net, dataset(seed, 'ood', 4096, streams, task, s))
                                         for s in ([1., 2., 3.] if task == 'selection' else [1.])},
                    diagnostic_initial=initial, diagnostic_selected=diagnose(net, diagnostic), history=history)
    return endpoint, best_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol_path = args.root/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    for relative, expected in protocol['source_sha256'].items():
        if sha(args.root/relative) != expected:
            raise RuntimeError(f'Source snapshot drift: {relative}')
    row = protocol['jobs'][args.index]
    for mode in (row.pop('mode'),):
        for rule in RULES:
            for rate in protocol['rates']:
                key = f"s{row['seed']}_n{row['somata']}_b{row['streams']}_{row['task']}_{mode}_{rule}_r{rate:g}"
                target = args.root/'results'/f'{key}.json'
                if target.exists():
                    existing = json.loads(target.read_text())
                    if existing['protocol_sha256'] != sha(protocol_path):
                        raise RuntimeError(f'Existing result has different protocol: {target}')
                    continue
                result, state = train_one(**row, mode=mode, rule=rule, rate=rate, steps=protocol['steps'], size=protocol['train_size'])
                result.update(protocol_sha256=sha(protocol_path), job_id=os.environ.get('SLURM_JOB_ID'),
                              phase=protocol['phase'], torch_version=torch.__version__)
                checkpoint = args.root/'checkpoints'/f'{key}.pt'
                checkpoint.parent.mkdir(exist_ok=True)
                if checkpoint.exists():
                    raise FileExistsError(checkpoint)
                torch.save(state, checkpoint)
                result['checkpoint_sha256'] = sha(checkpoint)
                save_new(target, result)
                print(key, result['test_nmse'], result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
