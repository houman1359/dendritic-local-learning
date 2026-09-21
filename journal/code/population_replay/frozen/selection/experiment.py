"""Selection and distractor robustness in production DendriNet.

Imports the hashed original transfer adapter via the execution PYTHONPATH.
The primary target removes the interaction that the original identity-output
configuration could not represent; the interaction arm adds a nonlinear parent.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from model import CreditNet, RULES, dataset as original_dataset
from run import diagnose, evaluate, save_new, sha

VARIANTS = {'separable': 0., 'interaction': .25}


class SelectionNet(CreditNet):
    def __init__(self, seed, mode='shunt', variant='separable'):
        self.nonlinear_parent = variant == 'interaction'
        super().__init__(seed, 16, 4, True, mode)
        if self.nonlinear_parent:
            self.core.branch_layers[1].reactivation = nn.Tanh()

    def _post_hook(self, index):
        base = super()._post_hook(index)
        def hook(layer, args, output):
            result = base(layer, args, output)
            if index == 1 and self.forward_mode == 'current' and self.nonlinear_parent:
                result = layer.reactivation(result)
                self.outputs[-1] = result
            return result
        return hook

    @torch.no_grad()
    def calibrate(self, excitation, inhibition):
        super().calibrate(excitation, inhibition)
        if self.nonlinear_parent:
            # Current matching refers to voltage before the parent nonlinearity.
            voltage = self.core.branch_layers[1]._last_branch_diagnostics['V']
            self.current_anchor.copy_(voltage.mean(0))


def dataset(seed, split, size, variant='separable', severity=1.):
    x, inhibition, _ = original_dataset(seed, split, size, 4, 'selection', severity)
    z = x[:, :8].log().reshape(-1, 4, 2)
    cue = x[:, -4:]
    features = z.tanh()
    evidence = .5 * features.sum(-1) + VARIANTS[variant] * features.prod(-1)
    evidence *= torch.tensor([1., -1., 1., -1.], dtype=x.dtype)
    return x, inhibition, (cue*evidence).sum(-1)


def train(seed, variant, mode, rule, rate, steps):
    started = time.monotonic()
    training = dataset(seed, 'train', 2048, variant)
    validation = dataset(seed, 'validation', 1024, variant)
    diagnostic = dataset(seed, 'diagnostic', 128, variant)
    net = SelectionNet(seed, mode, variant).double()
    net.calibrate(*training[:2])
    initial = diagnose(net, diagnostic)
    optimizer = torch.optim.Adam(net.parameters(), lr=rate)
    rng = torch.Generator().manual_seed(seed+100)
    variance = training[2].var(unbiased=False)
    best = evaluate(net, validation)
    best_state = copy.deepcopy(net.state_dict())
    best_step = 0
    history = [dict(step=0, validation_nmse=best)]
    bounds, clips = 0, 0
    masks = {}
    for layer in net.core.branch_layers:
        for name in ('branch_excitation', 'branch_inhibition'):
            synapse = getattr(layer, name, None)
            if synapse is not None:
                masks[id(synapse.pre_w)] = synapse.connection_mask
    for step in range(1, steps+1):
        ix = torch.randint(2048, (128,), generator=rng)
        net.gradients(training[0][ix], training[1][ix], training[2][ix], variance, rule)
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        if not torch.isfinite(norm):
            raise FloatingPointError(f'Nonfinite update: {seed, variant, mode, rule, rate, step}')
        clips += int(norm>10)
        optimizer.step()
        hit = False
        with torch.no_grad():
            for p in net.core.parameters():
                active = p[masks[id(p)]] if id(p) in masks else p
                hit |= bool(((active < -9) | (active > 9)).any())
                p.clamp_(-9, 9)
        bounds += int(hit)
        if step%128 == 0 or step == steps:
            val = evaluate(net, validation)
            if not np.isfinite(val):
                raise FloatingPointError('Nonfinite validation loss')
            history.append(dict(step=step, validation_nmse=val))
            if val < best:
                best, best_step = val, step
                best_state = copy.deepcopy(net.state_dict())
    endpoint_validation = evaluate(net, validation)
    net.load_state_dict(best_state)
    result = dict(seed=seed, variant=variant, beta=VARIANTS[variant], forward=mode,
                  rule=rule, rate=rate, steps=steps, selected_step=best_step,
                  validation_nmse=best, final_validation_nmse=endpoint_validation,
                  test_nmse=evaluate(net, dataset(seed, 'test', 4096, variant)),
                  ood_nmse={str(s): evaluate(net, dataset(seed, 'ood', 4096, variant, s))
                            for s in [1., 1.5, 2., 2.5, 3.]},
                  bounds= bounds, clips=clips, active_parameters=net.active_parameters(),
                  initial_diagnostics=initial, selected_diagnostics=diagnose(net, diagnostic),
                  history=history, elapsed_seconds=time.monotonic()-started)
    return result, best_state


def validate_sources(root, protocol):
    for relative, expected in protocol['source_sha256'].items():
        if sha(root/relative) != expected:
            raise RuntimeError(f'Execution source drift: {relative}')


def run(root, phase, index):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol_path = root/f'{phase}_protocol.json'
    protocol = json.loads(protocol_path.read_text())
    validate_sources(root, protocol)
    job = protocol['jobs'][index]
    for rule in RULES:
        rates = protocol['rates'] if phase=='development' else [protocol['fixed_rates'][job['variant']][job['mode']][rule]]
        for rate in rates:
            key = f"s{job['seed']}_{job['variant']}_{job['mode']}_{rule}_r{rate:g}"
            out = root/phase/'results'/f'{key}.json'
            if out.exists():
                raise FileExistsError(out)
            result, state = train(**job, rule=rule, rate=rate, steps=protocol['steps'])
            path = root/phase/'checkpoints'/f'{key}.pt'
            if path.exists():
                raise FileExistsError(path)
            torch.save(state, path)
            result.update(protocol_sha256=sha(protocol_path), checkpoint_sha256=sha(path),
                          phase=phase, job_id=os.environ.get('SLURM_JOB_ID'),
                          torch_version=torch.__version__)
            save_new(out, result)
            print(key, result['test_nmse'], result['ood_nmse']['3.0'], result['elapsed_seconds'], flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', choices=['development','fresh'], required=True)
    p.add_argument('--index', type=int, required=True)
    a=p.parse_args();run(a.root,a.phase,a.index)
