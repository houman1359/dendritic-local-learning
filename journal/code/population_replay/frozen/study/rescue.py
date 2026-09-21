"""Input-dependent parent sensitivity in the frozen production DendriNet.

The parent selection study and production runtime are imported from immutable
snapshots. No old experiment or checkpoint is modified by this follow-up.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiment import SelectionNet, dataset
from run import evaluate, save_new, sha

RULES = ('exact', 'broadcast', 'resistance', 'derivative',
         'shuffled_derivative', 'full_chain')


def context_shuffle(values, contexts, generator):
    """Permute examples independently per parent, preserving context strata.

Each column is one parent of one neuron. Marginals within every context and
parent are unchanged. Independent RNG leaves the minibatch stream untouched.
"""
    shuffled = values.clone()
    for context in torch.unique(contexts):
        indices = torch.where(contexts == context)[0]
        for column in range(values.shape[1]):
            perm = torch.randperm(len(indices), generator=generator)
            shuffled[indices, column] = values[indices[perm], column]
    return shuffled


class RescueNet(SelectionNet):
    def __init__(self, seed):
        super().__init__(seed, mode='shunt', variant='interaction')
        self.shuffle_rng = torch.Generator().manual_seed(seed + 7919)

    @torch.no_grad()
    def transport(self, rule):
        parent, soma = self.core.branch_layers[1:]
        h = self.distal_gate('resistance')
        voltage = parent._last_branch_diagnostics['V']
        derivative = 1 - voltage.tanh().square()
        if rule == 'shuffled_derivative':
            context = self.last_cue[:, -4:].argmin(-1)
            derivative = context_shuffle(derivative, context, self.shuffle_rng)
        terminal = h * derivative.repeat_interleave(2, -1)
        proximal = torch.ones_like(derivative)
        if rule == 'full_chain':
            # Diagnostic exact factorization: supplies all coupling ratios,
            # including the otherwise omitted proximal transport.
            soma_denominator = 1 + soma.branches_to_output.sum_conductances()
            proximal_factor = (soma.branches_to_output.weight()
                               / soma_denominator[:, None]).reshape(-1)
            base = 1 + parent.branches_to_output.sum_conductances()
            child_factor = (parent.branches_to_output.weight()
                            / base[:, None]).reshape(-1)
            terminal = terminal * child_factor * proximal_factor.repeat_interleave(2)
            proximal = proximal * proximal_factor
        return terminal, proximal

    def gradients(self, excitation, inhibition, target, variance, rule):
        if rule not in RULES:
            raise ValueError(rule)
        if rule in {'exact', 'broadcast', 'resistance'}:
            return super().gradients(excitation, inhibition, target, variance, rule)
        self.zero_grad(set_to_none=True)
        self.local = True
        prediction = self(excitation, inhibition)
        loss = .5 * (prediction - target).square().mean() / variance
        error = torch.autograd.grad(loss, self.outputs[-1], retain_graph=True)[0].detach()
        terminal, proximal = self.transport(rule)
        objective = loss
        for output, gain in zip(self.outputs[:-1], (terminal, proximal)):
            delivered = error.repeat_interleave(output.shape[1] // self.somata, -1) * gain
            objective = objective + (output * delivered).sum()
        objective.backward()
        self.local = False
        return float(loss.detach()) * 2


def flat(parameters):
    return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten()
                      for p in parameters]).detach().clone()


def diagnostics(net, data, variance):
    """Common-state terminal diagnostics, never used by the learning rule.

The interaction vector is the difference of gradients with full and separable
targets at identical predictions and a common normalization. It isolates the
target-driven interaction term, not the network's complete learning dynamics.
"""
    x, inhibition, target = data
    features = x[:, :8].log().reshape(-1, 4, 2).tanh()
    separable = (x[:, -4:] * .5 * features.sum(-1)
                 * x.new_tensor([1., -1., 1., -1.])).sum(-1)
    parameters = list(net.core.branch_layers[0].parameters())
    vectors, interaction_vectors = {}, {}
    rng_state = net.shuffle_rng.get_state()
    for rule in RULES:
        net.shuffle_rng.set_state(rng_state)
        net.gradients(x, inhibition, target, variance, rule)
        vectors[rule] = flat(parameters)
        # Identical shuffle in the two evaluations prevents a randomization
        # difference from masquerading as an interaction contribution.
        net.shuffle_rng.set_state(rng_state)
        net.gradients(x, inhibition, separable, variance, rule)
        interaction_vectors[rule] = vectors[rule] - flat(parameters)
    net.shuffle_rng.set_state(rng_state)
    reference = interaction_vectors['exact']
    exact = vectors['exact']
    rows = []
    for rule in RULES:
        v, interaction = vectors[rule], interaction_vectors[rule]
        rows.append(dict(rule=rule, terminal_norm=float(v.norm()),
                         terminal_cosine=float(v.dot(exact) / (v.norm()*exact.norm()).clamp_min(1e-30)),
                         interaction_norm=float(interaction.norm()),
                         exact_interaction_norm=float(reference.norm()),
                         interaction_projection=float(interaction.dot(reference) / reference.square().sum().clamp_min(1e-30)),
                         interaction_cosine=float(interaction.dot(reference) / (interaction.norm()*reference.norm()).clamp_min(1e-30))))
    return rows


def active_values(net):
    masks = {}
    for layer in net.core.branch_layers:
        for name in ('branch_excitation', 'branch_inhibition'):
            synapse = getattr(layer, name, None)
            if synapse is not None:
                masks[id(synapse.pre_w)] = synapse.connection_mask
    return [(p, masks.get(id(p))) for p in net.core.parameters()]


def bound_fraction(net, bound):
    active = torch.cat([(p[mask] if mask is not None else p.flatten())
                        for p, mask in active_values(net)])
    return float((active.abs() >= bound - 1e-10).double().mean())


def train(seed, rule, optimizer, bound, rate, steps, phase):
    started = time.monotonic()
    training = dataset(seed, 'train', 2048, 'interaction')
    validation = dataset(seed, 'validation', 1024, 'interaction')
    net = RescueNet(seed).double()
    net.calibrate(*training[:2])
    variance = training[2].var(unbiased=False)
    diagnostic = dataset(seed, 'diagnostic', 512, 'interaction')
    initial_diagnostics = diagnostics(net, diagnostic, variance) if phase == 'fresh' else []
    opt = (torch.optim.Adam if optimizer == 'adam' else torch.optim.SGD)(net.parameters(), lr=rate)
    minibatches = torch.Generator().manual_seed(seed + 100)
    best = evaluate(net, validation)
    best_state = copy.deepcopy(net.state_dict())
    best_step = 0
    history = [dict(step=0, validation_nmse=best, bound_fraction=bound_fraction(net, bound))]
    bounds, clips, first_bound = 0, 0, None
    active = active_values(net)
    for step in range(1, steps + 1):
        indices = torch.randint(2048, (128,), generator=minibatches)
        net.gradients(training[0][indices], training[1][indices], training[2][indices], variance, rule)
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10.)
        if not torch.isfinite(norm):
            raise FloatingPointError(f'Nonfinite gradient: {seed, rule, optimizer, bound, rate, step}')
        clips += int(norm > 10)
        opt.step()
        hit = False
        with torch.no_grad():
            for p, mask in active:
                values = p[mask] if mask is not None else p
                hit |= bool((values.abs() > bound).any())
                p.clamp_(-bound, bound)
        if hit and first_bound is None:
            first_bound = step
        bounds += int(hit)
        if step % 128 == 0 or step == steps:
            value = evaluate(net, validation)
            if not np.isfinite(value):
                raise FloatingPointError('Nonfinite validation loss')
            history.append(dict(step=step, validation_nmse=value, bound_fraction=bound_fraction(net, bound)))
            if value < best:
                best, best_step = value, step
                best_state = copy.deepcopy(net.state_dict())
    endpoint_state = copy.deepcopy(net.state_dict())
    endpoint_validation = evaluate(net, validation)
    net.load_state_dict(best_state)
    result = dict(seed=seed, rule=rule, optimizer=optimizer, bound=bound, rate=rate,
                  steps=steps, selected_step=best_step, validation_nmse=best,
                  endpoint_validation_nmse=endpoint_validation, bounds=bounds,
                  clips=clips, first_bound_step=first_bound,
                  selected_bound_fraction=bound_fraction(net, bound), history=history,
                  initial_diagnostics=initial_diagnostics)
    # Development selection does not inspect or even evaluate held-out test.
    if phase == 'fresh':
        result.update(test_nmse=evaluate(net, dataset(seed, 'test', 4096, 'interaction')),
                      ood_nmse={str(s): evaluate(net, dataset(seed, 'ood', 4096, 'interaction', s))
                                for s in [1., 2., 3.]},
                      selected_diagnostics=diagnostics(net, diagnostic, variance))
    result['elapsed_seconds'] = time.monotonic() - started
    return result, dict(selected=best_state, endpoint=endpoint_state)


def run(root, phase, index):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    path = root / f'{phase}_protocol.json'
    protocol = json.loads(path.read_text())
    for relative, expected in protocol['source_sha256'].items():
        if sha(root / relative) != expected:
            raise RuntimeError(f'Source drift: {relative}')
    job = protocol['jobs'][index]
    key = f"s{job['seed']}_{job['optimizer']}_b{job['bound']}_{job['rule']}_r{job['rate']:g}"
    output = root / phase / 'results' / f'{key}.json'
    checkpoint = root / phase / 'checkpoints' / f'{key}.pt'
    if output.exists() or checkpoint.exists():
        raise FileExistsError(key)
    result, states = train(**job, steps=protocol['steps'], phase=phase)
    torch.save(states, checkpoint)
    result.update(protocol_sha256=sha(path), checkpoint_sha256=sha(checkpoint),
                  job_id=os.environ.get('SLURM_JOB_ID'), torch_version=torch.__version__)
    save_new(output, result)
    print(key, result['validation_nmse'], result.get('test_nmse'), result['elapsed_seconds'], flush=True)
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['development', 'fresh'], required=True)
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    run(args.root, args.phase, args.index)
