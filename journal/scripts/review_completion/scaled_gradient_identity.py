"""Verify the terminal raw-gradient identity at archived nonlinear states.

This is a deterministic algebra/replay check, not a new training cohort.
The runtime is verified by the released population launcher's hash chain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(args):
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['WANDB_DISABLED'] = 'true'
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('population_launcher', args.launcher)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    frozen, identity, original = launcher.verified_sources()
    sys.path[:0] = [str(frozen / d) for d in ('study', 'selection', 'base', 'runtime/src')]
    import torch
    import pandas as pd
    from rescue import RescueNet, dataset
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    files = sorted((args.archive / 'fresh/results').glob('*_adam_b9_*.json'))
    assert len(files) == 120, len(files)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Deterministic checkpoint-only terminal gradient identity check; no training or inference',
        states='All 120 selected original-bound Adam states (six rules by twenty seeds), plus twenty paired initial states',
        equation='exact terminal log-conductance gradient = kappa * augmented terminal log-conductance gradient',
        coordinates='Raw log-conductance gradients before clipping or optimizer transformation',
        examples='512 frozen diagnostic examples per seed; loss normalized by variance of the 2048 training targets',
        tolerance='relative L2 residual <= 1e-10; all terminal kappa strictly positive',
        source_sha256=sha(__file__), runtime_identity_sha256=sha(frozen / 'identity.json'),
        original_protocol_sha256=identity['original_fresh_protocol_sha256'])
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    inputs, rows, seen = {}, [], set()

    def check(net, seed, state, trained_rule):
        x, inh, y = dataset(seed, 'diagnostic', 512, 'interaction')
        variance = dataset(seed, 'train', 2048, 'interaction')[2].var(unbiased=False)
        terminal, parent, soma = net.core.branch_layers
        parameters = list(terminal.named_parameters())
        gradients = {}
        for rule in ('exact', 'derivative'):
            net.gradients(x, inh, y, variance, rule)
            gradients[rule] = {n: p.grad.detach().clone() for n, p in parameters}
        with torch.no_grad():
            parent_base = 1 + parent.branches_to_output.sum_conductances()
            soma_den = 1 + soma.branches_to_output.sum_conductances()
            kappa = (parent.branches_to_output.weight() / parent_base[:, None]).reshape(-1)
            kappa = kappa * (soma.branches_to_output.weight() / soma_den[:, None]).reshape(-1).repeat_interleave(2)
            assert bool((kappa > 0).all())
            for name, parameter in parameters:
                exact = gradients['exact'][name]
                scaled = gradients['derivative'][name] * kappa.reshape((-1,) + (1,) * (parameter.ndim - 1))
                residual = scaled - exact
                relative = float(residual.norm() / exact.norm().clamp_min(1e-30))
                assert relative <= 1e-10, (seed, state, name, relative)
                rows.append(dict(seed=seed, state=state, trained_rule=trained_rule,
                    parameter=name, relative_l2_residual=relative,
                    max_absolute_residual=float(residual.abs().max()), exact_norm=float(exact.norm()),
                    kappa_min=float(kappa.min()), kappa_max=float(kappa.max())))

    for path in files:
        result = json.loads(path.read_text())
        seed = result['seed']
        checkpoint = args.archive / 'fresh/checkpoints' / (path.stem + '.pt')
        assert sha(checkpoint) == result['checkpoint_sha256']
        inputs[str(path)] = sha(path)
        inputs[str(checkpoint)] = sha(checkpoint)
        net = RescueNet(seed).double()
        net.calibrate(*dataset(seed, 'train', 2048, 'interaction')[:2])
        if seed not in seen:
            check(net, seed, 'initial', 'paired_initialization')
            seen.add(seed)
        net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['selected'])
        check(net, seed, 'selected', result['rule'])
    table = pd.DataFrame(rows)
    output = args.output / 'scaled_gradient_identity.csv'
    table.to_csv(output, index=False)
    report = dict(protocol=protocol, states=140, parameter_blocks=len(table),
        max_relative_l2_residual=float(table.relative_l2_residual.max()),
        max_absolute_residual=float(table.max_absolute_residual.max()),
        inputs=inputs, outputs={output.name:sha(output)}, torch=torch.__version__)
    (args.output / 'scaled_gradient_identity_provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs', 'protocol')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
