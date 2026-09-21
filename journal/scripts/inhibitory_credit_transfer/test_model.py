"""Numerical/semantic tests, independent of whether a proposed rule wins."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import CreditNet, FORWARDS, RULES, dataset

spec = importlib.util.spec_from_file_location("figure5_reference", Path(__file__).parents[1] / "conductance_local_gate/model.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def mapped_gradient(net):
    leaf, proximal, soma = net.core.branch_layers
    values = []
    for j in range(4):
        values.extend([leaf.branch_excitation.pre_w.grad[j, j],
                       leaf.branch_excitation.pre_w.grad[j, j+4],
                       leaf.branch_inhibition.pre_w.grad[j, j],
                       leaf.branch_inhibition.pre_w.grad[j, j+4]])
    values.extend([proximal.branch_inhibition.pre_w.grad[0, 8],
                   proximal.branch_inhibition.pre_w.grad[1, 9]])
    values.extend(proximal.branches_to_output.log_weight.grad.flatten())
    values.extend(soma.branches_to_output.log_weight.grad.flatten())
    return torch.stack(values).numpy()


@pytest.mark.parametrize("rule", ["exact", "broadcast", "resistance"])
def test_production_transfer_matches_independent_figure5(rule):
    theta = np.log(reference.NOMINAL) + np.random.default_rng(2).normal(0, .2, 24)
    x, y, _ = reference.data(3, "train", 19, True)
    net = CreditNet(2, 1, 2, False).double()
    net.load_figure5(theta)
    xx, yy = torch.tensor(x), torch.tensor(y)
    prediction = net(xx, xx)
    state = reference.forward(theta[None], x)
    np.testing.assert_allclose(prediction.detach().numpy(), state['output'][0], rtol=1e-12)
    net.gradients(xx, xx, yy, np.var(y), rule)
    delivery = reference.exact_path(state)[0] if rule == 'exact' else np.ones_like(state['voltage'][0])
    if rule == 'resistance':
        base = 1 + np.exp(theta[18:22]).reshape(2, 2).sum(1)
        delivery[:, :4] = np.repeat(base / state['denominator'][0, :, 4:6], 2, axis=1)
    expected = (((state['output'][0]-y) / np.var(y))[:, None] *
                reference.eligibility(state, x)[0] * delivery[:, reference.PARAM_UNIT]).mean(0)
    np.testing.assert_allclose(mapped_gradient(net), expected, rtol=1e-10, atol=1e-12)
    assert net.active_parameters() == 26  # 24 conductances plus affine readout


@pytest.mark.parametrize("mode", FORWARDS)
def test_local_forward_equals_exact_forward_and_detaches_paths(mode):
    net = CreditNet(4, 2, 2, True, mode).double()
    x, i, y = dataset(3, 'train', 11, 2)
    net.calibrate(x, i)
    normal = net(x, i)
    net.local = True
    local = net(x, i)
    torch.testing.assert_close(normal, local, rtol=0, atol=0)
    assert torch.autograd.grad(local.sum(), net.outputs[0], allow_unused=True)[0] is None


def test_gate_controls_preserve_marginals_or_rms():
    net = CreditNet(4, 3, 4).double()
    x, i, _ = dataset(5, 'train', 17)
    net(x, i)
    correct, swapped, uniform = [net.distal_gate(r).reshape(17, 3, 8)
                                  for r in ('resistance', 'swapped', 'uniform_rms')]
    torch.testing.assert_close(correct.sort(-1).values, swapped.sort(-1).values)
    torch.testing.assert_close(correct.square().mean(-1), uniform.square().mean(-1))
    assert not torch.allclose(correct, swapped)


@pytest.mark.parametrize("mode", FORWARDS)
def test_exact_gradient_finite_difference(mode):
    net = CreditNet(7, 2, 2, True, mode).double()
    x, i, y = dataset(9, 'train', 13, 2)
    net.calibrate(x, i)
    net.gradients(x, i, y, y.var(), 'exact')
    p = net.core.branch_layers[1].branch_inhibition.pre_w
    actual = p.grad[0, 8].item()
    with torch.no_grad():
        initial = p[0, 8].item()
        p[0, 8] = initial + 1e-6
        plus = .5 * (net(x, i)-y).square().mean()/y.var()
        p[0, 8] = initial - 1e-6
        minus = .5 * (net(x, i)-y).square().mean()/y.var()
        p[0, 8] = initial
    assert actual == pytest.approx(float((plus-minus)/2e-6), rel=1e-5, abs=1e-9)


def test_identical_tonic_operating_point():
    net = CreditNet(6, 2, 2).double()
    x, i, _ = dataset(9, 'train', 1, 2)
    i[:, 8:] = net.tonic_activity
    net.calibrate(x, i)
    results = []
    for mode in FORWARDS:
        net.forward_mode = mode
        results.append(net(x, i))
    for value in results[1:]:
        torch.testing.assert_close(value, results[0], rtol=1e-12, atol=1e-12)


def test_external_targets_splits_and_context():
    x, i, y = dataset(10, 'train', 30)
    x2, i2, y2 = dataset(10, 'train', 30)
    torch.testing.assert_close(x, x2)
    assert not torch.equal(x, dataset(10, 'test', 30)[0])
    torch.testing.assert_close(i[:, -4:], 4 * (1-x[:, -4:]))
    assert torch.isfinite(y).all()
