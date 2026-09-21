"""Scientific contract tests; run against the frozen parent runtime."""
import torch
import pytest
from rescue import RescueNet, context_shuffle, dataset, flat, diagnostics


def test_shuffle_preserves_parent_and_context_marginals():
    values = torch.arange(120).reshape(24, 5).double()
    contexts = torch.arange(24) % 4
    shuffled = context_shuffle(values, contexts, torch.Generator().manual_seed(9))
    assert not torch.equal(shuffled, values)
    for c in range(4):
        torch.testing.assert_close(shuffled[contexts == c].sort(0).values,
                                   values[contexts == c].sort(0).values)


@pytest.mark.parametrize('seed', [7, 11])
def test_full_chain_equals_exact_for_all_parameters(seed):
    net = RescueNet(seed).double()
    x, i, y = dataset(31, 'train', 48, 'interaction')
    variance = y.var(unbiased=False)
    net.gradients(x, i, y, variance, 'exact')
    expected = flat(list(net.parameters()))
    net.gradients(x, i, y, variance, 'full_chain')
    torch.testing.assert_close(flat(list(net.parameters())), expected, rtol=1e-10, atol=1e-12)


def test_new_rules_leave_forward_and_readout_unchanged():
    net = RescueNet(4).double()
    x, i, y = dataset(8, 'train', 48, 'interaction')
    prediction = net(x, i).detach().clone()
    net.gradients(x, i, y, y.var(), 'exact')
    reference = flat(list(net.readout.parameters()))
    for rule in ['derivative', 'shuffled_derivative', 'full_chain']:
        net.gradients(x, i, y, y.var(), rule)
        torch.testing.assert_close(flat(list(net.readout.parameters())), reference)
        torch.testing.assert_close(net(x, i), prediction, rtol=0, atol=0)


def test_resistance_eligibility_does_not_contain_parent_derivative():
    net = RescueNet(1).double()
    x, i, y = dataset(2, 'train', 64, 'interaction')
    net(x, i)
    actual, proximal = net.transport('derivative')
    voltage = net.core.branch_layers[1]._last_branch_diagnostics['V']
    expected = net.distal_gate('resistance') * (1 - voltage.tanh().square()).repeat_interleave(2, -1)
    torch.testing.assert_close(actual, expected)
    assert bool((proximal == 1).all())
    assert not torch.allclose(actual, net.distal_gate('resistance'))


def test_diagnostics_preserve_shuffle_stream_and_exact_projection():
    net = RescueNet(3).double()
    data = dataset(4, 'diagnostic', 64, 'interaction')
    state = net.shuffle_rng.get_state().clone()
    rows = diagnostics(net, data, data[2].var())
    assert torch.equal(state, net.shuffle_rng.get_state())
    exact = next(row for row in rows if row['rule'] == 'exact')
    assert exact['interaction_projection'] == pytest.approx(1.)
    assert exact['interaction_cosine'] == pytest.approx(1.)
