"""Locality and numerical tests; successful training is not an assertion."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("gate_extension", Path(__file__).with_name("experiment.py"))
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


def fixture(task="graded_sensory_mixture"):
    x, y, c = ex.dataset(191, "train", 31, task)
    theta = np.log(ex.model.NOMINAL)[None]
    return x, y, c, theta, np.array([[4., -1.]])


def test_exact_gradient_including_decoder():
    x, y, _, theta, decoder = fixture()
    p = np.column_stack((theta, decoder))
    exact = ex.gradients(theta, decoder, x, y, np.var(y), ["exact"])[0]
    finite = []
    for k in range(26):
        d = np.zeros_like(p)
        d[0, k] = 1e-6
        finite.append(float((ex.evaluate(p + d, x, y, np.var(y))[0] -
                             ex.evaluate(p - d, x, y, np.var(y))[0]) / 4e-6))
    np.testing.assert_allclose(exact, finite, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("rule", ex.RULES[1:])
def test_local_rules_do_not_read_exact_paths(monkeypatch, rule):
    x, y, _, theta, decoder = fixture()
    def forbidden(*args):
        raise AssertionError("oracle access")
    monkeypatch.setattr(ex.model, "exact_path", forbidden)
    assert np.isfinite(ex.gradients(theta, decoder, x, y, np.var(y), [rule])).all()


def test_external_target_does_not_use_teacher_or_forward(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("external target generated from conductance teacher")
    monkeypatch.setattr(ex.model, "teacher", forbidden)
    monkeypatch.setattr(ex.model, "forward", forbidden)
    for task in ex.TASKS[1:]:
        x, y, _ = ex.dataset(191, "train", 128, task)
        assert np.isfinite(x).all() and np.all(np.abs(y) <= 1)


def test_continuous_context_and_relative_resistance_gate():
    x, _, c, theta, _ = fixture()
    assert np.all((c > 0) & (c < 1))
    state = ex.model.forward(theta, x)
    q = ex.delivery(state, x, ["resistance_gate"])
    g = state["conductance"][0]
    expected = (1 + g[18:22].reshape(2, 2).sum(1)) / state["denominator"][0, :, 4:6]
    np.testing.assert_allclose(q[0, :, :4], np.repeat(expected, 2, axis=1))
    np.testing.assert_array_equal(q[0, :, 4:], 1)
    assert np.all((q > 0) & (q <= 1))


def test_splits_are_disjoint_and_reproducible():
    first = ex.dataset(191, "train", 64, ex.TASKS[1])
    repeat = ex.dataset(191, "train", 64, ex.TASKS[1])
    test = ex.dataset(191, "test", 64, ex.TASKS[1])
    for a, b in zip(first, repeat):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[0], test[0])
    assert set(first[2]) == {0, 1}


def test_conductance_circuit_matches_production_branch_dynamics():
    """Test the actual DendriNet branch balance, not another copied equation.

    This establishes equation-level agreement only: the experiment still
    trains an independent seven-compartment adapter, not a DendriNet layer.
    """
    import sys
    from types import SimpleNamespace
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics import forward_branch_dynamics

    def balance(excitation=None, inhibition=None, children=None, coupling=None):
        class Coupling:
            def __call__(self, value):
                return (value * coupling).sum(dim=-1)
            def sum_conductances(self):
                return coupling.sum(dim=-1)
        owner = SimpleNamespace(use_shunting=True, additive_mode="raw", _store_diagnostics=False,
                                input_excitatory=excitation is not None, input_recurrent=False,
                                input_branches=children is not None, input_inhibitory=inhibition is not None,
                                input_rec_inhibitory=False, branch_excitation=lambda _: excitation,
                                branch_inhibition=lambda _: inhibition, branches_to_output=Coupling(),
                                reactivation=torch.nn.Identity(), epsilon=0., training=False)
        return forward_branch_dynamics(owner, torch.ones(1),
                                       inhibitory_input=torch.ones(1), branch_input=children)

    x, y, _, theta, decoder = fixture()
    tt = torch.tensor(theta[0], dtype=torch.float64, requires_grad=True)
    xx = torch.tensor(x, dtype=torch.float64)
    g = tt.exp()
    voltages = []
    for k in range(4):
        voltages.append(balance(g[4*k]*xx[:, k] + g[4*k+1]*xx[:, k+4],
                                g[4*k+2]*xx[:, k] + g[4*k+3]*xx[:, k+4]))
    for k in range(2):
        voltages.append(balance(inhibition=g[16+k]*xx[:, 8+k],
                                children=torch.stack(voltages[2*k:2*k+2], dim=-1),
                                coupling=g[18+2*k:20+2*k]))
    voltages.append(balance(children=torch.stack(voltages[4:6], dim=-1), coupling=g[22:24]))
    np.testing.assert_allclose(torch.stack(voltages, dim=-1).detach().numpy(),
                               ex.model.forward(theta, x)["voltage"][0], rtol=1e-13, atol=1e-14)
    loss = ((decoder[0, 0]*voltages[-1] + decoder[0, 1] - torch.tensor(y))**2).mean() / (2*np.var(y))
    actual = torch.autograd.grad(loss, tt)[0].detach().numpy()
    expected = ex.gradients(theta, decoder, x, y, np.var(y), ["exact"])[0, :24]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
