"""Independent analytic checks added after, not inside, the frozen run snapshot."""
import itertools
import numpy as np
import torch

from rescue import RescueNet, diagnostics


def test_degree_identity_on_the_complete_binary_input_space():
    x = np.asarray(list(itertools.product([-1., 1.], repeat=8)))
    for groups in [[(0, 1), (2, 3), (4, 5), (6, 7)], [(0, 1, 2, 3), (4, 5, 6, 7)]]:
        y = sum(.5 * x[:, group].prod(1) for group in groups)
        gradient = np.zeros_like(x)
        for group in groups:
            for i in group:
                gradient[:, i] += .5 * x[:, [j for j in group if j != i]].prod(1)
        np.testing.assert_allclose(np.mean(np.sum(gradient**2, 1)), len(groups[0]) * y.var())
        np.testing.assert_allclose(gradient.T @ gradient / len(x), np.eye(8) / 4)


def test_cue_only_terminal_target_interaction_cancels_under_symmetry():
    # Exact Cartesian quadrature, not a finite random sample. Nonselected
    # features are zero; both selected features are independent and symmetric.
    latent, cues = [], []
    for context in range(4):
        for a, b in itertools.product([-1.3, -.7, .7, 1.3], repeat=2):
            z = np.zeros((4, 2)); z[context] = [a, b]
            latent.append(z.reshape(-1)); cues.append(np.eye(4)[context])
    z = torch.tensor(np.array(latent), dtype=torch.float64)
    cue = torch.tensor(np.array(cues), dtype=torch.float64)
    sensory = torch.cat([z.exp(), (-z).exp()], -1)
    x = torch.cat([sensory, cue], -1)
    inhibition = torch.cat([sensory, 4*(1-cue)], -1)
    t = z.reshape(-1, 4, 2).tanh()
    target = (cue * (.5*t.sum(-1) + .25*t.prod(-1)) * x.new_tensor([1, -1, 1, -1])).sum(-1)
    rows = {r['rule']: r for r in diagnostics(RescueNet(7).double(), (x, inhibition, target), target.var())}
    assert rows['broadcast']['interaction_norm'] < 1e-12
    assert rows['resistance']['interaction_norm'] < 1e-12
    assert rows['exact']['interaction_norm'] > 1e-6
    assert rows['derivative']['interaction_norm'] > 1e-6
