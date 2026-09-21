"""Production DendriNet interventions; no replacement of its shunting forward.

The current-inhibition arm is explicitly an intervention on proximal outputs.
Local updates detach inter-compartment edges and never evaluate exact paths.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import DendriNet

FORWARDS = ("shunt", "tonic", "current")
RULES = ("exact", "broadcast", "resistance", "swapped", "uniform_rms")


class CreditNet(nn.Module):
    def __init__(self, seed=0, somata=16, streams=4, cue_contacts=True, forward_mode="shunt"):
        super().__init__()
        if forward_mode not in FORWARDS:
            raise ValueError(forward_mode)
        torch.manual_seed(seed)
        self.somata, self.streams = somata, streams
        self.leaves = 2 * streams
        self.cue_contacts = cue_contacts
        self.forward_mode = forward_mode
        dim = 2 * self.leaves + streams
        self.core = DendriNet(n_soma=somata, branch_factors=[streams, 2],
                             excitatory_input_dim=dim, excitatory_synapses_per_branch=dim,
                             inhibitory_input_dim=dim, inhibitory_synapses_per_branch=dim,
                             reactivate=False, somatic_synapses=False, use_shunting=True,
                             weight_transform="exp", use_noise=False, topk_noise_level=0.,
                             topk_strategy="none", blocklinear_strategy="none")
        self.readout = nn.Linear(somata, 1)
        self.register_buffer("current_anchor", torch.zeros(somata * streams))
        self.register_buffer("tonic_activity", torch.full((streams,), 4. * (1 - 1 / streams)))
        self.local = False
        self.outputs = []
        self.last_cue = None
        self.current_calibrated = False
        leaf, proximal, soma = self.core.branch_layers
        # Fixed, explicitly grouped contacts: no adaptive TopK selection.
        for layer in self.core.branch_layers:
            layer.epsilon = 0.
            layer._store_diagnostics = True
        for synapse in (leaf.branch_excitation, leaf.branch_inhibition):
            mask = torch.zeros_like(synapse.pre_w, dtype=torch.bool)
            for row in range(somata * self.leaves):
                feature = row % self.leaves
                mask[row, feature] = True
                mask[row, feature + self.leaves] = True
            if synapse is leaf.branch_excitation and cue_contacts:
                mask[:, 2 * self.leaves:] = True
            synapse.connection_mask = mask
            synapse.K = dim
        proximal.input_excitatory = False
        proximal.branch_excitation = None
        mask = torch.zeros_like(proximal.branch_inhibition.pre_w, dtype=torch.bool)
        for row in range(somata * streams):
            mask[row, 2 * self.leaves + row % streams] = True
        proximal.branch_inhibition.connection_mask = mask
        proximal.branch_inhibition.K = dim
        with torch.no_grad():
            for layer in self.core.branch_layers:
                for name in ("branch_excitation", "branch_inhibition"):
                    synapse = getattr(layer, name, None)
                    if synapse is not None:
                        synapse.pre_w.normal_(0., .4)
                if layer.input_branches:
                    layer.branches_to_output.log_weight.normal_(0., .25)
            proximal.branch_inhibition.pre_w.add_(np.log(3.))
            self.readout.weight.normal_(0., 2. / np.sqrt(somata))
            self.readout.bias.zero_()
        for i, layer in enumerate(self.core.branch_layers):
            layer.register_forward_pre_hook(self._pre_hook(i))
            layer.register_forward_hook(self._post_hook(i))

    def _pre_hook(self, index):
        def hook(layer, args):
            args = list(args)
            if index == 1 and self.forward_mode == "tonic":
                inh = args[1].clone()
                inh[:, 2 * self.leaves:] = self.tonic_activity
                args[1] = inh
            if self.local and index > 0:
                args[2] = args[2].detach()
            return tuple(args)
        return hook

    def _post_hook(self, index):
        def hook(layer, args, output):
            if index == 1 and self.forward_mode == "current":
                if not self.current_calibrated:
                    raise RuntimeError("Current control requires a training-only anchor")
                # Same proximal conductance parameters and mean inhibitory load;
                # context varies injected current, not the conductance denominator.
                child_current = layer.branches_to_output(args[2])
                inh = layer.branch_inhibition(args[1])
                tonic_input = args[1].clone()
                tonic_input[:, 2 * self.leaves:] = self.tonic_activity
                tonic = layer.branch_inhibition(tonic_input)
                base = 1 + layer.branches_to_output.sum_conductances()
                output = (child_current - (inh - tonic) * self.current_anchor) / (base + tonic)
            self.outputs.append(output)
            return output
        return hook

    def forward(self, excitation, inhibition):
        self.outputs = []
        self.last_cue = inhibition
        soma = self.core(excitation, inhibition)
        return self.readout(soma).squeeze(-1)

    @torch.no_grad()
    def calibrate(self, excitation, inhibition):
        original = self.forward_mode
        self.forward_mode = "tonic"
        self(excitation, inhibition)
        self.current_anchor.copy_(self.outputs[1].mean(0))
        self.current_calibrated = True
        self.forward_mode = original

    @torch.no_grad()
    def distal_gate(self, rule):
        proximal = self.core.branch_layers[1]
        base = 1 + proximal.branches_to_output.sum_conductances()
        inhibition = proximal.branch_inhibition(self.last_cue)
        gate = (base / (base + inhibition)).reshape(-1, self.somata, self.streams)
        if rule == "swapped":
            gate = gate.roll(1, dims=-1)
        elif rule == "uniform_rms":
            gate = gate.square().mean(-1, keepdim=True).sqrt().expand_as(gate)
        elif rule != "resistance":
            raise ValueError(rule)
        return gate.reshape(-1, self.somata * self.streams).repeat_interleave(2, dim=-1)

    def gradients(self, excitation, inhibition, target, variance, rule):
        """Compute exact gradients or local eligibility × delivered soma error.

        Readout and soma use their exact local gradients in every arm. The
        source soma error is an assumption, not an endogenous teaching signal.
        """
        if rule not in RULES:
            raise ValueError(rule)
        self.zero_grad(set_to_none=True)
        self.local = rule != "exact"
        prediction = self(excitation, inhibition)
        loss = .5 * (prediction - target).square().mean() / variance
        objective = loss
        if self.local:
            error = torch.autograd.grad(loss, self.outputs[-1], retain_graph=True)[0].detach()
            for index, output in enumerate(self.outputs[:-1]):
                delivered = error.repeat_interleave(output.shape[1] // self.somata, dim=-1)
                if index == 0 and rule not in {"broadcast"}:
                    delivered = delivered * self.distal_gate(rule)
                objective = objective + (output * delivered).sum()
        objective.backward()
        self.local = False
        return float(loss.detach()) * 2

    def flat_gradient(self):
        return torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).flatten()
                          for p in self.parameters()])

    def active_parameters(self):
        masked = {}
        for layer in self.core.branch_layers:
            for name in ("branch_excitation", "branch_inhibition"):
                synapse = getattr(layer, name, None)
                if synapse is not None:
                    masked[id(synapse.pre_w)] = int(synapse.connection_mask.sum())
        return sum(masked.get(id(p), p.numel()) for p in self.parameters())

    @torch.no_grad()
    def load_figure5(self, theta):
        """Map all 24 published conductances onto genuine production modules."""
        if (self.somata, self.streams, self.cue_contacts) != (1, 2, False):
            raise ValueError("Figure 5 mapping needs one [2,2] neuron without cue bypass")
        t = torch.as_tensor(theta, dtype=self.readout.weight.dtype)
        leaf, proximal, soma = self.core.branch_layers
        for j in range(4):
            leaf.branch_excitation.pre_w[j, j] = t[4*j]
            leaf.branch_excitation.pre_w[j, j+4] = t[4*j+1]
            leaf.branch_inhibition.pre_w[j, j] = t[4*j+2]
            leaf.branch_inhibition.pre_w[j, j+4] = t[4*j+3]
        proximal.branch_inhibition.pre_w[0, 8] = t[16]
        proximal.branch_inhibition.pre_w[1, 9] = t[17]
        proximal.branches_to_output.log_weight.copy_(t[18:22].reshape(2, 2))
        soma.branches_to_output.log_weight.copy_(t[22:24].reshape(1, 2))
        self.readout.weight.fill_(1.)
        self.readout.bias.zero_()


def dataset(seed, split, size, streams=4, task="selection", severity=1., dose=4.):
    """Externally specified targets; contexts remain available to all forwards.

    Two independent features per stream. A common positive cue input reaches
    every terminal in all factorial arms; input supports never depend on y.
    """
    offsets = dict(train=1, validation=2, test=3, diagnostic=4, ood=5)
    rng = np.random.default_rng(np.random.SeedSequence([seed, offsets[split]]))
    z = rng.uniform(-2, 2, (size, streams, 2))
    if task == "selection":
        cue = np.eye(streams)[rng.integers(streams, size=size)]
        z *= 1 + (severity - 1) * (1 - cue[:, :, None])
    elif task == "mixture":
        cue = rng.dirichlet(np.ones(streams), size)
    else:
        raise ValueError(task)
    features = np.tanh(z)
    # A target-independent alternating tuning convention, including pairwise
    # interactions within each stream; not generated by a conductance teacher.
    evidence = (.5 * features.sum(-1) + .25 * features.prod(-1))
    evidence *= np.where(np.arange(streams) % 2, -1., 1.)
    y = (cue * evidence).sum(-1)
    flat = z.reshape(size, -1)
    sensory = np.column_stack((np.exp(flat), np.exp(-flat)))
    excitation = np.column_stack((sensory, cue))
    inhibition = np.column_stack((sensory, dose * (1 - cue)))
    return tuple(torch.tensor(v, dtype=torch.float64) for v in (excitation, inhibition, y))
