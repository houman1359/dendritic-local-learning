"""Compact ordinary ReLU controls and a Boolean production-signed mapping."""
import hashlib
import math

import torch
from torch import nn

from dendritic_modeling.scaling.order_spectrum import sample_raw_supports
from dendritic_modeling.scaling.local_composition import production_population


CONTROL_FAMILIES = ("local_relu", "sparse_relu2")


class LocalReLU(nn.Module):
    def __init__(self, raw):
        super().__init__()
        self.register_buffer("indices", torch.as_tensor(raw, dtype=torch.long))
        self.weight = nn.Parameter(torch.randn(*raw.shape) / math.sqrt(raw.shape[-1]))
        self.bias = nn.Parameter(torch.zeros(len(raw)))

    def forward(self, x):
        return ((x[:, self.indices] * self.weight).sum(-1) + self.bias).relu()


class GroupedReLU(nn.Module):
    def __init__(self, raw, width):
        super().__init__()
        self.local = LocalReLU(raw)
        self.coupling = nn.Parameter(torch.randn(width, 14) / math.sqrt(14))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        branch = self.local(x).reshape(len(x), -1, 14)
        return ((branch * self.coupling).sum(-1) + self.bias).relu()


class ReLUCircuit(nn.Module):
    def __init__(self, raw, width, grouped):
        super().__init__()
        self.hidden = GroupedReLU(raw, width) if grouped else LocalReLU(raw)
        self.readout = nn.Linear(width, 1)

    def forward(self, x):
        return self.readout(self.hidden(x))


def construct(family, ceiling, *, d=64, s=8, model_seed=751, support_seed=100751,
              width=None, raw_supports=None, dtype=torch.float32):
    if family not in CONTROL_FAMILIES: raise ValueError(family)
    grouped = family == "sparse_relu2"
    cost = 14 * (s + 2) + 2 if grouped else s + 2
    width = (ceiling - 1) // cost if width is None else width
    if width < 1: raise ValueError("Budget below one unit")
    sites = width * 14 if grouped else width
    raw = sample_raw_supports(d, s, sites, support_seed) if raw_supports is None else raw_supports
    assert raw.shape == (sites, s)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model_seed)
            model = ReLUCircuit(raw, width, grouped)
    finally: torch.set_default_dtype(previous)
    count = sum(p.numel() for p in model.parameters())
    assert count == cost * width + 1
    inventory = {"family": family, "d": d, "raw_fan_in": s, "width": width,
                 "actual_parameters": count, "budget_ceiling": ceiling, "budget_fraction": count / ceiling,
                 "raw_contact_sites": sites, "raw_supports": raw.tolist(),
                 "raw_support_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
                 "model_seed": model_seed, "support_seed": support_seed,
                 "pooling": "nonlinear_soma" if grouped else "affine",
                 "parameter_shapes": {n: list(p.shape) for n, p in model.named_parameters()},
                 "scope": "Ordinary ReLU control; compact Boolean representation of signed production is a separate mapping"}
    return model, inventory


def compact_signed_flat(production, raw_supports, d):
    """Exact on Rademacher inputs; not a global real-input equivalence."""
    pop = production_population(production)
    assert len(pop.branch_layers) == 2
    branch, soma = pop.branch_layers
    assert not branch.use_shunting and not soma.use_shunting
    assert branch.branch_inhibition is None and soma.branch_excitation is None
    assert soma.branch_inhibition is None
    parameter = next(production.parameters())
    width, s = production.readout.in_features, raw_supports.shape[1]
    compact, inv = construct("sparse_relu2", 1, d=d, s=s, width=width,
                             raw_supports=raw_supports, dtype=parameter.dtype)
    compact.to(parameter.device)
    with torch.no_grad():
        weights = branch.branch_excitation.pre_w.reshape(-1, 2 * s)
        pos, neg = weights[:, :s], weights[:, s:]
        gain = branch.reactivation.log_m.exp()
        compact.hidden.local.weight.copy_(gain[:, None] * (pos - neg) / 2)
        compact.hidden.local.bias.copy_(gain * ((pos + neg).sum(-1) / 2 - branch.reactivation.b))
        root_gain = soma.reactivation.log_m.exp()
        compact.hidden.coupling.copy_(root_gain[:, None] * soma.branches_to_output.log_weight)
        compact.hidden.bias.copy_(-root_gain * soma.reactivation.b)
        compact.readout.load_state_dict(production.readout.state_dict())
    return compact, inv
