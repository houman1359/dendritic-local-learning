"""Actual DendriNet and conventional local units with raw-coordinate contacts.

The primitive bank has affine pooling, while flat/binary production trees have
learned nonlinear somas. All E/I banks use both sign channels of the same raw
support; their separate conductance weights are counted, not shared implicitly.
"""
from copy import deepcopy
import hashlib
import math

import numpy as np
import torch
from torch import nn

from dendritic_modeling.scaling.models import build_model, model_report
from dendritic_modeling.scaling.local_composition import production_population, UnrolledTree, parameter_pairs
from dendritic_modeling.scaling.order_spectrum import sample_raw_supports, sign_split_contacts


BANK_FAMILIES = ("production_shunting", "production_signed", "local_tanh", "local_glu", "local_rational")


class LocalFeatures(nn.Module):
    def __init__(self, raw_supports, kind):
        super().__init__()
        self.kind = kind
        raw = torch.as_tensor(raw_supports, dtype=torch.long)
        self.register_buffer("raw_supports", raw)
        units, s = raw.shape
        projections = 1 if kind == "local_tanh" else 2
        self.weight = nn.Parameter(torch.randn(units, projections, s) / math.sqrt(s))
        self.bias = nn.Parameter(torch.zeros(units, projections))

    def forward(self, x):
        z = (x[:, self.raw_supports][:, :, None, :] * self.weight[None]).sum(-1) + self.bias
        if self.kind == "local_tanh":
            return z[..., 0].tanh()
        if self.kind == "local_glu":
            return z[..., 0] * z[..., 1].sigmoid()
        if self.kind == "local_rational":
            return z[..., 0] / (1 + z[..., 1].square())
        if self.kind == "local_bilinear":
            return z[..., 0] * z[..., 1]
        raise ValueError(self.kind)


class LocalBank(nn.Module):
    def __init__(self, raw_supports, kind):
        super().__init__()
        self.hidden = LocalFeatures(raw_supports, kind)
        self.readout = nn.Linear(len(raw_supports), 1)

    def features(self, x):
        return self.hidden(x)

    def forward(self, x):
        return self.readout(self.features(x))


def features(model, x):
    if isinstance(model, LocalBank):
        return model.features(x)
    return model.hidden(model.encoding(x))


def count_bank(family, units, s):
    per_unit = {"production_shunting": 4 * s + 3, "production_signed": 2 * s + 3,
                "local_tanh": s + 2, "local_glu": 2 * s + 3,
                "local_rational": 2 * s + 3, "local_bilinear": 2 * s + 3}[family]
    return units * per_unit + 1


def bank(family, *, d=64, s=8, units=8, model_seed=701, support_seed=100701,
         raw_supports=None, dtype=torch.float32):
    if not (1 <= s <= d and units >= 1):
        raise ValueError("Invalid bank shape")
    if raw_supports is None:
        raw_supports = sample_raw_supports(d, s, units, support_seed)
    raw = np.asarray(raw_supports)
    if raw.shape != (units, s):
        raise ValueError("Support shape does not match bank")
    encoded = sign_split_contacts(raw, d)
    before = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model_seed)
            if family.startswith("production_"):
                shunt = family == "production_shunting"
                if family not in BANK_FAMILIES:
                    raise ValueError(family)
                model = build_model({"family": "dendritic_shunting" if shunt else "dendritic_signed",
                                     "input_dim": d, "output_dim": 1, "width": units,
                                     "network_depth": 1, "branch_factors": [],
                                     "contacts_e": 2 * s, "contacts_i": 2 * s if shunt else 0,
                                     "activation": "param_relu", "activation_init": {"gain": 1., "threshold": 0.},
                                     "input_encoding": "signed_split", "seed": model_seed,
                                     "topology_seed": support_seed, "projection_backend": "eager"})
                pop = production_population(model)
                assert len(pop.branch_layers) == 1
                branch = pop.branch_layers[0]
                with torch.no_grad():
                    branch.branch_excitation.connection_indices.copy_(torch.as_tensor(encoded))
                    if shunt:
                        branch.branch_inhibition.connection_indices.copy_(torch.as_tensor(encoded))
            elif family in (*BANK_FAMILIES, "local_bilinear"):
                model = LocalBank(raw, family)
            else:
                raise ValueError(family)
    finally:
        torch.set_default_dtype(before)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == count_bank(family, units, s)
    inv = {"family": family, "d": d, "raw_fan_in": s, "units": units,
           "actual_parameters": actual, "optimized_parameters": actual,
           "model_seed": model_seed, "support_seed": support_seed,
           "raw_supports": raw.tolist(), "raw_support_sha256": hashlib.sha256(raw.astype(np.int64).tobytes()).hexdigest(),
           "pooling": "affine", "teacher_supports_given": False,
           "parameter_shapes": {n: list(p.shape) for n, p in model.named_parameters()},
           "encoding": "signed_split" if family.startswith("production_") else "raw",
           "encoded_contacts_per_unit": (4 * s if family == "production_shunting" else 2 * s) if family.startswith("production_") else s,
           "production_report": model_report(model) if family.startswith("production_") else None}
    return model, inv


def population_readout_oracle(model, x, y, rcond=1e-12):
    """Best unconstrained head in float64 on frozen, measured feature values.

    This is an optimistic population OLS reference, not a float32 SGD run.
    The numerical head realization error and feature conditioning are reported.
    """
    with torch.no_grad():
        f = features(model, x).detach().cpu().double().numpy()
    target = y.detach().cpu().double().numpy().reshape(-1)
    design = np.column_stack([f, np.ones(len(f))])
    coefficient, _, rank, singular = np.linalg.lstsq(design, target, rcond=rcond)
    prediction = design @ coefficient
    model_copy = deepcopy(model)
    with torch.no_grad():
        model_copy.readout.weight.copy_(torch.as_tensor(coefficient[:-1], device=x.device, dtype=x.dtype)[None])
        model_copy.readout.bias.copy_(torch.as_tensor(coefficient[-1:], device=x.device, dtype=x.dtype))
        realized = model_copy(x).detach().cpu().double().numpy().reshape(-1)
    return {"population_mse": float(np.mean((prediction - target) ** 2)),
            "realized_head_mse": float(np.mean((realized - target) ** 2)),
            "max_head_realization_difference": float(np.max(abs(realized - prediction))),
            "rank": int(rank), "columns": design.shape[1], "rcond": rcond,
            "singular_values": singular.tolist(), "coefficient_norm": float(np.linalg.norm(coefficient)),
            "head_coefficients": coefficient.tolist(), "head_parameters": len(coefficient),
            "scope": "Full-cube frozen-feature OLS in float64; head realization separately checked"}


def bank_independent_check(model, x):
    """Production versus ordinary operations, including every parameter gradient."""
    ordinary = UnrolledTree(model)
    a, b = model(x), ordinary(x)
    pairs = parameter_pairs(model, ordinary)
    ga = torch.autograd.grad(a.square().mean(), [p for p, _ in pairs])
    gb = torch.autograd.grad(b.square().mean(), [p for _, p in pairs])
    difference = max([float((a - b).abs().max())]
                     + [float((u - v).abs().max()) for u, v in zip(ga, gb)])
    tolerance = 2e-10 if x.dtype == torch.float64 else 3e-5
    if difference > tolerance:
        raise AssertionError(f"Production/ordinary discrepancy: {difference}")
    return {"max_output_or_parameter_gradient_difference": difference,
            "parameters_checked": sum(p.numel() for p, _ in pairs), "tolerance": tolerance}
