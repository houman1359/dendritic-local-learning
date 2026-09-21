"""64-raw-coordinate scaling adapters with counted contacts and soma controls."""
import hashlib
import math

import numpy as np
import torch
from torch import nn

from dendritic_modeling.scaling.models import build_model, model_report
from dendritic_modeling.scaling import interaction_models as primitive
from dendritic_modeling.scaling.local_composition import _canonical_parameters, _assign_canonical
from dendritic_modeling.scaling.order_spectrum import sample_raw_supports, sign_split_contacts


CORE = ("flat_shunting", "flat_signed", "local_tanh", "local_glu", "local_rational", "dense_tanh", "sparse_point2")


def parameter_count(family, width, s=8, d=64):
    if family in {"flat_shunting", "binary_shunting"}:
        return (56 * s + 45) * width + 1
    if family in {"flat_signed", "binary_signed"}:
        return (28 * s + 45) * width + 1
    if family in {"local_tanh", "local_glu", "local_rational"}:
        return primitive.count_bank(family, width, s)
    if family == "dense_tanh":
        return width ** 2 + (d + 7) * width + 1
    if family == "sparse_point2":
        return (14 * (s + 2) + 2) * width + 1
    raise ValueError(family)


def width_under_ceiling(family, ceiling, s, d):
    if ceiling < parameter_count(family, 1, s, d):
        raise ValueError("Budget below minimum architecture")
    lo, hi = 1, 2
    while parameter_count(family, hi, s, d) <= ceiling:
        lo, hi = hi, 2 * hi
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if parameter_count(family, mid, s, d) <= ceiling:
            lo = mid
        else:
            hi = mid
    return lo


class PointHidden(nn.Module):
    def __init__(self, raw, width):
        super().__init__()
        self.local = primitive.LocalFeatures(raw.reshape(-1, raw.shape[-1]), "local_tanh")
        self.coupling = nn.Parameter(torch.randn(width, 14) / math.sqrt(14))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        branch = self.local(x).reshape(len(x), -1, 14)
        return ((branch * self.coupling).sum(-1) + self.bias).tanh()


class SparsePoint2(nn.Module):
    def __init__(self, raw, width):
        super().__init__()
        self.encoding = nn.Identity()
        self.hidden = PointHidden(raw, width)
        self.readout = nn.Linear(width, 1)

    def forward(self, x):
        return self.readout(self.hidden(x))


def construct(family, ceiling, *, d=64, s=8, model_seed=701, support_seed=100701,
              width=None, dtype=torch.float32):
    width = width_under_ceiling(family, ceiling, s, d) if width is None else width
    structured = family.startswith(("flat_", "binary_"))
    grouped = structured or family == "sparse_point2"
    sites = 14 * width if grouped else width
    raw = sample_raw_supports(d, s, sites, support_seed) if family != "dense_tanh" else None
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model_seed)
            if family in {"local_tanh", "local_glu", "local_rational"}:
                model, _ = primitive.bank(family, d=d, s=s, units=width, model_seed=model_seed,
                                          support_seed=support_seed, raw_supports=raw, dtype=dtype)
            elif family == "sparse_point2":
                model = SparsePoint2(raw, width)
            elif family == "dense_tanh":
                model = build_model({"family": "dense", "input_dim": d, "output_dim": 1,
                                     "width": width, "network_depth": 2, "activation": "param_tanh",
                                     "activation_init": {"gain": 1., "threshold": 0.},
                                     "input_encoding": "raw", "seed": model_seed})
            elif structured:
                shunt = family.endswith("shunting")
                factors = [2, 2, 2] if family.startswith("binary") else [14]
                spec = {"family": "dendritic_shunting" if shunt else "dendritic_signed",
                        "input_dim": d, "output_dim": 1, "width": width, "network_depth": 1,
                        "branch_factors": [14], "contacts_e": 2 * s, "contacts_i": 2 * s if shunt else 0,
                        "activation": "param_relu", "activation_init": {"gain": 1., "threshold": 0.},
                        "input_encoding": "signed_split", "seed": model_seed, "topology_seed": support_seed}
                flat = build_model(spec)
                model = flat if factors == [14] else build_model({**spec, "branch_factors": factors})
                canonical = _canonical_parameters(flat)
                encoded = torch.as_tensor(sign_split_contacts(raw, d).reshape(width, 14, 2 * s))
                supports = {"e": encoded}
                if shunt:
                    supports["i"] = encoded.clone()
                _assign_canonical(model, {"branch_factors": factors}, canonical, supports)
                model.readout.load_state_dict(flat.readout.state_dict())
            else:
                raise ValueError(family)
    finally:
        torch.set_default_dtype(old)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == parameter_count(family, width, s, d), (family, actual)
    inv = {"family": family, "d": d, "raw_fan_in": s if raw is not None else d,
           "budget_ceiling": ceiling, "actual_parameters": actual, "width": width,
           "budget_fraction": actual / ceiling, "model_seed": model_seed, "support_seed": support_seed,
           "raw_contact_sites": sites if raw is not None else None,
           "raw_supports": raw.tolist() if raw is not None else None,
           "raw_support_sha256": hashlib.sha256(raw.tobytes()).hexdigest() if raw is not None else None,
           "raw_union_per_soma": [len(set(r.reshape(-1))) for r in raw.reshape(width, 14, s)] if grouped else None,
           "pooling": "nonlinear_soma" if grouped else "affine" if raw is not None else "two_dense_hidden_layers",
           "node_layout": "8+4+2 raw-contact nodes and one soma" if family.startswith("binary") else "14 parallel raw-contact nodes and one soma" if grouped else None,
           "parameter_shapes": {n: list(p.shape) for n, p in model.named_parameters()},
           "production_report": model_report(model) if structured else None,
           "scope": "Use actual P; equal ceilings need not yield identical P or contact counts"}
    return model, inv
