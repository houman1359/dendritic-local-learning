"""Higher-dimensional production trees with explicit, paired contact routing.

Canonical node order: eight leaves, four pairs, two quadruples, then soma.
The flat production factory supplies initial parameter values for both graphs.
Only the child graph changes in a flat/binary pair; no label-based initialization.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.scaling.models import build_model, model_report


CONDITIONS = ("local_additive", "local_compositional", "global_mixed_compositional")
SCOPES = [list(range(i, i + size)) for size in (4, 8, 16) for i in range(0, 32, size)]


def tensor_hash(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@lru_cache(maxsize=32)
def teacher_parameters(seed):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(8, 4))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    q, r = np.linalg.qr(np.random.default_rng(seed + 104729).normal(size=(32, 32)))
    q = q * np.sign(np.diag(r))[None, :]
    return torch.from_numpy(u), torch.from_numpy(q)


def target(x, condition, target_seed):
    if condition not in CONDITIONS or x.shape[-1] != 32:
        raise ValueError("Unknown condition or input dimension")
    u, q = (v.to(x) for v in teacher_parameters(target_seed))
    if condition == "global_mixed_compositional":
        x = x @ q.T
    z = (x.reshape(-1, 8, 4) * u).sum(-1)
    t = (1 + torch.erf(z / math.sqrt(2))) / 2

    def phi(v):
        return v * torch.log1p(1 / (v + 1))

    value = phi(t)
    if condition == "local_additive":
        return value.mean(-1, keepdim=True)
    while value.shape[-1] > 1:
        value = phi(value.reshape(len(x), -1, 2).mean(-1))
    return value


@dataclass
class DevelopmentData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    mean: float
    std: float
    identity: dict

    def receipt(self):
        return {
            **self.identity,
            "test_materialized": False,
            "normalization": {
                "mean": self.mean,
                "std": self.std,
                "source": "TRAIN only",
            },
            "arrays": {
                k: {
                    "shape": list(getattr(self, k).shape),
                    "sha256": tensor_hash(getattr(self, k)),
                }
                for k in ("train_x", "train_y", "validation_x", "validation_y")
            },
        }


def make_data(
    condition, target_seed, data_seed=2026091701, train_size=4096, validation_size=4096
):
    """Independent split streams and nested TRAIN prefixes; no TEST API here."""
    splits = []
    for split, n in enumerate((train_size, validation_size)):
        rng = np.random.default_rng(
            np.random.SeedSequence([data_seed, target_seed, split])
        )
        x = torch.from_numpy(rng.standard_normal((n, 32)))
        splits.append((x, target(x, condition, target_seed)))
    (x, y), (v, vy) = splits
    mean, std = float(y.mean()), float(y.std(unbiased=False))
    if not math.isfinite(std) or std**2 <= 1e-12:
        raise ValueError("Declared target has insufficient finite TRAIN variance")
    return DevelopmentData(
        x,
        (y - mean) / std,
        v,
        (vy - mean) / std,
        mean,
        std,
        {"condition": condition, "target_seed": target_seed, "data_seed": data_seed},
    )


def load_frozen_data(arrays_path, receipt_path):
    """Load byte-bound development arrays, independent of host BLAS rounding."""
    receipt = json.loads(Path(receipt_path).read_text())
    if receipt["test_materialized"]:
        raise ValueError("Development data receipt contains TEST")
    with np.load(arrays_path, allow_pickle=False) as archive:
        names = ("train_x", "train_y", "validation_x", "validation_y")
        if set(archive.files) != set(names):
            raise ValueError("Unexpected frozen development arrays")
        tensors = [torch.from_numpy(archive[name].copy()) for name in names]
    if any(x.dtype != torch.float64 or not torch.isfinite(x).all() for x in tensors):
        raise ValueError("Frozen development data must be finite float64")
    if (
        any(x.ndim != 2 for x in tensors)
        or tensors[0].shape[1] != 32
        or tensors[2].shape[1] != 32
    ):
        raise ValueError("Frozen development input dimensions differ")
    if tensors[1].shape != (len(tensors[0]), 1) or tensors[3].shape != (
        len(tensors[2]),
        1,
    ):
        raise ValueError("Frozen development label dimensions differ")
    data = DevelopmentData(
        *tensors,
        receipt["normalization"]["mean"],
        receipt["normalization"]["std"],
        {k: receipt[k] for k in ("condition", "target_seed", "data_seed")},
    )
    if data.receipt() != receipt:
        raise ValueError("Frozen data disagree with their original byte hashes")
    return data


def catalog():
    rows = []
    for mechanism in ("signed", "shunting"):
        for morphology, factors in (("flat14", [14]), ("binary222", [2, 2, 2])):
            for implementation in ("production_dendrinet", "ordinary_unrolled"):
                for routing in ("matched_hierarchy", "degree_preserving_scrambled"):
                    rows.append(
                        {
                            "id": f"{implementation}_{mechanism}_{morphology}_{routing}",
                            "implementation": implementation,
                            "mechanism": mechanism,
                            "branch_factors": factors,
                            "routing": routing,
                            "contacts_e": 4 if mechanism == "signed" else 2,
                            "contacts_i": 0 if mechanism == "signed" else 2,
                        }
                    )
    for implementation in ("dense", "sparse"):
        for depth in (2, 4):
            rows.append(
                {
                    "id": f"{implementation}_depth{depth}_iid",
                    "implementation": implementation,
                    "network_depth": depth,
                }
            )
    return rows


def parameter_count(arm, width):
    kind = arm["implementation"]
    if kind in {"production_dendrinet", "ordinary_unrolled"}:
        return (14 * (arm["contacts_e"] + arm["contacts_i"]) + 14 + 30 + 1) * width + 1
    depth = arm["network_depth"]
    if kind == "dense":
        return (depth - 1) * width**2 + (65 + 3 * depth) * width + 1
    if kind == "sparse":
        return (7 * depth + 1) * width + 1
    raise ValueError("Unknown architecture")


def match_width(arm, budget, tolerance=0.02):
    low, high = (4 if arm["implementation"] == "sparse" else 1), 1
    high = max(low, high)
    while parameter_count(arm, high) < budget:
        high *= 2
    while high - low > 1:
        mid = (low + high) // 2
        if parameter_count(arm, mid) < budget:
            low = mid
        else:
            high = mid
    width = min({low, high}, key=lambda w: (abs(parameter_count(arm, w) - budget), w))
    if abs(parameter_count(arm, width) - budget) / budget > tolerance:
        raise ValueError("No feasible whole-model budget within tolerance")
    return width


def production_population(model):
    populations = [m for m in model.modules() if isinstance(m, DendriNet)]
    if len(populations) != 1:
        raise ValueError("Expected one actual production DendriNet")
    return populations[0]


def level_slices(factors):
    if factors == [14]:
        return [(0, 14), (14, 15)]
    if factors == [2, 2, 2]:
        return [(0, 8), (8, 12), (12, 14), (14, 15)]
    raise ValueError("Only the declared equal-compartment graphs are supported")


def degree_scramble(indices, seed):
    """Sequential double-edge swaps, rejecting repeated contacts within a row."""
    a = np.asarray(indices, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    proposed, accepted = 100 * a.size, 0
    # Draw in chunks to bound temporary memory while preserving RNG order.
    flat = a.reshape(-1)
    k = a.shape[1]
    for start in range(0, proposed, 100000):
        pairs = rng.integers(a.size, size=(min(100000, proposed - start), 2))
        for i, j in pairs:
            ri, rj = i // k, j // k
            vi, vj = flat[i], flat[j]
            if ri == rj or vi == vj or vj in a[ri] or vi in a[rj]:
                continue
            flat[i], flat[j] = vj, vi
            accepted += 1
    if np.array_equal(a, indices):
        raise ValueError("Scrambling left routing unchanged")
    assert np.array_equal(
        np.bincount(a.ravel(), minlength=64),
        np.bincount(np.asarray(indices).ravel(), minlength=64),
    )
    return a, {"proposed_swaps": proposed, "accepted_swaps": accepted}


@lru_cache(maxsize=32)
def contact_support(width, contacts_e, contacts_i, support_seed, scrambled):
    rng = np.random.default_rng(support_seed)
    result, receipt = {}, {"seed": support_seed, "scopes": SCOPES, "banks": {}}
    for bank, k in (("e", contacts_e), ("i", contacts_i)):
        if not k:
            continue
        a = np.empty((width, 14, k), dtype=np.int64)
        for owner in range(width):
            for node, scope in enumerate(SCOPES):
                a[owner, node] = rng.choice(
                    scope + [x + 32 for x in scope], k, replace=False
                )
        base = a.copy()
        swaps = {}
        if scrambled:
            a, swaps = degree_scramble(
                a.reshape(-1, k), support_seed + (17 if bank == "e" else 29)
            )
            a = a.reshape(width, 14, k)
        result[bank] = torch.from_numpy(a)
        receipt["banks"][bank] = {
            **swaps,
            "sha256": tensor_hash(result[bank]),
            "matched_sha256": tensor_hash(torch.from_numpy(base)),
            "column_degrees": np.bincount(a.ravel(), minlength=64).tolist(),
            "contacts": int(a.size),
            "changed_slot_fraction": float(np.mean(a != base)),
        }
    receipt["index_bytes"] = sum(t.numel() * t.element_size() for t in result.values())
    return result, receipt


def _model_spec(arm, width, seed, support_seed, backend="eager"):
    structured = arm["implementation"] in {"production_dendrinet", "ordinary_unrolled"}
    return {
        "family": "dendritic_" + arm["mechanism"]
        if structured
        else arm["implementation"],
        "input_dim": 32,
        "output_dim": 1,
        "width": width,
        "network_depth": 1 if structured else arm["network_depth"],
        "branch_factors": arm.get("branch_factors", []),
        "contacts_e": arm.get("contacts_e", 4),
        "contacts_i": arm.get("contacts_i", 0),
        "activation": "param_relu",
        "activation_init": {"gain": 1.0, "threshold": 0.0},
        "input_encoding": "signed_split",
        "seed": seed,
        "topology_seed": support_seed,
        "projection_backend": backend,
    }


def _canonical_parameters(flat):
    pop = production_population(flat)
    leaf, root = pop.branch_layers
    w = flat.readout.weight.shape[1]
    p = {
        "e": leaf.branch_excitation.pre_w.reshape(w, 14, -1),
        "coupling": root.branches_to_output.log_weight.reshape(w, 14),
        "log_m": torch.cat(
            (leaf.reactivation.log_m.reshape(w, 14), root.reactivation.log_m[:, None]),
            1,
        ),
        "threshold": torch.cat(
            (leaf.reactivation.b.reshape(w, 14), root.reactivation.b[:, None]), 1
        ),
    }
    if leaf.branch_inhibition is not None:
        p["i"] = leaf.branch_inhibition.pre_w.reshape(w, 14, -1)
    return {k: v.detach().clone() for k, v in p.items()}


def _assign_canonical(model, arm, parameters, supports):
    pop = production_population(model)
    slices = level_slices(arm["branch_factors"])
    with torch.no_grad():
        for index, (layer, (lo, hi)) in enumerate(zip(pop.branch_layers, slices)):
            for bank, attr in (("e", "branch_excitation"), ("i", "branch_inhibition")):
                proj = getattr(layer, attr)
                if proj is not None:
                    proj.pre_w.copy_(parameters[bank][:, lo:hi].reshape_as(proj.pre_w))
                    proj.connection_indices.copy_(
                        supports[bank][:, lo:hi].reshape_as(proj.connection_indices)
                    )
            layer.reactivation.log_m.copy_(parameters["log_m"][:, lo:hi].reshape(-1))
            layer.reactivation.b.copy_(parameters["threshold"][:, lo:hi].reshape(-1))
            if index:
                prev_lo, prev_hi = slices[index - 1]
                layer.branches_to_output.log_weight.copy_(
                    parameters["coupling"][:, prev_lo:prev_hi].reshape_as(
                        layer.branches_to_output.log_weight
                    )
                )


class UnrolledBranch(nn.Module):
    """Independent ordinary tensor operations, with a bijection to production."""

    def __init__(self, branch):
        super().__init__()
        self.use_shunting, self.epsilon = branch.use_shunting, branch.epsilon
        for bank, attr in (("e", "branch_excitation"), ("i", "branch_inhibition")):
            proj = getattr(branch, attr)
            if proj is not None:
                self.register_parameter(bank, nn.Parameter(proj.pre_w.detach().clone()))
                self.register_buffer(
                    bank + "_indices", proj.connection_indices.detach().clone()
                )
        child = getattr(branch, "branches_to_output", None)
        if child is not None:
            self.coupling = nn.Parameter(child.log_weight.detach().clone())
        self.log_m = nn.Parameter(branch.reactivation.log_m.detach().clone())
        self.b = nn.Parameter(branch.reactivation.b.detach().clone())

    def transform(self, value):
        return F.softplus(value) if self.use_shunting else value

    def forward(self, x, previous):
        numerator, denominator = 0, 1
        if hasattr(self, "e"):
            e = (x[:, self.e_indices.long()] * self.transform(self.e)).sum(-1)
            numerator, denominator = numerator + e, denominator + e
        if hasattr(self, "coupling"):
            g = self.transform(self.coupling)
            current = (previous.reshape(len(x), *g.shape) * g).sum(-1)
            numerator, denominator = numerator + current, denominator + g.sum(-1)
        if hasattr(self, "i"):
            i = (x[:, self.i_indices.long()] * self.transform(self.i)).sum(-1)
            if self.use_shunting:
                denominator = denominator + i
            else:
                numerator = numerator - i
        v = numerator / (denominator + self.epsilon) if self.use_shunting else numerator
        return self.log_m.exp() * F.relu(v - self.b)


class UnrolledTree(nn.Module):
    def __init__(self, production):
        super().__init__()
        self.hidden = nn.ModuleList(
            [UnrolledBranch(b) for b in production_population(production).branch_layers]
        )
        self.readout = deepcopy(production.readout)

    def forward(self, x):
        x = torch.cat((x.relu(), (-x).relu()), -1)
        previous = None
        for branch in self.hidden:
            previous = branch(x, previous)
        return self.readout(previous)


def parameter_pairs(production, ordinary):
    pairs = []
    for b, u in zip(production_population(production).branch_layers, ordinary.hidden):
        for attr, name in (("branch_excitation", "e"), ("branch_inhibition", "i")):
            if getattr(b, attr) is not None:
                pairs.append((getattr(b, attr).pre_w, getattr(u, name)))
        if getattr(b, "branches_to_output", None) is not None:
            pairs.append((b.branches_to_output.log_weight, u.coupling))
        pairs.extend([(b.reactivation.log_m, u.log_m), (b.reactivation.b, u.b)])
    pairs.extend(
        [
            (production.readout.weight, ordinary.readout.weight),
            (production.readout.bias, ordinary.readout.bias),
        ]
    )
    assert sum(a.numel() for a, _ in pairs) == sum(
        p.numel() for p in production.parameters()
    )
    assert sum(b.numel() for _, b in pairs) == sum(
        p.numel() for p in ordinary.parameters()
    )
    return pairs


def construct(arm, budget, seed, *, width=None, dtype=torch.float32):
    width = match_width(arm, budget) if width is None else width
    support_seed = seed + 100000
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        if arm["implementation"] in {"dense", "sparse"}:
            model = build_model(_model_spec(arm, width, seed, support_seed))
            routing = {"seed": support_seed, "kind": arm["implementation"]}
        else:
            anchor = {**arm, "branch_factors": [14]}
            flat = build_model(_model_spec(anchor, width, seed, support_seed))
            canonical = _canonical_parameters(flat)
            model = (
                flat
                if arm["branch_factors"] == [14]
                else build_model(_model_spec(arm, width, seed, support_seed))
            )
            supports, routing = contact_support(
                width,
                arm["contacts_e"],
                arm["contacts_i"],
                support_seed,
                arm["routing"] == "degree_preserving_scrambled",
            )
            _assign_canonical(model, arm, canonical, supports)
            with torch.no_grad():
                model.readout.load_state_dict(flat.readout.state_dict())
            if arm["implementation"] == "ordinary_unrolled":
                model = UnrolledTree(model)
    finally:
        torch.set_default_dtype(previous_dtype)
    report = (
        model_report(model)
        if arm["implementation"] != "ordinary_unrolled"
        else {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }
    )
    assert (
        report["total_parameters"]
        == report["trainable_parameters"]
        == parameter_count(arm, width)
    )
    return model, {
        "arm": deepcopy(arm),
        "width": width,
        "requested_budget": budget,
        "actual_parameters": report["total_parameters"],
        "model_seed": seed,
        "support": deepcopy(routing),
        "model_report": report,
        "initialization": "production flat14 anchor copied by canonical node; readout preserved; no labels",
    }
