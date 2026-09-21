"""Unlabeled gate calibration in the existing production parameterization.

Initialization is an intervention, not a new operator or extra learned parameter.
The calibration input prefix must be reported as processed unlabeled exposure.
"""
import hashlib

import torch

from dendritic_modeling.scaling.local_composition import production_population


POLICIES = ("preserve", "center_all", "scale_all", "center_scale_all", "center_scale_leaves", "center_scale_soma")


def initialize_gates(model, x, policy, minimum_std=1e-3):
    if policy not in POLICIES or minimum_std <= 0:
        raise ValueError("Invalid gate initialization policy")
    layers = list(production_population(model).branch_layers)
    rows = []
    with torch.no_grad():
        for index, layer in enumerate(layers):
            gate = layer.reactivation
            if gate.__class__.__name__ != "ParametricReLU":
                raise ValueError("This intervention requires production parametric ReLU gates")
            is_soma = index == len(layers) - 1
            selected = policy != "preserve"
            if policy == "center_scale_leaves": selected = not is_soma
            if policy == "center_scale_soma": selected = is_soma
            values = []
            hook = gate.register_forward_pre_hook(lambda module, args: values.append(args[0].detach().clone()))
            model(x)
            hook.remove()
            assert len(values) == 1
            voltage = values[0]
            if selected and policy.startswith("center"):
                gate.b.copy_(voltage.quantile(.5, dim=0))
            rectified = (voltage - gate.b).relu()
            std = rectified.std(dim=0, correction=0)
            if selected and "scale" in policy:
                gate.log_m.copy_(-std.clamp_min(minimum_std).log())

            def summary(value):
                return {"min": float(value.min()), "median": float(value.median()), "max": float(value.max())}

            rows.append({"layer": index, "soma": is_soma, "changed": selected,
                         "voltage_mean": summary(voltage.mean(0)),
                         "voltage_std": summary(voltage.std(0, correction=0)),
                         "threshold": summary(gate.b), "gain": summary(gate.log_m.exp()),
                         "active_fraction": summary((voltage > gate.b).float().mean(0)),
                         "output_std": summary(std * gate.log_m.exp()),
                         "std_floor_fraction": float((std < minimum_std).float().mean())})
    return {"policy": policy, "unlabeled_input_draws": len(x), "minimum_std": minimum_std,
            "input_sha256": hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "parameter_count": sum(p.numel() for p in model.parameters()), "layers": rows,
            "scope": "Only existing gate thresholds/gains change; weights, contacts, couplings and head are preserved"}
