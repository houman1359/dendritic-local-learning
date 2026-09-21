"""Conventional centered-tanh and ReLU MLPs, with complete parameter counts."""
import torch
from torch import nn


CONTROL_FAMILIES = ("dense_centered_tanh", "dense_relu", "dense_shallow_tanh", "dense_shallow_relu")


class DenseCircuit(nn.Module):
    def __init__(self, d, width, family):
        super().__init__()
        activation = nn.Tanh if family.endswith("tanh") else nn.ReLU
        layers = [nn.Linear(d, width), activation()]
        if "shallow" not in family: layers += [nn.Linear(width, width), activation()]
        self.hidden = nn.Sequential(*layers)
        self.readout = nn.Linear(width, 1)
        for layer in self.hidden:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu" if family.endswith("relu") else "linear")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.readout(self.hidden(x))


def construct(family, ceiling, *, d=64, s=8, model_seed=751, support_seed=100751,
              width=None, dtype=torch.float32):
    if family not in CONTROL_FAMILIES: raise ValueError(family)
    count = (lambda w: (d + 2) * w + 1) if "shallow" in family else (lambda w: w * w + (d + 3) * w + 1)
    if width is None:
        lo, hi = 0, 1
        while count(hi) <= ceiling: lo, hi = hi, 2 * hi
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if count(mid) <= ceiling: lo = mid
            else: hi = mid
        width = lo
    if width < 1: raise ValueError("Budget below one hidden unit")
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model_seed)
            model = DenseCircuit(d, width, family)
    finally: torch.set_default_dtype(previous)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == count(width)
    return model, {"family": family, "d": d, "raw_fan_in": d, "width": width,
                   "actual_parameters": actual, "budget_ceiling": ceiling, "budget_fraction": actual / ceiling,
                   "model_seed": model_seed, "support_seed": None, "raw_contact_sites": None,
                   "raw_supports": None, "pooling": "one_dense_hidden_layer" if "shallow" in family else "two_dense_hidden_layers",
                   "parameter_shapes": {n: list(p.shape) for n, p in model.named_parameters()},
                   "initialization": "Hidden normal weights with fan-in variance1 for tanh,2 for ReLU; zero biases; ordinary head initialization",
                   "scope": "Conventional centered tanh or ReLU, without extra production gate gains or thresholds"}
