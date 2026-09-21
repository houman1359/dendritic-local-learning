"""Untrained, CPU float64 witnesses for the production shunting comparison.

The continuum target is fixed as width grows. Numerical integrals check the
construction; the accompanying theory note supplies the population bounds.
This module neither changes the training factory nor launches training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.scaling.models import build_model, model_report


def continuum_target(x: torch.Tensor) -> torch.Tensor:
    """Integral of x/(z+x) against the uniform probability law on [1, 2]."""
    return x * torch.log1p(1 / (1 + x))


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def gaussian_rule(width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Positive shifted Gauss-Legendre rule with total mass one."""
    _positive_integer(width, "width")
    nodes, weights = np.polynomial.legendre.leggauss(width)
    return (
        torch.tensor(1.5 + nodes / 2, dtype=torch.float64),
        torch.tensor(weights / 2, dtype=torch.float64),
    )


def continuum_bounds(relu_width: int, shunt_width: int) -> dict[str, float]:
    _positive_integer(relu_width, "relu_width")
    _positive_integer(shunt_width, "shunt_width")
    return {
        "relu_mse_lower": (7 / 36) ** 2 / (720 * (relu_width + 1) ** 4),
        "relu_interpolation_mse_upper": 1 / (120 * relu_width**4),
        "shunt_constructive_mse_upper": 3.0 ** (-4 * shunt_width),
    }


def _spec(family: str, width: int, encoding: str) -> dict:
    if encoding not in {"nonnegative", "signed_split"}:
        raise ValueError("encoding must be nonnegative or signed_split")
    return {
        "family": family,
        "input_dim": 1,
        "output_dim": 1,
        "width": width,
        "network_depth": 1,
        "branch_factors": [],
        "contacts_e": 1,
        "contacts_i": 0,
        "input_encoding": encoding,
        "activation": "relu",
        "projection_backend": "eager",
        "seed": 0,
    }


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


def production_mixture(
    nodes: torch.Tensor,
    coefficients: torch.Tensor,
    encoding: str = "nonnegative",
) -> tuple[nn.Module, dict]:
    """Assign a quadrature witness to the actual factory, including support.

    Each cell's positive effective weight is (1+epsilon)/node. All registered
    parameters remain trainable; explicit assignments are an existence witness.
    Signed splitting uses fixed contact zero, the positive channel on [0, 1].
    """
    nodes = torch.as_tensor(nodes, dtype=torch.float64, device="cpu")
    coefficients = torch.as_tensor(coefficients, dtype=torch.float64, device="cpu")
    if (
        nodes.ndim != 1
        or nodes.numel() < 1
        or coefficients.shape != nodes.shape
        or not torch.isfinite(nodes).all()
        or not torch.isfinite(coefficients).all()
        or not (nodes > 0).all()
        or not (coefficients > 0).all()
    ):
        raise ValueError("nodes and coefficients must be equal positive finite vectors")
    width = nodes.numel()
    model = build_model(_spec("dendritic_shunting", width, encoding)).double().eval()
    populations = [
        module for module in model.modules() if isinstance(module, DendriNet)
    ]
    if len(populations) != 1 or len(populations[0].branch_layers) != 1:
        raise RuntimeError("Witness requires precisely one flat production population")
    branch = populations[0].branch_layers[0]
    projection = branch.branch_excitation
    if (
        projection.pre_w.shape != (width, 1)
        or projection.weight_transform != "softplus"
    ):
        raise RuntimeError("Unexpected production excitation parameterization")
    epsilon = float(branch.epsilon)
    effective_weights = (1 + epsilon) / nodes
    with torch.no_grad():
        projection.pre_w.copy_(_inverse_softplus(effective_weights).reshape(width, 1))
        projection.connection_indices.zero_()
        model.readout.weight.copy_(coefficients.reshape(1, width))
        model.readout.bias.zero_()
    report = model_report(model)
    if report["total_parameters"] != 2 * width + 1:
        raise RuntimeError("Production inventory changed from 2W+1")
    return model, {
        "epsilon": epsilon,
        "nodes": nodes.tolist(),
        "coefficients": coefficients.tolist(),
        "assigned_effective_weights": effective_weights.tolist(),
        "observed_effective_weights": projection.weight().detach().flatten().tolist(),
        "connection_indices": projection.connection_indices.tolist(),
        "support_rule": "one fixed contact on encoded channel zero per cell",
        "all_parameters_trainable": all(p.requires_grad for p in model.parameters()),
        "model_report": report,
    }


class GenericDivisiveMixture(nn.Module):
    """Conventional arithmetic control with exactly W+W+1 trainable slots.

    This uses a standard softplus gain and division, with no DendriNet module.
    It has the same function class after absorbing the production leak in gains.
    """

    def __init__(self, nodes: torch.Tensor, coefficients: torch.Tensor):
        super().__init__()
        self.pre_weight = nn.Parameter(_inverse_softplus(1 / nodes).clone())
        self.readout_weight = nn.Parameter(coefficients.clone())
        self.readout_bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        excitation = x * F.softplus(self.pre_weight)
        return (excitation / (1 + excitation)) @ self.readout_weight[
            :, None
        ] + self.readout_bias


def relu_interpolant(width: int, encoding: str = "nonnegative") -> nn.Module:
    """Uniform H-interval interpolant, using H actual biased ReLU units."""
    _positive_integer(width, "width")
    model = build_model(_spec("dense", width, encoding)).double().eval()
    knots = torch.linspace(0, 1, width + 1, dtype=torch.float64)
    slopes = torch.diff(continuum_target(knots)) * width
    coefficients = torch.cat((slopes[:1], torch.diff(slopes)))
    with torch.no_grad():
        model.hidden[0][0].weight.zero_()
        model.hidden[0][0].weight[:, 0] = 1
        model.hidden[0][0].bias.copy_(-knots[:-1])
        model.readout.weight.copy_(coefficients.reshape(1, width))
        model.readout.bias.zero_()
    return model


def integration_rule(
    intervals: int, order: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gauss integration per affine interval, so no ReLU knot is missed."""
    _positive_integer(intervals, "intervals")
    _positive_integer(order, "order")
    x, weights = np.polynomial.legendre.leggauss(order)
    x = ((np.arange(intervals)[:, None] + (x + 1) / 2) / intervals).reshape(-1)
    weights = np.tile(weights / (2 * intervals), intervals)
    return torch.tensor(x, dtype=torch.float64)[:, None], torch.tensor(
        weights, dtype=torch.float64
    )


def _mse(model: nn.Module, intervals: int, order: int = 32) -> float:
    x, weights = integration_rule(intervals, order)
    with torch.no_grad():
        residual = (model(x) - continuum_target(x)).flatten()
    return float(weights @ residual.square())


def _state_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_witness(budgets: tuple[int, ...] = (7, 13, 19, 25, 37, 49, 97)) -> dict:
    """Construct both families at shared native-input budgets P=6k+1."""
    if not budgets or any(isinstance(p, bool) or p < 7 or (p - 1) % 6 for p in budgets):
        raise ValueError("budgets must have form 6k+1 with k>=1")
    rows, constructions, scalar_checks = [], [], []
    grid = torch.linspace(0, 1, 2049, dtype=torch.float64)[:, None]
    for encoding in ("nonnegative", "signed_split"):
        scalar, metadata = production_mixture(torch.ones(1), torch.ones(1), encoding)
        with torch.no_grad():
            residual = scalar(grid) - grid / (1 + grid)
        scalar_checks.append(
            {
                "encoding": encoding,
                "max_absolute_residual": float(residual.abs().max()),
                "parameters": metadata["model_report"]["total_parameters"],
                "epsilon": metadata["epsilon"],
                "effective_weight": metadata["observed_effective_weights"][0],
            }
        )
    for budget in budgets:
        shunt_width, relu_width = (budget - 1) // 2, (budget - 1) // 3
        nodes, coefficients = gaussian_rule(shunt_width)
        native, metadata = production_mixture(nodes, coefficients)
        split, split_metadata = production_mixture(nodes, coefficients, "signed_split")
        generic = GenericDivisiveMixture(nodes, coefficients)
        relu = relu_interpolant(relu_width)
        before = _state_hash(native)
        with torch.no_grad():
            prediction = native(grid)
            formula = (grid / (nodes + grid)) @ coefficients[:, None]
            generic_residual = float((prediction - generic(grid)).abs().max())
            formula_residual = float((prediction - formula).abs().max())
            split_residual = float((prediction - split(grid)).abs().max())
        row = {
            "budget": budget,
            "shunt_width": shunt_width,
            "relu_width": relu_width,
            "shunt_parameters": metadata["model_report"]["total_parameters"],
            "relu_parameters": model_report(relu)["total_parameters"],
            "generic_division_parameters": sum(p.numel() for p in generic.parameters()),
            "shunt_mse_gauss32": _mse(native, 1),
            "shunt_mse_gauss64": _mse(native, 1, 64),
            "relu_mse_gauss32_per_interval": _mse(relu, relu_width),
            "relu_mse_gauss64_per_interval": _mse(relu, relu_width, 64),
            "production_vs_formula_max_abs": formula_residual,
            "production_vs_generic_max_abs": generic_residual,
            "native_vs_signed_split_max_abs": split_residual,
            **continuum_bounds(relu_width, shunt_width),
        }
        row["certified_upper_below_lower"] = (
            row["shunt_constructive_mse_upper"] < row["relu_mse_lower"]
        )
        rows.append(row)
        constructions.append(
            {
                "budget": budget,
                "native": metadata,
                "signed_split": split_metadata,
                "state_sha256_before_evaluation": before,
                "state_sha256_after_evaluation": _state_hash(native),
            }
        )
    log2 = np.longdouble(2)
    log2 = np.log(log2)
    return {
        "schema": "production_shunting_scaling_witness_v1",
        "target": "F(x)=x*log((2+x)/(1+x)), fixed for every width",
        "distribution": "Uniform[0,1]",
        "training_steps": 0,
        "device": "cpu",
        "dtype": "float64",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scope": "Assigned constructive witnesses; no optimizer, learning exponent, or deep-ReLU lower bound.",
        "arithmetic_scope": "Real-arithmetic certificates are proved in the theory note. Numerical MSE uses quadrature; roundoff dominates sufficiently small errors.",
        "scalar_calibration": scalar_checks,
        "scalar_homogeneous_affine_mse_floor": float(
            36 * log2 - 28 * log2**2 - np.longdouble(23) / 2
        ),
        "rows": rows,
        "constructions": constructions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # Refuse to replace previous evidence; each invocation gets its own folder.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    receipt = run_witness()
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), root / "src/dendritic_modeling/scaling/models.py"]
    paths += sorted((root / "src/dendritic_modeling/networks").rglob("*.py"))
    receipt["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    (args.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    with (args.output_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(receipt["rows"][0]))
        writer.writeheader()
        writer.writerows(receipt["rows"])
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "rows": receipt["rows"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
