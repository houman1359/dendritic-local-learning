"""Counted shared-projection versus branch-specific local architectures."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__:
    from .smooth_block import (
        FitConfig,
        SmoothBlockModel,
        SmoothBlockTask,
        TaskConfig,
        evaluate,
        fit_intercept,
        readout_least_squares,
    )
else:
    from smooth_block import (
        FitConfig,
        SmoothBlockModel,
        SmoothBlockTask,
        TaskConfig,
        evaluate,
        fit_intercept,
        readout_least_squares,
    )


class ProjectionModel(SmoothBlockModel):
    def __init__(self, capacities, family, architecture="independent", **kwargs):
        if architecture not in ("independent", "shared"):
            raise ValueError("Unknown projection architecture")
        super().__init__(capacities, family, **kwargs)
        self.architecture = architecture
        if architecture == "shared":
            directions = []
            offset = 0
            for capacity in capacities:
                vector = self.projection[offset].detach()
                directions.append(vector / vector.norm())
                offset += capacity
            directions = torch.stack(directions)
            gains = (self.projection.detach() * directions[self.group]).sum(1)
            self.projection = nn.Parameter(directions)
            self.gain = nn.Parameter(gains)
        self.initialization_receipt = {
            **self.initialization_receipt,
            "architecture": architecture,
            "gain_scope": "Shared raw projection and signed freely trainable branch gains; initial effective affine vectors match the independent model at the same branch count.",
        }
        expected = (
            (self.inputs_per_block + 2) * sum(capacities) + len(capacities) + 1
            if architecture == "independent"
            else 3 * sum(capacities) + (self.inputs_per_block + 1) * len(capacities) + 1
        )
        assert self.parameter_count == expected

    def features(self, x):
        if self.architecture == "shared":
            projected = (x * self.projection).sum(2)
            pre = projected[:, self.group] * self.gain + self.threshold
        else:
            pre = (x[:, self.group] * self.projection).sum(2) + self.threshold
        if self.family == "tanh":
            return torch.tanh(pre)
        excitation = torch.relu(pre)
        return excitation if self.family == "relu" else excitation / (1 + excitation)

    def feature_parameters(self):
        return [
            self.projection,
            self.threshold,
            *([self.gain] if self.architecture == "shared" else []),
        ]

    def effective_projection(self):
        return (
            self.projection
            if self.architecture == "independent"
            else self.projection[self.group] * self.gain[:, None]
        )


class ProjectionTask:
    def __init__(self, kind="single_index", seed=2026102001):
        if kind not in ("single_index", "two_index"):
            raise ValueError("Unknown target")
        self.kind = kind
        self.base = SmoothBlockTask(
            TaskConfig(condition="mixed_range", teacher_seed=seed)
        )
        rng = np.random.default_rng(np.random.SeedSequence([seed, 71]))
        self._rotations = np.stack(
            [np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(4)]
        )

    def sample_inputs(self, seed, n):
        if self.kind == "single_index":
            return self.base.sample_inputs(seed, n)
        if n < 1:
            raise ValueError("Positive n required")
        latent = np.random.default_rng(seed).uniform(-0.5, 0.5, size=(n, 4, 3))
        return torch.from_numpy(np.einsum("nkj,kij->nki", latent, self._rotations))

    def components(self, x):
        if self.kind == "single_index":
            return self.base.components(x)
        latent = np.einsum("nki,kij->nkj", np.asarray(x), self._rotations)
        if np.max(np.abs(latent)) > 0.5 + 1e-10:
            raise ValueError("Query outside the rotated product cube")
        values = self.base._raw_mixtures(
            latent[:, :, 0] + 0.5
        ) + self.base._raw_mixtures(latent[:, :, 1] + 0.5)
        return torch.from_numpy(
            (values - 2 * self.base._means) / (2 * math.sqrt(2) * self.base._scales)
        )

    def evaluate(self, x):
        return self.components(x).sum(1)

    def sample(self, seed, n):
        x = self.sample_inputs(seed, n)
        return x, self.evaluate(x)

    def specification(self):
        return {
            **self.base.specification(),
            "kind": self.kind,
            "rotations": self._rotations.tolist(),
            "two_index_scope": "For the negative control only, X=Q*Uniform[-.5,.5]^3 and each normalized block sums the same curved response in two independent latent coordinates. Input mean0/covarianceI/12; private axes never reach learner.",
        }


def fit(model, x, y, config=None):
    config = config or FitConfig()
    start = time.monotonic()
    reference = {
        name: value.detach().clone() for name, value in model.named_parameters()
    }
    stages = [{"stage": "initial", **evaluate(model, x, y)}]
    closures = []
    body_names = [
        name
        for name, _ in model.named_parameters()
        if name in ("projection", "threshold", "gain")
    ]

    def gradient():
        model.zero_grad(set_to_none=True)
        loss = (model(x) - y).square().mean()
        loss.backward()
        norm = math.sqrt(
            sum(
                float(p.grad.square().sum())
                for p in model.parameters()
                if p.grad is not None
            )
        )
        model.zero_grad(set_to_none=True)
        return float(loss.detach()), norm

    _, initial_gradient = gradient()

    def optimize(stage, parameters, steps):
        previous = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if name in body_names
        }
        optimizer = torch.optim.LBFGS(
            parameters,
            lr=config.learning_rate,
            max_iter=steps,
            max_eval=steps * 3,
            history_size=config.history_size,
            tolerance_grad=config.tolerance_grad,
            tolerance_change=config.tolerance_change,
            line_search_fn="strong_wolfe",
        )

        def closure():
            model.zero_grad(set_to_none=True)
            loss = (model(x) - y).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite loss; preserve failed state")
            (loss * config.objective_scale).backward()
            closures.append({"stage": stage, "train_mse": float(loss.detach())})
            return loss * config.objective_scale

        optimizer.step(closure)
        state = optimizer.state[optimizer.param_groups[0]["params"][0]]
        stages.append(
            {
                "stage": stage,
                **evaluate(model, x, y),
                "iterations": int(state.get("n_iter", 0)),
                "closure_calls": int(state.get("func_evals", 0)),
                "last_step_size": float(state.get("t", 0)),
                "last_direction_max": (
                    float(state["d"].abs().max()) if "d" in state else 0.0
                ),
                "body_movement_l2": math.sqrt(
                    sum(
                        float((p.detach() - previous[name]).square().sum())
                        for name, p in model.named_parameters()
                        if name in body_names
                    )
                ),
            }
        )

    if config.recipe != "joint":
        stages.append(readout_least_squares(model, x, y, config.ls_rcond))
    if config.recipe == "alternating_ls_then_joint":
        for index in range(config.alternating_rounds):
            optimize(
                f"body_lbfgs_{index}",
                model.feature_parameters(),
                config.alternating_body_steps,
            )
            stages.append(readout_least_squares(model, x, y, config.ls_rcond))
    optimize("joint_lbfgs", list(model.parameters()), config.lbfgs_steps)
    loss, terminal_gradient = gradient()
    before = float(model.bias.detach())
    deployed = fit_intercept(model, x, y)
    movement = {
        name: float((value.detach() - reference[name]).norm())
        for name, value in model.named_parameters()
    }
    return {
        "config": asdict(config),
        "initialization": model.initialization_receipt,
        "parameters": model.parameter_count,
        "train_rows": len(x),
        "stages": stages,
        "closure_history": closures,
        "initial_train_mse": stages[0]["mse"],
        "terminal_train_mse": loss,
        "deployed_train_metrics": deployed,
        "initial_unscaled_gradient_l2": initial_gradient,
        "terminal_unscaled_gradient_l2": terminal_gradient,
        "intercept_delta": float(model.bias.detach()) - before,
        "parameter_movement_l2": movement,
        "body_movement_l2": math.sqrt(sum(movement[name] ** 2 for name in body_names)),
        "elapsed_seconds": time.monotonic() - start,
        "scope": "Same declared TRAIN-only optimizer budgets for both architectures; shared feature body includes its branch gains. All slots counted despite factorization redundancy.",
    }


def assemble(local_models, anchor_y=0):
    family, architecture = local_models[0].family, local_models[0].architecture
    if any(
        len(m.capacities) != 1 or m.family != family or m.architecture != architecture
        for m in local_models
    ):
        raise ValueError("Homogeneous one-block library required")
    model = ProjectionModel(
        [m.capacities[0] for m in local_models],
        family,
        architecture,
        initialization="random",
    )
    with torch.no_grad():
        model.projection.copy_(torch.cat([m.projection for m in local_models]))
        model.threshold.copy_(torch.cat([m.threshold for m in local_models]))
        model.internal_readout.copy_(
            torch.cat([m.internal_readout for m in local_models])
            * math.sqrt(len(local_models))
        )
        model.soma_readout.copy_(torch.cat([m.soma_readout for m in local_models]))
        model.bias.copy_(torch.stack([m.bias for m in local_models]).sum() + anchor_y)
        if architecture == "shared":
            model.gain.copy_(torch.cat([m.gain for m in local_models]))
    model.initialization_receipt = {
        "method": "Exact assembly; no subsequent fit",
        "local": [m.initialization_receipt for m in local_models],
    }
    return model


def save_state(model, path):
    path = Path(path)
    metadata = {
        "family": model.family,
        "architecture": model.architecture,
        "capacities": model.capacities,
        "initialization": model.initialization_receipt,
    }
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{name: p.detach().numpy() for name, p in model.state_dict().items()},
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
    return {
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameters": model.parameter_count,
    }


def load_state(path):
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata_json"]))
        model = ProjectionModel(
            metadata["capacities"],
            metadata["family"],
            metadata["architecture"],
            initialization="random",
        )
        model.load_state_dict(
            {name: torch.from_numpy(arrays[name].copy()) for name in model.state_dict()}
        )
        model.initialization_receipt = metadata["initialization"]
    return model
