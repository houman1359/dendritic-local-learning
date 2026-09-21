"""Development-only regression through actual production DendriNet modules.

No TEST generator is implemented here. The fixed plan determines targets,
initialization, optimizer settings and exact parameter counts. Preparation
freezes source and configs; execution refuses a different imported source.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import socket
import time
import traceback

import torch
from torch import nn
from torch.nn import functional as F

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from .models import build_model, model_report

FAMILIES = (
    "production_shunt",
    "ordinary_rational",
    "biased_relu_shallow",
    "biased_relu_depth2",
    "biased_tanh_shallow",
)
TASKS = ("continuum_narrow", "continuum_broad", "relu_spline_mismatch")
RECIPES = ("joint", "train_readout_ls_then_joint")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def tensor_hash(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def clone(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def geometry(family, budget):
    if (
        family not in FAMILIES
        or isinstance(budget, bool)
        or not isinstance(budget, int)
        or budget < 7
    ):
        raise ValueError("Unsupported family or parameter budget")
    if family == "biased_relu_depth2":
        width = math.isqrt(budget + 3) - 2
        actual = width * width + 4 * width + 1
    else:
        per_unit = 2 if family in FAMILIES[:2] else 3
        width = (budget - 1) // per_unit
        actual = per_unit * width + 1
    if actual != budget or width < 1:
        raise ValueError("Budget does not exactly match this family")
    return width


def target(x, task):
    if task == "relu_spline_mismatch":
        return (x - 0.2).abs() - 1.5 * (x - 0.5).abs() + 0.75 * (x - 0.8).abs()
    if task not in TASKS:
        raise ValueError("Unknown fixed target")
    a, b = (1.0, 2.0) if task == "continuum_narrow" else (0.01, 4.0)
    return x * torch.log1p((b - a) / (x + a)) / (b - a)


@dataclass(frozen=True)
class TrainingData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor

    def receipt(self):
        return {
            "test_materialized": False,
            **{
                name: {
                    "shape": list(getattr(self, name).shape),
                    "sha256": tensor_hash(getattr(self, name)),
                }
                for name in ("train_x", "train_y", "validation_x", "validation_y")
            },
        }


def make_data(task, train_points=2049, validation_points=2053):
    if min(train_points, validation_points) < 2:
        raise ValueError("At least two points per split are required")
    x = ((torch.arange(train_points, dtype=torch.float64) + 1 / 3) / train_points)[
        :, None
    ]
    v = (
        (torch.arange(validation_points, dtype=torch.float64) + 2 / 3)
        / validation_points
    )[:, None]
    if torch.unique(torch.cat((x, v))).numel() != train_points + validation_points:
        raise ValueError("Training and validation grids overlap")
    return TrainingData(x, target(x, task), v, target(v, task))


@contextmanager
def float64_construction():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_default_dtype(previous)


class DivisiveLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.pre_weight = nn.Parameter(torch.empty(width, 1))
        self.epsilon = 1e-8

    def forward(self, x):
        excitation = x * F.softplus(self.pre_weight).T
        return excitation / ((1 + excitation) + self.epsilon)


class OrdinaryModel(nn.Module):
    def __init__(self, family, width):
        super().__init__()
        self.encoding = nn.Identity()
        if family == "ordinary_rational":
            self.hidden = nn.Sequential(DivisiveLayer(width))
        else:
            activation = nn.Tanh if family == "biased_tanh_shallow" else nn.ReLU
            layers = [nn.Sequential(nn.Linear(1, width), activation())]
            if family == "biased_relu_depth2":
                layers.append(nn.Sequential(nn.Linear(width, width), nn.ReLU()))
            self.hidden = nn.Sequential(*layers)
        self.readout = nn.Linear(width, 1)

    def forward(self, x):
        return self.readout(self.hidden(x))


def features(model, x):
    return model.hidden(model.encoding(x))


def construct(family, budget, seed):
    width = geometry(family, budget)
    with float64_construction():
        if family == "production_shunt":
            model = build_model(
                {
                    "family": "dendritic_shunting",
                    "input_dim": 1,
                    "output_dim": 1,
                    "width": width,
                    "network_depth": 1,
                    "branch_factors": [],
                    "contacts_e": 1,
                    "contacts_i": 0,
                    "input_encoding": "nonnegative",
                    "activation": "relu",
                    "projection_backend": "eager",
                    "seed": seed,
                }
            )
        else:
            model = OrdinaryModel(family, width)
    generator = torch.Generator().manual_seed(seed)
    gains = torch.exp(
        math.log(0.25)
        + torch.rand(width, generator=generator, dtype=torch.float64) * math.log(400)
    )
    domain_examples = 0
    with torch.no_grad():
        if family in FAMILIES[:2]:
            if family == "production_shunt":
                populations = [m for m in model.modules() if isinstance(m, DendriNet)]
                assert len(populations) == 1 and len(populations[0].branch_layers) == 1
                branch = populations[0].branch_layers[0]
                projection = branch.branch_excitation
                assert (
                    projection.weight_transform == "softplus" and branch.epsilon == 1e-8
                )
                assert torch.count_nonzero(projection.connection_indices) == 0
                weight = projection.pre_w
            else:
                weight = model.hidden[0].pre_weight
            # Production uses torch softplus's linear shortcut above20.
            preweight = torch.where(
                gains > 20, gains, gains + torch.log(-torch.expm1(-gains))
            )
            weight.copy_(preweight[:, None])
            torch.testing.assert_close(
                F.softplus(weight).flatten(), gains, rtol=2e-15, atol=2e-15
            )
        else:
            knot = (
                torch.arange(width, dtype=torch.float64)
                + torch.rand(width, generator=generator, dtype=torch.float64)
            ) / width
            knot[0] = 0
            model.hidden[0][0].weight.copy_(gains[:, None])
            model.hidden[0][0].bias.copy_(-gains * knot)
            if family == "biased_relu_depth2":
                second = model.hidden[1][0]
                second.weight.copy_(
                    torch.randn(width, width, generator=generator, dtype=torch.float64)
                    / math.sqrt(width)
                )
                domain = torch.linspace(0, 1, 65, dtype=torch.float64)[:, None]
                pre = model.hidden[0](domain) @ second.weight.T
                second.bias.copy_(-0.5 * (pre.amin(0) + pre.amax(0)))
                domain_examples = 65
        model.readout.weight.copy_(
            torch.randn(1, width, generator=generator, dtype=torch.float64)
            / math.sqrt(width)
        )
        model.readout.bias.zero_()
    report = model_report(model)
    assert report["total_parameters"] == report["trainable_parameters"] == budget
    assert all(
        p.dtype == torch.float64 and p.device.type == "cpu" for p in model.parameters()
    )
    model.initialization_receipt = {
        "gain_span": [0.25, 100],
        "label_information_used": False,
        "target_quadrature_used": False,
        "domain_initialization_examples": domain_examples,
        "input_gain_sha256": tensor_hash(gains),
        "seed": seed,
    }
    return model


def mapped_parameters(model, family):
    if family == "production_shunt":
        pop = next(m for m in model.modules() if isinstance(m, DendriNet))
        return [
            pop.branch_layers[0].branch_excitation.pre_w,
            model.readout.weight,
            model.readout.bias,
        ]
    if family == "ordinary_rational":
        return [model.hidden[0].pre_weight, model.readout.weight, model.readout.bias]
    return list(model.parameters())


def least_squares_readout(model, data, rcond):
    before = {
        k: v.clone()
        for k, v in model.state_dict().items()
        if not k.startswith("readout.")
    }
    with torch.no_grad():
        design = torch.cat(
            (
                features(model, data.train_x),
                torch.ones(len(data.train_x), 1, dtype=torch.float64),
            ),
            dim=1,
        )
        solved = torch.linalg.lstsq(design, data.train_y, rcond=rcond, driver="gelsd")
        coefficient = solved.solution
        model.readout.weight.copy_(coefficient[:-1].T)
        model.readout.bias.copy_(coefficient[-1])
        if not torch.isfinite(coefficient).all():
            raise FloatingPointError("Nonfinite least-squares coefficients")
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    return {
        "solver": "torch.linalg.lstsq/gelsd",
        "rcond": rcond,
        "rank": int(solved.rank),
        "columns": design.shape[1],
        "singular_values": solved.singular_values.tolist(),
        "readout_max_abs": float(coefficient.abs().max()),
        "readout_l2": float(coefficient.norm()),
        "examples": len(data.train_x),
        "validation_used": False,
        "train_mse": float((design @ coefficient - data.train_y).square().mean()),
        "hidden_state_unchanged": True,
    }


@dataclass(frozen=True)
class FitSpec:
    family: str
    budget: int
    task: str
    seed: int
    recipe: str = "joint"
    lr: float = 1.0
    steps: int = 40

    def validate(self):
        geometry(self.family, self.budget)
        if (
            self.task not in TASKS
            or self.recipe not in RECIPES
            or self.steps < 1
            or self.lr not in (0.3, 1.0)
        ):
            raise ValueError("Unsupported development fit")


def fit(spec: FitSpec, data: TrainingData, settings: dict, output: Path):
    spec.validate()
    output.mkdir(parents=True, exist_ok=False)
    model = construct(spec.family, spec.budget, spec.seed)
    initial = clone(model)
    report = model_report(model)
    trace, exposures = (
        [],
        {
            "optimization_train_examples": 0,
            "measurement_train_examples": 0,
            "measurement_validation_examples": 0,
            "readout_initialization_train_examples": 0,
        },
    )
    receipt = {
        "status": "running",
        "spec": asdict(spec),
        "settings": settings,
        "data": data.receipt(),
        "model_report": report,
        "initialization": model.initialization_receipt,
        "runtime": {
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "threads": torch.get_num_threads(),
            "dtype": "float64",
            "device": "cpu",
        },
    }
    dump(output / "receipt.json", receipt)
    prepared, optimizer = None, None
    started = time.perf_counter()
    closure_calls = 0

    def record(step):
        prediction = model(data.train_x)
        loss = (prediction - data.train_y).square().mean()
        grads = torch.autograd.grad(loss, tuple(model.parameters()))
        with torch.no_grad():
            validation = (model(data.validation_x) - data.validation_y).square().mean()
        if (
            not torch.isfinite(loss)
            or not torch.isfinite(validation)
            or not all(torch.isfinite(g).all() for g in grads)
        ):
            raise FloatingPointError("Nonfinite training/validation diagnostic")
        exposures["measurement_train_examples"] += len(data.train_x)
        exposures["measurement_validation_examples"] += len(data.validation_x)
        trace.append(
            {
                "step": step,
                "closure_calls": closure_calls,
                "train_mse": float(loss.detach()),
                "validation_mse": float(validation),
                "unscaled_gradient_l2": math.sqrt(
                    sum(float(g.square().sum()) for g in grads)
                ),
                "zero_gradient_parameters": sum(int((g == 0).sum()) for g in grads),
            }
        )

    try:
        record(-1)
        if spec.recipe == "train_readout_ls_then_joint":
            receipt["least_squares"] = least_squares_readout(
                model, data, settings["ls_rcond"]
            )
            exposures["readout_initialization_train_examples"] += len(data.train_x)
        prepared = clone(model)
        record(0)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=spec.lr,
            max_iter=settings["max_iter_per_outer_step"],
            max_eval=settings["max_eval"],
            history_size=settings["history_size"],
            tolerance_grad=settings["tolerance_grad"],
            tolerance_change=settings["tolerance_change"],
            line_search_fn=settings["line_search"],
        )

        def closure():
            nonlocal closure_calls
            closure_calls += 1
            exposures["optimization_train_examples"] += len(data.train_x)
            optimizer.zero_grad(set_to_none=True)
            loss = (
                settings["objective_scale"]
                * (model(data.train_x) - data.train_y).square().mean()
            )
            loss.backward()
            if not torch.isfinite(loss) or any(
                p.grad is None or not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise FloatingPointError("Missing/nonfinite parameter gradient or loss")
            return loss

        steps_to_record = set(settings["trace_outer_steps"]) | {spec.steps}
        for step in range(1, spec.steps + 1):
            optimizer.step(closure)
            if step in steps_to_record:
                record(step)
        final_report = model_report(model)
        assert final_report["parameter_shapes"] == report["parameter_shapes"]
        assert final_report["topology_sha256"] == report["topology_sha256"]
        receipt.update(
            status="completed",
            terminal_model_report=final_report,
            terminal_train_mse=trace[-1]["train_mse"],
            terminal_validation_mse=trace[-1]["validation_mse"],
            hidden_parameter_movement=math.sqrt(
                sum(
                    float((p.detach() - prepared[name]).square().sum())
                    for name, p in model.named_parameters()
                    if not name.startswith("readout.")
                )
            ),
        )
    except Exception as exc:
        receipt.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    receipt.update(
        elapsed_seconds=time.perf_counter() - started,
        closure_calls=closure_calls,
        exposures=exposures,
        trace=trace,
    )
    torch.save(
        {
            "initial": initial,
            "prepared": prepared,
            "terminal": clone(model),
            "optimizer": optimizer.state_dict() if optimizer else None,
        },
        output / "states.pt",
    )
    receipt["state_sha256"] = sha(output / "states.pt")
    dump(output / "receipt.json", receipt)
    return receipt


def profile_specs():
    return [
        FitSpec(f, p, t, 7103)
        for f in FAMILIES
        for t in TASKS
        for p in ((13, 397) if f == "biased_relu_depth2" else (7, 397))
    ]


def development_specs(plan):
    cells = {(f, p) for f in FAMILIES for p in plan["parameter_budgets"]}
    cells |= set(
        itertools.product(
            plan["primary_mechanism_curve"]["families"],
            plan["primary_mechanism_curve"]["parameter_budgets"],
        )
    )
    return [
        FitSpec(f, p, t, seed, recipe, lr, 400)
        for f, p in sorted(cells)
        for t, seed, recipe, lr in itertools.product(
            TASKS, plan["development_seeds"], RECIPES, [0.3, 1.0]
        )
    ]


def validate_plan(plan):
    """Refuse silent divergence between the executable adapter and its plan."""
    expected = {
        "device": "cpu",
        "dtype": "float64",
        "torch_threads": 1,
        "optimizer": "LBFGS",
        "outer_steps": 400,
        "max_iter_per_outer_step": 1,
        "learning_rates": [0.3, 1.0],
        "recipes": list(RECIPES),
    }
    for key, value in expected.items():
        if plan["training"].get(key) != value:
            raise ValueError(f"Unsupported plan training field: {key}")
    if [t["id"] for t in plan["tasks"]] != list(TASKS):
        raise ValueError("Unexpected task grid")
    if [t.get("conductance_range") for t in plan["tasks"][:2]] != [[1, 2], [0.01, 4]]:
        raise ValueError("Target intervals differ from the implemented functions")
    if plan["initialization_contract"]["effective_input_gain_span"] != [0.25, 100]:
        raise ValueError("Unsupported initialization span")
    if plan["development_seeds"] != [7103, 7109]:
        raise ValueError("Development seeds changed")


def prepare(plan_path, destination, stage):
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if stage not in {"profile", "development"}:
        raise ValueError("Only profile and development are implemented")
    specs = profile_specs() if stage == "profile" else development_specs(plan)
    assert len(specs) == (30 if stage == "profile" else 1296)
    destination.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parents[1]
    frozen = destination / "source/src/dendritic_modeling"
    shutil.copytree(
        package, frozen, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copy2(plan_path, destination / "plan.json")
    source_files = [
        {"path": str(p.relative_to(destination)), "sha256": sha(p)}
        for p in sorted(frozen.rglob("*"))
        if p.is_file()
    ]
    manifest = {
        "schema": "production_continuum_development_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "plan_sha256": sha(destination / "plan.json"),
        "test_materialized": False,
        "source_files": source_files,
        "tasks": [{"index": i, "spec": asdict(spec)} for i, spec in enumerate(specs)],
    }
    dump(destination / "manifest.json", manifest)
    return manifest


def run(campaign, shard_index=0, shards=1):
    campaign = campaign.resolve()
    manifest = json.loads((campaign / "manifest.json").read_text())
    if not 0 <= shard_index < shards:
        raise ValueError("Invalid shard")
    if (
        Path(__file__).resolve()
        != campaign / "source/src/dendritic_modeling/scaling/production_continuum.py"
    ):
        raise ValueError("Execute using the campaign's frozen PYTHONPATH")
    assert sha(campaign / "plan.json") == manifest["plan_sha256"]
    for row in manifest["source_files"]:
        assert sha(campaign / row["path"]) == row["sha256"]
    plan = json.loads((campaign / "plan.json").read_text())
    validate_plan(plan)
    torch.set_num_threads(plan["training"]["torch_threads"])
    results = []
    for task_row in manifest["tasks"]:
        index = task_row["index"]
        if index % shards != shard_index:
            continue
        spec = FitSpec(**task_row["spec"])
        data = make_data(
            spec.task, plan["data"]["train_points"], plan["data"]["validation_points"]
        )
        result = fit(spec, data, plan["training"], campaign / f"runs/{index:04d}")
        results.append(
            {
                "index": index,
                "status": result["status"],
                "seconds": result["elapsed_seconds"],
                "validation_mse": result.get("terminal_validation_mse"),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    dump(
        campaign / f"shard_{shard_index:02d}.json",
        {"manifest_sha256": sha(campaign / "manifest.json"), "results": results},
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--stage", choices=("profile", "development"), required=True)
    r = commands.add_parser("run")
    r.add_argument("--campaign", type=Path, required=True)
    r.add_argument("--shard-index", type=int, default=0)
    r.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        m = prepare(args.plan, args.output_dir, args.stage)
        print(
            json.dumps(
                {
                    "tasks": len(m["tasks"]),
                    "source_files": len(m["source_files"]),
                    "stage": m["stage"],
                }
            )
        )
    else:
        results = run(args.campaign, args.shard_index, args.shards)
        if any(r["status"] != "completed" for r in results):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
