"""Exact QR gauge changes between declared, fresh-history LBFGS stages.

This is periodic factor balancing, not a permanent orthogonal-manifold
constraint. The frozen centered/RMS ridge objective and stored slot inventory
are unchanged. A restart-only control executes exactly the same stage schedule.
"""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from . import rank_learning as base


def rebalance_basis(model):
    """Replace V=QR and a by Ra; all effective projections are invariant."""
    if model.architecture == "full":
        return {"applied": False, "reason": "Full projections have no basis gauge"}
    with torch.no_grad():
        old = model.effective_projection().clone()
        before = torch.linalg.svdvals(model.basis)
        q, r = torch.linalg.qr(model.basis, mode="reduced")
        if not bool(torch.isfinite(q).all() and torch.isfinite(r).all()):
            raise FloatingPointError("Nonfinite QR factorization")
        signs = torch.where(torch.diagonal(r, dim1=-2, dim2=-1) < 0, -1.0, 1.0)
        q = q * signs[:, None, :]
        r = signs[:, :, None] * r
        coefficients = torch.einsum(
            "mij,mj->mi", r[model.group], model.branch_coefficients
        )
        if not bool(torch.isfinite(coefficients).all()):
            raise FloatingPointError("Nonfinite transformed branch coefficients")
        model.basis.copy_(q)
        model.branch_coefficients.copy_(coefficients)
        difference = model.effective_projection() - old
        drift = float(difference.abs().max())
        if drift > 1e-10 * max(1.0, float(old.abs().max())):
            raise FloatingPointError("QR gauge did not preserve effective projections")
        gram = q.transpose(1, 2) @ q
        tolerance = (
            torch.finfo(before.dtype).eps
            * max(model.inputs_per_block, model.rank)
            * before[:, 0]
        )
        numerical_rank = (before > tolerance[:, None]).sum(1)
        return {
            "applied": True,
            "before_basis_singular_values": before.cpu().tolist(),
            "before_basis_numerical_rank": numerical_rank.cpu().tolist(),
            "basis_rank_diagnostic_tolerance": tolerance.cpu().tolist(),
            "rank_diagnostic_used_to_truncate": False,
            "before_basis_condition": [
                (
                    float(values[0] / values[-1])
                    if values[-1] > 0 and bool(torch.isfinite(values[0] / values[-1]))
                    else None
                )
                for values in before
            ],
            "after_basis_singular_values": torch.linalg.svdvals(q).cpu().tolist(),
            "orthogonality_error": float(
                (gram - torch.eye(model.rank, dtype=q.dtype, device=q.device))
                .abs()
                .max()
            ),
            "maximum_effective_projection_drift": drift,
            "stored_parameters": model.parameter_count,
            "scope": "Function-preserving factor re-expression; an invertible gauge map only when the old basis has full column rank. No clipping, truncation or penalty. Numerical ranks are diagnostics only; all stored basis/branch slots retained.",
        }


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def fit_restarted(
    model,
    x,
    y,
    config=None,
    *,
    stages=3,
    rebalance=False,
    output_dir=None,
    optimizer="reduced",
):
    """Fit with declared restarts and optional exact QR after each stage.

    ``config.steps`` is the TOTAL iteration allowance, split deterministically
    across stages. Every arm gets fresh LBFGS history and the frozen readout
    warm start at each stage. Pass a width wrapper's counted ``.core`` and its
    duplicated raw input array. Saved model states use the frozen base schema.
    """
    config = config or base.FitConfig()
    objective_options = {"projection_rms_floor": config.projection_rms_floor}
    if int(stages) != stages or not 1 <= stages <= config.steps:
        raise ValueError("Whole stage count between one and total steps required")
    if optimizer not in ("reduced", "joint"):
        raise ValueError("Unknown optimizer")
    x = torch.as_tensor(x, dtype=torch.float64, device=model.bias.device)
    y = torch.as_tensor(y, dtype=torch.float64, device=model.bias.device)
    output = None if output_dir is None else Path(output_dir)
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    initial_projection = model.effective_projection().detach().clone()
    stage_fits, boundaries = [], []
    initial_state = (
        None if output is None else base.save_state(model, output / "initial.npz")
    )
    try:
        for stage in range(stages):
            steps = config.steps // stages + int(stage < config.steps % stages)
            stage_dir = None if output is None else output / f"stage_{stage:03d}"
            fitted = base.fit(
                model,
                x,
                y,
                replace(config, steps=steps),
                optimizer=optimizer,
                output_dir=stage_dir,
            )
            stage_fits.append(fitted)
            with torch.no_grad():
                before_prediction = model(x).clone()
            before_objective = base.joint_objective(
                model, x, y, config.ridge, **objective_options
            )[0].detach()
            boundary = (
                rebalance_basis(model)
                if rebalance
                else {"applied": False, "reason": "Declared restart-only control"}
            )
            with torch.no_grad():
                prediction_drift = float((model(x) - before_prediction).abs().max())
            after_objective = base.joint_objective(
                model, x, y, config.ridge, **objective_options
            )[0].detach()
            objective_drift = float(abs(after_objective - before_objective))
            if prediction_drift > 1e-8 * max(
                1.0, float(before_prediction.abs().max())
            ) or objective_drift > 1e-10 * max(1.0, float(abs(before_objective))):
                raise FloatingPointError(
                    "Gauge boundary changed prediction or objective"
                )
            boundary.update(
                stage=stage,
                prediction_max_drift=prediction_drift,
                objective_absolute_drift=objective_drift,
                before_objective=float(before_objective),
                after_objective=float(after_objective),
            )
            if output is not None:
                boundary["before_state"] = fitted["final_state"]
                boundary["after_state"] = base.save_state(
                    model, output / f"boundary_{stage:03d}.npz"
                )
            boundaries.append(boundary)
        model.zero_grad(set_to_none=True)
        # Commit only in reduced mode, as in the frozen fitter. Joint endpoints
        # retain their readouts; a new optimal solve would change the endpoint.
        terminal_objective, terminal = (
            base.reduced_objective(
                model, x, y, config.ridge, commit=True, **objective_options
            )
            if optimizer == "reduced"
            else base.joint_objective(model, x, y, config.ridge, **objective_options)
        )
        terminal_objective.backward()
        parameters = (
            model.body_parameters()
            if optimizer == "reduced"
            else [p for p in model.parameters() if p.requires_grad]
        )
        gradient = base._gradient_norm(parameters)
        model.zero_grad(set_to_none=True)
        closures = []
        for stage, receipt in enumerate(stage_fits):
            offset = len(closures)
            closures.extend(
                {
                    **item,
                    "stage": stage,
                    "stage_evaluation": item["evaluation"],
                    "evaluation": offset + i,
                }
                for i, item in enumerate(receipt["closure_history"])
            )
        receipt = {
            "status": "complete",
            "optimizer": optimizer,
            "config": asdict(config),
            "stored_parameters": model.parameter_count,
            "gradient_parameters": model.gradient_parameter_count,
            "initialization": model.initialization_receipt,
            "soma_gauge": stage_fits[0]["soma_gauge"],
            "restart_schedule": [item["config"]["steps"] for item in stage_fits],
            "rebalance": bool(rebalance),
            "boundary_records": boundaries,
            "stage_fits": stage_fits,
            "closure_history": closures,
            "closure_calls": len(closures),
            "iterations": sum(item["iterations"] for item in stage_fits),
            "initial_geometry": stage_fits[0]["initial_geometry"],
            "terminal_geometry": base.geometry_diagnostics(
                model, x, **objective_options
            ),
            "terminal_design_diagnostics": stage_fits[-1][
                "terminal_design_diagnostics"
            ],
            "stages": [
                {"stage": "common_ridge_warm_start", **stage_fits[0]["stages"][0]},
                {"stage": "terminal", **terminal},
            ],
            "warm_start_objective": stage_fits[0]["warm_start_objective"],
            "terminal_objective": float(terminal_objective.detach()),
            "terminal_unscaled_gradient_l2": gradient,
            "effective_projection_movement_l2": float(
                (model.effective_projection().detach() - initial_projection).norm()
            ),
            "parameter_movement_l2": {
                name: float((value.detach() - initial[name]).norm())
                for name, value in model.named_parameters()
            },
            "elapsed_seconds": time.monotonic() - started,
            "scope": "Exact unchanged TRAIN-centered/RMS ridge objective, total iteration allowance split across fresh-history stages; all terminal outcomes retained. QR is periodic gauge balancing, not an orthogonal-manifold constraint. Stage-wise warm starts are repeated and charged; joint mode also refits readouts at each stage start.",
        }
        if config.projection_rms_floor:
            receipt["scope"] = (
                "TRAIN-centered ridge with declared effective-projection penalty; "
                "all arms share the declared restart schedule and warm starts. "
                "QR preserves the effective projections and penalized objective. "
                "All endpoints retained; this changes training regularization, "
                "not the architecture or inference."
            )
        if output is not None:
            receipt["initial_state"] = initial_state
            receipt["warm_start_state"] = stage_fits[0]["warm_start_state"]
            receipt["final_state"] = base.save_state(model, output / "final.npz")
            _write(output / "fit.json", receipt)
        return receipt
    except Exception as error:
        if output is not None:
            _write(
                output / "failure.json",
                {
                    "status": "failed",
                    "error": repr(error),
                    "completed_stages": stage_fits,
                    "boundary_records": boundaries,
                    "partial_state": base.save_state(model, output / "partial.npz"),
                },
            )
        raise
