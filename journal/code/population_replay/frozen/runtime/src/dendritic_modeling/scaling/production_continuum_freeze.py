"""Freeze selection and size forecasts using development evidence only."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import production_continuum as pc

FLOOR = 1e-24


def forecasts(selected, plan):
    """Use the declared exact-common grid; never add sizes to rescue eligibility."""
    output = []
    for task in pc.TASKS:
        eligible = []
        for family in pc.FAMILIES:
            rows = sorted(
                (
                    r
                    for r in selected
                    if r["task"] == task
                    and r["family"] == family
                    and r["budget"] in plan["parameter_budgets"]
                ),
                key=lambda r: r["budget"],
            )
            used = [r for r in rows if r["mean_validation_mse"] > FLOOR]
            entry = {
                "task": task,
                "family": family,
                "grid": "exact_common",
                "used_budgets": [r["budget"] for r in used],
                "models": [],
            }
            allowed = len(used) >= 4 and used[-1]["budget"] / used[0]["budget"] >= 8
            entry["fit_eligible"] = allowed
            entry["floor_category_prediction"] = (
                "censored_at_or_below_1e-24"
                if all(r["mean_validation_mse"] <= FLOOR for r in rows[-2:])
                else "no_floor_prediction"
            )
            if allowed:
                x = np.array([r["budget"] for r in used], dtype=float)
                y = np.array([r["mean_validation_mse"] for r in used])
                held = np.array(plan["heldout_parameter_budgets"], dtype=float)
                for name, z, h in [
                    ("power", np.log(x), np.log(held)),
                    ("exponential", x, held),
                ]:
                    coef = np.linalg.lstsq(
                        np.column_stack((np.ones(len(x)), z)), np.log(y), rcond=None
                    )[0]
                    entry["models"].append(
                        {
                            "name": name,
                            "log_amplitude": float(coef[0]),
                            "decay": float(-coef[1]),
                            "predictions": dict(
                                zip(
                                    map(str, map(int, held)),
                                    np.exp(coef[0] + coef[1] * h).tolist(),
                                )
                            ),
                        }
                    )
                # Relative scaling avoids optimizing floor coordinates of vastly different magnitude.
                scale = float(y.min())
                logx, logy = np.log(x / x[0]), np.log(y / scale)

                def residual(theta, logx=logx, logy=logy):
                    floor_fraction, log_amp, alpha = theta
                    return (
                        np.logaddexp(
                            np.log(max(floor_fraction, 1e-300)), log_amp - alpha * logx
                        )
                        - logy
                    )

                fits = [
                    least_squares(
                        residual,
                        [f, float(logy[0]), 2.0],
                        bounds=([0, -700, 0], [0.999999, 700, 100]),
                        max_nfev=4000,
                    )
                    for f in [0, 0.1, 0.9]
                ]
                fit = min(fits, key=lambda v: float(np.sum(v.fun**2)))
                f, amplitude, alpha = fit.x
                prediction = scale * (
                    f + np.exp(amplitude - alpha * np.log(held / x[0]))
                )
                entry["models"].append(
                    {
                        "name": "floor_plus_power",
                        "floor": float(scale * f),
                        "alpha": float(alpha),
                        "optimizer_success": bool(fit.success),
                        "predictions": dict(
                            zip(map(str, map(int, held)), prediction.tolist())
                        ),
                    }
                )
                eligible.append((entry, x, y))
            output.append(entry)
        # A shared exponent, with independent family coefficients, on the same eligible data.
        if len(eligible) >= 2:
            design, response = [], []
            for index, (_, x, y) in enumerate(eligible):
                for p, value in zip(x, y):
                    row = [0.0] * len(eligible) + [float(np.log(p))]
                    row[index] = 1.0
                    design.append(row)
                    response.append(float(np.log(value)))
            coef = np.linalg.lstsq(np.array(design), np.array(response), rcond=None)[0]
            for index, (entry, _, _) in enumerate(eligible):
                entry["models"].append(
                    {
                        "name": "shared_exponent",
                        "alpha": float(-coef[-1]),
                        "predictions": {
                            str(p): float(np.exp(coef[index] + coef[-1] * np.log(p)))
                            for p in plan["heldout_parameter_budgets"]
                        },
                    }
                )
    return output


def freeze(campaign, audit_path, output):
    campaign, output = Path(campaign).resolve(), Path(output)
    manifest = json.loads((campaign / "manifest.json").read_text())
    plan = json.loads((campaign / "plan.json").read_text())
    audit = json.loads(Path(audit_path).read_text())
    assert manifest["stage"] == "extended_development"
    assert pc.sha(campaign / "manifest.json") == audit["manifest_sha256"]
    assert audit["status"] == "passed" and audit["training_steps"] == 3200
    assert audit["replay_exact_count"] == audit["exact_parent_prefix_count"] == 1296
    assert pc.sha(campaign / "plan.json") == manifest["plan_sha256"]
    for source in manifest["source_files"]:
        assert pc.sha(campaign / source["path"]) == source["sha256"]
    for path, digest in audit["input_sha256"].items():
        assert pc.sha(path) == digest
    selected = {(r["task"], r["family"], r["budget"]): r for r in audit["selected"]}
    assert len(selected) == 162
    specs = []
    for task, family, budget in sorted(selected):
        choice = selected[task, family, budget]
        for seed in plan["confirmation_seeds"]:
            specs.append(
                pc.FitSpec(
                    family, budget, task, seed, choice["recipe"], choice["lr"], 3200
                )
            )
    for task in pc.TASKS:
        for family in pc.FAMILIES:
            choice = selected[task, family, max(plan["parameter_budgets"])]
            for budget in plan["heldout_parameter_budgets"]:
                for seed in plan["confirmation_seeds"]:
                    specs.append(
                        pc.FitSpec(
                            family,
                            budget,
                            task,
                            seed,
                            choice["recipe"],
                            choice["lr"],
                            3200,
                        )
                    )
    assert len(specs) == len(set(specs)) == 576
    for spec in specs:
        spec.validate()
    result = {
        "schema": "production_continuum_confirmation_freeze_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection_frozen": True,
        "test_materialized": False,
        "development_campaign": str(campaign),
        "development_manifest_sha256": pc.sha(campaign / "manifest.json"),
        "development_audit_sha256": pc.sha(audit_path),
        "freezer_source_sha256": pc.sha(__file__),
        "base_plan_sha256": manifest["plan_sha256"],
        "training_steps": 3200,
        "selected": audit["selected"],
        "size_forecasts": forecasts(audit["selected"], plan),
        "forecast_rules": "Fit only exact-common development sizes above1e-24; at least4 sizes over factor8. All model forecasts retained without TEST selection. Log-MSE fit; no uncertainty claim from two initialization seeds. Floor-censored families receive a category forecast only. Primary low-P grid is reported separately and is not expanded to rescue a slope fit.",
        "evaluation_rules": {
            "test_points": 32768,
            "include_endpoints": True,
            "once_per_saved_state": True,
            "interpretation_floor": FLOOR,
            "independent_numpy_full_grid": True,
            "high_precision_points_if_censored": 33,
            "recovery": "Do not overwrite any fit or evaluation. Keep failures in every summary.",
        },
        "scope": "Fresh-initialization and held-out-size confirmation for three preselected fixed targets, not new target-function replication or convergence certification.",
        "tasks": [{"index": i, "spec": asdict(spec)} for i, spec in enumerate(specs)],
        "fresh_density_confirmation_plan": next(
            s for s in plan["stages"] if s["id"] == "fresh_density_confirmation"
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    pc.dump(output / "freeze.json", result)
    pc.dump(output / "base_plan.json", plan)
    import shutil

    shutil.copy2(__file__, output / "freezer.py")
    print(
        json.dumps(
            {"tasks": len(specs), "freeze_sha256": pc.sha(output / "freeze.json")}
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.campaign, args.audit, args.output_dir)
