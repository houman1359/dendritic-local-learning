"""Prospective cohort-preserving analysis; never fits a model or rate curve."""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/outputs/parameter_scaling/20260913"


def default_config():
    return {
        "schema": "rank_iid_pooled_analysis_v2",
        "cohorts": {
            "original": {
                "root": BASE + "/rank_iid_study_v1",
                "teachers": [10, 11, 12, 13],
                "all_fits": 1980,
            },
            "replication": {
                "root": BASE + "/rank_iid_replication_v1",
                "teachers": [14, 15, 16, 17],
                "all_fits": 576,
            },
        },
        "families": ["shunt", "relu", "tanh"],
        "architectures": ["full", "rank1", "rank2"],
        "intrinsic_ranks": [1, 2],
        "parameters": [65, 125, 245, 485],
        "replicates": [0, 1],
        "observation_seed_base": 2026106001,
        "primary_rows": 1152,
        "withheld_parameters": 485,
        "forecast_sha256": "f4dba0a5e34bbe9ce0b978476b226dd10567e64895b71837b7c9172d6a938b65",
        "selected_recipes_sha256": "538401882cf702c64ec89f7ac4f9eef6e744b765fa28615f8b2cdd43bdc8fa90",
        "reconciliation_relative_path": "roundoff_reconciliation_v1/result.json",
        "reconciliation_review_relative_path": "roundoff_reconciliation_v1/root_review.json",
        "bootstrap": "Exact empirical teacher bootstrap; multinomial countvectors with integer weights; inverse-CDF1/40 and39/40 quantiles",
        "primary_interaction": "D=mean_obs log(rank2/rank1)q1 - mean_obs log(rank2/rank1)q2; shuntP485; positive predicted",
        "scope": "Original4 teachers remain primary; replication4 and explicitly pooled8 are additional. No new model/recipe/rate fits or selected envelopes.",
    }


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


class Sources:
    def __init__(self):
        self.items = {}

    def bind(self, path, role):
        path = Path(path).resolve()
        digest = sha(path)
        self.items[str(path)] = {"role": role, "path": str(path), "sha256": digest}
        return digest

    def read(self, path, role):
        self.bind(path, role)
        return json.loads(Path(path).read_text())


@functools.lru_cache(maxsize=8)
def bootstrap_inventory(n):
    if not 1 <= n <= 8:
        raise ValueError("This bounded exact bootstrap supports1 through8 teachers")

    def compositions(total, slots):
        if slots == 1:
            yield (total,)
        else:
            for first in range(total + 1):
                for tail in compositions(total - first, slots - 1):
                    yield (first, *tail)

    counts = np.array(list(compositions(n, n)), dtype=np.int64)
    weights = np.array(
        [
            math.factorial(n) // math.prod(math.factorial(int(c)) for c in row)
            for row in counts
        ],
        dtype=np.int64,
    )
    assert len(counts) == math.comb(2 * n - 1, n - 1)
    assert int(weights.sum()) == n**n
    counts.setflags(write=False)
    weights.setflags(write=False)
    return counts, weights


def teacher_interval(values, exponentiate=False):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Teacher values must be a finite vector")
    n = len(values)
    counts, weights = bootstrap_inventory(n)
    means = counts @ values / n
    order = np.argsort(means, kind="stable")
    cumulative = np.cumsum(weights[order])
    total = n**n
    targets = [(total + 39) // 40, (39 * total + 39) // 40]
    bounds = [
        float(means[order[np.searchsorted(cumulative, target, side="left")]])
        for target in targets
    ]
    point = float(values.mean())
    if exponentiate:
        point, *bounds = np.exp([point, *bounds]).tolist()
    return {
        "estimate": point,
        "descriptive_bootstrap_low": bounds[0],
        "descriptive_bootstrap_high": bounds[1],
        "teachers": n,
        "distinct_bootstrap_countvectors": len(counts),
        "ordered_resamples_represented": total,
    }


def expected_keys(config):
    expected = set()
    for cohort, setting in config["cohorts"].items():
        for teacher, q, rep, p, family, architecture in itertools.product(
            setting["teachers"],
            config["intrinsic_ranks"],
            config["replicates"],
            config["parameters"],
            config["families"],
            config["architectures"],
        ):
            seed = config["observation_seed_base"] + 100 * teacher + 2 * rep
            expected.add((cohort, teacher, q, seed, p, family, architecture))
    return expected


def validate_rows(rows, config):
    keys = []
    for row in rows:
        key = tuple(
            row[k]
            for k in [
                "cohort",
                "teacher",
                "intrinsic_rank",
                "seed",
                "parameters",
                "family",
                "architecture",
            ]
        )
        keys.append(key)
        if not math.isfinite(row["observed_mse"]) or row["observed_mse"] <= 0:
            raise ValueError(
                "Positive finite raw MSE is required; no silent log flooring"
            )
        if (
            row["train_n"] != 2048
            or row["sigma"] != 0
            or row["initialization"] != "observational"
        ):
            raise ValueError("Non-primary observation regime")
        p, m = row["parameters"], row["branches"]
        expected = {"full": 5 * m + 5, "rank1": 3 * m + 17, "rank2": 4 * m + 29}[
            row["architecture"]
        ]
        if p != expected:
            raise ValueError("Incorrect actual parameter inventory")
    if (
        len(keys) != config["primary_rows"]
        or len(set(keys)) != len(keys)
        or set(keys) != expected_keys(config)
    ):
        raise ValueError("Missing, duplicate, or unexpected primary endpoint")


def qualify_audit(root, audit, setting, config, sources):
    """Preserve a failed audit and admit only root-reviewed exact exceptions."""
    expected = setting["all_fits"]
    if audit["status"] == "passed":
        if (
            audit["failures"]
            or audit["endpoint_count"] != expected
            or len(audit["rows"]) != expected
        ):
            raise ValueError("Passed audit has inconsistent inventory")
        return dict(audit) | {
            "qualification": {
                "original_status": "passed",
                "qualified_endpoints": expected,
                "documented_exception_endpoints": 0,
            },
            "accepted_exceptions": {},
        }
    if audit["status"] != "failed":
        raise ValueError("Unknown original audit status")
    result_path = root / config["reconciliation_relative_path"]
    review_path = root / config["reconciliation_review_relative_path"]
    review = sources.read(review_path, "root review of documented numerical exceptions")
    original_sha = sources.bind(
        root / "numpy_replay/audit.json", "unchanged original failed audit"
    )
    if (
        review["status"] != "approved"
        or review["reviewer"] != "root"
        or review["original_audit_sha256"] != original_sha
    ):
        raise ValueError("Numerical reconciliation lacks bound root approval")
    if (
        sources.bind(result_path, "separate documented numerical reconciliation")
        != review["reconciliation_sha256"]
    ):
        raise ValueError("Reconciliation changed after root review")
    result = sources.read(result_path, "separate documented numerical reconciliation")
    if result["status"] != "passed_with_documented_roundoff_exceptions":
        raise ValueError("Numerical reconciliation is not qualified")
    original = result["original_audit"]
    if (
        Path(original["path"]).resolve() != (root / "numpy_replay/audit.json").resolve()
        or original["sha256"] != original_sha
        or original["status"] != "failed"
        or original["passed_endpoint_count"] != audit["endpoint_count"]
    ):
        raise ValueError("Reconciliation does not preserve this failed audit")
    for descriptor, role in [
        (result["diagnostic"], "exact numerical diagnostic"),
        (result["reconciliation_source"], "reconciliation implementation"),
    ]:
        if sources.bind(descriptor["path"], role) != descriptor["sha256"]:
            raise ValueError("Changed diagnostic or reconciliation source")
    diagnostic = result["diagnostic"]
    if (
        sources.bind(diagnostic["source_path"], "numerical diagnostic source")
        != diagnostic["source_sha256"]
    ):
        raise ValueError("Changed numerical diagnostic implementation")
    if (
        result["expected_endpoints"] != expected
        or result["qualified_endpoint_count"] != expected
        or result["state_replays"] != 3 * expected
    ):
        raise ValueError("Reconciled endpoint inventory is incomplete")
    for filename, descriptor in result["selection_forecast_hashes"].items():
        if (
            sources.bind(root / filename, "reconciliation-bound selection/forecast")
            != descriptor["sha256"]
            or Path(descriptor["path"]).resolve() != (root / filename).resolve()
        ):
            raise ValueError("Reconciliation changed selection/forecast")
    rows = list(audit["rows"])
    present = {str(Path(row["evaluation"]).resolve()) for row in rows}
    accepted = {}
    for exception in result["accepted_exception_endpoints"]:
        path = str(Path(exception["evaluation_path"]).resolve())
        if (
            path in present
            or path in accepted
            or not exception["all_original_checks_completed"]
        ):
            raise ValueError("Duplicate or incompletely rechecked exception endpoint")
        for path_key, hash_key, role in [
            ("evaluation_path", "evaluation_sha256", "accepted endpoint evaluation"),
            ("fit_path", "fit_sha256", "accepted endpoint fit"),
            ("dataset_path", "dataset_sha256", "accepted endpoint dataset"),
        ]:
            if sources.bind(exception[path_key], role) != exception[hash_key]:
                raise ValueError("Changed accepted endpoint artifact")
        states = exception["state_archives"]
        if len(states) != 3 or {state["role"] for state in states} != {
            "initial",
            "warm_start",
            "final",
        }:
            raise ValueError("Reconciliation lacks all three original saved states")
        for state in states:
            if (
                sources.bind(state["path"], "accepted endpoint saved state")
                != state["sha256"]
            ):
                raise ValueError("Changed accepted endpoint state")
        if not exception["accepted_metric_exceptions"]:
            raise ValueError("No explicitly documented metric exception")
        row = exception["endpoint_row"]
        if str(Path(row["evaluation"]).resolve()) != path or row["status"] != "passed":
            raise ValueError("Qualified endpoint row was not fully replayed")
        rows.append(row)
        accepted[path] = exception["accepted_metric_exceptions"]
    for failure in audit["failures"]:
        if (
            failure["scope"] != "inventory"
            and str(Path(failure["scope"]).resolve()) not in accepted
        ):
            raise ValueError("Unresolved original audit failure")
    if len(rows) != expected or len({r["evaluation"] for r in rows}) != expected:
        raise ValueError("Qualified rows do not complete original inventory")
    return dict(audit) | {
        "rows": rows,
        "accepted_exceptions": accepted,
        "qualification": {
            "original_status": "failed",
            "qualified_endpoints": expected,
            "documented_exception_endpoints": len(accepted),
            "root_review_path": str(review_path.resolve()),
            "reconciliation_sha256": review["reconciliation_sha256"],
        },
    }


def preflight(config, sources):
    """Read only completion/audit seals until both cohorts are qualified."""
    qualified = {}
    for cohort, setting in config["cohorts"].items():
        root = Path(setting["root"])
        complete = sources.read(root / "campaign_complete.json", "campaign completion")
        if complete["status"] != "complete" or complete["fits"] != setting["all_fits"]:
            raise ValueError("Campaign incomplete: " + cohort)
    for cohort, setting in config["cohorts"].items():
        root = Path(setting["root"])
        audit = sources.read(
            root / "numpy_replay/audit.json", "independent all-state NumPy audit"
        )
        if Path(audit["campaign_root"]).resolve() != root.resolve():
            raise ValueError("Audit belongs to a different cohort")
        auditor = root / "numpy_replay_protocol/rank_study_replay.py"
        if (
            sources.bind(auditor, "independent auditor source")
            != audit["auditor_source_sha256"]
        ):
            raise ValueError("Changed auditor source")
        audit_manifest = sources.read(
            root / "numpy_replay_protocol/manifest.json", "auditor protocol manifest"
        )
        for filename, digest in audit_manifest["sha256"].items():
            if (
                sources.bind(
                    root / "numpy_replay_protocol" / filename, "auditor frozen artifact"
                )
                != digest
            ):
                raise ValueError("Changed auditor artifact")
        qualified[cohort] = qualify_audit(root, audit, setting, config, sources)
    return qualified


def load_rows(config, sources, qualified):
    rows = []
    frozen = None
    for cohort, setting in config["cohorts"].items():
        root = Path(setting["root"])
        for filename, expected, seal, field in [
            (
                "selected_recipes.json",
                config["selected_recipes_sha256"],
                "selection_complete.json",
                "selected_sha256",
            ),
            (
                "frozen_forecasts.json",
                config["forecast_sha256"],
                "forecast_complete.json",
                "forecast_sha256",
            ),
        ]:
            if (
                sources.bind(root / filename, "inherited fixed selection/forecast")
                != expected
            ):
                raise ValueError("Changed fixed selection or forecast")
            if sources.read(root / seal, "selection/forecast seal")[field] != expected:
                raise ValueError("Invalid selection or forecast seal")
        selected = sources.read(
            root / "selected_recipes.json", "inherited selected recipes"
        )
        current_forecasts = sources.read(
            root / "frozen_forecasts.json", "original frozen forecasts"
        )
        assert current_forecasts["selected_sha256"] == config["selected_recipes_sha256"]
        frozen = current_forecasts
        sources.bind(root / "config.json", "cohort config")
        sources.bind(root / "protocol.md", "cohort prospective protocol")
        initialized = sources.read(
            root / "initialized.json", "cohort initialized bindings"
        )
        for name, key in [
            ("config.json", "config_sha256"),
            ("choices.json", "choices_sha256"),
            ("protocol.md", "protocol_sha256"),
        ]:
            if (
                key in initialized
                and sources.bind(root / name, "cohort frozen configuration")
                != initialized[key]
            ):
                raise ValueError("Changed initialized configuration")
        for bound in initialized.get("sources", []):
            if (
                sources.bind(bound["path"], "cohort frozen model/task/driver")
                != bound["sha256"]
            ):
                raise ValueError("Changed initialized source")
        audited = {
            str(Path(r["evaluation"]).resolve()): r for r in qualified[cohort]["rows"]
        }
        paths = sorted((root / "fits/confirmation/primary").rglob("evaluation.json"))
        if len(paths) != 576:
            raise ValueError("Primary cohort must contain exactly576 endpoints")
        for path in paths:
            record = sources.read(path, "primary evaluation")
            replay = audited.get(str(path.resolve()))
            if (
                replay is None
                or replay["status"] != "passed"
                or replay["endpoint"] != "test"
            ):
                raise ValueError("Endpoint absent from passed replay")
            if record["stage"] != "confirmation" or record["variant"] != "primary":
                raise ValueError("Unexpected non-primary endpoint")
            for field in [
                "teacher",
                "intrinsic_rank",
                "seed",
                "family",
                "architecture",
                "parameters",
                "branches",
                "train_n",
                "sigma",
            ]:
                if record[field] != replay[field]:
                    raise ValueError("Replay/evaluation identity mismatch")
            mse = float(record["test"]["observed_mse"])
            if not np.isclose(
                mse, replay["endpoint_metrics"]["observed_mse"], rtol=3e-6, atol=5e-15
            ):
                exceptions = qualified[cohort]["accepted_exceptions"].get(
                    str(path.resolve()), []
                )
                if not any(
                    item["label"] == "test observed_mse"
                    and item["recorded"] == mse
                    and item["actual"] == replay["endpoint_metrics"]["observed_mse"]
                    for item in exceptions
                ):
                    raise ValueError(
                        "Evaluation differs from passed or exactly reconciled replay"
                    )
            chosen = selected["selections"][
                f"{record['architecture']}/{record['family']}/q{record['intrinsic_rank']}"
            ]["choice"]
            if record["choice_id"] != chosen["id"]:
                raise ValueError("Primary endpoint changed inherited recipe")
            sources.bind(record["fit_file"], "primary fit receipt bound by NumPy audit")
            rows.append(
                {
                    k: record[k]
                    for k in [
                        "teacher",
                        "intrinsic_rank",
                        "seed",
                        "family",
                        "architecture",
                        "parameters",
                        "branches",
                        "train_n",
                        "sigma",
                        "initialization",
                        "choice_id",
                        "optimizer",
                    ]
                }
                | {
                    "cohort": cohort,
                    "observed_mse": mse,
                    "clean_teacher_mse": float(record["test"]["clean_teacher_mse"]),
                    "evaluation_file": str(path.resolve()),
                    "evaluation_sha256": sha(path),
                    "fit_file": record["fit_file"],
                }
            )
    validate_rows(rows, config)
    return pd.DataFrame(rows), frozen


def scope_frames(frame):
    for scope in ("original", "replication", "pooled"):
        yield scope, frame if scope == "pooled" else frame[frame.cohort == scope]


def summarize(frame, frozen):
    risk_keys = [
        "cohort",
        "family",
        "architecture",
        "intrinsic_rank",
        "parameters",
        "teacher",
    ]
    teacher_risks = (
        frame.groupby(risk_keys, sort=True)
        .agg(
            mean_mse=("observed_mse", "mean"),
            mean_log_mse=("observed_mse", lambda x: float(np.log(x).mean())),
            replicates=("observed_mse", "size"),
        )
        .reset_index()
    )
    if not (teacher_risks.replicates == 2).all():
        raise ValueError("Teacher risk lacks two observation replicates")
    paired = frame.pivot(
        index=["cohort", "family", "parameters", "teacher", "seed", "intrinsic_rank"],
        columns="architecture",
        values="observed_mse",
    )
    if paired.isna().any().any():
        raise ValueError("Incomplete paired architecture comparison")
    paired_rows = []
    for index, values in paired.iterrows():
        ident = dict(zip(paired.index.names, index, strict=True))
        for numerator, denominator in [
            ("rank1", "full"),
            ("rank2", "full"),
            ("rank2", "rank1"),
        ]:
            logratio = float(np.log(values[numerator]) - np.log(values[denominator]))
            paired_rows.append(
                ident
                | {
                    "numerator": numerator,
                    "denominator": denominator,
                    "log_ratio": logratio,
                    "ratio": math.exp(logratio),
                }
            )
    contrasts = pd.DataFrame(paired_rows)
    contrast_keys = [
        "cohort",
        "family",
        "parameters",
        "teacher",
        "intrinsic_rank",
        "numerator",
        "denominator",
    ]
    teacher_contrasts = (
        contrasts.groupby(contrast_keys, sort=True)
        .agg(mean_log_ratio=("log_ratio", "mean"), replicates=("log_ratio", "size"))
        .reset_index()
    )
    teacher_contrasts["geometric_ratio"] = np.exp(teacher_contrasts.mean_log_ratio)
    targets = teacher_contrasts[
        (teacher_contrasts.numerator == "rank2")
        & (teacher_contrasts.denominator == "rank1")
    ]
    targets = targets.pivot(
        index=["cohort", "family", "parameters", "teacher"],
        columns="intrinsic_rank",
        values="mean_log_ratio",
    )
    interactions = targets.reset_index()[
        ["cohort", "family", "parameters", "teacher"]
    ].copy()
    interactions["D"] = np.asarray(targets[1] - targets[2])
    risk_summary = []
    contrast_summary = []
    interaction_summary = []
    forecasts = []
    architecture_forecasts = []
    for scope, subset in scope_frames(teacher_risks):
        for group, cell in subset.groupby(
            ["family", "architecture", "intrinsic_rank", "parameters"], sort=True
        ):
            risk_summary.append(
                dict(
                    zip(
                        ["family", "architecture", "intrinsic_rank", "parameters"],
                        group,
                        strict=True,
                    )
                )
                | {
                    "cohort_scope": scope,
                    "arithmetic_mean_mse": float(cell.mean_mse.mean()),
                    "geometric_mean_mse": float(np.exp(cell.mean_log_mse.mean())),
                    **teacher_interval(cell.mean_mse),
                }
            )
    for scope, subset in scope_frames(teacher_contrasts):
        for group, cell in subset.groupby(
            ["family", "parameters", "intrinsic_rank", "numerator", "denominator"],
            sort=True,
        ):
            contrast_summary.append(
                dict(
                    zip(
                        [
                            "family",
                            "parameters",
                            "intrinsic_rank",
                            "numerator",
                            "denominator",
                        ],
                        group,
                        strict=True,
                    )
                )
                | {
                    "cohort_scope": scope,
                    "mean_teacher_log_ratio": float(cell.mean_log_ratio.mean()),
                    "teachers_ratio_below_one": int((cell.mean_log_ratio < 0).sum()),
                    **teacher_interval(cell.mean_log_ratio, True),
                }
            )
    for scope, subset in scope_frames(interactions):
        for (family, p), cell in subset.groupby(["family", "parameters"], sort=True):
            interaction_summary.append(
                {
                    "cohort_scope": scope,
                    "family": family,
                    "parameters": p,
                    "positive_teachers": int((cell.D > 0).sum()),
                    **teacher_interval(cell.D),
                }
            )
    for scope, subset in scope_frames(frame):
        for prediction in frozen["forecasts"]:
            cell = subset[
                (subset.family == prediction["family"])
                & (subset.architecture == prediction["architecture"])
                & (subset.intrinsic_rank == prediction["intrinsic_rank"])
                & (subset.parameters == prediction["withheld_p"])
            ]
            actual = float(cell.observed_mse.mean())
            predicted = prediction["predicted_mse"]
            forecasts.append(
                {
                    "cohort_scope": scope,
                    **prediction,
                    "actual_arithmetic_mean_mse": actual,
                    "predicted_over_actual": predicted / actual,
                    "absolute_log_error": abs(
                        math.log(max(predicted, 1e-300) / actual)
                    ),
                    "observation_fits": len(cell),
                    "teachers": int(cell.teacher.nunique()),
                }
            )
        for prediction in frozen["architecture_predictions"]:
            cell = subset[
                (subset.family == prediction["family"])
                & (subset.intrinsic_rank == prediction["intrinsic_rank"])
                & (subset.parameters == 485)
            ]
            risks = cell.groupby("architecture").observed_mse.mean()
            observed_best = min(risks.index, key=lambda a: (risks[a], a))
            architecture_forecasts.append(
                {
                    "cohort_scope": scope,
                    **prediction,
                    "observed_best_architecture": observed_best,
                    "correct": prediction["predicted_architecture"] == observed_best,
                    "predicted_architecture_over_best": float(
                        risks[prediction["predicted_architecture"]] / risks.min()
                    ),
                }
            )
    return {
        "PrimaryRows": frame,
        "TeacherRisks": teacher_risks,
        "PairedRatios": contrasts,
        "TeacherRatios": teacher_contrasts,
        "TeacherInteractions": interactions,
        "RiskSummary": pd.DataFrame(risk_summary),
        "RatioSummary": pd.DataFrame(contrast_summary),
        "InteractionSummary": pd.DataFrame(interaction_summary),
        "FrozenForecasts": pd.DataFrame(forecasts),
        "ArchitectureForecasts": pd.DataFrame(architecture_forecasts),
    }


def plot_curves(tables, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"full": "#41464b", "rank1": "#278472", "rank2": "#cc7832"}
    for scope in ("original", "replication", "pooled"):
        fig, axes = plt.subplots(
            2, 3, figsize=(11, 6.6), sharex=True, constrained_layout=True
        )
        for row, q in enumerate((1, 2)):
            for col, family in enumerate(("shunt", "relu", "tanh")):
                ax = axes[row, col]
                for architecture in ("full", "rank1", "rank2"):
                    cell = tables["RiskSummary"]
                    cell = cell[
                        (cell.cohort_scope == scope)
                        & (cell.intrinsic_rank == q)
                        & (cell.family == family)
                        & (cell.architecture == architecture)
                    ].sort_values("parameters")
                    x = cell.parameters.to_numpy()
                    y = cell.arithmetic_mean_mse.to_numpy()
                    ax.plot(
                        x,
                        y,
                        "o-",
                        label=architecture,
                        color=colors[architecture],
                        linewidth=1.5,
                        markersize=4,
                    )
                    ax.fill_between(
                        x,
                        cell.descriptive_bootstrap_low.to_numpy(),
                        cell.descriptive_bootstrap_high.to_numpy(),
                        color=colors[architecture],
                        alpha=0.13,
                    )
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.axvline(485, color="#999999", linestyle=":", linewidth=0.7)
                ax.set_title(f"{family}, target rank{q}")
                ax.grid(alpha=0.15)
                if col == 0:
                    ax.set_ylabel("Arithmetic mean test MSE")
                if row == 1:
                    ax.set_xlabel("Complete stored parameters")
        axes[0, 0].legend(frameon=False, fontsize=8)
        n = 8 if scope == "pooled" else 4
        fig.suptitle(
            f"{scope.capitalize()} cohort: {n} teachers, two observations each\nShading: descriptive exact teacher-bootstrap percentiles; panel vertical ranges differ",
            fontsize=11,
        )
        fig.savefig(output / f"{scope}_curves.png", dpi=200)
        fig.savefig(output / f"{scope}_curves.pdf")
        plt.close(fig)


def serial_records(frame):
    return json.loads(json.dumps(frame.to_dict(orient="records"), allow_nan=False))


def run(protocol_dir, output):
    protocol_dir, output = Path(protocol_dir), Path(output)
    sources = Sources()
    manifest = sources.read(protocol_dir / "manifest.json", "pooled analysis freeze")
    for name, digest in manifest["files"].items():
        if (
            sources.bind(protocol_dir / name, "pooled analysis frozen artifact")
            != digest
        ):
            raise ValueError("Changed frozen pooled analysis")
    if sha(__file__) != manifest["files"]["rank_pooled_analysis.py"]:
        raise ValueError("Executable differs from frozen analysis")
    config = sources.read(
        protocol_dir / "protocol.json", "pooled prospective configuration"
    )
    if config != default_config():
        raise ValueError("Unexpected pooled protocol")
    qualified = preflight(config, sources)
    output.mkdir(parents=True, exist_ok=False)
    try:
        frame, frozen = load_rows(config, sources, qualified)
        tables = summarize(frame, frozen)
        tables["Sources"] = pd.DataFrame(
            sorted(sources.items.values(), key=lambda r: r["path"])
        )
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False)
        with pd.ExcelWriter(
            output / "supporting_data.xlsx", engine="openpyxl"
        ) as writer:
            for name, table in tables.items():
                safe = table.copy()
                for column in safe.columns:
                    safe[column] = safe[column].map(
                        lambda value: (
                            json.dumps(value)
                            if isinstance(value, (dict, list))
                            else value
                        )
                    )
                safe.to_excel(writer, sheet_name=name, index=False)
        plot_curves(tables, output)
        summary = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "status": "passed",
            "primary_rows": len(frame),
            "cohort_teachers": {"original": 4, "replication": 4, "pooled": 8},
            "bootstrap": config["bootstrap"],
            "scope": config["scope"],
            "input_audit_qualification": {
                cohort: value["qualification"] for cohort, value in qualified.items()
            },
            "table_rows": {name: len(table) for name, table in tables.items()},
            "risks": serial_records(tables["RiskSummary"]),
            "ratios": serial_records(tables["RatioSummary"]),
            "interactions": serial_records(tables["InteractionSummary"]),
            "forecast_checks": serial_records(tables["FrozenForecasts"]),
            "architecture_forecast_checks": serial_records(
                tables["ArchitectureForecasts"]
            ),
            "primary_shunt_p485": serial_records(
                tables["InteractionSummary"].query(
                    'family=="shunt" and parameters==485'
                )
            ),
            "source_manifest_sha256": sha(protocol_dir / "manifest.json"),
        }
        write(output / "analysis.json", summary)
        write(
            output / "output_manifest.json",
            {
                "utc": datetime.now(timezone.utc).isoformat(),
                "files": {
                    p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()
                },
                "raw_sources": sources.items,
            },
        )
        return summary
    except Exception as error:
        write(
            output / "analysis_failure.json",
            {"type": type(error).__name__, "error": str(error)},
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.protocol_dir, args.output_dir)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "primary_rows", "primary_shunt_p485")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
