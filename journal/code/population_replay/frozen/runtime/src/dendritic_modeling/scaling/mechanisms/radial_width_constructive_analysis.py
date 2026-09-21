"""Teacher-cluster analysis of the separately audited known-profile learner.

This module reads immutable artifacts; it imports no task, constructor or fitter.
Eight teacher rotations are the production clusters, with two paired observation
draws within each cluster. Pointwise bootstrap intervals are descriptive for the
fixed radial profile, not uncertainty across new target shapes or simultaneous
confidence bands. Parameter counts are actual stored counts, never generic
budget labels. No asymptotic exponent is estimated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _case_key(task, teacher, observation):
    return f"{task}_t{teacher}_o{observation}"


def validate_and_collect(constructive_root, audit_path):
    """Fail closed on independent replay and all artifacts used by this analysis."""
    root = Path(constructive_root).resolve()
    audit_path = Path(audit_path).resolve()
    checked = {}

    def check(path, value):
        path = Path(path)
        key = str(path.resolve())
        if key not in checked:
            checked[key] = sha(path)
        if checked[key] != value:
            raise ValueError(f"Artifact hash mismatch: {path}")
        return path

    complete = read(root / "complete.json")
    config = read(root / "config.json")
    if complete.get("status") != "complete" or complete.get("failed_states") != 0:
        raise ValueError("Constructive campaign is incomplete")
    frozen = read(check(root / "initialized.json", complete["initialized_sha256"]))
    for name, value in frozen["bindings"].items():
        check(root / name, value)
    if (
        complete["learned_states"] != config["expected_learned_states"]
        or complete["oracle_states"] != config["expected_oracle_states"]
    ):
        raise ValueError("Complete learned and oracle inventories are required")
    audit = read(audit_path)
    if audit.get("status") != "passed" or audit.get(
        "constructive_complete_sha256"
    ) != sha(root / "complete.json"):
        raise ValueError(
            "A passed independent audit bound to this completion is required"
        )
    for name in ("learned_states", "oracle_states", "oracle_evaluations"):
        expected = config["expected_" + name]
        if any(
            audit.get(prefix + name) != expected
            for prefix in ("", "passed_", "expected_")
        ):
            raise ValueError("Independent audit coverage is incomplete")
    expected_planes = (
        len(config["tasks"])
        * len(config["teachers"])
        * len(config["observations"])
        * len(config["train_sizes"])
    )
    if (
        audit.get("plane_diagnostics") != expected_planes
        or audit.get("passed_plane_diagnostics") != expected_planes
    ):
        raise ValueError("Independent plane audit coverage is incomplete")
    # Companion audit source paths are explicit; accept either mapping or lists
    # of path/sha256 bindings, but never a bare unbound source name.
    if (
        not isinstance(audit.get("source_bindings"), list)
        or not audit["source_bindings"]
    ):
        raise ValueError("Nonempty explicit independent audit source bindings required")
    for binding in audit["source_bindings"]:
        if not isinstance(binding, dict) or not {"path", "sha256"}.issubset(binding):
            raise ValueError("Explicit independent audit source bindings required")
        check(binding["path"], binding["sha256"])
    gate = read(root / "main_completion_gate.json")
    main = Path(frozen["main_root"])
    for name, value in gate["barriers"].items():
        check(main / name, value)
    main_frozen = read(main / "initialized.json")
    for name, value in main_frozen["bindings"].items():
        check(main / name, value)
    # Stage hashes bind all generic outcomes. Numerical generic losses are unused.
    for stage in ("development", "bridge", "confirmation"):
        barrier = read(main / f"{stage}_complete.json")
        if (
            barrier.get("status") != "complete"
            or len(barrier["results"]) != barrier["fits"]
        ):
            raise ValueError("Generic stage barrier is incomplete")
        for name, value in barrier["results"].items():
            check(main / name, value)
    public = read(root / "public_data_inventory.json")
    for entry in public.values():
        check(entry["path"], entry["sha256"])
        check(entry["sidecar_path"], entry["sidecar_sha256"])
    learned_barrier = read(
        check(root / "learned_complete.json", complete["learned_complete_sha256"])
    )
    oracle_barrier = read(
        check(root / "oracle_complete.json", complete["oracle_complete_sha256"])
    )
    for marker, expected in (
        (learned_barrier, config["expected_learned_states"]),
        (oracle_barrier, config["expected_oracle_states"]),
    ):
        if (
            marker.get("status") != "complete"
            or marker.get("states") != expected
            or marker.get("failed_states") != 0
        ):
            raise ValueError("Phase completion differs")
        for path, value in marker["packets"].items():
            check(path, value)
    if oracle_barrier["learned_complete_sha256"] != sha(root / "learned_complete.json"):
        raise ValueError("Oracle phase does not bind learned completion")
    learned, oracle = [], []
    for task, teacher, observation, n in itertools.product(
        config["tasks"],
        config["teachers"],
        config["observations"],
        config["train_sizes"],
    ):
        directory = root / "learned" / _case_key(task, teacher, observation) / f"N{n}"
        packet_path = directory / "packet_result.json"
        if str(packet_path) not in learned_barrier["packets"]:
            raise ValueError("Learned packet missing from barrier")
        packet = read(packet_path)
        expected_packet = {
            "task": task,
            "teacher": teacher,
            "observation": observation,
            "train_n": n,
        }
        if packet["status"] != "complete" or packet["packet"] != expected_packet:
            raise ValueError("Learned packet identity differs")
        if len(packet["states"]) != len(config["intervals"]):
            raise ValueError("Learned packet state count differs")
        for intervals, binding in zip(
            config["intervals"], packet["states"], strict=True
        ):
            row_path = check(binding["path"], binding["sha256"])
            row = read(row_path)
            expected_case = dict(expected_packet, intervals=intervals)
            if (
                row["status"] != "complete"
                or row["phase"] != "learned"
                or row["case"] != expected_case
            ):
                raise ValueError("Learned state identity differs")
            for stem in ("state", "data", "predictions", "estimator"):
                check(row[stem + "_path"], row[stem + "_sha256"])
            check(
                directory / "construction_complete.json",
                row["construction_barrier_sha256"],
            )
            entry = public[_case_key(task, teacher, observation)]
            if (
                Path(row["data_path"]).resolve() != Path(entry["path"]).resolve()
                or row["data_sha256"] != entry["sha256"]
            ):
                raise ValueError("Learned row uses wrong shared dataset")
            _validate_count_and_metrics(row, expected_case, config)
            learned.append(
                dict(
                    expected_case,
                    workbook_scope="known_profile",
                    stored_parameters=row["stored_parameters"],
                    train_mse=row["metrics"]["train_mse"],
                    endpoint_mse=row["metrics"]["endpoint_mse"],
                    endpoint_rows=row["endpoint_rows"],
                    construction_seconds=row["construction_seconds"],
                    evaluation_seconds=row["evaluation_seconds"],
                    estimator_seconds_shared_across_S=packet["estimation_seconds"],
                    result_path=str(row_path),
                    result_sha256=sha(row_path),
                    state_path=row["state_path"],
                    state_sha256=row["state_sha256"],
                    data_path=row["data_path"],
                    data_sha256=row["data_sha256"],
                )
            )
    for task, teacher in itertools.product(config["tasks"], config["teachers"]):
        directory = root / "oracle" / f"{task}_t{teacher}"
        packet_path = directory / "packet_result.json"
        if str(packet_path) not in oracle_barrier["packets"]:
            raise ValueError("Oracle packet missing from barrier")
        packet = read(packet_path)
        if packet["status"] != "complete" or packet["packet"] != {
            "task": task,
            "teacher": teacher,
        }:
            raise ValueError("Oracle packet identity differs")
        if packet["learned_complete_sha256"] != sha(root / "learned_complete.json"):
            raise ValueError("Oracle packet preceded or lost learned barrier")
        if len(packet["states"]) != len(config["intervals"]):
            raise ValueError("Oracle state count differs")
        for intervals, binding in zip(
            config["intervals"], packet["states"], strict=True
        ):
            row_path = check(binding["path"], binding["sha256"])
            row = read(row_path)
            expected_case = {"task": task, "teacher": teacher, "intervals": intervals}
            if (
                row["status"] != "complete"
                or row["phase"] != "privileged_oracle"
                or row["case"] != expected_case
            ):
                raise ValueError("Oracle state identity differs")
            for stem in ("state", "private_specification"):
                check(row[stem + "_path"], row[stem + "_sha256"])
            if [e["observation"] for e in row["evaluations"]] != config["observations"]:
                raise ValueError("Oracle observations differ")
            for evaluation in row["evaluations"]:
                for stem in ("data", "predictions"):
                    check(evaluation[stem + "_path"], evaluation[stem + "_sha256"])
                _validate_count_and_metrics(
                    dict(row, **evaluation), expected_case, config
                )
                oracle.append(
                    dict(
                        expected_case,
                        observation=evaluation["observation"],
                        train_n=evaluation["train_rows"],
                        workbook_scope="oracle",
                        stored_parameters=row["stored_parameters"],
                        train_mse=evaluation["metrics"]["train_mse"],
                        endpoint_mse=evaluation["metrics"]["endpoint_mse"],
                        endpoint_rows=evaluation["endpoint_rows"],
                        construction_seconds=row["construction_seconds"],
                        evaluation_seconds=evaluation["evaluation_seconds"],
                        result_path=str(row_path),
                        result_sha256=sha(row_path),
                        state_path=row["state_path"],
                        state_sha256=row["state_sha256"],
                        data_path=evaluation["data_path"],
                        data_sha256=evaluation["data_sha256"],
                    )
                )
    diagnostic_path = check(
        root / "private_plane_diagnostics.json",
        oracle_barrier["private_plane_diagnostics_sha256"],
    )
    diagnostics = read(diagnostic_path)
    expected = set(
        itertools.product(
            config["tasks"],
            config["teachers"],
            config["observations"],
            config["train_sizes"],
        )
    )
    keys = [
        (r["task"], r["teacher"], r["observation"], r["train_n"]) for r in diagnostics
    ]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Plane diagnostic grid differs")
    for row in diagnostics:
        check(row["learned_state_path"], row["learned_state_sha256"])
        errors = np.asarray(row["per_block_squared_projector_frobenius_error"])
        if (
            errors.shape != (config["blocks"],)
            or not np.isfinite(errors).all()
            or np.any(errors < 0)
            or np.any(errors > 2 + 1e-10)
        ):
            raise ValueError("Invalid squared plane errors")
        if not np.isclose(
            errors.mean(),
            row["mean_squared_projector_frobenius_error"],
            atol=1e-14,
            rtol=1e-14,
        ):
            raise ValueError("Plane diagnostic mean differs")
    if (
        len(learned) != config["expected_learned_states"]
        or len(oracle) != config["expected_oracle_evaluations"]
    ):
        raise ValueError("Collected inventory differs")
    receipt = {
        "status": "passed",
        "verified_artifacts": len(checked),
        "constructive_complete_sha256": sha(root / "complete.json"),
        "independent_audit_path": str(audit_path),
        "independent_audit_sha256": sha(audit_path),
        "initialized_sha256": sha(root / "initialized.json"),
        "main_completion_gate_sha256": sha(root / "main_completion_gate.json"),
    }
    return config, learned, oracle, diagnostics, receipt


def _validate_count_and_metrics(row, case, config):
    m = int(case["task"].removeprefix("radial_m"))
    expected = (
        4 * config["blocks"] * (m + 1) * case["intervals"] + 7 * config["blocks"] + 1
    )
    if (
        row["stored_parameters"] != expected
        or row["counted_inventory"]["stored_parameters"] != expected
    ):
        raise ValueError("Actual parameter count differs")
    if row["endpoint_rows"] != config["endpoint_rows"]:
        raise ValueError("Endpoint size differs")
    if "train_n" in case and row["train_rows"] != case["train_n"]:
        raise ValueError("TRAIN prefix differs")
    if any(
        not np.isfinite(row["metrics"][metric]) or row["metrics"][metric] < 0
        for metric in ("train_mse", "endpoint_mse")
    ):
        raise ValueError("Finite nonnegative raw risk required")


def bootstrap_summary(values, indices):
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) != indices.shape[1]
        or len(values) < 2
        or not np.isfinite(values).all()
    ):
        raise ValueError("Balanced finite teacher values required")
    samples = values[indices].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "pointwise_lower_95": float(lo),
        "pointwise_upper_95": float(hi),
        "teacher_clusters": len(values),
        "minimum_teacher_value": float(values.min()),
        "maximum_teacher_value": float(values.max()),
    }


def summarize(config, learned, oracle, diagnostics, *, draws=10000, seed=2026147101):
    if draws < 100 or not isinstance(draws, int):
        raise ValueError("At least 100 predeclared bootstrap draws required")
    teachers = config["teachers"]
    if len(teachers) < 2:
        raise ValueError("At least two teacher clusters required")
    indices = np.random.default_rng(seed).integers(
        0, len(teachers), size=(draws, len(teachers))
    )
    grouped = defaultdict(list)
    for row in learned + oracle:
        # Oracle coefficients do not use training observations; its train_n column
        # above names evaluation exposure only, not a learner sample size.
        n = row["train_n"] if row["workbook_scope"] == "known_profile" else 0
        grouped[
            (
                row["workbook_scope"],
                row["task"],
                n,
                row["intervals"],
                row["stored_parameters"],
                row["teacher"],
            )
        ].append(row)
    teacher_rows, lookup = [], {}
    for key, observations in sorted(grouped.items()):
        scope, task, n, intervals, parameters, teacher = key
        if sorted(r["observation"] for r in observations) != config["observations"]:
            raise ValueError("Two balanced observation draws per teacher are required")
        risk = float(np.mean([r["endpoint_mse"] for r in observations]))
        row = {
            "workbook_scope": scope,
            "task": task,
            "train_n": n,
            "intervals": intervals,
            "stored_parameters": parameters,
            "teacher": teacher,
            "observation_draws": len(observations),
            "endpoint_mse": risk,
            "train_mse": float(np.mean([r["train_mse"] for r in observations])),
        }
        teacher_rows.append(row)
        lookup[(scope, task, n, intervals, teacher)] = risk
    curves = []
    curve_keys = sorted(
        {
            (
                r["workbook_scope"],
                r["task"],
                r["train_n"],
                r["intervals"],
                r["stored_parameters"],
            )
            for r in teacher_rows
        }
    )
    for scope, task, n, intervals, parameters in curve_keys:
        values = [lookup[(scope, task, n, intervals, teacher)] for teacher in teachers]
        curves.append(
            dict(
                workbook_scope=scope,
                task=task,
                train_n=n,
                intervals=intervals,
                stored_parameters=parameters,
                **bootstrap_summary(values, indices),
            )
        )
    ratios = []
    for task, intervals, (small, large) in itertools.product(
        config["tasks"],
        config["intervals"],
        zip(config["train_sizes"][:-1], config["train_sizes"][1:], strict=True),
    ):
        denominator = np.array(
            [lookup[("known_profile", task, small, intervals, t)] for t in teachers]
        )
        numerator = np.array(
            [lookup[("known_profile", task, large, intervals, t)] for t in teachers]
        )
        if np.any(denominator <= 0) or np.any(numerator <= 0):
            ratios.append(
                {
                    "task": task,
                    "intervals": intervals,
                    "smaller_train_n": small,
                    "larger_train_n": large,
                    "status": "undefined_zero_risk",
                    "workbook_scope": "known_profile",
                }
            )
            continue
        logs = np.log(numerator / denominator)
        boot = bootstrap_summary(logs, indices)
        ratios.append(
            {
                "task": task,
                "intervals": intervals,
                "smaller_train_n": small,
                "larger_train_n": large,
                "status": "defined",
                "workbook_scope": "known_profile",
                "teacher_clusters": len(teachers),
                "geometric_paired_risk_ratio": float(np.exp(boot["mean"])),
                "pointwise_lower_95": float(np.exp(boot["pointwise_lower_95"])),
                "pointwise_upper_95": float(np.exp(boot["pointwise_upper_95"])),
                "teachers_improved": int(np.count_nonzero(numerator < denominator)),
                "all_teachers_improved": bool(np.all(numerator < denominator)),
                "teacher_log_ratios": json.dumps(logs.tolist()),
            }
        )
    plane_rows = []
    for task, n in itertools.product(config["tasks"], config["train_sizes"]):
        values = []
        for teacher in teachers:
            subset = [
                r
                for r in diagnostics
                if (r["task"], r["teacher"], r["train_n"]) == (task, teacher, n)
            ]
            if sorted(r["observation"] for r in subset) != config["observations"]:
                raise ValueError("Plane diagnostics are unbalanced")
            values.append(
                float(
                    np.mean(
                        [r["mean_squared_projector_frobenius_error"] for r in subset]
                    )
                )
            )
        plane_rows.append(
            dict(
                task=task,
                train_n=n,
                workbook_scope="oracle",
                **bootstrap_summary(values, indices),
            )
        )
    plateau = [
        dict(r)
        for r in curves
        if r["workbook_scope"] == "known_profile"
        and r["intervals"] == max(config["intervals"])
    ]
    for row in plateau:
        plane = next(
            p
            for p in plane_rows
            if (p["task"], p["train_n"]) == (row["task"], row["train_n"])
        )
        row.update(
            mean_squared_plane_error=plane["mean"],
            plane_lower_95=plane["pointwise_lower_95"],
            plane_upper_95=plane["pointwise_upper_95"],
        )
    return {
        "teacher_means": teacher_rows,
        "aggregate_curves": curves,
        "paired_sample_ratios": ratios,
        "plane_error_curves": plane_rows,
        "large_S_plateau": plateau,
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "clusters": teachers,
            "resampling_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "unit": "teacher; observations averaged within teacher",
            "interval": "percentile pointwise 95%; shared resamples across all curves",
            "scope": "Fixed-profile rotations; no new-shape, simultaneous-band or exponent claim",
        },
    }


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def make_plots(output, config, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    names = {"radial_m2": "Radial quartic", "radial_m4": "Radial degree eight"}

    def save(fig, name):
        if config.get("mode") == "SMOKE_ONLY":
            fig.text(
                0.5,
                -0.015,
                "TECHNICAL SMOKE ONLY — not scientific confirmation",
                ha="center",
                color="darkred",
                fontsize=9,
            )
        fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(
        1, len(config["tasks"]), figsize=(6 * len(config["tasks"]), 4.7), squeeze=False
    )
    for ax, task in zip(axes.ravel(), config["tasks"], strict=True):
        subset = [r for r in summary["aggregate_curves"] if r["task"] == task]
        for n in config["train_sizes"] + [0]:
            rows = sorted(
                [r for r in subset if r["train_n"] == n],
                key=lambda r: r["stored_parameters"],
            )
            x, y = [r["stored_parameters"] for r in rows], [r["mean"] for r in rows]
            style = {"color": "black", "linestyle": "--"} if n == 0 else {}
            label = (
                "Private true-plane reference"
                if n == 0
                else f"Estimated plane, N={n:,}"
            )
            (line,) = ax.plot(x, y, marker="o", label=label, **style)
            ax.fill_between(
                x,
                [r["pointwise_lower_95"] for r in rows],
                [r["pointwise_upper_95"] for r in rows],
                color=line.get_color(),
                alpha=0.12,
            )
        ax.set(
            xscale="log",
            yscale="log",
            title=names[task],
            xlabel="Actual stored parameter count",
            ylabel="Mean raw TEST MSE",
        )
        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Known-profile ReLU learner and privileged approximation reference")
    fig.tight_layout()
    save(fig, "actual_parameter_risk_curves")
    fig, axes = plt.subplots(
        1, len(config["tasks"]), figsize=(6 * len(config["tasks"]), 4.3), squeeze=False
    )
    for ax, task in zip(axes.ravel(), config["tasks"], strict=True):
        for small, large in zip(
            config["train_sizes"][:-1], config["train_sizes"][1:], strict=True
        ):
            rows = sorted(
                [
                    r
                    for r in summary["paired_sample_ratios"]
                    if r["task"] == task
                    and r["smaller_train_n"] == small
                    and r["status"] == "defined"
                ],
                key=lambda r: r["intervals"],
            )
            x, y = (
                [r["intervals"] for r in rows],
                [r["geometric_paired_risk_ratio"] for r in rows],
            )
            (line,) = ax.plot(x, y, "o-", label=f"N {large:,} / {small:,}")
            ax.fill_between(
                x,
                [r["pointwise_lower_95"] for r in rows],
                [r["pointwise_upper_95"] for r in rows],
                alpha=0.12,
                color=line.get_color(),
            )
        ax.axhline(1, color="black", lw=0.8, linestyle="--")
        ax.set(
            xscale="log",
            yscale="log",
            title=names[task],
            xlabel="Intervals per profile line (S)",
            ylabel="Paired TEST risk ratio; below 1 improves",
        )
        ax.set_xticks(config["intervals"])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "paired_sample_risk_ratios")
    fig, axes = plt.subplots(
        2, len(config["tasks"]), figsize=(6 * len(config["tasks"]), 7.5), squeeze=False
    )
    for column, task in enumerate(config["tasks"]):
        rows = sorted(
            [r for r in summary["large_S_plateau"] if r["task"] == task],
            key=lambda r: r["train_n"],
        )
        x = [r["train_n"] for r in rows]
        for ax, metric, lower, upper, label in (
            (
                axes[0, column],
                "mean",
                "pointwise_lower_95",
                "pointwise_upper_95",
                "Raw TEST MSE",
            ),
            (
                axes[1, column],
                "mean_squared_plane_error",
                "plane_lower_95",
                "plane_upper_95",
                "Squared projector error (private audit)",
            ),
        ):
            ax.plot(x, [r[metric] for r in rows], "o-")
            ax.fill_between(
                x, [r[lower] for r in rows], [r[upper] for r in rows], alpha=0.15
            )
            ax.set(
                xscale="log",
                yscale="log",
                xlabel="TRAIN observations (N)",
                ylabel=label,
            )
            ax.set_xticks(x)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(alpha=0.2)
        axes[0, column].set_title(names[task] + f", fixed S={max(config['intervals'])}")
        oracle = next(
            r
            for r in summary["aggregate_curves"]
            if r["task"] == task
            and r["train_n"] == 0
            and r["intervals"] == max(config["intervals"])
        )
        axes[0, column].axhline(
            oracle["mean"],
            color="black",
            linestyle="--",
            label="Private-plane interpolation reference",
        )
        axes[0, column].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "large_S_risk_and_plane_error")


def run_analysis(
    constructive_root, audit_path, output_dir, *, draws=10000, seed=2026147101
):
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(
            "Preserve prior analyses; choose a fresh output directory"
        )
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve()
    (output / "analysis_source.py").write_bytes(source.read_bytes())
    write(
        output / "analysis_config.json",
        {
            "bootstrap_draws": draws,
            "bootstrap_seed": seed,
            "constructive_root": str(Path(constructive_root).resolve()),
            "audit_path": str(Path(audit_path).resolve()),
            "source_sha256": sha(source),
            "status": "frozen_before_outcome_read",
            "rules": "Teacher arithmetic risks after observation averaging; shared-cluster percentile pointwise intervals; adjacent N ratios; max-S diagnostic; no exponent fits",
        },
    )
    started = time.monotonic()
    try:
        config, learned, oracle, diagnostics, gate = validate_and_collect(
            constructive_root, audit_path
        )
        summary = summarize(
            config, learned, oracle, diagnostics, draws=draws, seed=seed
        )
        raw = {
            "learned_rows": learned,
            "oracle_rows": oracle,
            "private_plane_diagnostics": [
                dict(r, workbook_scope="oracle") for r in diagnostics
            ],
        }
        for name, rows in {
            **raw,
            **{k: v for k, v in summary.items() if k != "bootstrap"},
        }.items():
            write_csv(output / f"{name}.csv", rows)
        write(output / "analysis.json", dict(config=config, validation=gate, **summary))
        make_plots(output, config, summary)
        note = [
            "# Known-profile constructive results",
            "",
            (
                "Technical smoke only; these figures are not scientific confirmation."
                if config["mode"] == "SMOKE_ONLY"
                else "Production confirmation cohort; fixed radial profiles with fresh plane rotations."
            ),
            "",
            "This analysis concerns an explicit learner that knows the radial degree, profile, normalization and input law. Its planes use raw scalar TRAIN labels. The private-plane curve is a privileged approximation reference and is never a learned result.",
            "",
            f"There are {len(config['teachers'])} teacher clusters per fixed power, with two observation draws averaged within each teacher. Teachers differ in plane orientation, not response shape within a power. Intervals are pointwise percentile bootstrap intervals across teacher clusters; they are descriptive and unadjusted for multiple comparisons.",
            "",
            "All parameter values are actual stored counts. No generic budget matching or asymptotic exponent is inferred. The largest-S view diagnoses a possible finite-sample plateau; it does not prove that plane error is the sole source of residual risk.",
            "",
            "| Task | N | Actual P | Mean TEST MSE | Pointwise 95% interval |",
            "|---|---:|---:|---:|---|",
        ]
        for row in summary["large_S_plateau"]:
            note.append(
                f"| {row['task']} | {row['train_n']} | {row['stored_parameters']} | {row['mean']:.6g} | [{row['pointwise_lower_95']:.6g}, {row['pointwise_upper_95']:.6g}] |"
            )
        (output / "results.md").write_text("\n".join(note) + "\n")
        outputs = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
        receipt = {
            "status": "passed",
            "utc": datetime.now(timezone.utc).isoformat(),
            "mode": config["mode"],
            "learned_rows": len(learned),
            "oracle_evaluation_rows": len(oracle),
            "plane_diagnostics": len(diagnostics),
            "teacher_clusters": len(config["teachers"]),
            "elapsed_seconds": time.monotonic() - started,
            "outputs": outputs,
            "validation": gate,
            "analysis_source_sha256": sha(source),
            "scope": "Known-profile learner and explicitly privileged reference, teacher-cluster descriptive inference",
        }
        write(output / "receipt.json", receipt)
        return receipt
    except Exception:
        import traceback

        write(
            output / "failed.json",
            {"status": "failed", "traceback": traceback.format_exc()},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constructive-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(args.constructive_root, args.audit, args.output_dir), indent=2
        )
    )


if __name__ == "__main__":
    main()
