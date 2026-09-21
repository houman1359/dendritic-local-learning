#!/usr/bin/env python3
"""Audit the between-by-within MNIST factorial and freeze publication tables.

The paper's feedback ladder varies only the within-neuron distribution of a
per-neuron error that exact readout backpropagation supplies. This factorial
crosses that within-neuron ladder with an approximate between-neuron source:
a fixed random soma-level feedback matrix (direct feedback alignment) inside
the LocalCA rule (``soma_error_source: dfa``), plus the soma-DFA trainer as
the DFA-by-exact-autograd anchor. The exact-readout row is not rerun: it is
the frozen ladder release in ``source_data/mnist_feedback_ladder``.

Tables are written only when every expected new run is present and valid,
mirroring ``collect_mnist_feedback_ladder.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


JOURNAL = Path(__file__).resolve().parents[1]
OUTPUT = JOURNAL / "source_data" / "mnist_between_within_factorial"
LADDER = JOURNAL / "source_data" / "mnist_feedback_ladder" / "seed_outcomes.csv"
PROJECT_RUN_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "journal_extension_20260820/sweep_runs/mnist_between_within_factorial"
)
EXPECTED_SEEDS = tuple(range(42, 57))
CORE_LABEL = {
    "dendritic_shunting": "shunting",
    "dendritic_additive": "additive",
}
MODE_LABEL = {
    "scalar": "scalar broadcast",
    "per_soma_shared": "neuron specific",
    "path_transport": "exact path",
}
WITHIN_ORDER = ("scalar broadcast", "neuron specific", "exact path",
                "exact autograd")
BETWEEN_ORDER = ("readout backprop", "dfa")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean(
    values: np.ndarray, seed: int, draws: int = 50_000
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True)
    sampled = sampled.mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def exact_sign_flip_p(values: np.ndarray) -> float:
    """Two-sided exact sign-flip test on paired seed differences."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    observed = abs(values.mean())
    count = 0
    total = 1 << n
    for mask in range(total):
        signs = np.array([1.0 if mask & (1 << i) else -1.0 for i in range(n)])
        if abs((values * signs).mean()) >= observed - 1e-15:
            count += 1
    return count / total


def newest_sweeps() -> dict[str, Path]:
    """One sweep directory per (family, dynamics), newest timestamp wins."""
    sweeps: dict[str, Path] = {}
    for run in sorted(PROJECT_RUN_ROOT.glob("journal_mnist_factorial_*")):
        stem = run.name
        key = stem.rsplit("_", 1)[0]
        sweeps[key] = run
    return sweeps


def collect_new_runs() -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    errors: list[str] = []
    sweeps = newest_sweeps()
    expected_keys = [
        f"journal_mnist_factorial_{family}_{dynamics}_15seed"
        for family in ("dfa", "somadfa")
        for dynamics in ("additive", "shunting")
    ]
    for key in expected_keys:
        run = sweeps.get(key)
        if run is None:
            errors.append(f"missing sweep {key}")
            continue
        manifest = json.loads(
            (run / "frozen_sweep_manifest.json").read_text(encoding="utf-8")
        )
        for entry in manifest["generated_configs"]:
            index = entry["index"]
            config_path = run / entry["path"]
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            core = config["model"]["core"]["type"]
            seed = int(config["experiment"]["seed"])
            strategy = config["training"]["main"]["strategy"]
            lsc = config["training"]["main"]["learning_strategy_config"]
            if strategy == "soma_dfa":
                within = "exact autograd"
                broadcast = "soma_dfa"
            else:
                broadcast = str(lsc["error_broadcast_mode"])
                within = MODE_LABEL[broadcast]
                if str(lsc.get("soma_error_source", "decoder")) != "dfa":
                    errors.append(f"{key}/{config_path.name}: not a DFA run")
                    continue
            result_dir = run / "results" / f"config_{index}"
            final_path = result_dir / "performance" / "final.json"
            checkpoint = result_dir / "final_model.pt"
            if not final_path.is_file() or not checkpoint.is_file():
                errors.append(
                    f"{key}/{config_path.name}: missing final metric or checkpoint"
                )
                continue
            final = json.loads(final_path.read_text(encoding="utf-8"))
            accuracy = float(final["accuracy"]["test"])
            if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
                errors.append(
                    f"{key}/{config_path.name}: invalid accuracy {accuracy}"
                )
                continue
            rows.append(
                {
                    "architecture": CORE_LABEL[core],
                    "core": core,
                    "seed": seed,
                    "between": "dfa",
                    "within": within,
                    "broadcast_mode": broadcast,
                    "test_accuracy": accuracy,
                    "run_dir": str(run),
                    "config_index": index,
                    "config_sha256": entry["sha256"],
                    "result_sha256": sha256(final_path),
                    "checkpoint_sha256": sha256(checkpoint),
                    "cohort": "between_within_factorial_2026-09-04",
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for (architecture, within), part in frame.groupby(
            ["architecture", "within"]
        ):
            seeds = tuple(sorted(part.seed))
            if seeds != EXPECTED_SEEDS:
                errors.append(
                    f"{architecture}/{within}: seeds {seeds} != expected"
                )
    return frame, errors


def ladder_row() -> pd.DataFrame:
    """The frozen exact-readout row of the factorial, from the ladder release."""
    ladder = pd.read_csv(LADDER)
    frame = ladder[
        [
            "architecture",
            "core",
            "seed",
            "feedback",
            "broadcast_mode",
            "test_accuracy",
            "run_dir",
            "config_index",
            "config_sha256",
            "result_sha256",
            "checkpoint_sha256",
            "cohort",
        ]
    ].rename(columns={"feedback": "within"})
    frame.insert(3, "between", "readout backprop")
    return frame


def summarize(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, ((architecture, between, within), part) in enumerate(
        outcomes.groupby(["architecture", "between", "within"], sort=False)
    ):
        mean, low, high = bootstrap_mean(
            part.test_accuracy.to_numpy(float), seed=9100 + index
        )
        rows.append(
            {
                "architecture": architecture,
                "between": between,
                "within": within,
                "n_seeds": len(part),
                "mean_test_accuracy": mean,
                "ci95_low_test_accuracy": low,
                "ci95_high_test_accuracy": high,
            }
        )
    return pd.DataFrame(rows)


def paired_contrasts(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Seed-paired contrasts: the within-ladder under DFA, the between gap
    at matched within-rung, and the transport-versus-autograd anchor."""

    indexed = outcomes.set_index(["architecture", "between", "within", "seed"])

    def diff(architecture, a, b):
        left = indexed.loc[(architecture, *a)].test_accuracy.sort_index()
        right = indexed.loc[(architecture, *b)].test_accuracy.sort_index()
        return (left - right).to_numpy(float)

    contrasts = []
    specs = []
    for architecture in ("additive", "shunting"):
        specs += [
            (architecture, "dfa within: neuron - scalar",
             ("dfa", "neuron specific"), ("dfa", "scalar broadcast")),
            (architecture, "dfa within: exact path - neuron",
             ("dfa", "exact path"), ("dfa", "neuron specific")),
            (architecture, "dfa within: autograd - exact path",
             ("dfa", "exact autograd"), ("dfa", "exact path")),
            (architecture, "between at scalar: readout - dfa",
             ("readout backprop", "scalar broadcast"),
             ("dfa", "scalar broadcast")),
            (architecture, "between at neuron: readout - dfa",
             ("readout backprop", "neuron specific"),
             ("dfa", "neuron specific")),
            (architecture, "between at exact path: readout - dfa",
             ("readout backprop", "exact path"), ("dfa", "exact path")),
        ]
    for index, (architecture, label, a, b) in enumerate(specs):
        values = diff(architecture, a, b)
        mean, low, high = bootstrap_mean(values, seed=9700 + index)
        contrasts.append(
            {
                "architecture": architecture,
                "contrast": label,
                "n_seeds": len(values),
                "mean_difference": mean,
                "ci95_low": low,
                "ci95_high": high,
                "positive_seed_fraction": float((values > 0).mean()),
                "exact_sign_flip_p": exact_sign_flip_p(values),
            }
        )
    return pd.DataFrame(contrasts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Report run completeness without writing tables.",
    )
    args = parser.parse_args()

    new_runs, errors = collect_new_runs()
    done = 0 if new_runs.empty else len(new_runs)
    print(f"collected {done}/120 new factorial runs")
    if errors:
        print(f"{len(errors)} problems; first 10:")
        for error in errors[:10]:
            print("  " + error)
    if args.check_only or errors or new_runs.empty:
        if not args.check_only and (errors or new_runs.empty):
            raise SystemExit("factorial incomplete; tables not written")
        return

    outcomes = pd.concat([ladder_row(), new_runs], ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(OUTPUT / "seed_outcomes.csv", index=False)
    summarize(outcomes).to_csv(OUTPUT / "condition_summary.csv", index=False)
    paired_contrasts(outcomes).to_csv(
        OUTPUT / "paired_contrasts.csv", index=False
    )
    audit = {
        "new_runs": int(len(new_runs)),
        "ladder_rows_reused": int(len(ladder_row())),
        "expected_seeds": list(EXPECTED_SEEDS),
        "between_levels": list(BETWEEN_ORDER),
        "within_levels": list(WITHIN_ORDER),
        "note": (
            "Exact-readout row reused from the frozen ladder release; "
            "DFA rows trained with soma_error_source=dfa (LocalCA) and the "
            "soma_dfa trainer; feedback seed equals the run seed."
        ),
    }
    (OUTPUT / "audit.json").write_text(
        json.dumps(audit, indent=1) + "\n", encoding="utf-8"
    )
    print(f"wrote tables to {OUTPUT}")
    with pd.option_context("display.width", 200):
        print(summarize(outcomes).to_string(index=False))


if __name__ == "__main__":
    main()
