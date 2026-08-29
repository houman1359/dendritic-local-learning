from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_supplementary_figure_s04_native.py"
SPEC = importlib.util.spec_from_file_location("build_s04_native", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_complete_package(path: Path) -> pd.DataFrame:
    labels = [name for name, *_ in MODULE.CONFIRMATORY_SPECS]
    rows = []
    offsets = {
        "strict scalar": 0.00,
        "neuron specific": 0.02,
        "exact path": 0.04,
        "backpropagation": 0.041,
    }
    for seed in range(10800, 10820):
        seed_offset = (seed - 10809.5) * 0.0002
        for label in labels:
            rows.append(
                {
                    "feedback": label,
                    "seed": seed,
                    "test_accuracy": 0.47 + seed_offset + offsets[label],
                }
            )
    outcomes = pd.DataFrame(rows)
    outcomes.to_csv(path / "seed_outcomes.csv", index=False)
    conditions = []
    for label in labels:
        values = outcomes.loc[outcomes.feedback.eq(label), "test_accuracy"]
        conditions.append(
            {
                "feedback": label,
                "n_seeds": len(values),
                "mean_test_accuracy": values.mean(),
            }
        )
    pd.DataFrame(conditions).to_csv(path / "condition_summary.csv", index=False)
    pd.DataFrame(
        {"contrast": sorted(MODULE.CONFIRMATORY_CONTRASTS)}
    ).to_csv(path / "paired_contrasts.csv", index=False)
    summary = {
        "contract_frozen_before_confirmatory_outcomes": True,
        "audit": {
            "status": "complete_and_validated",
            "integrity_valid": True,
            "convergence_valid": True,
            "n_expected": 80,
            "n_results_complete": 80,
        },
        "decision": {"audit_passes": True, "main_promotion": False},
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return outcomes


def test_loader_accepts_complete_negative_result(tmp_path: Path) -> None:
    expected = write_complete_package(tmp_path)
    observed, summary = MODULE.load_confirmatory_analysis(tmp_path)
    assert len(observed) == 80
    assert set(observed["seed"]) == set(expected["seed"])
    assert summary["decision"]["main_promotion"] is False


def test_loader_rejects_missing_package(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="finalized analyzer package"):
        MODULE.load_confirmatory_analysis(tmp_path)


def test_loader_rejects_convergence_flag(tmp_path: Path) -> None:
    write_complete_package(tmp_path)
    summary_path = tmp_path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["audit"]["status"] = "complete_with_convergence_flags"
    summary["audit"]["convergence_valid"] = False
    summary["decision"]["audit_passes"] = False
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="convergence audit did not pass"):
        MODULE.load_confirmatory_analysis(tmp_path)


def test_loader_rejects_unpaired_seed(tmp_path: Path) -> None:
    write_complete_package(tmp_path)
    outcomes_path = tmp_path / "seed_outcomes.csv"
    outcomes = pd.read_csv(outcomes_path)
    outcomes.loc[
        outcomes.feedback.eq("strict scalar") & outcomes.seed.eq(10800), "seed"
    ] = 99999
    outcomes.to_csv(outcomes_path, index=False)
    with pytest.raises(RuntimeError, match="same seed set"):
        MODULE.load_confirmatory_analysis(tmp_path)


def test_panel_draws_four_means_and_twenty_seed_trajectories(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    write_complete_package(tmp_path)
    outcomes, _ = MODULE.load_confirmatory_analysis(tmp_path)
    fig, ax = plt.subplots()
    MODULE.panel_confirmatory_cifar(ax, outcomes)
    # Twenty paired trajectories plus four mean/CI errorbar line artists.
    assert len(ax.lines) >= 24
    assert len(ax.get_xticklabels()) == 4
    plt.close(fig)


def test_extended_build_requires_and_uses_validated_package(tmp_path: Path) -> None:
    write_complete_package(tmp_path)
    output = tmp_path / "figure_S04_panels_A-D.pdf"
    problems = MODULE.build(output, confirmatory_analysis_dir=tmp_path)
    assert output.is_file()
    assert not problems
