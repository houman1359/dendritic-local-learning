from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_cifar10_credit_ladder_pilot.py"
SPEC = importlib.util.spec_from_file_location("analyze_cifar10_credit_ladder_pilot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_confirmation_gate_is_architecture_specific():
    rows = []
    for architecture in ("additive", "shunting"):
        for offset, seed in enumerate(MODULE.EXPECTED_SEEDS):
            scalar = 0.20 + 0.001 * offset
            neuron = scalar + 0.05
            exact = neuron + (0.012 if architecture == "shunting" else 0.004)
            bp = exact + 0.003
            for feedback, accuracy in zip(MODULE.FEEDBACK_ORDER, (scalar, neuron, exact, bp)):
                rows.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "feedback": feedback,
                        "test_accuracy": accuracy,
                    }
                )
    _, contrasts, decision = MODULE.summarize(pd.DataFrame(rows))
    assert not decision["gate_by_architecture"]["additive"]["passes"]
    assert decision["gate_by_architecture"]["shunting"]["passes"]
    assert decision["eligible_for_fresh_ten_seed_confirmation"]
    primary = contrasts[contrasts.contrast.eq("exact path minus neuron specific")]
    assert np.isclose(primary.set_index("architecture").loc["shunting", "mean_difference"], 0.012)


def test_execution_identity_accepts_quoted_and_unquoted_values(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    manifest = tmp_path / "original_manifest.yaml"
    manifest.write_text("study: test\n")
    launcher = jobs / "run_array_sweep.sh"
    launcher.write_text(
        "REPOSITORY_ROOT=/path/to/clean-worktree\n"
        'EXPECTED_REPOSITORY_HEAD="abc123"\n'
        'EXPECTED_TRACKED_DIFF_SHA256="deadbeef"\n'
    )
    identity = MODULE.execution_identity(tmp_path)
    assert identity["source_worktree"] == "/path/to/clean-worktree"
    assert identity["source_commit"] == "abc123"
    assert identity["source_tracked_diff_sha256"] == "deadbeef"
