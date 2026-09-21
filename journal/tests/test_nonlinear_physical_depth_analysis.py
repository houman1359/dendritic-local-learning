from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_nonlinear_physical_depth_confirmatory.py"


def _module():
    spec = importlib.util.spec_from_file_location("physical_depth_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_sign_flip_and_bootstrap_are_deterministic() -> None:
    module = _module()
    values = np.arange(1.0, 11.0)
    assert module.exact_sign_flip_p(values) == 2 / 1024
    first = module.bootstrap_mean(values, 1234, draws=2000)
    second = module.bootstrap_mean(values, 1234, draws=2000)
    assert first == second
    assert first[1] < first[0] < first[2]


def test_depth_effect_is_paired_by_seed() -> None:
    module = _module()
    frame = pd.DataFrame(
        {
            "seed": [1, 1, 2, 2],
            "depth": [1, 3, 1, 3],
            "test_accuracy": [0.5, 0.7, 0.6, 0.9],
            "regime": ["aligned"] * 4,
            "mechanism": ["shunting"] * 4,
            "method": ["bp"] * 4,
            "transport": ["backpropagation"] * 4,
        }
    )
    effect = module.depth_effect(
        frame,
        regime="aligned",
        mechanism="shunting",
        method="bp",
        transport="backpropagation",
    )
    assert np.allclose(effect.sort_index().to_numpy(), [0.2, 0.3])


def test_claim_gate_requires_a_positive_interval_and_seed_signs() -> None:
    module = _module()
    names = [
        "bp_depth_d3_minus_d1__aligned",
        "bp_depth_interaction__aligned_minus_zero_alignment",
        "bp_depth_interaction__aligned_minus_sensor_shuffled",
        "bp_depth_interaction__aligned_minus_rewired_tree",
        "local_depth_d3_minus_d1__aligned__per_soma_shared",
        "local_depth_d3_minus_d1__aligned__path_transport",
        "local_depth_interaction__aligned_minus_rewired__per_soma_shared",
        "local_depth_interaction__aligned_minus_rewired__path_transport",
    ]
    contrasts = pd.DataFrame(
        {
            "contrast": names,
            "mean_difference": [0.2] * len(names),
            "ci95_low": [0.1] * len(names),
            "ci95_high": [0.3] * len(names),
            "positive_pairs": [10] * len(names),
        }
    )
    contrasts.loc[0, ["mean_difference", "ci95_low", "ci95_high", "positive_pairs"]] = [
        -0.2,
        -0.3,
        -0.1,
        0,
    ]
    audit = module.audit_inference(contrasts, {})
    gates = audit["claim_gates"]
    assert gates[names[0]]["claim_gate_pass"] is False
    assert all(gates[name]["claim_gate_pass"] is True for name in names[1:])
