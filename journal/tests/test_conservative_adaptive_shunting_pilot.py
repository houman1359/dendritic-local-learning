from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_conservative_adaptive_shunting_pilot.py"
SPEC = importlib.util.spec_from_file_location("conservative_shunting_pilot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODEL
SPEC.loader.exec_module(MODEL)
CFG = json.loads(MODEL.ADAPTIVE.CONFIG.read_text(encoding="utf-8"))


def test_dose_zero_is_unity_and_dose_one_is_original_gain() -> None:
    signal = np.linspace(0.1, 1.0, 8)
    noise = np.linspace(1.0, 0.1, 8)
    original = MODEL.ORIGINAL_GAIN_FROM_MOMENTS(signal, noise, CFG)

    zero = json.loads(json.dumps(CFG))
    zero["estimator"]["attenuation_dose"] = 0.0
    assert np.allclose(
        MODEL.conservative_gain_from_moments(signal, noise, zero),
        np.ones_like(original),
    )

    one = json.loads(json.dumps(CFG))
    one["estimator"]["attenuation_dose"] = 1.0
    assert np.allclose(
        MODEL.conservative_gain_from_moments(signal, noise, one),
        original,
    )


def test_intermediate_dose_stays_between_original_and_unity() -> None:
    signal = np.linspace(0.1, 1.0, 8)
    noise = np.linspace(1.0, 0.1, 8)
    original = MODEL.ORIGINAL_GAIN_FROM_MOMENTS(signal, noise, CFG)
    middle = json.loads(json.dumps(CFG))
    middle["estimator"]["attenuation_dose"] = 0.25
    observed = MODEL.conservative_gain_from_moments(signal, noise, middle)
    assert np.all(observed >= original)
    assert np.all(observed <= 1.0)

