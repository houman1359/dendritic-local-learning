"""Lazy public interface for sweep plotters.

Plotter modules have different optional and scientific dependencies.  Loading
them on demand keeps focused plotting and test imports fast and lightweight.
"""

from __future__ import annotations

import importlib
from typing import Any

_PLOTTER_MODULES = {
    "AblationPlotter": "ablation_plotter",
    "BasePlotter": "base_plotter",
    "InformationPlotter": "information_plotter",
    "MultiplicativeGainPlotter": "multiplicative_gain_plotter",
    "NoisePlotter": "noise_plotter",
    "PerformancePlotter": "performance_plotter",
    "SNRPlotter": "snr_plotter",
    "WeightPlotter": "weight_plotter",
}


def __getattr__(name: str) -> Any:
    """Import a public plotter class on first access."""
    module_name = _PLOTTER_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_PLOTTER_MODULES))


__all__ = [
    "AblationPlotter",
    "BasePlotter",
    "InformationPlotter",
    "MultiplicativeGainPlotter",
    "NoisePlotter",
    "PerformancePlotter",
    "SNRPlotter",
    "WeightPlotter",
]
