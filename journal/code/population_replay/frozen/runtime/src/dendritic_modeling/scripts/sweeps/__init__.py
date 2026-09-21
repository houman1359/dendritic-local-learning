"""Sweep management and analysis tools.

Keep this package initializer lightweight.  Importing a focused utility (for
example, a plotter) should not eagerly import the training stack and every
sweep analyzer.
"""

from __future__ import annotations

import importlib
from types import ModuleType

_LAZY_SUBMODULES = {"analyzers", "unified"}


def __getattr__(name: str) -> ModuleType:
    """Import public sweep subpackages on first access."""
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_LAZY_SUBMODULES))


__all__ = ["analyzers", "unified"]
