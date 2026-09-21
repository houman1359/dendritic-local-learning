"""
Scripts Package

This module contains main training and experiment scripts.
"""

from typing import Any


def train_experiments_main(*args: Any, **kwargs: Any):
    """Lazy import wrapper for the main training entrypoint."""
    from dendritic_modeling.scripts.training.train_experiments import main

    return main(*args, **kwargs)


def run_analysis(*args: Any, **kwargs: Any):
    """Lazy import wrapper for sweep/run analysis entrypoint."""
    from dendritic_modeling.scripts.training.train_experiments import (
        run_analysis as _run,
    )

    return _run(*args, **kwargs)


__all__ = [
    "run_analysis",
    "train_experiments_main",
]
