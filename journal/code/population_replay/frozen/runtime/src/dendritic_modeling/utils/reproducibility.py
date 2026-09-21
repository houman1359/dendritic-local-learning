"""Independent random-stream resolution for reproducible experiments."""

from __future__ import annotations

import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

_INDEPENDENT_SEED_FIELDS = (
    "dataset_seed",
    "split_seed",
    "model_seed",
    "topology_seed",
    "loader_seed",
    "evaluation_seed",
    "probe_seed",
)


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _set_config_value(config: Any, key: str, value: Any) -> None:
    if isinstance(config, dict):
        config[key] = value
    else:
        setattr(config, key, value)


@dataclass(frozen=True)
class ResolvedExperimentSeeds:
    """Concrete RNG seeds used by one experiment run."""

    seed: int
    dataset_seed: int
    split_seed: int
    model_seed: int
    topology_seed: int
    loader_seed: int
    evaluation_seed: int
    probe_seed: int
    strict_deterministic: bool

    def asdict(self) -> dict[str, int | bool]:
        return asdict(self)


def resolve_experiment_seeds(
    experiment_config: Any,
    *,
    write_back: bool = False,
) -> ResolvedExperimentSeeds:
    """Resolve optional RNG streams against the legacy ``seed`` field.

    Resolution happens after configuration/CLI overrides. This means a legacy
    config containing only ``seed`` retains one familiar control, while an
    explicit stream remains independent from later changes to the base seed.
    """
    base_seed = int(_config_value(experiment_config, "seed", 42))
    resolved_values: dict[str, int] = {}
    for field_name in _INDEPENDENT_SEED_FIELDS:
        configured = _config_value(experiment_config, field_name, None)
        resolved_values[field_name] = (
            base_seed if configured is None else int(configured)
        )

    resolved = ResolvedExperimentSeeds(
        seed=base_seed,
        strict_deterministic=bool(
            _config_value(experiment_config, "strict_deterministic", False)
        ),
        **resolved_values,
    )
    if write_back:
        for field_name, value in resolved.asdict().items():
            _set_config_value(experiment_config, field_name, value)
    return resolved


@contextmanager
def preserved_random_state():
    """Restore Python, NumPy, CPU, and visible CUDA RNGs after a block."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@contextmanager
def isolated_random_seed(seed: int):
    """Temporarily seed Python, NumPy, CPU, and visible CUDA RNGs.

    Analysis code still contains a mixture of explicit generators and legacy
    global draws. This context makes the latter reproducible without allowing
    evaluation to perturb the subsequent training RNG state.
    """
    with preserved_random_state():
        normalized_seed = int(seed) % ((1 << 63) - 1)
        random.seed(normalized_seed)
        np.random.seed(normalized_seed % (1 << 32))
        torch.manual_seed(normalized_seed)
        yield


__all__ = [
    "ResolvedExperimentSeeds",
    "isolated_random_seed",
    "preserved_random_state",
    "resolve_experiment_seeds",
]
