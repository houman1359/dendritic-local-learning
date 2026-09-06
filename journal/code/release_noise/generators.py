"""Explicit, portable dispatch for the two historical noise-task meanings."""
from __future__ import annotations

import inspect
import numpy as np
import torch
from torch.utils.data import random_split

from frozen_generators import NoisyLineDataset, load_noise_resilience_mnist


def legacy_noisy_lines(*, seed=42, image_size=784, train_size=8000,
                       valid_size=1000, test_size=1000, noise_level=0.2):
    """Recover the clean-rerun generator, including its large image default.

    The caller controls the run seed. NumPy and Torch states are restored after
    generation so this explicit loader does not reset a caller's model RNG.
    """
    state = np.random.get_state()
    try:
        np.random.seed(int(seed))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            sizes = [int(train_size), int(valid_size), int(test_size)]
            dataset = NoisyLineDataset(sum(sizes), int(image_size), float(noise_level))
            return tuple(random_split(dataset, sizes))
    finally:
        np.random.set_state(state)


def projected_noise_mnist(**kwargs):
    """Newly frozen reference for the intended projected-MNIST protocol."""
    result = load_noise_resilience_mnist(**kwargs)
    return result['train'], result['valid'], result['test']


def _value(config, key, default=None):
    return config.get(key, default) if hasattr(config, 'get') else getattr(config, key, default)


def get_unified_datasets(task_cfg, train_cfg=None, output_dir=None, outputs_cfg=None):
    """Compatibility hook for the installed core's optional dataset fallback.

    Set task.dataset to an explicit name, or parameters.release_generator for a
    historical configuration. No undocumented interpretation is selected.
    """
    name = _value(task_cfg, 'dataset', _value(task_cfg, 'dataset_name'))
    parameters = dict(_value(task_cfg, 'parameters', {}) or {})
    requested = parameters.pop('release_generator', None)
    if name == 'noise_resilience' and requested is None:
        raise ValueError('Ambiguous noise_resilience: explicitly choose legacy_noisy_lines '
                         'or projected_noise_mnist; see task_identity.json.')
    name = requested or name
    if name == 'legacy_noisy_lines':
        keys = inspect.signature(legacy_noisy_lines).parameters
        unknown = set(parameters) - set(keys)
        if unknown:
            raise ValueError(f'Legacy noisy-line parameters not interpreted: {sorted(unknown)}')
        return legacy_noisy_lines(**parameters)
    if name == 'projected_noise_mnist':
        parameters.setdefault('task_data_path', _value(task_cfg, 'data_path'))
        return projected_noise_mnist(**parameters)
    raise ValueError(f'Noise release adapter does not provide dataset {name!r}')
