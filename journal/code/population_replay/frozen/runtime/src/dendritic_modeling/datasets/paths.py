"""Shared dataset path and split-seed helpers."""

import os

import torch


def get_data_directory(data_type, task_data_path=None):
    """
    Get or create a data directory for the specified data type.

    Args:
        data_type (str): Type of data (e.g., 'mnist', 'cifar')
        task_data_path (str, optional): Path from task.data_path config

    Returns:
        str: Path to the data directory
    """
    if task_data_path and task_data_path.strip():
        data_path = os.path.join(task_data_path, "datasets", data_type)
    else:
        data_path = os.path.join("./data", data_type)

    os.makedirs(data_path, exist_ok=True)
    return data_path


def split_generator(seed: int | None = 0) -> torch.Generator:
    """Return a CPU generator for reproducible train/validation splits."""
    return torch.Generator().manual_seed(int(seed or 0))


__all__ = [
    "get_data_directory",
    "split_generator",
]
