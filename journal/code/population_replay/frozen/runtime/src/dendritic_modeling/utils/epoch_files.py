"""Utilities for discovering numbered epoch files."""

from __future__ import annotations

import os
from collections.abc import Iterable


def _parse_epoch_filename(filename: str, *, require_json: bool) -> int | None:
    if not filename.startswith("epoch"):
        return None

    if require_json:
        epoch_text = filename.removeprefix("epoch")
        if not epoch_text.endswith(".json"):
            return None
        epoch_text = epoch_text.removesuffix(".json")
    else:
        epoch_text = filename.replace("epoch", "")
        if epoch_text.endswith(".json"):
            epoch_text = epoch_text.removesuffix(".json")

    if not epoch_text.isdigit():
        return None
    return int(epoch_text)


def epoch_files_by_number(
    directory: str | os.PathLike[str],
    *,
    require_json: bool = True,
) -> list[tuple[int, str]]:
    """Return ``(epoch_number, filename)`` pairs sorted by epoch number.

    ``require_json=False`` preserves the legacy parser used by performance
    plots and multi-stage training, including extensionless ``epochN`` files.
    """
    if not os.path.exists(directory):
        return []

    epoch_files: list[tuple[int, str]] = []
    for filename in os.listdir(directory):
        epoch_number = _parse_epoch_filename(filename, require_json=require_json)
        if epoch_number is not None:
            epoch_files.append((epoch_number, filename))

    return sorted(epoch_files, key=lambda item: item[0])


def epoch_file_paths_by_number(
    directory: str | os.PathLike[str],
    *,
    require_json: bool = True,
) -> Iterable[tuple[int, str]]:
    """Yield ``(epoch_number, file_path)`` pairs sorted by epoch number."""
    for epoch_number, filename in epoch_files_by_number(
        directory,
        require_json=require_json,
    ):
        yield epoch_number, os.path.join(directory, filename)


__all__ = ["epoch_file_paths_by_number", "epoch_files_by_number"]
