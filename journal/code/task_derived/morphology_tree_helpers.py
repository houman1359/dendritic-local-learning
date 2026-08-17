"""Minimal tree helpers used by the task-derived MICrONS analysis.

These functions are copied without algorithmic change from
``drafts/dendritic-credit-routing/analysis/analyze_microns_morphology_credit.py``.
They are kept here to make ``run_task_derived_credit_learning.py`` runnable
without importing the much larger structural-capacity analysis.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def parent_map(
    segments: pd.DataFrame,
) -> tuple[int, dict[int, int], dict[int, list[int]]]:
    """Return the root, parent map, and child map for one segment tree."""

    root_rows = segments[segments["parent_segment_id"] < 0]
    if len(root_rows) != 1:
        raise ValueError(f"expected one root segment, found {len(root_rows)}")
    root = int(root_rows.iloc[0]["segment_id"])
    parents: dict[int, int] = {}
    children: dict[int, list[int]] = defaultdict(list)
    for segment, parent in segments[
        ["segment_id", "parent_segment_id"]
    ].itertuples(index=False):
        if int(parent) >= 0:
            parents[int(segment)] = int(parent)
            children[int(parent)].append(int(segment))
    return root, parents, children


def ancestry_matrix(
    row_segments: list[int],
    column_segments: list[int],
    parent: dict[int, int],
) -> np.ndarray:
    """Mark whether each column segment lies on a row segment's soma path."""

    matrix = np.zeros((len(row_segments), len(column_segments)), dtype=float)
    columns = {segment: index for index, segment in enumerate(column_segments)}
    for row_index, segment in enumerate(row_segments):
        current = int(segment)
        while True:
            if current in columns:
                matrix[row_index, columns[current]] = 1.0
            if current not in parent:
                break
            current = parent[current]
    return matrix


__all__ = ["ancestry_matrix", "parent_map"]
