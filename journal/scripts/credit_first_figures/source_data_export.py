"""Preserve exact biological identifiers in heterogeneous plotted-data rows."""

import numbers

import pandas as pd


ID_COLUMNS = ("root_id", "target_root_id", "pre_pt_root_id")


def exact_id_table(records):
    """Build a frame without allowing missing IDs to promote integers to float.

    MICrONS/Pinky identifiers exceed the exact integer range of binary64.
    Convert them before DataFrame construction, not after precision is lost.
    A floating-point identifier is rejected even when it looks integral.
    """
    rows = []
    for record in records:
        row = dict(record)
        for column in ID_COLUMNS:
            if column not in row or pd.isna(row[column]):
                continue
            value = row[column]
            if isinstance(value, numbers.Integral):
                row[column] = str(value)
            elif isinstance(value, str) and value.isdecimal():
                row[column] = value
            else:
                raise ValueError(f"{column} must be an exact integer or decimal string: {value!r}")
        rows.append(row)
    table = pd.DataFrame(rows)
    for column in ID_COLUMNS:
        if column in table:
            table[column] = table[column].astype("string")
    return table


def concat_exact_id_tables(frames):
    """Concatenate differently shaped frames before pandas can round IDs."""
    return exact_id_table(record for frame in frames
                          for record in frame.to_dict("records"))
