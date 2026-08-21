#!/usr/bin/env python3
"""Freeze the outcome-independent Pinky v185 second-animal cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "pinky_v185" / "soma_valence_v185.csv"
DEFAULT_OUTDIR = ROOT / "source_data" / "pinky_v185_replication"
N_CELLS = 12
RECORD = "https://doi.org/10.5281/zenodo.3710459"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_position(value: object) -> tuple[int, int, int]:
    numbers = [int(item) for item in re.findall(r"-?\d+", str(value))]
    if len(numbers) != 3:
        raise ValueError(f"Expected three coordinates, found {value!r}")
    return numbers[0], numbers[1], numbers[2]


def select(frame: pd.DataFrame, n_cells: int = N_CELLS) -> pd.DataFrame:
    eligible = frame.loc[frame["cell_type"].eq("e")].copy()
    eligible["root_id"] = pd.to_numeric(
        eligible["pt_root_id"], errors="raise"
    ).astype("uint64")
    positions = np.asarray(
        [parse_position(value) for value in eligible["pt_position"]], dtype=int
    )
    eligible[["x_vox", "y_vox", "z_vox"]] = positions
    eligible = eligible.drop_duplicates("root_id").sort_values(
        ["y_vox", "root_id"]
    )
    if len(eligible) < n_cells:
        raise ValueError(f"Only {len(eligible)} eligible cells for {n_cells} strata")

    center_x = float(eligible["x_vox"].median())
    center_z = float(eligible["z_vox"].median())
    scale_x = max(float(eligible["x_vox"].std(ddof=1)), 1.0)
    scale_z = max(float(eligible["z_vox"].std(ddof=1)), 1.0)
    rows = []
    for stratum, indices in enumerate(np.array_split(np.arange(len(eligible)), n_cells)):
        block = eligible.iloc[indices].copy()
        block["centrality"] = (
            ((block["x_vox"] - center_x) / scale_x) ** 2
            + ((block["z_vox"] - center_z) / scale_z) ** 2
        )
        chosen = block.sort_values(["centrality", "root_id"]).iloc[0].copy()
        chosen["selection_stratum"] = int(stratum)
        chosen["eligible_in_stratum"] = int(len(block))
        rows.append(chosen)
    selected = pd.DataFrame(rows).sort_values("selection_stratum").reset_index(drop=True)
    selected["selection_order"] = np.arange(len(selected), dtype=int)
    selected["x_nm"] = 4 * selected["x_vox"]
    selected["y_nm"] = 4 * selected["y_vox"]
    selected["z_nm"] = 40 * selected["z_vox"]
    selected["dataset"] = "MICrONS phase-1 Pinky v185"
    selected["animal"] = "P36 male mouse, independent of minnie65"
    selected["selection_status"] = "selected_before_mesh_or_synapse_outcomes"
    return selected[
        [
            "selection_order",
            "selection_stratum",
            "eligible_in_stratum",
            "root_id",
            "id",
            "cell_type",
            "x_vox",
            "y_vox",
            "z_vox",
            "x_nm",
            "y_nm",
            "z_nm",
            "centrality",
            "dataset",
            "animal",
            "selection_status",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    selected = select(frame)
    manifest_path = args.outdir / "cohort_manifest.csv"
    summary_path = args.outdir / "selection_summary.json"
    summary = {
        "status": "frozen_before_mesh_or_synapse_outcomes",
        "dataset": "MICrONS phase-1 Pinky v185",
        "record": RECORD,
        "source_file": args.input.name,
        "source_sha256": sha256(args.input),
        "n_rows": int(len(frame)),
        "n_eligible_excitatory": int(frame["cell_type"].eq("e").sum()),
        "n_selected": int(len(selected)),
        "selection_rule": (
            "12 equal-count strata along pt_position coordinate 2; within each, "
            "minimum normalized distance to the global coordinate-1/3 median; "
            "root ID tie-break"
        ),
        "mesh_or_synapse_outcomes_used_for_selection": False,
        "coordinates": "pt_position parsed in nominal [4,4,40] nm voxels",
        "selected_root_ids": [int(value) for value in selected["root_id"]],
    }
    manifest_csv = selected.to_csv(index=False)
    summary_json = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not manifest_path.is_file() or manifest_path.read_text() != manifest_csv:
            raise SystemExit(f"Frozen manifest mismatch: {manifest_path}")
        if not summary_path.is_file() or summary_path.read_text() != summary_json:
            raise SystemExit(f"Frozen summary mismatch: {summary_path}")
        return
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest_csv, encoding="utf-8")
    summary_path.write_text(summary_json, encoding="utf-8")
    print(manifest_path.relative_to(ROOT))
    print(summary_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
