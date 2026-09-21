#!/usr/bin/env python3
"""Fetch a prospectively defined expanded MICrONS excitatory-cell cohort.

The selection rule is fixed before any journal-level routing outcome is
computed: include every morphologically summarized V1 excitatory neuron with
a stable nucleus identifier, then sort by layer/type and nucleus identifier.
``--max-cells`` is an operational cap only; zero means the full eligible
cohort.  The network and cache implementation is reused verbatim from the
earlier morphology project.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


JOURNAL = Path(__file__).resolve().parents[1]
REPO = JOURNAL.parents[2]
SOURCE_SCRIPT = REPO / "drafts" / "dendritic-credit-routing" / "analysis" / "fetch_microns_morphologies.py"
DEFAULT_ANATOMY = (
    REPO
    / "drafts"
    / "dendritic-credit-routing"
    / "imported"
    / "population"
    / "results"
    / "microns_rich_anatomy_summary"
    / "microns_rich_anatomy_summary.csv"
)
DEFAULT_OUTDIR = JOURNAL / "data" / "microns_expanded"


def load_source():
    spec = importlib.util.spec_from_file_location("routing_fetch", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_cells(path: Path, max_cells: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["post_pt_root_id"] = pd.to_numeric(frame["post_pt_root_id"], errors="coerce").astype("Int64")
    frame["nucleus_id"] = pd.to_numeric(frame["nucleus_id"], errors="coerce").astype("Int64")
    cell_type = frame["cell_type"].fillna("").astype(str)
    eligible = frame[
        frame["post_pt_root_id"].notna()
        & frame["nucleus_id"].notna()
        & cell_type.str.match(r"^L[2345](?:IT|ET)$")
        & frame["functional_area"].fillna("").eq("V1")
    ].copy()
    eligible["post_pt_root_id"] = eligible["post_pt_root_id"].astype("int64")
    eligible["nucleus_id"] = eligible["nucleus_id"].astype("int64")
    eligible = eligible.sort_values(["cell_type", "nucleus_id", "post_pt_root_id"]).reset_index(drop=True)
    if int(max_cells) > 0:
        eligible = eligible.head(int(max_cells)).copy()
    eligible["selection_order"] = np.arange(len(eligible), dtype=int)
    keep = [
        "post_pt_root_id",
        "cell_type",
        "functional_area",
        "has_coregistration",
        "has_digital_twin",
        "nucleus_id",
        "soma_position",
        "n_synapses",
        "soma_distance_um_q90",
        "selection_order",
    ]
    eligible = eligible[[col for col in keep if col in eligible.columns]].copy()
    return eligible.rename(columns={"post_pt_root_id": "source_root_id", "n_synapses": "previous_compartment_synapses"})


def serializable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--anatomy-csv", type=Path, default=DEFAULT_ANATOMY)
    parser.add_argument("--auth-secret-json", type=Path)
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--prediction-chunk-size", type=int, default=1000)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    source = load_source()
    args.outdir.mkdir(parents=True, exist_ok=True)
    selected = select_cells(args.anatomy_csv, args.max_cells)
    selected.to_csv(args.outdir / "prospective_selection.csv", index=False)
    client = source.make_client(args.auth_secret_json)
    cells = source.resolve_current_roots(client, selected)
    coarse = source.fetch_coarse_cell_types(client, args.outdir, args.refresh)

    rows: list[dict[str, Any]] = []
    for i, cell in cells.reset_index(drop=True).iterrows():
        root_id = int(cell["post_pt_root_id"])
        print(f"[{i + 1}/{len(cells)}] root {root_id} ({cell.get('cell_type', '?')})", flush=True)
        try:
            result = source.fetch_one_cell(
                client,
                root_id,
                args.outdir,
                coarse,
                args.prediction_chunk_size,
                args.refresh,
            )
        except Exception as exc:
            result = {"root_id": root_id, "status": "error", "error_type": type(exc).__name__, "error": str(exc)[:1000]}
        for column in cells.columns:
            result[column] = serializable(cell[column])
        rows.append(result)
        pd.DataFrame(rows).to_csv(args.outdir / "cell_manifest.csv", index=False)

    manifest = pd.DataFrame(rows)
    completed = manifest[manifest.status.isin(["fetched", "cached"])]
    summary = {
        "status": "complete" if len(completed) == len(manifest) else "partial",
        "selection_rule": "all summarized V1 L2/L3/L4/L5 IT or ET cells with stable nucleus identifiers, sorted by cell type and nucleus identifier",
        "selection_frozen_before_outcome_analysis": True,
        "operational_max_cells": int(args.max_cells),
        "datastack": "minnie65_public",
        "materialization_version": int(client.materialize.version),
        "n_eligible_cells": int(len(select_cells(args.anatomy_csv, 0))),
        "n_requested_cells": int(len(manifest)),
        "n_completed_cells": int(len(completed)),
        "n_total_synapses": int(pd.to_numeric(completed.get("n_synapses", 0), errors="coerce").fillna(0).sum()),
        "n_total_skeleton_nodes": int(pd.to_numeric(completed.get("n_skeleton_nodes", 0), errors="coerce").fillna(0).sum()),
        "class_rule": "presynaptic coarse E/I call when available; otherwise high-confidence spine/shaft/soma target proxy",
        "target_prediction_table": "synapse_target_predictions_ssa_v2",
    }
    (args.outdir / "fetch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
