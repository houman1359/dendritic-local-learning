#!/usr/bin/env python3
"""Build a disjoint MICrONS replication cohort from public version-661 files.

This path does not use CAVE authentication.  The cohort is fixed from the
prospective 55-cell list before routing outcomes are computed, and the eight
cells used in the original morphology pilot are excluded by stable nucleus ID.
Only direct presynaptic coarse E/I calls from the matching version-661 table
are retained by downstream analyses.  The public meshwork supplies synapse
positions and the public SWC supplies the dendritic tree from the same release.

The resulting directory matches the input schema expected by
``analyze_microns_morphology_credit.py``.  Raw meshworks are streamed through a
temporary file and are not retained; their URLs, byte sizes, and SHA-256 hashes
are written to the fetch manifest.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


JOURNAL = Path(__file__).resolve().parents[1]
REPO = JOURNAL.parents[2]
DEFAULT_SELECTION = JOURNAL / "data" / "microns_expanded" / "prospective_selection.csv"
DEFAULT_PILOT_MANIFEST = (
    REPO
    / "drafts"
    / "dendritic-credit-routing"
    / "data"
    / "microns_morphology"
    / "cell_manifest.csv"
)
DEFAULT_OUTDIR = JOURNAL / "data" / "microns_v661_replication"

TABLE_BASE = "https://storage.googleapis.com/mat_dbs/public/minnie65_phase3_v1/v661"
STATIC_BASE = (
    "https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/"
    "minnie65/skeletons/v661"
)
NUCLEUS_DATA = "nucleus_detection_v0_merged.csv.gz"
NUCLEUS_HEADER = "nucleus_detection_v0_merged_header.csv"
TYPE_DATA = "baylor_log_reg_cell_type_coarse_v1_merged.csv.gz"
TYPE_HEADER = "baylor_log_reg_cell_type_coarse_v1_merged_header.csv"


def download(url: str, timeout: int = 180) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "local-learning-journal/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_static_table(data_payload: bytes, header_payload: bytes) -> pd.DataFrame:
    header = pd.read_csv(io.BytesIO(header_payload), header=None, names=["column", "type"])
    with gzip.GzipFile(fileobj=io.BytesIO(data_payload)) as handle:
        frame = pd.read_csv(handle, header=None)
    frame.columns = header["column"].astype(str).tolist()
    return frame


def parse_swc(payload: bytes) -> pd.DataFrame:
    frame = pd.read_csv(
        io.BytesIO(payload),
        sep=r"\s+",
        comment="#",
        header=None,
        names=["id", "type", "x", "y", "z", "radius", "parent"],
    )
    if frame.empty:
        raise ValueError("empty SWC")
    # The static release distinguishes basal (3) and apical (4) dendrite.
    # The inherited compressor uses one dendrite code, so preserve both by
    # mapping apical nodes to the dendrite code.  Axon (2) remains excluded.
    frame["type"] = np.where(frame["type"].astype(int) == 4, 3, frame["type"].astype(int))
    return frame


def parse_postsynaptic_meshwork(payload: bytes) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".h5") as handle:
        handle.write(payload)
        handle.flush()
        with h5py.File(handle.name, "r") as meshwork:
            raw = meshwork["annotations/post_syn/data"][()]
    text = raw.decode("utf-8") if isinstance(raw, (bytes, np.bytes_)) else str(raw)
    frame = pd.read_json(io.StringIO(text))
    required = {
        "id",
        "size",
        "pre_pt_root_id",
        "post_pt_root_id",
        "post_pt_position",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"meshwork post_syn annotation missing columns: {missing}")
    return frame


def normalize_synapses(frame: pd.DataFrame, direct_types: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    for column in ["id", "pre_pt_root_id", "post_pt_root_id"]:
        rows[column] = pd.to_numeric(rows[column], errors="coerce").astype("Int64")
    rows["size"] = pd.to_numeric(rows["size"], errors="coerce")
    rows = rows.dropna(subset=["id", "pre_pt_root_id", "post_pt_root_id", "size", "post_pt_position"])
    rows[["id", "pre_pt_root_id", "post_pt_root_id"]] = rows[
        ["id", "pre_pt_root_id", "post_pt_root_id"]
    ].astype("int64")
    position = np.asarray(rows["post_pt_position"].tolist(), dtype=float)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"unexpected post_pt_position shape: {position.shape}")
    rows["post_pt_position_x"] = position[:, 0]
    rows["post_pt_position_y"] = position[:, 1]
    rows["post_pt_position_z"] = position[:, 2]

    types = direct_types[["pt_root_id", "cell_type"]].copy()
    types["pt_root_id"] = pd.to_numeric(types["pt_root_id"], errors="coerce").astype("Int64")
    types = types.dropna(subset=["pt_root_id"]).copy()
    types["pt_root_id"] = types["pt_root_id"].astype("int64")
    types = types.drop_duplicates("pt_root_id", keep="last").rename(
        columns={"pt_root_id": "pre_pt_root_id", "cell_type": "pre_cell_type"}
    )
    rows = rows.merge(types, on="pre_pt_root_id", how="left")
    calls = rows["pre_cell_type"].fillna("").astype(str).str.lower()
    rows["typed_class"] = np.where(
        calls.eq("excitatory"), "E", np.where(calls.eq("inhibitory"), "I", "?")
    )
    rows["target_tag"] = ""
    rows["target_tag_probability"] = np.nan
    rows["target_proxy_class"] = "?"
    rows["synapse_class"] = rows["typed_class"]
    rows["class_source"] = np.where(
        rows["typed_class"].isin(["E", "I"]), "presynaptic_cell_type_v661", "unclassified"
    )
    columns = [
        "id",
        "size",
        "pre_pt_root_id",
        "post_pt_root_id",
        "post_pt_position_x",
        "post_pt_position_y",
        "post_pt_position_z",
        "pre_cell_type",
        "typed_class",
        "target_tag",
        "target_tag_probability",
        "target_proxy_class",
        "synapse_class",
        "class_source",
    ]
    return rows[columns].copy()


def serializable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    selection = pd.read_csv(args.selection)
    pilot = pd.read_csv(args.pilot_manifest)
    pilot_nuclei = set(pd.to_numeric(pilot["nucleus_id"], errors="coerce").dropna().astype("int64"))

    table_payloads: dict[str, bytes] = {}
    for filename in [NUCLEUS_DATA, NUCLEUS_HEADER, TYPE_DATA, TYPE_HEADER]:
        table_payloads[filename] = download(f"{TABLE_BASE}/{filename}")
    nuclei = read_static_table(table_payloads[NUCLEUS_DATA], table_payloads[NUCLEUS_HEADER])
    direct_types = read_static_table(table_payloads[TYPE_DATA], table_payloads[TYPE_HEADER])
    nuclei["id"] = pd.to_numeric(nuclei["id"], errors="coerce").astype("Int64")
    nuclei["pt_root_id"] = pd.to_numeric(nuclei["pt_root_id"], errors="coerce").astype("Int64")
    nuclei = nuclei.dropna(subset=["id", "pt_root_id"]).copy()
    nuclei[["id", "pt_root_id"]] = nuclei[["id", "pt_root_id"]].astype("int64")
    selected_nuclei = set(pd.to_numeric(selection["nucleus_id"], errors="raise").astype("int64"))
    selected_mapping_counts = (
        nuclei[nuclei["id"].isin(selected_nuclei)]
        .groupby("id")["pt_root_id"]
        .nunique()
    )
    missing_nuclei = sorted(selected_nuclei - set(selected_mapping_counts.index.astype(int)))
    ambiguous_nuclei = sorted(
        int(value) for value in selected_mapping_counts[selected_mapping_counts.ne(1)].index
    )
    if missing_nuclei or ambiguous_nuclei:
        raise ValueError(
            "version-661 nucleus mapping is not one-to-one for the frozen selection: "
            f"missing={missing_nuclei}, ambiguous={ambiguous_nuclei}"
        )
    mapping = nuclei[["id", "pt_root_id"]].drop_duplicates("id", keep="last").rename(
        columns={"id": "nucleus_id", "pt_root_id": "v661_root_id"}
    )
    eligible = selection.merge(mapping, on="nucleus_id", how="left", validate="one_to_one")
    eligible["excluded_original_pilot"] = eligible["nucleus_id"].isin(pilot_nuclei)
    eligible["replication_eligible"] = (
        ~eligible["excluded_original_pilot"] & eligible["v661_root_id"].notna()
    )
    eligible["selection_basis"] = (
        "all V1 L2-L5 IT/ET cells in frozen 55-cell list; exclude original eight by nucleus ID"
    )
    eligible.to_csv(args.outdir / "eligibility_frozen_before_outcomes.csv", index=False)

    rows: list[dict[str, Any]] = []
    for index, cell in eligible[eligible["replication_eligible"]].reset_index(drop=True).iterrows():
        root_id = int(cell["v661_root_id"])
        nucleus_id = int(cell["nucleus_id"])
        stem = f"{root_id}_{nucleus_id}"
        swc_url = f"{STATIC_BASE}/skeletons/{stem}.swc"
        meshwork_url = f"{STATIC_BASE}/meshworks/{stem}.h5"
        print(f"[{index + 1}/{int(eligible['replication_eligible'].sum())}] nucleus {nucleus_id}", flush=True)
        record: dict[str, Any] = {
            "root_id": root_id,
            "v661_root_id": root_id,
            "nucleus_id": nucleus_id,
            "source_root_id": int(cell["source_root_id"]),
            "swc_url": swc_url,
            "meshwork_url": meshwork_url,
        }
        try:
            swc_payload = download(swc_url)
            meshwork_payload = download(meshwork_url)
            swc = parse_swc(swc_payload)
            synapses = normalize_synapses(parse_postsynaptic_meshwork(meshwork_payload), direct_types)
            swc.to_csv(args.outdir / f"skeleton_{root_id}.csv.gz", index=False, compression="gzip")
            synapses.to_csv(
                args.outdir / f"synapses_{root_id}.csv.gz", index=False, compression="gzip"
            )
            direct = synapses["typed_class"].isin(["E", "I"])
            record.update(
                {
                    "status": "fetched",
                    "n_skeleton_nodes": int(len(swc)),
                    "n_synapses": int(len(synapses)),
                    "n_direct_typed_synapses": int(direct.sum()),
                    "n_direct_e_synapses": int((synapses["typed_class"] == "E").sum()),
                    "n_direct_i_synapses": int((synapses["typed_class"] == "I").sum()),
                    "direct_type_coverage": float(direct.mean()),
                    "swc_bytes": int(len(swc_payload)),
                    "meshwork_bytes": int(len(meshwork_payload)),
                    "swc_sha256": sha256(swc_payload),
                    "meshwork_sha256": sha256(meshwork_payload),
                }
            )
        except Exception as exc:
            record.update(
                {"status": "error", "error_type": type(exc).__name__, "error": str(exc)[:1000]}
            )
        for column in ["cell_type", "functional_area", "has_coregistration", "has_digital_twin"]:
            if column in cell:
                record[column] = serializable(cell[column])
        rows.append(record)
        pd.DataFrame(rows).to_csv(args.outdir / "cell_manifest.csv", index=False)

    manifest = pd.DataFrame(rows)
    completed = manifest[manifest["status"] == "fetched"].copy()
    table_sources = {
        filename: {
            "url": f"{TABLE_BASE}/{filename}",
            "bytes": len(payload),
            "sha256": sha256(payload),
        }
        for filename, payload in table_payloads.items()
    }
    summary = {
        "status": "complete" if len(completed) == len(manifest) else "partial",
        "release": "MICrONS minnie65 version 661 static repository",
        "cave_authentication_used": False,
        "selection_frozen_before_routing_outcomes": True,
        "selection_universe": "55 V1 L2-L5 IT/ET cells in prospective_selection.csv",
        "exclusion_rule": "exclude the eight original pilot cells by stable nucleus ID",
        "n_selection_universe": int(len(eligible)),
        "n_original_pilot_excluded": int(eligible["excluded_original_pilot"].sum()),
        "n_replication_eligible": int(eligible["replication_eligible"].sum()),
        "n_completed": int(len(completed)),
        "n_total_postsynaptic_synapses": int(completed["n_synapses"].sum()),
        "n_total_direct_typed_synapses": int(completed["n_direct_typed_synapses"].sum()),
        "direct_type_coverage": float(
            completed["n_direct_typed_synapses"].sum() / completed["n_synapses"].sum()
        ),
        "classification_rule": "direct baylor_log_reg_cell_type_coarse_v1 call at version 661; no target-structure proxy",
        "table_sources": table_sources,
        "limitations": [
            "The cohort is disjoint from the original eight cells but comes from the same mouse.",
            "Eligibility is exhaustive only within a prior 64-cell compartment-annotation pilot, not the MICrONS population.",
            "Version 661 is a historical static reconstruction and is not identical to the current materialization.",
            "Direct presynaptic coarse E/I calls cover only a subset of incoming synapses.",
        ],
    }
    (args.outdir / "fetch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
