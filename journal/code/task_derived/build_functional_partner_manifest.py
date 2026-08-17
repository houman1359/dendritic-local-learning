#!/usr/bin/env python3
"""Join real MICrONS dendritic contacts to same-scan functional units.

The output is a coverage and extraction manifest, not a functional result.  It
uses expert/manual matches as the high-confidence tier and a residual-filtered
automatic table as an exploratory expansion tier.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MORPHOLOGY = PROJECT / "external_data" / "microns_morphology"
DEFAULT_MAPPED = PROJECT / "reproduced_results" / "microns_morphology_credit" / "mapped_synapses.csv.gz"
DEFAULT_SEGMENTS = PROJECT / "reproduced_results" / "microns_morphology_credit" / "segment_metrics.csv"
DEFAULT_TRIAL_MANIFEST = (
    PROJECT
    / "external_data"
    / "microns_dandi_trial_manifest"
    / "microns_dandi_trial_manifest.csv"
)
AUTO_TABLE = "coregistration_auto_phase3_fwd_apl_vess_combined_v2"
MANUAL_TABLE = "coregistration_manual_v4"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def auth_token(secret: Path | None) -> str | None:
    for name in ("CAVE_AUTH_TOKEN", "MICRONS_CAVE_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    if secret is not None and secret.exists():
        value = json.loads(secret.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            token = value.get("token")
            return token if isinstance(token, str) else None
    return None


def make_client(secret: Path | None):
    from caveclient import CAVEclient  # type: ignore

    token = auth_token(secret)
    return CAVEclient("minnie65_public", auth_token=token) if token else CAVEclient("minnie65_public")


def query_scan(client: Any, table: str, session: int, scan: int, cache: Path, refresh: bool) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{table}_ses{session}_scan{scan}.csv.gz"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    frame = client.materialize.query_table(
        table,
        filter_equal_dict={"session": int(session), "scan_idx": int(scan)},
        limit=200_000,
    )
    keep = [
        "target_id",
        "pt_root_id",
        "session",
        "scan_idx",
        "field",
        "unit_id",
        "residual",
        "score",
        "pt_position",
    ]
    frame = frame[[column for column in keep if column in frame.columns]].copy()
    frame.to_csv(path, index=False, compression="gzip")
    return frame


def aggregate_contact_rows(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_synapses_to_target": int(len(rows)),
        "n_mapped_synapses": int(rows["mapping_pass"].sum()),
        "n_contact_segments": int(rows.loc[rows["mapping_pass"], "segment_id"].nunique()),
        "contact_segment_ids": ";".join(
            str(int(value))
            for value in sorted(rows.loc[rows["mapping_pass"], "segment_id"].dropna().unique())
        ),
        "contact_path_um_mean": float(rows.loc[rows["mapping_pass"], "path_length_um"].mean()),
        "contact_depth_mean": float(rows.loc[rows["mapping_pass"], "topological_depth"].mean()),
        "contact_size_total": float(rows["size"].sum()),
        "direct_type_fraction": float(rows["typed_class"].isin(["E", "I"]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--morphology-dir", type=Path, default=DEFAULT_MORPHOLOGY)
    parser.add_argument("--mapped-synapses", type=Path, default=DEFAULT_MAPPED)
    parser.add_argument("--segment-metrics", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--trial-manifest", type=Path, default=DEFAULT_TRIAL_MANIFEST)
    parser.add_argument("--auth-secret-json", type=Path)
    parser.add_argument("--outdir", type=Path, default=PROJECT / "reproduced_results" / "microns_functional_partner_manifest")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.morphology_dir / "cell_manifest.csv")
    mapped = pd.read_csv(args.mapped_synapses)
    trials = pd.read_csv(args.trial_manifest)
    target_rows = trials[
        trials["nucleus_id"].isin(cells["nucleus_id"])
        & trials["has_dandi_asset"].eq(True)  # noqa: E712
    ].copy()
    scan_keys = sorted(
        {
            (int(session), int(scan))
            for session, scan in target_rows[["session", "scan_idx"]].itertuples(index=False, name=None)
        }
    )
    client = make_client(args.auth_secret_json)
    frames: dict[tuple[str, int, int], pd.DataFrame] = {}
    for table in (MANUAL_TABLE, AUTO_TABLE):
        for session, scan in scan_keys:
            frames[(table, session, scan)] = query_scan(
                client, table, session, scan, args.outdir / "coreg_cache", args.refresh
            )

    outputs: list[dict[str, Any]] = []
    for cell in cells.itertuples(index=False):
        target_synapses = mapped[
            (mapped["root_id"] == int(cell.root_id)) & mapped["mapping_pass"]
        ].copy()
        target_trials = target_rows[target_rows["nucleus_id"] == int(cell.nucleus_id)]
        for target_trial in target_trials.itertuples(index=False):
            session, scan = int(target_trial.session), int(target_trial.scan_idx)
            for tier, table in (("manual", MANUAL_TABLE), ("automatic_conservative", AUTO_TABLE)):
                coreg = frames[(table, session, scan)].copy()
                if tier == "automatic_conservative":
                    coreg = coreg[(coreg["residual"] <= 10.0) & (coreg["score"] >= 0.0)]
                coreg = coreg.sort_values(["residual", "score"], ascending=[True, False]).drop_duplicates("pt_root_id")
                connected = coreg[
                    coreg["pt_root_id"].isin(target_synapses["pre_pt_root_id"])
                    & coreg["pt_root_id"].ne(int(cell.root_id))
                ].copy()
                for match in connected.itertuples(index=False):
                    contact = target_synapses[target_synapses["pre_pt_root_id"] == int(match.pt_root_id)]
                    entry = {
                        "target_nucleus_id": int(cell.nucleus_id),
                        "target_root_id": int(cell.root_id),
                        "target_cell_type": str(cell.cell_type),
                        "match_tier": tier,
                        "coreg_table": table,
                        "pre_nucleus_id": int(match.target_id),
                        "pre_pt_root_id": int(match.pt_root_id),
                        "session": session,
                        "scan_idx": scan,
                        "field": int(match.field),
                        "unit_id": int(match.unit_id),
                        "residual": float(match.residual),
                        "score": float(match.score),
                        "dandi_asset_id": str(target_trial.dandi_asset_id),
                        "dandi_path": str(target_trial.dandi_path),
                        "has_dandi_asset": True,
                    }
                    entry.update(aggregate_contact_rows(contact))
                    outputs.append(entry)

    partners = pd.DataFrame(outputs)
    if partners.empty:
        raise RuntimeError("no connected functional partners were found")
    partners = partners.sort_values(
        ["match_tier", "target_nucleus_id", "session", "scan_idx", "residual"]
    )
    partners.to_csv(args.outdir / "functional_partner_manifest.csv", index=False)

    cohort = (
        partners.groupby(
            ["match_tier", "target_nucleus_id", "target_root_id", "target_cell_type", "session", "scan_idx"],
            dropna=False,
        )
        .agg(
            n_connected_roots=("pre_pt_root_id", "nunique"),
            n_synapses=("n_synapses_to_target", "sum"),
            n_mapped_synapses=("n_mapped_synapses", "sum"),
            n_contact_segments=("n_contact_segments", "sum"),
            median_residual=("residual", "median"),
        )
        .reset_index()
        .sort_values(["match_tier", "n_connected_roots"], ascending=[True, False])
    )
    cohort.to_csv(args.outdir / "functional_partner_cohorts.csv", index=False)
    summary = {
        "claim_level": "coverage audit for branch-mapped, same-scan functional presynaptic partners",
        "materialization_version": int(client.materialize.version),
        "n_morphology_cells": int(len(cells)),
        "n_cells_with_dandi_scans": int(target_rows["nucleus_id"].nunique()),
        "n_session_scans": int(len(scan_keys)),
        "tiers": {},
        "limitations": [
            "MICrONS calcium imaging measured excitatory neurons; inhibitory presynaptic drive is not directly observed.",
            "Automatic matches are exploratory even after residual/score filtering.",
            "Coverage is a small fraction of each target neuron's complete input set.",
        ],
    }
    for tier, tier_rows in partners.groupby("match_tier"):
        tier_cohorts = cohort[cohort["match_tier"] == tier]
        summary["tiers"][tier] = {
            "n_unique_connected_roots": int(tier_rows["pre_pt_root_id"].nunique()),
            "n_partner_rows": int(len(tier_rows)),
            "median_connected_roots_per_cohort": float(tier_cohorts["n_connected_roots"].median()),
            "max_connected_roots_per_cohort": int(tier_cohorts["n_connected_roots"].max()),
            "n_cohorts": int(len(tier_cohorts)),
        }
    write_json(args.outdir / "summary.json", summary)
    lines = [
        "# MICrONS functional-partner coverage",
        "",
        summary["claim_level"],
        "",
        f"- Morphology cells: {summary['n_morphology_cells']}.",
        f"- Cells with DANDI scans: {summary['n_cells_with_dandi_scans']}.",
        f"- Session/scans queried: {summary['n_session_scans']}.",
        "",
        "| match tier | cohorts | unique connected roots | median roots/cohort | maximum |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for tier, item in summary["tiers"].items():
        lines.append(
            f"| {tier} | {item['n_cohorts']} | {item['n_unique_connected_roots']} | "
            f"{item['median_connected_roots_per_cohort']:.1f} | {item['max_connected_roots_per_cohort']} |"
        )
    lines.extend(
        [
            "",
            "Manual matches are the primary evidence tier. Residual-filtered automatic matches are suitable for exploratory screening and cohort selection.",
            "The coverage audit does not establish functional clustering or shunting.",
        ]
    )
    (args.outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
