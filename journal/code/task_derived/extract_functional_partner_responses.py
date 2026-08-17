#!/usr/bin/env python3
"""Stream DANDI responses for one morphology target and its connected inputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
PARTNERS = PROJECT / "reproduced_results" / "microns_functional_partner_manifest" / "functional_partner_manifest.csv"
TRIAL_MANIFEST = (
    PROJECT
    / "external_data"
    / "microns_dandi_trial_manifest"
    / "microns_dandi_trial_manifest.csv"
)
LOADER = Path(__file__).resolve().with_name("microns_dandi_nwb_trial_extract.py")


def load_extractor():
    spec = importlib.util.spec_from_file_location("imported_microns_extract", LOADER)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {LOADER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-nucleus-id", type=int, required=True)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--scan-idx", type=int, required=True)
    parser.add_argument("--match-tier", choices=["manual", "automatic_conservative"], default="automatic_conservative")
    parser.add_argument("--partners", type=Path, default=PARTNERS)
    parser.add_argument("--trial-manifest", type=Path, default=TRIAL_MANIFEST)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--drop-nan-trials", action="store_true")
    args = parser.parse_args()

    partners = pd.read_csv(args.partners)
    selected = partners[
        (partners["target_nucleus_id"] == args.target_nucleus_id)
        & (partners["session"] == args.session)
        & (partners["scan_idx"] == args.scan_idx)
        & (partners["match_tier"] == args.match_tier)
    ].copy()
    if selected.empty:
        raise ValueError("selected target/scan/tier has no connected partners")
    selected = selected.sort_values(["residual", "score"], ascending=[True, False]).drop_duplicates("pre_pt_root_id")
    extraction = selected.rename(
        columns={"pre_nucleus_id": "nucleus_id", "pre_pt_root_id": "post_pt_root_id"}
    ).copy()
    extraction["role"] = "presynaptic_partner"

    trial_manifest = pd.read_csv(args.trial_manifest)
    target = trial_manifest[
        (trial_manifest["nucleus_id"] == args.target_nucleus_id)
        & (trial_manifest["session"] == args.session)
        & (trial_manifest["scan_idx"] == args.scan_idx)
        & trial_manifest["has_dandi_asset"].eq(True)  # noqa: E712
    ].copy()
    if target.empty:
        raise ValueError("target response row is absent from the DANDI manifest")
    target = target.sort_values(["residual", "score"], ascending=[True, False]).head(1)
    target["role"] = "postsynaptic_target"
    target["target_nucleus_id"] = args.target_nucleus_id
    target["target_root_id"] = int(selected["target_root_id"].iloc[0])

    keep = sorted(set(extraction.columns) | set(target.columns))
    combined = pd.concat(
        [extraction.reindex(columns=keep), target.reindex(columns=keep)],
        ignore_index=True,
        sort=False,
    )
    combined["has_dandi_asset"] = True

    outdir = args.outdir or (
        PROJECT
        / "reproduced_results"
        / "microns_functional_partner_responses"
        / f"target{args.target_nucleus_id}_ses{args.session}_scan{args.scan_idx}_{args.match_tier}"
    )
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "extraction_manifest.csv"
    combined.to_csv(manifest_path, index=False)

    extractor = load_extractor()
    run_args = SimpleNamespace(
        manifest=manifest_path,
        outdir=outdir,
        session=args.session,
        scan_idx=args.scan_idx,
        n_branches=8,
        feature_mode="metadata",
        start_offset=0.0,
        stop_offset=0.0,
        block_size=2**20,
        drop_nan_trials=args.drop_nan_trials,
    )
    summary = extractor.run_extract(run_args)
    summary["target_nucleus_id"] = args.target_nucleus_id
    summary["match_tier"] = args.match_tier
    summary["n_presynaptic_partners_requested"] = int(len(extraction))
    (outdir / "partner_extract_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
