"""Stream a MICrONS DANDI NWB file and export matched trial responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import fsspec
import h5py
import numpy as np
import pandas as pd
import requests


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    PROJECT
    / "external_data"
    / "microns_dandi_trial_manifest"
    / "microns_dandi_trial_manifest.csv"
)
DEFAULT_OUTDIR = PROJECT / "reproduced_results" / "microns_dandi_trial_extract"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def decode_array(values: np.ndarray) -> np.ndarray:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(value)
    return np.asarray(out)


def read_interval_table(h5: h5py.File, name: str) -> pd.DataFrame:
    group = h5["intervals"][name]
    n_rows = int(group["start_time"].shape[0])
    data: dict[str, Any] = {"stimulus_family": np.asarray([name] * n_rows, dtype=object)}
    for key in group.keys():
        obj = group[key]
        if not isinstance(obj, h5py.Dataset):
            continue
        arr = obj[:]
        if arr.dtype.kind in {"S", "O"}:
            arr = decode_array(arr)
        data[key] = arr
    return pd.DataFrame(data)


NUMERIC_FEATURE_COLUMNS = (
    "duration",
    "blue_green_saturation",
    "num_directions",
    "ori_coherence",
    "ori_fraction",
    "ori_mix",
    "pattern_aspect",
    "pattern_width",
    "rng_seed",
    "temp_bandwidth",
    "spatial_freq",
    "temp_freq",
    "temp_kernel_length",
    "texture_height",
    "texture_width",
    "up_factor",
    "xnodes",
    "ynodes",
)

CATEGORICAL_FEATURE_COLUMNS = (
    "stimulus_family",
    "stimulus_type",
    "temp_kernel",
    "short_movie_name",
    "movie_name",
)


def one_hot(values: pd.Series, prefix: str) -> tuple[np.ndarray, list[str]]:
    filled = values.astype("object").where(values.notna(), "missing").astype(str)
    categories = sorted(filled.unique().tolist())
    lookup = {value: idx for idx, value in enumerate(categories)}
    arr = np.zeros((len(filled), len(categories)), dtype=float)
    arr[np.arange(len(filled)), [lookup[value] for value in filled]] = 1.0
    names = [f"{prefix}={value}" for value in categories]
    return arr, names


def positive_numeric_features(trials: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    blocks = []
    names = []
    for col in NUMERIC_FEATURE_COLUMNS:
        if col not in trials.columns:
            continue
        values = pd.to_numeric(trials[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.any():
            continue
        filled = np.where(finite, values, 0.0)
        min_value = float(np.nanmin(filled))
        if min_value < 0.0:
            filled = filled - min_value
        scale = float(np.nanpercentile(np.abs(filled[finite]), 95.0))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        blocks.append((filled / scale)[:, None])
        names.append(col)
        if not finite.all():
            blocks.append((~finite).astype(float)[:, None])
            names.append(f"{col}:missing")
    if not blocks:
        return np.zeros((len(trials), 0), dtype=float), []
    return np.hstack(blocks), names


def metadata_features(trials: pd.DataFrame, include_hash: bool) -> tuple[np.ndarray, list[str]]:
    blocks = []
    names = []
    numeric, numeric_names = positive_numeric_features(trials)
    if numeric.shape[1]:
        blocks.append(numeric)
        names.extend(numeric_names)
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col not in trials.columns:
            continue
        encoded, encoded_names = one_hot(trials[col], col)
        blocks.append(encoded)
        names.extend(encoded_names)
    if include_hash:
        encoded, encoded_names = one_hot(trials["condition_hash"], "condition_hash")
        blocks.append(encoded)
        names.extend(encoded_names)
    if not blocks:
        encoded, encoded_names = one_hot(trials["condition_hash"], "condition_hash")
        blocks.append(encoded)
        names.extend(encoded_names)
    return np.hstack(blocks), names


def make_stimulus_features(
    trials: pd.DataFrame,
    n_branches: int,
    feature_mode: str = "hash",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hashes = trials["condition_hash"].astype(str).fillna("missing").to_numpy()
    unique = {value: idx for idx, value in enumerate(sorted(set(hashes)))}
    stimulus_ids = np.asarray([unique[value] for value in hashes], dtype=int)
    mode = feature_mode.lower()
    if mode == "hash":
        features = np.zeros((len(hashes), len(unique)), dtype=float)
        features[np.arange(len(hashes)), stimulus_ids] = 1.0
        feature_names = [f"condition_hash={value}" for value in sorted(unique)]
    elif mode == "metadata":
        features, feature_names = metadata_features(trials, include_hash=False)
    elif mode == "metadata_hash":
        features, feature_names = metadata_features(trials, include_hash=True)
    else:
        raise ValueError("feature_mode must be one of hash, metadata, metadata_hash")
    branch_ids = np.arange(features.shape[1], dtype=int) % max(1, int(n_branches))
    return stimulus_ids, features, branch_ids, np.asarray(feature_names, dtype="U")


def dandi_signed_url(asset_id: str) -> str:
    response = requests.get(
        f"https://api.dandiarchive.org/api/assets/{asset_id}/download/",
        allow_redirects=False,
        timeout=60,
    )
    response.raise_for_status()
    if "Location" not in response.headers:
        raise RuntimeError("DANDI download endpoint did not return a signed Location header")
    return response.headers["Location"]


def field_counts(h5: h5py.File) -> list[int]:
    fluorescence = h5["processing"]["ophys"]["Fluorescence"]
    counts = []
    for field in range(1, 9):
        key = f"RoiResponseSeries{field}"
        if key in fluorescence:
            counts.append(int(fluorescence[key]["data"].shape[1]))
        else:
            counts.append(0)
    return counts


def resolve_local_column(row: pd.Series, starts: np.ndarray, counts: list[int]) -> tuple[int | None, str]:
    field = int(row["field"])
    unit_id = int(row["unit_id"])
    if field < 1 or field > len(counts):
        return None, "field_out_of_range"
    local = unit_id - 1 - int(starts[field - 1])
    if 0 <= local < counts[field - 1]:
        return int(local), "global_1based_unit_id"
    local = unit_id - int(starts[field - 1])
    if 0 <= local < counts[field - 1]:
        return int(local), "global_0based_unit_id"
    local = unit_id - 1
    if 0 <= local < counts[field - 1]:
        return int(local), "field_1based_unit_id"
    local = unit_id
    if 0 <= local < counts[field - 1]:
        return int(local), "field_0based_unit_id"
    return None, "unresolved"


def interval_means(timestamps: np.ndarray, traces: np.ndarray, trials: pd.DataFrame, start_offset: float, stop_offset: float) -> np.ndarray:
    responses = np.full((len(trials), traces.shape[1]), np.nan, dtype=float)
    for trial_idx, trial in trials.iterrows():
        start = float(trial["start_time"]) + float(start_offset)
        stop = float(trial["stop_time"]) - float(stop_offset)
        if stop <= start:
            start = float(trial["start_time"])
            stop = float(trial["stop_time"])
        lo = int(np.searchsorted(timestamps, start, side="left"))
        hi = int(np.searchsorted(timestamps, stop, side="right"))
        if hi <= lo:
            continue
        responses[int(trial_idx), :] = np.nanmean(traces[lo:hi, :], axis=0)
    return responses


def run_extract(args: argparse.Namespace) -> dict[str, Any]:
    manifest = pd.read_csv(args.manifest)
    if args.session is None or args.scan_idx is None:
        grouped = (
            manifest.loc[manifest["has_dandi_asset"] == True]  # noqa: E712
            .groupby(["session", "scan_idx"], dropna=False)
            .agg(n_targets=("nucleus_id", "nunique"))
            .reset_index()
            .sort_values("n_targets", ascending=False)
        )
        if grouped.empty:
            raise ValueError("manifest has no rows with DANDI assets")
        args.session = int(grouped.iloc[0]["session"])
        args.scan_idx = int(grouped.iloc[0]["scan_idx"])

    rows = manifest[(manifest["session"] == args.session) & (manifest["scan_idx"] == args.scan_idx) & (manifest["has_dandi_asset"] == True)].copy()  # noqa: E712
    if rows.empty:
        raise ValueError(f"no DANDI-backed rows for session={args.session}, scan_idx={args.scan_idx}")
    rows = rows.sort_values(["residual", "score"], ascending=[True, False]).drop_duplicates("nucleus_id")
    asset_id = str(rows["dandi_asset_id"].dropna().iloc[0])
    dandi_path = str(rows["dandi_path"].dropna().iloc[0])
    signed_url = dandi_signed_url(asset_id)

    args.outdir.mkdir(parents=True, exist_ok=True)
    fs = fsspec.filesystem("http", block_size=args.block_size)
    with fs.open(signed_url, "rb") as file_obj, h5py.File(file_obj, "r") as h5:
        counts = field_counts(h5)
        starts = np.cumsum([0] + counts[:-1])
        mapping_rows = []
        for _, row in rows.iterrows():
            local_col, mode = resolve_local_column(row, starts, counts)
            entry = row.to_dict()
            entry["local_column"] = local_col
            entry["mapping_mode"] = mode
            entry["field_n_rois"] = counts[int(row["field"]) - 1] if pd.notna(row["field"]) else np.nan
            mapping_rows.append(entry)
        mapping = pd.DataFrame(mapping_rows)
        mapping = mapping[mapping["local_column"].notna()].copy()
        if mapping.empty:
            raise ValueError("no functional units could be mapped to NWB trace columns")
        mapping["local_column"] = mapping["local_column"].astype(int)
        mapping = mapping.reset_index(drop=True)

        trials = pd.concat(
            [read_interval_table(h5, name) for name in ["Monet2", "Trippy", "Clip"] if name in h5["intervals"]],
            ignore_index=True,
            sort=False,
        )
        trials = trials.sort_values("start_time").reset_index(drop=True)

        responses = np.full((len(trials), len(mapping)), np.nan, dtype=float)
        for field, field_rows in mapping.groupby("field", dropna=False):
            field = int(field)
            series_key = f"RoiResponseSeries{field}"
            if series_key not in h5["processing"]["ophys"]["Fluorescence"]:
                continue
            fluorescence = h5["processing"]["ophys"]["Fluorescence"][series_key]
            cols = field_rows["local_column"].to_numpy(dtype=int)
            order = np.argsort(cols)
            sorted_cols = cols[order]
            traces = np.asarray(fluorescence["data"][:, sorted_cols], dtype=float)
            timestamps = np.asarray(fluorescence["timestamps"][:], dtype=float)
            field_responses = interval_means(
                timestamps,
                traces,
                trials,
                start_offset=args.start_offset,
                stop_offset=args.stop_offset,
            )
            target_positions = field_rows.index.to_numpy(dtype=int)
            for sorted_idx, original_idx in enumerate(order):
                responses[:, target_positions[original_idx]] = field_responses[:, sorted_idx]

    valid_trial_mask = np.isfinite(responses).all(axis=1)
    if args.drop_nan_trials:
        responses = responses[valid_trial_mask]
        trials = trials.loc[valid_trial_mask].reset_index(drop=True)

    stimulus_ids, features, branch_ids, feature_names = make_stimulus_features(
        trials,
        args.n_branches,
        feature_mode=args.feature_mode,
    )
    population_gain = np.nanmean(responses, axis=1, keepdims=True)

    payload = {
        "responses": responses,
        "stimulus_ids": stimulus_ids,
        "features": features,
        "feature_branch_ids": branch_ids,
        "feature_names": feature_names,
        "gain_covariates": population_gain,
        "unit_target_ids": mapping["nucleus_id"].to_numpy(dtype=int),
        "unit_root_ids": mapping["post_pt_root_id"].to_numpy(dtype=np.int64),
        "unit_session": mapping["session"].to_numpy(dtype=int),
        "unit_scan_idx": mapping["scan_idx"].to_numpy(dtype=int),
        "unit_field": mapping["field"].to_numpy(dtype=int),
        "unit_id": mapping["unit_id"].to_numpy(dtype=int),
    }
    np.savez_compressed(args.outdir / "microns_dandi_trial_protocol.npz", **payload)
    mapping.to_csv(args.outdir / "microns_dandi_trial_unit_mapping.csv", index=False)
    trials.to_csv(args.outdir / "microns_dandi_trial_table.csv", index=False)

    stimulus_counts = pd.Series(stimulus_ids).value_counts()
    summary = {
        "claim_level": "streamed DANDI/NWB trial-response protocol for CAVE-matched MICrONS cells",
        "session": int(args.session),
        "scan_idx": int(args.scan_idx),
        "dandi_asset_id": asset_id,
        "dandi_path": dandi_path,
        "n_units": int(responses.shape[1]),
        "n_trials": int(responses.shape[0]),
        "n_unique_stimuli": int(len(stimulus_counts)),
        "feature_mode": str(args.feature_mode),
        "n_features": int(features.shape[1]),
        "n_branches": int(len(np.unique(branch_ids))),
        "stimulus_counts_min": int(stimulus_counts.min()) if len(stimulus_counts) else 0,
        "stimulus_counts_max": int(stimulus_counts.max()) if len(stimulus_counts) else 0,
        "stimulus_family_counts": {str(k): int(v) for k, v in trials["stimulus_family"].value_counts().items()},
        "response_finite_fraction": float(np.isfinite(responses).mean()),
        "response_mean": float(np.nanmean(responses)),
        "response_std": float(np.nanstd(responses)),
        "mapping_modes": {str(k): int(v) for k, v in mapping["mapping_mode"].value_counts().items()},
        "outputs": {
            "protocol": str(args.outdir / "microns_dandi_trial_protocol.npz"),
            "unit_mapping": str(args.outdir / "microns_dandi_trial_unit_mapping.csv"),
            "trial_table": str(args.outdir / "microns_dandi_trial_table.csv"),
        },
    }
    write_json(args.outdir / "microns_dandi_trial_extract_summary.json", summary)
    lines = [
        "# MICrONS DANDI Trial Response Extract",
        "",
        summary["claim_level"],
        "",
        f"- Session/scan: {summary['session']}/{summary['scan_idx']}.",
        f"- Units: {summary['n_units']}.",
        f"- Trials: {summary['n_trials']}.",
        f"- Unique stimuli: {summary['n_unique_stimuli']}.",
        f"- Stimulus families: {summary['stimulus_family_counts']}.",
        f"- Finite response fraction: {summary['response_finite_fraction']:.4f}.",
        "",
        "Interpretation: this is a real streamed trial-response export from DANDI/NWB for CAVE-matched cells. It is an access and protocol smoke, not yet a full multi-session fitted model result.",
    ]
    (args.outdir / "microns_dandi_trial_extract_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--session", type=int)
    parser.add_argument("--scan-idx", type=int)
    parser.add_argument("--n-branches", type=int, default=8)
    parser.add_argument(
        "--feature-mode",
        choices=["hash", "metadata", "metadata_hash"],
        default="hash",
        help="Feature bank exported to the five-model fitter.",
    )
    parser.add_argument("--start-offset", type=float, default=0.0)
    parser.add_argument("--stop-offset", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=2**20)
    parser.add_argument("--drop-nan-trials", action="store_true")
    args = parser.parse_args()

    summary = run_extract(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
