"""Collect declared completed sweeps into checkpoint-level metric tables.

This command is the non-inferential boundary between standard sweep outputs and
config-driven checkpoint summaries.  It uses the package ``DataCollector``,
requires an explicit expected size for every sweep, attaches only declared
design labels, and records hashes for all collected configs and held-out
performance files.  It never aggregates across independently trained seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dendritic_modeling.scripts.sweeps.collectors import DataCollector
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity

SCHEMA_VERSION = "sweep_metric_collection_v1"
RELEASE_SCHEMA_VERSION = "sweep_metric_collection_release_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported sweep metric collection manifest: {path}")
    return payload


def _artifact_inventory(
    root: Path,
    frame: pd.DataFrame,
    *,
    include_training_summary: bool,
    include_performance_epochs: bool,
) -> tuple[list[dict[str, Any]], str]:
    if "config_id" not in frame.columns:
        raise ValueError(f"collected sweep lacks config_id: {root}")
    records: list[dict[str, Any]] = []
    for config_id in sorted(int(value) for value in frame["config_id"].tolist()):
        config_candidates = sorted((root / "configs").glob(f"*config_{config_id}.yaml"))
        if len(config_candidates) != 1:
            raise ValueError(
                f"expected one generated config for config_id={config_id}, found "
                f"{len(config_candidates)} under {root}"
            )
        performance = (
            root / "results" / f"config_{config_id}" / "performance" / "final.json"
        )
        if not performance.is_file():
            raise FileNotFoundError(performance)
        paths = [("config", config_candidates[0]), ("performance", performance)]
        if include_training_summary:
            training_summary = (
                root / "results" / f"config_{config_id}" / "training_summary.json"
            )
            if not training_summary.is_file():
                raise FileNotFoundError(training_summary)
            paths.append(("training_summary", training_summary))
        if include_performance_epochs:
            epoch_dir = performance.parent / "epochs"
            epoch_paths = sorted(
                epoch_dir.glob("epoch*.json"),
                key=_performance_epoch_number,
            )
            if not epoch_paths:
                raise FileNotFoundError(f"no performance epochs under {epoch_dir}")
            paths.extend(("performance_epoch", path) for path in epoch_paths)
        for role, path in paths:
            records.append(
                {
                    "config_id": config_id,
                    "role": role,
                    "path": str(path.relative_to(root)),
                    "sha256": _sha256(path),
                }
            )
    payload = "".join(
        f"{record['config_id']}\t{record['role']}\t{record['sha256']}\t{record['path']}\n"
        for record in records
    ).encode("utf-8")
    return records, hashlib.sha256(payload).hexdigest()


def _performance_epoch_number(path: Path) -> int:
    match = re.fullmatch(r"epoch(\d+)", path.stem)
    if match is None:
        raise ValueError(f"invalid performance epoch filename: {path}")
    return int(match.group(1))


def _nested_metric(payload: Mapping[str, Any], path: str) -> float:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(path)
        value = value[part]
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"non-finite performance epoch metric {path!r}")
    return numeric


def _performance_epoch_history_rows(
    *,
    root: Path,
    frame: pd.DataFrame,
    sweep_name: str,
    labels: Mapping[str, Any],
    fields: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Collect declared metrics from standard ``performance/epochs`` outputs."""
    if not fields:
        return []
    rows: list[dict[str, Any]] = []
    for record in frame[["config_id", "seed"]].to_dict("records"):
        config_id = int(record["config_id"])
        epoch_dir = root / "results" / f"config_{config_id}" / "performance" / "epochs"
        epoch_paths = sorted(
            epoch_dir.glob("epoch*.json"), key=_performance_epoch_number
        )
        if not epoch_paths:
            raise FileNotFoundError(f"no performance epochs under {epoch_dir}")
        epoch_numbers = [_performance_epoch_number(path) for path in epoch_paths]
        if len(epoch_numbers) != len(set(epoch_numbers)):
            raise ValueError(
                f"duplicate performance epoch numbers for {sweep_name} "
                f"config_id={config_id}"
            )
        for path, epoch_number in zip(epoch_paths, epoch_numbers, strict=True):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for source, metric in fields.items():
                try:
                    value = _nested_metric(payload, source)
                except KeyError as error:
                    raise ValueError(
                        f"performance epoch metric {source!r} is absent for "
                        f"{sweep_name} config_id={config_id} epoch={epoch_number}"
                    ) from error
                rows.append(
                    {
                        "sweep_name": sweep_name,
                        "sweep_root": str(root),
                        "config_id": config_id,
                        "seed": record["seed"],
                        **labels,
                        "metric": metric,
                        "epoch_index": epoch_number - 1,
                        "epoch_number": epoch_number,
                        "value": value,
                    }
                )
    return rows


def _training_history_rows(
    *,
    root: Path,
    frame: pd.DataFrame,
    sweep_name: str,
    labels: Mapping[str, Any],
    fields: Mapping[str, Any],
    require_equal_length: bool,
) -> list[dict[str, Any]]:
    if not fields:
        return []
    rows: list[dict[str, Any]] = []
    for record in frame[["config_id", "seed"]].to_dict("records"):
        config_id = int(record["config_id"])
        path = root / "results" / f"config_{config_id}" / "training_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        histories: dict[str, list[float]] = {}
        for source, metric in fields.items():
            values = payload.get(str(source))
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"training history {source!r} is absent for {sweep_name} "
                    f"config_id={config_id}"
                )
            numeric = [float(value) for value in values]
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError(
                    f"non-finite training history for {sweep_name} config_id={config_id}"
                )
            histories[str(metric)] = numeric
        lengths = {len(values) for values in histories.values()}
        if require_equal_length and len(lengths) != 1:
            raise ValueError(
                f"training-history lengths differ for {sweep_name} config_id={config_id}"
            )
        for metric, values in histories.items():
            for epoch_index, value in enumerate(values):
                rows.append(
                    {
                        "sweep_name": sweep_name,
                        "sweep_root": str(root),
                        "config_id": config_id,
                        "seed": record["seed"],
                        **labels,
                        "metric": metric,
                        "epoch_index": epoch_index,
                        "epoch_number": epoch_index + 1,
                        "value": value,
                    }
                )
    return rows


def _validate_labels(
    labels: Mapping[str, Any], columns: Sequence[str]
) -> dict[str, Any]:
    normalized = {str(key): value for key, value in labels.items()}
    reserved = {"sweep_name", "sweep_root", "sweep_artifact_set_sha256"}
    collisions = sorted(set(normalized) & (set(columns) | reserved))
    if collisions:
        raise ValueError(
            f"declared design labels collide with collected columns: {collisions}"
        )
    for key, value in normalized.items():
        if not key or isinstance(value, (dict, list, tuple, set)):
            raise ValueError(f"design label {key!r} must be a scalar")
    return normalized


def collect_sweep_metrics(
    *, manifest_path: Path, output_dir: Path | None = None
) -> dict[str, Any]:
    """Collect all manifest sweeps and write one row per trained checkpoint."""

    manifest_path = manifest_path.resolve()
    manifest = _load_manifest(manifest_path)
    declarations = manifest.get("sweeps")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("sweeps must be a non-empty list")
    required_columns = [str(value) for value in manifest.get("required_columns", [])]
    if not required_columns:
        raise ValueError("required_columns must be declared explicitly")

    history_config = manifest.get("training_history", {})
    if not isinstance(history_config, dict):
        raise TypeError("training_history must be a mapping")
    history_fields = history_config.get("fields", {})
    if not isinstance(history_fields, dict):
        raise TypeError("training_history.fields must be a mapping")
    history_fields = {str(key): str(value) for key, value in history_fields.items()}
    performance_epoch_fields = history_config.get("performance_epoch_fields", {})
    if not isinstance(performance_epoch_fields, dict):
        raise TypeError("training_history.performance_epoch_fields must be a mapping")
    performance_epoch_fields = {
        str(key): str(value) for key, value in performance_epoch_fields.items()
    }
    require_equal_history_length = bool(
        history_config.get("require_equal_length_within_checkpoint", True)
    )

    frames: list[pd.DataFrame] = []
    history_rows: list[dict[str, Any]] = []
    sweep_records: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, declaration in enumerate(declarations):
        if not isinstance(declaration, dict):
            raise TypeError(f"sweeps[{index}] must be a mapping")
        name = str(declaration.get("name", ""))
        if not name or name in names:
            raise ValueError(f"duplicate or blank sweep name: {name!r}")
        names.add(name)
        root = _resolve_path(str(declaration["root"]), base=manifest_path.parent)
        if not (root / "configs").is_dir() or not (root / "results").is_dir():
            raise FileNotFoundError(f"invalid sweep root: {root}")
        expected = int(declaration["expected_config_count"])
        if expected < 1:
            raise ValueError("expected_config_count must be positive")

        frame = DataCollector(root).collect_all()
        if len(frame) != expected:
            raise RuntimeError(
                f"incomplete sweep {name}: collected {len(frame)}/{expected} checkpoints"
            )
        missing = sorted(set(required_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"sweep {name} lacks required columns: {missing}")
        labels = _validate_labels(declaration.get("labels", {}), frame.columns)
        artifacts, artifact_set_sha256 = _artifact_inventory(
            root,
            frame,
            include_training_summary=bool(history_fields),
            include_performance_epochs=bool(performance_epoch_fields),
        )
        declared_hash = declaration.get("artifact_set_sha256")
        if declared_hash is not None and str(declared_hash) != artifact_set_sha256:
            raise ValueError(f"artifact set hash differs for sweep {name}")

        selected = frame.copy()
        for key, value in labels.items():
            selected[key] = value
        selected["sweep_name"] = name
        selected["sweep_root"] = str(root)
        selected["sweep_artifact_set_sha256"] = artifact_set_sha256
        frames.append(selected)
        history_rows.extend(
            _training_history_rows(
                root=root,
                frame=frame,
                sweep_name=name,
                labels=labels,
                fields=history_fields,
                require_equal_length=require_equal_history_length,
            )
        )
        history_rows.extend(
            _performance_epoch_history_rows(
                root=root,
                frame=frame,
                sweep_name=name,
                labels=labels,
                fields=performance_epoch_fields,
            )
        )
        sweep_records.append(
            {
                "name": name,
                "root": str(root),
                "expected_config_count": expected,
                "collected_config_count": len(frame),
                "labels": labels,
                "artifact_set_sha256": artifact_set_sha256,
                "artifacts": artifacts,
            }
        )

    combined = pd.concat(frames, ignore_index=True, sort=False)
    identity = ["sweep_name", "config_id"]
    duplicate = combined.duplicated(identity, keep=False)
    if duplicate.any():
        rows = combined.loc[duplicate, identity].head().to_dict("records")
        raise ValueError(f"duplicate collected checkpoint identities: {rows}")

    destination = (
        output_dir.resolve()
        if output_dir is not None
        else _resolve_path(str(manifest["output_dir"]), base=manifest_path.parent)
    )
    destination.mkdir(parents=True, exist_ok=True)
    units_path = destination / "checkpoint_units.csv"
    combined.to_csv(units_path, index=False)
    output_paths = [units_path]
    if history_fields:
        history_path = destination / "training_history.csv"
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        output_paths.append(history_path)
    release = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "inferential_unit": "independently_trained_checkpoint",
        "aggregation_policy": "none",
        "required_columns": required_columns,
        "collected_checkpoint_count": len(combined),
        "sweeps": sweep_records,
        "source_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
        "training_history": {
            "fields": history_fields,
            "performance_epoch_fields": performance_epoch_fields,
            "require_equal_length_within_checkpoint": require_equal_history_length,
            "row_count": len(history_rows),
        },
        "outputs": {path.name: _sha256(path) for path in output_paths},
    }
    (destination / "collection_provenance.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return release


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    release = collect_sweep_metrics(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    print(
        "Collected "
        f"{release['collected_checkpoint_count']} checkpoints into "
        "checkpoint_units.csv"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
