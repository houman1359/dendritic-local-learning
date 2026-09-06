#!/usr/bin/env python3
"""Canonical explicit credit-first provenance; use --prepare before Figure5 is ready.

No main-panel assignments are inferred by historical renumbering. The retained
SI/supporting template and new panel map are versioned inputs. --prepare writes
an incomplete candidate under analysis, never the release manifest.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path

JOURNAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = JOURNAL_ROOT.parent
PROJECT_PREFIX = "drafts/dendritic-local-learning/journal/"
MANIFEST = JOURNAL_ROOT / "source_data/provenance_manifest.tsv"
MAP_PATH = JOURNAL_ROOT / "configs/credit_first_provenance/panel_sources.json"
INVENTORY = JOURNAL_ROOT / "source_data/credit_first_provenance/source_inventory.tsv"
FIELDS = ["entry_id", "record_type", "figure", "panel", "status", "source_path",
          "sha256", "generator_path", "replication_unit", "notes"]
_HASH_CACHE = {}


def resolve_project_path(raw_path):
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if raw_path.startswith(PROJECT_PREFIX):
        return JOURNAL_ROOT / raw_path[len(PROJECT_PREFIX):]
    legacy = "drafts/dendritic-local-learning/"
    if raw_path.startswith(legacy):
        return PROJECT_ROOT / raw_path[len(legacy):]
    return JOURNAL_ROOT / path


def sha256(path):
    path = path.resolve()
    if path in _HASH_CACHE:
        return _HASH_CACHE[path]
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Source changed while hashing: {path}")
    _HASH_CACHE[path] = digest.hexdigest()
    return _HASH_CACHE[path]


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if list(reader.fieldnames or []) != FIELDS:
            raise ValueError(f"Unexpected manifest columns: {path}")
        return list(reader)


def serialized(rows, fields=FIELDS):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def record(entry_id, path, figure, panel, *, role, generator="", unit="not applicable",
           notes="", record_type="panel_source"):
    source = JOURNAL_ROOT / path
    if not source.is_file():
        raise FileNotFoundError(source)
    if generator and not (JOURNAL_ROOT / generator).is_file():
        raise FileNotFoundError(JOURNAL_ROOT / generator)
    return dict(entry_id=entry_id, record_type=record_type, figure=figure,
                panel=panel, status="ready", source_path=PROJECT_PREFIX + path,
                sha256=sha256(source),
                generator_path=PROJECT_PREFIX + generator if generator else "",
                replication_unit=unit, notes=f"{role}. {notes}".strip())


def source_id(prefix, path, figure="", panel=""):
    token = hashlib.sha256(f"{path}|{figure}|{panel}".encode()).hexdigest()[:16]
    return f"creditfirst.{prefix}.{token}"


def build(*, prepare=False):
    _HASH_CACHE.clear()
    layout = json.loads(MAP_PATH.read_text())
    if layout.get("physical_figure_pending") and not prepare:
        raise RuntimeError("Physical-depth panel map is pending; --prepare cannot update the release manifest")
    for item in layout.get("source_hash_manifests", []):
        saved = json.loads((JOURNAL_ROOT / item["path"]).read_text())
        for key in ("source_sha256", "sources_sha256", "builders_sha256"):
            for source, expected in saved.get(key, {}).items():
                if sha256(JOURNAL_ROOT / source) != expected:
                    raise ValueError(f"Figure source changed since recorded rendering: {source}")
        if item.get("builder") and saved.get("builder_sha256"):
            if sha256(JOURNAL_ROOT / item["builder"]) != saved["builder_sha256"]:
                raise ValueError(f"Figure builder changed since recorded rendering: {item['builder']}")
    template = JOURNAL_ROOT / layout["retained_template"]
    rows = read_tsv(template)
    original_sources = {r["source_path"] for r in rows}
    for row in rows:
        row["sha256"] = sha256(resolve_project_path(row["source_path"]))
        generator = row.get("generator_path", "")
        if generator and not resolve_project_path(generator).is_file():
            raise FileNotFoundError(resolve_project_path(generator))
    for asset in layout["assets"]:
        canonical = JOURNAL_ROOT / asset["path"]
        component = JOURNAL_ROOT / asset["component"]
        if sha256(canonical) != sha256(component):
            raise ValueError(f"Canonical asset differs from declared builder output: {canonical}")
        rows.append(record(f"creditfirst.{asset['figure']}.asset", asset["path"],
                           asset["figure"], asset["panel"], role="Current publication asset",
                           generator=asset["generator"], notes=asset.get("notes", ""),
                           record_type="figure_asset"))
    for item in layout["records"]:
        rows.append(record(source_id("panel", item["path"], item["figure"], item["panel"]),
                           item["path"], item["figure"], item["panel"], role=item["role"],
                           generator=item.get("generator", ""), unit=item["replication_unit"],
                           notes=item.get("notes", "")))
    provenance_paths = {"scripts/update_provenance_hashes.py", str(MAP_PATH.relative_to(JOURNAL_ROOT)),
                        layout["retained_template"]}
    provenance_paths.update(a["generator"] for a in layout["assets"])
    for path in sorted(provenance_paths):
        rows.append(record(source_id("implementation", path), path, "methods", "implementation",
                           role="Versioned provenance map or figure implementation",
                           record_type="generator_snapshot"))
    excluded_extensions = set(layout.get("excluded_extensions", []))
    excluded_components = set(layout.get("excluded_path_components", []))
    allowed_documents = set(layout.get("allowed_document_names", []))
    excluded = []
    known = {r["source_path"] for r in rows}
    supporting_count = 0
    for group in layout["supporting_groups"]:
        directory = JOURNAL_ROOT / group["path"]
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths = []
        group_exclusions = set(group.get("exclude_subdirectories", []))
        group_extensions = set(group.get("excluded_extensions", []))
        for root, dirs, files in os.walk(directory):
            dirs[:] = sorted(d for d in dirs if d not in excluded_components
                             and str((Path(root) / d).relative_to(directory)) not in group_exclusions)
            paths.extend(Path(root) / name for name in sorted(files))
        for path in sorted(paths):
            relative = str(path.relative_to(JOURNAL_ROOT))
            if (path.suffix in excluded_extensions | group_extensions or excluded_components.intersection(path.parts)
                    or (path.suffix == ".md" and path.name not in allowed_documents)):
                excluded.append(relative)
                continue
            if PROJECT_PREFIX + relative in known:
                continue
            rows.append(record(source_id("support", relative), relative, group["figure"], group["panel"],
                               role="Supporting experimental source or scientific protocol",
                               unit=group["replication_unit"],
                               notes="Byte inventory; no plotted-panel association is inferred. Generating code and frozen choices are identified by the experimental protocol."))
            known.add(PROJECT_PREFIX + relative)
            supporting_count += 1
    ids = [r["entry_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate provenance entry IDs")
    if not original_sources.issubset({r["source_path"] for r in rows}):
        raise ValueError("A retained numerical or SI source disappeared")
    main = {r["figure"] for r in rows if r["record_type"] == "figure_asset" and re.fullmatch(r"fig\d+", r["figure"])}
    supplementary = {f for r in rows if r["record_type"] == "figure_asset" for f in r["figure"].split("/") if re.fullmatch(r"figS\d+", f)}
    expected_main = {f"fig{k}" for k in range(1, 9)}
    expected_supplementary = {f"figS{k}" for k in range(1, 48)}
    if not prepare and main != expected_main:
        raise ValueError(f"Main assets do not match eight-figure layout: {sorted(main)}")
    if supplementary != expected_supplementary:
        raise ValueError(f"SI assets differ from S1--S47: missing={sorted(expected_supplementary-supplementary)} extra={sorted(supplementary-expected_supplementary)}")
    for row in rows:
        for f in row["figure"].split("/"):
            if re.fullmatch(r"fig\d+", f) and f not in expected_main:
                raise ValueError(f"Obsolete main assignment remains: {row['entry_id']}")
    summary = dict(status="incomplete preparation" if prepare else "complete",
                   n_rows=len(rows), n_unique_sources=len({r["source_path"] for r in rows}),
                   n_retained_rows=len(read_tsv(template)), n_new_supporting_records=supporting_count,
                   main_figures=sorted(main), supplementary_figures=sorted(supplementary),
                   excluded_nondata_files=excluded,
                   excluded_raw_groups={g["path"]: g.get("exclude_subdirectories", [])
                                        for g in layout["supporting_groups"] if g.get("exclude_subdirectories")},
                   panel_map_sha256=sha256(MAP_PATH), template_sha256=sha256(template))
    return rows, summary


def inventory(rows):
    fields = ["entry_id", "figure", "panel", "source", "record_type", "independent_unit", "notes", "sha256"]
    records = [dict(entry_id=r["entry_id"], figure=r["figure"], panel=r["panel"],
                    source=r["source_path"].removeprefix(PROJECT_PREFIX),
                    record_type=r["record_type"], independent_unit=r["replication_unit"],
                    notes=r["notes"], sha256=r["sha256"]) for r in rows]
    return serialized(records, fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Write an incomplete analysis-only candidate")
    parser.add_argument("--check", action="store_true", help="Check current release ledger against actual sources")
    args = parser.parse_args()
    if args.prepare and args.check:
        parser.error("--prepare and --check are mutually exclusive")
    rows, summary = build(prepare=args.prepare)
    contents = serialized(rows)
    if args.prepare:
        target = JOURNAL_ROOT / "analysis/credit_first_provenance_20260906/prepared_manifest.tsv"
        write_atomic(target, contents)
        write_atomic(target.with_name("prepared_source_inventory.tsv"), inventory(rows))
        write_atomic(target.with_name("preparation_validation.json"), json.dumps(summary, indent=2) + "\n")
    elif args.check:
        if MANIFEST.read_text() != contents:
            raise RuntimeError("Canonical provenance is stale; run scripts/update_provenance_hashes.py")
        if INVENTORY.read_text() != inventory(rows):
            raise RuntimeError("Canonical flat source inventory is stale")
    else:
        write_atomic(MANIFEST, contents)
        write_atomic(INVENTORY, inventory(rows))
        write_atomic(INVENTORY.with_name("validation.json"), json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "excluded_nondata_files"}, indent=2))


if __name__ == "__main__":
    main()
