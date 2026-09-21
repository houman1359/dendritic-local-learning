"""Merge source-bound presentation tables without filtering or recomputing rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCOPES = {"known_profile", "oracle", "polynomial"}
COMPLETE = {
    "passed",
    "complete",
    "passed_with_documented_pointwise_roundoff_exceptions",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def merge(polynomial_manifest, constructive_analysis, output_dir):
    bindings = {str(Path(__file__).resolve()): sha(__file__)}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Bound reference artifact changed: {path}")
        bindings[str(path)] = digest
        return path

    manifest_path = bind(polynomial_manifest)
    tables = read(manifest_path)["tables"]
    if not tables or any(t["scope"] != "polynomial" for t in tables):
        raise ValueError("An explicit polynomial-only input manifest is required")
    tables = [dict(t) for t in tables]
    summaries = []

    def validate_table(table):
        path = bind(table["path"], table["sha256"])
        receipt_path = bind(table["receipt_path"], table["receipt_sha256"])
        receipt = read(receipt_path)
        if receipt["status"] not in COMPLETE:
            raise ValueError("Reference presentation must be complete")
        if (
            receipt.get("outputs", {}).get(path.name) != table["sha256"]
            and receipt.get("input_bindings", {}).get(str(path)) != table["sha256"]
        ):
            raise ValueError("Reference CSV is not bound by its declared receipt")
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            headers, rows = reader.fieldnames, list(reader)
        if not headers:
            raise ValueError("Reference table requires a header")
        scope = table["scope"]
        row_scopes = None
        if scope == "mixed_reference":
            column = table["scope_column"]
            if column not in headers:
                raise ValueError("Mixed reference scope column is absent")
            row_scopes = {row[column] for row in rows}
            if not row_scopes or not row_scopes <= SCOPES:
                raise ValueError("Every mixed reference row needs an explicit prior")
        elif scope not in SCOPES:
            raise ValueError("Unknown reference prior")
        elif "workbook_scope" in headers:
            row_scopes = {row["workbook_scope"] for row in rows}
            if row_scopes and row_scopes != {scope}:
                raise ValueError("Table and row prior scopes differ")
        summaries.append(
            {
                "label": table["label"],
                "path": str(path),
                "rows": len(rows),
                "columns": len(headers),
                "scope": scope,
                "row_scopes": sorted(row_scopes) if row_scopes else None,
                "presentation_receipt_status": receipt["status"],
                "underlying_audit_status": receipt.get(
                    "audit_status", "see bound audit"
                ),
            }
        )

    for table in tables:
        validate_table(table)
    analysis = Path(constructive_analysis).resolve()
    receipt_path = bind(analysis / "receipt.json")
    receipt = read(receipt_path)
    if receipt["status"] not in COMPLETE:
        raise ValueError("Constructive analysis is incomplete")
    csv_names = sorted(name for name in receipt["outputs"] if name.endswith(".csv"))
    actual_csv = sorted(p.name for p in analysis.glob("*.csv"))
    if not csv_names or csv_names != actual_csv:
        raise ValueError("Every constructive CSV must be bound and retained")
    for name in csv_names:
        path = bind(analysis / name, receipt["outputs"][name])
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            headers, rows = reader.fieldnames, list(reader)
        if "workbook_scope" not in (headers or []):
            raise ValueError("Constructive table lacks row-level prior scope")
        scopes = {row["workbook_scope"] for row in rows}
        if not scopes or not scopes <= SCOPES:
            raise ValueError("Constructive rows have absent or unknown prior scope")
        label = (
            "Profile risk + private audit"
            if name == "large_S_plateau.csv"
            else name.removesuffix(".csv").replace("_", " ").capitalize()
        )
        if len(label) > 31:
            raise ValueError("Explicit shorter Excel label required")
        table = {
            "label": label,
            "scope": (
                next(iter(scopes))
                if len(scopes) == 1 and name != "large_S_plateau.csv"
                else "mixed_reference"
            ),
            "path": str(path),
            "sha256": sha(path),
            "receipt_path": str(receipt_path),
            "receipt_sha256": sha(receipt_path),
        }
        if table["scope"] == "mixed_reference":
            table["scope_column"] = "workbook_scope"
        tables.append(table)
        validate_table(table)
    labels = [table["label"].casefold() for table in tables]
    paths = [str(Path(table["path"]).resolve()) for table in tables]
    if len(set(labels)) != len(labels) or len(set(paths)) != len(paths):
        raise ValueError("Repeated reference label or path")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    merged = output / "reference_manifest.json"
    write(merged, {"tables": tables})
    result = {
        "status": "complete",
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(__file__),
        "input_bindings": bindings,
        "outputs": {merged.name: sha(merged)},
        "tables": summaries,
        "total_reference_rows": sum(row["rows"] for row in summaries),
        "scope": "Presentation manifest only. Every bound input CSV and every raw row is retained without numerical changes. Generic, polynomial, known-profile and private-plane evidence have different assumptions; original audit statuses are not relabeled by presentation completion.",
    }
    if any(sha(path) != digest for path, digest in bindings.items()):
        raise ValueError("Reference source changed during merge")
    write(output / "merge_receipt.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--polynomial-manifest", type=Path, required=True)
    parser.add_argument("--constructive-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = merge(
        args.polynomial_manifest, args.constructive_analysis, args.output_dir
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "tables": len(result["tables"]),
                "rows": result["total_reference_rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
