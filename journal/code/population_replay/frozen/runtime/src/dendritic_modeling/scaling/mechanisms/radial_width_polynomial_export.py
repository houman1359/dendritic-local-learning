"""Export every declared ordinary polynomial reference from the sealed audit.

This is a presentation transform. It does not fit, score, choose a numerical
cutoff or read private teacher files. Failed/missing reference rows remain rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def export(audit_receipt, output_dir):
    audit_path = Path(audit_receipt).resolve()
    audit = read(audit_path)
    protocol_path = audit_path.parent / "protocol_receipt.json"
    protocol = read(protocol_path)
    if len(audit["dataset_receipts"]) != protocol["expected_datasets"]:
        raise ValueError("Audit dataset inventory is incomplete")
    train_sizes = sorted(protocol["reference_train_sizes"])
    if not train_sizes or len(set(train_sizes)) != len(train_sizes):
        raise ValueError("Unique declared polynomial TRAIN prefixes required")
    bindings = {
        str(audit_path): sha(audit_path),
        str(protocol_path): sha(protocol_path),
        str(Path(__file__).resolve()): sha(__file__),
    }

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Bound polynomial source changed: {path}")
        bindings[str(path)] = digest
        return path

    rows = []
    seen = set()
    for declared in sorted(audit["dataset_receipts"], key=lambda r: r["index"]):
        path = bind(declared["path"], declared["sha256"])
        dataset = read(path)
        data_path = Path(declared["data_path"]).resolve()
        metadata_path = bind(data_path.with_suffix(".json"))
        metadata = read(metadata_path)
        bind(data_path, metadata["sha256"])
        references = dataset.get("polynomial_references", [])
        by_n = {r["train_n"]: r for r in references}
        if len(by_n) != len(references) or not set(by_n) <= set(train_sizes):
            raise ValueError("Repeated or unexpected polynomial prefix")
        for n in train_sizes:
            key = (
                metadata["release"],
                metadata["task"],
                metadata["teacher"],
                metadata["observation"],
                n,
            )
            if key in seen:
                raise ValueError("Repeated polynomial scientific cell")
            seen.add(key)
            record = by_n.get(
                n,
                {
                    "status": "missing",
                    "train_n": n,
                    "error": "Declared prefix has no completed reference record",
                },
            )
            row = {
                "procedure": "polynomial",
                "task": metadata["task"],
                "train_n": n,
                "teacher": metadata["teacher"],
                "observation": metadata["observation"],
                "release": metadata["release"],
                "endpoint_split": metadata["endpoint"],
                "comparison_scope": record.get(
                    "comparison_scope", "Declared prefix; reference missing"
                ),
                "status": record.get("status", "failed"),
                "label_audit_status": dataset.get("label_audit", {}).get(
                    "status", "not_available"
                ),
                "power_m": int(metadata["task"].removeprefix("radial_m")),
                "polynomial_degree": 2 * int(metadata["task"].removeprefix("radial_m")),
                "actual_parameters": 61 if metadata["task"] == "radial_m2" else 181,
                "rcond": protocol["reference_rcond"],
                "stronger_prior": "Known homogeneous polynomial degree and supplied raw-coordinate blocks; TRAIN scalar labels only, no private plane/profile coefficients.",
                "data_path": str(data_path),
                "dataset_receipt_path": str(path),
            }
            if row["status"] == "passed":
                reference = record["reference"]
                if reference["status"] != "complete":
                    raise ValueError(
                        "Successful row lacks complete polynomial reference"
                    )
                model = reference["model"]
                if (
                    model["train_rows"] != n
                    or model["m"] != row["power_m"]
                    or model["stored_coefficients"] != row["actual_parameters"]
                    or model["rcond"] != protocol["reference_rcond"]
                ):
                    raise ValueError(
                        "Polynomial degree/prefix/count/cutoff differs from declared row"
                    )
                sources = {
                    Path(s["path"]).name: bind(s["path"], s["sha256"])
                    for s in reference["sources"]
                }
                state = sources["reference.npz"]
                reference_path = bind(state.parent / "result.json")
                if read(reference_path) != reference:
                    raise ValueError(
                        "Standalone and embedded polynomial receipts differ"
                    )
                with np.load(state, allow_pickle=False) as arrays:
                    singular = arrays["singular_values"]
                    coefficients = arrays["coefficients"]
                if (
                    coefficients.size != row["actual_parameters"]
                    or not np.isfinite(coefficients).all()
                    or not np.isfinite(singular).all()
                ):
                    raise ValueError("Invalid polynomial state inventory")
                rank = model["fit_rank"]
                row.update(
                    train_mse=reference["metrics"]["train_mse"],
                    endpoint_mse=reference["metrics"]["endpoint_mse"],
                    test_mse=(
                        reference["metrics"]["endpoint_mse"]
                        if metadata["endpoint"] == "test"
                        else None
                    ),
                    validation_mse=(
                        reference["metrics"]["endpoint_mse"]
                        if metadata["endpoint"] == "validation"
                        else None
                    ),
                    fit_rank=rank,
                    full_column_rank=rank == row["actual_parameters"],
                    singular_values_count=len(singular),
                    returned_singular_condition=(
                        float(singular[0] / singular[-1])
                        if len(singular) and singular[-1] > 0
                        else None
                    ),
                    retained_rank_condition=(
                        float(singular[0] / singular[rank - 1])
                        if rank > 0 and singular[rank - 1] > 0
                        else None
                    ),
                    conditioning_scope="Ratios of returned singular values; an underdetermined design can have a finite returned-spectrum condition and still lack full column rank.",
                    coefficient_l2=float(np.linalg.norm(coefficients)),
                    coefficient_max_abs=float(np.max(np.abs(coefficients))),
                    reference_receipt_path=str(reference_path),
                    state_path=str(state),
                    error=None,
                )
            else:
                row.update(
                    train_mse=None,
                    endpoint_mse=None,
                    test_mse=None,
                    validation_mse=None,
                    fit_rank=None,
                    full_column_rank=None,
                    singular_values_count=None,
                    returned_singular_condition=None,
                    retained_rank_condition=None,
                    conditioning_scope="Unavailable for failed/missing reference",
                    coefficient_l2=None,
                    coefficient_max_abs=None,
                    reference_receipt_path=None,
                    state_path=None,
                    error=json.dumps(record, sort_keys=True),
                )
            rows.append(row)
    if len(rows) != protocol["expected_datasets"] * len(train_sizes):
        raise ValueError("Not every declared polynomial reference row was retained")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "polynomial_rows.csv"
    headers = list(dict.fromkeys(k for r in rows for k in r))
    with csv_path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "status": "complete",
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha(__file__),
        "audit_status": audit["status"],
        "input_bindings": bindings,
        "outputs": {csv_path.name: sha(csv_path)},
        "rows": len(rows),
        "successful_rows": sum(r["status"] == "passed" for r in rows),
        "failed_or_missing_rows": sum(r["status"] != "passed" for r in rows),
        "scope": "Every declared prefix/teacher/observation retained; plain polynomial TRAIN-only OLS has a stronger degree/block prior. Source fit metrics copied without refitting or selecting; condition diagnostics read only from saved singular values. Original numerical-audit status is separate and unchanged.",
    }
    receipt_path = output / "export_receipt.json"
    write(receipt_path, receipt)
    manifest = {
        "tables": [
            {
                "label": "Polynomial reference rows",
                "scope": "polynomial",
                "path": str(csv_path),
                "sha256": sha(csv_path),
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha(receipt_path),
            }
        ]
    }
    write(output / "reference_manifest.json", manifest)
    if any(sha(p) != d for p, d in bindings.items()):
        raise ValueError("Polynomial input changed during export")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = export(args.audit_receipt, args.output_dir)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "status",
                    "rows",
                    "successful_rows",
                    "failed_or_missing_rows",
                    "audit_status",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
