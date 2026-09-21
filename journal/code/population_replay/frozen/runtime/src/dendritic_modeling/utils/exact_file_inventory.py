"""Deterministic distributed verification of a frozen exact-file inventory.

This module deliberately separates two boundaries:

* an application constructs and freezes an inventory only after its complete
  semantic verifier has accepted the inputs; and
* every execution rehashes every inventory file from live storage.

The inventory is not a mutable cache and no stat-only shortcut is accepted.
Files are assigned by a deterministic largest-first byte balancer so DDP ranks
can share hashing work without changing the exact-SHA boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

INVENTORY_SCHEMA = "dendritic_exact_file_inventory/v1"
SHARD_SCHEMA = "dendritic_exact_file_inventory_shard/v1"
VERIFICATION_SCHEMA = "dendritic_exact_file_inventory_verification/v1"


def canonical_sha256(value: object) -> str:
    """Hash one finite canonical-JSON value."""

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _lower_sha256(value: object, *, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _stat_identity(value: os.stat_result) -> list[int]:
    return [
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    ]


def file_snapshot(path: Path) -> dict[str, Any]:
    """Return a symlink-aware mutation snapshot without reading file bytes."""

    requested = Path(path).absolute()
    before = requested.lstat()
    resolved = requested.resolve(strict=True)
    after = requested.lstat()
    if _stat_identity(before) != _stat_identity(after):
        raise ValueError(f"input path changed while it was resolved: {requested}")
    target = resolved.stat()
    if not stat.S_ISREG(target.st_mode):
        raise ValueError(f"input does not resolve to a regular file: {requested}")
    return {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "requested_lstat": _stat_identity(after),
        "resolved_stat": _stat_identity(target),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(
    files: Sequence[Mapping[str, Any]],
    *,
    generation_binding: Mapping[str, Any],
    path_bindings: Sequence[Mapping[str, Any]] = (),
    directory_bindings: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Canonicalize a frozen application-supplied exact-file inventory."""

    by_path: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(files):
        if set(raw) != {"resolved_path", "bytes", "sha256", "roles"}:
            raise ValueError(f"inventory file row {index} has an invalid schema")
        resolved = Path(str(raw["resolved_path"]))
        if not resolved.is_absolute() or ".." in resolved.parts:
            raise ValueError("inventory resolved paths must be absolute and normalized")
        resolved_string = str(resolved)
        size = int(raw["bytes"])
        if size < 0:
            raise ValueError("inventory file bytes must be nonnegative")
        digest = _lower_sha256(raw["sha256"], label="inventory file sha256")
        roles = sorted({str(value) for value in raw["roles"]})
        if not roles or any(not value for value in roles):
            raise ValueError("inventory file roles must be non-empty strings")
        existing = by_path.get(resolved_string)
        if existing is None:
            by_path[resolved_string] = {
                "resolved_path": resolved_string,
                "bytes": size,
                "sha256": digest,
                "roles": roles,
            }
        else:
            if existing["bytes"] != size or existing["sha256"] != digest:
                raise ValueError(f"conflicting exact identities for {resolved_string}")
            existing["roles"] = sorted(set(existing["roles"]) | set(roles))

    rows = []
    for task_index, row in enumerate(
        sorted(by_path.values(), key=lambda value: value["resolved_path"])
    ):
        rows.append({"task_id": f"file_{task_index:05d}", **row})
    if not rows:
        raise ValueError("exact-file inventory must contain at least one file")

    normalized_path_bindings = sorted(
        [dict(value) for value in path_bindings],
        key=lambda value: (
            str(value.get("requested_path", "")),
            str(value.get("role", "")),
        ),
    )
    normalized_directory_bindings = sorted(
        [dict(value) for value in directory_bindings],
        key=lambda value: (str(value.get("kind", "")), str(value.get("root", ""))),
    )
    payload = {
        "schema": INVENTORY_SCHEMA,
        "status": "full_generation_verification_bound_live_rehash_required",
        "generation_binding": dict(generation_binding),
        "files": rows,
        "path_bindings": normalized_path_bindings,
        "directory_bindings": normalized_directory_bindings,
        "file_count": len(rows),
        "total_bytes": int(sum(int(row["bytes"]) for row in rows)),
        "largest_file_bytes": int(max(int(row["bytes"]) for row in rows)),
        "runtime_contract": {
            "every_file_live_sha256_required": True,
            "stat_only_cache_accepted": False,
            "semantic_verification_performed_during_generation": True,
            "semantic_reconstruction_repeated_at_runtime": False,
            "same_frozen_code_commit_required": True,
        },
    }
    return {**payload, "inventory_sha256": canonical_sha256(payload)}


def validate_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an inventory without trusting or reading its target files."""

    required = {
        "schema",
        "status",
        "generation_binding",
        "files",
        "path_bindings",
        "directory_bindings",
        "file_count",
        "total_bytes",
        "largest_file_bytes",
        "runtime_contract",
        "inventory_sha256",
    }
    if set(value) != required:
        raise ValueError(
            "exact-file inventory keys drifted: " f"{sorted(set(value) ^ required)}"
        )
    payload = {key: value[key] for key in required if key != "inventory_sha256"}
    if (
        value["schema"] != INVENTORY_SCHEMA
        or value["status"] != "full_generation_verification_bound_live_rehash_required"
        or canonical_sha256(payload) != value["inventory_sha256"]
    ):
        raise ValueError("exact-file inventory identity drifted")
    if value["runtime_contract"] != {
        "every_file_live_sha256_required": True,
        "stat_only_cache_accepted": False,
        "semantic_verification_performed_during_generation": True,
        "semantic_reconstruction_repeated_at_runtime": False,
        "same_frozen_code_commit_required": True,
    }:
        raise ValueError("exact-file inventory runtime contract drifted")
    files = value["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("exact-file inventory has no files")
    expected_ids = [f"file_{index:05d}" for index in range(len(files))]
    if [row.get("task_id") for row in files] != expected_ids:
        raise ValueError("exact-file inventory task IDs are not canonical")
    if [str(row.get("resolved_path")) for row in files] != sorted(
        str(row.get("resolved_path")) for row in files
    ):
        raise ValueError("exact-file inventory paths are not sorted")
    for index, row in enumerate(files):
        if set(row) != {"task_id", "resolved_path", "bytes", "sha256", "roles"}:
            raise ValueError(f"exact-file inventory row {index} schema drifted")
        path = Path(str(row["resolved_path"]))
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("exact-file inventory path is not absolute")
        if int(row["bytes"]) < 0:
            raise ValueError("exact-file inventory bytes are negative")
        _lower_sha256(row["sha256"], label="exact-file inventory sha256")
        roles = row["roles"]
        if not isinstance(roles, list) or roles != sorted(set(roles)) or not roles:
            raise ValueError("exact-file inventory roles are not canonical")
    if int(value["file_count"]) != len(files):
        raise ValueError("exact-file inventory count differs")
    sizes = [int(row["bytes"]) for row in files]
    if int(value["total_bytes"]) != sum(sizes):
        raise ValueError("exact-file inventory total bytes differ")
    if int(value["largest_file_bytes"]) != max(sizes):
        raise ValueError("exact-file inventory largest-file bytes differ")
    if not isinstance(value["generation_binding"], Mapping):
        raise ValueError("exact-file inventory generation binding is invalid")
    if not isinstance(value["path_bindings"], list) or not isinstance(
        value["directory_bindings"], list
    ):
        raise ValueError("exact-file inventory path bindings are invalid")
    return dict(value)


def balanced_assignment(
    inventory: Mapping[str, Any], *, world_size: int
) -> dict[str, int]:
    """Assign files deterministically by largest-first cumulative bytes."""

    verified = validate_inventory(inventory)
    world = int(world_size)
    if world < 1:
        raise ValueError("world_size must be positive")
    loads = [0] * world
    assignment: dict[str, int] = {}
    ordered = sorted(
        verified["files"],
        key=lambda row: (-int(row["bytes"]), str(row["task_id"])),
    )
    for row in ordered:
        rank = min(range(world), key=lambda candidate: (loads[candidate], candidate))
        assignment[str(row["task_id"])] = rank
        loads[rank] += int(row["bytes"])
    return assignment


def verify_shard(
    inventory: Mapping[str, Any], *, rank: int, world_size: int
) -> dict[str, Any]:
    """Live-hash one deterministic rank shard and bracket every read by stats."""

    verified = validate_inventory(inventory)
    world = int(world_size)
    worker = int(rank)
    if worker < 0 or worker >= world:
        raise ValueError("rank is outside world_size")
    assignment = balanced_assignment(verified, world_size=world)
    files = [
        row for row in verified["files"] if assignment[str(row["task_id"])] == worker
    ]
    receipts: list[dict[str, Any]] = []
    for row in files:
        path = Path(str(row["resolved_path"]))
        before = file_snapshot(path)
        if before["requested_path"] != before["resolved_path"]:
            raise ValueError(f"inventory target became a symbolic link: {path}")
        if before["resolved_path"] != str(path):
            raise ValueError(f"inventory target resolution drifted: {path}")
        if int(before["resolved_stat"][3]) != int(row["bytes"]):
            raise ValueError(f"inventory target size drifted: {path}")
        observed = _sha256_file(path)
        after = file_snapshot(path)
        if before != after:
            raise ValueError(f"inventory target changed while hashing: {path}")
        if observed != str(row["sha256"]):
            raise ValueError(f"inventory target SHA-256 drifted: {path}")
        receipts.append(
            {
                "task_id": str(row["task_id"]),
                "resolved_path": str(path),
                "bytes": int(row["bytes"]),
                "sha256": observed,
                "snapshot": after,
            }
        )
    for receipt in receipts:
        if file_snapshot(Path(receipt["resolved_path"])) != receipt["snapshot"]:
            raise ValueError(
                "inventory target changed before its rank shard completed: "
                f"{receipt['resolved_path']}"
            )
    return {
        "schema": SHARD_SCHEMA,
        "status": "verified",
        "inventory_sha256": verified["inventory_sha256"],
        "rank": worker,
        "world_size": world,
        "assignment_policy": "largest_first_bytes_then_lowest_rank",
        "file_count": len(receipts),
        "total_bytes": int(sum(row["bytes"] for row in receipts)),
        "files": receipts,
    }


def merge_verified_shards(
    inventory: Mapping[str, Any],
    shards: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Fail closed unless all and only canonical rank assignments were verified."""

    verified = validate_inventory(inventory)
    world = int(world_size)
    if len(shards) != world:
        raise ValueError("distributed inventory verification lacks one rank shard")
    assignment = balanced_assignment(verified, world_size=world)
    expected_by_id = {str(row["task_id"]): row for row in verified["files"]}
    observed_by_id: dict[str, dict[str, Any]] = {}
    ranks: set[int] = set()
    rank_rows: list[dict[str, Any]] = []
    for raw in shards:
        shard = dict(raw)
        required = {
            "schema",
            "status",
            "inventory_sha256",
            "rank",
            "world_size",
            "assignment_policy",
            "file_count",
            "total_bytes",
            "files",
        }
        if set(shard) != required:
            raise ValueError("distributed inventory shard schema drifted")
        rank = int(shard["rank"])
        if rank in ranks or rank < 0 or rank >= world:
            raise ValueError("distributed inventory rank is duplicate or invalid")
        ranks.add(rank)
        if (
            shard["schema"] != SHARD_SCHEMA
            or shard["status"] != "verified"
            or shard["inventory_sha256"] != verified["inventory_sha256"]
            or int(shard["world_size"]) != world
            or shard["assignment_policy"] != "largest_first_bytes_then_lowest_rank"
        ):
            raise ValueError("distributed inventory shard identity drifted")
        files = shard["files"]
        if not isinstance(files, list) or int(shard["file_count"]) != len(files):
            raise ValueError("distributed inventory shard count differs")
        if int(shard["total_bytes"]) != sum(int(row["bytes"]) for row in files):
            raise ValueError("distributed inventory shard byte total differs")
        for row in files:
            required_file = {
                "task_id",
                "resolved_path",
                "bytes",
                "sha256",
                "snapshot",
            }
            if set(row) != required_file:
                raise ValueError("distributed inventory file receipt schema drifted")
            task = str(row["task_id"])
            if task in observed_by_id or task not in expected_by_id:
                raise ValueError("distributed inventory task is duplicate or unknown")
            expected = expected_by_id[task]
            if assignment[task] != rank:
                raise ValueError("distributed inventory task was hashed by wrong rank")
            if (
                row["resolved_path"] != expected["resolved_path"]
                or int(row["bytes"]) != int(expected["bytes"])
                or row["sha256"] != expected["sha256"]
            ):
                raise ValueError("distributed inventory file receipt differs")
            observed_by_id[task] = dict(row)
        rank_rows.append(
            {
                "rank": rank,
                "file_count": len(files),
                "total_bytes": int(shard["total_bytes"]),
                "task_ids_sha256": canonical_sha256(
                    sorted(str(row["task_id"]) for row in files)
                ),
            }
        )
    if set(observed_by_id) != set(expected_by_id):
        missing = sorted(set(expected_by_id) - set(observed_by_id))
        raise ValueError(f"distributed inventory verification missed tasks: {missing}")
    for task, row in observed_by_id.items():
        current = file_snapshot(Path(str(row["resolved_path"])))
        if current != row["snapshot"]:
            raise ValueError(
                f"distributed inventory target changed before merge: {task}"
            )
    public = {
        "schema": VERIFICATION_SCHEMA,
        "status": "verified_before_execution",
        "inventory_sha256": verified["inventory_sha256"],
        "method": "deterministic_byte_balanced_all_rank_live_sha256",
        "world_size": world,
        "file_count": len(observed_by_id),
        "total_bytes": int(verified["total_bytes"]),
        "per_rank": sorted(rank_rows, key=lambda row: int(row["rank"])),
        "all_live_file_sha256_matched": True,
        "all_files_mutation_stable_through_rank_zero_merge": True,
        "generation_semantic_verification_bound": True,
        "stat_only_cache_used": False,
    }
    snapshots = {
        task: dict(row["snapshot"]) for task, row in sorted(observed_by_id.items())
    }
    return {**public, "receipt_sha256": canonical_sha256(public)}, snapshots
