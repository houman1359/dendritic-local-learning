"""Freeze an auditable, locally deduplicated corpus using a shared tokenizer.

This module deliberately does not promise complete near-duplicate removal.
LSH retrieves candidate pairs approximately; accepted edges use exact lexical
shingle Jaccard. Connected components, including historical exposure anchors,
are resolved before any split assignment. A manifest is published only after
packing, artifact hashing, and semantic index validation have succeeded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import re
import resource
import shutil
import sqlite3
import sys
import time
import unicodedata
from bisect import bisect_left
from collections import Counter
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from .token_data import file_hash

SPLITS = ("train", "validation", "test")
CONFIG_SCHEMA = "dendritic_scaling_main_corpus_config_v1"
WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Document text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def content_id(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _json_line(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _publish_manifest(path: Path, value) -> None:
    temporary = path.with_name(".manifest.json.tmp")
    with temporary.open("x") as stream:
        stream.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _local(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Artifact must be an existing local file: {name}")
    return path


def _verified(root: Path, name: str, digest: str) -> Path:
    path = _local(root, name)
    if file_hash(path) != digest:
        raise ValueError(f"Artifact hash mismatch: {name}")
    return path


def _ids(path: Path) -> list[str]:
    values = path.read_text().splitlines()
    if len(values) != len(set(values)) or any(
        not re.fullmatch(r"[0-9a-f]{64}", x) for x in values
    ):
        raise ValueError(f"Malformed or duplicate document IDs: {path}")
    return values


@contextmanager
def _timed(timings: dict, label: str):
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        timings[label] = {
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu,
        }


def _settings(config: dict) -> dict:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("Unknown main corpus configuration schema")
    settings = dict(config)
    for key in ("candidate_documents", "sequence_length"):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("selection_seed", "split_seed", "order_seed"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int):
            raise ValueError(f"{key} must be an integer")
    validation, test = settings["validation_fraction"], settings["test_fraction"]
    if not (0 < validation < 1 and 0 < test < 1 and validation + test < 1):
        raise ValueError("Require positive train, validation, and test fractions")
    required = settings.get("required_target_positions", [])
    if (
        not isinstance(required, list)
        or len(set(required)) != len(required)
        or any(
            isinstance(x, bool)
            or not isinstance(x, int)
            or x < 1
            or x % settings["sequence_length"]
            for x in required
        )
    ):
        raise ValueError(
            "Required target positions must be unique positive sequence-aligned integers"
        )
    settings["required_target_positions"] = sorted(required)
    near_defaults = {
        "num_hashes": 112,
        "bands": 14,
        "rows_per_band": 8,
        "shingle_words": 5,
        "seed": 2026091204,
        "jaccard_threshold": 0.8,
        "max_candidate_pairs": 2_000_000,
    }
    unknown = set(settings.get("near_duplicate", {})) - near_defaults.keys()
    if unknown:
        raise ValueError(f"Unknown near_duplicate settings: {sorted(unknown)}")
    near = {**near_defaults, **settings.get("near_duplicate", {})}
    for key in (
        "num_hashes",
        "bands",
        "rows_per_band",
        "shingle_words",
        "max_candidate_pairs",
    ):
        if (
            isinstance(near[key], bool)
            or not isinstance(near[key], int)
            or near[key] < 1
        ):
            raise ValueError(f"near_duplicate.{key} must be a positive integer")
    if near["bands"] * near["rows_per_band"] != near["num_hashes"]:
        raise ValueError("MinHash bands times rows_per_band must equal num_hashes")
    if isinstance(near["seed"], bool) or not isinstance(near["seed"], int):
        raise ValueError("near_duplicate.seed must be an integer")
    if not 0 < near["jaccard_threshold"] <= 1:
        raise ValueError("Jaccard threshold must be in (0,1]")
    settings["near_duplicate"] = near
    if not settings.get("sources") or not settings.get("intended_use"):
        raise ValueError("Sources and intended_use must be explicit")
    return settings


def word_shingles(text: str, words_per_shingle: int = 5) -> set[tuple[str, ...]]:
    """Lowercase Unicode words; documents shorter than n words use exact-only dedup."""
    words = WORD_RE.findall(text.lower())
    return {
        tuple(words[i : i + words_per_shingle])
        for i in range(max(0, len(words) - words_per_shingle + 1))
    }


def shingle_jaccard(left: str, right: str, words_per_shingle: int = 5) -> float:
    a, b = word_shingles(left, words_per_shingle), word_shingles(
        right, words_per_shingle
    )
    if not a or not b:
        return float(normalize_text(left) == normalize_text(right))
    return len(a & b) / len(a | b)


def minhash_signature(
    text: str,
    *,
    num_hashes: int,
    seed: int,
    shingle_words: int = 5,
    block_size: int = 256,
    statistics: dict | None = None,
) -> np.ndarray | None:
    """Bounded signature workspace, full text, SHA256 shingle hash + SplitMix64.

    Each seed defines a 64-bit permutation of the hashed shingle domain. The
    ideal independent-MinHash LSH probability is a reference, not a recall
    guarantee for these deterministic hash families or real documents.
    """
    shingles = word_shingles(text, shingle_words)
    if statistics is not None:
        statistics["unique_shingles"] = len(shingles)
    if not shingles:
        return None
    seeds = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{seed}:{i}".encode()).digest()[:8], "little"
            )
            for i in range(num_hashes)
        ],
        dtype=np.uint64,
    )
    signature = np.full(num_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
    iterator = iter(shingles)
    while block := list(itertools.islice(iterator, block_size)):
        base = np.asarray(
            [
                int.from_bytes(
                    hashlib.sha256(
                        json.dumps(
                            x, ensure_ascii=False, separators=(",", ":")
                        ).encode()
                    ).digest()[:8],
                    "little",
                )
                for x in block
            ],
            dtype=np.uint64,
        )
        values = base[:, None] ^ seeds[None, :]
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
        signature = np.minimum(signature, values.min(axis=0))
    return signature


def _load_qualification(config: dict):
    descriptor = config["qualification_manifest"]
    path = Path(descriptor["path"]).resolve()
    if file_hash(path) != descriptor["sha256"]:
        raise ValueError("Qualification manifest hash mismatch")
    manifest = json.loads(path.read_text())
    if manifest["normalization"] != "NFC, collapse whitespace":
        raise ValueError(
            "Qualification normalization differs from anchor normalization"
        )
    root = path.parent
    tokenizer_path = _verified(
        root, manifest["tokenizer_path"], manifest["tokenizer_sha256"]
    )
    fit_path = _verified(
        root,
        manifest["tokenizer_training_ids_path"],
        manifest["tokenizer_training_ids_sha256"],
    )
    fit_ids = _ids(fit_path)
    fit_set = set(fit_ids)
    split_ids, anchors, artifacts = {}, {}, []
    for split in SPLITS:
        description = manifest["splits"][split]
        ids_path = _verified(
            root, description["document_ids_path"], description["document_ids_sha256"]
        )
        expected = set(_ids(ids_path))
        if len(expected) != description["documents"]:
            raise ValueError("Qualification split document count mismatch")
        if any(expected & other for other in split_ids.values()):
            raise ValueError("Qualification split IDs overlap")
        split_ids[split] = expected
        spool = _local(root, f"{split}.documents.jsonl")
        seen = set()
        with spool.open() as stream:
            for line in stream:
                document = json.loads(line)
                identity = document["id"]
                normalized = normalize_text(document["text"])
                if (
                    identity not in expected
                    or identity in seen
                    or content_id(normalized) != identity
                    or not normalized
                ):
                    raise ValueError("Qualification text/ID coverage mismatch")
                seen.add(identity)
                anchors[identity] = {
                    "text": normalized,
                    "qualification_split": split,
                    "tokenizer_fit": identity in fit_set,
                }
        if seen != expected:
            raise ValueError(
                "Qualification anchor spool does not cover all declared IDs"
            )
        artifacts.append(
            {
                "split": split,
                "path": str(spool),
                "sha256": file_hash(spool),
                "document_ids_sha256": file_hash(ids_path),
                "documents": len(seen),
            }
        )
    if (
        not set(fit_ids) <= split_ids["train"]
        or len(fit_ids) != manifest["tokenizer_training_documents"]
    ):
        raise ValueError(
            "Tokenizer-fit IDs must match declared count and qualification training IDs"
        )
    return manifest, tokenizer_path, fit_path, fit_ids, anchors, artifacts


def _reserve_order(config: dict, destination: Path):
    """Canonical source order and full SHA256-ranked rows; hash without text decoding."""
    import pyarrow.parquet as pq

    sources = sorted(config["sources"], key=lambda source: source["sha256"])
    if len({source["sha256"] for source in sources}) != len(sources):
        raise ValueError("Source content hashes must be unique")
    descriptions, total = [], 0
    for source in sources:
        path = Path(source["path"]).resolve()
        if file_hash(path) != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {path}")
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        if source.get("size_bytes", path.stat().st_size) != path.stat().st_size:
            raise ValueError("Source size mismatch")
        if source.get("rows", rows) != rows:
            raise ValueError("Source row count mismatch")
        if "text" not in parquet.schema.names:
            raise ValueError("Parquet source requires a text column")
        descriptions.append(
            {
                **source,
                "path": str(path),
                "rows": rows,
                "size_bytes": path.stat().st_size,
                "row_group_rows": [
                    parquet.metadata.row_group(i).num_rows
                    for i in range(parquet.num_row_groups)
                ],
            }
        )
        total += rows
    count = config["candidate_documents"]
    if count > total:
        raise ValueError("Candidate count exceeds frozen source universe")
    ranked = np.empty(total, dtype=[("rank", "V32"), ("source", "<u4"), ("row", "<u8")])
    offset = 0
    for source_index, source in enumerate(descriptions):
        for row in range(source["rows"]):
            rank = hashlib.sha256(
                f"{config['selection_seed']}:{source['sha256']}:{row}".encode()
            ).digest()
            ranked[offset] = (rank, source_index, row)
            offset += 1
    ranked.sort(order=("rank", "source", "row"))
    reserve = np.empty(total, dtype=[("source", "<u4"), ("row", "<u8")])
    reserve["source"], reserve["row"] = ranked["source"], ranked["row"]
    np.save(destination / "source_row_reserve.npy", reserve, allow_pickle=False)
    with (destination / "selected_source_rows.jsonl").open("w") as stream:
        for rank, row in enumerate(reserve[:count]):
            stream.write(
                _json_line(
                    {
                        "selection_rank": rank,
                        "source_sha256": descriptions[int(row["source"])]["sha256"],
                        "source_index": int(row["source"]),
                        "row_index": int(row["row"]),
                    }
                )
            )
    return descriptions, reserve[:count].copy()


def _candidate_documents(sources: list[dict], selected: np.ndarray, counts: Counter):
    import pyarrow.parquet as pq

    for source_index, source in enumerate(sources):
        ranks = {
            int(row["row"]): rank
            for rank, row in enumerate(selected)
            if row["source"] == source_index
        }
        if not ranks:
            continue
        selected_rows = sorted(ranks)
        parquet = pq.ParquetFile(source["path"])
        columns = [
            name
            for name in ("text", "id", "url", "dump", "date", "file_path")
            if name in parquet.schema.names
        ]
        group_start = 0
        for group, group_rows in enumerate(source["row_group_rows"]):
            lo = bisect_left(selected_rows, group_start)
            hi = bisect_left(selected_rows, group_start + group_rows)
            if lo < hi:
                counts["source_row_groups_decoded"] += 1
                counts["source_rows_in_decoded_groups"] += group_rows
                counts["source_compressed_column_bytes_decoded"] += sum(
                    parquet.metadata.row_group(group)
                    .column(column)
                    .total_compressed_size
                    for column, name in enumerate(parquet.schema.names)
                    if name in columns
                )
                # Arrow batches bound decoded text memory even for large row groups.
                row_index = group_start
                for batch in parquet.iter_batches(
                    batch_size=128, row_groups=[group], columns=columns
                ):
                    for document in batch.to_pylist():
                        if row_index in ranks:
                            text = document.pop("text")
                            yield {
                                "text": text,
                                "selection_rank": ranks[row_index],
                                "source_index": source_index,
                                "source_sha256": source["sha256"],
                                "row_index": row_index,
                                "row_group": group,
                                "row_in_group": row_index - group_start,
                                "original_id": document.get("id"),
                                "url_host": urlsplit(
                                    document.get("url") or ""
                                ).hostname,
                                "dump": document.get("dump"),
                                "date": (
                                    str(document.get("date"))
                                    if document.get("date") is not None
                                    else None
                                ),
                                "original_file_path": document.get("file_path"),
                            }
                        row_index += 1
            group_start += group_rows


def _database(destination: Path):
    database = sqlite3.connect(destination / "document_provenance.sqlite")
    database.execute(
        "CREATE TABLE documents (id TEXT PRIMARY KEY, text TEXT NOT NULL, anchor INTEGER NOT NULL, candidate INTEGER NOT NULL, position INTEGER)"
    )
    database.execute(
        "CREATE TABLE observations (rank INTEGER PRIMARY KEY, id TEXT, source_json TEXT NOT NULL, status TEXT NOT NULL)"
    )
    database.execute("CREATE INDEX observations_content_id ON observations(id,rank)")
    database.execute(
        "CREATE TABLE pairs (left_pos INTEGER, right_pos INTEGER, similarity REAL, accepted INTEGER, PRIMARY KEY(left_pos,right_pos)) WITHOUT ROWID"
    )
    return database


def _freeze_database(database, path: Path):
    """Close a committed, exclusively owned database and enter its read-only phase.

    SQLite immutable mode skips filesystem locking and change-detection checks.
    It is appropriate here only because this preparation creates the database,
    has completed every write, and never permits another writer afterwards.
    """
    path = path.resolve()
    if database.in_transaction:
        raise ValueError("Database must be committed before its immutable read phase")
    attached = {row[1]: row[2] for row in database.execute("PRAGMA database_list")}
    if Path(attached.get("main", "")).resolve() != path:
        raise ValueError("Immutable database path differs from the committed writer")

    def reject_sidecars():
        if any(
            Path(str(path) + suffix).exists() for suffix in ("-journal", "-wal", "-shm")
        ):
            raise ValueError(
                "Unexpected SQLite journal/WAL sidecar before immutable read phase"
            )

    reject_sidecars()
    database.close()
    reject_sidecars()
    readonly = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)
    readonly.execute("PRAGMA query_only=ON")
    return readonly


def _ingest(database, anchors: dict, sources: list[dict], selected: np.ndarray):
    for identity in sorted(anchors):
        database.execute(
            "INSERT INTO documents VALUES (?,?,1,0,NULL)",
            (identity, anchors[identity]["text"]),
        )
    counts = Counter()
    for document in _candidate_documents(sources, selected, counts):
        text = normalize_text(document.pop("text"))
        rank = document["selection_rank"]
        identity = content_id(text) if text else None
        counts["candidate_rows_seen"] += 1
        if not text:
            status = "empty"
        else:
            row = database.execute(
                "SELECT candidate,anchor FROM documents WHERE id=?", (identity,)
            ).fetchone()
            status = (
                "exact_duplicate_candidate"
                if row and row[0]
                else "exact_match_protected_anchor" if row else "new_content"
            )
            database.execute(
                "INSERT INTO documents VALUES (?,?,0,1,NULL) ON CONFLICT(id) DO UPDATE SET candidate=1",
                (identity, text),
            )
        counts[status] += 1
        database.execute(
            "INSERT INTO observations VALUES (?,?,?,?)",
            (rank, identity, _json_line(document).strip(), status),
        )
    if counts["candidate_rows_seen"] != len(selected):
        raise ValueError("Selected source rows were not read exactly once")
    database.commit()
    return dict(counts)


def _cluster(database, destination: Path, settings: dict, timings: dict):
    near = settings["near_duplicate"]
    identities = [
        row[0] for row in database.execute("SELECT id FROM documents ORDER BY id")
    ]
    size = len(identities)
    parents = np.arange(size, dtype=np.int64)
    short = np.zeros(size, dtype=bool)
    signature_measurements = Counter()
    path = destination / "minhash_signatures.npy"
    signatures = np.lib.format.open_memmap(
        path, mode="w+", dtype="<u8", shape=(size, near["num_hashes"])
    )
    with _timed(timings, "minhash_signatures"):
        for position, identity in enumerate(identities):
            text, anchor = database.execute(
                "SELECT text,anchor FROM documents WHERE id=?", (identity,)
            ).fetchone()
            before = time.perf_counter()
            statistics = {}
            signature = minhash_signature(
                text,
                num_hashes=near["num_hashes"],
                seed=near["seed"],
                shingle_words=near["shingle_words"],
                statistics=statistics,
            )
            label = "anchor" if anchor else "candidate_only"
            signature_measurements[f"signature_{label}_wall_seconds"] += (
                time.perf_counter() - before
            )
            signature_measurements[
                f"signature_{label}_unique_shingles"
            ] += statistics.get("unique_shingles", 0)
            signature_measurements["signature_max_document_unique_shingles"] = max(
                signature_measurements["signature_max_document_unique_shingles"],
                statistics.get("unique_shingles", 0),
            )
            short[position] = signature is None
            signatures[position] = 0 if signature is None else signature
            database.execute(
                "UPDATE documents SET position=? WHERE id=?", (position, identity)
            )
        database.commit()
        signatures.flush()
    (destination / "signature_document_ids.txt").write_text(
        "".join(x + "\n" for x in identities)
    )

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return int(index)

    counts = {
        "signature_documents": size,
        "short_documents_exact_only": int(short.sum()),
        "candidate_pairs": 0,
        "accepted_edges": 0,
        "rejected_edges": 0,
        "largest_lsh_bucket": 0,
        "bucket_pair_occurrences": 0,
    }
    counts.update(signature_measurements)
    counts["signature_protected_anchor_documents"] = database.execute(
        "SELECT count(*) FROM documents WHERE anchor=1"
    ).fetchone()[0]
    counts["signature_candidate_only_documents"] = database.execute(
        "SELECT count(*) FROM documents WHERE anchor=0 AND candidate=1"
    ).fetchone()[0]
    counts["signature_anchor_also_selected_documents"] = database.execute(
        "SELECT count(*) FROM documents WHERE anchor=1 AND candidate=1"
    ).fetchone()[0]
    with _timed(timings, "lsh_candidate_verification"):
        eligible = np.flatnonzero(~short)
        for band in range(near["bands"]):
            start = band * near["rows_per_band"]
            values = np.ascontiguousarray(
                signatures[eligible, start : start + near["rows_per_band"]]
            )
            keys = values.view(
                np.dtype((np.void, values.dtype.itemsize * near["rows_per_band"]))
            ).reshape(-1)
            order = np.argsort(keys, kind="stable")
            ordered_keys = keys[order]
            boundaries = np.r_[
                0, np.flatnonzero(ordered_keys[1:] != ordered_keys[:-1]) + 1, len(order)
            ]
            for lo, hi in itertools.pairwise(boundaries):
                bucket = eligible[order[lo:hi]]
                counts["largest_lsh_bucket"] = max(
                    counts["largest_lsh_bucket"], len(bucket)
                )
                if len(bucket) < 2:
                    continue
                # A single bucket already exceeding the whole pair budget cannot be processed safely.
                if len(bucket) * (len(bucket) - 1) // 2 > near["max_candidate_pairs"]:
                    raise ValueError(
                        "LSH bucket exceeds explicit candidate-pair budget; no bucket was silently skipped"
                    )
                for left, right in itertools.combinations(sorted(map(int, bucket)), 2):
                    counts["bucket_pair_occurrences"] += 1
                    inserted = database.execute(
                        "INSERT OR IGNORE INTO pairs VALUES (?,?,NULL,NULL)",
                        (left, right),
                    ).rowcount
                    if not inserted:
                        continue
                    counts["candidate_pairs"] += 1
                    if counts["candidate_pairs"] > near["max_candidate_pairs"]:
                        raise ValueError(
                            "LSH candidate-pair budget exceeded; increase prospectively and rerun"
                        )
                    texts = [
                        database.execute(
                            "SELECT text FROM documents WHERE id=?", (identities[p],)
                        ).fetchone()[0]
                        for p in (left, right)
                    ]
                    shingles = [
                        word_shingles(text, near["shingle_words"]) for text in texts
                    ]
                    intersection, union = len(shingles[0] & shingles[1]), len(
                        shingles[0] | shingles[1]
                    )
                    similarity = intersection / union
                    threshold = Fraction(str(near["jaccard_threshold"]))
                    accepted = (
                        intersection * threshold.denominator
                        >= union * threshold.numerator
                    )
                    database.execute(
                        "UPDATE pairs SET similarity=?,accepted=? WHERE left_pos=? AND right_pos=?",
                        (similarity, int(accepted), left, right),
                    )
                    counts["accepted_edges" if accepted else "rejected_edges"] += 1
                    if accepted:
                        a, b = find(left), find(right)
                        parents[max(a, b)] = min(a, b)
            database.commit()
    del signatures
    groups = {}
    for position, identity in enumerate(identities):
        groups[identity] = identities[find(position)]
    with (destination / "near_duplicate_edges.jsonl").open("w") as stream:
        for left, right, similarity in database.execute(
            "SELECT left_pos,right_pos,similarity FROM pairs WHERE accepted=1 ORDER BY left_pos,right_pos"
        ):
            stream.write(
                _json_line(
                    {
                        "left_id": identities[left],
                        "right_id": identities[right],
                        "jaccard": similarity,
                    }
                )
            )
    return groups, counts


def _assign(database, groups: dict[str, str], config: dict, destination: Path):
    members = {}
    for identity, anchor, candidate in database.execute(
        "SELECT id,anchor,candidate FROM documents ORDER BY id"
    ):
        group = members.setdefault(
            groups[identity], {"members": [], "anchor": False, "candidates": set()}
        )
        group["members"].append(identity)
        group["anchor"] |= bool(anchor)
        if candidate:
            group["candidates"].add(identity)
    assignments, counts = {}, Counter()
    with (destination / "document_groups.jsonl").open("w") as stream:
        for group_id, group in sorted(members.items()):
            if group["anchor"]:
                split, representative = "excluded_protected", None
            else:
                if not group["candidates"]:
                    raise ValueError("Unprotected group has no candidate document")
                representative = min(group["candidates"])
                value = (
                    int.from_bytes(
                        hashlib.sha256(
                            f"{config['split_seed']}:{group_id}".encode()
                        ).digest()[:8],
                        "big",
                    )
                    / 2**64
                )
                split = (
                    "validation"
                    if value < config["validation_fraction"]
                    else (
                        "test"
                        if value
                        < config["validation_fraction"] + config["test_fraction"]
                        else "train"
                    )
                )
            counts[f"groups_{split}"] += 1
            for identity in group["members"]:
                record = {
                    "id": identity,
                    "group_id": group_id,
                    "split": split,
                    "representative_id": representative,
                    "protected_component": group["anchor"],
                    "retained": identity == representative,
                    "candidate": identity in group["candidates"],
                }
                assignments[identity] = record
                stream.write(_json_line(record))
                if record["candidate"]:
                    counts[
                        (
                            "candidate_contents_excluded_protected"
                            if group["anchor"]
                            else (
                                "candidate_contents_retained"
                                if record["retained"]
                                else "candidate_contents_removed_near_duplicate"
                            )
                        )
                    ] += 1
    with (destination / "source_observations.jsonl").open("w") as stream:
        for _rank, identity, source_json, status in database.execute(
            "SELECT rank,id,source_json,status FROM observations ORDER BY rank"
        ):
            stream.write(
                _json_line(
                    {
                        **json.loads(source_json),
                        "id": identity,
                        "ingest_status": status,
                        **(
                            assignments[identity]
                            if identity
                            else {"split": "excluded_empty", "retained": False}
                        ),
                    }
                )
            )
    return assignments, dict(counts)


def _pack(database, assignments: dict, destination: Path, tokenizer, config: dict):
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None or not 0 <= eos_id < tokenizer.get_vocab_size() <= 65536:
        raise ValueError("Shared tokenizer needs EOS and a uint16 vocabulary")
    descriptions = {}
    for split in SPLITS:
        retained = [
            row
            for row in assignments.values()
            if row["split"] == split and row["retained"]
        ]
        retained.sort(
            key=lambda row: (
                hashlib.sha256(
                    f"{config['order_seed']}:{row['group_id']}".encode()
                ).digest(),
                row["id"],
            )
        )
        token_path, id_path = (
            destination / f"{split}.tokens.bin",
            destination / f"{split}.document_ids.txt",
        )
        index_path = destination / f"{split}.document_index.jsonl"
        count = 0
        with (
            token_path.open("wb", buffering=1024 * 1024) as tokens,
            id_path.open("w") as ids,
            index_path.open("w") as index,
        ):
            for order, record in enumerate(retained):
                identity = record["id"]
                text = database.execute(
                    "SELECT text FROM documents WHERE id=?", (identity,)
                ).fetchone()[0]
                provenance = [
                    json.loads(row[0])
                    for row in database.execute(
                        "SELECT source_json FROM observations WHERE id=? ORDER BY rank",
                        (identity,),
                    )
                ]
                if not provenance:
                    raise ValueError(
                        "Retained representative has no selected source provenance"
                    )
                encoded = [*tokenizer.encode(text).ids, eos_id]
                tokens.write(np.asarray(encoded, dtype="<u2").tobytes())
                index.write(
                    _json_line(
                        {
                            "id": identity,
                            "group_id": record["group_id"],
                            "split": split,
                            "document_order": order,
                            "token_start": count,
                            "token_end": count + len(encoded),
                            "includes_eos": True,
                            "normalized_utf8_bytes": len(text.encode()),
                            "source_rows": provenance,
                        }
                    )
                )
                ids.write(identity + "\n")
                count += len(encoded)
        if not retained or count <= config["sequence_length"]:
            raise ValueError(f"Split {split} has insufficient retained targets")
        descriptions[split] = {
            "path": token_path.name,
            "dtype": "<u2",
            "num_tokens": count,
            "sha256": file_hash(token_path),
            "documents": len(retained),
            "document_ids_path": id_path.name,
            "document_ids_sha256": file_hash(id_path),
            "document_index_path": index_path.name,
            "document_index_sha256": file_hash(index_path),
            "available_target_positions": (count - 1)
            // config["sequence_length"]
            * config["sequence_length"],
        }
    return descriptions


def _prefix_index(destination: Path, splits: dict, unique: int, sequence_length: int):
    description = splits["train"]
    available = description["available_target_positions"]
    if unique > available:
        raise ValueError(
            f"Required U={unique} exceeds measured usable training positions {available}; extend frozen reserve prospectively"
        )
    path = destination / f"train.prefix_{unique}.jsonl"
    records, partial = 0, []
    with (
        (destination / description["document_index_path"]).open() as source,
        path.open("w") as output,
    ):
        for line in source:
            record = json.loads(line)
            start, end = record["token_start"], record["token_end"]
            target_start, target_end = max(1, start), min(unique + 1, end)
            if target_start >= target_end:
                continue
            overlap = {
                "id": record["id"],
                "group_id": record["group_id"],
                "token_start": start,
                "token_end": end,
                "target_start": target_start,
                "target_end": target_end,
                "target_positions": target_end - target_start,
                "partial_document": target_start != start or target_end != end,
            }
            output.write(_json_line(overlap))
            records += 1
            if overlap["partial_document"]:
                partial.append(record["id"])
    return {
        "unique_target_positions": unique,
        "sequence_length": sequence_length,
        "target_interval": [1, unique + 1],
        "input_interval": [0, unique],
        "path": path.name,
        "sha256": file_hash(path),
        "documents_with_targets": records,
        "partial_document_ids": partial,
    }


def validate_main_corpus(
    manifest_path: str | Path, *, manifest: dict | None = None
) -> dict:
    """Verify artifact hashes plus split/group/offset/prefix semantics; no test loss."""
    path = Path(manifest_path).resolve()
    root = path.parent
    manifest = json.loads(path.read_text()) if manifest is None else manifest
    if manifest.get("main_corpus_schema") != "dendritic_scaling_main_corpus_v1":
        raise ValueError("Unknown main corpus schema")
    for name, digest in manifest["artifacts"].items():
        _verified(root, name, digest)
    fit_ids = set(
        _ids(
            _verified(
                root,
                manifest["tokenizer_training_ids_path"],
                manifest["tokenizer_training_ids_sha256"],
            )
        )
    )
    protected_ids = set(_ids(_local(root, "protected_document_ids.txt")))
    if not fit_ids <= protected_ids:
        raise ValueError("Tokenizer-fit IDs are not protected")
    groups, retained = {}, {}
    with _local(root, "document_groups.jsonl").open() as stream:
        for line in stream:
            record = json.loads(line)
            identity, group_id = record["id"], record["group_id"]
            if identity in groups:
                raise ValueError("Document group identity repeated")
            groups[identity] = record
            if identity in protected_ids and (
                not record["protected_component"]
                or record["split"] != "excluded_protected"
            ):
                raise ValueError("Protected document was assigned to an active split")
            if record["retained"]:
                if (
                    record["split"] not in SPLITS
                    or record["protected_component"]
                    or record["representative_id"] != identity
                ):
                    raise ValueError("Invalid retained group representative")
                if group_id in retained:
                    raise ValueError("Multiple retained documents in one group")
                retained[group_id] = identity
    if not protected_ids <= groups.keys():
        raise ValueError("Not all protected IDs have group assignments")
    component_splits = {}
    for record in groups.values():
        group_id = record["group_id"]
        if group_id not in groups or groups[group_id]["group_id"] != group_id:
            raise ValueError("Group ID is not its canonical member")
        value = (
            record["split"],
            record["protected_component"],
            record["representative_id"],
        )
        if component_splits.setdefault(group_id, value) != value:
            raise ValueError("Component assignments disagree")
        if record["split"] == "excluded_protected":
            if record["representative_id"] is not None or record["retained"]:
                raise ValueError("Protected component has a retained representative")
        elif retained.get(group_id) != record["representative_id"]:
            raise ValueError("Component representative is missing")
    all_ids, all_groups, split_records = set(), set(), {}
    for split, description in manifest["splits"].items():
        ids = _ids(
            _verified(
                root,
                description["document_ids_path"],
                description["document_ids_sha256"],
            )
        )
        token_path = _verified(root, description["path"], description["sha256"])
        if token_path.stat().st_size != 2 * description["num_tokens"]:
            raise ValueError("Token file length mismatch")
        tokens = np.memmap(token_path, mode="r", dtype="<u2")
        expected_start, observed, indexed = 0, [], []
        with _verified(
            root,
            description["document_index_path"],
            description["document_index_sha256"],
        ).open() as stream:
            for order, line in enumerate(stream):
                record = json.loads(line)
                identity, group_id = record["id"], record["group_id"]
                if (
                    record["split"] != split
                    or record["document_order"] != order
                    or record["token_start"] != expected_start
                    or not record["includes_eos"]
                    or record["token_end"] <= expected_start
                    or record["token_end"] > len(tokens)
                ):
                    raise ValueError(
                        "Document token index has a gap, overlap, or invalid interval"
                    )
                if (
                    identity in all_ids
                    or group_id in all_groups
                    or identity in protected_ids
                    or identity in fit_ids
                    or retained.get(group_id) != identity
                    or groups[identity]["split"] != split
                ):
                    raise ValueError(
                        "Split/group/retained-document assignment mismatch"
                    )
                if not record["source_rows"] or any(
                    source["source_sha256"]
                    not in {s["sha256"] for s in manifest["sources"]}
                    for source in record["source_rows"]
                ):
                    raise ValueError("Document source provenance mismatch")
                if int(tokens[record["token_end"] - 1]) != manifest["eos_id"]:
                    raise ValueError("Document index does not terminate with EOS")
                all_ids.add(identity)
                all_groups.add(group_id)
                observed.append(identity)
                indexed.append(record)
                expected_start = record["token_end"]
        if (
            expected_start != description["num_tokens"]
            or observed != ids
            or len(ids) != description["documents"]
        ):
            raise ValueError("Document index does not cover its token file and ID list")
        for start in range(0, len(tokens), 1_000_000):
            if tokens[start : start + 1_000_000].max() >= manifest["vocab_size"]:
                raise ValueError("Token exceeds vocabulary")
        split_records[split] = indexed
        del tokens
    if all_ids != set(retained.values()):
        raise ValueError("Not all retained representatives were packed")
    for description in manifest["prefixes"]:
        unique = description["unique_target_positions"]
        if description["target_interval"] != [1, unique + 1] or description[
            "input_interval"
        ] != [0, unique]:
            raise ValueError("Invalid prefix interval convention")
        expected = []
        for record in split_records["train"]:
            start, end = max(1, record["token_start"]), min(
                unique + 1, record["token_end"]
            )
            if start < end:
                expected.append(
                    (
                        record["id"],
                        record["group_id"],
                        record["token_start"],
                        record["token_end"],
                        start,
                        end,
                    )
                )
        observed = []
        total = 0
        with _verified(
            root, description["path"], description["sha256"]
        ).open() as stream:
            for line in stream:
                row = json.loads(line)
                observed.append(
                    tuple(
                        row[k]
                        for k in (
                            "id",
                            "group_id",
                            "token_start",
                            "token_end",
                            "target_start",
                            "target_end",
                        )
                    )
                )
                if row["target_positions"] != row["target_end"] - row["target_start"]:
                    raise ValueError("Prefix target overlap count mismatch")
                total += row["target_positions"]
        if observed != expected or total != unique:
            raise ValueError("Prefix index does not cover declared target positions")
    return {
        "passed": True,
        "protected_documents": len(protected_ids),
        "tokenizer_fit_documents": len(fit_ids),
        "retained_documents": len(all_ids),
        "groups": len(component_splits),
        "prefixes": len(manifest["prefixes"]),
    }


def prepare_main_corpus(config: dict, output_dir: str | Path) -> dict:
    """Prepare a new immutable corpus directory; failures retain diagnostic files."""
    from tokenizers import Tokenizer

    settings = _settings(config)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    _json(destination / "preparation_config.json", settings)
    timings, counts = {}, {}
    started = time.time()
    database = None
    try:
        with _timed(timings, "qualification_anchor_verification"):
            (
                qualification,
                tokenizer_path,
                fit_path,
                fit_ids,
                anchors,
                anchor_artifacts,
            ) = _load_qualification(settings)
            shutil.copyfile(tokenizer_path, destination / "tokenizer.json")
            shutil.copyfile(
                fit_path, destination / "tokenizer_training_document_ids.txt"
            )
            _verified(destination, "tokenizer.json", qualification["tokenizer_sha256"])
            _verified(
                destination,
                "tokenizer_training_document_ids.txt",
                qualification["tokenizer_training_ids_sha256"],
            )
            (destination / "protected_document_ids.txt").write_text(
                "".join(x + "\n" for x in sorted(anchors))
            )
            tokenizer = Tokenizer.from_file(str(destination / "tokenizer.json"))
            if (
                tokenizer.get_vocab_size() != qualification["vocab_size"]
                or tokenizer.token_to_id("<eos>") != qualification["eos_id"]
            ):
                raise ValueError(
                    "Shared tokenizer vocabulary/EOS disagrees with qualification manifest"
                )
        with _timed(timings, "source_verification_and_reserve_order"):
            sources, selected = _reserve_order(settings, destination)
        database = _database(destination)
        with _timed(timings, "candidate_source_read_and_exact_dedup"):
            counts.update(_ingest(database, anchors, sources, selected))
        groups, cluster_counts = _cluster(database, destination, settings, timings)
        counts.update(cluster_counts)
        with _timed(timings, "database_freeze_for_readonly_stages"):
            database = _freeze_database(
                database, destination / "document_provenance.sqlite"
            )
        with _timed(timings, "component_assignment"):
            assignments, group_counts = _assign(database, groups, settings, destination)
            counts.update(group_counts)
        with _timed(timings, "tokenizer_auxiliary_exposure"):
            fit_bytes = fit_tokens = 0
            with (destination / "tokenizer_fit_exposure.jsonl").open("w") as stream:
                for identity in fit_ids:
                    text = anchors[identity]["text"]
                    size, tokens = len(text.encode()), len(tokenizer.encode(text).ids)
                    fit_bytes += size
                    fit_tokens += tokens
                    stream.write(
                        _json_line(
                            {
                                "id": identity,
                                "normalized_utf8_bytes": size,
                                "encoded_tokens_without_eos": tokens,
                                "group_id": groups[identity],
                                "new_corpus_split": "excluded_protected",
                            }
                        )
                    )
        with _timed(timings, "encoding_and_packing"):
            splits = _pack(database, assignments, destination, tokenizer, settings)
        database.commit()
        database.close()
        database = None
        with _timed(timings, "prefix_indices"):
            prefixes = [
                _prefix_index(destination, splits, unique, settings["sequence_length"])
                for unique in settings["required_target_positions"]
            ]
        _json(
            destination / "preparation_measurements.json",
            {
                "timings": timings,
                "counts": counts,
                "elapsed_seconds_before_final_validation": time.time() - started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "artifact_bytes_before_final_validation": sum(
                    p.stat().st_size for p in destination.iterdir() if p.is_file()
                ),
            },
        )
        artifacts = {
            path.name: file_hash(path)
            for path in sorted(destination.iterdir())
            if path.is_file()
        }
        near = settings["near_duplicate"]
        manifest = {
            "schema": "dendritic_scaling_packed_tokens_v1",
            "main_corpus_schema": "dendritic_scaling_main_corpus_v1",
            "vocab_size": tokenizer.get_vocab_size(),
            "requested_vocab_size": qualification.get(
                "requested_vocab_size", qualification["vocab_size"]
            ),
            "tokenizer_path": "tokenizer.json",
            "tokenizer_sha256": file_hash(destination / "tokenizer.json"),
            "eos_id": tokenizer.token_to_id("<eos>"),
            "tokenizer_training_documents": len(fit_ids),
            "tokenizer_training_ids_path": "tokenizer_training_document_ids.txt",
            "tokenizer_training_ids_sha256": file_hash(
                destination / "tokenizer_training_document_ids.txt"
            ),
            "tokenizer_exposure": {
                "policy": "Frozen shared auxiliary tokenizer; all qualification components excluded from new corpus",
                "fit_normalized_utf8_bytes": fit_bytes,
                "fit_encoded_tokens_without_eos": fit_tokens,
                "included_in_lm_U_or_T": False,
                "all_fit_documents_in_smallest_U": False,
            },
            "normalization": "NFC, collapse whitespace",
            "source_documents_seen": len(selected),
            "selection_policy": "SHA256(selection_seed:source_sha256:row_index) rank over full canonical source universe; selected prefix of frozen reserve",
            "selection_seed": settings["selection_seed"],
            "source_row_reserve_path": "source_row_reserve.npy",
            "source_row_reserve_sha256": artifacts["source_row_reserve.npy"],
            "selected_source_rows_sha256": artifacts["selected_source_rows.jsonl"],
            "split_policy": "Whole exact/near-duplicate graph components; protected components excluded; SHA256(split_seed:canonical_group_id)",
            "split_seed": settings["split_seed"],
            "validation_fraction": settings["validation_fraction"],
            "test_fraction": settings["test_fraction"],
            "order_policy": "One representative per retained component; SHA256(order_seed:canonical_group_id), then content ID",
            "order_seed": settings["order_seed"],
            "near_duplicate_filter": True,
            "near_duplicate": {
                **near,
                "signature_algorithm": "SHA256 shingle tuple to 64 bits, independently seeded SplitMix64 permutations, uint64 minima",
                "text_policy": "Lowercase Unicode regex \\w+ words; full-document shingle sets; no boilerplate removal or truncation",
                "short_document_policy": "Fewer than shingle_words words: exact normalized-content dedup only",
                "edge_rule": "Exact lexical shingle-set Jaccard >= threshold among approximate LSH candidates",
                "group_rule": "Connected components; endpoints need not meet pairwise threshold; minimum member content SHA256 group ID and representative",
                "idealized_candidate_probability_at_threshold": 1
                - (1 - near["jaccard_threshold"] ** near["rows_per_band"])
                ** near["bands"],
                "limitations": "Approximate retrieval can miss near duplicates and substring/containment contamination; independent residual audit still required",
            },
            "packing_policy": "Concatenate documents with EOS; causal attention may cross EOS inside a window; no cross-window attention",
            "target_position_policy": "U counts stream target positions [1,U+1), conditional on shared tokenizer; sequence-aligned nested prefixes",
            "sequence_length": settings["sequence_length"],
            "splits": splits,
            "prefixes": prefixes,
            "sources": sources,
            "qualification_anchor_artifacts": anchor_artifacts,
            "counts": counts,
            "artifacts": artifacts,
            "final_test_access": "Blocked by PackedTokenStore unless allow_test=True; preparation validates integrity without evaluating models",
            "readiness": "Prepared and internally validated; independent residual near-duplicate audit and main-study protocol checks remain",
            "provenance": {
                "source_config": settings,
                "preparation_source_sha256": file_hash(__file__),
                "token_data_source_sha256": file_hash(
                    Path(__file__).with_name("token_data.py")
                ),
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "tokenizers", "pyarrow")
                },
                "python_version": sys.version,
                "unicode_version": unicodedata.unidata_version,
                "sqlite_version": sqlite3.sqlite_version,
                "database_read_policy": "Exclusive preparation writer committed and closed after clustering; no journal/WAL sidecars; immutable read-only connection for assignment and packing; no subsequent writers",
                "intended_use": settings["intended_use"],
            },
        }
        with _timed(timings, "final_integrity_and_semantic_validation"):
            validation = validate_main_corpus(
                destination / "manifest.json", manifest=manifest
            )
        _json(
            destination / "completion_report.json",
            {
                "validation": validation,
                "timings": timings,
                "elapsed_seconds": time.time() - started,
                "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            },
        )
        manifest["artifacts"]["completion_report.json"] = file_hash(
            destination / "completion_report.json"
        )
        _publish_manifest(destination / "manifest.json", manifest)
        return manifest
    except BaseException as error:
        _json(
            destination / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "timings": timings,
                "counts": counts,
                "elapsed_seconds": time.time() - started,
            },
        )
        raise
    finally:
        if database is not None:
            database.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.source_config.read_text())
    config["invocation_source_config_sha256"] = file_hash(args.source_config)
    result = prepare_main_corpus(config, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "counts": result["counts"],
                "available_target_positions": {
                    s: d["available_target_positions"]
                    for s, d in result["splits"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
