"""Auditable train-only BPE preparation and memory-mapped causal-LM windows.

Document splits use normalized-content hashes. Exact duplicates are removed;
near-duplicate detection is deliberately not claimed. Windows expose already
shifted labels, so callers must not ask a causal-LM wrapper to shift them again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import torch


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _verify_id_artifact(
    root: Path,
    description: dict,
    *,
    path_key: str,
    hash_key: str,
    label: str,
    verify_hash: bool,
    allow_hash_only: bool = False,
) -> bool:
    """Check a declared local ID artifact; return whether its hash was checked."""
    if path_key not in description:
        if hash_key in description and not allow_hash_only:
            raise ValueError(f"{label} requires both a path and a hash")
        return False
    if hash_key not in description:
        raise ValueError(f"{label} requires both a path and a hash")
    declared_path = description[path_key]
    if not isinstance(declared_path, str) or not declared_path:
        raise ValueError(f"{label} must be an existing local file")
    path = (root / declared_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{label} must be an existing local file")
    if verify_hash and file_hash(path) != description[hash_key]:
        raise ValueError(f"{label} hash differs from its manifest")
    return bool(verify_hash)


def prepare_corpus(
    documents,
    output_dir: str | Path,
    *,
    vocab_size: int = 16384,
    tokenizer_train_documents: int = 10000,
    split_seed: int = 20260911,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    provenance: dict | None = None,
) -> dict:
    """Freeze document splits before fitting a byte-level BPE on training only.

    Input documents are strings or dictionaries with a text field. The tokenizer
    sees at most the first tokenizer_train_documents accepted training documents.
    A disk-backed exact-duplicate index bounds RAM. Existing outputs are refused;
    manifest.json is written last and marks successful completion.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    if isinstance(vocab_size, bool) or not 260 <= vocab_size <= 65536:
        raise ValueError("uint16 byte-level BPE requires vocab_size in [260,65536]")
    if tokenizer_train_documents < 1:
        raise ValueError("tokenizer_train_documents must be positive")
    if not (
        0 < validation_fraction < 1
        and 0 < test_fraction < 1
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError("Require positive train, validation, and test fractions")
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    split_names = ("train", "validation", "test")
    counts = dict.fromkeys(split_names, 0)
    source_count = duplicates = empty = 0
    spool_paths = {
        split: destination / f"{split}.documents.jsonl" for split in split_names
    }
    streams = {split: path.open("w") for split, path in spool_paths.items()}
    database = sqlite3.connect(destination / "document_index.sqlite")
    database.execute(
        "CREATE TABLE documents (sha256 TEXT PRIMARY KEY, split TEXT NOT NULL)"
    )
    try:
        for document in documents:
            source_count += 1
            text = document if isinstance(document, str) else document["text"]
            normalized = " ".join(unicodedata.normalize("NFC", text).split())
            if not normalized:
                empty += 1
                continue
            identity = hashlib.sha256(normalized.encode()).hexdigest()
            bucket = (
                int.from_bytes(
                    hashlib.sha256(f"{split_seed}:{identity}".encode()).digest()[:8],
                    "big",
                )
                / 2**64
            )
            split = (
                "validation"
                if bucket < validation_fraction
                else "test" if bucket < validation_fraction + test_fraction else "train"
            )
            cursor = database.execute(
                "INSERT OR IGNORE INTO documents VALUES (?,?)", (identity, split)
            )
            if cursor.rowcount == 0:
                duplicates += 1
                continue
            streams[split].write(
                json.dumps({"id": identity, "text": normalized}, ensure_ascii=False)
                + "\n"
            )
            counts[split] += 1
        database.commit()
    finally:
        database.close()
        for stream in streams.values():
            stream.close()
    if not all(counts.values()):
        raise ValueError(f"Every document split needs data; observed {counts}")

    training_ids = []

    def tokenizer_documents():
        with spool_paths["train"].open() as stream:
            for index, line in enumerate(stream):
                if index >= tokenizer_train_documents:
                    break
                document = json.loads(line)
                training_ids.append(document["id"])
                yield document["text"]

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        show_progress=False,
    )
    tokenizer.train_from_iterator(tokenizer_documents(), trainer=trainer)
    tokenizer_path = destination / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    eos_id = tokenizer.token_to_id("<eos>")
    assert eos_id == 2
    splits = {}
    for split in split_names:
        token_path = destination / f"{split}.tokens.bin"
        id_path = destination / f"{split}.document_ids.txt"
        count = 0
        with (
            spool_paths[split].open() as source,
            token_path.open("wb") as target,
            id_path.open("w") as ids,
        ):
            for line in source:
                document = json.loads(line)
                tokens = [*tokenizer.encode(document["text"]).ids, eos_id]
                np.asarray(tokens, dtype="<u2").tofile(target)
                count += len(tokens)
                ids.write(document["id"] + "\n")
        splits[split] = {
            "path": token_path.name,
            "dtype": "<u2",
            "num_tokens": count,
            "sha256": file_hash(token_path),
            "documents": counts[split],
            "document_ids_path": id_path.name,
            "document_ids_sha256": file_hash(id_path),
        }
    training_ids_path = destination / "tokenizer_training_document_ids.txt"
    training_ids_path.write_text("".join(identity + "\n" for identity in training_ids))
    manifest = {
        "schema": "dendritic_scaling_packed_tokens_v1",
        "vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": vocab_size,
        "tokenizer_path": tokenizer_path.name,
        "tokenizer_sha256": file_hash(tokenizer_path),
        "eos_id": eos_id,
        "tokenizer_training_documents": len(training_ids),
        "tokenizer_training_ids_sha256": file_hash(training_ids_path),
        "tokenizer_training_ids_path": training_ids_path.name,
        "split_seed": split_seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "normalization": "NFC, collapse whitespace",
        "split_policy": "SHA256(seed:normalized_content_sha256), exact duplicates removed",
        "near_duplicate_filter": False,
        "source_documents_seen": source_count,
        "exact_duplicates_removed": duplicates,
        "empty_documents_removed": empty,
        "packing_policy": "Concatenate documents with EOS; causal attention may cross EOS inside a window; no cross-window attention",
        "splits": splits,
        "provenance": provenance or {},
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


class PackedTokenStore:
    """Nonoverlapping prediction-target windows in a verified local uint16 file.

    Each L-target window reads L+1 tokens. Adjacent windows share one context
    token but never a target position. unique_tokens selects a nested prefix of
    target positions, and must be a positive multiple of L.

    Declared document-ID and tokenizer-fit ID files must exist locally and their
    hashes are checked unless verify_hash=False. Legacy manifests without an ID
    artifact path remain loadable, but their document-ID hash is explicitly
    unverified. Tokenizer-fit metadata is optional; if supplied it needs both a
    path and hash. These checks establish file integrity, not split membership.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        sequence_length: int,
        unique_tokens: int | None = None,
        *,
        allow_test: bool = False,
        verify_hash: bool = True,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError("Unknown token split")
        if split == "test" and not allow_test:
            raise ValueError("Test tokens require explicit final-evaluation access")
        if (
            isinstance(sequence_length, bool)
            or not isinstance(sequence_length, int)
            or sequence_length < 1
        ):
            raise ValueError("sequence_length must be a positive integer")
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema"] != "dendritic_scaling_packed_tokens_v1":
            raise ValueError("Unknown token manifest schema")
        self.vocab_size = manifest["vocab_size"]
        if not 3 <= self.vocab_size <= 65536:
            raise ValueError("Invalid uint16 vocabulary size")
        description = manifest["splits"][split]
        path = (manifest_path.parent / description["path"]).resolve()
        if (
            not path.is_relative_to(manifest_path.parent)
            or description["dtype"] != "<u2"
        ):
            raise ValueError("Token files must be local uint16 artifacts")
        if path.stat().st_size != 2 * description["num_tokens"]:
            raise ValueError("Token file length differs from its manifest")
        document_ids_verified = _verify_id_artifact(
            manifest_path.parent,
            description,
            path_key="document_ids_path",
            hash_key="document_ids_sha256",
            label="Document IDs",
            verify_hash=verify_hash,
            allow_hash_only=True,
        )
        tokenizer_training_ids_verified = _verify_id_artifact(
            manifest_path.parent,
            manifest,
            path_key="tokenizer_training_ids_path",
            hash_key="tokenizer_training_ids_sha256",
            label="Tokenizer training IDs",
            verify_hash=verify_hash,
        )
        if verify_hash:
            if file_hash(path) != description["sha256"]:
                raise ValueError("Token file hash differs from its manifest")
            if (
                file_hash(manifest_path.parent / manifest["tokenizer_path"])
                != manifest["tokenizer_sha256"]
            ):
                raise ValueError("Tokenizer hash differs from its manifest")
        self.sequence_length = sequence_length
        available = (
            (description["num_tokens"] - 1) // sequence_length
        ) * sequence_length
        if unique_tokens is None:
            unique_tokens = available
        if (
            isinstance(unique_tokens, bool)
            or not isinstance(unique_tokens, int)
            or unique_tokens < 1
            or unique_tokens % sequence_length
            or unique_tokens > available
        ):
            raise ValueError(
                "unique_tokens must be a positive multiple of sequence_length within the split"
            )
        self.unique_tokens = unique_tokens
        self.tokens = np.memmap(path, dtype="<u2", mode="r")
        self.identity = {
            "manifest_sha256": file_hash(manifest_path),
            "split": split,
            "token_sha256": description["sha256"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "document_ids_sha256": description.get("document_ids_sha256"),
            "document_ids_hash_verified": document_ids_verified,
            "tokenizer_training_ids_sha256": manifest.get(
                "tokenizer_training_ids_sha256"
            ),
            "tokenizer_training_ids_hash_verified": tokenizer_training_ids_verified,
            "sequence_length": sequence_length,
            "unique_target_positions": unique_tokens,
            "available_target_positions": available,
            "test_access": split == "test",
            "hash_verified": verify_hash,
            "vocab_size": self.vocab_size,
        }

    def __len__(self):
        return self.unique_tokens // self.sequence_length

    def get_batch(self, indices) -> dict[str, torch.Tensor]:
        indices = np.asarray(indices)
        if (
            indices.ndim != 1
            or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
        ):
            raise ValueError("Batch indices must be a nonempty integer vector")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Token window outside the declared unique-data prefix")
        offsets = indices[:, None] * self.sequence_length + np.arange(
            self.sequence_length + 1
        )
        batch = np.asarray(self.tokens[offsets], dtype=np.int64)
        if batch.max() >= self.vocab_size:
            raise ValueError("Packed token exceeds the declared vocabulary")
        return {
            "input_ids": torch.from_numpy(batch[:, :-1].copy()),
            "labels": torch.from_numpy(batch[:, 1:].copy()),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.source_config.read_text())
    sources = config["sources"]
    for source in sources:
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {source['path']}")

    def documents():
        import pyarrow.parquet as pq

        seen = 0
        for source in sources:
            for batch in pq.ParquetFile(source["path"]).iter_batches(
                batch_size=128, columns=["text"]
            ):
                for text in batch.column(0).to_pylist():
                    if seen >= config["max_documents"]:
                        return
                    seen += 1
                    yield text

    result = prepare_corpus(
        documents(),
        args.output_dir,
        vocab_size=config.get("vocab_size", 16384),
        tokenizer_train_documents=config.get("tokenizer_train_documents", 10000),
        split_seed=config.get("split_seed", 20260911),
        provenance={
            "source_config": config,
            "source_config_sha256": file_hash(args.source_config),
            "preparation_source_sha256": file_hash(__file__),
            "intended_use": config.get("intended_use", "systems_qualification"),
        },
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "vocab_size": result["vocab_size"],
                "tokens": {
                    key: value["num_tokens"] for key, value in result["splits"].items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
