"""Data-source loaders and activation capture for transformer replacement training."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from dendritic_modeling.config import Config
from dendritic_modeling.networks.architectures.replacement.cells import (
    require_runtime_tensor_contract,
)
from dendritic_modeling.networks.architectures.transformer import (
    DenseMLPControl,
    ReplacementRecord,
    apply_collapsed_population_spans,
    build_selected_transformer_replacement_from_dims,
    build_transformer_replacement_for_mlp,
    resolve_transformer_layers,
    restore_transformer_mlp_layers,
    validate_collapsed_replacement_spans,
)
from dendritic_modeling.networks.architectures.transformer.utils import _get_attr_path
from dendritic_modeling.networks.checkpoints import sha256_file
from dendritic_modeling.scripts.text.frozen_text_windows import (
    VerifiedFrozenTextWindows,
    _tensor_content_sha256,
    verify_frozen_text_windows,
)
from dendritic_modeling.training._transformer_replacement.builders import (
    _apply_parameter_tied_population_groups_to_units,
    _make_replacement,
    _make_single_layer_ei_stack_replacement,
    _replacement_kind,
)
from dendritic_modeling.training._transformer_replacement.common import (
    DistillationUnit,
    _to_plain_mapping,
    resolve_configured_replacement_layers,
)
from dendritic_modeling.training.replacement_common import _load_tensor_payload
from dendritic_modeling.utils import exact_file_inventory
from dendritic_modeling.utils.hooks import ForwardHookRemovalMixin, register_hook_groups

from .curriculum_stream import (
    CurriculumOrderedWindowBatchSource,
    build_curriculum_window_source,
)
from .model_sources import (
    TransformerModelSourceSession,
    normalize_transformer_model_source,
)
from .token_stream import (
    FrozenWindowBatchSource,
    StreamingTokenBatchSource,
    resolve_data_files,
)

_FROZEN_EXECUTION_BINDING_SCHEMA = "dendritic_campaign_bound_frozen_window_execution/v1"
_FROZEN_EXECUTION_BINDINGS: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class _ExecutionBoundFrozenTextWindows:
    """Compact frozen-window identity backed by a live distributed SHA gate."""

    path: Path
    manifest_file_sha256: str
    semantic_sha256: str
    tensor_path: Path
    tensor_file_sha256: str
    tensor_content_sha256: str
    sequence_length: int
    window_count: int
    primary_tokenizer_artifact_set_sha256: str
    primary_tokenizer_model_root: Path
    data_file_set_sha256: str
    evidentiary: bool
    source_file_count: int
    source_document_count: int
    comparison_tokenizer_equivalence_audited: bool
    comparison_tokenizer_equivalence_scope: str | None
    tensor_preflight_snapshot: dict[str, Any]
    campaign_manifest_file_sha256: str
    inventory_sha256: str
    verification_receipt_sha256: str

    def load_input_ids(self) -> torch.Tensor:
        """Load one tensor only while its exact-hash snapshot remains stable."""

        before = exact_file_inventory.file_snapshot(self.tensor_path)
        if before != self.tensor_preflight_snapshot:
            raise ValueError(
                "frozen tensor changed after distributed exact-SHA verification"
            )
        try:
            value = torch.load(self.tensor_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ValueError(f"cannot load frozen tensor {self.tensor_path}") from exc
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.long
            or value.ndim != 2
        ):
            raise TypeError("frozen .pt tensor must be rank-two CPU torch.int64")
        input_ids = value.contiguous()
        if list(input_ids.shape) != [self.window_count, self.sequence_length]:
            raise ValueError("frozen tensor shape differs from bound execution receipt")
        if _tensor_content_sha256(input_ids) != self.tensor_content_sha256:
            raise ValueError("frozen tensor semantic content differs from receipt")
        if exact_file_inventory.file_snapshot(self.tensor_path) != before:
            raise ValueError("frozen tensor changed while it was being loaded")
        return input_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "campaign_bound_distributed_exact_sha256_verified",
            "manifest_path": str(self.path),
            "manifest_file_sha256": self.manifest_file_sha256,
            "semantic_sha256": self.semantic_sha256,
            "tensor_path": str(self.tensor_path),
            "tensor_file_sha256": self.tensor_file_sha256,
            "tensor_content_sha256": self.tensor_content_sha256,
            "shape": [self.window_count, self.sequence_length],
            "primary_tokenizer_artifact_set_sha256": (
                self.primary_tokenizer_artifact_set_sha256
            ),
            "primary_tokenizer_model_root": str(self.primary_tokenizer_model_root),
            "data_file_set_sha256": self.data_file_set_sha256,
            "evidentiary": self.evidentiary,
            "source_file_count": self.source_file_count,
            "source_document_count": self.source_document_count,
            "comparison_tokenizer_equivalence_audited": (
                self.comparison_tokenizer_equivalence_audited
            ),
            "comparison_tokenizer_equivalence_scope": (
                self.comparison_tokenizer_equivalence_scope
            ),
            "execution_binding": {
                "campaign_manifest_file_sha256": self.campaign_manifest_file_sha256,
                "inventory_sha256": self.inventory_sha256,
                "verification_receipt_sha256": self.verification_receipt_sha256,
                "semantic_reconstruction_repeated_at_runtime": False,
                "live_exact_sha256_completed_before_load": True,
            },
        }


def install_frozen_window_execution_binding(value: Mapping[str, Any]) -> str:
    """Install one process-local token after distributed verification succeeds."""

    required = {
        "schema",
        "campaign_manifest_file_sha256",
        "inventory_sha256",
        "verification_receipt_sha256",
        "verification",
        "source_document_group_overlap_count",
        "windows",
        "binding_sha256",
    }
    if set(value) != required or value.get("schema") != (
        _FROZEN_EXECUTION_BINDING_SCHEMA
    ):
        raise ValueError("frozen-window execution binding schema drifted")
    payload = {key: value[key] for key in required if key != "binding_sha256"}
    token = exact_file_inventory.canonical_sha256(payload)
    if token != value["binding_sha256"]:
        raise ValueError("frozen-window execution binding identity drifted")
    verification = value["verification"]
    verification_payload = (
        {key: verification[key] for key in verification if key != "receipt_sha256"}
        if isinstance(verification, Mapping)
        else {}
    )
    if (
        not isinstance(verification, Mapping)
        or verification.get("schema") != exact_file_inventory.VERIFICATION_SCHEMA
        or verification.get("status") != "verified_before_execution"
        or verification.get("inventory_sha256") != value["inventory_sha256"]
        or verification.get("receipt_sha256") != value["verification_receipt_sha256"]
        or exact_file_inventory.canonical_sha256(verification_payload)
        != verification.get("receipt_sha256")
        or verification.get("all_live_file_sha256_matched") is not True
        or verification.get("stat_only_cache_used") is not False
        or int(value["source_document_group_overlap_count"]) != 0
    ):
        raise ValueError("frozen-window execution verification is incomplete")
    windows = value["windows"]
    if not isinstance(windows, Mapping) or set(windows) != {"training", "validation"}:
        raise ValueError("frozen-window execution binding must contain one exact pair")
    for role in ("training", "validation"):
        row = windows[role]
        required_window = {
            "manifest_path",
            "manifest_file_sha256",
            "semantic_sha256",
            "tensor_path",
            "tensor_file_sha256",
            "tensor_content_sha256",
            "shape",
            "primary_tokenizer_artifact_set_sha256",
            "primary_tokenizer_model_root",
            "data_file_set_sha256",
            "evidentiary",
            "source_file_count",
            "source_document_count",
            "comparison_tokenizer_equivalence_audited",
            "comparison_tokenizer_equivalence_scope",
            "tensor_preflight_snapshot",
        }
        if not isinstance(row, Mapping) or set(row) != required_window:
            raise ValueError(f"frozen-window {role} execution descriptor drifted")
        if (
            not bool(row["evidentiary"])
            or len(row["shape"]) != 2
            or any(int(size) < 1 for size in row["shape"])
            or exact_file_inventory.file_snapshot(Path(row["tensor_path"]))
            != row["tensor_preflight_snapshot"]
        ):
            raise ValueError(f"frozen-window {role} execution descriptor is invalid")
    frozen_value = copy.deepcopy(dict(value))
    existing = _FROZEN_EXECUTION_BINDINGS.get(token)
    if existing is not None and existing != frozen_value:
        raise ValueError("frozen-window execution token was reused for another binding")
    _FROZEN_EXECUTION_BINDINGS[token] = frozen_value
    return token


def clear_frozen_window_execution_binding(token: str) -> None:
    """Remove one process-local operational verification token."""

    _FROZEN_EXECUTION_BINDINGS.pop(str(token), None)


def _execution_bound_frozen_pair(
    token: str,
    *,
    manifest_paths: tuple[Path, Path],
    expected_sequence_length: int,
    minimum_training_windows: int,
    minimum_validation_windows: int,
    expected_artifact_set: str,
    expected_model_root: Path | None,
    require_evidentiary: bool,
) -> tuple[_ExecutionBoundFrozenTextWindows, _ExecutionBoundFrozenTextWindows]:
    value = _FROZEN_EXECUTION_BINDINGS.get(str(token))
    if value is None:
        raise ValueError(
            "frozen_execution_binding_token was not installed by a live distributed "
            "verification in this process"
        )
    result: list[_ExecutionBoundFrozenTextWindows] = []
    for role, requested_manifest, minimum in zip(
        ("training", "validation"),
        manifest_paths,
        (minimum_training_windows, minimum_validation_windows),
        strict=True,
    ):
        row = value["windows"][role]
        manifest = Path(str(row["manifest_path"])).resolve(strict=True)
        if manifest != Path(requested_manifest).resolve(strict=True):
            raise ValueError(f"{role} frozen manifest differs from execution binding")
        shape = [int(size) for size in row["shape"]]
        if shape[0] < int(minimum) or shape[1] != int(expected_sequence_length):
            raise ValueError(f"{role} frozen-window geometry differs from execution")
        if (
            expected_artifact_set
            and row["primary_tokenizer_artifact_set_sha256"] != expected_artifact_set
        ):
            raise ValueError(f"{role} tokenizer artifact set differs from execution")
        model_root = Path(str(row["primary_tokenizer_model_root"])).resolve(strict=True)
        if expected_model_root is not None and model_root != Path(
            expected_model_root
        ).resolve(strict=True):
            raise ValueError(f"{role} tokenizer model root differs from execution")
        if require_evidentiary and not bool(row["evidentiary"]):
            raise ValueError(f"{role} frozen windows are not evidentiary")
        result.append(
            _ExecutionBoundFrozenTextWindows(
                path=manifest,
                manifest_file_sha256=str(row["manifest_file_sha256"]),
                semantic_sha256=str(row["semantic_sha256"]),
                tensor_path=Path(str(row["tensor_path"])).resolve(strict=True),
                tensor_file_sha256=str(row["tensor_file_sha256"]),
                tensor_content_sha256=str(row["tensor_content_sha256"]),
                sequence_length=shape[1],
                window_count=shape[0],
                primary_tokenizer_artifact_set_sha256=str(
                    row["primary_tokenizer_artifact_set_sha256"]
                ),
                primary_tokenizer_model_root=model_root,
                data_file_set_sha256=str(row["data_file_set_sha256"]),
                evidentiary=bool(row["evidentiary"]),
                source_file_count=int(row["source_file_count"]),
                source_document_count=int(row["source_document_count"]),
                comparison_tokenizer_equivalence_audited=bool(
                    row["comparison_tokenizer_equivalence_audited"]
                ),
                comparison_tokenizer_equivalence_scope=(
                    None
                    if row["comparison_tokenizer_equivalence_scope"] is None
                    else str(row["comparison_tokenizer_equivalence_scope"])
                ),
                tensor_preflight_snapshot=dict(row["tensor_preflight_snapshot"]),
                campaign_manifest_file_sha256=str(
                    value["campaign_manifest_file_sha256"]
                ),
                inventory_sha256=str(value["inventory_sha256"]),
                verification_receipt_sha256=str(value["verification_receipt_sha256"]),
            )
        )
    training, validation = result
    if training.semantic_sha256 == validation.semantic_sha256 or (
        training.tensor_content_sha256 == validation.tensor_content_sha256
    ):
        raise ValueError(
            "execution-bound frozen training and validation identities overlap"
        )
    return training, validation


def _sample_hidden_pairs(
    unit: DistillationUnit,
    *,
    n_samples: int,
    sequence_length: int,
    hidden_size: int,
    input_std: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> TensorDataset:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chunks_x: list[torch.Tensor] = []
    chunks_y: list[torch.Tensor] = []
    teacher = unit.teacher_mlp
    if teacher is None:
        raise ValueError("Synthetic hidden-pair sampling requires a teacher MLP")

    remaining = int(n_samples)
    with torch.no_grad():
        while remaining > 0:
            current = min(int(batch_size), remaining)
            x_cpu = torch.randn(
                current,
                int(sequence_length),
                int(hidden_size),
                generator=generator,
                dtype=torch.float32,
            )
            x_cpu = x_cpu * float(input_std)
            x = x_cpu.to(device=device, dtype=dtype)
            y = teacher(x).detach().to("cpu", dtype=torch.float32)
            chunks_x.append(x_cpu)
            chunks_y.append(y)
            remaining -= current

    return TensorDataset(torch.cat(chunks_x, dim=0), torch.cat(chunks_y, dim=0))


def _build_synthetic_datasets(
    units: Sequence[DistillationUnit],
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[TensorDataset], list[TensorDataset]]:
    train_cfg = config.training.transformer_replacement
    teacher_cfg = _to_plain_mapping(train_cfg.synthetic_teacher)
    hidden_size = int(train_cfg.hidden_size or 32)
    input_std = float(teacher_cfg.get("input_std", 1.0))
    base_seed = int(
        train_cfg.seed if train_cfg.seed is not None else config.experiment.seed
    )

    train_datasets = []
    valid_datasets = []
    for idx, unit in enumerate(units):
        train_datasets.append(
            _sample_hidden_pairs(
                unit,
                n_samples=train_cfg.train_samples,
                sequence_length=train_cfg.sequence_length,
                hidden_size=hidden_size,
                input_std=input_std,
                seed=base_seed + 17 * idx,
                device=device,
                dtype=dtype,
                batch_size=train_cfg.batch_size,
            )
        )
        valid_datasets.append(
            _sample_hidden_pairs(
                unit,
                n_samples=train_cfg.valid_samples,
                sequence_length=train_cfg.sequence_length,
                hidden_size=hidden_size,
                input_std=input_std,
                seed=base_seed + 10_003 + 17 * idx,
                device=device,
                dtype=dtype,
                batch_size=train_cfg.batch_size,
            )
        )
    return train_datasets, valid_datasets


def _load_tensor(path: str) -> Any:
    return _load_tensor_payload(
        path,
        empty_path_message="Hidden-cache tensor path is empty",
    )


def _select_layer_tensor(payload: Any, layer_index: int) -> torch.Tensor:
    if torch.is_tensor(payload):
        return payload
    if isinstance(payload, Mapping):
        for key in (layer_index, str(layer_index), f"layer_{layer_index}"):
            if key in payload:
                value = payload[key]
                if torch.is_tensor(value):
                    return value
        raise KeyError(f"No tensor for layer {layer_index} in hidden cache")
    raise TypeError("Hidden-cache payload must be a Tensor or dict of Tensors")


def _build_hidden_cache_units_and_datasets(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[DistillationUnit], list[TensorDataset], list[TensorDataset]]:
    train_cfg = config.training.transformer_replacement
    cache_cfg = _to_plain_mapping(train_cfg.hidden_cache)
    train_inputs = _load_tensor(str(cache_cfg.get("train_inputs", "")))
    train_targets = _load_tensor(str(cache_cfg.get("train_targets", "")))
    valid_inputs = _load_tensor(str(cache_cfg.get("valid_inputs", "")))
    valid_targets = _load_tensor(str(cache_cfg.get("valid_targets", "")))

    layer_indices = resolve_configured_replacement_layers(config, default=[0])
    units = []
    train_datasets = []
    valid_datasets = []
    for layer_index in layer_indices:
        x_train = _select_layer_tensor(train_inputs, int(layer_index)).float()
        y_train = _select_layer_tensor(train_targets, int(layer_index)).float()
        x_valid = _select_layer_tensor(valid_inputs, int(layer_index)).float()
        y_valid = _select_layer_tensor(valid_targets, int(layer_index)).float()
        hidden_size = int(x_train.shape[-1])
        selection = _to_plain_mapping(config.model.transformer_replacement.selection)
        plans_by_layer = _to_plain_mapping(
            config.model.transformer_replacement.compiled_plans_by_layer
        )
        if bool(selection.get("enabled", False)) or plans_by_layer:
            intermediate_size = getattr(train_cfg, "intermediate_size", None)
            if intermediate_size is None:
                raise ValueError(
                    "hidden_cache with FMI selection or compiled plans requires "
                    "training.transformer_replacement.intermediate_size"
                )
            if bool(selection.get("enabled", False)):
                replacement = build_selected_transformer_replacement_from_dims(
                    hidden_size=hidden_size,
                    teacher_intermediate_size=int(intermediate_size),
                    transformer_replacement=config.model.transformer_replacement,
                    layer_index=int(layer_index),
                )
            else:
                boundary = nn.Module()
                boundary.hidden_size = int(hidden_size)
                boundary.intermediate_size = int(intermediate_size)
                replacement = build_transformer_replacement_for_mlp(
                    boundary,
                    config.model.transformer_replacement,
                    config.model.core,
                    layer_index=int(layer_index),
                )
            replacement = replacement.to(device=device, dtype=dtype)
        else:
            replacement = _make_replacement(
                hidden_size=hidden_size,
                config=config,
                dtype=dtype,
                device=device,
            )
        units.append(
            DistillationUnit(layer_index=int(layer_index), replacement=replacement)
        )
        train_datasets.append(TensorDataset(x_train, y_train))
        valid_datasets.append(TensorDataset(x_valid, y_valid))
    return (
        _apply_parameter_tied_population_groups_to_units(config, units),
        train_datasets,
        valid_datasets,
    )


def _extract_mlp_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(
        "Captured MLP output must be a Tensor or tuple/list starting with one"
    )


class _MLPIOCapture(ForwardHookRemovalMixin):
    def __init__(
        self,
        model: nn.Module,
        layer_indices: Sequence[int],
        *,
        layers_attr: str | None,
        mlp_attr: str,
        capture_layer_hidden: bool = False,
    ):
        self.records: dict[int, dict[str, torch.Tensor]] = {}
        self.handles = []
        try:
            layers = resolve_transformer_layers(model, layers_attr=layers_attr)

            def _register_layer_hooks(layer_index_raw: int):
                layer_index = int(layer_index_raw)
                layer = layers[int(layer_index)]
                mlp = _get_attr_path(layer, mlp_attr)
                if not isinstance(mlp, nn.Module):
                    raise TypeError(
                        f"layer {layer_index} target {mlp_attr!r} is not an nn.Module"
                    )
                self.records[layer_index] = {}

                def _register_hook(hook_type: str):
                    if hook_type == "pre":
                        return [
                            mlp.register_forward_pre_hook(
                                self._make_pre_hook(layer_index)
                            )
                        ]
                    if hook_type == "layer_pre":
                        return [
                            layer.register_forward_pre_hook(
                                self._make_layer_hidden_hook(layer_index)
                            )
                        ]
                    return [mlp.register_forward_hook(self._make_hook(layer_index))]

                hook_types = ("pre", "forward")
                if capture_layer_hidden:
                    hook_types = ("pre", "forward", "layer_pre")
                return register_hook_groups(
                    hook_types,
                    _register_hook,
                )

            self.handles = register_hook_groups(layer_indices, _register_layer_hooks)
        except Exception:
            self.close()
            raise

    def _make_pre_hook(self, layer_index: int):
        def hook(module: nn.Module, args: tuple[Any, ...]) -> None:
            del module
            self.records[layer_index]["input"] = args[0].detach()

        return hook

    def _make_layer_hidden_hook(self, layer_index: int):
        def hook(module: nn.Module, args: tuple[Any, ...]) -> None:
            del module
            self.records[layer_index]["hidden"] = args[0].detach()

        return hook

    def _make_hook(self, layer_index: int):
        def hook(module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
            del module, args
            self.records[layer_index]["target"] = _extract_mlp_output(output).detach()

        return hook

    def clear(self) -> None:
        for value in self.records.values():
            value.clear()

    def close(self) -> None:
        self.remove_forward_hooks(self.handles)
        self.handles.clear()


def _flatten_tokenizer_input_ids(value: Any) -> list[int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, Sequence) and value and isinstance(value[0], Sequence):
        value = value[0]
    return [int(token_id) for token_id in value]


def _tokenize_joined_documents(
    documents: Iterable[str],
    tokenizer: Any,
    *,
    max_tokens: int | None,
) -> torch.Tensor:
    """Tokenize bounded documents as one text without artificial EOS labels."""

    text = "\n\n".join(str(value) for value in documents)
    encoded = tokenizer(text, add_special_tokens=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise TypeError("Tokenizer output must contain an input_ids field")
    token_ids = _flatten_tokenizer_input_ids(encoded["input_ids"])
    if max_tokens is not None:
        del token_ids[int(max_tokens) :]
    return torch.tensor(token_ids, dtype=torch.long)


def _load_text_dataset(
    text_cfg: Mapping[str, Any],
    *,
    split: str,
    data_files: Any,
    streaming: bool,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("datasets is required for hf_text dataset sources") from exc

    dataset_name = str(text_cfg.get("dataset_name", "") or "")
    normalized_files = resolve_data_files(data_files)
    if not dataset_name:
        if normalized_files is None:
            raise ValueError(
                "hf_text source requires text.text_path, text.dataset_name, "
                "or text.data_files"
            )
        dataset_name = "json"
    dataset_config = text_cfg.get("dataset_config") or None
    args = [dataset_name]
    if dataset_config is not None:
        args.append(str(dataset_config))
    kwargs: dict[str, Any] = {
        "split": str(split),
        "streaming": bool(streaming),
    }
    if normalized_files is not None:
        kwargs["data_files"] = normalized_files
    return load_dataset(*args, **kwargs)


def _validation_text_config(text_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve an optional evaluation corpus without mutating training data."""

    resolved = dict(text_cfg)
    validation_name = str(text_cfg.get("validation_dataset_name", "") or "")
    if validation_name:
        resolved["dataset_name"] = validation_name
        resolved["dataset_config"] = text_cfg.get("validation_dataset_config") or None
    return resolved


def _iter_text_field(
    dataset: Iterable[Mapping[str, Any]],
    *,
    text_field: str,
    max_documents: int | None,
) -> Iterable[str]:
    for index, row in enumerate(dataset):
        if max_documents is not None and index >= int(max_documents):
            break
        value = str(row.get(text_field, ""))
        if value.strip():
            yield value


SYNTHETIC_COPY_KIND = "synthetic_copy"


def _normalize_synthetic_copy_source(
    source: Mapping[str, Any], *, index: int
) -> dict[str, Any]:
    """Validate one ``kind: synthetic_copy`` mixture source (defaults filled).

    The source emits documents of the form ``X X`` where ``X`` is a run of
    ``tokens_per_half`` random vocabulary words, so the packed training
    stream carries the random-copy (induction) task that plain prose does
    not exercise.  ``seed`` must differ from the chain's copy-probe seed so
    the probe stays held out; ``documents_per_cycle`` bounds one cycle of
    the auxiliary source before it reshuffles.
    """

    def _int(name: str, default: int, minimum: int) -> int:
        raw = source.get(name, default)
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"text.mixture[{index}].{name} must be an integer"
            ) from exc
        if value < minimum:
            raise ValueError(
                f"text.mixture[{index}].{name} must be >= {minimum}, got {value}"
            )
        return value

    return {
        "tokens_per_half": _int("tokens_per_half", 255, 8),
        "documents_per_cycle": _int("documents_per_cycle", 4096, 1),
        "seed": _int("seed", 90210, 0),
    }


def _copyable_vocabulary(tokenizer: Any) -> list[int]:
    """Token ids that decode to one leading-space word and re-encode to themselves.

    Restricting the copy alphabet to such tokens makes ``decode(ids)`` a
    text whose re-tokenization reproduces ``ids`` (pre-tokenization splits
    at the leading space, so BPE cannot merge across neighbours), i.e. the
    packed training windows really contain token-exact repeats.
    """

    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    pool: list[int] = []
    for token_id in range(vocab_size):
        text = tokenizer.decode([token_id])
        if (
            len(text) < 2
            or not text.startswith(" ")
            or any(ch.isspace() for ch in text[1:])
            or not text[1:].isprintable()
        ):
            continue
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if list(encoded) == [token_id]:
            pool.append(token_id)
    if len(pool) < 64:
        raise ValueError(
            f"synthetic_copy: only {len(pool)} round-trippable vocabulary "
            "tokens; the tokenizer cannot host the copy task (fail closed)"
        )
    return pool


def _synthetic_copy_document_factory(
    source_cfg: Mapping[str, Any],
    *,
    tokenizer: Any,
    base_seed: int,
    world_size: int,
    rank: int,
) -> Callable[[int], Iterable[str]]:
    """Per-cycle generator of ``X X`` random-word copy documents."""

    tokens_per_half = int(source_cfg["tokens_per_half"])
    documents_per_cycle = int(source_cfg["documents_per_cycle"])
    pool_holder: list[list[int]] = []

    def factory(cycle: int) -> Iterable[str]:
        if not pool_holder:
            pool_holder.append(_copyable_vocabulary(tokenizer))
        pool = pool_holder[0]
        rng = random.Random(
            int(source_cfg["seed"]) + int(base_seed) + 104729 * int(cycle)
        )
        for document_index in range(documents_per_cycle):
            ids = [rng.choice(pool) for _ in range(tokens_per_half)]
            if world_size > 1 and document_index % world_size != rank:
                continue
            half = tokenizer.decode(ids)
            yield half + half

    return factory


def _append_synthetic_copy_tokens(
    train_tokens: torch.Tensor,
    sources: Sequence[Mapping[str, Any]] | None,
    tokenizer: Any,
    *,
    seed_offset: int = 0,
) -> torch.Tensor:
    """Fixed-token-path counterpart of the streaming ``synthetic_copy`` source.

    Appends ``[X X eos]`` copy documents (token ids drawn from the
    round-trippable vocabulary) so that a fraction ``weight`` of the final
    training tensor is copy text: ``n_copy = n_base * w / (1 - w)``.  The
    fixed path samples training windows uniformly over the flat tensor, so
    this matches the streaming source's document-level weight.  Validation
    tokens are never touched.
    """

    copy_sources = [
        source
        for source in (sources or [])
        if source.get("kind") == SYNTHETIC_COPY_KIND
    ]
    if not copy_sources:
        return train_tokens
    pool = _copyable_vocabulary(tokenizer)
    eos = getattr(tokenizer, "eos_token_id", None)
    n_base = int(train_tokens.numel())
    pieces = [train_tokens]
    for source in copy_sources:
        weight = float(source["weight"])
        target = round(n_base * weight / (1.0 - weight))
        rng = random.Random(int(source["seed"]) + int(seed_offset))
        ids: list[int] = []
        while len(ids) < target:
            half = [rng.choice(pool) for _ in range(int(source["tokens_per_half"]))]
            ids.extend(half)
            ids.extend(half)
            if eos is not None:
                ids.append(int(eos))
        pieces.append(torch.tensor(ids[:target], dtype=train_tokens.dtype))
    return torch.cat(pieces)


def _normalize_mixture_sources(
    text_cfg: Mapping[str, Any],
) -> list[dict[str, Any]] | None:
    """Validate ``text.mixture`` (default off) into weighted source configs."""

    raw = text_cfg.get("mixture")
    if raw in (None, "", []):
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("text.mixture must be a list of source mappings")
    sources: list[dict[str, Any]] = []
    total_weight = 0.0
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ValueError(f"text.mixture[{index}] must be a mapping")
        source = dict(entry)
        try:
            weight = float(source.get("weight"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"text.mixture[{index}].weight must be a number in (0, 1)"
            ) from exc
        if not 0.0 < weight < 1.0:
            raise ValueError(
                f"text.mixture[{index}].weight must be in (0, 1), got {weight}"
            )
        kind = str(source.get("kind", "dataset") or "dataset")
        if kind == SYNTHETIC_COPY_KIND:
            source.update(_normalize_synthetic_copy_source(source, index=index))
        elif kind != "dataset":
            raise ValueError(
                f"text.mixture[{index}].kind must be 'dataset' or "
                f"'{SYNTHETIC_COPY_KIND}', got {kind!r}"
            )
        else:
            has_name = bool(str(source.get("dataset_name", "") or ""))
            has_files = source.get("data_files") not in (None, "", [])
            if not has_name and not has_files:
                raise ValueError(
                    f"text.mixture[{index}] requires dataset_name or data_files"
                )
        source["kind"] = kind
        source["weight"] = weight
        total_weight += weight
        sources.append(source)
    if total_weight >= 1.0:
        raise ValueError(
            "text.mixture weights must sum below 1 so the primary corpus "
            f"keeps positive probability, got {total_weight}"
        )
    return sources


def _iter_source_text(
    dataset: Iterable[Mapping[str, Any]],
    *,
    source_cfg: Mapping[str, Any],
    max_documents: int | None,
) -> Iterable[str]:
    """Yield one text per row, joining ``text_fields`` when configured."""

    fields = source_cfg.get("text_fields")
    if fields in (None, "", []):
        yield from _iter_text_field(
            dataset,
            text_field=str(source_cfg.get("text_field", "text")),
            max_documents=max_documents,
        )
        return
    if isinstance(fields, (str, bytes)) or not isinstance(fields, Sequence):
        raise ValueError("text_fields must be a list of row field names")
    field_names = [str(name) for name in fields]
    for index, row in enumerate(dataset):
        if max_documents is not None and index >= int(max_documents):
            break
        parts = [str(row.get(name, "")) for name in field_names]
        value = "\n".join(part for part in parts if part.strip())
        if value.strip():
            yield value


def _mixture_document_factory(
    source_cfg: Mapping[str, Any],
    *,
    base_seed: int,
    shuffle_buffer: int,
    world_size: int,
    rank: int,
) -> Callable[[int], Iterable[str]]:
    """Build a per-cycle document loader for one auxiliary mixture source.

    Each cycle reshuffles with a distinct seed so a small corpus that is
    cycled during one primary epoch is oversampled in a different order.
    """

    def factory(cycle: int) -> Iterable[str]:
        streaming_flag = bool(source_cfg.get("streaming", True))
        dataset = _load_text_dataset(
            source_cfg,
            split=str(source_cfg.get("split", "train")),
            data_files=source_cfg.get("data_files"),
            streaming=streaming_flag,
        )
        seed = int(base_seed) + 104729 * int(cycle)
        if streaming_flag:
            if shuffle_buffer > 1:
                dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
        else:
            dataset = dataset.shuffle(seed=seed)
        if world_size > 1:
            dataset = dataset.shard(num_shards=world_size, index=rank)
        max_documents_raw = source_cfg.get("max_documents")
        return _iter_source_text(
            dataset,
            source_cfg=source_cfg,
            max_documents=(
                None if max_documents_raw is None else int(max_documents_raw)
            ),
        )

    return factory


def _interleave_documents(
    primary_documents: Iterable[str],
    source_factories: Sequence[Callable[[int], Iterable[str]]],
    weights: Sequence[float],
    *,
    seed: int,
) -> Iterator[str]:
    """Weighted document interleave: primary exhaustion ends the epoch,
    auxiliary sources cycle (reshuffled) when exhausted."""

    if len(source_factories) != len(weights):
        raise ValueError("mixture factories and weights must align")
    rng = random.Random(int(seed))
    weight_vector = [1.0 - float(sum(weights)), *[float(w) for w in weights]]
    population = list(range(len(weight_vector)))
    iterators: list[Iterator[str]] = [iter(primary_documents)]
    iterators.extend(iter(factory(0)) for factory in source_factories)
    cycles = [0] * len(iterators)
    while True:
        index = rng.choices(population, weights=weight_vector, k=1)[0]
        if index == 0:
            try:
                value = next(iterators[0])
            except StopIteration:
                return
            yield value
            continue
        value = None
        for _attempt in range(2):
            try:
                value = next(iterators[index])
                break
            except StopIteration:
                cycles[index] += 1
                iterators[index] = iter(source_factories[index - 1](cycles[index]))
        if value is None:
            raise ValueError(f"text.mixture source {index - 1} produced no documents")
        yield value


def _build_streaming_token_source(
    config: Config,
    tokenizer: Any,
    text_cfg: Mapping[str, Any],
) -> tuple[
    StreamingTokenBatchSource | CurriculumOrderedWindowBatchSource,
    torch.Tensor,
]:
    """Build a packed training stream and a bounded deterministic validation set."""
    train_cfg = config.training.transformer_replacement
    split = str(text_cfg.get("split", "train"))
    text_field = str(text_cfg.get("text_field", "text"))
    train_files = text_cfg.get("data_files")
    validation_files = text_cfg.get("validation_data_files")
    validation_split = str(text_cfg.get("validation_split", "") or "")
    validation_text_cfg = _validation_text_config(text_cfg)
    validation_text_field = str(
        text_cfg.get("validation_text_field", text_field) or text_field
    )
    validation_documents = max(1, int(text_cfg.get("validation_documents", 64)))
    separate_validation = (
        validation_files not in (None, "", [])
        or bool(validation_split)
        or bool(text_cfg.get("validation_dataset_name"))
    )

    validation_dataset = _load_text_dataset(
        validation_text_cfg,
        split=validation_split or split,
        data_files=validation_files if separate_validation else train_files,
        streaming=True,
    )
    validation_max_tokens_raw = text_cfg.get("validation_max_tokens")
    if validation_max_tokens_raw is None:
        validation_max_tokens_raw = max(
            int(train_cfg.sequence_length) + 1,
            int(train_cfg.valid_samples) * int(train_cfg.sequence_length),
        )
    validation_tokens = _tokenize_joined_documents(
        _iter_text_field(
            validation_dataset,
            text_field=validation_text_field,
            max_documents=validation_documents,
        ),
        tokenizer,
        max_tokens=int(validation_max_tokens_raw),
    )
    min_tokens = max(4, int(train_cfg.sequence_length) + 1)
    if validation_tokens.numel() < min_tokens:
        raise ValueError(
            f"Need at least {min_tokens} validation tokens for hf_text distillation"
        )

    base_seed = int(
        text_cfg.get("shuffle_seed")
        if text_cfg.get("shuffle_seed") is not None
        else (train_cfg.seed or config.experiment.seed)
    )
    shuffle_buffer = int(text_cfg.get("shuffle_buffer_size", 10_000))
    max_documents_raw = text_cfg.get("max_documents")
    max_documents = None if max_documents_raw is None else int(max_documents_raw)
    world_size = int(text_cfg.get("world_size") or os.environ.get("WORLD_SIZE", 1))
    rank = int(text_cfg.get("rank") or os.environ.get("RANK", 0))
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid text-stream rank/world_size: {rank}/{world_size}")

    curriculum_manifest = str(
        text_cfg.get("curriculum_order_manifest", "") or ""
    ).strip()
    if curriculum_manifest:
        # Default-off curriculum mode: stream the sealed manifest's windows
        # in teacher-loss order instead of shuffled packed documents.  The
        # validation stream above stays byte-identical to the shuffled path.
        stream = build_curriculum_window_source(
            manifest_path=curriculum_manifest,
            text_cfg=text_cfg,
            expected_tokenizer_name=str(
                text_cfg.get("tokenizer_name")
                or config.model.transformer_replacement.model_name
            ),
            expected_sequence_length=int(train_cfg.sequence_length),
            expected_separator_token_id=getattr(tokenizer, "eos_token_id", None),
            rank=rank,
            world_size=world_size,
        )
        resume_stream_state = _resolve_resume_stream_state(
            train_cfg,
            text_cfg=text_cfg,
            rank=rank,
            world_size=world_size,
        )
        if resume_stream_state:
            stream.load_state(resume_stream_state)
        return stream, validation_tokens

    mixture_sources = _normalize_mixture_sources(text_cfg)

    def document_factory(epoch: int) -> Iterable[str]:
        dataset = _load_text_dataset(
            text_cfg,
            split=split,
            data_files=train_files,
            streaming=True,
        )
        if not separate_validation and validation_documents:
            dataset = dataset.skip(validation_documents)
        if shuffle_buffer > 1:
            dataset = dataset.shuffle(
                seed=base_seed + int(epoch),
                buffer_size=shuffle_buffer,
            )
        if world_size > 1:
            dataset = dataset.shard(num_shards=world_size, index=rank)
        primary_documents = _iter_text_field(
            dataset,
            text_field=text_field,
            max_documents=max_documents,
        )
        if not mixture_sources:
            return primary_documents
        factories = [
            (
                _synthetic_copy_document_factory(
                    source,
                    tokenizer=tokenizer,
                    base_seed=base_seed + 1000003 * int(epoch) + 31 * index,
                    world_size=world_size,
                    rank=rank,
                )
                if source.get("kind") == SYNTHETIC_COPY_KIND
                else _mixture_document_factory(
                    source,
                    base_seed=base_seed + 1000003 * int(epoch) + 31 * index,
                    shuffle_buffer=shuffle_buffer,
                    world_size=world_size,
                    rank=rank,
                )
            )
            for index, source in enumerate(mixture_sources)
        ]
        return _interleave_documents(
            primary_documents,
            factories,
            [float(source["weight"]) for source in mixture_sources],
            seed=base_seed + 7919 * int(epoch) + rank,
        )

    separator = getattr(tokenizer, "eos_token_id", None)
    stream = StreamingTokenBatchSource(
        document_factory,
        tokenizer,
        sequence_length=int(train_cfg.sequence_length),
        separator_token_id=separator,
        max_tokens_per_document=(
            None
            if text_cfg.get("max_tokens_per_document") is None
            else int(text_cfg["max_tokens_per_document"])
        ),
    )
    resume_stream_state = _resolve_resume_stream_state(
        train_cfg,
        text_cfg=text_cfg,
        rank=rank,
        world_size=world_size,
    )
    if resume_stream_state:
        stream.load_state(resume_stream_state)
    return stream, validation_tokens


def _frozen_manifest_paths(
    text_cfg: Mapping[str, Any],
) -> tuple[Path, Path] | None:
    training_raw = str(text_cfg.get("frozen_training_manifest", "") or "").strip()
    validation_raw = str(text_cfg.get("frozen_validation_manifest", "") or "").strip()
    if bool(training_raw) != bool(validation_raw):
        raise ValueError(
            "text.frozen_training_manifest and text.frozen_validation_manifest "
            "must be configured together"
        )
    if not training_raw:
        return None
    return Path(training_raw), Path(validation_raw)


def _resolve_resume_stream_state(
    train_cfg: Any,
    *,
    text_cfg: Mapping[str, Any],
    rank: int,
    world_size: int,
) -> str:
    """Resolve legacy external stream state without shadowing joint state.

    Exact joint checkpoints carry each rank's token-stream state in the
    corresponding ``training_checkpoint.rankN.pt`` worker payload.  Loading a
    sidecar before that checkpoint is both redundant and unsafe: a stale or
    absent sidecar can prevent an otherwise complete joint restart.  The
    layerwise checkpoint path retains its historical sidecar contract.
    """

    explicit = str(text_cfg.get("resume_stream_state", "") or "")
    resume_checkpoint = str(getattr(train_cfg, "resume_checkpoint", "") or "")
    mode = str(getattr(train_cfg, "mode", "") or "").strip().lower()
    if resume_checkpoint and mode == "joint_lm_distillation":
        if explicit:
            raise ValueError(
                "joint_lm_distillation restart state is embedded rank-locally in "
                "the joint checkpoint; text.resume_stream_state must be empty"
            )
        return ""
    if explicit or not resume_checkpoint:
        return explicit
    state_name = (
        "token_stream_checkpoint_state.json"
        if world_size == 1
        else f"token_stream_checkpoint_state.rank{rank}.json"
    )
    return os.path.join(
        os.path.dirname(resume_checkpoint) or str(train_cfg.save_dir),
        state_name,
    )


def _reject_mixed_frozen_text_sources(text_cfg: Mapping[str, Any]) -> None:
    conflicts = []
    for key in (
        "text_path",
        "dataset_name",
        "dataset_config",
        "data_files",
        "validation_dataset_name",
        "validation_dataset_config",
        "validation_data_files",
        "validation_split",
    ):
        if text_cfg.get(key) not in (None, "", []):
            conflicts.append(key)
    if bool(text_cfg.get("streaming", False)):
        conflicts.append("streaming")
    if conflicts:
        raise ValueError(
            "frozen text windows cannot be combined with raw/Hugging Face source "
            f"fields: {', '.join(sorted(conflicts))}"
        )


def _flatten_window_groups(verified: VerifiedFrozenTextWindows) -> set[str]:
    return {
        str(group)
        for window_groups in verified.window_source_document_group_sha256s
        for group in window_groups
    }


def _write_frozen_input_receipt(
    *,
    train_cfg: Any,
    training: VerifiedFrozenTextWindows,
    validation: VerifiedFrozenTextWindows,
    seed: int,
    rank: int,
    world_size: int,
) -> None:
    if int(rank) != 0:
        return
    output = Path(str(train_cfg.save_dir)) / "frozen_text_window_inputs.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "dendritic_frozen_training_inputs/v1",
        "status": "verified",
        "training": training.as_dict(),
        "validation": validation.as_dict(),
        "source_document_group_overlap_count": 0,
        "stream": {
            "seed": int(seed),
            "rank_count": int(world_size),
            "policy": "epoch_seeded_permutation_without_replacement_per_epoch",
            "ddp_sharding": "equal_strided_columns_after_shuffled_tail_drop",
        },
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)


def _build_frozen_token_source(
    config: Config,
    text_cfg: Mapping[str, Any],
    manifest_paths: tuple[Path, Path],
) -> tuple[FrozenWindowBatchSource, torch.Tensor]:
    """Strictly verify and load disjoint frozen training/validation windows."""

    _reject_mixed_frozen_text_sources(text_cfg)
    train_cfg = config.training.transformer_replacement
    model_cfg = config.model.transformer_replacement
    require_evidentiary = text_cfg.get("frozen_require_evidentiary", True)
    if not isinstance(require_evidentiary, bool):
        raise TypeError("text.frozen_require_evidentiary must be boolean")

    expected_artifact_set = str(
        text_cfg.get("frozen_primary_tokenizer_artifact_set_sha256", "") or ""
    ).strip()
    expected_model_root_raw = str(
        text_cfg.get("frozen_primary_tokenizer_model_root", "") or ""
    ).strip()
    expected_model_root: Path | None = None
    if expected_model_root_raw:
        expected_model_root = Path(expected_model_root_raw)
    else:
        configured_model = Path(str(model_cfg.model_name)).expanduser()
        if configured_model.is_dir():
            expected_model_root = configured_model
    if not expected_artifact_set and expected_model_root is None:
        raise ValueError(
            "frozen windows require either a local transformer model path or "
            "text.frozen_primary_tokenizer_artifact_set_sha256"
        )

    world_size = int(text_cfg.get("world_size") or os.environ.get("WORLD_SIZE", 1))
    rank = int(text_cfg.get("rank") or os.environ.get("RANK", 0))
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid frozen text rank/world_size: {rank}/{world_size}")
    minimum_training_windows = int(train_cfg.batch_size) * world_size
    minimum_validation_windows = max(
        int(train_cfg.batch_size), int(train_cfg.valid_samples)
    )
    execution_token = str(
        text_cfg.get("frozen_execution_binding_token", "") or ""
    ).strip()
    if execution_token:
        training, validation = _execution_bound_frozen_pair(
            execution_token,
            manifest_paths=manifest_paths,
            expected_sequence_length=int(train_cfg.sequence_length),
            minimum_training_windows=minimum_training_windows,
            minimum_validation_windows=minimum_validation_windows,
            expected_artifact_set=expected_artifact_set,
            expected_model_root=expected_model_root,
            require_evidentiary=require_evidentiary,
        )
    else:
        verification_kwargs = {
            "expected_sequence_length": int(train_cfg.sequence_length),
            "expected_primary_tokenizer_artifact_set_sha256": (
                expected_artifact_set or None
            ),
            "expected_primary_tokenizer_model_root": expected_model_root,
            "require_evidentiary": require_evidentiary,
        }
        training = verify_frozen_text_windows(
            manifest_paths[0],
            minimum_windows=minimum_training_windows,
            **verification_kwargs,
        )
        validation = verify_frozen_text_windows(
            manifest_paths[1],
            minimum_windows=minimum_validation_windows,
            **verification_kwargs,
        )
    if training.semantic_sha256 == validation.semantic_sha256:
        raise ValueError("training and validation frozen manifests are identical")
    if training.tensor_content_sha256 == validation.tensor_content_sha256:
        raise ValueError("training and validation frozen tensors are identical")
    if execution_token:
        binding = _FROZEN_EXECUTION_BINDINGS[execution_token]
        if int(binding["source_document_group_overlap_count"]) != 0:
            raise ValueError("execution-bound frozen windows are not group-disjoint")
    else:
        overlap = _flatten_window_groups(training) & _flatten_window_groups(validation)
        if overlap:
            raise ValueError(
                "training and validation frozen windows share "
                f"{len(overlap)} source document group(s)"
            )

    training_input_ids = training.load_input_ids()
    validation_input_ids = validation.load_input_ids()
    stream_seed_raw = text_cfg.get("frozen_shuffle_seed")
    if stream_seed_raw is None:
        stream_seed_raw = text_cfg.get("shuffle_seed")
    stream_seed = int(
        stream_seed_raw
        if stream_seed_raw is not None
        else (train_cfg.seed if train_cfg.seed is not None else config.experiment.seed)
    )
    stream = FrozenWindowBatchSource(
        training_input_ids,
        seed=stream_seed,
        rank=rank,
        world_size=world_size,
        manifest_semantic_sha256=training.semantic_sha256,
        tensor_content_sha256=training.tensor_content_sha256,
    )
    resume_stream_state = _resolve_resume_stream_state(
        train_cfg,
        text_cfg=text_cfg,
        rank=rank,
        world_size=world_size,
    )
    if resume_stream_state:
        stream.load_state(resume_stream_state)
    _write_frozen_input_receipt(
        train_cfg=train_cfg,
        training=training,
        validation=validation,
        seed=stream_seed,
        rank=rank,
        world_size=world_size,
    )
    return stream, validation_input_ids


def _read_token_source(
    config: Config,
) -> tuple[
    torch.Tensor | StreamingTokenBatchSource | FrozenWindowBatchSource,
    torch.Tensor,
]:
    train_cfg = config.training.transformer_replacement
    text_cfg = _to_plain_mapping(train_cfg.text)
    model_cfg = config.model.transformer_replacement
    frozen_manifests = _frozen_manifest_paths(text_cfg)
    mixture_requested = text_cfg.get("mixture") not in (None, "", [])
    curriculum_requested = bool(
        str(text_cfg.get("curriculum_order_manifest", "") or "").strip()
    )
    if mixture_requested and frozen_manifests is not None:
        raise ValueError("text.mixture cannot be combined with frozen text manifests")
    if curriculum_requested and frozen_manifests is not None:
        raise ValueError(
            "text.curriculum_order_manifest cannot be combined with frozen "
            "text manifests"
        )
    if curriculum_requested and mixture_requested:
        raise ValueError(
            "text.curriculum_order_manifest cannot be combined with text.mixture"
        )
    if frozen_manifests is not None:
        return _build_frozen_token_source(config, text_cfg, frozen_manifests)
    tokenizer_name = str(text_cfg.get("tokenizer_name") or model_cfg.model_name)
    text_path = str(text_cfg.get("text_path", ""))
    fixed_path_copy_sources: list[dict[str, Any]] | None = None
    if mixture_requested and (not bool(text_cfg.get("streaming", False)) or text_path):
        normalized = _normalize_mixture_sources(text_cfg) or []
        if text_path or any(
            source.get("kind") != SYNTHETIC_COPY_KIND for source in normalized
        ):
            raise ValueError(
                "text.mixture requires text.streaming: true and no text.text_path "
                f"(only {SYNTHETIC_COPY_KIND} sources may augment the fixed-token "
                "dataset path)"
            )
        fixed_path_copy_sources = normalized
    if curriculum_requested and (
        not bool(text_cfg.get("streaming", False)) or text_path
    ):
        raise ValueError(
            "text.curriculum_order_manifest requires text.streaming: true and "
            "no text.text_path"
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for hf_text distillation") from exc

    tokenizer_kwargs = _to_plain_mapping(text_cfg.get("tokenizer_kwargs", {}))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tokenizer_kwargs)

    if bool(text_cfg.get("streaming", False)) and not text_path:
        return _build_streaming_token_source(config, tokenizer, text_cfg)

    if text_path:
        if text_cfg.get("validation_data_files") or text_cfg.get("validation_split"):
            raise ValueError(
                "text.validation_data_files and text.validation_split require a "
                "dataset source rather than text.text_path"
            )
        with open(text_path, encoding="utf-8") as handle:
            text = handle.read()
    else:
        dataset_name = str(text_cfg.get("dataset_name", "") or "")
        if not dataset_name and not text_cfg.get("data_files"):
            raise ValueError(
                "hf_text source requires text.text_path, text.dataset_name, "
                "or text.data_files"
            )
        split = str(text_cfg.get("split", "train"))
        text_field = str(text_cfg.get("text_field", "text"))
        dataset = _load_text_dataset(
            text_cfg,
            split=split,
            data_files=text_cfg.get("data_files"),
            streaming=False,
        )
        max_documents = text_cfg.get("max_documents")
        max_documents = None if max_documents is None else int(max_documents)
        docs = list(
            _iter_text_field(
                dataset,
                text_field=text_field,
                max_documents=max_documents,
            )
        )
        if not docs:
            raise ValueError(
                f"Dataset {dataset_name or 'json'!r} produced no non-empty "
                f"{text_field!r} text"
            )
        text = "\n\n".join(docs)

        validation_files = text_cfg.get("validation_data_files")
        validation_split = str(text_cfg.get("validation_split", "") or "")
        validation_text_cfg = _validation_text_config(text_cfg)
        validation_text_field = str(
            text_cfg.get("validation_text_field", text_field) or text_field
        )
        separate_validation = (
            validation_files not in (None, "", [])
            or bool(validation_split)
            or bool(text_cfg.get("validation_dataset_name"))
        )
        if separate_validation:
            validation_dataset = _load_text_dataset(
                validation_text_cfg,
                split=validation_split or split,
                data_files=(
                    validation_files
                    if validation_files not in (None, "", [])
                    else text_cfg.get("data_files")
                ),
                streaming=False,
            )
            validation_documents = max(1, int(text_cfg.get("validation_documents", 64)))
            validation_tokens = _tokenize_joined_documents(
                _iter_text_field(
                    validation_dataset,
                    text_field=validation_text_field,
                    max_documents=validation_documents,
                ),
                tokenizer,
                max_tokens=(
                    None
                    if text_cfg.get("validation_max_tokens") is None
                    else int(text_cfg["validation_max_tokens"])
                ),
            )
            train_tokens = _tokenize_joined_documents(
                docs,
                tokenizer,
                max_tokens=(
                    None
                    if text_cfg.get("max_tokens") is None
                    else int(text_cfg["max_tokens"])
                ),
            )
            min_tokens = max(4, int(train_cfg.sequence_length) + 1)
            if train_tokens.numel() < min_tokens:
                raise ValueError(
                    f"Need at least {min_tokens} training tokens for hf_text "
                    "distillation"
                )
            if validation_tokens.numel() < min_tokens:
                raise ValueError(
                    f"Need at least {min_tokens} validation tokens for hf_text "
                    "distillation"
                )
            train_tokens = _append_synthetic_copy_tokens(
                train_tokens, fixed_path_copy_sources, tokenizer
            )
            return train_tokens, validation_tokens

    tokenized = tokenizer(text, return_tensors="pt")
    input_ids = tokenized["input_ids"].squeeze(0).long()
    max_tokens = text_cfg.get("max_tokens")
    if max_tokens is not None:
        input_ids = input_ids[: int(max_tokens)]
    min_tokens = max(4, int(train_cfg.sequence_length) + 1)
    if input_ids.numel() < min_tokens:
        raise ValueError(f"Need at least {min_tokens} tokens for hf_text distillation")
    split = max(int(0.9 * input_ids.numel()), min_tokens)
    return (
        _append_synthetic_copy_tokens(
            input_ids[:split], fixed_path_copy_sources, tokenizer
        ),
        input_ids[max(0, split - train_cfg.sequence_length) :],
    )


def _sample_token_batch(
    tokens: torch.Tensor | StreamingTokenBatchSource | FrozenWindowBatchSource,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(tokens, (StreamingTokenBatchSource, FrozenWindowBatchSource)):
        if int(sequence_length) != tokens.sequence_length:
            raise ValueError(
                "Requested sequence_length does not match the streaming source"
            )
        return tokens.next_batch(int(batch_size), device=device)
    if tokens.ndim == 2:
        if int(tokens.shape[1]) != int(sequence_length):
            raise ValueError(
                "Rank-two token windows must match the requested sequence_length"
            )
        indices = torch.randint(0, int(tokens.shape[0]), (int(batch_size),))
        return tokens.index_select(0, indices).to(device=device)
    if tokens.ndim != 1:
        raise ValueError("In-memory token sources must be rank one or rank two")
    max_start = tokens.numel() - int(sequence_length)
    if max_start <= 0:
        raise ValueError("Token source is shorter than sequence_length")
    starts = torch.randint(0, max_start, (int(batch_size),))
    batch = torch.stack([tokens[start : start + sequence_length] for start in starts])
    return batch.to(device=device)


def read_transformer_token_source(
    config: Config,
) -> tuple[
    torch.Tensor | StreamingTokenBatchSource | FrozenWindowBatchSource,
    torch.Tensor,
]:
    """Public calibration-data loader shared by training and FMI profiling."""

    return _read_token_source(config)


def sample_transformer_token_batch(
    tokens: torch.Tensor | StreamingTokenBatchSource | FrozenWindowBatchSource,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample one packed token batch using the training path's exact policy."""

    return _sample_token_batch(
        tokens,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
    )


logger = logging.getLogger(__name__)


def load_frozen_replacement_from_checkpoint(
    mlp: nn.Module,
    config: Config,
    checkpoint: str,
    *,
    device: torch.device,
    layer_index: int,
    verify_semantic_identity: bool = False,
    replacement_dtype_override: torch.dtype | None = None,
) -> nn.Module:
    """Build a replacement at the CHECKPOINT's shape and load it frozen.

    Mixed-density chains: the replacement must be built at the checkpoint's
    shape, not the current config's density (audit of nonuniform-chain
    failure, 2026-08-16). Shared by staged composition pre-patching and the
    paired deletion-vs-replacement control evaluator.
    """
    from dendritic_modeling.training._transformer_replacement.checkpoints import (
        _load_replacement_state_dict,
    )

    payload = torch.load(checkpoint, map_location=device)
    if isinstance(payload, Mapping) and payload.get("parameter_tied_replacement"):
        raise NotImplementedError(
            "single-site pre-patching cannot reconstruct a shared replacement "
            "alias graph; load the complete config plus the v2 replacement "
            "export manifest instead"
        )
    if isinstance(payload, Mapping) and payload.get("collapsed_replacement_span"):
        raise NotImplementedError(
            "single-site pre-patching cannot reconstruct the zero branches of a "
            "collapsed replacement span; load the complete config plus the v3 "
            "replacement export manifest instead"
        )
    state_dict = payload.get("state_dict", payload)
    overrides = {}
    gate_key = "gate_core.branch_layers.0.branch_excitation.pre_w"
    out_key = "output_projection.pre_w"
    if gate_key in state_dict:
        overrides["synapses_per_branch"] = int(state_dict[gate_key].shape[1])
    if out_key in state_dict and state_dict[out_key].dim() == 2:
        n_units = int(state_dict[gate_key].shape[0]) if gate_key in state_dict else None
        if n_units is None or state_dict[out_key].shape[1] != n_units:
            overrides["output_topk"] = int(state_dict[out_key].shape[1])
    replacement = _make_replacement_for_teacher_mlp(
        mlp,
        config,
        device=device,
        dtype=next(mlp.parameters()).dtype,
        kwargs_overrides=overrides,
        layer_index=layer_index,
    )
    if replacement_dtype_override is not None:
        if replacement_dtype_override not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }:
            raise ValueError(
                "replacement_dtype_override must be a floating-point dtype"
            )
        replacement = replacement.to(
            device=device,
            dtype=replacement_dtype_override,
        )
    load_receipt = _load_replacement_state_dict(
        replacement,
        state_dict,
        (
            payload.get("sparse_topology_manifest")
            if isinstance(payload, dict)
            else None
        ),
        verify_semantic_identity=verify_semantic_identity,
    )
    if load_receipt is not None:
        replacement._checkpoint_load_receipt = load_receipt
    replacement.eval()
    for param in replacement.parameters():
        param.requires_grad_(False)
    return replacement


def _apply_pre_patched_replacements(
    teacher: nn.Module,
    layers,
    config: Config,
    *,
    layer_indices,
    device: torch.device,
) -> list[ReplacementRecord]:
    """Patch earlier-stage trained replacements into the model before capture.

    Staged (student-context) composition: with stages < k already patched in,
    this run's capture sees the composed student's activations at layer k
    while the still-dense layer-k module provides the teacher-function target
    on those same inputs. Patched modules are frozen.
    """
    entries = list(getattr(config.model.transformer_replacement, "pre_patched", []))
    if not entries:
        return []
    model_cfg = config.model.transformer_replacement
    training_set = {int(i) for i in layer_indices}
    records: list[ReplacementRecord] = []
    for entry in entries:
        layer_index = int(entry["layer_index"])
        checkpoint = str(entry["checkpoint"])
        if layer_index in training_set:
            raise ValueError(
                f"pre_patched layer {layer_index} is also listed for training"
            )
        layer = layers[layer_index]
        mlp = _get_attr_path(layer, model_cfg.target_module)
        replacement = load_frozen_replacement_from_checkpoint(
            mlp,
            config,
            checkpoint,
            device=device,
            layer_index=layer_index,
        )
        require_runtime_tensor_contract(
            replacement,
            boundary=f"pre-patched transformer layer {layer_index}",
        )
        path = model_cfg.target_module.split(".")
        parent = layer if len(path) == 1 else _get_attr_path(layer, ".".join(path[:-1]))
        setattr(parent, path[-1], replacement)
        records.append(
            ReplacementRecord(
                layer_index=layer_index,
                mlp_attr=str(model_cfg.target_module),
                original_mlp=mlp,
                replacement=replacement,
            )
        )
        # Record the swap so trajectory-anchored distillation can restore the
        # clean teacher for its second forward. A plain list attribute is
        # deliberately not registered as a submodule.
        swaps = getattr(teacher, "_dendritic_prepatch_swaps", None)
        if swaps is None:
            swaps = []
            teacher._dendritic_prepatch_swaps = swaps
        swaps.append((parent, path[-1], mlp, replacement))
        logger.info("Pre-patched layer %d from %s (frozen)", layer_index, checkpoint)
    return records


def _apply_pre_patched_collapsed_spans(
    student: nn.Module,
    config: Config,
    *,
    layer_indices: Sequence[int],
    device: torch.device,
) -> list[ReplacementRecord]:
    """Load frozen v3 span exports before fitting new disjoint spans.

    Every source manifest is hash-pinned and its artifacts are verified before
    the model is mutated. The embedded compiler plans reconstruct the exact
    prior cells; learned weights are then loaded through the ordinary v3
    checkpoint verifier. This is staged student-context composition, not a
    warm start of the newly trained span.
    """

    entries = list(
        getattr(
            config.model.transformer_replacement,
            "pre_patched_collapsed_spans",
            [],
        )
        or []
    )
    if not entries:
        return []
    from dendritic_modeling.training._transformer_replacement.checkpoints import (
        _load_replacement_record_checkpoints,
        _resolve_replacement_artifact_paths,
    )

    model_cfg = config.model.transformer_replacement
    current_layers = {int(layer) for layer in layer_indices}
    single_site_prepatch_layers = {
        int(entry["layer_index"]) for entry in list(model_cfg.pre_patched or [])
    }
    specs: list[dict[str, Any]] = []
    all_prior_layers: set[int] = set()
    for entry_index, raw_entry in enumerate(entries):
        entry = _to_plain_mapping(raw_entry)
        checkpoint_dir = os.path.realpath(str(entry.get("checkpoint_dir", "") or ""))
        if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"pre_patched_collapsed_spans[{entry_index}] checkpoint_dir is "
                f"not a directory: {checkpoint_dir!r}"
            )
        if entry.get("freeze", True) is not True:
            raise ValueError("pre-patched collapsed spans must remain frozen")
        expected_manifest_sha256 = str(entry.get("manifest_sha256", "") or "").lower()
        if len(expected_manifest_sha256) != 64:
            raise ValueError(
                "pre-patched collapsed spans require a 64-character manifest_sha256 pin"
            )
        manifest_path = os.path.join(checkpoint_dir, "replacement_export_manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(manifest_path)
        observed_manifest_sha256 = sha256_file(manifest_path)
        if observed_manifest_sha256 != expected_manifest_sha256:
            raise RuntimeError(
                "pre-patched collapsed-span manifest SHA-256 mismatch: "
                f"expected {expected_manifest_sha256}, observed "
                f"{observed_manifest_sha256}"
            )
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema") != "dendritic_replacement_artifact_manifest/v3":
            raise RuntimeError(
                "pre-patched collapsed spans require a v3 replacement export"
            )
        raw_spans = entry.get("collapsed_replacement_spans")
        if not raw_spans:
            raise ValueError(
                "each pre-patched collapsed-span entry must explicitly declare "
                "collapsed_replacement_spans"
            )
        flat_layers = [layer for span in raw_spans for layer in span]
        spans = validate_collapsed_replacement_spans(raw_spans, flat_layers)
        manifest_spans = [
            [int(layer) for layer in span]
            for span in manifest.get("collapsed_replacement_spans", [])
        ]
        if [list(span) for span in spans] != manifest_spans:
            raise RuntimeError(
                "pre-patched collapsed-span declaration does not match its "
                "hash-pinned v3 manifest"
            )
        prior_layers = {layer for span in spans for layer in span}
        overlap = sorted(
            prior_layers
            & (current_layers | single_site_prepatch_layers | all_prior_layers)
        )
        if overlap:
            raise ValueError(
                "pre-patched collapsed spans must be disjoint from current and "
                f"previously patched layers; overlap={overlap}"
            )
        all_prior_layers.update(prior_layers)
        exits = [int(span[-1]) for span in spans]
        artifact_paths = _resolve_replacement_artifact_paths(checkpoint_dir, exits)
        plans: dict[str, dict[str, Any]] = {}
        norm_attrs: set[str] = set()
        for span in spans:
            exit_layer = int(span[-1])
            payload = torch.load(
                artifact_paths[exit_layer],
                map_location="cpu",
                weights_only=False,
            )
            plan = payload.get("compiled_replacement_plan")
            boundary = payload.get("collapsed_replacement_span", {})
            norm_attr = boundary.get("post_mlp_norm_attr", "")
            if (
                not isinstance(norm_attr, str)
                or (
                    list(boundary.get("post_mlp_norm_removed_layers", []))
                    != (list(span) if norm_attr else [])
                )
                or (
                    norm_attr
                    and boundary.get("cell_output_boundary")
                    != "post_mlp_norm_residual_branch"
                )
            ):
                raise RuntimeError(
                    "pre-patched collapsed-span post-MLP norm boundary is invalid"
                )
            norm_attrs.add(norm_attr)
            if not isinstance(plan, Mapping) or not plan:
                raise RuntimeError(
                    "pre-patched collapsed-span artifact lacks its compiler plan"
                )
            plans[f"{int(span[0])}:{exit_layer}"] = dict(plan)
        if len(norm_attrs) != 1:
            raise RuntimeError(
                "pre-patched collapsed spans require one consistent post-MLP norm path"
            )
        specs.append(
            {
                "checkpoint_dir": checkpoint_dir,
                "manifest_sha256": expected_manifest_sha256,
                "spans": spans,
                "layers": flat_layers,
                "plans": plans,
                "post_mlp_norm_attr": next(iter(norm_attrs)),
            }
        )

    records: list[ReplacementRecord] = []
    try:
        for spec in specs:
            first_plan = next(iter(spec["plans"].values()))
            patch_config = {
                "enabled": True,
                "model_family": str(model_cfg.model_family),
                "layers": list(spec["layers"]),
                "collapsed_replacement_spans": [list(span) for span in spec["spans"]],
                "collapsed_span_post_mlp_norm_attr": spec["post_mlp_norm_attr"],
                "target_module": str(model_cfg.target_module),
                "layers_attr": model_cfg.layers_attr,
                "preserve_device_dtype": bool(model_cfg.preserve_device_dtype),
                "replacement_kwargs": dict(first_plan["replacement_kwargs"]),
                "compiled_plans_by_collapsed_span": dict(spec["plans"]),
                "selection": {},
                "pre_patched": [],
            }
            current = apply_collapsed_population_spans(
                student,
                patch_config,
                config.model.core,
            )
            records.extend(current)
            _load_replacement_record_checkpoints(
                current,
                str(spec["checkpoint_dir"]),
                device=device,
            )
            for record in current:
                for parameter in record.replacement.parameters():
                    parameter.requires_grad_(False)
                cell = getattr(record.replacement, "span_cell", record.replacement)
                selection_manifest = dict(getattr(cell, "selection_manifest", {}) or {})
                selection_manifest["staged_student_context_prepatch"] = {
                    "schema": "dendritic_collapsed_span_prepatch/v1",
                    "checkpoint_dir": str(spec["checkpoint_dir"]),
                    "manifest_sha256": str(spec["manifest_sha256"]),
                    "frozen": True,
                }
                cell.selection_manifest = selection_manifest
    except Exception:
        if records:
            restore_transformer_mlp_layers(
                student,
                list(reversed(records)),
                layers_attr=model_cfg.layers_attr,
            )
        raise
    student._dendritic_collapsed_prepatch_records = list(records)
    return records


def _resolve_replacement_dtype(config, model_dtype):
    """FP32 master weights by default; 'model' keeps the surrounding dtype."""
    train_cfg = getattr(
        getattr(config, "training", None), "transformer_replacement", None
    )
    name = str(getattr(train_cfg, "replacement_dtype", "float32")).lower()
    if name in ("model", "auto", "same"):
        return model_dtype
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _make_replacement_for_teacher_mlp(
    mlp: nn.Module,
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    kwargs_overrides: dict | None = None,
    layer_index: int | None = None,
) -> nn.Module:
    """Build the configured FFN replacement for one teacher mlp module."""
    replacement = build_transformer_replacement_for_mlp(
        mlp,
        config.model.transformer_replacement,
        config.model.core,
        layer_index=layer_index,
        replacement_kwargs=kwargs_overrides,
    )
    replacement_dtype = _resolve_replacement_dtype(config, dtype)
    return replacement.to(device=device, dtype=replacement_dtype)


def _load_hf_teacher_and_units(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, list[DistillationUnit]]:
    model_cfg = config.model.transformer_replacement
    source_declaration = _to_plain_mapping(getattr(model_cfg, "model_source", {}) or {})
    source_session = (
        TransformerModelSourceSession(source_declaration)
        if source_declaration
        else None
    )
    teacher = _load_hf_causal_lm_for_joint_training(
        config,
        device=device,
        dtype=dtype,
        initialization="pretrained",
        model_source_session=source_session,
    )
    if source_session is not None:
        teacher._dendritic_model_source_receipt = source_session.finalize()
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    layer_indices = resolve_configured_replacement_layers(config)
    if not layer_indices:
        raise ValueError("At least one transformer replacement layer is required")
    layers = resolve_transformer_layers(teacher, layers_attr=model_cfg.layers_attr)

    _apply_pre_patched_replacements(
        teacher,
        layers,
        config,
        layer_indices=layer_indices,
        device=device,
    )

    units: list[DistillationUnit] = []
    for layer_index in layer_indices:
        mlp = _get_attr_path(layers[int(layer_index)], model_cfg.target_module)
        if not isinstance(mlp, nn.Module):
            raise TypeError(
                f"layer {layer_index} target {model_cfg.target_module!r} "
                "is not an nn.Module"
            )
        kind = _replacement_kind(config)
        if kind in {"dense_mlp_control", "dense_control", "dense_mlp"}:
            replacement = DenseMLPControl.from_mlp(mlp).to(
                device=device,
                dtype=_resolve_replacement_dtype(config, dtype),
            )
        elif kind == "dense_swiglu_surrogate":
            replacement = build_transformer_replacement_for_mlp(
                mlp,
                model_cfg,
                config.model.core,
                layer_index=int(layer_index),
            ).to(device=device, dtype=_resolve_replacement_dtype(config, dtype))
        elif kind in {
            "ei_stack",
            "layerwise_ei_stack",
            "einet_stack",
        }:
            param = next(mlp.parameters(), None)
            if param is None or param.ndim < 2:
                raise ValueError(
                    "Could not infer hidden size for EI-stack replacement from "
                    f"layer {layer_index}"
                )
            hidden_size = int(param.shape[1])
            replacement = _make_single_layer_ei_stack_replacement(
                hidden_size=hidden_size,
                config=config,
                dtype=dtype,
                device=device,
            )
        else:
            replacement = build_transformer_replacement_for_mlp(
                mlp,
                config.model.transformer_replacement,
                config.model.core,
                layer_index=int(layer_index),
            ).to(
                device=device,
                dtype=_resolve_replacement_dtype(config, dtype),
            )
        units.append(
            DistillationUnit(
                layer_index=int(layer_index),
                teacher_mlp=mlp,
                replacement=replacement,
            )
        )
    return teacher, _apply_parameter_tied_population_groups_to_units(config, units)


def _load_hf_causal_lm_for_joint_training(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
    initialization: str = "pretrained",
    model_source_session: TransformerModelSourceSession | None = None,
) -> nn.Module:
    """Load one causal LM through HF or a provenance-bound external source."""
    model_cfg = config.model.transformer_replacement
    model_kwargs = _to_plain_mapping(model_cfg.model_kwargs)
    if "device_map" in model_kwargs:
        raise ValueError(
            "joint_lm_distillation expects a single-device model; remove device_map "
            "from model.transformer_replacement.model_kwargs"
        )
    normalized = str(initialization).lower()
    source_declaration = _to_plain_mapping(getattr(model_cfg, "model_source", {}) or {})
    if source_declaration:
        if str(model_cfg.model_family).lower() != "causal_lm" or str(
            model_cfg.model_loader
        ).lower() not in {"auto", "causal_lm"}:
            raise ValueError(
                "external transformer model sources currently require causal_lm"
            )
        if model_kwargs:
            raise ValueError(
                "model_kwargs must be empty for an external transformer model_source; "
                "its loader semantics are fully declared by model_source"
            )
        expected_source = normalize_transformer_model_source(source_declaration)
        owns_session = model_source_session is None
        session = model_source_session or TransformerModelSourceSession(
            source_declaration
        )
        if session.declaration != expected_source:
            raise ValueError("external model-source session declaration differs")
        model = session.load(
            device=device,
            dtype=dtype,
            initialization=normalized,
        )
        if owns_session:
            model._dendritic_model_source_receipt = session.finalize()
    else:
        if model_source_session is not None:
            raise ValueError(
                "a model-source session was supplied without model_source config"
            )
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
        except ImportError as exc:
            raise ImportError(
                "transformers is required for hf_text distillation"
            ) from exc
        if normalized in {"pretrained", "checkpoint"}:
            model = AutoModelForCausalLM.from_pretrained(
                model_cfg.model_name,
                **model_kwargs,
            )
        elif normalized in {"from_config", "scratch", "random"}:
            config_kwargs = dict(model_kwargs)
            config_kwargs.pop("torch_dtype", None)
            hf_config = AutoConfig.from_pretrained(
                model_cfg.model_name,
                **config_kwargs,
            )
            model = AutoModelForCausalLM.from_config(
                hf_config,
                trust_remote_code=bool(config_kwargs.get("trust_remote_code", False)),
            )
        else:
            raise ValueError(
                "student_initialization must be 'pretrained' or 'from_config'"
            )
        model = model.to(device=device, dtype=dtype)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model
