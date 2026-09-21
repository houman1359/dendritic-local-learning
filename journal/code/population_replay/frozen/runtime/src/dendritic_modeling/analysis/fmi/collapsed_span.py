"""Auditable FMI capture for true collapsed transformer FFN spans.

A collapsed span is not a teacher module boundary.  Its replacement cell sees
the hidden state entering the span-exit FFN, while its supervised correction is

``dense teacher span-exit hidden - zero-cell collapsed-student span-exit hidden``.

This module measures that exact pair.  It deliberately does not reuse or
relabel a single-layer FFN fingerprint.  Dense-teacher exits are sampled in a
first pass and retained on CPU, so a large-model caller may release the teacher
before loading the collapsed student.  Every sampled row retains its source
window group and content address for group-disjoint validation.

The observed correction is generally not an exact local callable of the cell
input.  Output-spectrum statistics can be measured directly, but gradient,
support, sign, interaction, and mechanism estimators need a differentiable
function.  :func:`profile_collapsed_span_boundary` therefore fits a declared
random-feature kernel-ridge surrogate on source-disjoint groups and profiles it
only on held-out groups.  The surrogate status is recorded, and downstream
plan compilation must fail closed when its predictive gate is not met.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.analysis.fmi.profiler import (
    FMIProfilerConfig,
    profile_teacher_component,
)
from dendritic_modeling.analysis.fmi.rank_validation import validate_rank_estimate
from dendritic_modeling.networks.architectures.transformer import (
    ReplacementRecord,
    ZeroFFNResidualBranch,
    resolve_transformer_layers,
)
from dendritic_modeling.networks.architectures.transformer.patching import (
    _validate_collapsed_span_norm_placement,
)
from dendritic_modeling.networks.architectures.transformer.utils import (
    _get_attr_path,
    _set_attr_path,
)


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Hash a tensor's exact dtype, shape, and contiguous bytes."""

    canonical = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    if canonical.numel():
        digest.update(canonical.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_hidden(output: Any, *, name: str) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"{name} does not expose a hidden-state tensor")


def _flatten_hidden_rows(
    hidden: torch.Tensor,
    *,
    batch_size: int,
    name: str,
) -> tuple[torch.Tensor, int]:
    if hidden.ndim < 2 or int(hidden.shape[0]) != int(batch_size):
        raise ValueError(
            f"{name} must have batch-leading shape [B,...,D] with B={batch_size}"
        )
    positions_per_group = math.prod(int(size) for size in hidden.shape[1:-1])
    if positions_per_group < 1:
        positions_per_group = 1
    rows = hidden.reshape(-1, hidden.shape[-1])
    if int(rows.shape[0]) != int(batch_size) * positions_per_group:
        raise RuntimeError(f"{name} row flattening is inconsistent")
    return rows, positions_per_group


def _hash_model_kwargs_row(model_kwargs: Mapping[str, Any], row: int) -> str:
    digest = hashlib.sha256()
    for key in sorted(model_kwargs):
        value = model_kwargs[key]
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        if torch.is_tensor(value):
            if value.ndim < 1 or row >= int(value.shape[0]):
                raise ValueError(
                    f"batched model input {key!r} does not contain source row {row}"
                )
            digest.update(tensor_content_sha256(value[row]).encode("ascii"))
        else:
            digest.update(
                json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
            )
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class CollapsedSpanCalibrationBatch:
    """One replayable model input batch with source-window identities."""

    model_kwargs: Mapping[str, Any]
    source_group_ids: torch.Tensor
    source_group_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = self.source_group_ids
        if groups.ndim != 1 or groups.dtype != torch.int64:
            raise TypeError("source_group_ids must be a one-dimensional int64 tensor")
        if int(groups.numel()) < 1 or bool((groups < 0).any()):
            raise ValueError("source_group_ids must be non-empty and nonnegative")
        if int(torch.unique(groups).numel()) != int(groups.numel()):
            raise ValueError("source group ids must be unique within each batch")
        if len(self.source_group_sha256) != int(groups.numel()):
            raise ValueError("one source content hash is required per source group")
        for value in self.source_group_sha256:
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError("source group hashes must be lowercase SHA-256 values")
        for key, value in self.model_kwargs.items():
            if torch.is_tensor(value) and value.ndim >= 1:
                if int(value.shape[0]) != int(groups.numel()):
                    raise ValueError(
                        f"batched model input {key!r} does not align with groups"
                    )

    @property
    def batch_size(self) -> int:
        return int(self.source_group_ids.numel())

    def content_manifest(self) -> dict[str, Any]:
        return {
            "source_group_ids": [int(value) for value in self.source_group_ids],
            "source_group_sha256": list(self.source_group_sha256),
            "batch_sha256": _canonical_json_sha256(
                {
                    "ids": [int(value) for value in self.source_group_ids],
                    "sha256": list(self.source_group_sha256),
                }
            ),
        }


def content_addressed_calibration_batch(
    model_kwargs: Mapping[str, Any],
    *,
    source_group_ids: torch.Tensor,
) -> CollapsedSpanCalibrationBatch:
    """Construct a batch whose individual source examples are content-addressed."""

    groups = source_group_ids.detach().to(device="cpu", dtype=torch.int64)
    hashes = tuple(
        _hash_model_kwargs_row(model_kwargs, row) for row in range(int(groups.numel()))
    )
    return CollapsedSpanCalibrationBatch(
        model_kwargs=dict(model_kwargs),
        source_group_ids=groups,
        source_group_sha256=hashes,
    )


@dataclass(frozen=True)
class CollapsedSpanSpec:
    """Placement identity for one true collapsed span."""

    span_layers: tuple[int, ...]
    mlp_attr: str = "mlp"

    def __post_init__(self) -> None:
        layers = tuple(int(layer) for layer in self.span_layers)
        if len(layers) < 2:
            raise ValueError("a collapsed span requires at least two layers")
        if layers != tuple(range(layers[0], layers[-1] + 1)):
            raise ValueError("collapsed span layers must be sorted and contiguous")
        if layers[0] < 0:
            raise ValueError("collapsed span layers must be nonnegative")
        object.__setattr__(self, "span_layers", layers)

    @property
    def key(self) -> str:
        return f"{self.span_layers[0]}:{self.span_layers[-1]}"

    @property
    def exit_layer(self) -> int:
        return int(self.span_layers[-1])


def collapsed_span_specs_from_records(
    records: Sequence[ReplacementRecord],
) -> tuple[CollapsedSpanSpec, ...]:
    specs = []
    for record in records:
        layers = tuple(int(layer) for layer in record.collapsed_span_layers)
        if not layers:
            continue
        spec = CollapsedSpanSpec(layers, str(record.mlp_attr))
        if spec.exit_layer != int(record.layer_index):
            raise ValueError("collapsed record layer index must equal its span exit")
        specs.append(spec)
    if not specs:
        raise ValueError("no collapsed replacement records were supplied")
    if len({spec.key for spec in specs}) != len(specs):
        raise ValueError("collapsed span record keys must be unique")
    return tuple(sorted(specs, key=lambda spec: spec.exit_layer))


@dataclass(frozen=True)
class DenseTeacherSpanSample:
    """A bounded teacher-exit sample with exact replay row references."""

    spec: CollapsedSpanSpec
    teacher_exits: torch.Tensor
    batch_indices: torch.Tensor
    row_indices: torch.Tensor
    row_group_ids: torch.Tensor
    rows_seen: int

    def __post_init__(self) -> None:
        count = int(self.teacher_exits.shape[0])
        if self.teacher_exits.ndim != 2 or count < 4:
            raise ValueError("teacher span sample must contain at least four rows")
        for name, value in (
            ("batch_indices", self.batch_indices),
            ("row_indices", self.row_indices),
            ("row_group_ids", self.row_group_ids),
        ):
            if value.dtype != torch.int64 or tuple(value.shape) != (count,):
                raise TypeError(f"{name} must be int64 with one value per row")
        if int(self.rows_seen) < count:
            raise ValueError("rows_seen cannot be smaller than sampled rows")


@dataclass(frozen=True)
class DenseTeacherSpanExitCache:
    """Content-bound first-pass dense-teacher exit cache."""

    samples: Mapping[str, DenseTeacherSpanSample]
    calibration_batches: tuple[dict[str, Any], ...]
    sample_seed: int
    max_examples: int

    @property
    def calibration_sha256(self) -> str:
        return _canonical_json_sha256(list(self.calibration_batches))


class _TeacherReservoir:
    def __init__(self, *, max_examples: int, seed: int) -> None:
        self.max_examples = int(max_examples)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.keys = torch.empty(0)
        self.outputs = torch.empty((0, 0))
        self.batch_indices = torch.empty(0, dtype=torch.int64)
        self.row_indices = torch.empty(0, dtype=torch.int64)
        self.group_ids = torch.empty(0, dtype=torch.int64)
        self.rows_seen = 0

    def update(
        self,
        outputs: torch.Tensor,
        *,
        batch_index: int,
        group_ids: torch.Tensor,
    ) -> None:
        rows = outputs.detach().to(device="cpu", dtype=torch.float32).contiguous()
        count = int(rows.shape[0])
        if rows.ndim != 2 or tuple(group_ids.shape) != (count,):
            raise ValueError("teacher rows and source groups do not align")
        self.rows_seen += count
        keys = torch.rand(count, generator=self.generator)
        batch_indices = torch.full((count,), int(batch_index), dtype=torch.int64)
        row_indices = torch.arange(count, dtype=torch.int64)
        groups = group_ids.detach().to(device="cpu", dtype=torch.int64)
        if self.keys.numel():
            keys = torch.cat((self.keys, keys))
            rows = torch.cat((self.outputs, rows))
            batch_indices = torch.cat((self.batch_indices, batch_indices))
            row_indices = torch.cat((self.row_indices, row_indices))
            groups = torch.cat((self.group_ids, groups))
        keep = min(self.max_examples, int(keys.numel()))
        selected = torch.topk(keys, keep, largest=False).indices
        self.keys = keys[selected]
        self.outputs = rows[selected]
        self.batch_indices = batch_indices[selected]
        self.row_indices = row_indices[selected]
        self.group_ids = groups[selected]

    def finish(self, spec: CollapsedSpanSpec) -> DenseTeacherSpanSample:
        # Canonical traversal order makes replay and hashes independent of the
        # internal top-k ordering returned by torch.
        order = sorted(
            range(int(self.keys.numel())),
            key=lambda index: (
                int(self.batch_indices[index]),
                int(self.row_indices[index]),
            ),
        )
        selected = torch.tensor(order, dtype=torch.int64)
        return DenseTeacherSpanSample(
            spec=spec,
            teacher_exits=self.outputs[selected],
            batch_indices=self.batch_indices[selected],
            row_indices=self.row_indices[selected],
            row_group_ids=self.group_ids[selected],
            rows_seen=self.rows_seen,
        )


def _expanded_group_ids(
    batch: CollapsedSpanCalibrationBatch,
    *,
    positions_per_group: int,
) -> torch.Tensor:
    return batch.source_group_ids.repeat_interleave(int(positions_per_group))


def _validate_calibration_batches(
    batches: Sequence[CollapsedSpanCalibrationBatch],
) -> tuple[dict[str, Any], ...]:
    if not batches:
        raise ValueError("at least one calibration batch is required")
    seen: set[int] = set()
    manifests = []
    for batch in batches:
        current = {int(value) for value in batch.source_group_ids}
        overlap = seen & current
        if overlap:
            raise ValueError(
                f"source group ids must be globally unique; repeated {sorted(overlap)}"
            )
        seen.update(current)
        manifests.append(batch.content_manifest())
    return tuple(manifests)


def _default_forward(model: nn.Module, model_kwargs: Mapping[str, Any]) -> Any:
    return model(**dict(model_kwargs))


def capture_dense_teacher_span_exits(
    teacher: nn.Module,
    specs: Sequence[CollapsedSpanSpec],
    batches: Sequence[CollapsedSpanCalibrationBatch],
    *,
    layers_attr: str | None = None,
    max_examples: int = 512,
    seed: int = 0,
    forward_fn: Callable[[nn.Module, Mapping[str, Any]], Any] | None = None,
) -> DenseTeacherSpanExitCache:
    """First pass: uniformly retain dense-teacher exits and replay row ids."""

    specs = tuple(specs)
    if not specs or len({spec.key for spec in specs}) != len(specs):
        raise ValueError("span specifications must be non-empty and unique")
    if int(max_examples) < 4:
        raise ValueError("max_examples must be at least four")
    manifests = _validate_calibration_batches(batches)
    layers = resolve_transformer_layers(teacher, layers_attr=layers_attr)
    runner = forward_fn or _default_forward
    reservoirs = {
        spec.key: _TeacherReservoir(
            max_examples=int(max_examples),
            seed=(
                int(seed)
                + int.from_bytes(hashlib.sha256(spec.key.encode()).digest()[:4], "big")
            ),
        )
        for spec in specs
    }
    captured: dict[str, list[torch.Tensor]] = {spec.key: [] for spec in specs}
    handles = []
    training_states = [(module, bool(module.training)) for module in teacher.modules()]
    teacher.eval()
    try:
        for spec in specs:
            layer = layers[spec.exit_layer]

            def capture(_module, _args, output, *, key=spec.key):
                captured[key].append(
                    _extract_hidden(output, name=f"dense teacher exit {key}").detach()
                )

            handles.append(layer.register_forward_hook(capture))
        with torch.no_grad():
            for batch_index, batch in enumerate(batches):
                before = {key: len(values) for key, values in captured.items()}
                runner(teacher, batch.model_kwargs)
                for spec in specs:
                    values = captured[spec.key]
                    if len(values) != before[spec.key] + 1:
                        raise RuntimeError(
                            f"dense teacher did not traverse span exit {spec.key} once"
                        )
                    rows, positions = _flatten_hidden_rows(
                        values.pop(),
                        batch_size=batch.batch_size,
                        name=f"dense teacher exit {spec.key}",
                    )
                    reservoirs[spec.key].update(
                        rows,
                        batch_index=batch_index,
                        group_ids=_expanded_group_ids(
                            batch, positions_per_group=positions
                        ),
                    )
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in training_states:
            module.training = was_training
    samples = {spec.key: reservoirs[spec.key].finish(spec) for spec in specs}
    return DenseTeacherSpanExitCache(
        samples=samples,
        calibration_batches=manifests,
        sample_seed=int(seed),
        max_examples=int(max_examples),
    )


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor_content_sha256(value).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _plan_sha256(module: nn.Module) -> str | None:
    plan = getattr(module, "compiled_replacement_plan", None)
    if not isinstance(plan, Mapping) or not plan:
        return None
    return _canonical_json_sha256(dict(plan))


@contextmanager
def _collapsed_student_context(
    student_layers: Sequence[nn.Module],
    records: Sequence[ReplacementRecord],
    current: ReplacementRecord,
    *,
    context_policy: str,
):
    normalized = str(context_policy).strip().lower()
    if normalized not in {"configured", "isolated_dense", "all_zero"}:
        raise ValueError(
            "context_policy must be configured, isolated_dense, or all_zero"
        )
    mutations: list[tuple[nn.Module, str, nn.Module]] = []
    current_zero: ZeroFFNResidualBranch | None = None
    try:
        for record in records:
            _validate_collapsed_span_norm_placement(student_layers, record)
            span = tuple(int(layer) for layer in record.collapsed_span_layers)
            installed = tuple(record.collapsed_span_installed_modules)
            originals = tuple(record.collapsed_span_original_mlps)
            if not span or len(installed) != len(span) or len(originals) != len(span):
                raise RuntimeError("collapsed replacement record is incomplete")
            for offset, layer_index in enumerate(span):
                layer = student_layers[layer_index]
                existing = _get_attr_path(layer, record.mlp_attr)
                if existing is not installed[offset]:
                    raise RuntimeError(
                        f"student layer {layer_index} no longer holds its configured "
                        "collapsed-span module"
                    )
                desired: nn.Module = existing
                if record is current:
                    if offset < len(span) - 1:
                        if not isinstance(existing, ZeroFFNResidualBranch):
                            raise RuntimeError(
                                "earlier FFN branches in the current span are not zero"
                            )
                    else:
                        current_zero = ZeroFFNResidualBranch(
                            layer_index=layer_index,
                            span_index=int(record.collapsed_span_index),
                            span_layers=span,
                        )
                        desired = current_zero
                elif normalized == "isolated_dense":
                    desired = originals[offset]
                elif normalized == "all_zero" and offset == len(span) - 1:
                    desired = ZeroFFNResidualBranch(
                        layer_index=layer_index,
                        span_index=int(record.collapsed_span_index),
                        span_layers=span,
                    )
                if desired is not existing:
                    mutations.append((layer, str(record.mlp_attr), existing))
                    _set_attr_path(layer, record.mlp_attr, desired)
                if (
                    record is not current
                    and normalized == "isolated_dense"
                    and record.collapsed_span_post_mlp_norm_attr
                ):
                    norm_attr = record.collapsed_span_post_mlp_norm_attr
                    existing_norm = _get_attr_path(layer, norm_attr)
                    mutations.append((layer, norm_attr, existing_norm))
                    _set_attr_path(
                        layer,
                        norm_attr,
                        record.collapsed_span_original_post_mlp_norms[offset],
                    )
        if current_zero is None:
            raise RuntimeError("current collapsed span exit was not replaced by zero")
        yield current_zero
    finally:
        for layer, mlp_attr, installed in reversed(mutations):
            _set_attr_path(layer, mlp_attr, installed)


@dataclass(frozen=True)
class CapturedCollapsedSpanBoundary:
    """Measured cell input and span correction on aligned sampled rows."""

    spec: CollapsedSpanSpec
    inputs: torch.Tensor
    targets: torch.Tensor
    dense_teacher_exits: torch.Tensor
    zero_student_exits: torch.Tensor
    row_group_ids: torch.Tensor
    source_group_sha256: Mapping[int, str]
    rows_seen: int
    context: Mapping[str, Any]
    calibration_sha256: str

    def __post_init__(self) -> None:
        shape = tuple(self.inputs.shape)
        if self.inputs.ndim != 2 or int(self.inputs.shape[0]) < 4:
            raise ValueError("captured span boundary needs at least four rows")
        if tuple(self.targets.shape) != shape:
            raise ValueError("collapsed cell inputs and correction targets must align")
        if tuple(self.dense_teacher_exits.shape) != shape:
            raise ValueError("dense teacher exits do not align with captured inputs")
        if tuple(self.zero_student_exits.shape) != shape:
            raise ValueError("zero-student exits do not align with captured inputs")
        if self.row_group_ids.dtype != torch.int64 or tuple(
            self.row_group_ids.shape
        ) != (shape[0],):
            raise TypeError("row_group_ids must be int64 with one value per row")
        observed_groups = {int(value) for value in torch.unique(self.row_group_ids)}
        supplied_groups = {int(key) for key in self.source_group_sha256}
        if supplied_groups != observed_groups:
            raise ValueError(
                "source_group_sha256 must cover sampled row groups exactly"
            )
        for value in self.source_group_sha256.values():
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError("source group hashes must be lowercase SHA-256 values")
        if not torch.equal(
            self.targets,
            self.dense_teacher_exits - self.zero_student_exits,
        ):
            raise ValueError("captured targets do not equal teacher minus zero student")

    @property
    def content_sha256(self) -> str:
        return _canonical_json_sha256(
            {
                "schema": "dendritic_collapsed_span_boundary_content/v1",
                "span_key": self.spec.key,
                "inputs_sha256": tensor_content_sha256(self.inputs),
                "targets_sha256": tensor_content_sha256(self.targets),
                "dense_teacher_exits_sha256": tensor_content_sha256(
                    self.dense_teacher_exits
                ),
                "zero_student_exits_sha256": tensor_content_sha256(
                    self.zero_student_exits
                ),
                "row_group_ids_sha256": tensor_content_sha256(self.row_group_ids),
                "source_group_sha256": {
                    str(key): value
                    for key, value in sorted(self.source_group_sha256.items())
                },
                "calibration_sha256": self.calibration_sha256,
                "context": dict(self.context),
            }
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "dendritic_collapsed_span_fmi_boundary/v1",
            "span_key": self.spec.key,
            "span_layers": list(self.spec.span_layers),
            "exit_layer": self.spec.exit_layer,
            "mlp_attr": self.spec.mlp_attr,
            "input": "hidden_state_entering_zero_cell_at_span_exit_ffn_site",
            "target": (
                "dense_teacher_span_exit_hidden_minus_"
                "zero_cell_collapsed_student_span_exit_hidden"
            ),
            "rows_seen": int(self.rows_seen),
            "rows_profiled": int(self.inputs.shape[0]),
            "input_dim": int(self.inputs.shape[1]),
            "output_dim": int(self.targets.shape[1]),
            "row_group_semantics": "content_addressed_source_window",
            "row_group_count": int(torch.unique(self.row_group_ids).numel()),
            "source_groups": [
                {"group_id": int(key), "sha256": value}
                for key, value in sorted(self.source_group_sha256.items())
            ],
            "calibration_sha256": self.calibration_sha256,
            "content_sha256": self.content_sha256,
            "tensor_sha256": {
                "inputs": tensor_content_sha256(self.inputs),
                "targets": tensor_content_sha256(self.targets),
                "dense_teacher_exits": tensor_content_sha256(self.dense_teacher_exits),
                "zero_student_exits": tensor_content_sha256(self.zero_student_exits),
                "row_group_ids": tensor_content_sha256(self.row_group_ids),
            },
            "student_context": dict(self.context),
        }


def _record_context_manifest(
    records: Sequence[ReplacementRecord],
    current: ReplacementRecord,
    *,
    context_policy: str,
) -> dict[str, Any]:
    cells = []
    for record in records:
        replacement = record.replacement
        cells.append(
            {
                "span_key": f"{record.collapsed_span_layers[0]}:"
                f"{record.collapsed_span_layers[-1]}",
                "role": "profiled_zero_cell" if record is current else "context",
                "configured_plan_sha256": _plan_sha256(replacement),
                "configured_state_sha256": _module_state_sha256(replacement),
            }
        )
    return {
        "schema": "dendritic_collapsed_span_student_context/v1",
        "policy": str(context_policy),
        "profiled_span_exit": int(current.layer_index),
        "profiled_span_exit_cell": "exact_zero",
        "profiled_span_earlier_ffns": "exact_zero",
        "other_spans": {
            "configured": "configured_cells_and_zero_branches_active",
            "isolated_dense": "original_dense_ffns_restored",
            "all_zero": "all_declared_span_ffns_exact_zero",
        }[str(context_policy).strip().lower()],
        "configured_cells": cells,
    }


def capture_collapsed_student_span_boundaries(
    student: nn.Module,
    records: Sequence[ReplacementRecord],
    batches: Sequence[CollapsedSpanCalibrationBatch],
    teacher_cache: DenseTeacherSpanExitCache,
    *,
    layers_attr: str | None = None,
    context_policy: str = "isolated_dense",
    profile_span_keys: Sequence[str] | None = None,
    forward_fn: Callable[[nn.Module, Mapping[str, Any]], Any] | None = None,
) -> dict[str, CapturedCollapsedSpanBoundary]:
    """Second pass: replay selected rows through the zero-cell student."""

    manifests = _validate_calibration_batches(batches)
    if _canonical_json_sha256(list(manifests)) != teacher_cache.calibration_sha256:
        raise ValueError("student calibration batches differ from the teacher pass")
    records = tuple(record for record in records if record.collapsed_span_layers)
    all_specs = collapsed_span_specs_from_records(records)
    all_keys = {spec.key for spec in all_specs}
    requested_keys = (
        set(teacher_cache.samples)
        if profile_span_keys is None
        else {str(key) for key in profile_span_keys}
    )
    if requested_keys != set(teacher_cache.samples):
        raise ValueError("profile_span_keys must match the teacher cache exactly")
    if not requested_keys.issubset(all_keys):
        raise ValueError("teacher cache contains a span absent from the student")
    profile_records = tuple(
        record
        for record in records
        if f"{record.collapsed_span_layers[0]}:{record.collapsed_span_layers[-1]}"
        in requested_keys
    )
    specs = collapsed_span_specs_from_records(profile_records)
    layers = resolve_transformer_layers(student, layers_attr=layers_attr)
    runner = forward_fn or _default_forward
    source_hash_by_group = {
        int(group_id): source_hash
        for batch in batches
        for group_id, source_hash in zip(
            batch.source_group_ids.tolist(), batch.source_group_sha256
        )
    }
    training_states = [(module, bool(module.training)) for module in student.modules()]
    student.eval()
    results: dict[str, CapturedCollapsedSpanBoundary] = {}
    try:
        for record, spec in zip(
            sorted(profile_records, key=lambda item: int(item.layer_index)), specs
        ):
            sample = teacher_cache.samples[spec.key]
            selected_by_batch: dict[int, list[tuple[int, int]]] = {}
            for sample_index, (batch_index, row_index) in enumerate(
                zip(sample.batch_indices.tolist(), sample.row_indices.tolist())
            ):
                selected_by_batch.setdefault(int(batch_index), []).append(
                    (int(row_index), int(sample_index))
                )
            sampled_inputs: list[torch.Tensor | None] = [None] * len(
                sample.batch_indices
            )
            sampled_base: list[torch.Tensor | None] = [None] * len(sample.batch_indices)
            layer = layers[spec.exit_layer]
            with _collapsed_student_context(
                layers,
                records,
                record,
                context_policy=context_policy,
            ) as zero:
                cell_inputs: list[torch.Tensor] = []
                base_exits: list[torch.Tensor] = []

                def capture_input(_module, args, *, captures=cell_inputs):
                    captures.append(args[0].detach())

                def capture_base(
                    _module,
                    _args,
                    output,
                    *,
                    captures=base_exits,
                    span_key=spec.key,
                ):
                    captures.append(
                        _extract_hidden(
                            output, name=f"zero student exit {span_key}"
                        ).detach()
                    )

                handles = [
                    zero.register_forward_pre_hook(capture_input),
                    layer.register_forward_hook(capture_base),
                ]
                try:
                    with torch.no_grad():
                        for batch_index, batch in enumerate(batches):
                            if batch_index not in selected_by_batch:
                                continue
                            before_inputs = len(cell_inputs)
                            before_exits = len(base_exits)
                            runner(student, batch.model_kwargs)
                            if (
                                len(cell_inputs) != before_inputs + 1
                                or len(base_exits) != before_exits + 1
                            ):
                                raise RuntimeError(
                                    f"student did not traverse span exit {spec.key} once"
                                )
                            input_rows, input_positions = _flatten_hidden_rows(
                                cell_inputs.pop(),
                                batch_size=batch.batch_size,
                                name=f"span cell input {spec.key}",
                            )
                            base_rows, base_positions = _flatten_hidden_rows(
                                base_exits.pop(),
                                batch_size=batch.batch_size,
                                name=f"zero student exit {spec.key}",
                            )
                            if input_positions != base_positions:
                                raise ValueError(
                                    "cell inputs and zero-student exits do not align"
                                )
                            for row_index, sample_index in selected_by_batch[
                                batch_index
                            ]:
                                sampled_inputs[sample_index] = (
                                    input_rows[row_index]
                                    .detach()
                                    .to(device="cpu", dtype=torch.float32)
                                )
                                sampled_base[sample_index] = (
                                    base_rows[row_index]
                                    .detach()
                                    .to(device="cpu", dtype=torch.float32)
                                )
                finally:
                    for handle in handles:
                        handle.remove()
            if any(value is None for value in sampled_inputs + sampled_base):
                raise RuntimeError("student replay did not recover every selected row")
            inputs = torch.stack(
                [value for value in sampled_inputs if value is not None]
            )
            base = torch.stack([value for value in sampled_base if value is not None])
            teacher_exits = sample.teacher_exits.to(dtype=torch.float32)
            if inputs.shape != base.shape or inputs.shape != teacher_exits.shape:
                raise ValueError(
                    "collapsed-span correction requires equal hidden dimensions at "
                    "cell input, zero-student exit, and dense-teacher exit"
                )
            targets = teacher_exits - base
            results[spec.key] = CapturedCollapsedSpanBoundary(
                spec=spec,
                inputs=inputs,
                targets=targets,
                dense_teacher_exits=teacher_exits,
                zero_student_exits=base,
                row_group_ids=sample.row_group_ids.clone(),
                source_group_sha256={
                    group_id: source_hash_by_group[group_id]
                    for group_id in sorted(
                        {int(value) for value in sample.row_group_ids}
                    )
                },
                rows_seen=int(sample.rows_seen),
                context=_record_context_manifest(
                    records,
                    record,
                    context_policy=context_policy,
                ),
                calibration_sha256=teacher_cache.calibration_sha256,
            )
    finally:
        for module, was_training in training_states:
            module.training = was_training
    return results


@dataclass(frozen=True)
class SpanSurrogateConfig:
    """Registered estimator and acceptance gate for span correction FMI."""

    random_features: int = 128
    ridge: float = 1.0e-3
    validation_group_fraction: float = 0.25
    profile_group_fraction: float = 0.25
    minimum_groups_per_partition: int = 4
    minimum_validation_r2: float = 0.25
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.random_features) < 1:
            raise ValueError("random_features must be positive")
        if not math.isfinite(float(self.ridge)) or float(self.ridge) <= 0.0:
            raise ValueError("surrogate ridge must be positive and finite")
        if not 0.0 < float(self.validation_group_fraction) < 1.0:
            raise ValueError("validation_group_fraction must lie in (0, 1)")
        if not 0.0 < float(self.profile_group_fraction) < 1.0:
            raise ValueError("profile_group_fraction must lie in (0, 1)")
        if (
            float(self.validation_group_fraction) + float(self.profile_group_fraction)
            >= 1.0
        ):
            raise ValueError("validation and profile group fractions must sum below 1")
        if int(self.minimum_groups_per_partition) < 2:
            raise ValueError("minimum_groups_per_partition must be at least two")
        if not math.isfinite(float(self.minimum_validation_r2)):
            raise ValueError("minimum_validation_r2 must be finite")


@dataclass(frozen=True)
class SpanRankValidationConfig:
    """Preregistered group-disjoint resolution gate for the correction rank."""

    sample_sizes: tuple[int, ...] = (128, 256, 384)
    holdout_rows: int = 128
    max_capacity_fraction: float = 0.8
    max_relative_rank_drift: float = 0.1
    max_heldout_residual: float = 0.02
    max_heldout_residual_drift: float = 0.01
    tail_points: int = 2

    def __post_init__(self) -> None:
        sizes = tuple(sorted({int(value) for value in self.sample_sizes}))
        if len(sizes) < int(self.tail_points) or any(value < 4 for value in sizes):
            raise ValueError(
                "rank validation needs tail_points distinct sample sizes >= 4"
            )
        if int(self.holdout_rows) < 4:
            raise ValueError("rank validation holdout_rows must be at least four")
        object.__setattr__(self, "sample_sizes", sizes)


class RandomFeatureSpanCorrectionSurrogate(nn.Module):
    """Differentiable linear-plus-GELU kernel-ridge correction surrogate."""

    def __init__(
        self,
        *,
        x_mean: torch.Tensor,
        x_scale: torch.Tensor,
        random_weight: torch.Tensor,
        random_bias: torch.Tensor,
        train_x: torch.Tensor,
        train_random_features: torch.Tensor,
        target_mean: torch.Tensor,
        dual: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("x_mean", x_mean)
        self.register_buffer("x_scale", x_scale)
        self.register_buffer("random_weight", random_weight)
        self.register_buffer("random_bias", random_bias)
        self.register_buffer("train_x", train_x)
        self.register_buffer("train_random_features", train_random_features)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("dual", dual)

    def _normalized(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs.float() - self.x_mean) / self.x_scale

    def _random(self, normalized: torch.Tensor) -> torch.Tensor:
        return F.gelu(normalized @ self.random_weight + self.random_bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self._normalized(inputs)
        random = self._random(normalized)
        kernel = (normalized @ self.train_x.T) / normalized.shape[1]
        kernel = kernel + (random @ self.train_random_features.T) / random.shape[1]
        return self.target_mean + kernel @ self.dual


def _group_disjoint_indices(
    group_ids: torch.Tensor,
    *,
    validation_fraction: float,
    profile_fraction: float,
    minimum_groups_per_partition: int,
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    groups = torch.unique(group_ids.to(device="cpu", dtype=torch.int64), sorted=True)
    minimum = int(minimum_groups_per_partition)
    if int(groups.numel()) < 3 * minimum:
        raise ValueError(
            "three-way span FMI requires at least "
            f"{3 * minimum} source groups ({minimum} each for fit, gate-validation, "
            "and untouched fingerprint-profile partitions)"
        )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(int(groups.numel()), generator=generator)
    validation_count = max(minimum, round(float(validation_fraction) * len(groups)))
    profile_count = max(minimum, round(float(profile_fraction) * len(groups)))
    if validation_count + profile_count > len(groups) - minimum:
        raise ValueError(
            "configured three-way group fractions leave fewer than the required "
            f"{minimum} fit groups"
        )
    validation_groups = groups[order[:validation_count]].sort().values
    profile_groups = (
        groups[order[validation_count : validation_count + profile_count]].sort().values
    )
    train_groups = groups[order[validation_count + profile_count :]].sort().values
    ids = group_ids.to(device="cpu", dtype=torch.int64)
    validation_index = torch.nonzero(
        torch.isin(ids, validation_groups), as_tuple=False
    ).reshape(-1)
    profile_index = torch.nonzero(
        torch.isin(ids, profile_groups), as_tuple=False
    ).reshape(-1)
    train_index = torch.nonzero(torch.isin(ids, train_groups), as_tuple=False).reshape(
        -1
    )
    if (
        min(
            int(train_index.numel()),
            int(validation_index.numel()),
            int(profile_index.numel()),
        )
        < 4
    ):
        raise ValueError(
            "three-way group-disjoint split needs at least four rows per partition"
        )
    return (
        train_index,
        validation_index,
        profile_index,
        train_groups,
        validation_groups,
        profile_groups,
    )


def _source_group_partition(
    boundary: CapturedCollapsedSpanBoundary,
    groups: torch.Tensor,
) -> list[dict[str, Any]]:
    return [
        {
            "group_id": int(group_id),
            "source_sha256": boundary.source_group_sha256[int(group_id)],
        }
        for group_id in groups.tolist()
    ]


def fit_span_correction_surrogate(
    boundary: CapturedCollapsedSpanBoundary,
    *,
    config: SpanSurrogateConfig | None = None,
    device: torch.device | str = "cpu",
) -> tuple[RandomFeatureSpanCorrectionSurrogate, dict[str, Any], torch.Tensor]:
    """Fit and validate the declared observational correction surrogate."""

    cfg = config or SpanSurrogateConfig()
    (
        train_index,
        validation_index,
        profile_index,
        train_groups,
        validation_groups,
        profile_groups,
    ) = _group_disjoint_indices(
        boundary.row_group_ids,
        validation_fraction=cfg.validation_group_fraction,
        profile_fraction=cfg.profile_group_fraction,
        minimum_groups_per_partition=cfg.minimum_groups_per_partition,
        seed=cfg.seed,
    )
    target_device = torch.device(device)
    train_x = boundary.inputs[train_index].to(device=target_device, dtype=torch.float32)
    train_y = boundary.targets[train_index].to(
        device=target_device, dtype=torch.float32
    )
    validation_x = boundary.inputs[validation_index].to(
        device=target_device, dtype=torch.float32
    )
    validation_y = boundary.targets[validation_index].to(
        device=target_device, dtype=torch.float32
    )
    x_mean = train_x.mean(dim=0)
    x_scale = (train_x - x_mean).square().mean(dim=0).sqrt().clamp_min(1.0e-6)
    normalized_train = (train_x - x_mean) / x_scale
    generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed) + 7_919)
    random_weight = torch.randn(
        normalized_train.shape[1],
        int(cfg.random_features),
        generator=generator,
    ).to(device=target_device) / math.sqrt(int(normalized_train.shape[1]))
    random_bias = (
        2.0 * math.pi * torch.rand(int(cfg.random_features), generator=generator)
    ).to(device=target_device)
    train_random = F.gelu(normalized_train @ random_weight + random_bias)
    kernel = (normalized_train @ normalized_train.T) / normalized_train.shape[1]
    kernel = kernel + (train_random @ train_random.T) / train_random.shape[1]
    kernel_scale = torch.diagonal(kernel).mean().clamp_min(1.0e-12)
    regularizer = float(cfg.ridge) * kernel_scale
    regularized = kernel + regularizer * torch.eye(
        kernel.shape[0], device=target_device, dtype=kernel.dtype
    )
    target_mean = train_y.mean(dim=0)
    factor, info = torch.linalg.cholesky_ex(regularized)
    if bool((info != 0).any()):
        raise RuntimeError("span correction surrogate ridge system is not SPD")
    dual = torch.cholesky_solve(train_y - target_mean, factor)
    surrogate = RandomFeatureSpanCorrectionSurrogate(
        x_mean=x_mean,
        x_scale=x_scale,
        random_weight=random_weight,
        random_bias=random_bias,
        train_x=normalized_train,
        train_random_features=train_random,
        target_mean=target_mean,
        dual=dual,
    )
    with torch.no_grad():
        prediction = surrogate(validation_x)
    squared_error = (prediction - validation_y).square().sum()
    baseline_error = (validation_y - target_mean).square().sum().clamp_min(1.0e-12)
    r2 = float(1.0 - squared_error / baseline_error)
    normalized_rmse = float(
        (squared_error / validation_y.numel()).sqrt()
        / validation_y.square().mean().sqrt().clamp_min(1.0e-12)
    )
    centered_prediction = prediction - prediction.mean(dim=0)
    centered_target = validation_y - validation_y.mean(dim=0)
    cosine = float(
        (centered_prediction * centered_target).sum()
        / (
            centered_prediction.square().sum().sqrt()
            * centered_target.square().sum().sqrt()
        ).clamp_min(1.0e-12)
    )
    passed = math.isfinite(r2) and r2 >= float(cfg.minimum_validation_r2)
    report = {
        "schema": "dendritic_collapsed_span_surrogate_validation/v1",
        "estimator": "linear_plus_fixed_gelu_random_features_kernel_ridge",
        "config": asdict(cfg),
        "fit_device": str(target_device),
        "split_unit": "content_addressed_source_group",
        "group_disjoint": True,
        "three_way_group_disjoint": True,
        "partition_roles": {
            "fit": "surrogate parameter estimation only",
            "gate_validation": "surrogate acceptance only",
            "fingerprint_profile": (
                "untouched groups used for measured spectrum and differential "
                "FMI features only after the gate passes"
            ),
        },
        "train_rows": int(train_index.numel()),
        "gate_validation_rows": int(validation_index.numel()),
        "fingerprint_profile_rows": int(profile_index.numel()),
        "train_group_count": int(train_groups.numel()),
        "gate_validation_group_count": int(validation_groups.numel()),
        "fingerprint_profile_group_count": int(profile_groups.numel()),
        "configured_group_fractions": {
            "fit": float(
                1.0 - cfg.validation_group_fraction - cfg.profile_group_fraction
            ),
            "gate_validation": float(cfg.validation_group_fraction),
            "fingerprint_profile": float(cfg.profile_group_fraction),
        },
        "realized_group_fractions": {
            "fit": float(
                train_groups.numel() / torch.unique(boundary.row_group_ids).numel()
            ),
            "gate_validation": float(
                validation_groups.numel() / torch.unique(boundary.row_group_ids).numel()
            ),
            "fingerprint_profile": float(
                profile_groups.numel() / torch.unique(boundary.row_group_ids).numel()
            ),
        },
        "train_group_ids_sha256": tensor_content_sha256(train_groups),
        "gate_validation_group_ids_sha256": tensor_content_sha256(validation_groups),
        "fingerprint_profile_group_ids_sha256": tensor_content_sha256(profile_groups),
        "source_group_partitions": {
            "fit": _source_group_partition(boundary, train_groups),
            "gate_validation": _source_group_partition(boundary, validation_groups),
            "fingerprint_profile": _source_group_partition(boundary, profile_groups),
        },
        "regularizer": float(regularizer),
        "validation_r2": r2,
        "validation_normalized_rmse": normalized_rmse,
        "validation_centered_cosine": cosine,
        "minimum_validation_r2": float(cfg.minimum_validation_r2),
        "passed": bool(passed),
        "interpretation": (
            "output-spectrum statistics use measured held-out correction targets; "
            "gradient/support/sign/interaction/mechanism statistics use this "
            "observational surrogate and are admissible only when passed=true"
        ),
    }
    return surrogate, report, profile_index


def profile_collapsed_span_boundary(
    boundary: CapturedCollapsedSpanBoundary,
    *,
    profiler_config: FMIProfilerConfig | None = None,
    surrogate_config: SpanSurrogateConfig | None = None,
    rank_validation_config: SpanRankValidationConfig | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Profile one measured span correction with a validated local surrogate."""

    profiler_cfg = profiler_config or FMIProfilerConfig()
    rank_cfg = rank_validation_config or SpanRankValidationConfig()
    target_device = torch.device(device)
    rank_validation = validate_rank_estimate(
        boundary.targets.to(device=target_device),
        sample_sizes=rank_cfg.sample_sizes,
        holdout_rows=rank_cfg.holdout_rows,
        epsilon=profiler_cfg.spectrum_epsilon,
        estimator=profiler_cfg.spectrum_estimator,
        winsorize_quantile=profiler_cfg.spectrum_winsorize_quantile,
        huber_c=profiler_cfg.spectrum_huber_c,
        huber_max_iters=profiler_cfg.spectrum_huber_max_iters,
        mom_blocks=profiler_cfg.spectrum_mom_blocks,
        seed=profiler_cfg.seed,
        max_capacity_fraction=rank_cfg.max_capacity_fraction,
        max_relative_rank_drift=rank_cfg.max_relative_rank_drift,
        max_heldout_residual=rank_cfg.max_heldout_residual,
        max_heldout_residual_drift=rank_cfg.max_heldout_residual_drift,
        tail_points=rank_cfg.tail_points,
        group_ids=boundary.row_group_ids,
        require_group_disjoint=True,
        source_provenance={
            "schema": "dendritic_collapsed_span_rank_source/v1",
            "span_key": boundary.spec.key,
            "boundary_content_sha256": boundary.content_sha256,
            "target": (
                "dense_teacher_span_exit_hidden_minus_"
                "zero_cell_collapsed_student_span_exit_hidden"
            ),
        },
    )
    surrogate, validation, profile_index = fit_span_correction_surrogate(
        boundary,
        config=surrogate_config,
        device=target_device,
    )
    result: dict[str, Any] = {
        "boundary": boundary.metadata(),
        "rank_validation_config": asdict(rank_cfg),
        "rank_validation": rank_validation,
        "surrogate_validation": validation,
        "status": (
            "validated_observational_span_fingerprint"
            if validation["passed"] and rank_validation["passed"]
            else (
                "surrogate_quality_gate_failed_no_compilation"
                if not validation["passed"]
                else "rank_resolution_gate_failed_no_compilation"
            )
        ),
    }
    if not validation["passed"]:
        result["fingerprint"] = None
        return result
    surrogate = surrogate.to(device=target_device)
    inputs = boundary.inputs[profile_index].to(device=target_device)
    targets = boundary.targets[profile_index].to(device=target_device)
    fingerprint = profile_teacher_component(
        surrogate,
        inputs,
        outputs=targets,
        config=profiler_cfg,
    )
    profile_partition_rank = int(fingerprint["output_spectrum_rank_995"])
    if rank_validation["passed"]:
        fingerprint["output_spectrum_rank_995"] = int(rank_validation["validated_rank"])
    fingerprint["span_rank_resolution"] = {
        "schema": "dendritic_collapsed_span_rank_resolution/v1",
        "passed": bool(rank_validation["passed"]),
        "selection_rank": rank_validation["validated_rank"],
        "profile_partition_candidate_rank": profile_partition_rank,
        "selection_basis": (
            "full_capture_group_disjoint_resolution_validated_rank"
            if rank_validation["passed"]
            else "unresolved_no_selection_rank"
        ),
    }
    fingerprint["span_correction_scope"] = {
        "schema": "dendritic_collapsed_span_fingerprint_scope/v1",
        "span_key": boundary.spec.key,
        "target": (
            "dense_teacher_span_exit_hidden_minus_"
            "zero_cell_collapsed_student_span_exit_hidden"
        ),
        "output_spectrum_source": "measured_group_disjoint_targets",
        "differential_statistics_source": "validated_observational_surrogate",
        "fingerprint_partition": "untouched_source_groups_after_fit_and_gate",
        "surrogate_validation_r2": validation["validation_r2"],
        "not_equivalent_to_single_layer_fmi": True,
    }
    result["fingerprint"] = fingerprint
    return result


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    try:
        from omegaconf import OmegaConf

        converted = OmegaConf.to_container(value, resolve=True)
    except (ImportError, TypeError, ValueError):
        converted = None
    return dict(converted) if isinstance(converted, Mapping) else {}


def _span_selection_config(
    configured: Any,
    fingerprint: Mapping[str, Any],
    *,
    allow_non_biological: bool,
) -> dict[str, Any]:
    selection = deepcopy(_plain_mapping(configured))
    selection.update(
        {
            "enabled": True,
            "mode": "fmi",
            "rule_set": str(selection.get("rule_set", "theory_v4")),
            "allow_non_biological": bool(allow_non_biological),
            "fingerprint": {"values": dict(fingerprint)},
        }
    )
    selection.pop("path", None)
    return selection


def compile_collapsed_span_profiles(
    profiles: Mapping[str, Mapping[str, Any]],
    *,
    configured_selection: Any,
    teacher_intermediate_sizes: Mapping[str, int],
    model_artifact_set_sha256: str,
    include_non_biological: bool = False,
) -> dict[str, Any]:
    """Compile exact-coverage config plans from validated span fingerprints.

    The returned ``compiled_plans_by_collapsed_span`` value can be copied
    directly into ``model.transformer_replacement``.  Compilation is atomic:
    one missing/failed surrogate, size, or span key prevents a partial mapping
    that the collapsed-span patcher could accidentally treat as complete.
    """

    from dendritic_modeling.networks.architectures.replacement import (
        resolve_replacement_selection,
    )

    if not profiles:
        raise ValueError("at least one collapsed-span profile is required")
    model_artifact_set_sha256 = str(model_artifact_set_sha256).lower()
    if len(model_artifact_set_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in model_artifact_set_sha256
    ):
        raise ValueError(
            "compiled span FMI plans require a verified model artifact-set SHA-256"
        )
    if set(profiles) != set(teacher_intermediate_sizes):
        raise ValueError(
            "teacher_intermediate_sizes must cover the profiled spans exactly"
        )
    compiled: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    for key in sorted(
        profiles,
        key=lambda value: tuple(int(part) for part in value.split(":")),
    ):
        entry = dict(profiles[key])
        boundary = entry.get("boundary")
        fingerprint = entry.get("fingerprint")
        validation = entry.get("surrogate_validation")
        if not isinstance(boundary, Mapping) or str(boundary.get("span_key")) != key:
            raise ValueError(f"profile {key!r} lacks its exact span boundary manifest")
        if (
            not isinstance(validation, Mapping)
            or not bool(validation.get("passed", False))
            or not isinstance(fingerprint, Mapping)
        ):
            raise ValueError(
                f"span {key} did not pass its observational-surrogate quality gate"
            )
        rank_validation = entry.get("rank_validation")
        if not isinstance(rank_validation, Mapping) or not bool(
            rank_validation.get("passed", False)
        ):
            raise ValueError(
                f"span {key} did not pass its group-disjoint rank-resolution gate"
            )
        hidden_size = int(boundary.get("input_dim", 0))
        output_size = int(boundary.get("output_dim", 0))
        intermediate_size = int(teacher_intermediate_sizes[key])
        if hidden_size < 1 or output_size != hidden_size or intermediate_size < 1:
            raise ValueError(
                f"span {key} has incompatible hidden/intermediate dimensions"
            )
        domains = {"biological_only": False}
        if include_non_biological:
            domains["biological_and_non_biological"] = True
        selections: dict[str, dict[str, Any]] = {}
        for domain, allow_non_biological in domains.items():
            resolved = resolve_replacement_selection(
                _span_selection_config(
                    configured_selection,
                    fingerprint,
                    allow_non_biological=allow_non_biological,
                ),
                hidden_size=hidden_size,
                teacher_intermediate_size=intermediate_size,
                layer_index=int(boundary["exit_layer"]),
            )
            selections[domain] = resolved.as_dict()
        primary_domain = (
            "biological_and_non_biological"
            if include_non_biological
            else "biological_only"
        )
        plan = selections[primary_domain]["plan"]
        plan_sha256 = _canonical_json_sha256(plan)
        compiled[key] = plan
        manifests[key] = {
            "schema": "dendritic_compiled_collapsed_span_fmi_plan/v1",
            "span_key": key,
            "span_layers": list(boundary["span_layers"]),
            "exit_layer": int(boundary["exit_layer"]),
            "boundary_content_sha256": str(boundary["content_sha256"]),
            "fingerprint_sha256": _canonical_json_sha256(fingerprint),
            "surrogate_validation": dict(validation),
            "rank_validation": dict(rank_validation),
            "teacher_intermediate_size": intermediate_size,
            "candidate_domains": selections,
            "primary_domain": primary_domain,
            "compiled_plan_sha256": plan_sha256,
            "model_artifact_set_sha256": model_artifact_set_sha256,
            "status": (
                "prospectively_compiled_hypothesis_requires_reference_grid_"
                "and_disjoint_recovery_validation"
            ),
            "not_single_layer_fmi": True,
        }
    return {
        "schema": "dendritic_compiled_collapsed_span_plans/v1",
        "compiled_plans_by_collapsed_span": compiled,
        "plan_manifests_by_collapsed_span": manifests,
        "teacher_binding": {
            "schema": "dendritic_collapsed_span_teacher_binding/v1",
            "model_artifact_set_sha256": model_artifact_set_sha256,
            "weight_content_addressed": True,
        },
        "coverage": {
            "required_span_keys": list(compiled),
            "complete": True,
            "partial_plans_emitted": False,
        },
    }


def save_collapsed_span_captures(
    captured: Mapping[str, CapturedCollapsedSpanBoundary],
    output_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Write content-named capture artifacts and return file manifests."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, dict[str, Any]] = {}
    for key, boundary in captured.items():
        if key != boundary.spec.key:
            raise ValueError("capture mapping key does not match its span")
        filename = f"span_{key.replace(':', '_')}_{boundary.content_sha256[:16]}.pt"
        path = destination / filename
        payload = {
            "schema": "dendritic_collapsed_span_fmi_capture/v1",
            "metadata": boundary.metadata(),
            "inputs": boundary.inputs,
            "targets": boundary.targets,
            "dense_teacher_exits": boundary.dense_teacher_exits,
            "zero_student_exits": boundary.zero_student_exits,
            "row_group_ids": boundary.row_group_ids,
            "source_group_sha256": dict(boundary.source_group_sha256),
        }
        torch.save(payload, path)
        manifests[key] = {
            "path": str(path.resolve()),
            "bytes": int(path.stat().st_size),
            "sha256": _file_sha256(path),
            "semantic_content_sha256": boundary.content_sha256,
        }
    return manifests


__all__ = [
    "CapturedCollapsedSpanBoundary",
    "CollapsedSpanCalibrationBatch",
    "CollapsedSpanSpec",
    "DenseTeacherSpanExitCache",
    "DenseTeacherSpanSample",
    "RandomFeatureSpanCorrectionSurrogate",
    "SpanRankValidationConfig",
    "SpanSurrogateConfig",
    "capture_collapsed_student_span_boundaries",
    "capture_dense_teacher_span_exits",
    "collapsed_span_specs_from_records",
    "compile_collapsed_span_profiles",
    "content_addressed_calibration_batch",
    "fit_span_correction_surrogate",
    "profile_collapsed_span_boundary",
    "save_collapsed_span_captures",
    "tensor_content_sha256",
]
