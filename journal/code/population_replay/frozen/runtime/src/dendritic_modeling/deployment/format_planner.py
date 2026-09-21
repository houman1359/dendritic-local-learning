"""Pure, fail-closed planning for projection deployment formats.

The planner in this module does not convert a module, choose a kernel, or
predict speed.  It turns an explicit logical projection and optional verified
topology layouts into a set of byte- and MAC-accounted representation
candidates.  In the absence of paired measured-latency receipts it returns a
resource-only Pareto set and deliberately leaves ``selected_candidate_id``
unset.

Latency-based selection is possible only over an explicit, completely
measured comparison scope.  Receipts are bound to both the projection and the
candidate fingerprints and must share one benchmark artifact, device,
operation, comparison identifier, and code commit.  This makes the API usable
by the replacement compiler later without allowing ``auto`` to become an
unrecorded deployment decision.

All byte counts describe persistent runtime projection payload (values,
indices, and required encoding metadata).  Activation/workspace bytes and
allocator overhead are outside this contract.  A MAC is one multiply-
accumulate for one input row; no latency, throughput, or energy is inferred
from MAC count.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

PLAN_SCHEMA = "dendritic_projection_deployment_format_plan/v1"
RECEIPT_SCHEMA = "dendritic_projection_latency_receipt/v1"

_VALUE_BYTES = {"fp32": 4, "bf16": 2, "fp8": 1}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,64}$")

__all__ = [
    "BlockLocalUInt16Layout",
    "BlockSparseLayout",
    "DeploymentFormatCandidate",
    "DeploymentFormatPlan",
    "FP8ScaleLayout",
    "MeasuredLatencyReceipt",
    "ProjectionFormatRequest",
    "StructuredNMLayout",
    "plan_projection_formats",
]


def _ceil_div(numerator: int, denominator: int) -> int:
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def _align_up(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_storage_scalar(value: int, *, name: str) -> int:
    resolved = int(value)
    if resolved not in (1, 2, 4, 8):
        raise ValueError(f"{name} must be one of 1, 2, 4, or 8 bytes")
    return resolved


def _validate_value_dtype(value: str, *, allow_fp8: bool = False) -> str:
    normalized = str(value).strip().lower()
    allowed = {"bf16", "fp32"} | ({"fp8"} if allow_fp8 else set())
    if normalized not in allowed:
        raise ValueError(f"value_dtype must be one of {sorted(allowed)}")
    return normalized


@dataclass(frozen=True)
class FP8ScaleLayout:
    """Explicit block-scale storage for an FP8 dense projection.

    One scale (and, optionally, one zero point) is retained for each
    ``block_rows x block_cols`` weight block.  Edge blocks use one full
    metadata entry even when only partly occupied.
    """

    block_rows: int
    block_cols: int
    scale_bytes: int
    zero_point_bytes: int = 0

    def __post_init__(self) -> None:
        if min(int(self.block_rows), int(self.block_cols)) < 1:
            raise ValueError("FP8 scale block dimensions must be positive")
        _validate_storage_scalar(self.scale_bytes, name="scale_bytes")
        if int(self.zero_point_bytes) not in (0, 1, 2, 4, 8):
            raise ValueError("zero_point_bytes must be 0, 1, 2, 4, or 8")

    def metadata_bytes(self, *, in_features: int, out_features: int) -> int:
        blocks = _ceil_div(out_features, self.block_rows) * _ceil_div(
            in_features, self.block_cols
        )
        return int(blocks * (self.scale_bytes + self.zero_point_bytes))


@dataclass(frozen=True)
class BlockLocalUInt16Layout:
    """Exact grouped encoding for uint16 indices in large input spaces.

    Contacts are grouped by output and input block.  The runtime payload is:

    * one uint16 local index per active contact;
    * ``group_count`` input-block identifiers;
    * ``out_features + 1`` pointers from outputs to groups; and
    * ``group_count + 1`` pointers from groups to contacts.

    ``topology_verified`` must attest that the existing fixed topology was
    encoded this way; the planner never rearranges or blockifies contacts.
    """

    block_size: int
    group_count: int
    block_id_bytes: int = 4
    pointer_bytes: int = 4
    topology_verified: bool = False

    def __post_init__(self) -> None:
        if int(self.block_size) < 1:
            raise ValueError("block_size must be positive")
        if int(self.group_count) < 1:
            raise ValueError("group_count must be positive")
        _validate_storage_scalar(self.block_id_bytes, name="block_id_bytes")
        _validate_storage_scalar(self.pointer_bytes, name="pointer_bytes")


@dataclass(frozen=True)
class StructuredNMLayout:
    """Verified N:M topology and its exact row-packed metadata layout."""

    nonzero: int
    group_size: int
    metadata_bits_per_group: int
    metadata_alignment_bytes: int = 1
    value_dtype: str = "bf16"
    topology_verified: bool = False

    def __post_init__(self) -> None:
        if not 1 <= int(self.nonzero) <= int(self.group_size):
            raise ValueError("structured nonzero must be in [1, group_size]")
        if int(self.metadata_bits_per_group) < 0:
            raise ValueError("metadata_bits_per_group must be nonnegative")
        if int(self.metadata_alignment_bytes) < 1:
            raise ValueError("metadata_alignment_bytes must be positive")
        _validate_value_dtype(self.value_dtype)


@dataclass(frozen=True)
class BlockSparseLayout:
    """Verified block-CSR topology with fully materialized active blocks."""

    block_rows: int
    block_cols: int
    active_blocks: int
    value_dtype: str = "bf16"
    column_index_bytes: int = 4
    row_pointer_bytes: int = 4
    topology_verified: bool = False

    def __post_init__(self) -> None:
        if (
            min(
                int(self.block_rows),
                int(self.block_cols),
                int(self.active_blocks),
            )
            < 1
        ):
            raise ValueError("block dimensions and active_blocks must be positive")
        _validate_value_dtype(self.value_dtype)
        _validate_storage_scalar(self.column_index_bytes, name="column_index_bytes")
        _validate_storage_scalar(self.row_pointer_bytes, name="row_pointer_bytes")


@dataclass(frozen=True)
class ProjectionFormatRequest:
    """Logical projection and already-observed topology facts.

    ``active_contacts`` is the number of retained nonzero/contact slots.
    Uniform indexed ELL candidates additionally require
    ``contacts_per_output``.  Nonuniform block-local and block-CSR layouts use
    their explicit pointer payload instead.
    """

    projection_id: str
    in_features: int
    out_features: int
    active_contacts: int
    contacts_per_output: int | None = None
    fixed_topology: bool = True
    topology_sha256: str | None = None
    tensor_core_multiple: int = 16
    fp8_scale_layout: FP8ScaleLayout | None = None
    block_local_uint16: BlockLocalUInt16Layout | None = None
    structured_nm: StructuredNMLayout | None = None
    block_sparse: BlockSparseLayout | None = None

    def __post_init__(self) -> None:
        if not str(self.projection_id).strip():
            raise ValueError("projection_id must be nonempty")
        in_features = int(self.in_features)
        out_features = int(self.out_features)
        active = int(self.active_contacts)
        if min(in_features, out_features, active) < 1:
            raise ValueError(
                "projection dimensions and active_contacts must be positive"
            )
        dense_contacts = in_features * out_features
        if active > dense_contacts:
            raise ValueError("active_contacts cannot exceed the dense projection")
        if self.contacts_per_output is not None:
            contacts = int(self.contacts_per_output)
            if not 1 <= contacts <= in_features:
                raise ValueError("contacts_per_output must be in [1, in_features]")
            if contacts * out_features != active:
                raise ValueError(
                    "active_contacts must equal out_features * contacts_per_output"
                )
        if int(self.tensor_core_multiple) < 1:
            raise ValueError("tensor_core_multiple must be positive")
        if self.topology_sha256 is not None and not _SHA256_RE.fullmatch(
            str(self.topology_sha256)
        ):
            raise ValueError("topology_sha256 must be a lowercase SHA-256")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeploymentFormatCandidate:
    """One representation candidate with no implied runtime support."""

    projection_fingerprint: str
    format_id: str
    family: str
    legal: bool
    illegality_reasons: tuple[str, ...]
    value_dtype: str
    index_dtype: str | None
    stored_value_elements: int | None
    value_bytes: int | None
    index_bytes: int | None
    metadata_bytes: int | None
    tensor_core_padding_bytes: int | None
    runtime_bytes: int | None
    active_macs_per_input_row: int | None
    executed_macs_per_input_row: int | None
    logical_in_features: int
    logical_out_features: int
    stored_in_features: int | None
    stored_out_features: int | None
    details: dict[str, Any]
    candidate_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MeasuredLatencyReceipt:
    """A paired benchmark observation bound to an exact planner candidate."""

    candidate_id: str
    projection_fingerprint: str
    candidate_fingerprint: str
    comparison_id: str
    operation: str
    latency_ms: float
    sample_count: int
    device: str
    code_commit: str
    artifact_sha256: str
    numerical_gate_id: str
    numerical_gate_passed: bool
    max_abs_error: float | None = None
    max_relative_error: float | None = None
    atol: float | None = None
    rtol: float | None = None
    quality_criterion: str | None = None

    def __post_init__(self) -> None:
        if not str(self.candidate_id).strip():
            raise ValueError("receipt candidate_id must be nonempty")
        for name, value in (
            ("projection_fingerprint", self.projection_fingerprint),
            ("candidate_fingerprint", self.candidate_fingerprint),
            ("artifact_sha256", self.artifact_sha256),
        ):
            if not _SHA256_RE.fullmatch(str(value)):
                raise ValueError(f"receipt {name} must be a lowercase SHA-256")
        if not str(self.comparison_id).strip():
            raise ValueError("receipt comparison_id must be nonempty")
        if str(self.operation) not in {"forward", "forward_backward"}:
            raise ValueError("receipt operation must be forward or forward_backward")
        latency = float(self.latency_ms)
        if not math.isfinite(latency) or latency <= 0.0:
            raise ValueError("receipt latency_ms must be finite and positive")
        if int(self.sample_count) < 1:
            raise ValueError("receipt sample_count must be positive")
        if not str(self.device).strip():
            raise ValueError("receipt device must be nonempty")
        if not _COMMIT_RE.fullmatch(str(self.code_commit)):
            raise ValueError("receipt code_commit must be a hexadecimal commit id")
        if not str(self.numerical_gate_id).strip():
            raise ValueError("receipt numerical_gate_id must be nonempty")
        if not isinstance(self.numerical_gate_passed, bool):
            raise ValueError("receipt numerical_gate_passed must be boolean")
        parity_values = (
            self.max_abs_error,
            self.max_relative_error,
            self.atol,
            self.rtol,
        )
        has_any_parity_value = any(value is not None for value in parity_values)
        has_all_parity_values = all(value is not None for value in parity_values)
        criterion = (
            None
            if self.quality_criterion is None
            else str(self.quality_criterion).strip()
        )
        if has_any_parity_value and not has_all_parity_values:
            raise ValueError(
                "receipt parity gate requires max_abs_error, max_relative_error, "
                "atol, and rtol together"
            )
        if not has_all_parity_values and not criterion:
            raise ValueError(
                "receipt requires finite parity measurements/tolerances or a "
                "documented quality_criterion"
            )
        if has_all_parity_values:
            for name, value in zip(
                ("max_abs_error", "max_relative_error", "atol", "rtol"),
                parity_values,
                strict=True,
            ):
                resolved = float(value)
                if not math.isfinite(resolved) or resolved < 0.0:
                    raise ValueError(f"receipt {name} must be finite and nonnegative")
        if self.quality_criterion is not None and not criterion:
            raise ValueError("receipt quality_criterion must be nonempty when set")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": RECEIPT_SCHEMA,
            "measurement_kind": "measured",
            **asdict(self),
        }


@dataclass(frozen=True)
class DeploymentFormatPlan:
    """Serializable planner result."""

    schema: str
    request: dict[str, Any]
    projection_fingerprint: str
    candidates: tuple[DeploymentFormatCandidate, ...]
    resource_pareto_candidate_ids: tuple[str, ...]
    selection: dict[str, Any]
    claim_boundary: dict[str, str]

    @property
    def selected_candidate_id(self) -> str | None:
        selected = self.selection.get("selected_candidate_id")
        return None if selected is None else str(selected)

    def candidate(self, format_id: str) -> DeploymentFormatCandidate:
        matches = [item for item in self.candidates if item.format_id == format_id]
        if len(matches) != 1:
            raise KeyError(format_id)
        return matches[0]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "request": self.request,
            "projection_fingerprint": self.projection_fingerprint,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "resource_pareto_candidate_ids": list(self.resource_pareto_candidate_ids),
            "selection": self.selection,
            "claim_boundary": self.claim_boundary,
        }


def _candidate(
    *,
    projection_fingerprint: str,
    format_id: str,
    family: str,
    request: ProjectionFormatRequest,
    reasons: Sequence[str] = (),
    value_dtype: str,
    index_dtype: str | None,
    stored_value_elements: int | None,
    value_bytes: int | None,
    index_bytes: int | None,
    metadata_bytes: int | None,
    tensor_core_padding_bytes: int | None,
    active_macs: int | None,
    executed_macs: int | None,
    stored_in_features: int | None,
    stored_out_features: int | None,
    details: dict[str, Any] | None = None,
) -> DeploymentFormatCandidate:
    normalized_reasons = tuple(dict.fromkeys(str(reason) for reason in reasons))
    legal = not normalized_reasons
    runtime_bytes = None
    if legal:
        if None in (value_bytes, index_bytes, metadata_bytes):
            raise AssertionError("legal candidates require complete byte accounting")
        runtime_bytes = int(value_bytes) + int(index_bytes) + int(metadata_bytes)
    payload = {
        "projection_fingerprint": projection_fingerprint,
        "format_id": format_id,
        "family": family,
        "legal": legal,
        "illegality_reasons": normalized_reasons,
        "value_dtype": value_dtype,
        "index_dtype": index_dtype,
        "stored_value_elements": stored_value_elements,
        "value_bytes": value_bytes,
        "index_bytes": index_bytes,
        "metadata_bytes": metadata_bytes,
        "tensor_core_padding_bytes": tensor_core_padding_bytes,
        "runtime_bytes": runtime_bytes,
        "active_macs_per_input_row": active_macs,
        "executed_macs_per_input_row": executed_macs,
        "logical_in_features": int(request.in_features),
        "logical_out_features": int(request.out_features),
        "stored_in_features": stored_in_features,
        "stored_out_features": stored_out_features,
        "details": details or {},
    }
    return DeploymentFormatCandidate(
        **payload,
        candidate_fingerprint=_canonical_sha256(payload),
    )


def _uniform_index_reasons(
    request: ProjectionFormatRequest, *, index_cardinality: int
) -> list[str]:
    reasons = []
    if not request.fixed_topology:
        reasons.append("fixed_topology_required")
    if request.contacts_per_output is None:
        reasons.append("uniform_contacts_per_output_required")
    if int(request.in_features) > int(index_cardinality):
        reasons.append("input_dimension_exceeds_index_cardinality")
    return reasons


def _dense_candidates(
    request: ProjectionFormatRequest, projection_fingerprint: str
) -> list[DeploymentFormatCandidate]:
    in_features = int(request.in_features)
    out_features = int(request.out_features)
    dense_elements = in_features * out_features
    multiple = int(request.tensor_core_multiple)
    padded_in = _align_up(in_features, multiple)
    padded_out = _align_up(out_features, multiple)
    padded_elements = padded_in * padded_out
    candidates = [
        _candidate(
            projection_fingerprint=projection_fingerprint,
            format_id="dense_bf16",
            family="dense",
            request=request,
            value_dtype="bf16",
            index_dtype=None,
            stored_value_elements=dense_elements,
            value_bytes=dense_elements * _VALUE_BYTES["bf16"],
            index_bytes=0,
            metadata_bytes=0,
            tensor_core_padding_bytes=0,
            active_macs=dense_elements,
            executed_macs=dense_elements,
            stored_in_features=in_features,
            stored_out_features=out_features,
            details={"scale_metadata_included": False},
        )
    ]
    if (padded_in, padded_out) != (in_features, out_features):
        candidates.append(
            _candidate(
                projection_fingerprint=projection_fingerprint,
                format_id="dense_bf16_tensor_core_padded",
                family="dense",
                request=request,
                value_dtype="bf16",
                index_dtype=None,
                stored_value_elements=padded_elements,
                value_bytes=padded_elements * _VALUE_BYTES["bf16"],
                index_bytes=0,
                metadata_bytes=0,
                tensor_core_padding_bytes=(padded_elements - dense_elements)
                * _VALUE_BYTES["bf16"],
                active_macs=dense_elements,
                executed_macs=padded_elements,
                stored_in_features=padded_in,
                stored_out_features=padded_out,
                details={
                    "padding_multiple": multiple,
                    "zero_padding_required": True,
                    "dynamic_padding_cost_included": False,
                },
            )
        )

    scale_layout = request.fp8_scale_layout
    fp8_reason = [] if scale_layout is not None else ["fp8_scale_layout_required"]
    fp8_metadata = (
        scale_layout.metadata_bytes(in_features=in_features, out_features=out_features)
        if scale_layout is not None
        else None
    )
    candidates.append(
        _candidate(
            projection_fingerprint=projection_fingerprint,
            format_id="dense_fp8",
            family="dense",
            request=request,
            reasons=fp8_reason,
            value_dtype="fp8",
            index_dtype=None,
            stored_value_elements=dense_elements,
            value_bytes=dense_elements,
            index_bytes=0 if scale_layout is not None else None,
            metadata_bytes=fp8_metadata,
            tensor_core_padding_bytes=0,
            active_macs=dense_elements,
            executed_macs=dense_elements,
            stored_in_features=in_features,
            stored_out_features=out_features,
            details={
                "scale_layout": (
                    asdict(scale_layout) if scale_layout is not None else None
                ),
                "runtime_kernel_support_not_inferred": True,
            },
        )
    )
    if (padded_in, padded_out) != (in_features, out_features):
        padded_metadata = (
            scale_layout.metadata_bytes(in_features=padded_in, out_features=padded_out)
            if scale_layout is not None
            else None
        )
        exact_runtime = (
            dense_elements + int(fp8_metadata) if fp8_metadata is not None else None
        )
        padded_runtime = (
            padded_elements + int(padded_metadata)
            if padded_metadata is not None
            else None
        )
        candidates.append(
            _candidate(
                projection_fingerprint=projection_fingerprint,
                format_id="dense_fp8_tensor_core_padded",
                family="dense",
                request=request,
                reasons=fp8_reason,
                value_dtype="fp8",
                index_dtype=None,
                stored_value_elements=padded_elements,
                value_bytes=padded_elements,
                index_bytes=0 if scale_layout is not None else None,
                metadata_bytes=padded_metadata,
                tensor_core_padding_bytes=(
                    int(padded_runtime) - int(exact_runtime)
                    if padded_runtime is not None and exact_runtime is not None
                    else None
                ),
                active_macs=dense_elements,
                executed_macs=padded_elements,
                stored_in_features=padded_in,
                stored_out_features=padded_out,
                details={
                    "padding_multiple": multiple,
                    "scale_layout": (
                        asdict(scale_layout) if scale_layout is not None else None
                    ),
                    "zero_padding_required": True,
                    "dynamic_padding_cost_included": False,
                    "runtime_kernel_support_not_inferred": True,
                },
            )
        )
    return candidates


def _indexed_candidates(
    request: ProjectionFormatRequest, projection_fingerprint: str
) -> list[DeploymentFormatCandidate]:
    active = int(request.active_contacts)
    result = []
    for format_id, value_dtype, index_dtype, index_bytes_per_contact in (
        ("indexed_fp32_int32", "fp32", "int32", 4),
        ("indexed_bf16_int32", "bf16", "int32", 4),
        ("indexed_bf16_uint16", "bf16", "uint16", 2),
    ):
        cardinality = 1 << (31 if index_dtype == "int32" else 16)
        reasons = _uniform_index_reasons(request, index_cardinality=cardinality)
        value_bytes = active * _VALUE_BYTES[value_dtype]
        index_bytes = active * index_bytes_per_contact
        result.append(
            _candidate(
                projection_fingerprint=projection_fingerprint,
                format_id=format_id,
                family="indexed",
                request=request,
                reasons=reasons,
                value_dtype=value_dtype,
                index_dtype=index_dtype,
                stored_value_elements=active,
                value_bytes=value_bytes,
                index_bytes=index_bytes,
                metadata_bytes=0,
                tensor_core_padding_bytes=0,
                active_macs=active,
                executed_macs=active,
                stored_in_features=int(request.in_features),
                stored_out_features=int(request.out_features),
                details={
                    "uniform_ell_encoding": True,
                    "contacts_per_output": request.contacts_per_output,
                    "runtime_kernel_support_not_inferred": True,
                },
            )
        )

    layout = request.block_local_uint16
    reasons = []
    if not request.fixed_topology:
        reasons.append("fixed_topology_required")
    if layout is None:
        reasons.append("verified_block_local_layout_required")
        metadata_bytes = None
        details: dict[str, Any] = {"layout": None}
    else:
        if not layout.topology_verified:
            reasons.append("block_local_topology_not_verified")
        if int(layout.block_size) > (1 << 16):
            reasons.append("block_size_exceeds_uint16_cardinality")
        input_blocks = _ceil_div(request.in_features, layout.block_size)
        if input_blocks > (1 << (8 * int(layout.block_id_bytes))):
            reasons.append("input_block_count_exceeds_block_id_cardinality")
        if int(layout.group_count) > active:
            reasons.append("group_count_exceeds_active_contacts")
        metadata_bytes = int(
            layout.group_count * layout.block_id_bytes
            + (request.out_features + 1) * layout.pointer_bytes
            + (layout.group_count + 1) * layout.pointer_bytes
        )
        details = {
            "layout": asdict(layout),
            "input_blocks": int(input_blocks),
            "encoding": (
                "uint16 local contact indices + group block ids + "
                "output-to-group and group-to-contact pointers"
            ),
        }
    result.append(
        _candidate(
            projection_fingerprint=projection_fingerprint,
            format_id="indexed_bf16_uint16_block_local",
            family="indexed_block_local",
            request=request,
            reasons=reasons,
            value_dtype="bf16",
            index_dtype="uint16_local",
            stored_value_elements=active,
            value_bytes=active * _VALUE_BYTES["bf16"],
            index_bytes=active * 2,
            metadata_bytes=metadata_bytes,
            tensor_core_padding_bytes=0,
            active_macs=active,
            executed_macs=active,
            stored_in_features=int(request.in_features),
            stored_out_features=int(request.out_features),
            details=details,
        )
    )
    return result


def _structured_candidate(
    request: ProjectionFormatRequest, projection_fingerprint: str
) -> DeploymentFormatCandidate:
    layout = request.structured_nm
    reasons = []
    if not request.fixed_topology:
        reasons.append("fixed_topology_required")
    if layout is None:
        reasons.append("verified_structured_nm_layout_required")
        return _candidate(
            projection_fingerprint=projection_fingerprint,
            format_id="structured_nm",
            family="structured_sparse",
            request=request,
            reasons=reasons,
            value_dtype="bf16",
            index_dtype="packed_nm_metadata",
            stored_value_elements=None,
            value_bytes=None,
            index_bytes=None,
            metadata_bytes=None,
            tensor_core_padding_bytes=0,
            active_macs=None,
            executed_macs=None,
            stored_in_features=int(request.in_features),
            stored_out_features=int(request.out_features),
            details={"layout": None},
        )
    if not layout.topology_verified:
        reasons.append("structured_nm_topology_not_verified")
    if request.in_features % layout.group_size:
        reasons.append("input_dimension_not_divisible_by_nm_group")
        groups_per_output = None
        encoded_active = None
    else:
        groups_per_output = request.in_features // layout.group_size
        encoded_active = request.out_features * groups_per_output * layout.nonzero
        if encoded_active != request.active_contacts:
            reasons.append("structured_nm_active_contact_mismatch")
    if groups_per_output is None:
        metadata_bytes = None
        value_bytes = None
    else:
        row_metadata_unaligned = _ceil_div(
            groups_per_output * layout.metadata_bits_per_group, 8
        )
        row_metadata_bytes = _align_up(
            row_metadata_unaligned, layout.metadata_alignment_bytes
        )
        metadata_bytes = int(request.out_features * row_metadata_bytes)
        value_bytes = int(encoded_active * _VALUE_BYTES[layout.value_dtype])
    return _candidate(
        projection_fingerprint=projection_fingerprint,
        format_id="structured_nm",
        family="structured_sparse",
        request=request,
        reasons=reasons,
        value_dtype=layout.value_dtype,
        index_dtype="packed_nm_metadata",
        stored_value_elements=encoded_active,
        value_bytes=value_bytes,
        index_bytes=0 if metadata_bytes is not None else None,
        metadata_bytes=metadata_bytes,
        tensor_core_padding_bytes=0,
        active_macs=encoded_active,
        executed_macs=encoded_active,
        stored_in_features=int(request.in_features),
        stored_out_features=int(request.out_features),
        details={
            "layout": asdict(layout),
            "groups_per_output": groups_per_output,
            "runtime_kernel_support_not_inferred": True,
        },
    )


def _block_sparse_candidate(
    request: ProjectionFormatRequest, projection_fingerprint: str
) -> DeploymentFormatCandidate:
    layout = request.block_sparse
    reasons = []
    if not request.fixed_topology:
        reasons.append("fixed_topology_required")
    if layout is None:
        reasons.append("verified_block_sparse_layout_required")
        return _candidate(
            projection_fingerprint=projection_fingerprint,
            format_id="block_sparse",
            family="block_sparse",
            request=request,
            reasons=reasons,
            value_dtype="bf16",
            index_dtype="block_csr",
            stored_value_elements=None,
            value_bytes=None,
            index_bytes=None,
            metadata_bytes=None,
            tensor_core_padding_bytes=0,
            active_macs=None,
            executed_macs=None,
            stored_in_features=int(request.in_features),
            stored_out_features=int(request.out_features),
            details={"layout": None},
        )
    if not layout.topology_verified:
        reasons.append("block_sparse_topology_not_verified")
    if request.in_features % layout.block_cols:
        reasons.append("input_dimension_not_divisible_by_block_cols")
    if request.out_features % layout.block_rows:
        reasons.append("output_dimension_not_divisible_by_block_rows")
    total_block_rows = _ceil_div(request.out_features, layout.block_rows)
    total_block_cols = _ceil_div(request.in_features, layout.block_cols)
    total_blocks = total_block_rows * total_block_cols
    if layout.active_blocks > total_blocks:
        reasons.append("active_blocks_exceed_dense_block_grid")
    encoded_active = layout.active_blocks * layout.block_rows * layout.block_cols
    if encoded_active != request.active_contacts:
        reasons.append("block_sparse_active_contact_mismatch")
    if total_block_cols > (1 << (8 * layout.column_index_bytes)):
        reasons.append("block_column_count_exceeds_index_cardinality")
    value_bytes = encoded_active * _VALUE_BYTES[layout.value_dtype]
    metadata_bytes = (
        layout.active_blocks * layout.column_index_bytes
        + (total_block_rows + 1) * layout.row_pointer_bytes
    )
    return _candidate(
        projection_fingerprint=projection_fingerprint,
        format_id="block_sparse",
        family="block_sparse",
        request=request,
        reasons=reasons,
        value_dtype=layout.value_dtype,
        index_dtype="block_csr",
        stored_value_elements=encoded_active,
        value_bytes=value_bytes,
        index_bytes=0,
        metadata_bytes=metadata_bytes,
        tensor_core_padding_bytes=0,
        active_macs=encoded_active,
        executed_macs=encoded_active,
        stored_in_features=int(request.in_features),
        stored_out_features=int(request.out_features),
        details={
            "layout": asdict(layout),
            "total_block_rows": int(total_block_rows),
            "total_block_cols": int(total_block_cols),
            "runtime_kernel_support_not_inferred": True,
        },
    )


def _resource_pareto(
    candidates: Sequence[DeploymentFormatCandidate],
) -> tuple[str, ...]:
    """Pareto set in bytes and executed MACs only.

    Precision, quality, numerical error, latency, workspace, and kernel
    availability are intentionally absent, so this set is not a deployment
    recommendation.
    """

    legal = [candidate for candidate in candidates if candidate.legal]
    frontier = []
    for candidate in legal:
        assert candidate.runtime_bytes is not None
        assert candidate.executed_macs_per_input_row is not None
        dominated = False
        for other in legal:
            if other is candidate:
                continue
            assert other.runtime_bytes is not None
            assert other.executed_macs_per_input_row is not None
            no_worse = (
                other.runtime_bytes <= candidate.runtime_bytes
                and other.executed_macs_per_input_row
                <= candidate.executed_macs_per_input_row
            )
            strictly_better = (
                other.runtime_bytes < candidate.runtime_bytes
                or other.executed_macs_per_input_row
                < candidate.executed_macs_per_input_row
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate.format_id)
    return tuple(sorted(frontier))


def _selection_record(
    *,
    request: ProjectionFormatRequest,
    projection_fingerprint: str,
    candidates: Sequence[DeploymentFormatCandidate],
    receipts: Sequence[MeasuredLatencyReceipt],
    selection_candidate_ids: Sequence[str] | None,
) -> dict[str, Any]:
    legal = {
        candidate.format_id: candidate for candidate in candidates if candidate.legal
    }
    illegal = {
        candidate.format_id: candidate
        for candidate in candidates
        if not candidate.legal
    }
    if selection_candidate_ids is None:
        scope = tuple(sorted(legal))
        scope_source = "all_legal_candidates"
    else:
        scope = tuple(str(value) for value in selection_candidate_ids)
        scope_source = "explicit_selection_candidate_ids"
        if len(set(scope)) != len(scope):
            raise ValueError("selection_candidate_ids must be unique")
        unknown = sorted(set(scope) - set(legal) - set(illegal))
        if unknown:
            raise ValueError(f"unknown selection candidates: {unknown}")
        illegal_scope = sorted(set(scope) & set(illegal))
        if illegal_scope:
            details = {
                key: list(illegal[key].illegality_reasons) for key in illegal_scope
            }
            raise ValueError(f"selection scope contains illegal candidates: {details}")
    base = {
        "criterion": "minimum paired measured latency_ms",
        "scope_source": scope_source,
        "selection_candidate_ids": list(scope),
        "excluded_legal_candidate_ids": sorted(set(legal) - set(scope)),
        "selected_candidate_id": None,
        "receipt_schema": RECEIPT_SCHEMA,
        "boundary": (
            "No speedup or cross-device claim is inferred. Selection, when "
            "present, is only the minimum observed latency in the exact "
            "complete paired comparison scope."
        ),
    }
    if not receipts:
        return {**base, "status": "not_selected_no_measured_latency_receipts"}
    if len(scope) < 2:
        return {**base, "status": "not_selected_scope_requires_two_candidates"}

    receipt_by_candidate: dict[str, MeasuredLatencyReceipt] = {}
    for receipt in receipts:
        if receipt.candidate_id in receipt_by_candidate:
            raise ValueError(f"duplicate latency receipt for {receipt.candidate_id!r}")
        if receipt.candidate_id not in legal:
            raise ValueError(
                f"latency receipt targets a non-legal candidate: "
                f"{receipt.candidate_id!r}"
            )
        candidate = legal[receipt.candidate_id]
        if receipt.projection_fingerprint != projection_fingerprint:
            raise ValueError(
                f"projection fingerprint mismatch for {receipt.candidate_id!r}"
            )
        if receipt.candidate_fingerprint != candidate.candidate_fingerprint:
            raise ValueError(
                f"candidate fingerprint mismatch for {receipt.candidate_id!r}"
            )
        receipt_by_candidate[receipt.candidate_id] = receipt

    if set(receipt_by_candidate) != set(scope):
        return {
            **base,
            "status": "not_selected_incomplete_or_extra_receipt_scope",
            "missing_receipt_candidate_ids": sorted(
                set(scope) - set(receipt_by_candidate)
            ),
            "extra_receipt_candidate_ids": sorted(
                set(receipt_by_candidate) - set(scope)
            ),
        }
    if request.topology_sha256 is None and any(
        legal[format_id].family != "dense" for format_id in scope
    ):
        return {
            **base,
            "status": "not_selected_sparse_topology_identity_required",
        }
    failed_numerical_gates = sorted(
        candidate_id
        for candidate_id, receipt in receipt_by_candidate.items()
        if not receipt.numerical_gate_passed
    )
    if failed_numerical_gates:
        return {
            **base,
            "status": "not_selected_numerical_gate_failed",
            "failed_numerical_gate_candidate_ids": failed_numerical_gates,
        }
    numerical_gate_definitions = {
        (
            receipt.numerical_gate_id,
            receipt.atol,
            receipt.rtol,
            receipt.quality_criterion,
        )
        for receipt in receipt_by_candidate.values()
    }
    if len(numerical_gate_definitions) != 1:
        raise ValueError(
            "selection receipts must share one numerical gate id and the same "
            "parity tolerances or documented quality criterion"
        )
    comparison_fields = {
        (
            receipt.comparison_id,
            receipt.operation,
            receipt.device,
            receipt.code_commit,
            receipt.artifact_sha256,
        )
        for receipt in receipt_by_candidate.values()
    }
    if len(comparison_fields) != 1:
        raise ValueError(
            "selection receipts must share comparison_id, operation, device, "
            "code_commit, and benchmark artifact"
        )
    selected = min(
        scope,
        key=lambda format_id: (
            receipt_by_candidate[format_id].latency_ms,
            legal[format_id].runtime_bytes,
            legal[format_id].executed_macs_per_input_row,
            format_id,
        ),
    )
    receipt = receipt_by_candidate[selected]
    return {
        **base,
        "status": "selected_from_complete_paired_measured_scope",
        "selected_candidate_id": selected,
        "selected_observed_latency_ms": float(receipt.latency_ms),
        "comparison_id": receipt.comparison_id,
        "operation": receipt.operation,
        "device": receipt.device,
        "code_commit": receipt.code_commit,
        "artifact_sha256": receipt.artifact_sha256,
        "numerical_gate": {
            "id": receipt.numerical_gate_id,
            "passed_for_all_candidates": True,
            "atol": receipt.atol,
            "rtol": receipt.rtol,
            "quality_criterion": receipt.quality_criterion,
        },
        "sample_counts": {
            key: int(value.sample_count)
            for key, value in sorted(receipt_by_candidate.items())
        },
    }


def plan_projection_formats(
    request: ProjectionFormatRequest,
    *,
    latency_receipts: Sequence[MeasuredLatencyReceipt] = (),
    selection_candidate_ids: Sequence[str] | None = None,
) -> DeploymentFormatPlan:
    """Enumerate legal formats and optionally select from paired measurements.

    The function is pure: it does not inspect or mutate a model, load a kernel,
    or read receipt artifacts.  Callers that later materialize a selected
    format must separately verify the cited artifact and implement that exact
    candidate fingerprint.
    """

    if not isinstance(request, ProjectionFormatRequest):
        raise TypeError("request must be a ProjectionFormatRequest")
    request_payload = request.as_dict()
    projection_fingerprint = _canonical_sha256(
        {"schema": PLAN_SCHEMA, "request": request_payload}
    )
    candidates = [
        *_dense_candidates(request, projection_fingerprint),
        *_indexed_candidates(request, projection_fingerprint),
        _structured_candidate(request, projection_fingerprint),
        _block_sparse_candidate(request, projection_fingerprint),
    ]
    ids = [candidate.format_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise AssertionError("planner emitted duplicate candidate identifiers")
    selection = _selection_record(
        request=request,
        projection_fingerprint=projection_fingerprint,
        candidates=candidates,
        receipts=latency_receipts,
        selection_candidate_ids=selection_candidate_ids,
    )
    return DeploymentFormatPlan(
        schema=PLAN_SCHEMA,
        request=request_payload,
        projection_fingerprint=projection_fingerprint,
        candidates=tuple(candidates),
        resource_pareto_candidate_ids=_resource_pareto(candidates),
        selection=selection,
        claim_boundary={
            "mutation": "planner only; no model or topology is altered",
            "storage": (
                "exact persistent runtime projection values, indices, and "
                "declared encoding metadata; excludes activations, workspace, "
                "allocator overhead, and serialized-container overhead"
            ),
            "macs": (
                "analytical multiply-accumulates per input row; zero-padding "
                "is included in executed MACs but not active MACs"
            ),
            "pareto": (
                "resource-only bytes/MAC frontier; ignores numerical quality, "
                "kernel availability, workspace, latency, and energy"
            ),
            "latency": (
                "never estimated; an optional selection requires a complete "
                "explicit scope of fingerprint-matched paired measurements"
            ),
        },
    )
