"""Exploratory, group-disjoint diagnostics for cached collapsed-span targets.

This module deliberately consumes the immutable capture format written by
``save_collapsed_span_captures``.  Candidate surrogates are fit on one set of
content-addressed source groups and evaluated on a second set.  The third,
fingerprint-profile partition is identified and hashed but its tensor values
are never indexed for fitting or metrics.  Consequently these reports can be used to choose a
surrogate for a *future* prospectively frozen campaign, but are not themselves
evidence for an FMI prescription.

The alternatives here avoid the quadratic-in-row dual system used by the
original random-feature kernel surrogate:

* a no-input target-mean control;
* reduced-rank primal ridge with a fixed input projection;
* fixed-budget compact GELU and SwiGLU probes.

All methods expose explicit learned, active, and stored-value accounting.  The
compact probes use a train-only output basis, so the cost of a wide teacher
hidden state can be controlled independently from the hidden width.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.analysis.fmi.collapsed_span import (
    CapturedCollapsedSpanBoundary,
    CollapsedSpanSpec,
    _group_disjoint_indices,
    tensor_content_sha256,
)

CAPTURE_SCHEMA = "dendritic_collapsed_span_fmi_capture/v1"
REPORT_SCHEMA = "dendritic_collapsed_span_surrogate_zoo/v1"
EVIDENCE_STATUS = "exploratory_non_evidentiary"
CLAIM_BOUNDARY = (
    "Post hoc surrogate diagnostic. Candidate methods are compared on the "
    "gate-validation partition, so neither the selected method nor this report "
    "is evidentiary. Fingerprint-profile values are used only for immutable "
    "capture hashing, never fitting or metrics; a method and all hyperparameters "
    "must be frozen before a prospective run."
)


def canonical_json_sha256(value: Any) -> str:
    """Return a stable digest for a finite JSON value."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without reading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CachedSpanCapture:
    """A fully verified immutable capture and its file identity."""

    boundary: CapturedCollapsedSpanBoundary
    path: str
    file_sha256: str
    file_bytes: int


def _require_tensor(
    payload: Mapping[str, Any], name: str, *, ndim: int
) -> torch.Tensor:
    value = payload.get(name)
    if not torch.is_tensor(value) or value.ndim != ndim:
        raise TypeError(f"capture field {name!r} must be a {ndim}-D tensor")
    return value.detach().to(device="cpu").contiguous()


def load_cached_span_capture(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_semantic_sha256: str | None = None,
) -> CachedSpanCapture:
    """Load a capture only after file, tensor, and semantic identities agree.

    ``expected_file_sha256`` is mandatory so a mutable path cannot silently
    select different calibration tensors.  The capture's own metadata is then
    independently reconstructed and checked, including every tensor digest and
    the teacher-minus-zero-student target identity.
    """

    capture_path = Path(path).resolve()
    if not capture_path.is_file():
        raise FileNotFoundError(capture_path)
    observed_file_sha256 = file_sha256(capture_path)
    if observed_file_sha256 != str(expected_file_sha256).lower():
        raise ValueError(
            "capture file SHA-256 mismatch: "
            f"expected {expected_file_sha256}, observed {observed_file_sha256}"
        )
    try:
        payload = torch.load(capture_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with old torch only
        payload = torch.load(capture_path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != CAPTURE_SCHEMA:
        raise ValueError(f"capture must use schema {CAPTURE_SCHEMA!r}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("capture metadata must be a mapping")
    if metadata.get("schema") != "dendritic_collapsed_span_fmi_boundary/v1":
        raise ValueError("capture boundary metadata schema is unsupported")
    span_layers = metadata.get("span_layers")
    if not isinstance(span_layers, Sequence) or isinstance(span_layers, (str, bytes)):
        raise TypeError("capture span_layers must be a sequence")
    source_hashes_raw = payload.get("source_group_sha256")
    if not isinstance(source_hashes_raw, Mapping):
        raise TypeError("capture source_group_sha256 must be a mapping")
    boundary = CapturedCollapsedSpanBoundary(
        spec=CollapsedSpanSpec(
            tuple(int(layer) for layer in span_layers),
            mlp_attr=str(metadata.get("mlp_attr", "mlp")),
        ),
        inputs=_require_tensor(payload, "inputs", ndim=2),
        targets=_require_tensor(payload, "targets", ndim=2),
        dense_teacher_exits=_require_tensor(payload, "dense_teacher_exits", ndim=2),
        zero_student_exits=_require_tensor(payload, "zero_student_exits", ndim=2),
        row_group_ids=_require_tensor(payload, "row_group_ids", ndim=1).to(
            dtype=torch.int64
        ),
        source_group_sha256={
            int(group): str(digest) for group, digest in source_hashes_raw.items()
        },
        rows_seen=int(metadata["rows_seen"]),
        context=dict(metadata.get("student_context", {})),
        calibration_sha256=str(metadata["calibration_sha256"]),
    )
    recorded_tensor_hashes = metadata.get("tensor_sha256")
    if not isinstance(recorded_tensor_hashes, Mapping):
        raise TypeError("capture metadata lacks tensor_sha256")
    tensors = {
        "inputs": boundary.inputs,
        "targets": boundary.targets,
        "dense_teacher_exits": boundary.dense_teacher_exits,
        "zero_student_exits": boundary.zero_student_exits,
        "row_group_ids": boundary.row_group_ids,
    }
    for name, tensor in tensors.items():
        observed = tensor_content_sha256(tensor)
        if recorded_tensor_hashes.get(name) != observed:
            raise ValueError(f"capture tensor hash mismatch for {name!r}")
    recorded_semantic = str(metadata.get("content_sha256", ""))
    if boundary.content_sha256 != recorded_semantic:
        raise ValueError("capture semantic content hash does not reconstruct")
    if (
        expected_semantic_sha256 is not None
        and recorded_semantic != str(expected_semantic_sha256).lower()
    ):
        raise ValueError("capture semantic SHA-256 does not match expectation")
    return CachedSpanCapture(
        boundary=boundary,
        path=str(capture_path),
        file_sha256=observed_file_sha256,
        file_bytes=int(capture_path.stat().st_size),
    )


@dataclass(frozen=True)
class SurrogateSplitConfig:
    """The exact registered three-way group split controls."""

    validation_group_fraction: float = 0.25
    profile_group_fraction: float = 0.25
    minimum_groups_per_partition: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.validation_group_fraction) < 1.0:
            raise ValueError("validation_group_fraction must lie in (0, 1)")
        if not 0.0 < float(self.profile_group_fraction) < 1.0:
            raise ValueError("profile_group_fraction must lie in (0, 1)")
        if (
            float(self.validation_group_fraction) + float(self.profile_group_fraction)
            >= 1.0
        ):
            raise ValueError("validation and profile fractions must sum below one")
        if int(self.minimum_groups_per_partition) < 2:
            raise ValueError("minimum_groups_per_partition must be at least two")


@dataclass(frozen=True)
class RidgeProbeConfig:
    """Reduced-rank primal ridge controls."""

    input_features: int = 256
    output_rank: int = 128
    ridge: float = 1.0e-3
    seed: int = 17

    def __post_init__(self) -> None:
        if int(self.input_features) < 1:
            raise ValueError("input_features must be positive")
        if int(self.output_rank) < 0:
            raise ValueError("output_rank must be nonnegative")
        if not math.isfinite(float(self.ridge)) or float(self.ridge) <= 0.0:
            raise ValueError("ridge must be positive and finite")


@dataclass(frozen=True)
class MLPProbeConfig:
    """Fixed-training-budget compact nonlinear probe controls."""

    hidden_features: int = 256
    output_rank: int = 128
    steps: int = 500
    batch_size: int = 256
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    seed: int = 29
    amp_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        for name, value in (
            ("hidden_features", self.hidden_features),
            ("steps", self.steps),
            ("batch_size", self.batch_size),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if int(self.output_rank) < 0:
            raise ValueError("output_rank must be nonnegative")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(float(self.weight_decay)) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")
        if self.amp_dtype not in {"none", "bfloat16", "float16"}:
            raise ValueError("amp_dtype must be none, bfloat16, or float16")


@dataclass(frozen=True)
class SurrogateZooConfig:
    """Complete diagnostic configuration."""

    methods: tuple[str, ...] = (
        "target_mean",
        "reduced_rank_primal_ridge",
        "gelu_mlp",
        "swiglu_mlp",
    )
    split: SurrogateSplitConfig = SurrogateSplitConfig()
    ridge: RidgeProbeConfig = RidgeProbeConfig()
    mlp: MLPProbeConfig = MLPProbeConfig()
    evaluation_batch_size: int = 256

    def __post_init__(self) -> None:
        supported = {
            "target_mean",
            "reduced_rank_primal_ridge",
            "gelu_mlp",
            "swiglu_mlp",
        }
        methods = tuple(dict.fromkeys(str(method) for method in self.methods))
        if not methods or any(method not in supported for method in methods):
            raise ValueError(
                f"methods must be a nonempty subset of {sorted(supported)}"
            )
        if int(self.evaluation_batch_size) < 1:
            raise ValueError("evaluation_batch_size must be positive")
        object.__setattr__(self, "methods", methods)


@dataclass(frozen=True)
class SurrogatePartitions:
    """Exact row and group identities for the three registered roles."""

    fit_index: torch.Tensor
    gate_index: torch.Tensor
    profile_index: torch.Tensor
    fit_groups: torch.Tensor
    gate_groups: torch.Tensor
    profile_groups: torch.Tensor


def reproduce_registered_partitions(
    boundary: CapturedCollapsedSpanBoundary,
    config: SurrogateSplitConfig,
) -> SurrogatePartitions:
    """Call the production split routine and verify content-level disjointness."""

    source_hashes = list(boundary.source_group_sha256.values())
    if len(source_hashes) != len(set(source_hashes)):
        raise ValueError(
            "distinct source_group_ids share a content hash; a group-id split "
            "would not be content-disjoint"
        )
    values = _group_disjoint_indices(
        boundary.row_group_ids,
        validation_fraction=config.validation_group_fraction,
        profile_fraction=config.profile_group_fraction,
        minimum_groups_per_partition=config.minimum_groups_per_partition,
        seed=config.seed,
    )
    partitions = SurrogatePartitions(*values)
    group_sets = [
        set(partitions.fit_groups.tolist()),
        set(partitions.gate_groups.tolist()),
        set(partitions.profile_groups.tolist()),
    ]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("registered group split unexpectedly overlaps")
    return partitions


def _output_basis(
    centered_targets: torch.Tensor, *, rank: int, seed: int
) -> torch.Tensor | None:
    maximum = min(int(centered_targets.shape[0]), int(centered_targets.shape[1]))
    requested = int(rank)
    if requested == 0 or requested >= int(centered_targets.shape[1]):
        return None
    effective = min(requested, maximum)
    devices = [centered_targets.device.index or 0] if centered_targets.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        _, _, basis = torch.pca_lowrank(
            centered_targets,
            q=effective,
            center=False,
            niter=2,
        )
    return basis[:, :effective].contiguous()


class TargetMeanSurrogate(nn.Module):
    """No-input control fitted only on the fit partition."""

    def __init__(self, target_mean: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("target_mean", target_mean.detach().clone())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.target_mean.expand(int(inputs.shape[0]), -1)


class ReducedRankPrimalRidge(nn.Module):
    """Primal ridge using fixed input features and an optional output basis."""

    def __init__(
        self,
        *,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        input_projection: torch.Tensor | None,
        target_mean: torch.Tensor,
        coefficients: torch.Tensor,
        output_basis: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_scale", input_scale)
        self.register_buffer("input_projection", input_projection)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("output_basis", output_basis)

    def _features(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = (inputs.float() - self.input_mean) / self.input_scale
        if self.input_projection is not None:
            normalized = normalized @ self.input_projection
        return normalized

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        correction = self._features(inputs) @ self.coefficients
        if self.output_basis is not None:
            correction = correction @ self.output_basis.T
        return self.target_mean + correction


class CompactMLPSurrogate(nn.Module):
    """Compact GELU or SwiGLU correction model with a train-only output basis."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        mechanism: str,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        target_mean: torch.Tensor,
        output_basis: torch.Tensor | None,
    ) -> None:
        super().__init__()
        if mechanism not in {"gelu", "swiglu"}:
            raise ValueError("mechanism must be gelu or swiglu")
        self.mechanism = mechanism
        latent_dim = (
            int(output_basis.shape[1]) if output_basis is not None else int(output_dim)
        )
        self.input_projection = nn.Linear(int(input_dim), int(hidden_dim))
        self.gate_projection = (
            nn.Linear(int(input_dim), int(hidden_dim))
            if mechanism == "swiglu"
            else None
        )
        self.output_projection = nn.Linear(int(hidden_dim), latent_dim)
        self.register_buffer("input_mean", input_mean)
        self.register_buffer("input_scale", input_scale)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("output_basis", output_basis)

    def latent(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = (inputs.float() - self.input_mean) / self.input_scale
        values = self.input_projection(normalized)
        if self.mechanism == "gelu":
            hidden = F.gelu(values)
        else:
            assert self.gate_projection is not None
            hidden = values * F.silu(self.gate_projection(normalized))
        return self.output_projection(hidden)

    def target_latent(self, targets: torch.Tensor) -> torch.Tensor:
        centered = targets.float() - self.target_mean
        if self.output_basis is not None:
            return centered @ self.output_basis
        return centered

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        correction = self.latent(inputs)
        if self.output_basis is not None:
            correction = correction @ self.output_basis.T
        return self.target_mean + correction


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _state_accounting(
    model: nn.Module,
    *,
    learned_parameter_values: int,
    inference_macs_per_row: int,
) -> dict[str, Any]:
    state = model.state_dict()
    stored_values = sum(int(value.numel()) for value in state.values())
    stored_bytes = sum(
        int(value.numel()) * int(value.element_size()) for value in state.values()
    )
    trainable_parameters = sum(int(value.numel()) for value in model.parameters())
    return {
        "active_parameter_values": int(stored_values),
        "trainable_parameter_values": int(trainable_parameters),
        "learned_parameter_values": int(learned_parameter_values),
        "stored_values": int(stored_values),
        "stored_bytes": int(stored_bytes),
        "inference_macs_per_row": int(inference_macs_per_row),
        "definitions": {
            "active_parameter_values": (
                "persistent model-state scalars consulted by one inference, "
                "including fixed projections and fitted buffers"
            ),
            "trainable_parameter_values": "nn.Parameter scalars in the fitted module",
            "learned_parameter_values": (
                "fitted scalars including fitted bases and offsets, whether stored "
                "as parameters or buffers"
            ),
            "stored_values": "all persistent state_dict tensor scalars",
            "stored_bytes": "exact in-memory dtype bytes of persistent tensors",
            "inference_macs_per_row": "theoretical dense multiply-accumulates",
        },
    }


def _evaluate_metrics(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_mean: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    count = int(inputs.shape[0])
    output_dim = int(targets.shape[1])
    squared_error = 0.0
    baseline_error = 0.0
    target_square = 0.0
    prediction_square = 0.0
    prediction_target = 0.0
    prediction_sum = torch.zeros(output_dim, dtype=torch.float64)
    target_sum = torch.zeros(output_dim, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, count, int(batch_size)):
            stop = min(count, start + int(batch_size))
            x = inputs[start:stop].to(device=device, dtype=torch.float32)
            y = targets[start:stop].to(device=device, dtype=torch.float32)
            prediction = model(x).float()
            mean = target_mean.to(device=device, dtype=torch.float32)
            squared_error += float((prediction - y).square().sum())
            baseline_error += float((y - mean).square().sum())
            target_square += float(y.square().sum())
            prediction_square += float(prediction.square().sum())
            prediction_target += float((prediction * y).sum())
            prediction_sum += prediction.sum(dim=0).double().cpu()
            target_sum += y.sum(dim=0).double().cpu()
    baseline_error = max(baseline_error, 1.0e-12)
    target_square = max(target_square, 1.0e-12)
    centered_dot = prediction_target - float(prediction_sum @ target_sum) / count
    centered_prediction_square = (
        prediction_square - float(prediction_sum @ prediction_sum) / count
    )
    centered_target_square = target_square - float(target_sum @ target_sum) / count
    # The running square sums originate in float32 model outputs.  Direct
    # subtraction can leave a tiny spurious centered norm for a mathematically
    # constant predictor (notably the target-mean control), which previously
    # produced unbounded cosine values.  Treat variance below the accumulated
    # float32 roundoff scale as the degenerate zero-variance case.
    prediction_roundoff = (
        16.0 * torch.finfo(torch.float32).eps * max(prediction_square, 1.0)
    )
    target_roundoff = 16.0 * torch.finfo(torch.float32).eps * max(target_square, 1.0)
    centered_prediction_square = max(centered_prediction_square, 0.0)
    centered_target_square = max(centered_target_square, 0.0)
    cosine_defined = (
        centered_prediction_square > prediction_roundoff
        and centered_target_square > target_roundoff
    )
    centered_denominator = math.sqrt(
        centered_prediction_square * centered_target_square
    )
    return {
        "r2_against_fit_target_mean": float(1.0 - squared_error / baseline_error),
        "normalized_rmse_target_rms": float(
            math.sqrt(squared_error / (count * output_dim))
            / math.sqrt(target_square / (count * output_dim))
        ),
        "centered_cosine": (
            float(centered_dot / centered_denominator) if cosine_defined else 0.0
        ),
        "rmse": float(math.sqrt(squared_error / (count * output_dim))),
    }


def _output_basis_oracle_r2(
    model: nn.Module,
    targets: torch.Tensor,
    *,
    target_mean: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> float:
    """Best gate R2 possible when only the fitted output basis is limiting."""

    output_basis = getattr(model, "output_basis", None)
    if output_basis is None:
        return 1.0
    residual_square = 0.0
    baseline_square = 0.0
    basis = output_basis.to(device=device, dtype=torch.float32)
    mean = target_mean.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, int(targets.shape[0]), int(batch_size)):
            stop = min(int(targets.shape[0]), start + int(batch_size))
            centered = targets[start:stop].to(device=device, dtype=torch.float32) - mean
            projected = (centered @ basis) @ basis.T
            residual_square += float((centered - projected).square().sum())
            baseline_square += float(centered.square().sum())
    return float(1.0 - residual_square / max(baseline_square, 1.0e-12))


def _summarize_group_metrics(
    values: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    if len(values) < 2:
        raise ValueError("source-group summaries require at least two groups")
    result: dict[str, Any] = {"group_count": len(values)}
    for metric in values[0]:
        tensor = torch.tensor(
            [float(entry[metric]) for entry in values], dtype=torch.float64
        )
        mean = float(tensor.mean())
        standard_error = float(tensor.std(unbiased=True) / math.sqrt(len(values)))
        result[metric] = {
            "macro_mean": mean,
            "macro_median": float(tensor.median()),
            "macro_standard_error": standard_error,
            "normal_95_ci": [
                float(mean - 1.96 * standard_error),
                float(mean + 1.96 * standard_error),
            ],
            "minimum": float(tensor.min()),
            "maximum": float(tensor.max()),
        }
    result["interval_note"] = (
        "Descriptive normal intervals over source-group macro metrics; not a "
        "prospectively registered inferential test."
    )
    return result


def _evaluate(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    target_mean: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    if group_ids.dtype != torch.int64 or tuple(group_ids.shape) != (
        int(inputs.shape[0]),
    ):
        raise TypeError("evaluation group_ids must be int64 with one id per row")
    aggregate = _evaluate_metrics(
        model,
        inputs,
        targets,
        target_mean=target_mean,
        device=device,
        batch_size=batch_size,
    )
    group_values = []
    for group in torch.unique(group_ids, sorted=True):
        index = torch.nonzero(group_ids == group, as_tuple=False).reshape(-1)
        group_values.append(
            _evaluate_metrics(
                model,
                inputs[index],
                targets[index],
                target_mean=target_mean,
                device=device,
                batch_size=batch_size,
            )
        )
    return aggregate, _summarize_group_metrics(group_values)


def _fixed_input_projection(
    input_dim: int, features: int, *, seed: int, device: torch.device
) -> torch.Tensor | None:
    if int(features) >= int(input_dim):
        return None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    projection = torch.randn(
        int(input_dim), int(features), generator=generator, dtype=torch.float32
    ) / math.sqrt(int(features))
    return projection.to(device=device)


def _fit_ridge(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    config: RidgeProbeConfig,
    device: torch.device,
) -> tuple[ReducedRankPrimalRidge, dict[str, Any]]:
    x = train_x.to(device=device, dtype=torch.float32)
    y = train_y.to(device=device, dtype=torch.float32)
    input_mean = x.mean(dim=0)
    input_scale = (x - input_mean).square().mean(dim=0).sqrt().clamp_min(1.0e-6)
    normalized = (x - input_mean) / input_scale
    input_projection = _fixed_input_projection(
        int(x.shape[1]), int(config.input_features), seed=config.seed, device=device
    )
    features = normalized if input_projection is None else normalized @ input_projection
    target_mean = y.mean(dim=0)
    centered_y = y - target_mean
    output_basis = _output_basis(
        centered_y, rank=config.output_rank, seed=int(config.seed) + 1
    )
    fit_target = centered_y if output_basis is None else centered_y @ output_basis
    scale = max(int(features.shape[0]), 1)
    gram = features.T @ features / scale
    ridge_scale = torch.diagonal(gram).mean().clamp_min(1.0e-12)
    regularizer = float(config.ridge) * ridge_scale
    system = gram + regularizer * torch.eye(
        int(gram.shape[0]), dtype=gram.dtype, device=device
    )
    rhs = features.T @ fit_target / scale
    factor, info = torch.linalg.cholesky_ex(system)
    if bool((info != 0).any()):
        raise RuntimeError("reduced-rank primal ridge system is not SPD")
    coefficients = torch.cholesky_solve(rhs, factor)
    model = ReducedRankPrimalRidge(
        input_mean=input_mean,
        input_scale=input_scale,
        input_projection=input_projection,
        target_mean=target_mean,
        coefficients=coefficients,
        output_basis=output_basis,
    )
    feature_dim = int(features.shape[1])
    latent_dim = int(coefficients.shape[1])
    learned = int(
        input_mean.numel()
        + input_scale.numel()
        + target_mean.numel()
        + coefficients.numel()
    )
    if output_basis is not None:
        learned += int(output_basis.numel())
    macs = feature_dim * latent_dim
    if input_projection is not None:
        macs += int(x.shape[1]) * feature_dim
    if output_basis is not None:
        macs += int(y.shape[1]) * latent_dim
    details = {
        "regularizer": float(regularizer),
        "realized_input_features": feature_dim,
        "realized_output_rank": (
            int(y.shape[1]) if output_basis is None else int(output_basis.shape[1])
        ),
        "output_basis_fit_partition_only": True,
        "accounting": _state_accounting(
            model,
            learned_parameter_values=learned,
            inference_macs_per_row=macs,
        ),
    }
    return model, details


def _amp_context(device: torch.device, dtype_name: str):
    enabled = device.type == "cuda" and dtype_name != "none"
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _fit_mlp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    mechanism: str,
    config: MLPProbeConfig,
    device: torch.device,
) -> tuple[CompactMLPSurrogate, dict[str, Any]]:
    x_for_stats = train_x.to(device=device, dtype=torch.float32)
    y_for_stats = train_y.to(device=device, dtype=torch.float32)
    input_mean = x_for_stats.mean(dim=0)
    input_scale = (
        (x_for_stats - input_mean).square().mean(dim=0).sqrt().clamp_min(1.0e-6)
    )
    target_mean = y_for_stats.mean(dim=0)
    output_basis = _output_basis(
        y_for_stats - target_mean,
        rank=config.output_rank,
        seed=int(config.seed) + (101 if mechanism == "gelu" else 211),
    )
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(config.seed) + (0 if mechanism == "gelu" else 1))
        model = CompactMLPSurrogate(
            input_dim=int(train_x.shape[1]),
            hidden_dim=int(config.hidden_features),
            output_dim=int(train_y.shape[1]),
            mechanism=mechanism,
            input_mean=input_mean,
            input_scale=input_scale,
            target_mean=target_mean,
            output_basis=output_basis,
        ).to(device)
    del x_for_stats, y_for_stats
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed) + 31_337)
    order = torch.randperm(int(train_x.shape[0]), generator=generator)
    cursor = 0
    loss_trace: list[dict[str, float | int]] = []
    model.train()
    for step in range(int(config.steps)):
        if cursor + int(config.batch_size) > int(order.numel()):
            order = torch.randperm(int(train_x.shape[0]), generator=generator)
            cursor = 0
        index = order[cursor : cursor + int(config.batch_size)]
        cursor += int(config.batch_size)
        x = train_x[index].to(device=device, dtype=torch.float32)
        y = train_y[index].to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with _amp_context(device, config.amp_dtype):
            latent = model.latent(x)
            target_latent = model.target_latent(y)
            loss = F.mse_loss(latent.float(), target_latent.float())
        loss.backward()
        optimizer.step()
        if step in {0, int(config.steps) - 1}:
            loss_trace.append({"step": int(step + 1), "fit_latent_mse": float(loss)})
    latent_dim = int(model.output_projection.out_features)
    hidden = int(config.hidden_features)
    input_dim = int(train_x.shape[1])
    output_dim = int(train_y.shape[1])
    macs = input_dim * hidden + hidden * latent_dim
    if mechanism == "swiglu":
        macs += input_dim * hidden
    if output_basis is not None:
        macs += output_dim * latent_dim
    learned = sum(int(value.numel()) for value in model.parameters())
    if output_basis is not None:
        learned += int(output_basis.numel())
    learned += int(input_mean.numel() + input_scale.numel() + target_mean.numel())
    details = {
        "mechanism": mechanism,
        "realized_output_rank": latent_dim,
        "output_basis_fit_partition_only": True,
        "fixed_step_training_no_gate_early_stopping": True,
        "loss_trace": loss_trace,
        "accounting": _state_accounting(
            model,
            learned_parameter_values=learned,
            inference_macs_per_row=macs,
        ),
    }
    return model, details


def _partition_manifest(
    boundary: CapturedCollapsedSpanBoundary,
    partitions: SurrogatePartitions,
) -> dict[str, Any]:
    def describe(index: torch.Tensor, groups: torch.Tensor) -> dict[str, Any]:
        return {
            "rows": int(index.numel()),
            "groups": int(groups.numel()),
            "group_ids_sha256": tensor_content_sha256(groups),
            "row_indices_sha256": tensor_content_sha256(index),
            "source_groups": [
                {
                    "group_id": int(group),
                    "source_sha256": boundary.source_group_sha256[int(group)],
                }
                for group in groups.tolist()
            ],
        }

    return {
        "split_implementation": (
            "dendritic_modeling.analysis.fmi.collapsed_span._group_disjoint_indices"
        ),
        "split_unit": "content_addressed_source_group",
        "group_disjoint": True,
        "fit": describe(partitions.fit_index, partitions.fit_groups),
        "gate_validation": describe(partitions.gate_index, partitions.gate_groups),
        "fingerprint_profile": {
            **describe(partitions.profile_index, partitions.profile_groups),
            "values_used_for_model_fit_or_metrics": False,
            "values_covered_by_capture_integrity_hash": True,
            "reserved_role": "future_fingerprint_profile_after_prospective_freeze",
        },
    }


def benchmark_cached_span_surrogates(
    capture: CachedSpanCapture,
    *,
    config: SurrogateZooConfig | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Fit candidates on fit groups and evaluate only fit and gate groups."""

    cfg = config or SurrogateZooConfig()
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    boundary = capture.boundary
    partitions = reproduce_registered_partitions(boundary, cfg.split)
    # Intentionally construct no profile_x/profile_y values here.  Only group and
    # row identities enter the report's sealed-partition manifest.
    fit_x = boundary.inputs[partitions.fit_index]
    fit_y = boundary.targets[partitions.fit_index]
    gate_x = boundary.inputs[partitions.gate_index]
    gate_y = boundary.targets[partitions.gate_index]
    target_mean = fit_y.float().mean(dim=0).to(target_device)
    methods: dict[str, Any] = {}
    for method in cfg.methods:
        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target_device)
        _synchronize(target_device)
        started = time.perf_counter()
        if method == "target_mean":
            model = TargetMeanSurrogate(target_mean).to(target_device)
            details = {
                "accounting": _state_accounting(
                    model,
                    learned_parameter_values=int(target_mean.numel()),
                    inference_macs_per_row=0,
                )
            }
        elif method == "reduced_rank_primal_ridge":
            model, details = _fit_ridge(
                fit_x, fit_y, config=cfg.ridge, device=target_device
            )
        elif method in {"gelu_mlp", "swiglu_mlp"}:
            model, details = _fit_mlp(
                fit_x,
                fit_y,
                mechanism="gelu" if method == "gelu_mlp" else "swiglu",
                config=cfg.mlp,
                device=target_device,
            )
        else:  # pragma: no cover - guarded by the config dataclass
            raise AssertionError(method)
        _synchronize(target_device)
        fit_seconds = float(time.perf_counter() - started)
        fit_metrics, fit_group_metrics = _evaluate(
            model,
            fit_x,
            fit_y,
            boundary.row_group_ids[partitions.fit_index],
            target_mean=target_mean,
            device=target_device,
            batch_size=cfg.evaluation_batch_size,
        )
        gate_metrics, gate_group_metrics = _evaluate(
            model,
            gate_x,
            gate_y,
            boundary.row_group_ids[partitions.gate_index],
            target_mean=target_mean,
            device=target_device,
            batch_size=cfg.evaluation_batch_size,
        )
        if method != "target_mean":
            details["gate_output_basis_oracle_r2_against_fit_target_mean"] = (
                _output_basis_oracle_r2(
                    model,
                    gate_y,
                    target_mean=target_mean,
                    device=target_device,
                    batch_size=cfg.evaluation_batch_size,
                )
            )
        peak_cuda_bytes = (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else None
        )
        methods[method] = {
            "fit_seconds": fit_seconds,
            "peak_cuda_allocated_bytes_fit_and_evaluation": peak_cuda_bytes,
            "fit_metrics": fit_metrics,
            "fit_source_group_macro_metrics": fit_group_metrics,
            "gate_validation_metrics": gate_metrics,
            "gate_validation_source_group_macro_metrics": gate_group_metrics,
            "gate_used_for_training_or_early_stopping": False,
            "fingerprint_profile_values_used_for_fit_or_metrics": False,
            **details,
        }
        del model
    report = {
        "schema": REPORT_SCHEMA,
        "evidence_status": EVIDENCE_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "selection_policy": (
            "No winner is declared. Any post hoc ranking on gate-validation is "
            "hypothesis generation only; freeze one method before new data."
        ),
        "metric_definitions": {
            "centered_cosine": (
                "Global centered cosine over rows and output coordinates; set "
                "to 0 by convention when either centered variance is below "
                "the accumulated float32 roundoff scale."
            ),
            "gate_output_basis_oracle_r2_against_fit_target_mean": (
                "Gate R2 of the orthogonal projection onto the output basis "
                "fitted only on fit groups; 1 for a full-output model. This "
                "is a representation ceiling, not a fitted surrogate score."
            ),
        },
        "capture": {
            "path": capture.path,
            "bytes": int(capture.file_bytes),
            "file_sha256": capture.file_sha256,
            "semantic_content_sha256": boundary.content_sha256,
            "calibration_sha256": boundary.calibration_sha256,
            "span_key": boundary.spec.key,
            "input_dim": int(boundary.inputs.shape[1]),
            "output_dim": int(boundary.targets.shape[1]),
            "rows": int(boundary.inputs.shape[0]),
        },
        "config": asdict(cfg),
        "device": str(target_device),
        "partitions": _partition_manifest(boundary, partitions),
        "methods": methods,
        "report_content_sha256": None,
    }
    report["report_content_sha256"] = canonical_json_sha256(
        {**report, "report_content_sha256": None}
    )
    return report


__all__ = [
    "CAPTURE_SCHEMA",
    "CLAIM_BOUNDARY",
    "EVIDENCE_STATUS",
    "REPORT_SCHEMA",
    "CachedSpanCapture",
    "CompactMLPSurrogate",
    "MLPProbeConfig",
    "ReducedRankPrimalRidge",
    "RidgeProbeConfig",
    "SurrogatePartitions",
    "SurrogateSplitConfig",
    "SurrogateZooConfig",
    "TargetMeanSurrogate",
    "benchmark_cached_span_surrogates",
    "canonical_json_sha256",
    "file_sha256",
    "load_cached_span_capture",
    "reproduce_registered_partitions",
]
