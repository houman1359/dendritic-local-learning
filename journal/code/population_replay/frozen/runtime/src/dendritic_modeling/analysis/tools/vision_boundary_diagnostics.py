"""Matched diagnostics for pretrained vision-layer replacements.

The analyzer compares a fitted replacement with the exact pretrained span on
the same encoded inputs. It reports scale, sparsity, normalized error, cosine
alignment, projected linear CKA, and projected effective dimension without
changing or retraining the checkpoint.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dendritic_modeling.analysis.tools.representation_accessibility import (
    population_geometry,
)
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    iter_analysis_batches,
)
from dendritic_modeling.config import (
    EvaluationRuntimeConfig,
    VisionBoundaryDiagnosticsAnalysisParams,
)
from dendritic_modeling.networks.architectures.classical.pretrained import (
    build_pretrained_core_from_plan,
    build_split_plan,
    list_backbone_layers,
)
from dendritic_modeling.utils.general import save_dict


@dataclass
class _StreamingMoments:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    zero_count: int = 0
    positive_count: int = 0
    sample_norm_total: float = 0.0
    sample_count: int = 0

    def update(self, values: torch.Tensor, *, zero_tolerance: float) -> None:
        flattened = values.detach().to(dtype=torch.float64).reshape(values.shape[0], -1)
        if not torch.isfinite(flattened).all():
            raise ValueError("vision boundary diagnostics require finite activations")
        self.count += int(flattened.numel())
        self.total += float(flattened.sum().item())
        self.total_square += float(flattened.square().sum().item())
        self.zero_count += int((flattened.abs() <= zero_tolerance).sum().item())
        self.positive_count += int((flattened > zero_tolerance).sum().item())
        self.sample_norm_total += float(flattened.norm(dim=1).sum().item())
        self.sample_count += int(flattened.shape[0])

    def summary(self) -> dict[str, float | int]:
        if self.count < 1 or self.sample_count < 1:
            raise RuntimeError("vision boundary diagnostics observed no activations")
        mean = self.total / self.count
        variance = max(self.total_square / self.count - mean * mean, 0.0)
        return {
            "scalar_count": self.count,
            "mean": mean,
            "standard_deviation": variance**0.5,
            "zero_fraction": self.zero_count / self.count,
            "positive_fraction": self.positive_count / self.count,
            "mean_sample_l2_norm": self.sample_norm_total / self.sample_count,
        }


def _inferred_replacement_names(model: torch.nn.Module, backbone: str) -> list[str]:
    encoder = getattr(model, "encoder_network", None)
    decoder = getattr(model, "decoder_network", None)
    if encoder is None or decoder is None:
        raise TypeError("vision boundary diagnostics require encoder/core/decoder")
    prefix = list(getattr(encoder, "_layer_names", []))
    suffix = list(getattr(decoder, "_layer_names", []))
    all_names = list_backbone_layers(backbone)
    start = all_names.index(prefix[-1]) + 1 if prefix else 0
    end = all_names.index(suffix[0]) if suffix else len(all_names)
    names = all_names[start:end]
    if not names:
        raise ValueError("could not infer a nonempty replaced vision span")
    return names


def _exact_reference(
    model: torch.nn.Module,
    params: VisionBoundaryDiagnosticsAnalysisParams,
    *,
    spatial: bool,
) -> tuple[torch.nn.Module, list[str]]:
    target_modules = _inferred_replacement_names(model, params.backbone)
    plan = build_split_plan(
        backbone=params.backbone,
        weights=params.weights,
        target_modules=target_modules,
        spatial=spatial,
    )
    return build_pretrained_core_from_plan(plan, freeze=True), target_modules


def _projection_map(
    feature_count: int,
    projection_dim: int,
    seed: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    sequence = np.random.SeedSequence([int(seed), int(feature_count), 0xB0A7D4])
    rng = np.random.default_rng(sequence)
    buckets_np = rng.integers(0, projection_dim, size=feature_count, dtype=np.int64)
    signs_np = rng.choice(np.asarray([-1.0, 1.0]), size=feature_count)
    digest = hashlib.sha256()
    digest.update(buckets_np.tobytes())
    digest.update(signs_np.astype(np.float32).tobytes())
    buckets = torch.from_numpy(buckets_np).to(device=device)
    signs = torch.from_numpy(signs_np).to(device=device, dtype=torch.float32)
    return buckets, signs, digest.hexdigest()


def _project(
    values: torch.Tensor,
    buckets: torch.Tensor,
    signs: torch.Tensor,
    projection_dim: int,
) -> torch.Tensor:
    flattened = values.detach().to(dtype=torch.float32).reshape(values.shape[0], -1)
    projected = flattened.new_zeros((flattened.shape[0], projection_dim))
    projected.scatter_add_(
        1,
        buckets.unsqueeze(0).expand(flattened.shape[0], -1),
        flattened * signs.unsqueeze(0),
    )
    return projected


def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("linear CKA requires matched samples-by-features matrices")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    left_covariance = x.T @ x
    right_covariance = y.T @ y
    denominator = np.linalg.norm(left_covariance) * np.linalg.norm(right_covariance)
    return float(np.square(cross).sum() / denominator) if denominator > 0 else 0.0


class VisionBoundaryDiagnosticsAnalyzer:
    """Compare one fitted vision replacement with its exact pretrained span."""

    def __init__(self, params: VisionBoundaryDiagnosticsAnalysisParams):
        self.params = params

    def analyze(
        self,
        model: torch.nn.Module,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "final",
        runtime: EvaluationRuntimeConfig | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        if not hasattr(model, "encoder_network") or not hasattr(model, "core_network"):
            raise TypeError("vision boundary diagnostics require a replacement model")

        student_moments = _StreamingMoments()
        teacher_moments = _StreamingMoments()
        projected_student: list[np.ndarray] = []
        projected_teacher: list[np.ndarray] = []
        squared_error = 0.0
        teacher_square = 0.0
        cosine_total = 0.0
        sample_count = 0
        target_modules: list[str] | None = None
        exact_core: torch.nn.Module | None = None
        buckets: torch.Tensor | None = None
        signs: torch.Tensor | None = None
        projection_sha256: str | None = None
        feature_shape: list[int] | None = None

        with analysis_device_context(model, device) as analysis_device:
            with torch.no_grad():
                for batch in iter_analysis_batches(
                    test_dataset,
                    runtime,
                    self.params.max_samples,
                    device=analysis_device,
                ):
                    inputs = batch[0].to(analysis_device)
                    encoded = model.encoder_network(inputs)
                    if exact_core is None:
                        exact_core, target_modules = _exact_reference(
                            model,
                            self.params,
                            spatial=encoded.ndim == 4,
                        )
                        exact_core = exact_core.to(analysis_device).eval()
                    student = model.core_network(encoded)
                    teacher = exact_core(encoded)
                    if not isinstance(student, torch.Tensor) or not isinstance(
                        teacher, torch.Tensor
                    ):
                        raise TypeError("vision boundary outputs must be tensors")
                    if student.shape != teacher.shape:
                        raise ValueError(
                            "replacement and exact boundary shapes differ: "
                            f"{list(student.shape)} != {list(teacher.shape)}"
                        )
                    if feature_shape is None:
                        feature_shape = list(student.shape[1:])
                        feature_count = int(student[0].numel())
                        buckets, signs, projection_sha256 = _projection_map(
                            feature_count,
                            self.params.projection_dim,
                            self.params.projection_seed,
                            device=student.device,
                        )
                    student_moments.update(
                        student, zero_tolerance=self.params.zero_tolerance
                    )
                    teacher_moments.update(
                        teacher, zero_tolerance=self.params.zero_tolerance
                    )
                    student_flat = student.to(dtype=torch.float64).reshape(
                        student.shape[0], -1
                    )
                    teacher_flat = teacher.to(dtype=torch.float64).reshape(
                        teacher.shape[0], -1
                    )
                    squared_error += float(
                        (student_flat - teacher_flat).square().sum().item()
                    )
                    teacher_square += float(teacher_flat.square().sum().item())
                    cosine_total += float(
                        torch.nn.functional.cosine_similarity(
                            student_flat,
                            teacher_flat,
                            dim=1,
                            eps=1e-12,
                        )
                        .sum()
                        .item()
                    )
                    sample_count += int(student.shape[0])
                    if buckets is None or signs is None:
                        raise RuntimeError("projection map was not initialized")
                    projected_student.append(
                        _project(
                            student,
                            buckets,
                            signs,
                            self.params.projection_dim,
                        )
                        .cpu()
                        .numpy()
                    )
                    projected_teacher.append(
                        _project(
                            teacher,
                            buckets,
                            signs,
                            self.params.projection_dim,
                        )
                        .cpu()
                        .numpy()
                    )

        if sample_count < 2 or target_modules is None or feature_shape is None:
            raise RuntimeError("vision boundary diagnostics observed too few samples")
        student_projection = np.concatenate(projected_student, axis=0)
        teacher_projection = np.concatenate(projected_teacher, axis=0)
        student_summary = student_moments.summary()
        teacher_summary = teacher_moments.summary()
        teacher_norm = float(teacher_summary["mean_sample_l2_norm"])
        results = {
            "contract": {
                "backbone": self.params.backbone,
                "weights": self.params.weights,
                "target_modules": target_modules,
                "feature_shape": feature_shape,
                "sample_count": sample_count,
                "projection_dim": self.params.projection_dim,
                "projection_seed": self.params.projection_seed,
                "projection_sha256": projection_sha256,
                "reference": "exact pretrained replacement span on matched inputs",
            },
            "student": student_summary,
            "exact": teacher_summary,
            "comparison": {
                "relative_mse": squared_error / max(teacher_square, 1e-12),
                "mean_sample_cosine": cosine_total / sample_count,
                "mean_l2_norm_ratio": (
                    float(student_summary["mean_sample_l2_norm"])
                    / max(teacher_norm, 1e-12)
                ),
                "projected_linear_cka": _linear_cka(
                    student_projection, teacher_projection
                ),
            },
            "projected_geometry": {
                "student": population_geometry(
                    student_projection,
                    variance_epsilon=1e-12,
                ),
                "exact": population_geometry(
                    teacher_projection,
                    variance_epsilon=1e-12,
                ),
            },
        }
        if save_path is not None:
            output_filename = (
                filename if filename.endswith(".json") else f"{filename}.json"
            )
            save_dict(results, save_path, output_filename)
        return results


__all__ = ["VisionBoundaryDiagnosticsAnalyzer"]
