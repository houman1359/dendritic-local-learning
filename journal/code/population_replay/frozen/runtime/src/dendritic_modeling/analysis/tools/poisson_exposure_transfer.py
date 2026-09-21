"""Posthoc Poisson exposure-transfer evaluation with common random numbers.

This analyzer is deliberately not registered with :class:`AnalysisManager`.
The manager receives one already-built test split, whereas exposure transfer
must rebuild the stochastic test dataset at several exposures and coordinate
draw seeds across independently trained checkpoints.  Keeping it as an
explicit posthoc tool makes that cross-checkpoint protocol visible to callers.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from dendritic_modeling.analysis.utils.runtime import analysis_device_context
from dendritic_modeling.config import PoissonExposureTransferParams
from dendritic_modeling.utils.dataset_fingerprint import fingerprint_dataset
from dendritic_modeling.utils.reproducibility import (
    isolated_random_seed,
    preserved_random_state,
)

POISSON_EXPOSURE_DRAW_SEED_RULE = "base + 100 * round(test_duration * 1e6) + draw"
POSTHOC_INTEGRATION_NOTE = (
    "Poisson exposure transfer is posthoc-only because it rebuilds one stochastic "
    "test dataset per exposure and coordinates common random numbers across "
    "independently trained checkpoints; AnalysisManager operates on one fixed test "
    "dataset for one model run."
)


def poisson_exposure_duration_key(duration: float) -> int:
    """Return the stable micro-exposure key used by the draw-seed protocol."""

    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(
            f"Test durations must be finite and positive, got {duration!r}"
        )
    key = round(duration * 1_000_000)
    if key <= 0:
        raise ValueError(
            f"Test duration is below micro-duration resolution: {duration}"
        )
    return key


def validate_poisson_exposure_durations(
    durations: tuple[float, ...] | list[float],
) -> tuple[float, ...]:
    """Normalize exposures and reject empty or micro-precision collisions."""

    values = tuple(float(value) for value in durations)
    if not values:
        raise ValueError("At least one test duration is required")
    keys = [poisson_exposure_duration_key(value) for value in values]
    if len(set(keys)) != len(keys):
        raise ValueError(
            f"Test durations collide at micro-duration precision: {values}"
        )
    return values


def poisson_exposure_draw_seed(base_seed: int, duration: float, draw: int) -> int:
    """Return a model-independent seed for one exposure and draw.

    The seed depends on the nominal exposure rather than its position in a
    sweep.  Draw indices occupy a collision-free namespace of size 100.
    """

    if isinstance(draw, bool) or int(draw) != draw:
        raise ValueError(f"draw must be an integer, got {draw!r}")
    draw = int(draw)
    if draw < 0:
        raise ValueError(f"draw must be non-negative, got {draw}")
    if draw >= 100:
        raise ValueError("At most 100 draws per duration are supported")
    return int(base_seed) + 100 * poisson_exposure_duration_key(duration) + draw


def _realized_dataset_fingerprint(
    inputs: list[torch.Tensor],
    labels: list[torch.Tensor],
) -> dict[str, Any]:
    if not inputs or not labels:
        raise ValueError("Cannot fingerprint an empty realized dataset")
    realized = TensorDataset(torch.cat(inputs, dim=0), torch.cat(labels, dim=0))
    return fingerprint_dataset(realized)


@torch.inference_mode()
def evaluate_poisson_exposure_accuracy(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    draw_seed: int,
    fingerprint_realized_dataset: bool = False,
) -> tuple[int, int, dict[str, Any] | None]:
    """Evaluate one stochastic draw without leaking any RNG-state changes.

    ``num_workers=0`` and the global CPU stream are intentional compatibility
    requirements: :class:`PoissonGeneratorDataset` samples with
    :func:`torch.poisson` in ``__getitem__``.  Seeding inside
    :func:`isolated_random_seed` preserves the historical realization exactly
    while restoring Python, NumPy, CPU, and visible CUDA streams afterward.
    """

    realized_inputs: list[torch.Tensor] = []
    realized_labels: list[torch.Tensor] = []
    correct = 0
    total = 0
    with isolated_random_seed(draw_seed):
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
        )
        for batch in loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                raise TypeError(
                    "Poisson exposure datasets must yield (inputs, labels) batches"
                )
            inputs, labels = batch[0], batch[1]
            if fingerprint_realized_dataset:
                realized_inputs.append(inputs.detach().cpu().clone())
                realized_labels.append(labels.detach().cpu().clone())
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            if not isinstance(logits, torch.Tensor):
                raise TypeError("Model must return a tensor of class logits")
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    "Model produced non-finite Poisson-transfer logits"
                )
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())

    fingerprint = None
    if fingerprint_realized_dataset:
        fingerprint = _realized_dataset_fingerprint(
            realized_inputs,
            realized_labels,
        )
    return correct, total, fingerprint


class PoissonExposureTransferAnalyzer:
    """Evaluate a checkpoint across Poisson exposures using paired draws.

    Parameters are represented by :class:`PoissonExposureTransferParams`, so
    the protocol can be saved to and loaded from YAML independently of a paper
    runner.  ``dataset_factory`` must rebuild the test dataset for the supplied
    exposure while holding the base split fixed.

    This is a posthoc analyzer rather than an ``AnalysisManager`` component;
    see :data:`POSTHOC_INTEGRATION_NOTE`.
    """

    def __init__(
        self,
        params: PoissonExposureTransferParams | Mapping[str, Any] | None = None,
    ) -> None:
        if params is None:
            params = PoissonExposureTransferParams()
        elif isinstance(params, Mapping):
            params = PoissonExposureTransferParams(**dict(params))
        if not isinstance(params, PoissonExposureTransferParams):
            raise TypeError("params must be PoissonExposureTransferParams or a mapping")
        self.params = params
        self.test_durations = validate_poisson_exposure_durations(params.test_durations)

    def analyze(
        self,
        model: torch.nn.Module,
        *,
        dataset_factory: Callable[[float], Dataset],
        train_duration: float,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Return protocol metadata and one accuracy record per paired draw."""

        train_duration = float(train_duration)
        if not math.isfinite(train_duration) or train_duration <= 0:
            raise ValueError(
                f"Training duration must be finite and positive, got {train_duration!r}"
            )
        if not callable(dataset_factory):
            raise TypeError("dataset_factory must be callable")

        records: list[dict[str, Any]] = []
        with analysis_device_context(model, device) as analysis_device:
            for duration in self.test_durations:
                # Dataset construction is normally deterministic from a split
                # seed. Preserve all ambient streams as a defensive boundary
                # for custom factories without changing the state they see.
                with preserved_random_state():
                    dataset = dataset_factory(duration)
                if not isinstance(dataset, Dataset):
                    raise TypeError(
                        "dataset_factory must return a torch.utils.data.Dataset"
                    )
                if (
                    self.params.max_samples is not None
                    and self.params.max_samples < len(dataset)
                ):
                    dataset = Subset(dataset, range(self.params.max_samples))
                for draw in range(self.params.draws_per_duration):
                    draw_seed = poisson_exposure_draw_seed(
                        self.params.base_draw_seed,
                        duration,
                        draw,
                    )
                    n_correct, n_samples, fingerprint = (
                        evaluate_poisson_exposure_accuracy(
                            model,
                            dataset,
                            device=analysis_device,
                            batch_size=self.params.batch_size,
                            draw_seed=draw_seed,
                            fingerprint_realized_dataset=(
                                self.params.fingerprint_realized_datasets
                            ),
                        )
                    )
                    record: dict[str, Any] = {
                        "train_duration": train_duration,
                        "test_duration": float(duration),
                        "test_duration_key": poisson_exposure_duration_key(duration),
                        "draw": draw,
                        "draw_seed": draw_seed,
                        "n_correct": n_correct,
                        "n_samples": n_samples,
                        "accuracy": n_correct / max(n_samples, 1),
                    }
                    if fingerprint is not None:
                        record["realized_dataset_fingerprint"] = fingerprint
                    records.append(record)

        result: dict[str, Any] = {
            "test_durations": list(self.test_durations),
            "draws_per_duration": self.params.draws_per_duration,
            "base_draw_seed": self.params.base_draw_seed,
            "draw_seed_rule": POISSON_EXPOSURE_DRAW_SEED_RULE,
            "max_samples": self.params.max_samples,
            "common_random_numbers": True,
            "records": records,
        }
        if self.params.fingerprint_realized_datasets:
            result["fingerprint_realized_datasets"] = True
        return result


__all__ = [
    "POISSON_EXPOSURE_DRAW_SEED_RULE",
    "POSTHOC_INTEGRATION_NOTE",
    "PoissonExposureTransferAnalyzer",
    "evaluate_poisson_exposure_accuracy",
    "poisson_exposure_draw_seed",
    "poisson_exposure_duration_key",
    "validate_poisson_exposure_durations",
]
