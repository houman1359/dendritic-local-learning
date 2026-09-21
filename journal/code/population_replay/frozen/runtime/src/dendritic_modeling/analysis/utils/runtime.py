"""Shared dataset/runtime helpers for analysis modules."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dendritic_modeling.analysis.utils.dataset_runtime import (
    effective_sample_cap,
    estimate_dataset_size_mb,
    should_materialize_dataset,
    subset_dataset_for_runtime,
)
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.training.dataloader_utils import (
    dataloader_supports_kwarg,
    seeded_dataloader_kwargs,
)
from dendritic_modeling.utils.hooks import HookHandle, run_with_forward_hooks

logger = logging.getLogger(__name__)
AnalysisBatchProcessor = Callable[[tuple[torch.Tensor, ...], torch.device], None]


def _resolve_analysis_prefetch_factor(
    prefetch_factor: int | None,
    *,
    num_workers: int,
) -> int | None:
    if prefetch_factor is not None and num_workers > 0:
        return max(1, int(prefetch_factor))
    return None


def _resolve_analysis_num_workers(runtime: EvaluationRuntimeConfig) -> int:
    return max(0, int(getattr(runtime, "num_workers", 0)))


def _resolve_analysis_persistent_workers(
    persistent_workers: bool,
    *,
    num_workers: int,
) -> bool:
    resolved_persistent = bool(persistent_workers)
    if resolved_persistent and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0")
    return resolved_persistent


def analysis_loader_kwargs_from_runtime(
    runtime: EvaluationRuntimeConfig | None,
    device: str | torch.device | None = None,
) -> dict[str, object]:
    """Return DataLoader kwargs governed by analysis runtime settings."""
    if runtime is None:
        runtime = EvaluationRuntimeConfig()
    pin = resolve_analysis_pin_memory(runtime, device)
    num_workers = _resolve_analysis_num_workers(runtime)
    persistent_workers = _resolve_analysis_persistent_workers(
        getattr(runtime, "persistent_workers", False),
        num_workers=num_workers,
    )

    loader_kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin,
        "persistent_workers": persistent_workers,
    }
    prefetch_factor = _resolve_analysis_prefetch_factor(
        getattr(runtime, "prefetch_factor", None),
        num_workers=num_workers,
    )
    if prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    in_order = getattr(runtime, "in_order", None)
    if in_order is not None and dataloader_supports_kwarg("in_order"):
        loader_kwargs["in_order"] = bool(in_order)
    return loader_kwargs


def make_analysis_loader(
    dataset: Dataset,
    runtime: EvaluationRuntimeConfig | None,
    explicit_max_samples: int | None = None,
    device: str | torch.device | None = None,
) -> DataLoader:
    """Build a deterministic evaluation loader for analysis."""
    if runtime is None:
        runtime = EvaluationRuntimeConfig()
    dataset_view = subset_dataset_for_runtime(dataset, runtime, explicit_max_samples)
    loader_kwargs = analysis_loader_kwargs_from_runtime(runtime, device)
    return DataLoader(
        dataset_view,
        batch_size=runtime.batch_size,
        shuffle=False,
        **seeded_dataloader_kwargs(int(getattr(runtime, "seed", 0))),
        **loader_kwargs,
    )


def materialize_dataset(
    dataset: Dataset,
    runtime: EvaluationRuntimeConfig | None,
    explicit_max_samples: int | None = None,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, ...]:
    """Return dataset tensors on CPU using either slicing or streaming."""
    if runtime is None:
        runtime = EvaluationRuntimeConfig()
    dataset_view = subset_dataset_for_runtime(dataset, runtime, explicit_max_samples)

    if should_materialize_dataset(dataset_view, runtime):
        try:
            items = dataset_view[:]
            if isinstance(items, torch.Tensor):
                return (items,)
            return tuple(items)
        except Exception:
            pass

    loader = make_analysis_loader(
        dataset,
        runtime,
        explicit_max_samples,
        device=device,
    )
    accumulated: list[list[torch.Tensor]] = []
    for batch in loader:
        if isinstance(batch, torch.Tensor):
            batch = (batch,)
        if not accumulated:
            accumulated = [[] for _ in range(len(batch))]
        for idx, value in enumerate(batch):
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            accumulated[idx].append(value.cpu())

    if not accumulated:
        return ()
    return tuple(torch.cat(chunks, dim=0) for chunks in accumulated)


def evaluation_kwargs_from_runtime(
    runtime: EvaluationRuntimeConfig | None,
) -> dict[str, int | bool | str | None]:
    """Translate runtime config into metric evaluation kwargs."""
    if runtime is None:
        runtime = EvaluationRuntimeConfig()
    kwargs: dict[str, int | bool | str | None] = {
        "eval_mode": runtime.mode,
        "eval_batch_size": runtime.batch_size,
        "eval_num_workers": runtime.num_workers,
        "eval_pin_memory": runtime.pin_memory,
        "eval_persistent_workers": runtime.persistent_workers,
        "eval_prefetch_factor": runtime.prefetch_factor,
        "eval_max_batches": runtime.max_batches,
        "eval_materialize_threshold_mb": runtime.materialize_threshold_mb,
        "eval_seed": runtime.seed,
    }
    if runtime.in_order is not None:
        kwargs["eval_in_order"] = runtime.in_order
    return kwargs


def iter_analysis_batches(
    dataset: Dataset,
    runtime: EvaluationRuntimeConfig | None,
    explicit_max_samples: int | None = None,
    device: str | torch.device | None = None,
) -> Iterable[tuple[torch.Tensor, ...]]:
    """Yield dataset batches under the shared runtime policy.

    In ``materialize`` mode, or when ``auto`` selects materialization and the
    dataset supports slicing, this yields a single full batch. This preserves
    the old single-pass behavior used by hook-based analyzers on small datasets.
    """
    if runtime is None:
        runtime = EvaluationRuntimeConfig()
    dataset_view = subset_dataset_for_runtime(dataset, runtime, explicit_max_samples)

    if should_materialize_dataset(dataset_view, runtime):
        try:
            items = dataset_view[:]
            if isinstance(items, torch.Tensor):
                yield (items,)
            else:
                yield tuple(items)
            return
        except Exception:
            pass

    loader = make_analysis_loader(
        dataset,
        runtime,
        explicit_max_samples,
        device=device,
    )
    for batch in loader:
        if isinstance(batch, torch.Tensor):
            yield (batch,)
        else:
            yield tuple(batch)


# ---------------------------------------------------------------------------
# Analysis device context manager
# ---------------------------------------------------------------------------


def get_model_device(model: nn.Module) -> torch.device:
    """Best-effort device inference for modules with parameters or buffers."""
    try:
        return next(model.parameters()).device
    except (StopIteration, TypeError):
        pass
    try:
        return next(model.buffers()).device
    except (StopIteration, TypeError):
        return torch.device("cpu")


@contextlib.contextmanager
def analysis_device_context(
    model: nn.Module,
    device: str | torch.device,
):
    """Context manager that moves a model to *device* for analysis and restores
    it to the original device and train/eval state afterward.

    Usage::

        with analysis_device_context(model, "cpu") as dev:
            # model is on dev, in eval mode
            out = model(inputs.to(dev))
        # model is back on the original device and in its original mode

    This prevents the common bug where an analyzer moves a model to the
    analysis device in-place and never moves it back, breaking subsequent
    training or analysis steps that expect the model on the training device.
    """
    original_device = get_model_device(model)
    was_training = getattr(model, "training", False)
    target = torch.device(device)

    try:
        model.to(target)
        model.eval()
        yield target
    finally:
        model.to(original_device)
        model.train(was_training)


def run_no_grad_analysis_batches(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: str | torch.device,
    runtime: EvaluationRuntimeConfig | None,
    explicit_max_samples: int | None,
    process_batch: AnalysisBatchProcessor,
) -> None:
    """Run a no-grad analysis pass over runtime-governed dataset batches."""
    with analysis_device_context(model, device) as analysis_device:
        _run_no_grad_batches(
            dataset=dataset,
            runtime=runtime,
            explicit_max_samples=explicit_max_samples,
            device=analysis_device,
            process_batch=process_batch,
        )


def _run_no_grad_batches(
    *,
    dataset: Dataset,
    runtime: EvaluationRuntimeConfig | None,
    explicit_max_samples: int | None,
    device: torch.device,
    process_batch: AnalysisBatchProcessor,
) -> None:
    with torch.no_grad():
        for batch in iter_analysis_batches(
            dataset,
            runtime,
            explicit_max_samples,
            device=device,
        ):
            process_batch(batch, device)


def run_model_over_analysis_batches(
    *,
    model: nn.Module,
    dataset: Dataset,
    device: str | torch.device,
    runtime: EvaluationRuntimeConfig | None,
    explicit_max_samples: int | None,
    attach_hooks: Callable[[], Iterable[HookHandle]],
    remove_hooks: Callable[[Iterable[HookHandle]], None],
) -> None:
    """Run a model over analysis batches with managed hooks and device state."""

    def _process_batch(batch: tuple[torch.Tensor, ...], analysis_device) -> None:
        _ = model(batch[0].to(analysis_device))

    with analysis_device_context(model, device) as analysis_device:
        run_with_forward_hooks(
            attach=attach_hooks,
            remove=remove_hooks,
            body=lambda: _run_no_grad_batches(
                dataset=dataset,
                runtime=runtime,
                explicit_max_samples=explicit_max_samples,
                device=analysis_device,
                process_batch=_process_batch,
            ),
        )


def resolve_analysis_pin_memory(
    runtime: EvaluationRuntimeConfig | None,
    device: str | torch.device | None = None,
) -> bool:
    """Return the effective pin_memory setting, auto-disabling on CPU."""
    if runtime is None:
        pin = True
    else:
        pin = getattr(runtime, "pin_memory", True)
    if device is not None and str(device) == "cpu":
        return False
    if device is None and not torch.cuda.is_available():
        return False
    return pin


__all__ = [
    "analysis_device_context",
    "analysis_loader_kwargs_from_runtime",
    "effective_sample_cap",
    "estimate_dataset_size_mb",
    "evaluation_kwargs_from_runtime",
    "get_model_device",
    "iter_analysis_batches",
    "make_analysis_loader",
    "materialize_dataset",
    "resolve_analysis_pin_memory",
    "run_model_over_analysis_batches",
    "run_no_grad_analysis_batches",
    "should_materialize_dataset",
    "subset_dataset_for_runtime",
]
