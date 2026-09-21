"""Runtime helpers for materialized and streaming metric evaluation."""

from __future__ import annotations

import inspect
import logging
import math
from collections.abc import Callable

import torch
from torch.utils.data import DataLoader

from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    estimate_dataset_size_mb,
    make_analysis_loader,
    should_materialize_dataset,
)
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import Classifier, RecurrentClassifier, Regressor
from dendritic_modeling.training.utils._evaluation.score import (
    _metric_accepts_seq_lengths,
)
from dendritic_modeling.utils.general import save_dict

logger = logging.getLogger("dendritic_modeling.training.utils.evaluation")


def _should_materialize(
    dataset: torch.utils.data.Dataset,
    mode: str = "auto",
    threshold_mb: float = 1024,
) -> bool:
    """Decide whether to materialize *dataset* into a single tensor."""

    runtime = EvaluationRuntimeConfig(
        mode=mode,
        materialize_threshold_mb=threshold_mb,
    )
    return should_materialize_dataset(dataset, runtime)


def _model_accepts_seq_lengths(model: torch.nn.Module) -> bool:
    """Return whether ``model.forward`` accepts variable sequence lengths."""
    return "seq_lengths" in inspect.signature(model.forward).parameters


def _stream_model_outputs(
    model: Classifier | Regressor,
    loader: DataLoader,
    device: str,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run *model* on batches from *loader*, accumulating outputs + labels on CPU.

    Model outputs (logits/predictions) are much smaller than raw inputs
    (e.g. Nx1000 floats vs Nx3x224x224 for ImageNet), so accumulating them
    on CPU is safe even for very large datasets.

    Returns (all_inputs_placeholder, all_labels, all_seq_lengths) where
    ``all_inputs_placeholder`` is a dummy since we only need labels+outputs
    for metrics.  We actually return (all_outputs, all_labels, seq_lengths).
    """
    all_outputs: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_seq_lengths: list[torch.Tensor] = []
    batches_seen = 0

    with analysis_device_context(model, device) as analysis_device:
        with torch.no_grad():
            for batch in loader:
                if max_batches is not None and batches_seen >= max_batches:
                    break
                inputs = batch[0].to(analysis_device)
                labels = batch[1]
                seq_lengths = batch[2] if len(batch) > 2 else None

                outputs = model(
                    inputs,
                    **(
                        {"seq_lengths": seq_lengths.to(analysis_device)}
                        if seq_lengths is not None and _model_accepts_seq_lengths(model)
                        else {}
                    ),
                )
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())
                if seq_lengths is not None:
                    all_seq_lengths.append(seq_lengths.cpu())
                batches_seen += 1

    if not all_outputs:
        return torch.tensor([]), torch.tensor([]), None

    cat_outputs = torch.cat(all_outputs, dim=0)
    cat_labels = torch.cat(all_labels, dim=0)
    cat_seq_lengths = torch.cat(all_seq_lengths, dim=0) if all_seq_lengths else None
    return cat_outputs, cat_labels, cat_seq_lengths


def _make_eval_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    device: str = "cpu",
    persistent_workers: bool = False,
    prefetch_factor: int | None = 2,
    in_order: bool | None = None,
    seed: int = 0,
    **_extra,
) -> DataLoader:
    """Build a DataLoader suitable for evaluation (no shuffle, no drop_last)."""

    runtime = EvaluationRuntimeConfig(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        in_order=in_order,
        seed=seed,
    )
    return make_analysis_loader(
        dataset=dataset,
        runtime=runtime,
        device=device,
    )


class _CachedOutputModel:
    """Wraps a model so that ``__call__`` / ``predict`` return pre-computed outputs.

    Preserves ``isinstance`` checks (Classifier / Regressor) and attribute
    access (``_is_recurrent_core``, etc.) by delegating to the wrapped model.
    """

    def __init__(self, real_model: torch.nn.Module, cached_outputs: torch.Tensor):
        # Store on the instance directly so __getattr__ doesn't intercept them.
        object.__setattr__(self, "_real_model", real_model)
        object.__setattr__(self, "_cached_outputs", cached_outputs)

    # --- calls that metric functions make ---
    def __call__(self, *_args, **_kwargs):
        return object.__getattribute__(self, "_cached_outputs")

    def forward(self, *_args, **_kwargs):
        return object.__getattribute__(self, "_cached_outputs")

    def predict(self, *_args, **_kwargs):
        cached = object.__getattribute__(self, "_cached_outputs")
        real = object.__getattribute__(self, "_real_model")
        # Classifier.predict() returns argmax of logits; Regressor.predict()
        # returns raw outputs.  Replicate that from cached forward outputs
        # so metrics that call .predict() (e.g. pred_label_mi) see correct
        # class indices rather than raw logits.
        if isinstance(real, (Classifier, RecurrentClassifier)):
            return cached.argmax(dim=-1)
        return cached

    # --- delegate everything else (isinstance, attributes) to real model ---
    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real_model"), name)

    def __isinstance_check__(self, cls):
        return isinstance(object.__getattribute__(self, "_real_model"), cls)

    # Make isinstance() work by forwarding class checks.
    @property
    def __class__(self):
        return object.__getattribute__(self, "_real_model").__class__


def _scalarize_metric_result(result):
    """Return Python scalars for scalar tensors while preserving vector metrics."""
    if hasattr(result, "numel") and result.numel() == 1:
        return result.item()
    return result


def _is_nan_metric_result(result) -> bool:
    """Return True when a scalar metric result is NaN."""
    if result is None:
        return False
    if torch.is_tensor(result):
        return result.numel() == 1 and bool(torch.isnan(result).item())
    if isinstance(result, float):
        return math.isnan(result)
    return False


def _metric_call_kwargs(
    *,
    reduce_dim: int,
    move_device: bool,
    device: str,
    metric_kwargs: dict,
) -> dict:
    """Build the common metric keyword payload."""
    return dict(
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        **metric_kwargs,
    )


def _evaluate_materialized_metric(
    *,
    metric_func: Callable,
    model: Classifier | Regressor,
    items,
    reduce_dim: int,
    move_device: bool,
    device: str,
    metric_accepts_seq_lengths: bool,
    metric_kwargs: dict,
):
    """Evaluate one metric on a fully materialized dataset slice."""
    seq_lengths = items[2] if len(items) > 2 else None
    call_kwargs = _metric_call_kwargs(
        reduce_dim=reduce_dim,
        move_device=move_device,
        device=device,
        metric_kwargs=metric_kwargs,
    )
    if metric_accepts_seq_lengths:
        call_kwargs["seq_lengths"] = seq_lengths
    return _scalarize_metric_result(
        metric_func(
            model,
            items[0],
            items[1],
            **call_kwargs,
        )
    )


def _evaluate_streamed_metric(
    *,
    metric_func: Callable,
    metric_name: str,
    model: Classifier | Regressor,
    dataset: torch.utils.data.Dataset,
    reduce_dim: int,
    device: str,
    metric_accepts_seq_lengths: bool,
    eval_batch_size: int,
    eval_num_workers: int,
    eval_pin_memory: bool,
    eval_persistent_workers: bool,
    eval_prefetch_factor: int | None,
    eval_in_order: bool | None,
    eval_max_batches: int | None,
    eval_materialize_threshold_mb: float,
    eval_seed: int,
    eval_loader_factory: Callable[..., DataLoader] | None,
    metric_kwargs: dict,
):
    """Evaluate one metric by streaming model outputs through a DataLoader."""
    est_mb = estimate_dataset_size_mb(dataset)
    logger.info(
        "Streaming evaluation for %s (est. %.0f MB, threshold %.0f MB, "
        "max_batches=%s)",
        metric_name,
        est_mb,
        eval_materialize_threshold_mb,
        eval_max_batches,
    )
    loader_factory = eval_loader_factory or _make_eval_loader
    loader = loader_factory(
        dataset,
        batch_size=eval_batch_size,
        num_workers=eval_num_workers,
        pin_memory=eval_pin_memory,
        device=device,
        persistent_workers=eval_persistent_workers,
        prefetch_factor=eval_prefetch_factor,
        in_order=eval_in_order,
        seed=eval_seed,
    )
    cat_outputs, cat_labels, cat_seq_lengths = _stream_model_outputs(
        model=model,
        loader=loader,
        device=device,
        max_batches=eval_max_batches,
    )
    if cat_outputs.numel() == 0:
        return None

    # The _compute_* functions expect (model, inputs, labels) and call
    # model(inputs) internally.  We wrap the model so forward() returns
    # cached outputs, avoiding a second forward pass.  The wrapper
    # preserves isinstance checks and attributes like _is_recurrent_core.
    cached_model = _CachedOutputModel(model, cat_outputs)

    stream_kwargs = _metric_call_kwargs(
        reduce_dim=reduce_dim,
        move_device=False,
        device="cpu",
        metric_kwargs=metric_kwargs,
    )
    if metric_accepts_seq_lengths:
        stream_kwargs["seq_lengths"] = cat_seq_lengths
    return _scalarize_metric_result(
        metric_func(
            cached_model,
            cat_outputs,  # inputs arg (ignored by cached forward)
            cat_labels,
            **stream_kwargs,
        )
    )


def _evaluate_metric_generic(
    metric_func: Callable,
    metric_name: str,
    model: Classifier | Regressor,
    train_ds: torch.utils.data.Dataset | None = None,
    valid_ds: torch.utils.data.Dataset | None = None,
    test_ds: torch.utils.data.Dataset | None = None,
    reduce_dim: int = 0,
    move_device: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str | None = None,
    filename: str | None = None,
    eval_mode: str = "auto",
    eval_batch_size: int = 256,
    eval_num_workers: int = 4,
    eval_pin_memory: bool = True,
    eval_persistent_workers: bool = False,
    eval_prefetch_factor: int | None = 2,
    eval_in_order: bool | None = None,
    eval_max_batches: int | None = None,
    eval_materialize_threshold_mb: float = 1024,
    eval_seed: int = 0,
    eval_loader_factory: Callable[..., DataLoader] | None = None,
    **metric_kwargs,
) -> tuple[
    float | torch.Tensor | None,
    float | torch.Tensor | None,
    float | torch.Tensor | None,
]:
    """
    Generic function to evaluate any metric on multiple datasets.

    Supports two modes:
    - **materialize**: load dataset into memory via ``dataset[:]`` (old behaviour,
      fast for small datasets).
    - **stream**: iterate batch-by-batch via DataLoader (memory-safe for large
      datasets like ImageNet).

    The ``"auto"`` mode (default) picks based on estimated dataset size.

    Args:
        metric_func: The metric function to call (e.g., accuracy_score, auc_score)
        metric_name: Name of the metric for saving (e.g., "accuracy", "auc")
        model: The model (classifier or regressor)
        train_ds: Training dataset (optional)
        valid_ds: Validation dataset (optional)
        test_ds: Test dataset (optional)
        reduce_dim: Dimension along which to compute mean
        move_device: Whether to move tensors to device
        device: Device to use for computation
        save_path: Path to save results (optional)
        filename: Filename for saved results (defaults to metric_name)
        eval_mode: "auto", "materialize", or "stream"
        eval_batch_size: Batch size for streaming evaluation
        eval_num_workers: DataLoader workers for streaming evaluation
        eval_max_batches: Cap evaluation to this many batches (None = all)
        eval_materialize_threshold_mb: Size threshold for auto mode switch
        **metric_kwargs: Additional keyword arguments for the metric function

    Returns:
        Tuple of (train_metric, valid_metric, test_metric)
    """
    if filename is None:
        filename = metric_name

    datasets = [train_ds, valid_ds, test_ds]
    results = []

    # Check once whether metric_func accepts seq_lengths.
    metric_accepts_seq_lengths = _metric_accepts_seq_lengths(metric_func)

    for dataset in datasets:
        if dataset is None:
            results.append(None)
            continue

        materialize = _should_materialize(
            dataset, mode=eval_mode, threshold_mb=eval_materialize_threshold_mb
        )

        if materialize:
            # --- fast path: full materialization (old behaviour) ---
            try:
                items = dataset[:]
            except Exception:
                # Some wrappers don't support slicing; fall back to streaming.
                pass
            else:
                materialized_result = _evaluate_materialized_metric(
                    metric_func=metric_func,
                    model=model,
                    items=items,
                    reduce_dim=reduce_dim,
                    move_device=move_device,
                    device=device,
                    metric_accepts_seq_lengths=metric_accepts_seq_lengths,
                    metric_kwargs=metric_kwargs,
                )
                if not _is_nan_metric_result(materialized_result):
                    results.append(materialized_result)
                    continue
                logger.warning(
                    "Materialized evaluation for %s returned NaN; retrying with "
                    "streaming evaluation. This usually indicates a recoverable "
                    "device-memory failure during full-dataset scoring.",
                    metric_name,
                )

        results.append(
            _evaluate_streamed_metric(
                metric_func=metric_func,
                metric_name=metric_name,
                model=model,
                dataset=dataset,
                reduce_dim=reduce_dim,
                device=device,
                metric_accepts_seq_lengths=metric_accepts_seq_lengths,
                eval_batch_size=eval_batch_size,
                eval_num_workers=eval_num_workers,
                eval_pin_memory=eval_pin_memory,
                eval_persistent_workers=eval_persistent_workers,
                eval_prefetch_factor=eval_prefetch_factor,
                eval_in_order=eval_in_order,
                eval_max_batches=eval_max_batches,
                eval_materialize_threshold_mb=eval_materialize_threshold_mb,
                eval_seed=eval_seed,
                eval_loader_factory=eval_loader_factory,
                metric_kwargs=metric_kwargs,
            )
        )

    train_metric, valid_metric, test_metric = results

    # Save results if path provided
    if save_path is not None:
        save_dict(
            {
                f"train {metric_name}": train_metric,
                f"valid {metric_name}": valid_metric,
                f"test {metric_name}": test_metric,
            },
            save_path,
            filename,
        )

    return train_metric, valid_metric, test_metric


__all__ = [
    "_CachedOutputModel",
    "_evaluate_metric_generic",
    "_make_eval_loader",
    "_model_accepts_seq_lengths",
    "_should_materialize",
    "_stream_model_outputs",
]
