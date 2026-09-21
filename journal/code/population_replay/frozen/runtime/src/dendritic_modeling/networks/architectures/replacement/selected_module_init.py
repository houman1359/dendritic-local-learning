"""Teacher capture and initialization for selected-module replacements.

``replace_modules_with_selected_population_networks`` places compiled cells at
named ``nn.Linear``, 1x1-conv, and general KxK-conv boundaries.  This module
supplies the other half of teacher conditioning at those boundaries:

* capture each target module's INPUTS while the intact teacher model runs its
  own forward (the canonical calibration context — after patching, a
  downstream site would already see a partially replaced stream);
* turn (captured inputs, the record's original module) into the flat
  vector-boundary pairs the canonical initializer understands — for a KxK
  convolution that means unfolded patches per output location, with the
  teacher's bias removed from the targets because the patch wrapper carries
  that bias outside the compiled cell;
* delegate to ``initialize_population_topology_from_targets_`` on the inner
  cell, so conv and attention sites get the identical teacher-saliency
  support selection and ridge readout fit as FFN sites, rather than a
  parallel implementation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.networks.architectures.replacement.module_patching import (
    ChannelMapReplacement,
    PatchMapReplacement,
    SelectedModuleReplacementRecord,
    _get_module_path,
)
from dendritic_modeling.networks.architectures.replacement.teacher_topology import (
    DEFAULT_TOPOLOGY_ROW_CHUNK_SIZE,
    initialize_population_topology_from_targets_,
)


def capture_module_inputs(
    model: nn.Module,
    paths: Sequence[str],
    batches: Iterable[Any],
    *,
    max_rows_per_path: int = 4096,
    forward: Any = None,
) -> dict[str, torch.Tensor]:
    """Run ``model`` on ``batches`` and collect each named module's inputs.

    Run this on the INTACT teacher model, before patching: the captured
    tensors are then the calibration context every replacement is initialized
    and locally distilled against.  ``forward`` may adapt how a batch is fed
    (default: ``model(batch)``).  Capture is truncated per path once
    ``max_rows_per_path`` boundary rows have been collected, counting conv
    inputs by their batch dimension.
    """

    captured: dict[str, list[torch.Tensor]] = {path: [] for path in paths}
    rows: dict[str, int] = dict.fromkeys(paths, 0)
    handles = []
    for path in paths:
        module = _get_module_path(model, path)

        def hook(_module, args, *, key=path):
            if rows[key] >= max_rows_per_path:
                return
            value = args[0]
            if not torch.is_tensor(value):
                raise TypeError(
                    f"capture at {key!r} expects a tensor first input, got "
                    f"{type(value).__name__}"
                )
            captured[key].append(value.detach().cpu())
            rows[key] += int(value.shape[0])

        handles.append(module.register_forward_pre_hook(hook))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                if forward is not None:
                    forward(model, batch)
                else:
                    model(batch)
                if all(rows[path] >= max_rows_per_path for path in paths):
                    break
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    missing = [path for path in paths if not captured[path]]
    if missing:
        raise RuntimeError(
            f"no inputs captured for {missing}: the batches never reached those modules"
        )
    return {path: torch.cat(captured[path], dim=0) for path in paths}


def selected_module_boundary_pairs(
    record: SelectedModuleReplacementRecord,
    module_inputs: torch.Tensor,
    *,
    max_rows: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, nn.Module]:
    """Return (flat inputs, flat targets, inner cell) for one record.

    The targets are what the INNER compiled cell must reproduce: for a patch
    wrapper the teacher bias is subtracted, because the wrapper adds its own
    (teacher-initialized, trainable) bias outside the cell.
    """

    original = record.original
    replacement = record.replacement
    original_parameter = next(original.parameters(), None)
    if original_parameter is not None:
        # Captured inputs live on CPU by design; the original module may be
        # on an accelerator. Targets are computed on the original's device
        # and brought back so the returned pairs stay device-neutral.
        module_inputs = module_inputs.to(original_parameter.device)
    with torch.no_grad():
        if isinstance(replacement, PatchMapReplacement):
            if not isinstance(original, nn.Conv2d):
                raise TypeError(
                    f"{record.path!r}: patch wrapper over non-conv original"
                )
            outputs = original(module_inputs).cpu()
            flat_targets = (
                outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1]).float()
            )
            if replacement.bias is not None:
                flat_targets = flat_targets - replacement.bias.detach().cpu().float()
            patches = F.unfold(
                module_inputs,
                kernel_size=replacement.kernel_size,
                dilation=replacement.dilation,
                padding=replacement.padding,
                stride=replacement.stride,
            )
            flat_inputs = (
                patches.transpose(1, 2).reshape(-1, replacement.patch_dim).cpu().float()
            )
            inner = replacement.replacement
        elif isinstance(replacement, ChannelMapReplacement):
            if not isinstance(original, nn.Conv2d):
                raise TypeError(
                    f"{record.path!r}: channel-map wrapper over non-conv original"
                )
            outputs = original(module_inputs).cpu()
            flat_targets = (
                outputs.permute(0, 2, 3, 1).reshape(-1, outputs.shape[1]).float()
            )
            strided = module_inputs
            if replacement.stride != (1, 1):
                strided = strided[
                    :, :, :: replacement.stride[0], :: replacement.stride[1]
                ]
            flat_inputs = (
                strided.permute(0, 2, 3, 1)
                .reshape(-1, replacement.input_channels)
                .cpu()
                .float()
            )
        elif isinstance(original, nn.Linear):
            flat_inputs = (
                module_inputs.reshape(-1, module_inputs.shape[-1]).cpu().float()
            )
            flat_targets = (
                original(module_inputs).reshape(-1, original.out_features).float()
            )
            inner = replacement
        else:
            raise TypeError(
                f"{record.path!r}: unsupported original module "
                f"{type(original).__name__}"
            )
    if isinstance(replacement, ChannelMapReplacement):
        inner = replacement.replacement
    if flat_inputs.shape[0] != flat_targets.shape[0]:
        raise RuntimeError(
            f"{record.path!r}: {flat_inputs.shape[0]} boundary inputs do not "
            f"align with {flat_targets.shape[0]} targets"
        )
    if flat_inputs.shape[0] > max_rows:
        flat_inputs = flat_inputs[:max_rows]
        flat_targets = flat_targets[:max_rows]
    return flat_inputs, flat_targets, inner


def selected_module_row_partition(
    flat_inputs: torch.Tensor, *, valid_fraction: float = 0.25, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Reserve validation rows before any target-conditioned initialization.

    This is a row partition, not a document/source-group independence claim.
    Its receipt binds the exact flattened input values and ordered indices so
    later fitting cannot silently change the calibration data or partition.
    """
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in (0, 1)")
    n_rows = int(flat_inputs.shape[0])
    n_valid = max(1, round(n_rows * valid_fraction))
    if n_rows - n_valid < 1:
        raise ValueError(
            f"{n_rows} calibration rows cannot support a validation partition"
        )
    permutation = torch.randperm(
        n_rows, generator=torch.Generator().manual_seed(int(seed))
    )
    valid_index, train_index = permutation[:n_valid], permutation[n_valid:]
    digest = hashlib.sha256()
    digest.update(str((tuple(flat_inputs.shape), str(flat_inputs.dtype))).encode())
    # Bounded transfers/copies even for large captured transformer boundaries.
    row_bytes = max(1, flat_inputs[0].numel() * flat_inputs.element_size())
    chunk_rows = max(1, (16 * 1024 * 1024) // row_bytes)
    for start in range(0, n_rows, chunk_rows):
        block = flat_inputs[start : start + chunk_rows].detach().contiguous().cpu()
        digest.update(memoryview(block.view(torch.uint8).numpy()).cast("B"))
    receipt = {
        "schema": "selected_module_initialization_partition/v1",
        "rows": n_rows,
        "seed": int(seed),
        "valid_fraction": float(valid_fraction),
        "input_sha256": digest.hexdigest(),
        "train_indices_sha256": hashlib.sha256(
            train_index.numpy().tobytes()
        ).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(
            valid_index.numpy().tobytes()
        ).hexdigest(),
        "claim": "disjoint flattened rows; source-group independence not established",
    }
    return train_index, valid_index, receipt


def initialize_selected_module_replacements_from_teacher_(
    records: Sequence[SelectedModuleReplacementRecord],
    module_inputs: Mapping[str, torch.Tensor],
    *,
    metric: str = "activation_weighted",
    max_rows: int = 4096,
    ridge: float = 1e-4,
    row_chunk_size: int = DEFAULT_TOPOLOGY_ROW_CHUNK_SIZE,
    valid_fraction: float = 0.25,
    seed: int = 0,
    partition_max_rows: int = 8192,
) -> dict[str, dict[str, Any]]:
    """Teacher-condition every record's inner cell from captured inputs.

    Fails closed on a record without captured inputs rather than silently
    leaving that cell at its random topology while the rest are conditioned.
    Validation is reserved first using the trainer's row budget, fraction and
    seed. ``max_rows`` then caps initialization within the training partition.
    """

    diagnostics: dict[str, dict[str, Any]] = {}
    for record in records:
        inputs = module_inputs.get(record.path)
        if inputs is None:
            raise KeyError(
                f"no captured inputs for {record.path!r}; capture_module_inputs "
                "must cover every record being initialized"
            )
        flat_inputs, flat_targets, inner = selected_module_boundary_pairs(
            record, inputs, max_rows=partition_max_rows
        )
        train_index, _, partition = selected_module_row_partition(
            flat_inputs, valid_fraction=valid_fraction, seed=seed
        )
        train_index = train_index[:max_rows]
        previous = record.selection_manifest.get("initialization_partition")
        if previous is not None and previous != partition:
            raise ValueError(
                f"{record.path!r}: initialization partition changed; rebuild the replacement"
            )
        diagnostics[record.path] = initialize_population_topology_from_targets_(
            inner,
            flat_inputs[train_index],
            flat_targets[train_index],
            metric=metric,
            max_rows=max_rows,
            ridge=ridge,
            row_chunk_size=row_chunk_size,
        )
        record.selection_manifest["initialization_partition"] = partition
        diagnostics[record.path]["initialization_partition"] = partition
        diagnostics[record.path]["initialization_rows"] = int(train_index.numel())
    return diagnostics


__all__ = [
    "capture_module_inputs",
    "initialize_selected_module_replacements_from_teacher_",
    "selected_module_boundary_pairs",
    "selected_module_row_partition",
]
