"""Local distillation for selected-module replacements.

The named-module patcher places compiled cells at Linear, 1x1-conv, and
KxK-conv boundaries; ``selected_module_init`` teacher-conditions them from
captured inputs.  This trainer closes the loop: each record's replacement is
fitted locally against its original module's outputs on the same captured
calibration inputs, with a held-out slice for validation-best restoration.

Scope: this is the LOCAL fit stage (the vision/CNN counterpart of layerwise
distillation).  End-to-end recovery of a fully patched model remains the
task-level trainers' job.  House rules carried over from the transformer
loops: the train/validation split is disjoint and seeded, indexed-rewire
topology updates advance once per optimizer step, and the validation-best
state is restored before returning.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from dendritic_modeling.networks.architectures.replacement.module_patching import (
    PatchMapReplacement,
    SelectedModuleReplacementRecord,
)
from dendritic_modeling.networks.architectures.replacement.selected_module_init import (
    selected_module_boundary_pairs,
    selected_module_row_partition,
)
from dendritic_modeling.training.optimizers.custom import (
    apply_sparse_topology_updates_after_step,
)
from dendritic_modeling.training.replacement_common import _compute_distillation_loss


def _record_forward(
    record: SelectedModuleReplacementRecord,
    inner: torch.nn.Module,
    flat_inputs: torch.Tensor,
) -> torch.Tensor:
    prediction = inner(flat_inputs)
    replacement = record.replacement
    if isinstance(replacement, PatchMapReplacement) and replacement.bias is not None:
        prediction = prediction + replacement.bias
    return prediction


def train_selected_module_replacements_(
    records: Sequence[SelectedModuleReplacementRecord],
    module_inputs: Mapping[str, torch.Tensor],
    *,
    steps: int = 300,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    valid_fraction: float = 0.25,
    eval_every: int = 25,
    loss_name: str = "relative_mse",
    max_rows: int = 8192,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Locally fit every record against its original module's outputs.

    Returns one history per path: initial/best/final validation loss, the
    best step, and the per-step train losses.  The replacement is left at its
    validation-best state.
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in (0, 1)")
    histories: dict[str, dict[str, Any]] = {}
    for record in records:
        inputs = module_inputs.get(record.path)
        if inputs is None:
            raise KeyError(
                f"no captured inputs for {record.path!r}; every trained record "
                "needs calibration inputs"
            )
        flat_inputs, cell_targets, inner = selected_module_boundary_pairs(
            record, inputs, max_rows=max_rows
        )
        replacement = record.replacement
        targets = cell_targets
        if (
            isinstance(replacement, PatchMapReplacement)
            and replacement.bias is not None
        ):
            # The pairs helper subtracts the INITIAL bias so the inner cell's
            # teacher-init sees the linear part; training instead optimizes
            # the cell and the bias jointly against the raw teacher outputs.
            targets = cell_targets + replacement.bias.detach().cpu().float()

        train_index, valid_index, partition = selected_module_row_partition(
            flat_inputs, valid_fraction=valid_fraction, seed=seed
        )
        initialization_partition = record.selection_manifest.get(
            "initialization_partition"
        )
        if (
            initialization_partition is not None
            and initialization_partition != partition
        ):
            raise ValueError(
                f"{record.path!r}: training partition differs from supervised initialization; "
                "use the same inputs, seed, valid_fraction, and row budget, or rebuild the replacement"
            )

        parameter = next(replacement.parameters())
        device = parameter.device
        train_inputs = flat_inputs[train_index].to(device)
        train_targets = targets[train_index].to(device)
        valid_inputs = flat_inputs[valid_index].to(device)
        valid_targets = targets[valid_index].to(device)

        optimizer = torch.optim.AdamW(
            replacement.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )

        def evaluate(
            *,
            record=record,
            replacement=replacement,
            inner=inner,
            valid_inputs=valid_inputs,
            valid_targets=valid_targets,
        ) -> float:
            replacement.eval()
            with torch.no_grad():
                prediction = _record_forward(record, inner, valid_inputs)
                loss = _compute_distillation_loss(
                    prediction.float(),
                    valid_targets,
                    loss_name=loss_name,
                    error_context=record.path,
                )
            return float(loss.item())

        initial_valid = evaluate()
        best_valid = initial_valid
        best_step = 0
        best_state = copy.deepcopy(replacement.state_dict())
        train_losses: list[float] = []
        valid_losses: list[float] = [initial_valid]

        batch_generator = torch.Generator().manual_seed(int(seed) + 1)
        for step in range(1, int(steps) + 1):
            replacement.train()
            optimizer.zero_grad(set_to_none=True)
            batch = torch.randint(
                0,
                train_inputs.shape[0],
                (min(int(batch_size), int(train_inputs.shape[0])),),
                generator=batch_generator,
            )
            prediction = _record_forward(record, inner, train_inputs[batch])
            loss = _compute_distillation_loss(
                prediction.float(),
                train_targets[batch],
                loss_name=loss_name,
                error_context=record.path,
            )
            loss.backward()
            optimizer.step()
            apply_sparse_topology_updates_after_step(replacement, optimizer)
            train_losses.append(float(loss.item()))
            if step % int(eval_every) == 0 or step == int(steps):
                valid = evaluate()
                valid_losses.append(valid)
                if valid < best_valid:
                    best_valid = valid
                    best_step = step
                    best_state = copy.deepcopy(replacement.state_dict())

        replacement.load_state_dict(best_state)
        histories[record.path] = {
            "initial_valid_loss": initial_valid,
            "best_valid_loss": best_valid,
            "best_step": best_step,
            "final_valid_loss": valid_losses[-1],
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_rows": int(train_inputs.shape[0]),
            "valid_rows": int(valid_inputs.shape[0]),
            "initialization_partition": initialization_partition,
            "validation_partition": partition,
        }
    return histories


def jointly_recover_selected_module_replacements_(
    model: torch.nn.Module,
    records: Sequence[SelectedModuleReplacementRecord],
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    steps: int = 600,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    valid_fraction: float = 0.25,
    eval_every: int = 25,
    seed: int = 0,
) -> dict[str, Any]:
    """Jointly recover every replacement against precomputed teacher logits.

    Local fits do not compose: the measured six-encoder ViT replacement
    retained 79.1% of teacher top-1 where independent compounding of the
    single-encoder retention predicted ~93.6% (2026-08-24) — errors interact
    super-linearly through depth, exactly as in the transformer stacks.  This
    is the vision counterpart of the transformer joint-recovery stage: one
    optimizer over every record's replacement parameters, forward through the
    fully patched model, KL against the TEACHER's logits captured before
    replacement.  The backbone stays in eval mode (frozen normalization and
    dropout statistics) and its parameters receive no gradient; sparse
    topology stays frozen during recovery, matching the transformer joint
    recovery's frozen-export contract.

    ``batches`` holds ``(images, teacher_logits)`` pairs on CPU; a seeded
    split holds out validation batches, and every replacement is left at the
    joint validation-best state.
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in (0, 1)")
    n_batches = len(batches)
    n_valid = max(1, round(n_batches * valid_fraction))
    if n_batches - n_valid < 1:
        raise ValueError(
            f"{n_batches} recovery batches cannot support a "
            f"{valid_fraction:.2f} validation fraction"
        )
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(n_batches, generator=generator).tolist()
    valid_batches = [batches[i] for i in permutation[:n_valid]]
    train_batches = [batches[i] for i in permutation[n_valid:]]

    parameters = [p for record in records for p in record.replacement.parameters()]
    device = parameters[0].device
    optimizer = torch.optim.AdamW(
        parameters, lr=float(lr), weight_decay=float(weight_decay)
    )

    def _kd_loss(images: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        student_logits = model(images.to(device))
        return torch.nn.functional.kl_div(
            torch.log_softmax(student_logits.float(), dim=-1),
            torch.softmax(teacher_logits.to(device).float(), dim=-1),
            reduction="batchmean",
        )

    model.eval()

    def evaluate() -> float:
        for record in records:
            record.replacement.eval()
        with torch.no_grad():
            total = 0.0
            for images, teacher_logits in valid_batches:
                total += float(_kd_loss(images, teacher_logits).item())
        return total / len(valid_batches)

    def snapshot() -> dict[str, Any]:
        return {
            record.path: copy.deepcopy(record.replacement.state_dict())
            for record in records
        }

    initial_valid = evaluate()
    best_valid = initial_valid
    best_step = 0
    best_state = snapshot()
    train_losses: list[float] = []
    valid_losses: list[float] = [initial_valid]

    batch_generator = torch.Generator().manual_seed(int(seed) + 1)
    for step in range(1, int(steps) + 1):
        for record in records:
            record.replacement.train()
        optimizer.zero_grad(set_to_none=True)
        pick = int(
            torch.randint(0, len(train_batches), (1,), generator=batch_generator)
        )
        images, teacher_logits = train_batches[pick]
        loss = _kd_loss(images, teacher_logits)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))
        if step % int(eval_every) == 0 or step == int(steps):
            valid = evaluate()
            valid_losses.append(valid)
            if valid < best_valid:
                best_valid = valid
                best_step = step
                best_state = snapshot()

    for record in records:
        record.replacement.load_state_dict(best_state[record.path])
        record.replacement.eval()
    return {
        "initial_valid_loss": initial_valid,
        "best_valid_loss": best_valid,
        "best_step": best_step,
        "final_valid_loss": valid_losses[-1],
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "train_batches": len(train_batches),
        "valid_batches": len(valid_batches),
    }


def jointly_recover_selected_module_features_(
    model: torch.nn.Module,
    records: Sequence[SelectedModuleReplacementRecord],
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    steps: int = 600,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    valid_fraction: float = 0.25,
    eval_every: int = 25,
    patch_weight: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Jointly recover every replacement against precomputed teacher FEATURES.

    Feature backbones (DINOv2 and kin) have no classifier head, so the logits
    KD recovery above cannot apply — and skipping joint recovery is measured
    to collapse the features (the quarantined 2026-08 DINOv2 six-encoder
    screen retained a CLS cosine of 0.062 from local fits alone).  This is
    the feature-space counterpart: one optimizer over every record's
    replacement parameters, forward through the fully patched model, and a
    combined cosine objective against the INTACT teacher's token features
    captured before replacement::

        loss = mean(1 - cos(CLS_student, CLS_teacher))
             + patch_weight * mean(1 - cos(patch_student, patch_teacher))

    ``model(images)`` must return the full token features ``(batch, tokens,
    dim)`` with token 0 the CLS token, and each batch is ``(images,
    teacher_features)`` with the teacher features in that same shape — a
    2-D target means classifier logits and fails closed toward
    ``jointly_recover_selected_module_replacements_``.  Teacher features stay
    on CPU and move to the compute device one batch at a time.  Discipline is
    identical to the logits path: the backbone stays in eval mode and its
    parameters receive no gradient, sparse topology stays frozen during
    recovery, the batch split is seeded and disjoint, and every replacement
    is left at the joint validation-best state.
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in (0, 1)")
    if patch_weight < 0.0:
        raise ValueError("patch_weight must be >= 0")
    for images, teacher_features in batches:
        if teacher_features.ndim != 3:
            raise ValueError(
                "teacher features must be (batch, tokens, dim) token features "
                f"from the intact backbone, got {tuple(teacher_features.shape)}"
                " — 2-D targets are classifier logits; use "
                "jointly_recover_selected_module_replacements_ for those "
                "(fail closed rather than misreading the objective)"
            )
        if int(images.shape[0]) != int(teacher_features.shape[0]):
            raise ValueError(
                f"images and teacher features disagree on batch size: "
                f"{tuple(images.shape)} vs {tuple(teacher_features.shape)}"
            )
    n_batches = len(batches)
    n_valid = max(1, round(n_batches * valid_fraction))
    if n_batches - n_valid < 1:
        raise ValueError(
            f"{n_batches} recovery batches cannot support a "
            f"{valid_fraction:.2f} validation fraction"
        )
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(n_batches, generator=generator).tolist()
    valid_batches = [batches[i] for i in permutation[:n_valid]]
    train_batches = [batches[i] for i in permutation[n_valid:]]

    parameters = [p for record in records for p in record.replacement.parameters()]
    device = parameters[0].device
    optimizer = torch.optim.AdamW(
        parameters, lr=float(lr), weight_decay=float(weight_decay)
    )

    def _feature_loss(
        images: torch.Tensor, teacher_features: torch.Tensor
    ) -> torch.Tensor:
        student = model(images.to(device)).float()
        teacher = teacher_features.to(device).float()
        if student.shape != teacher.shape:
            raise ValueError(
                f"student features {tuple(student.shape)} do not match the "
                f"precomputed teacher features {tuple(teacher.shape)}: the "
                "model passed here must return the full (batch, tokens, dim) "
                "token features, not a CLS-only or logits view (fail closed)"
            )
        cls_cosine = torch.nn.functional.cosine_similarity(
            student[:, 0], teacher[:, 0], dim=-1
        )
        loss = (1.0 - cls_cosine).mean()
        if student.shape[1] > 1 and patch_weight > 0.0:
            patch_cosine = torch.nn.functional.cosine_similarity(
                student[:, 1:], teacher[:, 1:], dim=-1
            )
            loss = loss + float(patch_weight) * (1.0 - patch_cosine).mean()
        return loss

    model.eval()

    def evaluate() -> float:
        for record in records:
            record.replacement.eval()
        with torch.no_grad():
            total = 0.0
            for images, teacher_features in valid_batches:
                total += float(_feature_loss(images, teacher_features).item())
        return total / len(valid_batches)

    def snapshot() -> dict[str, Any]:
        return {
            record.path: copy.deepcopy(record.replacement.state_dict())
            for record in records
        }

    initial_valid = evaluate()
    best_valid = initial_valid
    best_step = 0
    best_state = snapshot()
    train_losses: list[float] = []
    valid_losses: list[float] = [initial_valid]

    batch_generator = torch.Generator().manual_seed(int(seed) + 1)
    for step in range(1, int(steps) + 1):
        for record in records:
            record.replacement.train()
        optimizer.zero_grad(set_to_none=True)
        pick = int(
            torch.randint(0, len(train_batches), (1,), generator=batch_generator)
        )
        images, teacher_features = train_batches[pick]
        loss = _feature_loss(images, teacher_features)
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.item()))
        if step % int(eval_every) == 0 or step == int(steps):
            valid = evaluate()
            valid_losses.append(valid)
            if valid < best_valid:
                best_valid = valid
                best_step = step
                best_state = snapshot()

    for record in records:
        record.replacement.load_state_dict(best_state[record.path])
        record.replacement.eval()
    return {
        "objective": "feature_cosine",
        "patch_weight": float(patch_weight),
        "initial_valid_loss": initial_valid,
        "best_valid_loss": best_valid,
        "best_step": best_step,
        "final_valid_loss": valid_losses[-1],
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "train_batches": len(train_batches),
        "valid_batches": len(valid_batches),
    }


__all__ = [
    "jointly_recover_selected_module_features_",
    "jointly_recover_selected_module_replacements_",
    "train_selected_module_replacements_",
]
