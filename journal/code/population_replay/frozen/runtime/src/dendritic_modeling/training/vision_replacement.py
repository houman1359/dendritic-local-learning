"""Training helpers for vision-backbone dendritic replacements."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.config import Config
from dendritic_modeling.config.compression import ModelCompressionConfig
from dendritic_modeling.deployment import (
    freeze_sparse_topology_,
    prepare_model_for_compact_state_,
)
from dendritic_modeling.deployment.ledger import (
    deployment_storage_manifest,
    module_storage_ledger,
)
from dendritic_modeling.networks.architectures.replacement import (
    initialize_population_topology_from_targets_,
    supports_teacher_topology_initialization,
)
from dendritic_modeling.networks.checkpoints import (
    atomic_torch_save_candidates,
    compact_model_state_dict_candidates,
    decode_sparse_bitmask_state_dict,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training.optimizers import create_optimizer
from dendritic_modeling.training.optimizers.custom import (
    apply_sparse_topology_updates_after_scaled_step,
)
from dendritic_modeling.training.replacement_common import (
    ReplacementTrainingHistory,
    _capture_requires_grad_states,
    _clip_replacement_grad_norm,
    _compute_vision_distillation_loss as _compute_distillation_loss,
    _infinite_loader,
    _load_required_tensor,
    _make_cuda_grad_scaler,
    _make_replacement_loader,
    _move_replacement_batch_tensors,
    _resolve_device,
    _resolve_dtype,
    _resolve_replacement_loader_tuning,
    _restore_requires_grad_states,
    _should_run_periodic_step,
    _to_plain_mapping,
    _write_metrics_json,
)

logger = logging.getLogger(__name__)


@dataclass
class VisionReplacementTrainingResult:
    """Summary returned by vision replacement distillation."""

    train_losses: list[float]
    valid_losses: list[float]
    initial_valid_loss: float
    final_valid_loss: float
    best_valid_loss: float
    best_step: int
    save_dir: str
    train_target: str
    teacher_source: str
    replacement_stored_bytes: int | None = None
    replacement_index_bytes: int | None = None
    storage_scope: str = "runtime_state_before_export"

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _load_tensor(path: str) -> torch.Tensor:
    return _load_required_tensor(
        path,
        empty_path_message="Vision replacement tensor-cache path is empty",
        type_error_message="Vision replacement tensor cache entries must be tensors",
    )


def _build_tensor_datasets(config: Config) -> tuple[TensorDataset, TensorDataset]:
    train_cfg = config.training.vision_replacement
    cache_cfg = _to_plain_mapping(train_cfg.tensor_cache)
    train_inputs = _load_tensor(str(cache_cfg.get("train_inputs", ""))).float()
    train_targets = _load_tensor(str(cache_cfg.get("train_targets", ""))).float()
    valid_inputs = _load_tensor(str(cache_cfg.get("valid_inputs", ""))).float()
    valid_targets = _load_tensor(str(cache_cfg.get("valid_targets", ""))).float()
    return (
        TensorDataset(train_inputs, train_targets),
        TensorDataset(valid_inputs, valid_targets),
    )


def _unique_trainable_parameters(modules: Sequence[nn.Module]) -> list[nn.Parameter]:
    trainable: list[nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for param in module.parameters():
            param.requires_grad_(True)
            param_id = id(param)
            if param_id not in seen:
                seen.add(param_id)
                trainable.append(param)
    if not trainable:
        raise ValueError("No vision replacement parameters are trainable")
    return trainable


def _select_vision_trainable_parameters(
    model: nn.Module,
    train_target: str,
) -> list[nn.Parameter]:
    """Select trainable parameters for vision replacement distillation."""
    normalized = str(train_target).lower()
    if normalized in {"replacement_only", "core_only", "dendritic_only"}:
        states = _capture_requires_grad_states(model.parameters())
        try:
            for param in model.parameters():
                param.requires_grad_(False)
            return _unique_trainable_parameters([model.core_network])
        except Exception:
            _restore_requires_grad_states(states)
            raise
    if normalized in {"replacement_and_decoder", "core_decoder"}:
        states = _capture_requires_grad_states(model.parameters())
        try:
            for param in model.parameters():
                param.requires_grad_(False)
            return _unique_trainable_parameters(
                [model.core_network, model.decoder_network]
            )
        except Exception:
            _restore_requires_grad_states(states)
            raise
    if normalized in {"full_model", "all", "student"}:
        for param in model.parameters():
            param.requires_grad_(True)
        params = [param for param in model.parameters() if param.requires_grad]
        if not params:
            raise ValueError("No vision model parameters are trainable")
        return params
    raise ValueError(
        "vision replacement train_target must be 'replacement_only', "
        "'replacement_and_decoder', or 'full_model'"
    )


def _forward_source(
    model: nn.Module,
    teacher_source: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    normalized = str(teacher_source).lower()
    if normalized == "tensor_cache":
        return model.core_network
    if normalized == "image_tensor_cache":
        return model
    raise ValueError(
        "vision replacement teacher_source must be 'tensor_cache' or "
        "'image_tensor_cache'"
    )


def _validate_source_train_target(teacher_source: str, train_target: str) -> None:
    if str(teacher_source).lower() != "tensor_cache":
        return
    if str(train_target).lower() not in {
        "replacement_only",
        "core_only",
        "dendritic_only",
    }:
        raise ValueError(
            "teacher_source='tensor_cache' contains inputs to the replacement "
            "core, so it supports only train_target='replacement_only'. Use "
            "teacher_source='image_tensor_cache' for full-model distillation."
        )


def _make_vision_loader(
    dataset: torch.utils.data.Dataset,
    train_cfg: Any,
    *,
    device: torch.device,
    shuffle: bool,
    log_tuning: bool = False,
) -> DataLoader:
    tuning = _resolve_replacement_loader_tuning(dataset, train_cfg, device=device)
    if log_tuning:
        logger.info("Vision replacement DataLoader tuning: %s", tuning.summary(dataset))
    return _make_replacement_loader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        tuning=tuning,
    )


def _evaluate_dataset(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    dataset: TensorDataset,
    *,
    train_cfg: Any,
    device: torch.device,
    dtype: torch.dtype,
    loss_name: str,
    cosine_weight: float,
) -> float:
    loader = _make_vision_loader(dataset, train_cfg, device=device, shuffle=False)
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, target in loader:
            x, target = _move_replacement_batch_tensors(
                x,
                target,
                device=device,
                dtype=dtype,
            )
            prediction = forward_fn(x)
            loss = _compute_distillation_loss(
                prediction,
                target,
                loss_name=loss_name,
                cosine_weight=cosine_weight,
            )
            total += float(loss.item())
            count += 1
    return total / max(count, 1)


def _save_result(
    result: VisionReplacementTrainingResult,
    model: nn.Module,
    *,
    save_replacement: bool,
    freeze_sparse_topology_on_export: bool = False,
    topology_encoding: str = "raw",
    stochastic_topology_freeze_policy: str = "reject",
    stochastic_topology_freeze_seed: int = 0,
    ragged_topology_format: str = "csr",
) -> None:
    if not save_replacement:
        ledger = module_storage_ledger(model.core_network)
        result.replacement_stored_bytes = int(ledger["total_bytes"])
        result.replacement_index_bytes = int(ledger["index_bytes"])
        _write_metrics_json(result)
        return
    normalized_encoding = str(topology_encoding).strip().lower()
    if normalized_encoding not in {"auto", "raw", "uint", "bitmask"}:
        raise ValueError(
            "vision replacement checkpoint encoding must be auto, raw, uint, or "
            "bitmask"
        )
    topology_records = []
    if freeze_sparse_topology_on_export:
        model.core_network.eval()
        policy_encoding = (
            "uint" if normalized_encoding == "raw" else normalized_encoding
        )
        topology_records = freeze_sparse_topology_(
            model.core_network,
            ModelCompressionConfig(
                sparse_topology={
                    "topology_encoding": policy_encoding,
                    "stochastic_topk_policy": stochastic_topology_freeze_policy,
                    "stochastic_seed": int(stochastic_topology_freeze_seed),
                    "ragged_topology_format": ragged_topology_format,
                }
            ),
        )
        result.storage_scope = "runtime_state_after_fixed_topology_export"
    else:
        result.storage_scope = "runtime_state_at_checkpoint_export"
    ledger = module_storage_ledger(model.core_network)
    result.replacement_stored_bytes = int(ledger["total_bytes"])
    result.replacement_index_bytes = int(ledger["index_bytes"])
    _write_metrics_json(result)
    topology_manifest = [asdict(record) for record in topology_records]

    def compact_candidates(state_dict):
        if normalized_encoding == "raw":
            return {"raw": (dict(state_dict), {})}
        return compact_model_state_dict_candidates(
            state_dict,
            topology_encoding=normalized_encoding,
        )

    core_candidates = compact_candidates(model.core_network.state_dict())
    selection_manifests = {
        path or "core": module.selection_manifest
        for path, module in model.core_network.named_modules()
        if hasattr(module, "selection_manifest")
    }
    compiled_plans = {
        path or "core": module.compiled_replacement_plan
        for path, module in model.core_network.named_modules()
        if hasattr(module, "compiled_replacement_plan")
    }
    topology_initialization = {
        path or "core": module.teacher_topology_diagnostics
        for path, module in model.core_network.named_modules()
        if hasattr(module, "teacher_topology_diagnostics")
    }
    core_payloads = {
        encoding: {
            "schema_version": 3,
            "state_dict": core_state,
            "sparse_index_encoding": core_index_encoding,
            "sparse_topology_manifest": topology_manifest,
            "topology_encoding_requested": normalized_encoding,
            "topology_encoding_selected": encoding,
            "train_target": result.train_target,
            "teacher_source": result.teacher_source,
            "selection_manifests": selection_manifests,
            "compiled_replacement_plans": compiled_plans,
            "teacher_topology_initialization": topology_initialization,
            "deployment_storage_manifest": deployment_storage_manifest(
                model.core_network,
                core_state,
                topology_encoding=encoding,
                sparse_topology_manifest=topology_manifest,
                scope="vision_replacement_core",
            ),
            **(
                {"parameter_estimate": model.core_network.parameter_estimate()}
                if hasattr(model.core_network, "parameter_estimate")
                else {}
            ),
        }
        for encoding, (core_state, core_index_encoding) in core_candidates.items()
    }
    artifact_records: list[dict[str, Any]] = []

    def save_candidates(payloads, filename: str, scope: str) -> None:
        checkpoint_path = os.path.join(result.save_dir, filename)
        report = atomic_torch_save_candidates(payloads, checkpoint_path)
        artifact_records.append(
            {
                "scope": scope,
                "checkpoint": filename,
                "compact_checkpoint_bytes": int(report["output_bytes"]),
                "sha256": str(report["output_sha256"]),
                "topology_encoding_requested": normalized_encoding,
                "topology_encoding_selected": str(report["selected"]),
                "topology_encoding_candidate_bytes": dict(report["candidate_bytes"]),
                "selection_basis": str(report["selection_basis"]),
            }
        )

    save_candidates(core_payloads, "replacement_core.pt", "replacement_core")

    if result.train_target in {"replacement_and_decoder", "core_decoder"}:
        payloads = {
            encoding: {
                "schema_version": 3,
                "core_state_dict": core_state,
                "decoder_state_dict": model.decoder_network.state_dict(),
                "sparse_index_encoding": core_index_encoding,
                "sparse_topology_manifest": topology_manifest,
                "topology_encoding_requested": normalized_encoding,
                "topology_encoding_selected": encoding,
                "train_target": result.train_target,
                "teacher_source": result.teacher_source,
                "selection_manifests": selection_manifests,
                "compiled_replacement_plans": compiled_plans,
                "core_deployment_storage_manifest": deployment_storage_manifest(
                    model.core_network,
                    core_state,
                    topology_encoding=encoding,
                    sparse_topology_manifest=topology_manifest,
                    scope="vision_replacement_core_with_decoder",
                ),
                "decoder_runtime_storage": module_storage_ledger(model.decoder_network),
            }
            for encoding, (core_state, core_index_encoding) in core_candidates.items()
        }
        save_candidates(
            payloads,
            "replacement_with_decoder.pt",
            "replacement_with_decoder",
        )
    elif result.train_target in {"full_model", "all", "student"}:
        student_candidates = compact_candidates(model.state_dict())
        student_manifest = [
            {**record, "path": f"core_network.{record['path']}"}
            for record in topology_manifest
        ]
        payloads = {
            encoding: {
                "schema_version": 3,
                "state_dict": student_state,
                "sparse_index_encoding": student_index_encoding,
                "sparse_topology_manifest": student_manifest,
                "topology_encoding_requested": normalized_encoding,
                "topology_encoding_selected": encoding,
                "train_target": result.train_target,
                "teacher_source": result.teacher_source,
                "selection_manifests": selection_manifests,
                "compiled_replacement_plans": compiled_plans,
                "deployment_storage_manifest": deployment_storage_manifest(
                    model,
                    student_state,
                    topology_encoding=encoding,
                    sparse_topology_manifest=student_manifest,
                    scope="vision_full_student",
                ),
            }
            for encoding, (
                student_state,
                student_index_encoding,
            ) in student_candidates.items()
        }
        save_candidates(payloads, "student_model.pt", "full_student")

    with open(
        os.path.join(result.save_dir, "replacement_export_manifest.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "schema": "dendritic_replacement_artifact_manifest/v1",
                "scope": "vision_replacement",
                "artifacts": artifact_records,
                "compact_checkpoint_bytes": int(
                    sum(
                        record["compact_checkpoint_bytes"]
                        for record in artifact_records
                    )
                ),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def load_vision_replacement_core_checkpoint(
    core_network: nn.Module,
    checkpoint_path: str,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a raw or self-describing compact vision replacement core."""

    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state_dict = decode_sparse_bitmask_state_dict(payload.get("state_dict", payload))
    manifest = payload.get("sparse_topology_manifest", [])
    if manifest:
        prepare_model_for_compact_state_(core_network, manifest, state_dict)
    core_network.load_state_dict(state_dict)
    return payload


def run_vision_replacement_training(
    config: Config,
) -> VisionReplacementTrainingResult:
    """Run block-output distillation for a configured vision replacement."""
    train_cfg = config.training.vision_replacement
    if not train_cfg.enabled:
        raise ValueError("training.vision_replacement.enabled must be true")
    if str(train_cfg.mode).lower() != "blockwise_distillation":
        raise ValueError(
            "training.vision_replacement.mode must be 'blockwise_distillation'. "
            "Use training.main for supervised full-model training."
        )

    teacher_source = str(train_cfg.teacher_source).lower()
    train_target = str(train_cfg.train_target).lower()
    _validate_source_train_target(teacher_source, train_target)

    seed = int(train_cfg.seed if train_cfg.seed is not None else config.experiment.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _resolve_device(train_cfg.device)
    dtype = _resolve_dtype(train_cfg.dtype)
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        logger.warning("Using float32 for CPU vision replacement training")
        dtype = torch.float32

    train_dataset, valid_dataset = _build_tensor_datasets(config)
    model, _encoder_network = initialize_model(config.model)
    model = model.to(device=device, dtype=dtype)
    trainable_params = _select_vision_trainable_parameters(model, train_target)
    forward_fn = _forward_source(model, teacher_source)

    if teacher_source == "tensor_cache" and supports_teacher_topology_initialization(
        model.core_network
    ):
        calibration_rows = min(
            len(valid_dataset),
            max(1, int(train_cfg.batch_size))
            * max(1, int(getattr(train_cfg, "reactivation_calibration_batches", 3))),
        )
        initialize_population_topology_from_targets_(
            model.core_network,
            valid_dataset.tensors[0][:calibration_rows].to(device=device, dtype=dtype),
            valid_dataset.tensors[1][:calibration_rows].to(device=device, dtype=dtype),
            metric=str(model.core_network.teacher_support_metric),
        )

    optimizer = create_optimizer(trainable_params, config.training.main.optimizer)
    scaler = _make_cuda_grad_scaler(train_cfg.use_amp, device)
    train_loader = _make_vision_loader(
        train_dataset,
        train_cfg,
        device=device,
        shuffle=True,
        log_tuning=True,
    )
    loader = _infinite_loader(train_loader)

    initial_valid = _evaluate_dataset(
        forward_fn,
        valid_dataset,
        train_cfg=train_cfg,
        device=device,
        dtype=dtype,
        loss_name=train_cfg.loss,
        cosine_weight=train_cfg.cosine_weight,
    )
    history = ReplacementTrainingHistory.start(initial_valid)

    for step in range(1, int(train_cfg.max_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        x, target = next(loader)
        x, target = _move_replacement_batch_tensors(
            x,
            target,
            device=device,
            dtype=dtype,
        )
        model.train()
        if teacher_source == "tensor_cache":
            model.encoder_network.eval()
            model.decoder_network.eval()
        with autocast("cuda", enabled=scaler.is_enabled()):
            prediction = forward_fn(x)
            loss = _compute_distillation_loss(
                prediction,
                target,
                loss_name=train_cfg.loss,
                cosine_weight=train_cfg.cosine_weight,
            )
        scaler.scale(loss).backward()
        _clip_replacement_grad_norm(
            trainable_params,
            max_grad_norm=getattr(train_cfg, "max_grad_norm", None),
            scaler=scaler,
            optimizer=optimizer,
        )
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        apply_sparse_topology_updates_after_scaled_step(
            model.core_network,
            optimizer,
            scaler,
            scale_before=scale_before,
        )
        history.record_train_loss(float(loss.detach().item()))

        if _should_run_periodic_step(step, train_cfg.eval_every, train_cfg.max_steps):
            model.eval()
            valid = _evaluate_dataset(
                forward_fn,
                valid_dataset,
                train_cfg=train_cfg,
                device=device,
                dtype=dtype,
                loss_name=train_cfg.loss,
                cosine_weight=train_cfg.cosine_weight,
            )
            history.record_valid_loss(step, valid)
        if _should_run_periodic_step(step, train_cfg.log_every, 1):
            logger.info(
                "vision replacement distillation step=%s/%s train_loss=%.6f "
                "valid_loss=%.6f",
                step,
                train_cfg.max_steps,
                history.train_losses[-1],
                history.valid_losses[-1],
            )

    result = VisionReplacementTrainingResult(
        **history.as_result_kwargs(),
        save_dir=train_cfg.save_dir,
        train_target=train_target,
        teacher_source=teacher_source,
    )
    _save_result(
        result,
        model,
        save_replacement=train_cfg.save_replacement,
        freeze_sparse_topology_on_export=bool(
            getattr(train_cfg, "freeze_sparse_topology_on_export", False)
        ),
        topology_encoding=str(
            getattr(train_cfg, "replacement_checkpoint_encoding", "raw")
        ),
        stochastic_topology_freeze_policy=str(
            getattr(train_cfg, "stochastic_topology_freeze_policy", "reject")
        ),
        stochastic_topology_freeze_seed=int(
            getattr(train_cfg, "stochastic_topology_freeze_seed", 0)
        ),
        ragged_topology_format=str(getattr(train_cfg, "ragged_topology_format", "csr")),
    )
    return result


__all__ = [
    "VisionReplacementTrainingResult",
    "load_vision_replacement_core_checkpoint",
    "run_vision_replacement_training",
]
