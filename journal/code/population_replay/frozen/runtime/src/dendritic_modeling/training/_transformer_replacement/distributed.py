"""Small torchrun/DDP runtime for transformer replacement experiments."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger(__name__)


_TOPOLOGY_CONTROL_ATTRIBUTES = (
    "K",
    "_rewire_seed",
    "_rewire_step_host",
    "freeze_connectivity",
    "indexed_rewire_frequency",
    "indexed_rewire_quantile",
    "indexed_rewire_until_step",
    "rewire_frequency",
    "rewire_init_policy",
    "rewire_quantile",
    "rewire_until_step",
    "topk_type",
)


@dataclass
class TransformerDistributedContext:
    """Resolved process topology for one transformer replacement worker."""

    mode: str = "none"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device | None = None
    initialized_here: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode == "ddp"

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()
            self.initialized_here = False


def initialize_transformer_distributed(
    train_cfg: Any,
) -> TransformerDistributedContext:
    """Initialize a process group when ``distributed_mode: ddp`` is requested."""
    mode = str(getattr(train_cfg, "distributed_mode", "none") or "none").lower()
    if mode in {"", "none", "off", "false"}:
        return TransformerDistributedContext()
    if mode != "ddp":
        raise ValueError(
            "transformer replacement distributed_mode must be 'none' or 'ddp'; "
            "full-model FSDP is a separate scale gate"
        )
    required = [
        name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ
    ]
    if required:
        raise RuntimeError(
            "distributed_mode='ddp' requires torchrun environment variables; "
            f"missing {', '.join(required)}"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise RuntimeError(f"Invalid torchrun rank/world size: {rank}/{world_size}")

    requested_device = str(getattr(train_cfg, "device", "auto") or "auto").lower()
    force_cpu = requested_device == "cpu"
    if torch.cuda.is_available() and not force_cpu:
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local_rank} is not a visible CUDA device")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        default_backend = "nccl"
    else:
        device = torch.device("cpu")
        default_backend = "gloo"

    initialized_here = False
    if not dist.is_initialized():
        backend = str(getattr(train_cfg, "distributed_backend", "") or default_backend)
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True
    actual_rank = dist.get_rank()
    actual_world_size = dist.get_world_size()
    if actual_rank != rank or actual_world_size != world_size:
        if initialized_here:
            dist.destroy_process_group()
        raise RuntimeError(
            "Initialized process-group topology does not match torchrun environment"
        )
    return TransformerDistributedContext(
        mode="ddp",
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        initialized_here=initialized_here,
    )


def wrap_transformer_ddp(
    module: nn.Module,
    train_cfg: Any,
    device: torch.device,
) -> nn.Module:
    """Wrap a trainable module in DDP when configured."""
    mode = str(getattr(train_cfg, "distributed_mode", "none") or "none").lower()
    if mode != "ddp":
        return module
    if not dist.is_initialized():
        raise RuntimeError("DDP wrapping requested before process-group initialization")
    kwargs: dict[str, Any] = {
        "broadcast_buffers": False,
        "find_unused_parameters": bool(
            getattr(train_cfg, "ddp_find_unused_parameters", False)
        ),
    }
    if device.type == "cuda":
        kwargs.update(device_ids=[device.index], output_device=device.index)
    return DistributedDataParallel(module, **kwargs)


def transformer_process_rank() -> int:
    """Return the active distributed rank without requiring initialization."""
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", 0))


def transformer_process_world_size() -> int:
    """Return the active process count without requiring initialization."""
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    return int(os.environ.get("WORLD_SIZE", 1))


def is_transformer_main_process() -> bool:
    return transformer_process_rank() == 0


def transformer_distributed_mean(value: float, device: torch.device) -> float:
    """Average a scalar across the active process group for rank-0 reporting."""
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    scalar = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
    scalar /= dist.get_world_size()
    return float(scalar.item())


def transformer_distributed_sum_count_mean(
    total: float,
    count: int,
    device: torch.device,
) -> float:
    """Return a true global mean from a local sum and observation count."""

    if int(count) < 0:
        raise ValueError("distributed observation count cannot be negative")
    if not dist.is_available() or not dist.is_initialized():
        return float(total) / max(int(count), 1)
    pair = torch.tensor(
        [float(total), float(count)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(pair, op=dist.ReduceOp.SUM)
    global_count = int(pair[1].item())
    if global_count < 1:
        raise RuntimeError("distributed mean has no observations")
    return float((pair[0] / pair[1]).item())


def transformer_distributed_gather_row(
    values: list[float] | tuple[float, ...],
    device: torch.device,
) -> list[list[float]]:
    """Gather one fixed-width numeric row from every rank in rank order."""

    local = torch.tensor(list(values), device=device, dtype=torch.float64)
    if not dist.is_available() or not dist.is_initialized():
        return [[float(value) for value in local.cpu().tolist()]]
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [[float(value) for value in row.detach().cpu().tolist()] for row in gathered]


def transformer_distributed_sum_tensor(
    value: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sum a detached tensor across ranks and return it on the CPU."""

    result = value.detach().to(device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result.cpu()


def transformer_module_state_digest(
    modules: Sequence[nn.Module],
) -> dict[str, Any]:
    """Hash tensor state and latent topology controls of unique modules exactly."""

    digest = hashlib.sha256()
    seen: dict[int, int] = {}
    alias_layout: list[int] = []
    tensor_count = 0
    total_bytes = 0
    topology_controls: list[dict[str, Any]] = []
    for module in modules:
        identity = id(module)
        physical_index = seen.get(identity)
        if physical_index is None:
            physical_index = len(seen)
            seen[identity] = physical_index
            state = module.state_dict()
            for name in sorted(state):
                tensor = state[name]
                if not torch.is_tensor(tensor) or tensor.layout != torch.strided:
                    raise TypeError(
                        "DDP state digest requires strided tensor state entries; "
                        f"module {physical_index} entry {name!r} is unsupported"
                    )
                cpu = tensor.detach().cpu().contiguous()
                metadata = json.dumps(
                    {
                        "module": physical_index,
                        "module_type": (
                            f"{type(module).__module__}.{type(module).__qualname__}"
                        ),
                        "name": name,
                        "dtype": str(cpu.dtype),
                        "shape": list(cpu.shape),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
                digest.update(len(metadata).to_bytes(8, "big"))
                digest.update(metadata)
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
                tensor_count += 1
                total_bytes += len(raw)
            for submodule_path, submodule in module.named_modules(
                remove_duplicate=True
            ):
                controls: dict[str, bool | float | int | str | None] = {}
                for attribute in _TOPOLOGY_CONTROL_ATTRIBUTES:
                    if not hasattr(submodule, attribute):
                        continue
                    value = getattr(submodule, attribute)
                    if value is None or isinstance(value, (bool, float, int, str)):
                        controls[attribute] = value
                    else:
                        raise TypeError(
                            "DDP topology control must be a JSON scalar; "
                            f"module {physical_index} path {submodule_path!r} "
                            f"attribute {attribute!r} has type {type(value).__name__}"
                        )
                if controls:
                    topology_controls.append(
                        {
                            "module": physical_index,
                            "path": submodule_path,
                            "module_type": (
                                f"{type(submodule).__module__}."
                                f"{type(submodule).__qualname__}"
                            ),
                            "controls": controls,
                        }
                    )
        alias_layout.append(physical_index)
    alias_payload = json.dumps(alias_layout, separators=(",", ":")).encode("utf-8")
    digest.update(len(alias_payload).to_bytes(8, "big"))
    digest.update(alias_payload)
    topology_payload = json.dumps(
        topology_controls,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(topology_payload).to_bytes(8, "big"))
    digest.update(topology_payload)
    return {
        "sha256": digest.hexdigest(),
        "physical_module_count": len(seen),
        "logical_module_count": len(modules),
        "alias_layout": alias_layout,
        "tensor_count": tensor_count,
        "tensor_bytes": total_bytes,
        "topology_control_module_count": len(topology_controls),
        "topology_control_sha256": hashlib.sha256(topology_payload).hexdigest(),
        "scope": "parameters_persistent_buffers_and_topology_controls",
    }


def validate_transformer_state_digest_records(
    records: Sequence[dict[str, Any]],
    *,
    stage: str,
) -> dict[str, Any]:
    """Fail closed unless every rank reports the same exact state digest."""

    if not records:
        raise RuntimeError("DDP state digest gathered no rank records")
    required = {
        "rank",
        "sha256",
        "physical_module_count",
        "logical_module_count",
        "alias_layout",
        "tensor_count",
        "tensor_bytes",
        "topology_control_module_count",
        "topology_control_sha256",
        "scope",
    }
    if any(set(record) != required for record in records):
        raise RuntimeError("DDP state digest rank record schema differs")
    ordered = sorted(records, key=lambda record: int(record["rank"]))
    expected_ranks = list(range(len(ordered)))
    if [int(record["rank"]) for record in ordered] != expected_ranks:
        raise RuntimeError("DDP state digest ranks are incomplete or duplicated")
    comparable = [
        {key: value for key, value in record.items() if key != "rank"}
        for record in ordered
    ]
    if any(record != comparable[0] for record in comparable[1:]):
        summary = ", ".join(
            f"rank{record['rank']}={str(record['sha256'])[:12]}" for record in ordered
        )
        raise RuntimeError(
            f"DDP replacement state/topology divergence at {stage}: {summary}"
        )
    return {
        "stage": str(stage),
        "world_size": len(ordered),
        "all_ranks_exact": True,
        **comparable[0],
    }


def assert_transformer_distributed_module_state_equal(
    modules: Sequence[nn.Module],
    *,
    stage: str,
) -> dict[str, Any]:
    """Cryptographically verify replacement state equality across all ranks."""

    local = {
        "rank": transformer_process_rank(),
        **transformer_module_state_digest(modules),
    }
    if not dist.is_available() or not dist.is_initialized():
        gathered = [local]
    else:
        gathered: list[dict[str, Any] | None] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered, local)
        if any(record is None for record in gathered):
            raise RuntimeError("DDP state digest collective returned an empty record")
        gathered = [record for record in gathered if record is not None]
    result = validate_transformer_state_digest_records(gathered, stage=stage)
    if transformer_process_rank() == 0:
        logger.info(
            "transformer DDP state digest stage=%s sha256=%s tensors=%d bytes=%d",
            stage,
            result["sha256"],
            result["tensor_count"],
            result["tensor_bytes"],
        )
    return result


def assert_transformer_distributed_finite_row(
    values: Sequence[float],
    device: torch.device,
    *,
    stage: str,
) -> None:
    """Fail every rank when any worker reports a non-finite scalar."""

    local_finite = int(all(math.isfinite(float(value)) for value in values))
    status = torch.tensor(local_finite, device=device, dtype=torch.uint8)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if int(status.item()) != 1:
        raise FloatingPointError(f"non-finite distributed scalar at {stage}")


__all__ = [
    "TransformerDistributedContext",
    "assert_transformer_distributed_finite_row",
    "assert_transformer_distributed_module_state_equal",
    "initialize_transformer_distributed",
    "is_transformer_main_process",
    "transformer_distributed_gather_row",
    "transformer_distributed_mean",
    "transformer_distributed_sum_count_mean",
    "transformer_distributed_sum_tensor",
    "transformer_module_state_digest",
    "transformer_process_rank",
    "transformer_process_world_size",
    "validate_transformer_state_digest_records",
    "wrap_transformer_ddp",
]
