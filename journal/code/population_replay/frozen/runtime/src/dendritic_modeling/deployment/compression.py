"""Lossless compaction of trained dendritic sparse topology."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.config.compression import ModelCompressionConfig
from dendritic_modeling.deployment.csr_sparse import (
    CSRInferenceLinear,
    masked_weight_to_csr,
)
from dendritic_modeling.deployment.ledger import deployment_storage_manifest
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    CreditGatedDeepstLinear,
    DeepstLinear,
    DenseToSparseLinear,
    IndexedDynamicTopKLinear,
    IndexedRewireLinear,
    IndexedSparseLinear,
    StochasticTopKLinear,
    TopKLinear,
    VarianceTopKLinear,
)
from dendritic_modeling.networks.checkpoints import (
    atomic_torch_save_candidates,
    compact_model_state_dict_candidates,
    decode_sparse_bitmask_state_dict,
    extract_model_state_dict,
    sha256_file,
)

COMPACT_CHECKPOINT_FORMAT = "dendritic_modeling.compact_checkpoint"
COMPACT_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class FrozenSparseModuleRecord:
    """Reconstruction record for one fixed indexed sparse module."""

    path: str
    source_type: str
    target_type: str
    in_features: int
    out_features: int
    synapses_per_output: int
    param_space: str
    init_method: str
    init_gain: float
    weight_transform: str
    weight_norm_order: int | None
    gamma: float
    output_chunk_size: int
    workspace_mb: float | None
    cache_transformed_weights: bool
    projection_backend: str
    support_group_rows: int
    support_col_block: int
    row_degrees: list[int] | None = None
    freeze_policy: str = "active_topology"


@dataclass(frozen=True)
class _FrozenSupport:
    """Internal frozen topology before its runtime representation is built."""

    representation: str
    indices: torch.Tensor | None
    raw_weights: torch.Tensor | None
    effective_weight: torch.Tensor
    mask: torch.Tensor
    freeze_policy: str


def _module_device_and_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.get_default_dtype()
    return parameter.device, parameter.dtype


def _sorted_selected_topology(
    indices: torch.Tensor,
    raw_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sorted_indices, order = indices.to(dtype=torch.long).sort(dim=1)
    sorted_weights = raw_weights.gather(1, order.to(raw_weights.device))
    return sorted_indices, sorted_weights


def _dense_topk_support(module: TopKLinear) -> tuple[torch.Tensor, torch.Tensor]:
    mask = module.weight_mask().to(dtype=torch.bool)
    counts = mask.sum(dim=1)
    if counts.numel() == 0 or bool((counts != counts[0]).any()):
        raise RuntimeError("TopK support must have a constant row-wise synapse count")
    k = int(counts[0].item())
    coordinates = torch.nonzero(mask, as_tuple=False)
    indices = coordinates[:, 1].reshape(module.out_features, k)
    raw_weights = module.pre_w.gather(1, indices.to(module.pre_w.device))
    return _sorted_selected_topology(indices, raw_weights)


def _support_from_mask(
    module: nn.Module,
    mask: torch.Tensor,
    *,
    effective_weight: torch.Tensor,
    freeze_policy: str,
    ragged_topology_format: str,
) -> _FrozenSupport:
    """Resolve a dense training mask to uniform indexed or ragged CSR storage."""

    active = mask.to(device=module.pre_w.device, dtype=torch.bool)
    counts = active.sum(dim=1)
    uniform = counts.numel() > 0 and bool((counts == counts[0]).all())
    if uniform and int(counts[0]) > 0:
        k = int(counts[0])
        coordinates = torch.nonzero(active, as_tuple=False)
        indices = coordinates[:, 1].reshape(module.out_features, k)
        raw_weights = module.pre_w.gather(1, indices.to(module.pre_w.device))
        indices, raw_weights = _sorted_selected_topology(indices, raw_weights)
        return _FrozenSupport(
            representation="indexed",
            indices=indices,
            raw_weights=raw_weights,
            effective_weight=effective_weight,
            mask=active,
            freeze_policy=freeze_policy,
        )
    if ragged_topology_format == "reject":
        raise ValueError(
            f"{type(module).__name__} has ragged row degrees "
            f"{counts.tolist()}, but ragged_topology_format='reject'"
        )
    return _FrozenSupport(
        representation="csr",
        indices=None,
        raw_weights=None,
        effective_weight=effective_weight,
        mask=active,
        freeze_policy=freeze_policy,
    )


def _deterministic_topk_mask(module: TopKLinear) -> torch.Tensor:
    """Select standard TopK from learned scores without stochastic noise."""

    scores = (
        module.pre_w.abs() if module.weight_transform == "identity" else module.pre_w
    )
    scores = module._apply_forbidden_scores(scores)
    indices = scores.topk(module.K, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(module.pre_w)
    rows = torch.arange(module.out_features, device=module.pre_w.device)[:, None]
    mask[rows, indices] = 1
    return module._apply_forbidden_mask(mask)


def _sample_stochastic_mask_once(
    module: StochasticTopKLinear,
    *,
    seed: int,
) -> torch.Tensor:
    devices = []
    if module.pre_w.device.type == "cuda":
        devices = [module.pre_w.device.index or torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed_all(int(seed))
        return module.weight_mask().detach()


def _dynamic_indexed_support(
    module: IndexedDynamicTopKLinear,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = module.active_sparse_mask().to(dtype=torch.bool)
    counts = active.sum(dim=1)
    if bool((counts != module.K).any()):
        raise RuntimeError(
            "Indexed dynamic support must select exactly K candidates per row"
        )
    slots = torch.nonzero(active, as_tuple=False)[:, 1].reshape(
        module.out_features, module.K
    )
    indices = module.connection_indices.gather(
        1, slots.to(module.connection_indices.device)
    )
    raw_weights = module.pre_w.gather(1, slots.to(module.pre_w.device))
    return _sorted_selected_topology(indices, raw_weights)


def _fixed_indexed_support(
    module: IndexedSparseLinear,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _sorted_selected_topology(module.connection_indices, module.pre_w)


def _replacement_options(module: nn.Module) -> dict[str, Any]:
    return {
        "param_space": str(getattr(module, "param_space", "log")),
        "init_method": str(getattr(module, "init_method", "xavier_normal")),
        "init_gain": float(getattr(module, "init_gain", 1.0)),
        "weight_transform": str(getattr(module, "weight_transform", "exp")),
        "weight_norm_order": getattr(module, "weight_norm_order", None),
        "gamma": float(getattr(module, "gamma", 1.0)),
        "output_chunk_size": int(getattr(module, "output_chunk_size", 2048)),
        "workspace_mb": getattr(module, "workspace_mb", None),
        "cache_transformed_weights": bool(
            getattr(module, "cache_transformed_weights", False)
        ),
        "projection_backend": str(getattr(module, "projection_backend", "eager")),
        "support_group_rows": int(getattr(module, "support_group_rows", 1)),
        "support_col_block": int(getattr(module, "support_col_block", 1)),
    }


def _make_indexed_replacement(
    source: nn.Module,
    indices: torch.Tensor,
    raw_weights: torch.Tensor,
    *,
    options: Mapping[str, Any] | None = None,
) -> IndexedSparseLinear:
    replacement_options = dict(options or _replacement_options(source))
    device, dtype = _module_device_and_dtype(source)
    replacement = IndexedSparseLinear(
        in_features=int(source.in_features),
        out_features=int(source.out_features),
        K=int(indices.shape[1]),
        connection_indices=indices.detach().cpu(),
        # "auto" stores int32 indices whenever in_features fits, halving index
        # memory and letting the CUDA/Triton kernels consume the buffer without
        # a per-forward dtype conversion. Serialized uint16/uint32 indices cast
        # losslessly into the runtime buffer during load_state_dict.
        index_dtype="auto",
        persistent_indices=True,
        **replacement_options,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        replacement.connection_indices.copy_(
            indices.to(device=replacement.connection_indices.device)
        )
        replacement.pre_w.copy_(raw_weights.to(device=device, dtype=dtype))
    replacement.pre_w.requires_grad_(source.pre_w.requires_grad)
    replacement.train(source.training)
    return replacement


def _verify_layer_equivalence(
    source: nn.Module,
    replacement: nn.Module,
    *,
    atol: float,
    rtol: float,
) -> None:
    device, dtype = _module_device_and_dtype(source)
    if not (dtype.is_floating_point or dtype.is_complex):
        return
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    sample = torch.randn(
        2,
        int(source.in_features),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    source_training = source.training
    replacement_training = replacement.training
    try:
        source.eval()
        replacement.eval()
        with torch.no_grad():
            expected = source(sample)
            observed = replacement(sample)
    finally:
        source.train(source_training)
        replacement.train(replacement_training)
    dtype_atol, dtype_rtol = _verification_tolerances(
        dtype,
        atol=atol,
        rtol=rtol,
        reduction_backend=getattr(
            replacement, "_last_resolved_projection_backend", None
        ),
    )
    if not torch.allclose(expected, observed, atol=dtype_atol, rtol=dtype_rtol):
        maximum_error = float((expected - observed).abs().max().item())
        raise RuntimeError(
            "Sparse topology conversion changed layer outputs; "
            f"maximum absolute error={maximum_error:.6g}"
        )


def _verification_tolerances(
    dtype: torch.dtype,
    *,
    atol: float,
    rtol: float,
    reduction_backend: str | None = None,
) -> tuple[float, float]:
    reordered_sparse_reduction = reduction_backend in {
        "triton_transposed",
        "triton_fused",
        "triton_ell",
    }
    if dtype in {torch.float16, torch.bfloat16} or reordered_sparse_reduction:
        return max(atol, 5e-3), max(rtol, 5e-3)
    return atol, rtol


def _convertible_support(
    module: nn.Module,
    config: ModelCompressionConfig,
) -> _FrozenSupport | None:
    sparse = config.sparse_topology
    if type(module) is DenseToSparseLinear:
        if module.noise_level != 0:
            raise ValueError("Cannot losslessly freeze noisy DenseToSparseLinear")
        if sparse.require_final_dense_to_sparse_step and (
            module.current_k != module.target_k
        ):
            raise ValueError(
                "DenseToSparseLinear has not reached target K; finish its pruning "
                "schedule or explicitly disable the final-step requirement"
            )
        mask = module.weight_mask()
        indices, raw_weights = _dense_topk_support(module)
        return _FrozenSupport(
            "indexed",
            indices,
            raw_weights,
            module._pruned_weight_from_mask(mask).detach(),
            mask.to(torch.bool),
            "final_dense_to_sparse_topk",
        )
    if type(module) is TopKLinear:
        if module.noise_level != 0:
            raise ValueError("Cannot losslessly freeze noisy TopKLinear")
        mask = module.weight_mask()
        indices, raw_weights = _dense_topk_support(module)
        return _FrozenSupport(
            "indexed",
            indices,
            raw_weights,
            module._pruned_weight_from_mask(mask).detach(),
            mask.to(torch.bool),
            "deterministic_standard_topk",
        )
    if type(module) is VarianceTopKLinear:
        if not sparse.allow_variance_topk:
            return None
        if module.training or module.stats_tracking:
            raise ValueError(
                "VarianceTopKLinear must be in eval mode so activation statistics "
                "are frozen before export"
            )
        if module.noise_level != 0:
            raise ValueError("Cannot losslessly freeze noisy VarianceTopKLinear")
        mask = module.weight_mask()
        return _support_from_mask(
            module,
            mask,
            effective_weight=module._pruned_weight_from_mask(mask).detach(),
            freeze_policy="frozen_variance_statistics_topk",
            ragged_topology_format=sparse.ragged_topology_format,
        )
    if type(module) is StochasticTopKLinear:
        policy = str(sparse.stochastic_topk_policy)
        if policy == "reject":
            raise ValueError(
                "StochasticTopKLinear requires an explicit deployment policy: "
                "deterministic_topk or sample_once"
            )
        mask = (
            _deterministic_topk_mask(module)
            if policy == "deterministic_topk"
            else _sample_stochastic_mask_once(module, seed=sparse.stochastic_seed)
        )
        return _support_from_mask(
            module,
            mask,
            effective_weight=module._pruned_weight_from_mask(mask).detach(),
            freeze_policy=(
                "deterministic_topk_of_stochastic_scores"
                if policy == "deterministic_topk"
                else f"single_stochastic_draw_seed_{sparse.stochastic_seed}"
            ),
            ragged_topology_format=sparse.ragged_topology_format,
        )
    if type(module) in {DeepstLinear, CreditGatedDeepstLinear}:
        if not sparse.allow_deepst_snapshot:
            return None
        if module.training:
            raise ValueError("DeepST topology snapshot requires model.eval()")
        mask = module.weight_mask()
        if sparse.ragged_topology_format == "reject":
            raise ValueError(
                "DeepST snapshot export requires ragged_topology_format='csr'; "
                "CSR also represents constant-branch masks without changing "
                "their already-transformed effective conductances"
            )
        return _FrozenSupport(
            representation="csr",
            indices=None,
            raw_weights=None,
            effective_weight=module.pruned_weight().detach(),
            mask=mask.to(dtype=torch.bool),
            freeze_policy="snapshot_current_deepst_mask",
        )
    if type(module) is IndexedDynamicTopKLinear:
        if not sparse.allow_indexed_dynamic:
            return None
        if module.selection != "standard" or module.noise_level != 0:
            raise ValueError(
                "IndexedDynamicTopKLinear must use deterministic standard "
                "selection with noise_level=0 before topology freezing"
            )
        indices, raw_weights = _dynamic_indexed_support(module)
        mask = torch.zeros(
            module.out_features,
            module.in_features,
            dtype=torch.bool,
            device=indices.device,
        )
        mask.scatter_(1, indices.to(torch.long), True)
        return _FrozenSupport(
            "indexed",
            indices,
            raw_weights,
            module.pruned_weight().detach(),
            mask,
            "active_indexed_candidate_topk",
        )
    if type(module) is IndexedRewireLinear:
        if not sparse.allow_indexed_rewire:
            return None
        indices, raw_weights = _fixed_indexed_support(module)
        mask = torch.zeros(
            module.out_features,
            module.in_features,
            dtype=torch.bool,
            device=indices.device,
        )
        mask.scatter_(1, indices.to(torch.long), True)
        return _FrozenSupport(
            "indexed",
            indices,
            raw_weights,
            module.pruned_weight().detach(),
            mask,
            "current_indexed_rewire_topology",
        )
    return None


def _record_for(
    path: str,
    source: nn.Module,
    replacement: IndexedSparseLinear | CSRInferenceLinear,
    *,
    freeze_policy: str,
) -> FrozenSparseModuleRecord:
    if isinstance(replacement, CSRInferenceLinear):
        degrees = [int(value) for value in replacement.row_degrees.tolist()]
        synapses_per_output = max(degrees, default=0)
        options = {
            "param_space": "folded_effective_weight",
            "init_method": "none",
            "init_gain": 1.0,
            "weight_transform": "identity",
            "weight_norm_order": None,
            "gamma": 1.0,
            "output_chunk_size": 2048,
            "workspace_mb": None,
            "cache_transformed_weights": False,
            "projection_backend": "torch_csr",
            "support_group_rows": 1,
            "support_col_block": 1,
        }
    else:
        degrees = None
        synapses_per_output = replacement.K
        options = _replacement_options(replacement)
    return FrozenSparseModuleRecord(
        path=path,
        source_type=type(source).__name__,
        target_type=type(replacement).__name__,
        in_features=replacement.in_features,
        out_features=replacement.out_features,
        synapses_per_output=synapses_per_output,
        row_degrees=degrees,
        freeze_policy=freeze_policy,
        **options,
    )


def freeze_sparse_topology_(
    model: nn.Module,
    config: ModelCompressionConfig | None = None,
) -> list[FrozenSparseModuleRecord]:
    """Replace deterministic dynamic sparse layers with exact fixed topology.

    The operation mutates ``model`` and is intended for inference/export after
    training. Existing :class:`IndexedSparseLinear` modules are already compact
    and are left unchanged.
    """

    policy = config or ModelCompressionConfig()
    if not policy.sparse_topology.enabled:
        return []

    candidates: list[
        tuple[
            nn.Module,
            str,
            str,
            nn.Module,
            _FrozenSupport,
        ]
    ] = []
    converted_paths: dict[int, str] = {}

    def collect(parent: nn.Module, prefix: str) -> None:
        for name, child in list(parent._modules.items()):
            if child is None:
                continue
            path = f"{prefix}.{name}" if prefix else name
            support = _convertible_support(child, policy)
            if support is None:
                collect(child, path)
                continue
            previous_path = converted_paths.get(id(child))
            if previous_path is not None:
                raise ValueError(
                    "Shared dynamic sparse modules cannot be frozen safely: "
                    f"{previous_path!r} and {path!r} reference the same module"
                )
            converted_paths[id(child)] = path
            candidates.append((parent, name, path, child, support))

    collect(model, "")
    prepared: list[
        tuple[
            nn.Module,
            str,
            IndexedSparseLinear | CSRInferenceLinear,
            FrozenSparseModuleRecord,
        ]
    ] = []
    for parent, name, path, child, support in candidates:
        if support.representation == "indexed":
            assert support.indices is not None and support.raw_weights is not None
            replacement = _make_indexed_replacement(
                child, support.indices, support.raw_weights
            )
        elif support.representation == "csr":
            replacement = masked_weight_to_csr(
                support.effective_weight,
                support.mask,
            )
        else:
            raise AssertionError(
                f"Unknown frozen representation {support.representation}"
            )
        if policy.verification.enabled:
            device, dtype = _module_device_and_dtype(child)
            generator = torch.Generator(device=device).manual_seed(0)
            sample = torch.randn(
                2,
                int(child.in_features),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            with torch.no_grad():
                expected = torch.nn.functional.linear(sample, support.effective_weight)
                observed = replacement(sample)
            dtype_atol, dtype_rtol = _verification_tolerances(
                dtype,
                atol=policy.verification.atol,
                rtol=policy.verification.rtol,
                reduction_backend=getattr(
                    replacement, "_last_resolved_projection_backend", None
                ),
            )
            if not torch.allclose(
                expected,
                observed,
                atol=dtype_atol,
                rtol=dtype_rtol,
            ):
                maximum_error = float((expected - observed).abs().max().item())
                raise RuntimeError(
                    "Sparse topology conversion changed layer outputs; "
                    f"maximum absolute error={maximum_error:.6g}"
                )
        prepared.append(
            (
                parent,
                name,
                replacement,
                _record_for(
                    path,
                    child,
                    replacement,
                    freeze_policy=support.freeze_policy,
                ),
            )
        )

    for parent, name, replacement, _ in prepared:
        parent._modules[name] = replacement
    return [record for _, _, _, record in prepared]


def describe_fixed_sparse_topology(
    model: nn.Module,
) -> list[FrozenSparseModuleRecord]:
    """Manifest records for modules that are ALREADY fixed indexed layers.

    ``freeze_sparse_topology_`` only records modules it converts, so a model
    whose topology was changed by some earlier structural operation — a
    pruning-ladder rung replaces ``IndexedRewireLinear`` with a plain
    ``IndexedSparseLinear`` at a smaller contact count — exports with no
    record of its realized shapes, and a reload that rebuilds from the source
    config then fails on the size mismatch.  These records make such an
    export self-describing: ``prepare_model_for_compact_state_`` rebuilds
    each module at the recorded shape before the strict load.  Subclasses
    (rewiring, dynamic) are excluded; they are the freeze pass's job.
    """

    records: list[FrozenSparseModuleRecord] = []
    for path, child in model.named_modules():
        if not path or type(child) is not IndexedSparseLinear:
            continue
        records.append(_record_for(path, child, child, freeze_policy="already_fixed"))
    return records


def _module_at_path(model: nn.Module, path: str) -> tuple[nn.Module, str, nn.Module]:
    parts = path.split(".")
    if not parts or not path:
        raise ValueError("Compact module paths must be non-empty")
    parent = model
    for part in parts[:-1]:
        child = parent._modules.get(part)
        if child is None:
            raise KeyError(f"Model has no module at compact path {path!r}")
        parent = child
    name = parts[-1]
    child = parent._modules.get(name)
    if child is None:
        raise KeyError(f"Model has no module at compact path {path!r}")
    return parent, name, child


def prepare_model_for_compact_state_(
    model: nn.Module,
    manifest: list[Mapping[str, Any]],
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Reconstruct fixed indexed modules before strict compact-state loading."""

    for item in manifest:
        path = str(item["path"])
        parent, name, source = _module_at_path(model, path)
        if item.get("target_type") == "CSRInferenceLinear":
            keys = {
                field: f"{path}.{field}"
                for field in ("crow_indices", "col_indices", "values")
            }
            missing = [key for key in keys.values() if key not in state_dict]
            if missing:
                raise KeyError(
                    "Compact CSR state is missing " + ", ".join(sorted(missing))
                )
            device, dtype = _module_device_and_dtype(source)
            replacement = CSRInferenceLinear(
                int(item["in_features"]),
                int(item["out_features"]),
                crow_indices=state_dict[keys["crow_indices"]],
                col_indices=state_dict[keys["col_indices"]],
                values=state_dict[keys["values"]],
            ).to(device=device, dtype=dtype)
            parent._modules[name] = replacement
            continue
        index_key = f"{path}.connection_indices"
        if index_key not in state_dict:
            raise KeyError(f"Compact state is missing {index_key}")
        indices = state_dict[index_key].to(dtype=torch.long)
        expected_shape = (
            int(item["out_features"]),
            int(item["synapses_per_output"]),
        )
        if tuple(indices.shape) != expected_shape:
            raise ValueError(
                f"Compact topology shape for {path} is {tuple(indices.shape)}, "
                f"expected {expected_shape}"
            )
        weight_key = f"{path}.pre_w"
        if weight_key not in state_dict:
            raise KeyError(f"Compact state is missing {weight_key}")
        replacement = _make_indexed_replacement(
            source,
            indices,
            state_dict[weight_key],
            options={
                key: (
                    item.get(key, 1)
                    if key in {"support_group_rows", "support_col_block"}
                    else item[key]
                )
                for key in (
                    "param_space",
                    "init_method",
                    "init_gain",
                    "weight_transform",
                    "weight_norm_order",
                    "gamma",
                    "output_chunk_size",
                    "workspace_mb",
                    "cache_transformed_weights",
                    "projection_backend",
                    "support_group_rows",
                    "support_col_block",
                )
            },
        )
        parent._modules[name] = replacement


def compact_model_checkpoint(
    model: nn.Module,
    source_checkpoint: str | Path,
    output_checkpoint: str | Path,
    config: ModelCompressionConfig | None = None,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze native sparse topology and save a self-describing checkpoint."""

    policy = config or ModelCompressionConfig()
    source = Path(source_checkpoint).expanduser().resolve()
    output = Path(output_checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("Compact output must differ from the source checkpoint")
    if output.exists():
        raise FileExistsError(output)

    records = freeze_sparse_topology_(model, policy)
    compact_candidates = compact_model_state_dict_candidates(
        model.state_dict(),
        topology_encoding=policy.sparse_topology.topology_encoding,
    )
    source_bytes = source.stat().st_size
    topology_manifest = [asdict(record) for record in records]
    payloads = {
        encoding: {
            "format": COMPACT_CHECKPOINT_FORMAT,
            "format_version": COMPACT_CHECKPOINT_VERSION,
            "state_dict": compact_state,
            "sparse_topology_manifest": topology_manifest,
            "sparse_index_encoding": encodings,
            "topology_encoding_requested": policy.sparse_topology.topology_encoding,
            "topology_encoding_selected": encoding,
            "deployment_storage_manifest": deployment_storage_manifest(
                model,
                compact_state,
                topology_encoding=encoding,
                sparse_topology_manifest=topology_manifest,
                scope="full_model",
            ),
            "compression_config": asdict(policy),
            "provenance": dict(provenance or {}),
            "source": {
                "path": str(source),
                "sha256": sha256_file(source),
                "bytes": source_bytes,
            },
        }
        for encoding, (compact_state, encodings) in compact_candidates.items()
    }
    encoding_report = atomic_torch_save_candidates(payloads, output)
    output_bytes = int(encoding_report["output_bytes"])
    return {
        "output": str(output),
        "output_sha256": encoding_report["output_sha256"],
        "output_bytes": output_bytes,
        "source": str(source),
        "source_bytes": source_bytes,
        "size_reduction_factor": source_bytes / max(output_bytes, 1),
        "converted_modules": len(records),
        "topology_encoding_requested": policy.sparse_topology.topology_encoding,
        "topology_encoding": encoding_report["selected"],
        "topology_encoding_selection": encoding_report,
    }


def freeze_model_for_inference_(model: nn.Module) -> int:
    """Fold constant inference tensors across every dendritic module.

    Walks the model and calls ``freeze_for_inference`` on each
    :class:`IndexedSparseLinear` and :class:`BlockLinear` (including
    ``EfficientBlockLinear``): the weight transforms and branch conductance
    sums are computed once into non-persistent buffers instead of on every
    forward. Also switches the model to eval mode. The folds are exact (same
    tensors the training forward computes) and are dropped automatically if
    the model returns to training or loads new weights, so calling this never
    changes results — only inference cost. Returns the number of folded
    modules.
    """
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
        BlockLinear,
    )

    model.eval()
    folded = 0
    for module in model.modules():
        if isinstance(module, (IndexedSparseLinear, BlockLinear)):
            module.freeze_for_inference()
            folded += 1
    return folded


def load_compact_model_checkpoint(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    strict: bool = True,
    apply_dense_runtime: bool = False,
    freeze_for_inference: bool = False,
) -> dict[str, Any]:
    """Load a compact checkpoint into a model built from its training config."""

    resolved = Path(checkpoint).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != (
        COMPACT_CHECKPOINT_FORMAT
    ):
        raise ValueError("Checkpoint is not a dendritic compact checkpoint")
    if payload.get("format_version") != COMPACT_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported compact checkpoint version {payload.get('format_version')!r}"
        )
    state = extract_model_state_dict(payload)
    state = decode_sparse_bitmask_state_dict(state)
    manifest = payload.get("sparse_topology_manifest", [])
    if not isinstance(manifest, list):
        raise TypeError("sparse_topology_manifest must be a list")
    prepare_model_for_compact_state_(model, manifest, state)
    incompatible = model.load_state_dict(state, strict=strict)
    runtime_report: dict[str, Any] | None = None
    if apply_dense_runtime:
        compression_payload = payload.get("compression_config", {})
        if not isinstance(compression_payload, dict):
            raise TypeError("compression_config must be a mapping")
        policy = ModelCompressionConfig(**compression_payload)
        from dendritic_modeling.deployment.torchao import optimize_dense_runtime_

        model.eval()
        runtime_report = optimize_dense_runtime_(model, policy.dense_runtime)
    frozen_modules = 0
    if freeze_for_inference:
        frozen_modules = freeze_model_for_inference_(model)
    return {
        "checkpoint": str(resolved),
        "sha256": sha256_file(resolved),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "converted_modules": len(manifest),
        "dense_runtime": runtime_report,
        "frozen_inference_modules": frozen_modules,
    }


__all__ = [
    "COMPACT_CHECKPOINT_FORMAT",
    "COMPACT_CHECKPOINT_VERSION",
    "FrozenSparseModuleRecord",
    "compact_model_checkpoint",
    "freeze_sparse_topology_",
    "load_compact_model_checkpoint",
    "prepare_model_for_compact_state_",
]
