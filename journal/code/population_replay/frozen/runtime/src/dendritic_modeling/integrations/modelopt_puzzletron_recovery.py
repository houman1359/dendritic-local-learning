"""Structure-preserving recovery utilities for OLMo-3 Puzzletron models.

An ordinary ``AutoModelForCausalLM.from_pretrained`` call constructs every
OLMo-3 MLP at the global ``intermediate_size`` and therefore cannot faithfully
restore a Puzzletron checkpoint whose ``block_configs`` specify heterogeneous
per-layer widths.  This module keeps the NVIDIA dependency optional, registers
the local OLMo-3 descriptor, constructs the model inside ModelOpt's
``deci_x_patcher``, and validates the realized tensor geometry against every
block configuration.

The training policy is equally explicit: only FFNs whose realized structure
differs from the teacher-width FFN are made trainable.  Embeddings, attention,
normalization, the language-model head, and unchanged FFNs remain frozen.  The
selected MLP modules can be passed directly to composable FSDP2 before a root
wrap that ignores the frozen parameters.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dendritic_modeling.integrations.modelopt_puzzletron_olmo3 import (
    get_registered_olmo3_puzzletron_classes,
    register_olmo3_puzzletron_adapter,
    validate_olmo3_attention_contract,
)

STRUCTURE_SCHEMA = "dendritic_modelopt_olmo3_recovery_structure/v1"
SELECTION_SCHEMA = "dendritic_modelopt_olmo3_replacement_selection/v1"
COMPACT_CHECKPOINT_SCHEMA = "dendritic_modelopt_olmo3_compact_checkpoint/v1"
RECOVERY_SMOKE_SCHEMA = "dendritic_modelopt_olmo3_recovery_smoke/v1"

_AUXILIARY_CHECKPOINT_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


@dataclass(frozen=True)
class Olmo3PuzzletronStructure:
    """Exact correspondence between ``block_configs`` and realized MLPs."""

    schema: str
    model_type: str
    hidden_size: int
    num_layers: int
    teacher_intermediate_size: int
    widths_by_layer: tuple[int | None, ...]
    no_op_layers: tuple[int, ...]
    changed_ffn_layers: tuple[int, ...]
    unchanged_ffn_layers: tuple[int, ...]
    ffn_parameter_values_by_layer: tuple[int, ...]
    ffn_parameter_bytes_by_layer: tuple[int, ...]
    total_parameter_values: int
    total_parameter_bytes: int
    total_buffer_bytes: int
    parameter_dtypes: tuple[str, ...]


@dataclass(frozen=True)
class Olmo3PuzzletronTrainableSelection:
    """Replacement-only parameters and their enclosing FSDP2 units."""

    schema: str
    structurally_changed_layers: tuple[int, ...]
    trainable_layers: tuple[int, ...]
    parameter_names: tuple[str, ...]
    trainable_parameter_values: int
    frozen_parameter_values: int
    modules: tuple[nn.Module, ...] = field(repr=False, compare=False)
    parameters: tuple[nn.Parameter, ...] = field(repr=False, compare=False)
    frozen_parameters: tuple[nn.Parameter, ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class LoadedOlmo3PuzzletronCheckpoint:
    """A patched model together with its verified structural receipt."""

    checkpoint: Path
    model: nn.Module = field(repr=False, compare=False)
    descriptor: type = field(repr=False, compare=False)
    structure: Olmo3PuzzletronStructure
    state_key_count: int
    safetensors_shard_count: int
    safetensors_bytes: int


@dataclass(frozen=True)
class Olmo3PuzzletronCompactCheckpoint:
    """Atomic ModelOpt checkpoint-save receipt."""

    schema: str
    checkpoint: Path
    state_key_count: int
    safetensors_shard_count: int
    safetensors_bytes: int
    auxiliary_files: tuple[str, ...]
    structure: Olmo3PuzzletronStructure


@dataclass(frozen=True)
class _RecoveryAPI:
    load_model_config: Callable[..., Any]
    deci_x_patcher: Callable[..., AbstractContextManager[Any]]
    auto_model_for_causal_lm: Any
    save_checkpoint: Callable[..., None]


class _StateDictModelView:
    """Minimal public-API view accepted by ModelOpt ``save_checkpoint``."""

    def __init__(self, model: nn.Module, state_dict: Mapping[str, torch.Tensor]):
        self.config = model.config
        self._state_dict = dict(state_dict)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self._state_dict


def _load_recovery_api() -> _RecoveryAPI:
    try:
        from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher
        from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import (
            load_model_config,
            save_checkpoint,
        )
        from transformers import AutoModelForCausalLM
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "OLMo-3 Puzzletron recovery requires the pinned "
            "nvidia-modelopt[puzzletron] environment"
        ) from exc
    return _RecoveryAPI(
        load_model_config=load_model_config,
        deci_x_patcher=deci_x_patcher,
        auto_model_for_causal_lm=AutoModelForCausalLM,
        save_checkpoint=save_checkpoint,
    )


def _value(value: Any, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping):
            value = value.get(name)
        else:
            value = getattr(value, name, None)
    return value


def _local_checkpoint_config(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    root = Path(checkpoint).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Puzzletron checkpoint is not a directory: {root}")
    config_path = root / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError(f"Puzzletron checkpoint has no regular config.json: {root}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Puzzletron config: {config_path}") from exc
    if config.get("model_type") != "olmo3":
        raise ValueError("Puzzletron recovery only supports model_type='olmo3'")
    blocks = config.get("block_configs")
    num_layers = config.get("num_hidden_layers")
    if (
        isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
        or num_layers <= 0
        or not isinstance(blocks, list)
        or len(blocks) != num_layers
    ):
        raise ValueError("Puzzletron block_configs do not cover every OLMo-3 layer")
    validate_olmo3_attention_contract(config)
    return root, config


def _checkpoint_state_layout(
    checkpoint: Path,
    *,
    expected_state_keys: set[str] | None = None,
    tied_lm_head: bool = False,
) -> tuple[set[str], tuple[Path, ...], int]:
    index_path = checkpoint / "model.safetensors.index.json"
    single_path = checkpoint / "model.safetensors"
    if index_path.is_symlink() or single_path.is_symlink():
        raise ValueError("Puzzletron safetensors metadata may not be a symlink")
    if index_path.is_file() == single_path.is_file():
        raise ValueError(
            "Puzzletron checkpoint must contain exactly one safetensors layout"
        )
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid safetensors index: {index_path}") from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            raise ValueError("safetensors index has no non-empty weight_map")
        state_keys = set(weight_map)
        relative_files = set(weight_map.values())
        if any(not isinstance(item, str) or not item for item in relative_files):
            raise ValueError("safetensors index contains an invalid shard name")
        shards: list[Path] = []
        for relative in sorted(relative_files):
            relative_path = Path(relative)
            if relative_path.is_absolute():
                raise ValueError("safetensors index contains an absolute shard path")
            shard = (checkpoint / relative_path).resolve(strict=True)
            if not shard.is_relative_to(checkpoint) or not shard.is_file():
                raise ValueError("safetensors shard escapes the checkpoint")
            unresolved = checkpoint / relative_path
            if unresolved.is_symlink() or shard.suffix != ".safetensors":
                raise ValueError("Puzzletron shard must be a regular safetensors file")
            shards.append(shard)
    else:
        try:
            from safetensors import safe_open
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "safetensors is required for Puzzletron recovery"
            ) from exc
        with safe_open(single_path, framework="pt", device="cpu") as handle:
            state_keys = set(handle.keys())
        shards = [single_path.resolve(strict=True)]

    if expected_state_keys is not None:
        allowed_missing = {"lm_head.weight"} if tied_lm_head else set()
        missing = expected_state_keys - state_keys
        unexpected = state_keys - expected_state_keys
        if unexpected or not missing.issubset(allowed_missing):
            raise RuntimeError(
                "Puzzletron safetensors/state_dict keys differ: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
    return state_keys, tuple(shards), sum(path.stat().st_size for path in shards)


def inspect_olmo3_puzzletron_structure(
    model: nn.Module,
) -> Olmo3PuzzletronStructure:
    """Prove that every realized OLMo-3 FFN matches its block configuration."""

    config = getattr(model, "config", None)
    if _value(config, "model_type") != "olmo3":
        raise ValueError("model config is not OLMo-3")
    validate_olmo3_attention_contract(config)
    hidden_size = _value(config, "hidden_size")
    teacher_width = _value(config, "intermediate_size")
    num_layers = _value(config, "num_hidden_layers")
    blocks = _value(config, "block_configs")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (hidden_size, teacher_width, num_layers)
    ):
        raise ValueError("OLMo-3 config has invalid global dimensions")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise ValueError("OLMo-3 config has no block_configs sequence")
    layers = _value(model, "model", "layers")
    if not isinstance(layers, (nn.ModuleList, list, tuple)):
        raise ValueError("OLMo-3 model has no decoder-layer sequence")
    if len(blocks) != num_layers or len(layers) != num_layers:
        raise ValueError("OLMo-3 model, global config, and block_configs disagree")

    widths: list[int | None] = []
    no_ops: list[int] = []
    changed: list[int] = []
    unchanged: list[int] = []
    values_by_layer: list[int] = []
    bytes_by_layer: list[int] = []
    for layer_index, (layer, block) in enumerate(zip(layers, blocks)):
        no_op = bool(_value(block, "ffn", "no_op"))
        configured_width = _value(block, "ffn", "intermediate_size")
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, nn.Module):
            raise RuntimeError(f"layer {layer_index} has no MLP module")
        layer_parameters = list(mlp.named_parameters())
        if no_op:
            if configured_width is not None or layer_parameters:
                raise RuntimeError(
                    f"FFN no-op at layer {layer_index} retains width or parameters"
                )
            widths.append(None)
            no_ops.append(layer_index)
            changed.append(layer_index)
            values_by_layer.append(0)
            bytes_by_layer.append(0)
            continue
        if (
            isinstance(configured_width, bool)
            or not isinstance(configured_width, int)
            or configured_width <= 0
        ):
            raise ValueError(f"layer {layer_index} has no positive FFN width")
        projections = {
            name: getattr(mlp, name, None)
            for name in ("gate_proj", "up_proj", "down_proj")
        }
        if any(not isinstance(module, nn.Linear) for module in projections.values()):
            raise RuntimeError(f"layer {layer_index} is not a realized SwiGLU FFN")
        expected_shapes = {
            "gate_proj": (configured_width, hidden_size),
            "up_proj": (configured_width, hidden_size),
            "down_proj": (hidden_size, configured_width),
        }
        observed_shapes = {
            name: tuple(module.weight.shape) for name, module in projections.items()
        }
        if observed_shapes != expected_shapes:
            raise RuntimeError(
                f"layer {layer_index} FFN geometry differs from block_configs: "
                f"{observed_shapes} != {expected_shapes}"
            )
        if any(module.bias is not None for module in projections.values()):
            raise RuntimeError("the OLMo-3 Puzzletron bridge requires bias-free FFNs")
        expected_names = {
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        }
        if {name for name, _ in layer_parameters} != expected_names:
            raise RuntimeError(f"layer {layer_index} has unexpected FFN parameters")
        realized_values = sum(parameter.numel() for _, parameter in layer_parameters)
        expected_values = 3 * hidden_size * configured_width
        if realized_values != expected_values:
            raise RuntimeError(f"layer {layer_index} FFN parameter count drifted")
        realized_bytes = sum(
            parameter.numel() * parameter.element_size()
            for _, parameter in layer_parameters
        )
        widths.append(configured_width)
        values_by_layer.append(realized_values)
        bytes_by_layer.append(realized_bytes)
        (changed if configured_width != teacher_width else unchanged).append(
            layer_index
        )

    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    return Olmo3PuzzletronStructure(
        schema=STRUCTURE_SCHEMA,
        model_type="olmo3",
        hidden_size=hidden_size,
        num_layers=num_layers,
        teacher_intermediate_size=teacher_width,
        widths_by_layer=tuple(widths),
        no_op_layers=tuple(no_ops),
        changed_ffn_layers=tuple(changed),
        unchanged_ffn_layers=tuple(unchanged),
        ffn_parameter_values_by_layer=tuple(values_by_layer),
        ffn_parameter_bytes_by_layer=tuple(bytes_by_layer),
        total_parameter_values=sum(parameter.numel() for parameter in parameters),
        total_parameter_bytes=sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
        total_buffer_bytes=sum(
            buffer.numel() * buffer.element_size() for buffer in buffers
        ),
        parameter_dtypes=tuple(
            sorted({str(parameter.dtype) for parameter in parameters})
        ),
    )


def load_olmo3_puzzletron_checkpoint(
    checkpoint: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
) -> LoadedOlmo3PuzzletronCheckpoint:
    """Load an exact heterogeneous AnyModel checkpoint through ``deci_x_patcher``."""

    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("Puzzletron recovery dtype must be a floating torch dtype")
    root, raw_config = _local_checkpoint_config(checkpoint)
    register_olmo3_puzzletron_adapter()
    descriptor, _ = get_registered_olmo3_puzzletron_classes()
    api = _load_recovery_api()
    config = api.load_model_config(root, trust_remote_code=False)
    if _value(config, "model_type") != "olmo3":
        raise RuntimeError("ModelOpt loaded a non-OLMo-3 config")
    if len(_value(config, "block_configs")) != len(raw_config["block_configs"]):
        raise RuntimeError("ModelOpt changed the block-config count while loading")
    validate_olmo3_attention_contract(config)

    target_device = torch.device(device)
    load_kwargs: dict[str, Any] = {
        "config": config,
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "output_loading_info": True,
    }
    if target_device.type != "cpu":
        load_kwargs["device_map"] = {"": target_device}
    with api.deci_x_patcher(
        model_descriptor=descriptor,
        block_configs=config.block_configs,
    ):
        model, loading_info = api.auto_model_for_causal_lm.from_pretrained(
            root, **load_kwargs
        )
    if not isinstance(loading_info, Mapping):
        raise RuntimeError("Transformers returned no Puzzletron loading receipt")
    nonempty_loading_fields = {
        name: loading_info.get(name)
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(name)
    }
    if nonempty_loading_fields:
        raise RuntimeError(
            f"Puzzletron checkpoint did not load exactly: {nonempty_loading_fields}"
        )
    if target_device.type == "cpu":
        model.to(device=target_device)

    floating_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if parameter.is_floating_point()
    }
    if floating_dtypes != {dtype}:
        raise RuntimeError(
            f"Puzzletron parameter dtype drifted: {floating_dtypes} != {{{dtype}}}"
        )
    parameter_devices = {parameter.device.type for parameter in model.parameters()}
    if parameter_devices != {target_device.type}:
        raise RuntimeError(f"Puzzletron parameter devices drifted: {parameter_devices}")
    structure = inspect_olmo3_puzzletron_structure(model)
    state_keys = set(model.state_dict())
    indexed_keys, shards, shard_bytes = _checkpoint_state_layout(
        root,
        expected_state_keys=state_keys,
        tied_lm_head=bool(_value(config, "tie_word_embeddings")),
    )
    return LoadedOlmo3PuzzletronCheckpoint(
        checkpoint=root,
        model=model,
        descriptor=descriptor,
        structure=structure,
        state_key_count=len(indexed_keys),
        safetensors_shard_count=len(shards),
        safetensors_bytes=shard_bytes,
    )


def select_changed_olmo3_ffns_for_training(
    model: nn.Module,
) -> Olmo3PuzzletronTrainableSelection:
    """Freeze the model and enable gradients only for narrowed/changed FFNs."""

    structure = inspect_olmo3_puzzletron_structure(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    modules: list[nn.Module] = []
    trainable_layers: list[int] = []
    for layer_index in structure.changed_ffn_layers:
        if layer_index in structure.no_op_layers:
            continue
        mlp = model.model.layers[layer_index].mlp
        layer_parameters = tuple(mlp.parameters())
        if not layer_parameters:
            raise RuntimeError(f"changed FFN at layer {layer_index} has no parameters")
        for parameter in layer_parameters:
            parameter.requires_grad_(True)
        modules.append(mlp)
        trainable_layers.append(layer_index)

    named_parameters = tuple(model.named_parameters())
    selected_names = tuple(
        name for name, value in named_parameters if value.requires_grad
    )
    selected_parameters = tuple(
        value for _, value in named_parameters if value.requires_grad
    )
    selected_ids = {id(parameter) for parameter in selected_parameters}
    if len(selected_ids) != len(selected_parameters):
        raise RuntimeError("replacement-only selection contains aliased parameters")
    expected_prefixes = tuple(
        f"model.layers.{index}.mlp." for index in trainable_layers
    )
    if not selected_names or any(
        not name.startswith(expected_prefixes) for name in selected_names
    ):
        raise RuntimeError(
            "replacement-only selection leaked outside structurally changed FFNs"
        )
    module_parameter_ids = {
        id(parameter) for module in modules for parameter in module.parameters()
    }
    if module_parameter_ids != selected_ids:
        raise RuntimeError("FSDP2 module and trainable-parameter selections disagree")
    frozen_parameters = tuple(
        value for _, value in named_parameters if not value.requires_grad
    )
    return Olmo3PuzzletronTrainableSelection(
        schema=SELECTION_SCHEMA,
        structurally_changed_layers=structure.changed_ffn_layers,
        trainable_layers=tuple(trainable_layers),
        parameter_names=selected_names,
        trainable_parameter_values=sum(value.numel() for value in selected_parameters),
        frozen_parameter_values=sum(value.numel() for value in frozen_parameters),
        modules=tuple(modules),
        parameters=selected_parameters,
        frozen_parameters=frozen_parameters,
    )


def collect_modelopt_full_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect a CPU full state dict from an unwrapped model or FSDP2 model.

    Every distributed rank must call this function.  With FSDP2 and CPU
    offload, rank zero receives the full mapping and nonzero ranks receive an
    empty mapping, which is the form accepted by
    :func:`save_olmo3_puzzletron_checkpoint` on rank zero.
    """

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    state = get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            ignore_frozen_params=False,
            strict=True,
        ),
    )
    if not isinstance(state, Mapping):
        raise RuntimeError("FSDP2 full-state collection returned no mapping")
    normalized: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError(f"ModelOpt cannot serialize state entry {name!r}")
        normalized[name] = value.detach().to(device="cpu").contiguous()
    return normalized


def save_olmo3_puzzletron_checkpoint(
    model: nn.Module,
    output_dir: Path,
    *,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    source_checkpoint: Path | None = None,
) -> Olmo3PuzzletronCompactCheckpoint:
    """Atomically save a compact, heterogeneity-preserving ModelOpt checkpoint."""

    output = Path(output_dir).absolute()
    if output.exists():
        raise FileExistsError(f"Puzzletron recovery output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    structure = inspect_olmo3_puzzletron_structure(model)
    expected_state_keys = set(model.state_dict())
    if state_dict is None:
        state_dict = collect_modelopt_full_state_dict(model)
    normalized = dict(state_dict)
    if set(normalized) != expected_state_keys:
        raise RuntimeError(
            "ModelOpt save state is not the complete model state: "
            f"missing={sorted(expected_state_keys - set(normalized))}, "
            f"unexpected={sorted(set(normalized) - expected_state_keys)}"
        )
    if any(not isinstance(value, torch.Tensor) for value in normalized.values()):
        raise TypeError("ModelOpt save state contains a non-tensor value")

    register_olmo3_puzzletron_adapter()
    descriptor, _ = get_registered_olmo3_puzzletron_classes()
    api = _load_recovery_api()
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    auxiliary: list[str] = []
    try:
        api.save_checkpoint(
            _StateDictModelView(model, normalized), temporary, descriptor
        )
        if source_checkpoint is not None:
            source, _ = _local_checkpoint_config(source_checkpoint)
            for name in _AUXILIARY_CHECKPOINT_FILES:
                source_file = source / name
                if source_file.is_symlink():
                    raise ValueError(f"checkpoint auxiliary file is a symlink: {name}")
                if source_file.is_file():
                    shutil.copy2(source_file, temporary / name)
                    auxiliary.append(name)
        _, saved_config = _local_checkpoint_config(temporary)
        if saved_config["block_configs"] != json.loads(
            json.dumps(
                _value(model.config, "block_configs"),
                default=lambda item: item.to_dict(),
            )
        ):
            raise RuntimeError("ModelOpt save changed heterogeneous block_configs")
        indexed_keys, shards, shard_bytes = _checkpoint_state_layout(
            temporary,
            expected_state_keys=expected_state_keys,
            tied_lm_head=bool(_value(model.config, "tie_word_embeddings")),
        )
        os.replace(temporary, output)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return Olmo3PuzzletronCompactCheckpoint(
        schema=COMPACT_CHECKPOINT_SCHEMA,
        checkpoint=output,
        state_key_count=len(indexed_keys),
        safetensors_shard_count=len(shards),
        safetensors_bytes=shard_bytes,
        auxiliary_files=tuple(auxiliary),
        structure=structure,
    )


def run_olmo3_puzzletron_recovery_smoke(
    source_checkpoint: Path,
    output_checkpoint: Path,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    sequence_length: int = 8,
    learning_rate: float = 1e-4,
) -> dict[str, Any]:
    """Exercise load, update, compact save, and exact patched reload.

    This is an integration gate, not a recovery-quality experiment.  The
    default CPU/FP32 path is intended for a tiny heterogeneous OLMo-3 fixture;
    the 7B matched-recovery campaign must use its separately frozen objective,
    data, distributed topology, and optimizer protocol.
    """

    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer")
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least two")
    if not isinstance(learning_rate, (int, float)) or not 0.0 < learning_rate < 1.0:
        raise ValueError("learning_rate must lie strictly between zero and one")

    loaded = load_olmo3_puzzletron_checkpoint(
        source_checkpoint,
        dtype=dtype,
        device=device,
    )
    model = loaded.model
    selection = select_changed_olmo3_ffns_for_training(model)
    if not selection.parameters:
        raise RuntimeError("recovery smoke has no structurally changed FFN to update")
    vocab_size = _value(model.config, "vocab_size")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size <= 1
    ):
        raise ValueError("OLMo-3 recovery smoke requires a valid vocabulary size")
    target_device = torch.device(device)
    input_ids = (
        torch.arange(sequence_length, device=target_device, dtype=torch.long)
        .remainder(vocab_size - 1)
        .add(1)
        .unsqueeze(0)
    )
    model.config.use_cache = False
    model.eval()
    probes_before = tuple(
        parameter.detach().reshape(-1)[: min(parameter.numel(), 4096)].cpu().clone()
        for parameter in selection.parameters
    )
    optimizer = torch.optim.AdamW(
        selection.parameters,
        lr=float(learning_rate),
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    result = model(input_ids=input_ids, use_cache=False)
    logits = getattr(result, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.shape[:2] != input_ids.shape:
        raise RuntimeError("OLMo-3 recovery smoke returned invalid logits")
    loss = nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("OLMo-3 recovery smoke produced a non-finite loss")
    loss.backward()
    if any(parameter.grad is None for parameter in selection.parameters):
        raise RuntimeError("a selected Puzzletron FFN parameter received no gradient")
    if any(
        not bool(torch.isfinite(parameter.grad).all())
        for parameter in selection.parameters
    ):
        raise RuntimeError("a selected Puzzletron FFN gradient is non-finite")
    loss_value = float(loss.detach().cpu().item())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    parameter_update_observed = any(
        not torch.equal(
            before,
            parameter.detach().reshape(-1)[: before.numel()].cpu(),
        )
        for before, parameter in zip(probes_before, selection.parameters)
    )
    if not parameter_update_observed:
        raise RuntimeError("one optimizer step changed no probed Puzzletron parameter")
    del optimizer, logits, loss, result

    with torch.inference_mode():
        updated_logits = (
            model(input_ids=input_ids, use_cache=False).logits.detach().clone()
        )
    state = collect_modelopt_full_state_dict(model)
    saved = save_olmo3_puzzletron_checkpoint(
        model,
        output_checkpoint,
        state_dict=state,
        source_checkpoint=source_checkpoint,
    )
    reloaded = load_olmo3_puzzletron_checkpoint(
        saved.checkpoint,
        dtype=dtype,
        device=device,
    )
    reloaded.model.config.use_cache = False
    reloaded.model.eval()
    with torch.inference_mode():
        replay_logits = reloaded.model(
            input_ids=input_ids, use_cache=False
        ).logits.detach()
    maximum_absolute_logit_difference = float(
        (updated_logits - replay_logits).abs().max().item()
    )
    if not torch.equal(updated_logits, replay_logits):
        raise RuntimeError(
            "patched reload changed updated logits: "
            f"max_abs={maximum_absolute_logit_difference}"
        )
    if reloaded.structure != saved.structure:
        raise RuntimeError(
            "patched reload changed Puzzletron width/resource accounting"
        )
    return {
        "schema": RECOVERY_SMOKE_SCHEMA,
        "status": "passed",
        "source_checkpoint": str(loaded.checkpoint),
        "output_checkpoint": str(saved.checkpoint),
        "dtype": str(dtype),
        "device": str(target_device),
        "sequence_length": sequence_length,
        "learning_rate": float(learning_rate),
        "loss_before_update": loss_value,
        "parameter_update_observed": parameter_update_observed,
        "trainable_layers": list(selection.trainable_layers),
        "trainable_parameter_values": selection.trainable_parameter_values,
        "frozen_parameter_values": selection.frozen_parameter_values,
        "widths_by_layer": list(saved.structure.widths_by_layer),
        "ffn_parameter_values_by_layer": list(
            saved.structure.ffn_parameter_values_by_layer
        ),
        "total_parameter_values": saved.structure.total_parameter_values,
        "saved_state_key_count": saved.state_key_count,
        "saved_safetensors_shard_count": saved.safetensors_shard_count,
        "saved_safetensors_bytes": saved.safetensors_bytes,
        "exact_logit_parity": True,
        "maximum_absolute_logit_difference": maximum_absolute_logit_difference,
        "claim_boundary": (
            "A one-step structure/save/reload integration smoke on one checkpoint. "
            "It is not a recovery-quality, compression, runtime, memory, energy, "
            "hardware, or superiority result."
        ),
    }


__all__ = [
    "COMPACT_CHECKPOINT_SCHEMA",
    "RECOVERY_SMOKE_SCHEMA",
    "SELECTION_SCHEMA",
    "STRUCTURE_SCHEMA",
    "LoadedOlmo3PuzzletronCheckpoint",
    "Olmo3PuzzletronCompactCheckpoint",
    "Olmo3PuzzletronStructure",
    "Olmo3PuzzletronTrainableSelection",
    "collect_modelopt_full_state_dict",
    "inspect_olmo3_puzzletron_structure",
    "load_olmo3_puzzletron_checkpoint",
    "run_olmo3_puzzletron_recovery_smoke",
    "save_olmo3_puzzletron_checkpoint",
    "select_changed_olmo3_ffns_for_training",
]
