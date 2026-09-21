"""Version-pinned OLMo-3 adapter for NVIDIA ModelOpt Puzzletron.

ModelOpt 0.46.0 does not ship an OLMo-3 AnyModel descriptor.  This module
registers a local descriptor and converter under the explicit name ``olmo3``
without changing the installed NVIDIA package.  Importing this module alone is
cheap and has no ModelOpt side effects; every worker must call
:func:`register_olmo3_puzzletron_adapter` before invoking a Puzzletron entry
point.

The supported search surface is deliberately narrow: heterogeneous SwiGLU FFN
widths, FFN no-ops, and attention no-ops.  KV-head pruning is excluded because
OLMo-3's ``k_norm`` width changes with the KV-head count while ModelOpt's stock
KV pruning path only slices Q/K/V/O tensors.
"""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from dendritic_modeling.integrations.modelopt import (
    PINNED_MODELOPT_VERSION,
    ModelOptEnvironment,
    require_pinned_modelopt,
)

OLMO3_PUZZLETRON_DESCRIPTOR = "olmo3"
PINNED_TRANSFORMERS_VERSION = "4.57.6"
SUPPORTED_OLMO3_ATTENTION_TYPES = frozenset({"sliding_attention", "full_attention"})
PUZZLETRON_HYDRA_RESOLVERS = (
    "to_path",
    "random_int",
    "timedelta_minutes",
    "warmup_steps",
    "get_object",
)

# These are the dependencies declared by the nvidia-modelopt ``puzzletron``
# extra.  Only Hydra is exact-pinned by NVIDIA's 0.46.0 wheel metadata.
PUZZLETRON_EXTRA_REQUIREMENTS: Mapping[str, str | None] = {
    "fire": None,
    "hydra-core": "1.3.2",
    "immutabledict": None,
    "lru-dict": None,
    "pandas": None,
    "typeguard": None,
}

OLMO3_PUZZLETRON_CLAIM_BOUNDARY = (
    "This local adapter makes OLMo-3 structurally usable by NVIDIA ModelOpt "
    "0.46.0 Puzzletron. It is not an NVIDIA-supplied OLMo recipe, does not "
    "validate search quality or deployment speed, and does not support KV-head "
    "pruning. A process-local, version-pinned dispatch corrects ModelOpt 0.46.0's "
    "base-descriptor call while calculating per-layer subblock statistics and is "
    "restored after the run. Compression, quality, memory, and runtime claims "
    "require separate matched experiments and realized-artifact accounting."
)


@dataclass(frozen=True)
class PuzzletronDependencyStatus:
    """Installed identities required before importing Puzzletron."""

    modelopt: ModelOptEnvironment
    transformers_version: str
    puzzletron_extra_versions: Mapping[str, str]


@dataclass(frozen=True)
class Olmo3PuzzletronRegistration:
    """Process-local factory registration retained by ModelOpt."""

    descriptor: str
    modelopt_version: str
    transformers_version: str
    descriptor_class_name: str
    converter_class_name: str
    supported_pruning_mixins: tuple[str, ...]
    kv_head_pruning_supported: bool
    claim_boundary: str


@dataclass(frozen=True)
class BF16FFNParameterBudget:
    """Planning ledger for a BF16 FFN-only, parameter-byte-matched arm.

    The ledger assumes bias-free SwiGLU projections (gate, up, and down) and
    parameter storage of two bytes per value.  Puzzletron's own realized
    subblock statistics and the serialized/runtime tensor bytes remain the
    authoritative measurements.
    """

    target_runtime_bytes: int
    maximum_parameter_count: int
    teacher_parameter_count: int
    fixed_non_ffn_parameter_count: int
    hidden_size: int
    num_layers: int
    width_alignment: int
    maximum_aggregate_intermediate_width: int
    realized_parameter_count: int
    realized_runtime_bytes: int
    target_slack_bytes: int
    claim_boundary: str


@dataclass(frozen=True)
class Olmo3PuzzletronExecution:
    """Isolated pre-run contract and ModelOpt's returned runtime config.

    ``preregistered_config`` has recursively immutable configuration containers
    and is never passed to ModelOpt. ``runtime_config`` is the separate
    execution copy after the official NVIDIA entrypoint has applied its
    in-place runtime transforms. Consumers must validate the two objects
    independently rather than treating ModelOpt's returned object as the
    preregistered experiment contract.
    """

    preregistered_config: Mapping[str, Any]
    runtime_config: Any


@dataclass(frozen=True)
class _PuzzletronAPI:
    environment: ModelOptEnvironment
    transformers_version: str
    model_descriptor_base: type
    model_descriptor_factory: type
    converter_base: type
    converter_factory: type
    ffn_layer_descriptor_base: type
    ffn_pruning_mixin: type
    block_config: type
    attention_config: type
    ffn_config: type
    dummy_block: type[nn.Module]
    matching_zeros: type[nn.Module]
    same: type[nn.Module]
    return_tuple_of_size: Callable[[type[nn.Module], int], type[nn.Module]]
    olmo3_decoder_layer: type[nn.Module]
    olmo3_for_causal_lm: type[nn.Module]
    olmo3_rotary_embedding: type[nn.Module]
    convert_model: Callable[..., Any]
    puzzletron: Callable[..., Any]
    puzzletron_entrypoint_module: Any
    initialize_hydra_config_for_dir: Callable[..., Any]
    instantiate_hydra_config: Callable[..., Any]
    register_hydra_resolvers: Callable[[], None]
    hydra_resolver_is_registered: Callable[[str], bool]
    extra_versions: Mapping[str, str]


@dataclass(frozen=True)
class _AdapterClasses:
    descriptor: type
    converter: type
    ffn_layer_descriptor: type


class _AttentionTypeCarrier(nn.Module):
    """Attribute carrier used where OLMo's outer loop inspects self-attention."""

    def __init__(self, attention_type: str) -> None:
        super().__init__()
        self.attention_type = attention_type


class _Float32RotaryModuleDict(nn.ModuleDict):
    """Keep OLMo-3's precision-sensitive nonpersistent RoPE state in FP32.

    ModelOpt rebuilds rotary modules after assigning checkpoint tensors and then
    calls ``model_shard.type(torch.bfloat16)``.  OLMo-3 computes ``inv_freq`` in
    FP32 and a BF16 round-trip is not reversible, so preserve the original
    values while still honoring device transformations applied to the model.
    """

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True):
        preserved: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        destinations: dict[str, torch.device] = {}
        for name, module in self.items():
            inv_freq = module.inv_freq.detach().clone().float()
            original_inv_freq = module.original_inv_freq.detach().clone().float()
            preserved[name] = (inv_freq, original_inv_freq)
            destinations[name] = fn(inv_freq).device

        result = super()._apply(fn, recurse=recurse)
        for name, (inv_freq, original_inv_freq) in preserved.items():
            module = self[name]
            device = destinations[name]
            module.register_buffer(
                "inv_freq", inv_freq.to(device=device), persistent=False
            )
            module.original_inv_freq = original_inv_freq.to(device=device)
        return result


_REGISTRATION_LOCK = threading.RLock()
_PUZZLETRON_CONFIG_DISPATCH_LOCK = threading.RLock()
_ACTIVE_REGISTRATION: Olmo3PuzzletronRegistration | None = None
_ACTIVE_API: _PuzzletronAPI | None = None
_ACTIVE_CLASSES: _AdapterClasses | None = None
_HYDRA_RESOLVERS_REGISTERED = False
_SUBBLOCK_STATS_DISPATCH_LOCK = threading.RLock()


def inspect_puzzletron_dependencies() -> dict[str, str | None]:
    """Inspect optional Puzzletron distributions without importing ModelOpt."""

    identities: dict[str, str | None] = {}
    for distribution in ("transformers", *PUZZLETRON_EXTRA_REQUIREMENTS):
        try:
            identities[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            identities[distribution] = None
    return identities


def require_pinned_puzzletron_environment() -> PuzzletronDependencyStatus:
    """Fail before factory imports if ModelOpt or its OLMo-facing API drifted."""

    environment = require_pinned_modelopt()
    identities = inspect_puzzletron_dependencies()
    transformers_version = identities.pop("transformers")
    if transformers_version != PINNED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Transformers version drift for the OLMo-3 Puzzletron adapter: "
            f"installed {transformers_version}, required "
            f"{PINNED_TRANSFORMERS_VERSION}"
        )

    missing = sorted(name for name, version in identities.items() if version is None)
    if missing:
        raise RuntimeError(
            "ModelOpt Puzzletron extras are incomplete; install the pinned "
            f"nvidia-modelopt[puzzletron]=={PINNED_MODELOPT_VERSION} "
            f"environment (missing {missing})"
        )
    drifted = {
        name: (identities[name], expected)
        for name, expected in PUZZLETRON_EXTRA_REQUIREMENTS.items()
        if expected is not None and identities[name] != expected
    }
    if drifted:
        raise RuntimeError(f"ModelOpt Puzzletron dependency drift: {drifted}")

    return PuzzletronDependencyStatus(
        modelopt=environment,
        transformers_version=str(transformers_version),
        puzzletron_extra_versions={
            name: str(version) for name, version in identities.items()
        },
    )


def _symbol(module_name: str, name: str) -> Any:
    module = importlib.import_module(module_name)
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise RuntimeError(
            f"ModelOpt 0.46.0 Puzzletron API mismatch: {module_name}.{name} is absent"
        ) from exc


def _load_pinned_puzzletron_api() -> _PuzzletronAPI:
    status = require_pinned_puzzletron_environment()
    try:
        model_descriptor_module = "modelopt.torch.puzzletron.anymodel.model_descriptor"
        converter_module = "modelopt.torch.puzzletron.anymodel.converter"
        pruning_module = "modelopt.torch.puzzletron.pruning"
        block_module = "modelopt.torch.puzzletron.block_config"
        dummy_module = "modelopt.torch.puzzletron.utils.dummy_modules"
        no_op_module = "modelopt.torch.puzzletron.anymodel.puzzformer.no_op"
        olmo_module = "transformers.models.olmo3.modeling_olmo3"

        omega_conf = _symbol("omegaconf", "OmegaConf")
        return _PuzzletronAPI(
            environment=status.modelopt,
            transformers_version=status.transformers_version,
            model_descriptor_base=_symbol(model_descriptor_module, "ModelDescriptor"),
            model_descriptor_factory=_symbol(
                model_descriptor_module, "ModelDescriptorFactory"
            ),
            converter_base=_symbol(converter_module, "Converter"),
            converter_factory=_symbol(converter_module, "ConverterFactory"),
            ffn_layer_descriptor_base=_symbol(
                pruning_module, "FFNIntermediateLayerDescriptor"
            ),
            ffn_pruning_mixin=_symbol(pruning_module, "FFNIntermediatePruningMixIn"),
            block_config=_symbol(block_module, "BlockConfig"),
            attention_config=_symbol(block_module, "AttentionConfig"),
            ffn_config=_symbol(block_module, "FFNConfig"),
            dummy_block=_symbol(dummy_module, "DummyBlock"),
            matching_zeros=_symbol(no_op_module, "MatchingZeros"),
            same=_symbol(no_op_module, "Same"),
            return_tuple_of_size=_symbol(no_op_module, "return_tuple_of_size"),
            olmo3_decoder_layer=_symbol(olmo_module, "Olmo3DecoderLayer"),
            olmo3_for_causal_lm=_symbol(olmo_module, "Olmo3ForCausalLM"),
            olmo3_rotary_embedding=_symbol(olmo_module, "Olmo3RotaryEmbedding"),
            convert_model=_symbol(converter_module, "convert_model"),
            puzzletron=_symbol("modelopt.torch.puzzletron.entrypoint", "puzzletron"),
            puzzletron_entrypoint_module=importlib.import_module(
                "modelopt.torch.puzzletron.entrypoint"
            ),
            initialize_hydra_config_for_dir=_symbol(
                "modelopt.torch.puzzletron.tools.hydra_utils",
                "initialize_hydra_config_for_dir",
            ),
            instantiate_hydra_config=_symbol("hydra.utils", "instantiate"),
            register_hydra_resolvers=_symbol(
                "modelopt.torch.puzzletron.tools.hydra_utils",
                "register_hydra_resolvers",
            ),
            hydra_resolver_is_registered=omega_conf.has_resolver,
            extra_versions=status.puzzletron_extra_versions,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "the pinned ModelOpt Puzzletron API could not be imported; rebuild "
            "the dedicated Puzzletron environment from "
            f"nvidia-modelopt[puzzletron]=={PINNED_MODELOPT_VERSION}"
        ) from exc


def _attention_type_from_layer(layer: nn.Module) -> str:
    attention = getattr(layer, "self_attn", None)
    attention_type = getattr(attention, "attention_type", None)
    if attention_type not in SUPPORTED_OLMO3_ATTENTION_TYPES:
        raise RuntimeError(
            "OLMo-3 layer has no supported self_attn.attention_type: "
            f"{attention_type!r}"
        )
    return str(attention_type)


def _validated_layer_types(config: Any) -> list[str]:
    values = getattr(config, "layer_types", None)
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("OLMo-3 config must define a non-empty layer_types sequence")
    layer_types = [str(value) for value in values]
    unsupported = sorted(set(layer_types) - SUPPORTED_OLMO3_ATTENTION_TYPES)
    if unsupported:
        raise ValueError(f"unsupported OLMo-3 layer_types: {unsupported}")
    return layer_types


def _build_adapter_classes(api: _PuzzletronAPI) -> _AdapterClasses:
    @dataclass
    class Olmo3FFNIntermediateLayerDescriptor(api.ffn_layer_descriptor_base):
        down_proj_name: str = "mlp.down_proj"
        ffn_prefix_name: str = "model.layers.{layer_idx}.mlp"
        linear_weight_names: list[str] = field(
            default_factory=lambda: ["down_proj", "gate_proj", "up_proj"]
        )

    class Olmo3ModelDescriptor(api.model_descriptor_base):
        """OLMo-3 AnyModel structure with post-normalized subblocks."""

        @staticmethod
        def decoder_layer_cls() -> type[nn.Module]:
            return api.olmo3_decoder_layer

        @staticmethod
        def block_config_to_layer_overrides(block_config: Any) -> dict[str, Any]:
            return {
                "intermediate_size": block_config.ffn.intermediate_size,
                "num_key_value_heads": block_config.attention.num_key_value_heads,
            }

        @classmethod
        def create_dummy_block(
            cls, original_layer: nn.Module, block_index: int
        ) -> nn.Module:
            attention_type = _attention_type_from_layer(original_layer)
            dummy = api.dummy_block(block_index=block_index)
            dummy.self_attn = _AttentionTypeCarrier(attention_type)
            return dummy

        @staticmethod
        def attn_no_op_post_init(decoder_layer: nn.Module) -> None:
            attention_type = _attention_type_from_layer(decoder_layer)
            no_op = api.return_tuple_of_size(api.matching_zeros, size=2)()
            no_op.attention_type = attention_type
            decoder_layer.self_attn = no_op
            decoder_layer.post_attention_layernorm = api.same()

        @staticmethod
        def mlp_no_op_post_init(decoder_layer: nn.Module) -> None:
            decoder_layer.mlp = api.matching_zeros()
            decoder_layer.post_feedforward_layernorm = api.same()

        @staticmethod
        def init_rotary_embedding(model: nn.Module, runtime: Any) -> None:
            if not hasattr(runtime, "device"):
                raise ValueError("Puzzletron runtime has no device for rotary rebuild")
            # Match the HF teacher's construction route exactly: generate RoPE
            # frequencies on CPU, transfer them to the runtime device in FP32,
            # and protect them from ModelOpt's subsequent whole-model BF16 cast.
            rotary_embs = _Float32RotaryModuleDict(
                {
                    "sliding_attention": api.olmo3_rotary_embedding(
                        config=model.config,
                        device=torch.device("cpu"),
                        rope_type="default",
                    ),
                    "full_attention": api.olmo3_rotary_embedding(
                        config=model.config,
                        device=torch.device("cpu"),
                    ),
                }
            )
            model.model.rotary_embs = rotary_embs.to(runtime.device)

        @staticmethod
        def input_embedding_name() -> str:
            return "model.embed_tokens"

        @staticmethod
        def output_embedding_name() -> str:
            return "lm_head"

        @staticmethod
        def final_norm_name() -> str:
            return "model.norm"

        @staticmethod
        def layer_block_name(index: int) -> str:
            return f"model.layers.{index}"

        @staticmethod
        def layer_name_predicates(num_layers: int) -> dict[str, Any]:
            import re

            if num_layers <= 0:
                raise ValueError("num_layers must be positive")
            predicates: dict[str, Any] = {
                "embeddings": re.compile(r"^model\.embed_tokens\.weight$"),
                "lm_head": re.compile(r"^(model\.norm\.weight|lm_head\.weight)$"),
            }
            for layer_index in range(num_layers):
                prefix = rf"^model\.layers\.{layer_index}\."
                predicates[f"block_{layer_index}_attention"] = re.compile(
                    prefix
                    + r"(post_attention_layernorm\.weight|"
                    + r"self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)|"
                    + r"self_attn\.(q_norm|k_norm)\.weight)$"
                )
                predicates[f"block_{layer_index}_ffn"] = re.compile(
                    prefix
                    + r"(post_feedforward_layernorm\.weight|"
                    + r"mlp\.(gate_proj|up_proj|down_proj)\.weight)$"
                )
            return predicates

        @staticmethod
        def truncate_pattern_for_subblock(
            lm_config: Any, parent_layer_index: int | None = None
        ) -> None:
            layer_types = _validated_layer_types(lm_config)
            index = 0 if parent_layer_index is None else int(parent_layer_index)
            if index < 0 or index >= len(layer_types):
                raise IndexError(
                    f"parent_layer_index {index} is outside {len(layer_types)} layers"
                )
            lm_config.layer_types = [layer_types[index]]

        @staticmethod
        def pruning_mixins() -> dict[str, Any]:
            # Deliberately no ``kv_heads`` entry: k_norm must be pruned jointly
            # with K/V before that search axis is scientifically admissible.
            return {
                "ffn_intermediate": api.ffn_pruning_mixin(
                    Olmo3FFNIntermediateLayerDescriptor()
                )
            }

    class Olmo3Converter(api.converter_base):
        @staticmethod
        def create_block_configs_from_main_config(config: Any) -> list[dict[str, Any]]:
            layer_types = _validated_layer_types(config)
            num_layers = int(config.num_hidden_layers)
            if len(layer_types) != num_layers:
                raise ValueError(
                    "OLMo-3 layer_types length does not match num_hidden_layers: "
                    f"{len(layer_types)} != {num_layers}"
                )
            return [
                api.block_config(
                    attention=api.attention_config(
                        no_op=False,
                        num_key_value_heads=int(config.num_key_value_heads),
                    ),
                    ffn=api.ffn_config(
                        no_op=False,
                        intermediate_size=int(config.intermediate_size),
                    ),
                ).to_dict()
                for _ in layer_types
            ]

    for cls, name in (
        (Olmo3FFNIntermediateLayerDescriptor, "Olmo3FFNIntermediateLayerDescriptor"),
        (Olmo3ModelDescriptor, "Olmo3ModelDescriptor"),
        (Olmo3Converter, "Olmo3Converter"),
    ):
        cls.__name__ = name
        cls.__qualname__ = name
        cls.__module__ = __name__

    return _AdapterClasses(
        descriptor=Olmo3ModelDescriptor,
        converter=Olmo3Converter,
        ffn_layer_descriptor=Olmo3FFNIntermediateLayerDescriptor,
    )


def _factory_mapping(factory: type) -> dict[str, type]:
    mapping = getattr(factory, "CLASS_MAPPING", None)
    if not isinstance(mapping, dict):
        raise RuntimeError("ModelOpt 0.46.0 factory has no mutable CLASS_MAPPING")
    return mapping


def validate_olmo3_attention_contract(config: Any) -> None:
    """Reject active block configs that change OLMo-3's KV-head count.

    Attention no-ops are admissible and conventionally carry ``None`` heads.
    Every retained attention subblock must use the teacher's global KV-head
    count until an OLMo-specific K/V/``k_norm`` pruning mixin is implemented.
    """

    expected_heads = _config_value(config, "num_key_value_heads")
    blocks = _config_value(config, "block_configs")
    num_layers = _config_value(config, "num_hidden_layers")
    if not isinstance(expected_heads, int) or expected_heads <= 0:
        raise ValueError("OLMo-3 config has no positive num_key_value_heads")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        raise ValueError("OLMo-3 AnyModel config has no block_configs sequence")
    if not isinstance(num_layers, int) or len(blocks) != num_layers:
        raise ValueError("OLMo-3 block_configs do not cover every decoder layer")
    for layer_index, block in enumerate(blocks):
        no_op = bool(_config_value(block, "attention", "no_op"))
        heads = _config_value(block, "attention", "num_key_value_heads")
        if no_op:
            if heads is not None:
                raise ValueError(
                    f"attention no-op at layer {layer_index} retains KV-head metadata"
                )
            continue
        if heads != expected_heads:
            raise ValueError(
                "unsupported OLMo-3 KV-head change at layer "
                f"{layer_index}: {heads!r} != {expected_heads}"
            )


def _is_ffn_intermediate_mixin(value: Any, api: _PuzzletronAPI) -> bool:
    if isinstance(value, api.ffn_pruning_mixin):
        return True
    if isinstance(value, Mapping):
        target = value.get("_target_")
        return isinstance(target, str) and target.rsplit(".", 1)[-1] == (
            "FFNIntermediatePruningMixIn"
        )
    return False


def _is_olmo3_ffn_mixin(
    value: Any,
    api: _PuzzletronAPI,
    classes: _AdapterClasses,
) -> bool:
    """Require both NVIDIA's FFN mixin and this port's OLMo layer descriptor."""

    if not _is_ffn_intermediate_mixin(value, api):
        return False
    layer_descriptor = _config_value(value, "layer_descriptor")
    if isinstance(layer_descriptor, classes.ffn_layer_descriptor):
        return True
    target = _config_value(layer_descriptor, "_target_")
    expected = (
        f"{classes.ffn_layer_descriptor.__module__}."
        f"{classes.ffn_layer_descriptor.__qualname__}"
    )
    return target == expected


def _ensure_puzzletron_hydra_resolvers(api: _PuzzletronAPI) -> None:
    """Accept all NVIDIA resolvers, register none, and reject partial state."""

    global _HYDRA_RESOLVERS_REGISTERED
    with _REGISTRATION_LOCK:
        states = {
            name: bool(api.hydra_resolver_is_registered(name))
            for name in PUZZLETRON_HYDRA_RESOLVERS
        }
        if all(states.values()):
            _HYDRA_RESOLVERS_REGISTERED = True
            return
        if any(states.values()):
            raise RuntimeError(f"partial Puzzletron Hydra resolver state: {states}")
        if _HYDRA_RESOLVERS_REGISTERED:
            raise RuntimeError(
                "Puzzletron Hydra resolvers disappeared after registration"
            )
        api.register_hydra_resolvers()
        missing = [
            name
            for name in PUZZLETRON_HYDRA_RESOLVERS
            if not api.hydra_resolver_is_registered(name)
        ]
        if missing:
            raise RuntimeError(
                f"ModelOpt did not retain Puzzletron Hydra resolvers: {missing}"
            )
        _HYDRA_RESOLVERS_REGISTERED = True


def register_olmo3_puzzletron_adapter() -> Olmo3PuzzletronRegistration:
    """Permanently register the local adapter in this worker process."""

    global _ACTIVE_API, _ACTIVE_CLASSES, _ACTIVE_REGISTRATION
    with _REGISTRATION_LOCK:
        if _ACTIVE_REGISTRATION is not None:
            assert _ACTIVE_API is not None and _ACTIVE_CLASSES is not None
            descriptor_mapping = _factory_mapping(_ACTIVE_API.model_descriptor_factory)
            converter_mapping = _factory_mapping(_ACTIVE_API.converter_factory)
            if (
                descriptor_mapping.get(OLMO3_PUZZLETRON_DESCRIPTOR)
                is not _ACTIVE_CLASSES.descriptor
                or converter_mapping.get(OLMO3_PUZZLETRON_DESCRIPTOR)
                is not _ACTIVE_CLASSES.converter
            ):
                raise RuntimeError("active OLMo-3 Puzzletron factory ownership changed")
            return _ACTIVE_REGISTRATION

        api = _load_pinned_puzzletron_api()
        descriptor_mapping = _factory_mapping(api.model_descriptor_factory)
        converter_mapping = _factory_mapping(api.converter_factory)
        if OLMO3_PUZZLETRON_DESCRIPTOR in descriptor_mapping:
            raise RuntimeError("ModelOpt already has a conflicting olmo3 descriptor")
        if OLMO3_PUZZLETRON_DESCRIPTOR in converter_mapping:
            raise RuntimeError("ModelOpt already has a conflicting olmo3 converter")

        classes = _build_adapter_classes(api)
        api.model_descriptor_factory.register(
            **{OLMO3_PUZZLETRON_DESCRIPTOR: classes.descriptor}
        )
        try:
            api.converter_factory.register(
                **{OLMO3_PUZZLETRON_DESCRIPTOR: classes.converter}
            )
        except Exception:
            if (
                descriptor_mapping.get(OLMO3_PUZZLETRON_DESCRIPTOR)
                is classes.descriptor
            ):
                descriptor_mapping.pop(OLMO3_PUZZLETRON_DESCRIPTOR)
            raise

        if (
            api.model_descriptor_factory.get(OLMO3_PUZZLETRON_DESCRIPTOR)
            is not classes.descriptor
            or api.converter_factory.get(OLMO3_PUZZLETRON_DESCRIPTOR)
            is not classes.converter
        ):
            descriptor_mapping.pop(OLMO3_PUZZLETRON_DESCRIPTOR, None)
            converter_mapping.pop(OLMO3_PUZZLETRON_DESCRIPTOR, None)
            raise RuntimeError("ModelOpt did not retain the local OLMo-3 factories")

        # Hydra object paths can now resolve the local classes, while factory
        # users continue to use the explicit short name ``olmo3``.
        globals()[classes.descriptor.__name__] = classes.descriptor
        globals()[classes.converter.__name__] = classes.converter
        globals()[classes.ffn_layer_descriptor.__name__] = classes.ffn_layer_descriptor

        registration = Olmo3PuzzletronRegistration(
            descriptor=OLMO3_PUZZLETRON_DESCRIPTOR,
            modelopt_version=str(api.environment.version),
            transformers_version=api.transformers_version,
            descriptor_class_name=classes.descriptor.__name__,
            converter_class_name=classes.converter.__name__,
            supported_pruning_mixins=("ffn_intermediate",),
            kv_head_pruning_supported=False,
            claim_boundary=OLMO3_PUZZLETRON_CLAIM_BOUNDARY,
        )
        _ACTIVE_API = api
        _ACTIVE_CLASSES = classes
        _ACTIVE_REGISTRATION = registration
        return registration


def get_registered_olmo3_puzzletron_classes() -> tuple[type, type]:
    """Return descriptor and converter after import-first registration."""

    register_olmo3_puzzletron_adapter()
    assert _ACTIVE_CLASSES is not None
    return _ACTIVE_CLASSES.descriptor, _ACTIVE_CLASSES.converter


def _require_explicit_olmo3_descriptor(descriptor: str) -> None:
    if descriptor != OLMO3_PUZZLETRON_DESCRIPTOR:
        raise ValueError(
            "the local OLMo-3 port requires the explicit Puzzletron descriptor "
            f"{OLMO3_PUZZLETRON_DESCRIPTOR!r}; got {descriptor!r}"
        )


def _load_local_olmo3_config(checkpoint: Path) -> dict[str, Any]:
    root = Path(checkpoint).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"OLMo-3 checkpoint is not a local directory: {root}")
    config_path = root / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError(f"OLMo-3 checkpoint has no regular config.json: {root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "olmo3":
        raise ValueError(
            f"checkpoint model_type is not olmo3: {config.get('model_type')!r}"
        )
    layer_types = config.get("layer_types")
    num_layers = config.get("num_hidden_layers")
    if not isinstance(layer_types, list) or len(layer_types) != num_layers:
        raise ValueError("checkpoint layer_types do not cover every decoder layer")
    unsupported = sorted(set(layer_types) - SUPPORTED_OLMO3_ATTENTION_TYPES)
    if unsupported:
        raise ValueError(
            f"checkpoint has unsupported OLMo-3 layer types: {unsupported}"
        )
    return config


def _load_converted_olmo3_config(checkpoint: Path) -> dict[str, Any]:
    """Load an AnyModel OLMo config and reject unsupported attention changes."""

    config = _load_local_olmo3_config(checkpoint)
    if "block_configs" not in config:
        raise ValueError("converted OLMo-3 config has no block_configs")
    validate_olmo3_attention_contract(config)
    return config


def convert_olmo3_to_anymodel(
    *,
    input_dir: Path,
    output_dir: Path,
    descriptor: str,
) -> Olmo3PuzzletronRegistration:
    """Convert a local OLMo-3 checkpoint after process-local registration."""

    _require_explicit_olmo3_descriptor(descriptor)
    _load_local_olmo3_config(input_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"AnyModel output directory is not empty: {destination}")
    registration = register_olmo3_puzzletron_adapter()
    assert _ACTIVE_API is not None
    _ACTIVE_API.convert_model(
        input_dir=str(Path(input_dir).resolve()),
        output_dir=str(destination.resolve()),
        converter=OLMO3_PUZZLETRON_DESCRIPTOR,
    )
    _load_converted_olmo3_config(destination)
    return registration


def _config_value(config: Any, *names: str) -> Any:
    value = config
    for name in names:
        if isinstance(value, Mapping):
            value = value.get(name)
        else:
            value = getattr(value, name, None)
    return value


def _freeze_preregistered_config(value: Any) -> Any:
    """Recursively freeze an isolated instantiated Hydra configuration."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_preregistered_config(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_preregistered_config(item) for item in value)
    return value


@contextmanager
def _temporary_puzzletron_config_dispatch(
    api: _PuzzletronAPI,
    *,
    execution_config: Any,
    config_dir: Path,
    config_name: str,
    overrides: Sequence[str],
) -> Iterator[None]:
    """Give the official entrypoint one isolated copy of the validated config.

    ModelOpt 0.46.0 composes its Hydra config inside ``puzzletron`` and then
    mutates that object across pipeline stages. Temporarily replace only the
    entrypoint module's imported initializer so the exact separately copied
    config validated by this adapter is the object instantiated for execution.
    The original initializer is restored on every exit and the entrypoint must
    request the registered identity exactly once.
    """

    owner = api.puzzletron_entrypoint_module
    expected = {
        "config_dir": str(config_dir.resolve(strict=True)),
        "config_name": config_name,
        "overrides": list(overrides),
    }
    with _PUZZLETRON_CONFIG_DISPATCH_LOCK:
        original = vars(owner).get("initialize_hydra_config_for_dir")
        if original is not api.initialize_hydra_config_for_dir:
            raise RuntimeError(
                "ModelOpt Puzzletron entrypoint config-initializer ownership changed"
            )
        calls = 0

        def dispatch(**kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls != 1 or kwargs != expected:
                raise RuntimeError(
                    "ModelOpt Puzzletron requested a Hydra config outside the "
                    f"registered execution identity: {kwargs!r}"
                )
            return execution_config

        owner.initialize_hydra_config_for_dir = dispatch
        try:
            yield
        finally:
            ownership_changed = (
                vars(owner).get("initialize_hydra_config_for_dir") is not dispatch
            )
            owner.initialize_hydra_config_for_dir = original
            if ownership_changed:
                raise RuntimeError(
                    "ModelOpt Puzzletron config-initializer ownership changed "
                    "during execution"
                )
            if calls != 1:
                raise RuntimeError(
                    "ModelOpt Puzzletron did not consume exactly one isolated "
                    f"execution config; observed {calls} calls"
                )


@contextmanager
def _temporary_olmo3_subblock_stats_dispatch(
    api: _PuzzletronAPI,
    classes: _AdapterClasses,
) -> Iterator[None]:
    """Route ModelOpt 0.46.0's hard-coded stats call to the OLMo descriptor.

    ``calc_subblock_stats`` in the pinned NVIDIA release calls
    ``ModelDescriptor.truncate_pattern_for_subblock`` directly instead of the
    registered descriptor. That leaves OLMo's 32-entry ``layer_types`` list
    untrimmed when a one-layer analytical model is built and can attribute a
    full-attention layer to the first (sliding-attention) pattern. Patch only
    the process-local base class for the synchronous Puzzletron call, dispatch
    only OLMo configs to our descriptor, preserve the original behavior for
    every other model type, and restore exact class ownership on every exit.
    """

    base = api.model_descriptor_base
    with _SUBBLOCK_STATS_DISPATCH_LOCK:
        original_descriptor = vars(base).get("truncate_pattern_for_subblock")
        if not isinstance(original_descriptor, staticmethod):
            raise RuntimeError(
                "ModelOpt 0.46.0 base stats truncation API changed; expected "
                "one directly owned staticmethod"
            )
        original = original_descriptor.__func__

        def dispatch(lm_config: Any, parent_layer_index: int | None = None) -> Any:
            if _config_value(lm_config, "model_type") == "olmo3":
                return classes.descriptor.truncate_pattern_for_subblock(
                    lm_config, parent_layer_index
                )
            return original(lm_config, parent_layer_index)

        patched_descriptor = staticmethod(dispatch)
        base.truncate_pattern_for_subblock = patched_descriptor
        try:
            yield
        finally:
            ownership_changed = (
                vars(base).get("truncate_pattern_for_subblock")
                is not patched_descriptor
            )
            base.truncate_pattern_for_subblock = original_descriptor
            if ownership_changed:
                raise RuntimeError(
                    "ModelOpt base stats truncation ownership changed during "
                    "the OLMo-3 Puzzletron run"
                )


def run_olmo3_puzzletron(
    *,
    hydra_config_dir: Path,
    hydra_config: str,
    puzzle_dir: Path,
    dataset_path: Path,
    descriptor: str,
) -> Olmo3PuzzletronExecution:
    """Invoke Puzzletron only after validating its converted OLMo-3 teacher.

    The wrapper verifies the teacher AnyModel block configurations, composes the
    same Hydra configuration once for validation, and then calls NVIDIA's entry
    point. It admits only the FFN-intermediate pruning mixin; a KV-head change or
    mixed pruning recipe fails before any search work.
    """

    _require_explicit_olmo3_descriptor(descriptor)
    puzzle_root = Path(puzzle_dir).resolve(strict=True)
    register_olmo3_puzzletron_adapter()
    assert _ACTIVE_API is not None
    _ensure_puzzletron_hydra_resolvers(_ACTIVE_API)
    overrides = [
        f"puzzle_dir={puzzle_root}",
        f"dataset_path={Path(dataset_path).resolve()}",
    ]
    config_dir = Path(hydra_config_dir).resolve(strict=True)
    dataset = Path(dataset_path).resolve()
    preview = _ACTIVE_API.initialize_hydra_config_for_dir(
        config_dir=str(config_dir),
        config_name=hydra_config,
        overrides=overrides,
    )
    configured_descriptor = _config_value(preview, "descriptor")
    if configured_descriptor != OLMO3_PUZZLETRON_DESCRIPTOR:
        raise ValueError(
            "the composed Puzzletron config must explicitly set descriptor: olmo3; "
            f"got {configured_descriptor!r}"
        )
    pruning_mixin = _config_value(preview, "pruning", "pruning_mixin")
    assert _ACTIVE_CLASSES is not None
    if not _is_olmo3_ffn_mixin(pruning_mixin, _ACTIVE_API, _ACTIVE_CLASSES):
        raise ValueError(
            "OLMo-3 Puzzletron requires FFNIntermediatePruningMixIn with the "
            "registered Olmo3FFNIntermediateLayerDescriptor; got "
            f"{pruning_mixin!r}"
        )
    teacher_dir = _config_value(preview, "teacher_dir")
    if not isinstance(teacher_dir, (str, Path)):
        raise ValueError("the composed Puzzletron config has no resolved teacher_dir")
    _load_converted_olmo3_config(Path(teacher_dir))
    preregistered_config = _freeze_preregistered_config(
        _ACTIVE_API.instantiate_hydra_config(copy.deepcopy(preview))
    )
    execution_config = copy.deepcopy(preview)
    with (
        _temporary_puzzletron_config_dispatch(
            _ACTIVE_API,
            execution_config=execution_config,
            config_dir=config_dir,
            config_name=hydra_config,
            overrides=overrides,
        ),
        _temporary_olmo3_subblock_stats_dispatch(_ACTIVE_API, _ACTIVE_CLASSES),
    ):
        runtime_config = _ACTIVE_API.puzzletron(
            hydra_config_dir=str(config_dir),
            hydra_config=hydra_config,
            puzzle_dir=str(puzzle_root),
            dataset_path=str(dataset),
        )
    return Olmo3PuzzletronExecution(
        preregistered_config=preregistered_config,
        runtime_config=runtime_config,
    )


def plan_bf16_ffn_parameter_budget(
    *,
    target_runtime_bytes: int,
    teacher_parameter_count: int,
    hidden_size: int,
    teacher_intermediate_sizes: Sequence[int],
    width_alignment: int = 256,
) -> BF16FFNParameterBudget:
    """Compute the largest aligned aggregate FFN width under a byte target."""

    integers = {
        "target_runtime_bytes": target_runtime_bytes,
        "teacher_parameter_count": teacher_parameter_count,
        "hidden_size": hidden_size,
        "width_alignment": width_alignment,
    }
    if any(not isinstance(value, int) for value in integers.values()):
        raise TypeError("all BF16 budget scalar inputs must be integers")
    if any(value <= 0 for value in integers.values()):
        raise ValueError(f"BF16 budget scalar inputs must be positive: {integers}")
    widths = [int(width) for width in teacher_intermediate_sizes]
    if not widths or any(width <= 0 for width in widths):
        raise ValueError("teacher_intermediate_sizes must be non-empty and positive")

    maximum_parameters = target_runtime_bytes // 2
    teacher_ffn_parameters = 3 * hidden_size * sum(widths)
    fixed_parameters = teacher_parameter_count - teacher_ffn_parameters
    if fixed_parameters < 0:
        raise ValueError("teacher FFN parameter count exceeds teacher_parameter_count")
    available_ffn_parameters = maximum_parameters - fixed_parameters
    if available_ffn_parameters < 0:
        raise ValueError("target byte budget is smaller than fixed non-FFN parameters")

    aggregate_width = available_ffn_parameters // (3 * hidden_size)
    aggregate_width = (aggregate_width // width_alignment) * width_alignment
    realized_parameters = fixed_parameters + 3 * hidden_size * aggregate_width
    realized_bytes = 2 * realized_parameters
    assert realized_bytes <= target_runtime_bytes
    return BF16FFNParameterBudget(
        target_runtime_bytes=target_runtime_bytes,
        maximum_parameter_count=maximum_parameters,
        teacher_parameter_count=teacher_parameter_count,
        fixed_non_ffn_parameter_count=fixed_parameters,
        hidden_size=hidden_size,
        num_layers=len(widths),
        width_alignment=width_alignment,
        maximum_aggregate_intermediate_width=aggregate_width,
        realized_parameter_count=realized_parameters,
        realized_runtime_bytes=realized_bytes,
        target_slack_bytes=target_runtime_bytes - realized_bytes,
        claim_boundary=(
            "Planning arithmetic only for a bias-free BF16 SwiGLU FFN-only arm. "
            "Use Puzzletron subblock statistics and realized serialized/runtime "
            "tensor accounting for evidence."
        ),
    )


__all__ = [
    "OLMO3_PUZZLETRON_CLAIM_BOUNDARY",
    "OLMO3_PUZZLETRON_DESCRIPTOR",
    "PINNED_TRANSFORMERS_VERSION",
    "PUZZLETRON_EXTRA_REQUIREMENTS",
    "PUZZLETRON_HYDRA_RESOLVERS",
    "BF16FFNParameterBudget",
    "Olmo3PuzzletronRegistration",
    "PuzzletronDependencyStatus",
    "convert_olmo3_to_anymodel",
    "get_registered_olmo3_puzzletron_classes",
    "inspect_puzzletron_dependencies",
    "plan_bf16_ffn_parameter_budget",
    "register_olmo3_puzzletron_adapter",
    "require_pinned_puzzletron_environment",
    "run_olmo3_puzzletron",
    "validate_olmo3_attention_contract",
]
