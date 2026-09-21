"""Model-weight-free parameter profiles for dendritic language models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

from dendritic_modeling.networks.architectures.transformer.config_translation import (
    build_dendritic_ffn_kwargs_from_core_config,
)
from dendritic_modeling.networks.architectures.transformer.estimates import (
    estimate_dendritic_ffn_params,
    estimate_gated_dendritic_ffn_params,
)
from dendritic_modeling.networks.architectures.transformer.utils import (
    _to_plain_mapping,
)

_MISSING = object()


def _value(source: Any, names: Sequence[str], default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            value = source[name]
        else:
            value = getattr(source, name, _MISSING)
        if value is not _MISSING and value is not None:
            return value
    if default is _MISSING:
        raise ValueError(f"Missing transformer config field; tried {list(names)}")
    return default


@dataclass(frozen=True)
class TransformerParameterSpec:
    """Minimal decoder-only LM architecture needed for parameter accounting."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    gated_mlp: bool = True
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    mlp_bias: bool = False
    norm_parameters_per_hidden: int = 1
    norms_per_layer: int = 2
    attention_norms_per_layer: int = 0
    final_norm: bool = True
    positional_embedding_parameters: int = 0
    model_type: str = ""

    def __post_init__(self) -> None:
        positive = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "vocab_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
        for field_name in positive:
            if int(getattr(self, field_name)) < 1:
                raise ValueError(f"{field_name} must be >= 1")

    @classmethod
    def from_hf_config(
        cls,
        config: Any,
        *,
        gated_mlp: bool = True,
        norm_parameters_per_hidden: int | None = None,
    ) -> TransformerParameterSpec:
        """Build a spec from a Hugging Face config object or ``config.json``."""
        hidden_size = int(_value(config, ("hidden_size", "d_model", "n_embd")))
        attention_heads = int(
            _value(config, ("num_attention_heads", "n_head", "num_heads"))
        )
        head_dim = int(_value(config, ("head_dim",), hidden_size // attention_heads))
        norm_params = norm_parameters_per_hidden
        if norm_params is None:
            norm_params = 1 if _value(config, ("rms_norm_eps",), None) else 2
        model_type = str(_value(config, ("model_type",), ""))
        attention_norms = int(
            _value(
                config,
                ("attention_norms_per_layer",),
                2 if model_type in {"olmo2", "olmo3"} else 0,
            )
        )
        return cls(
            hidden_size=hidden_size,
            intermediate_size=int(
                _value(config, ("intermediate_size", "ffn_dim", "n_inner"))
            ),
            num_hidden_layers=int(
                _value(config, ("num_hidden_layers", "n_layer", "num_layers"))
            ),
            vocab_size=int(_value(config, ("vocab_size",))),
            num_attention_heads=attention_heads,
            num_key_value_heads=int(
                _value(config, ("num_key_value_heads",), attention_heads)
            ),
            head_dim=head_dim,
            gated_mlp=bool(gated_mlp),
            tie_word_embeddings=bool(_value(config, ("tie_word_embeddings",), True)),
            attention_bias=bool(_value(config, ("attention_bias",), False)),
            mlp_bias=bool(_value(config, ("mlp_bias",), False)),
            norm_parameters_per_hidden=int(norm_params),
            attention_norms_per_layer=attention_norms,
            model_type=model_type,
        )

    def parameter_breakdown(self) -> dict[str, int]:
        """Return an analytical parameter breakdown for the dense baseline."""
        hidden = int(self.hidden_size)
        query_width = int(self.num_attention_heads) * int(self.head_dim)
        kv_width = int(self.num_key_value_heads) * int(self.head_dim)
        attention = hidden * (query_width + 2 * kv_width) + query_width * hidden
        if self.attention_bias:
            attention += query_width + 2 * kv_width + hidden

        projections = 3 if self.gated_mlp else 2
        dense_ffn = projections * hidden * int(self.intermediate_size)
        if self.mlp_bias:
            dense_ffn += (2 if self.gated_mlp else 1) * int(
                self.intermediate_size
            ) + hidden

        block_norm_per_layer = (
            int(self.norms_per_layer) * hidden * int(self.norm_parameters_per_hidden)
        )
        attention_norm_per_layer = (
            int(self.attention_norms_per_layer)
            * hidden
            * int(self.norm_parameters_per_hidden)
        )
        norm_per_layer = block_norm_per_layer + attention_norm_per_layer
        embedding = int(self.vocab_size) * hidden
        lm_head = 0 if self.tie_word_embeddings else embedding
        final_norm = (
            hidden * int(self.norm_parameters_per_hidden) if self.final_norm else 0
        )
        decoder_layer = attention + dense_ffn + norm_per_layer
        total = (
            embedding
            + lm_head
            + int(self.positional_embedding_parameters)
            + final_norm
            + int(self.num_hidden_layers) * decoder_layer
        )
        return {
            "token_embedding": embedding,
            "lm_head": lm_head,
            "positional_embedding": int(self.positional_embedding_parameters),
            "attention_per_layer": attention,
            "dense_ffn_per_layer": dense_ffn,
            "block_norm_per_layer": block_norm_per_layer,
            "attention_norm_per_layer": attention_norm_per_layer,
            "norm_per_layer": norm_per_layer,
            "decoder_layer": decoder_layer,
            "final_norm": final_norm,
            "total": total,
        }


def _reactivation_parameters_per_unit(kwargs: Mapping[str, Any]) -> int:
    if not bool(kwargs.get("reactivate", True)):
        return 0
    activation = str(kwargs.get("reactivation_type", "param_tanh")).lower()
    if activation == "param_tanh":
        return 2
    if activation in {
        "param_tanh_only_m",
        "param_linear_sigmoid",
        "param_linear_tanh",
        "linear_sigmoid",
        "linear_tanh",
    }:
        return 1
    if activation in {"param_linear_tanh_transition", "linear_tanh_transition"}:
        return 2
    return 0


def estimate_dendritic_replacement_from_config(
    *,
    hidden_size: int,
    core_config: Any,
    transformer_replacement: Any,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Translate a replacement config and estimate one dendritic FFN slot."""
    kwargs = build_dendritic_ffn_kwargs_from_core_config(
        core_config,
        transformer_replacement=transformer_replacement,
    )
    tr = _to_plain_mapping(transformer_replacement)
    replacement_kwargs = _to_plain_mapping(tr.get("replacement_kwargs", {}))
    replacement_kind = str(
        replacement_kwargs.get("kind", replacement_kwargs.get("type", "dendritic_ffn"))
    ).lower()
    estimator = (
        estimate_gated_dendritic_ffn_params
        if replacement_kind
        in {"gated_dendritic_ffn", "gated_dendritic", "dendritic_glu"}
        else estimate_dendritic_ffn_params
    )
    estimate_kwargs = {
        "hidden_size": int(hidden_size),
        "dendritic_units": int(kwargs["dendritic_units"]),
        "branch_factors": list(kwargs["branch_factors"]),
        "synapses_per_branch": int(kwargs["synapses_per_branch"]),
        "candidate_size": kwargs.get("indexed_candidate_size"),
        "input_transform": str(kwargs.get("input_transform", "signed_split")),
        "topk_type": str(kwargs.get("topk_type", "indexed_rewire")),
        "output_topk": kwargs.get("output_topk"),
        "output_bias": bool(kwargs.get("output_bias", False)),
        "somatic_synapses": bool(kwargs.get("somatic_synapses", True)),
        "reactivation_parameters_per_unit": _reactivation_parameters_per_unit(kwargs),
        "pre_norm": bool(kwargs.get("pre_norm", False)),
    }
    estimate = estimator(**estimate_kwargs)
    architecture = {
        "replacement_kind": replacement_kind,
        "dendritic_units": int(kwargs["dendritic_units"]),
        "branch_factors": [int(value) for value in kwargs["branch_factors"]],
        "synapses_per_branch": int(kwargs["synapses_per_branch"]),
        "input_transform": str(kwargs.get("input_transform", "signed_split")),
        "topk_type": str(kwargs.get("topk_type", "indexed_rewire")),
        "candidate_size": kwargs.get("indexed_candidate_size"),
        "somatic_synapses": bool(kwargs.get("somatic_synapses", True)),
        "reactivation_type": str(kwargs.get("reactivation_type", "param_tanh")),
        "pre_norm": bool(kwargs.get("pre_norm", False)),
        "output_topk": kwargs.get("output_topk"),
    }
    return estimate, architecture


def estimate_indexed_attention_params(
    transformer: TransformerParameterSpec,
    *,
    query_topk: int,
    key_value_topk: int,
    output_topk: int,
) -> dict[str, Any]:
    """Estimate indexed-sparse Q/K/V/O projections for one attention layer.

    The attention operation, RoPE, softmax, and KV cache are unchanged. Only
    the four learned linear projections are counted as sparse.
    """
    hidden = int(transformer.hidden_size)
    query_width = int(transformer.num_attention_heads) * int(transformer.head_dim)
    key_value_width = int(transformer.num_key_value_heads) * int(transformer.head_dim)
    topks = {
        "query_topk": (int(query_topk), hidden),
        "key_value_topk": (int(key_value_topk), hidden),
        "output_topk": (int(output_topk), query_width),
    }
    for name, (value, input_width) in topks.items():
        if value < 1 or value > input_width:
            raise ValueError(f"{name} must be between 1 and {input_width}")

    query = query_width * int(query_topk)
    key = key_value_width * int(key_value_topk)
    value = key_value_width * int(key_value_topk)
    output = hidden * int(output_topk)
    topology_entries = query + key + value + output
    if transformer.attention_bias:
        query += query_width
        key += key_value_width
        value += key_value_width
        output += hidden
    stored_total = query + key + value + output
    dense_total = int(transformer.parameter_breakdown()["attention_per_layer"])
    row_bitmask_bytes = (
        query_width * ceil(hidden / 8)
        + 2 * key_value_width * ceil(hidden / 8)
        + hidden * ceil(query_width / 8)
    )
    return {
        "dense_total": dense_total,
        "stored_total": stored_total,
        "active_weight_count_proxy": stored_total,
        "query_projection": query,
        "key_projection": key,
        "value_projection": value,
        "output_projection": output,
        "topology_index_entries": topology_entries,
        "runtime_int64_topology_bytes": topology_entries * 8,
        "compact_uint16_topology_bytes": topology_entries * 2,
        "row_bitmask_topology_bytes": row_bitmask_bytes,
        "bfloat16_weight_plus_bitmask_bytes": stored_total * 2 + row_bitmask_bytes,
        "stored_fraction_of_dense": stored_total / dense_total,
        "stored_reduction_factor": dense_total / max(stored_total, 1),
        "query_topk": int(query_topk),
        "key_value_topk": int(key_value_topk),
        "output_topk": int(output_topk),
    }


def _validate_factorization_rank(
    transformer: TransformerParameterSpec,
    rank: int,
) -> int:
    rank = int(rank)
    maximum = min(int(transformer.hidden_size), int(transformer.vocab_size))
    if rank < 1 or rank > maximum:
        raise ValueError(f"rank must be between 1 and {maximum}")
    return rank


def estimate_factorized_token_embedding_params(
    transformer: TransformerParameterSpec,
    *,
    rank: int,
) -> dict[str, Any]:
    """Estimate a rank-factorized token embedding table.

    Storage can fall substantially, but a factorized lookup performs a dense
    rank-to-hidden projection instead of a direct row lookup.
    """
    rank = _validate_factorization_rank(transformer, rank)
    hidden = int(transformer.hidden_size)
    vocab = int(transformer.vocab_size)
    dense_total = vocab * hidden
    stored_total = vocab * rank + rank * hidden
    return {
        "dense_total": dense_total,
        "stored_total": stored_total,
        "token_factor": vocab * rank,
        "hidden_factor": rank * hidden,
        "rank": rank,
        "dense_weights_touched_per_token": hidden,
        "factorized_weights_touched_per_token": rank + rank * hidden,
        "stored_fraction_of_dense": stored_total / dense_total,
        "stored_reduction_factor": dense_total / max(stored_total, 1),
    }


def estimate_factorized_lm_head_params(
    transformer: TransformerParameterSpec,
    *,
    rank: int,
) -> dict[str, Any]:
    """Estimate a rank-factorized full-vocabulary LM output projection."""
    rank = _validate_factorization_rank(transformer, rank)
    hidden = int(transformer.hidden_size)
    vocab = int(transformer.vocab_size)
    dense_total = vocab * hidden
    stored_total = hidden * rank + rank * vocab
    baseline_stored = int(transformer.parameter_breakdown()["lm_head"])
    return {
        "logical_dense_total": dense_total,
        "baseline_stored_total": baseline_stored,
        "stored_total": stored_total,
        "hidden_factor": hidden * rank,
        "vocabulary_factor": rank * vocab,
        "rank": rank,
        "full_logits_weight_count_proxy": stored_total,
        "stored_fraction_of_logical_dense": stored_total / dense_total,
        "stored_reduction_factor": dense_total / max(stored_total, 1),
        "independent_replacement_applicable": not transformer.tie_word_embeddings,
    }


def profile_transformer_component_options(
    transformer: TransformerParameterSpec,
    *,
    attention_layers: Sequence[int] = (),
    query_topk: int | None = None,
    key_value_topk: int | None = None,
    output_topk: int | None = None,
    embedding_rank: int | None = None,
    lm_head_rank: int | None = None,
) -> dict[str, Any]:
    """Profile attention, embedding, and LM-head compression candidates.

    The combined total applies every requested option when the baseline has an
    independent LM head. For tied embeddings, an LM-head-only factorization is
    reported as inapplicable because it would also change the shared embedding.
    """
    baseline = transformer.parameter_breakdown()
    baseline_total = int(baseline["total"])
    removed = 0
    added = 0
    options: dict[str, Any] = {}

    attention_values = (query_topk, key_value_topk, output_topk)
    if any(value is not None for value in attention_values):
        if not all(value is not None for value in attention_values):
            raise ValueError(
                "query_topk, key_value_topk, and output_topk must be set together"
            )
        requested_layers = [int(layer) for layer in attention_layers]
        if len(requested_layers) != len(set(requested_layers)):
            raise ValueError("attention_layers must not contain duplicates")
        layers = sorted(requested_layers)
        if not layers:
            raise ValueError("attention_layers is required for attention profiling")
        if layers[0] < 0 or layers[-1] >= int(transformer.num_hidden_layers):
            raise IndexError("attention layer index is outside transformer depth")
        attention = estimate_indexed_attention_params(
            transformer,
            query_topk=int(query_topk),
            key_value_topk=int(key_value_topk),
            output_topk=int(output_topk),
        )
        dense_removed = len(layers) * int(attention["dense_total"])
        sparse_added = len(layers) * int(attention["stored_total"])
        options["attention"] = {
            "layers": layers,
            "num_layers": len(layers),
            "per_layer": attention,
            "dense_parameters_removed": dense_removed,
            "stored_parameters_added": sparse_added,
            "student_stored_parameters": baseline_total - dense_removed + sparse_added,
        }
        removed += dense_removed
        added += sparse_added
    elif attention_layers:
        raise ValueError(
            "attention TopK values are required when attention_layers is set"
        )

    if embedding_rank is not None:
        embedding = estimate_factorized_token_embedding_params(
            transformer,
            rank=embedding_rank,
        )
        options["token_embedding"] = {
            **embedding,
            "student_stored_parameters": baseline_total
            - int(embedding["dense_total"])
            + int(embedding["stored_total"]),
        }
        removed += int(embedding["dense_total"])
        added += int(embedding["stored_total"])

    lm_head_applicable = True
    if lm_head_rank is not None:
        lm_head = estimate_factorized_lm_head_params(
            transformer,
            rank=lm_head_rank,
        )
        lm_head_applicable = bool(lm_head["independent_replacement_applicable"])
        options["lm_head"] = dict(lm_head)
        if lm_head_applicable:
            options["lm_head"]["student_stored_parameters"] = (
                baseline_total
                - int(lm_head["baseline_stored_total"])
                + int(lm_head["stored_total"])
            )
            removed += int(lm_head["baseline_stored_total"])
            added += int(lm_head["stored_total"])
        else:
            options["lm_head"]["student_stored_parameters"] = None
            options["lm_head"]["inapplicable_reason"] = (
                "The baseline ties its LM head to the token embedding; profile a "
                "shared factorized embedding/head instead of replacing the head alone."
            )

    combined_applicable = lm_head_applicable
    combined_total = baseline_total - removed + added if combined_applicable else None
    return {
        "schema_version": 1,
        "definitions": {
            "stored_parameters": "Learned scalar parameters stored by the model.",
            "attention_active_weight_count_proxy": (
                "Selected projection weights touched per token; attention-score and "
                "value-mixing costs are excluded."
            ),
            "attention_topology_bytes": (
                "Non-learned connectivity metadata. Runtime int64 indices can exceed "
                "the sparse weights; compact checkpoint encodings do not imply a "
                "faster runtime kernel."
            ),
            "full_logits_weight_count_proxy": (
                "Factor weights used to produce every vocabulary logit; this is not "
                "a measured FLOP or latency value."
            ),
        },
        "transformer": asdict(transformer),
        "baseline": baseline,
        "options": options,
        "combined_candidate": {
            "applicable": combined_applicable,
            "stored_parameters": combined_total,
            "stored_fraction_of_baseline": (
                combined_total / baseline_total if combined_total is not None else None
            ),
            "stored_reduction_factor": (
                baseline_total / max(combined_total, 1)
                if combined_total is not None
                else None
            ),
        },
    }


def profile_dendritic_lm(
    transformer: TransformerParameterSpec,
    *,
    dendritic_per_layer: Mapping[str, int],
    replaced_layers: Sequence[int],
    replacement_architecture: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Profile baseline and dendritic-student parameter counts.

    Active counts are an active-weight-count proxy. They are deliberately not
    reported as FLOPs: hardware speedups must be established by measured
    kernel benchmarks.
    """
    requested_layers = [int(layer) for layer in replaced_layers]
    if len(requested_layers) != len(set(requested_layers)):
        raise ValueError("replaced_layers must not contain duplicates")
    layers = sorted(requested_layers)
    if not layers:
        raise ValueError("replaced_layers must contain at least one layer")
    if layers[0] < 0 or layers[-1] >= int(transformer.num_hidden_layers):
        raise IndexError("replaced layer index is outside the transformer depth")

    baseline = transformer.parameter_breakdown()
    count = len(layers)
    removed = count * int(baseline["dense_ffn_per_layer"])
    stored_added = count * int(dendritic_per_layer["stored_total"])
    active_added = count * int(dendritic_per_layer["active_total"])
    stored_total = int(baseline["total"]) - removed + stored_added
    active_proxy_total = int(baseline["total"]) - removed + active_added
    baseline_total = int(baseline["total"])

    return {
        "schema_version": 1,
        "definitions": {
            "stored_parameters": "Learned scalar parameters stored by the model.",
            "active_weight_count_proxy": (
                "Stored non-FFN parameters plus dendritic weights used in one "
                "forward pass; this is not a FLOP or latency estimate."
            ),
        },
        "transformer": asdict(transformer),
        "baseline": baseline,
        "replacement": {
            "layers": layers,
            "num_layers": count,
            "architecture": dict(replacement_architecture or {}),
            "per_layer": dict(dendritic_per_layer),
            "dense_ffn_parameters_removed": removed,
            "stored_parameters_added": stored_added,
            "active_parameters_added": active_added,
        },
        "student": {
            "stored_parameters": stored_total,
            "active_weight_count_proxy": active_proxy_total,
            "stored_fraction_of_baseline": stored_total / baseline_total,
            "active_proxy_fraction_of_baseline": active_proxy_total / baseline_total,
            "stored_reduction_factor": baseline_total / max(stored_total, 1),
            "active_proxy_reduction_factor": baseline_total
            / max(active_proxy_total, 1),
            "ffn_stored_reduction_factor": removed / max(stored_added, 1),
            "ffn_active_reduction_factor": removed / max(active_added, 1),
        },
    }


def profile_dendritic_lm_config(
    experiment_config: Any,
    hf_config: Any,
    *,
    gated_mlp: bool = True,
) -> dict[str, Any]:
    """Profile a typed experiment config against an HF config object/dict."""
    transformer = TransformerParameterSpec.from_hf_config(
        hf_config,
        gated_mlp=gated_mlp,
    )
    model_config = experiment_config.model
    replacement_config = model_config.transformer_replacement
    layers = list(replacement_config.layers)
    estimate, architecture = estimate_dendritic_replacement_from_config(
        hidden_size=transformer.hidden_size,
        core_config=model_config.core,
        transformer_replacement=replacement_config,
    )
    report = profile_dendritic_lm(
        transformer,
        dendritic_per_layer=estimate,
        replaced_layers=layers,
        replacement_architecture=architecture,
    )
    report["model_name"] = str(replacement_config.model_name)
    return report
