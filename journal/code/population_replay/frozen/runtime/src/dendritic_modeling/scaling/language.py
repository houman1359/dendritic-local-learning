"""Experimental, from-scratch Llama shell for dendritic FFN scaling studies.

This is a model/accounting prototype, not a language-training framework. It
loads no pretrained model, tokenizer, or dataset. Native attention, RMS norms,
RoPE, residual connections, and tied embedding/head weights are shared across
families; only the block MLP changes when ``ffn_model`` is supplied.
"""

from __future__ import annotations

import math
import random
from copy import deepcopy
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from dendritic_modeling.scaling.models import (
    _IndexedAffineReadout,
    build_model,
    match_parameter_budget,
    model_report,
)

_DEFAULTS = {
    "max_position_embeddings": 512,
    "seed": 0,
    "ffn_model": None,
    "token_chunk_size": 0,
    "checkpoint_ffn": False,
}
_REQUIRED = {
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
}


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _normalize_spec(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise TypeError("language model spec must be a dict")
    unknown = set(spec) - _REQUIRED - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown language model spec keys: {sorted(unknown)}")
    result = {**deepcopy(_DEFAULTS), **deepcopy(spec)}
    for field in _REQUIRED | {"max_position_embeddings"}:
        result[field] = _integer(result.get(field), field)
    if result["vocab_size"] < 3:
        raise ValueError("vocab_size must accommodate default BOS/EOS IDs 1 and 2")
    hidden, heads = result["hidden_size"], result["num_attention_heads"]
    if hidden % heads or (hidden // heads) % 2:
        raise ValueError("hidden_size must divide into even attention head dimensions")
    result["seed"] = _integer(result["seed"], "seed", 0)
    result["token_chunk_size"] = _integer(
        result["token_chunk_size"], "token_chunk_size", 0
    )
    if not isinstance(result["checkpoint_ffn"], bool):
        raise ValueError("checkpoint_ffn must be a bool")
    if result["ffn_model"] is None:
        if result["token_chunk_size"] or result["checkpoint_ffn"]:
            raise ValueError(
                "FFN chunk/checkpoint options require ffn_model; the native "
                "baseline keeps its native MLP modules"
            )
    elif not isinstance(result["ffn_model"], dict):
        raise TypeError("ffn_model must be a dict or None")
    else:
        for field in ("input_dim", "output_dim"):
            if field in result["ffn_model"] and result["ffn_model"][field] != hidden:
                raise ValueError(f"ffn_model.{field} must equal hidden_size")
    return result


class TokenwiseScalingFFN(nn.Module):
    """Apply an existing scaling model independently to each hidden token.

    Chunking bounds each temporary gather but does not, by itself, reduce the
    sum of saved training activations. Non-reentrant checkpointing recomputes
    each chunk's FFN during backward and is active only in training with grads.
    The full output and the inputs needed for recomputation still occupy memory.
    """

    def __init__(
        self,
        core: nn.Module,
        *,
        hidden_size: int,
        token_chunk_size: int = 0,
        checkpoint_ffn: bool = False,
    ):
        super().__init__()
        self.core = core
        self.hidden_size = hidden_size
        self.token_chunk_size = token_chunk_size
        self.checkpoint_ffn = checkpoint_ffn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim < 2 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"FFN expects [..., {self.hidden_size}] hidden states")
        original_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, self.hidden_size)
        if not flat.shape[0]:
            raise ValueError("FFN requires at least one token")
        chunk_size = self.token_chunk_size or flat.shape[0]
        use_checkpoint = (
            self.checkpoint_ffn and self.training and torch.is_grad_enabled()
        )
        outputs = []
        for chunk in flat.split(chunk_size, dim=0):
            outputs.append(
                checkpoint(self.core, chunk, use_reentrant=False)
                if use_checkpoint
                else self.core(chunk)
            )
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        return output.reshape(original_shape)


def build_language_model(spec: dict) -> nn.Module:
    """Build a random FP32 causal LM entirely from local installed code.

    Required outer fields are vocab_size, hidden_size, num_hidden_layers,
    num_attention_heads, and intermediate_size. Defaults use tied embeddings,
    full multi-head attention, native SwiGLU, zero attention dropout, SDPA,
    and disabled KV caching. No teacher or pretrained weights are loaded.

    ``ffn_model=None`` leaves native SwiGLU modules intact. Otherwise the dict
    uses ``scaling.models.build_model``'s schema, with input/output dimensions
    inferred from hidden_size. Its width, contacts, and morphology affect only
    the FFNs; intermediate_size remains the native baseline's reference width.
    FFN parameter/support seeds default to the outer seed and are offset by
    104729 per layer. Explicit readout seeds receive the same layer offset;
    omitted readout seeds inherit the already offset FFN seeds. The common
    outer model is initialized before patching,
    preserving identical outer parameters for identical outer specs and seed.

    Save the resolved spec plus state_dict and reconstruct through this factory.
    Plain Hugging Face from_pretrained cannot reconstruct the custom FFN types
    from the stock Llama config alone. ``match_language_parameter_budget``
    matches actual whole-model counts; optimizer/data state remains outside
    this prototype.
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    resolved = _normalize_spec(spec)
    config = LlamaConfig(
        vocab_size=resolved["vocab_size"],
        hidden_size=resolved["hidden_size"],
        intermediate_size=resolved["intermediate_size"],
        num_hidden_layers=resolved["num_hidden_layers"],
        num_attention_heads=resolved["num_attention_heads"],
        num_key_value_heads=resolved["num_attention_heads"],
        max_position_embeddings=resolved["max_position_embeddings"],
        hidden_act="silu",
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    ffn_receipts = []
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(resolved["seed"])
        model = LlamaForCausalLM(config).float()
        base_ffn = resolved["ffn_model"]
        if base_ffn is not None:
            initial_seed = _integer(
                base_ffn.get("seed", resolved["seed"]), "ffn_model.seed", 0
            )
            topology_seed = _integer(
                base_ffn.get("topology_seed", initial_seed),
                "ffn_model.topology_seed",
                0,
            )
            for index, layer in enumerate(model.model.layers):
                ffn_spec = {
                    **base_ffn,
                    "input_dim": resolved["hidden_size"],
                    "output_dim": resolved["hidden_size"],
                    "seed": initial_seed + index * 104729,
                    "topology_seed": topology_seed + index * 104729,
                }
                if "readout" in base_ffn:
                    readout = deepcopy(base_ffn["readout"])
                    if isinstance(readout, dict):
                        for field in ("parameter_seed", "topology_seed"):
                            if field in readout:
                                readout[field] = (
                                    _integer(
                                        readout[field], f"ffn_model.readout.{field}", 0
                                    )
                                    + index * 104729
                                )
                    ffn_spec["readout"] = readout
                core = build_model(ffn_spec).float()
                ffn_receipts.append(model_report(core)["resolved_spec"])
                layer.mlp = TokenwiseScalingFFN(
                    core,
                    hidden_size=resolved["hidden_size"],
                    token_chunk_size=resolved["token_chunk_size"],
                    checkpoint_ffn=resolved["checkpoint_ffn"],
                )
        model._scaling_language_spec = deepcopy(resolved)
        model._scaling_language_ffn_specs = ffn_receipts
    return model


def _rms_scale_parameters(projection: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Select one affine output stage whose scaling scales the whole map once."""
    if type(projection) is nn.Linear:
        return list(projection.named_parameters())
    if (
        type(projection) is nn.Sequential
        and len(projection) == 2
        and all(type(factor) is nn.Linear for factor in projection)
        and projection[0].bias is None
        and projection[0].out_features == projection[1].in_features
    ):
        return [
            (f"1.{name}", value) for name, value in projection[1].named_parameters()
        ]
    if (
        type(projection) is _IndexedAffineReadout
        and projection.projection.weight_transform == "identity"
    ):
        parameters = [("projection.pre_w", projection.projection.pre_w)]
        if projection.bias is not None:
            parameters.append(("bias", projection.bias))
        return parameters
    raise ValueError("Unsupported final affine projection for FFN RMS scaling")


def condition_ffn_output_rms(
    model: nn.Module, input_ids: torch.Tensor, *, target_rms: float
) -> dict:
    """Rescale existing FFN output projections once, using training inputs only.

    Call before optimization with an unpadded batch selected exclusively from
    training data; this model helper cannot verify data provenance. One decoder
    pass runs in eval/no_grad with autocast disabled. Each layer's complete FFN
    output is measured, its final affine projection weight and bias are scaled,
    and the corrected output is returned to the downstream residual stream in
    that same pass. Native SwiGLU uses ``mlp.down_proj``; every custom family
    uses ``mlp.core.readout``. A low-rank readout scales only its final factor;
    an indexed readout scales its signed stored weights and optional bias.
    Positive weight transforms and other projection types are unsupported.
    No continuing forward multiplier is installed.

    RMS averages squared values across all supplied tokens and hidden features.
    Token variation is RMS after removing each feature's mean across tokens;
    it distinguishes variation from a large featurewise constant output. The
    reported after values measure the corrected output in this pass. A fresh
    forward can differ by floating-point rounding from scaling the projection
    before its matrix multiplication.

    Learned parameters, supports and buffer inventories do not change. Caller
    torch/Python/NumPy RNG and individual module train/eval flags are preserved.
    Existing projection values and buffers are restored if calibration fails.
    This is an optional initialization control, not a performance claim.
    """
    if (
        isinstance(target_rms, bool)
        or not isinstance(target_rms, Real)
        or not math.isfinite(target_rms)
        or target_rms <= 0
    ):
        raise ValueError("target_rms must be positive and finite")
    target_rms = float(target_rms)
    if not hasattr(model, "_scaling_language_spec"):
        raise ValueError("Expected a model created by build_language_model")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or not input_ids.numel()
        or input_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            "input_ids must be a nonempty [batch, sequence] integer tensor"
        )
    devices = {parameter.device for parameter in model.parameters()}
    if len(devices) != 1 or input_ids.device not in devices:
        raise ValueError("Model and input_ids must share a single device")
    device = input_ids.device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("RMS initialization currently supports CPU or CUDA")

    projections = []
    for index, layer in enumerate(model.model.layers):
        if isinstance(layer.mlp, TokenwiseScalingFFN):
            projection, suffix = layer.mlp.core.readout, "core.readout"
        else:
            projection, suffix = getattr(layer.mlp, "down_proj", None), "down_proj"
        try:
            scale_parameters = _rms_scale_parameters(projection)
        except ValueError as error:
            raise ValueError(
                f"Layer {index} lacks a supported final affine projection"
            ) from error
        projections.append(
            (
                layer.mlp,
                projection,
                f"model.layers.{index}.mlp.{suffix}",
                scale_parameters,
            )
        )
    saved_parameters = [
        (parameter, parameter.detach().clone())
        for _, projection, _, _ in projections
        for parameter in projection.parameters()
    ]
    saved_buffers = [(buffer, buffer.detach().clone()) for buffer in model.buffers()]
    training_flags = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cuda_devices = [device.index] if device.type == "cuda" else []
    records, hooks = [], []
    successful = False

    def statistics(output):
        if not isinstance(output, torch.Tensor) or not output.is_floating_point():
            raise ValueError("Expected floating FFN output")
        flat = output.detach().reshape(-1, output.shape[-1]).double()
        rms = flat.square().mean().sqrt().item()
        if not math.isfinite(rms) or rms <= 0:
            raise ValueError("Observed FFN output RMS must be positive and finite")
        variation = (
            (flat - flat.mean(dim=0, keepdim=True)).square().mean().sqrt().item()
        )
        return rms, variation

    def hook_for(index, scale_parameters, projection_name):
        def condition(_module, _inputs, output):
            if len(records) != index:
                raise ValueError(
                    "Expected exactly one FFN call per layer in decoder order"
                )
            before, variation_before = statistics(output)
            multiplier = target_rms / before
            if not math.isfinite(multiplier) or multiplier <= 0:
                raise ValueError(
                    "FFN RMS scaling multiplier is not positive and finite"
                )
            for _, parameter in scale_parameters:
                parameter.mul_(multiplier)
                if not torch.isfinite(parameter).all():
                    raise ValueError(
                        "FFN RMS scaling produced nonfinite projection parameters"
                    )
            corrected = output * multiplier
            after, variation_after = statistics(corrected)
            records.append(
                {
                    "layer": index,
                    "projection_name": projection_name,
                    "scaled_parameters": [
                        f"{projection_name}.{name}" for name, _ in scale_parameters
                    ],
                    "before_rms": before,
                    "after_rms": after,
                    "multiplier": multiplier,
                    "before_token_variation_rms": variation_before,
                    "after_token_variation_rms": variation_after,
                }
            )
            return corrected

        return condition

    try:
        model.eval()
        with (
            torch.random.fork_rng(devices=cuda_devices),
            torch.no_grad(),
            torch.autocast(device_type=device.type, enabled=False),
        ):
            for index, (ffn, _, name, scale_parameters) in enumerate(projections):
                hooks.append(
                    ffn.register_forward_hook(hook_for(index, scale_parameters, name))
                )
            model.model(input_ids=input_ids, use_cache=False, return_dict=True)
            if len(records) != len(projections):
                raise ValueError("Calibration did not observe every FFN layer")
        successful = True
    finally:
        for hook in hooks:
            hook.remove()
        with torch.no_grad():
            if not successful:
                for parameter, original in saved_parameters:
                    parameter.copy_(original)
            for buffer, original in saved_buffers:
                if not torch.equal(buffer, original):
                    buffer.copy_(original)
        for module, training in training_flags:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    return {
        "schema": "dendritic_scaling_language_ffn_rms_initialization_v1",
        "target_rms": target_rms,
        "calibration_batch_shape": list(input_ids.shape),
        "calibration_token_count": input_ids.numel(),
        "token_variation_definition": "RMS after subtracting each hidden feature's mean across supplied tokens",
        "layers": records,
    }


def language_model_report(model: nn.Module) -> dict:
    """Report actual unique whole-LM parameters, counting tied weights once."""
    if not hasattr(model, "_scaling_language_spec"):
        raise ValueError("Expected a model created by build_language_model")
    report = model_report(model)
    all_parameters = {id(p): p for p in model.parameters()}
    ffn_parameters = {
        id(p): p for layer in model.model.layers for p in layer.mlp.parameters()
    }
    embedding_parameters = {id(p): p for p in model.get_input_embeddings().parameters()}
    head_parameters = {id(p): p for p in model.get_output_embeddings().parameters()}
    shared_ids = set(embedding_parameters) & set(head_parameters)
    shell_parameters = {
        key: p for key, p in all_parameters.items() if key not in ffn_parameters
    }

    def count(values):
        return sum(p.numel() for p in values)

    spec = deepcopy(model._scaling_language_spec)
    report.update(
        {
            "schema": "dendritic_parameter_scaling_language_prototype_v1",
            "experimental_prototype": True,
            "budget_matched": False,
            "resolved_spec": spec,
            "ffn_family": (
                "native_swiglu"
                if spec["ffn_model"] is None
                else spec["ffn_model"]["family"]
            ),
            "ffn_parameters": count(ffn_parameters.values()),
            "shell_parameters": count(shell_parameters.values()),
            "embedding_parameters": count(embedding_parameters.values()),
            "lm_head_unique_parameters": count(
                p for key, p in head_parameters.items() if key not in shared_ids
            ),
            "tied_embedding_head_parameters": count(
                embedding_parameters[key] for key in shared_ids
            ),
            "embedding_head_tied": bool(shared_ids),
            "attention_parameters": sum(
                p.numel()
                for layer in model.model.layers
                for p in layer.self_attn.parameters()
            ),
            "ffn_parameters_per_layer": [
                sum(p.numel() for p in layer.mlp.parameters())
                for layer in model.model.layers
            ],
            "ffn_reports": [
                (
                    model_report(layer.mlp.core)
                    if isinstance(layer.mlp, TokenwiseScalingFFN)
                    else model_report(layer.mlp)
                )
                for layer in model.model.layers
            ],
            "ffn_layer_specs": deepcopy(model._scaling_language_ffn_specs),
            "attention_implementation": model.config._attn_implementation,
            "token_chunk_size": spec["token_chunk_size"],
            "checkpoint_ffn": spec["checkpoint_ffn"],
            "parameter_dtypes": sorted({str(p.dtype) for p in all_parameters.values()}),
        }
    )
    return report


def _minimum_ffn_width(spec: dict) -> int:
    """Smallest width preserving hidden contacts and output support/rank."""
    depth = _integer(spec.get("network_depth", 1), "ffn_model.network_depth")
    minimum = 1
    if (
        depth > 1
        and not spec.get("clip_contacts", False)
        and spec.get("family") != "dense"
    ):
        contacts_e = _integer(spec.get("contacts_e", 4), "ffn_model.contacts_e")
        contacts_i = _integer(spec.get("contacts_i", 0), "ffn_model.contacts_i", 0)
        minimum = (
            contacts_e + contacts_i
            if spec.get("family") == "sparse"
            else max(contacts_e, contacts_i)
        )
    readout = spec.get("readout", {})
    if not isinstance(readout, dict):
        raise ValueError("ffn_model.readout must be a dict")
    for field in ("topk", "rank"):
        if field in readout:
            minimum = max(
                minimum, _integer(readout[field], f"ffn_model.readout.{field}")
            )
    return minimum


def match_language_parameter_budget(
    spec: dict, target_parameters: int, tolerance: float = 0.02
) -> tuple[dict, dict]:
    """Match whole unique learned parameters while preserving the outer shell.

    Native SwiGLU changes only intermediate_size. Custom models change only
    ffn_model.width (besides explicitly resolving defaults). Vocabulary,
    hidden size, layer count, attention, tied embeddings, contacts, and
    morphology remain fixed. The custom model's intermediate_size is its
    native initialization reference and is preserved.

    Instantiate a minimal native reference once to count the actual shell.
    Search custom widths using single-FFN models, then instantiate the final
    complete LM and enforce the requested tolerance. No padding parameters
    fill gaps. Fixed sparse readout supports can leave hidden outputs unused;
    their learned parameters still count and support coverage is reported.
    Targets at or below the shell floor are rejected. Discrete
    widths can make otherwise valid targets unattainable at a strict tolerance.

    For paired outer initial weights across families, first match the native
    model, then pass its resolved intermediate_size into every custom spec.
    This makes the common native construction consume the same RNG stream
    before FFN replacement. Matching alone equates neither FLOPs nor training
    dynamics. Residual-output initialization is not changed by this function.
    """
    target = _integer(target_parameters, "target_parameters")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(tolerance)
        or not 0 <= tolerance < 1
    ):
        raise ValueError("tolerance must be finite and in [0, 1)")
    tolerance = float(tolerance)
    resolved = _normalize_spec(spec)
    reference_spec = {
        **resolved,
        "ffn_model": None,
        "intermediate_size": 1,
        "token_chunk_size": 0,
        "checkpoint_ffn": False,
    }
    reference_model = build_language_model(reference_spec)
    reference_report = language_model_report(reference_model)
    shell = reference_report["shell_parameters"]
    native_parameters_per_intermediate_unit = reference_report["ffn_parameters"]
    del reference_model
    if target <= shell:
        raise ValueError(
            f"target_parameters={target} is at or below the fixed shell floor "
            f"of {shell}; no positive FFN budget remains"
        )

    if resolved["ffn_model"] is None:
        ideal = (target - shell) / native_parameters_per_intermediate_unit
        choices = {max(1, math.floor(ideal)), max(1, math.ceil(ideal))}
        width = min(
            choices,
            key=lambda value: (
                abs(shell + value * native_parameters_per_intermediate_unit - target),
                value,
            ),
        )
        resolved["intermediate_size"] = width
        adjusted_field = "intermediate_size"
    else:
        base = deepcopy(resolved["ffn_model"])
        seed = _integer(base.get("seed", resolved["seed"]), "ffn_model.seed", 0)
        base.update(
            input_dim=resolved["hidden_size"],
            output_dim=resolved["hidden_size"],
            seed=seed,
            topology_seed=_integer(
                base.get("topology_seed", seed), "ffn_model.topology_seed", 0
            ),
        )
        minimum_width = _minimum_ffn_width(base)
        minimum_model = build_model({**base, "width": minimum_width})
        minimum_report = model_report(minimum_model)
        del minimum_model
        per_layer_target = (target - shell) / resolved["num_hidden_layers"]
        if per_layer_target <= minimum_report["total_parameters"]:
            candidate = minimum_report["resolved_spec"]
        else:
            # Request the nearest per-FFN width without prematurely imposing a
            # relative FFN tolerance: the contract is on whole-model P.
            candidate, _ = match_parameter_budget(
                base,
                max(1, round(per_layer_target)),
                tolerance=math.nextafter(1.0, 0.0),
            )
        # Integer division/rounding can place the per-layer target on a tie.
        # Inspect adjacent real widths against the original whole-model target.
        candidates = []
        for width in sorted(
            {
                max(minimum_width, candidate["width"] - 1),
                candidate["width"],
                candidate["width"] + 1,
            }
        ):
            core = build_model({**candidate, "width": width})
            report = model_report(core)
            del core
            predicted = (
                shell + resolved["num_hidden_layers"] * report["total_parameters"]
            )
            candidates.append((abs(predicted - target), predicted, report))
        best = min(candidates, key=lambda value: (value[0], value[1]))[2]
        resolved["ffn_model"] = deepcopy(best["resolved_spec"])
        adjusted_field = "ffn_model.width"

    model = build_language_model(resolved)
    report = language_model_report(model)
    del model
    if report["shell_parameters"] != shell:
        raise RuntimeError("Language budget matching changed the instantiated shell")
    error = abs(report["total_parameters"] - target) / target
    if error > tolerance:
        width = (
            resolved["intermediate_size"]
            if resolved["ffn_model"] is None
            else resolved["ffn_model"]["width"]
        )
        raise ValueError(
            f"No {adjusted_field} matches target_parameters={target} within "
            f"tolerance={tolerance:g}; nearest feasible value={width} has "
            f"{report['total_parameters']} parameters (relative error={error:.6g}, "
            f"shell_parameters={shell})"
        )
    report.update(
        {
            "budget_matched": True,
            "target_parameters": target,
            "budget_tolerance": tolerance,
            "budget_relative_error": error,
            "budget_adjusted_field": adjusted_field,
            "budget_scope": "whole_model_unique_learned_parameters",
            "native_reference_intermediate_size": resolved["intermediate_size"],
        }
    )
    return deepcopy(report["resolved_spec"]), report


__all__ = [
    "TokenwiseScalingFFN",
    "build_language_model",
    "language_model_report",
    "match_language_parameter_budget",
]
