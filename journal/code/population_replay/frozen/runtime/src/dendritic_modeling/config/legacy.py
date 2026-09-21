"""Compatibility adapters for deprecated config spellings.

Active builders should consume canonical config keys.  Deprecated public YAML
spellings are normalized here so compatibility stays explicit and contained.
"""

from __future__ import annotations

import logging
from typing import Any

from dendritic_modeling.config.conversion import to_plain_dict

logger = logging.getLogger(__name__)


def normalize_transfer_config(transfer: Any) -> dict[str, Any]:
    """Return transfer params with deprecated aliases mapped to canonical keys."""
    normalized = to_plain_dict(transfer)

    # Historical code treated this alias as an opt-in OR with the canonical
    # field, so preserve that behavior even when both are present.
    if "allow_direct_inhibitory_input" in normalized:
        normalized["allow_direct_inhibitory_stream"] = bool(
            normalized.get("allow_direct_inhibitory_stream", False)
        ) or bool(normalized["allow_direct_inhibitory_input"])
        normalized.pop("allow_direct_inhibitory_input", None)

    # Historical code gave the canonical field precedence when both were set.
    if "build_inhibitory_population_for_input_mode1" in normalized:
        normalized.setdefault(
            "input_mode1_build_inhibitory_population",
            normalized["build_inhibitory_population_for_input_mode1"],
        )
        normalized.pop("build_inhibitory_population_for_input_mode1", None)

    return normalized


def _is_legacy_unified_ei_layer(layer: dict[str, Any]) -> bool:
    """Return True for the pre-population unified-EI layer shorthand."""
    return any(
        key in layer
        for key in (
            "tau",
            "ff_synapses",
            "rec_synapses",
            "ie_synapses",
            "compartment",
        )
    )


def _legacy_branch_factors(layer: dict[str, Any], *, inhibitory: bool) -> list[int]:
    """Translate legacy recurrent compartment names to a population morphology."""
    key = "inhibitory_branch_factors" if inhibitory else "branch_factors"
    factors = layer.get(key)
    if factors is not None:
        return list(factors)
    return [1]


def _canonicalize_legacy_unified_ei_layer(
    layer: dict[str, Any],
    unified_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Convert old recurrent layer shorthand to explicit population configs."""
    ff_synapses = int(layer.get("ff_synapses", layer.get("ff_excitatory_synapses", 0)))
    rec_synapses = int(
        layer.get("rec_synapses", layer.get("rec_excitatory_synapses", 0))
    )
    ie_synapses = int(layer.get("ie_synapses", layer.get("ff_inhibitory_synapses", 0)))

    if (
        ff_synapses == 0
        and rec_synapses > 0
        and not bool(unified_cfg.get("allow_recurrent_only_layers", False))
    ):
        # A zero-initialized recurrent-only layer has no upstream drive and can
        # make lower layers/input projections inert.  Keep the migration adapter
        # behavior aligned with the canonical recurrent examples.
        ff_synapses = int(unified_cfg.get("legacy_recurrent_only_ff_synapses", 2))

    tau = float(layer.get("tau", unified_cfg.get("tau_base", 50.0)))
    i_tau = float(unified_cfg.get("i_tau", tau))
    learnable_tau = bool(
        unified_cfg.get("learnable_tau", unified_cfg.get("learn_taus", False))
    )
    soma_mode = str(unified_cfg.get("soma_mode", layer.get("soma_mode", ""))).lower()
    use_shunting = bool(layer.get("use_shunting", soma_mode not in {"additive", "sum"}))
    recurrent = bool(
        layer.get(
            "recurrent",
            rec_synapses > 0 or bool(unified_cfg.get("recurrent", False)),
        )
    )

    base_population = {
        "branch_factors": _legacy_branch_factors(layer, inhibitory=False),
        "ff_excitatory_synapses": ff_synapses,
        "ff_inhibitory_synapses": ie_synapses,
        "rec_excitatory_synapses": rec_synapses,
        "rec_inhibitory_synapses": ie_synapses,
        "use_shunting": use_shunting,
        "use_additive_normalization": soma_mode in {"additive", "sum"},
        "reactivate": bool(layer.get("reactivate", True)),
        "reactivation_type": layer.get("reactivation_type", "param_tanh"),
        "weight_transform": str(layer.get("weight_transform", "softplus")),
        "tau_base": tau,
        "tau_ratio": float(layer.get("tau_ratio", 1.0)),
        "learnable_tau": learnable_tau,
        "somatic_synapses": bool(layer.get("somatic_synapses", True)),
    }

    n_inhibitory = int(unified_cfg.get("n_inhibitory", 16) or 0)
    inhibitory = None
    if n_inhibitory > 0 and (ff_synapses > 0 or rec_synapses > 0):
        inhibitory = {
            **base_population,
            "n_neurons": n_inhibitory,
            "branch_factors": _legacy_branch_factors(layer, inhibitory=True),
            "ff_excitatory_synapses": ff_synapses,
            "ff_inhibitory_synapses": 0,
            "rec_excitatory_synapses": rec_synapses,
            "rec_inhibitory_synapses": 0,
            "tau_base": i_tau,
        }

    canonical = {
        key: value
        for key, value in layer.items()
        if key
        not in {
            "tau",
            "tau_ratio",
            "ff_synapses",
            "rec_synapses",
            "ie_synapses",
            "compartment",
            "use_shunting",
            "reactivate",
            "reactivation_type",
            "weight_transform",
            "somatic_synapses",
        }
    }
    canonical.update(
        {
            "recurrent": recurrent,
            "dt": float(layer.get("dt", unified_cfg.get("dt", 1.0))),
            "excitatory": {
                **base_population,
                "n_neurons": int(unified_cfg.get("n_excitatory", 64)),
            },
            "inhibitory": inhibitory,
        }
    )
    return canonical


def canonicalize_unified_ei_config(unified_cfg: Any) -> dict[str, Any]:
    """Normalize old/new unified-EI config spellings before dataclass build."""
    canonical = to_plain_dict(unified_cfg)
    if (
        "input_projection_dims" not in canonical
        and "input_projection_layers" in canonical
    ):
        logger.warning(
            "unified_ei.input_projection_layers is deprecated; use "
            "unified_ei.input_projection_dims instead."
        )
        canonical["input_projection_dims"] = list(canonical["input_projection_layers"])

    if "transfer_params" in canonical:
        canonical["transfer_params"] = normalize_transfer_config(
            canonical["transfer_params"]
        )

    canonical_layers = []
    for layer in canonical.get("layers", []) or []:
        layer_payload = to_plain_dict(layer)
        if layer_payload and _is_legacy_unified_ei_layer(layer_payload):
            logger.warning(
                "Legacy unified_ei.layers[] shorthand is deprecated; use explicit "
                "unified_ei.layers[].excitatory / inhibitory population blocks."
            )
            canonical_layers.append(
                _canonicalize_legacy_unified_ei_layer(layer_payload, canonical)
            )
        else:
            canonical_layers.append(layer)
    canonical["layers"] = canonical_layers

    allowed_keys = {
        "layers",
        "input_dim",
        "use_transfer",
        "transfer_params",
        "input_projection_dims",
        "output_mode",
        "store_routing",
    }
    return {key: value for key, value in canonical.items() if key in allowed_keys}


__all__ = [
    "canonicalize_unified_ei_config",
    "normalize_transfer_config",
]
