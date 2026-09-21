"""Build-option validation for structured recurrent E/I factory configs."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.conversion import layer_value
from dendritic_modeling.config.model_aliases import (
    get_core_morphology_alias_overrides,
    get_structured_recurrent_alias_overrides,
    warn_alias_conflicts,
)
from dendritic_modeling.networks.activations import resolve_dendritic_activation
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_types import (
    _RecurrentReactivationOptions,
    _StructuredRecurrentBuildOptions,
    _StructuredRecurrentSections,
)


def _validate_structured_recurrent_type(core_type: str) -> None:
    if core_type in {
        "dendritic_mlp",
        "flat_shunting",
        "flat_additive",
        "flat_normalized_additive",
    }:
        raise ValueError(
            f"Recurrent mode is not supported for architecture type '{core_type}'. "
            "Use 'EINet', 'dendritic_shunting', 'dendritic_additive', or "
            "'dendritic_normalized_additive'."
        )


def _require_recurrent_excitatory_sizes(
    architecture: dict[str, Any],
) -> list[Any]:
    excitatory_sizes = list(architecture.get("excitatory_layer_sizes", []))
    if not excitatory_sizes:
        raise ValueError(
            "Recurrent mode requires non-empty architecture.excitatory_layer_sizes"
        )
    return excitatory_sizes


def _validate_direct_inhibitory_stream(
    *,
    transfer: dict[str, Any],
    architecture: dict[str, Any],
    connectivity: dict[str, Any],
    excitatory_sizes: list[Any],
) -> tuple[list[Any], int]:
    inhibitory_sizes = list(architecture.get("inhibitory_layer_sizes", []))
    input_mode = int(transfer.get("input_mode", 0))
    has_inhibitory_population = any(int(size or 0) > 0 for size in inhibitory_sizes)
    has_direct_ie_synapses = any(
        int(
            layer_value(
                connectivity.get("ie_synapses_per_branch_per_layer", []),
                layer_idx,
                default=0,
            )
        )
        > 0
        for layer_idx in range(len(excitatory_sizes))
    )
    allow_direct_inhibitory_stream = bool(
        transfer.get("allow_direct_inhibitory_stream", False)
    )
    if (
        input_mode == 1
        and not has_inhibitory_population
        and has_direct_ie_synapses
        and not allow_direct_inhibitory_stream
    ):
        raise ValueError(
            "input_mode=1 with empty inhibitory_layer_sizes and nonzero "
            "ie_synapses_per_branch_per_layer would use the transferred "
            "inhibitory stream directly instead of building inhibitory "
            "neurons. Set architecture.inhibitory_layer_sizes to positive "
            "values for an inhibitory population, or set "
            "transfer.allow_direct_inhibitory_stream=true to opt in to the "
            "direct stream explicitly."
        )
    return inhibitory_sizes, input_mode


def _resolve_recurrent_use_transfer(recurrent_cfg: dict[str, Any]) -> bool:
    # Keep legacy EINet behavior by default: TransferLayer is on unless disabled.
    if "use_transfer" in recurrent_cfg:
        return bool(recurrent_cfg["use_transfer"])
    return True


def _resolve_recurrent_morphology_options(
    *,
    core_type: str,
    morphology: dict[str, Any],
) -> tuple[dict[str, object], bool, bool]:
    core_morphology_overrides = get_core_morphology_alias_overrides(core_type)
    warn_alias_conflicts(
        core_type,
        morphology,
        core_morphology_overrides,
        context="model.core.morphology",
    )
    structured_alias_overrides = get_structured_recurrent_alias_overrides(core_type)

    use_shunting = bool(morphology.get("use_shunting", True))
    if "use_shunting" in core_morphology_overrides:
        use_shunting = bool(core_morphology_overrides["use_shunting"])

    use_additive_normalization = bool(
        morphology.get("use_additive_normalization", False)
    )
    if "use_additive_normalization" in core_morphology_overrides:
        use_additive_normalization = bool(
            core_morphology_overrides["use_additive_normalization"]
        )
    if "use_additive_normalization" in structured_alias_overrides:
        use_additive_normalization = bool(
            structured_alias_overrides["use_additive_normalization"]
        )
    return structured_alias_overrides, use_shunting, use_additive_normalization


def _resolve_recurrent_reactivation_options(
    *,
    core_type: str,
    reactivation: dict[str, Any],
    structured_alias_overrides: dict[str, object],
) -> _RecurrentReactivationOptions:
    reactivation_init_m = float(reactivation.get("init_m", 1.5))
    reactivation_init_b = float(reactivation.get("init_b", 0.5))
    reactivation_init_policy = str(reactivation.get("init_policy", "analytical"))
    reactivate, reactivation_type = resolve_dendritic_activation(
        reactivation.get("dendritic_activation"),
        bool(reactivation.get("enabled", True)),
        reactivation.get("type", "param_tanh"),
    )
    if structured_alias_overrides:
        warn_alias_conflicts(
            core_type,
            reactivation,
            {
                "init_m": structured_alias_overrides.get("reactivation_init_m"),
                "init_b": structured_alias_overrides.get("reactivation_init_b"),
                "init_policy": structured_alias_overrides.get(
                    "reactivation_init_policy"
                ),
            },
            context="model.core.reactivation",
        )
        reactivation_init_m = float(
            structured_alias_overrides.get("reactivation_init_m", reactivation_init_m)
        )
        reactivation_init_b = float(
            structured_alias_overrides.get("reactivation_init_b", reactivation_init_b)
        )
        reactivation_init_policy = str(
            structured_alias_overrides.get(
                "reactivation_init_policy", reactivation_init_policy
            )
        )
    return _RecurrentReactivationOptions(
        reactivate=reactivate,
        reactivation_type=reactivation_type,
        init_m=reactivation_init_m,
        init_b=reactivation_init_b,
        init_policy=reactivation_init_policy,
    )


def _resolve_transfer_inhibitory_mode(
    *,
    transfer: dict[str, Any],
    recurrent_cfg: dict[str, Any],
) -> str:
    transfer_inhibitory_mode = str(
        transfer.get(
            "inhibitory_mode",
            recurrent_cfg.get("transfer_inhibitory_mode", "first"),
        )
    ).lower()
    if transfer_inhibitory_mode not in {"none", "first", "all"}:
        raise ValueError(
            "transfer.inhibitory_mode must be one of ('none', 'first', 'all')"
        )
    return transfer_inhibitory_mode


def _resolve_structured_recurrent_build_options(
    *,
    core_type: str,
    sections: _StructuredRecurrentSections,
) -> _StructuredRecurrentBuildOptions:
    """Resolve validated build options before constructing recurrent layers."""
    recurrent_cfg = sections.recurrent_cfg
    architecture = sections.architecture
    connectivity = sections.connectivity
    transfer = sections.transfer

    _validate_structured_recurrent_type(core_type)
    excitatory_sizes = _require_recurrent_excitatory_sizes(architecture)
    inhibitory_sizes, input_mode = _validate_direct_inhibitory_stream(
        transfer=transfer,
        architecture=architecture,
        connectivity=connectivity,
        excitatory_sizes=excitatory_sizes,
    )
    use_transfer = _resolve_recurrent_use_transfer(recurrent_cfg)
    (
        structured_alias_overrides,
        use_shunting,
        use_additive_normalization,
    ) = _resolve_recurrent_morphology_options(
        core_type=core_type,
        morphology=sections.morphology,
    )
    reactivation_options = _resolve_recurrent_reactivation_options(
        core_type=core_type,
        reactivation=sections.reactivation,
        structured_alias_overrides=structured_alias_overrides,
    )

    # Per-layer recurrence: infer from synapse counts, backward compat with explicit flag.
    explicit_recurrent_layers = recurrent_cfg.get(
        "recurrent_layers", recurrent_cfg.get("recurrent_per_layer", None)
    )

    transfer_inhibitory_mode = _resolve_transfer_inhibitory_mode(
        transfer=transfer,
        recurrent_cfg=recurrent_cfg,
    )

    return _StructuredRecurrentBuildOptions(
        excitatory_sizes=excitatory_sizes,
        inhibitory_sizes=inhibitory_sizes,
        input_mode=input_mode,
        use_transfer=use_transfer,
        transfer_inhibitory_mode=transfer_inhibitory_mode,
        explicit_recurrent_layers=explicit_recurrent_layers,
        use_shunting=use_shunting,
        use_additive_normalization=use_additive_normalization,
        reactivation_options=reactivation_options,
    )


__all__ = [
    "_require_recurrent_excitatory_sizes",
    "_resolve_recurrent_morphology_options",
    "_resolve_recurrent_reactivation_options",
    "_resolve_recurrent_use_transfer",
    "_resolve_structured_recurrent_build_options",
    "_resolve_transfer_inhibitory_mode",
    "_validate_direct_inhibitory_stream",
    "_validate_structured_recurrent_type",
]
