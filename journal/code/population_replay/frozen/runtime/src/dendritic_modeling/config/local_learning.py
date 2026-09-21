"""Config normalization for local learning strategies."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.training import (
    FiveFactorConfig,
    FourFactorConfig,
    HSICConfig,
    InhibitoryHomeostasisConfig,
    LocalGateHomeostasisConfig,
    LocalRuleConfig,
    LocalVoltageHomeostasisConfig,
    MorphologyAwareConfig,
    STDPConfig,
    ThreeFactorConfig,
)


def section_to_dict(section: Any) -> dict[str, Any]:
    """Convert a config section object to a plain dict."""

    if section is None:
        return {}
    if isinstance(section, dict):
        return dict(section)
    if hasattr(section, "asdict") and callable(section.asdict):
        return section.asdict()
    if hasattr(section, "__dict__"):
        return {
            key: value
            for key, value in vars(section).items()
            if not key.startswith("_")
        }
    return {}


def coerce_legacy_local_rule_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy flat local learning config to nested LocalRuleConfig layout."""

    cfg = dict(config_dict)
    alias_mode = cfg.pop("inhibitory_cell_update_mode", None)
    if alias_mode is not None:
        existing_mode = cfg.get("explicit_inhibitory_update_mode")
        if (
            existing_mode is not None
            and LocalRuleConfig._normalize_explicit_inhibitory_update_mode(
                existing_mode
            )
            != LocalRuleConfig._normalize_explicit_inhibitory_update_mode(alias_mode)
        ):
            raise ValueError(
                "Conflicting inhibitory-cell update aliases in LocalRuleConfig"
            )
        cfg["explicit_inhibitory_update_mode"] = alias_mode

    def ensure_section(section_name: str) -> dict[str, Any]:
        section = section_to_dict(cfg.get(section_name))
        cfg[section_name] = section
        return section

    legacy_group_fields = {
        "three_factor": [
            "dynamics_mode",
            "use_conductance_scaling",
            "use_driving_force",
            "theta",
            "e_rev_exc",
            "e_rev_inh",
            "additive_gain_mode",
            "additive_stats_ema_alpha",
        ],
        "four_factor": [
            "rho_mode",
            "rho_estimator",
            "ema_alpha",
            "layer_wise_rho_scale",
            "augment_k",
            "augment_noise_sigma",
        ],
        "five_factor": [
            "phi_mode",
            "phi_estimator",
            "phi_ridge_lambda",
            "layer_wise_phi_scale",
            "rls_forgetting",
        ],
        "morphology_aware": [
            "use_path_propagation",
            "path_factor_mode",
            "morphology_modulator_mode",
            "morphology_depth_offset",
            "morphology_centrality_metric",
            "use_dendritic_normalization",
            "use_branch_type_rules",
            "apical_branch_scale",
            "basal_branch_scale",
            "use_branch_length_modulation",
            "use_branch_role_rules",
            "branch_role_source",
            "specialized_branch_scale",
            "mixed_branch_scale",
            "branch_role_power",
            "branch_role_alignment_weight",
        ],
        "inhibitory_homeostasis": [
            "mode",
            "weight",
            "target_r_tot",
            "target_voltage",
        ],
        "voltage_homeostasis": [
            "weight",
            "target_voltage",
        ],
    }

    for section_name, field_names in legacy_group_fields.items():
        section = ensure_section(section_name)
        for field_name in field_names:
            if field_name in cfg and field_name not in section:
                section[field_name] = cfg.pop(field_name)

    hsic_section = ensure_section("hsic")
    hsic_field_map = {
        "hsic_enabled": "enabled",
        "hsic_weight": "weight",
        "hsic_self_weight": "self_weight",
        "hsic_target_weight": "target_weight",
        "hsic_target_source": "target_source",
        "hsic_kernel": "kernel",
        "hsic_sigma": "sigma",
        "hsic_degree": "degree",
        "hsic_coef0": "coef0",
        "hsic_grad_clip_value": "grad_clip_value",
        "hsic_warmup_epochs": "warmup_epochs",
        "hsic_apply_last_layer_only": "apply_last_layer_only",
    }
    for legacy_name, nested_name in hsic_field_map.items():
        if legacy_name in cfg and nested_name not in hsic_section:
            hsic_section[nested_name] = cfg.pop(legacy_name)

    stdp_section = ensure_section("stdp")
    stdp_field_map = {
        "stdp_enabled": "enabled",
        "stdp_apply_to": "apply_to",
        "stdp_activity_mode": "activity_mode",
        "stdp_pre_threshold": "pre_threshold",
        "stdp_post_threshold": "post_threshold",
        "stdp_tau_pre": "tau_pre",
        "stdp_tau_post": "tau_post",
        "stdp_a_plus": "a_plus",
        "stdp_a_minus": "a_minus",
        "stdp_learning_rate_scale": "learning_rate_scale",
        "stdp_inhibitory_update_sign": "inhibitory_update_sign",
        "stdp_use_error_modulation": "use_error_modulation",
        "stdp_error_modulation_mode": "error_modulation_mode",
        "stdp_clamp_update": "clamp_update",
        "stdp_detach_traces": "detach_traces",
    }
    for legacy_name, nested_name in stdp_field_map.items():
        if legacy_name in cfg and nested_name not in stdp_section:
            stdp_section[nested_name] = cfg.pop(legacy_name)

    # Drop known legacy-only keys that are not part of LocalRuleConfig.
    for legacy_only_key in [
        "rho_value",
        "weight_lr_multiplier",
        "compute_gradient_stats",
    ]:
        cfg.pop(legacy_only_key, None)

    return cfg


def build_local_rule_config(config_dict: dict[str, Any]) -> LocalRuleConfig:
    """Build LocalRuleConfig from dict with proper nested config handling."""

    cfg_kwargs = coerce_legacy_local_rule_config(dict(config_dict))

    # Convenience aliases: allow readable shorthand like "5f_vh" while
    # keeping the actual implementation as ordinary LocalCA + a strictly
    # local voltage-homeostasis modifier.
    raw_rule_variant = str(cfg_kwargs.get("rule_variant", "3f")).lower()
    vh_alias_map = {
        "3f_vh": "3f",
        "3f+vh": "3f",
        "3f-vh": "3f",
        "4f_vh": "4f",
        "4f+vh": "4f",
        "4f-vh": "4f",
        "5f_vh": "5f",
        "5f+vh": "5f",
        "5f-vh": "5f",
    }
    if raw_rule_variant in vh_alias_map:
        cfg_kwargs["rule_variant"] = vh_alias_map[raw_rule_variant]
        vh_cfg = cfg_kwargs.get("voltage_homeostasis")
        if isinstance(vh_cfg, dict):
            vh_cfg = dict(vh_cfg)
            vh_cfg.setdefault("enabled", True)
            vh_cfg.setdefault("weight", 0.02)
            vh_cfg.setdefault("target_voltage", 0.5)
            cfg_kwargs["voltage_homeostasis"] = vh_cfg
        elif vh_cfg is None:
            cfg_kwargs["voltage_homeostasis"] = {
                "enabled": True,
                "weight": 0.02,
                "target_voltage": 0.5,
            }

    if "three_factor" in cfg_kwargs and isinstance(cfg_kwargs["three_factor"], dict):
        cfg_kwargs["three_factor"] = ThreeFactorConfig(**cfg_kwargs["three_factor"])

    if "four_factor" in cfg_kwargs and isinstance(cfg_kwargs["four_factor"], dict):
        cfg_kwargs["four_factor"] = FourFactorConfig(**cfg_kwargs["four_factor"])

    if "five_factor" in cfg_kwargs and isinstance(cfg_kwargs["five_factor"], dict):
        cfg_kwargs["five_factor"] = FiveFactorConfig(**cfg_kwargs["five_factor"])

    if "morphology_aware" in cfg_kwargs and isinstance(
        cfg_kwargs["morphology_aware"], dict
    ):
        cfg_kwargs["morphology_aware"] = MorphologyAwareConfig(
            **cfg_kwargs["morphology_aware"]
        )

    if "hsic" in cfg_kwargs and isinstance(cfg_kwargs["hsic"], dict):
        cfg_kwargs["hsic"] = HSICConfig(**cfg_kwargs["hsic"])

    if "stdp" in cfg_kwargs and isinstance(cfg_kwargs["stdp"], dict):
        cfg_kwargs["stdp"] = STDPConfig(**cfg_kwargs["stdp"])

    if "inhibitory_homeostasis" in cfg_kwargs and isinstance(
        cfg_kwargs["inhibitory_homeostasis"], dict
    ):
        cfg_kwargs["inhibitory_homeostasis"] = InhibitoryHomeostasisConfig(
            **cfg_kwargs["inhibitory_homeostasis"]
        )

    if "voltage_homeostasis" in cfg_kwargs and isinstance(
        cfg_kwargs["voltage_homeostasis"], dict
    ):
        cfg_kwargs["voltage_homeostasis"] = LocalVoltageHomeostasisConfig(
            **cfg_kwargs["voltage_homeostasis"]
        )

    if "gate_homeostasis" in cfg_kwargs and isinstance(
        cfg_kwargs["gate_homeostasis"], dict
    ):
        cfg_kwargs["gate_homeostasis"] = LocalGateHomeostasisConfig(
            **cfg_kwargs["gate_homeostasis"]
        )

    return LocalRuleConfig(**cfg_kwargs)


__all__ = [
    "build_local_rule_config",
    "coerce_legacy_local_rule_config",
    "section_to_dict",
]
