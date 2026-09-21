"""Synapse-type configuration helpers for recurrent dendritic populations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.conversion import to_plain_dict as _to_plain_mapping


@dataclass
class SynapseTypeConfig(BaseConfig):
    """Configuration for one conductance/receptor-like synapse type.

    Dendritic branch voltages in this model are normalized/unitless. Voltage
    gates therefore map the previous branch voltage as
    ``V_mV = voltage_scale_mv * V + mg_v_offset`` before applying Mg-block or
    sigmoid gating constants.
    """

    name: str = "AMPA"
    polarity: str = "excitatory"
    fraction: float = 1.0
    tau: float | None = None
    tau_rise: float | None = None
    tau_decay: float | None = None
    reversal_potential: float | None = None
    nonlinearity: str = "linear"
    conductance_power: float = 1.0
    magnesium_block: bool = False
    voltage_gate: str = "none"
    mg_concentration: float = 1.0
    mg_scale: float = 3.57
    mg_slope: float = 16.13
    mg_v_offset: float = 0.0
    voltage_scale_mv: float = 100.0
    stp_enabled: bool = False
    stp_u0: float = 0.2
    stp_tau_u: float = 200.0
    stp_tau_x: float = 700.0
    stp_facilitation: float = 0.0
    stp_depression: float = 1.0
    stp_differentiable_state: bool = False
    enabled: bool = True

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise ValueError("synapse type name must be non-empty")
        self.polarity = str(self.polarity).lower()
        if self.polarity not in {"excitatory", "inhibitory"}:
            raise ValueError(
                "synapse type polarity must be 'excitatory' or 'inhibitory', "
                f"got {self.polarity!r}"
            )
        self.fraction = float(self.fraction)
        if self.fraction < 0.0:
            raise ValueError(f"synapse type fraction must be >= 0, got {self.fraction}")
        if self.tau is not None:
            self.tau = float(self.tau)
            if self.tau <= 0.0:
                raise ValueError(f"synapse type tau must be > 0, got {self.tau}")
        if self.tau_decay is not None:
            self.tau_decay = float(self.tau_decay)
            if self.tau_decay <= 0.0:
                raise ValueError(
                    f"synapse type tau_decay must be > 0, got {self.tau_decay}"
                )
        if self.tau_rise is not None:
            self.tau_rise = float(self.tau_rise)
            if self.tau_rise <= 0.0:
                raise ValueError(
                    f"synapse type tau_rise must be > 0, got {self.tau_rise}"
                )
        if self.tau_decay is None:
            self.tau_decay = self.tau
        if self.tau_rise is not None and self.tau_decay is not None:
            if self.tau_rise >= self.tau_decay:
                raise ValueError(
                    "tau_rise must be smaller than tau_decay for double-exponential "
                    f"synapses, got tau_rise={self.tau_rise}, "
                    f"tau_decay={self.tau_decay}"
                )
        if self.reversal_potential is None:
            self.reversal_potential = 1.0 if self.polarity == "excitatory" else 0.0
        else:
            self.reversal_potential = float(self.reversal_potential)
        if not -1.0 <= self.reversal_potential <= 2.0:
            raise ValueError(
                "synapse type reversal_potential must be in [-1, 2] for the "
                f"normalized conductance convention, got {self.reversal_potential}"
            )
        self.nonlinearity = str(self.nonlinearity).lower()
        if self.nonlinearity not in {"linear", "relu", "sigmoid", "tanh", "softplus"}:
            raise ValueError(
                "synapse type nonlinearity must be one of 'linear', 'relu', "
                f"'sigmoid', 'tanh', or 'softplus', got {self.nonlinearity!r}"
            )
        self.conductance_power = float(self.conductance_power)
        if self.conductance_power <= 0.0:
            raise ValueError(
                "synapse type conductance_power must be > 0, "
                f"got {self.conductance_power}"
            )
        if self.magnesium_block:
            self.voltage_gate = "nmda_magnesium"
        self.voltage_gate = str(self.voltage_gate).lower()
        if self.voltage_gate not in {"none", "nmda_magnesium", "sigmoid"}:
            raise ValueError(
                "synapse type voltage_gate must be 'none', 'nmda_magnesium', "
                f"or 'sigmoid', got {self.voltage_gate!r}"
            )
        self.magnesium_block = self.voltage_gate == "nmda_magnesium"
        self.mg_concentration = float(self.mg_concentration)
        self.mg_scale = float(self.mg_scale)
        self.mg_slope = float(self.mg_slope)
        self.mg_v_offset = float(self.mg_v_offset)
        self.voltage_scale_mv = float(self.voltage_scale_mv)
        if self.mg_scale <= 0.0:
            raise ValueError(f"mg_scale must be > 0, got {self.mg_scale}")
        if self.mg_slope <= 0.0:
            raise ValueError(f"mg_slope must be > 0, got {self.mg_slope}")
        if self.voltage_scale_mv <= 0.0:
            raise ValueError(
                f"voltage_scale_mv must be > 0, got {self.voltage_scale_mv}"
            )
        self.stp_enabled = bool(self.stp_enabled)
        self.stp_u0 = float(self.stp_u0)
        self.stp_tau_u = float(self.stp_tau_u)
        self.stp_tau_x = float(self.stp_tau_x)
        self.stp_facilitation = float(self.stp_facilitation)
        self.stp_depression = float(self.stp_depression)
        self.stp_differentiable_state = bool(self.stp_differentiable_state)
        if not 0.0 <= self.stp_u0 <= 1.0:
            raise ValueError(f"stp_u0 must be in [0, 1], got {self.stp_u0}")
        if self.stp_tau_u <= 0.0:
            raise ValueError(f"stp_tau_u must be > 0, got {self.stp_tau_u}")
        if self.stp_tau_x <= 0.0:
            raise ValueError(f"stp_tau_x must be > 0, got {self.stp_tau_x}")


@dataclass
class SynapseTypeGroup:
    """Normalized synapse types for one polarity."""

    polarity: str
    types: list[SynapseTypeConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.polarity = str(self.polarity).lower()
        if self.polarity not in {"excitatory", "inhibitory"}:
            raise ValueError(f"unknown synapse polarity {self.polarity!r}")
        enabled = [syn for syn in self.types if bool(syn.enabled) and syn.fraction > 0]
        if not enabled:
            if self.types:
                raise ValueError(
                    f"{self.polarity} synapse types must include at least one "
                    "enabled entry with fraction > 0"
                )
            enabled = [_legacy_synapse_type(self.polarity)]
        total = sum(syn.fraction for syn in enabled)
        if total <= 0.0:
            raise ValueError(f"{self.polarity} synapse fractions must sum to > 0")
        self.types = [
            SynapseTypeConfig(
                name=syn.name,
                polarity=self.polarity,
                fraction=syn.fraction / total,
                tau=syn.tau,
                tau_rise=syn.tau_rise,
                tau_decay=syn.tau_decay,
                reversal_potential=syn.reversal_potential,
                nonlinearity=syn.nonlinearity,
                conductance_power=syn.conductance_power,
                magnesium_block=syn.magnesium_block,
                voltage_gate=syn.voltage_gate,
                mg_concentration=syn.mg_concentration,
                mg_scale=syn.mg_scale,
                mg_slope=syn.mg_slope,
                mg_v_offset=syn.mg_v_offset,
                voltage_scale_mv=syn.voltage_scale_mv,
                stp_enabled=syn.stp_enabled,
                stp_u0=syn.stp_u0,
                stp_tau_u=syn.stp_tau_u,
                stp_tau_x=syn.stp_tau_x,
                stp_facilitation=syn.stp_facilitation,
                stp_depression=syn.stp_depression,
                stp_differentiable_state=syn.stp_differentiable_state,
                enabled=True,
            )
            for syn in enabled
        ]


@dataclass
class SynapseTypeSet:
    """Opt-in synapse-type definitions for a recurrent population."""

    enabled: bool = False
    excitatory: SynapseTypeGroup = field(
        default_factory=lambda: SynapseTypeGroup(
            polarity="excitatory", types=[_legacy_synapse_type("excitatory")]
        )
    )
    inhibitory: SynapseTypeGroup = field(
        default_factory=lambda: SynapseTypeGroup(
            polarity="inhibitory", types=[_legacy_synapse_type("inhibitory")]
        )
    )


def _legacy_synapse_type(polarity: str) -> SynapseTypeConfig:
    name = "AMPA" if polarity == "excitatory" else "GABA_A"
    return SynapseTypeConfig(name=name, polarity=polarity, fraction=1.0, tau=None)


def _as_synapse_type(
    item: SynapseTypeConfig | dict[str, Any],
    *,
    polarity: str,
) -> SynapseTypeConfig:
    if isinstance(item, SynapseTypeConfig):
        payload = item.asdict()
    else:
        payload = _to_plain_mapping(item)
    if "weight" in payload and "fraction" not in payload:
        payload["fraction"] = payload.pop("weight")
    if "tau_d" in payload and "tau_decay" not in payload:
        payload["tau_decay"] = payload.pop("tau_d")
    if "tau_r" in payload and "tau_rise" not in payload:
        payload["tau_rise"] = payload.pop("tau_r")
    if "mg_block" in payload and "magnesium_block" not in payload:
        payload["magnesium_block"] = payload.pop("mg_block")
    if "mg_voltage_scale" in payload and "voltage_scale_mv" not in payload:
        payload["voltage_scale_mv"] = payload.pop("mg_voltage_scale")
    if "mg_voltage_scale_mv" in payload and "voltage_scale_mv" not in payload:
        payload["voltage_scale_mv"] = payload.pop("mg_voltage_scale_mv")
    if "voltage_scale" in payload and "voltage_scale_mv" not in payload:
        payload["voltage_scale_mv"] = payload.pop("voltage_scale")
    stp = payload.pop("short_term_plasticity", None)
    if stp is None:
        stp = payload.pop("stp", None)
    if isinstance(stp, dict):
        payload["stp_enabled"] = bool(stp.get("enabled", True))
        for key, value in stp.items():
            if key == "enabled":
                continue
            payload[f"stp_{key}"] = value
    payload["polarity"] = str(payload.get("polarity", polarity)).lower()
    return SynapseTypeConfig(**payload)


def _resolve_type_list(
    root: dict[str, Any],
    *,
    polarity: str,
) -> list[SynapseTypeConfig]:
    aliases = (
        ("excitatory", "exc", "e")
        if polarity == "excitatory"
        else ("inhibitory", "inh", "i")
    )
    raw_items = None
    for alias in aliases:
        if alias in root:
            raw_items = root[alias]
            break

    if raw_items is None:
        raw_types = root.get("types", None)
        if raw_types is not None:
            raw_items = [
                item
                for item in raw_types
                if str(_to_plain_mapping(item).get("polarity", polarity)).lower()
                == polarity
            ]

    if raw_items is None:
        return [_legacy_synapse_type(polarity)]
    if isinstance(raw_items, dict):
        raw_items = [
            {"name": name, **_to_plain_mapping(config)}
            for name, config in raw_items.items()
        ]
    return [_as_synapse_type(item, polarity=polarity) for item in raw_items]


def _has_configured_type_list(root: dict[str, Any], *, polarity: str) -> bool:
    aliases = (
        ("excitatory", "exc", "e")
        if polarity == "excitatory"
        else ("inhibitory", "inh", "i")
    )
    for alias in aliases:
        if alias not in root:
            continue
        raw_items = root[alias]
        if isinstance(raw_items, dict):
            return bool(raw_items)
        return bool(raw_items)

    raw_types = root.get("types", None)
    if raw_types is None:
        return False
    return any(
        str(_to_plain_mapping(item).get("polarity", polarity)).lower() == polarity
        for item in raw_types
    )


def build_synapse_type_set(config: Any) -> SynapseTypeSet:
    """Build a normalized synapse-type set from a plain config object.

    ``enabled: false`` returns the legacy disabled set. When enabled, both
    excitatory and inhibitory type lists must be explicit so a partial YAML
    payload cannot silently fall back to legacy traces.
    """
    root = _to_plain_mapping(config)
    if not bool(root.get("enabled", False)):
        return SynapseTypeSet(enabled=False)
    missing = [
        polarity
        for polarity in ("excitatory", "inhibitory")
        if not _has_configured_type_list(root, polarity=polarity)
    ]
    if missing:
        raise ValueError(
            "synapse_types.enabled=true requires explicit non-empty type lists "
            f"for both polarities; missing: {', '.join(missing)}"
        )

    return SynapseTypeSet(
        enabled=True,
        excitatory=SynapseTypeGroup(
            polarity="excitatory",
            types=_resolve_type_list(root, polarity="excitatory"),
        ),
        inhibitory=SynapseTypeGroup(
            polarity="inhibitory",
            types=_resolve_type_list(root, polarity="inhibitory"),
        ),
    )


def validate_additive_synapse_reversals(synapse_types: SynapseTypeSet) -> None:
    """Validate normalized reversal potentials for additive synapse dynamics."""
    for synapse_type in synapse_types.excitatory.types:
        if not math.isclose(float(synapse_type.reversal_potential), 1.0):
            raise ValueError(
                "synapse_types with use_shunting=false require normalized "
                "additive reversal potentials: excitatory=1.0 and "
                "inhibitory=0.0. Set use_shunting=true for non-normalized "
                f"reversal_potential on {synapse_type.name!r}."
            )
    for synapse_type in synapse_types.inhibitory.types:
        if not math.isclose(float(synapse_type.reversal_potential), 0.0):
            raise ValueError(
                "synapse_types with use_shunting=false require normalized "
                "additive reversal potentials: excitatory=1.0 and "
                "inhibitory=0.0. Set use_shunting=true for non-normalized "
                f"reversal_potential on {synapse_type.name!r}."
            )


__all__ = [
    "SynapseTypeConfig",
    "SynapseTypeGroup",
    "SynapseTypeSet",
    "build_synapse_type_set",
    "validate_additive_synapse_reversals",
]
