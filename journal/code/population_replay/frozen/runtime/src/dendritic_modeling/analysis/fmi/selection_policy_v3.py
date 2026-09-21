"""Risk-aware FMI-v3 selection with a calibrated morphology promotion gate.

FMI-v2 maps teacher-side statistics directly to an executable morphology.  The
Qwen2.5-72B development grid showed that this is too optimistic: most sites
were nearly tied, but projected branching caused two large tail failures.  This
module therefore separates three decisions that must not be conflated:

* teacher-side FMI proposes capacity axes such as width and pathway density;
* an explicit biological/manual policy fixes non-inferable constraints;
* a matched local calibration may promote gating, E/I, integration, or
  branching away from a flat incumbent only with preregistered confidence.

The v1/v2 Qwen sites are development evidence for this policy.  They are not a
prospective validation set for FMI-v3.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from dendritic_modeling.analysis.fmi.shortlist_safety import LocalFitEvidence
from dendritic_modeling.networks.architectures.replacement.compiler import (
    CANONICAL_FAMILIES,
)
from dendritic_modeling.networks.architectures.replacement.selection import (
    ALL_SELECTION_AXES,
    FMI_SELECTABLE_AXES,
    MANUAL_ONLY_AXES,
    resolve_replacement_selection,
)

V3_ADVISORY_SCHEMA = "dendritic_fmi_selection_advisory/v3"
V3_PROMOTION_POLICY_SCHEMA = "dendritic_fmi_promotion_policy/v3"
V3_PROMOTION_DECISION_SCHEMA = "dendritic_fmi_promotion_decision/v3"

AXIS_SOURCES = frozenset({"fmi", "manual", "local_calibration"})
LOCAL_CALIBRATION_AXES = frozenset(
    {
        "explicit_ei",
        "branch_factors",
        "excitatory_branch_factors",
        "inhibitory_branch_factors",
        "somatic_synapses",
        "somatic_excitatory_synapses",
        "somatic_inhibitory_synapses",
        "inhibitory_population_somatic_synapses",
        "integration_rule",
        "gated",
    }
)
CAPACITY_AXES = (
    "density",
    "input_to_excitatory_density",
    "input_to_inhibitory_density",
    "inhibitory_to_excitatory_density",
    "output_density",
    "population_width",
)
STRUCTURAL_FIELDS = (
    "biological_neuron",
    "explicit_ei",
    "gated",
    "morphology_class",
    "integration_rule",
)
REQUIRED_RESOURCE_BUDGET_AXES = frozenset(
    {"active_weight_macs", "serialized_bytes", "stored_parameters"}
)
REQUIRED_COMMON_PROVENANCE = frozenset(
    {
        "teacher_artifact_sha256",
        "data_manifest_sha256",
        "target_manifest_sha256",
        "fit_protocol_sha256",
        "gate_protocol_sha256",
    }
)


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _sha256(value: object, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
        raise ValueError(f"{label} must be a full lowercase SHA-256 digest")
    return normalized


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    normalized = value.strip()
    if any(character in normalized for character in "\r\n\t"):
        raise ValueError(f"{label} cannot contain control whitespace")
    return normalized


def _plain(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    try:
        from omegaconf import OmegaConf

        converted = OmegaConf.to_container(value, resolve=True)
        return dict(converted) if isinstance(converted, Mapping) else {}
    except (ImportError, TypeError, ValueError):
        return {}


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _normalize_axis_policy(
    *,
    dimension_sources: Mapping[str, object],
    manual_values: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Require an explicit source for every executable selection axis."""

    sources = {
        str(axis): str(source).strip().lower()
        for axis, source in dimension_sources.items()
    }
    missing = sorted(set(ALL_SELECTION_AXES) - set(sources))
    extra = sorted(set(sources) - set(ALL_SELECTION_AXES))
    if missing or extra:
        raise ValueError(
            "FMI-v3 dimension_sources must cover every selection axis exactly; "
            f"missing={missing}, extra={extra}"
        )
    invalid = {
        axis: source for axis, source in sources.items() if source not in AXIS_SOURCES
    }
    if invalid:
        raise ValueError(f"invalid FMI-v3 dimension sources: {invalid}")
    illegal_fmi = sorted(axis for axis in MANUAL_ONLY_AXES if sources[axis] != "manual")
    if illegal_fmi:
        raise ValueError(
            f"runtime/experimental axes are manual-only in FMI-v3: {illegal_fmi}"
        )
    illegal_calibration = sorted(
        axis
        for axis, source in sources.items()
        if source == "local_calibration" and axis not in LOCAL_CALIBRATION_AXES
    )
    if illegal_calibration:
        raise ValueError(
            "only structural morphology axes may use local_calibration: "
            f"{illegal_calibration}"
        )
    if sources["biological_neuron"] != "manual":
        raise ValueError(
            "biological_neuron must be an explicit manual policy in FMI-v3"
        )

    manual = dict(manual_values)
    unknown_manual = sorted(set(manual) - set(ALL_SELECTION_AXES))
    if unknown_manual:
        raise ValueError(f"unknown FMI-v3 manual axes: {unknown_manual}")
    missing_manual = sorted(
        axis
        for axis, source in sources.items()
        if source == "manual" and axis not in manual
    )
    if missing_manual:
        raise ValueError(
            "every manual FMI-v3 axis must carry an explicit value; "
            f"missing={missing_manual}"
        )
    stray_manual = sorted(axis for axis in manual if sources.get(axis) != "manual")
    if stray_manual:
        raise ValueError(
            "manual_values cannot silently override FMI/calibrated axes; "
            f"stray={stray_manual}"
        )
    return dict(sorted(sources.items())), manual


def _risk_capacity_audit(
    *,
    proposed_densities: Mapping[str, float],
    proposed_width: int,
    teacher_intermediate_size: int,
    layer_index: int,
    layer_count: int,
    deletion_ratio: float,
    risk_config: Mapping[str, object],
) -> dict[str, Any]:
    if layer_count < 2 or not 0 <= layer_index < layer_count:
        raise ValueError("layer_index must lie in a model with at least two layers")
    deletion = _finite_float(deletion_ratio, label="deletion_ratio")
    if deletion <= 0:
        raise ValueError("deletion_ratio must be positive")
    cfg = dict(risk_config)
    required = {
        "base_density_floor",
        "depth_density_gain",
        "deletion_density_gain",
        "maximum_density_floor",
        "base_width_fraction_floor",
        "depth_width_fraction_gain",
        "deletion_width_fraction_gain",
        "maximum_width_fraction_floor",
        "depth_power",
    }
    missing = sorted(required - set(cfg))
    unknown = sorted(set(cfg) - required)
    if missing or unknown:
        raise ValueError(
            f"risk_capacity must be fully frozen; missing={missing}, unknown={unknown}"
        )
    values = {
        key: _finite_float(cfg[key], label=f"risk_capacity.{key}") for key in required
    }
    if any(values[key] < 0 for key in required):
        raise ValueError("risk-capacity coefficients must be nonnegative")
    if values["maximum_density_floor"] > 1.0:
        raise ValueError("maximum_density_floor cannot exceed one")
    if values["maximum_width_fraction_floor"] > 1.0:
        raise ValueError("maximum_width_fraction_floor cannot exceed one")
    if values["depth_power"] <= 0:
        raise ValueError("depth_power must be positive")

    depth_fraction = float(layer_index) / float(layer_count - 1)
    deletion_excess = max(0.0, deletion - 1.0)
    density_floor_unclamped = (
        values["base_density_floor"]
        + values["depth_density_gain"] * depth_fraction ** values["depth_power"]
        + values["deletion_density_gain"] * deletion_excess
    )
    density_floor = min(values["maximum_density_floor"], density_floor_unclamped)
    width_fraction_unclamped = (
        values["base_width_fraction_floor"]
        + values["depth_width_fraction_gain"] * depth_fraction ** values["depth_power"]
        + values["deletion_width_fraction_gain"] * deletion_excess
    )
    width_fraction_floor = min(
        values["maximum_width_fraction_floor"], width_fraction_unclamped
    )
    width_floor = max(1, math.ceil(width_fraction_floor * teacher_intermediate_size))
    density_proposals = {
        axis: _finite_float(value, label=f"proposed_densities.{axis}")
        for axis, value in proposed_densities.items()
    }
    expected_density_axes = set(CAPACITY_AXES) - {"population_width"}
    if set(density_proposals) != expected_density_axes:
        raise ValueError(
            "proposed_densities must cover every density capacity axis exactly"
        )
    if any(not 0 < value <= 1 for value in density_proposals.values()):
        raise ValueError("proposed densities must lie in (0, 1]")
    realized_densities = {
        axis: max(value, density_floor) for axis, value in density_proposals.items()
    }
    return {
        "schema": "dendritic_fmi_risk_capacity/v1",
        "inputs": {
            "layer_index": int(layer_index),
            "layer_count": int(layer_count),
            "depth_fraction": depth_fraction,
            "deletion_ratio": deletion,
            "deletion_excess": deletion_excess,
            "teacher_intermediate_size": int(teacher_intermediate_size),
        },
        "frozen_coefficients": dict(sorted(values.items())),
        "proposed": {
            "densities": density_proposals,
            "population_width": int(proposed_width),
        },
        "floors": {
            "density_unclamped": density_floor_unclamped,
            "density": density_floor,
            "width_fraction_unclamped": width_fraction_unclamped,
            "width_fraction": width_fraction_floor,
            "population_width": width_floor,
        },
        "realized": {
            "densities": realized_densities,
            "population_width": max(int(proposed_width), width_floor),
        },
        "interpretation": (
            "capacity safety floor from depth and teacher-only deletion sensitivity; "
            "not evidence that the realized capacity is optimal"
        ),
    }


def _incumbent_values(*, biological: bool) -> dict[str, Any]:
    return {
        "biological_neuron": bool(biological),
        "explicit_ei": bool(biological),
        "branch_factors": [],
        "excitatory_branch_factors": [],
        "inhibitory_branch_factors": [],
        "somatic_synapses": True,
        "somatic_excitatory_synapses": None,
        "somatic_inhibitory_synapses": None,
        "inhibitory_population_somatic_synapses": None,
        "integration_rule": "raw_additive",
        "gated": False,
    }


def _candidate_metadata(*, biological: bool) -> list[dict[str, Any]]:
    if biological:
        rows = (
            ("gated_positive_ei_flat", False, True, True, "raw_additive"),
            ("positive_ei_branched_additive", True, False, True, "raw_additive"),
            (
                "gated_positive_ei_branched_additive",
                True,
                True,
                True,
                "raw_additive",
            ),
            ("positive_ei_branched_shunting", True, False, True, "shunting"),
            ("gated_positive_ei_branched_shunting", True, True, True, "shunting"),
        )
    else:
        rows = (
            ("signed_ei_flat", False, False, True, "raw_additive"),
            ("gated_signed_flat", False, True, False, "raw_additive"),
            ("gated_signed_ei_flat", False, True, True, "raw_additive"),
            ("signed_branched", True, False, False, "raw_additive"),
            ("signed_ei_branched_additive", True, False, True, "raw_additive"),
            ("gated_signed_branched", True, True, False, "raw_additive"),
            (
                "gated_signed_ei_branched_additive",
                True,
                True,
                True,
                "raw_additive",
            ),
        )
    result = []
    for family, branched, gated, explicit_ei, integration in rows:
        promoted = []
        if branched:
            promoted.append("branching")
        if gated:
            promoted.append("gating")
        if explicit_ei != biological:
            promoted.append("explicit_ei")
        if integration != "raw_additive":
            promoted.append("integration_rule")
        result.append(
            {
                "candidate_id": family,
                "family": family,
                "biological_neuron": biological,
                "explicit_ei": explicit_ei,
                "gated": gated,
                "morphology_class": "branched" if branched else "flat",
                "integration_rule": integration,
                "promoted_dimensions": promoted,
                "status": "requires_matched_local_calibration",
            }
        )
    return result


def build_fmi_v3_advisory(
    *,
    selection_config: Mapping[str, object],
    hidden_size: int,
    teacher_intermediate_size: int,
    layer_index: int,
    layer_count: int,
    deletion_ratio: float,
) -> dict[str, Any]:
    """Resolve teacher capacity but execute a flat incumbent until calibration.

    ``selection_config`` must declare ``dimension_sources`` for every current
    selection axis.  Structural axes may be marked ``local_calibration``;
    teacher-only FMI never promotes those axes in the returned executable plan.
    """

    cfg = _plain(selection_config)
    if str(cfg.get("policy", "conservative_local_calibration_v3")) != (
        "conservative_local_calibration_v3"
    ):
        raise ValueError("unsupported FMI-v3 policy")
    sources, manual = _normalize_axis_policy(
        dimension_sources=_plain(cfg.get("dimension_sources")),
        manual_values=_plain(cfg.get("manual_values")),
    )
    allow_non_biological = bool(cfg.get("allow_non_biological", False))
    domain = str(cfg.get("constraint_domain", "biological")).strip().lower()
    if domain not in {"biological", "relaxed"}:
        raise ValueError("constraint_domain must be 'biological' or 'relaxed'")
    if domain == "relaxed" and not allow_non_biological:
        raise ValueError("constraint_domain=relaxed requires allow_non_biological=true")
    biological = domain == "biological"
    if bool(manual["biological_neuron"]) != biological:
        raise ValueError(
            "manual biological_neuron must agree with the declared constraint_domain"
        )

    base = {
        "enabled": True,
        "mode": "hybrid",
        "rule_set": str(cfg.get("rule_set", "theory_v4")),
        "allow_non_biological": allow_non_biological,
        "fingerprint": deepcopy(cfg.get("fingerprint", {})),
        "thresholds": deepcopy(cfg.get("thresholds", {})),
        "compiler": deepcopy(cfg.get("compiler", {})),
    }

    # The teacher advisory exposes what the old direct-projection policy would
    # have requested.  It is retained for audit and candidate generation only.
    advisory_axes: dict[str, str] = {}
    advisory_manual = dict(manual)
    for axis, source in sources.items():
        if source == "local_calibration":
            advisory_axes[axis] = "fmi" if axis in FMI_SELECTABLE_AXES else "manual"
        else:
            advisory_axes[axis] = source
    advisory_manual.setdefault("biological_neuron", biological)
    teacher_selection = resolve_replacement_selection(
        {**base, "axes": advisory_axes, "manual": advisory_manual},
        hidden_size=int(hidden_size),
        teacher_intermediate_size=int(teacher_intermediate_size),
        layer_index=int(layer_index),
    )

    proposed_densities = {
        "density": float(teacher_selection.plan.density),
        "input_to_excitatory_density": float(
            teacher_selection.plan.input_to_excitatory_density
        ),
        "input_to_inhibitory_density": float(
            teacher_selection.plan.input_to_inhibitory_density
        ),
        "inhibitory_to_excitatory_density": float(
            teacher_selection.plan.inhibitory_to_excitatory_density
        ),
        "output_density": float(teacher_selection.plan.output_density),
    }
    risk = _risk_capacity_audit(
        proposed_densities=proposed_densities,
        proposed_width=int(teacher_selection.plan.population_width),
        teacher_intermediate_size=int(teacher_intermediate_size),
        layer_index=int(layer_index),
        layer_count=int(layer_count),
        deletion_ratio=float(deletion_ratio),
        risk_config=_plain(cfg.get("risk_capacity")),
    )
    realized_densities = dict(risk["realized"]["densities"])
    realized_width = int(risk["realized"]["population_width"])

    incumbent_axes: dict[str, str] = {}
    incumbent_manual = dict(manual)
    local_values = _incumbent_values(biological=biological)
    for axis, source in sources.items():
        if source == "local_calibration":
            incumbent_axes[axis] = "manual"
            incumbent_manual[axis] = local_values[axis]
        else:
            incumbent_axes[axis] = source
    for axis in CAPACITY_AXES:
        incumbent_axes[axis] = "manual"
        incumbent_manual[axis] = (
            realized_width if axis == "population_width" else realized_densities[axis]
        )

    incumbent_selection = resolve_replacement_selection(
        {**base, "axes": incumbent_axes, "manual": incumbent_manual},
        hidden_size=int(hidden_size),
        teacher_intermediate_size=int(teacher_intermediate_size),
        layer_index=int(layer_index),
    )
    expected_family = "positive_ei_flat" if biological else "signed_flat"
    if incumbent_selection.plan.family != expected_family:
        raise RuntimeError(
            "FMI-v3 incumbent failed to compile to the declared flat family: "
            f"{incumbent_selection.plan.family!r} != {expected_family!r}"
        )

    candidates = _candidate_metadata(biological=biological)
    payload = {
        "schema": V3_ADVISORY_SCHEMA,
        "policy": "conservative_local_calibration_v3",
        "status": "flat_incumbent_until_matched_local_calibration_passes",
        "claim_boundary": (
            "Teacher-side FMI supplies a capacity prior and candidate shortlist; "
            "it does not establish morphology optimality. Structural promotion "
            "requires separately registered matched local evidence."
        ),
        "allow_non_biological": allow_non_biological,
        "constraint_domain": domain,
        "dimension_sources": sources,
        "manual_values": manual,
        "site": {
            "layer_index": int(layer_index),
            "layer_count": int(layer_count),
            "deletion_ratio": float(deletion_ratio),
        },
        "teacher_advisory": {
            "plan": teacher_selection.plan.as_dict(),
            "selection_manifest": teacher_selection.manifest,
            "plan_sha256": _canonical_sha256(teacher_selection.plan.as_dict()),
            "execution_status": "not_promoted_by_teacher_statistics_alone",
        },
        "risk_capacity": risk,
        "incumbent": {
            "candidate_id": expected_family,
            "plan": incumbent_selection.plan.as_dict(),
            "selection_manifest": incumbent_selection.manifest,
            "plan_sha256": _canonical_sha256(incumbent_selection.plan.as_dict()),
            "execution_status": "safe_default_executable_before_calibration",
        },
        "calibration_candidates": candidates,
        "development_evidence_boundary": {
            "qwen72b_v1_v2_sites": "development_only",
            "prospective_v3_sites": "must_be_disjoint_and_frozen_before_training",
        },
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


def _normalize_candidate_metadata(
    values: Mapping[str, Mapping[str, object]],
    *,
    incumbent_candidate_id: str,
    biological_domain: bool,
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("candidate_metadata must be a non-empty mapping")
    normalized: dict[str, dict[str, Any]] = {}
    allowed = set(STRUCTURAL_FIELDS) | {"family"}
    for raw_id, raw in values.items():
        candidate_id = _identifier(str(raw_id), label="candidate_id")
        if not isinstance(raw, Mapping):
            raise TypeError(
                f"candidate metadata for {candidate_id!r} must be a mapping"
            )
        missing = sorted(allowed - set(raw))
        extra = sorted(set(raw) - allowed)
        if missing or extra:
            raise ValueError(
                f"candidate metadata for {candidate_id!r} is incomplete; "
                f"missing={missing}, extra={extra}"
            )
        row = {
            "family": _identifier(raw["family"], label="family"),
            "biological_neuron": bool(raw["biological_neuron"]),
            "explicit_ei": bool(raw["explicit_ei"]),
            "gated": bool(raw["gated"]),
            "morphology_class": str(raw["morphology_class"]).strip().lower(),
            "integration_rule": str(raw["integration_rule"]).strip().lower(),
        }
        if row["morphology_class"] not in {"flat", "branched"}:
            raise ValueError("morphology_class must be flat or branched")
        if row["integration_rule"] not in {
            "raw_additive",
            "conductance_normalized",
            "tangent_matched",
            "shunting",
        }:
            raise ValueError("unsupported integration_rule in candidate metadata")
        if row["family"] not in CANONICAL_FAMILIES:
            raise ValueError(f"candidate family {row['family']!r} is not canonical")
        expected_structure = {
            "biological_neuron": "positive_ei" in row["family"],
            "explicit_ei": "_ei_" in row["family"],
            "gated": row["family"].startswith("gated_"),
            "morphology_class": ("branched" if "branched" in row["family"] else "flat"),
            "integration_rule": (
                "shunting" if row["family"].endswith("_shunting") else "raw_additive"
            ),
        }
        contradictions = {
            key: {"declared": row[key], "family_implies": value}
            for key, value in expected_structure.items()
            if row[key] != value
        }
        if contradictions:
            raise ValueError(
                f"candidate metadata contradicts family {row['family']!r}: "
                f"{contradictions}"
            )
        if biological_domain and not row["biological_neuron"]:
            raise ValueError(
                "a biological-domain promotion policy cannot contain relaxed candidates"
            )
        normalized[candidate_id] = row
    if incumbent_candidate_id not in normalized:
        raise ValueError("candidate_metadata must include the incumbent")
    incumbent = normalized[incumbent_candidate_id]
    if incumbent["morphology_class"] != "flat" or incumbent["gated"]:
        raise ValueError("the FMI-v3 incumbent must be flat and ungated")
    for candidate_id, row in normalized.items():
        if candidate_id == incumbent_candidate_id:
            continue
        changed = [
            field for field in STRUCTURAL_FIELDS if row[field] != incumbent[field]
        ]
        if not changed:
            raise ValueError(
                f"promotion candidate {candidate_id!r} changes no structural dimension"
            )
    return dict(sorted(normalized.items()))


def _normalize_hash_map(
    values: Mapping[str, object], *, label: str, expected: set[str]
) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping")
    normalized = {
        _identifier(str(key), label=f"{label} key"): _sha256(value, label=label)
        for key, value in values.items()
    }
    if set(normalized) != expected:
        raise ValueError(
            f"{label} must cover candidates exactly; "
            f"missing={sorted(expected - set(normalized))}, "
            f"extra={sorted(set(normalized) - expected)}"
        )
    return dict(sorted(normalized.items()))


def _normalize_resource_budget(values: Mapping[str, object]) -> dict[str, int]:
    if not isinstance(values, Mapping):
        raise TypeError("resource_budget must be a mapping")
    missing = sorted(REQUIRED_RESOURCE_BUDGET_AXES - set(values))
    if missing:
        raise ValueError(f"resource_budget is missing matched axes: {missing}")
    result: dict[str, int] = {}
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"resource_budget[{key!r}] must be a positive integer")
        result[str(key)] = int(value)
    return dict(sorted(result.items()))


def _normalize_group_hashes(
    values: Mapping[str, object], *, label: str
) -> dict[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} must be a non-empty mapping")
    result = {
        _identifier(str(key), label=f"{label} key"): _sha256(value, label=label)
        for key, value in values.items()
    }
    if len(set(result.values())) != len(result):
        raise ValueError(f"{label} cannot alias content under multiple group IDs")
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class MorphologyPromotionPolicyV3:
    """Content-addressed, prospective matched-calibration policy."""

    schema: str
    policy_id: str
    advisory_sha256: str
    incumbent_candidate_id: str
    biological_domain: bool
    candidate_metadata: dict[str, dict[str, Any]]
    candidate_artifact_sha256: dict[str, str]
    candidate_plan_sha256: dict[str, str]
    resource_budget: dict[str, int]
    fit_groups: dict[str, str]
    gate_groups: dict[str, str]
    target_sha256: dict[str, str]
    common_provenance: dict[str, str]
    relative_improvement_margin: float
    confidence_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    minimum_gate_groups: int
    multiple_comparison_correction: str
    selection_tie_break: str
    require_no_target_mean_harm: bool
    registration_record_sha256: str
    registered_before_gate_evidence: bool
    content_sha256: str

    def content_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("content_sha256")
        return payload

    def validate_content_identity(self) -> None:
        if self.schema != V3_PROMOTION_POLICY_SCHEMA:
            raise ValueError("unsupported FMI-v3 promotion policy schema")
        if _canonical_sha256(self.content_payload()) != self.content_sha256:
            raise ValueError("FMI-v3 promotion policy content hash mismatch")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_morphology_promotion_policy_v3(
    *,
    policy_id: str,
    advisory_sha256: str,
    incumbent_candidate_id: str,
    biological_domain: bool,
    candidate_metadata: Mapping[str, Mapping[str, object]],
    candidate_artifact_sha256: Mapping[str, object],
    candidate_plan_sha256: Mapping[str, object],
    resource_budget: Mapping[str, object],
    fit_groups: Mapping[str, object],
    gate_groups: Mapping[str, object],
    target_sha256: Mapping[str, object],
    common_provenance: Mapping[str, object],
    relative_improvement_margin: float,
    confidence_level: float,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    minimum_gate_groups: int,
    registration_record_sha256: str,
    registered_before_gate_evidence: bool,
    require_no_target_mean_harm: bool = True,
) -> MorphologyPromotionPolicyV3:
    """Freeze the margin and uncertainty rule before local gate evidence."""

    if registered_before_gate_evidence is not True:
        raise ValueError("the promotion policy must precede gate evidence")
    incumbent_id = _identifier(incumbent_candidate_id, label="incumbent_candidate_id")
    metadata = _normalize_candidate_metadata(
        candidate_metadata,
        incumbent_candidate_id=incumbent_id,
        biological_domain=bool(biological_domain),
    )
    candidates = set(metadata)
    artifacts = _normalize_hash_map(
        candidate_artifact_sha256,
        label="candidate_artifact_sha256",
        expected=candidates,
    )
    plans = _normalize_hash_map(
        candidate_plan_sha256,
        label="candidate_plan_sha256",
        expected=candidates,
    )
    if len(set(artifacts.values())) != len(artifacts):
        raise ValueError("candidate artifacts must be content-distinct")
    budget = _normalize_resource_budget(resource_budget)
    fit = _normalize_group_hashes(fit_groups, label="fit_groups")
    gate = _normalize_group_hashes(gate_groups, label="gate_groups")
    if set(fit).intersection(gate) or set(fit.values()).intersection(gate.values()):
        raise ValueError("fit and gate groups must be content-disjoint")
    minimum = _positive_int(minimum_gate_groups, label="minimum_gate_groups")
    if len(gate) < minimum:
        raise ValueError("gate_groups do not meet minimum_gate_groups")
    targets = _normalize_group_hashes(target_sha256, label="target_sha256")
    if set(common_provenance) != REQUIRED_COMMON_PROVENANCE:
        raise ValueError(
            "common_provenance must cover the complete matched calibration contract"
        )
    provenance = {
        key: _sha256(common_provenance[key], label=key)
        for key in sorted(common_provenance)
    }
    margin = _finite_float(
        relative_improvement_margin, label="relative_improvement_margin"
    )
    confidence = _finite_float(confidence_level, label="confidence_level")
    if not 0 <= margin < 1:
        raise ValueError("relative_improvement_margin must lie in [0, 1)")
    if not 0.5 < confidence < 1:
        raise ValueError("confidence_level must lie strictly between 0.5 and 1")
    resamples = _positive_int(bootstrap_resamples, label="bootstrap_resamples")
    if resamples < 200:
        raise ValueError("bootstrap_resamples must be at least 200")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise TypeError("bootstrap_seed must be an integer")
    payload = {
        "schema": V3_PROMOTION_POLICY_SCHEMA,
        "policy_id": _identifier(policy_id, label="policy_id"),
        "advisory_sha256": _sha256(advisory_sha256, label="advisory_sha256"),
        "incumbent_candidate_id": incumbent_id,
        "biological_domain": bool(biological_domain),
        "candidate_metadata": metadata,
        "candidate_artifact_sha256": artifacts,
        "candidate_plan_sha256": plans,
        "resource_budget": budget,
        "fit_groups": fit,
        "gate_groups": gate,
        "target_sha256": targets,
        "common_provenance": provenance,
        "relative_improvement_margin": margin,
        "confidence_level": confidence,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": int(bootstrap_seed),
        "minimum_gate_groups": minimum,
        "multiple_comparison_correction": "bonferroni_one_sided_percentile",
        "selection_tie_break": (
            "largest_lower_bound_then_mean_then_lexicographically_first_id"
        ),
        "require_no_target_mean_harm": bool(require_no_target_mean_harm),
        "registration_record_sha256": _sha256(
            registration_record_sha256, label="registration_record_sha256"
        ),
        "registered_before_gate_evidence": True,
    }
    return MorphologyPromotionPolicyV3(
        **payload,
        content_sha256=_canonical_sha256(payload),
    )


def _quantile_order_statistic(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = max(
        0, min(len(ordered) - 1, math.ceil(probability * (len(ordered) + 1)) - 1)
    )
    return ordered[index]


def _paired_relative_improvements(
    candidate: LocalFitEvidence,
    incumbent: LocalFitEvidence,
) -> tuple[dict[str, float], dict[str, float], list[float]]:
    assert candidate.paired_group_losses is not None
    assert incumbent.paired_group_losses is not None
    target_means: dict[str, float] = {}
    per_group: list[float] = []
    for target in sorted(incumbent.target_sha256):
        improvements: list[float] = []
        for group in sorted(incumbent.gate_groups):
            baseline = float(incumbent.paired_group_losses[target][group])
            proposed = float(candidate.paired_group_losses[target][group])
            if baseline <= 0:
                # A zero-loss incumbent cannot be improved by a nonnegative loss.
                improvement = 0.0 if proposed == 0 else -1.0
            else:
                improvement = (baseline - proposed) / baseline
            improvements.append(improvement)
        target_means[target] = sum(improvements) / len(improvements)
    for group in sorted(incumbent.gate_groups):
        values = []
        for target in sorted(incumbent.target_sha256):
            baseline = float(incumbent.paired_group_losses[target][group])
            proposed = float(candidate.paired_group_losses[target][group])
            values.append(
                0.0
                if baseline == proposed == 0
                else ((baseline - proposed) / baseline if baseline > 0 else -1.0)
            )
        per_group.append(sum(values) / len(values))
    aggregate_losses = {
        "candidate_mean_loss": sum(candidate.aggregate_losses.values())
        / len(candidate.aggregate_losses),
        "incumbent_mean_loss": sum(incumbent.aggregate_losses.values())
        / len(incumbent.aggregate_losses),
    }
    return target_means, aggregate_losses, per_group


def apply_morphology_promotion_v3(
    *,
    policy: MorphologyPromotionPolicyV3,
    advisory_sha256: str,
    evidence_by_candidate: Mapping[str, LocalFitEvidence],
    candidate_plan_sha256_by_candidate: Mapping[str, object],
) -> dict[str, Any]:
    """Promote only a confidence-supported, matched-budget structural winner."""

    policy.validate_content_identity()
    advisory = _sha256(advisory_sha256, label="advisory_sha256")
    if advisory != policy.advisory_sha256:
        raise ValueError("advisory hash does not match the preregistered policy")
    expected = set(policy.candidate_metadata)
    if set(evidence_by_candidate) != expected:
        raise ValueError(
            "evidence_by_candidate must cover every registered candidate exactly"
        )
    observed_plan_hashes = _normalize_hash_map(
        candidate_plan_sha256_by_candidate,
        label="candidate_plan_sha256_by_candidate",
        expected=expected,
    )
    if observed_plan_hashes != policy.candidate_plan_sha256:
        raise ValueError("observed candidate plan hashes differ from registration")
    for candidate_id, evidence in evidence_by_candidate.items():
        if not isinstance(evidence, LocalFitEvidence):
            raise TypeError(f"evidence for {candidate_id!r} must be LocalFitEvidence")
        evidence.validate_content_identity()
        if evidence.candidate_id != candidate_id:
            raise ValueError("evidence key and candidate_id differ")
        if evidence.paired_group_losses is None:
            raise ValueError("FMI-v3 promotion requires paired gate-group losses")
        if (
            evidence.candidate_artifact_sha256
            != policy.candidate_artifact_sha256[candidate_id]
        ):
            raise ValueError(
                "candidate artifact hash differs from the registered policy"
            )
        if evidence.resource_budget != policy.resource_budget:
            raise ValueError("candidate resource budget is not exactly matched")
        if (
            evidence.fit_groups != policy.fit_groups
            or evidence.gate_groups != policy.gate_groups
        ):
            raise ValueError(
                "candidate fit/gate groups differ from the registered policy"
            )
        if evidence.target_sha256 != policy.target_sha256:
            raise ValueError("candidate target set differs from the registered policy")
        if evidence.common_provenance != policy.common_provenance:
            raise ValueError("candidate provenance differs from the registered policy")
        expected_morphology = policy.candidate_metadata[candidate_id][
            "morphology_class"
        ]
        if evidence.morphology_class != expected_morphology:
            raise ValueError("candidate morphology differs from the registered policy")

    incumbent_id = policy.incumbent_candidate_id
    incumbent = evidence_by_candidate[incumbent_id]
    alternatives = sorted(expected - {incumbent_id})
    corrected_alpha = (1.0 - policy.confidence_level) / max(1, len(alternatives))
    rng = random.Random(policy.bootstrap_seed)
    gate_count = len(policy.gate_groups)
    bootstrap_indices = [
        [rng.randrange(gate_count) for _ in range(gate_count)]
        for _ in range(policy.bootstrap_resamples)
    ]

    rows: list[dict[str, Any]] = []
    for candidate_id in alternatives:
        candidate = evidence_by_candidate[candidate_id]
        target_means, aggregate_losses, per_group = _paired_relative_improvements(
            candidate, incumbent
        )
        observed = sum(per_group) / len(per_group)
        samples = [
            sum(per_group[index] for index in indices) / gate_count
            for indices in bootstrap_indices
        ]
        lower = _quantile_order_statistic(samples, corrected_alpha)
        upper = _quantile_order_statistic(samples, 1.0 - corrected_alpha)
        no_target_harm = all(value >= 0.0 for value in target_means.values())
        passes = (
            observed > policy.relative_improvement_margin
            and lower > policy.relative_improvement_margin
            and (no_target_harm or not policy.require_no_target_mean_harm)
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_metadata": policy.candidate_metadata[candidate_id],
                "candidate_plan_sha256": policy.candidate_plan_sha256[candidate_id],
                "paired_gate_group_count": gate_count,
                "target_mean_relative_improvement": target_means,
                "aggregate_losses": aggregate_losses,
                "observed_mean_relative_improvement": observed,
                "bootstrap_one_sided_corrected_alpha": corrected_alpha,
                "bootstrap_confidence_interval": {"lower": lower, "upper": upper},
                "required_relative_improvement_margin": policy.relative_improvement_margin,
                "no_target_mean_harm": no_target_harm,
                "promotion_gate_passed": passes,
            }
        )

    best_observed = max(
        [0.0] + [row["observed_mean_relative_improvement"] for row in rows]
    )
    for row in rows:
        row["empirical_regret_to_best_observed_improvement"] = max(
            0.0, best_observed - row["observed_mean_relative_improvement"]
        )
    eligible = [row for row in rows if row["promotion_gate_passed"]]
    if eligible:
        winner = sorted(
            eligible,
            key=lambda row: (
                -row["bootstrap_confidence_interval"]["lower"],
                -row["observed_mean_relative_improvement"],
                row["candidate_id"],
            ),
        )[0]
        selected = winner["candidate_id"]
        decision = "calibrated_structural_promotion"
    else:
        selected = incumbent_id
        decision = "retain_robust_flat_incumbent"
    payload = {
        "schema": V3_PROMOTION_DECISION_SCHEMA,
        "decision": decision,
        "selected_candidate_id": selected,
        "selected_plan_sha256": policy.candidate_plan_sha256[selected],
        "claim_status": "local_calibration_selection_not_global_optimality",
        "policy": policy.as_dict(),
        "uncertainty": {
            "method": "paired_gate_group_nonparametric_bootstrap",
            "confidence_level": policy.confidence_level,
            "multiple_comparison_correction": policy.multiple_comparison_correction,
            "corrected_one_sided_alpha": corrected_alpha,
            "bootstrap_resamples": policy.bootstrap_resamples,
            "bootstrap_seed": policy.bootstrap_seed,
        },
        "candidate_audit": rows,
        "incumbent_audit": {
            "candidate_id": incumbent_id,
            "candidate_metadata": policy.candidate_metadata[incumbent_id],
            "candidate_plan_sha256": policy.candidate_plan_sha256[incumbent_id],
            "empirical_regret_to_best_observed_improvement": best_observed,
        },
        "provenance": {
            "advisory_sha256": advisory,
            "policy_content_sha256": policy.content_sha256,
            "registration_record_sha256": policy.registration_record_sha256,
            "candidate_artifact_sha256": policy.candidate_artifact_sha256,
            "evidence_content_sha256": {
                candidate_id: evidence_by_candidate[candidate_id].content_sha256
                for candidate_id in sorted(evidence_by_candidate)
            },
            "candidate_plan_sha256": policy.candidate_plan_sha256,
            "fit_groups": policy.fit_groups,
            "gate_groups": policy.gate_groups,
            "target_sha256": policy.target_sha256,
            "common_provenance": policy.common_provenance,
            "matched_resource_budget": policy.resource_budget,
        },
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "AXIS_SOURCES",
    "CAPACITY_AXES",
    "LOCAL_CALIBRATION_AXES",
    "V3_ADVISORY_SCHEMA",
    "V3_PROMOTION_DECISION_SCHEMA",
    "V3_PROMOTION_POLICY_SCHEMA",
    "MorphologyPromotionPolicyV3",
    "apply_morphology_promotion_v3",
    "build_fmi_v3_advisory",
    "build_morphology_promotion_policy_v3",
]
