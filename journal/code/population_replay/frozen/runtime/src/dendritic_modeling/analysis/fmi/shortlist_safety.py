"""Conservative local-fit safety veto for FMI morphology shortlists.

Functional Morphology Inference (FMI) is an advisory architecture prior, not
an optimality oracle.  This module adds a cheap, prospectively specified guard:
fit the FMI-shortlisted cells and one matched-budget flat fallback on one set
of content-addressed source groups, evaluate them on disjoint gate groups, and
remove a shortlist candidate only when the fallback Pareto-dominates it on the
paired group losses.  An optional preregistered finite loss-ratio cap can catch
a catastrophic loss on one target even when another target improves.

Aggregate-only evidence is accepted for diagnostics but is explicitly
exploratory and can never trigger the registered deterministic veto.  The
public decision labels are intentionally restricted to ``advisory_shortlist``
and ``safety_veto_fallback``; neither means that an architecture is optimal.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

LOCAL_FIT_EVIDENCE_SCHEMA = "dendritic_fmi_local_fit_evidence/v1"
SAFETY_VETO_POLICY_SCHEMA = "dendritic_fmi_shortlist_safety_policy/v1"
SAFETY_VETO_DECISION_SCHEMA = "dendritic_fmi_shortlist_safety_decision/v1"

ADVISORY_SHORTLIST = "advisory_shortlist"
SAFETY_VETO_FALLBACK = "safety_veto_fallback"
DECISION_LABELS = frozenset({ADVISORY_SHORTLIST, SAFETY_VETO_FALLBACK})

PAIRED_EVIDENCE = "source_group_disjoint_paired_gate_losses"
AGGREGATE_EXPLORATORY = "aggregate_only_exploratory"

_REQUIRED_COMMON_PROVENANCE = frozenset(
    {
        "teacher_artifact_sha256",
        "data_manifest_sha256",
        "target_manifest_sha256",
        "fit_protocol_sha256",
        "gate_protocol_sha256",
    }
)
_REQUIRED_RESOURCE_BUDGET_AXES = frozenset(
    {"active_weight_macs", "serialized_bytes", "stored_parameters"}
)
_NUMERIC_COMPARISON_RULE = "exact_finite_binary64_order_no_tolerance"
_GROUP_AGGREGATION_RULE = "unweighted_gate_group_arithmetic_mean_per_target"
_EVIDENCE_INTERPRETATION = "deterministic_safety_screen_not_inferential_test"


def _canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _validate_sha256(value: object, *, label: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a full lowercase SHA-256 digest")
    return normalized


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    normalized = value.strip()
    if any(character in normalized for character in "\r\n\t"):
        raise ValueError(f"{label} cannot contain control whitespace")
    return normalized


def _normalize_hash_mapping(
    values: Mapping[object, object], *, label: str
) -> dict[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in values.items():
        key = _identifier(str(raw_key), label=f"{label} key")
        if key in normalized:
            raise ValueError(f"{label} has duplicate normalized key {key!r}")
        normalized[key] = _validate_sha256(raw_value, label=f"{label}[{key!r}]")
    return dict(sorted(normalized.items()))


def _normalize_identifier_mapping(
    values: Mapping[object, object], *, label: str, lowercase_values: bool = False
) -> dict[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for raw_key, raw_value in values.items():
        key = _identifier(str(raw_key), label=f"{label} key")
        if key in normalized:
            raise ValueError(f"{label} has duplicate normalized key {key!r}")
        value = _identifier(raw_value, label=f"{label}[{key!r}]")
        normalized[key] = value.lower() if lowercase_values else value
    return dict(sorted(normalized.items()))


def _require_unique_content_hashes(values: Mapping[str, str], *, label: str) -> None:
    hashes = list(values.values())
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"{label} cannot alias the same content under multiple IDs")


def _normalize_common_provenance(values: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("common_provenance must be a mapping")
    if set(values) != _REQUIRED_COMMON_PROVENANCE:
        missing = sorted(_REQUIRED_COMMON_PROVENANCE - set(values))
        extra = sorted(set(values) - _REQUIRED_COMMON_PROVENANCE)
        raise ValueError(
            "common_provenance fields differ from the required protocol; "
            f"missing={missing}, extra={extra}"
        )
    return {key: _validate_sha256(values[key], label=key) for key in sorted(values)}


def _normalize_candidate_ids(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"{label} must be a non-empty sequence")
    normalized = tuple(_identifier(value, label=label) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} entries must be unique")
    return normalized


def _normalize_budget(values: Mapping[str, object]) -> dict[str, int]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("resource_budget must be a non-empty mapping")
    normalized: dict[str, int] = {}
    for raw_key, raw_value in values.items():
        key = _identifier(raw_key, label="resource budget key")
        if key in normalized:
            raise ValueError(f"resource_budget has duplicate normalized key {key!r}")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise TypeError(
                f"resource_budget[{key!r}] must be an exact integer counter"
            )
        if raw_value < 0:
            raise ValueError(f"resource_budget[{key!r}] must be nonnegative")
        normalized[key] = int(raw_value)
    missing = sorted(_REQUIRED_RESOURCE_BUDGET_AXES - set(normalized))
    if missing:
        raise ValueError(
            f"resource_budget is missing required matched-cost axes: {missing}"
        )
    nonpositive = sorted(
        axis for axis in _REQUIRED_RESOURCE_BUDGET_AXES if normalized[axis] <= 0
    )
    if nonpositive:
        raise ValueError(
            f"required resource-budget axes must be positive: {nonpositive}"
        )
    return dict(sorted(normalized.items()))


def _finite_loss(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return normalized


def _normalize_aggregate_losses(
    values: Mapping[str, object] | None,
    *,
    targets: Mapping[str, str],
) -> dict[str, float] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError("aggregate_losses must be a target-to-loss mapping")
    if set(values) != set(targets):
        raise ValueError(
            "aggregate_losses must cover the content-addressed targets exactly"
        )
    return {
        target: _finite_loss(values[target], label=f"aggregate loss {target!r}")
        for target in sorted(targets)
    }


def _normalize_paired_losses(
    values: Mapping[str, Mapping[object, object]] | None,
    *,
    targets: Mapping[str, str],
    gate_groups: Mapping[str, str],
) -> dict[str, dict[str, float]] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping) or set(values) != set(targets):
        raise ValueError(
            "paired_group_losses must cover the content-addressed targets exactly"
        )
    normalized: dict[str, dict[str, float]] = {}
    for target in sorted(targets):
        group_losses = values[target]
        if not isinstance(group_losses, Mapping):
            raise TypeError(f"paired losses for target {target!r} must be a mapping")
        string_losses = {str(group): loss for group, loss in group_losses.items()}
        if len(string_losses) != len(group_losses):
            raise ValueError(
                f"paired losses for target {target!r} have duplicate group IDs"
            )
        if set(string_losses) != set(gate_groups):
            raise ValueError(
                f"paired losses for target {target!r} must cover gate groups exactly"
            )
        normalized[target] = {
            group: _finite_loss(
                string_losses[group],
                label=f"paired loss {target!r}/{group!r}",
            )
            for group in sorted(gate_groups)
        }
    return normalized


def _paired_aggregates(
    losses: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    aggregates: dict[str, float] = {}
    for target, group_losses in losses.items():
        values = tuple(group_losses.values())
        scale = max(values)
        if scale == 0:
            aggregates[target] = 0.0
            continue
        # Losses are finite and nonnegative.  Scaling first keeps the mean
        # finite even near the largest representable binary64 value.
        normalized_mean = math.fsum(value / scale for value in values) / len(values)
        mean = scale * normalized_mean
        if not math.isfinite(mean):  # pragma: no cover - defensive invariant
            raise ValueError(f"paired mean loss overflowed for target {target!r}")
        aggregates[target] = float(mean)
    return aggregates


@dataclass(frozen=True)
class LocalFitEvidence:
    """One candidate's fit/gate evidence and immutable content identity."""

    schema: str
    candidate_id: str
    morphology_class: str
    resource_budget: dict[str, int]
    fit_groups: dict[str, str]
    gate_groups: dict[str, str]
    target_sha256: dict[str, str]
    common_provenance: dict[str, str]
    candidate_artifact_sha256: str
    paired_group_losses: dict[str, dict[str, float]] | None
    aggregate_losses: dict[str, float]
    evidence_tier: str
    content_sha256: str

    def content_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("content_sha256")
        return payload

    def validate_content_identity(self) -> None:
        if self.schema != LOCAL_FIT_EVIDENCE_SCHEMA:
            raise ValueError("unsupported local-fit evidence schema")
        expected = _canonical_sha256(self.content_payload())
        if self.content_sha256 != expected:
            raise ValueError(
                f"local-fit evidence content hash mismatch for {self.candidate_id!r}"
            )
        rebuilt = build_local_fit_evidence(
            candidate_id=self.candidate_id,
            morphology_class=self.morphology_class,
            resource_budget=self.resource_budget,
            fit_groups=self.fit_groups,
            gate_groups=self.gate_groups,
            target_sha256=self.target_sha256,
            common_provenance=self.common_provenance,
            candidate_artifact_sha256=self.candidate_artifact_sha256,
            paired_group_losses=self.paired_group_losses,
            aggregate_losses=self.aggregate_losses,
        )
        if self != rebuilt:
            raise ValueError(
                f"local-fit evidence is not canonical for {self.candidate_id!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_local_fit_evidence(
    *,
    candidate_id: str,
    morphology_class: str,
    resource_budget: Mapping[str, object],
    fit_groups: Mapping[object, object],
    gate_groups: Mapping[object, object],
    target_sha256: Mapping[object, object],
    common_provenance: Mapping[str, object],
    candidate_artifact_sha256: str,
    paired_group_losses: Mapping[str, Mapping[object, object]] | None = None,
    aggregate_losses: Mapping[str, object] | None = None,
) -> LocalFitEvidence:
    """Validate and content-address one candidate's held-out local-fit evidence."""

    normalized_id = _identifier(candidate_id, label="candidate_id")
    normalized_morphology = _identifier(
        morphology_class, label="morphology_class"
    ).lower()
    normalized_budget = _normalize_budget(resource_budget)
    normalized_fit = _normalize_hash_mapping(fit_groups, label="fit_groups")
    normalized_gate = _normalize_hash_mapping(gate_groups, label="gate_groups")
    _require_unique_content_hashes(normalized_fit, label="fit_groups")
    _require_unique_content_hashes(normalized_gate, label="gate_groups")
    overlap = sorted(set(normalized_fit).intersection(normalized_gate))
    if overlap:
        raise ValueError(
            f"fit and gate source groups must be disjoint; overlap={overlap}"
        )
    content_overlap = sorted(
        set(normalized_fit.values()).intersection(normalized_gate.values())
    )
    if content_overlap:
        raise ValueError(
            "fit and gate source-group contents must be disjoint; "
            f"overlapping_sha256={content_overlap}"
        )
    normalized_targets = _normalize_hash_mapping(target_sha256, label="target_sha256")
    normalized_provenance = _normalize_common_provenance(common_provenance)
    normalized_paired = _normalize_paired_losses(
        paired_group_losses,
        targets=normalized_targets,
        gate_groups=normalized_gate,
    )
    normalized_aggregate = _normalize_aggregate_losses(
        aggregate_losses,
        targets=normalized_targets,
    )
    if normalized_paired is None and normalized_aggregate is None:
        raise ValueError("local-fit evidence requires paired or aggregate losses")
    if normalized_paired is not None:
        derived = _paired_aggregates(normalized_paired)
        if normalized_aggregate is not None:
            for target in sorted(normalized_targets):
                if not math.isclose(
                    normalized_aggregate[target],
                    derived[target],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    raise ValueError(
                        "aggregate loss does not equal the paired group mean for "
                        f"target {target!r}"
                    )
        normalized_aggregate = derived
        evidence_tier = PAIRED_EVIDENCE
    else:
        assert normalized_aggregate is not None
        evidence_tier = AGGREGATE_EXPLORATORY

    payload = {
        "schema": LOCAL_FIT_EVIDENCE_SCHEMA,
        "candidate_id": normalized_id,
        "morphology_class": normalized_morphology,
        "resource_budget": normalized_budget,
        "fit_groups": normalized_fit,
        "gate_groups": normalized_gate,
        "target_sha256": normalized_targets,
        "common_provenance": normalized_provenance,
        "candidate_artifact_sha256": _validate_sha256(
            candidate_artifact_sha256,
            label="candidate_artifact_sha256",
        ),
        "paired_group_losses": normalized_paired,
        "aggregate_losses": normalized_aggregate,
        "evidence_tier": evidence_tier,
    }
    return LocalFitEvidence(
        **payload,
        content_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True)
class SafetyVetoPolicy:
    """Prospectively registered comparison policy."""

    schema: str
    policy_id: str
    fmi_advisory_sha256: str
    advisory_shortlist: tuple[str, ...]
    fallback_candidate_id: str
    fallback_morphology_class: str
    candidate_morphology_class: dict[str, str]
    candidate_artifact_sha256: dict[str, str]
    resource_budget: dict[str, int]
    fit_groups: dict[str, str]
    gate_groups: dict[str, str]
    target_sha256: dict[str, str]
    common_provenance: dict[str, str]
    primary_rule: str
    optional_ratio_cap: float | None
    group_aggregation_rule: str
    numeric_comparison_rule: str
    evidence_interpretation: str
    minimum_gate_group_count: int
    registration_record_sha256: str
    registered_before_gate_evidence: bool
    registration_status: str
    content_sha256: str

    def content_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("content_sha256")
        return payload

    def validate_content_identity(self) -> None:
        if self.schema != SAFETY_VETO_POLICY_SCHEMA:
            raise ValueError("unsupported safety-veto policy schema")
        if _canonical_sha256(self.content_payload()) != self.content_sha256:
            raise ValueError("safety-veto policy content hash mismatch")
        rebuilt = build_safety_veto_policy(
            policy_id=self.policy_id,
            fmi_advisory_sha256=self.fmi_advisory_sha256,
            advisory_shortlist=self.advisory_shortlist,
            fallback_candidate_id=self.fallback_candidate_id,
            candidate_morphology_class=self.candidate_morphology_class,
            candidate_artifact_sha256=self.candidate_artifact_sha256,
            resource_budget=self.resource_budget,
            fit_groups=self.fit_groups,
            gate_groups=self.gate_groups,
            target_sha256=self.target_sha256,
            common_provenance=self.common_provenance,
            registration_record_sha256=self.registration_record_sha256,
            optional_ratio_cap=self.optional_ratio_cap,
            minimum_gate_group_count=self.minimum_gate_group_count,
            registered_before_gate_evidence=self.registered_before_gate_evidence,
        )
        if self != rebuilt:
            raise ValueError("safety-veto policy is not canonical")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_safety_veto_policy(
    *,
    policy_id: str,
    fmi_advisory_sha256: str,
    advisory_shortlist: Sequence[str],
    fallback_candidate_id: str,
    candidate_morphology_class: Mapping[object, object],
    candidate_artifact_sha256: Mapping[object, object],
    resource_budget: Mapping[str, object],
    fit_groups: Mapping[object, object],
    gate_groups: Mapping[object, object],
    target_sha256: Mapping[object, object],
    common_provenance: Mapping[str, object],
    registration_record_sha256: str,
    optional_ratio_cap: float | None = None,
    minimum_gate_group_count: int = 2,
    registered_before_gate_evidence: bool,
) -> SafetyVetoPolicy:
    """Bind a prospective gate policy to its full comparison contract."""

    if registered_before_gate_evidence is not True:
        raise ValueError(
            "the safety-veto policy must be registered before gate evidence"
        )
    shortlist = _normalize_candidate_ids(advisory_shortlist, label="advisory_shortlist")
    fallback_id = _identifier(fallback_candidate_id, label="fallback_candidate_id")
    if fallback_id in shortlist:
        raise ValueError("the preregistered fallback cannot be in the FMI shortlist")
    normalized_morphologies = _normalize_identifier_mapping(
        candidate_morphology_class,
        label="candidate_morphology_class",
        lowercase_values=True,
    )
    expected_candidates = {*shortlist, fallback_id}
    if set(normalized_morphologies) != expected_candidates:
        missing = sorted(expected_candidates - set(normalized_morphologies))
        extra = sorted(set(normalized_morphologies) - expected_candidates)
        raise ValueError(
            "candidate_morphology_class must cover shortlist plus fallback exactly; "
            f"missing={missing}, extra={extra}"
        )
    if normalized_morphologies[fallback_id] != "flat":
        raise ValueError("the preregistered fallback morphology must be 'flat'")
    normalized_artifacts = _normalize_hash_mapping(
        candidate_artifact_sha256,
        label="candidate_artifact_sha256",
    )
    _require_unique_content_hashes(
        normalized_artifacts, label="candidate_artifact_sha256"
    )
    expected_artifacts = expected_candidates
    if set(normalized_artifacts) != expected_artifacts:
        missing = sorted(expected_artifacts - set(normalized_artifacts))
        extra = sorted(set(normalized_artifacts) - expected_artifacts)
        raise ValueError(
            "candidate_artifact_sha256 must cover shortlist plus fallback exactly; "
            f"missing={missing}, extra={extra}"
        )
    normalized_budget = _normalize_budget(resource_budget)
    normalized_fit = _normalize_hash_mapping(fit_groups, label="fit_groups")
    normalized_gate = _normalize_hash_mapping(gate_groups, label="gate_groups")
    _require_unique_content_hashes(normalized_fit, label="fit_groups")
    _require_unique_content_hashes(normalized_gate, label="gate_groups")
    overlap_ids = sorted(set(normalized_fit).intersection(normalized_gate))
    if overlap_ids:
        raise ValueError(
            f"fit and gate source groups must be disjoint; overlap={overlap_ids}"
        )
    overlap_hashes = sorted(
        set(normalized_fit.values()).intersection(normalized_gate.values())
    )
    if overlap_hashes:
        raise ValueError(
            "fit and gate source-group contents must be disjoint; "
            f"overlapping_sha256={overlap_hashes}"
        )
    if isinstance(minimum_gate_group_count, bool) or not isinstance(
        minimum_gate_group_count, int
    ):
        raise TypeError("minimum_gate_group_count must be an integer")
    if minimum_gate_group_count < 1:
        raise ValueError("minimum_gate_group_count must be positive")
    if len(normalized_gate) < minimum_gate_group_count:
        raise ValueError(
            "gate partition has fewer groups than minimum_gate_group_count"
        )
    normalized_targets = _normalize_hash_mapping(target_sha256, label="target_sha256")
    normalized_provenance = _normalize_common_provenance(common_provenance)
    ratio_cap = None
    if optional_ratio_cap is not None:
        if isinstance(optional_ratio_cap, bool) or not isinstance(
            optional_ratio_cap, (int, float)
        ):
            raise TypeError("optional_ratio_cap must be numeric")
        ratio_cap = float(optional_ratio_cap)
        if not math.isfinite(ratio_cap) or ratio_cap <= 1.0:
            raise ValueError("optional_ratio_cap must be finite and greater than one")
    payload = {
        "schema": SAFETY_VETO_POLICY_SCHEMA,
        "policy_id": _identifier(policy_id, label="policy_id"),
        "fmi_advisory_sha256": _validate_sha256(
            fmi_advisory_sha256, label="fmi_advisory_sha256"
        ),
        "advisory_shortlist": shortlist,
        "fallback_candidate_id": fallback_id,
        "fallback_morphology_class": "flat",
        "candidate_morphology_class": normalized_morphologies,
        "candidate_artifact_sha256": normalized_artifacts,
        "resource_budget": normalized_budget,
        "fit_groups": normalized_fit,
        "gate_groups": normalized_gate,
        "target_sha256": normalized_targets,
        "common_provenance": normalized_provenance,
        "primary_rule": "threshold_free_paired_group_pareto_dominance",
        "optional_ratio_cap": ratio_cap,
        "group_aggregation_rule": _GROUP_AGGREGATION_RULE,
        "numeric_comparison_rule": _NUMERIC_COMPARISON_RULE,
        "evidence_interpretation": _EVIDENCE_INTERPRETATION,
        "minimum_gate_group_count": minimum_gate_group_count,
        "registration_record_sha256": _validate_sha256(
            registration_record_sha256,
            label="registration_record_sha256",
        ),
        "registered_before_gate_evidence": True,
        "registration_status": (
            "declared_preregistered_before_gate_with_external_record_hash"
        ),
    }
    return SafetyVetoPolicy(
        **payload,
        content_sha256=_canonical_sha256(payload),
    )


def _matched_comparison_contract(
    candidate: LocalFitEvidence,
    fallback: LocalFitEvidence,
) -> None:
    candidate.validate_content_identity()
    fallback.validate_content_identity()
    if candidate.resource_budget != fallback.resource_budget:
        raise ValueError(
            f"resource budgets are not exactly matched for {candidate.candidate_id!r}"
        )
    if candidate.fit_groups != fallback.fit_groups:
        raise ValueError("fit source-group IDs or content hashes do not match")
    if candidate.gate_groups != fallback.gate_groups:
        raise ValueError("gate source-group IDs or content hashes do not match")
    if candidate.target_sha256 != fallback.target_sha256:
        raise ValueError("local-fit target IDs or content hashes do not match")
    if candidate.common_provenance != fallback.common_provenance:
        raise ValueError(
            "teacher, data, target, or fit/gate protocol hashes do not match"
        )


def _loss_ratio(candidate: float, fallback: float) -> tuple[float | None, str]:
    if fallback > 0:
        ratio = candidate / fallback
        if not math.isfinite(ratio):
            return None, "overflow_candidate_over_fallback"
        return ratio, "finite"
    if candidate == 0:
        return 1.0, "both_zero"
    return None, "infinite_candidate_over_zero_fallback"


def _compare_candidate(
    candidate: LocalFitEvidence,
    fallback: LocalFitEvidence,
    *,
    ratio_cap: float | None,
) -> dict[str, Any]:
    _matched_comparison_contract(candidate, fallback)
    paired = (
        candidate.paired_group_losses is not None
        and fallback.paired_group_losses is not None
    )
    target_rows: dict[str, Any] = {}
    for target in sorted(candidate.target_sha256):
        candidate_mean = candidate.aggregate_losses[target]
        fallback_mean = fallback.aggregate_losses[target]
        ratio, ratio_status = _loss_ratio(candidate_mean, fallback_mean)
        target_rows[target] = {
            "target_sha256": candidate.target_sha256[target],
            "candidate_mean_loss": candidate_mean,
            "fallback_mean_loss": fallback_mean,
            "candidate_over_fallback_loss_ratio": ratio,
            "loss_ratio_status": ratio_status,
        }

    if paired:
        axes = [
            (
                target,
                group,
                candidate.paired_group_losses[target][group],
                fallback.paired_group_losses[target][group],
            )
            for target in sorted(candidate.target_sha256)
            for group in sorted(candidate.gate_groups)
        ]
        paired_rows = {
            target: {
                group: {
                    "candidate_loss": candidate.paired_group_losses[target][group],
                    "fallback_loss": fallback.paired_group_losses[target][group],
                    "candidate_minus_fallback_loss": (
                        candidate.paired_group_losses[target][group]
                        - fallback.paired_group_losses[target][group]
                    ),
                }
                for group in sorted(candidate.gate_groups)
            }
            for target in sorted(candidate.target_sha256)
        }
        fallback_weakly_better = all(
            fallback_loss <= candidate_loss
            for _target, _group, candidate_loss, fallback_loss in axes
        )
        fallback_strictly_better = any(
            fallback_loss < candidate_loss
            for _target, _group, candidate_loss, fallback_loss in axes
        )
        candidate_weakly_better = all(
            candidate_loss <= fallback_loss
            for _target, _group, candidate_loss, fallback_loss in axes
        )
        candidate_strictly_better = any(
            candidate_loss < fallback_loss
            for _target, _group, candidate_loss, fallback_loss in axes
        )
        fallback_dominates = fallback_weakly_better and fallback_strictly_better
        candidate_dominates = candidate_weakly_better and candidate_strictly_better
        if fallback_dominates:
            relationship = "fallback_pareto_dominates_advisory"
        elif candidate_dominates:
            relationship = "advisory_pareto_dominates_fallback"
        elif all(
            candidate_loss == fallback_loss
            for *_, candidate_loss, fallback_loss in axes
        ):
            relationship = "exact_tie"
        else:
            relationship = "mixed_or_uncertain"

        ratio_breaches = []
        if ratio_cap is not None:
            for target, row in target_rows.items():
                ratio = row["candidate_over_fallback_loss_ratio"]
                if ratio is None or ratio > ratio_cap:
                    ratio_breaches.append(target)
        if fallback_dominates:
            vetoed = True
            veto_rule = "threshold_free_paired_group_pareto_dominance"
        elif ratio_breaches:
            vetoed = True
            veto_rule = "preregistered_target_mean_loss_ratio_cap"
        else:
            vetoed = False
            veto_rule = None
        return {
            "candidate_id": candidate.candidate_id,
            "candidate_morphology_class": candidate.morphology_class,
            "fallback_morphology_class": fallback.morphology_class,
            "evidence_tier": PAIRED_EVIDENCE,
            "candidate_source_evidence_tier": candidate.evidence_tier,
            "fallback_source_evidence_tier": fallback.evidence_tier,
            "comparison_downgrade_reason": None,
            "exploratory_only": False,
            "evidence_interpretation": _EVIDENCE_INTERPRETATION,
            "pareto_relationship": relationship,
            "paired_axis_count": len(axes),
            "ratio_cap": ratio_cap,
            "ratio_cap_breached_targets": ratio_breaches,
            "vetoed": vetoed,
            "veto_rule": veto_rule,
            "paired_group_losses": paired_rows,
            "targets": target_rows,
        }

    # Aggregate-only comparisons are descriptive.  They deliberately do not
    # alter the shortlist, even if every aggregate favors the fallback or a
    # preregistered cap would have fired with paired evidence.
    fallback_weakly_better = all(
        row["fallback_mean_loss"] <= row["candidate_mean_loss"]
        for row in target_rows.values()
    )
    fallback_strictly_better = any(
        row["fallback_mean_loss"] < row["candidate_mean_loss"]
        for row in target_rows.values()
    )
    exploratory_ratio_breaches = []
    if ratio_cap is not None:
        exploratory_ratio_breaches = [
            target
            for target, row in target_rows.items()
            if row["candidate_over_fallback_loss_ratio"] is None
            or row["candidate_over_fallback_loss_ratio"] > ratio_cap
        ]
    if candidate.paired_group_losses is None and fallback.paired_group_losses is None:
        downgrade_reason = "paired_group_losses_missing_for_both_arms"
    elif candidate.paired_group_losses is None:
        downgrade_reason = "paired_group_losses_missing_for_advisory_arm"
    else:
        downgrade_reason = "paired_group_losses_missing_for_fallback_arm"
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_morphology_class": candidate.morphology_class,
        "fallback_morphology_class": fallback.morphology_class,
        "evidence_tier": AGGREGATE_EXPLORATORY,
        "candidate_source_evidence_tier": candidate.evidence_tier,
        "fallback_source_evidence_tier": fallback.evidence_tier,
        "comparison_downgrade_reason": downgrade_reason,
        "exploratory_only": True,
        "evidence_interpretation": _EVIDENCE_INTERPRETATION,
        "pareto_relationship": (
            "exploratory_aggregate_fallback_dominance"
            if fallback_weakly_better and fallback_strictly_better
            else "exploratory_aggregate_mixed_or_tied"
        ),
        "paired_axis_count": 0,
        "ratio_cap": ratio_cap,
        "ratio_cap_breached_targets": exploratory_ratio_breaches,
        "vetoed": False,
        "veto_rule": None,
        "paired_group_losses": None,
        "targets": target_rows,
    }


def apply_shortlist_safety_veto(
    *,
    advisory_shortlist: Sequence[str],
    fmi_advisory_sha256: str,
    evidence_by_candidate: Mapping[str, LocalFitEvidence],
    policy: SafetyVetoPolicy,
) -> dict[str, Any]:
    """Apply a conservative matched-budget veto to an ordered FMI shortlist.

    Unsafe candidates are removed individually.  The flat fallback replaces
    the shortlist only if every advisory candidate is vetoed.  Ties, mixed
    paired results, and all aggregate-only results remain advisory.
    """

    policy.validate_content_identity()
    advisory_source_sha256 = _validate_sha256(
        fmi_advisory_sha256, label="fmi_advisory_sha256"
    )
    shortlist_tuple = _normalize_candidate_ids(
        advisory_shortlist, label="advisory_shortlist"
    )
    shortlist = list(shortlist_tuple)
    if shortlist_tuple != policy.advisory_shortlist:
        raise ValueError(
            "supplied advisory shortlist does not match the preregistered policy"
        )
    if advisory_source_sha256 != policy.fmi_advisory_sha256:
        raise ValueError(
            "supplied FMI advisory hash does not match the preregistered policy"
        )
    fallback_id = policy.fallback_candidate_id
    if fallback_id in shortlist:
        raise ValueError("the preregistered fallback cannot be in the FMI shortlist")
    expected_evidence = {*shortlist, fallback_id}
    if set(evidence_by_candidate) != expected_evidence:
        missing = sorted(expected_evidence - set(evidence_by_candidate))
        extra = sorted(set(evidence_by_candidate) - expected_evidence)
        raise ValueError(
            "evidence_by_candidate must cover shortlist plus fallback exactly; "
            f"missing={missing}, extra={extra}"
        )
    for key, evidence in evidence_by_candidate.items():
        if not isinstance(evidence, LocalFitEvidence):
            raise TypeError(f"evidence for {key!r} must be LocalFitEvidence")
        if evidence.candidate_id != key:
            raise ValueError(f"evidence key {key!r} does not match candidate_id")
        evidence.validate_content_identity()
        if evidence.morphology_class != policy.candidate_morphology_class[key]:
            raise ValueError(
                f"morphology class for {key!r} does not match the preregistered policy"
            )
        if evidence.candidate_artifact_sha256 != policy.candidate_artifact_sha256[key]:
            raise ValueError(
                f"artifact hash for {key!r} does not match the preregistered policy"
            )
        registered_contract = (
            ("resource budget", evidence.resource_budget, policy.resource_budget),
            ("fit groups", evidence.fit_groups, policy.fit_groups),
            ("gate groups", evidence.gate_groups, policy.gate_groups),
            ("targets", evidence.target_sha256, policy.target_sha256),
            (
                "common provenance",
                evidence.common_provenance,
                policy.common_provenance,
            ),
        )
        for label, observed, registered in registered_contract:
            if observed != registered:
                raise ValueError(
                    f"{label} for {key!r} does not match the preregistered policy"
                )
    fallback = evidence_by_candidate[fallback_id]
    if fallback.morphology_class != policy.fallback_morphology_class:
        raise ValueError("preregistered fallback must have morphology_class='flat'")

    comparisons = [
        _compare_candidate(
            evidence_by_candidate[candidate],
            fallback,
            ratio_cap=policy.optional_ratio_cap,
        )
        for candidate in shortlist
    ]
    vetoed = [row["candidate_id"] for row in comparisons if row["vetoed"]]
    retained = [candidate for candidate in shortlist if candidate not in vetoed]
    decision = SAFETY_VETO_FALLBACK if not retained else ADVISORY_SHORTLIST
    if decision not in DECISION_LABELS:  # pragma: no cover - invariant guard
        raise RuntimeError("invalid safety-veto decision label")
    evidence_tiers = sorted({row["evidence_tier"] for row in comparisons})
    payload = {
        "schema": SAFETY_VETO_DECISION_SCHEMA,
        "decision": decision,
        "claim_status": "advisory_only_never_optimal",
        "original_advisory_shortlist": shortlist,
        "retained_advisory_shortlist": retained,
        "vetoed_advisory_candidates": vetoed,
        "fallback_candidate_id": fallback_id,
        "resolved_candidate_ids": (
            [fallback_id] if decision == SAFETY_VETO_FALLBACK else retained
        ),
        "policy": policy.as_dict(),
        "evidence_tiers": evidence_tiers,
        "comparisons": comparisons,
        "provenance": {
            "fmi_advisory_sha256": advisory_source_sha256,
            "advisory_shortlist_sha256": _canonical_sha256(shortlist),
            "registration_record_sha256": policy.registration_record_sha256,
            "registration_status": policy.registration_status,
            "policy_content_sha256": policy.content_sha256,
            "evidence_content_sha256": {
                candidate: evidence_by_candidate[candidate].content_sha256
                for candidate in sorted(evidence_by_candidate)
            },
            "candidate_artifact_sha256": policy.candidate_artifact_sha256,
            "common_provenance": policy.common_provenance,
            "fit_groups": policy.fit_groups,
            "fit_group_partition_sha256": _canonical_sha256(policy.fit_groups),
            "gate_groups": policy.gate_groups,
            "gate_group_partition_sha256": _canonical_sha256(policy.gate_groups),
            "target_sha256": policy.target_sha256,
            "target_set_sha256": _canonical_sha256(policy.target_sha256),
            "matched_resource_budget": policy.resource_budget,
            "matched_resource_budget_sha256": _canonical_sha256(policy.resource_budget),
        },
    }
    payload["content_sha256"] = _canonical_sha256(payload)
    return payload


__all__ = [
    "ADVISORY_SHORTLIST",
    "AGGREGATE_EXPLORATORY",
    "DECISION_LABELS",
    "LOCAL_FIT_EVIDENCE_SCHEMA",
    "PAIRED_EVIDENCE",
    "SAFETY_VETO_DECISION_SCHEMA",
    "SAFETY_VETO_FALLBACK",
    "SAFETY_VETO_POLICY_SCHEMA",
    "LocalFitEvidence",
    "SafetyVetoPolicy",
    "apply_shortlist_safety_veto",
    "build_local_fit_evidence",
    "build_safety_veto_policy",
]
