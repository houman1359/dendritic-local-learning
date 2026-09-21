"""Teacher-conditioned selection of executable replacement architectures.

FMI is deliberately separated from the replacement compiler.  This module
turns a *precomputed* teacher fingerprint into compiler arguments, resolving
every architectural axis independently as ``manual``, ``fmi``, or a declared
fallback.  It never profiles a teacher during model construction: profiling is
an explicit, reproducible calibration job in :mod:`dendritic_modeling.analysis.fmi`.

``theory_v2`` is a versioned prospective rule set.  It extends, but does not
modify, the frozen ``frozen_prospective_fmi_v1`` experiment.  Until its own
reference grid is run, its outputs are hypotheses to test rather than claims
of optimality.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dendritic_modeling.networks.architectures.replacement.compiler import (
    TOPOLOGY_MODES,
    CompiledReplacementPlan,
    compile_replacement_candidate,
)

FMI_SELECTABLE_AXES = (
    "biological_neuron",
    "density",
    "input_to_excitatory_density",
    "input_to_inhibitory_density",
    "inhibitory_to_excitatory_density",
    "population_width",
    "inhibitory_fraction",
    "explicit_ei",
    "branch_factors",
    "excitatory_branch_factors",
    "inhibitory_branch_factors",
    "somatic_synapses",
    "somatic_excitatory_synapses",
    "somatic_inhibitory_synapses",
    "inhibitory_population_somatic_synapses",
    "affine_bypass_mode",
    "affine_bypass_density",
    "affine_bypass_rank",
    "integration_rule",
    "gated",
    "output_density",
    "output_rank",
    "teacher_support_metric",
)
MANUAL_ONLY_AXES = (
    "topology_mode",
    "output_topology_mode",
    "reactivation_type",
    "gate_activation",
    "input_transform",
    "output_projection_mode",
    "affine_bypass_topology_mode",
)
ALL_SELECTION_AXES = FMI_SELECTABLE_AXES + MANUAL_ONLY_AXES


@dataclass(frozen=True)
class ResolvedReplacementSelection:
    """An executable plan plus a complete per-axis decision journal."""

    plan: CompiledReplacementPlan
    manifest: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"plan": self.plan.as_dict(), "manifest": self.manifest}


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


def _extract_artifact_entry(payload: Mapping[str, Any], key: str | None) -> dict:
    """Resolve direct fingerprints and the committed ``targets`` format."""

    if key:
        targets = payload.get("targets")
        if isinstance(targets, Mapping) and key in targets:
            entry = targets[key]
        elif key in payload:
            entry = payload[key]
        else:
            available = sorted(targets)[:8] if isinstance(targets, Mapping) else []
            raise KeyError(
                f"FMI fingerprint key {key!r} was not found; available target "
                f"keys include {available}"
            )
        if not isinstance(entry, Mapping):
            raise TypeError(f"FMI fingerprint entry {key!r} must be a mapping")
        return dict(entry)
    return dict(payload)


def load_fmi_fingerprint(
    fingerprint_config: Any,
    *,
    layer_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load ``(fingerprint, prior_decision, provenance)`` from config.

    Accepted forms are ``values: {...}``, a direct fingerprint mapping, or a
    JSON ``path`` plus optional ``key``.  ``{layer}`` in the key is expanded
    for heterogeneous transformer-layer selection.
    """

    cfg = _plain(fingerprint_config)
    path = cfg.get("path")
    key = cfg.get("key")
    if key is not None and layer_index is not None:
        key = str(key).format(layer=int(layer_index), layer_index=int(layer_index))
    if path:
        artifact_path = Path(str(path)).expanduser().resolve()
        artifact_bytes = artifact_path.read_bytes()
        payload = json.loads(artifact_bytes)
        if not isinstance(payload, Mapping):
            raise TypeError("FMI fingerprint JSON root must be a mapping")
        entry = _extract_artifact_entry(payload, None if key is None else str(key))
        protocol = payload.get("protocol", entry.get("protocol", "unknown"))
        provenance = {
            "path": str(artifact_path),
            "key": key,
            "protocol": protocol,
            "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        }
    else:
        values = cfg.get("values", cfg)
        if not isinstance(values, Mapping):
            raise TypeError("selection.fingerprint.values must be a mapping")
        entry = dict(values)
        canonical = json.dumps(entry, sort_keys=True, separators=(",", ":")).encode()
        provenance = {
            "path": None,
            "key": key,
            "protocol": entry.get("protocol"),
            "sha256": hashlib.sha256(canonical).hexdigest(),
        }

    fingerprint = entry.get("fingerprint", entry)
    decision = entry.get("decision", {})
    if not isinstance(fingerprint, Mapping) or not isinstance(decision, Mapping):
        raise TypeError("FMI fingerprint and decision payloads must be mappings")
    if isinstance(entry.get("boundary"), Mapping):
        provenance["boundary"] = dict(entry["boundary"])
    return dict(fingerprint), dict(decision), provenance


def validate_profile_boundary(
    boundary: Mapping[str, Any],
    *,
    module_path: str | None,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> dict[str, Any]:
    """Reject known profile/execution identity mismatches, including whole FFN
    profiles reused at one of their projections. Missing identity stays unknown.
    """

    def canonical(path: str) -> str:
        path = re.sub(r"^layer\[(\d+)\]", r"layers.\1", path)
        # Normalize the known shorthand and HF wrapper only. Dropping arbitrary
        # prefixes would equate encoder.layers.N with decoder.layers.N.
        return path.removeprefix("model.") if path.startswith("model.layers.") else path

    profiled_path = boundary.get("module_path")
    if (
        profiled_path
        and module_path
        and canonical(str(profiled_path)) != canonical(module_path)
    ):
        raise ValueError(
            f"FMI boundary mismatch: profile {profiled_path!r} cannot prescribe "
            f"replacement {module_path!r}; profile the exact executed boundary. "
            "Whole-module to projection transfer is an unvalidated heuristic."
        )
    for key, actual in (("input_dim", input_dim), ("output_dim", output_dim)):
        if (
            boundary.get(key) is not None
            and actual is not None
            and int(boundary[key]) != actual
        ):
            raise ValueError(
                f"FMI boundary mismatch: profiled {key}={boundary[key]} but replacement {key}={actual}"
            )
    return {
        "status": "matched_declared_identity"
        if profiled_path and module_path
        else "identity_unrecorded",
        "profiled_module_path": profiled_path,
        "replacement_module_path": module_path,
    }


def _rank_assessment(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    """Keep empirical rank diagnostics separate from population-rank evidence."""
    candidate = next(
        (
            fingerprint[k]
            for k in ("task_rank_995", "output_spectrum_rank_995", "task_rank")
            if k in fingerprint
        ),
        None,
    )
    result: dict[str, Any] = {
        "status": "advisory_unvalidated" if candidate is not None else "not_available",
        "candidate_rank": candidate,
        "validated_rank": None,
        "claim": "empirical output-spectrum candidate; population rank and heldout reconstruction not established",
    }
    rows, dim = fingerprint.get("n_examples"), fingerprint.get("output_dim")
    if candidate is not None and rows is not None and dim is not None:
        from dendritic_modeling.analysis.fmi.rank_validation import (
            rank_capacity_diagnostics,
        )

        diagnostics = rank_capacity_diagnostics(
            int(candidate), sample_size=int(rows), feature_dimension=int(dim)
        )
        if int(candidate) > diagnostics["centered_sample_ceiling"]:
            raise ValueError(
                "FMI rank exceeds its centered sample ceiling; check profile metadata"
            )
        result.update(diagnostics)
        result["near_sample_ceiling"] = bool(
            diagnostics["sample_limited"] and diagnostics["capacity_fraction"] >= 0.8
        )
    else:
        result["capacity_diagnostics"] = (
            "unavailable_without_rank_sample_count_and_output_dimension"
        )
    return result


def _fmi_predictions(
    fingerprint: Mapping[str, Any],
    prior_decision: Mapping[str, Any],
    *,
    hidden_size: int,
    teacher_intermediate_size: int,
    thresholds: Mapping[str, Any],
    rule_set: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Apply the versioned v2 theory rules without hiding their assumptions."""

    explanations: dict[str, str] = {}
    predictions: dict[str, Any] = {}

    rank = int(
        fingerprint.get(
            "task_rank_995",
            fingerprint.get(
                "output_spectrum_rank_995",
                fingerprint.get("task_rank", teacher_intermediate_size),
            ),
        )
    )
    # Degenerate-covariance gate (2026-08-25 review): the spectrum rank comes
    # from a small-sample covariance whose estimate can collapse when a few
    # extreme outputs dominate (measured: OLMo-2-32B layers 29-33 inferred
    # rank 1 against output dimension 5120, one output norm ~440x the
    # median). A collapsed rank must fail closed, never compile a width-1
    # population as if it were a measurement.
    degenerate_floor = int(thresholds.get("population_width_degenerate_floor", 8))
    # The gate targets collapsed estimates on REAL boundaries (rank 1 against
    # output dimension 5120); genuinely tiny boundaries (unit tests, toy
    # models) may have honest single-digit ranks, so trigger only when the
    # boundary is large enough that such a rank cannot be genuine structure.
    if rank < degenerate_floor and teacher_intermediate_size >= 16 * degenerate_floor:
        raise ValueError(
            f"inferred spectrum rank {rank} is below the degenerate-covariance "
            f"floor {degenerate_floor} for a boundary of intermediate size "
            f"{teacher_intermediate_size}: the calibration covariance is "
            "dominated by outliers or too few samples. Re-profile with more "
            "calibration data or robust rank estimation; refusing to emit a "
            "near-zero population_width as a prescription."
        )
    width_floor = int(thresholds.get("population_width_min", 1))
    width_cap = int(thresholds.get("population_width_max", teacher_intermediate_size))
    predictions["population_width"] = max(width_floor, min(rank, width_cap))
    explanations["population_width"] = (
        "unweighted output-spectrum rank (identity task metric unless a task "
        "metric was registered) is a sample-limited lower-bound CANDIDATE for "
        f"the soma/readout width, clamped to [{width_floor}, {width_cap}]"
    )

    density = fingerprint.get("k95_frac", fingerprint.get("support_fraction"))
    shortlist = prior_decision.get("shortlist", [])
    if bool(thresholds.get("use_prior_shortlist_density", False)) and shortlist:
        first = str(shortlist[0])
        if "@d" in first:
            density = float(first.rsplit("@d", 1)[1])
            explanations["density"] = (
                "density from the frozen/registered candidate shortlist"
            )
    if density is None:
        density = float(thresholds.get("density_fallback", 0.25))
        explanations["density"] = "fallback: fingerprint has no support estimate"
    density_floor = float(thresholds.get("density_min", 1.0 / hidden_size))
    density_cap = float(thresholds.get("density_max", 1.0))
    predictions["density"] = max(density_floor, min(float(density), density_cap))
    explanations.setdefault(
        "density", "activation-weighted support fraction needed for target energy"
    )

    excitatory_density = fingerprint.get("excitatory_k95_frac")
    inhibitory_density = fingerprint.get("inhibitory_k95_frac")
    if excitatory_density is None:
        excitatory_density = density
        excitatory_reason = (
            "shared total-support fallback because this fingerprint predates "
            "polarity-conditioned support estimation"
        )
    else:
        excitatory_reason = (
            "support fraction within sign-stable positive functional attribution"
        )
    if inhibitory_density is None:
        inhibitory_density = max(
            density_floor,
            float(density) * float(fingerprint.get("inhib_frac", 0.25)),
        )
        inhibitory_reason = (
            "total support scaled by the functional negative-energy share; a "
            "fallback allocation prior requiring grid confirmation"
        )
    else:
        inhibitory_reason = (
            "support fraction within sign-stable negative functional attribution"
        )
    predictions["input_to_excitatory_density"] = max(
        density_floor, min(float(excitatory_density), density_cap)
    )
    predictions["input_to_inhibitory_density"] = max(
        density_floor, min(float(inhibitory_density), density_cap)
    )
    explanations["input_to_excitatory_density"] = excitatory_reason
    explanations["input_to_inhibitory_density"] = inhibitory_reason

    inhibitory = float(fingerprint.get("inhib_frac", 0.25))
    inhibitory_floor = float(thresholds.get("inhibitory_fraction_min", 0.05))
    inhibitory_cap = float(thresholds.get("inhibitory_fraction_max", 0.5))
    predictions["inhibitory_fraction"] = max(
        inhibitory_floor, min(inhibitory, inhibitory_cap)
    )
    explanations["inhibitory_fraction"] = (
        "gradient-sign-consistent inhibitory share; an empirical allocation "
        "prior, not a theorem"
    )
    inhibitory_readout_density = fingerprint.get("inhibitory_readout_support_fraction")
    if inhibitory_readout_density is None:
        inhibitory_readout_density = max(density_floor, min(1.0, inhibitory))
        inhibitory_readout_reason = (
            "functional negative-energy share used only as an I-to-E readout "
            "density prior because teacher-only observations cannot identify "
            "the student's inhibitory basis; this axis requires validation"
        )
    else:
        inhibitory_readout_reason = (
            "measured support of a registered inhibitory-basis readout"
        )
    predictions["inhibitory_to_excitatory_density"] = float(inhibitory_readout_density)
    explanations["inhibitory_to_excitatory_density"] = inhibitory_readout_reason
    stable_fraction = float(fingerprint.get("sign_stable_energy_fraction", 1.0))
    positive_ei_regret = fingerprint.get(
        "positive_ei_regret",
        prior_decision.get("positive_ei_regret"),
    )
    selected_family = str(
        prior_decision.get(
            "selected_family",
            prior_decision.get("selected_candidate", ""),
        )
    ).lower()
    if not selected_family:
        shortlist = prior_decision.get("shortlist", [])
        if shortlist:
            selected_family = str(shortlist[0]).lower()
    if positive_ei_regret is not None:
        max_regret = float(thresholds.get("positive_ei_max_regret", 0.02))
        predictions["biological_neuron"] = float(positive_ei_regret) <= max_regret
        explanations["biological_neuron"] = (
            "positive-E/I validation regret compared with the best relaxed "
            f"candidate ({float(positive_ei_regret):.4g}) versus the allowed "
            f"threshold ({max_regret:.4g})"
        )
    elif selected_family:
        predictions["biological_neuron"] = "positive_ei" in selected_family
        explanations["biological_neuron"] = (
            "constraint class of the registered validation-selected candidate"
        )
    else:
        stable_threshold = float(thresholds.get("positive_ei_min_stable_energy", 0.8))
        inhibitory_threshold = float(
            thresholds.get("positive_ei_min_inhibitory_fraction", 0.05)
        )
        predictions["biological_neuron"] = (
            stable_fraction >= stable_threshold and inhibitory >= inhibitory_threshold
        )
        explanations["biological_neuron"] = (
            "prospective positive-E/I feasibility rule from sign-stable "
            "teacher attribution; relaxed signed execution is selected when "
            "the stable-polarity evidence is insufficient"
        )
    predictions["explicit_ei"] = inhibitory >= float(
        thresholds.get("explicit_ei_min_fraction", 0.05)
    ) and stable_fraction >= float(thresholds.get("explicit_ei_min_stable_energy", 0.5))
    explanations["explicit_ei"] = (
        "explicit I population retained when sign-consistent inhibitory "
        "energy and total stable-polarity energy exceed configured minima"
    )

    r_mult = float(fingerprint.get("r_mult", 0.0))
    predictions["gated"] = r_mult > float(thresholds.get("r_mult_gate", 0.0))
    explanations["gated"] = (
        "finite-amplitude product-probe advantage over additive and saturating controls"
    )

    mechanism = fingerprint.get("mechanism_scores", {})
    shunting_probability = fingerprint.get("shunting_win_probability")
    shunting_threshold = float(thresholds.get("shunting_win_probability", 0.95))
    if (
        shunting_probability is not None
        and float(shunting_probability) >= shunting_threshold
    ):
        integration = "shunting"
        integration_reason = (
            "paired finite-amplitude bootstrap supports shunting: "
            f"P(win)={float(shunting_probability):.3f} >= {shunting_threshold:.3f}"
        )
    elif isinstance(mechanism, Mapping) and mechanism:
        admissible = {
            name: value for name, value in mechanism.items() if str(name) != "shunting"
        }
        best = min(admissible, key=lambda name: float(admissible[name]))
        integration = {
            "raw_additive": "raw_additive",
            "normalized_additive": "conductance_normalized",
            "tangent_additive": "tangent_matched",
        }.get(str(best), "raw_additive")
        # Tangent execution needs calibrated operating-point anchors. Without
        # them the result remains scientific evidence but cannot be compiled.
        if integration == "tangent_matched" and not all(
            key in fingerprint for key in ("additive_tangent_n0", "additive_tangent_t0")
        ):
            integration = "raw_additive"
            integration_reason = (
                "tangent probe won but no operating-point anchors were stored; "
                "using the executable raw-additive control"
            )
        else:
            integration_reason = f"lowest matched-probe held-out risk ({best})"
    else:
        integration = "raw_additive"
        integration_reason = (
            "no paired finite-amplitude bootstrap licensed shunting; using the "
            "registered additive fallback"
        )
    predictions["integration_rule"] = integration
    explanations["integration_rule"] = integration_reason

    branch_factors = fingerprint.get(
        "interaction_branch_factors", fingerprint.get("branch_factors")
    )
    # An explicitly measured empty tree means "flat", not "missing". Only a
    # null/absent estimator should fall through to a manual/default morphology.
    if branch_factors is not None:
        predictions["branch_factors"] = [int(value) for value in branch_factors]
        explanations["branch_factors"] = (
            "recursive interaction-graph separability on high-attribution inputs; "
            "an empty measured tree selects a flat population"
        )
        e_branches = fingerprint.get(
            "excitatory_interaction_branch_factors", branch_factors
        )
        i_branches = fingerprint.get(
            "inhibitory_interaction_branch_factors", branch_factors
        )
        predictions["excitatory_branch_factors"] = [int(value) for value in e_branches]
        predictions["inhibitory_branch_factors"] = [int(value) for value in i_branches]
        explanations["excitatory_branch_factors"] = (
            "positive-polarity interaction tree when available, otherwise the "
            "registered joint interaction tree"
        )
        explanations["inhibitory_branch_factors"] = (
            "negative-polarity interaction tree when available, otherwise the "
            "registered joint interaction tree"
        )

    linear_bypass = fingerprint.get("linear_bypass_fraction")
    if linear_bypass is not None:
        threshold = float(thresholds.get("somatic_bypass_fraction", 0.15))
        bypass_enabled = float(linear_bypass) >= threshold
        predictions["somatic_synapses"] = bypass_enabled
        explanations["somatic_synapses"] = (
            "held-out variance captured by the affine main-effect probe "
            f"({float(linear_bypass):.4g}) compared with the configured "
            f"bypass threshold ({threshold:.4g}); prospective heuristic"
        )
        bypass_support = float(fingerprint.get("linear_bypass_k95_frac", density))
        total_bypass_contacts = (
            max(1, round(bypass_support * hidden_size)) if bypass_enabled else 0
        )
        inhibitory_bypass_contacts = round(total_bypass_contacts * inhibitory)
        excitatory_bypass_contacts = total_bypass_contacts - inhibitory_bypass_contacts
        predictions["somatic_excitatory_synapses"] = excitatory_bypass_contacts
        predictions["somatic_inhibitory_synapses"] = inhibitory_bypass_contacts
        predictions["inhibitory_population_somatic_synapses"] = 0
        explanations["somatic_excitatory_synapses"] = (
            "affine-probe support budget allocated to the positive functional path"
        )
        explanations["somatic_inhibitory_synapses"] = (
            "affine-probe support budget allocated by negative attribution energy"
        )
        explanations["inhibitory_population_somatic_synapses"] = (
            "no teacher-identifiable direct bypass into the latent I population; "
            "kept at zero pending a measured student-basis grid"
        )
        if rule_set == "theory_v4":
            predictions["affine_bypass_mode"] = "sparse" if bypass_enabled else "none"
            predictions["affine_bypass_density"] = max(
                density_floor, min(float(bypass_support), density_cap)
            )
            predictions["affine_bypass_rank"] = max(1, min(rank, hidden_size))
            explanations["affine_bypass_mode"] = (
                "explicit affine main-effect path is enabled when the held-out "
                "linear probe clears the somatic-bypass threshold"
            )
            explanations["affine_bypass_density"] = (
                "coordinate support of the held-out affine main-effect fit"
            )
            explanations["affine_bypass_rank"] = (
                "task-weighted output rank, used only when the affine path is "
                "manually or experimentally assigned low-rank storage"
            )

    predictions["output_rank"] = max(1, min(rank, hidden_size))
    explanations["output_rank"] = (
        "unvalidated empirical output-spectrum candidate; executable only when the registered "
        "output projection mode is low_rank"
    )
    output_density = fingerprint.get("output_support_fraction")
    if output_density is None:
        output_density = predictions["density"]
        output_density_reason = (
            "input-support density used as a conservative readout-grid center; "
            "teacher observations do not identify the learned student soma "
            "basis, so this is not an output-topology estimate"
        )
    else:
        output_density_reason = (
            "support fraction from a registered student-basis/readout probe"
        )
    predictions["output_density"] = max(
        1.0 / max(teacher_intermediate_size, 1),
        min(float(output_density), 1.0),
    )
    explanations["output_density"] = output_density_reason

    predictions["teacher_support_metric"] = prior_decision.get(
        "selection_metric", "activation_weighted"
    )
    explanations["teacher_support_metric"] = (
        "teacher-support scoring policy from the fingerprint decision record"
    )
    return predictions, explanations


def _family_name(
    *, biological: bool, explicit_ei: bool, branched: bool, gated: bool, shunting: bool
) -> str:
    prefix = "gated_" if gated else ""
    if biological:
        if not explicit_ei:
            raise ValueError(
                "biological_neuron=true requires explicit_ei=true so signed "
                "effects are represented by separate positive E/I pathways"
            )
        stem = "positive_ei"
    elif explicit_ei:
        stem = "signed_ei"
    else:
        stem = "signed"
    if not branched:
        return f"{prefix}{stem}_flat"
    suffix = "shunting" if shunting else "additive"
    return (
        f"{prefix}{stem}_branched_{suffix}"
        if explicit_ei
        else f"{prefix}{stem}_branched"
    )


def _robustness_assessment(
    fingerprint: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Turn teacher sample tails into validation requirements, not a claim."""

    profile = fingerprint.get("sample_tail_profile", {})
    if not isinstance(profile, Mapping):
        profile = {}
    observed = []
    for section in ("input", "output"):
        values = profile.get(section, {})
        if isinstance(values, Mapping):
            for metric in (
                "row_l2_p99_over_median",
                "row_l2_max_over_median",
                "standardized_row_rms_p99_over_median",
                "standardized_row_rms_max_over_median",
            ):
                if metric in values:
                    observed.append(float(values[metric]))
    tail_score = max(observed) if observed else None
    threshold = float(thresholds.get("sample_tail_validation_threshold", 3.0))
    requires_tail_validation = tail_score is None or tail_score >= threshold
    # Hard tail gate (2026-08-25 review, second pass): a moderate tail is a
    # validation requirement, but an EXTREME tail (one row hundreds of times
    # the median) means the calibration covariance -- and every quantity
    # derived from it, the spectrum rank included -- is dominated by outliers.
    # Emitting a prescription from that state is not a candidate, it is
    # noise; fail closed instead of merely recording tail_risk_detected.
    hard_threshold = float(thresholds.get("sample_tail_hard_threshold", 100.0))
    estimator_record = fingerprint.get("spectrum_estimator", {})
    spectrum_estimator = (
        str(estimator_record.get("name", "ordinary"))
        if isinstance(estimator_record, Mapping)
        else "ordinary"
    )
    tail_hard_exceeded = tail_score is not None and tail_score >= hard_threshold
    if tail_hard_exceeded and spectrum_estimator == "ordinary":
        raise ValueError(
            f"sample tail score {tail_score:.1f} exceeds the hard gate "
            f"{hard_threshold:.1f}: the calibration statistics are dominated "
            "by outlier rows and no architecture prescription can be derived "
            "from them. Re-profile with more/cleaner calibration data or a "
            "robust estimator."
        )
    # With a robust spectrum estimator the gate's premise no longer holds --
    # the rank is not derived from the outlier-dominated ordinary covariance
    # (frozen huber, 2026-08-26). The extreme tail stays RECORDED and forces
    # tail validation; it just no longer aborts pure prescription.
    hard_gate_record = (
        {
            "tail_hard_exceeded": float(tail_score),
            "hard_threshold": hard_threshold,
            "handled_by_robust_estimator": spectrum_estimator,
            # Tenth review: only the spectrum/rank is robustified. Every
            # other fingerprint-derived choice still comes from the
            # extreme-tailed samples and is NOT resolved by the robust
            # estimator -- the prescription's width is rank-derived and
            # admissible; these components require tail validation (or
            # manual confirmation) before the plan is treated as complete.
            "prescription_scope": "width_only_rank_derived",
            "non_spectrum_statistics_unresolved": [
                "support (k95)",
                "sign statistics",
                "interaction structure",
                "mechanism scores",
                "densities",
            ],
        }
        if tail_hard_exceeded
        else None
    )
    return {
        **({"hard_tail_gate": hard_gate_record} if hard_gate_record else {}),
        "sample_tail_score": tail_score,
        "tail_validation_threshold": threshold,
        "requires_tail_validation": requires_tail_validation,
        "status": (
            "tail_risk_detected"
            if tail_score is not None and requires_tail_validation
            else (
                "tail_profile_missing"
                if tail_score is None
                else "tail_profile_below_threshold"
            )
        ),
        "required_measurements": [
            "disjoint examples not used for architecture or checkpoint selection",
            "multiple context lengths for sequence models",
            "per-window median, p95, maximum, and mean loss",
            "tail-stratified local teacher-function error",
        ],
        "selection_policy": (
            "Teacher tails do not silently enlarge the cell. They trigger a "
            "registered density/width/recovery control and a tail-quality gate."
        ),
    }


def _one_axis_confirmation_controls(
    resolved: Mapping[str, Any],
    *,
    hidden_size: int,
    teacher_intermediate_size: int,
) -> list[dict[str, Any]]:
    """Return compact falsification controls around the FMI best guess.

    These are not claimed to be posterior samples. They are one-axis changes
    that identify which teacher-conditioned choice produced any measured gain
    without exploding into an opaque Cartesian sweep.
    """

    density = float(resolved["density"])
    density_controls = sorted(
        {max(1.0 / hidden_size, min(1.0, density * factor)) for factor in (0.5, 2.0)}
        - {density}
    )

    def density_controls_for(axis: str) -> list[float]:
        value = float(resolved[axis])
        return sorted(
            {max(1.0 / hidden_size, min(1.0, value * factor)) for factor in (0.5, 2.0)}
            - {value}
        )

    pathway_density_controls = {
        axis: density_controls_for(axis)
        for axis in (
            "input_to_excitatory_density",
            "input_to_inhibitory_density",
            "inhibitory_to_excitatory_density",
        )
    }
    width = int(resolved["population_width"])
    width_controls = sorted(
        {
            max(1, min(teacher_intermediate_size, round(width * factor)))
            for factor in (0.75, 1.25)
        }
        - {width}
    )
    branches = [int(value) for value in resolved["branch_factors"]]
    e_branches = [int(value) for value in resolved["excitatory_branch_factors"]]
    i_branches = [int(value) for value in resolved["inhibitory_branch_factors"]]
    integration = str(resolved["integration_rule"])
    if integration == "shunting" and not bool(resolved["biological_neuron"]):
        # R12 empirical safety default (2026-08-25 review): automatic
        # selection must never emit signed-weight shunting — all three
        # measured signed shunting arms diverged. The shunting evidence is
        # preserved in the journal; the executable choice falls back to the
        # additive control. Deliberate research probes go through the
        # compiler's allow_measured_unstable override, never through FMI.
        integration = "raw_additive"
        resolved["integration_rule"] = "raw_additive"
    integration_controls = [
        value
        for value in ("raw_additive", "conductance_normalized", "shunting")
        if value != integration
        and (
            value != "shunting"
            or (bool(resolved["explicit_ei"]) and bool(resolved["biological_neuron"]))
        )
    ]

    def soma_controls_for(axis: str) -> list[int]:
        value = resolved[axis]
        if value is None:
            return []
        count = int(value)
        alternative = 0 if count > 0 else 1
        return [alternative]

    controls = [
        {
            "axis": "density",
            "values": density_controls,
            "reason": "support-budget sensitivity around the inferred K",
        },
        {
            "axis": "population_width",
            "values": width_controls,
            "reason": "rank truncation sensitivity around the inferred soma count",
        },
        *[
            {
                "axis": axis,
                "values": values,
                "reason": (
                    "pathway-specific support sensitivity at fixed morphology; "
                    "compare using complete deployed bytes"
                ),
            }
            for axis, values in pathway_density_controls.items()
        ],
        {
            "axis": "branch_factors",
            "values": [[]] if branches else [[2]],
            "reason": "flat-versus-branched falsification at matched pathway budget",
        },
        {
            "axis": "somatic_synapses",
            "values": (
                [not bool(resolved["somatic_synapses"])]
                if e_branches or i_branches
                else []
            ),
            "reason": "direct-bypass ablation for the affine main-effect proxy",
        },
        {
            "axis": "affine_bypass_mode",
            "values": (
                ["none"]
                if str(resolved["affine_bypass_mode"]) != "none"
                else ["sparse"]
            ),
            "reason": (
                "explicit affine-main-effect path ablation, distinct from "
                "within-cell somatic contacts"
            ),
        },
        {
            "axis": "affine_bypass_density",
            "values": (
                density_controls_for("affine_bypass_density")
                if str(resolved["affine_bypass_mode"]) == "sparse"
                else []
            ),
            "reason": "affine-path coordinate-support sensitivity",
        },
        {
            "axis": "affine_bypass_rank",
            "values": (
                sorted(
                    {
                        max(1, round(int(resolved["affine_bypass_rank"]) * factor))
                        for factor in (0.5, 1.5)
                    }
                    - {int(resolved["affine_bypass_rank"])}
                )
                if str(resolved["affine_bypass_mode"]) == "low_rank"
                else []
            ),
            "reason": "affine-main-effect rank sensitivity",
        },
        {
            "axis": "somatic_excitatory_synapses",
            "values": soma_controls_for("somatic_excitatory_synapses"),
            "reason": "numeric positive-path soma-bypass contact ablation",
        },
        {
            "axis": "somatic_inhibitory_synapses",
            "values": soma_controls_for("somatic_inhibitory_synapses"),
            "reason": "numeric negative-path soma-bypass contact ablation",
        },
        {
            "axis": "inhibitory_population_somatic_synapses",
            "values": soma_controls_for("inhibitory_population_somatic_synapses"),
            "reason": "numeric input-to-I soma-bypass contact ablation",
        },
        {
            "axis": "excitatory_branch_factors",
            "values": [[]] if e_branches else [[2]],
            "reason": "E-tree ablation independent of the I morphology",
        },
        {
            "axis": "inhibitory_branch_factors",
            "values": [[]] if i_branches else [[2]],
            "reason": "I-tree ablation independent of the E morphology",
        },
        {
            "axis": "integration_rule",
            "values": integration_controls,
            "reason": "matched additive/normalized/shunting mechanism controls",
        },
        {
            "axis": "gated",
            "values": [not bool(resolved["gated"])],
            "reason": "matched single-path/product-path control",
        },
        {
            "axis": "topology_mode",
            "values": [
                mode
                for mode in TOPOLOGY_MODES
                if mode != str(resolved["topology_mode"])
            ],
            "reason": (
                "selection/training-rule comparison at the same compiled final "
                "contact budgets"
            ),
        },
        {
            "axis": "output_topology_mode",
            "values": (
                [
                    mode
                    for mode in ("standard", "indexed_rewire")
                    if mode != str(resolved["output_topology_mode"])
                ]
                if str(resolved["output_projection_mode"]) == "sparse"
                else []
            ),
            "reason": (
                "readout-selection control independent of the dendritic core "
                "selection method"
            ),
        },
        {
            "axis": "output_density",
            "values": (
                density_controls_for("output_density")
                if str(resolved["output_projection_mode"]) == "sparse"
                else []
            ),
            "reason": "readout sparsity sensitivity in the learned soma basis",
        },
        {
            "axis": "output_rank",
            "values": (
                sorted(
                    {
                        max(1, round(int(resolved["output_rank"]) * factor))
                        for factor in (0.5, 1.5)
                    }
                    - {int(resolved["output_rank"])}
                )
                if str(resolved["output_projection_mode"]) == "low_rank"
                else []
            ),
            "reason": "task-weighted readout-rank sensitivity",
        },
        {
            "axis": "output_projection_mode",
            "values": [
                (
                    "low_rank"
                    if str(resolved["output_projection_mode"]) == "sparse"
                    else "sparse"
                )
            ],
            "reason": (
                "sparse-versus-low-rank readout control using the separately "
                "registered density and rank budgets"
            ),
        },
    ]
    inactive_without_ei = {
        "input_to_inhibitory_density",
        "inhibitory_to_excitatory_density",
        "inhibitory_branch_factors",
        "somatic_inhibitory_synapses",
        "inhibitory_population_somatic_synapses",
    }
    inactive_without_branches = {
        "somatic_synapses",
        "somatic_excitatory_synapses",
        "somatic_inhibitory_synapses",
        "inhibitory_population_somatic_synapses",
    }
    return [
        control
        for control in controls
        if control["values"]
        and (
            bool(resolved["explicit_ei"]) or control["axis"] not in inactive_without_ei
        )
        and (
            e_branches or i_branches or control["axis"] not in inactive_without_branches
        )
    ]


def resolve_replacement_selection(
    selection_config: Any,
    *,
    hidden_size: int,
    teacher_intermediate_size: int,
    layer_index: int | None = None,
    module_path: str | None = None,
    boundary_output_dim: int | None = None,
) -> ResolvedReplacementSelection:
    """Resolve an optional FMI/manual hybrid into a compiled EI replacement."""

    cfg = _plain(selection_config)
    if not bool(cfg.get("enabled", False)):
        raise ValueError("selection.enabled must be true when resolving FMI selection")
    mode = str(cfg.get("mode", "hybrid")).strip().lower()
    if mode not in {"manual", "fmi", "hybrid"}:
        raise ValueError("selection.mode must be 'manual', 'fmi', or 'hybrid'")
    rule_set = str(cfg.get("rule_set", "theory_v2"))
    if rule_set not in {"theory_v2", "theory_v3", "theory_v4"}:
        raise ValueError(
            "The runtime architecture compiler currently supports rule_set="
            "'theory_v2', 'theory_v3', or 'theory_v4'. Frozen v1 artifacts may be consumed "
            "as fingerprints, but their files and rules remain immutable."
        )
    manual = _plain(cfg.get("manual", {}))
    unknown_manual = sorted(set(manual) - set(ALL_SELECTION_AXES))
    if unknown_manual:
        # A dropped manual key silently reverts that axis to its default and
        # was observed producing signed_ei_flat cells from a request that
        # named a completely different family. Same failure class as the
        # silently-dropped training keys: fail loudly instead.
        raise ValueError(
            f"selection.manual has unknown axes {unknown_manual}; valid axes "
            f"are {sorted(ALL_SELECTION_AXES)}. (There is no 'family' key: "
            "families are combinations of explicit_ei/gated/"
            "biological_neuron/integration_rule.)"
        )
    thresholds = _plain(cfg.get("thresholds", {}))
    axes_cfg = _plain(cfg.get("axes", {}))
    allow_non_biological = bool(cfg.get("allow_non_biological", False))
    unknown_axes = sorted(set(axes_cfg) - set(ALL_SELECTION_AXES))
    if unknown_axes:
        raise ValueError(f"Unknown selection axes: {unknown_axes}")
    if not allow_non_biological and manual.get("biological_neuron") is False:
        raise ValueError(
            "biological_neuron=false requires selection.allow_non_biological=true"
        )
    if (
        not allow_non_biological
        and str(axes_cfg.get("biological_neuron", "manual")).lower() == "fmi"
    ):
        raise ValueError(
            "FMI may select biological_neuron only when "
            "selection.allow_non_biological=true; otherwise biological "
            "constraints are enforced"
        )

    fingerprint: dict[str, Any] = {}
    prior_decision: dict[str, Any] = {}
    provenance = {"path": None, "key": None, "protocol": None, "sha256": None}
    if mode != "manual" or any(source == "fmi" for source in axes_cfg.values()):
        fingerprint, prior_decision, provenance = load_fmi_fingerprint(
            cfg.get("fingerprint", {}), layer_index=layer_index
        )
    boundary = dict(provenance.get("boundary", {}))
    for key in ("input_dim", "output_dim"):
        if key not in boundary and key in fingerprint:
            boundary[key] = fingerprint[key]
    boundary_assessment = validate_profile_boundary(
        boundary,
        module_path=module_path,
        input_dim=int(hidden_size),
        output_dim=boundary_output_dim,
    )
    rank_assessment = _rank_assessment(fingerprint)
    if bool(cfg.get("require_validated_rank", False)):
        raise ValueError(
            "Generic FMI selection supplies advisory ranks only; use the boundary-matched "
            "rank-validation pipeline before requesting an evidence-backed population rank"
        )
    predictions, explanations = _fmi_predictions(
        fingerprint,
        prior_decision,
        hidden_size=int(hidden_size),
        teacher_intermediate_size=int(teacher_intermediate_size),
        thresholds=thresholds,
        rule_set=rule_set,
    )

    fallback = {
        "biological_neuron": True,
        "density": 0.25,
        "input_to_excitatory_density": 0.25,
        "input_to_inhibitory_density": 0.25,
        "inhibitory_to_excitatory_density": 0.25,
        "population_width": int(teacher_intermediate_size),
        "inhibitory_fraction": 0.25,
        "explicit_ei": True,
        "branch_factors": [2, 2],
        "excitatory_branch_factors": [2, 2],
        "inhibitory_branch_factors": [2, 2],
        "somatic_synapses": False,
        "somatic_excitatory_synapses": None,
        "somatic_inhibitory_synapses": None,
        "inhibitory_population_somatic_synapses": None,
        "affine_bypass_mode": "none",
        "affine_bypass_density": 0.25,
        "affine_bypass_rank": max(1, min(hidden_size, teacher_intermediate_size)),
        "integration_rule": "raw_additive",
        "gated": False,
        "output_density": 0.25,
        "output_rank": max(1, min(hidden_size, teacher_intermediate_size)),
        "teacher_support_metric": "activation_weighted",
        "topology_mode": "indexed_rewire",
        "output_topology_mode": "indexed_rewire",
        "reactivation_type": "param_tanh",
        "gate_activation": "silu",
        "input_transform": "identity",
        "output_projection_mode": "sparse",
        "affine_bypass_topology_mode": "indexed_rewire",
    }
    resolved: dict[str, Any] = {}
    journal: dict[str, dict[str, Any]] = {}
    for axis in ALL_SELECTION_AXES:
        default_source = (
            "manual" if axis in MANUAL_ONLY_AXES or mode == "manual" else "fmi"
        )
        if axis == "biological_neuron":
            if not allow_non_biological:
                default_source = "manual"
            elif axis in manual:
                default_source = "manual"
            elif mode != "manual":
                default_source = "fmi"
        source = str(axes_cfg.get(axis, default_source)).strip().lower()
        if source not in {"manual", "fmi"}:
            raise ValueError(f"selection.axes.{axis} must be 'manual' or 'fmi'")
        if axis in MANUAL_ONLY_AXES and source == "fmi":
            raise ValueError(
                f"selection axis {axis!r} is manual-only; FMI does not infer "
                "experimental constraints or runtime policies"
            )
        if source == "manual":
            if axis in manual:
                value = manual[axis]
                reason = "explicit manual configuration"
                realized_source = "manual"
            else:
                value = fallback[axis]
                reason = "declared default because no manual value was supplied"
                realized_source = "fallback"
        elif axis in predictions:
            value = predictions[axis]
            reason = explanations[axis]
            realized_source = "fmi"
        elif axis in manual:
            value = manual[axis]
            reason = "manual fallback: fingerprint lacks the required estimator"
            realized_source = "manual_fallback"
        else:
            value = fallback[axis]
            reason = "declared fallback: fingerprint lacks the required estimator"
            realized_source = "fallback"
        resolved[axis] = value
        journal[axis] = {
            "requested_source": source,
            "realized_source": realized_source,
            "value": value,
            "reason": reason,
        }

    # A shared tree remains a supported backward-compatible axis. Specific
    # polarity trees override it only when they were explicitly requested or
    # measured in the fingerprint.
    for axis, fingerprint_key in (
        ("excitatory_branch_factors", "excitatory_interaction_branch_factors"),
        ("inhibitory_branch_factors", "inhibitory_interaction_branch_factors"),
    ):
        if (
            axis not in axes_cfg
            and axis not in manual
            and fingerprint_key not in fingerprint
        ):
            resolved[axis] = list(resolved["branch_factors"])
            journal[axis].update(
                {
                    "realized_source": "shared_branch_alias",
                    "value": list(resolved["branch_factors"]),
                    "reason": (
                        "no polarity-specific interaction tree was registered; "
                        "using the shared branch_factors axis"
                    ),
                }
            )

    # A top-level density request must reach every pathway density the caller
    # did not explicitly pin. Historically these axes silently kept the
    # declared 0.25 default even when a different top-level density was
    # requested, so the request was label-only: the compiled cell stayed at
    # 0.25 regardless. Restore the compiler's own contract (pathway densities
    # default to ``density`` when omitted) and record the decision in the
    # journal. A request of exactly the declared default writes the value the
    # axis already holds, so those selections stay bit-for-bit unchanged.
    def _densities_close(left: float, right: float) -> bool:
        # Tolerance covers float-representation noise only (YAML/JSON round
        # trips), never scientifically distinct densities.
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12)

    top_level_density = float(resolved["density"])
    density_pinned = journal["density"]["realized_source"] == "manual"
    density_is_declared_default = _densities_close(
        top_level_density, float(fallback["density"])
    )
    density_propagation_axes = [
        "input_to_excitatory_density",
        "input_to_inhibitory_density",
        "inhibitory_to_excitatory_density",
        "output_density",
    ]
    if str(resolved["affine_bypass_mode"]) != "none":
        # The affine bypass density obeys the same compiler fallback contract
        # but only participates when the bypass path exists; gating on the
        # mode keeps manifests for bypass-free configs untouched.
        density_propagation_axes.append("affine_bypass_density")
    for axis in density_propagation_axes:
        entry = journal[axis]
        if entry["realized_source"] == "fallback":
            if _densities_close(float(resolved[axis]), top_level_density):
                continue
            resolved[axis] = top_level_density
            entry.update(
                {
                    "realized_source": "density_propagation",
                    "value": top_level_density,
                    "reason": (
                        "propagated from the top-level density axis because "
                        "no pathway-specific value was pinned; mirrors the "
                        "compiler contract that pathway densities default to "
                        "the global density"
                    ),
                }
            )
        elif (
            axis != "affine_bypass_density"
            and density_pinned
            and not density_is_declared_default
            and entry["realized_source"] == "manual"
            and not _densities_close(float(resolved[axis]), top_level_density)
        ):
            # Pin-overrides-density is the documented historical contract
            # (the v3 policy pins capacity-audited per-axis floors that
            # legitimately diverge from the top-level density), so an
            # explicit pin always wins. The disagreement is journaled so it
            # can never again pass silently — that was the original failure
            # mode, and visibility (not refusal) is the correct guard. The
            # affine bypass is exempt because its support is measured in an
            # independently profiled reference space.
            entry["density_conflict"] = {
                "pinned_value": float(resolved[axis]),
                "requested_top_level_density": top_level_density,
                "resolution": "explicit_pathway_pin_overrides_top_level_density",
            }

    # Biological constraints are the default and are enforced unless the
    # experiment explicitly permits relaxed engineering candidates.  With
    # that opt-in, FMI or a manual axis may select either constraint class.
    if not allow_non_biological:
        resolved["biological_neuron"] = True
        journal["biological_neuron"].update(
            {
                "realized_source": "constraint",
                "value": True,
                "reason": (
                    "biological constraints are enforced because "
                    "allow_non_biological is false"
                ),
            }
        )

    # A strict cell cannot consume signed values directly. An omitted input
    # transform is promoted visibly; an explicit manual contradiction fails.
    if bool(resolved["biological_neuron"]) and str(resolved["input_transform"]) not in {
        "relu",
        "signed_split",
    }:
        if journal["input_transform"]["realized_source"] == "manual":
            raise ValueError(
                "biological_neuron=true requires input_transform relu or signed_split"
            )
        resolved["input_transform"] = "signed_split"
        journal["input_transform"].update(
            {
                "realized_source": "constraint",
                "value": "signed_split",
                "reason": "strict biological execution requires nonnegative inputs",
            }
        )

    # Reconcile hard executable constraints after independent evidence is
    # recorded. A manual contradiction fails; an FMI/default proposal may be
    # promoted with the constraint and the promotion remains visible.
    requires_explicit_ei = (
        bool(resolved["biological_neuron"])
        or str(resolved["integration_rule"]) == "shunting"
    )
    if requires_explicit_ei and not bool(resolved["explicit_ei"]):
        if journal["explicit_ei"]["realized_source"] == "manual":
            raise ValueError(
                "explicit_ei=false conflicts with biological_neuron=true or "
                "shunting integration"
            )
        resolved["explicit_ei"] = True
        journal["explicit_ei"].update(
            {
                "realized_source": "constraint",
                "value": True,
                "reason": (
                    "promoted because strict biological execution and "
                    "shunting integration require an explicit inhibitory "
                    "population; the original FMI evidence remains in the "
                    "fingerprint"
                ),
            }
        )

    # Teacher-side bypass support is measured in the input coordinate system,
    # whereas the compiled inhibitory soma pathway reads from the inferred I
    # population. A low-rank teacher can therefore imply more negative bypass
    # coordinates than the selected I population physically contains. Clamp
    # inferred/default budgets to the executable source spaces and record the
    # constraint; retain compiler errors for contradictory manual budgets.
    adapted_input_dim = (
        2 * int(hidden_size)
        if str(resolved["input_transform"]) == "signed_split"
        else int(hidden_size)
    )
    inhibitory_width = (
        max(
            1,
            round(
                int(resolved["population_width"])
                * float(resolved["inhibitory_fraction"])
            ),
        )
        if bool(resolved["explicit_ei"])
        else 0
    )
    soma_limits = {
        "somatic_excitatory_synapses": adapted_input_dim,
        "somatic_inhibitory_synapses": inhibitory_width,
        "inhibitory_population_somatic_synapses": (
            adapted_input_dim if bool(resolved["explicit_ei"]) else 0
        ),
    }
    for axis, limit in soma_limits.items():
        value = resolved[axis]
        if value is None or int(value) <= limit:
            continue
        if journal[axis]["realized_source"] == "manual":
            raise ValueError(
                f"manual {axis}={int(value)} exceeds its compiled source "
                f"dimension {limit}"
            )
        resolved[axis] = int(limit)
        journal[axis].update(
            {
                "realized_source": "constraint",
                "value": int(limit),
                "reason": (
                    f"inferred bypass budget {int(value)} was clamped to its "
                    f"compiled source dimension {limit}"
                ),
            }
        )

    branches = tuple(int(value) for value in resolved["branch_factors"])
    excitatory_branches = tuple(
        int(value) for value in resolved["excitatory_branch_factors"]
    )
    inhibitory_branches = tuple(
        int(value) for value in resolved["inhibitory_branch_factors"]
    )
    branched = bool(excitatory_branches or inhibitory_branches)
    if not branched:
        if journal["somatic_synapses"]["realized_source"] == "manual" and not bool(
            resolved["somatic_synapses"]
        ):
            raise ValueError(
                "flat populations have no distal compartment and require "
                "somatic_synapses=true"
            )
        resolved["somatic_synapses"] = True
        journal["somatic_synapses"].update(
            {
                "realized_source": "constraint",
                "value": True,
                "reason": (
                    "the measured flat morphology has no compartment distinct "
                    "from the soma"
                ),
            }
        )
        flat_soma_contacts = {
            "somatic_excitatory_synapses": max(
                1,
                round(
                    float(resolved["input_to_excitatory_density"]) * int(hidden_size)
                ),
            ),
            "somatic_inhibitory_synapses": (
                max(
                    1,
                    round(
                        float(resolved["inhibitory_to_excitatory_density"])
                        * inhibitory_width
                    ),
                )
                if bool(resolved["explicit_ei"])
                else 0
            ),
            "inhibitory_population_somatic_synapses": (
                max(
                    1,
                    round(
                        float(resolved["input_to_inhibitory_density"])
                        * int(hidden_size)
                    ),
                )
                if bool(resolved["explicit_ei"])
                else 0
            ),
        }
        for axis, expected in flat_soma_contacts.items():
            if (
                journal[axis]["realized_source"] == "manual"
                and resolved[axis] is not None
                and int(resolved[axis]) != expected
            ):
                raise ValueError(
                    f"manual {axis}={int(resolved[axis])} conflicts with the "
                    f"flat pathway contact budget {expected}"
                )
            resolved[axis] = int(expected)
            journal[axis].update(
                {
                    "realized_source": "constraint",
                    "value": int(expected),
                    "reason": (
                        "flat populations place the complete pathway contact "
                        "budget at the soma"
                    ),
                }
            )
    integration = str(resolved["integration_rule"])
    if integration == "shunting" and not bool(resolved["biological_neuron"]):
        # R12 empirical safety default (2026-08-25 review, second pass):
        # signed-weight shunting is measured-unstable (all three arms
        # diverged). A MANUAL scientific request must not be silently
        # rewritten — it raises and directs the caller to the compiler's
        # labeled unsafe route. Automatic FMI selection falls back to the
        # additive control with an explicit constraint record so the
        # journal never contradicts the compiled plan.
        entry = journal.get("integration_rule")
        source = (
            str(entry.get("realized_source", "")) if isinstance(entry, dict) else ""
        )
        if source == "manual":
            raise ValueError(
                "manual selection requested shunting integration on a "
                "signed-weight domain, a measured-unstable pairing (strategy "
                "R12). Refusing to silently rewrite a scientific request: "
                "either choose a biological (positive_ei) domain, request "
                "raw_additive, or compile the probe explicitly with "
                "allow_measured_unstable=True outside FMI selection."
            )
        integration = "raw_additive"
        resolved["integration_rule"] = "raw_additive"
        constraint = {
            "requested_value": "shunting",
            "realized_value": "raw_additive",
            "realized_source": "constraint",
            "constraint": (
                "R12: signed-weight shunting is measured-unstable; automatic "
                "selection falls back to the additive control"
            ),
        }
        if isinstance(entry, dict):
            entry.update(constraint)
        else:
            journal["integration_rule"] = constraint
    family = _family_name(
        biological=bool(resolved["biological_neuron"]),
        explicit_ei=bool(resolved["explicit_ei"]),
        branched=branched,
        gated=bool(resolved["gated"]),
        shunting=integration == "shunting",
    )
    compile_kwargs = _plain(cfg.get("compiler", {}))
    scientific_keys = {
        "hidden_size",
        "teacher_intermediate_size",
        "density",
        "input_to_excitatory_density",
        "input_to_inhibitory_density",
        "inhibitory_to_excitatory_density",
        "population_width",
        "inhibitory_fraction",
        "branch_factors",
        "excitatory_branch_factors",
        "inhibitory_branch_factors",
        "somatic_synapses",
        "somatic_excitatory_synapses",
        "somatic_inhibitory_synapses",
        "inhibitory_population_somatic_synapses",
        "output_density",
        "output_rank",
        "output_projection_mode",
        "topology_mode",
        "output_topology_mode",
        "reactivation_type",
        "gate_activation",
        "integration_rule",
        "additive_tangent_n0",
        "additive_tangent_t0",
        "input_transform",
        "affine_bypass_mode",
        "affine_bypass_density",
        "affine_bypass_rank",
        "affine_bypass_topology_mode",
    }
    conflicts = sorted(scientific_keys & set(compile_kwargs))
    if conflicts:
        raise ValueError(
            "selection.compiler cannot override resolved scientific axes; set "
            f"their source/value under selection.axes/manual instead: {conflicts}"
        )
    compiler_options: dict[str, Any] = {
        "hidden_size": int(hidden_size),
        "teacher_intermediate_size": int(teacher_intermediate_size),
        "density": float(resolved["density"]),
        "input_to_excitatory_density": float(resolved["input_to_excitatory_density"]),
        "input_to_inhibitory_density": float(resolved["input_to_inhibitory_density"]),
        "inhibitory_to_excitatory_density": float(
            resolved["inhibitory_to_excitatory_density"]
        ),
        "population_width": int(resolved["population_width"]),
        "inhibitory_fraction": float(resolved["inhibitory_fraction"]),
        "branch_factors": branches,
        "excitatory_branch_factors": excitatory_branches,
        "inhibitory_branch_factors": inhibitory_branches,
        "somatic_synapses": bool(resolved["somatic_synapses"]),
        "somatic_excitatory_synapses": resolved["somatic_excitatory_synapses"],
        "somatic_inhibitory_synapses": resolved["somatic_inhibitory_synapses"],
        "inhibitory_population_somatic_synapses": resolved[
            "inhibitory_population_somatic_synapses"
        ],
        "output_density": float(resolved["output_density"]),
        "output_rank": (
            int(resolved["output_rank"])
            if str(resolved["output_projection_mode"]) == "low_rank"
            else None
        ),
        "output_projection_mode": str(resolved["output_projection_mode"]),
        "topology_mode": str(resolved["topology_mode"]),
        "output_topology_mode": str(resolved["output_topology_mode"]),
        "reactivation_type": str(resolved["reactivation_type"]),
        "gate_activation": str(resolved["gate_activation"]),
        "integration_rule": integration,
        "additive_tangent_n0": fingerprint.get("additive_tangent_n0"),
        "additive_tangent_t0": fingerprint.get("additive_tangent_t0"),
        "input_transform": str(resolved["input_transform"]),
        "teacher_support_metric": str(resolved["teacher_support_metric"]),
        "affine_bypass_mode": str(resolved["affine_bypass_mode"]),
        "affine_bypass_density": float(resolved["affine_bypass_density"]),
        "affine_bypass_rank": (
            int(resolved["affine_bypass_rank"])
            if str(resolved["affine_bypass_mode"]) == "low_rank"
            else None
        ),
        "affine_bypass_topology_mode": str(resolved["affine_bypass_topology_mode"]),
    }
    compiler_options.update(compile_kwargs)
    plan = compile_replacement_candidate(family, **compiler_options)
    manifest = {
        "schema": "dendritic_replacement_selection/v3",
        "rule_set": rule_set,
        "status": "prospective_hypothesis_requires_reference_grid",
        "mode": mode,
        "allow_non_biological": allow_non_biological,
        "layer_index": layer_index,
        "fingerprint_provenance": provenance,
        "boundary_assessment": boundary_assessment,
        "rank_assessment": rank_assessment,
        "fingerprint": fingerprint,
        "prior_decision": prior_decision,
        "axes": journal,
        "compiled_family": plan.family,
        "teacher_support_metric": resolved["teacher_support_metric"],
        "execution": {
            "architecture": "compiled_and_executable",
            "teacher_support_metric": (
                "requested_for_calibration; execution evidence is recorded "
                "only after the teacher-conditioned initializer runs"
            ),
        },
        "robustness_assessment": _robustness_assessment(
            fingerprint,
            thresholds=thresholds,
        ),
        "recommended_confirmation": {
            "design": "one_axis_at_a_time_around_primary",
            "controls": _one_axis_confirmation_controls(
                resolved,
                hidden_size=int(hidden_size),
                teacher_intermediate_size=int(teacher_intermediate_size),
            ),
            "note": (
                "controls are falsification experiments, not a calibrated "
                "posterior or evidence that the primary is optimal"
            ),
        },
    }
    return ResolvedReplacementSelection(plan=plan, manifest=manifest)


__all__ = [
    "ALL_SELECTION_AXES",
    "FMI_SELECTABLE_AXES",
    "MANUAL_ONLY_AXES",
    "ResolvedReplacementSelection",
    "load_fmi_fingerprint",
    "resolve_replacement_selection",
    "validate_profile_boundary",
]
