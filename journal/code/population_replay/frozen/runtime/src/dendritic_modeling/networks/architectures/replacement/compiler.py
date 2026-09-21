"""Compile theory or sweep choices into executable replacement configurations.

This module belongs to the reusable replacement architecture layer rather than
to a particular analysis method. FMI, a preregistered sweep, or a manually
specified candidate may all call the same compiler. It emits ordinary package
configuration payloads that can be trained by the standard transformer or
vision replacement paths and compacted by the deployment exporter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

CANONICAL_FAMILIES = (
    "signed_flat",
    "signed_branched",
    "signed_ei_flat",
    "signed_ei_branched_additive",
    "signed_ei_branched_shunting",
    "positive_ei_flat",
    "positive_ei_branched_additive",
    "positive_ei_branched_shunting",
    "gated_signed_flat",
    "gated_signed_branched",
    "gated_signed_ei_flat",
    "gated_signed_ei_branched_additive",
    "gated_signed_ei_branched_shunting",
    "gated_positive_ei_flat",
    "gated_positive_ei_branched_additive",
    "gated_positive_ei_branched_shunting",
)

LEGACY_FAMILY_ALIASES = {
    "gated_sparse": "gated_signed_flat",
    "ungated_sparse": "signed_flat",
    "hier_gated": "gated_signed_branched",
    "shunting_ei": "positive_ei_branched_shunting",
    "wide_narrow": "gated_signed_flat",
    "rank_readout": "gated_signed_flat",
}

TOPOLOGY_MODES = (
    "standard",
    "stochastic",
    "variance",
    "indexed",
    "indexed_rewire",
    "indexed_dynamic",
    "dense_to_sparse",
)
INTEGRATION_RULES = (
    "raw_additive",
    "conductance_normalized",
    "tangent_matched",
    "shunting",
)


@dataclass(frozen=True)
class CompiledReplacementPlan:
    """Serializable bridge from a theory/sweep decision to training config."""

    requested_candidate: str
    family: str
    density: float
    hidden_size: int
    population_width: int
    inhibitory_width: int
    branch_factors: tuple[int, ...]
    excitatory_branch_factors: tuple[int, ...]
    inhibitory_branch_factors: tuple[int, ...]
    somatic_synapses: bool
    synapses_per_branch: int
    input_to_excitatory_density: float
    input_to_inhibitory_density: float
    inhibitory_to_excitatory_density: float
    input_to_excitatory_contacts: int
    input_to_inhibitory_contacts: int
    inhibitory_to_excitatory_contacts: int
    somatic_excitatory_synapses: int
    somatic_inhibitory_synapses: int
    inhibitory_population_somatic_synapses: int
    output_density: float
    output_rank: int | None
    output_projection_mode: str
    output_topk: int | None
    topology_mode: str
    integration_rule: str
    biological_neuron: bool
    input_transform: str
    reactivation_type: str
    gate_activation: str | None
    teacher_support_metric: str | None
    replacement_kind: str
    core_config: dict[str, Any]
    replacement_kwargs: dict[str, Any]
    training_overrides: dict[str, Any]
    export_contract: dict[str, Any]
    #: populated ONLY when an unsafe research override authorized a
    #: measured-unstable pairing (e.g. signed shunting under R12); absent
    #: (None) for every normal compile so existing frozen plans stay valid.
    safety: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compiled_replacement_plan_from_mapping(
    payload: Mapping[str, Any],
) -> CompiledReplacementPlan:
    """Strictly reconstruct one versioned compiler output from configuration."""

    from dataclasses import MISSING

    fields = set(CompiledReplacementPlan.__dataclass_fields__)
    required = {
        name
        for name, spec in CompiledReplacementPlan.__dataclass_fields__.items()
        if spec.default is MISSING and spec.default_factory is MISSING
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - fields)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unknown:
            details.append(f"unknown={unknown}")
        raise ValueError("Invalid compiled replacement plan: " + ", ".join(details))
    normalized = deepcopy(dict(payload))
    for key in (
        "branch_factors",
        "excitatory_branch_factors",
        "inhibitory_branch_factors",
    ):
        normalized[key] = tuple(int(value) for value in normalized[key])
    plan = CompiledReplacementPlan(**normalized)
    schema = str(plan.export_contract.get("schema", ""))
    if schema != "dendritic_replacement_export_contract/v2":
        raise ValueError(
            "Compiled replacement plan must carry the current v2 export contract"
        )
    if plan.hidden_size < 1 or plan.population_width < 1:
        raise ValueError("Compiled replacement dimensions must be positive")
    return plan


def _validate_density(value: float, *, name: str) -> float:
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return resolved


def _validate_contacts(
    value: int | None,
    *,
    fallback_density: float,
    reference_dim: int,
    source_dim: int,
    name: str,
    require_nonzero: bool,
) -> int:
    """Resolve an absolute contact budget without hiding its reference space."""

    minimum = 1 if require_nonzero else 0
    contacts = (
        max(minimum, round(float(fallback_density) * int(reference_dim)))
        if value is None
        else int(value)
    )
    if contacts < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {contacts}")
    if contacts > int(source_dim):
        raise ValueError(f"{name}={contacts} exceeds its source dimension {source_dim}")
    return contacts


def _distal_and_somatic_layout(
    *,
    distal_contacts_per_soma: int,
    branch_factors: tuple[int, ...],
    source_dim: int,
    somatic_contacts: int,
    pathway: str,
) -> tuple[list[int], dict[str, Any]]:
    """Place a declared dendritic budget at leaves and a bypass at the soma.

    PopulationNetwork level lists are distal-to-soma. Intermediate levels
    perform child aggregation but receive no direct feedforward contacts in
    this matched-budget compiler. This makes the declared pathway budget
    independent of depth. Integer K requires rounding up across equally sized
    leaves; the exact realized total and rounding overhead are returned.
    """

    if source_dim < 1:
        raise ValueError(f"{pathway} source_dim must be positive")
    if somatic_contacts < 0 or somatic_contacts > source_dim:
        raise ValueError(
            f"{pathway} somatic contacts must be in [0, {source_dim}], got "
            f"{somatic_contacts}"
        )
    if not branch_factors:
        flat_contacts = somatic_contacts or distal_contacts_per_soma
        if flat_contacts < 1:
            raise ValueError(f"flat {pathway} requires at least one soma contact")
        if flat_contacts > source_dim:
            raise ValueError(
                f"flat {pathway} contacts={flat_contacts} exceed source dimension "
                f"{source_dim}"
            )
        return [int(flat_contacts)], {
            "pathway": pathway,
            "source_dim": int(source_dim),
            "branch_factors": [],
            "leaf_branches_per_soma": 1,
            "declared_distal_contacts_per_soma": int(distal_contacts_per_soma),
            "distal_synapses_per_leaf": 0,
            "realized_distal_contacts_per_soma": 0,
            "somatic_contacts_per_soma": int(flat_contacts),
            "realized_total_contacts_per_soma": int(flat_contacts),
            "integer_rounding_overhead_per_soma": 0,
            "synapses_by_level_distal_to_soma": [int(flat_contacts)],
        }

    leaf_branches = math.prod(branch_factors)
    distal_k = (
        0
        if distal_contacts_per_soma == 0
        else math.ceil(distal_contacts_per_soma / leaf_branches)
    )
    if distal_k > source_dim:
        raise ValueError(
            f"{pathway} requires {distal_k} contacts per leaf, exceeding source "
            f"dimension {source_dim}"
        )
    realized_distal = int(distal_k * leaf_branches)
    by_level = [
        int(distal_k),
        *([0] * (len(branch_factors) - 1)),
        int(somatic_contacts),
    ]
    return by_level, {
        "pathway": pathway,
        "source_dim": int(source_dim),
        "branch_factors": list(branch_factors),
        "leaf_branches_per_soma": int(leaf_branches),
        "declared_distal_contacts_per_soma": int(distal_contacts_per_soma),
        "distal_synapses_per_leaf": int(distal_k),
        "realized_distal_contacts_per_soma": realized_distal,
        "somatic_contacts_per_soma": int(somatic_contacts),
        "realized_total_contacts_per_soma": int(realized_distal + somatic_contacts),
        "integer_rounding_overhead_per_soma": int(
            realized_distal - distal_contacts_per_soma
        ),
        "synapses_by_level_distal_to_soma": by_level,
    }


def parse_candidate(
    candidate: str, *, density: float | None = None
) -> tuple[str, float]:
    """Parse ``family@d0.125`` while also accepting an explicit density."""

    requested = str(candidate).strip().lower()
    parsed_density = density
    family = requested
    if "@d" in requested:
        family, encoded = requested.rsplit("@d", 1)
        encoded_density = float(encoded)
        if parsed_density is not None and not math.isclose(
            float(parsed_density), encoded_density
        ):
            raise ValueError("candidate density conflicts with explicit density")
        parsed_density = encoded_density
    family = LEGACY_FAMILY_ALIASES.get(family, family)
    if family not in CANONICAL_FAMILIES:
        raise ValueError(
            f"Unknown replacement family {family!r}; expected one of "
            f"{CANONICAL_FAMILIES} or legacy aliases {tuple(LEGACY_FAMILY_ALIASES)}"
        )
    if parsed_density is None:
        raise ValueError("density is required, either explicitly or as '@d<value>'")
    parsed_density = float(parsed_density)
    if not 0.0 < parsed_density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    return family, parsed_density


def _topology_payload(
    *,
    mode: str,
    seed: int,
    pruning_end_step: int,
    initial_density: float,
    rewire_frequency: int,
    rewire_quantile: float,
    rewire_until_step: int,
    projection_backend: str = "auto",
) -> dict[str, Any]:
    if mode not in TOPOLOGY_MODES:
        raise ValueError(f"topology_mode must be one of {TOPOLOGY_MODES}")
    payload: dict[str, Any] = {
        "topk_type": mode,
        "topk_init_method": "xavier_normal",
        "topk_noise_level": 0.0,
        "indexed_seed": int(seed),
        "indexed_selection": "standard",
        "indexed_index_dtype": "auto",
        # "auto" resolves per device: the Triton transposed/fused kernels on a
        # Triton-capable GPU (measured 2.4-21x over the other sparse backends),
        # portable recompute everywhere else. Pinning "recompute" here silently
        # forced every compiled plan onto the slow path even on H100s.
        "indexed_projection_backend": str(projection_backend),
        "indexed_rewire_frequency": int(rewire_frequency),
        "indexed_rewire_quantile": float(rewire_quantile),
        "indexed_rewire_until_step": int(rewire_until_step),
    }
    if mode == "dense_to_sparse":
        payload.update(
            {
                "dense_to_sparse_initial_density": float(initial_density),
                "dense_to_sparse_start_step": 0,
                "dense_to_sparse_end_step": int(pruning_end_step),
                "dense_to_sparse_update_interval": 1,
                "dense_to_sparse_schedule": "cubic",
                "dense_to_sparse_freeze_on_end": True,
                # Transformer trainers advance this once after each optimizer
                # step. Forward-driven schedules can otherwise count
                # calibration or auxiliary passes as pruning steps.
                "dense_to_sparse_advance_on_forward": False,
                "dense_to_sparse_prune_metric": "magnitude",
            }
        )
    return payload


def compile_replacement_candidate(
    candidate: str,
    *,
    hidden_size: int,
    teacher_intermediate_size: int,
    density: float | None = None,
    input_to_excitatory_density: float | None = None,
    input_to_inhibitory_density: float | None = None,
    inhibitory_to_excitatory_density: float | None = None,
    input_to_excitatory_contacts: int | None = None,
    input_to_inhibitory_contacts: int | None = None,
    inhibitory_to_excitatory_contacts: int | None = None,
    population_width: int | None = None,
    inhibitory_fraction: float = 0.25,
    branch_factors: tuple[int, ...] = (2, 2),
    excitatory_branch_factors: tuple[int, ...] | None = None,
    inhibitory_branch_factors: tuple[int, ...] | None = None,
    somatic_synapses: bool | None = None,
    somatic_excitatory_synapses: int | None = None,
    somatic_inhibitory_synapses: int | None = None,
    inhibitory_population_somatic_synapses: int | None = None,
    topology_mode: str = "indexed_rewire",
    projection_backend: str = "auto",
    pruning_end_step: int = 500,
    initial_density: float = 1.0,
    rewire_frequency: int = 25,
    rewire_quantile: float = 0.1,
    rewire_until_step: int | None = None,
    output_rewire_frequency: int | None = None,
    output_rewire_quantile: float | None = None,
    output_rewire_until_step: int | None = None,
    seed: int = 7,
    output_rank: int | None = None,
    output_density: float | None = None,
    output_projection_mode: str = "sparse",
    output_topology_mode: str | None = None,
    reactivation_type: str | None = None,
    gate_activation: str = "silu",
    integration_rule: str | None = None,
    additive_tangent_n0: float | None = None,
    additive_tangent_t0: float | None = None,
    input_transform: str | None = None,
    teacher_support_metric: str | None = None,
    affine_bypass_mode: str = "none",
    affine_bypass_density: float | None = None,
    affine_bypass_rank: int | None = None,
    affine_bypass_topology_mode: str | None = None,
    allow_measured_unstable: bool = False,
) -> CompiledReplacementPlan:
    """Compile one theory- or sweep-selected morphology into package configs.

    ``density`` is a backward-compatible global prior. Pathway-specific
    densities or absolute contact counts may override it for input-to-E,
    input-to-I, and I-to-E paths. Densities are defined relative to each
    pathway's declared reference/source width and the compiled plan records
    both declared and exactly realized counts.

    For branched cells, direct dendritic contacts are placed on distal leaves;
    intermediate compartments aggregate children without silently adding
    contacts. Numeric soma counts are separate affine-bypass budgets. Integer
    K may round a distal budget upward by fewer than one contact per leaf, and
    that overhead is explicit in ``export_contract``. Dense-to-sparse starts
    with ``initial_density`` and anneals to final K; indexed-rewire stores only
    final K while allowing contact identities to change.
    """

    family, resolved_density = parse_candidate(candidate, density=density)
    hidden_size = int(hidden_size)
    teacher_intermediate_size = int(teacher_intermediate_size)
    if hidden_size < 1 or teacher_intermediate_size < 1:
        raise ValueError("hidden and teacher intermediate sizes must be positive")
    if not 0.0 < float(inhibitory_fraction) <= 1.0:
        raise ValueError("inhibitory_fraction must be in (0, 1]")
    if not 0.0 < float(initial_density) <= 1.0:
        raise ValueError("initial_density must be in (0, 1]")
    resolved_input_to_e_density = _validate_density(
        (
            resolved_density
            if input_to_excitatory_density is None
            else input_to_excitatory_density
        ),
        name="input_to_excitatory_density",
    )
    resolved_input_to_i_density = _validate_density(
        (
            resolved_density
            if input_to_inhibitory_density is None
            else input_to_inhibitory_density
        ),
        name="input_to_inhibitory_density",
    )
    resolved_i_to_e_density = _validate_density(
        (
            resolved_density
            if inhibitory_to_excitatory_density is None
            else inhibitory_to_excitatory_density
        ),
        name="inhibitory_to_excitatory_density",
    )
    resolved_output_density = float(
        resolved_density if output_density is None else output_density
    )
    if not 0.0 < resolved_output_density <= 1.0:
        raise ValueError("output_density must be in (0, 1]")
    legacy_family = str(candidate).split("@d", 1)[0].strip().lower()
    normalized_output_mode = str(output_projection_mode).strip().lower()
    if legacy_family == "rank_readout" and normalized_output_mode == "sparse":
        normalized_output_mode = "low_rank"
    if normalized_output_mode not in {"sparse", "low_rank"}:
        raise ValueError("output_projection_mode must be 'sparse' or 'low_rank'")
    if normalized_output_mode == "low_rank":
        if output_rank is None and legacy_family != "rank_readout":
            raise ValueError("low_rank output projection requires output_rank")
        if output_rank is not None and int(output_rank) < 1:
            raise ValueError("output_rank must be positive")
    elif output_rank is not None:
        raise ValueError(
            "output_rank is only executable when output_projection_mode='low_rank'"
        )
    if int(pruning_end_step) < 1:
        raise ValueError("pruning_end_step must be >= 1")
    resolved_rewire_until = (
        int(pruning_end_step) if rewire_until_step is None else int(rewire_until_step)
    )
    resolved_output_rewire_frequency = (
        int(rewire_frequency)
        if output_rewire_frequency is None
        else int(output_rewire_frequency)
    )
    resolved_output_rewire_quantile = (
        float(rewire_quantile)
        if output_rewire_quantile is None
        else float(output_rewire_quantile)
    )
    resolved_output_rewire_until = (
        resolved_rewire_until
        if output_rewire_until_step is None
        else int(output_rewire_until_step)
    )
    for name, value in (
        ("rewire_frequency", rewire_frequency),
        ("output_rewire_frequency", resolved_output_rewire_frequency),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1")
    for name, value in (
        ("rewire_quantile", rewire_quantile),
        ("output_rewire_quantile", resolved_output_rewire_quantile),
    ):
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
    for name, value in (
        ("rewire_until_step", resolved_rewire_until),
        ("output_rewire_until_step", resolved_output_rewire_until),
    ):
        if int(value) < 1:
            raise ValueError(f"{name} must be >= 1")

    normalized_topology_mode = str(topology_mode).strip().lower()
    if normalized_topology_mode not in TOPOLOGY_MODES:
        raise ValueError(f"topology_mode must be one of {TOPOLOGY_MODES}")
    normalized_output_topology_mode = (
        normalized_topology_mode
        if output_topology_mode is None
        else str(output_topology_mode).strip().lower()
    )
    if normalized_output_topology_mode not in TOPOLOGY_MODES:
        raise ValueError(f"output_topology_mode must be one of {TOPOLOGY_MODES}")
    normalized_bypass_mode = str(affine_bypass_mode).strip().lower()
    if normalized_bypass_mode not in {"none", "sparse", "low_rank"}:
        raise ValueError("affine_bypass_mode must be none, sparse, or low_rank")
    normalized_bypass_topology_mode = (
        normalized_output_topology_mode
        if affine_bypass_topology_mode is None
        else str(affine_bypass_topology_mode).strip().lower()
    )
    if normalized_bypass_topology_mode not in TOPOLOGY_MODES:
        raise ValueError(f"affine_bypass_topology_mode must be one of {TOPOLOGY_MODES}")
    active_topology_modes = {normalized_topology_mode}
    if normalized_output_mode == "sparse":
        active_topology_modes.add(normalized_output_topology_mode)
    width = int(
        teacher_intermediate_size if population_width is None else population_width
    )
    if legacy_family == "wide_narrow" and population_width is None:
        width = 2 * teacher_intermediate_size
    if width < 1:
        raise ValueError("population_width must be positive")

    gated = family.startswith("gated_")
    explicit_ei = "_ei_" in family
    positive_ei = "positive_ei" in family
    branched = "branched" in family
    family_shunting = family.endswith("shunting")
    normalized_integration = "shunting" if family_shunting else "raw_additive"
    if integration_rule is not None:
        normalized_integration = str(integration_rule).strip().lower()
    integration_aliases = {
        "additive": "raw_additive",
        "raw": "raw_additive",
        "normalized_additive": "conductance_normalized",
        "tangent_additive": "tangent_matched",
    }
    normalized_integration = integration_aliases.get(
        normalized_integration, normalized_integration
    )
    if normalized_integration not in INTEGRATION_RULES:
        raise ValueError(
            f"integration_rule must be one of {INTEGRATION_RULES}, got "
            f"{normalized_integration!r}"
        )
    if normalized_integration == "tangent_matched" and (
        additive_tangent_n0 is None or additive_tangent_t0 is None
    ):
        raise ValueError(
            "tangent_matched integration requires calibrated "
            "additive_tangent_n0 and additive_tangent_t0 anchors"
        )
    shunting = normalized_integration == "shunting"
    safety_record: dict[str, Any] | None = None
    if shunting and not positive_ei:
        # Empirical safety default (strategy R12, 2026-08-25): all three
        # signed shunting arms diverged (peaked by step 50; hybrid feature
        # matrix v1). Automatic selection must never produce this pairing;
        # deliberate research probes pass allow_measured_unstable=True and
        # carry the label in their provenance.
        if not allow_measured_unstable:
            raise ValueError(
                "shunting integration on signed-weight families is a "
                "measured-unstable pairing (strategy R12: all three signed "
                "shunting arms diverged). Use a positive_ei family, raw "
                "additive integration, or pass allow_measured_unstable=True "
                "for an explicitly labeled unsafe research probe."
            )
        safety_record = {
            "measured_unstable_pairing": "signed_shunting",
            "override_authorized": True,
            "evidence_rule": "R12",
        }
    if shunting and not explicit_ei:
        raise ValueError(
            "shunting integration requires an explicit E/I family so the "
            "denominator has a declared inhibitory pathway"
        )
    if explicit_ei and branched:
        old_suffix = "shunting" if family.endswith("shunting") else "additive"
        new_suffix = "shunting" if shunting else "additive"
        if old_suffix != new_suffix:
            family = f"{family.rsplit('_', 1)[0]}_{new_suffix}"
    biological = bool(positive_ei)
    resolved_input_transform = (
        ("signed_split" if biological else "identity")
        if input_transform is None
        else str(input_transform).strip().lower()
    )
    if resolved_input_transform not in {"identity", "relu", "signed_split"}:
        raise ValueError(
            "input_transform must be identity, relu, or signed_split, got "
            f"{resolved_input_transform!r}"
        )
    if biological and resolved_input_transform not in {"relu", "signed_split"}:
        raise ValueError(
            "biological replacement families require nonnegative core inputs; "
            "use input_transform='relu' or 'signed_split'"
        )
    default_branches = tuple(int(value) for value in branch_factors)
    realized_excitatory_branches = tuple(
        int(value)
        for value in (
            default_branches
            if excitatory_branch_factors is None
            else excitatory_branch_factors
        )
    )
    realized_inhibitory_branches = tuple(
        int(value)
        for value in (
            default_branches
            if inhibitory_branch_factors is None
            else inhibitory_branch_factors
        )
    )
    if not branched:
        realized_excitatory_branches = ()
        realized_inhibitory_branches = ()
    for name, values in (
        ("excitatory_branch_factors", realized_excitatory_branches),
        ("inhibitory_branch_factors", realized_inhibitory_branches),
    ):
        if any(value < 1 for value in values):
            raise ValueError(f"{name} entries must be positive")
    if branched and not (realized_excitatory_branches or realized_inhibitory_branches):
        raise ValueError(
            "branched families require at least one non-empty E or I branch plan"
        )

    inhibitory_width = (
        max(1, round(width * float(inhibitory_fraction))) if explicit_ei else 0
    )
    adapted_input_dim = (
        2 * hidden_size if resolved_input_transform == "signed_split" else hidden_size
    )
    resolved_input_to_e_contacts = _validate_contacts(
        input_to_excitatory_contacts,
        fallback_density=resolved_input_to_e_density,
        reference_dim=hidden_size,
        source_dim=adapted_input_dim,
        name="input_to_excitatory_contacts",
        require_nonzero=True,
    )
    resolved_input_to_i_contacts = (
        _validate_contacts(
            input_to_inhibitory_contacts,
            fallback_density=resolved_input_to_i_density,
            reference_dim=hidden_size,
            source_dim=adapted_input_dim,
            name="input_to_inhibitory_contacts",
            require_nonzero=True,
        )
        if explicit_ei
        else 0
    )
    resolved_i_to_e_contacts = (
        _validate_contacts(
            inhibitory_to_excitatory_contacts,
            fallback_density=resolved_i_to_e_density,
            reference_dim=inhibitory_width,
            source_dim=inhibitory_width,
            name="inhibitory_to_excitatory_contacts",
            require_nonzero=True,
        )
        if explicit_ei
        else 0
    )

    requested_somatic = (
        not branched if somatic_synapses is None else bool(somatic_synapses)
    )
    if not branched and somatic_synapses is False:
        raise ValueError(
            "flat populations have no non-somatic compartment and therefore "
            "cannot set somatic_synapses=false"
        )
    numeric_somatic = (
        somatic_excitatory_synapses,
        somatic_inhibitory_synapses,
        inhibitory_population_somatic_synapses,
    )
    if somatic_synapses is False and any(
        value is not None and int(value) > 0 for value in numeric_somatic
    ):
        raise ValueError(
            "somatic_synapses=false conflicts with a positive numeric soma budget"
        )

    def soma_budget(value: int | None, fallback: int, source_dim: int) -> int:
        if not requested_somatic:
            return 0
        resolved = fallback if value is None else int(value)
        if resolved < 0 or resolved > source_dim:
            raise ValueError(
                f"somatic contact budget must be in [0, {source_dim}], got {resolved}"
            )
        return resolved

    # Flat populations have no dendritic level separate from the soma, so
    # their entire incoming pathway budget is necessarily somatic. Branched
    # populations use the explicit numeric bypass or the legacy Boolean
    # fallback (one distal-leaf K) when no number was supplied.
    excitatory_leaf_count = (
        math.prod(realized_excitatory_branches) if realized_excitatory_branches else 1
    )
    inhibitory_leaf_count = (
        math.prod(realized_inhibitory_branches) if realized_inhibitory_branches else 1
    )
    default_e_soma = (
        resolved_input_to_e_contacts
        if not branched
        else math.ceil(resolved_input_to_e_contacts / excitatory_leaf_count)
    )
    default_i_path_soma = (
        resolved_i_to_e_contacts
        if not branched
        else math.ceil(resolved_i_to_e_contacts / excitatory_leaf_count)
    )
    default_i_population_soma = (
        resolved_input_to_i_contacts
        if not branched
        else math.ceil(resolved_input_to_i_contacts / inhibitory_leaf_count)
    )
    resolved_e_soma = soma_budget(
        somatic_excitatory_synapses,
        default_e_soma,
        adapted_input_dim,
    )
    resolved_i_path_soma = (
        soma_budget(
            somatic_inhibitory_synapses,
            default_i_path_soma,
            inhibitory_width,
        )
        if explicit_ei
        else 0
    )
    resolved_i_population_soma = (
        soma_budget(
            inhibitory_population_somatic_synapses,
            default_i_population_soma,
            adapted_input_dim,
        )
        if explicit_ei
        else 0
    )
    if not branched:
        resolved_e_soma = resolved_input_to_e_contacts
        resolved_i_path_soma = resolved_i_to_e_contacts
        resolved_i_population_soma = resolved_input_to_i_contacts

    e_exc_by_level, e_exc_contract = _distal_and_somatic_layout(
        distal_contacts_per_soma=resolved_input_to_e_contacts,
        branch_factors=realized_excitatory_branches,
        source_dim=adapted_input_dim,
        somatic_contacts=resolved_e_soma,
        pathway="input_to_excitatory_population",
    )
    e_inh_by_level: list[int] = [0] * len(e_exc_by_level)
    e_inh_contract: dict[str, Any] | None = None
    i_exc_by_level: list[int] = []
    i_exc_contract: dict[str, Any] | None = None
    if explicit_ei:
        e_inh_by_level, e_inh_contract = _distal_and_somatic_layout(
            distal_contacts_per_soma=resolved_i_to_e_contacts,
            branch_factors=realized_excitatory_branches,
            source_dim=inhibitory_width,
            somatic_contacts=resolved_i_path_soma,
            pathway="inhibitory_to_excitatory_population",
        )
        i_exc_by_level, i_exc_contract = _distal_and_somatic_layout(
            distal_contacts_per_soma=resolved_input_to_i_contacts,
            branch_factors=realized_inhibitory_branches,
            source_dim=adapted_input_dim,
            somatic_contacts=resolved_i_population_soma,
            pathway="input_to_inhibitory_population",
        )
    resolved_somatic_synapses = any(
        value > 0
        for value in (
            resolved_e_soma,
            resolved_i_path_soma,
            resolved_i_population_soma,
        )
    )
    synapses_per_branch = int(e_exc_by_level[0])
    if reactivation_type is None:
        resolved_reactivation_type = "param_tanh"
        reactivate = bool(branched or explicit_ei)
    else:
        resolved_reactivation_type = str(reactivation_type).strip().lower()
        reactivate = resolved_reactivation_type not in {"none", "identity"}
    if not resolved_reactivation_type:
        raise ValueError("reactivation_type must be non-empty")
    if biological and resolved_reactivation_type not in {
        "param_tanh",
        "param_tanh_only_m",
        "param_relu",
        "relu",
        "sigmoid",
        "softplus",
    }:
        raise ValueError(
            "biological replacement families require a nonnegative population "
            f"reactivation, got {resolved_reactivation_type!r}"
        )
    normalized_support_metric = (
        None
        if teacher_support_metric is None or not str(teacher_support_metric).strip()
        else str(teacher_support_metric).strip().lower()
    )
    if normalized_support_metric not in {
        None,
        "gradient",
        "activation_weighted",
        "activation_weighted_robust",
        "activation_weighted_structured",
        "row_taylor",
        "row_empirical_fisher",
        "row_signed_covariance",
    }:
        raise ValueError(
            "teacher_support_metric must be gradient, activation_weighted, "
            "activation_weighted_robust, activation_weighted_structured, "
            "row_taylor, row_empirical_fisher, row_signed_covariance, or omitted"
        )
    normalized_gate_activation = str(gate_activation).strip().lower()
    if gated and normalized_gate_activation not in {
        "silu",
        "swish",
        "gelu",
        "relu",
        "identity",
        "linear",
        "none",
    }:
        raise ValueError("unsupported gate_activation")

    if "dense_to_sparse" in active_topology_modes:
        realized_path_densities: list[float] = []
        if normalized_topology_mode == "dense_to_sparse":
            realized_path_densities.append(max(e_exc_by_level) / adapted_input_dim)
        if explicit_ei and normalized_topology_mode == "dense_to_sparse":
            realized_path_densities.extend(
                [
                    max(e_inh_by_level) / inhibitory_width,
                    max(i_exc_by_level) / adapted_input_dim,
                ]
            )
        if (
            normalized_output_mode == "sparse"
            and normalized_output_topology_mode == "dense_to_sparse"
        ):
            realized_path_densities.append(resolved_output_density)
        maximum_final_density = max(realized_path_densities)
        if float(initial_density) + 1e-12 < maximum_final_density:
            raise ValueError(
                "dense-to-sparse initial_density must be at least every "
                "realized final pathway density; required >= "
                f"{maximum_final_density:.6g}"
            )

    topology = _topology_payload(
        mode=normalized_topology_mode,
        seed=int(seed),
        pruning_end_step=int(pruning_end_step),
        initial_density=float(initial_density),
        rewire_frequency=int(rewire_frequency),
        rewire_quantile=float(rewire_quantile),
        rewire_until_step=resolved_rewire_until,
        projection_backend=str(projection_backend),
    )
    common = {
        # Every population below installs explicit, distal-to-soma pathway
        # lists. Zero defaults prevent a newly added population from silently
        # inheriting an unaccounted contact bank.
        "ff_excitatory_synapses": 0,
        "ff_inhibitory_synapses": 0,
        "rec_excitatory_synapses": 0,
        "rec_inhibitory_synapses": 0,
        "somatic_synapses": False,
        "use_shunting": bool(shunting),
        "use_additive_normalization": False,
        "additive_mode": (
            "raw"
            if shunting
            else {
                "raw_additive": "raw",
                "conductance_normalized": "conductance_normalized",
                "tangent_matched": "tangent_matched",
            }[normalized_integration]
        ),
        "additive_tangent_n0": additive_tangent_n0,
        "additive_tangent_t0": additive_tangent_t0,
        "reactivate": reactivate,
        "reactivation_type": resolved_reactivation_type,
        "reactivation_init_policy": "occupancy_quantile",
        "reactivation_memory_efficient": True,
        "weight_transform": "softplus" if biological else "identity",
        "efficient_blocklinear": True,
        **topology,
    }

    if explicit_ei:
        populations = [
            {
                "name": "i",
                "polarity": "inhibitory",
                "n_neurons": inhibitory_width,
                "branch_factors": list(realized_inhibitory_branches),
                "population": {
                    "ff_excitatory_synapses": int(i_exc_by_level[0]),
                    "ff_excitatory_synapses_by_level": list(i_exc_by_level),
                    "ff_inhibitory_synapses": 0,
                    "ff_inhibitory_synapses_by_level": [0] * len(i_exc_by_level),
                    "somatic_synapses": bool(i_exc_by_level[-1] > 0),
                    "use_shunting": False,
                    "indexed_seed": int(seed) + 1,
                },
            },
            {
                "name": "e",
                "polarity": "excitatory",
                "n_neurons": width,
                "branch_factors": list(realized_excitatory_branches),
                "population": {
                    "ff_excitatory_synapses": int(e_exc_by_level[0]),
                    "ff_inhibitory_synapses": int(e_inh_by_level[0]),
                    "ff_excitatory_synapses_by_level": list(e_exc_by_level),
                    "ff_inhibitory_synapses_by_level": list(e_inh_by_level),
                    "somatic_synapses": bool(
                        e_exc_by_level[-1] > 0 or e_inh_by_level[-1] > 0
                    ),
                },
            },
        ]
        connections = [
            {"source": "input", "target": "i"},
            {"source": "input", "target": "e"},
            {"source": "i", "target": "e"},
        ]
        readout = "e"
    else:
        populations = [
            {
                "name": "ffn",
                "polarity": "excitatory",
                "n_neurons": width,
                "branch_factors": list(realized_excitatory_branches),
                "population": {
                    "ff_excitatory_synapses": int(e_exc_by_level[0]),
                    "ff_inhibitory_synapses": 0,
                    "ff_excitatory_synapses_by_level": list(e_exc_by_level),
                    "ff_inhibitory_synapses_by_level": [0] * len(e_exc_by_level),
                    "somatic_synapses": bool(e_exc_by_level[-1] > 0),
                },
            }
        ]
        connections = [{"source": "input", "target": "ffn"}]
        readout = "ffn"

    population_network = {
        # Transformer wrappers consume and remove this field before building
        # the core; direct vision/classification PopulationNetworks apply it
        # internally. Thus one compiled plan has the same input semantics in
        # both replacement pipelines.
        "input_transform": resolved_input_transform,
        "layers": [
            {
                "name": "replacement",
                "readout_population": readout,
                "population_defaults": common,
                "populations": populations,
                "connections": connections,
            }
        ],
    }
    core_config = {
        "type": "population_network",
        "biological_neuron": biological,
        # Retain a concise operator declaration for existing validation/cards;
        # the executable values live in population_defaults above.
        "morphology": {
            "weight_transform": common["weight_transform"],
            "use_shunting": bool(shunting),
            "additive_mode": common["additive_mode"],
        },
        "population_network": population_network,
    }
    replacement_kind = "gated_population_network" if gated else "population_network_ffn"
    effective_output_rank = output_rank
    if legacy_family == "rank_readout" and effective_output_rank is None:
        effective_output_rank = max(1, min(hidden_size, width) // 8)
    if effective_output_rank is not None and int(effective_output_rank) > min(
        hidden_size, width
    ):
        raise ValueError(
            "output_rank must not exceed min(hidden_size, population_width)"
        )
    output_topk = (
        None
        if normalized_output_mode == "low_rank"
        else max(1, min(width, round(resolved_output_density * width)))
    )
    replacement_kwargs: dict[str, Any] = {
        "kind": replacement_kind,
        "input_transform": resolved_input_transform,
        "biological_neuron": biological,
    }
    if normalized_support_metric is not None:
        replacement_kwargs["teacher_support_metric"] = normalized_support_metric
    adapted_input_dim = (
        2 * hidden_size if resolved_input_transform == "signed_split" else hidden_size
    )
    bypass_topk = None
    effective_bypass_rank = None
    bypass_density = (
        resolved_density
        if affine_bypass_density is None
        else _validate_density(affine_bypass_density, name="affine_bypass_density")
    )
    if normalized_bypass_mode == "sparse":
        if affine_bypass_rank is not None:
            raise ValueError("sparse affine bypass cannot also declare a rank")
        bypass_topk = max(
            1, min(adapted_input_dim, round(bypass_density * hidden_size))
        )
        active_topology_modes.add(normalized_bypass_topology_mode)
        replacement_kwargs.update(
            {
                "affine_bypass_topk": int(bypass_topk),
                "affine_bypass_topology_mode": normalized_bypass_topology_mode,
                "affine_bypass_seed": int(seed) + 2_000_003,
            }
        )
    elif normalized_bypass_mode == "low_rank":
        effective_bypass_rank = (
            max(1, min(hidden_size, adapted_input_dim) // 8)
            if affine_bypass_rank is None
            else int(affine_bypass_rank)
        )
        if not 1 <= effective_bypass_rank <= min(hidden_size, adapted_input_dim):
            raise ValueError(
                "affine_bypass_rank must be in [1, min(hidden_size, adapted_input_dim)]"
            )
        replacement_kwargs["affine_bypass_rank"] = int(effective_bypass_rank)
    if gated:
        replacement_kwargs["gate_activation"] = normalized_gate_activation
    if normalized_output_mode == "low_rank":
        assert effective_output_rank is not None
        replacement_kwargs["output_rank"] = int(effective_output_rank)
    else:
        replacement_kwargs.update(
            {
                "output_topk": int(output_topk),
                "output_topology_mode": normalized_output_topology_mode,
                "output_seed": int(seed) + 1_000_003,
                "output_selection": "standard",
                "output_noise_level": 0.0,
                "output_temperature": 0.5,
                "output_ultrafast": True,
                "output_variance_momentum": 0.9,
                "output_rewire_frequency": resolved_output_rewire_frequency,
                "output_rewire_quantile": resolved_output_rewire_quantile,
                "output_rewire_until_step": resolved_output_rewire_until,
                "output_dense_to_sparse_initial_density": float(initial_density),
                "output_dense_to_sparse_start_step": 0,
                "output_dense_to_sparse_end_step": int(pruning_end_step),
                "output_dense_to_sparse_update_interval": 1,
                "output_dense_to_sparse_schedule": "cubic",
                "output_dense_to_sparse_freeze_on_end": True,
                "output_dense_to_sparse_advance_on_forward": False,
                "output_dense_to_sparse_prune_metric": "magnitude",
                "output_index_dtype": "auto",
                "output_projection_backend": str(projection_backend),
            }
        )

    dynamic_training_modes = set(TOPOLOGY_MODES) - {"indexed"}
    freeze_on_export = bool(active_topology_modes & dynamic_training_modes)
    training_overrides = {
        "restore_best_replacement": "dense_to_sparse" not in active_topology_modes,
        "freeze_sparse_topology_on_export": freeze_on_export,
        "stochastic_topology_freeze_policy": (
            "deterministic_topk" if "stochastic" in active_topology_modes else "reject"
        ),
        "stochastic_topology_freeze_seed": int(seed),
        "ragged_topology_format": "csr",
    }
    pathway_contracts = {
        "input_to_excitatory_population": e_exc_contract,
    }
    if e_inh_contract is not None:
        pathway_contracts["inhibitory_to_excitatory_population"] = e_inh_contract
    if i_exc_contract is not None:
        pathway_contracts["input_to_inhibitory_population"] = i_exc_contract
    core_contacts_per_cell = sum(
        int(contract["realized_total_contacts_per_soma"])
        * (width if "excitatory_population" in name else inhibitory_width)
        for name, contract in pathway_contracts.items()
    )
    output_path_multiplier = 2 if biological else 1
    if normalized_output_mode == "sparse":
        assert output_topk is not None
        output_weight_parameters = hidden_size * output_topk * output_path_multiplier
    else:
        assert effective_output_rank is not None
        output_weight_parameters = (
            width * int(effective_output_rank)
            + int(effective_output_rank) * hidden_size
        ) * output_path_multiplier
    if normalized_bypass_mode == "none":
        bypass_weight_parameters = 0
    elif normalized_bypass_mode == "sparse":
        assert bypass_topk is not None
        bypass_weight_parameters = hidden_size * bypass_topk * output_path_multiplier
    else:
        assert effective_bypass_rank is not None
        bypass_weight_parameters = (
            adapted_input_dim * effective_bypass_rank
            + effective_bypass_rank * hidden_size
        ) * output_path_multiplier
    export_contract = {
        "schema": "dendritic_replacement_export_contract/v2",
        "contact_semantics": (
            "distal dendritic budget per soma plus separately declared soma bypass"
        ),
        "pathways": pathway_contracts,
        "realized_core_contact_parameters_per_core": int(core_contacts_per_cell),
        "gated_core_multiplier": 2 if gated else 1,
        "realized_core_contact_parameters_per_replacement": int(
            core_contacts_per_cell * (2 if gated else 1)
        ),
        "final_contacts_per_output_pathway_budget": int(resolved_input_to_e_contacts),
        "final_synapses_per_leaf_branch": int(synapses_per_branch),
        "leaf_branches_per_unit": int(excitatory_leaf_count),
        "excitatory_leaf_branches_per_unit": int(excitatory_leaf_count),
        "inhibitory_leaf_branches_per_unit": int(inhibitory_leaf_count),
        "output_projection_mode": normalized_output_mode,
        "core_topology_selection_method": normalized_topology_mode,
        "output_topology_selection_method": normalized_output_topology_mode,
        "output_rank": (
            None if effective_output_rank is None else int(effective_output_rank)
        ),
        "output_contacts_per_hidden_unit": output_topk,
        "output_pathway_multiplier": int(output_path_multiplier),
        "estimated_output_weight_parameters": int(output_weight_parameters),
        "affine_bypass_mode": normalized_bypass_mode,
        "affine_bypass_density": (
            None if normalized_bypass_mode == "none" else float(bypass_density)
        ),
        "affine_bypass_topology_selection_method": (
            normalized_bypass_topology_mode
            if normalized_bypass_mode == "sparse"
            else None
        ),
        "affine_bypass_contacts_per_output": bypass_topk,
        "affine_bypass_rank": effective_bypass_rank,
        "estimated_affine_bypass_weight_parameters": int(bypass_weight_parameters),
        "freeze_dynamic_topology": freeze_on_export,
        "checkpoint_representation": "fixed_indexed_contacts",
        "report_deployed_not_training_storage": True,
        "required_optimizer_steps_before_export": (
            int(pruning_end_step) if "dense_to_sparse" in active_topology_modes else 0
        ),
    }
    return CompiledReplacementPlan(
        requested_candidate=str(candidate),
        family=family,
        density=resolved_density,
        hidden_size=hidden_size,
        population_width=width,
        inhibitory_width=inhibitory_width,
        branch_factors=realized_excitatory_branches,
        excitatory_branch_factors=realized_excitatory_branches,
        inhibitory_branch_factors=realized_inhibitory_branches,
        somatic_synapses=resolved_somatic_synapses,
        synapses_per_branch=int(synapses_per_branch),
        input_to_excitatory_density=resolved_input_to_e_density,
        input_to_inhibitory_density=resolved_input_to_i_density,
        inhibitory_to_excitatory_density=resolved_i_to_e_density,
        input_to_excitatory_contacts=resolved_input_to_e_contacts,
        input_to_inhibitory_contacts=resolved_input_to_i_contacts,
        inhibitory_to_excitatory_contacts=resolved_i_to_e_contacts,
        somatic_excitatory_synapses=resolved_e_soma,
        somatic_inhibitory_synapses=resolved_i_path_soma,
        inhibitory_population_somatic_synapses=resolved_i_population_soma,
        output_density=resolved_output_density,
        output_rank=(
            None if effective_output_rank is None else int(effective_output_rank)
        ),
        output_projection_mode=normalized_output_mode,
        output_topk=output_topk,
        topology_mode=normalized_topology_mode,
        integration_rule=normalized_integration,
        biological_neuron=biological,
        input_transform=resolved_input_transform,
        reactivation_type=resolved_reactivation_type,
        gate_activation=(normalized_gate_activation if gated else None),
        teacher_support_metric=normalized_support_metric,
        replacement_kind=replacement_kind,
        core_config=core_config,
        replacement_kwargs=replacement_kwargs,
        training_overrides=training_overrides,
        export_contract=export_contract,
        safety=safety_record,
    )


def apply_compiled_plan_to_config(
    base_config: Mapping[str, Any],
    plan: CompiledReplacementPlan,
    *,
    layers: Sequence[int] | None = None,
    target: str = "transformer",
) -> dict[str, Any]:
    """Merge a compiled plan into a standard experiment configuration.

    Teacher, data, optimizer, and model-loading settings remain inherited from
    the base config. The replacement core and its topology/export policy come
    from the compiled plan. The returned mapping can be written directly as
    YAML and consumed by the maintained transformer replacement trainer.
    """

    if not isinstance(base_config, Mapping):
        raise TypeError("base_config must be a mapping")
    payload = deepcopy(dict(base_config))
    model = payload.setdefault("model", {})
    if not isinstance(model, dict):
        raise TypeError("base_config.model must be a mapping")
    model["core"] = deepcopy(plan.core_config)
    training = payload.setdefault("training", {})
    if not isinstance(training, dict):
        raise TypeError("base_config.training must be a mapping")
    normalized_target = str(target).strip().lower()
    if normalized_target == "transformer":
        transformer = model.setdefault("transformer_replacement", {})
        if not isinstance(transformer, dict):
            raise TypeError(
                "base_config.model.transformer_replacement must be a mapping"
            )
        transformer["enabled"] = True
        transformer["input_transform"] = plan.replacement_kwargs["input_transform"]
        transformer["replacement_kwargs"] = deepcopy(plan.replacement_kwargs)
        transformer["compiled_plan"] = plan.as_dict()
        # A uniform compiled plan, per-layer frozen plans, and runtime FMI
        # selection are mutually exclusive authorities. Base configs are often
        # reused from a profiled campaign, so retaining either stale field can
        # make a nominal sweep execute the inherited architecture instead.
        transformer.pop("compiled_plans_by_layer", None)
        transformer["selection"] = {"enabled": False}
        replacement_training = training.setdefault("transformer_replacement", {})
        if not isinstance(replacement_training, dict):
            raise TypeError(
                "base_config.training.transformer_replacement must be a mapping"
            )
        replacement_training["enabled"] = True
        replacement_training.update(deepcopy(plan.training_overrides))

        if layers is not None:
            normalized_layers = [int(layer) for layer in layers]
            if not normalized_layers:
                raise ValueError("layers must not be empty")
            if len(normalized_layers) != len(set(normalized_layers)):
                raise ValueError("layers must not contain duplicates")
            transformer["layers"] = normalized_layers
            replacement_training["layers"] = normalized_layers
    elif normalized_target == "vision":
        if layers is not None:
            raise ValueError(
                "layers applies only to transformer targets; configure the vision "
                "replacement span in model.vision_replacement"
            )
        vision = model.setdefault("vision_replacement", {})
        if not isinstance(vision, dict):
            raise TypeError("base_config.model.vision_replacement must be a mapping")
        vision["enabled"] = True
        replacement_training = training.setdefault("vision_replacement", {})
        if not isinstance(replacement_training, dict):
            raise TypeError("base_config.training.vision_replacement must be a mapping")
        replacement_training["enabled"] = True
        replacement_training.update(deepcopy(plan.training_overrides))
        replacement_training.setdefault("replacement_checkpoint_encoding", "auto")
    else:
        raise ValueError("target must be 'transformer' or 'vision'")

    required_steps = int(plan.export_contract["required_optimizer_steps_before_export"])
    configured_steps = int(replacement_training.get("max_steps", 200))
    if required_steps and configured_steps < required_steps:
        raise ValueError(
            "dense-to-sparse export requires max_steps >= pruning_end_step; "
            f"got max_steps={configured_steps}, pruning_end_step={required_steps}"
        )
    return payload


def apply_compiled_plans_by_layer_to_config(
    base_config: Mapping[str, Any],
    plans_by_layer: Mapping[int | str, CompiledReplacementPlan | Mapping[str, Any]],
) -> dict[str, Any]:
    """Materialize a transformer config with a distinct frozen plan per layer.

    This is the explicit non-uniform counterpart of
    :func:`apply_compiled_plan_to_config`. Every layer is compiled before the
    training job starts, so the executed morphology cannot drift with config
    defaults or a later fingerprint edit.
    """

    if not plans_by_layer:
        raise ValueError("plans_by_layer must not be empty")
    normalized: dict[int, CompiledReplacementPlan] = {}
    for raw_layer, raw_plan in plans_by_layer.items():
        layer = int(raw_layer)
        if layer < 0 or layer in normalized:
            raise ValueError(
                "compiled plan layer indices must be unique non-negative integers"
            )
        normalized[layer] = (
            raw_plan
            if isinstance(raw_plan, CompiledReplacementPlan)
            else compiled_replacement_plan_from_mapping(raw_plan)
        )
    ordered = dict(sorted(normalized.items()))
    hidden_sizes = {plan.hidden_size for plan in ordered.values()}
    if len(hidden_sizes) != 1:
        raise ValueError(
            "One transformer replacement campaign must use a common hidden size"
        )

    first = next(iter(ordered.values()))
    payload = deepcopy(dict(base_config))
    model = payload.setdefault("model", {})
    if not isinstance(model, dict):
        raise TypeError("base_config.model must be a mapping")
    # The first core remains a readable schema exemplar. Runtime construction
    # uses each layer's plan.core_config, never this shared fallback.
    model["core"] = deepcopy(first.core_config)
    transformer = model.setdefault("transformer_replacement", {})
    if not isinstance(transformer, dict):
        raise TypeError("base_config.model.transformer_replacement must be a mapping")
    transformer.update(
        {
            "enabled": True,
            "layers": list(ordered),
            "input_transform": first.input_transform,
            "replacement_kwargs": deepcopy(first.replacement_kwargs),
            "compiled_plan": {},
            "compiled_plans_by_layer": {
                str(layer): plan.as_dict() for layer, plan in ordered.items()
            },
            "selection": {"enabled": False},
        }
    )

    training = payload.setdefault("training", {})
    if not isinstance(training, dict):
        raise TypeError("base_config.training must be a mapping")
    replacement_training = training.setdefault("transformer_replacement", {})
    if not isinstance(replacement_training, dict):
        raise TypeError(
            "base_config.training.transformer_replacement must be a mapping"
        )
    replacement_training.update(
        {
            "enabled": True,
            "layers": list(ordered),
            "freeze_sparse_topology_on_export": any(
                bool(plan.training_overrides["freeze_sparse_topology_on_export"])
                for plan in ordered.values()
            ),
            "restore_best_replacement": all(
                bool(plan.training_overrides["restore_best_replacement"])
                for plan in ordered.values()
            ),
        }
    )
    replacement_training.setdefault("replacement_checkpoint_encoding", "auto")
    required_steps = max(
        int(plan.export_contract["required_optimizer_steps_before_export"])
        for plan in ordered.values()
    )
    configured_steps = int(replacement_training.get("max_steps", 200))
    if required_steps and configured_steps < required_steps:
        raise ValueError(
            "non-uniform dense-to-sparse export requires max_steps >= the largest "
            f"pruning_end_step; got {configured_steps} < {required_steps}"
        )
    return payload


__all__ = [
    "CANONICAL_FAMILIES",
    "INTEGRATION_RULES",
    "LEGACY_FAMILY_ALIASES",
    "TOPOLOGY_MODES",
    "CompiledReplacementPlan",
    "apply_compiled_plan_to_config",
    "apply_compiled_plans_by_layer_to_config",
    "compile_replacement_candidate",
    "compiled_replacement_plan_from_mapping",
    "parse_candidate",
]
