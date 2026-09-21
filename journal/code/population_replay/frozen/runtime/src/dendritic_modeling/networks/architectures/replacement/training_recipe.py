"""Evidence-based training recipes for compiled replacement families.

This module turns the 2026-08-24 init-policy x integration-rule measurement
campaign into an executable recommendation: given a family's constraint axes
(``biological_neuron``, ``explicit_ei``, ``gated``), return the integration
rule, reactivation init policy, calibration mode, reactivation switch, and
training-budget multiplier the frozen-architecture evidence supports.  Policy
normalization, the policy-to-calibration mapping, and the constructor-time
pairing guard live in :mod:`dendritic_modeling.config.reactivation` and are
reused, never duplicated.

Evidence record (every number below is from the frozen Pythia-70M L2
architecture — width 222, d0.4414 — single seed, local relative MSE.  These
are frozen-architecture measurements, NOT general truths; the 600-step
verdicts were budget-confounded against biological constraints, so budgets
are treated as explicit treatments throughout):

* Init-integration matrix v2 (2026-08-24, positive_ei_flat, 600 steps): the
  init-policy x integration-rule pairing decides whether strict-positive
  cells train within the screen budget.  Under raw additive integration only
  occupancy_quantile fits (0.625 valid vs analytical 0.991, analytical_slope
  0.990, occupancy_slope 0.988); under conductance-normalized integration
  likewise (0.622 vs 0.999/0.998/0.997 — exonerating conductance
  integration, whose earlier "no fit" was the architecture confound);
  shunting fits under EVERY policy (0.593 across all four, within 4% of the
  frozen 0.571 whose median_mad calibration is slightly better than
  quantile-mode calibration).  The non-quantile pairings were never given a
  long-budget arm, so their "untrainable" verdicts are screen-budget-scoped.
  The checkpoint audit confirmed shunting's policy-insensitivity is genuine
  convergence, not a calibration bypass: trained reactivation gate
  parameters agree to ~1e-6 between the audited analytical and
  occupancy_quantile arms (all four agree to four decimals in loss).

* Slow, not unable (2026-08-24, 5000-step frozen-architecture arms): strict
  positive cells keep descending with no plateau — quantile-paired additive
  0.625 -> 0.383 (best AT the final step), shunting 0.593 -> 0.462 — at
  roughly the ~5-8x approximate pace gap to the signed families.
  :data:`STRICT_POSITIVE_BUDGET_MULTIPLIER` = 8.0 is the upper end of that
  approximate gap; budgets remain explicit treatments — no 8x-budget arm has
  been run and the 5000-step curves had not plateaued.  With the same-budget
  signed anchors (signed_flat 0.1696, gated_signed_flat 0.1482 at 5000
  steps) the ungated biological-constraint tax is ~2.3-2.6x and roughly
  stable across budgets — a ratio at these budgets, not a converged
  asymptote.

* Non-biological health matrix (2026-08-24, 600 steps): no untrainable
  pairing on the signed side — every init policy fits every measured signed
  family within a ~10% band, and occupancy_quantile is best or tied in each
  (signed_flat 0.2503, gated_signed_flat 0.2176, signed_ei_flat 0.2372;
  hybrid matrix v1 adds gated_signed_ei_flat 0.1961).  The one caveat is
  nonlinearity, not init: removing reactivation from UNGATED signed cells
  roughly doubles the loss (0.4608 / 0.4659 vs ~0.25) because reactivation
  was the cell's only nonlinearity, while BOTH gated signed families improve
  without it (gated_signed_flat 0.1997, gated_signed_ei_flat 0.1916 — each
  its family's best cell).  Two same-day single-seed replications at one
  layer; ``reactivate`` stays True here and the gated no-reactivation arm is
  recorded as a twice-replicated candidate, not a default.

* Hybrid feature matrix v1 (2026-08-24, 18 frozen-architecture cells):
  (1) gating rescues biological cells — gated_positive_ei with
  quantile-paired raw additive reads 0.3360 at 600 steps (better than the
  ungated family's 5000-step 0.383) and 0.2467 at 5000, still descending,
  cutting the tax vs the matched gated signed family to ~1.7x; under gating
  additive+quantile beats shunting outright at 600 steps (0.336 vs 0.578).
  (2) The quantile-only rule replicates on the gated biological family
  (0.9502/0.9478/0.9385 no-fit at the screen budget).  (3) Shunting on
  signed weights is measured-unstable: all three signed shunting arms peak
  at step 50 and diverge (signed_flat best 0.8639; gated_signed 0.9694,
  final 1.0174; signed_ei 0.7085, final 1.2670) — the pairing fails closed
  here.  (4) Biological cells without reactivation are structurally invalid
  by validator contract (nonnegative population reactivation required), not
  merely unmeasured.  (5) Signed 5000-step anchors: signed_ei 0.1661,
  gated_signed_ei 0.1402.

Unmeasured open cells — recipes touching them fail closed or carry an
explicit extrapolation label, never assertions: conductance-normalized
integration on any signed family and on gated_positive_ei; gated_positive_ei
shunting under the non-quantile policies; the non-quantile pairings at long
budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from dendritic_modeling.config.reactivation import (
    DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY,
    reactivation_policy_to_calibration_mode,
)

#: Integration rules with a measured training recipe.  ``tangent_matched`` is
#: a valid compiler rule but has no measured init-policy cell, so the recipe
#: fails closed for it rather than guessing.
RECIPE_INTEGRATION_RULES = ("raw_additive", "conductance_normalized", "shunting")

#: Budget multiplier for strict-positive (biological) families: the 5000-step
#: frozen curves measured roughly a ~5-8x (approximate) pace gap to the
#: signed families, still descending with no plateau; 8.0 is the upper end of
#: that approximate gap.  No 8x-budget arm has been run — budgets remain
#: explicit treatments.
STRICT_POSITIVE_BUDGET_MULTIPLIER = 8.0

#: Budget multiplier for signed families: trainable within 600-step screens
#: (not converged — the 5000-step anchors still improve markedly, e.g.
#: signed_flat 0.2503 -> 0.1696).
SIGNED_BUDGET_MULTIPLIER = 1.0

_INTEGRATION_ALIASES = {
    "additive": "raw_additive",
    "raw": "raw_additive",
    "normalized_additive": "conductance_normalized",
}

_SCOPE_NOTE = (
    "scope: frozen Pythia-70M L2 architecture (width 222, d0.4414), single "
    "seed, local relative MSE — frozen-architecture evidence, not a general "
    "claim; budgets are explicit treatments"
)


@dataclass(frozen=True)
class TrainingRecipe:
    """An evidence-cited training recommendation for one replacement family."""

    integration_rule: str
    reactivation_init_policy: str
    reactivation_calibration_mode: str | None
    reactivate: bool
    budget_multiplier: float
    rationale: tuple[str, ...]


def _normalize_integration_rule(integration_rule: str) -> str:
    canonical = str(integration_rule).strip().lower()
    canonical = _INTEGRATION_ALIASES.get(canonical, canonical)
    if canonical == "tangent_matched":
        raise ValueError(
            "tangent_matched integration has no measured training recipe: no "
            "init-policy x integration cell was run for it (2026-08-24 "
            f"matrices). Pin one of {RECIPE_INTEGRATION_RULES} or omit "
            "integration_rule."
        )
    if canonical not in RECIPE_INTEGRATION_RULES:
        raise ValueError(
            f"integration_rule must be one of {RECIPE_INTEGRATION_RULES}, got "
            f"{canonical!r}"
        )
    return canonical


def recommend_training_recipe(
    *,
    biological_neuron: bool,
    explicit_ei: bool,
    gated: bool,
    integration_rule: str | None = None,
) -> TrainingRecipe:
    """Recommend the measured-trainable recipe for a replacement family.

    ``integration_rule=None`` selects the evidence-based default for the
    constraint class; a pinned rule is honored when a measured-trainable
    pairing exists for it, and unknown, unmeasured, or measured-unstable
    rules fail closed with :class:`ValueError` (mirroring the selection guard
    style: no silent fallbacks).  Every recommendation pairs with
    :func:`~dendritic_modeling.config.reactivation.warn_untrainable_init_integration_pairing`
    without triggering it.
    """

    if biological_neuron and not explicit_ei:
        raise ValueError(
            "biological_neuron=true requires explicit_ei=true so signed "
            "effects are represented by separate positive E/I pathways"
        )
    pinned = (
        None
        if integration_rule is None
        else _normalize_integration_rule(integration_rule)
    )
    if pinned == "shunting" and not explicit_ei:
        raise ValueError(
            "shunting integration requires an explicit E/I family so the "
            "denominator has a declared inhibitory pathway"
        )

    policy = DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY
    rationale: list[str] = []
    if biological_neuron:
        budget_multiplier = STRICT_POSITIVE_BUDGET_MULTIPLIER
        rule = pinned if pinned is not None else "raw_additive"
        if rule in ("raw_additive", "conductance_normalized"):
            rationale.append(
                f"occupancy_quantile is REQUIRED under {rule}: it is the only "
                "measured-trainable init policy for strict-positive cells at "
                "the 600-step screen budget — 0.625 (raw additive) / 0.622 "
                "(conductance-normalized) valid relative MSE vs ~0.99 no-fit "
                "for analytical and both hybrid policies (matrix v2, "
                "2026-08-24; no long-budget arm has tested the non-quantile "
                "pairings), replicated on the gated biological family "
                "(0.3360 vs 0.9502/0.9478/0.9385; hybrid matrix v1)"
            )
            if rule == "conductance_normalized":
                rationale.append(
                    "conductance-normalized integration is exonerated: it "
                    "trains when paired with occupancy_quantile on the frozen "
                    "architecture (0.622); its earlier 'no fit' was the "
                    "architecture confound (matrix v2, 2026-08-24)"
                )
            if pinned is None:
                rationale.append(
                    "raw additive + occupancy_quantile is the default for "
                    "strict-positive cells: the best value at the long "
                    "budgets this recipe prescribes (0.383 vs shunting 0.462 "
                    "at 5000 steps, ungated; 0.336 vs shunting 0.578 at 600 "
                    "steps under gating — R11), superseding the 600-step "
                    "shunting-best ordering, which was budget-confounded "
                    "(2026-08-24)"
                )
        else:
            rationale.append(
                "shunting pinned: the policy-robust rule for ungated strict "
                "positive cells (fits under every init policy, 0.593 across "
                "all four at 600 steps, within 4% of the frozen 0.571 whose "
                "median_mad calibration is slightly better); note the "
                "long-budget ordering favors additive+quantile (0.383 vs "
                "0.462 at 5000 steps) and under gating additive+quantile "
                "beats shunting outright (0.336 vs 0.578; hybrid matrix v1)"
            )
            rationale.append(
                "occupancy_quantile kept as the policy: shunting is "
                "policy-insensitive by design (trained reactivation gate "
                "parameters agree to ~1e-6 between the audited analytical "
                "and occupancy_quantile arms; checkpoint audit, 2026-08-24), "
                "and quantile remains the only measured-trainable policy if "
                "integration is later switched to additive/conductance"
            )
        rationale.append(
            "reactivate=True is required by contract for biological cells: "
            "the config validator rejects biological_neuron=true with "
            "reactivate=False (nonnegative population reactivation required; "
            "hybrid matrix v1 — both bio no-reactivation arms failed closed)"
        )
        if gated:
            rationale.append(
                "gating rescues biological cells (measured, hybrid matrix "
                "v1): gated_positive_ei additive+quantile 0.3360 at 600 "
                "steps and 0.2467 at 5000, still descending — the tax vs "
                "the matched gated signed family is ~1.7x instead of ~2.5x"
            )
        rationale.append(
            "budget_multiplier=8.0: strict-positive cells are slow learners, "
            "not broken — 5000-step frozen curves keep descending with no "
            "plateau (quantile-paired additive 0.625 -> 0.383, best at the "
            "final step; shunting 0.593 -> 0.462; gated 0.336 -> 0.247) at "
            "roughly the ~5-8x approximate pace gap; every 600-step verdict "
            "was budget-confounded (2026-08-24). Matched-budget tax vs "
            "signed anchors: ~2.3-2.6x ungated, ~1.7x gated — ratios at "
            "these budgets, not converged asymptotes"
        )
    else:
        budget_multiplier = SIGNED_BUDGET_MULTIPLIER
        if pinned == "shunting":
            raise ValueError(
                "shunting integration on signed weights is measured-unstable "
                "and prohibited: all three signed shunting arms peaked at "
                "step 50 and diverged (signed_flat best 0.8639; gated_signed "
                "0.9694, final 1.0174; signed_ei 0.7085, final 1.2670; "
                "hybrid matrix v1, 2026-08-24). Use raw_additive."
            )
        if pinned == "conductance_normalized":
            raise ValueError(
                "conductance-normalized integration has no measured cell on "
                "any signed family (2026-08-24 matrices measured it on "
                "strict-positive cells only); the recipe fails closed rather "
                "than extrapolating. Use raw_additive or run the frozen-"
                "architecture cell first."
            )
        rule = "raw_additive"
        rationale.append(
            "occupancy_quantile: best or tied in every measured signed "
            "family (signed_flat 0.2503, gated_signed_flat 0.2176, "
            "signed_ei_flat 0.2372, gated_signed_ei_flat 0.1961) and every "
            "policy fits within a ~10% band (non-biological health matrix "
            "and hybrid matrix v1, 600 steps, 2026-08-24)"
        )
        if pinned is None:
            rationale.append(
                "raw additive integration by default for signed families; "
                "shunting-on-signed is measured-unstable and prohibited "
                "(hybrid matrix v1), and conductance-on-signed is unmeasured"
            )
        if gated:
            rationale.append(
                "candidate, not encoded: BOTH gated signed families improve "
                "WITHOUT reactivation (gated_signed_flat 0.1997, "
                "gated_signed_ei_flat 0.1916 — each its family's best cell), "
                "a twice-replicated same-day, single-seed, one-layer result; "
                "reactivate stays True until replicated at another site or "
                "seed (2026-08-24)"
            )
        else:
            rationale.append(
                "reactivate=True is load-bearing for ungated signed cells: "
                "removing reactivation roughly doubles the loss (signed_flat "
                "0.4608, signed_ei_flat 0.4659 vs ~0.25) because it is the "
                "cell's only nonlinearity (2026-08-24)"
            )
        rationale.append(
            "budget_multiplier=1.0: signed families are trainable within "
            "600-step screens (not converged — 5000-step anchors still "
            "improve: signed_flat 0.1696, gated_signed_flat 0.1482, "
            "signed_ei_flat 0.1661, gated_signed_ei_flat 0.1402; 2026-08-24)"
        )
    rationale.append(_SCOPE_NOTE)
    return TrainingRecipe(
        integration_rule=rule,
        reactivation_init_policy=policy,
        reactivation_calibration_mode=reactivation_policy_to_calibration_mode(policy),
        reactivate=True,
        budget_multiplier=budget_multiplier,
        rationale=tuple(rationale),
    )


__all__ = [
    "RECIPE_INTEGRATION_RULES",
    "SIGNED_BUDGET_MULTIPLIER",
    "STRICT_POSITIVE_BUDGET_MULTIPLIER",
    "TrainingRecipe",
    "recommend_training_recipe",
]
