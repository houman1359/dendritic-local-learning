"""Empirical replacement-selection rules learned from measured matrices.

``theory_v2`` predicts a replacement architecture from teacher fingerprints
before anything is trained.  This module is its measured counterpart: it reads
the collected campaign matrices (``measured_matrix.json`` files produced by
``collect_population_replacement_matrix``) and extracts per-axis rules from
CONTROLLED pairs — two completed cells that are identical on every other
architecture axis and differ only on the axis under study.  The output is a
versioned manifest a selector can consume alongside, or instead of, the
theory rules.

Design constraints, in house style:

* **Fail closed.**  Only matrices whose ``status`` is ``measured`` are
  accepted, and an axis with fewer than ``min_pairs`` controlled pairs yields
  ``insufficient_evidence`` rather than a direction.
* **Never mix instruments.**  Pairs are formed only within one source matrix,
  so cells trained under different protocols, layers, seeds, or collectors
  are never compared as if they were controls for each other.
* **Descriptive, not causal.**  The manifest's claim boundary records that
  rules describe the campaigns they were fitted on — typically one teacher,
  one layer, one seed per cell — and must be revalidated prospectively before
  being used as a selector on a new model family.  The Pythia-70M campaign is
  the standing warning: local relative MSE did not reliably predict recovered
  end-to-end quality, so rules fitted on a local metric must not be quoted as
  end-to-end conclusions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

EMPIRICAL_RULES_SCHEMA = "dendritic_replacement_empirical_rules/v1"

#: Categorical architecture axes a v1 rule can be fitted over.  Numeric axes
#: (widths, densities) participate in the MATCHING complement — two cells
#: differing in width are never treated as a controlled pair for any axis —
#: but are not themselves rule targets in v1.
CATEGORICAL_AXES: tuple[str, ...] = (
    "family",
    "biological_neuron",
    "explicit_ei",
    "gated",
    "shunting",
    "integration_rule",
    "branch_factors",
    "teacher_support_metric",
    "affine_bypass_mode",
    "topology_mode",
)

#: Every field that must agree between two records before they count as a
#: controlled pair for some axis: all other categorical axes plus the numeric
#: resource axes.
MATCHING_FIELDS: tuple[str, ...] = (
    *CATEGORICAL_AXES,
    "population_width",
    "inhibitory_width",
    "output_density",
    "output_topk",
    "affine_bypass_density",
    "affine_bypass_contacts_per_output",
)

#: ``family`` is a derived label that ENCODES the mechanism axes (an E/I cell
#: and its no-I control never share a family string), so requiring it to
#: match would make every mechanism comparison impossible.  It stays a rule
#: axis but never joins another axis's complement.
_DERIVED_LABEL_FIELDS: frozenset[str] = frozenset({"family"})

#: Fields that mechanically co-vary with an axis and must therefore be
#: dropped from that axis's matching complement.  Removing an inhibitory
#: population necessarily zeroes the inhibitory width, and the campaign's
#: shunting cells are exactly its conductance-integration cells, so demanding
#: equality there would exclude every genuine controlled pair.  The price is
#: visible rather than hidden: each emitted pair records both cells' active
#: parameter counts, because an E/I cell and its no-I control at equal soma
#: width are NOT parameter-matched (the campaign's measured 1.49x gap).
AXIS_DEPENDENT_FIELDS: dict[str, frozenset[str]] = {
    "explicit_ei": frozenset({"inhibitory_width"}),
    "biological_neuron": frozenset({"teacher_support_metric"}),
    "shunting": frozenset({"integration_rule"}),
    "integration_rule": frozenset({"shunting"}),
}

_REQUIRED_RECORD_FIELDS = ("candidate_id",)


class EmpiricalRuleError(RuntimeError):
    """A matrix or axis cannot support the requested rule extraction."""


def _canonical(value: Any) -> Any:
    """Make list-valued axes (branch factors) hashable and order-stable."""
    if isinstance(value, list):
        return tuple(_canonical(item) for item in value)
    return value


@dataclass(frozen=True)
class MeasuredSource:
    """One accepted measured matrix and the provenance a rule inherits."""

    path: str
    sha256: str
    record_count: int
    git_commits: tuple[str, ...] = ()


@dataclass
class AxisRule:
    """Controlled-pair evidence for one categorical axis."""

    axis: str
    outcome_key: str
    lower_is_better: bool
    pair_count: int
    values: tuple[Any, Any] | None
    wins_first: int
    wins_second: int
    ties: int
    mean_outcome_delta_first_minus_second: float | None
    verdict: str
    pairs: list[dict[str, Any]] = field(default_factory=list)

    def asdict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "outcome_key": self.outcome_key,
            "lower_is_better": self.lower_is_better,
            "pair_count": self.pair_count,
            "values": list(self.values) if self.values is not None else None,
            "wins_first": self.wins_first,
            "wins_second": self.wins_second,
            "ties": self.ties,
            "mean_outcome_delta_first_minus_second": (
                self.mean_outcome_delta_first_minus_second
            ),
            "verdict": self.verdict,
            "pairs": self.pairs,
        }


def load_measured_records(
    paths: Iterable[str | Path],
) -> tuple[list[dict[str, Any]], list[MeasuredSource]]:
    """Load records from measured matrices, refusing anything unmeasured.

    Every returned record carries ``_source_index`` so downstream pairing can
    stay within one matrix; the parallel source list preserves path, content
    hash, and the commits the collector recorded.
    """
    records: list[dict[str, Any]] = []
    sources: list[MeasuredSource] = []
    for path in paths:
        resolved = Path(path)
        raw = resolved.read_bytes()
        payload = json.loads(raw)
        status = payload.get("status")
        # The collector writes "measured_local_screen" for a complete local
        # matrix and "partially_measured" when any run is missing; only the
        # complete statuses are admissible evidence.
        if status not in {"measured", "measured_local_screen"}:
            raise EmpiricalRuleError(
                f"{resolved}: status is {status!r}, not a complete measured "
                "status -- rules must not be fitted on partial or "
                "invalidated campaigns"
            )
        matrix_records = payload.get("records")
        if not isinstance(matrix_records, list) or not matrix_records:
            raise EmpiricalRuleError(f"{resolved}: no records to fit on")
        source_index = len(sources)
        sources.append(
            MeasuredSource(
                path=str(resolved),
                sha256=hashlib.sha256(raw).hexdigest(),
                record_count=len(matrix_records),
                git_commits=tuple(payload.get("source_git_commits") or ()),
            )
        )
        for record in matrix_records:
            for required in _REQUIRED_RECORD_FIELDS:
                if required not in record:
                    raise EmpiricalRuleError(f"{resolved}: record missing {required!r}")
            tagged = dict(record)
            tagged["_source_index"] = source_index
            records.append(tagged)
    return records, sources


def controlled_axis_pairs(
    records: Sequence[Mapping[str, Any]],
    axis: str,
    outcome_key: str,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Pairs identical on every other matching field, differing only on ``axis``.

    Pairing never crosses matrices (``_source_index`` is part of the match
    key), so records from different layers, protocols, or collectors are
    never treated as controls for one another.  Records without a finite
    outcome are excluded before matching rather than silently losing inside a
    pair.
    """
    if axis not in CATEGORICAL_AXES:
        raise EmpiricalRuleError(
            f"axis must be one of {CATEGORICAL_AXES}, got {axis!r}"
        )
    excluded = (
        {axis} | _DERIVED_LABEL_FIELDS | AXIS_DEPENDENT_FIELDS.get(axis, frozenset())
    )
    complement = tuple(name for name in MATCHING_FIELDS if name not in excluded)
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for record in records:
        outcome = record.get(outcome_key)
        if not isinstance(outcome, (int, float)):
            continue
        key = (
            record.get("_source_index"),
            *(_canonical(record.get(name)) for name in complement),
        )
        buckets.setdefault(key, []).append(record)
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        ordered = sorted(
            bucket, key=lambda r: (str(_canonical(r.get(axis))), r["candidate_id"])
        )
        for i, first in enumerate(ordered):
            for second in ordered[i + 1 :]:
                if _canonical(first.get(axis)) != _canonical(second.get(axis)):
                    pairs.append((first, second))
    return pairs


def fit_axis_rule(
    records: Sequence[Mapping[str, Any]],
    axis: str,
    *,
    outcome_key: str = "mean_layer_relative_mse",
    lower_is_better: bool = True,
    min_pairs: int = 3,
) -> AxisRule:
    """Fit one axis rule from controlled pairs, failing closed when thin.

    A directional verdict is only emitted when every pair compares the same
    two axis values and at least ``min_pairs`` such pairs exist; mixed
    three-way axes report ``heterogeneous_values`` so a caller cannot quote a
    direction the pairs do not actually share.
    """
    pairs = controlled_axis_pairs(records, axis, outcome_key)
    value_pairs = {
        tuple(sorted((str(_canonical(a.get(axis))), str(_canonical(b.get(axis))))))
        for a, b in pairs
    }
    if not pairs or len(pairs) < min_pairs:
        return AxisRule(
            axis=axis,
            outcome_key=outcome_key,
            lower_is_better=lower_is_better,
            pair_count=len(pairs),
            values=None,
            wins_first=0,
            wins_second=0,
            ties=0,
            mean_outcome_delta_first_minus_second=None,
            verdict="insufficient_evidence",
        )
    if len(value_pairs) > 1:
        return AxisRule(
            axis=axis,
            outcome_key=outcome_key,
            lower_is_better=lower_is_better,
            pair_count=len(pairs),
            values=None,
            wins_first=0,
            wins_second=0,
            ties=0,
            mean_outcome_delta_first_minus_second=None,
            verdict="heterogeneous_values",
        )
    first_value, second_value = sorted(value_pairs.pop())
    wins_first = wins_second = ties = 0
    deltas: list[float] = []
    pair_records: list[dict[str, Any]] = []
    for a, b in pairs:
        if str(_canonical(a.get(axis))) != first_value:
            a, b = b, a
        delta = float(a[outcome_key]) - float(b[outcome_key])
        deltas.append(delta)
        better_first = delta < 0 if lower_is_better else delta > 0
        if delta == 0:
            ties += 1
        elif better_first:
            wins_first += 1
        else:
            wins_second += 1
        pair_records.append(
            {
                "first_candidate": a["candidate_id"],
                "second_candidate": b["candidate_id"],
                "outcome_first": float(a[outcome_key]),
                "outcome_second": float(b[outcome_key]),
                "active_parameters_first": a.get("active_parameters"),
                "active_parameters_second": b.get("active_parameters"),
            }
        )
    mean_delta = sum(deltas) / len(deltas)
    preferred = (
        first_value
        if (mean_delta < 0) == lower_is_better and mean_delta != 0
        else second_value
    )
    dominant = max(wins_first, wins_second)
    minority = min(wins_first, wins_second)
    # A direction requires the winning value to take at least twice as many
    # controlled pairs as the losing one and to win at least once; 1-0 is a
    # direction (thin, but every pair agrees), 10-6 is not.
    verdict = (
        f"prefer_{preferred}"
        if dominant > 0 and dominant >= 2 * minority
        else "no_consistent_direction"
    )
    return AxisRule(
        axis=axis,
        outcome_key=outcome_key,
        lower_is_better=lower_is_better,
        pair_count=len(pairs),
        values=(first_value, second_value),
        wins_first=wins_first,
        wins_second=wins_second,
        ties=ties,
        mean_outcome_delta_first_minus_second=mean_delta,
        verdict=verdict,
        pairs=pair_records,
    )


def empirical_rule_manifest(
    matrix_paths: Iterable[str | Path],
    *,
    axes: Sequence[str] = CATEGORICAL_AXES,
    outcome_key: str = "mean_layer_relative_mse",
    lower_is_better: bool = True,
    min_pairs: int = 3,
) -> dict[str, Any]:
    """Fit every requested axis over the given matrices into one manifest."""
    records, sources = load_measured_records(matrix_paths)
    rules = [
        fit_axis_rule(
            records,
            axis,
            outcome_key=outcome_key,
            lower_is_better=lower_is_better,
            min_pairs=min_pairs,
        ).asdict()
        for axis in axes
    ]
    return {
        "schema": EMPIRICAL_RULES_SCHEMA,
        "status": "fitted",
        "claim_boundary": (
            "Rules are descriptive controlled-pair summaries of the exact "
            "campaigns listed under sources -- typically one teacher, one "
            "layer, one seed per cell. They are not causal, not cross-model, "
            "and rules fitted on a local metric must not be quoted as "
            "end-to-end conclusions (local relative MSE did not reliably "
            "predict recovered quality in the Pythia-70M campaign). "
            "Revalidate prospectively before using any rule as a selector."
        ),
        "outcome_key": outcome_key,
        "lower_is_better": lower_is_better,
        "min_pairs": int(min_pairs),
        "record_count": len(records),
        "sources": [
            {
                "path": source.path,
                "sha256": source.sha256,
                "record_count": source.record_count,
                "git_commits": list(source.git_commits),
            }
            for source in sources
        ],
        "rules": rules,
    }
