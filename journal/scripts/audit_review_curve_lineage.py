#!/usr/bin/env python3
"""Bounded, read-only audit of historical supplementary curve lineage.

No training, model loading, figure mutation, or numerical reanalysis occurs.
Validity below concerns the signed-input/positive-conductance issue only;
it is deliberately separate from independently verified execution lineage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT.parent
REPO = DRAFT.parents[1]
OUT = ROOT / "source_data/review_curve_lineage"
INHERITED = ROOT / "source_data/inherited_neurips"
VALIDITY = ROOT / "source_data/prospective_input_validity"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    theory = INHERITED / "theory_diag_by_condition.csv"
    oracle = INHERITED / "path_transport_upper_bound_summary.csv"
    rank = INHERITED / "noise_resilience_rank_bridge_summary.csv"
    gradient_config = DRAFT / "neurips/configs/sweeps/sweep_neurips_gradient_fidelity_vs_ie_activation_corrected.yaml"
    oracle_config = DRAFT / "neurips/configs/sweeps/sweep_neurips_path_transport_upper_bound_activation_corrected_5seed.yaml"
    rank_config = DRAFT / "neurips/configs/sweeps/sweep_neurips_noise_resilience_rank_bridge_activation_corrected.yaml"
    rank_base = DRAFT / "neurips/configs/noise_resilience_shunting_localca_strong.yaml"

    def add_legacy(source: Path, panel: str, metric: str, feedback: str,
                   config: Path, advertised_analysis: str) -> None:
        frame = pd.read_csv(source)
        frame = frame[frame.dataset.eq("noise_resilience")]
        for index, row in frame.iterrows():
            additive = row.network_type == "dendritic_additive"
            rows.append({
                "figure": "S1", "panel": panel, "record_type": "plotted_summary_point",
                "source_csv": relative(source), "source_sha256": sha(source),
                "source_csv_row_1based_including_header": int(index) + 2,
                "task": row.dataset, "core": row.network_type, "feedback": feedback,
                "x_parameter": "ie_value", "x_value": int(row.ie_value),
                "metric": metric, "plotted_mean_native_units": float(row[metric]),
                "n_runs": int(row.n_runs), "input_validity_status": "proven_valid" if additive else "unresolved",
                "validity_evidence_level": "declared_additive_model_domain" if additive else "aggregate_only",
                "execution_lineage_status": "unresolved", "run_ids_present_in_plotted_source": False,
                "resolved_config_rechecked": False,
                "available_intended_config": relative(config),
                "available_intended_config_sha256": sha(config),
                "advertised_analysis": advertised_analysis,
                "reason": ("Signed inputs are allowed in the declared additive model; execution provenance is not independently resolved."
                           if additive else "Current intended config specifies ReLU and generator advertises a nonnegative-input correction, but the aggregate has no run IDs or executed-config evidence; neither validity nor invalidity is established."),
                "publication_status": "displayed_additive_only" if additive else "archival_only_not_displayed",
            })

    for metric, feedback in [("per_soma_weighted_cosine_mean", "matched_width_fallback"),
                             ("path_transport_weighted_cosine_mean", "exact_path_diagnostic")]:
        add_legacy(theory, "D", metric, feedback, gradient_config,
                   "theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix_summary")
    add_legacy(theory, "E", "test_accuracy_mean", "matched_width_fallback", gradient_config,
               "theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix_summary")
    add_legacy(oracle, "E", "test_accuracy_mean", "exact_path_training", oracle_config,
               "path_transport_upper_bound_nonnegativeinput_fix_5seed")

    frame = pd.read_csv(rank)
    rungs = [("per_soma", 4, False), ("per_soma", 4, True),
             ("low_rank", 2, False), ("low_rank", 4, False),
             ("low_rank", 8, False), ("path_transport", 4, False)]
    for mode, budget, path_prop in rungs:
        selected = frame[(frame.broadcast_mode == mode) & (frame.broadcast_rank == budget)
                         & (frame.use_path_propagation == path_prop)]
        assert len(selected) == 1
        index, row = next(selected.iterrows())
        rows.append({
            "figure": "S3", "panel": "D", "record_type": "plotted_summary_point",
            "source_csv": relative(rank), "source_sha256": sha(rank),
            "source_csv_row_1based_including_header": int(index) + 2,
            "task": "noise_resilience", "core": "dendritic_shunting", "feedback": mode,
            "x_parameter": "broadcast_rank", "x_value": budget,
            "use_path_propagation": path_prop, "metric": "test_accuracy_mean",
            "plotted_mean_native_units": float(row.test_accuracy_mean), "n_runs": int(row.n_runs),
            "input_validity_status": "unresolved", "validity_evidence_level": "aggregate_only",
            "execution_lineage_status": "unresolved", "run_ids_present_in_plotted_source": False,
            "resolved_config_rechecked": False,
            "available_intended_config": relative(rank_config),
            "available_intended_config_sha256": sha(rank_config),
            "available_intended_base_config": relative(rank_base),
            "advertised_analysis": "rank_bridge_nonnegativeinput_fix",
            "reason": "Available referenced base config specifies ReLU; aggregate-to-executed-run linkage is absent. Related diagnostic run IDs establish an associated cohort name only, not this panel's complete run membership or execution convention.",
            "publication_status": "archival_only_not_displayed",
        })

    audit_path = VALIDITY / "historical_run_validity.csv"
    audit = pd.read_csv(audit_path)
    fixed = audit[audit.family.eq("fixed_budget")].copy()
    outcomes_path = ROOT / "source_data/prospective_followup/seed_outcomes.csv"
    outcomes = pd.read_csv(outcomes_path)
    outcomes = outcomes[outcomes.family.eq("fixed_budget")]
    merged = fixed.merge(outcomes[["run_dir", "config_index", "test_accuracy", "status", "epochs_recorded"]],
                         on=["run_dir", "config_index"], validate="one_to_one")
    assert len(merged) == 320 and merged.status.eq("pass").all()
    configs = [ROOT / row.run_dir / "results" / f"config_{int(row.config_index)}" / "config.json"
               for _, row in merged.iterrows()]
    merged["resolved_config_path"] = [relative(p) for p in configs]
    merged["resolved_config_available_for_recheck"] = [p.is_file() for p in configs]
    merged["input_validity_status"] = merged.input_convention_valid.map({True: "proven_valid", False: "proven_invalid"})
    merged["validity_evidence_level"] = "retained_run_level_validity_audit"
    merged["execution_lineage_status"] = "retained_audit_and_completed_outcome_join; resolved_config_not_available_for_recheck"
    merged["validity_audit_sha256"] = sha(audit_path)
    merged["outcome_source_sha256"] = sha(outcomes_path)
    merged.to_csv(OUT / "s8_fixed_budget_run_lineage.csv", index=False)

    for keys, group in merged.groupby(["core", "strategy", "feedback", "depth"], sort=True):
        core, strategy, feedback, depth = keys
        rows.append({
            "figure": "S8", "panel": {"backprop": "D", "per_soma": "A", "per_soma_shared": "B", "path_transport": "C"}[feedback],
            "record_type": "qualified_or_excluded_summary_point", "source_csv": relative(audit_path),
            "source_sha256": sha(audit_path), "task": "noise_resilience", "core": core,
            "strategy": strategy, "feedback": feedback, "x_parameter": "depth", "x_value": int(depth),
            "metric": "test_accuracy", "plotted_mean_native_units": group.test_accuracy.mean(),
            "n_runs": len(group), "input_validity_status": group.input_validity_status.iloc[0],
            "validity_evidence_level": "retained_run_level_validity_audit",
            "execution_lineage_status": "retained_audit_and_completed_outcome_join; resolved_config_not_available_for_recheck",
            "run_ids_present_in_plotted_source": True,
            "resolved_config_rechecked": False,
            "reason": "Existing audit script opened each resolved config and asserted null output activation for every excluded noise/shunting run. Preserved audit joins one-to-one to completed seed outcomes; raw config directories are unavailable in this checkout.",
            "publication_status": "included_qualified_additive" if group.publication_included.all() else "excluded_signed_shunting",
        })
    pd.DataFrame(rows).to_csv(OUT / "curve_lineage.csv", index=False)

    related_path = INHERITED / "error_field_decomposition_runs.csv"
    related = pd.read_csv(related_path)
    related = related[related.dataset.eq("noise_resilience")][["run_dir", "run_name", "seed", "dataset", "network_type", "error_broadcast_mode"]].drop_duplicates()
    related["resolved_config_available"] = related.run_dir.map(lambda p: (Path(p) / "config.json").is_file())
    related["link_scope"] = "related diagnostic cohort only; not a verified join to S3D aggregate"
    related.to_csv(OUT / "related_noise_diagnostic_run_ids.csv", index=False)
    files = [theory, oracle, rank, gradient_config, oracle_config, rank_config, rank_base,
             audit_path, outcomes_path, related_path,
             ROOT / "scripts/build_supplementary_figure_s01_native.py",
             ROOT / "scripts/build_supplementary_figure_s03_native.py",
             ROOT / "scripts/build_prospective_input_validity_audit.py"]
    pd.DataFrame([{"path": relative(p), "exists": p.exists(), "sha256": sha(p)} for p in files]).to_csv(OUT / "evidence_files.csv", index=False)
    summary = {
        "scope": "Signed-input validity of S1D/E, S3D and S8; not a general validation of model or results.",
        "inherited_source_summary_points": 46,
        "historical_shunting_points_removed_from_publication": 26,
        "inherited_shunting_points_unresolved": 26,
        "inherited_additive_points_unaffected_by_signed_input_criterion": 20,
        "all_inherited_aggregate_execution_lineage_unresolved": True,
        "s8_retained_run_audit": {"valid_additive_included": 160, "invalid_shunting_excluded": 160},
        "s8_existing_completed_outcome_rows_joined": len(merged),
        "s8_resolved_configs_available_for_independent_recheck": int(merged.resolved_config_available_for_recheck.sum()),
        "no_execution_inferred_from_config_names": True,
        "no_training_rerun": True,
        "proof_scope": "proven_valid/proven_invalid in the S8 rows report the retained explicit run-level validity audit, not a new inspection of missing raw configs. Additive validity concerns the signed-input domain criterion under the declared model. Legacy shunting status remains unresolved.",
        "archive_limitation": "The documented archive/pre-overleaf-prune-20260820 ref is not available in this checkout; no external/raw archive was fetched.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "README.md").write_text("""# Historical curve lineage audit

Historical S1D/E and former S3D are **unresolved for the shunting input convention**.
The revised publication removes these unsupported shunting series from display;
S1D/E retain additive results and S3 now has panels A–C. Aggregate data remain
archival sources, with `publication_status` recording their exclusion.
Their native generators consume aggregate CSVs without run IDs. The archived
generator paths explicitly name nonnegative-input corrections, and available
intended configurations specify ReLU. This contradicts an unconditional claim
that these historical shunting series used signed inputs, but does not prove their
executed convention. Current configuration intent is not execution evidence.

Five retained noise-task run IDs in a related error-field diagnostic identify
the named rank-bridge cohort, but cannot establish all six plotted rungs or
join those aggregates to resolved training configs. Raw legacy run directories
and the documented pre-prune archive ref are unavailable in this checkout.

The retained S8 run-level audit explicitly records 160 additive runs included
and 160 signed-shunting runs excluded. Its generator opens each resolved config
and asserts null output activation for excluded shunting runs. Every retained
audit row joins one-to-one to a completed outcome record. The raw resolved
configs themselves are currently unavailable, so this report preserves that
audited classification and explicitly records the independent-recheck limit.
The qualified source table and current plot generator include additive only.

`curve_lineage.csv` separates input-domain validity from execution-lineage
status. **The proof label is scoped to the stated evidence level**: an additive
model accepts signed inputs under its declared equations; S8 validity labels
reproduce the explicit retained run-level audit; no historical S1/S3 shunting
curve receives a proved-valid or proved-invalid label. This is not a general
endorsement of the calculations, implementation, or other scientific claims.

Historical shunting tables are retained only for provenance. Their executed
input convention could not be established from the retained aggregates, so
these series are absent from the current publication plots.
S8 can retain its explicit signed-shunting exclusion statement.

Rebuild with `python scripts/audit_review_curve_lineage.py`. The script only
reads existing records and writes this audit folder; it never reruns training.
""")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
