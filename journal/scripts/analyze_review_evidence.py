#!/usr/bin/env python3
"""Reanalyse frozen evidence after review; no historical fits are changed.

Scores are moments of the actual routed update, not an assumed linear map of
the exact gradient. Historical training was full batch. Consequently the
deterministic score is the recipe-matched score; the archived size-64 noise is
a hypothetical minibatch diagnostic. All selection analyses are retrospective.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

import analyze_credit_phase_existing as existing

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
OUT = SOURCE / "review_evidence_reanalysis"
SEED = 20260906
BOOT = 20000
KEY = ["seed", "condition_id", "architecture", "feedback_family", "budget_k"]
CONTROLS = ["within_neuron_route_derangement", "depth_interleaved_bins",
            "random_sparse_matched", "random_rank_k"]
ANCESTRY = "correct_ancestry_subtrees"


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, float_format="%.12g")


def interval(values, offset=0):
    x = np.asarray(values, float)
    rng = np.random.default_rng(SEED + offset)
    means = x[rng.integers(0, len(x), size=(BOOT, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * (len(p)-np.arange(len(p)))))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def exact_signflip(values):
    """Exhaust all paired signs; statistic is the mean (sum is equivalent)."""
    values=np.asarray(values,float)
    sums=np.zeros(1)
    for value in values:
        sums=np.concatenate([sums+value,sums-value])
    return float(np.mean(np.abs(sums)>=abs(values.sum())-1e-10))


def rho(x, y):
    return float(spearmanr(x, y).statistic) if np.ptp(x) > 1e-12 and np.ptp(y) > 1e-12 else np.nan


def global_curvature(cfg):
    rows = []
    for seed in cfg["confirmatory_seeds"]:
        data, _, _, initial, _, _ = existing.reconstruct(seed, cfg)
        scale = float(cfg["training"]["gradient_context_rescaling"])
        maximum = 0.0
        for context in np.unique(data.context):
            x = data.x[data.context == context, int(context), :]
            maximum = max(maximum, float(np.linalg.eigvalsh(scale*x.T@x/(4*len(data.label))).max()))
        local = existing.objective_smoothness(data, initial, scale)
        assert maximum >= local - 1e-12
        rows.append(dict(seed=seed, global_logistic_smoothness=maximum,
                         initial_hessian_spectral_norm=local,
                         global_over_initial=maximum/local))
    return pd.DataFrame(rows)


def moment_scores(cfg):
    archived = pd.read_csv(SOURCE / "credit_phase_existing/operator_metrics.csv")
    outcomes = pd.read_csv(SOURCE / "trained_subtree_address_full_factorial/seed_outcomes.csv")
    curvature = global_curvature(cfg)
    frame = archived.merge(curvature, on="seed", validate="many_to_one").merge(
        outcomes, on=KEY, validate="one_to_one")
    assert len(frame) == len(archived) == 2700
    assert np.max(np.abs(frame.objective_smoothness-frame.initial_hessian_spectral_norm)) < 1e-8
    r = frame.retained_signal_inner_product.to_numpy()
    v = frame.operator_gradient_norm_sq.to_numpy()
    noise = frame.admitted_gradient_noise.to_numpy()
    L = frame.global_logistic_smoothness.to_numpy()
    assert np.all(np.isfinite(np.concatenate([r,v,noise,L])))
    assert np.all(v>0) and np.all(noise>=0) and np.all(L>0)
    eta = float(cfg["training"]["learning_rate"])
    for name, second in [("deterministic", v), ("hypothetical_minibatch64", v+noise)]:
        frame[f"{name}_second_moment"] = second
        frame[f"{name}_maximum_bound_decrease"] = np.maximum(r, 0)**2/(2*L*second)
        frame[f"{name}_bound_at_training_step"] = eta*r-0.5*L*eta**2*second
        frame[f"{name}_optimal_bound_step"] = np.maximum(r, 0)/(L*second)
    write(curvature, "global_logistic_curvature_by_seed.csv")
    write(frame, "moment_scores.csv")
    return frame


def validate_actual_updates(frame,cfg):
    rows=[]
    for seed in [min(cfg["confirmatory_seeds"]),max(cfg["confirmatory_seeds"])]:
        train,conditions,permutations,initial_task,initial,routes=existing.reconstruct(seed,cfg)
        scale=float(cfg["training"]["gradient_context_rescaling"])
        gradient=existing.FACTORIAL.gradients_all(initial,train,routes,conditions,permutations,scale)
        eta=float(cfg["training"]["learning_rate"])
        before=existing.FACTORIAL.loss_accuracy(existing.FACTORIAL.logits_all(initial,train,conditions,permutations),train.label)[0]*scale
        after=existing.FACTORIAL.loss_accuracy(existing.FACTORIAL.logits_all(initial-eta*gradient,train,conditions,permutations),train.label)[0]*scale
        reference={a:next(i for i,c in enumerate(conditions) if c.architecture_index==a and c.family=="exact_compartment_transport") for a in range(len(permutations))}
        metrics=frame[frame.seed.eq(seed)].set_index("condition_id")
        maximum_moment_error=0.
        minimum_bound_slack=np.inf
        for i,c in enumerate(conditions):
            exact=gradient[reference[c.architecture_index]].ravel()
            current=gradient[i].ravel()
            row=metrics.loc[c.condition_id]
            error=max(abs(exact@current-row.retained_signal_inner_product),abs(current@current-row.operator_gradient_norm_sq))
            maximum_moment_error=max(maximum_moment_error,error)
            slack=before[i]-after[i]-row.deterministic_bound_at_training_step
            minimum_bound_slack=min(minimum_bound_slack,slack)
        assert maximum_moment_error<1e-8
        assert minimum_bound_slack>=-1e-8
        rows.append(dict(seed=seed,n_conditions=len(conditions),maximum_archived_moment_reconstruction_error=maximum_moment_error,
            minimum_actual_fullbatch_decrease_minus_global_bound=minimum_bound_slack,
            all_actual_step_bounds_pass=True))
    result=pd.DataFrame(rows)
    write(result,"actual_update_validation.csv")
    return rows


def pooled_correlations(frame):
    """Seed-block bootstrap; duplicate architecture implementations excluded."""
    part = frame[frame.architecture.eq("dendritic_tree")].sort_values(["seed", "condition_id"])
    seeds = sorted(part.seed.unique())
    blocks = np.stack([part.index[part.seed.eq(s)].to_numpy() for s in seeds])
    # Reset index, then use a compact index array for repeated seed-block samples.
    part = part.reset_index(drop=True)
    blocks = np.arange(len(part)).reshape(len(seeds), -1)
    rng = np.random.default_rng(SEED)
    boot_index = blocks[rng.integers(0, len(seeds), (5000, len(seeds)))].reshape(5000, -1)
    scores = ["maximum_guaranteed_decrease", "deterministic_maximum_bound_decrease",
              "hypothetical_minibatch64_maximum_bound_decrease",
              "deterministic_bound_at_training_step", "initial_gradient_capture", "budget_k"]
    rows = []
    for target in ["heldout_accuracy", "norm_matched_one_step_progress"]:
        y = part[target].to_numpy()
        ry = rankdata(y[boot_index], axis=1)
        ry -= ry.mean(axis=1, keepdims=True)
        yden = np.sum(ry**2, axis=1)
        for score in scores:
            x = part[score].to_numpy()
            rx = rankdata(x[boot_index], axis=1)
            rx -= rx.mean(axis=1, keepdims=True)
            values = np.sum(rx*ry, axis=1)/np.sqrt(np.sum(rx**2, axis=1)*yden)
            lo, hi = np.quantile(values, [.025, .975])
            rows.append(dict(score=score, endpoint=target, spearman_rho=rho(x,y),
                             ci95_low=lo, ci95_high=hi, n_seed_blocks=len(seeds),
                             n_conditions=len(part), bootstrap_draws=len(values)))
    result = pd.DataFrame(rows)
    write(result, "pooled_correlations.csv")
    return result


def selection(frame):
    """Within-family K selection; constants abstain, selectable ties pick small K."""
    part = frame[frame.architecture.eq("dendritic_tree") & frame.feedback_family.isin(
        [ANCESTRY] + CONTROLS + ["learned_rank_k_upper_bound"])].copy()
    criteria = ["deterministic_maximum_bound_decrease",
                "hypothetical_minibatch64_maximum_bound_decrease",
                "deterministic_bound_at_training_step", "initial_gradient_capture", "budget_k"]
    records = []
    for (seed, family), group in part.groupby(["seed", "feedback_family"]):
        group = group.sort_values("budget_k")
        for criterion in criteria:
            values = group[criterion].to_numpy()
            degenerate = np.ptp(values) <= 1e-12
            if degenerate:
                chosen = None
            else:
                chosen = group.loc[group[criterion].idxmax()]
            for endpoint in ["heldout_accuracy", "norm_matched_one_step_progress"]:
                best = group[endpoint].max()
                records.append(dict(seed=seed, feedback_family=family, criterion=criterion,
                    endpoint=endpoint, degenerate=degenerate,
                    predicted_k=np.nan if chosen is None else int(chosen.budget_k),
                    observed_best_k=int(group.loc[group[endpoint].idxmax(), "budget_k"]),
                    selected_endpoint=np.nan if chosen is None else float(chosen[endpoint]),
                    best_endpoint=best,
                    selection_regret=np.nan if chosen is None else max(0., best-float(chosen[endpoint])),
                    argmax_match=np.nan if chosen is None else bool(np.isclose(chosen[endpoint], best, atol=1e-10, rtol=0)),
                    within_family_spearman=rho(values, group[endpoint].to_numpy())))
    selected = pd.DataFrame(records)
    summaries = []
    for i, ((family, criterion, endpoint), g) in enumerate(selected.groupby(["feedback_family", "criterion", "endpoint"])):
        valid = g[~g.degenerate]
        mean, low, high = interval(valid.selection_regret, i) if len(valid) else (np.nan,)*3
        summaries.append(dict(feedback_family=family, criterion=criterion, endpoint=endpoint,
            n_seed_curves=len(g), n_degenerate=int(g.degenerate.sum()),
            n_scored=len(valid), match_count=int(valid.argmax_match.sum()),
            match_fraction=valid.argmax_match.mean(), mean_regret=mean,
            regret_ci95_low=low, regret_ci95_high=high,
            mean_within_family_spearman=valid.within_family_spearman.mean()))
    group_curves = part.groupby(["feedback_family", "budget_k"], as_index=False)[criteria+[
        "heldout_accuracy", "norm_matched_one_step_progress"]].mean()
    group_rows=[]
    for family, g in group_curves.groupby("feedback_family"):
        g=g.sort_values("budget_k")
        for criterion in criteria:
            varying = np.ptp(g[criterion]) > 1e-12
            chosen = g.loc[g[criterion].idxmax()]
            group_rows.append(dict(feedback_family=family, criterion=criterion, degenerate=not varying,
                predicted_k=int(chosen.budget_k) if varying else np.nan,
                observed_best_k=int(g.loc[g.heldout_accuracy.idxmax(), "budget_k"]),
                accuracy_regret=float(g.heldout_accuracy.max()-chosen.heldout_accuracy) if varying else np.nan,
                within_family_spearman=rho(g[criterion], g.heldout_accuracy)))
    write(selected,"within_family_seed_predictions.csv")
    write(pd.DataFrame(summaries),"within_family_prediction_summary.csv")
    write(group_curves,"within_family_mean_curves.csv")
    write(pd.DataFrame(group_rows),"within_family_mean_predictions.csv")
    # Count unique numerical utility curves across representations, no pseudoreplication.
    allcurves = frame.groupby(["architecture","feedback_family","budget_k"],as_index=False).deterministic_maximum_bound_decrease.mean()
    signatures=[]
    curve_map=[]
    unique_curves=[]
    for (arch,family), g in allcurves.groupby(["architecture","feedback_family"]):
        if len(g)>1:
            v=g.sort_values("budget_k").deterministic_maximum_bound_decrease.to_numpy()
            if np.ptp(v)>1e-12:
                signatures.append(tuple(np.round(v,10)))
                match=next((j for j,u in enumerate(unique_curves) if np.allclose(v,u,rtol=1e-8,atol=1e-10)),None)
                if match is None:
                    match=len(unique_curves)
                    unique_curves.append(v)
                reference=allcurves[allcurves.architecture.eq("dendritic_tree") & allcurves.feedback_family.eq(family)].sort_values("budget_k").deterministic_maximum_bound_decrease.to_numpy()
                curve_map.append(dict(architecture=arch,feedback_family=family,
                    numerical_curve_id=match,maximum_absolute_difference_from_dendritic_reference=np.max(np.abs(v-reference)),
                    mean_utility_k1=v[0],mean_utility_k2=v[1],mean_utility_k4=v[2],mean_utility_k8=v[3]))
    write(pd.DataFrame(curve_map),"unique_curve_mapping.csv")
    return dict(nonconstant_representation_family_cells=len(signatures),
                distinct_nonconstant_mean_utility_curves=len(unique_curves),
                numerical_equivalence_rtol=1e-8,numerical_equivalence_atol=1e-10,
                total_dendritic_seed_family_curves=len(part.groupby(["seed","feedback_family"])))


def ancestry_contrasts(frame):
    part=frame[frame.architecture.eq("dendritic_tree")]
    wide=part.pivot(index=["seed","budget_k"],columns="feedback_family",values="heldout_accuracy")
    # Each accuracy is an integer count among the same 2,048 test examples.
    # Restore its exact grid before ranking ties; CSV decimal truncation otherwise
    # spuriously separates theoretically identical absolute paired differences.
    counts=np.rint(wide*2048)
    assert np.nanmax(np.abs(wide*2048-counts))<1e-6
    wide=counts/2048
    rows=[]
    definitions={name:[name] for name in CONTROLS+["learned_rank_k_upper_bound"]}
    definitions["best_matched_nonanatomical_oracle"]=CONTROLS
    definitions["legacy_interior_oracle_including_pca"]=["depth_interleaved_bins","learned_rank_k_upper_bound","random_rank_k","random_sparse_matched"]
    for control, names in definitions.items():
        contrast=100*(wide[ANCESTRY]-wide[names].max(axis=1))
        for (seed,k), value in contrast.items():
            rows.append(dict(seed=seed,budget_k=k,control=control,accuracy_difference_pp=value,
                comparator_families=";".join(names)))
    differences=pd.DataFrame(rows)
    summaries=[]
    for i, ((k,control), g) in enumerate(differences.groupby(["budget_k","control"])):
        x=g.accuracy_difference_pp.to_numpy()
        mean,lo,hi=interval(x,2000+i)
        p=1. if np.allclose(x,0,atol=1e-10) else float(wilcoxon(x,method="auto").pvalue)
        summaries.append(dict(budget_k=k,control=control,n_pairs=len(x),mean_difference_pp=mean,
             ci95_low_pp=lo,ci95_high_pp=hi,positive_seeds=int((x>1e-10).sum()),
             tied_seeds=int((np.abs(x)<=1e-10).sum()),p_two_sided_wilcoxon=p,
             p_exact_two_sided_mean_signflip=exact_signflip(x)))
    summary=pd.DataFrame(summaries)
    summary["p_holm_four_budgets"]=np.nan
    for control,g in summary.groupby("control"):
        summary.loc[g.index,"p_holm_four_budgets"]=holm(g.p_two_sided_wilcoxon)
    summary["p_holm_four_individual_controls_at_k4"]=np.nan
    idx=summary.index[summary.budget_k.eq(4)&summary.control.isin(CONTROLS)]
    summary.loc[idx,"p_holm_four_individual_controls_at_k4"]=holm(summary.loc[idx,"p_two_sided_wilcoxon"])
    summary["p_exact_mean_signflip_holm_four_budgets"]=np.nan
    for control,g in summary.groupby("control"):
        summary.loc[g.index,"p_exact_mean_signflip_holm_four_budgets"]=holm(g.p_exact_two_sided_mean_signflip)
    summary["p_exact_mean_signflip_holm_four_individual_controls_at_k4"]=np.nan
    summary.loc[idx,"p_exact_mean_signflip_holm_four_individual_controls_at_k4"]=holm(summary.loc[idx,"p_exact_two_sided_mean_signflip"])
    # Comparator-selection audit at every K, including historical PCA substitution.
    write(differences,"ancestry_control_paired_differences.csv")
    write(summary,"ancestry_control_contrasts.csv")
    write(summary[summary.budget_k.eq(4)],"ancestry_k4_control_contrasts.csv")
    optimum_rows=[]
    utility_rows=[]
    accuracy_contrast=differences[differences.control.eq("best_matched_nonanatomical_oracle")].set_index(["seed","budget_k"]).accuracy_difference_pp
    for criterion in ["deterministic_maximum_bound_decrease","hypothetical_minibatch64_maximum_bound_decrease","maximum_guaranteed_decrease"]:
        uwide=part.pivot(index=["seed","budget_k"],columns="feedback_family",values=criterion)
        ucontrast=uwide[ANCESTRY]-uwide[CONTROLS].max(axis=1)
        for (seed,k),value in ucontrast.items():
            utility_rows.append(dict(seed=seed,budget_k=k,criterion=criterion,ancestry_minus_best_control_utility=value))
        for seed in sorted(part.seed.unique()):
            u=ucontrast.loc[seed].sort_index()
            a=accuracy_contrast.loc[seed].sort_index()
            chosen=int(u.idxmax())
            optimum_rows.append(dict(seed=seed,criterion=criterion,predicted_k=chosen,
                observed_best_k=int(a.idxmax()),argmax_match=chosen==int(a.idxmax()),
                observed_k4_is_best=int(a.idxmax())==4,chosen_accuracy_contrast_pp=float(a.loc[chosen]),
                best_accuracy_contrast_pp=float(a.max()),contrast_selection_regret_pp=float(a.max()-a.loc[chosen])))
    write(pd.DataFrame(optimum_rows),"ancestry_contrast_optimum_predictions.csv")
    urows=pd.DataFrame(utility_rows)
    write(urows,"ancestry_utility_contrast_by_seed.csv")
    usummary=[]
    for i,((criterion,k),g) in enumerate(urows.groupby(["criterion","budget_k"])):
        mean,lo,hi=interval(g.ancestry_minus_best_control_utility,6000+i)
        usummary.append(dict(criterion=criterion,budget_k=k,mean=mean,ci95_low=lo,ci95_high=hi,
            positive_seeds=int((g.ancestry_minus_best_control_utility>0).sum()),n_seeds=len(g)))
    write(pd.DataFrame(usummary),"ancestry_utility_contrast_summary.csv")
    return summary[summary.budget_k.eq(4)].to_dict("records")


def native_functional():
    tree=pd.read_csv(SOURCE/"fulltree_boundary/output/cell_method_means.csv")
    original=pd.read_csv(SOURCE/"figure5/functional_target_metrics.csv",dtype={"target_root_id":str})
    scans=pd.read_csv(SOURCE/"functional_topology_all_scans/cell_metrics.csv",dtype={"target_root_id":str})
    rows=[]
    for name,data,column in [("selected_scans",original,"partial_shared_path_r"),
                             ("scan_complete",scans,"shared_path_partial_r")]:
        for _,r in data.iterrows():
            rows.append(dict(target_root_id=r.target_root_id,endpoint="structure_function_partial_r",
                comparison=name,unit="partial rank correlation",effect=r[column]))
    for endpoint,unit,sign in [("heldout_normalized_mse","normalized MSE",-1),
                                ("common_checkpoint_update_capture","update match",1)]:
        wide=tree.pivot(index="target_root_id",columns="method",values=endpoint)
        for control in ["site-shuffled routes","random anatomical routes"]:
            differences=sign*(wide["topology-matched routes"]-wide[control])
            for target,value in differences.items():
                rows.append(dict(target_root_id=str(target),endpoint=endpoint,
                    comparison=control,unit=unit,effect=value))
    values=pd.DataFrame(rows)
    summaries=[]
    for i,((endpoint,comparison,unit),g) in enumerate(values.groupby(["endpoint","comparison","unit"])):
        mean,lo,hi=interval(g.effect,4000+i)
        summaries.append(dict(endpoint=endpoint,comparison=comparison,unit=unit,n_targets=len(g),
            mean=mean,ci95_low=lo,ci95_high=hi,positive_targets=int((g.effect>0).sum())))
    write(values,"functional_native_target_effects.csv")
    write(pd.DataFrame(summaries),"functional_native_contrasts.csv")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(existing.CONFIG.read_text())
    frame=moment_scores(cfg)
    validation=validate_actual_updates(frame,cfg)
    corr=pooled_correlations(frame)
    counts=selection(frame)
    contrasts=ancestry_contrasts(frame)
    native_functional()
    report=dict(status="completed_retrospective_reanalysis",moment_definition="r=grad(L)^T E[d], second=E[||d||^2]",
        global_bound="max_context lambda_max(c X_c^T X_c/(4N)); c=8 matches stored gradient context rescaling",
        historical_optimizer="full-batch gradient descent at eta=0.3; noise-free score is recipe matched",
        accuracy_precision="Exact integer/2048 grid restored before signed-rank tests to preserve true ties obscured by decimal CSV truncation",
        test_variants="Wilcoxon uses scipy auto with tied absolute differences; separate exact paired mean-signflip enumerates all 2^20 signs. Holm families are explicitly named in CSV columns.",
        noise_status="archived 64 hypothetical minibatches of size 64, not the actual training algorithm",
        selection_status="retrospective, within fixed route families; no morphology held-out validation",
        architecture_duplicate_counts=counts, k4_contrasts=contrasts,
        direct_actual_update_validation=validation,
        correlations=corr.to_dict("records"),
        global_to_initial_curvature_range=[frame.global_over_initial.min(),frame.global_over_initial.max()],
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            existing.CONFIG,existing.OUTCOMES,SOURCE/"credit_phase_existing/operator_metrics.csv",Path(__file__),
            ROOT/"scripts/analyze_credit_phase_existing.py",ROOT/"scripts/run_trained_subtree_address_full_factorial.py"]})
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUT/"README.md").write_text("# Review evidence reanalysis\n\nRun `python scripts/analyze_review_evidence.py` from the journal directory. This secondary analysis never changes or reruns historical fits. `moment_scores.csv` separates deterministic full-batch scores from hypothetical size-64 minibatch scores and uses the global logistic curvature bound. Scores apply to actual routed-update moments. `within_family_*` excludes duplicated representations from seed-level inference and reports abstention for constant criteria, rank concordance and held-out endpoint regret. `ancestry_*` exposes each control, the original four-control oracle and the historical interior-analysis oracle that substituted PCA for route derangement. P values are two-sided paired Wilcoxon, with explicit Holm families. `functional_native_*` preserves the native units and seven-target nesting. Bootstrap means use 20,000 draws; pooled Spearman intervals use 5,000 paired seed-block draws. Every selection statistic is retrospective and is not a held-out morphology-selection experiment.\n")
    with (OUT/"README.md").open("a") as handle:
        handle.write("\nAccuracy is reconstructed as exact integer counts among 2,048 test trials before rank tests; the decimal CSV otherwise spuriously separates tied absolute differences. Wilcoxon uses SciPy's automatic treatment of ties. Separately named exact paired mean-signflip P values enumerate all 2^20 assignments; do not label these Wilcoxon. Holm columns identify either four budgets or four individual controls at K=4. `actual_update_validation.csv` reconstructs both update moments and a complete actual one-step objective decrease for 135 conditions at each of two seeds; all 270 global-L bounds pass. `unique_curve_mapping.csv` counts numerical equivalence with atol=1e-10 and rtol=1e-8; unique curves are still related observations, not independent task families.\n")
    print(json.dumps({"counts":counts,"curvature_ratio":report["global_to_initial_curvature_range"],"k4":contrasts},indent=2))
    print(corr.to_string(index=False))


if __name__=="__main__":
    main()
