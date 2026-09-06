#!/usr/bin/env python3
"""Develop, freeze, seal predictions, train, then evaluate the finite-data bridge."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from bridge import (ALPHA_GRID, CALIBRATION_NOISE, CALIBRATION_SIZES, FAMILIES,
    FAILURE_LOSS, STRUCTURE, TIE_TOLERANCE, adaptive_tree, arrays_hash, candidates,
    estimate_coefficients, input_gradient_covariance, pilot, sample,
    score_coefficients, select_scores, task, train_candidate, tree_from_payload,
    tree_payload)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "source_data/morphology_calibration"
DEVELOPMENT_SEEDS = tuple(range(720910, 720915))
CONFIRMATORY_SEEDS = tuple(range(820910, 820930))
TRAINING = dict(training_rows=1024, validation_rows=512, test_rows=4096,
                training_noise=0.3, sweeps=32, restarts=2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def immutable(path, payload):
    if path.exists():
        assert json.loads(path.read_text()) == payload, f"Frozen artifact mismatch: {path}"
    else:
        dump(path, payload)


def source_hashes():
    files = [HERE/name for name in ("bridge.py", "run.py", "test_bridge.py")] + [STRUCTURE/name for name in
        ("model.py", "constructive.py", "constructive_dp_v2.py")]
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(files)}


def base_protocol():
    return dict(schema="finite_noisy_morphology_calibration/v1", model=
        "Eight Rademacher input leaves;seven bilinear internal nodes;28 coefficients;14 edges;root-only output",
        candidates=[tree_payload(tree) for tree in candidates()],
        families=list(FAMILIES), task_generation=
        "Independent permutations, Rademacher signs, amplitudes Uniform(.5,1.5), coefficient-L2 normalization; random control selects four distinct masks uniformly from all degree2..8 masks independently of candidates",
        known_prior="Eight named Rademacher inputs, generic multilinear polynomial dictionary of all255 nonconstant monomials; sparse estimation prior; no family/support/true-coefficient access",
        distribution_scope="New signed/random-amplitude functions; covariance is NOT held isospectral across these tasks; old exact experiment supplies that separate control",
        calibration_sizes=list(CALIBRATION_SIZES), calibration_noise_sd=list(CALIBRATION_NOISE),
        primary_condition=dict(calibration_rows=256, calibration_noise_sd=0.5),
        calibration_split="First75% fit, remaining25% independent-draw gate; both selector and pilot have the same total query budget; samples have independent fresh noise even when input patterns repeat",
        estimator="Lasso all255 Walsh monomials, intercept separately fitted then excluded from interaction score, alpha=c*sd(fit_y)*sqrt(2log256/n_fit), cyclic coordinate descent,tol1e-8,max20000",
        alpha_grid=list(ALPHA_GRID), selector=
        "Smallest estimated centered-cut maximum; among scores within1e-10 choose smallest summed cut tails; among those within1e-10 choose lexicographic candidate. Sum is heuristic only, not a bound.",
        rank_baseline="99.5%-energy input-gradient covariance rank inferred from same estimated polynomial; development lookup to candidate, no family labels, bins with fewer than2 development tasks use development-best fixed",
        pilot="2ALS sweeps,one initialization per candidate,fit calibration75%,score remaining25%;not continued into final training",
        training=TRAINING, restart_selection="Minimum independent noisy validation MSE;never test or full-population error",
        streams="Calibration IDs100..105;training10,validation11,test12,task1; SeedSequence(seed,family,stream);all independent draws, input patterns may repeat",
        primary_endpoint="Independent4096-draw clean-target MSE normalized by true target variance1; noisy-test MSE and exact256-point population error are secondary evaluation-only outputs",
        primary_regret="Policy test NMSE minus retrospective minimum test NMSE over12 fixed candidates, each with validation-selected restart",
        primary_comparisons=["estimated_cut versus development_best_fixed", "estimated_cut versus two_sweep_pilot"],
        meaningful_margin_nmse=0.01, inference=
        "20paired seed blocks;four families averaged within seed;10000 seed bootstrap;descriptive95%CI and Bonferroni97.5% two-sided CIs for two primary comparisons;declare meaningful superiority only if adjusted lower>0 and mean improvement>=.01",
        adaptive_secondary="Estimated-polynomial DP only at256/.5, same28coefficients/14edges and training; can select outside12-tree menu, so signed regret against that menu is retained",
        oracle_references="Full-target cut choice,full-target DP,exact population errors and best-trained menu use truth only after all confirmatory calibration choices are sealed;privileged retrospective references",
        failure_policy=f"Retain every fit. Nonfinite or linear algebra failures receive declared loss {FAILURE_LOSS}; report all failure counts and convergence warnings.",
        development_seeds=list(DEVELOPMENT_SEEDS), confirmatory_seeds=list(CONFIRMATORY_SEEDS),
        source_hashes=source_hashes())


def calibration(seed, alpha_scales, stage, adaptive=False):
    directory = OUT/stage
    rows, scores, audits, payloads, arrays = [], [], [], {}, {}
    for family, family_name in enumerate(FAMILIES):
        coeff = task(seed, family)
        for m_index, count in enumerate(CALIBRATION_SIZES):
            for noise_index, noise in enumerate(CALIBRATION_NOISE):
                stream = 100 + m_index * 2 + noise_index
                x, y, _, patterns = sample(coeff, seed, family, stream, count, noise)
                nfit = 3 * count // 4
                key = f"{family_name}_{count}_{noise_index}"
                arrays[key+"_x"] = x.astype(np.int8); arrays[key+"_y"] = y
                fields = dict(seed=seed, family=family_name, family_index=family,
                    calibration_rows=count, calibration_noise_sd=noise)
                pilot_choice, pilot_rows, pilot_time = pilot(x[:nfit], y[:nfit], x[nfit:], y[nfit:],
                    seed*100 + family, candidates())
                pilots = {row["candidate_id"]: row for row in pilot_rows}
                for alpha_scale in alpha_scales:
                    estimated, diagnostic = estimate_coefficients(x[:nfit], y[:nfit], alpha_scale)
                    tick = time.perf_counter()
                    measured = score_coefficients(estimated, candidates())
                    selected = select_scores(measured)
                    score_seconds = time.perf_counter()-tick
                    rows.append(dict(**fields, alpha_scale=alpha_scale, selected_candidate=selected,
                        pilot_candidate=pilot_choice, observed_rank995=diagnostic["observed_gradient_rank995"],
                        estimator_seconds=diagnostic["seconds"], score_seconds=score_seconds,
                        pilot_seconds=pilot_time, coefficient_nonzero=diagnostic["nonzero_coefficients"],
                        lasso_converged=diagnostic["converged"], alpha=diagnostic["alpha"]))
                    for name, bound, total in measured:
                        scores.append(dict(**fields, alpha_scale=alpha_scale, candidate_id=name,
                            estimated_max_tail=bound, estimated_sum_tail=total,
                            pilot_gate_mse=pilots[name]["pilot_gate_mse"], pilot_status=pilots[name]["pilot_status"]))
                    if adaptive and count == 256 and noise == 0.5:
                        tick = time.perf_counter()
                        tree, bound = adaptive_tree(estimated, "estimated_adaptive_dp")
                        payloads[family_name] = dict(tree=tree_payload(tree), estimated_bound=bound,
                            selection_seconds=time.perf_counter()-tick, coefficient_sha256=arrays_hash(estimated))
                audits.append(dict(**fields, stream=stream, fit_rows=nfit, gate_rows=count-nfit,
                    label_queries=count, distinct_input_patterns=int(len(np.unique(patterns))),
                    repeated_inputs_have_fresh_noise=True, sample_sha256=arrays_hash(x,y),
                    source_id_prefix=f"seed{seed}/family{family}/stream{stream}",
                    calibration_fit_gate_row_ids_disjoint=True))
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory/f"selection_{seed}.csv", index=False)
    pd.DataFrame(scores).to_csv(directory/f"scores_{seed}.csv", index=False)
    dump(directory/f"calibration_audit_{seed}.json", audits)
    np.savez_compressed(directory/f"calibration_samples_{seed}.npz", **arrays)
    if adaptive:
        dump(directory/f"adaptive_choices_{seed}.json", payloads)


def outcomes(seed, stage, config, include_adaptive=False):
    directory = OUT/stage
    rows, metadata = [], []
    chosen = json.loads((directory/f"adaptive_choices_{seed}.json").read_text()) if include_adaptive else {}
    for family, family_name in enumerate(FAMILIES):
        coeff = task(seed, family)
        trees = candidates()
        oracle_choice = select_scores(score_coefficients(coeff, trees))
        if include_adaptive:
            trees.append(tree_from_payload(chosen[family_name]["tree"]))
            oracle_tree, oracle_bound = adaptive_tree(coeff, "oracle_adaptive_dp")
            trees.append(oracle_tree)
        else:
            oracle_bound = None
        streams = []
        for stream, count in ((10,config["training_rows"]),(11,config["validation_rows"]),(12,config["test_rows"])):
            x,y,_,patterns = sample(coeff, seed, family, stream, count, config["training_noise"])
            streams.append(dict(stream=stream,rows=count,sample_sha256=arrays_hash(x,y),
                source_id_prefix=f"seed{seed}/family{family}/stream{stream}",
                distinct_patterns=int(len(np.unique(patterns)))))
        for tree in trees:
            result = train_candidate(tree, coeff, seed, family, config)
            for row in result:
                row.update(seed=seed,family=family_name,family_index=family,
                    oracle_cut_candidate=oracle_choice, parameters=28,edges=14)
                row["training_curve"] = json.dumps(row["training_curve"],sort_keys=True)
            rows.extend(result)
            print(stage,seed,family_name,tree.name,"complete",flush=True)
        metadata.append(dict(seed=seed,family=family_name,coefficients=coeff.tolist(),
            variance=float(coeff@coeff),gradient_eigenvalues=np.linalg.eigvalsh(input_gradient_covariance(coeff)).tolist(),
            streams=streams,oracle_dp_bound=oracle_bound,
            true_coefficients_used_by_calibration_selector=False))
    pd.DataFrame(rows).to_csv(directory/f"outcomes_{seed}.csv", index=False)
    dump(directory/f"outcome_audit_{seed}.json", metadata)


def develop(index):
    immutable(OUT/"development_protocol.json", base_protocol())
    seed = DEVELOPMENT_SEEDS[index]
    calibration(seed, ALPHA_GRID, "development")
    outcomes(seed, "development", TRAINING)


def freeze():
    protocol = base_protocol()
    assert json.loads((OUT/"development_protocol.json").read_text()) == protocol
    frames = [pd.read_csv(OUT/"development"/f"outcomes_{seed}.csv") for seed in DEVELOPMENT_SEEDS]
    trained = pd.concat(frames,ignore_index=True)
    selected = trained[trained.selected_by_validation].copy()
    assert len(selected) == 5*4*12
    cal = pd.concat([pd.read_csv(OUT/"development"/f"selection_{seed}.csv") for seed in DEVELOPMENT_SEEDS])
    primary = cal[(cal.calibration_rows == 256)&(cal.calibration_noise_sd == .5)]
    joined = primary.merge(selected[["seed","family","candidate_id","test_nmse"]],
        left_on=["seed","family","selected_candidate"], right_on=["seed","family","candidate_id"],validate="many_to_one")
    alpha_table = joined.groupby("alpha_scale").test_nmse.mean()
    alpha = float(alpha_table.idxmin())
    fixed = str(selected.groupby("candidate_id").test_nmse.mean().idxmin())
    rank_mapping, rank_counts = {}, {}
    primary = primary[primary.alpha_scale == alpha]
    rank_join = primary[["seed","family","observed_rank995"]].merge(selected,on=["seed","family"])
    for rank in range(9):
        subset = rank_join[rank_join.observed_rank995 == rank]
        count = len(subset)//12
        rank_mapping[str(rank)] = str(subset.groupby("candidate_id").test_nmse.mean().idxmin()) if count >= 2 else fixed
        rank_counts[str(rank)] = count
    protocol.update(selected_alpha_scale=alpha,development_best_fixed=fixed,
        development_alpha_objectives={str(k):float(v) for k,v in alpha_table.items()},
        rank_candidate_mapping=rank_mapping,rank_development_task_counts=rank_counts)
    immutable(OUT/"protocol.json", protocol)
    if not (OUT/"protocol_freeze.json").exists():
        dump(OUT/"protocol_freeze.json",dict(protocol_sha256=sha(OUT/"protocol.json"),
            utc=pd.Timestamp.now(tz="UTC").isoformat(),python=platform.python_version(),
            numpy=np.__version__,development_hashes={str(p.relative_to(OUT)):sha(p)
                for p in sorted((OUT/"development").glob("*")) if p.is_file()},
            confirmatory_outcomes_exist=bool(list((OUT/"confirmatory").glob("outcomes_*")))))
    assert not json.loads((OUT/"protocol_freeze.json").read_text())["confirmatory_outcomes_exist"]
    print(json.dumps({key:protocol[key] for key in ("selected_alpha_scale","development_best_fixed","rank_candidate_mapping","development_alpha_objectives")},indent=2))


def frozen_protocol():
    protocol = json.loads((OUT/"protocol.json").read_text())
    assert protocol["source_hashes"] == source_hashes(), "Frozen source changed"
    assert sha(OUT/"protocol.json") == json.loads((OUT/"protocol_freeze.json").read_text())["protocol_sha256"]
    return protocol


def select(index):
    protocol = frozen_protocol()
    assert not (OUT/"confirmatory_selection_seal.json").exists(), "Selection phase already sealed"
    assert not list((OUT/"confirmatory").glob("outcomes_*")), "Cannot select after confirmatory outcomes"
    calibration(CONFIRMATORY_SEEDS[index], [protocol["selected_alpha_scale"]], "confirmatory", adaptive=True)


def seal():
    frozen_protocol()
    paths=[]
    for seed in CONFIRMATORY_SEEDS:
        for stem,suffix in (("selection","csv"),("scores","csv"),("calibration_audit","json"),
                            ("calibration_samples","npz"),("adaptive_choices","json")):
            path=OUT/"confirmatory"/f"{stem}_{seed}.{suffix}"
            assert path.exists(),path
            paths.append(path)
    assert not list((OUT/"confirmatory").glob("outcomes_*"))
    payload=dict(utc=pd.Timestamp.now(tz="UTC").isoformat(),protocol_sha256=sha(OUT/"protocol.json"),
        selection_source_hashes={str(path.relative_to(OUT)):sha(path) for path in paths},
        seed_count=20,task_count=80,calibration_conditions=6,
        all_candidate_final_training_and_test_outcomes_unopened=True,
        declared_two_sweep_calibration_pilot_already_observed=True)
    assert not (OUT/"confirmatory_selection_seal.json").exists()
    dump(OUT/"confirmatory_selection_seal.json",payload)


def confirm(index):
    protocol=frozen_protocol(); seed=CONFIRMATORY_SEEDS[index]
    seal=json.loads((OUT/"confirmatory_selection_seal.json").read_text())
    assert seal["protocol_sha256"]==sha(OUT/"protocol.json")
    for name,value in seal["selection_source_hashes"].items():
        if str(seed) in name:
            assert sha(OUT/name)==value,name
    outcomes(seed,"confirmatory",protocol["training"],include_adaptive=True)


def bootstrap(values, confidence=.95):
    values=np.asarray(values,dtype=float)
    generator=np.random.default_rng(650905)
    means=values[generator.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    q=(1-confidence)/2
    return list(map(float,np.quantile(means,[q,1-q])))


def summarize():
    protocol=frozen_protocol()
    seal=json.loads((OUT/"confirmatory_selection_seal.json").read_text())
    for path,value in seal["selection_source_hashes"].items():assert sha(OUT/path)==value,path
    all_fit=pd.concat([pd.read_csv(OUT/"confirmatory"/f"outcomes_{seed}.csv") for seed in CONFIRMATORY_SEEDS],ignore_index=True)
    assert len(all_fit)==20*4*14*2
    all_fit.to_csv(OUT/"all_fit_outcomes.csv",index=False)
    end=all_fit[all_fit.selected_by_validation].copy(); assert len(end)==20*4*14
    end.to_csv(OUT/"candidate_outcomes.csv",index=False)
    cal=pd.concat([pd.read_csv(OUT/"confirmatory"/f"selection_{seed}.csv") for seed in CONFIRMATORY_SEEDS],ignore_index=True)
    cal.to_csv(OUT/"calibration_selection_records.csv",index=False)
    rows=[]; fixed_names={tree.name for tree in candidates()}
    for _,selection in cal.iterrows():
        group=end[(end.seed==selection.seed)&(end.family==selection.family)].set_index("candidate_id")
        fixed_group=group.loc[sorted(fixed_names)]
        oracle=float(fixed_group.test_nmse.min())
        choices={"estimated_cut":selection.selected_candidate,"two_sweep_pilot":selection.pilot_candidate,
            "development_best_fixed":protocol["development_best_fixed"],
            "observed_rank_only":protocol["rank_candidate_mapping"][str(int(selection.observed_rank995))],
            "fixed_balanced":"balanced_p0","fixed_comb":"comb_p0","fixed_mixed":"mixed_p0",
            "oracle_full_target_cut":str(fixed_group.oracle_cut_candidate.iloc[0])}
        if selection.calibration_rows==256 and selection.calibration_noise_sd==.5:
            choices.update(estimated_adaptive_dp="estimated_adaptive_dp",oracle_adaptive_dp="oracle_adaptive_dp")
        common=dict(seed=int(selection.seed),family=selection.family,calibration_rows=int(selection.calibration_rows),
            calibration_noise_sd=float(selection.calibration_noise_sd),observed_rank995=int(selection.observed_rank995),
            oracle_menu_test_nmse=oracle,label_queries=int(selection.calibration_rows))
        for policy,name in choices.items():
            record=group.loc[name]
            rows.append(dict(**common,policy=policy,selected_candidate=name,test_nmse=float(record.test_nmse),
                regret=float(record.test_nmse)-oracle,exact_population_nmse=float(record.exact_population_nmse)))
        rows.append(dict(**common,policy="uniform_random_expectation",selected_candidate="exact_expectation",
            test_nmse=float(fixed_group.test_nmse.mean()),regret=float(fixed_group.test_nmse.mean())-oracle,
            exact_population_nmse=float(fixed_group.exact_population_nmse.mean())))
        rows.append(dict(**common,policy="oracle_best_trained_menu",selected_candidate=str(fixed_group.test_nmse.idxmin()),
            test_nmse=oracle,regret=0.,exact_population_nmse=float(fixed_group.loc[fixed_group.test_nmse.idxmin()].exact_population_nmse)))
    policies=pd.DataFrame(rows);policies.to_csv(OUT/"policy_outcomes.csv",index=False)
    summaries=[]
    for family_label,data in [("all",policies)]+list(policies.groupby("family")):
        for (count,noise,policy),group in data.groupby(["calibration_rows","calibration_noise_sd","policy"]):
            units=group.groupby("seed")[["test_nmse","regret"]].mean()
            lo,hi=bootstrap(units.regret)
            summaries.append(dict(family=family_label,calibration_rows=count,calibration_noise_sd=noise,policy=policy,
                seed_count=len(units),mean_test_nmse=units.test_nmse.mean(),mean_regret=units.regret.mean(),
                regret_ci_low=lo,regret_ci_high=hi))
    pd.DataFrame(summaries).to_csv(OUT/"policy_summary.csv",index=False)
    primary=policies[(policies.calibration_rows==256)&(policies.calibration_noise_sd==.5)]
    contrasts=[]
    for baseline in ("development_best_fixed","two_sweep_pilot","observed_rank_only","uniform_random_expectation"):
        pivot=primary[primary.policy.isin((baseline,"estimated_cut"))].pivot(index=["seed","family"],columns="policy",values="regret")
        difference=pivot[baseline]-pivot.estimated_cut
        difference.rename("baseline_minus_selector_regret").reset_index().assign(baseline=baseline).to_csv(
            OUT/f"paired_contrast_{baseline}.csv",index=False)
        units=difference.groupby(level="seed").mean();lo,hi=bootstrap(units)
        adjusted=bootstrap(units,.975)
        contrasts.append(dict(baseline=baseline,mean_improvement=units.mean(),ci95_low=lo,ci95_high=hi,
            ci975_low=adjusted[0],ci975_high=adjusted[1],seed_count=len(units),
            primary_comparison=baseline in ("development_best_fixed","two_sweep_pilot"),
            meaningful_superiority=bool(adjusted[0]>0 and units.mean()>=.01)))
    pd.DataFrame(contrasts).to_csv(OUT/"primary_contrasts.csv",index=False)
    timing=cal.groupby(["calibration_rows","calibration_noise_sd"])[["estimator_seconds","score_seconds","pilot_seconds"]].median().reset_index()
    timing.to_csv(OUT/"calibration_timing.csv",index=False)
    report=dict(status="complete",protocol_sha256=sha(OUT/"protocol.json"),
        selection_seal_sha256=sha(OUT/"confirmatory_selection_seal.json"),all_sealed_input_hashes_verified=True,
        independent_seed_blocks=20,task_count=80,calibration_conditions=6,
        final_fit_count=len(all_fit),final_fit_failures=int((all_fit.status!="completed").sum()),
        lasso_nonconvergences=int((~cal.lasso_converged).sum()),
        alpha=protocol["selected_alpha_scale"],training=TRAINING,
        primary_contrasts=contrasts,retrospective_oracle_information_excluded_from_calibration=True,
        exact_population_used_only_after_seal=True,
        limits=["Synthetic algebraic tree with known8variableWalsh dictionary, not biophysical neuron or task-general compiler",
                "New random coefficients change covariance; this test is not the earlier isospectral capacity theorem",
                "AdaptiveDP secondary exceeds the12fixed-candidate menu; signed menu regret may be negative",
                "Independent repeated input draws can share patterns; their observation noise and source IDs are independent"])
    dump(OUT/"report.json",report)
    print(pd.DataFrame(contrasts).to_string(index=False))
    print(pd.DataFrame(summaries).query("family=='all' and calibration_rows==256 and calibration_noise_sd==.5").to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["develop","freeze","select","seal","confirm","summarize"])
    parser.add_argument("--index",type=int,default=0);args=parser.parse_args()
    if args.action=="develop":develop(args.index)
    elif args.action=="freeze":freeze()
    elif args.action=="select":select(args.index)
    elif args.action=="seal":seal()
    elif args.action=="confirm":confirm(args.index)
    else:summarize()
