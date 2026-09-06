#!/usr/bin/env python3
"""Post-seal independent reconstruction checks; never changes frozen rules/data."""
from __future__ import annotations
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from bridge import (FAMILIES,arrays_hash,candidates,estimate_coefficients,
    score_coefficients,select_scores,task,train_candidate,cut_scores,adaptive_tree)
from run_serialization_fix_v2 import decode_numeric_order,run
OUT,CONFIRMATORY_SEEDS,TRAINING=run.OUT,run.CONFIRMATORY_SEEDS,run.TRAINING
frozen_protocol,sha,dump=run.frozen_protocol,run.sha,run.dump


def main():
    start=time.perf_counter(); protocol=frozen_protocol()
    seal=json.loads((OUT/"confirmatory_selection_seal.json").read_text())
    for filename,runner in (("serialization_amendment.json","run_serialization_fix.py"),
                            ("serialization_dispatch_amendment.json","run_serialization_fix_v2.py")):
        amendment=json.loads((OUT/filename).read_text())
        assert amendment["runtime_fix_sha256"]==sha(Path(__file__).with_name(runner))
        assert amendment["protocol_sha256"]==sha(OUT/"protocol.json")
        assert amendment["selection_seal_sha256"]==sha(OUT/"confirmatory_selection_seal.json")
    seal_time=pd.Timestamp(seal["utc"]).timestamp()
    for path,expected in seal["selection_source_hashes"].items():
        assert sha(OUT/path)==expected,path
    # Read original roundtrip floats: the convenience aggregate has undergone a
    # second CSV parse/export, which can collapse last-bit validation differences.
    all_fit=pd.concat([pd.read_csv(OUT/"confirmatory"/f"outcomes_{seed}.csv",
        float_precision="round_trip") for seed in CONFIRMATORY_SEEDS],ignore_index=True)
    assert len(all_fit)==2240
    assert not all_fit[["validation_mse","test_nmse","noisy_test_mse","exact_population_nmse"]].isna().any().any()
    assert np.isfinite(all_fit[["validation_mse","test_nmse","noisy_test_mse","exact_population_nmse"]].to_numpy()).all()
    restart_groups=0
    for _,group in all_fit.groupby(["seed","family","candidate_id"]):
        expected=group.sort_values(["validation_mse","restart"]).iloc[0]
        chosen=group[group.selected_by_validation]
        assert len(chosen)==1 and chosen.iloc[0].restart==expected.restart
        restart_groups+=1
    counts=dict(calibration_records=0,coefficient_selections_reproduced=0,
                source_stream_contracts=0,source_sample_hashes=0)
    query_count=0;max_score_delta=0.;bounds=[]
    for seed in CONFIRMATORY_SEEDS:
        path=OUT/"confirmatory"/f"outcomes_{seed}.csv"
        assert path.stat().st_mtime>=seal_time,"Outcome artifact predates global seal"
        cal=pd.read_csv(OUT/"confirmatory"/f"selection_{seed}.csv")
        scores=pd.read_csv(OUT/"confirmatory"/f"scores_{seed}.csv")
        audits=json.loads((OUT/"confirmatory"/f"calibration_audit_{seed}.json").read_text())
        outcome_audits=json.loads((OUT/"confirmatory"/f"outcome_audit_{seed}.json").read_text())
        samples=np.load(OUT/"confirmatory"/f"calibration_samples_{seed}.npz")
        assert len(cal)==24 and len(audits)==24
        for record in audits:
            noise_index=int(record["calibration_noise_sd"]>0)
            key=f'{record["family"]}_{record["calibration_rows"]}_{noise_index}'
            x=samples[key+"_x"].astype(float);y=samples[key+"_y"]
            assert arrays_hash(x,y)==record["sample_sha256"]
            nfit=record["fit_rows"]
            assert set(range(nfit)).isdisjoint(range(nfit,len(y)))
            estimate,_=estimate_coefficients(x[:nfit],y[:nfit],protocol["selected_alpha_scale"])
            recomputed=score_coefficients(estimate,candidates())
            match=cal[(cal.family==record["family"])&(cal.calibration_rows==len(y))&
                      (cal.calibration_noise_sd==record["calibration_noise_sd"])]
            assert len(match)==1 and select_scores(recomputed)==match.iloc[0].selected_candidate
            stored=scores[(scores.family==record["family"])&(scores.calibration_rows==len(y))&
                      (scores.calibration_noise_sd==record["calibration_noise_sd"])].set_index("candidate_id")
            for candidate,bound,total in recomputed:
                delta=max(abs(stored.loc[candidate,"estimated_max_tail"]-bound),
                          abs(stored.loc[candidate,"estimated_sum_tail"]-total))
                max_score_delta=max(max_score_delta,float(delta));assert delta<1e-10
            counts["calibration_records"]+=1;counts["coefficient_selections_reproduced"]+=1
            counts["source_sample_hashes"]+=1;query_count+=len(y)
        for audit in outcome_audits:
            streams={entry["stream"] for entry in audit["streams"]}
            assert streams=={10,11,12} and streams.isdisjoint(range(100,106))
            assert len({entry["sample_sha256"] for entry in audit["streams"]})==3
            counts["source_stream_contracts"]+=1
        adaptive=json.loads((OUT/"confirmatory"/f"adaptive_choices_{seed}.json").read_text())
        for family,name in enumerate(FAMILIES):
            coeff=task(seed,family)
            trees=candidates()+[decode_numeric_order(adaptive[name]["tree"]),
                               adaptive_tree(coeff,"oracle_adaptive_dp")[0]]
            for tree in trees:
                bound=cut_scores(coeff,tree)["centered_cut_bound"]
                selected=all_fit[(all_fit.seed==seed)&(all_fit.family==name)&
                    (all_fit.candidate_id==tree.name)&all_fit.selected_by_validation].iloc[0]
                gap=float(selected.exact_population_nmse-bound)
                assert gap>=-1e-7
                bounds.append(dict(seed=seed,family=name,candidate_id=tree.name,
                    true_centered_cut_bound=bound,exact_population_nmse=selected.exact_population_nmse,
                    excess_above_lower_bound=gap))
    # Replay one entire candidate fit per family at the first confirmatory seed.
    # This includes both32-sweep restarts and independent validation/test streams.
    replays=[]
    for family,name in enumerate(FAMILIES):
        seed=CONFIRMATORY_SEEDS[0];tree=candidates()[0]
        rerun=train_candidate(tree,task(seed,family),seed,family,TRAINING)
        stored=all_fit[(all_fit.seed==seed)&(all_fit.family==name)&(all_fit.candidate_id==tree.name)].set_index("restart")
        maximum=0.
        for record in rerun:
            for metric in ("validation_mse","test_nmse","noisy_test_mse","exact_population_nmse"):
                maximum=max(maximum,abs(record[metric]-stored.loc[record["restart"],metric]))
            assert record["selected_by_validation"]==stored.loc[record["restart"],"selected_by_validation"]
        assert maximum<1e-10
        replays.append(dict(seed=seed,family=name,candidate=tree.name,restarts=2,max_absolute_error=maximum))
    pd.DataFrame(bounds).to_csv(OUT/"candidate_bound_diagnostics.csv",index=False)
    dump(OUT/"validation_report.json",dict(status="passed",seconds=time.perf_counter()-start,
        frozen_source_hashes_verified=True,runtime_amendments_verified=2,sealed_artifact_hashes_verified=len(seal["selection_source_hashes"]),
        all_outcome_file_times_after_global_seal=True,total_fits=2240,
        validation_only_restart_choices_verified=restart_groups,**counts,
        total_confirmatory_calibration_label_queries=query_count,max_cut_score_reconstruction_error=max_score_delta,
        exact_population_lower_bound_checks=len(bounds),
        minimum_excess_above_bound=min(row["excess_above_lower_bound"] for row in bounds),
        maximum_final_parameter_norm=float(all_fit.final_parameter_norm.max()),
        final_parameter_norm_above10000=int((all_fit.final_parameter_norm>10000).sum()),
        full_fit_replays=replays,scope="Numerical/data-contract validation in current environment; no new scientific endpoints or model selection tuning",
        observation_independence="Independent draw/source stream IDs, not disjoint input patterns; repeated inputs have fresh independent noise"))
    print((OUT/"validation_report.json").read_text())


if __name__=="__main__":main()
