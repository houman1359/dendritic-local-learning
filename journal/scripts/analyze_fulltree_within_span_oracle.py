#!/usr/bin/env python3
"""Replay exact checkpoints and separate fixed-gain from fixed-span error.

This is a retrospective diagnostic, not new learned-feedback performance.
Every dictionary is frozen before task fitting. Exact fits are re-executed only
because full checkpoint states were not stored in Source Data. Endpoint drift
is reported for every run, without outcome-dependent exclusions. If any drift
exceeds the tolerance, these are labelled newly re-executed checkpoints, not
the exact archived checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_reconstructed_tree_task_learning as base

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"source_data"
OUT=SOURCE/"fulltree_within_span_oracle"
REPLAY_TOLERANCE=1e-5


def one_replay(extract,segments,args,replicate,recorded,metadata):
    data=base.load_target(extract)
    root=int(data["root_id"])
    stream=100*int(data["session"])+int(data["scan_idx"])
    train,test=base.grouped_split(data["stimulus_ids"],base.stable_rng(args.seed,root,1000+stream,replicate),args.test_fraction)
    x=base.scale_inputs(data["x_raw"],train,args.input_lower_quantile,args.input_upper_quantile,args.input_maximum)
    y=base.scale_target(data["y_raw"],train)
    tree=base.build_passive_tree(segments[segments.root_id.eq(root)].copy(),data["contacts"],args.channels,
        base.stable_rng(args.seed,root,2000+stream,replicate),args.background_e_scale,
        args.background_i_scale,args.excitatory_reversal,args.inhibitory_reversal)
    # Random candidate ordering is not portable across sorting-library versions.
    # Replay the ACTUAL archived route identities rather than resampling them.
    original_metadata=metadata[(root,int(data["session"]),int(data["scan_idx"]),replicate)]
    assert tree.site_segment_ids.tolist()==original_metadata["site_segment_ids"]
    assert tree.selected_route_segments==original_metadata["selected_route_segments"]
    electrical=base.electrical_geometry(segments[segments.root_id.eq(root)].copy(),
        e_scale=args.background_e_scale,i_scale=args.background_i_scale)
    _,parents,_=base.parent_map(electrical)
    route_ids=original_metadata["random_route_segments"]
    lookup=electrical.set_index("segment_id")
    beta=lookup.loc[route_ids,"g_i"].to_numpy()/np.maximum(lookup.loc[route_ids,"g_total"].to_numpy(),1e-12)
    tree.random_dictionary=base.ancestry_matrix(tree.site_segment_ids.tolist(),route_ids,parents)*beta[None,:]
    tree.random_route_segments=route_ids
    initial=base.initialize_state(tree,x[train],y[train],args.initial_conductance)
    state,_=base.train(tree,x[train],y[train],initial,None,args.steps,args.learning_rate,args.weight_decay,args.readout_decay)
    _,_,cache=base.objective_and_gradients(tree,x[test],y[test],state,None,args.weight_decay,args.readout_decay)
    baseline=np.mean((y[test]-np.mean(y[train]))**2)
    mse=np.mean(cache["error"]**2)/baseline
    original=recorded[recorded.target_root_id.eq(root)&recorded.session.eq(int(data["session"])) &
                      recorded.scan_idx.eq(int(data["scan_idx"]))&recorded.replicate.eq(replicate)&
                      recorded.method.eq("exact compartment error")]
    assert len(original)==1
    mismatch=abs(mse-float(original.iloc[0].heldout_normalized_mse))
    exact=cache["exact_error"]
    eligibility=cache["eligibility"]
    updates=eligibility*exact
    # Trial-dependent coefficients are exact-error oracles in a fixed dictionary.
    # They do not alter which compartments the dictionary can address.
    rows=[]
    fields={"unprojected baseline transfer":cache["error"][:,None]*state.readout*tree.root_to_sites[None,:]}
    for method,D in [("topology-matched routes",tree.morphology_dictionary),
                     ("site-shuffled routes",tree.shuffled_dictionary),
                     ("random anatomical routes",tree.random_dictionary)]:
        projection=base.projector(D)
        fixed=cache["error"][:,None]*state.readout*(projection@tree.root_to_sites)[None,:]
        dynamic=exact@projection.T
        oracle_updates=np.empty_like(updates)
        for i in range(len(exact)):
            weighted=eligibility[i,:,None]*D
            coefficients=np.linalg.lstsq(weighted,updates[i],rcond=1e-10)[0]
            oracle_updates[i]=weighted@coefficients
        for mode,field,update in [("frozen_baseline",fixed,eligibility*fixed),
                                  ("trialwise_field_projection",dynamic,eligibility*dynamic),
                                  ("trialwise_update_oracle",None,oracle_updates)]:
            rows.append(dict(method=method,mode=mode,field_match=np.nan if field is None else base.capture(exact,field),
                update_match=base.capture(updates,update),update_cosine=base.flattened_cosine(updates,update)))
        assert base.capture(updates,oracle_updates)>=base.capture(updates,eligibility*fixed)-1e-9
        assert base.capture(updates,oracle_updates)>=base.capture(updates,eligibility*dynamic)-1e-9
        # Preserve direct linkage to the historical static field statistic.
        historical=recorded[recorded.target_root_id.eq(root)&recorded.session.eq(int(data["session"]))&
                            recorded.scan_idx.eq(int(data["scan_idx"]))&recorded.replicate.eq(replicate)&recorded.method.eq(method)]
        static_error=abs(float(historical.iloc[0].common_checkpoint_update_capture)-base.capture(updates,eligibility*fixed))
        for row in rows:
            if row["method"]==method:
                row["static_update_match_replay_error"]=static_error
    field=fields["unprojected baseline transfer"]
    rows.append(dict(method="unprojected baseline transfer",mode="frozen_baseline",
        field_match=base.capture(exact,field),update_match=base.capture(updates,eligibility*field),
        update_cosine=base.flattened_cosine(updates,eligibility*field)))
    for row in rows:
        row.update(target_root_id=root,session=int(data["session"]),scan_idx=int(data["scan_idx"]),
            replicate=replicate,n_sites=x.shape[1],dictionary_channels=tree.used_channels,
            exact_heldout_normalized_mse=mse,exact_endpoint_replay_error=mismatch)
    return rows


def summarize(frame,out):
    metrics=["field_match","update_match","update_cosine","exact_endpoint_replay_error","static_update_match_replay_error"]
    scan=frame.groupby(["target_root_id","session","scan_idx","method","mode"],as_index=False)[metrics].mean()
    cell=scan.groupby(["target_root_id","method","mode"],as_index=False)[metrics].mean()
    rows=[]
    for index,((method,mode),g) in enumerate(cell.groupby(["method","mode"])):
        rng=np.random.default_rng(20260906+index)
        x=g.update_match.to_numpy()
        means=x[rng.integers(0,len(x),(20000,len(x)))].mean(axis=1)
        low,high=np.quantile(means,[.025,.975])
        rows.append(dict(method=method,mode=mode,n_targets=len(g),mean_update_match=x.mean(),
                         ci95_low=low,ci95_high=high,mean_field_match=g.field_match.mean()))
    for name,data in [("run_metrics",frame),("scan_metrics",scan),("cell_metrics",cell),("condition_summary",pd.DataFrame(rows))]:
        data.to_csv(out/f"{name}.csv",index=False,float_format="%.12g")
    differences=[]
    wide=cell.pivot(index="target_root_id",columns=["method","mode"],values="update_match")
    for mode in ["frozen_baseline","trialwise_field_projection","trialwise_update_oracle"]:
        for control in ["site-shuffled routes","random anatomical routes"]:
            x=wide[("topology-matched routes",mode)]-wide[(control,mode)]
            rng=np.random.default_rng(20260916+len(differences))
            means=x.to_numpy()[rng.integers(0,len(x),(20000,len(x)))].mean(axis=1)
            low,high=np.quantile(means,[.025,.975])
            differences.append(dict(mode=mode,control=control,n_targets=len(x),
                ancestry_minus_control_mean=x.mean(),ci95_low=low,ci95_high=high,
                positive_targets=int((x>0).sum())))
    pd.DataFrame(differences).to_csv(out/"paired_ancestry_contrasts.csv",index=False,float_format="%.12g")
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true",help="Replay one scan/replicate; no seven-target inference")
    parsed=parser.parse_args()
    out=OUT/"smoke" if parsed.smoke else OUT
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((SOURCE/"fulltree_boundary/output/config.json").read_text())
    args=SimpleNamespace(**cfg)
    segments=pd.read_csv(base.DEFAULT_SEGMENTS)
    recorded=pd.read_csv(SOURCE/"fulltree_boundary/output/runs.csv")
    metadata={}
    for line in (SOURCE/"fulltree_boundary/output/dictionary_and_validation_metadata.jsonl").read_text().splitlines():
        m=json.loads(line)
        metadata[(m["target_root_id"],m["session"],m["scan_idx"],m["replicate"])]=m
    extracts=sorted(base.DEFAULT_EXTRACT_ROOT.glob("target*_automatic_conservative"))
    if parsed.smoke: extracts=extracts[:1]
    rows=[]
    input_manifest=[]
    started=time.monotonic()
    for scan in extracts:
        for rep in range(1 if parsed.smoke else args.replicates):
            rows.extend(one_replay(scan,segments,args,rep,recorded,metadata))
        data=base.load_target(scan)
        input_manifest.append(dict(target_root_id=data["root_id"],session=data["session"],scan_idx=data["scan_idx"],inputs=data["input_files"]))
        print(f"completed {scan.name}; {len(rows)} diagnostic rows; {time.monotonic()-started:.1f} s",flush=True)
    frame=pd.DataFrame(rows)
    summary=summarize(frame,out)
    concordant=bool(frame.exact_endpoint_replay_error.max()<REPLAY_TOLERANCE and frame.static_update_match_replay_error.max()<REPLAY_TOLERANCE)
    report=dict(status="smoke_only" if parsed.smoke else "complete_reexecuted_checkpoint_diagnostic",
        n_exact_replays=len(frame.drop_duplicates(["target_root_id","session","scan_idx","replicate"])),
        n_targets=int(frame.target_root_id.nunique()),n_rows=len(frame),
        maximum_exact_endpoint_replay_error=float(frame.exact_endpoint_replay_error.max()),
        maximum_static_update_match_replay_error=float(frame.static_update_match_replay_error.max()),
        all_replays_within_tolerance=concordant,
        checkpoints_with_endpoint_drift_over_tolerance=int(frame.drop_duplicates(["target_root_id","session","scan_idx","replicate"]).exact_endpoint_replay_error.gt(REPLAY_TOLERANCE).sum()),
        replay_absolute_tolerance=REPLAY_TOLERANCE,
        scope="Fixed dictionaries; exact trialwise field or eligibility-weighted update projection at re-executed exact-learning checkpoints. Historical checkpoint identity is not claimed if replay tolerance fails. No oracle learning trajectories or biology tested.",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_seconds=time.monotonic()-started,
        source_hashes={base.portable_path(p):base.sha256(p) for p in [Path(__file__),Path(base.__file__),
            base.TREE_CODE/"analyze_microns_morphology_credit.py",base.DEFAULT_SEGMENTS,
            SOURCE/"fulltree_boundary/output/config.json",SOURCE/"fulltree_boundary/output/runs.csv",
            SOURCE/"fulltree_boundary/output/dictionary_and_validation_metadata.jsonl"]},
        conditions=summary)
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (out/"input_source_manifest.json").write_text(json.dumps(input_manifest,indent=2)+"\n")
    (out/"README.md").write_text("# Fixed-dictionary within-span oracle diagnostic\n\nRun `OPENBLAS_NUM_THREADS=1 python scripts/analyze_fulltree_within_span_oracle.py`. `--smoke` runs a single exact-fit re-execution without population claims. Full mode re-executes 130 exact-learning fits from 13 scans and ten replicates, reports both exact held-out MSE and static-field update-match drift against frozen outcomes, then compares frozen-baseline coefficients, trialwise exact-field projection, and trialwise eligibility-weighted least-squares update projection in the same fixed route dictionaries. The last is a true within-span update upper bound; the field projector need not maximize update match. These are common-state diagnostics at newly re-executed checkpoints, not trained oracle accuracies or an implemented biological encoder. Original full checkpoint states were not archived. Random-route identities are restored from the original manifest, avoiding sorting-dependent resampling. All fits are retained regardless of numerical drift; report.json states the number above the 1e-5 absolute tolerance and maximum discrepancies. This is not a claim of bit-identical historical reproduction. Replicates average within scan, scans within seven targets, with 20,000 target-bootstrap draws. The raw response caches remain in their upstream dataset directory; input_source_manifest.json records their exact hashes and relative paths.\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
