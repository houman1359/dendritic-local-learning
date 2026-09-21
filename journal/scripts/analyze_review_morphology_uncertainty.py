#!/usr/bin/env python3
"""Audit cached labels, mapping and radius sensitivity in the eight-cell cohort.

No new contact labels are imputed. Sensitivity probes use the nominal hybrid
model's field ensemble held fixed across perturbations, rather than defining a
new target to favor each perturbed dictionary. Cell summaries are descriptive
within one mouse. Radius variations are controlled perturbations, not a
calibrated posterior over reconstruction errors.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code/reconstructed_tree"))
import analyze_microns_morphology_credit as base
DATA=ROOT.parent.parent/"dendritic-credit-routing/data/microns_morphology"
OUT=ROOT/"source_data/review_morphology_uncertainty"


def classify(synapses,mode):
    result=synapses.copy()
    direct=result.typed_class.isin(["E","I"])
    confident=pd.to_numeric(result.target_tag_probability,errors="coerce").fillna(0).ge(.8)
    result["synapse_class"]=np.where(direct,result.typed_class,
        np.where(confident,result.target_proxy_class,"?")) if mode=="hybrid" else result.typed_class
    return result


def dictionary(segments,sites,weights,axial_mode="mean_radius"):
    electrical=base.electrical_geometry(segments,1.,.5).set_index("segment_id")
    _,parents,_=base.parent_map(segments)
    if axial_mode=="series_resistance":
        valid=segments.parent_segment_id.ge(0).to_numpy()
        raw=1/segments.loc[valid,"raw_axial_resistance"].to_numpy(float)
        new_edge=np.zeros(len(segments));new_edge[valid]=np.clip(raw/np.median(raw[raw>0]),.05,20.)
        child_sum={int(i):0. for i in segments.segment_id}
        for row,g in zip(segments.itertuples(),new_edge):
            if row.parent_segment_id>=0:child_sum[int(row.parent_segment_id)]+=g
        replacement=electrical.index.map(child_sum).to_numpy(float)
        electrical["g_total"]+=replacement-electrical["g_children"].to_numpy(float)
    routes=segments.loc[segments.I_size.gt(0),"segment_id"].astype(int).tolist()
    ancestry=base.ancestry_matrix(sites,routes,parents)
    beta=(electrical.loc[routes,"g_i"]/electrical.loc[routes,"g_total"]).to_numpy()
    kernel=ancestry*beta[None,:]
    leverage=beta*(ancestry.T@weights)/max(weights.sum(),1e-30)
    selected=np.argsort(-leverage,kind="stable")[:min(4,len(routes))]
    return kernel,kernel[:,selected],[routes[i] for i in selected]


def capture(target,d,weights):
    target=np.sqrt(weights[:,None]/weights.sum())*target
    d=np.sqrt(weights[:,None]/weights.sum())*d
    projection=d@np.linalg.pinv(d,rcond=1e-10)
    denominator=np.sum(target*target)
    return float(np.sum((projection@target)**2)/max(denominator,1e-30)),int(np.linalg.matrix_rank(d))


def compression_audit(swc,segments,nodes,root_id):
    lookup=swc.set_index("id")
    records=[]
    for segment,group in nodes.groupby("segment_id"):
        lengths=[];radii=[]
        for row in group.itertuples():
            parent=int(lookup.loc[row.id,"parent"])
            if parent<0: continue
            point=lookup.loc[row.id,["x","y","z"]].to_numpy(float)
            proximal=lookup.loc[parent,["x","y","z"]].to_numpy(float)
            length=np.linalg.norm(point-proximal)
            if length<=0: continue
            lengths.append(length);radii.append((float(row.radius)+float(lookup.loc[parent,"radius"]))/2)
        if not lengths: continue
        lengths=np.asarray(lengths);radii=np.maximum(radii,1e-12)
        compressed=segments[segments.segment_id.eq(segment)].iloc[0]
        actual_resistance=np.sum(lengths/radii**2)
        approx=compressed.edge_length_um/max(compressed.mean_radius_um,1e-12)**2
        records.append(dict(root_id=root_id,segment_id=int(segment),raw_length_um=lengths.sum(),
            compressed_length_um=compressed.edge_length_um,
            raw_axial_resistance=actual_resistance,
            relative_axial_resistance_error=approx/actual_resistance-1,
            radius_length_product_error=compressed.mean_radius_um*compressed.edge_length_um-np.sum(lengths*radii)))
    return records


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(DATA/"cell_manifest.csv")
    confusion=[];missingness=[];sensitivity=[];compression=[];input_hashes={}
    for root_id in manifest.root_id.astype(int):
        paths=[DATA/f"skeleton_{root_id}.csv.gz",DATA/f"synapses_{root_id}.csv.gz"]
        for path in paths: input_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        swc=pd.read_csv(paths[0]);syn=pd.read_csv(paths[1])
        segments,nodes=base.compress_dendritic_tree(swc)
        cell_compression=compression_audit(swc,segments,nodes,root_id)
        compression.extend(cell_compression)
        raw_r={r["segment_id"]:r["raw_axial_resistance"] for r in cell_compression}
        segments["raw_axial_resistance"]=segments.segment_id.map(raw_r).fillna(1.)
        mapped=base.map_synapses_to_segments(syn,nodes,5.)
        meta=segments.set_index("segment_id")
        mapped["path_length_um"]=mapped.segment_id.map(meta.path_length_um)
        mapped["compartment"]=np.where(mapped.segment_id.map(meta.parent_segment_id).lt(0),"soma",
            np.where(mapped.segment_id.map(meta.is_leaf),"terminal","internal"))
        mapped["path_quartile"]=pd.qcut(mapped.path_length_um.rank(method="first"),4,labels=False)
        keep=mapped.mapping_pass
        joint=keep & mapped.typed_class.isin(["E","I"]) & mapped.target_proxy_class.isin(["E","I"])
        for (direct,proxy),g in mapped[joint].groupby(["typed_class","target_proxy_class"]):
            confusion.append(dict(root_id=root_id,direct_class=direct,proxy_class=proxy,n_contacts=len(g)))
        for grouping in ["compartment","path_quartile"]:
            for value,g in mapped.groupby(grouping):
                direct=g.typed_class.isin(["E","I"])
                missingness.append(dict(root_id=root_id,grouping=grouping,stratum=str(value),
                    n_contacts=len(g),n_mapped=int(g.mapping_pass.sum()),
                    n_direct=int(direct.sum()),n_mapped_direct=int((direct & g.mapping_pass).sum()),
                    direct_fraction=float(direct.mean()),mapping_fraction=float(g.mapping_pass.mean())))
        nominal=base.add_segment_synapses(segments,base.map_synapses_to_segments(classify(syn,"hybrid"),nodes,5.))
        sites=nominal.loc[nominal.E_size.gt(0),"segment_id"].astype(int).tolist()
        weights=nominal.set_index("segment_id").loc[sites,"E_size"].to_numpy(float)
        target,_,nominal_selected=dictionary(nominal,sites,weights)
        for threshold in [2.,5.,10.]:
            for mode in ["hybrid","direct_only"]:
                mapped_case=base.map_synapses_to_segments(classify(syn,mode),nodes,threshold)
                case=base.add_segment_synapses(segments,mapped_case)
                for radius_sd in [0.,.25,.5]:
                    for replicate in range(1 if radius_sd==0 else 5):
                        perturbed=case.copy()
                        rng=np.random.default_rng(np.random.SeedSequence([root_id & 0xffffffff,replicate,927]))
                        radius_factor=np.exp(radius_sd*rng.normal(size=len(case))-.5*radius_sd**2)
                        perturbed["mean_radius_um"]*=radius_factor
                        perturbed["raw_axial_resistance"]/=radius_factor**2
                        for axial_mode in ["mean_radius","series_resistance"]:
                            _,d,selected=dictionary(perturbed,sites,weights,axial_mode)
                            value,rank=capture(target,d,weights)
                            union=set(selected)|set(nominal_selected)
                            sensitivity.append(dict(root_id=root_id,mapping_threshold_um=threshold,
                                label_mode=mode,radius_log_sd=radius_sd,replicate=replicate,axial_mode=axial_mode,
                                n_mapped_classified=int((mapped_case.mapping_pass & mapped_case.synapse_class.isin(["E","I"])).sum()),
                                n_candidate_routes=int(perturbed.I_size.gt(0).sum()),selected_rank=rank,
                                selected_route_ids=";".join(map(str,selected)),
                                selected_nominal_jaccard=len(set(selected)&set(nominal_selected))/max(len(union),1),
                                fixed_nominal_field_capture=value))
        print(f"completed {root_id}",flush=True)
    frames={"confusion_by_cell":pd.DataFrame(confusion),"label_missingness":pd.DataFrame(missingness),
        "mapping_radius_sensitivity":pd.DataFrame(sensitivity),"compression_geometry_audit":pd.DataFrame(compression)}
    for name,frame in frames.items():frame.to_csv(OUT/f"{name}.csv",index=False,float_format="%.12g")
    pooled=frames["confusion_by_cell"].groupby(["direct_class","proxy_class"],as_index=False).n_contacts.sum()
    pooled.to_csv(OUT/"confusion_pooled.csv",index=False)
    cell=frames["mapping_radius_sensitivity"].groupby(["root_id","mapping_threshold_um","label_mode","radius_log_sd","axial_mode"],as_index=False)[["selected_nominal_jaccard","fixed_nominal_field_capture","selected_rank"]].mean()
    cell.to_csv(OUT/"cell_sensitivity_means.csv",index=False,float_format="%.12g")
    c=frames["compression_geometry_audit"]
    report={"status":"complete","n_cells":len(manifest),"n_jointly_labeled_mapped_contacts":int(pooled.n_contacts.sum()),
        "direct_proxy_agreement":float(pooled.loc[pooled.direct_class.eq(pooled.proxy_class),"n_contacts"].sum()/pooled.n_contacts.sum()),
        "n_sensitivity_conditions":len(frames["mapping_radius_sensitivity"]),
        "maximum_chain_length_error_um":float((c.raw_length_um-c.compressed_length_um).abs().max()),
        "maximum_radius_length_product_error":float(c.radius_length_product_error.abs().max()),
        "median_relative_axial_resistance_error":float(c.relative_axial_resistance_error.median()),
        "most_negative_relative_axial_resistance_error":float(c.relative_axial_resistance_error.min()),
        "scope":"Eight cells from one mouse. Direct label availability is observed, not assumed random. Radius perturbations and mapping thresholds are sensitivity analyses, not calibrated uncertainty. Field-capture probes are fixed nominal model-generated fields; they do not test endogenous task credit. Chain compression preserves path length and the length-weighted radius product, but not axial resistance for heterogeneous radii.",
        "input_sha256":input_hashes,"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUT/"README.md").write_text("# Morphology preprocessing and label uncertainty\n\n"+report["scope"]+"\n\nAll perturbations retained. Five radius draws are averaged within each cell before summaries; no additional animal replication is inferred.\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
