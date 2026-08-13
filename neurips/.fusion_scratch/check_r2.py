import csv, math, statistics
# Inspect path_transport r2: how many are positive vs hugely negative, and what best_scale/scaled_relative_l2 look like
paths = {
 "A":"/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/activation_audit_frozen_localca_theory_nonnegativeinput_fix/compartment_error_fidelity.csv",
 "D":"/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/theory_diag_gradient_fidelity_vs_ie_activation_corrected/compartment_error_fidelity.csv",
}
def fnum(x):
    try:
        if x is None or x=="" or x.lower()=="nan": return None
        return float(x)
    except: return None
for tag,p in paths.items():
    rows=[r for r in csv.DictReader(open(p)) if r["dataset"]=="noise_resilience" and r["broadcast_mode"]=="path_transport"]
    cos=[fnum(r["cosine"]) for r in rows]
    r2=[fnum(r["r2"]) for r in rows]
    srl2=[fnum(r["scaled_relative_l2"]) for r in rows]
    rl2=[fnum(r["relative_l2"]) for r in rows]
    bs=[fnum(r["best_scale"]) for r in rows]
    nr=[fnum(r["norm_ratio"]) for r in rows]
    pos=sum(1 for v in r2 if v is not None and v>0)
    print(f"--- {tag} path_transport (n={len(rows)}) ---")
    print(f"  cosine>=0.95: {sum(1 for v in cos if v is not None and v>=0.95)}/{len([v for v in cos if v is not None])}")
    print(f"  r2 positive count: {pos}; r2 NA: {sum(1 for v in r2 if v is None)}")
    valid=[v for v in r2 if v is not None]
    print(f"  r2 median={statistics.median(valid):.2f}  max={max(valid):.4f}")
    valid_srl2=[v for v in srl2 if v is not None]
    valid_rl2=[v for v in rl2 if v is not None]
    print(f"  relative_l2 mean={statistics.mean(valid_rl2):.4f} (median={statistics.median(valid_rl2):.4f})")
    print(f"  scaled_relative_l2 mean={statistics.mean(valid_srl2):.4f} (median={statistics.median(valid_srl2):.4f})")
    valid_bs=[v for v in bs if v is not None]
    print(f"  best_scale mean={statistics.mean(valid_bs):.4g} (median={statistics.median(valid_bs):.4g})")
    valid_nr=[v for v in nr if v is not None]
    print(f"  norm_ratio mean={statistics.mean(valid_nr):.4g} (median={statistics.median(valid_nr):.4g})")
