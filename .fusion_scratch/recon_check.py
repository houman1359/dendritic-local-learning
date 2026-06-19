import csv, statistics

P = "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix/run_summary.csv"
rows = list(csv.DictReader(open(P)))
print(f"n runs = {len(rows)}")

for col in ["factorization_weighted_relative_l2", "factorization_weighted_scale_mismatch",
            "factorization_weighted_cosine", "factorization_weighted_scaled_relative_l2"]:
    vals = [float(r[col]) for r in rows if r[col] not in ("", "nan")]
    nan_n = sum(1 for r in rows if r[col] in ("", "nan"))
    print(f"\n{col}:  (n={len(vals)}, nan/empty={nan_n})")
    print(f"   min={min(vals):.6e}  max={max(vals):.6e}  mean={statistics.mean(vals):.6e}  median={statistics.median(vals):.6e}")

# threshold checks for the paper claims
rel = [float(r["factorization_weighted_relative_l2"]) for r in rows]
sm  = [float(r["factorization_weighted_scale_mismatch"]) for r in rows]
print("\n--- Paper claim checks ---")
print(f"weighted relative reconstruction error < 2e-7 ?  max={max(rel):.6e}  -> {max(rel) < 2e-7}")
print(f"weighted scale mismatch < 4e-8 ?                 max={max(sm):.6e}  -> {max(sm) < 4e-8}")
print(f"  (# runs with rel >= 2e-7: {sum(1 for x in rel if x>=2e-7)})")
print(f"  (# runs with sm  >= 4e-8: {sum(1 for x in sm  if x>=4e-8)})")
