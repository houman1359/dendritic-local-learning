import csv
f = "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix_summary/theory_diag_by_condition.csv"
rows = list(csv.DictReader(open(f)))
hdr = list(rows[0].keys())
cvcols = [h for h in hdr if 'path_gain_cv' in h]
countcols = [h for h in hdr if h.lower() in ('n_runs', 'count', 'n', 'num_runs') or h.endswith('_count') or 'n_run' in h.lower()]
print("ALL HEADERS:")
print(hdr)
print()
print("CV columns:", cvcols)
print("count columns:", countcols)
print()
for r in rows:
    if r.get('dataset') == 'mnist' and r.get('ie_value') == '5':
        print("network_type =", r['network_type'], " ie_value =", r['ie_value'])
        for c in cvcols:
            print("   ", c, "=", r[c])
        for c in countcols:
            print("   ", c, "=", r[c])
        print()
