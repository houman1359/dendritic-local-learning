import csv, statistics

files = {
 "nonnegativeinput_fix": "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix/path_gain_stats.csv",
 "activation_corrected": "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/theory_diag_gradient_fidelity_vs_ie_activation_corrected/path_gain_stats.csv",
 "original": "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/theory_diag_gradient_fidelity_vs_ie/path_gain_stats.csv",
}

def ms(v):
    return (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0, len(v))

for tag, f in files.items():
    rows = [r for r in csv.DictReader(open(f)) if r['dataset'] == 'mnist' and r['ie_value'] == '5']
    print("=" * 72)
    print(tag, " (n_mnist_ie5 rows =", len(rows), ")")
    for nt in ['dendritic_shunting', 'dendritic_additive']:
        sub = [r for r in rows if r['network_type'] == nt]
        byL = {}
        for r in sub:
            byL.setdefault(r['layer_index'], []).append(float(r['path_gain_cv']))
        ncfg = len(set(r['run_name'] for r in sub))
        print("  [" + nt + "]  n_configs=" + str(ncfg))
        for L in sorted(byL):
            m, s, n = ms(byL[L])
            print("     L%s: cv mean=%.4f std=%.4f (n=%d)" % (L, m, s, n))
        cfgs = {}
        for r in sub:
            cfgs.setdefault(r['run_name'], {})[r['layer_index']] = float(r['path_gain_cv'])
        per_all, per_l01, per_l0 = [], [], []
        for c, d in cfgs.items():
            per_all.append(statistics.mean(list(d.values())))
            l01 = [d[k] for k in ('0', '1') if k in d]
            per_l01.append(statistics.mean(l01))
            if '0' in d:
                per_l0.append(d['0'])
        a = ms(per_all); b = ms(per_l01); c0 = ms(per_l0)
        print("     mean over L0,L1,L2 per-config: %.4f +/- %.4f" % (a[0], a[1]))
        print("     mean over L0,L1   per-config: %.4f +/- %.4f" % (b[0], b[1]))
        print("     L0 only           per-config: %.4f +/- %.4f" % (c0[0], c0[1]))
