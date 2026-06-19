import csv, math, sys, statistics
from collections import defaultdict, OrderedDict

FILES = OrderedDict([
    ("A_actaudit_nonneginput_fix",
     "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/activation_audit_frozen_localca_theory_nonnegativeinput_fix/compartment_error_fidelity.csv"),
    ("B_actaudit_activation_corrected",
     "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/activation_audit_frozen_localca_theory_activation_corrected/compartment_error_fidelity.csv"),
    ("C_theorydiag_nonneginput_fix",
     "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix/compartment_error_fidelity.csv"),
    ("D_theorydiag_activation_corrected",
     "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/theory_diag_gradient_fidelity_vs_ie_activation_corrected/compartment_error_fidelity.csv"),
])

def fnum(x):
    try:
        if x is None or x == "" or x.lower() == "nan":
            return None
        return float(x)
    except Exception:
        return None

def fmt(v, nd=4):
    if v is None:
        return "  NA  "
    return f"{v:.{nd}f}"

def summ(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v,float) and math.isnan(v))]
    n = len(vals)
    if n == 0:
        return (0, None, None, None, None)
    mean = sum(vals)/n
    sd = statistics.pstdev(vals) if n > 1 else 0.0
    return (n, mean, sd, min(vals), max(vals))

def load(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

for tag, path in FILES.items():
    rows = load(path)
    print("="*100)
    print(f"FILE {tag}")
    print(path)
    print("="*100)
    print(f"TOTAL ROWS (excl header): {len(rows)}")

    def distinct(col):
        s = set()
        for row in rows:
            s.add(row.get(col, ""))
        return s

    for col in ["dataset","network_type","broadcast_mode","error_broadcast_mode","ie_value","strategy","rule_variant","loss_name","layer_name"]:
        vals = distinct(col)
        # sort ie numerically
        if col == "ie_value":
            try:
                disp = sorted(vals, key=lambda x: (fnum(x) is None, fnum(x)))
            except Exception:
                disp = sorted(vals)
        else:
            disp = sorted(vals)
        print(f"  distinct {col} ({len(vals)}): {disp}")

    # Restrict to noise_resilience if present
    dsvals = distinct("dataset")
    if "noise_resilience" in dsvals:
        work = [r for r in rows if r.get("dataset") == "noise_resilience"]
        print(f"\n  -> Restricting to dataset=='noise_resilience': {len(work)} rows")
    else:
        work = rows
        print(f"\n  -> 'noise_resilience' NOT present; using ALL rows ({len(work)}). dataset values: {sorted(dsvals)}")

    # Group by (broadcast_mode, network_type, ie_value): cosine and r2
    groups_cos = defaultdict(list)
    groups_r2 = defaultdict(list)
    groups_runs = defaultdict(set)
    for r in work:
        bm = r.get("broadcast_mode","")
        nt = r.get("network_type","")
        ie = r.get("ie_value","")
        key = (bm, nt, ie)
        groups_cos[key].append(fnum(r.get("cosine")))
        groups_r2[key].append(fnum(r.get("r2")))
        groups_runs[key].add(r.get("run_dir",""))

    print("\n  GROUPED by (broadcast_mode, network_type, ie_value):")
    hdr = f"  {'broadcast_mode':<20}{'net':<12}{'ie':<8}{'n':<6}{'runs':<6}| cos_mean cos_sd  cos_min cos_max |  r2_mean  r2_sd   r2_min  r2_max"
    print(hdr)
    print("  "+"-"*(len(hdr)-2))
    def keysort(k):
        bm, nt, ie = k
        return (bm, nt, (fnum(ie) is None, fnum(ie)))
    for key in sorted(groups_cos.keys(), key=keysort):
        bm, nt, ie = key
        nc, cm, cs, cmin, cmax = summ(groups_cos[key])
        nr, rm, rs, rmin, rmax = summ(groups_r2[key])
        nruns = len(groups_runs[key])
        print(f"  {bm:<20}{nt:<12}{str(ie):<8}{nc:<6}{nruns:<6}| {fmt(cm)} {fmt(cs)} {fmt(cmin)} {fmt(cmax)} | {fmt(rm)} {fmt(rs)} {fmt(rmin)} {fmt(rmax)}")

    # Focused: path_transport overall and by network_type, ie>=5
    print("\n  === FOCUS: broadcast_mode == 'path_transport' ===")
    pt = [r for r in work if r.get("broadcast_mode") == "path_transport"]
    if not pt:
        print("    (no path_transport rows)")
    else:
        cos_all = [fnum(r.get("cosine")) for r in pt]
        r2_all  = [fnum(r.get("r2")) for r in pt]
        n,m,s,mn,mx = summ(cos_all)
        print(f"    ALL path_transport: cosine n={n} mean={fmt(m)} sd={fmt(s)} min={fmt(mn)} max={fmt(mx)}")
        n,m,s,mn,mx = summ(r2_all)
        print(f"    ALL path_transport:     r2 n={n} mean={fmt(m)} sd={fmt(s)} min={fmt(mn)} max={fmt(mx)}")
        # by network type
        for nt in sorted(set(r.get("network_type") for r in pt)):
            sub = [r for r in pt if r.get("network_type")==nt]
            n,m,s,mn,mx = summ([fnum(r.get("cosine")) for r in sub])
            n2,m2,s2,mn2,mx2 = summ([fnum(r.get("r2")) for r in sub])
            print(f"    net={nt:<10} cosine mean={fmt(m)} [{fmt(mn)},{fmt(mx)}] (n={n}) | r2 mean={fmt(m2)} [{fmt(mn2)},{fmt(mx2)}]")
        # ie>=5 focus by network
        print("    -- restricted ie_value>=5 --")
        pt5 = [r for r in pt if (fnum(r.get("ie_value")) is not None and fnum(r.get("ie_value"))>=5)]
        if pt5:
            n,m,s,mn,mx = summ([fnum(r.get("cosine")) for r in pt5])
            n2,m2,s2,mn2,mx2 = summ([fnum(r.get("r2")) for r in pt5])
            print(f"    ie>=5 ALL: cosine mean={fmt(m)} [{fmt(mn)},{fmt(mx)}] (n={n}) | r2 mean={fmt(m2)} [{fmt(mn2)},{fmt(mx2)}]")
            for nt in sorted(set(r.get("network_type") for r in pt5)):
                sub = [r for r in pt5 if r.get("network_type")==nt]
                n,m,s,mn,mx = summ([fnum(r.get("cosine")) for r in sub])
                n2,m2,s2,mn2,mx2 = summ([fnum(r.get("r2")) for r in sub])
                print(f"      ie>=5 net={nt:<10} cosine mean={fmt(m)} [{fmt(mn)},{fmt(mx)}] (n={n}) | r2 mean={fmt(m2)} [{fmt(mn2)},{fmt(mx2)}]")
        else:
            print("    (no ie>=5 rows for path_transport)")

    # Focused: scalar (rank-1) contrast
    print("\n  === CONTRAST: broadcast_mode == 'scalar' (rank-1) ===")
    sc = [r for r in work if r.get("broadcast_mode") == "scalar"]
    if not sc:
        print("    (no scalar rows)")
    else:
        n,m,s,mn,mx = summ([fnum(r.get("cosine")) for r in sc])
        print(f"    ALL scalar: cosine n={n} mean={fmt(m)} sd={fmt(s)} min={fmt(mn)} max={fmt(mx)}")
        n,m,s,mn,mx = summ([fnum(r.get("r2")) for r in sc])
        print(f"    ALL scalar:     r2 n={n} mean={fmt(m)} sd={fmt(s)} min={fmt(mn)} max={fmt(mx)}")
        for nt in sorted(set(r.get("network_type") for r in sc)):
            sub = [r for r in sc if r.get("network_type")==nt]
            n,m,s,mn,mx = summ([fnum(r.get("cosine")) for r in sub])
            n2,m2,s2,mn2,mx2 = summ([fnum(r.get("r2")) for r in sub])
            print(f"    net={nt:<10} cosine mean={fmt(m)} [{fmt(mn)},{fmt(mx)}] (n={n}) | r2 mean={fmt(m2)} [{fmt(mn2)},{fmt(mx2)}]")
        pt5 = [r for r in sc if (fnum(r.get("ie_value")) is not None and fnum(r.get("ie_value"))>=5)]
        if pt5:
            n,m,s,mn,mx = summ([fnum(r.get("cosine")) for r in pt5])
            n2,m2,s2,mn2,mx2 = summ([fnum(r.get("r2")) for r in pt5])
            print(f"    ie>=5 ALL scalar: cosine mean={fmt(m)} [{fmt(mn)},{fmt(mx)}] (n={n}) | r2 mean={fmt(m2)} [{fmt(mn2)},{fmt(mx2)}]")

    # Other broadcast modes quick contrast (per_soma, path_factor_scalar)
    print("\n  === Other modes (overall cosine/r2) ===")
    for bm in sorted(set(r.get("broadcast_mode") for r in work)):
        if bm in ("path_transport","scalar"):
            continue
        sub = [r for r in work if r.get("broadcast_mode")==bm]
        n,m,s,mn,mx = summ([fnum(r.get("cosine")) for r in sub])
        n2,m2,s2,mn2,mx2 = summ([fnum(r.get("r2")) for r in sub])
        print(f"    {bm:<22} cosine mean={fmt(m)} [{fmt(mn)},{fmt(mx)}] (n={n}) | r2 mean={fmt(m2)} [{fmt(mn2)},{fmt(mx2)}]")

    # distinct run_dir total
    print(f"\n  distinct run_dir total (in working set): {len(set(r.get('run_dir') for r in work))}")
    print()
