import csv, math, statistics, os

BASE = "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis/gradient_fidelity"
SUMMARY = os.path.join(BASE, "gradient_norm_dynamics_summary.csv")

def fmt(xs):
    xs = list(xs)
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if len(xs) > 1 else 0.0  # population std
    ssamp = statistics.stdev(xs) if len(xs) > 1 else 0.0  # sample std
    return m, s, ssamp

print("="*80)
print("PART A: from gradient_norm_dynamics_summary.csv (aggregate per config/epoch)")
print("="*80)
rows = list(csv.DictReader(open(SUMMARY)))
# group final epoch (max epoch per config)
by_cfg = {}
for r in rows:
    by_cfg.setdefault(r["config"], []).append(r)
finals = {}
for cfg, rs in by_cfg.items():
    rs.sort(key=lambda r: int(r["epoch"]))
    last = rs[-1]
    finals[cfg] = last
    print(f"  {cfg:9s} {last['core_type']:9s} epoch={last['epoch']:>3s} "
          f"weighted_cos={float(last['weighted_cosine']):+.6f} "
          f"local={float(last['local_grad_norm']):.6f} bp={float(last['backprop_grad_norm']):.6f} "
          f"|log10(L/BP)|={abs(math.log10(float(last['local_grad_norm'])/float(last['backprop_grad_norm']))):.6f}")

for ct in ["shunting", "additive"]:
    cfgs = [c for c,l in finals.items() if l["core_type"]==ct]
    coss = [float(finals[c]["weighted_cosine"]) for c in cfgs]
    sms  = [abs(math.log10(float(finals[c]["local_grad_norm"])/float(finals[c]["backprop_grad_norm"]))) for c in cfgs]
    mc, sc, scc = fmt(coss)
    ms, ss, ssc = fmt(sms)
    print(f"\n  GROUP {ct} (n={len(cfgs)}: {cfgs})")
    print(f"    weighted_cosine: mean={mc:+.4f} popstd={sc:.4f} samplestd={scc:.4f}  values={[round(x,4) for x in coss]}")
    print(f"    scale_mismatch : mean={ms:.4f} popstd={ss:.4f} samplestd={ssc:.4f}  values={[round(x,4) for x in sms]}")

print()
print("="*80)
print("PART B: from per-config gradient_fidelity_trajectory.csv (numel-weighted at final epoch)")
print("="*80)
# determine core_type per config from summary
cfg_ct = {c: finals[c]["core_type"] for c in finals}
traj_results = {}
for i in range(6):
    cfg = f"config_{i}"
    path = os.path.join(BASE, cfg, "gradient_fidelity_trajectory.csv")
    trows = list(csv.DictReader(open(path)))
    epochs = sorted(set(int(r["epoch"]) for r in trows))
    last_ep = epochs[-1]
    lr = [r for r in trows if int(r["epoch"])==last_ep]
    # numel-weighted cosine
    wsum = sum(float(r["numel"]) for r in lr)
    wcos = sum(float(r["cosine_similarity"])*float(r["numel"]) for r in lr)/wsum
    # aggregate L2 norms: sqrt(sum of squares of per-param norms)
    Ltot = math.sqrt(sum(float(r["local_grad_norm"])**2 for r in lr))
    BPtot = math.sqrt(sum(float(r["backprop_grad_norm"])**2 for r in lr))
    sm_agg = abs(math.log10(Ltot/BPtot))
    # also numel-weighted norm_ratio based scale mismatch
    wnr = sum(float(r["norm_ratio"])*float(r["numel"]) for r in lr)/wsum
    sm_wnr = abs(math.log10(wnr)) if wnr>0 else float('nan')
    traj_results[cfg] = dict(core=cfg_ct.get(cfg,"?"), last_ep=last_ep, wcos=wcos,
                             Ltot=Ltot, BPtot=BPtot, sm_agg=sm_agg, wnr=wnr, sm_wnr=sm_wnr,
                             nparams=len(lr), totnumel=wsum)
    print(f"  {cfg:9s} {cfg_ct.get(cfg,'?'):9s} last_epoch={last_ep:>3d} nparams={len(lr):3d} "
          f"wcos={wcos:+.6f}  L2agg_L={Ltot:.6f} L2agg_BP={BPtot:.6f} sm_agg|log10(L/BP)|={sm_agg:.6f}  "
          f"wnr={wnr:.4f} sm_wnr={sm_wnr:.4f}")

for ct in ["shunting", "additive"]:
    cfgs = [c for c,v in traj_results.items() if v["core"]==ct]
    coss = [traj_results[c]["wcos"] for c in cfgs]
    sm_agg = [traj_results[c]["sm_agg"] for c in cfgs]
    sm_wnr = [traj_results[c]["sm_wnr"] for c in cfgs]
    mc, sc, scc = fmt(coss)
    print(f"\n  GROUP {ct} (n={len(cfgs)}: {cfgs})")
    print(f"    weighted_cos      : mean={mc:+.4f} popstd={sc:.4f} samplestd={scc:.4f}  values={[round(x,4) for x in coss]}")
    m,s,ssamp = fmt(sm_agg)
    print(f"    sm_agg(L2 norms)  : mean={m:.4f} popstd={s:.4f} samplestd={ssamp:.4f}  values={[round(x,4) for x in sm_agg]}")
    m,s,ssamp = fmt(sm_wnr)
    print(f"    sm_wnr(weighted)  : mean={m:.4f} popstd={s:.4f} samplestd={ssamp:.4f}  values={[round(x,4) for x in sm_wnr]}")
