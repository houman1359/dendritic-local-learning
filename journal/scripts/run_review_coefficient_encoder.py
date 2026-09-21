#!/usr/bin/env python3
"""Learn trial-dependent subtree coefficients from an explicit local context cue.

The cue and supervised branch-activation targets are additional information
resources. This is an implementable rate-model estimator, not evidence that a
biological neuron learns these coefficients without a context signal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import run_trained_subtree_address_full_factorial as task

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/review_completion/coefficient_encoder.json"
OUT = ROOT / "source_data/review_coefficient_encoder"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def softmax(z):
    a = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return a / a.sum(axis=-1, keepdims=True)


def context_cues(context, noise, rng):
    return np.eye(8)[context] + noise * rng.normal(size=(len(context), 8))


def fit_encoder(cues, context, cfg, rng):
    """Local delta-rule updates; targets encode observed subtree activation."""
    x = np.column_stack([cues, np.ones(len(cues))])
    y = np.eye(4)[np.asarray(context) // 2]
    w = np.zeros((9, 4))
    for _ in range(cfg["epochs"]):
        order = rng.permutation(len(x))
        for start in range(0, len(x), cfg["batch_size"]):
            ids = order[start:start + cfg["batch_size"]]
            residual = softmax(x[ids] @ w) - y[ids]
            gradient = x[ids].T @ residual / len(ids)
            gradient[:-1] += cfg["weight_decay"] * w[:-1]
            w -= cfg["learning_rate"] * gradient
    return w


def route_fields(coefficients):
    dictionary = np.repeat(np.eye(4), 2, axis=0) / np.sqrt(2)
    fields = coefficients @ dictionary.T
    return fields / np.maximum(np.linalg.norm(fields, axis=1, keepdims=True), 1e-30)


def coefficient_predictions(cues, w, delay):
    # Delay moves the actual trial cue; no labels enter this operation.
    delayed = np.roll(cues, int(delay), axis=0)
    return softmax(np.column_stack([delayed, np.ones(len(delayed))]) @ w)


def train_paired(initial, data, test, routes, checkpoints, training):
    weights = np.repeat(initial[None], len(routes), axis=0)
    selected_x = data.x[np.arange(len(data.label)), data.context]
    selected_test = test.x[np.arange(len(test.label)), test.context]
    histories = []
    ids = np.arange(len(weights))[:, None]
    for epoch in range(training["epochs"] + 1):
        z = np.einsum("mnf,nf->mn", weights[ids, data.context], selected_x)
        delta = task.sigmoid(z) - data.label[None]
        if epoch in checkpoints:
            ztest = np.einsum("mnf,nf->mn", weights[ids, test.context], selected_test)
            losses, accuracy = task.loss_accuracy(ztest, test.label)
            exact_routes = np.eye(8)[data.context]
            for m in range(len(weights)):
                # Compare exact and delivered updates at each method's OWN state.
                exact = np.einsum("n,nb,nbf->bf", delta[m], exact_routes, data.x) / len(data.label)
                delivered = np.einsum("n,nb,nbf->bf", delta[m], routes[m], data.x) / len(data.label)
                denom = np.linalg.norm(exact) * np.linalg.norm(delivered)
                histories.append({"model": m, "epoch": epoch, "heldout_loss": losses[m],
                                  "heldout_accuracy": accuracy[m],
                                  "gradient_cosine_own_state": np.sum(exact * delivered) / max(denom, 1e-30)})
        if epoch == training["epochs"]:
            break
        gradient = training["gradient_context_rescaling"] * np.einsum(
            "mn,mnb,nbf->mbf", delta, routes, data.x, optimize=True) / len(data.label)
        weights -= training["learning_rate"] * gradient
    if not np.isfinite(weights).all():
        raise FloatingPointError("nonfinite encoder-task fit; retain failure and diagnose")
    return pd.DataFrame(histories)


def run(seed, cfg, out):
    old = json.loads(task.CONFIG.read_text())
    rng = np.random.default_rng(seed)
    teachers = task.normalized_rows(rng.normal(size=(8, 8)))
    train = task.make_dataset(rng, old["task"]["train_examples"], teachers, old)
    test = task.make_dataset(rng, old["task"]["test_examples"], teachers, old)
    initial = rng.normal(scale=old["training"]["initialization_sd"], size=(8, 8))
    rows, routes, diagnostics = [], [], []
    for noise_idx, noise in enumerate(cfg["cue_noise_sd"]):
        cue_rng = np.random.default_rng(seed + 100000 + noise_idx)
        train_cues = context_cues(train.context, noise, cue_rng)
        test_cues = context_cues(test.context, noise, cue_rng)
        # Balanced calibration contexts are independent of classification data.
        max_n = max(cfg["calibration_samples"])
        cal_context = np.tile(np.arange(8), max_n // 8)
        cal_cues = context_cues(cal_context, noise, cue_rng)
        for n in cfg["calibration_samples"]:
            w = fit_encoder(cal_cues[:n], cal_context[:n], cfg["encoder"],
                            np.random.default_rng(seed + 200000 + n))
            for delay in cfg["cue_delay_trials"]:
                learned = coefficient_predictions(train_cues, w, delay)
                heldout = coefficient_predictions(test_cues, w, delay)
                for method in cfg["methods"]:
                    if method == "oracle_context":
                        coeff = np.eye(4)[train.context // 2]
                        eval_coeff = np.eye(4)[test.context // 2]
                    elif method == "frozen_profile":
                        coeff = np.full((len(train.context), 4), .25)
                        eval_coeff = np.full((len(test.context), 4), .25)
                    elif method == "mismatched_encoder":
                        coeff = np.roll(learned, 1, axis=1)
                        eval_coeff = np.roll(heldout, 1, axis=1)
                    else:
                        coeff, eval_coeff = learned, heldout
                    model = len(rows)
                    rows.append(dict(model=model, seed=seed, calibration_samples=n,
                                     cue_noise_sd=noise, cue_delay_trials=delay, method=method))
                    fields = route_fields(coeff)
                    routes.append(fields)
                    diagnostics.append(dict(model=model,
                        coefficient_group_accuracy=float(np.mean(eval_coeff.argmax(1) == test.context // 2)),
                        coefficient_target_mse=float(np.mean((eval_coeff - np.eye(4)[test.context // 2])**2)),
                        maximum_route_norm_error=float(np.max(np.abs(np.linalg.norm(fields, axis=1)-1)))))
    history = train_paired(initial, train, test, np.asarray(routes), set(cfg["checkpoints"]), old["training"])
    frame = history.merge(pd.DataFrame(rows), on="model", validate="many_to_one").merge(
        pd.DataFrame(diagnostics), on="model", validate="many_to_one")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / f"seed_{seed}.csv", index=False, float_format="%.12g")
    (out / f"seed_{seed}.json").write_text(json.dumps({"seed":seed,"protocol_sha256":sha(CONFIG),
        "script_sha256":sha(__file__),"task_script_sha256":sha(task.__file__),
        "numpy_version":np.__version__,"python":platform.python_version(),
        "rows":len(frame),"all_finite":bool(np.isfinite(frame.select_dtypes(include=np.number)).all().all())},indent=2)+"\n")
    print(f"seed {seed}: {len(frame)} rows", flush=True)


def aggregate(cfg):
    files = [OUT / "runs" / f"seed_{seed}.csv" for seed in cfg["seeds"]]
    frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    expected = len(cfg["seeds"]) * 3 * 3 * 3 * 4 * len(cfg["checkpoints"])
    assert len(frame) == expected
    assert frame.groupby(["seed","model","epoch"]).size().eq(1).all()
    frame.to_csv(OUT / "trajectories.csv", index=False, float_format="%.12g")
    final = frame[frame.epoch.eq(max(cfg["checkpoints"]))]
    keys = ["calibration_samples","cue_noise_sd","cue_delay_trials","method"]
    summary = final.groupby(keys).agg(mean_accuracy=("heldout_accuracy","mean"),
        mean_loss=("heldout_loss","mean"),mean_coefficient_accuracy=("coefficient_group_accuracy","mean"),
        n_seeds=("seed","nunique")).reset_index()
    contrasts = []
    rng = np.random.default_rng(20260905)
    for key, group in final.groupby(keys[:-1]):
        wide = group.pivot(index="seed", columns="method", values="heldout_accuracy")
        for control in ["oracle_context","frozen_profile","mismatched_encoder"]:
            # Restore the exact integer test-count grid before signed-rank tests.
            difference = (np.rint(wide.learned_local_cue.to_numpy()*2048)
                          - np.rint(wide[control].to_numpy()*2048)) * (100/2048)
            samples = rng.choice(difference, (cfg["bootstrap_draws"],len(difference)), replace=True).mean(1)
            lo,hi = np.quantile(samples,[.025,.975])
            p = 1.0 if np.allclose(difference,0) else float(wilcoxon(difference,zero_method="pratt").pvalue)
            contrasts.append(dict(zip(keys[:-1], key),control=control,mean_pp=difference.mean(),
                                  ci95_low_pp=lo,ci95_high_pp=hi,positive_seeds=int((difference>0).sum()),p_wilcoxon=p))
    contrasts = pd.DataFrame(contrasts)
    primary = np.ones(len(contrasts),dtype=bool)
    for k,v in cfg["primary_condition"].items(): primary &= contrasts[k].eq(v)
    p = contrasts.loc[primary,"p_wilcoxon"].to_numpy()
    order = np.argsort(p)
    adjusted = np.empty(len(p)); adjusted[order] = np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    contrasts.loc[primary,"primary_family_holm_p"] = adjusted
    summary.to_csv(OUT / "condition_summary.csv",index=False,float_format="%.12g")
    contrasts.to_csv(OUT / "paired_contrasts.csv",index=False,float_format="%.12g")
    report = {"complete":True,"n_seeds":len(cfg["seeds"]),"models_per_seed":108,"trajectory_rows":len(frame),
              "protocol_sha256":sha(CONFIG),"primary_contrasts":contrasts.loc[primary].to_dict("records"),
              "scope":cfg["scope"]}
    (OUT / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    (OUT / "README.md").write_text("# Learned local-context route coefficients\n\n"+cfg["scope"]+
        "\n\nTwenty fresh paired seeds. Oracle/frozen fits repeated across nuisance cells are identical controls, not additional independent observations. Confidence intervals resample seeds; three primary contrasts use Holm correction. Other grid contrasts are descriptive sensitivity analyses. The protocol was locally fixed and hashed before outcomes, not publicly preregistered.\n")
    print(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--seed",type=int); parser.add_argument("--aggregate",action="store_true")
    args=parser.parse_args(); cfg=json.loads(CONFIG.read_text())
    if args.aggregate: aggregate(cfg)
    elif args.seed is not None:
        assert args.seed in cfg["seeds"]+[cfg["canary_seed"]]
        run(args.seed,cfg,OUT/("canary" if args.seed==cfg["canary_seed"] else "runs"))
    else: parser.error("provide --seed or --aggregate")


if __name__ == "__main__": main()
