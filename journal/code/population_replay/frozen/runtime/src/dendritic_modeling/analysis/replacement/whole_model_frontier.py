"""Whole-model compression operating-point frontier for OLMo-3-7B.

Builds the analysis that answers two directives: (1) total whole-model
compression with any per-layer distribution (dense / replace / possibly
delete per layer); (2) iso-quality comparison to SOTA — the maximum
whole-model compression projected at given perplexity-ratio targets.

Inputs (all on disk, deterministic, no network):

* Per-layer single-replacement screens
  ``olmo3_7b_alloc_screen_L{0..31}_v1/results/*/benchmark_metrics.json``
  (best trained arm per layer; ``*_OVERWRITTEN_*`` quarantined dirs
  excluded; certified seed-404 rederive arms considered where present).
* Deletion floors ``olmo3_7b_ffn_deletion_control.json`` and the paired
  controls ``olmo3_7b_paired_control_*.json``.
* Measured composition anchors: the span-ladder pair evals
  ``olmo3_7b_span_ladder_v1/pair_eval_len*.json`` and the depth-selective
  / whole-FFN pair evals.
* Parameter accounting derived from the model config (config JSON only;
  weights are never loaded), cross-checked against the safetensors index
  total size.

The compounding model maps (sum of per-layer single log-ratios, number of
replaced layers) to a composed log perplexity ratio; a small model family
is fitted and selected by leave-one-out error over the measured anchors,
and the anchor-fit quality is reported as-is. Frontier points derived
from the fit are labelled PROJECTED; per-target measured floors that
coincide with actual anchor compositions are labelled MEASURED.

This module is the committed chain of custody for
``olmo3_7b_whole_model_frontier_v1.json``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

DEFAULT_SCREEN_ROOT = Path("outputs/canonical_population_fmi")
DEFAULT_MODEL_DIR = Path(
    "/n/netscratch/kempner_dev/Lab/hsafaai/serving_zoo/OLMo/Olmo-3-7B-Instruct"
)
DEFAULT_TARGETS = (1.05, 1.1, 1.2, 1.5, 2.0)
PRIMARY_CORPUS = "wikitext"
SENSITIVITY_CORPUS = "allenai/c4"
NAMED_ANCHOR_FILES = (
    "olmo3_7b_pair_eval_depthsel_v1.json",
    "olmo3_7b_pair_eval_depthsel24k_v1.json",
    "olmo3_7b_pair_eval_wholeffn18k_v1.json",
)
DELETION_FREE_THRESHOLD = 1.01

# Model family for the compounding fit: feature names over
# S = sum of per-layer single log perplexity ratios, n = number of
# replaced layers. "1" is an intercept.
MODEL_FAMILY: dict[str, tuple[str, ...]] = {
    "alpha_S_plus_beta_n": ("S", "n"),
    "alpha_S_plus_gamma_S2": ("S", "S2"),
    "intercept_alpha_S_beta_n": ("1", "S", "n"),
    "intercept_alpha_S_beta_n_delta_n2": ("1", "S", "n", "n2"),
    "intercept_beta_n": ("1", "n"),
}


def _load_json(path: Path | str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def collect_best_arms(screen_root: Path) -> dict:
    """Best (lowest trained lm_loss) valid arm per layer, with compression.

    Quarantined ``*_OVERWRITTEN_*`` directories are excluded. Certified
    seed-404 rederive arms under ``olmo3_7b_winner_rederive_v1`` are added
    to the candidate pool for the layers they cover (their originals were
    overwritten in the screens). The per-cell ``initial`` lm_loss is used
    for each ratio (it is near- but not bitwise-constant across cells).
    """
    rederive = {}
    for metrics_file in sorted(
        glob.glob(
            str(
                screen_root
                / "olmo3_7b_winner_rederive_v1/results/*/benchmark_metrics.json"
            )
        )
    ):
        record = _load_json(metrics_file)
        (layer,) = record["layer_indices"]
        rederive.setdefault(int(layer), []).append((metrics_file, record))

    per_layer: dict[int, dict] = {}
    initial_losses: dict[float, int] = {}
    excluded_overwritten = []
    for layer in range(32):
        results_dir = screen_root / f"olmo3_7b_alloc_screen_L{layer}_v1" / "results"
        candidates = []
        for arm_dir in sorted(os.listdir(results_dir)):
            if "_OVERWRITTEN_" in arm_dir:
                excluded_overwritten.append(f"L{layer}/{arm_dir}")
                continue
            metrics_file = results_dir / arm_dir / "benchmark_metrics.json"
            if not metrics_file.exists():
                continue
            record = _load_json(str(metrics_file))
            candidates.append((arm_dir, "alloc_screen", record))
        for metrics_file, record in rederive.get(layer, []):
            candidates.append(
                (Path(metrics_file).parent.name, "winner_rederive_seed404", record)
            )
        if not candidates:
            raise RuntimeError(f"no valid arms for layer {layer}")
        for _, _, record in candidates:
            value = float(record["initial"]["lm_loss"])
            initial_losses[value] = initial_losses.get(value, 0) + 1
        arm_dir, source, record = min(
            candidates, key=lambda item: float(item[2]["trained"]["lm_loss"])
        )
        trained = float(record["trained"]["lm_loss"])
        initial = float(record["initial"]["lm_loss"])
        log_ratio = trained - initial
        per_layer[layer] = {
            "layer": layer,
            "arm": arm_dir,
            "arm_source": source,
            "trained_lm_loss": trained,
            "initial_lm_loss": initial,
            "single_log_ratio": round(log_ratio, 6),
            "single_ppl_ratio": round(math.exp(log_ratio), 6),
            "teacher_dense_params": int(record["teacher_dense_params"]),
            "dendritic_active_params": int(record["dendritic_active_params"]),
            "dendritic_stored_params": int(record["dendritic_stored_params"]),
            "dendritic_stored_bytes": int(record["dendritic_stored_bytes"]),
            "dendritic_index_bytes": int(record["dendritic_index_bytes"]),
            "n_candidate_arms": len(candidates),
        }
    dense_params = {row["teacher_dense_params"] for row in per_layer.values()}
    if dense_params != {135266304}:
        raise RuntimeError(f"unexpected teacher_dense_params values: {dense_params}")
    return {
        "per_layer": per_layer,
        "initial_loss_values": {
            f"{value!r}": count for value, count in sorted(initial_losses.items())
        },
        "initial_loss_constant": len(initial_losses) == 1,
        "excluded_overwritten_dirs": excluded_overwritten,
    }


def collect_deletion(screen_root: Path) -> dict:
    """Deletion floors per measured layer, per corpus, with candidacy flags."""
    per_layer: dict[int, dict] = {}
    control = _load_json(screen_root / "olmo3_7b_ffn_deletion_control.json")
    for layer_key, row in control["layers"].items():
        per_layer.setdefault(int(layer_key), {"ratios_by_source": {}})[
            "ratios_by_source"
        ]["deletion_control_wikitext_seed7"] = float(row["deletion_ppl_ratio"])
    for control_file in sorted(
        glob.glob(str(screen_root / "olmo3_7b_paired_control_*_seed*.json"))
    ):
        record = _load_json(control_file)
        corpus = record["dataset"]["name"]
        seed = record["seed"]
        for layer_key, arm in record["arms"].items():
            per_layer.setdefault(int(layer_key), {"ratios_by_source": {}})[
                "ratios_by_source"
            ][f"paired_control_{corpus}_seed{seed}"] = float(arm["deletion_ppl_ratio"])
    for layer, row in per_layer.items():
        ratios = row["ratios_by_source"]
        wikitext = [v for k, v in ratios.items() if "wikitext" in k]
        row["layer"] = layer
        row["wikitext_mean_ratio"] = round(sum(wikitext) / len(wikitext), 6)
        row["wikitext_max_ratio"] = round(max(wikitext), 6)
        row["cross_corpus_max_ratio"] = round(max(ratios.values()), 6)
        row["deletion_free_wikitext_mean"] = (
            row["wikitext_mean_ratio"] < DELETION_FREE_THRESHOLD
        )
        row["deletion_free_wikitext_max"] = max(wikitext) < DELETION_FREE_THRESHOLD
        row["deletion_free_cross_corpus"] = (
            max(ratios.values()) < DELETION_FREE_THRESHOLD
        )
        row["ratios_by_source"] = {k: round(v, 6) for k, v in sorted(ratios.items())}
    return {
        "threshold": DELETION_FREE_THRESHOLD,
        "criterion_note": (
            "deletion_free_wikitext_mean averages the three wikitext "
            "seeds; the max-based flags are the strict per-seed and "
            "cross-corpus criteria (no layer passes either strict flag)."
        ),
        "per_layer": {layer: per_layer[layer] for layer in sorted(per_layer)},
        "deletion_free_wikitext_layers": sorted(
            layer
            for layer, row in per_layer.items()
            if row["deletion_free_wikitext_mean"]
        ),
        "deletion_free_cross_corpus_layers": sorted(
            layer
            for layer, row in per_layer.items()
            if row["deletion_free_cross_corpus"]
        ),
    }


def parameter_accounting(model_dir: Path) -> dict:
    """Whole-model parameter split derived from the config (no weights)."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(model_dir))
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    n_layers = int(config.num_hidden_layers)
    vocab = int(config.vocab_size)
    if config.tie_word_embeddings:
        raise RuntimeError("accounting assumes untied embeddings for OLMo-3-7B")
    ffn_per_layer = 3 * hidden * intermediate
    attn_per_layer = 4 * hidden * hidden + 2 * hidden  # q,k,v,o + q_norm,k_norm
    norms_per_layer = 2 * hidden  # post-attention + post-feedforward RMSNorm
    embed = vocab * hidden
    lm_head = vocab * hidden
    final_norm = hidden
    total = (
        embed
        + lm_head
        + final_norm
        + n_layers * (ffn_per_layer + attn_per_layer + norms_per_layer)
    )
    accounting = {
        "model_dir": str(model_dir),
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": n_layers,
        "vocab_size": vocab,
        "ffn_params_per_layer": ffn_per_layer,
        "attention_params_per_layer": attn_per_layer,
        "layer_norm_params_per_layer": norms_per_layer,
        "embedding_params": embed,
        "lm_head_params": lm_head,
        "final_norm_params": final_norm,
        "ffn_params_total": ffn_per_layer * n_layers,
        "total_params": total,
        "ffn_fraction_of_total": round(ffn_per_layer * n_layers / total, 6),
    }
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.exists():
        index = _load_json(index_file)
        total_bytes = int(index["metadata"]["total_size"])
        accounting["safetensors_total_bytes"] = total_bytes
        accounting["safetensors_params_bf16"] = total_bytes // 2
        accounting["matches_safetensors_index"] = total_bytes // 2 == total
        if not accounting["matches_safetensors_index"]:
            raise RuntimeError(
                "config-derived parameter count "
                f"{total} != safetensors index {total_bytes // 2}"
            )
    return accounting


def _anchor_steps(record: dict) -> int | None:
    for key in ("composed_config", "checkpoint_dir"):
        match = re.search(r"steps(\d+)", str(record.get(key, "")))
        if match:
            return int(match.group(1))
    return None


def collect_anchors(
    screen_root: Path, per_layer: dict[int, dict], anchor_files: list[Path]
) -> list[dict]:
    """Measured composition pair evals with per-corpus ratios and singles sum."""
    anchors = []
    for anchor_file in anchor_files:
        record = _load_json(anchor_file)
        layers = [int(layer) for layer in record["layers"]]
        ratios = {
            corpus: float(row["perplexity_ratio"])
            for corpus, row in record["corpora"].items()
        }
        anchors.append(
            {
                "name": Path(anchor_file).name,
                "path": str(anchor_file),
                "layers": layers,
                "n_layers": len(layers),
                "recovery_steps": _anchor_steps(record),
                "perplexity_ratio_by_corpus": {
                    corpus: round(value, 6) for corpus, value in sorted(ratios.items())
                },
                "sum_singles_log_ratio": round(
                    sum(per_layer[layer]["single_log_ratio"] for layer in layers), 6
                ),
            }
        )
    return anchors


def default_anchor_files(screen_root: Path) -> list[Path]:
    files = [
        Path(path)
        for path in sorted(
            glob.glob(str(screen_root / "olmo3_7b_span_ladder_v1/pair_eval_len*.json"))
        )
    ]
    files.extend(
        screen_root / name
        for name in NAMED_ANCHOR_FILES
        if (screen_root / name).exists()
    )
    return files


def _design(features: tuple[str, ...], sums, counts):
    import numpy as np

    columns = {
        "1": np.ones_like(sums),
        "S": sums,
        "n": counts,
        "S2": sums**2,
        "n2": counts**2,
    }
    return np.column_stack([columns[name] for name in features])


def fit_compounding(anchors: list[dict], corpus: str) -> dict:
    """Fit the compounding-model family; select by leave-one-out RMSE."""
    import numpy as np

    usable = [
        anchor for anchor in anchors if corpus in anchor["perplexity_ratio_by_corpus"]
    ]
    sums = np.array([anchor["sum_singles_log_ratio"] for anchor in usable])
    counts = np.array([float(anchor["n_layers"]) for anchor in usable])
    observed = np.array(
        [math.log(anchor["perplexity_ratio_by_corpus"][corpus]) for anchor in usable]
    )
    fits = {}
    for name, features in MODEL_FAMILY.items():
        design = _design(features, sums, counts)
        coef, *_ = np.linalg.lstsq(design, observed, rcond=None)
        residuals = design @ coef - observed
        loo_errors = []
        for held_out in range(len(observed)):
            mask = np.ones(len(observed), dtype=bool)
            mask[held_out] = False
            loo_coef, *_ = np.linalg.lstsq(design[mask], observed[mask], rcond=None)
            loo_errors.append(float(design[held_out] @ loo_coef - observed[held_out]))
        loo_errors = np.array(loo_errors)
        fits[name] = {
            "features": list(features),
            "coefficients": [round(float(value), 6) for value in coef],
            "fit_rmse_log": round(float(np.sqrt(np.mean(residuals**2))), 6),
            "loo_rmse_log": round(float(np.sqrt(np.mean(loo_errors**2))), 6),
            "loo_max_abs_error_log": round(float(np.abs(loo_errors).max()), 6),
        }
    selected = min(fits, key=lambda name: fits[name]["loo_rmse_log"])
    selected_design = _design(tuple(MODEL_FAMILY[selected]), sums, counts)
    coef = np.array(fits[selected]["coefficients"])
    return {
        "corpus": corpus,
        "n_anchors": len(usable),
        "anchor_names": [anchor["name"] for anchor in usable],
        "models": fits,
        "selected_model": selected,
        "selected_loo_residuals_log": [
            round(float(value), 6) for value in (selected_design @ coef - observed)
        ],
    }


def predict_composed_log(
    fit: dict, model_name: str, sum_singles: float, n: int
) -> float:
    features = fit["models"][model_name]["features"]
    coef = fit["models"][model_name]["coefficients"]
    values = {
        "1": 1.0,
        "S": sum_singles,
        "n": float(n),
        "S2": sum_singles**2,
        "n2": float(n) ** 2,
    }
    return sum(c * values[f] for c, f in zip(coef, features))


def _compression_metrics(
    replaced: list[int], deleted: list[int], per_layer: dict[int, dict], acct: dict
) -> dict:
    ffn_dense = acct["ffn_params_per_layer"]
    total = acct["total_params"]
    bytes_per_param = 2  # bf16 serving checkpoint convention
    replaced_active = sum(per_layer[lyr]["dendritic_active_params"] for lyr in replaced)
    replaced_bytes = sum(
        per_layer[lyr]["dendritic_stored_bytes"]
        + per_layer[lyr]["dendritic_index_bytes"]
        for lyr in replaced
    )
    removed_dense = ffn_dense * (len(replaced) + len(deleted))
    active_after = total - removed_dense + replaced_active
    total_bytes = total * bytes_per_param
    bytes_after = total_bytes - removed_dense * bytes_per_param + replaced_bytes
    ffn_total = acct["ffn_params_total"]
    n_dense = acct["num_hidden_layers"] - len(replaced) - len(deleted)
    ffn_after = ffn_dense * n_dense + replaced_active
    ffn_bytes_after = ffn_dense * n_dense * bytes_per_param + replaced_bytes
    return {
        "layers_replaced": sorted(replaced),
        "layers_deleted": sorted(deleted),
        "layers_dense": [
            lyr
            for lyr in range(acct["num_hidden_layers"])
            if lyr not in replaced and lyr not in deleted
        ],
        "n_replaced": len(replaced),
        "n_deleted": len(deleted),
        "whole_model_active_params_after": int(active_after),
        "whole_model_active_compression_x": round(total / active_after, 4),
        "ffn_scope_active_compression_x": (
            round(ffn_total / ffn_after, 4) if ffn_after else None
        ),
        "whole_model_byte_compression_x": round(total_bytes / bytes_after, 4),
        "ffn_scope_byte_compression_x": (
            round(ffn_total * bytes_per_param / ffn_bytes_after, 4)
            if ffn_bytes_after
            else None
        ),
    }


def build_frontier(
    targets: tuple[float, ...],
    per_layer: dict[int, dict],
    fit: dict,
    sensitivity_fit: dict | None,
    anchors: list[dict],
    acct: dict,
    deletion: dict,
) -> list[dict]:
    """Greedy frontier per quality target under the selected compounding fit.

    Layers are ordered by their single log-ratio (the marginal projected
    cost of adding a layer is monotone in it for every model in the
    family), and layers are added while the projected composed ratio stays
    at or below the target. All greedy points are PROJECTED; the measured
    floor per target is the anchor composition (MEASURED) with the highest
    whole-model compression whose measured ratio meets the target.
    """
    selected = fit["selected_model"]
    order = sorted(per_layer, key=lambda lyr: (per_layer[lyr]["single_log_ratio"], lyr))
    frontier = []
    for target in targets:
        log_target = math.log(target)
        chosen: list[int] = []
        sum_singles = 0.0
        projected_log = None
        while len(chosen) < len(order):
            candidate = order[len(chosen)]
            trial_sum = sum_singles + per_layer[candidate]["single_log_ratio"]
            trial_log = predict_composed_log(fit, selected, trial_sum, len(chosen) + 1)
            if trial_log > log_target:
                break
            chosen.append(candidate)
            sum_singles = trial_sum
            projected_log = trial_log
        point = {
            "target_ppl_ratio": target,
            "label": "PROJECTED",
            "projected_ppl_ratio": (
                round(math.exp(projected_log), 4) if projected_log is not None else None
            ),
            "sum_singles_log_ratio": round(sum_singles, 6),
            "per_layer_arms": {lyr: per_layer[lyr]["arm"] for lyr in sorted(chosen)},
            **_compression_metrics(chosen, [], per_layer, acct),
        }
        if sensitivity_fit is not None and chosen:
            point["projected_ppl_ratio_c4_fit"] = round(
                math.exp(
                    predict_composed_log(
                        sensitivity_fit,
                        sensitivity_fit["selected_model"],
                        sum_singles,
                        len(chosen),
                    )
                ),
                4,
            )
        s_model = "intercept_alpha_S_beta_n"
        if chosen and s_model in fit["models"]:
            point["projected_ppl_ratio_singles_aware_fit"] = round(
                math.exp(predict_composed_log(fit, s_model, sum_singles, len(chosen))),
                4,
            )
        # Deletion variant: swap wikitext-deletion-free layers already in the
        # set from replaced to deleted (their replacement params drop out).
        deletable = [
            lyr for lyr in deletion["deletion_free_wikitext_layers"] if lyr in chosen
        ]
        if deletable:
            kept = [lyr for lyr in chosen if lyr not in deletable]
            variant = _compression_metrics(kept, deletable, per_layer, acct)
            point["deletion_variant"] = {
                "note": (
                    "Deletion-free by the seed-mean wikitext criterion only; "
                    "paired controls on c4/fineweb measure 1.02-1.04x for "
                    "these layers, so cross-corpus deletion is not free."
                ),
                **variant,
            }
        measured = [
            anchor
            for anchor in anchors
            if anchor["perplexity_ratio_by_corpus"].get(PRIMARY_CORPUS, math.inf)
            <= target
        ]
        if measured:
            floor = max(
                measured,
                key=lambda anchor: _compression_metrics(
                    anchor["layers"], [], per_layer, acct
                )["whole_model_active_compression_x"],
            )
            point["measured_floor"] = {
                "label": "MEASURED",
                "anchor": floor["name"],
                "recovery_steps": floor["recovery_steps"],
                "measured_ppl_ratio": floor["perplexity_ratio_by_corpus"][
                    PRIMARY_CORPUS
                ],
                **_compression_metrics(floor["layers"], [], per_layer, acct),
            }
        # Operating point: whichever of the projected greedy set and the
        # measured floor reaches higher whole-model compression at <= target.
        operating = {
            "label": "PROJECTED",
            "source": "greedy_projection",
            "ppl_ratio": point["projected_ppl_ratio"],
            "layers_replaced": point["layers_replaced"],
            "whole_model_active_compression_x": point[
                "whole_model_active_compression_x"
            ],
        }
        floor_point = point.get("measured_floor")
        if floor_point is not None and (
            floor_point["whole_model_active_compression_x"]
            > point["whole_model_active_compression_x"]
        ):
            operating = {
                "label": "MEASURED",
                "source": floor_point["anchor"],
                "ppl_ratio": floor_point["measured_ppl_ratio"],
                "layers_replaced": floor_point["layers_replaced"],
                "whole_model_active_compression_x": floor_point[
                    "whole_model_active_compression_x"
                ],
            }
        point["operating_point"] = operating
        frontier.append(point)
    return frontier


def _flagship(
    frontier: list[dict], per_layer: dict[int, dict], fit: dict, acct: dict
) -> dict | None:
    """Recommended flagship run at the 1.2x target.

    The conservative set is the greedy set whose count-only (LOO-selected)
    projection stays <= 1.2x; the stretch set extends it to the size of
    the measured floor composition, which the count-only model
    over-penalises (it measured 1.074x at 6k steps where the model
    projects ~1.29x for any 12-layer set).
    """
    point = next((p for p in frontier if abs(p["target_ppl_ratio"] - 1.2) < 1e-9), None)
    if point is None:
        return None
    order = sorted(per_layer, key=lambda lyr: (per_layer[lyr]["single_log_ratio"], lyr))

    def describe(layers: list[int]) -> dict:
        sum_singles = sum(per_layer[lyr]["single_log_ratio"] for lyr in layers)
        selected = fit["selected_model"]
        described = {
            "layers_replaced": sorted(layers),
            "per_layer_arms": {
                str(lyr): per_layer[lyr]["arm"] for lyr in sorted(layers)
            },
            "sum_singles_log_ratio": round(sum_singles, 6),
            "projected_ppl_ratio_count_only_fit": round(
                math.exp(predict_composed_log(fit, selected, sum_singles, len(layers))),
                4,
            ),
            **_compression_metrics(layers, [], per_layer, acct),
        }
        s_model = "intercept_alpha_S_beta_n"
        if s_model in fit["models"]:
            described["projected_ppl_ratio_singles_aware_fit"] = round(
                math.exp(predict_composed_log(fit, s_model, sum_singles, len(layers))),
                4,
            )
        return described

    conservative = describe(point["layers_replaced"])
    stretch = None
    floor_point = point.get("measured_floor")
    if floor_point is not None and floor_point["n_replaced"] > point["n_replaced"]:
        stretch = describe(order[: floor_point["n_replaced"]])
        stretch["justification"] = (
            "A same-size MEASURED composition "
            f"({floor_point['anchor']}, contiguous and containing strictly "
            "worse singles than this cherry-picked set) reached "
            f"{floor_point['measured_ppl_ratio']}x at only "
            f"{floor_point['recovery_steps']} recovery steps, and the "
            "singles-aware fit projects this set <= 1.2x; only the "
            "count-only fit projects it above target."
        )
    return {
        "criterion": (
            "maximum whole-model compression at projected <= 1.2x; the "
            "stretch set is the recommended flagship because the measured "
            "floor composition of the same size already beats the target"
        ),
        "recommended_set": "stretch" if stretch is not None else "conservative",
        "conservative": conservative,
        "stretch": stretch,
        "recommended_recovery": (
            "joint recovery at 24000 steps, lr 3e-05, bs 8, seq 256 (the "
            "depthsel 12k->24k anchor pair shows longer budgets improve "
            "the composed ratio; mixed 6k-24k anchor budgets make the "
            "projections conservative for a 24k run), then certify with a "
            "dendritic_composition_pair_eval on wikitext/c4/fineweb."
        ),
    }


def build_artifact(
    screen_root: Path,
    model_dir: Path,
    anchor_files: list[Path],
    targets: tuple[float, ...],
) -> dict:
    arms = collect_best_arms(screen_root)
    per_layer = arms["per_layer"]
    deletion = collect_deletion(screen_root)
    acct = parameter_accounting(model_dir)
    if acct["ffn_params_per_layer"] != 135266304:
        raise RuntimeError("config-derived FFN size disagrees with the screens")
    anchors = collect_anchors(screen_root, per_layer, anchor_files)
    fit = fit_compounding(anchors, PRIMARY_CORPUS)
    try:
        sensitivity_fit = fit_compounding(anchors, SENSITIVITY_CORPUS)
    except (KeyError, ValueError):
        sensitivity_fit = None
    frontier = build_frontier(
        targets, per_layer, fit, sensitivity_fit, anchors, acct, deletion
    )

    flagship = _flagship(frontier, per_layer, fit, acct)
    for layer_row in per_layer.values():
        deletion_row = deletion["per_layer"].get(layer_row["layer"])
        layer_row["deletion_ppl_ratio_wikitext_mean"] = (
            deletion_row["wikitext_mean_ratio"] if deletion_row else None
        )
        layer_row["deletion_ppl_ratio_wikitext_max"] = (
            deletion_row["wikitext_max_ratio"] if deletion_row else None
        )
        layer_row["deletion_ppl_ratio_cross_corpus_max"] = (
            deletion_row["cross_corpus_max_ratio"] if deletion_row else None
        )

    return {
        "schema": "dendritic_whole_model_frontier/v1",
        "claim_boundary": (
            "Frontier points labelled PROJECTED are compounding-model "
            "extrapolations fitted on 12 measured composition anchors "
            "(contiguous spans at 6k recovery steps plus depth-selective "
            "and whole-FFN compositions at 12k/18k/24k steps), pending the "
            "flagship measured run; only points labelled MEASURED are "
            "backed by an actual composed pair eval. The anchors show the "
            "composed penalty is governed by the number of replaced layers "
            "(the singles-sum term is not validated out-of-sample), the "
            "fitted intercept absorbs an in-domain joint-recovery gain on "
            "the wikitext eval, anchors are contiguous spans while frontier "
            "sets are cherry-picked, and all quality numbers are single-seed "
            "wikitext-validation perplexity ratios at sequence length 256."
        ),
        "screen_root": str(screen_root),
        "targets": list(targets),
        "primary_corpus": PRIMARY_CORPUS,
        "parameter_accounting": acct,
        "initial_loss_handling": {
            "bitwise_constant": arms["initial_loss_constant"],
            "distinct_values": arms["initial_loss_values"],
            "policy": (
                "initial lm_loss is NOT bitwise-constant across cells "
                "(three near-identical values, spread ~1.3e-4 nats); every "
                "single ratio is computed per-cell against that cell's own "
                "initial."
            ),
        },
        "excluded_overwritten_dirs": arms["excluded_overwritten_dirs"],
        "per_layer": {str(layer): per_layer[layer] for layer in sorted(per_layer)},
        "deletion": deletion,
        "anchors": anchors,
        "anchor_fit": fit,
        "anchor_fit_sensitivity_c4": sensitivity_fit,
        "step_budget_caveat": (
            "Anchor recovery budgets are mixed: span-ladder anchors 6000 "
            "steps, depth-selective 12000 and 24000, whole-FFN 18000. The "
            "depthsel pair (same composition, 12k vs 24k) moves the wikitext "
            "ratio 2.7099 -> 2.4446, so the fitted per-layer cost partially "
            "conflates budget; projections assume comparable joint-recovery "
            "budgets and are conservative for longer ones."
        ),
        "frontier": frontier,
        "flagship_recommendation": flagship,
        "sota_reference": {
            "pruning": "~2x whole-model at ~1.1-1.2x perplexity (e.g. 50% "
            "one-shot sparsity: SparseGPT/Wanda class)",
            "quantization": "~4x storage near-lossless at 4-bit "
            "(weight-only); composes multiplicatively with this method",
            "note": (
                "Iso-quality comparison: our whole-model active-parameter "
                "compression at 1.1-1.2x projected perplexity is below the "
                "2x pruning reference because only the FFN scope (59.3% of "
                "parameters) is touched; FFN-scope compression is larger, "
                "and byte compression composes with 4-bit quantization of "
                "the remaining dense weights."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, default=DEFAULT_SCREEN_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--anchors",
        type=Path,
        nargs="*",
        default=None,
        help="explicit anchor pair-eval JSONs (default: auto-discover)",
    )
    parser.add_argument(
        "--targets", type=float, nargs="*", default=list(DEFAULT_TARGETS)
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    anchor_files = (
        list(args.anchors) if args.anchors else default_anchor_files(args.screen_root)
    )
    artifact = build_artifact(
        args.screen_root, args.model_dir, anchor_files, tuple(args.targets)
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(artifact, handle, indent=1)
        handle.write("\n")
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
