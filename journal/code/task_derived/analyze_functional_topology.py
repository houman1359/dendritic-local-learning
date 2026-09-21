#!/usr/bin/env python3
"""Exploratory functional-similarity test on real MICrONS dendritic topology.

This analysis deliberately treats the neuron, not synapse pairs, as the future
unit of replication.  Pairwise permutation p-values reported for a single
target are screening statistics and must not be interpreted as population
inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPED = PROJECT / "reproduced_results" / "microns_morphology_credit" / "mapped_synapses.csv.gz"
DEFAULT_SEGMENTS = PROJECT / "reproduced_results" / "microns_morphology_credit" / "segment_metrics.csv"
DEFAULT_PARTNERS = PROJECT / "reproduced_results" / "microns_functional_partner_manifest" / "functional_partner_manifest.csv"


def condition_means(
    responses: np.ndarray, stimulus_ids: np.ndarray, min_repeats: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(
        [
            value
            for value in np.unique(stimulus_ids)
            if np.sum(stimulus_ids == value) >= int(min_repeats)
        ],
        dtype=int,
    )
    means = np.vstack([np.nanmean(responses[stimulus_ids == value], axis=0) for value in unique])
    return unique, means


def repeat_reliability(responses: np.ndarray, stimulus_ids: np.ndarray) -> np.ndarray:
    left, right = [], []
    for value in np.unique(stimulus_ids):
        indices = np.flatnonzero(stimulus_ids == value)
        if len(indices) < 2:
            continue
        left.append(np.nanmean(responses[indices[::2]], axis=0))
        right.append(np.nanmean(responses[indices[1::2]], axis=0))
    if len(left) < 3:
        return np.full(responses.shape[1], np.nan)
    a, b = np.asarray(left), np.asarray(right)
    return np.asarray(
        [spearmanr(a[:, index], b[:, index]).statistic for index in range(a.shape[1])],
        dtype=float,
    )


def ancestor_chain(segment: int, parents: dict[int, int]) -> list[int]:
    chain = [int(segment)]
    seen = set(chain)
    while parents.get(chain[-1], -1) >= 0:
        parent = int(parents[chain[-1]])
        if parent in seen:
            raise ValueError("cycle in segment tree")
        chain.append(parent)
        seen.add(parent)
    return chain


def lca(a: int, b: int, parents: dict[int, int]) -> int:
    ancestors_a = set(ancestor_chain(a, parents))
    for value in ancestor_chain(b, parents):
        if value in ancestors_a:
            return int(value)
    return 0


def first_level_branch(segment: int, parents: dict[int, int]) -> int:
    chain = ancestor_chain(segment, parents)
    return int(chain[-2]) if len(chain) >= 2 else int(chain[-1])


def dominant_contact(contact: pd.DataFrame) -> pd.Series:
    grouped = (
        contact.groupby("segment_id", dropna=False)
        .agg(size=("size", "sum"), n_synapses=("id", "size"), x=("x_um", "mean"), y=("y_um", "mean"), z=("z_um", "mean"))
        .reset_index()
        .sort_values(["size", "n_synapses"], ascending=[False, False])
    )
    return grouped.iloc[0]


def partial_rank_correlation(y: np.ndarray, x: np.ndarray, controls: np.ndarray) -> float:
    ranked_y = rankdata(y)
    ranked_x = rankdata(x)
    ranked_controls = np.column_stack([rankdata(controls[:, index]) for index in range(controls.shape[1])])
    design = np.column_stack([np.ones(len(y)), ranked_controls])
    residual_y = ranked_y - design @ np.linalg.lstsq(design, ranked_y, rcond=None)[0]
    residual_x = ranked_x - design @ np.linalg.lstsq(design, ranked_x, rcond=None)[0]
    return float(np.corrcoef(residual_y, residual_x)[0, 1])


def permutation_p(observed: float, null: np.ndarray, alternative: str = "greater") -> float:
    if alternative == "greater":
        exceed = np.sum(null >= observed)
    elif alternative == "less":
        exceed = np.sum(null <= observed)
    else:
        exceed = np.sum(np.abs(null) >= abs(observed))
    return float((exceed + 1) / (len(null) + 1))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument("--mapped-synapses", type=Path, default=DEFAULT_MAPPED)
    parser.add_argument("--segment-metrics", type=Path, default=DEFAULT_SEGMENTS)
    parser.add_argument("--partner-manifest", type=Path, default=DEFAULT_PARTNERS)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    extract_dir = args.extract_dir.resolve()
    outdir = args.outdir or extract_dir / "functional_topology"
    outdir.mkdir(parents=True, exist_ok=True)
    protocol = np.load(extract_dir / "microns_dandi_trial_protocol.npz")
    mapping = pd.read_csv(extract_dir / "microns_dandi_trial_unit_mapping.csv")
    if "role" not in mapping.columns or (mapping["role"] == "postsynaptic_target").sum() != 1:
        raise ValueError("extraction must contain exactly one postsynaptic target")
    target_row = mapping[mapping["role"] == "postsynaptic_target"].iloc[0]
    target_nucleus = int(target_row["nucleus_id"])
    target_root = int(target_row["target_root_id"] if pd.notna(target_row.get("target_root_id")) else target_row["post_pt_root_id"])

    responses = np.asarray(protocol["responses"], dtype=float)
    stimulus_ids = np.asarray(protocol["stimulus_ids"], dtype=int)
    repeated_ids, tuning = condition_means(responses, stimulus_ids, min_repeats=2)
    _, tuning_all = condition_means(responses, stimulus_ids, min_repeats=1)
    reliability = repeat_reliability(responses, stimulus_ids)
    partner_indices = np.flatnonzero(mapping["role"].eq("presynaptic_partner").to_numpy())
    partner_mapping = mapping.iloc[partner_indices].reset_index(drop=True)
    partner_tuning = tuning[:, partner_indices]
    partner_tuning_all = tuning_all[:, partner_indices]
    partner_reliability = reliability[partner_indices]
    functional_similarity = np.corrcoef(partner_tuning.T)
    functional_similarity_all = np.corrcoef(partner_tuning_all.T)

    mapped = pd.read_csv(args.mapped_synapses)
    contacts = mapped[
        (mapped["root_id"] == target_root)
        & mapped["mapping_pass"]
        & mapped["pre_pt_root_id"].isin(partner_mapping["post_pt_root_id"])
    ].copy()
    segments = pd.read_csv(args.segment_metrics)
    segments = segments[segments["root_id"] == target_root].copy()
    parents = dict(zip(segments["segment_id"].astype(int), segments["parent_segment_id"].astype(int)))
    path_um = dict(zip(segments["segment_id"].astype(int), segments["path_length_um"].astype(float)))
    depth = dict(zip(segments["segment_id"].astype(int), segments["topological_depth"].astype(float)))

    contact_rows = []
    for unit_index, unit in partner_mapping.iterrows():
        root = int(unit["post_pt_root_id"])
        contact = contacts[contacts["pre_pt_root_id"] == root]
        if contact.empty:
            continue
        dominant = dominant_contact(contact)
        segment = int(dominant["segment_id"])
        contact_rows.append(
            {
                "unit_index": int(unit_index),
                "pre_pt_root_id": root,
                "nucleus_id": int(unit["nucleus_id"]),
                "segment_id": segment,
                "major_branch": first_level_branch(segment, parents),
                "path_um": float(path_um[segment]),
                "depth": float(depth[segment]),
                "x_um": float(dominant["x"]),
                "y_um": float(dominant["y"]),
                "z_um": float(dominant["z"]),
                "n_synapses": int(len(contact)),
                "contact_size": float(contact["size"].sum()),
                "repeat_reliability": float(partner_reliability[unit_index]),
            }
        )
    contact_table = pd.DataFrame(contact_rows).sort_values("unit_index").reset_index(drop=True)
    if len(contact_table) < 5:
        raise ValueError("fewer than five functional partners have mapped dendritic contacts")

    pair_rows = []
    for left in range(len(contact_table)):
        for right in range(left + 1, len(contact_table)):
            a, b = contact_table.iloc[left], contact_table.iloc[right]
            seg_a, seg_b = int(a.segment_id), int(b.segment_id)
            common = lca(seg_a, seg_b, parents)
            tree_distance = path_um[seg_a] + path_um[seg_b] - 2.0 * path_um[common]
            euclidean = float(
                np.linalg.norm(
                    a[["x_um", "y_um", "z_um"]].to_numpy(float)
                    - b[["x_um", "y_um", "z_um"]].to_numpy(float)
                )
            )
            pair_rows.append(
                {
                    "left": left,
                    "right": right,
                    "functional_similarity": float(functional_similarity[int(a.unit_index), int(b.unit_index)]),
                    "tree_distance_um": float(tree_distance),
                    "shared_path_um": float(path_um[common]),
                    "shared_path_fraction": float(path_um[common] / max(min(path_um[seg_a], path_um[seg_b]), 1e-9)),
                    "lca_depth": float(depth[common]),
                    "euclidean_distance_um": euclidean,
                    "path_depth_difference_um": abs(path_um[seg_a] - path_um[seg_b]),
                    "same_major_branch": bool(a.major_branch == b.major_branch),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    y = pairs["functional_similarity"].to_numpy(float)
    shared = pairs["shared_path_fraction"].to_numpy(float)
    tree_distance = pairs["tree_distance_um"].to_numpy(float)
    controls = np.column_stack(
        [
            np.log1p(pairs["euclidean_distance_um"].to_numpy(float)),
            np.log1p(pairs["path_depth_difference_um"].to_numpy(float)),
        ]
    )
    observed_shared = float(spearmanr(y, shared).statistic)
    observed_tree = float(spearmanr(y, -tree_distance).statistic)
    observed_partial = partial_rank_correlation(y, shared, controls)
    same_mask = pairs["same_major_branch"].to_numpy(bool)
    observed_branch_delta = float(np.mean(y[same_mask]) - np.mean(y[~same_mask])) if same_mask.any() and (~same_mask).any() else float("nan")
    y_all = functional_similarity_all[pairs["left"].to_numpy(int), pairs["right"].to_numpy(int)]
    all_condition_sensitivity = {
        "shared_path_spearman_r": float(spearmanr(y_all, shared).statistic),
        "negative_tree_distance_spearman_r": float(spearmanr(y_all, -tree_distance).statistic),
        "shared_path_partial_r": partial_rank_correlation(y_all, shared, controls),
    }

    rng = np.random.default_rng(args.seed)
    null_shared = np.empty(args.permutations)
    null_tree = np.empty(args.permutations)
    null_partial = np.empty(args.permutations)
    null_branch = np.empty(args.permutations)
    pair_left = pairs["left"].to_numpy(int)
    pair_right = pairs["right"].to_numpy(int)
    for index in range(args.permutations):
        order = rng.permutation(len(contact_table))
        permuted_y = functional_similarity[order[pair_left], order[pair_right]]
        null_shared[index] = spearmanr(permuted_y, shared).statistic
        null_tree[index] = spearmanr(permuted_y, -tree_distance).statistic
        null_partial[index] = partial_rank_correlation(permuted_y, shared, controls)
        null_branch[index] = np.mean(permuted_y[same_mask]) - np.mean(permuted_y[~same_mask]) if np.isfinite(observed_branch_delta) else np.nan

    manual_manifest = pd.read_csv(args.partner_manifest)
    manual_roots = set(
        manual_manifest[
            (manual_manifest["target_nucleus_id"] == target_nucleus)
            & (manual_manifest["session"] == int(target_row["session"]))
            & (manual_manifest["scan_idx"] == int(target_row["scan_idx"]))
            & (manual_manifest["match_tier"] == "manual")
        ]["pre_pt_root_id"].astype(np.int64)
    )
    contact_table["manual_match"] = contact_table["pre_pt_root_id"].isin(manual_roots)
    contact_table.to_csv(outdir / "functional_contacts.csv", index=False)
    pairs.to_csv(outdir / "functional_contact_pairs.csv", index=False)

    summary = {
        "claim_level": "single-target exploratory topology/function screening test",
        "target_nucleus_id": target_nucleus,
        "target_root_id": target_root,
        "session": int(target_row["session"]),
        "scan_idx": int(target_row["scan_idx"]),
        "n_presynaptic_partners": int(len(contact_table)),
        "n_manual_partner_subset": int(contact_table["manual_match"].sum()),
        "n_pairs": int(len(pairs)),
        "functional_similarity_definition": "Pearson correlation across condition-mean responses for conditions with at least two repeats",
        "n_repeated_conditions_for_reliability": int(len(repeated_ids)),
        "median_repeat_reliability": float(np.nanmedian(contact_table["repeat_reliability"])),
        "tests": {
            "shared_path_fraction": {
                "spearman_r": observed_shared,
                "permutation_p_greater": permutation_p(observed_shared, null_shared),
            },
            "negative_tree_distance": {
                "spearman_r": observed_tree,
                "permutation_p_greater": permutation_p(observed_tree, null_tree),
            },
            "shared_path_partial_euclidean_and_depth": {
                "partial_rank_r": observed_partial,
                "permutation_p_greater": permutation_p(observed_partial, null_partial),
            },
            "same_major_branch": {
                "mean_similarity_delta": observed_branch_delta,
                "permutation_p_greater": permutation_p(observed_branch_delta, null_branch),
            },
        },
        "all_condition_sensitivity": all_condition_sensitivity,
        "interpretation": (
            "Screening evidence only: automatic coregistration expansion, one target neuron, incomplete excitatory-input coverage, and pairwise observations are not independent."
        ),
    }
    write_json(outdir / "summary.json", summary)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    axes[0].scatter(shared, y, s=22, alpha=0.72, color="#33658a")
    axes[0].set_xlabel("shared path fraction")
    axes[0].set_ylabel("stimulus-response similarity")
    axes[0].set_title(f"r={observed_shared:.2f}, perm. p={summary['tests']['shared_path_fraction']['permutation_p_greater']:.3f}")
    axes[1].scatter(tree_distance, y, s=22, alpha=0.72, color="#5b8c5a")
    axes[1].set_xlabel("tree distance (µm)")
    axes[1].set_ylabel("response similarity")
    axes[1].set_title(f"r with -distance={observed_tree:.2f}")
    axes[2].hist(null_partial, bins=35, color="#d8d8d8", edgecolor="white")
    axes[2].axvline(observed_partial, color="#b33a3a", linewidth=2.2)
    axes[2].set_xlabel("partial rank correlation under label shuffle")
    axes[2].set_ylabel("permutations")
    axes[2].set_title(f"observed={observed_partial:.2f}")
    fig.suptitle(
        f"MICrONS target {target_nucleus}: functional input similarity versus real dendritic topology",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outdir / "functional_topology.png", dpi=220)
    fig.savefig(outdir / "functional_topology.pdf")
    plt.close(fig)

    lines = [
        "# Functional similarity on a real MICrONS dendritic tree",
        "",
        f"Target nucleus `{target_nucleus}`, session/scan {summary['session']}/{summary['scan_idx']}.",
        "",
        f"- Connected presynaptic partners with streamed responses and mapped contacts: {summary['n_presynaptic_partners']}.",
        f"- Expert/manual subset: {summary['n_manual_partner_subset']}.",
        f"- Pairwise comparisons: {summary['n_pairs']}.",
        f"- Median repeat reliability: {summary['median_repeat_reliability']:.3f}.",
        f"- Functional similarity versus shared path: r={observed_shared:.3f}, permutation p={summary['tests']['shared_path_fraction']['permutation_p_greater']:.4f}.",
        f"- Functional similarity versus negative tree distance: r={observed_tree:.3f}, permutation p={summary['tests']['negative_tree_distance']['permutation_p_greater']:.4f}.",
        f"- Shared path after Euclidean/depth controls: partial r={observed_partial:.3f}, permutation p={summary['tests']['shared_path_partial_euclidean_and_depth']['permutation_p_greater']:.4f}.",
        f"- Same-major-branch similarity delta: {observed_branch_delta:.3f}, permutation p={summary['tests']['same_major_branch']['permutation_p_greater']:.4f}.",
        "",
        "This is a screening result from one target and an automatic-match expansion tier. It is not population evidence and must not enter the abstract as confirmation unless it replicates across independently selected targets using the manual tier or a preregistered quality threshold.",
    ]
    (outdir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
