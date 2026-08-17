#!/usr/bin/env python3
"""Summarize whether routed models learn and use pathway structure.

This script is intentionally complementary to the activation-based
``analyze_inhibitory_path_gain.py`` diagnostic. It reads finished run folders
without replaying data and reports:

- router assignment to known pathway groups, when the router is learned;
- excitatory branch specialization to pathway chunks;
- inhibitory branch specialization to pathway chunks;
- seed-level and grouped performance for learned-I versus no-I controls.

The static branch metrics answer "did the learned synapses become pathway
selective?". The activation diagnostic answers "is inhibition active on the
task-relevant or task-irrelevant path for a given input?".
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
DRAFT_ROOT = REPO_ROOT / "drafts" / "dendritic-local-learning"
ANALYSIS_ROOT = DRAFT_ROOT / "analysis"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _nested(data: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _strip_seed(name: str) -> tuple[str, int | None]:
    match = re.match(r"(.+)_s(\d+)$", name)
    if match:
        return match.group(1), int(match.group(2))
    return name, None


def _load_state_dict(run_dir: Path) -> dict[str, torch.Tensor] | None:
    for candidate in (
        run_dir / "final_model.pt",
        run_dir / "main_network" / "local_learning_best_model.pt",
        run_dir / "main_network" / "standard_best_model.pt",
    ):
        if candidate.exists():
            state = torch.load(candidate, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if isinstance(state, dict):
                return state
    return None


def _positive_weight(pre_w: torch.Tensor, transform: str) -> torch.Tensor:
    transform = str(transform or "softplus").lower()
    if transform == "exp":
        return torch.exp(pre_w)
    if transform == "softplus":
        return torch.nn.functional.softplus(pre_w)
    if transform == "relu":
        return torch.relu(pre_w)
    if transform == "identity":
        return pre_w
    raise ValueError(f"Unsupported weight_transform={transform!r}")


def _topk_pruned_weight(
    pre_w: torch.Tensor,
    *,
    k: int,
    transform: str,
) -> torch.Tensor:
    if k <= 0:
        return torch.zeros_like(pre_w)
    k = min(int(k), int(pre_w.shape[1]))
    scores = pre_w.abs() if str(transform).lower() == "identity" else pre_w
    indices = torch.topk(scores, k, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(pre_w)
    mask.scatter_(1, indices, 1.0)
    return mask * _positive_weight(pre_w, transform)


def _pathway_groups_from_config(cfg: dict[str, Any]) -> tuple[list[list[int]], list[int]]:
    params = _nested(cfg, ("model", "encoder", "params"), {}) or {}
    input_dim = int(params.get("input_dim", 0) or 0)
    shared_indices = [int(x) for x in params.get("shared_indices", []) or []]
    shared_set = set(shared_indices)

    raw_groups = params.get("pathway_groups")
    if raw_groups:
        groups = [[int(idx) for idx in group] for group in raw_groups]
    else:
        n_pathways = int(params.get("n_pathways", 2) or 2)
        routed_pool = [idx for idx in range(input_dim) if idx not in shared_set]
        step = max(1, len(routed_pool) // max(1, n_pathways))
        groups = []
        start = 0
        for pathway_idx in range(n_pathways):
            end = len(routed_pool) if pathway_idx == n_pathways - 1 else start + step
            groups.append(routed_pool[start:end])
            start = end

    routed_indices = sorted(idx for group in groups for idx in group)
    return groups, routed_indices


def _pathway_output_slices(cfg: dict[str, Any]) -> list[slice]:
    params = _nested(cfg, ("model", "encoder", "params"), {}) or {}
    raw_dims = params.get("pathway_dims")
    if raw_dims:
        dims = [int(x) for x in raw_dims]
    else:
        n_pathways = int(params.get("n_pathways", 2) or 2)
        if params.get("pathway_groups"):
            n_pathways = len(params["pathway_groups"])
        pathway_dim = int(params.get("pathway_dim", 0) or 0)
        if pathway_dim <= 0:
            groups, _ = _pathway_groups_from_config(cfg)
            shared = params.get("shared_indices", []) or []
            dims = [len(shared) + len(group) for group in groups]
        else:
            dims = [pathway_dim] * n_pathways

    slices: list[slice] = []
    start = 0
    for dim in dims:
        stop = start + int(dim)
        slices.append(slice(start, stop))
        start = stop
    return slices


def _router_metrics(
    cfg: dict[str, Any],
    state: dict[str, torch.Tensor] | None,
) -> dict[str, float | str | bool]:
    params = _nested(cfg, ("model", "encoder", "params"), {}) or {}
    if _nested(cfg, ("model", "encoder", "type")) != "pathway_router":
        return {}

    router_mode = str(params.get("router_mode", "fixed")).lower()
    groups, routed_indices = _pathway_groups_from_config(cfg)
    out: dict[str, float | str | bool] = {
        "router_mode": router_mode,
        "router_learned": router_mode == "learned",
        "n_pathway_groups": float(len(groups)),
        "n_routed_features": float(len(routed_indices)),
    }

    if router_mode != "learned" or state is None:
        out.update(
            {
                "router_expected_assignment_accuracy": 1.0
                if router_mode == "fixed"
                else float("nan"),
                "router_expected_probability": 1.0
                if router_mode == "fixed"
                else float("nan"),
                "router_assignment_entropy_norm": 0.0
                if router_mode == "fixed"
                else float("nan"),
                "router_assignment_margin": 1.0
                if router_mode == "fixed"
                else float("nan"),
            }
        )
        return out

    logits_key = next((key for key in state if key.endswith("assignment_logits")), None)
    if logits_key is None:
        return out
    logits = state[logits_key].float()
    temperature = float(params.get("learned_router_temperature", 1.0) or 1.0)
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)

    expected_by_feature: dict[int, int] = {}
    for pathway_idx, group in enumerate(groups):
        for feature_idx in group:
            expected_by_feature[int(feature_idx)] = int(pathway_idx)
    expected = torch.tensor(
        [expected_by_feature[int(feature_idx)] for feature_idx in routed_indices],
        dtype=torch.long,
    )
    if expected.numel() != probs.shape[0]:
        out["router_shape_mismatch"] = True
        return out

    chosen = probs.argmax(dim=-1)
    expected_prob = probs[torch.arange(probs.shape[0]), expected]
    other = probs.clone()
    other[torch.arange(probs.shape[0]), expected] = -float("inf")
    entropy = -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=-1)
    entropy_norm = entropy / math.log(max(probs.shape[1], 2))

    out.update(
        {
            "router_shape_mismatch": False,
            "router_expected_assignment_accuracy": float(
                (chosen == expected).float().mean().item()
            ),
            "router_expected_probability": float(expected_prob.mean().item()),
            "router_assignment_entropy_norm": float(entropy_norm.mean().item()),
            "router_assignment_margin": float(
                (expected_prob - other.max(dim=-1).values).mean().item()
            ),
        }
    )
    return out


def _branch_pathway_metrics(
    cfg: dict[str, Any],
    state: dict[str, torch.Tensor] | None,
    *,
    population: str,
) -> dict[str, float | bool]:
    if state is None or _nested(cfg, ("model", "encoder", "type")) != "pathway_router":
        return {}

    if population == "excitatory":
        key_suffix = "branch_excitation.pre_w"
        k_values = _nested(
            cfg,
            ("model", "core", "connectivity", "ee_synapses_per_branch_per_layer"),
            [],
        )
    elif population == "inhibitory":
        key_suffix = "branch_inhibition.pre_w"
        k_values = _nested(
            cfg,
            ("model", "core", "connectivity", "ie_synapses_per_branch_per_layer"),
            [],
        )
    else:
        raise ValueError(population)

    k = int(k_values[0]) if k_values else 0
    prefix = "e_pathway" if population == "excitatory" else "i_pathway"
    if k <= 0:
        return {
            f"{prefix}_available": False,
            f"{prefix}_branch_purity": float("nan"),
            f"{prefix}_best_branch_alignment": float("nan"),
            f"{prefix}_diagonal_branch_alignment": float("nan"),
            f"{prefix}_specialized_branch_fraction": float("nan"),
        }

    pre_key = next(
        (
            key
            for key in state
            if key.endswith(key_suffix)
            and ".branch_layers.0." in key
            and ".excitatory_cells." in key
        ),
        None,
    )
    if pre_key is None:
        return {f"{prefix}_available": False}

    pre_w = state[pre_key].float()
    transform = _nested(cfg, ("model", "core", "morphology", "weight_transform"), "softplus")
    w = _topk_pruned_weight(pre_w, k=k, transform=transform)
    layer_sizes = _nested(cfg, ("model", "core", "architecture", "excitatory_layer_sizes"), [])
    n_soma = int(layer_sizes[0]) if layer_sizes else 0
    if n_soma <= 0 or w.shape[0] % n_soma != 0:
        return {f"{prefix}_available": False}

    branch_factor = int(w.shape[0] // n_soma)
    slices = _pathway_output_slices(cfg)
    n_paths = len(slices)
    if n_paths < 2:
        return {f"{prefix}_available": False}

    w_by_branch = w.reshape(n_soma, branch_factor, w.shape[1])
    masses = torch.stack(
        [w_by_branch[:, :, sl].sum(dim=-1) for sl in slices],
        dim=-1,
    )  # [n_soma, branch_factor, n_paths]
    total = masses.sum().clamp_min(1e-12)
    branch_totals = masses.sum(dim=-1).clamp_min(1e-12)
    purity = (masses.max(dim=-1).values / branch_totals).mean()
    specialized = ((masses.max(dim=-1).values / branch_totals) >= 0.60).float().mean()

    diagonal = float("nan")
    if branch_factor == n_paths:
        diagonal = float(
            sum(masses[:, idx, idx].sum() for idx in range(n_paths)).item()
            / float(total.item())
        )

    best = float("nan")
    per_soma_best = float("nan")
    if branch_factor >= n_paths:
        best_val = 0.0
        per_soma_vals = torch.zeros(n_soma, dtype=masses.dtype)
        per_soma_total = masses.sum(dim=(1, 2)).clamp_min(1e-12)
        for perm in itertools.permutations(range(branch_factor), n_paths):
            per_soma_val = sum(
                masses[:, branch_idx, path_idx]
                for path_idx, branch_idx in enumerate(perm)
            )
            val = per_soma_val.sum()
            best_val = max(best_val, float(val.item()))
            per_soma_vals = torch.maximum(per_soma_vals, per_soma_val)
        best = best_val / float(total.item())
        per_soma_best = float((per_soma_vals / per_soma_total).mean().item())
    elif branch_factor == n_paths:
        best = diagonal

    return {
        f"{prefix}_available": True,
        f"{prefix}_branch_purity": float(purity.item()),
        f"{prefix}_best_branch_alignment": best,
        f"{prefix}_per_soma_best_branch_alignment": per_soma_best,
        f"{prefix}_diagonal_branch_alignment": diagonal,
        f"{prefix}_specialized_branch_fraction": float(specialized.item()),
        f"{prefix}_branch_factor": float(branch_factor),
    }


def _run_row(run_dir: Path) -> dict[str, Any] | None:
    cfg = _read_json(run_dir / "config.json")
    if cfg is None:
        return None
    final = _read_json(run_dir / "performance" / "final.json")
    run_name = str(_nested(cfg, ("outputs", "run_name"), run_dir.name))
    condition, parsed_seed = _strip_seed(run_name)
    state = _load_state_dict(run_dir)

    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "condition": condition,
        "seed": _nested(cfg, ("experiment", "seed"), parsed_seed),
        "status": "complete" if final is not None else "incomplete",
        "dataset": _nested(cfg, ("data", "dataset_name")),
        "core": _nested(cfg, ("model", "core", "type")),
        "strategy": _nested(cfg, ("training", "main", "strategy")),
        "broadcast_mode": _nested(
            cfg,
            ("training", "main", "learning_strategy_config", "error_broadcast_mode"),
        ),
        "broadcast_rank": _nested(
            cfg,
            ("training", "main", "learning_strategy_config", "broadcast_rank"),
        ),
    }
    ie_values = _nested(
        cfg,
        ("model", "core", "connectivity", "ie_synapses_per_branch_per_layer"),
        [],
    )
    row["has_i_to_e"] = bool(ie_values) and any(float(v) > 0 for v in ie_values)
    if final is not None:
        acc = final.get("accuracy", {})
        row["train_accuracy"] = acc.get("train")
        row["valid_accuracy"] = acc.get("valid")
        row["test_accuracy"] = acc.get("test")

    row.update(_router_metrics(cfg, state))
    row.update(_branch_pathway_metrics(cfg, state, population="excitatory"))
    row.update(_branch_pathway_metrics(cfg, state, population="inhibitory"))
    return row


def summarize(sweep_root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dirs = sorted(
        (sweep_root / "results").glob("config_*"),
        key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else p.name,
    )
    rows = [row for run_dir in run_dirs if (row := _run_row(run_dir)) is not None]
    detailed = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(out_dir / "pathway_learning_detailed.csv", index=False)

    complete = detailed[detailed.get("status", "") == "complete"].copy()
    metric_cols = [
        col
        for col in complete.columns
        if col.endswith("_accuracy")
        or col.startswith(("router_", "e_pathway_", "i_pathway_"))
    ]
    metric_cols = [
        col
        for col in metric_cols
        if pd.api.types.is_numeric_dtype(complete[col])
        and not col.endswith("_available")
    ]
    if complete.empty:
        grouped = pd.DataFrame()
    else:
        grouped = (
            complete.groupby(
                [
                    "condition",
                    "dataset",
                    "core",
                    "strategy",
                    "broadcast_mode",
                    "broadcast_rank",
                    "has_i_to_e",
                ],
                dropna=False,
            )[metric_cols]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped.columns = [
            "_".join(str(part) for part in col if str(part))
            if isinstance(col, tuple)
            else str(col)
            for col in grouped.columns
        ]
    grouped.to_csv(out_dir / "pathway_learning_grouped.csv", index=False)
    return detailed, grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Defaults to analysis/<sweep-name>_pathway_learning_<date>.",
    )
    args = parser.parse_args()

    sweep_root = args.sweep_root.resolve()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (
            ANALYSIS_ROOT
            / f"{sweep_root.name}_pathway_learning_{date.today().strftime('%Y%m%d')}"
        )
    detailed, grouped = summarize(sweep_root, out_dir)
    n_complete = int((detailed["status"] == "complete").sum()) if not detailed.empty else 0
    print(f"Sweep: {sweep_root}")
    print(f"Detailed rows: {len(detailed)} ({n_complete} complete)")
    print(f"Grouped rows: {len(grouped)}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
