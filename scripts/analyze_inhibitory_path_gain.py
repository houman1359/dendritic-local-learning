#!/usr/bin/env python3
"""Analyze input-driven inhibition and path-gain suppression in cue runs.

This diagnostic is aimed at the paper-facing ``input_mode=1`` setting, where
the same nonnegative routed pathway vector drives E and I synapses directly.
For cue integration, the encoder exposes two routed pathways:

    path 0 = [context, cue A]
    path 1 = [context, cue B]

The script measures whether first-layer inhibitory synaptic activation is
larger on the task-irrelevant pathway and whether larger inhibition coincides
with lower local input resistance, the shunting mechanism that lowers path gain.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dendritic_modeling.config import load_config
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment


ANALYSIS_DATE = "20260425"


def _strip_legacy_fields(obj: Any) -> None:
    """Remove fields produced by older sweeps that current config dataclasses reject."""
    if isinstance(obj, dict):
        obj.pop("match_additive_init_to_shunting", None)
        for value in obj.values():
            _strip_legacy_fields(value)
    elif isinstance(obj, list):
        for value in obj:
            _strip_legacy_fields(value)


def _load_config_compat(config_path: Path):
    raw = json.loads(config_path.read_text())
    _strip_legacy_fields(raw)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(raw, handle)
        tmp_path = handle.name
    load_config.cache_clear()
    return load_config(tmp_path)


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")
    return state


def _safe_float(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _pearsonr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((x @ y / denom).item())


def _binned_mi(values: torch.Tensor, labels: torch.Tensor, n_bins: int = 8) -> float:
    """Small discrete MI estimate I(bin(values); labels), in bits."""
    values = values.reshape(-1).float()
    labels = labels.reshape(-1).long()
    if values.numel() == 0:
        return float("nan")
    quantiles = torch.linspace(0, 1, n_bins + 1, dtype=values.dtype)
    edges = torch.quantile(values, quantiles)
    edges[0] = -torch.inf
    edges[-1] = torch.inf
    bins = torch.bucketize(values, edges[1:-1])
    n_label = int(labels.max().item()) + 1
    joint = torch.zeros(n_bins, n_label, dtype=torch.float64)
    for b, c in zip(bins.tolist(), labels.tolist()):
        joint[b, c] += 1.0
    joint = joint / joint.sum().clamp_min(1.0)
    px = joint.sum(dim=1, keepdim=True)
    py = joint.sum(dim=0, keepdim=True)
    expected = px @ py
    mask = joint > 0
    return float((joint[mask] * torch.log2(joint[mask] / expected[mask])).sum().item())


@dataclass
class RoutedPathwaySpec:
    n_pathways: int
    pathway_dim: int
    shared_dim: int

    @property
    def path0_all(self) -> list[int]:
        return list(range(self.pathway_dim))

    @property
    def path1_all(self) -> list[int]:
        return list(range(self.pathway_dim, 2 * self.pathway_dim))

    @property
    def path0_signal(self) -> list[int]:
        return list(range(self.shared_dim, self.pathway_dim))

    @property
    def path1_signal(self) -> list[int]:
        start = self.pathway_dim + self.shared_dim
        return list(range(start, 2 * self.pathway_dim))


def _pathway_spec(config) -> RoutedPathwaySpec:
    params = config.model.encoder.params
    n_pathways = int(getattr(params, "n_pathways", 2) or 2)
    pathway_dim = int(getattr(params, "pathway_dim", 0) or 0)
    shared_indices = list(getattr(params, "shared_indices", []) or [])
    if n_pathways != 2 or pathway_dim <= 0:
        raise ValueError(
            "This diagnostic currently expects a two-pathway PathwayRouter encoder."
        )
    return RoutedPathwaySpec(
        n_pathways=n_pathways,
        pathway_dim=pathway_dim,
        shared_dim=len(shared_indices),
    )


def _context_indices(config, input_dim: int) -> list[int]:
    params = config.model.encoder.params
    shared_indices = list(getattr(params, "shared_indices", []) or [])
    if len(shared_indices) == 2:
        return [int(idx) for idx in shared_indices]
    if input_dim >= 2:
        return [0, 1]
    raise ValueError("Cannot infer two-dimensional context cue from the input.")


def _context_selects_single_path(config) -> bool:
    dataset_name = str(config.data.dataset_name)
    return dataset_name in {"cue_integration", "contextual_stream_gain_shift"}


def _first_inhibitory_branch_pair(model) -> tuple[str, DendriticBranchLayer, str, DendriticBranchLayer | None]:
    layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DendriticBranchLayer)
        and getattr(module, "branch_inhibition", None) is not None
    ]
    if not layers:
        raise ValueError("No DendriticBranchLayer with inhibitory synapses found.")
    distal_name, distal = layers[0]

    all_branch_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, DendriticBranchLayer)
    ]
    soma_name = ""
    soma = None
    for idx, (_name, module) in enumerate(all_branch_layers):
        if module is distal and idx + 1 < len(all_branch_layers):
            next_name, next_module = all_branch_layers[idx + 1]
            if getattr(next_module, "branches_to_output", None) is not None:
                soma_name, soma = next_name, next_module
            break
    return distal_name, distal, soma_name, soma


def analyze_run(run_dir: Path, out_dir: Path, max_samples: int) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "final_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "main_network" / "local_learning_best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found under {run_dir}")

    config = _load_config_compat(config_path)
    config.model.core.implementation.compile_forward = False
    config.wandb.use_wandb = False
    spec = _pathway_spec(config)

    _run_save_path, _train_ds, _valid_ds, test_ds, model, _encoder = setup_environment(
        model_config=config.model,
        training_config=config.training,
        data_config=config.data,
        wandb_config=config.wandb,
        outputs_config=config.outputs,
        experiment_config=config.experiment,
        is_main=False,
    )
    state = _load_state_dict(checkpoint_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            f"Loaded with missing={len(missing)} unexpected={len(unexpected)} "
            f"for {run_dir}"
        )
    model.eval()

    distal_name, distal_layer, soma_name, soma_layer = _first_inhibitory_branch_pair(model)
    for _name, module in model.named_modules():
        if isinstance(module, DendriticBranchLayer):
            module._store_diagnostics = True

    captures: dict[str, list[torch.Tensor]] = {
        "raw_x": [],
        "x_inh": [],
        "inh_conductance": [],
        "inh_activation": [],
        "distal_r": [],
        "soma_r": [],
    }

    def distal_hook(module, inputs, _output):
        x_inh = inputs[1].detach()
        w_inh = module.branch_inhibition.pruned_weight().detach()
        inh_cond = module.branch_inhibition(x_inh).detach()
        inh_act = x_inh[:, :, None] * w_inh.t()[None, :, :]
        g_tot = module._diag_g_tot.detach()
        captures["x_inh"].append(x_inh.cpu())
        captures["inh_conductance"].append(inh_cond.cpu())
        captures["inh_activation"].append(inh_act.cpu())
        captures["distal_r"].append((1.0 / g_tot.clamp_min(1e-8)).cpu())

    def soma_hook(module, _inputs, _output):
        g_tot = getattr(module, "_diag_g_tot", None)
        if isinstance(g_tot, torch.Tensor):
            captures["soma_r"].append((1.0 / g_tot.detach().clamp_min(1e-8)).cpu())

    handles = [distal_layer.register_forward_hook(distal_hook)]
    if soma_layer is not None:
        handles.append(soma_layer.register_forward_hook(soma_hook))

    try:
        loader = DataLoader(test_ds, batch_size=256, shuffle=False)
        with torch.no_grad():
            seen = 0
            for batch in loader:
                x = batch[0]
                captures["raw_x"].append(x.detach().cpu())
                _ = model(x)
                seen += x.size(0)
                if seen >= max_samples:
                    break
    finally:
        for handle in handles:
            handle.remove()
        for _name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                module._store_diagnostics = False

    raw_x = torch.cat(captures["raw_x"], dim=0)[:max_samples]
    inh_conductance = torch.cat(captures["inh_conductance"], dim=0)[:max_samples]
    inh_activation = torch.cat(captures["inh_activation"], dim=0)[:max_samples]
    distal_r = torch.cat(captures["distal_r"], dim=0)[:max_samples]
    soma_r = (
        torch.cat(captures["soma_r"], dim=0)[:max_samples]
        if captures["soma_r"]
        else None
    )

    context_indices = _context_indices(config, input_dim=raw_x.size(1))
    contexts = raw_x[:, context_indices].argmax(dim=1)
    context_selects_single_path = _context_selects_single_path(config)
    branch_factor = 2
    if inh_conductance.size(1) % branch_factor != 0:
        raise ValueError(
            f"Expected first dendritic layer width divisible by 2, got "
            f"{inh_conductance.size(1)}"
        )
    n_soma = inh_conductance.size(1) // branch_factor
    inh_by_branch = inh_conductance.reshape(-1, n_soma, branch_factor)
    r_by_branch = distal_r.reshape(-1, n_soma, branch_factor)
    act_by_branch = inh_activation.reshape(
        inh_activation.size(0), inh_activation.size(1), n_soma, branch_factor
    )

    path0_total = act_by_branch[:, spec.path0_all].sum(dim=1).mean(dim=(1, 2))
    path1_total = act_by_branch[:, spec.path1_all].sum(dim=1).mean(dim=(1, 2))
    path0_signal = act_by_branch[:, spec.path0_signal].sum(dim=1).mean(dim=(1, 2))
    path1_signal = act_by_branch[:, spec.path1_signal].sum(dim=1).mean(dim=(1, 2))

    if context_selects_single_path:
        relevant = torch.where(contexts == 0, path0_total, path1_total)
        irrelevant = torch.where(contexts == 0, path1_total, path0_total)
        irrelevant_over_relevant = _safe_float(
            irrelevant.mean() / relevant.mean().clamp_min(1e-12)
        )
        irrelevant_inhibition_mean = _safe_float(irrelevant.mean())
        relevant_inhibition_mean = _safe_float(relevant.mean())
    else:
        irrelevant_over_relevant = float("nan")
        irrelevant_inhibition_mean = float("nan")
        relevant_inhibition_mean = float("nan")
    context_pred = (path0_total > path1_total).long()
    context_from_inhibition_acc = (context_pred == contexts).float().mean()
    context_from_inhibition_best = torch.maximum(
        context_from_inhibition_acc, 1.0 - context_from_inhibition_acc
    )
    balance = (path0_total - path1_total) / (path0_total + path1_total + 1e-12)

    rows: list[dict[str, Any]] = []
    group_dims = {
        "path0_all": spec.path0_all,
        "path1_all": spec.path1_all,
        "path0_signal": spec.path0_signal,
        "path1_signal": spec.path1_signal,
    }
    for context in (0, 1):
        mask = contexts == context
        for group_name, dims in group_dims.items():
            vals = act_by_branch[mask][:, dims].sum(dim=1).mean(dim=1)
            means = vals.mean(dim=0)
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "context": context,
                    "quantity": "inhibitory_activation",
                    "group": group_name,
                    "branch0": _safe_float(means[0]),
                    "branch1": _safe_float(means[1]),
                    "branch0_minus_branch1": _safe_float(means[0] - means[1]),
                }
            )

        r_means = r_by_branch[mask].mean(dim=(0, 1))
        rows.append(
            {
                "run_dir": str(run_dir),
                "context": context,
                "quantity": "local_input_resistance",
                "group": "all_inputs",
                "branch0": _safe_float(r_means[0]),
                "branch1": _safe_float(r_means[1]),
                "branch0_minus_branch1": _safe_float(r_means[0] - r_means[1]),
            }
        )

    gain_proxy_available = False
    if soma_layer is not None and soma_r is not None:
        edge = soma_layer.branches_to_output.weight().detach().cpu().reshape(
            n_soma, branch_factor
        )
        gain_proxy = r_by_branch * edge.unsqueeze(0)
        if soma_r.ndim == 2 and soma_r.size(1) == n_soma:
            gain_proxy = gain_proxy * soma_r.unsqueeze(-1)
        gain_proxy_available = True
        for context in (0, 1):
            mask = contexts == context
            means = gain_proxy[mask].mean(dim=(0, 1))
            rows.append(
                {
                    "run_dir": str(run_dir),
                    "context": context,
                    "quantity": "path_gain_proxy",
                    "group": "distal_to_soma",
                    "branch0": _safe_float(means[0]),
                    "branch1": _safe_float(means[1]),
                    "branch0_minus_branch1": _safe_float(means[0] - means[1]),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "inhibitory_path_gain_summary.csv"
    header = [
        "run_dir",
        "context",
        "quantity",
        "group",
        "branch0",
        "branch1",
        "branch0_minus_branch1",
    ]
    with csv_path.open("w") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in header) + "\n")

    diagnostics = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "distal_layer": distal_name,
        "soma_layer": soma_name,
        "n_samples": int(raw_x.size(0)),
        "dataset": str(config.data.dataset_name),
        "context_indices": context_indices,
        "context_selects_single_path": context_selects_single_path,
        "context_counts": torch.bincount(contexts, minlength=2).tolist(),
        "path0_total_mean": _safe_float(path0_total.mean()),
        "path1_total_mean": _safe_float(path1_total.mean()),
        "path0_signal_mean": _safe_float(path0_signal.mean()),
        "path1_signal_mean": _safe_float(path1_signal.mean()),
        "irrelevant_inhibition_mean": irrelevant_inhibition_mean,
        "relevant_inhibition_mean": relevant_inhibition_mean,
        "irrelevant_over_relevant": irrelevant_over_relevant,
        "context_from_inhibition_accuracy": _safe_float(context_from_inhibition_acc),
        "context_from_inhibition_best_polarity_accuracy": _safe_float(
            context_from_inhibition_best
        ),
        "context_balance_mi_bits": _binned_mi(balance, contexts),
        "inhibition_vs_local_resistance_r": _pearsonr(inh_by_branch, r_by_branch),
        "gain_proxy_available": gain_proxy_available,
    }
    json_path = out_dir / "inhibitory_path_gain_diagnostics.json"
    json_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))

    md_path = out_dir / "inhibitory_path_gain_report.md"
    md_path.write_text(
        "\n".join(
            [
                "# Inhibitory Path-Gain Diagnostic",
                "",
                f"- Run: `{run_dir}`",
                f"- Distal layer: `{distal_name}`",
                f"- Samples: {diagnostics['n_samples']}",
                f"- Context counts: {diagnostics['context_counts']}",
                f"- Context selects a single relevant path: "
                f"{diagnostics['context_selects_single_path']}",
                f"- Irrelevant/relevant inhibitory activation: "
                f"{diagnostics['irrelevant_over_relevant']:.3f}"
                if diagnostics["context_selects_single_path"]
                else "- Irrelevant/relevant inhibitory activation: n/a "
                "(both routed streams are task-relevant)",
                f"- Context decoded from pathway-inhibition balance: "
                f"{diagnostics['context_from_inhibition_accuracy']:.3f}",
                f"- Context decoded from pathway-inhibition balance, best polarity: "
                f"{diagnostics['context_from_inhibition_best_polarity_accuracy']:.3f}",
                f"- MI(context; inhibition balance), quantile-binned: "
                f"{diagnostics['context_balance_mi_bits']:.3f} bits",
                f"- Corr(inhibitory conductance, local input resistance): "
                f"{diagnostics['inhibition_vs_local_resistance_r']:.3f}",
                "",
                "Detailed context and branch summaries are in "
                "`inhibitory_path_gain_summary.csv`.",
                "",
            ]
        )
    )
    print(md_path.read_text())
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, action="append")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("analysis") / f"inhibitory_path_gain_{ANALYSIS_DATE}",
    )
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    if not math.isfinite(args.max_samples) or args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    diagnostics = []
    if len(args.run_dir) == 1:
        diagnostics.append(analyze_run(args.run_dir[0], args.out_dir, args.max_samples))
    else:
        for run_dir in args.run_dir:
            diagnostics.append(
                analyze_run(run_dir, args.out_dir / run_dir.name, args.max_samples)
            )

        numeric_keys = [
            "irrelevant_over_relevant",
            "context_from_inhibition_accuracy",
            "context_from_inhibition_best_polarity_accuracy",
            "context_balance_mi_bits",
            "inhibition_vs_local_resistance_r",
        ]
        grouped: dict[str, Any] = {"n_runs": len(diagnostics)}
        for key in numeric_keys:
            vals = torch.tensor(
                [float(item[key]) for item in diagnostics], dtype=torch.float64
            )
            grouped[f"{key}_mean"] = float(vals.mean().item())
            grouped[f"{key}_std"] = float(vals.std(unbiased=False).item())
        args.out_dir.mkdir(parents=True, exist_ok=True)
        grouped_path = args.out_dir / "inhibitory_path_gain_grouped_diagnostics.json"
        grouped_path.write_text(json.dumps(grouped, indent=2, sort_keys=True))
        print(json.dumps(grouped, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
