#!/usr/bin/env python3
"""Measure low-rank structure of exact compartment-error fields.

This is a checkpoint-only diagnostic. For each completed run, it loads one
batch, backpropagates the task loss, collects exact compartment errors
``dL/dV_n`` from all dendritic branch layers, and computes SVD-based
compressibility metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (  # noqa: E402
    TopKLinear,
)
from dendritic_modeling.scripts.script_utils.setup_utils import (  # noqa: E402
    initialize_model,
)
from measure_theory_diagnostics import (  # noqa: E402
    _capture_decoder_input,
    _compute_loss,
    _discover_run_dirs,
    _get_batch,
    _get_value,
    _hook_branch_layers,
    _load_config_from_run,
    _locate_model_path,
    _make_helper,
)


def _residual_to_exact(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return ||exact - approx|| / ||exact|| for matched matrices."""
    exact = exact.detach().float()
    approx = approx.detach().float().to(device=exact.device, dtype=exact.dtype)
    denom = exact.norm().clamp_min(1e-30)
    return float(((exact - approx).norm() / denom).item())


def _cosine_to_exact(approx: torch.Tensor, exact: torch.Tensor) -> float:
    """Return flattened cosine between an approximation and exact matrix."""
    flat_approx = approx.detach().float().reshape(-1)
    flat_exact = exact.detach().float().reshape(-1)
    if flat_approx.norm() <= 0 or flat_exact.norm() <= 0:
        return float("nan")
    return float(
        torch.nn.functional.cosine_similarity(flat_approx, flat_exact, dim=0).item()
    )


def _shared_field_metrics(matrix: torch.Tensor) -> dict[str, float]:
    """Metrics for the best per-example scalar all-ones approximation.

    This is the constrained object closest to a global scalar broadcast: each row
    may choose its best scalar, but the spatial template is fixed to all ones.
    """
    matrix = matrix.detach().float()
    finite = torch.isfinite(matrix).all(dim=1)
    matrix = matrix[finite]
    if matrix.numel() == 0 or matrix.dim() != 2:
        return {
            "shared_field_residual": float("nan"),
            "shared_field_cosine": float("nan"),
            "centered_shared_field_residual": float("nan"),
            "centered_shared_field_cosine": float("nan"),
        }

    shared = matrix.mean(dim=1, keepdim=True).expand_as(matrix)
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    centered_shared = centered.mean(dim=1, keepdim=True).expand_as(centered)
    return {
        "shared_field_residual": _residual_to_exact(shared, matrix),
        "shared_field_cosine": _cosine_to_exact(shared, matrix),
        "centered_shared_field_residual": _residual_to_exact(
            centered_shared,
            centered,
        ),
        "centered_shared_field_cosine": _cosine_to_exact(centered_shared, centered),
    }


def _per_soma_shared_field_metrics(
    matrix: torch.Tensor,
    *,
    n_soma: int | None,
    segment_widths: list[int] | None = None,
) -> dict[str, float]:
    """Metrics for the best field shared within each soma's compartments.

    The practical ``per_soma`` mode supplies one coefficient per example and
    soma, repeated over that soma's dendritic compartments. This projection
    fixes exactly that blockwise spatial form and chooses its least-squares
    coefficient independently for every example and soma.
    """
    matrix = matrix.detach().float()
    finite = torch.isfinite(matrix).all(dim=1)
    matrix = matrix[finite]
    if (
        matrix.numel() == 0
        or matrix.dim() != 2
        or n_soma is None
        or n_soma <= 0
    ):
        return {
            "per_soma_shared_residual": float("nan"),
            "per_soma_shared_cosine": float("nan"),
        }

    grouped = _group_segments_by_soma(
        matrix,
        n_soma=n_soma,
        segment_widths=segment_widths,
    )
    if grouped is None:
        return {
            "per_soma_shared_residual": float("nan"),
            "per_soma_shared_cosine": float("nan"),
        }
    blocks, widths = grouped
    projected_blocks = blocks.mean(dim=2, keepdim=True).expand_as(blocks)
    projected = _ungroup_soma_segments(
        projected_blocks,
        segment_compartments_per_soma=widths,
    )
    return {
        "per_soma_shared_residual": _residual_to_exact(projected, matrix),
        "per_soma_shared_cosine": _cosine_to_exact(projected, matrix),
    }


def _group_segments_by_soma(
    matrix: torch.Tensor,
    *,
    n_soma: int,
    segment_widths: list[int] | None,
) -> tuple[torch.Tensor, list[int]] | None:
    """Regroup layer-major flattened fields into [sample, soma, compartment]."""
    widths = segment_widths or [matrix.size(1)]
    if sum(widths) != matrix.size(1) or any(
        width <= 0 or width % n_soma != 0 for width in widths
    ):
        return None
    start = 0
    grouped_segments = []
    compartments_per_soma = []
    for width in widths:
        segment = matrix[:, start : start + width]
        per_soma = width // n_soma
        grouped_segments.append(segment.reshape(matrix.size(0), n_soma, per_soma))
        compartments_per_soma.append(per_soma)
        start += width
    return torch.cat(grouped_segments, dim=2), compartments_per_soma


def _ungroup_soma_segments(
    grouped: torch.Tensor,
    *,
    segment_compartments_per_soma: list[int],
) -> torch.Tensor:
    """Restore layer-major flattening after a soma-grouped projection."""
    segments = []
    start = 0
    for per_soma in segment_compartments_per_soma:
        segment = grouped[:, :, start : start + per_soma]
        segments.append(segment.reshape(grouped.size(0), -1))
        start += per_soma
    return torch.cat(segments, dim=1)


def _template_projection_metrics(
    matrix: torch.Tensor,
    *,
    template: torch.Tensor | None,
    n_soma: int | None,
    segment_widths: list[int] | None,
) -> dict[str, float]:
    """Constrained residuals for scalar and per-soma template coefficients.

    ``template`` is the local voltage-conversion pattern available to the rule
    (the post-voltage activation derivative; all ones under identity transfer).
    The scalar projection permits one coefficient per example. The per-soma
    projection permits one coefficient per example and soma, shared across all
    of that soma's dendritic layers and compartments.
    """
    keys = {
        "scalar_template_residual": float("nan"),
        "scalar_template_cosine": float("nan"),
        "per_soma_template_residual": float("nan"),
        "per_soma_template_cosine": float("nan"),
    }
    if template is None or template.shape != matrix.shape:
        return keys
    exact = matrix.detach().float()
    basis = template.detach().float().to(device=exact.device, dtype=exact.dtype)

    scalar_den = basis.square().sum(dim=1, keepdim=True).clamp_min(1e-30)
    scalar_coeff = (exact * basis).sum(dim=1, keepdim=True) / scalar_den
    scalar_projection = scalar_coeff * basis
    keys["scalar_template_residual"] = _residual_to_exact(scalar_projection, exact)
    keys["scalar_template_cosine"] = _cosine_to_exact(scalar_projection, exact)

    if n_soma is None or n_soma <= 0:
        return keys
    grouped_exact = _group_segments_by_soma(
        exact,
        n_soma=n_soma,
        segment_widths=segment_widths,
    )
    grouped_basis = _group_segments_by_soma(
        basis,
        n_soma=n_soma,
        segment_widths=segment_widths,
    )
    if grouped_exact is None or grouped_basis is None:
        return keys
    exact_blocks, widths = grouped_exact
    basis_blocks, _ = grouped_basis
    block_den = basis_blocks.square().sum(dim=2, keepdim=True).clamp_min(1e-30)
    block_coeff = (exact_blocks * basis_blocks).sum(dim=2, keepdim=True) / block_den
    projected_blocks = block_coeff * basis_blocks
    per_soma_projection = _ungroup_soma_segments(
        projected_blocks,
        segment_compartments_per_soma=widths,
    )
    keys["per_soma_template_residual"] = _residual_to_exact(
        per_soma_projection,
        exact,
    )
    keys["per_soma_template_cosine"] = _cosine_to_exact(
        per_soma_projection,
        exact,
    )
    return keys


def _globally_rescaled_metrics(
    approx: torch.Tensor,
    exact: torch.Tensor,
) -> dict[str, float]:
    """Residual and scale after the best single scalar rescaling of a field."""
    approx = approx.detach().float().to(device=exact.device, dtype=exact.dtype)
    exact = exact.detach().float()
    denom = approx.square().sum()
    if denom <= 0 or not torch.isfinite(denom):
        return {
            "rescaled_residual": float("nan"),
            "optimal_scale": float("nan"),
        }
    scale = (approx * exact).sum() / denom
    return {
        "rescaled_residual": _residual_to_exact(scale * approx, exact),
        "optimal_scale": float(scale.item()),
    }


def _expand_soma_error_to_matrix(
    delta: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor | None:
    """Expand soma-coordinate errors to a layer matrix when dimensions allow it."""
    if delta.dim() == 1:
        delta = delta.unsqueeze(-1)
    elif delta.dim() != 2:
        delta = delta.reshape(delta.size(0), -1)
    if delta.size(0) != matrix.size(0):
        return None
    if delta.size(1) == matrix.size(1):
        return delta.to(dtype=matrix.dtype, device=matrix.device)
    if matrix.size(1) % delta.size(1) == 0:
        repeats = matrix.size(1) // delta.size(1)
        return delta.repeat_interleave(repeats, dim=1).to(
            dtype=matrix.dtype,
            device=matrix.device,
        )
    return None


def _legacy_per_soma_error_to_matrix(
    delta: torch.Tensor,
    delta_scalar: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the submitted ``per_soma`` implementation exactly.

    The legacy mode preserves the soma vector only when its width exactly
    matches the current compartment layer. Wider dendritic layers fall back to
    the global scalar. This differs from an ancestry-preserving per-soma field,
    which repeats each soma coordinate over all descendant compartments.
    """
    if delta.dim() == 1:
        delta = delta.unsqueeze(-1)
    elif delta.dim() != 2:
        delta = delta.reshape(delta.size(0), -1)
    if delta.size(0) == matrix.size(0) and delta.size(1) == matrix.size(1):
        return delta.to(dtype=matrix.dtype, device=matrix.device)
    return delta_scalar.to(dtype=matrix.dtype, device=matrix.device).expand_as(matrix)


def _broadcast_conditioned_metrics(
    matrix: torch.Tensor,
    *,
    scalar_broadcast: torch.Tensor | None,
    per_soma_broadcast: torch.Tensor | None,
    actual_broadcast: torch.Tensor | None,
) -> dict[str, float]:
    """Return residuals for the broadcast fields available to the run."""
    metrics: dict[str, float] = {}
    for name, approx in [
        ("scalar_broadcast", scalar_broadcast),
        ("per_soma_broadcast", per_soma_broadcast),
        ("actual_broadcast", actual_broadcast),
    ]:
        if approx is None:
            metrics[f"{name}_residual"] = float("nan")
            metrics[f"{name}_cosine"] = float("nan")
            metrics[f"{name}_rescaled_residual"] = float("nan")
            metrics[f"{name}_optimal_scale"] = float("nan")
            continue
        metrics[f"{name}_residual"] = _residual_to_exact(approx, matrix)
        metrics[f"{name}_cosine"] = _cosine_to_exact(approx, matrix)
        rescaled = _globally_rescaled_metrics(approx, matrix)
        metrics[f"{name}_rescaled_residual"] = rescaled["rescaled_residual"]
        metrics[f"{name}_optimal_scale"] = rescaled["optimal_scale"]
    return metrics


def _matrix_rank_metrics(matrix: torch.Tensor) -> dict[str, float]:
    """Return SVD compressibility metrics for a samples x compartments matrix."""
    matrix = matrix.detach().float()
    finite = torch.isfinite(matrix).all(dim=1)
    matrix = matrix[finite]
    if matrix.numel() == 0 or matrix.shape[0] < 2:
        return {
            "n_samples": int(matrix.shape[0]),
            "n_compartments": int(matrix.shape[1]) if matrix.dim() == 2 else 0,
            "rank1_residual": float("nan"),
            "rank2_residual": float("nan"),
            "rank4_residual": float("nan"),
            "rank8_residual": float("nan"),
            "effective_rank_entropy": float("nan"),
            "effective_rank_participation": float("nan"),
            "top1_energy_fraction": float("nan"),
            "top2_energy_fraction": float("nan"),
            "top4_energy_fraction": float("nan"),
            "top8_energy_fraction": float("nan"),
            **_shared_field_metrics(matrix),
        }

    singular_values = torch.linalg.svdvals(matrix)
    energy = singular_values.square()
    total = energy.sum().clamp_min(1e-30)
    probs = energy / total

    def residual(k: int) -> float:
        kept = energy[: min(k, energy.numel())].sum()
        return float(torch.sqrt((total - kept).clamp_min(0.0) / total).item())

    def energy_fraction(k: int) -> float:
        return float((energy[: min(k, energy.numel())].sum() / total).item())

    entropy = -(probs * torch.log(probs.clamp_min(1e-30))).sum()
    participation = 1.0 / probs.square().sum().clamp_min(1e-30)
    return {
        "n_samples": int(matrix.shape[0]),
        "n_compartments": int(matrix.shape[1]),
        "rank1_residual": residual(1),
        "rank2_residual": residual(2),
        "rank4_residual": residual(4),
        "rank8_residual": residual(8),
        "effective_rank_entropy": float(torch.exp(entropy).item()),
        "effective_rank_participation": float(participation.item()),
        "top1_energy_fraction": energy_fraction(1),
        "top2_energy_fraction": energy_fraction(2),
        "top4_energy_fraction": energy_fraction(4),
        "top8_energy_fraction": energy_fraction(8),
        **_shared_field_metrics(matrix),
    }


def _collect_exact_error_matrices(
    run_dir: Path,
    batch_size: int,
    split: str,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    dict[str, torch.Tensor],
    dict[str, dict[str, Any]],
]:
    config = _load_config_from_run(run_dir)
    if hasattr(config.model.core, "implementation"):
        config.model.core.implementation.compile_forward = False
    x_batch, y_batch = _get_batch(config, split=split, batch_size=batch_size)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    input_dim = int(x_batch[0].numel())
    encoder_params = getattr(config.model.encoder, "params", None)
    if encoder_params is None:
        config.model.encoder.params = {"input_dim": input_dim}
    elif isinstance(encoder_params, dict):
        encoder_params["input_dim"] = input_dim
    else:
        encoder_params.input_dim = input_dim

    model, _ = initialize_model(config.model)
    state = torch.load(_locate_model_path(run_dir), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    helper = _make_helper()
    decoder_cache: dict[str, Any] = {}
    dec_handle = _capture_decoder_input(model, decoder_cache)
    layer_records, layer_handles = _hook_branch_layers(model)
    topk_modules = [module for module in model.modules() if isinstance(module, TopKLinear)]
    for module in topk_modules:
        module.cache_mask = True

    try:
        y_hat = model(x_batch)
    finally:
        if dec_handle is not None:
            dec_handle.remove()
        for handle in layer_handles:
            handle.remove()
        for module in topk_modules:
            module.cache_mask = False

    common_cfg = _get_value(_get_value(config.training, "main"), "common", {})
    loss_name = _get_value(common_cfg, "loss_function", "cat_nll")
    loss = _compute_loss(loss_name, y_hat, y_batch)
    helper.loss_function = type("Loss", (), {"_loss_name": str(loss_name)})()
    helper._decoder_cache = decoder_cache
    delta_out = helper._compute_soma_error(y_hat.detach(), y_batch.detach())
    _, delta_local = helper._resolve_local_soma_signals(
        model=model,
        y_hat=y_hat,
        delta_out=delta_out.detach(),
    )
    delta_local = delta_local.detach()
    delta_scalar = helper._reduce_error_to_scalar(delta_local).detach()
    model.zero_grad(set_to_none=True)
    loss.backward()

    matrices: dict[str, torch.Tensor] = {}
    broadcasts: dict[str, dict[str, Any]] = {}
    all_layers = []
    all_scalar_broadcasts = []
    all_per_soma_broadcasts = []
    actual_mode = _get_value(
        _get_value(_get_value(config.training, "main"), "learning_strategy_config"),
        "error_broadcast_mode",
    )
    for layer_idx, rec in enumerate(layer_records):
        exact = rec["v_n"].grad.detach().cpu().float()
        matrices[f"layer_{layer_idx}"] = exact.reshape(exact.shape[0], -1)
        layer_matrix = matrices[f"layer_{layer_idx}"]
        activation_derivative = helper._get_layer_activation_derivative(
            rec,
            v_n=rec.get("v_n"),
            v_out=rec.get("v_out"),
        )
        if isinstance(activation_derivative, torch.Tensor):
            activation_derivative = activation_derivative.detach().cpu().float()
        else:
            activation_derivative = torch.ones_like(exact.detach().cpu().float())
        activation_derivative = activation_derivative.reshape(
            activation_derivative.shape[0],
            -1,
        )
        scalar_broadcast = (
            delta_scalar.detach()
            .cpu()
            .float()
            .expand(-1, layer_matrix.size(1))
            * activation_derivative
        )
        per_soma = _expand_soma_error_to_matrix(
            delta_local.detach().cpu().float(),
            layer_matrix,
        )
        per_soma_broadcast = (
            per_soma * activation_derivative if isinstance(per_soma, torch.Tensor) else None
        )
        if str(actual_mode).lower() == "per_soma":
            actual_broadcast = (
                _legacy_per_soma_error_to_matrix(
                    delta_local.detach().cpu().float(),
                    delta_scalar.detach().cpu().float(),
                    layer_matrix,
                )
                * activation_derivative
            )
        elif str(actual_mode).lower() in {"per_soma_shared", "per_soma_tree"}:
            actual_broadcast = per_soma_broadcast
        elif str(actual_mode).lower() in {"scalar", "rank1", "rank_1"}:
            actual_broadcast = scalar_broadcast
        else:
            actual_broadcast = None
        broadcasts[f"layer_{layer_idx}"] = {
            "layer_name": str(rec.get("layer_name", f"layer_{layer_idx}")),
            "scalar_broadcast": scalar_broadcast,
            "per_soma_broadcast": per_soma_broadcast,
            "actual_broadcast": actual_broadcast,
            "template": activation_derivative,
            "n_soma": int(delta_local.size(1)) if delta_local.dim() == 2 else None,
            "segment_widths": [int(layer_matrix.size(1))],
        }
        all_layers.append(matrices[f"layer_{layer_idx}"])
        all_scalar_broadcasts.append(scalar_broadcast)
        if per_soma_broadcast is not None:
            all_per_soma_broadcasts.append(per_soma_broadcast)
    if all_layers:
        matrices["all_layers"] = torch.cat(all_layers, dim=1)
        all_scalar = torch.cat(all_scalar_broadcasts, dim=1)
        all_per_soma = (
            torch.cat(all_per_soma_broadcasts, dim=1)
            if len(all_per_soma_broadcasts) == len(all_layers)
            else None
        )
        if str(actual_mode).lower() == "per_soma":
            all_actual = torch.cat(
                [
                    broadcasts[f"layer_{idx}"]["actual_broadcast"]
                    for idx in range(len(layer_records))
                ],
                dim=1,
            )
        elif str(actual_mode).lower() in {"per_soma_shared", "per_soma_tree"}:
            all_actual = all_per_soma
        elif str(actual_mode).lower() in {"scalar", "rank1", "rank_1"}:
            all_actual = all_scalar
        else:
            all_actual = None
        broadcasts["all_layers"] = {
            "scalar_broadcast": all_scalar,
            "per_soma_broadcast": all_per_soma,
            "actual_broadcast": all_actual,
            "template": torch.cat(
                [
                    broadcasts[f"layer_{idx}"]["template"]
                    for idx in range(len(layer_records))
                ],
                dim=1,
            ),
            "n_soma": int(delta_local.size(1)) if delta_local.dim() == 2 else None,
            "segment_widths": [int(layer.size(1)) for layer in all_layers],
        }

    connectivity = getattr(config.model.core, "connectivity", None)
    ie_values = _get_value(connectivity, "ie_synapses_per_branch_per_layer", [])
    ie_value = ie_values[0] if isinstance(ie_values, list) and ie_values else None
    meta = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "seed": _get_value(config.experiment, "seed"),
        "dataset": config.data.dataset_name,
        "network_type": config.model.core.type,
        "ie_value": ie_value,
        "strategy": _get_value(_get_value(config.training, "main"), "strategy"),
        "rule_variant": _get_value(
            _get_value(_get_value(config.training, "main"), "learning_strategy_config"),
            "rule_variant",
        ),
        "error_broadcast_mode": _get_value(
            _get_value(_get_value(config.training, "main"), "learning_strategy_config"),
            "error_broadcast_mode",
        ),
        "test_accuracy": _get_value(
            _get_value(_get_value(config.training, "main"), "metrics", {}),
            "test_accuracy",
            None,
        ),
    }
    return meta, matrices, broadcasts


def analyze_run(run_dir: Path, batch_size: int, split: str, device: torch.device) -> list[dict[str, Any]]:
    meta, matrices, broadcasts = _collect_exact_error_matrices(
        run_dir,
        batch_size,
        split,
        device,
    )
    rows: list[dict[str, Any]] = []
    for scope, matrix in matrices.items():
        uncentered = _matrix_rank_metrics(matrix)
        centered = _matrix_rank_metrics(matrix - matrix.mean(dim=0, keepdim=True))
        broadcast = broadcasts.get(scope, {})
        broadcast_metrics = _broadcast_conditioned_metrics(
            matrix,
            scalar_broadcast=broadcast.get("scalar_broadcast"),
            per_soma_broadcast=broadcast.get("per_soma_broadcast"),
            actual_broadcast=broadcast.get("actual_broadcast"),
        )
        constrained_metrics = _per_soma_shared_field_metrics(
            matrix,
            n_soma=(
                int(broadcast["n_soma"])
                if isinstance(broadcast.get("n_soma"), int)
                else None
            ),
            segment_widths=broadcast.get("segment_widths"),
        )
        template_metrics = _template_projection_metrics(
            matrix,
            template=broadcast.get("template"),
            n_soma=(
                int(broadcast["n_soma"])
                if isinstance(broadcast.get("n_soma"), int)
                else None
            ),
            segment_widths=broadcast.get("segment_widths"),
        )
        row = {
            **meta,
            "scope": scope,
            **uncentered,
            **constrained_metrics,
            **template_metrics,
            **broadcast_metrics,
        }
        for key, value in centered.items():
            if key.startswith("n_"):
                continue
            if key.startswith("shared_field") or key.startswith("centered_shared_field"):
                continue
            row[f"centered_{key}"] = value
        rows.append(row)
    return rows


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    group_cols = [
        col
        for col in ["dataset", "network_type", "strategy", "rule_variant", "error_broadcast_mode", "scope"]
        if col in rows.columns
    ]
    metric_cols = [
        col
        for col in rows.columns
        if col.endswith("_residual")
        or col.endswith("_cosine")
        or col.startswith(("top", "effective_rank", "centered_"))
    ]
    summary = (
        rows.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-dirs", type=Path, nargs="+")
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    supplied = sum(bool(x) for x in (args.run_dir, args.run_dirs, args.sweep_dir))
    if supplied != 1:
        raise ValueError("Specify exactly one of --run-dir, --run-dirs, or --sweep-dir.")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    if args.run_dirs:
        run_dirs = list(args.run_dirs)
    else:
        root = args.run_dir or args.sweep_dir
        run_dirs = _discover_run_dirs(root)
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]

    all_rows: list[dict[str, Any]] = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        print(f"[{idx}/{len(run_dirs)}] {run_dir}")
        try:
            all_rows.extend(analyze_run(run_dir, args.batch_size, args.split, device))
        except Exception as exc:
            print(f"WARNING: failed {run_dir}: {exc}")
            all_rows.append({"run_dir": str(run_dir), "error": str(exc)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(all_rows)
    rows.to_csv(args.output_dir / "error_rank_diagnostics.csv", index=False)
    summary = _summarize(rows[rows.get("error").isna()] if "error" in rows else rows)
    summary.to_csv(args.output_dir / "error_rank_summary.csv", index=False)
    print(f"Saved diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
