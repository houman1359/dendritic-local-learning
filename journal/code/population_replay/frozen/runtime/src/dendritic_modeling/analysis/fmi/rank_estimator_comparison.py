"""Compare rank_995 covariance estimators across sample sizes with bootstraps.

The FMI profiler's output-spectrum effective rank (``output_spectrum_rank_995``)
collapsed to 1-2 on OLMo-2-32B layers 28-38 and quadrupling calibration samples
did not fix it, implicating the ordinary covariance estimator under
heavy-tailed activations.  The prescribed remedy (external review, 2026-08)
is: "Compare ordinary, winsorized, Huber, and median-of-means estimators
across sample sizes, then freeze the winning estimator prospectively."

This harness runs all four estimators of
:func:`dendritic_modeling.analysis.fmi.spectrum.robust_task_weighted_spectrum`
over a saved captured-boundary tensor (or a synthetic heavy-tailed matrix with
planted rank), bootstrap-resampling rows at each requested sample size, and
writes a JSON of the resulting rank_995 distributions per estimator per
sample size.  Bootstrap index sets are shared across estimators at each
(size, draw), so the comparison is paired.

Examples:
    # Self-contained synthetic check (no cluster data needed):
    python -m dendritic_modeling.analysis.fmi.rank_estimator_comparison \
        --synthetic --output /tmp/rank_comparison.json

    # Real captured boundary activations (a saved [M, d] tensor or a dict
    # holding one under --tensor-key):
    python -m dendritic_modeling.analysis.fmi.rank_estimator_comparison \
        --captured /path/to/layer33_outputs.pt --tensor-key outputs \
        --sample-sizes 256 1024 4096 --bootstrap 200 \
        --output /path/to/layer33_rank_comparison.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from dendritic_modeling.analysis.fmi.spectrum import (
    SPECTRUM_ESTIMATORS,
    n_min,
    robust_task_weighted_spectrum,
)

__all__ = [
    "build_parser",
    "compare_rank_estimators",
    "main",
    "synthetic_heavy_tailed_matrix",
]


def synthetic_heavy_tailed_matrix(
    *,
    rows: int = 8192,
    dim: int = 64,
    rank: int = 10,
    signal_scale: float = 10.0,
    noise_scale: float = 0.02,
    student_df: float = 1.5,
    corrupt_fraction: float = 0.002,
    corrupt_scale: float = 10000.0,
    seed: int = 0,
) -> tuple[torch.Tensor, int]:
    """Low-rank Gaussian signal under the standard gross-error tail model.

    Mirrors the failure mode seen on deep OLMo-2-32B boundaries (rare
    enormous activation rows, documented "high tail risk" flags): a genuine
    ``rank``-dimensional signal plus (i) multivariate Student-t bulk noise —
    one chi-square scale per ROW, infinite variance for ``student_df < 2`` —
    and (ii) a small ``corrupt_fraction`` of rows (at least two, kept below
    the winsorization clip budget) hit by Cauchy-amplitude spikes along
    random directions.  The spikes dominate the ordinary empirical
    covariance, collapsing its rank_995 toward the spike count, while robust
    estimators clip/downweight them and recover the planted rank.  Returns
    the matrix and the planted signal rank.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    latent = torch.randn(rows, rank, generator=generator)
    mixing = torch.randn(rank, dim, generator=generator)
    # Orthonormalize mixing rows so the planted spectrum is well conditioned.
    mixing = torch.linalg.qr(mixing.T)[0].T * float(signal_scale)
    normals = torch.randn(rows, dim, generator=generator)
    # Multivariate Student-t via its per-row chi-square scale mixture.
    # torch.distributions ignores explicit generators, so the global RNG is
    # seeded for the chi-square draw and restored afterward — the
    # construction stays fully deterministic under a fixed seed.
    chi2 = torch.distributions.Chi2(torch.tensor(float(student_df)))
    rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(int(seed) + 1)
        gamma_draws = chi2.sample((rows,))
    finally:
        torch.random.set_rng_state(rng_state)
    row_scales = (gamma_draws / float(student_df)).sqrt().clamp_min(1e-12)
    matrix = latent @ mixing + float(noise_scale) * normals / row_scales.unsqueeze(1)
    corrupt_count = max(2, round(float(corrupt_fraction) * rows))
    corrupt_rows = torch.randperm(rows, generator=generator)[:corrupt_count]
    directions = torch.nn.functional.normalize(
        torch.randn(corrupt_count, dim, generator=generator), dim=1
    )
    uniforms = torch.rand(corrupt_count, generator=generator).clamp(1e-6, 1 - 1e-6)
    cauchy_amplitudes = torch.tan(torch.pi * (uniforms - 0.5)).abs() + 1.0
    matrix[corrupt_rows] += (
        float(corrupt_scale) * cauchy_amplitudes.unsqueeze(1) * directions
    )
    return matrix, int(rank)


def _summary(ranks: list[int]) -> dict[str, Any]:
    ordered = sorted(ranks)
    return {
        "ranks": ranks,
        "median": statistics.median(ordered),
        "iqr": [
            ordered[int(0.25 * (len(ordered) - 1))],
            ordered[int(0.75 * (len(ordered) - 1))],
        ],
        "min": ordered[0],
        "max": ordered[-1],
    }


def compare_rank_estimators(
    matrix: torch.Tensor,
    *,
    sample_sizes: list[int],
    bootstrap_draws: int = 200,
    epsilon: float = 0.005,
    winsorize_quantile: float = 0.995,
    huber_c: float = 1.345,
    huber_max_iters: int = 5,
    mom_blocks: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap rank_995 distributions for all estimators and sample sizes."""
    if matrix.ndim != 2 or matrix.shape[0] < 4:
        raise ValueError("matrix must be [M, d] with at least four rows")
    pool_rows = int(matrix.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    def rank_for(rows: torch.Tensor, estimator: str, draw_seed: int) -> int:
        eigenvalues, _ = robust_task_weighted_spectrum(
            rows,
            estimator=estimator,
            winsorize_quantile=winsorize_quantile,
            huber_c=huber_c,
            huber_max_iters=huber_max_iters,
            mom_blocks=mom_blocks,
            seed=draw_seed,
        )
        return n_min(eigenvalues, epsilon)

    results: dict[str, Any] = {
        estimator: {
            "full_pool_rank_995": rank_for(matrix, estimator, int(seed)),
            "by_sample_size": {},
        }
        for estimator in SPECTRUM_ESTIMATORS
    }
    skipped: list[int] = []
    for size in sample_sizes:
        if size > pool_rows:
            skipped.append(int(size))
            continue
        # One shared index set per draw keeps the comparison paired.
        draws = [
            torch.randint(0, pool_rows, (int(size),), generator=generator)
            for _ in range(int(bootstrap_draws))
        ]
        for estimator in SPECTRUM_ESTIMATORS:
            ranks = [
                rank_for(matrix[indices.to(matrix.device)], estimator, seed + draw)
                for draw, indices in enumerate(draws)
            ]
            results[estimator]["by_sample_size"][str(int(size))] = _summary(ranks)
    return {
        "schema": "dendritic_fmi_rank_estimator_comparison/v1",
        "pool_rows": pool_rows,
        "dim": int(matrix.shape[1]),
        "epsilon": float(epsilon),
        "bootstrap_draws": int(bootstrap_draws),
        "sample_sizes": [int(size) for size in sample_sizes],
        "skipped_sample_sizes_exceeding_pool": skipped,
        "seed": int(seed),
        "estimator_parameters": {
            "winsorize_quantile": float(winsorize_quantile),
            "huber_c": float(huber_c),
            "huber_max_iters": int(huber_max_iters),
            "median_of_means_blocks": int(mom_blocks),
        },
        "results": results,
    }


def _load_captured(path: Path, tensor_key: str | None) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        key = tensor_key or "outputs"
        if key not in payload:
            raise KeyError(
                f"tensor key {key!r} not in artifact keys {sorted(payload)}; "
                "pass --tensor-key"
            )
        payload = payload[key]
    if not torch.is_tensor(payload):
        raise TypeError("captured artifact must be a tensor or a dict holding one")
    return payload.reshape(-1, payload.shape[-1]).float()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--captured",
        type=Path,
        help="saved captured-boundary tensor (.pt): a [M, d] tensor or a dict",
    )
    source.add_argument(
        "--synthetic",
        action="store_true",
        help="planted rank-10 signal + Student-t(1.5) noise; runs without data",
    )
    parser.add_argument(
        "--tensor-key",
        default=None,
        help="dict key holding the [M, d] tensor in --captured (default: outputs)",
    )
    parser.add_argument(
        "--sample-sizes", type=int, nargs="+", default=[256, 1024, 4096]
    )
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.005)
    parser.add_argument("--winsorize-quantile", type=float, default=0.995)
    parser.add_argument("--huber-c", type=float, default=1.345)
    parser.add_argument("--huber-max-iters", type=int, default=5)
    parser.add_argument("--mom-blocks", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--synthetic-rows", type=int, default=8192)
    parser.add_argument("--synthetic-dim", type=int, default=64)
    parser.add_argument("--synthetic-rank", type=int, default=10)
    parser.add_argument("--synthetic-noise-scale", type=float, default=0.02)
    parser.add_argument("--synthetic-df", type=float, default=1.5)
    parser.add_argument("--synthetic-corrupt-fraction", type=float, default=0.002)
    parser.add_argument("--synthetic-corrupt-scale", type=float, default=10000.0)
    parser.add_argument("--output", type=Path, required=True, help="JSON output path")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.synthetic:
        matrix, planted_rank = synthetic_heavy_tailed_matrix(
            rows=args.synthetic_rows,
            dim=args.synthetic_dim,
            rank=args.synthetic_rank,
            noise_scale=args.synthetic_noise_scale,
            student_df=args.synthetic_df,
            corrupt_fraction=args.synthetic_corrupt_fraction,
            corrupt_scale=args.synthetic_corrupt_scale,
            seed=args.seed,
        )
        source: dict[str, Any] = {
            "mode": "synthetic",
            "planted_rank": planted_rank,
            "rows": args.synthetic_rows,
            "dim": args.synthetic_dim,
            "student_df": args.synthetic_df,
            "noise_scale": args.synthetic_noise_scale,
            "corrupt_fraction": args.synthetic_corrupt_fraction,
            "corrupt_scale": args.synthetic_corrupt_scale,
        }
    else:
        matrix = _load_captured(args.captured, args.tensor_key)
        source = {
            "mode": "captured",
            "path": str(args.captured),
            "tensor_key": args.tensor_key or "outputs",
        }
    report = compare_rank_estimators(
        matrix.to(device=args.device),
        sample_sizes=args.sample_sizes,
        bootstrap_draws=args.bootstrap,
        epsilon=args.epsilon,
        winsorize_quantile=args.winsorize_quantile,
        huber_c=args.huber_c,
        huber_max_iters=args.huber_max_iters,
        mom_blocks=args.mom_blocks,
        seed=args.seed,
    )
    report["source"] = source
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for estimator, entry in report["results"].items():
        line = ", ".join(
            f"n={size}: median {summary['median']} (iqr {summary['iqr']})"
            for size, summary in entry["by_sample_size"].items()
        )
        print(f"{estimator:>16} | full-pool {entry['full_pool_rank_995']:>3} | {line}")
    print(f"wrote {args.output}")
    return report


if __name__ == "__main__":
    main()
