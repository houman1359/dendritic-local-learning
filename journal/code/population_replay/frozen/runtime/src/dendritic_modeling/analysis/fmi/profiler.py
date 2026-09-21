"""Reusable teacher-component profiler for Functional Morphology Inference.

This is the maintained, model-agnostic counterpart of the original frozen
OLMo prospective script.  It accepts any differentiable callable
``teacher_fn: [M,d_in] -> [M,d_out]`` and emits a JSON-serializable fingerprint
covering the architecture axes used by the v2 replacement selector.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from dendritic_modeling.analysis.fmi.gain_load import (
    challenge_score_distribution,
    gain_load_score,
    gain_sensitivity,
    paired_shunting_win_probability,
)
from dendritic_modeling.analysis.fmi.interactions import infer_tree, interaction_matrix
from dendritic_modeling.analysis.fmi.mechanism import (
    delta_shunt,
    mechanism_scores,
    multiplicative_advantage,
)
from dendritic_modeling.analysis.fmi.spectrum import (
    n_min,
    robust_task_weighted_spectrum,
    spectrum_slope_beta,
)
from dendritic_modeling.analysis.fmi.support import (
    attribution_energy,
    gradient_participation_ratio,
    sign_consistency,
    support_jaccard,
    support_size,
)


@dataclass(frozen=True)
class FMIProfilerConfig:
    """Cost/accuracy controls for one teacher-component fingerprint."""

    spectrum_epsilon: float = 0.005
    # Covariance/second-moment estimator behind the output-spectrum rank.
    # "ordinary" is the historical estimator and is bit-identical to the
    # pre-option behavior; "winsorized", "huber", and "median_of_means" are
    # heavy-tail-resistant alternatives (2026-08-25 review prescription for
    # the OLMo-2-32B layer 28-38 rank collapse). See
    # dendritic_modeling.analysis.fmi.spectrum.robust_task_weighted_spectrum.
    spectrum_estimator: str = "ordinary"
    spectrum_winsorize_quantile: float = 0.995
    spectrum_huber_c: float = 1.345
    spectrum_huber_max_iters: int = 5
    spectrum_mom_blocks: int = 16
    support_energy: float = 0.95
    sign_threshold: float = 0.2
    linear_probe_ridge: float = 1.0e-4
    latent_count: int = 4
    mechanism_latent_count: int = 2
    probe_examples: int = 512
    support_pairs: int = 64
    interaction_coordinates: int = 24
    interaction_latent_count: int = 2
    interaction_probe_points: int = 32
    interaction_step: float = 0.5
    interaction_max_depth: int = 2
    interaction_min_gain: float = 0.02
    interaction_eta_tree: float = 0.65
    probe_steps: int = 400
    product_probe_steps: int = 800
    shunting_bootstrap_samples: int = 2000
    seed: int = 0


def profiler_config_from_preset(name: str, *, seed: int = 0) -> FMIProfilerConfig:
    """Return a versioned reference or inexpensive smoke configuration."""

    normalized = str(name).strip().lower()
    if normalized == "reference":
        return FMIProfilerConfig(seed=int(seed))
    if normalized == "smoke":
        return FMIProfilerConfig(
            latent_count=2,
            mechanism_latent_count=1,
            probe_examples=64,
            support_pairs=8,
            interaction_coordinates=8,
            interaction_latent_count=1,
            interaction_probe_points=8,
            probe_steps=10,
            product_probe_steps=20,
            shunting_bootstrap_samples=200,
            seed=int(seed),
        )
    raise ValueError("FMI profiler preset must be 'reference' or 'smoke'")


def _kurtosis(values: torch.Tensor) -> float:
    centered = values.float() - values.float().mean()
    variance = centered.square().mean()
    if float(variance) <= 0.0:
        return 0.0
    return float(centered.pow(4).mean() / variance.square() - 3.0)


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    return float(sum(value * weight for value, weight in zip(values, weights)))


def _sample_tail_summary(values: torch.Tensor) -> dict[str, float]:
    """Summarize across-example tails without depending on absolute scale."""

    rows = values.detach().float().reshape(-1, values.shape[-1])
    row_l2 = rows.norm(dim=-1)
    coordinate_rms = rows.square().mean(dim=0).sqrt().clamp_min(1e-8)
    standardized_row_rms = (rows / coordinate_rms).square().mean(dim=-1).sqrt()

    def ratios(samples: torch.Tensor, prefix: str) -> dict[str, float]:
        median = samples.median().clamp_min(1e-12)
        return {
            f"{prefix}_p95_over_median": float(torch.quantile(samples, 0.95) / median),
            f"{prefix}_p99_over_median": float(torch.quantile(samples, 0.99) / median),
            f"{prefix}_max_over_median": float(samples.max() / median),
        }

    return {
        **ratios(row_l2, "row_l2"),
        **ratios(standardized_row_rms, "standardized_row_rms"),
    }


def _branch_factors_from_levels(levels: list[torch.Tensor]) -> list[int]:
    counts = [int(labels.unique().numel()) for labels in levels]
    factors: list[int] = []
    previous = 1
    for count in counts:
        if count > previous:
            factors.append(max(2, math.ceil(count / previous)))
            previous = count
    return factors


def _support_fraction(
    attribution: torch.Tensor,
    mask: torch.Tensor,
    *,
    energy: float,
) -> float:
    """Return full-input-normalized support within a declared coordinate class."""

    restricted = attribution * mask.to(dtype=attribution.dtype)
    if float(restricted.sum()) <= 0.0:
        return 0.0
    return support_size(restricted, energy) / attribution.numel()


def _linear_bypass_profile(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    support_energy: float,
    ridge: float,
) -> tuple[float, float]:
    """Held-out variance captured by an affine main-effect probe.

    The statistic is intentionally modest: it detects a globally affine
    component that can bypass branch composition, but does not claim to
    recover a full functional-ANOVA decomposition.  The selector therefore
    labels its somatic-bypass rule as prospective.
    """

    split = max(2, int(0.75 * inputs.shape[0]))
    if not math.isfinite(float(ridge)) or float(ridge) <= 0.0:
        raise ValueError("linear probe ridge must be positive and finite")
    train_x = inputs[:split].detach().float()
    test_x = inputs[split:].detach().float()
    train_y = targets[:split].detach().float()
    test_y = targets[split:].detach().float()
    x_mean = train_x.mean(dim=0)
    x_scale = (train_x - x_mean).square().mean(dim=0).sqrt().clamp_min(1.0e-6)
    normalized_train = (train_x - x_mean) / x_scale
    normalized_test = (test_x - x_mean) / x_scale
    y_mean = train_y.mean()
    centered_y = train_y - y_mean

    # The real-teacher setting commonly has fewer calibration rows than input
    # dimensions. A direct underdetermined least-squares solve can return NaN
    # on CUDA for ill-conditioned activation matrices. Solve the equivalent
    # dual ridge system instead; scaling lambda by the mean Gram diagonal makes
    # ``ridge`` dimensionless and comparable across teacher boundaries.
    gram = normalized_train @ normalized_train.T
    gram_scale = torch.diagonal(gram).mean().clamp_min(1.0e-12)
    regularized = gram + float(ridge) * gram_scale * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    dual = torch.linalg.solve(regularized, centered_y.unsqueeze(1)).squeeze(1)
    prediction = y_mean + normalized_test @ normalized_train.T @ dual
    residual = (prediction - test_y).square().mean()
    variance = (test_y - test_y.mean()).square().mean().clamp_min(1e-12)
    explained = float((1.0 - residual / variance).clamp(0.0, 1.0))
    if not math.isfinite(explained):
        raise FloatingPointError("affine bypass probe produced non-finite R-squared")
    if explained <= 0.0:
        return 0.0, 0.0
    coefficients = (normalized_train.T @ dual) / x_scale
    input_rms = train_x.square().mean(dim=0).sqrt()
    coefficient_energy = (coefficients * input_rms).square()
    support_fraction = (
        support_size(coefficient_energy, support_energy) / inputs.shape[1]
        if float(coefficient_energy.sum()) > 0.0
        else 0.0
    )
    return explained, float(support_fraction)


def profile_teacher_component(
    teacher_fn: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    *,
    outputs: torch.Tensor | None = None,
    task_weight: torch.Tensor | None = None,
    config: FMIProfilerConfig | None = None,
) -> dict[str, Any]:
    """Profile a frozen teacher component without training a replacement.

    Inputs and outputs are flattened calibration examples.  The caller owns
    sampling and task conditioning; passing ``task_weight`` supplies the PSD
    metric used by the task-weighted output spectrum.
    """

    cfg = config or FMIProfilerConfig()
    if inputs.ndim != 2 or inputs.shape[0] < 4:
        raise ValueError("inputs must be [M,d_in] with at least four examples")
    if outputs is None:
        with torch.no_grad():
            outputs = teacher_fn(inputs).detach()
    else:
        outputs = outputs.detach()
    if outputs.ndim != 2 or outputs.shape[0] != inputs.shape[0]:
        raise ValueError("outputs must be [M,d_out] and align with inputs")

    # With spectrum_estimator == "ordinary" this delegates verbatim to
    # task_weighted_spectrum and is bit-identical to the historical behavior.
    eigenvalues, eigenvectors, weight_root = robust_task_weighted_spectrum(
        outputs,
        task_weight,
        estimator=cfg.spectrum_estimator,
        winsorize_quantile=cfg.spectrum_winsorize_quantile,
        huber_c=cfg.spectrum_huber_c,
        huber_max_iters=cfg.spectrum_huber_max_iters,
        mom_blocks=cfg.spectrum_mom_blocks,
        seed=cfg.seed,
        return_root=True,
    )
    task_rank = n_min(eigenvalues, cfg.spectrum_epsilon)
    beta = spectrum_slope_beta(eigenvalues, task_rank)
    latent_count = min(
        int(cfg.latent_count),
        int((eigenvalues > 0).sum()),
        outputs.shape[1],
    )
    if latent_count < 1:
        raise ValueError("teacher outputs have no non-constant latent direction")
    latent_weights_tensor = eigenvalues[:latent_count]
    latent_weights_tensor = (
        latent_weights_tensor / latent_weights_tensor.sum().clamp_min(1e-12)
    )
    latent_weights = [float(value) for value in latent_weights_tensor]

    generator = torch.Generator(device="cpu").manual_seed(int(cfg.seed))
    probe_count = min(int(cfg.probe_examples), inputs.shape[0])
    indices = torch.randperm(inputs.shape[0], generator=generator)[:probe_count]
    indices = indices.to(inputs.device)
    probe_inputs = inputs[indices]

    supports: list[float] = []
    active_ranks: list[float] = []
    stability: list[float] = []
    inhibitory: list[float] = []
    sign_stability: list[float] = []
    excitatory_supports: list[float] = []
    inhibitory_supports: list[float] = []
    context_dependent_supports: list[float] = []
    r_mults: list[float] = []
    linear_bypass: list[float] = []
    linear_bypass_supports: list[float] = []
    attribution_vectors: list[torch.Tensor] = []
    latent_functions = []
    for latent_index in range(latent_count):
        direction = eigenvectors[:, latent_index].detach()
        # Eigensolver signs are arbitrary. Canonicalize each latent before any
        # signed-attribution statistic so repeated runs and platforms cannot
        # silently swap the two functional polarity channels.
        pivot = int(direction.abs().argmax().item())
        if float(direction[pivot]) < 0.0:
            direction = -direction

        def z_fn(x: torch.Tensor, direction=direction) -> torch.Tensor:
            y = teacher_fn(x)
            if weight_root is not None:
                y = y @ weight_root
            return y @ direction

        latent_functions.append(z_fn)
        attribution = attribution_energy(z_fn, probe_inputs)
        attribution_vectors.append(attribution)
        k = support_size(attribution, cfg.support_energy)
        supports.append(k / inputs.shape[1])
        active_ranks.append(gradient_participation_ratio(z_fn, probe_inputs))
        stability.append(
            support_jaccard(
                z_fn,
                probe_inputs,
                max(1, k),
                pairs=cfg.support_pairs,
                generator=generator,
            )
        )
        signs = sign_consistency(z_fn, probe_inputs)
        consistent = signs.abs() >= cfg.sign_threshold
        stable_energy = attribution[consistent]
        inhibitory_energy = attribution[consistent & (signs < 0)].sum()
        sign_stability.append(
            float(stable_energy.sum() / attribution.sum().clamp_min(1e-12))
        )
        negative_share = (
            float(inhibitory_energy / stable_energy.sum().clamp_min(1e-12))
            if bool(consistent.any())
            else 0.0
        )
        inhibitory.append(negative_share)
        excitatory_supports.append(
            _support_fraction(
                attribution,
                consistent & (signs > 0),
                energy=cfg.support_energy,
            )
        )
        inhibitory_supports.append(
            _support_fraction(
                attribution,
                consistent & (signs < 0),
                energy=cfg.support_energy,
            )
        )
        context_dependent_supports.append(
            _support_fraction(
                attribution,
                ~consistent,
                energy=cfg.support_energy,
            )
        )
        with torch.no_grad():
            targets = z_fn(probe_inputs)
        bypass_fraction, bypass_support = _linear_bypass_profile(
            probe_inputs,
            targets,
            support_energy=cfg.support_energy,
            ridge=cfg.linear_probe_ridge,
        )
        linear_bypass.append(bypass_fraction)
        linear_bypass_supports.append(bypass_support)
        r_mults.append(
            multiplicative_advantage(
                probe_inputs,
                targets,
                steps=cfg.product_probe_steps,
                seed=cfg.seed + latent_index,
            )
        )

    mechanism_count = min(int(cfg.mechanism_latent_count), latent_count)
    nonnegative = torch.cat([F.relu(probe_inputs), F.relu(-probe_inputs)], dim=1)
    natural_mechanisms: list[dict[str, float]] = []
    challenge_mechanisms: list[dict[str, float]] = []
    shunting_deltas: list[float] = []
    gain_scores: list[float] = []
    gain_sensitivities: list[float] = []
    shunting_probabilities: list[float] = []
    challenge_residuals: list[dict[str, torch.Tensor]] = []
    for latent_index in range(mechanism_count):
        z_fn = latent_functions[latent_index]
        with torch.no_grad():
            natural_targets = z_fn(probe_inputs)
        natural = mechanism_scores(
            nonnegative,
            natural_targets,
            steps=cfg.probe_steps,
            seed=cfg.seed + latent_index,
        )

        def split_target(u: torch.Tensor, z_fn=z_fn) -> torch.Tensor:
            width = u.shape[1] // 2
            return z_fn(u[:, :width] - u[:, width:])

        challenge, residuals = challenge_score_distribution(
            nonnegative,
            split_target,
            steps=cfg.probe_steps,
            seed=cfg.seed + latent_index,
        )
        natural_mechanisms.append(natural)
        challenge_mechanisms.append(challenge)
        shunting_deltas.append(delta_shunt(natural))
        gain_scores.append(gain_load_score(natural, challenge))
        gain_sensitivities.append(gain_sensitivity(z_fn, probe_inputs))
        shunting_probabilities.append(
            paired_shunting_win_probability(
                residuals,
                bootstrap_samples=cfg.shunting_bootstrap_samples,
                seed=cfg.seed + 10_000 + latent_index,
            )
        )
        challenge_residuals.append(residuals)

    mechanism_weights = latent_weights[:mechanism_count]
    mechanism_weight_sum = sum(mechanism_weights)
    mechanism_weights = [weight / mechanism_weight_sum for weight in mechanism_weights]
    aggregate_challenge_residuals = {
        mechanism: sum(
            weight * residuals[mechanism]
            for weight, residuals in zip(mechanism_weights, challenge_residuals)
        )
        for mechanism in challenge_residuals[0]
    }
    shunting_win_probability = paired_shunting_win_probability(
        aggregate_challenge_residuals,
        bootstrap_samples=cfg.shunting_bootstrap_samples,
        seed=cfg.seed + 20_000,
    )

    aggregate_attribution = sum(
        weight * attribution
        for weight, attribution in zip(latent_weights, attribution_vectors)
    )
    interaction_count = min(int(cfg.interaction_coordinates), inputs.shape[1])
    interaction_coords = (
        aggregate_attribution.topk(interaction_count).indices.sort().values.tolist()
    )
    interaction_latent_count = min(cfg.interaction_latent_count, latent_count)
    interactions = None
    has_interaction_signal = False
    for latent_index in range(interaction_latent_count):
        matrix = interaction_matrix(
            latent_functions[latent_index],
            probe_inputs,
            coordinates=interaction_coords,
            probe_points=cfg.interaction_probe_points,
            step=cfg.interaction_step,
        )
        has_interaction_signal = has_interaction_signal or float(matrix.sum()) > 1e-12
        # Normalize each latent before spectral aggregation so a high-amplitude
        # output direction cannot erase a lower-energy but distinct hierarchy.
        matrix = matrix / matrix.sum().clamp_min(1e-12)
        contribution = latent_weights[latent_index] * matrix
        interactions = (
            contribution if interactions is None else interactions + contribution
        )
    assert interactions is not None
    if not has_interaction_signal:
        interaction_depth = 0
        interaction_levels = [torch.zeros(interaction_count, dtype=torch.long)]
    else:
        interaction_depth, interaction_levels = infer_tree(
            interactions,
            max_depth=cfg.interaction_max_depth,
            min_gain=cfg.interaction_min_gain,
            eta_tree=cfg.interaction_eta_tree,
        )

    scales = inputs.detach().float().square().mean(dim=0).sqrt()
    median = float(scales.median().clamp_min(1e-12))
    mean_mechanisms = {
        name: statistics.mean(scores[name] for scores in natural_mechanisms)
        for name in natural_mechanisms[0]
    }
    return {
        "schema": "dendritic_fmi_fingerprint/v2",
        "profiler_config": asdict(cfg),
        "n_examples": int(inputs.shape[0]),
        "input_dim": int(inputs.shape[1]),
        "output_dim": int(outputs.shape[1]),
        # Honest metric provenance (2026-08-25 review, second pass): without a
        # supplied task metric the spectrum weight is identity, so the rank is
        # the UNWEIGHTED output-spectrum rank -- a sample-limited candidate,
        # not a task-conditioned estimate. task_rank_995 is populated only
        # when a real task metric was supplied; consumers fall back to
        # output_spectrum_rank_995 otherwise.
        "metric_basis": (
            "task_weighted" if task_weight is not None else "identity_output_spectrum"
        ),
        "task_weight_supplied": task_weight is not None,
        "output_spectrum_rank_995": int(task_rank),
        **({"task_rank_995": int(task_rank)} if task_weight is not None else {}),
        # New field (2026-08-25): the covariance/second-moment estimator the
        # spectrum rank above was computed under, with its parameters.
        # Existing fields keep their meanings; "ordinary" reproduces the
        # historical estimator bit-for-bit.
        "spectrum_estimator": {
            "name": str(cfg.spectrum_estimator),
            "winsorize_quantile": float(cfg.spectrum_winsorize_quantile),
            "huber_c": float(cfg.spectrum_huber_c),
            "huber_max_iters": int(cfg.spectrum_huber_max_iters),
            "median_of_means_blocks": int(cfg.spectrum_mom_blocks),
        },
        "beta": float(beta),
        "k95_frac": _weighted_mean(supports, latent_weights),
        "k95_per_latent": supports,
        "active_rank": _weighted_mean(active_ranks, latent_weights),
        "active_rank_per_latent": active_ranks,
        "support_jaccard": _weighted_mean(stability, latent_weights),
        "support_jaccard_per_latent": stability,
        "inhib_frac": _weighted_mean(inhibitory, latent_weights),
        "inhib_frac_per_latent": inhibitory,
        "functional_negative_energy_fraction": _weighted_mean(
            inhibitory, latent_weights
        ),
        "excitatory_k95_frac": _weighted_mean(excitatory_supports, latent_weights),
        "excitatory_k95_frac_per_latent": excitatory_supports,
        "inhibitory_k95_frac": _weighted_mean(inhibitory_supports, latent_weights),
        "inhibitory_k95_frac_per_latent": inhibitory_supports,
        "context_dependent_k95_frac": _weighted_mean(
            context_dependent_supports, latent_weights
        ),
        "context_dependent_k95_frac_per_latent": context_dependent_supports,
        "sign_stable_energy_fraction": _weighted_mean(sign_stability, latent_weights),
        "sign_stable_energy_fraction_per_latent": sign_stability,
        "r_mult": max(r_mults),
        "r_mult_per_latent": r_mults,
        "linear_bypass_fraction": _weighted_mean(linear_bypass, latent_weights),
        "linear_bypass_fraction_per_latent": linear_bypass,
        "linear_bypass_k95_frac": _weighted_mean(
            linear_bypass_supports, latent_weights
        ),
        "linear_bypass_k95_frac_per_latent": linear_bypass_supports,
        "mechanism_scores": mean_mechanisms,
        "mechanism_scores_per_latent": natural_mechanisms,
        "challenge_scores_per_latent": challenge_mechanisms,
        "delta_shunt": statistics.median(shunting_deltas),
        "delta_shunt_per_latent": shunting_deltas,
        "gain_load_score": statistics.median(gain_scores),
        "gain_load_score_per_latent": gain_scores,
        "gain_sensitivity": statistics.median(gain_sensitivities),
        "shunting_win_probability": shunting_win_probability,
        "shunting_win_probability_per_latent": shunting_probabilities,
        "interaction_coordinates": interaction_coords,
        "interaction_depth": int(interaction_depth),
        "interaction_branch_factors": _branch_factors_from_levels(interaction_levels),
        "interaction_labels_per_level": [
            labels.tolist() for labels in interaction_levels
        ],
        "latent_weights": latent_weights,
        "input_scale_profile": {
            "rms_kurtosis": _kurtosis(scales),
            "rms_p99_over_median": float(torch.quantile(scales, 0.99)) / median,
            "rms_max_over_median": float(scales.max()) / median,
        },
        "sample_tail_profile": {
            "schema": "dendritic_fmi_sample_tail/v1",
            "input": _sample_tail_summary(inputs),
            "output": _sample_tail_summary(outputs),
        },
    }


__all__ = [
    "FMIProfilerConfig",
    "profile_teacher_component",
    "profiler_config_from_preset",
]
