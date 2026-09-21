"""Biological loss functions for training dendritic neurons to match visual properties.

All functions are differentiable and can be used in gradient-based optimization.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as functional

__all__ = [
    "compute_bio_losses",
    "loss_dog",
    "loss_dsi",
    "loss_endstop",
    "loss_osi",
    "loss_phase_invariance",
    "loss_sparsity",
]


def loss_dog(sta: torch.Tensor, target_ratio: float = 1.6) -> torch.Tensor:
    """Difference of Gaussians loss for center-surround organization.

    Fits the spatial receptive field to a DoG model and penalizes
    deviation from expected center/surround ratio.

    Parameters
    ----------
    sta : torch.Tensor
        Spike-triggered average or receptive field, shape (H, W)
    target_ratio : float
        Target ratio of surround_sigma / center_sigma

    Returns
    -------
    loss : torch.Tensor
        DoG fit loss (scalar)
    """
    H, W = sta.shape

    # Create coordinate grids
    y = torch.arange(H, dtype=sta.dtype, device=sta.device) - H / 2
    x = torch.arange(W, dtype=sta.dtype, device=sta.device) - W / 2
    Y, X = torch.meshgrid(y, x, indexing="ij")
    R2 = X**2 + Y**2

    # Estimate center size from peak response location
    center_mask = sta.abs() > 0.7 * sta.abs().max()
    if center_mask.sum() > 0:
        center_size = torch.sqrt(R2[center_mask].mean())
    else:
        center_size = torch.tensor(3.0, device=sta.device)

    # Fit DoG parameters
    sigma_c = center_size
    sigma_s = sigma_c * target_ratio

    # Generate ideal DoG
    center_g = torch.exp(-R2 / (2 * sigma_c**2))
    surround_g = torch.exp(-R2 / (2 * sigma_s**2))

    # Normalize
    center_g = center_g / center_g.sum()
    surround_g = surround_g / surround_g.sum()

    # Determine polarity (ON or OFF center)
    if sta.sum() > 0:  # ON-center
        dog_ideal = center_g - 0.5 * surround_g
    else:  # OFF-center
        dog_ideal = -center_g + 0.5 * surround_g

    # Scale to match
    scale = (sta * dog_ideal).sum() / (dog_ideal * dog_ideal).sum()
    dog_ideal = dog_ideal * scale

    # MSE loss
    loss = functional.mse_loss(sta, dog_ideal)

    # Add penalty for deviation from target ratio
    actual_ratio = sigma_s / sigma_c
    ratio_loss = (actual_ratio - target_ratio).abs()

    return loss + 0.1 * ratio_loss


def loss_osi(
    responses: torch.Tensor, orientations: torch.Tensor, target_osi: float = 0.8
) -> torch.Tensor:
    """Orientation selectivity index loss.

    Parameters
    ----------
    responses : torch.Tensor
        Response magnitudes, shape (n_orientations,)
    orientations : torch.Tensor
        Stimulus orientations in degrees
    target_osi : float
        Target OSI value

    Returns
    -------
    loss : torch.Tensor
        OSI loss (scalar)
    """
    # Convert to radians
    ori_rad = orientations * np.pi / 180

    # Find preferred orientation
    pref_idx = responses.argmax()
    pref_ori = ori_rad[pref_idx]

    # Find orthogonal orientation
    orth_ori = (pref_ori + np.pi / 2) % np.pi
    orth_diffs = ((ori_rad - orth_ori + np.pi) % np.pi - np.pi / 2).abs()
    orth_idx = orth_diffs.argmin()

    # Calculate OSI
    r_pref = responses[pref_idx]
    r_orth = responses[orth_idx]
    osi = (r_pref - r_orth) / (r_pref + r_orth + 1e-6)

    # Loss: deviation from target
    loss = (osi - target_osi) ** 2

    # Add penalty for low overall response
    mean_response = responses.mean()
    if mean_response < 0.1:
        loss = loss + (0.1 - mean_response) ** 2

    return loss


def loss_dsi(
    responses: torch.Tensor, directions: torch.Tensor, target_dsi: float = 0.5
) -> torch.Tensor:
    """Direction selectivity index loss.

    Parameters
    ----------
    responses : torch.Tensor
        Response magnitudes, shape (n_directions,)
    directions : torch.Tensor
        Motion directions in degrees
    target_dsi : float
        Target DSI value

    Returns
    -------
    loss : torch.Tensor
        DSI loss (scalar)
    """
    # Convert to radians
    dir_rad = directions * np.pi / 180

    # Find preferred direction
    pref_idx = responses.argmax()
    pref_dir = dir_rad[pref_idx]

    # Find null direction (opposite)
    null_dir = (pref_dir + np.pi) % (2 * np.pi)
    null_diffs = ((dir_rad - null_dir) % (2 * np.pi)).abs()
    null_idx = null_diffs.argmin()

    # Calculate DSI
    r_pref = responses[pref_idx]
    r_null = responses[null_idx]
    dsi = (r_pref - r_null) / (r_pref + r_null + 1e-6)

    # Loss: deviation from target
    loss = (dsi - target_dsi) ** 2

    return loss


def loss_phase_invariance(
    responses_phases: torch.Tensor, target_f1_f0: float = 0.5
) -> torch.Tensor:
    """Phase invariance loss (for complex cells).

    Parameters
    ----------
    responses_phases : torch.Tensor
        Responses to different phases, shape (n_phases,)
    target_f1_f0 : float
        Target F1/F0 ratio (low = complex cell)

    Returns
    -------
    loss : torch.Tensor
        Phase invariance loss
    """
    # Calculate F1 (fundamental frequency) and F0 (mean)
    f0 = responses_phases.mean()
    f1 = (responses_phases.max() - responses_phases.min()) / 2

    # F1/F0 ratio
    f1_f0 = f1 / (f0 + 1e-6)

    # Loss: complex cells should have low F1/F0
    loss = (f1_f0 - target_f1_f0) ** 2

    # Add variance penalty to encourage invariance
    variance = responses_phases.var()
    target_var = (f0 * 0.1) ** 2  # 10% of mean
    var_loss = torch.relu(variance - target_var)

    return loss + 0.5 * var_loss


def loss_endstop(
    length_curve: torch.Tensor, preferred_idx: Optional[int] = None
) -> torch.Tensor:
    """End-stopping (length tuning) loss.

    Parameters
    ----------
    length_curve : torch.Tensor
        Responses to different bar lengths, shape (n_lengths,)
    preferred_idx : int or None
        Expected peak position (None = auto-detect)

    Returns
    -------
    loss : torch.Tensor
        End-stopping loss
    """
    n_lengths = len(length_curve)

    if preferred_idx is None:
        # Find peak
        preferred_idx = length_curve.argmax().item()

    # Ideal end-stopped curve: peak then decline
    ideal_curve = torch.zeros_like(length_curve)

    # Rising phase
    for i in range(preferred_idx + 1):
        ideal_curve[i] = (i + 1) / (preferred_idx + 1)

    # Falling phase (end-stopping)
    for i in range(preferred_idx + 1, n_lengths):
        fall_rate = 0.7  # How much it falls
        t = (i - preferred_idx) / (n_lengths - preferred_idx - 1)
        ideal_curve[i] = 1.0 - fall_rate * t

    # Scale to match
    scale = length_curve.max()
    ideal_curve = ideal_curve * scale

    # MSE loss
    loss = functional.mse_loss(length_curve, ideal_curve)

    # Add penalty if no clear peak
    if length_curve.max() < 1.5 * length_curve.min():
        loss = loss + 0.5

    return loss


def loss_sparsity(
    activations: torch.Tensor, target_sparsity: float = 0.1
) -> torch.Tensor:
    """Sparsity loss to encourage sparse activations.

    Parameters
    ----------
    activations : torch.Tensor
        Neural activations of any shape
    target_sparsity : float
        Target fraction of active units

    Returns
    -------
    loss : torch.Tensor
        Sparsity loss
    """
    # L1 sparsity
    l1_loss = activations.abs().mean()

    # Hoyer sparsity (ratio of L1 to L2 norm)
    l1_norm = activations.abs().sum()
    l2_norm = (activations**2).sum().sqrt()
    n = activations.numel()

    hoyer = (np.sqrt(n) - l1_norm / (l2_norm + 1e-6)) / (np.sqrt(n) - 1)
    hoyer_loss = (hoyer - target_sparsity) ** 2

    return l1_loss * 0.01 + hoyer_loss


def compute_bio_losses(
    model_responses: dict, loss_weights: dict, probe_metadata: dict
) -> tuple[torch.Tensor, dict]:
    """Compute weighted combination of biological losses.

    Parameters
    ----------
    model_responses : dict
        Dictionary of responses for different probe types:
        - 'gratings': (batch, n_orientations)
        - 'motion': (batch, n_directions)
        - 'dots': (batch, h, w) spike-triggered average
        - 'bars': (batch, n_lengths)
        - 'phases': (batch, n_phases)

    loss_weights : dict
        Weights for each loss component

    probe_metadata : dict
        Metadata about probes (orientations, directions, etc.)

    Returns
    -------
    total_loss : torch.Tensor
        Weighted sum of all losses
    loss_dict : dict
        Individual loss values for logging
    """
    losses = {}

    # DoG loss (if dots/noise responses available)
    if "dots" in model_responses and loss_weights.get("dog", 0) > 0:
        sta = model_responses["dots"].mean(0)  # Average over batch
        losses["dog"] = loss_dog(sta)

    # OSI loss (if grating responses available)
    if "gratings" in model_responses and loss_weights.get("osi", 0) > 0:
        responses = model_responses["gratings"].mean(0)  # Average over batch
        orientations = torch.tensor(probe_metadata["orientations"])
        losses["osi"] = loss_osi(responses, orientations)

    # DSI loss (if motion responses available)
    if "motion" in model_responses and loss_weights.get("dsi", 0) > 0:
        responses = model_responses["motion"].mean(0)
        directions = torch.tensor(probe_metadata["directions"])
        losses["dsi"] = loss_dsi(responses, directions)

    # Phase invariance loss
    if "phases" in model_responses and loss_weights.get("phase", 0) > 0:
        responses = model_responses["phases"].mean(0)
        losses["phase"] = loss_phase_invariance(responses)

    # End-stopping loss
    if "bars" in model_responses and loss_weights.get("endstop", 0) > 0:
        length_curve = model_responses["bars"].mean(0)
        losses["endstop"] = loss_endstop(length_curve)

    # Sparsity loss (always included if weight > 0)
    if loss_weights.get("sparse", 0) > 0:
        # Collect all activations
        all_activations = []
        for _key, resp in model_responses.items():
            all_activations.append(resp.flatten())
        activations = torch.cat(all_activations)
        losses["sparse"] = loss_sparsity(activations)

    # Compute weighted total
    total_loss = torch.tensor(0.0, device=next(iter(model_responses.values())).device)
    for key, loss in losses.items():
        weight = loss_weights.get(key, 0.0)
        total_loss = total_loss + weight * loss

    return total_loss, losses
