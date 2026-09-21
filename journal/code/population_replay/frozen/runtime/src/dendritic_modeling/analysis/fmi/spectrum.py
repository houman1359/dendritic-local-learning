"""Task-weighted output spectrum: soma-count bounds and latent targets.

Level-1 estimators of the FMI theory: the eigen-decomposition of the
(optionally task-weighted) second moment of teacher outputs lower-bounds the
number of scalar somas any linear-readout replacement needs, and its top
eigen-directions define the scalar latent targets every unit-level estimator
operates on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "SPECTRUM_ESTIMATORS",
    "SpectrumFit",
    "fit_robust_task_weighted_spectrum",
    "latent_target",
    "n_min",
    "robust_task_weighted_spectrum",
    "spectrum_slope_beta",
    "task_weighted_spectrum",
]

SPECTRUM_ESTIMATORS = ("ordinary", "winsorized", "huber", "median_of_means")


@dataclass(frozen=True)
class SpectrumFit:
    """Fitted output-spectrum estimator and its row preprocessing state.

    The historical spectrum API exposes only an eigensystem.  Rank validation
    additionally needs to apply the *same* fitted center, clipping, and robust
    row weights to a disjoint holdout.  This object carries that state without
    changing the return contract of :func:`task_weighted_spectrum` or
    :func:`robust_task_weighted_spectrum`.

    ``pre_center`` is the optional ordinary mean removed before task-metric
    whitening. ``estimator_center`` and the optional robust parameters live in
    the whitened coordinate system.
    """

    estimator: str
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    weight_root: torch.Tensor | None
    pre_center: torch.Tensor
    estimator_center: torch.Tensor
    winsor_low: torch.Tensor | None = None
    winsor_high: torch.Tensor | None = None
    huber_scale: torch.Tensor | None = None
    huber_tau: torch.Tensor | None = None

    def transform_rows(self, outputs: torch.Tensor) -> torch.Tensor:
        """Apply the fitted estimator preprocessing to new output rows."""
        if outputs.ndim != 2 or outputs.shape[1] != self.pre_center.numel():
            raise ValueError(
                "outputs must be [M, d_out] with d_out matching the fitted "
                f"dimension {self.pre_center.numel()}, got {tuple(outputs.shape)}"
            )
        y = outputs.to(
            device=self.pre_center.device,
            dtype=self.pre_center.dtype,
        )
        y = y - self.pre_center.unsqueeze(0)
        if self.weight_root is not None:
            y = y @ self.weight_root
        if self.estimator == "winsorized":
            assert self.winsor_low is not None and self.winsor_high is not None
            y = y.clamp(
                min=self.winsor_low.unsqueeze(0),
                max=self.winsor_high.unsqueeze(0),
            )
        centered = y - self.estimator_center.unsqueeze(0)
        if self.estimator == "huber":
            assert self.huber_scale is not None and self.huber_tau is not None
            row_norm = (centered / self.huber_scale).square().mean(dim=1).sqrt()
            row_weight = (self.huber_tau / row_norm.clamp_min(1e-12)).clamp(max=1.0)
            centered = centered * row_weight.sqrt().unsqueeze(1)
        return centered


def task_weighted_spectrum(
    outputs: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    return_root: bool = False,
    center: bool = True,
):
    """Eigenvalues (descending) and eigenvectors of the weighted second moment.

    Args:
        outputs: teacher outputs ``[M, d_out]`` on calibration data.
        weight: optional PSD task metric ``Q`` ``[d_out, d_out]``; identity
            when omitted. Eigenvectors are returned in the whitened basis
            ``Q^(1/2) f``.
        return_root: also return ``Q^(1/2)`` (``None`` when no weight was
            given) — required by :func:`latent_target` whenever ``Q != I``.

    Returns:
        ``(eigenvalues, eigenvectors)`` with eigenvectors as columns, plus
        the whitening root when ``return_root`` is set.
    """
    if outputs.ndim != 2:
        raise ValueError(f"outputs must be [M, d_out], got {tuple(outputs.shape)}")
    y = outputs
    if center:
        # The affine readout carries the mean, so a constant output direction
        # must not consume a soma dimension: rank of a constant output is 0
        # after centering, not 1 (audit fix 2026-08-16; call sites that
        # pre-centered remain correct — centering is idempotent).
        y = y - y.mean(dim=0, keepdim=True)
    root = None
    if weight is not None:
        w_vals, w_vecs = torch.linalg.eigh(0.5 * (weight + weight.T))
        root = w_vecs @ torch.diag(w_vals.clamp_min(0).sqrt()) @ w_vecs.T
        y = y @ root
    if y.shape[1] > y.shape[0]:
        # The empirical covariance has rank at most M-1 after centering. SVD in
        # sample space avoids materializing a d_out x d_out matrix for LLM
        # boundaries while returning the same nonzero eigensystem.
        _, singular_values, right_vectors = torch.linalg.svd(y, full_matrices=False)
        eigenvalues = singular_values.square().div(y.shape[0]).clamp_min(0.0)
        eigenvectors = right_vectors.T
    else:
        second_moment = y.T @ y / y.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        eigenvectors = eigenvectors[:, order]
    if return_root:
        return eigenvalues, eigenvectors, root
    return eigenvalues, eigenvectors


def _descending_eigh(second_moment: torch.Tensor):
    """Symmetric eigendecomposition sorted descending with eigenvalues >= 0."""
    eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvalues[order].clamp_min(0.0), eigenvectors[:, order]


def _row_matrix_spectrum(rows: torch.Tensor, denominator: torch.Tensor | float):
    """Spectrum of ``rows.T @ rows / denominator`` via the cheaper of two routes.

    Mirrors the sample-space/feature-space branch of
    :func:`task_weighted_spectrum` so robust estimators inherit the same
    LLM-width scalability.
    """
    if rows.shape[1] > rows.shape[0]:
        _, singular_values, right_vectors = torch.linalg.svd(rows, full_matrices=False)
        eigenvalues = singular_values.square().div(denominator).clamp_min(0.0)
        return eigenvalues, right_vectors.T
    return _descending_eigh(rows.T @ rows / denominator)


def _column_quantiles(
    y: torch.Tensor, lower: float, upper: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column quantiles, chunked so torch.quantile's element cap holds."""
    probabilities = torch.tensor([lower, upper], dtype=y.dtype, device=y.device)
    chunk = max(1, (2**24) // max(1, y.shape[0]))
    lows, highs = [], []
    for start in range(0, y.shape[1], chunk):
        block = torch.quantile(y[:, start : start + chunk], probabilities, dim=0)
        lows.append(block[0])
        highs.append(block[1])
    return torch.cat(lows), torch.cat(highs)


def _winsorized_spectrum(
    y: torch.Tensor, quantile: float, *, return_state: bool = False
):
    """Per-dimension two-sided winsorization, then the ordinary spectrum.

    Each column is clamped to its own ``[1 - quantile, quantile]`` empirical
    quantiles (default 0.995 clips 0.5% of mass per tail per dimension); the
    clamped data is re-centered at its mean because clamping shifts columns
    with asymmetric tails.  Quantile clamping is translation-equivariant, so
    the result does not depend on any centering applied beforehand.
    """
    if not 0.5 < quantile < 1.0:
        raise ValueError(f"winsorize quantile must be in (0.5, 1), got {quantile}")
    low, high = _column_quantiles(y, 1.0 - quantile, quantile)
    clamped = y.clamp(min=low.unsqueeze(0), max=high.unsqueeze(0))
    center = clamped.mean(dim=0)
    clamped = clamped - center.unsqueeze(0)
    spectrum = _row_matrix_spectrum(clamped, float(clamped.shape[0]))
    if return_state:
        return (*spectrum, center, low, high)
    return spectrum


def _huber_spectrum(
    y: torch.Tensor, c: float, max_iters: int, *, return_state: bool = False
):
    """Huber row-weighted covariance (documented standard construction).

    Construction: standardize coordinates by their median/MAD (robust scale
    ``1.4826 * MAD``); iterate ``max_iters`` rounds of (i) per-row
    standardized RMS norm ``r_i`` about the current center, (ii) Huber
    threshold ``tau = median(r) + c * 1.4826 * MAD(r)`` with the classical
    ``c = 1.345``, (iii) Huber weights ``w_i = min(1, tau / r_i)``, and
    (iv) center update to the ``w``-weighted mean.  The final spectrum is the
    eigendecomposition of ``sum_i w_i (y_i - mu)(y_i - mu)^T / sum_i w_i`` —
    a fixed-shape (row-norm rather than full-Mahalanobis) Huber M-estimator
    of scatter, chosen to avoid inverting a d x d matrix at LLM widths.
    """
    if not c > 0.0:
        raise ValueError(f"huber c must be positive, got {c}")
    if int(max_iters) < 1:
        raise ValueError(f"huber max_iters must be >= 1, got {max_iters}")
    median = y.median(dim=0).values
    scale = (y - median).abs().median(dim=0).values.mul(1.4826).clamp_min(1e-12)
    center = median
    weights = torch.ones(y.shape[0], dtype=y.dtype, device=y.device)
    for _ in range(int(max_iters)):
        standardized = (y - center) / scale
        row_norm = standardized.square().mean(dim=1).sqrt()
        norm_median = row_norm.median()
        norm_mad = (row_norm - norm_median).abs().median().mul(1.4826)
        tau = norm_median + float(c) * norm_mad
        weights = (tau / row_norm.clamp_min(1e-12)).clamp(max=1.0)
        center = (weights.unsqueeze(1) * y).sum(dim=0) / weights.sum().clamp_min(1e-12)
    weighted_rows = (y - center) * weights.sqrt().unsqueeze(1)
    spectrum = _row_matrix_spectrum(weighted_rows, weights.sum().clamp_min(1e-12))
    if return_state:
        return (*spectrum, center, scale, tau)
    return spectrum


def _median_of_means_spectrum(
    y: torch.Tensor, blocks: int, seed: int, *, return_state: bool = False
):
    """Median-of-means covariance: per-block second moments, elementwise median.

    Rows are centered at the global coordinatewise median (robust center),
    shuffled by a seeded CPU generator, and split into ``blocks`` near-equal
    blocks (reduced so every block keeps at least 8 rows).  Each block's
    second moment is computed and the elementwise median across blocks is
    eigendecomposed.  The elementwise median of PSD matrices need not be PSD,
    so eigenvalues are clamped at zero.  The d x d median matrix is
    materialized (assembled in column chunks to bound peak memory).
    """
    if int(blocks) < 1:
        raise ValueError(f"median-of-means blocks must be >= 1, got {blocks}")
    m, d = y.shape
    effective_blocks = max(1, min(int(blocks), m // 8))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(m, generator=generator).to(y.device)
    block_rows = [rows for rows in permutation.chunk(effective_blocks) if rows.numel()]
    center = y.median(dim=0).values
    centered = y - center
    column_chunk = max(1, (2**22) // max(1, d * len(block_rows)))
    median_matrix = torch.empty((d, d), dtype=y.dtype, device=y.device)
    for start in range(0, d, column_chunk):
        end = min(start + column_chunk, d)
        stacked = torch.stack(
            [
                centered[rows].T @ centered[rows][:, start:end] / rows.numel()
                for rows in block_rows
            ]
        )
        median_matrix[:, start:end] = stacked.median(dim=0).values
    spectrum = _descending_eigh(0.5 * (median_matrix + median_matrix.T))
    if return_state:
        return (*spectrum, center)
    return spectrum


def robust_task_weighted_spectrum(
    outputs: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    estimator: str = "ordinary",
    winsorize_quantile: float = 0.995,
    huber_c: float = 1.345,
    huber_max_iters: int = 5,
    mom_blocks: int = 16,
    seed: int = 0,
    return_root: bool = False,
    center: bool = True,
):
    """Task-weighted spectrum with a selectable covariance estimator.

    ``estimator="ordinary"`` delegates verbatim to
    :func:`task_weighted_spectrum` and is bit-identical to it.  The robust
    estimators share the ordinary pipeline's centering/whitening contract
    (eigenvectors live in the whitened basis ``Q^(1/2) f``) but replace the
    empirical second moment with a heavy-tail-resistant estimate:

    - ``"winsorized"``: per-dimension two-sided winsorization at
      ``winsorize_quantile`` before the spectrum (see
      :func:`_winsorized_spectrum`).
    - ``"huber"``: iteratively reweighted rows with Huber weights on a
      MAD-standardized per-row norm, ``tau = median + huber_c * 1.4826 * MAD``
      (see :func:`_huber_spectrum`).
    - ``"median_of_means"``: ``mom_blocks`` seeded row blocks, per-block
      second moments, elementwise median (see
      :func:`_median_of_means_spectrum`).

    Robust paths promote sub-float32 inputs to float32 (order statistics and
    eigensolvers need a float dtype); the ordinary path is left untouched.
    """
    name = str(estimator).strip().lower()
    if name not in SPECTRUM_ESTIMATORS:
        raise ValueError(
            f"spectrum estimator must be one of {SPECTRUM_ESTIMATORS}, got {estimator!r}"
        )
    if name == "ordinary":
        return task_weighted_spectrum(
            outputs, weight, return_root=return_root, center=center
        )
    if outputs.ndim != 2:
        raise ValueError(f"outputs must be [M, d_out], got {tuple(outputs.shape)}")
    y = outputs
    if y.dtype not in (torch.float32, torch.float64):
        y = y.float()
    if center:
        # Robust estimators re-derive their own robust centers below; the
        # ordinary mean-centering here only fixes the translation frame and
        # keeps the whitening applied to the same data as the ordinary path.
        y = y - y.mean(dim=0, keepdim=True)
    root = None
    if weight is not None:
        # Identical whitening-root construction to task_weighted_spectrum.
        w_vals, w_vecs = torch.linalg.eigh(0.5 * (weight + weight.T))
        root = w_vecs @ torch.diag(w_vals.clamp_min(0).sqrt()) @ w_vecs.T
        root = root.to(dtype=y.dtype)
        y = y @ root
    if name == "winsorized":
        eigenvalues, eigenvectors = _winsorized_spectrum(y, float(winsorize_quantile))
    elif name == "huber":
        eigenvalues, eigenvectors = _huber_spectrum(
            y, float(huber_c), int(huber_max_iters)
        )
    else:
        eigenvalues, eigenvectors = _median_of_means_spectrum(
            y, int(mom_blocks), int(seed)
        )
    if return_root:
        return eigenvalues, eigenvectors, root
    return eigenvalues, eigenvectors


def fit_robust_task_weighted_spectrum(
    outputs: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    estimator: str = "ordinary",
    winsorize_quantile: float = 0.995,
    huber_c: float = 1.345,
    huber_max_iters: int = 5,
    mom_blocks: int = 16,
    seed: int = 0,
    center: bool = True,
) -> SpectrumFit:
    """Fit a spectrum estimator and retain preprocessing for held-out rows.

    This is the stateful counterpart to
    :func:`robust_task_weighted_spectrum`.  The ordinary eigensystem is
    obtained by calling the historical implementation directly, preserving
    its exact numerical behavior.  Robust estimators expose the fitted
    centers and transformations that their existing private implementations
    already compute.
    """
    name = str(estimator).strip().lower()
    if name not in SPECTRUM_ESTIMATORS:
        raise ValueError(
            f"spectrum estimator must be one of {SPECTRUM_ESTIMATORS}, "
            f"got {estimator!r}"
        )
    if outputs.ndim != 2:
        raise ValueError(f"outputs must be [M, d_out], got {tuple(outputs.shape)}")

    if name == "ordinary":
        eigenvalues, eigenvectors, root = task_weighted_spectrum(
            outputs,
            weight,
            return_root=True,
            center=center,
        )
        pre_center = (
            outputs.mean(dim=0)
            if center
            else torch.zeros(
                outputs.shape[1], dtype=outputs.dtype, device=outputs.device
            )
        )
        estimator_center = torch.zeros_like(pre_center)
        if root is not None:
            estimator_center = estimator_center.to(dtype=root.dtype) @ root
        return SpectrumFit(
            estimator=name,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            weight_root=root,
            pre_center=pre_center,
            estimator_center=estimator_center,
        )

    y = outputs
    if y.dtype not in (torch.float32, torch.float64):
        y = y.float()
    pre_center = (
        y.mean(dim=0)
        if center
        else torch.zeros(y.shape[1], dtype=y.dtype, device=y.device)
    )
    y = y - pre_center.unsqueeze(0)
    root = None
    if weight is not None:
        w_vals, w_vecs = torch.linalg.eigh(0.5 * (weight + weight.T))
        root = w_vecs @ torch.diag(w_vals.clamp_min(0).sqrt()) @ w_vecs.T
        root = root.to(dtype=y.dtype)
        y = y @ root

    if name == "winsorized":
        eigenvalues, eigenvectors, estimator_center, low, high = _winsorized_spectrum(
            y, float(winsorize_quantile), return_state=True
        )
        return SpectrumFit(
            estimator=name,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            weight_root=root,
            pre_center=pre_center,
            estimator_center=estimator_center,
            winsor_low=low,
            winsor_high=high,
        )
    if name == "huber":
        eigenvalues, eigenvectors, estimator_center, scale, tau = _huber_spectrum(
            y,
            float(huber_c),
            int(huber_max_iters),
            return_state=True,
        )
        return SpectrumFit(
            estimator=name,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            weight_root=root,
            pre_center=pre_center,
            estimator_center=estimator_center,
            huber_scale=scale,
            huber_tau=tau,
        )

    eigenvalues, eigenvectors, estimator_center = _median_of_means_spectrum(
        y,
        int(mom_blocks),
        int(seed),
        return_state=True,
    )
    return SpectrumFit(
        estimator=name,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        weight_root=root,
        pre_center=pre_center,
        estimator_center=estimator_center,
    )


def n_min(eigenvalues: torch.Tensor, epsilon: float) -> int:
    """Smallest rank capturing ``1 - epsilon`` of the spectral energy."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0
    cumulative = torch.cumsum(eigenvalues, dim=0) / total
    return min(int((cumulative < 1.0 - epsilon).sum()) + 1, eigenvalues.numel())


def spectrum_slope_beta(eigenvalues: torch.Tensor, rank: int) -> float:
    """Least-squares slope of ``log lambda_i`` vs ``log i`` over ``i <= rank``.

    Returned as the positive decay exponent ``beta_y`` (steeper spectrum ->
    larger beta). Fewer than two strictly positive eigenvalues carry no slope
    information, so ``0.0`` is returned (never a clamp artifact).
    """
    positives = int((eigenvalues > 0).sum())
    if positives < 2:
        return 0.0
    rank = max(2, min(int(rank), positives))
    lam = eigenvalues[:rank]
    log_i = torch.log(torch.arange(1, rank + 1, dtype=lam.dtype, device=lam.device))
    log_lam = torch.log(lam)
    centered_i = log_i - log_i.mean()
    slope = float(
        (centered_i * (log_lam - log_lam.mean())).sum() / (centered_i**2).sum()
    )
    return -slope


def latent_target(
    eigenvectors: torch.Tensor,
    index: int,
    weight_root: torch.Tensor | None = None,
):
    """Return a closure mapping teacher outputs to the ``index``-th latent.

    The eigenvectors of :func:`task_weighted_spectrum` live in the whitened
    basis ``Q^(1/2) f``; when a task weight was used, its root must be passed
    here so the closure whitens raw outputs before projecting (with ``Q = I``
    the root may be omitted).
    """
    direction = eigenvectors[:, index]

    def target(outputs: torch.Tensor) -> torch.Tensor:
        if weight_root is not None:
            outputs = outputs @ weight_root
        return outputs @ direction

    return target
