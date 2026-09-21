"""Finite-horizon operators for the frozen linear morphology experiment.

No manuscript data or fitted selector choices are imported here. The Gaussian
operator is exact for fresh iid Gaussian inputs, conditional linear targets and
independent zero-mean label noise. It is not exact for repeatedly sampling a
finite training cache. The empirical spectral operator is exact full-batch GD
on its supplied fit set and evaluates on a separately supplied evaluation set.
"""
from __future__ import annotations

import numpy as np


def gaussian_risk(mean, covariance_gram, context_features, targets, probabilities, noise_variances):
    """Expected half-MSE from M=E[V] and CovGram=E[(V-M)(V-M)^T]."""
    residual = np.einsum('crk,ckd->crd', context_features, mean) - targets[None]
    variance = np.einsum('crk,ckl,crl->cr', context_features, covariance_gram, context_features)
    return .5 * ((np.sum(residual**2, axis=-1) + variance + noise_variances) @ probabilities)


def gaussian_moments(context_features, targets, probabilities, noise_variances,
                     eta, batch_size, checkpoints, initial_mean=None, initial_covariance_gram=None):
    """Exact first/Gram-second moments for independent fresh-example SGD.

    context_features has shape (candidates, contexts, K); targets is (r,d).
    Zero padding allows different K in one call. A minibatch averages B iid
    examples with replacement from the specified *population*. Only isotropic
    Gaussian x and label-noise first/second moments are assumed.
    """
    c = np.asarray(context_features, float)
    t = np.asarray(targets, float)
    p = np.asarray(probabilities, float)
    noise = np.asarray(noise_variances, float)
    n, _, k = c.shape
    d = t.shape[1]
    h = np.einsum('r,crk,crl->ckl', p, c, c)
    cross = np.einsum('r,crk,rd->ckd', p, c, t)
    mean = np.zeros((n, k, d)) if initial_mean is None else np.array(initial_mean, copy=True)
    cov = np.zeros((n, k, k)) if initial_covariance_gram is None else np.array(initial_covariance_gram, copy=True)
    contraction = np.eye(k)[None] - eta*h
    inv_batch = 0. if np.isinf(batch_size) else 1./batch_size
    out = {}
    for step in range(1, max(checkpoints)+1):
        residual = np.einsum('crk,ckd->crd', c, mean) - t[None]
        residual2 = np.sum(residual**2, axis=-1) + np.einsum('crk,ckl,crl->cr', c, cov, c)
        gradient_mean = h @ mean - cross
        # Gaussian fourth-moment contraction: E[(e^T x-eps)^2 ||x||^2].
        single_gradient_gram = np.einsum('r,cr,crk,crl->ckl', p, (d+2)*residual2+d*noise, c, c)
        conditional_gradient_gram = h @ cov @ h + gradient_mean @ gradient_mean.transpose(0, 2, 1)
        cov = contraction @ cov @ contraction.transpose(0, 2, 1) + eta**2*inv_batch*(single_gradient_gram-conditional_gradient_gram)
        cov = .5*(cov+cov.transpose(0, 2, 1))
        mean = mean-eta*gradient_mean
        if step in checkpoints:
            out[step] = dict(loss=gaussian_risk(mean, cov, c, t, p, noise), mean=mean.copy(),
                             covariance_gram=cov.copy())
    return out


def gaussian_fullbatch_spectral(context_features, targets, probabilities, noise_variances, eta, checkpoints):
    """Closed-form GD mean, including exactly unlearnable zero-eigenvalue modes."""
    c = np.asarray(context_features, float)
    h = np.einsum('r,crk,crl->ckl', probabilities, c, c)
    cross = np.einsum('r,crk,rd->ckd', probabilities, c, targets)
    eigenvalues, vectors = np.linalg.eigh(h)
    # PSD round-off is tolerated; true negative eigenvalues are an error.
    if eigenvalues.min() < -1e-10:
        raise ValueError('Context Hessian is not positive semidefinite')
    eigenvalues = np.maximum(eigenvalues, 0.)
    projected = vectors.transpose(0, 2, 1) @ cross
    out = {}
    for steps in checkpoints:
        gain = np.full_like(eigenvalues, eta*steps)
        valid = (eigenvalues > 1e-14) & (eta*eigenvalues < 1.)
        gain[valid] = -np.expm1(steps*np.log1p(-eta*eigenvalues[valid]))/eigenvalues[valid]
        # Generic real formula also covers a stable negative contraction factor.
        negative_contraction = eta*eigenvalues >= 1.
        gain[negative_contraction] = (1.-(1.-eta*eigenvalues[negative_contraction])**steps)/eigenvalues[negative_contraction]
        mean = vectors @ (gain[..., None]*projected)
        cov = np.zeros_like(h)
        out[steps] = dict(loss=gaussian_risk(mean, cov, c, targets, probabilities, noise_variances), mean=mean)
    return out


def empirical_fullbatch_spectral(fit_design, fit_y, evaluation_design, evaluation_y, eta, checkpoints):
    """Exact fit-set GD trajectory, evaluated on explicitly separate observations."""
    z = np.asarray(fit_design, float)
    h = z.T @ z/len(z)
    b = z.T @ fit_y/len(z)
    values, vectors = np.linalg.eigh(h)
    values = np.maximum(values, 0.)
    projected = vectors.T @ b
    out = {}
    for steps in checkpoints:
        gain = np.full_like(values, eta*steps)
        valid = values > 1e-14
        gain[valid] = (1.-(1.-eta*values[valid])**steps)/values[valid]
        weights = vectors @ (gain*projected)
        loss = .5*np.mean((evaluation_design @ weights-evaluation_y)**2)
        out[steps] = dict(loss=float(loss), weights=weights)
    return out


def fit_gaussian_context_model(x, a, y):
    """Calibration-only conditional OLS with unbiased residual variance estimates.

    The observed context frequencies are estimated; x~N(0,I) is an explicit
    model assumption. Context vectors themselves are supplied observations.
    """
    _, first, labels = np.unique(np.round(a, 12), axis=0, return_index=True, return_inverse=True)
    contexts = a[first]
    targets, variances, counts = [], [], []
    for label in range(len(first)):
        mask = labels == label
        xx, yy = x[mask], y[mask]
        target, _, rank, _ = np.linalg.lstsq(xx, yy, rcond=None)
        if rank != x.shape[1] or len(xx) <= x.shape[1]:
            raise ValueError('Insufficient calibration observations for conditional OLS')
        targets.append(target)
        variances.append(np.sum((xx @ target-yy)**2)/(len(xx)-rank))
        counts.append(len(xx))
    return dict(contexts=contexts, targets=np.asarray(targets), probabilities=np.asarray(counts)/len(x),
                noise_variances=np.asarray(variances), counts=counts)
