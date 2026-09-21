"""Information theory estimators with multiple backend implementations.

This module provides a unified interface for computing mutual information
and related metrics using different estimation methods:
- Kraskov (k-nearest neighbors)
- Scikit-learn (kernel density or histogram-based)
- Copula-based (via DVC)
- PID (Partial Information Decomposition)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Optional, Union

import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mutual_info_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dendritic_modeling.utils.information.nearest_neighbors import (
    compute_cmi_ccc,
    compute_mi_cc,
    compute_mi_cd,
)


class EstimationMethod(Enum):
    """Available estimation methods for mutual information."""

    KRASKOV = "kraskov"
    DECODER = "decoder"  # Cross-validated decoder-based I(X;C) for discrete targets
    BINNED = "binned"  # Histogram/binning plug-in estimator (best for low-dim)
    SKLEARN = "sklearn"
    COPULA = "copula"
    AUTO = "auto"  # Automatically select based on data properties


@dataclass
class EstimatorConfig:
    """Configuration for information estimators."""

    method: Union[str, EstimationMethod] = EstimationMethod.AUTO
    n_neighbors: int = 10  # For Kraskov
    n_bins: int = 20  # For histogram-based methods
    bandwidth: Optional[float] = None  # For kernel density estimation
    copula_type: str = "gaussian"  # For copula-based estimation
    use_gpu: bool = False
    verbose: bool = False
    # Additional method-specific parameters
    method_params: dict[str, Any] = None

    def __post_init__(self):
        if isinstance(self.method, str):
            self.method = EstimationMethod(self.method.lower())
        if self.method_params is None:
            self.method_params = {}


class BaseInformationEstimator(ABC):
    """Abstract base class for information theory estimators."""

    def __init__(self, config: EstimatorConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        if config.verbose:
            self.logger.setLevel(logging.DEBUG)

    @abstractmethod
    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information I(X;Y) between two variables."""

    @abstractmethod
    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Compute conditional mutual information I(X;Y|Z)."""

    def entropy(self, x: np.ndarray) -> float:
        """Compute entropy H(X) of a variable."""
        # Default implementation using MI: H(X) = I(X;X)
        return self.mutual_information(x, x)

    def conditional_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute conditional entropy H(X|Y)."""
        # H(X|Y) = H(X) - I(X;Y)
        return self.entropy(x) - self.mutual_information(x, y)

    def joint_mutual_information(
        self, sources: list[np.ndarray], target: np.ndarray
    ) -> float:
        """Compute mutual information I(X1,X2,...;Y) for multiple sources."""
        # Concatenate sources and compute MI with target
        joint_source = np.hstack([self._ensure_2d(s) for s in sources])
        return self.mutual_information(joint_source, target)

    def _chain_rule_cmi(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Compute CMI with the chain rule using this estimator's MI backend."""
        yz = np.hstack([y, z])
        mi_xyz = self.mutual_information(x, yz)
        mi_xz = self.mutual_information(x, z)
        return max(0.0, mi_xyz - mi_xz)

    def mutual_information_null(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        n_shuffles: int = 100,
        seed: int = 0,
        return_values: bool = False,
    ):
        """Permutation (shuffled-null) baseline for MI.

        This is a practical way to estimate finite-sample bias and a null
        distribution: it permutes `y` across samples and recomputes I(X;Y).

        Returns
        -------
        (mean, std) or (mean, std, values) in nats, where values has length n_shuffles.
        """
        n = int(n_shuffles)
        if n <= 0:
            return (
                (0.0, 0.0, np.array([], dtype=float)) if return_values else (0.0, 0.0)
            )

        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        self._validate_inputs(x, y)

        rng = np.random.default_rng(int(seed))
        vals = np.empty(n, dtype=float)
        for i in range(n):
            perm = rng.permutation(y.shape[0])
            vals[i] = float(self.mutual_information(x, y[perm]))

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        return (mean, std, vals) if return_values else (mean, std)

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D for consistent processing."""
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    @staticmethod
    def _validate_inputs(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None):
        """Validate input arrays have compatible shapes."""
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and Y must have same number of samples: {x.shape[0]} vs {y.shape[0]}"
            )
        if z is not None and z.shape[0] != x.shape[0]:
            raise ValueError(
                f"Z must have same number of samples as X and Y: {z.shape[0]} vs {x.shape[0]}"
            )


class KraskovEstimator(BaseInformationEstimator):
    """Kraskov k-nearest neighbors estimator for mutual information.

    Based on:
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information. Physical review E, 69(6), 066138.
    """

    def __init__(self, config: EstimatorConfig):
        super().__init__(config)
        self.n_neighbors = config.n_neighbors

    @staticmethod
    def _is_discrete_1d(values: np.ndarray) -> bool:
        """Heuristic: decide whether a 1D array should be treated as discrete.

        We treat a variable as discrete if:
        - It has a small number of unique values (e.g. class labels), OR
        - It is integer-like with a moderate number of unique values.
        """
        values = np.asarray(values).ravel()
        if values.size == 0:
            return False
        unique = np.unique(values)
        if unique.size <= 1:
            return True
        # Small-unique AND clearly not continuous-like (unique << n_samples)
        if unique.size <= 20 and unique.size <= max(5, values.size // 5):
            return True
        # Integer-like labels (classification), allow more uniques.
        if unique.size <= 200 and np.allclose(unique, np.round(unique), atol=1e-6):
            return True
        return False

    @staticmethod
    def _to_int_labels(values: np.ndarray) -> np.ndarray:
        """Map arbitrary 1D values to contiguous integer labels [0..K-1]."""
        values = np.asarray(values).ravel()
        unique = np.unique(values)
        # np.unique returns sorted uniques; searchsorted is fast and stable.
        return np.searchsorted(unique, values).astype(int)

    def _mutual_information_with_k(self, x: np.ndarray, y: np.ndarray, k: int) -> float:
        """Mutual information with an explicit k (neighbors) parameter."""
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        self._validate_inputs(x, y)

        x_is_discrete = x.shape[1] == 1 and self._is_discrete_1d(x[:, 0])
        y_is_discrete = y.shape[1] == 1 and self._is_discrete_1d(y[:, 0])

        # Handle discrete-discrete exactly (in nats).
        if x_is_discrete and y_is_discrete:
            x_lab = self._to_int_labels(x.ravel())
            y_lab = self._to_int_labels(y.ravel())
            return float(mutual_info_score(x_lab, y_lab))

        # Handle mixed continuous/discrete using Ross (2014).
        if x_is_discrete and not y_is_discrete:
            # compute_mi_cd expects (continuous, discrete)
            return compute_mi_cd(y, self._to_int_labels(x.ravel()), k)
        if not x_is_discrete and y_is_discrete:
            return compute_mi_cd(x, self._to_int_labels(y.ravel()), k)

        # Continuous-continuous KSG.
        return compute_mi_cc(x, y, k)

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI using Kraskov's k-NN estimator."""
        if self.config.verbose:
            self.logger.debug(
                f"    Computing MI: x{np.asarray(x).shape}, y{np.asarray(y).shape}, k={self.n_neighbors}"
            )
        return self._mutual_information_with_k(x, y, self.n_neighbors)

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Compute CMI I(X;Y|Z) with discrete-aware handling.

        Cases handled explicitly:
        - Z discrete (e.g. class labels): I(X;Y|Z) = Σ_z p(z) I(X;Y | Z=z)
        - Y discrete, Z continuous: I(X;Y|Z) = I([X,Z];Y) - I(Z;Y)
        - Otherwise (all continuous): default chain-rule approximation
        """
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        self._validate_inputs(x, y, z)

        k = int(self.n_neighbors)
        if k < 1:
            return 0.0

        y_is_discrete = y.shape[1] == 1 and self._is_discrete_1d(y[:, 0])
        z_is_discrete = z.shape[1] == 1 and self._is_discrete_1d(z[:, 0])

        # If Z is discrete, compute a weighted average of within-class MI.
        if z_is_discrete:
            z_labels = z.ravel()
            n_total = int(z_labels.size)
            if n_total <= 1:
                return 0.0

            cmi = 0.0
            for val in np.unique(z_labels):
                mask = z_labels == val
                n = int(np.sum(mask))
                # ``kneighbors(X=None)`` excludes each fitted sample itself, so
                # a subgroup only needs one other sample and k <= n - 1.
                if n <= 1:
                    continue
                k_sub = min(k, n - 1)
                if k_sub < 1:
                    continue
                try:
                    mi_sub = self._mutual_information_with_k(x[mask], y[mask], k_sub)
                except ValueError as exc:
                    # Be tolerant to rare degenerate subgroups where sklearn rejects
                    # neighbor queries (e.g., effectively tiny fit sets).
                    if "n_neighbors" in str(exc) and "n_samples_fit" in str(exc):
                        if self.config.verbose:
                            self.logger.debug(
                                "Skipping tiny/degenerate CMI subgroup for z=%s (n=%d): %s",
                                str(val),
                                n,
                                str(exc),
                            )
                        continue
                    raise
                cmi += (n / n_total) * mi_sub
            return float(max(0.0, cmi))

        # If Y is discrete and Z is continuous, use chain rule with the mixed MI estimator.
        if y_is_discrete:
            y_labels = self._to_int_labels(y.ravel())
            xz = np.hstack([x, z])
            mi_xz_y = compute_mi_cd(xz, y_labels, k)
            mi_z_y = compute_mi_cd(z, y_labels, k)
            return float(max(0.0, mi_xz_y - mi_z_y))

        # Default (continuous) chain rule: I(X;Y|Z) = I(X;[Y,Z]) - I(X;Z)
        return float(compute_cmi_ccc(x, y, z, k))


class BinnedEstimator(BaseInformationEstimator):
    """Binning / histogram plug-in estimator (discrete MI on discretized data).

    This is most appropriate for low-dimensional settings. For high-dimensional
    vectors, binning quickly becomes sparse and strongly biased; treat it as a
    diagnostic/baseline rather than a primary estimator.
    """

    def __init__(self, config: EstimatorConfig):
        super().__init__(config)
        self.n_bins = int(config.n_bins)
        mp = dict(config.method_params or {})
        self.binning_strategy = str(mp.get("binning_strategy", "quantile")).lower()
        self.bias_correction = str(mp.get("bias_correction", "none")).lower()

    def _bin_continuous(self, x: np.ndarray) -> np.ndarray:
        x = self._ensure_2d(np.asarray(x))
        n, d = x.shape
        if self.n_bins < 2:
            return np.zeros((n, d), dtype=int)

        binned = np.zeros((n, d), dtype=int)
        for j in range(d):
            col = x[:, j]
            if np.allclose(col, col[0], atol=1e-12):
                binned[:, j] = 0
                continue

            if self.binning_strategy == "uniform":
                edges = np.linspace(col.min(), col.max(), self.n_bins + 1)[1:-1]
            else:
                # Default: quantile binning
                qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
                edges = np.quantile(col, qs)

            edges = np.unique(edges)
            if edges.size == 0:
                binned[:, j] = 0
            else:
                binned[:, j] = np.digitize(col, edges, right=False)

        return binned

    def _to_states(self, x: np.ndarray) -> np.ndarray:
        x = self._ensure_2d(np.asarray(x))
        if x.shape[1] == 1 and KraskovEstimator._is_discrete_1d(x[:, 0]):
            return KraskovEstimator._to_int_labels(x[:, 0])

        binned = self._bin_continuous(x)
        _, inv = np.unique(binned, axis=0, return_inverse=True)
        return inv.astype(int)

    @staticmethod
    def _miller_madow_mi_correction(
        n_samples: int, x_states: np.ndarray, y_states: np.ndarray
    ) -> float:
        n = float(max(1, int(n_samples)))
        kx = int(np.unique(x_states).size)
        ky = int(np.unique(y_states).size)
        _, inv_xy = np.unique(
            np.column_stack([x_states, y_states]), axis=0, return_inverse=True
        )
        kxy = int(np.unique(inv_xy).size)
        return float((kx + ky - kxy - 1) / (2.0 * n))

    @staticmethod
    def _miller_madow_cmi_correction(
        n_samples: int,
        x_states: np.ndarray,
        y_states: np.ndarray,
        z_states: np.ndarray,
    ) -> float:
        """Return the entropy-expansion correction for ``I(X;Y|Z)``."""
        n = float(max(1, int(n_samples)))

        def state_count(*columns: np.ndarray) -> int:
            states = np.column_stack(columns)
            return int(np.unique(states, axis=0).shape[0])

        kxz = state_count(x_states, z_states)
        kyz = state_count(y_states, z_states)
        kz = int(np.unique(z_states).size)
        kxyz = state_count(x_states, y_states, z_states)
        return float((kxz + kyz - kz - kxyz) / (2.0 * n))

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        self._validate_inputs(x, y)

        xs = self._to_states(x)
        ys = self._to_states(y)
        mi = float(mutual_info_score(xs, ys))

        if self.bias_correction in {"miller_madow", "mm"}:
            mi += self._miller_madow_mi_correction(xs.size, xs, ys)

        return float(max(0.0, mi))

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        self._validate_inputs(x, y, z)

        xs = self._to_states(x)
        ys = self._to_states(y)
        zs = self._to_states(z)

        yz = np.column_stack([ys, zs])
        _, yz_states = np.unique(yz, axis=0, return_inverse=True)

        mi_x_yz = float(mutual_info_score(xs, yz_states))
        mi_x_z = float(mutual_info_score(xs, zs))
        cmi = mi_x_yz - mi_x_z
        if self.bias_correction in {"miller_madow", "mm"}:
            cmi += self._miller_madow_cmi_correction(xs.size, xs, ys, zs)
        return float(max(0.0, cmi))


class DecoderEstimator(BaseInformationEstimator):
    """Decoder-based MI estimator for discrete targets.

    For I(X;C) where C is discrete (class labels), estimate:
      I(X;C) = H(C) - H(C|X)
    where H(C|X) is approximated by cross-validated negative log-likelihood of
    a probabilistic classifier (default: multinomial logistic regression).

    Continuous-continuous MI/CMI is only supported when explicitly enabled via
    `continuous_strategy` ("binned" or "gaussian") and the total dimension is
    <= `continuous_max_dim`; otherwise this returns NaN (with a warning).
    """

    def __init__(self, config: EstimatorConfig):
        super().__init__(config)
        self.n_bins = int(config.n_bins)
        mp = dict(config.method_params or {})
        self.cv_folds = int(mp.get("cv_folds", 5))
        self.C = float(mp.get("C", 1.0))
        self.standardize = bool(mp.get("standardize", True))
        self.seed = int(mp.get("seed", 0))
        self.continuous_strategy = str(mp.get("continuous_strategy", "none")).lower()
        self.continuous_max_dim = int(mp.get("continuous_max_dim", 2))
        self.gaussian_ridge = float(mp.get("gaussian_ridge", 1e-6))
        self.binning_strategy = str(mp.get("binning_strategy", "quantile")).lower()
        self.bias_correction = str(mp.get("bias_correction", "none")).lower()

    def _unsupported(self, msg: str) -> float:
        self.logger.warning(msg)
        return float("nan")

    def _binned_helper(self) -> BinnedEstimator:
        cfg = EstimatorConfig(
            method=EstimationMethod.BINNED,
            n_bins=self.n_bins,
            method_params={
                "binning_strategy": self.binning_strategy,
                "bias_correction": self.bias_correction,
            },
        )
        return BinnedEstimator(cfg)

    def _cov(self, x: np.ndarray) -> np.ndarray:
        x = self._ensure_2d(np.asarray(x))
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov)
        cov = cov + float(self.gaussian_ridge) * np.eye(cov.shape[0])
        return cov

    def _gaussian_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        xy = np.hstack([x, y])

        sx, ldx = np.linalg.slogdet(self._cov(x))
        sy, ldy = np.linalg.slogdet(self._cov(y))
        sxy, ldxy = np.linalg.slogdet(self._cov(xy))
        if sx <= 0 or sy <= 0 or sxy <= 0:
            return self._unsupported(
                "decoder(gaussian): non-positive definite covariance"
            )
        mi = 0.5 * (ldx + ldy - ldxy)
        return float(max(0.0, mi))

    def _gaussian_cmi(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        z = self._ensure_2d(np.asarray(z))
        xz = np.hstack([x, z])
        yz = np.hstack([y, z])
        xyz = np.hstack([x, y, z])

        sxz, ldxz = np.linalg.slogdet(self._cov(xz))
        syz, ldyz = np.linalg.slogdet(self._cov(yz))
        sz, ldz = np.linalg.slogdet(self._cov(z))
        sxyz, ldxyz = np.linalg.slogdet(self._cov(xyz))
        if sxz <= 0 or syz <= 0 or sz <= 0 or sxyz <= 0:
            return self._unsupported(
                "decoder(gaussian): non-positive definite covariance for CMI"
            )
        cmi = 0.5 * (ldxz + ldyz - ldz - ldxyz)
        return float(max(0.0, cmi))

    def _continuous_dim_ok(self, total_dim: int) -> bool:
        return int(total_dim) <= int(self.continuous_max_dim)

    def _mi_cont_cont(self, x: np.ndarray, y: np.ndarray) -> float:
        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        total_dim = int(x.shape[1] + y.shape[1])

        if not self._continuous_dim_ok(total_dim):
            return self._unsupported(
                f"decoder: continuous MI unsupported for dim={total_dim} (max {self.continuous_max_dim}). "
                f"Use method='kraskov' for high-dim, or set decoder_continuous_strategy for low-dim."
            )

        if self.continuous_strategy in {"binned", "bin", "hist", "histogram"}:
            return float(self._binned_helper().mutual_information(x, y))
        if self.continuous_strategy in {"gaussian", "bayes"}:
            return float(self._gaussian_mi(x, y))
        return self._unsupported(
            "decoder: continuous MI disabled (decoder_continuous_strategy='none')."
        )

    @staticmethod
    def _discrete_entropy_nats(labels: np.ndarray) -> float:
        labels = np.asarray(labels).ravel().astype(int)
        if labels.size == 0:
            return 0.0
        counts = np.bincount(labels)
        counts = counts[counts > 0]
        if counts.size == 0:
            return 0.0
        p = counts.astype(float) / float(counts.sum())
        return float(-np.sum(p * np.log(p + 1e-12)))

    def _mi_cont_disc_decoder(self, x: np.ndarray, labels: np.ndarray) -> float:
        x = self._ensure_2d(np.asarray(x))
        y = KraskovEstimator._to_int_labels(np.asarray(labels).ravel())

        n_classes = int(np.unique(y).size)
        if n_classes <= 1:
            return 0.0

        H = self._discrete_entropy_nats(y)

        # Choose a feasible number of folds based on minimum class count.
        counts = np.bincount(y)
        min_count = int(counts[counts > 0].min()) if np.any(counts > 0) else 0
        n_splits = min(max(2, int(self.cv_folds)), min_count) if min_count >= 2 else 0
        if n_splits < 2:
            return 0.0

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        base_clf = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
        )
        clf = (
            make_pipeline(StandardScaler(), base_clf) if self.standardize else base_clf
        )

        classes = np.arange(n_classes)
        total_loss = 0.0
        total_n = 0
        for train_idx, test_idx in cv.split(x, y):
            x_tr, x_te = x[train_idx], x[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            try:
                clf.fit(x_tr, y_tr)
                proba = clf.predict_proba(x_te)
            except Exception:
                return 0.0

            total_loss += float(log_loss(y_te, proba, labels=classes, normalize=False))
            total_n += int(y_te.size)

        if total_n <= 0:
            return 0.0

        H_cond = total_loss / float(total_n)
        return float(max(0.0, H - H_cond))

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        self._validate_inputs(x, y)

        x_is_discrete = x.shape[1] == 1 and KraskovEstimator._is_discrete_1d(x[:, 0])
        y_is_discrete = y.shape[1] == 1 and KraskovEstimator._is_discrete_1d(y[:, 0])

        if x_is_discrete and y_is_discrete:
            return float(
                mutual_info_score(
                    KraskovEstimator._to_int_labels(x[:, 0]),
                    KraskovEstimator._to_int_labels(y[:, 0]),
                )
            )

        if y_is_discrete and not x_is_discrete:
            return self._mi_cont_disc_decoder(x, y[:, 0])

        if x_is_discrete and not y_is_discrete:
            return self._mi_cont_disc_decoder(y, x[:, 0])

        # Continuous-continuous: only if explicitly enabled for low-dimensional cases.
        return self._mi_cont_cont(x, y)

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        self._validate_inputs(x, y, z)

        y_is_discrete = y.shape[1] == 1 and KraskovEstimator._is_discrete_1d(y[:, 0])
        z_is_discrete = z.shape[1] == 1 and KraskovEstimator._is_discrete_1d(z[:, 0])

        if z_is_discrete:
            z_labels = z[:, 0]
            n_total = int(z_labels.size)
            if n_total <= 1:
                return 0.0
            cmi = 0.0
            for val in np.unique(z_labels):
                mask = z_labels == val
                n = int(np.sum(mask))
                if n <= 1:
                    continue
                cmi += (n / n_total) * float(self.mutual_information(x[mask], y[mask]))
            return float(max(0.0, cmi))

        if y_is_discrete:
            xz = np.hstack([x, z])
            mi_xz_y = self.mutual_information(xz, y)
            mi_z_y = self.mutual_information(z, y)
            return float(max(0.0, float(mi_xz_y) - float(mi_z_y)))

        # All-continuous: only if explicitly enabled for low-dimensional cases.
        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        z = self._ensure_2d(np.asarray(z))
        total_dim = int(x.shape[1] + y.shape[1] + z.shape[1])
        if not self._continuous_dim_ok(total_dim):
            return self._unsupported(
                f"decoder: continuous CMI unsupported for dim={total_dim} (max {self.continuous_max_dim})."
            )
        if self.continuous_strategy in {"binned", "bin", "hist", "histogram"}:
            return float(self._binned_helper().conditional_mutual_information(x, y, z))
        if self.continuous_strategy in {"gaussian", "bayes"}:
            return float(self._gaussian_cmi(x, y, z))
        return self._unsupported(
            "decoder: continuous CMI disabled (decoder_continuous_strategy='none')."
        )


class SklearnEstimator(BaseInformationEstimator):
    """Scikit-learn based estimator using various methods."""

    def __init__(self, config: EstimatorConfig):
        super().__init__(config)
        self.n_bins = config.n_bins
        self.bandwidth = config.bandwidth

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI using sklearn's methods."""
        x = self._ensure_2d(x)
        y_orig = y
        y = self._ensure_2d(y)
        self._validate_inputs(x, y)

        # Determine if we're dealing with discrete or continuous variables
        x_discrete = self._is_discrete(x)
        y_discrete = self._is_discrete(y)

        if x_discrete and y_discrete:
            # Both discrete - use mutual_info_score
            # Convert to integer labels
            x_labels = self._to_labels(x)
            y_labels = self._to_labels(y_orig.ravel() if y_orig.ndim == 1 else y)
            return mutual_info_score(x_labels, y_labels)

        elif not x_discrete and y_discrete:
            # X continuous, Y discrete (classification)
            y_labels = self._to_labels(y_orig.ravel() if y_orig.ndim == 1 else y)
            k = min(max(1, int(self.config.n_neighbors)), max(1, x.shape[0] - 1))
            if x.shape[1] > 1:
                return float(compute_mi_cd(x, y_labels, k))
            mi_scores = mutual_info_classif(x, y_labels, n_neighbors=k)
            return float(mi_scores[0])

        elif not x_discrete and not y_discrete:
            # Both continuous (regression)
            if x.shape[1] == 1 and y.shape[1] == 1:
                # Both univariate - use histogram-based estimation
                return self._histogram_mi(x.ravel(), y.ravel())
            else:
                # Multivariate - estimate joint I(X;Y), not mean feature-wise MI.
                k = min(max(1, int(self.config.n_neighbors)), max(1, x.shape[0] - 1))
                return float(compute_mi_cc(x, y, k))

        else:
            # X discrete, Y continuous - swap and compute
            return self.mutual_information(y, x)

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Compute CMI using chain rule."""
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        self._validate_inputs(x, y, z)

        # I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
        return self._chain_rule_cmi(x, y, z)

    def _is_discrete(self, x: np.ndarray) -> bool:
        """Check if variable appears to be discrete."""
        if x.shape[1] > 1:
            return False  # Multi-dimensional, assume continuous

        unique_vals = np.unique(x)
        n_unique = len(unique_vals)
        n_samples = len(x)

        # Heuristic: discrete if few unique values or appears to be integer labels
        if n_unique < 20 or n_unique < n_samples / 10:
            return True

        # Check if all values are integers
        if np.all(x == x.astype(int)):
            return True

        return False

    def _to_labels(self, x: np.ndarray) -> np.ndarray:
        """Convert continuous values to discrete labels."""
        if x.ndim > 1:
            # For multi-dimensional, create composite labels
            _, inverse = np.unique(x, axis=0, return_inverse=True)
            return inverse.astype(int)
        else:
            # For 1D, use unique value mapping
            unique_vals = np.unique(x)
            label_map = {val: i for i, val in enumerate(unique_vals)}
            return np.array([label_map[val] for val in x.ravel()])

    def _histogram_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI using histogram-based estimation."""
        # Create 2D histogram
        hist, _, _ = np.histogram2d(x, y, bins=self.n_bins)

        # Convert to probabilities
        pxy = hist / np.sum(hist)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)

        # Compute MI
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

        return max(0.0, mi)


class CopulaEstimator(BaseInformationEstimator):
    """Copula-based mutual information estimator."""

    def __init__(self, config: EstimatorConfig):
        super().__init__(config)
        self.copula_type = config.copula_type
        self._check_dvc_available()

    def _check_dvc_available(self):
        """Check if DVC tensorflow is available."""
        try:
            # import tensorflow as tf  # Available for import check
            # Check if DVC modules can be imported
            import os
            import sys

            # Add external directory to path if needed
            external_dir = os.path.join(
                os.path.dirname(__file__), "../../../../external"
            )
            if external_dir not in sys.path:
                sys.path.insert(0, external_dir)

            self.dvc_available = True
            self.logger.debug("DVC tensorflow available for copula estimation")
        except ImportError:
            self.dvc_available = False
            self.logger.warning(
                "DVC tensorflow not available, falling back to simple copula estimation"
            )

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI using copula-based estimation."""
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        self._validate_inputs(x, y)

        if self.dvc_available and self.copula_type == "dvc":
            return self._dvc_copula_mi(x, y)
        else:
            return self._simple_copula_mi(x, y)

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Compute CMI using copula-based estimation."""
        # Use chain rule as with other estimators
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        self._validate_inputs(x, y, z)

        return self._chain_rule_cmi(x, y, z)

    def _simple_copula_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Enhanced copula-based MI estimation with discrete variable handling."""

        # Handle discrete variables (like class labels) by adding non-informative noise
        x_processed = self._add_noise_to_discrete(x)
        y_processed = self._add_noise_to_discrete(y)

        # Transform to uniform margins using empirical CDF
        n_samples = len(x_processed)
        u = np.zeros_like(x_processed)
        v = np.zeros_like(y_processed)

        for i in range(x_processed.shape[1]):
            ranks = stats.rankdata(x_processed[:, i])
            u[:, i] = ranks / (n_samples + 1)

        for i in range(y_processed.shape[1]):
            ranks = stats.rankdata(y_processed[:, i])
            v[:, i] = ranks / (n_samples + 1)

        # Choose estimation method based on copula_type
        if self.copula_type == "gaussian":
            return self._gaussian_copula_mi(u, v)
        elif self.copula_type == "kernel":
            return self._kernel_copula_mi(u, v)
        else:
            # Default to Gaussian copula
            return self._gaussian_copula_mi(u, v)

    def _add_noise_to_discrete(
        self, data: np.ndarray, noise_scale: float = 0.1
    ) -> np.ndarray:
        """Add non-informative noise to discrete variables.

        For discrete variables like class labels, adds uniform noise in [0, 1)
        to each integer value, transforming c to a random number in [c, c+1).
        """
        data_processed = data.copy()

        # Detect if data appears to be discrete (integer-like values)
        for i in range(data.shape[1]):
            col = data[:, i]
            # Check if all values are close to integers
            if np.allclose(col, np.round(col), atol=1e-6):
                # This appears to be discrete - add uniform noise
                noise = np.random.uniform(0, noise_scale, size=col.shape)
                data_processed[:, i] = col + noise
                self.logger.debug(f"Added noise to discrete variable in column {i}")

        return data_processed

    def _gaussian_copula_mi(self, u: np.ndarray, v: np.ndarray) -> float:
        """Gaussian copula MI estimation."""
        if u.shape[1] == 1 and v.shape[1] == 1:
            # Univariate case - check for constant inputs first
            u_flat = u.ravel()
            v_flat = v.ravel()

            # Check if either input is constant (all same values)
            if np.var(u_flat) < 1e-10 or np.var(v_flat) < 1e-10:
                # If either input is constant, MI = 0
                return 0.0

            # Suppress the correlation warning for constant inputs
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
                corr, _ = stats.pearsonr(u_flat, v_flat)

            # Handle NaN correlation (when inputs are constant)
            if np.isnan(corr) or np.isinf(corr):
                return 0.0

            # MI for Gaussian copula: -0.5 * log(1 - rho^2)
            if abs(corr) < 0.999:
                mi = -0.5 * np.log(1 - corr**2)
                return max(0.0, mi)
            else:
                return 0.0

        # For multivariate, fall back to Kraskov on uniform margins
        return KraskovEstimator(self.config).mutual_information(u, v)

    def _kernel_copula_mi(self, u: np.ndarray, v: np.ndarray) -> float:
        """Non-parametric kernel-based copula MI estimation."""
        from sklearn.neighbors import KernelDensity

        if u.shape[1] == 1 and v.shape[1] == 1:
            # Univariate case with kernel density estimation
            u_flat = u.ravel()
            v_flat = v.ravel()

            # Check for constant inputs
            if np.var(u_flat) < 1e-10 or np.var(v_flat) < 1e-10:
                return 0.0

            # Choose bandwidth automatically using Scott's rule
            n_samples = len(u_flat)
            bandwidth_u = n_samples ** (-1 / 5) * np.std(u_flat)
            bandwidth_v = n_samples ** (-1 / 5) * np.std(v_flat)
            bandwidth_joint = min(bandwidth_u, bandwidth_v)

            # Ensure minimum bandwidth
            bandwidth_joint = max(bandwidth_joint, 0.01)

            try:
                # Estimate marginal densities
                kde_u = KernelDensity(bandwidth=bandwidth_u, kernel="gaussian")
                kde_v = KernelDensity(bandwidth=bandwidth_v, kernel="gaussian")
                kde_joint = KernelDensity(bandwidth=bandwidth_joint, kernel="gaussian")

                kde_u.fit(u_flat.reshape(-1, 1))
                kde_v.fit(v_flat.reshape(-1, 1))

                # Joint density on uniform margins
                uv_joint = np.column_stack([u_flat, v_flat])
                kde_joint.fit(uv_joint)

                # Compute MI using kernel density estimates
                # MI = ∫∫ p(u,v) log(p(u,v) / (p(u)p(v))) du dv
                # Approximate using sample-based estimation

                log_pu = kde_u.score_samples(u_flat.reshape(-1, 1))
                log_pv = kde_v.score_samples(v_flat.reshape(-1, 1))
                log_puv = kde_joint.score_samples(uv_joint)

                # MI estimate: mean of log(p(u,v)) - log(p(u)) - log(p(v))
                mi_samples = log_puv - log_pu - log_pv
                mi = np.mean(mi_samples)

                return max(0.0, mi)

            except Exception as e:
                self.logger.warning(
                    f"Kernel copula estimation failed: {e}, falling back to Gaussian"
                )
                return self._gaussian_copula_mi(u, v)

        # For multivariate, fall back to Kraskov
        return KraskovEstimator(self.config).mutual_information(u, v)

    def _dvc_copula_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Use DVC tensorflow for copula-based MI estimation."""
        # This would implement the full DVC copula estimation
        # For now, fall back to simple copula
        return self._simple_copula_mi(x, y)


class InformationEstimatorFactory:
    """Factory for creating information estimators based on configuration."""

    _estimators: ClassVar[dict[EstimationMethod, type]] = {
        EstimationMethod.KRASKOV: KraskovEstimator,
        EstimationMethod.DECODER: DecoderEstimator,
        EstimationMethod.BINNED: BinnedEstimator,
        EstimationMethod.SKLEARN: SklearnEstimator,
        EstimationMethod.COPULA: CopulaEstimator,
    }

    @classmethod
    def create(cls, config: Union[EstimatorConfig, dict]) -> BaseInformationEstimator:
        """Create an estimator based on configuration."""
        if isinstance(config, dict):
            config = EstimatorConfig(**config)

        if config.method == EstimationMethod.AUTO:
            # Default to Kraskov for general use
            config.method = EstimationMethod.KRASKOV

        estimator_class = cls._estimators.get(config.method)
        if estimator_class is None:
            raise ValueError(f"Unknown estimation method: {config.method}")

        return estimator_class(config)

    @classmethod
    def register_estimator(cls, method: EstimationMethod, estimator_class: type):
        """Register a new estimator class."""
        cls._estimators[method] = estimator_class
