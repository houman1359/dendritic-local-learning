"""
Noise perturbation analysis module.

This module contains classes for evaluating model robustness under various noise conditions
including uniform and Gaussian noise perturbations.
"""

import math
from collections.abc import Callable, Mapping
from typing import Optional

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    evaluation_kwargs_from_runtime,
    materialize_dataset,
)
from dendritic_modeling.config import NoisePerturbationAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.datasets.augmentation.noise import (
    ClampRange,
    add_gaussian_noise,
    add_uniform_noise,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
)
from dendritic_modeling.utils import save_dict


class NoisePerturbationAnalyzer(AbstractAnalyzer):
    """
    Analyzer for evaluating model robustness under noise perturbations.

    Supports uniform and Gaussian noise at various magnitudes, evaluating
    accuracy and categorical log-likelihood on perturbed test data.
    """

    def __init__(self, params: NoisePerturbationAnalysisParams):
        """
        Initialize the noise perturbation analyzer.

        Args:
            uniform_noise: Whether to evaluate uniform noise robustness
            uniform_magnitudes: list of uniform noise magnitudes to test
            gaussian_noise: Whether to evaluate Gaussian noise robustness
            gaussian_sdevs: list of Gaussian noise standard deviations to test
            n_samples: Number of noise samples to generate per test point
            accuracy: Whether to compute accuracy on noisy data
            categorical_loglikelihood: Whether to compute categorical log-likelihood on noisy data
        """
        super().__init__("NoisePerturbationAnalyzer")
        self.uniform_noise = getattr(params, "uniform_noise", False)
        self.uniform_magnitudes = getattr(params, "uniform_magnitudes", [0.1, 0.2])
        self.gaussian_noise = getattr(params, "gaussian_noise", False)
        self.gaussian_sdevs = getattr(params, "gaussian_sdevs", [0.1, 0.2])
        self.n_samples = getattr(params, "n_samples", 10)
        self.accuracy = getattr(params, "accuracy", False)
        self.auc = getattr(params, "auc", False)
        self.categorical_ll = getattr(params, "categorical_loglikelihood", False)
        self.mse = getattr(params, "mse", False)
        self.cosine_similarity = getattr(params, "cosine_similarity", False)
        self.seed = int(getattr(params, "seed", 0) or 0)
        self.clamp_range = self._normalize_clamp_range(
            getattr(params, "clamp_range", (0.0, 1.0))
        )

    @staticmethod
    def _normalize_clamp_range(value) -> ClampRange:
        if value is None:
            return None
        if (
            isinstance(value, (str, bytes))
            or isinstance(value, Mapping)
            or not hasattr(value, "__len__")
            or not hasattr(value, "__getitem__")
        ):
            raise ValueError(
                "noise_perturbation.params.clamp_range must be null or a "
                "[min, max] pair."
            )
        if len(value) != 2:
            raise ValueError(
                "noise_perturbation.params.clamp_range must be null or a "
                "[min, max] pair."
            )
        try:
            lo, hi = float(value[0]), float(value[1])
        except (TypeError, ValueError, KeyError, IndexError):
            raise ValueError(
                "noise_perturbation.params.clamp_range must be null or a "
                "[min, max] pair."
            ) from None
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(
                "noise_perturbation.params.clamp_range values must be finite."
            )
        if lo > hi:
            raise ValueError(
                "noise_perturbation.params.clamp_range must satisfy min <= max."
            )
        return (lo, hi)

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ):
        """
        Analyze model robustness under noise perturbations.

        Args:
            model: The model to evaluate
            test_dataset: Test dataset to perturb with noise
            device: Device to run evaluation on
            save_path: Path to save results (optional)
            filename: Filename for saved results

        Returns:
            Dictionary containing noise robustness results if save_path is None, otherwise None
        """
        eval_kwargs = evaluation_kwargs_from_runtime(runtime)
        with analysis_device_context(model, device) as analysis_device:
            items = materialize_dataset(test_dataset, runtime, device=analysis_device)
            x: torch.Tensor = items[0].to(analysis_device)
            y: torch.Tensor = items[1].to(analysis_device)
            y = y[None, ...].expand(self.n_samples, *y.shape)
            generator = torch.Generator(device=analysis_device)
            generator.manual_seed(self.seed)

            noise_results = {}

            def _evaluate_metric(
                noise_type: str,
                noise_level: float,
                metric_func: Callable,
                metric_key: str,
                noisy_test_ds: torch.utils.data.Dataset,
            ):
                metric_result = metric_func(
                    model,
                    test_ds=noisy_test_ds,
                    reduce_dim=1,
                    move_device=False,
                    **eval_kwargs,
                )[-1]

                if isinstance(metric_result, torch.Tensor):
                    metric_list = metric_result.cpu().tolist()
                else:
                    metric_list = float(metric_result)
                noise_results[noise_type][f"{noise_level}"][metric_key] = metric_list

            if self.uniform_noise:
                noise_results["uniform"] = {}
                for mag in self.uniform_magnitudes:
                    noise_results["uniform"][f"{mag}"] = {}
                    noisy_x = add_uniform_noise(
                        x,
                        spread=mag,
                        n_samples=self.n_samples,
                        generator=generator,
                        clamp_range=self.clamp_range,
                    )
                    noisy_test_ds = TensorDataset(noisy_x, y)
                    if self.accuracy:
                        _evaluate_metric(
                            noise_type="uniform",
                            noise_level=mag,
                            metric_func=evaluate_accuracy,
                            metric_key="accuracy",
                            noisy_test_ds=noisy_test_ds,
                        )
                    if self.auc:
                        _evaluate_metric(
                            noise_type="uniform",
                            noise_level=mag,
                            metric_func=evaluate_auc,
                            metric_key="auc",
                            noisy_test_ds=noisy_test_ds,
                        )
                    if self.categorical_ll:
                        _evaluate_metric(
                            noise_type="uniform",
                            noise_level=mag,
                            metric_func=evaluate_categorical_loglikelihood,
                            metric_key="categorical_ll",
                            noisy_test_ds=noisy_test_ds,
                        )
                    if self.mse:
                        _evaluate_metric(
                            noise_type="uniform",
                            noise_level=mag,
                            metric_func=evaluate_mse,
                            metric_key="mse",
                            noisy_test_ds=noisy_test_ds,
                        )
                    if self.cosine_similarity:
                        _evaluate_metric(
                            noise_type="uniform",
                            noise_level=mag,
                            metric_func=evaluate_cosine_similarity,
                            metric_key="cosine_similarity",
                            noisy_test_ds=noisy_test_ds,
                        )

            if self.gaussian_noise:
                noise_results["gaussian"] = {}
                for sdev in self.gaussian_sdevs:
                    noise_results["gaussian"][f"{sdev}"] = {}
                    noisy_x = add_gaussian_noise(
                        x,
                        std=sdev,
                        n_samples=self.n_samples,
                        generator=generator,
                        clamp_range=self.clamp_range,
                    )
                    noisy_test_ds = TensorDataset(noisy_x, y)
                    if self.accuracy:
                        _evaluate_metric(
                            noise_type="gaussian",
                            noise_level=sdev,
                            metric_func=evaluate_accuracy,
                            metric_key="accuracy",
                            noisy_test_ds=noisy_test_ds,
                        )
                    if self.auc:
                        _evaluate_metric(
                            noise_type="gaussian",
                            noise_level=sdev,
                            metric_func=evaluate_auc,
                            metric_key="auc",
                            noisy_test_ds=noisy_test_ds,
                        )
                    if self.categorical_ll:
                        _evaluate_metric(
                            noise_type="gaussian",
                            noise_level=sdev,
                            metric_func=evaluate_categorical_loglikelihood,
                            metric_key="categorical_ll",
                            noisy_test_ds=noisy_test_ds,
                        )
                    if self.mse:
                        _evaluate_metric(
                            noise_type="gaussian",
                            noise_level=sdev,
                            metric_func=evaluate_mse,
                            metric_key="mse",
                            noisy_test_ds=noisy_test_ds,
                        )
                    if self.cosine_similarity:
                        _evaluate_metric(
                            noise_type="gaussian",
                            noise_level=sdev,
                            metric_func=evaluate_cosine_similarity,
                            metric_key="cosine_similarity",
                            noisy_test_ds=noisy_test_ds,
                        )

        if save_path is not None:
            save_dict(noise_results, save_path, filename)
        else:
            return noise_results
