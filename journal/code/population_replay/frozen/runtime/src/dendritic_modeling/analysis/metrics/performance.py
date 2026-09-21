"""
Performance metrics analysis module.

This module contains classes for evaluating model performance including
accuracy, categorical log-likelihood, and mean squared error.

Supports both in-memory (materialize) and streaming (DataLoader) evaluation,
selected automatically based on dataset size.
"""

import logging
import os
from collections.abc import Callable
from typing import Optional

# Disable Dynamo to fix fake tensor device issues during analysis
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import evaluation_kwargs_from_runtime
from dendritic_modeling.config import PerformanceAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
)
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


class PerformanceAnalyzer(AbstractAnalyzer):
    """
    Analyzer for evaluating model performance on various metrics.

    Supports accuracy, categorical log-likelihood, and MSE evaluation
    on train, validation, and test datasets.

    Uses streaming evaluation for large datasets (e.g., ImageNet) and
    in-memory evaluation for small datasets (e.g., CIFAR-10).
    """

    def __init__(self, params: PerformanceAnalysisParams):
        super().__init__("PerformanceAnalyzer")
        self.accuracy = getattr(params, "accuracy", False)
        self.auc = getattr(params, "auc", False)
        self.categorical_loglikelihood = getattr(
            params, "categorical_loglikelihood", False
        )
        self.mse = getattr(params, "mse", False)
        self.cosine_similarity = getattr(params, "cosine_similarity", False)

    def analyze(
        self,
        model: BaseModel,
        train_ds: Optional[torch.utils.data.Dataset] = None,
        valid_ds: Optional[torch.utils.data.Dataset] = None,
        test_ds: Optional[torch.utils.data.Dataset] = None,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
        runtime: Optional[EvaluationRuntimeConfig] = None,
        splits: Optional[list[str]] = None,
    ):
        """
        Analyze model performance on specified datasets.

        Args:
            model: The model to evaluate
            train_ds: Training dataset (optional)
            valid_ds: Validation dataset (optional)
            test_ds: Test dataset (optional)
            device: Device to run evaluation on
            save_path: Path to save results (optional)
            filename: Filename for saved results
            training: Whether this is being called during training
            runtime: Shared analysis runtime config for dataset iteration.
            splits: Dataset splits to evaluate. Defaults to all available splits.

        Returns:
            Dictionary containing results if save_path is None, otherwise None
        """
        if runtime is None:
            runtime = EvaluationRuntimeConfig()

        split_map = {"train": train_ds, "valid": valid_ds, "test": test_ds}
        if splits is None:
            splits = ["train", "valid", "test"]
        active_splits = {k: v for k, v in split_map.items() if k in splits}
        active_train = active_splits.get("train")
        active_valid = active_splits.get("valid")
        active_test = active_splits.get("test")

        eval_kwargs = evaluation_kwargs_from_runtime(runtime)

        results = {}

        def _evaluate_metric(metric_func: Callable, metric_key: str):
            results[metric_key] = {}
            train_metric, valid_metric, test_metric = metric_func(
                model,
                train_ds=active_train,
                valid_ds=active_valid,
                test_ds=active_test,
                move_device=True,
                device=device,
                **eval_kwargs,
            )
            if active_train is not None and train_metric is not None:
                results[metric_key]["train"] = train_metric
            if active_valid is not None and valid_metric is not None:
                results[metric_key]["valid"] = valid_metric
            if active_test is not None and test_metric is not None:
                results[metric_key]["test"] = test_metric

        if self.accuracy:
            _evaluate_metric(metric_func=evaluate_accuracy, metric_key="accuracy")

        if self.auc:
            _evaluate_metric(metric_func=evaluate_auc, metric_key="auc")

        if self.categorical_loglikelihood:
            _evaluate_metric(
                metric_func=evaluate_categorical_loglikelihood,
                metric_key="categorical_loglikelihood",
            )

        if self.mse:
            _evaluate_metric(metric_func=evaluate_mse, metric_key="mse")

        if self.cosine_similarity:
            _evaluate_metric(
                metric_func=evaluate_cosine_similarity, metric_key="cosine_similarity"
            )

        if save_path is not None:
            if not filename.endswith(".json"):
                filename = f"{filename}.json"
            save_dict(results, save_path, filename)
        else:
            return results
