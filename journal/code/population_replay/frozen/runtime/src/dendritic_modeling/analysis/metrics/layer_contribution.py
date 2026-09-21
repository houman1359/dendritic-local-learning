"""
Layer contribution analysis module.

This module contains classes for analyzing the contribution of individual layers to overall
model performance by testing each layer in isolation and measuring its individual impact.
"""

import logging
import os
from collections.abc import Callable
from copy import deepcopy
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    evaluation_kwargs_from_runtime,
    subset_dataset_for_runtime,
)
from dendritic_modeling.config import SingleLayerContributionAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
)
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import iter_named_modules_of_type


class SingleLayerContributionAnalyzer(AbstractAnalyzer):
    """
    Analyzes the contribution of individual layers by isolating them.
    This is the renamed version of "layer_ablation" - it removes all other layers
    to test what each layer can do on its own.
    """

    def __init__(self, params: SingleLayerContributionAnalysisParams):
        super().__init__("SingleLayerContributionAnalyzer")
        self.excitation_contribution = getattr(params, "excitation_contribution", False)
        self.inhibition_contribution = getattr(params, "inhibition_contribution", False)
        self.both_contribution = getattr(params, "both_contribution", False)
        self.accuracy = getattr(params, "accuracy", False)
        self.auc = getattr(params, "auc", False)
        self.categorical_ll = getattr(params, "categorical_loglikelihood", False)
        self.mse = getattr(params, "mse", False)
        self.cosine_similarity = getattr(params, "cosine_similarity", False)
        self.logger = logging.getLogger(__name__)

    def evaluate_layer_contribution(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        module: DendriticBranchLayer,
        module_name: str,
        contribution_type: str,
        results_dict: dict,
        state_dict: dict,
        eval_kwargs: Optional[dict] = None,
    ):
        if eval_kwargs is None:
            eval_kwargs = {}
        results_dict[contribution_type][module_name] = {}
        results_dict[contribution_type][module_name]["depth"] = module.layer_idx

        for name, mod in iter_named_modules_of_type(model, DendriticBranchLayer):
            if (
                name != module_name
                and mod.branch_excitation is not None
                and mod.branch_inhibition is not None
            ):
                mod.branch_excitation.pre_w.data.fill_(-1000)
                mod.branch_inhibition.pre_w.data.fill_(-1000)

        if contribution_type == "excitation":
            module.branch_inhibition.pre_w.data.fill_(-1000)
        if contribution_type == "inhibition":
            module.branch_excitation.pre_w.data.fill_(-1000)

        def _evaluate_metric(metric_func: Callable, metric_key: str):
            metric_score = metric_func(
                model, test_ds=test_dataset, move_device=False, **eval_kwargs
            )[-1]
            results_dict[contribution_type][module_name][metric_key] = metric_score

        if self.accuracy:
            _evaluate_metric(metric_func=evaluate_accuracy, metric_key="accuracy")
        if self.auc:
            _evaluate_metric(metric_func=evaluate_auc, metric_key="auc")
        if self.categorical_ll:
            _evaluate_metric(
                metric_func=evaluate_categorical_loglikelihood,
                metric_key="log likelihood",
            )
        if self.mse:
            _evaluate_metric(metric_func=evaluate_mse, metric_key="mse")
        if self.cosine_similarity:
            _evaluate_metric(
                metric_func=evaluate_cosine_similarity, metric_key="cosine similarity"
            )

        model.load_state_dict(state_dict)

        return results_dict

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cuda",
        save_path: Optional[str] = None,
        filename: str = "final",
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ):
        """Analyze each layer's contribution in isolation"""
        self.logger.info("Starting single layer contribution analysis...")

        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        eval_kwargs = evaluation_kwargs_from_runtime(runtime)
        test_dataset = subset_dataset_for_runtime(test_dataset, runtime)
        with analysis_device_context(model, device):
            state_dict = deepcopy(model.state_dict())

            results = {}
            if self.excitation_contribution:
                results["excitation"] = {}
            if self.inhibition_contribution:
                results["inhibition"] = {}
            if self.both_contribution:
                results["both"] = {}

            for name, module in iter_named_modules_of_type(model, DendriticBranchLayer):
                if (
                    module.branch_excitation is not None
                    and module.branch_inhibition is not None
                ):
                    if self.both_contribution:
                        results = self.evaluate_layer_contribution(
                            model=model,
                            test_dataset=test_dataset,
                            module=module,
                            module_name=name,
                            contribution_type="both",
                            results_dict=results,
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )

                    if self.excitation_contribution:
                        results = self.evaluate_layer_contribution(
                            model=model,
                            test_dataset=test_dataset,
                            module=module,
                            module_name=name,
                            contribution_type="excitation",
                            results_dict=results,
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )

                    if self.inhibition_contribution:
                        results = self.evaluate_layer_contribution(
                            model=model,
                            test_dataset=test_dataset,
                            module=module,
                            module_name=name,
                            contribution_type="inhibition",
                            results_dict=results,
                            state_dict=state_dict,
                            eval_kwargs=eval_kwargs,
                        )

        if save_path is not None:
            save_dict(results, save_path, filename)
            plot_layer_contribution(
                results=results, save_path=save_path, filename=filename
            )
        else:
            return results


def plot_layer_contribution(
    results: dict, save_path: Optional[str] = None, filename: str = "final"
) -> Optional[list[plt.Figure]]:
    """
    Plot layer ablation results showing accuracy/log-likelihood decrease vs depth.

    Args:
        layer_results: Dictionary containing ablation results by type and layer
        save_path: Path to save plots. If None, returns list of figures
        filename: Base filename for saved plots

    Returns:
        List of figures if save_path is None, otherwise None
    """
    if save_path is None:
        fig_list = []

    metric_mapping = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "log likelihood": "Log Likelihood",
        "mse": "MSE",
        "cosine similarity": "Cosine Similarity",
    }

    def has_key_nested(dictionary, key):
        """
        Check if key exists at any level in a nested dictionary
        """
        if isinstance(dictionary, dict):
            if key in dictionary:
                return True
            return any(has_key_nested(value, key) for value in dictionary.values())
        elif isinstance(dictionary, (list, tuple)):
            return any(has_key_nested(item, key) for item in dictionary)
        return False

    metrics = []
    for metric in metric_mapping.keys():
        if has_key_nested(results, metric):
            metrics.append(metric)

    synapse_ablation_types = []
    if "excitation" in results:
        synapse_ablation_types.append("excitation")
    if "inhibition" in results:
        synapse_ablation_types.append("inhibition")
    if "both" in results:
        synapse_ablation_types.append("both")

    for metric in metrics:
        res = {}
        for ablation_type in synapse_ablation_types:
            data_dict = {"depth": [], "performance": []}
            type_dict: dict = results[ablation_type]
            for layer_dict in type_dict.values():
                data_dict["depth"].append(layer_dict["depth"])
                data_dict["performance"].append(layer_dict[metric])
            res[ablation_type] = data_dict

        fig, ax = plt.subplots(figsize=(10, 8))
        n_ablation_types = len(synapse_ablation_types)
        depths = res[synapse_ablation_types[0]]["depth"]
        width = 0.25
        for i, ablation_type in enumerate(synapse_ablation_types):
            x = np.array(depths) + width * (i - (n_ablation_types - 1) / 2)
            ax.bar(
                x, res[ablation_type]["performance"], width=width, label=ablation_type
            )
        # Create depth labels - correct soma/distal ordering
        # Lower depth values = closer to soma, higher = more distal
        depth_labels = []
        for depth in depths:
            if depth == min(depths):
                depth_labels.append("Soma")
            else:
                # Calculate distal layer number based on distance from soma
                distal_layer = depth - min(depths)
                depth_labels.append(f"Distal {distal_layer}")

        ax.set_xticks(depths)
        ax.set_xticklabels(depth_labels, rotation=45)
        ax.set_xlabel("Branch Depth (Soma to Distal)")
        ax.set_ylabel(f"{metric_mapping[metric]}")
        ax.set_title(f"{metric_mapping[metric]} vs. Branch Depth")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

        if save_path is not None:
            plot_path = os.path.join(save_path, f"{filename}_{metric}.png")
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            fig_list.append(fig)

    if save_path is None:
        return fig_list
