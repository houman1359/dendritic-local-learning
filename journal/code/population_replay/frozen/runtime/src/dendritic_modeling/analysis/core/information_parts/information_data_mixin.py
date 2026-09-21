"""Dataset materialization and activation collection for information analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dendritic_modeling.analysis.utils.runtime import materialize_dataset
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils.hooks import run_with_forward_hooks


@dataclass(frozen=True)
class _InformationAnalysisData:
    """Materialized inputs and labels used by information analysis."""

    inputs: torch.Tensor
    labels: torch.Tensor
    C: np.ndarray
    entropy_C: float | None


class InformationDataMixin:
    """Collect dataset tensors and recorded layer activations."""

    def _materialize_analysis_dataset(
        self,
        *,
        test_dataset: torch.utils.data.Dataset,
        device: str,
        runtime: EvaluationRuntimeConfig | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize the configured analysis dataset on the analysis device."""
        self.logger.info("Loading test dataset...")
        items = materialize_dataset(
            test_dataset,
            runtime,
            explicit_max_samples=self.max_samples,
            device=device,
        )
        inputs: torch.Tensor = items[0].to(device)
        labels: torch.Tensor = items[1].to(device)
        self.logger.info(
            f"Loaded dataset: {len(inputs)} samples, input shape: {inputs.shape}"
        )
        return inputs, labels

    def _collect_forward_hook_activations(
        self,
        *,
        model: BaseModel,
        inputs: torch.Tensor,
    ) -> None:
        """Run a no-grad forward pass while collecting attached hook payloads."""

        def _forward_pass() -> None:
            with torch.no_grad():
                _ = model(inputs)

        run_with_forward_hooks(
            attach=lambda: self.attach_forward_hooks(model),
            remove=self.remove_forward_hooks,
            body=_forward_pass,
        )

    def _labels_to_class_array(
        self,
        labels: torch.Tensor,
    ) -> tuple[np.ndarray, float | None]:
        """Return labels as a NumPy class array plus optional discrete entropy."""
        C = labels.detach().cpu().numpy()
        entropy_C: float | None
        if self._is_discrete_class_labels(C):
            entropy_C = self._discrete_entropy_nats(C)
        else:
            entropy_C = None

        return C, entropy_C

    def _collect_analysis_data(
        self,
        *,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str,
        runtime: EvaluationRuntimeConfig | None,
    ) -> _InformationAnalysisData:
        """Materialize data and collect branch activations through forward hooks."""
        inputs, labels = self._materialize_analysis_dataset(
            test_dataset=test_dataset,
            device=device,
            runtime=runtime,
        )
        self._collect_forward_hook_activations(model=model, inputs=inputs)
        self.attach_parent_soma_outputs_to_branch_records()
        C, entropy_C = self._labels_to_class_array(labels)
        self.logger.debug(
            f"Data collection completed. Found {len(self.data_dict)} layers."
        )

        return _InformationAnalysisData(
            inputs=inputs,
            labels=labels,
            C=C,
            entropy_C=entropy_C,
        )


__all__ = ["InformationDataMixin", "_InformationAnalysisData"]
