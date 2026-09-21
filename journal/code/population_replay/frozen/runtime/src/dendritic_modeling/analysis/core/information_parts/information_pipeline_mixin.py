"""Analysis pipeline orchestration for information analysis."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel


class InformationPipelineMixin:
    """Runs data collection, metric dispatch, and finalization."""

    def _run_analysis_pipeline(
        self,
        *,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str,
        save_path: str | None,
        filename: str,
        runtime: EvaluationRuntimeConfig | None,
        analysis_start_time: float,
    ) -> dict[str, Any] | None:
        """Run data collection, metric dispatch, enhanced outputs, and finalization."""
        analysis_data = self._collect_analysis_data(
            model=model,
            test_dataset=test_dataset,
            device=device,
            runtime=runtime,
        )
        inputs = analysis_data.inputs
        labels = analysis_data.labels
        C = analysis_data.C
        entropy_C = analysis_data.entropy_C

        results = self._compute_results_for_level(
            model=model,
            inputs=inputs,
            C=C,
            entropy_C=entropy_C,
        )
        if results is None:
            return None

        self._add_enhanced_analysis_results(results, C)
        return self._complete_analysis_results(
            results=results,
            model=model,
            inputs=inputs,
            labels=labels,
            analysis_start_time=analysis_start_time,
            device=device,
            save_path=save_path,
            filename=filename,
        )


__all__ = ["InformationPipelineMixin"]
