"""Result summarization helpers for information analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from dendritic_modeling.analysis.core.information_parts.information_helpers import (
    convert_information_units_in_place,
    discrete_entropy_nats,
    extract_metric_tokens,
    is_discrete_class_labels,
    keep_metric_key_given_availability,
    metric_targets_class,
    rename_metric_key_tokens,
    should_convert_information_key_to_bits,
)
from dendritic_modeling.analysis.core.information_parts.information_metric_merge import (
    merge_average_renamed_numeric_metrics_in_place,
    merge_renamed_metric_dicts_in_place,
)
from dendritic_modeling.analysis.core.information_parts.information_null_baseline_mixin import (
    InformationNullBaselineMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_results import (
    average_mi_results,
    compute_layer_statistics,
    compute_variance_analysis,
    filter_result_metrics_in_place,
)
from dendritic_modeling.analysis.core.information_parts.information_summary import (
    format_information_summary,
)


class InformationResultHelpersMixin(InformationNullBaselineMixin):
    """Formats, filters, and summarizes information-analysis results."""

    def get_summary(self, results: dict[str, dict[str, dict | float]]) -> str:
        """Generate a summary string of the results."""
        return format_information_summary(results)

    def _average_mi_results(
        self, results_list: list[dict[str, dict[str, dict | float]]]
    ) -> dict[str, Any]:
        """Average MI results across multiple branches."""
        return average_mi_results(results_list, method=self.method)

    @staticmethod
    def _is_discrete_class_labels(labels: np.ndarray) -> bool:
        """Heuristic: decide whether labels are discrete class IDs."""
        return is_discrete_class_labels(labels)

    @staticmethod
    def _discrete_entropy_nats(labels: np.ndarray) -> float:
        """Discrete entropy H(C) in natural units (nats)."""
        return discrete_entropy_nats(labels)

    @staticmethod
    def _metric_targets_class(metric_key: str) -> bool:
        """Return True if the metric is of the form I(X;C) or I(X;C|...)."""
        return metric_targets_class(metric_key)

    def _compute_layer_statistics(
        self,
        results_list: list[dict[str, Any]],
        *,
        entropy_C: float | None = None,
        topk: int = 10,
    ) -> dict[str, float]:
        """Compute per-layer summary stats from per-branch MI results."""
        return compute_layer_statistics(
            results_list,
            entropy_C=entropy_C,
            topk=topk,
        )

    @staticmethod
    def _extract_metric_tokens(metric_key: str) -> list[str]:
        """Extract variable tokens from metric keys like 'I(E;Vb|C)'.

        Returns an empty list for non-metric keys.
        """
        return extract_metric_tokens(metric_key)

    @staticmethod
    def _rename_metric_key_tokens(metric_key: str, token_map: dict[str, str]) -> str:
        """Rename variable tokens inside MI keys.

        This is used to generate LDA/shuffled metric names (e.g. E -> E_lin) while
        preserving separators, and it works for tokens in both the main and
        conditioning parts (after '|').
        """
        return rename_metric_key_tokens(metric_key, token_map)

    def _merge_renamed_metric_dicts_in_place(
        self,
        *,
        target: dict[str, Any],
        source: dict[str, Any],
        token_map: dict[str, str],
    ) -> None:
        """Merge nested metric dictionaries after renaming variable tokens."""
        merge_renamed_metric_dicts_in_place(
            target=target,
            source=source,
            token_map=token_map,
            rename_metric_key_tokens=self._rename_metric_key_tokens,
        )

    def _merge_average_renamed_numeric_metrics_in_place(
        self,
        *,
        target: dict[str, Any],
        sources: list[dict[str, Any]],
        token_map: dict[str, str],
    ) -> None:
        """Average numeric nested metric values, rename them, and merge into target."""
        merge_average_renamed_numeric_metrics_in_place(
            target=target,
            sources=sources,
            token_map=token_map,
            rename_metric_key_tokens=self._rename_metric_key_tokens,
        )

    def _keep_metric_key_given_availability(
        self,
        metric_key: str,
        *,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
    ) -> bool:
        """Return True if the metric is meaningful given available variables.

        We drop any metric that references:
        - E / E_* when there are no excitatory synapses
        - I / I_* when there are no inhibitory synapses
        - Vb / Vb_* when branch input does not exist (e.g., outermost distal layer)
        """
        return keep_metric_key_given_availability(
            metric_key,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input,
        )

    def _filter_result_metrics_in_place(
        self,
        result: dict[str, Any],
        *,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
    ) -> None:
        """Remove metrics that depend on non-existent variables (in-place)."""
        filter_result_metrics_in_place(
            result,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input,
        )

    @staticmethod
    def _should_convert_information_key_to_bits(key: Any) -> bool:
        """Heuristic: return True if `key` is an MI/CMI-like metric name in nats.

        We convert keys that contain an MI-style token (e.g. "I(E;C)") and are
        *not* clearly dimensionless or in non-MI units.
        """
        return should_convert_information_key_to_bits(key)

    def _convert_information_units_in_place(self, obj: Any) -> None:
        """Convert MI/CMI-like quantities in-place from nats -> bits.

        Estimators operate in nats internally. Many plots and summaries are in bits,
        so we convert any MI/CMI-like metrics (including *_std and *_null variants)
        before saving/plotting when `compute.output_units="bits"`.
        """
        convert_information_units_in_place(
            obj,
            output_units=getattr(self, "output_units", "bits"),
        )

    def _compute_variance_analysis(self, results: dict) -> dict:
        """Compute variance bookkeeping from information-analysis summaries."""
        return compute_variance_analysis(results)


__all__ = ["InformationResultHelpersMixin"]
