"""Result finalization and persistence helpers for information analysis."""

import os
import time
from typing import Any

import torch

from dendritic_modeling.analysis.core.information_parts.information_results import (
    LAYER_INFORMATION_PROXY_SEMANTICS,
)
from dendritic_modeling.analysis.utils.dendritic_depth import (
    SOMA_RELATIVE_DEPTH_REFERENCE,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils import save_dict


class InformationFinalizationMixin:
    """Finalize information-analysis results and optional saved artifacts."""

    def _complete_analysis_results(
        self,
        *,
        results: dict[str, Any],
        model: BaseModel,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        analysis_start_time: float,
        device: str,
        save_path: str | None,
        filename: str,
    ) -> dict[str, Any]:
        """Add metadata, finalize units/plots, save outputs, and return results."""
        self._add_analysis_metadata(results)

        total_analysis_time = time.time() - analysis_start_time
        self._log_analysis_summary(results, total_analysis_time)

        somatic_synapses = self._finalize_analysis_payload(
            results=results,
            model=model,
            inputs=inputs,
            labels=labels,
            device=device,
            save_path=save_path,
        )
        self._save_analysis_outputs(
            results=results,
            save_path=save_path,
            filename=filename,
            somatic_synapses=somatic_synapses,
        )
        return results

    def _add_analysis_metadata(self, results: dict[str, Any]) -> None:
        """Attach analyzer configuration metadata to result dictionaries."""
        results["per_neuron_analysis_enabled"] = self.per_neuron_analysis
        results["per_einet_analysis_enabled"] = self.per_einet_analysis
        results["per_layer_analysis_enabled"] = self.per_layer_analysis
        results["analysis_split"] = self.analysis_split
        results["has_inhibitory_data"] = self.has_inhibitory_data(self.data_dict)
        results["dendritic_depth_reference"] = SOMA_RELATIVE_DEPTH_REFERENCE
        results["signal_semantics"] = {
            "E": "pre_gate_excitation_current",
            "I": "pre_gate_inhibition_current",
            "Vb": "pre_gate_upstream_current",
            "V": "pre_gate_voltage",
            "Vout": "post_gate_output",
            "S": "post_gate_parent_soma_output",
        }
        results["layer_information_proxy_semantics"] = dict(
            LAYER_INFORMATION_PROXY_SEMANTICS
        )

    def _log_analysis_summary(
        self, results: dict[str, Any], total_analysis_time: float
    ) -> None:
        """Log a concise summary of completed information analysis."""
        self.logger.info("Information analysis completed successfully!")
        self.logger.info("Analysis summary:")
        self.logger.info(f"   - Total time: {total_analysis_time:.1f}s")
        self.logger.info(f"   - Computation level: {self.computation_level}")
        self.logger.info(f"   - Method: {self.method}")
        self.logger.info(f"   - Sample size: {results.get('n_samples', 'unknown')}")
        if "num_branches" in results:
            self.logger.info(f"   - Branches processed: {results['num_branches']}")
        if "num_layers" in results:
            self.logger.info(f"   - Layers processed: {results['num_layers']}")

    def _finalize_analysis_payload(
        self,
        *,
        results: dict[str, Any],
        model: BaseModel,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        device: str,
        save_path: str | None,
    ) -> bool:
        """Finalize in-memory analysis results before optional persistence."""
        somatic_synapses = self._add_synthetic_soma_entry_if_needed(results, model)
        self._finalize_information_units_and_variance(results)
        self._maybe_add_lda_vs_trained_performance(
            results=results,
            model=model,
            inputs=inputs,
            labels=labels,
            device=device,
            save_path=save_path,
        )
        return somatic_synapses

    def _finalize_information_units_and_variance(self, results: dict[str, Any]) -> None:
        results["information_units"] = getattr(self, "output_units", "nats")
        if results["information_units"] == "bits":
            self._convert_information_units_in_place(results)

        self.logger.info("Computing variance analysis for plotting...")
        variance_analysis = self._compute_variance_analysis(results)
        results["variance_analysis"] = variance_analysis
        self.logger.info(
            f"Variance analysis computed with {len(variance_analysis.get('layer_variances', {}))} layers"
        )

    def _maybe_add_lda_vs_trained_performance(
        self,
        *,
        results: dict[str, Any],
        model: BaseModel,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        device: str,
        save_path: str | None,
    ) -> None:
        if not (self.compute_lda_weights and save_path is not None):
            return

        try:
            lda_comparison = self.compare_lda_vs_trained_performance(
                model, inputs, labels, device
            )
            results["lda_vs_trained_performance"] = lda_comparison

            self._plot_lda_vs_trained_performance_if_available(
                lda_comparison=lda_comparison,
                save_path=save_path,
            )
        except Exception as e:
            self._log_exception_with_traceback(
                "Error computing LDA vs trained performance",
                e,
            )

    def _plot_lda_vs_trained_performance_if_available(
        self,
        *,
        lda_comparison: dict[str, Any],
        save_path: str,
    ) -> None:
        """Plot LDA comparison results when the comparison includes an accuracy."""
        if lda_comparison.get("lda_accuracy") is None:
            self.logger.warning("LDA accuracy not available, skipping performance plot")
            return

        from dendritic_modeling.plotting.visualizations.information_plots import (
            plot_lda_vs_trained_performance,
        )

        plot_lda_vs_trained_performance(lda_comparison, save_path, self.logger)
        self.logger.info("LDA vs trained performance comparison completed!")

    def _save_analysis_outputs(
        self,
        *,
        results: dict[str, Any],
        save_path: str | None,
        filename: str,
        somatic_synapses: bool,
    ) -> None:
        if save_path is None:
            return

        self.logger.info(f"Saving results to {save_path}/{filename}")
        save_dict(results, save_path, filename)

        self._write_analysis_summary_file(
            results=results, save_path=save_path, filename=filename
        )
        self._run_post_save_visualizations(
            results=results,
            save_path=save_path,
            filename=filename,
            somatic_synapses=somatic_synapses,
        )
        self.logger.info("Results saved successfully!")

    def _write_analysis_summary_file(
        self,
        *,
        results: dict[str, Any],
        save_path: str,
        filename: str,
    ) -> None:
        """Write the text summary next to serialized analysis results."""
        summary = self.get_summary(results)
        summary_path = os.path.join(save_path, f"{filename}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary)

    def _run_post_save_visualizations(
        self,
        *,
        results: dict[str, Any],
        save_path: str,
        filename: str,
        somatic_synapses: bool,
    ) -> None:
        """Generate optional post-save summaries and plots."""
        try:
            if "per_layer_analysis" in results:
                self._save_per_layer_ei_summaries(
                    results["per_layer_analysis"], save_path, filename
                )
            self._generate_information_plots(results, save_path, somatic_synapses)
        except Exception as plot_err:
            self.logger.warning(f"Post-save visualization failed: {plot_err}")


__all__ = ["InformationFinalizationMixin"]
