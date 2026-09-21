"""Enhanced information-analysis result helpers."""

from typing import Any

import numpy as np


class InformationEnhancedAnalysisMixin:
    """Add optional per-layer, per-neuron, and E/I-specific analyses."""

    def _collect_enhanced_analysis_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Collect concatenated E/I/Vout arrays for enhanced analysis modes."""
        all_E_data = []
        all_I_data = []
        all_Vout_data = []

        for layer_data in self.data_dict.values():
            if "excitation" in layer_data:
                E_filtered, I_filtered = self.ensure_inhibitory_data(
                    layer_data["excitation"], layer_data["inhibition"]
                )
                all_E_data.append(E_filtered.detach().cpu().numpy())
                all_I_data.append(I_filtered.detach().cpu().numpy())
                all_Vout_data.append(layer_data["output"].detach().cpu().numpy())

        if not all_E_data:
            return None

        E_all = np.concatenate(all_E_data, axis=1)
        I_all = np.concatenate(all_I_data, axis=1)
        Vout_all = np.concatenate(all_Vout_data, axis=1)
        return E_all, I_all, Vout_all

    def _add_enhanced_analysis_results(
        self, results: dict[str, Any], C: np.ndarray
    ) -> None:
        """Add optional per-layer, per-neuron, and EI-specific analysis results."""
        if getattr(self, "network_population_information", False):
            network_layer_results = self._compute_network_layer_information(C)
            if network_layer_results:
                results["network_layer_information"] = network_layer_results

        if not self._enhanced_analysis_requested():
            return

        self.logger.info("Running enhanced analysis modes...")

        if self.per_layer_analysis:
            per_layer_results = self.compute_per_layer_enhanced_analysis(C)
            results["per_layer_analysis"] = per_layer_results

        if not self._enhanced_array_analysis_requested():
            return

        enhanced_arrays = self._collect_enhanced_analysis_arrays()
        if enhanced_arrays is None:
            return

        E_all, I_all, Vout_all = enhanced_arrays

        if self.per_neuron_analysis:
            per_neuron_results = self.compute_per_neuron_mi(E_all, I_all, Vout_all, C)
            results["per_neuron_analysis"] = per_neuron_results

        if self.per_einet_analysis:
            ei_specific_results = self.compute_ei_specific_mi(E_all, I_all, Vout_all, C)
            results["per_einet_analysis"] = ei_specific_results

    def _compute_network_layer_information(
        self, C: np.ndarray
    ) -> dict[str, dict[str, Any]]:
        """Compute class information from each population-network layer readout."""
        layer_results: dict[str, dict[str, Any]] = {}
        labels = np.asarray(C).reshape(-1)
        for module_name, captured in getattr(self, "network_layer_data", {}).items():
            readout = captured.get("post_gate_readout_output")
            if readout is None or int(readout.shape[0]) != int(labels.shape[0]):
                continue

            payload: dict[str, Any] = {
                key: captured[key]
                for key in (
                    "network_layer_index",
                    "network_layer_name",
                    "readout_population",
                    "forward_call_count",
                    "capture_reduction",
                )
                if key in captured
            }
            try:
                payload["readout_class_information"] = {
                    "I(post_gate_readout;C)": float(
                        self.compute_class_mi(readout.detach().cpu().numpy(), labels)
                    )
                }
                population_information = {}
                for population_name, soma_output in captured.get(
                    "post_gate_soma_outputs", {}
                ).items():
                    if int(soma_output.shape[0]) != int(labels.shape[0]):
                        continue
                    population_information[population_name] = {
                        "I(post_gate_soma;C)": float(
                            self.compute_class_mi(
                                soma_output.detach().cpu().numpy(), labels
                            )
                        )
                    }
                payload["population_soma_class_information"] = population_information
            except Exception as exc:
                self.logger.warning(
                    "Network-layer information failed for %s: %s",
                    module_name,
                    exc,
                )
                continue
            layer_results[module_name] = payload
        return layer_results

    def _enhanced_analysis_requested(self) -> bool:
        return (
            self.per_neuron_analysis
            or self.per_einet_analysis
            or self.per_layer_analysis
        )

    def _enhanced_array_analysis_requested(self) -> bool:
        return self.per_neuron_analysis or self.per_einet_analysis


__all__ = ["InformationEnhancedAnalysisMixin"]
