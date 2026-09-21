"""Enhanced E/I and per-neuron information-analysis summaries."""

from __future__ import annotations

import os

import numpy as np
import torch

from dendritic_modeling.utils import save_dict

_EI_SUMMARY_SPECS = (
    ("excitatory_results", "excitatory", "Excitatory Network Summary", True),
    ("inhibitory_results", "inhibitory", "Inhibitory Network Summary", False),
    ("combined_results", "combined", "Combined E+I Network Summary", False),
)


class InformationSignalSummaryMixin:
    """Compute per-neuron, per-layer, and E/I-specific information summaries."""

    def ensure_inhibitory_data(
        self, excitation_data: torch.Tensor, inhibition_data: torch.Tensor
    ):
        """Ensure inhibitory data is available, replacing with zeros if missing.

        Args:
            excitation_data: Shape [n_samples, n_branches] - excitatory activations
            inhibition_data: Shape [n_samples, n_branches] - inhibitory activations (can be None)

        Returns:
            Tuple of (excitation_data, inhibition_data) with guaranteed inhibitory data
        """
        # Handle case where inhibitory data might be missing/None
        if inhibition_data is None:
            inhibition_data = torch.zeros_like(excitation_data)
            self.logger.warning(
                "Inhibitory data is missing - using zeros for inhibitory activations"
            )

        return excitation_data, inhibition_data

    def has_inhibitory_data(self, layer_data_dict: dict) -> bool:
        """Check if the network has valid inhibitory data.

        Args:
            layer_data_dict: Dictionary containing layer data from all layers

        Returns:
            True if inhibitory data exists and is not all zeros
        """
        for layer_data in layer_data_dict.values():
            if "inhibition" in layer_data and layer_data["inhibition"] is not None:
                # Check if inhibitory data is not all zeros
                if torch.any(layer_data["inhibition"] != 0):
                    return True
        return False

    def compute_per_neuron_mi(
        self,
        E_data: np.ndarray,
        I_data: np.ndarray,
        Vout_data: np.ndarray,
        C: np.ndarray,
    ):
        """Compute MI for each neuron separately.

        Args:
            E_data: Shape [n_samples, n_neurons] - excitatory activations
            I_data: Shape [n_samples, n_neurons] - inhibitory activations
            Vout_data: Shape [n_samples, n_neurons] - output activations
            C: Shape [n_samples,] - class labels

        Returns:
            Dictionary with per-neuron MI results
        """
        _, n_neurons = E_data.shape
        per_neuron_results = []

        self.logger.info(f"Computing MI for {n_neurons} neurons individually...")

        for neuron_idx in range(n_neurons):
            # Extract data for single neuron
            E_neuron = E_data[:, neuron_idx : neuron_idx + 1]  # Keep 2D shape
            I_neuron = I_data[:, neuron_idx : neuron_idx + 1]
            Vout_neuron = Vout_data[:, neuron_idx : neuron_idx + 1]

            # Compute MI for this neuron
            neuron_results = self.compute_metrics(
                E_neuron, I_neuron, Vout_neuron, C, None, None
            )
            neuron_results["neuron_index"] = neuron_idx
            per_neuron_results.append(neuron_results)

        # Also compute average across neurons
        averaged_results = self._average_mi_results(per_neuron_results)

        return {
            "per_neuron_results": per_neuron_results,
            "averaged_results": averaged_results,
            "num_neurons": n_neurons,
        }

    def compute_ei_specific_mi(
        self,
        E_data: np.ndarray,
        I_data: np.ndarray,
        Vout_data: np.ndarray,
        C: np.ndarray,
    ):
        """Compute MI separately for excitatory and inhibitory networks.

        Args:
            E_data: Shape [n_samples, n_neurons] - excitatory activations
            I_data: Shape [n_samples, n_neurons] - inhibitory activations
            Vout_data: Shape [n_samples, n_neurons] - output activations
            C: Shape [n_samples,] - class labels

        Returns:
            Dictionary with separate E and I network MI results
        """
        self.logger.info(
            "Computing MI separately for excitatory and inhibitory networks..."
        )

        results = {}

        # Always compute excitatory network MI
        e_results = self.compute_metrics(
            E_data, np.zeros_like(I_data), Vout_data, C, None, None
        )
        e_results["network_type"] = "excitatory"
        results["excitatory_results"] = e_results

        # Check if inhibitory data is meaningful (not all zeros)
        has_inhibitory = np.any(I_data != 0)

        if has_inhibitory:
            # Compute MI for inhibitory network only
            i_results = self.compute_metrics(
                np.zeros_like(E_data), I_data, Vout_data, C, None, None
            )
            i_results["network_type"] = "inhibitory"
            results["inhibitory_results"] = i_results

            # Compute MI for combined E+I network (current default)
            combined_results = self.compute_metrics(
                E_data, I_data, Vout_data, C, None, None
            )
            combined_results["network_type"] = "combined"
            results["combined_results"] = combined_results

            self.logger.info(
                "Computed MI for excitatory, inhibitory, and combined networks"
            )
        else:
            # No inhibitory data available
            self.logger.warning(
                "No inhibitory data detected - skipping inhibitory and combined network analysis"
            )
            results["inhibitory_results"] = None
            results["combined_results"] = (
                e_results  # Use excitatory results as combined
            )
            results["inhibitory_data_available"] = False

        results["inhibitory_data_available"] = has_inhibitory
        return results

    def compute_per_layer_enhanced_analysis(self, C: np.ndarray):
        """Compute enhanced analysis (per-neuron and EI-specific) for each layer separately.

        Args:
            C: Class labels for all samples

        Returns:
            Dictionary with per-layer enhanced analysis results
        """
        self.logger.info("Computing per-layer enhanced analysis...")

        per_layer_results = {}

        for layer_name, layer_data in self.data_dict.items():
            if "excitation" not in layer_data:
                continue

            self.logger.info(f"Processing enhanced analysis for layer: {layer_name}")

            # Ensure inhibitory data is available for this layer
            E_filtered, I_filtered = self.ensure_inhibitory_data(
                layer_data["excitation"], layer_data["inhibition"]
            )

            E_layer = E_filtered.detach().cpu().numpy()
            I_layer = I_filtered.detach().cpu().numpy()
            Vout_layer = layer_data["output"].detach().cpu().numpy()

            layer_results = {}

            # Add per-neuron analysis for this layer
            if self.per_neuron_analysis:
                per_neuron_results = self.compute_per_neuron_mi(
                    E_layer, I_layer, Vout_layer, C
                )
                layer_results["per_neuron_analysis"] = per_neuron_results

            # Add EI-network analysis for this layer
            if self.per_einet_analysis:
                ei_specific_results = self.compute_ei_specific_mi(
                    E_layer, I_layer, Vout_layer, C
                )
                layer_results["per_einet_analysis"] = ei_specific_results

            # Add layer metadata
            layer_results["layer_name"] = layer_name
            layer_results["layer_shape"] = {
                "excitation": E_layer.shape,
                "inhibition": I_layer.shape,
                "output": Vout_layer.shape,
            }
            layer_results["has_inhibitory_data"] = np.any(I_layer != 0)

            per_layer_results[layer_name] = layer_results

        return per_layer_results

    def _save_per_layer_ei_summaries(
        self, per_layer_data: dict, save_path: str, base_filename: str
    ):
        """Generate and save individual summaries for each layer's E and I networks.

        Args:
            per_layer_data: Dictionary containing per-layer analysis results
            save_path: Directory to save summary files
            base_filename: Base filename for the summaries
        """
        self.logger.info("Generating individual layer E/I network summaries...")

        # Create subfolder for per-layer results
        per_layer_path = os.path.join(save_path, "per_layer_results")
        os.makedirs(per_layer_path, exist_ok=True)

        for layer_name, layer_results in per_layer_data.items():
            if "per_einet_analysis" in layer_results:
                ei_analysis = layer_results["per_einet_analysis"]

                # Clean layer name for filename
                clean_layer_name = layer_name.replace(".", "_").replace("/", "_")

                for result_key, filename_suffix, title, allow_none in _EI_SUMMARY_SPECS:
                    if not _should_save_ei_summary(
                        ei_analysis,
                        result_key=result_key,
                        allow_none=allow_none,
                    ):
                        continue

                    self._save_ei_summary_artifact(
                        results=ei_analysis[result_key],
                        per_layer_path=per_layer_path,
                        base_filename=base_filename,
                        clean_layer_name=clean_layer_name,
                        layer_name=layer_name,
                        filename_suffix=filename_suffix,
                        title=title,
                    )

        self.logger.info(
            f"Generated individual summaries for {len(per_layer_data)} layers"
        )

    def _save_ei_summary_artifact(
        self,
        *,
        results: dict,
        per_layer_path: str,
        base_filename: str,
        clean_layer_name: str,
        layer_name: str,
        filename_suffix: str,
        title: str,
    ) -> None:
        """Write one per-layer E/I summary text file and matching JSON payload."""
        summary = self.get_summary(results)
        summary_path = os.path.join(
            per_layer_path,
            f"{base_filename}_{clean_layer_name}_{filename_suffix}_summary.txt",
        )
        with open(summary_path, "w") as f:
            f.write(f"{title} - {layer_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(summary)

        save_dict(
            results,
            per_layer_path,
            f"{base_filename}_{clean_layer_name}_{filename_suffix}_results",
        )


def _should_save_ei_summary(
    ei_analysis: dict,
    *,
    result_key: str,
    allow_none: bool,
) -> bool:
    if allow_none:
        return result_key in ei_analysis
    return ei_analysis.get(result_key) is not None


__all__ = ["InformationSignalSummaryMixin"]
