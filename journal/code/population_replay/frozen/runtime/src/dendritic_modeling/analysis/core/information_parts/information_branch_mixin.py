"""Layer-branch information-analysis helpers."""

from typing import Any

import numpy as np
import torch
from tqdm import tqdm


class InformationBranchAnalysisMixin:
    """Compute branch-aggregated information metrics."""

    def _apply_layer_synapse_availability_in_place(
        self,
        layer_stats: dict[str, Any],
        layer_name: str,
    ) -> None:
        """Annotate layer statistics and remove synapse metrics when none exist."""
        if layer_name not in self.data_dict:
            return

        layer_stats["has_synapses"] = self.data_dict[layer_name].get(
            "has_synapses", True
        )
        if layer_stats["has_synapses"]:
            return

        for key in list(layer_stats.keys()):
            if any(
                metric_token in key
                for metric_token in [
                    "I(E;",
                    "I(I;",
                    "I(E_lin;",
                    "I(I_lin;",
                ]
            ):
                layer_stats.pop(key, None)

    def _aggregate_layer_branch_arrays_for_metrics(
        self,
        *,
        E_layer_data: torch.Tensor,
        I_layer_data: torch.Tensor,
        Vout_layer_data: torch.Tensor,
        Vb_layer_data: torch.Tensor | None,
        C: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Prepare one layer's branch arrays for layer-level information metrics."""
        C_for_layer = C

        if self.branch_aggregation == "multivariate":
            E_layer = E_layer_data.detach().cpu().numpy()
            I_layer = I_layer_data.detach().cpu().numpy()
            Vb_layer = (
                Vb_layer_data.detach().cpu().numpy()
                if Vb_layer_data is not None
                else None
            )
            Vout_layer = Vout_layer_data.detach().cpu().numpy()
        elif self.branch_aggregation == "sample":
            E_layer = E_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
            I_layer = I_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
            Vb_layer = (
                Vb_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
                if Vb_layer_data is not None
                else None
            )
            Vout_layer = Vout_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
            _n_samples, n_branches = E_layer_data.shape
            C_for_layer = np.repeat(C, n_branches, axis=0)
        else:
            E_layer = E_layer_data.mean(dim=1).detach().cpu().numpy()
            I_layer = I_layer_data.mean(dim=1).detach().cpu().numpy()
            if Vb_layer_data is not None:
                Vb_layer = Vb_layer_data.mean(dim=1).detach().cpu().numpy()
                Vb_layer = Vb_layer.reshape(-1, 1)
            else:
                Vb_layer = None
            Vout_layer = Vout_layer_data.mean(dim=1).detach().cpu().numpy()
            E_layer = E_layer.reshape(-1, 1)
            I_layer = I_layer.reshape(-1, 1)
            Vout_layer = Vout_layer.reshape(-1, 1)

        return E_layer, I_layer, Vout_layer, C_for_layer, Vb_layer

    def _try_add_label_shuffle_null_baselines_in_place(
        self,
        results: dict[str, Any],
        *,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vb: np.ndarray | None,
        Vinf: np.ndarray | None,
        S: np.ndarray | None = None,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
        seed_offset: int,
        failure_context: str,
    ) -> None:
        """Add label-shuffle null baselines when enabled, logging failures."""
        if self.mi_null_shuffles <= 0:
            return

        try:
            self._add_label_shuffle_null_baselines_in_place(
                results,
                E=E,
                I_var=I_var,
                Vout=Vout,
                C=C,
                Vb=Vb,
                Vinf=Vinf,
                S=S,
                has_exc_synapses=has_exc_synapses,
                has_inh_synapses=has_inh_synapses,
                has_branch_input=has_branch_input,
                seed_offset=seed_offset,
            )
        except Exception as e:
            self.logger.warning(
                f"Null (label-shuffle) baseline failed for {failure_context}: {e}"
            )

    def _compute_branch_metrics_with_null_baselines(
        self,
        *,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vb: np.ndarray | None,
        S: np.ndarray | None = None,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
        seed_offset: int,
        failure_context: str,
    ) -> dict[str, Any]:
        """Compute branch metrics and attach optional label-shuffle baselines."""
        branch_results = self.compute_metrics(E, I_var, Vout, C, None, Vb, S)
        self._try_add_label_shuffle_null_baselines_in_place(
            branch_results,
            E=E,
            I_var=I_var,
            Vout=Vout,
            C=C,
            Vb=Vb,
            Vinf=None,
            S=S,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input and (Vb is not None),
            seed_offset=seed_offset,
            failure_context=failure_context,
        )
        return branch_results

    def _compute_layer_branch_result(
        self,
        *,
        layer_name: str,
        layer_data: dict[str, Any],
        C: np.ndarray,
        seed_offset: int,
    ) -> dict[str, Any]:
        """Compute layer-level metrics after aggregating one layer's branches."""
        E_layer_data_raw = layer_data["excitation"]
        I_layer_data_raw = layer_data["inhibition"]
        Vb_layer_data_raw = layer_data.get("branch_input")
        Vout_layer_data = layer_data["output"]

        has_exc_synapses = bool(layer_data.get("has_exc_synapses", True))
        has_inh_synapses = bool(layer_data.get("has_inh_synapses", True))
        has_branch_input = Vb_layer_data_raw is not None

        E_layer_data, I_layer_data = self.ensure_inhibitory_data(
            E_layer_data_raw, I_layer_data_raw
        )

        # Keep Vb as None when it does not exist, e.g. in the outermost distal layer.
        Vb_layer_data = Vb_layer_data_raw if has_branch_input else None
        E_layer, I_layer, Vout_layer, C_for_layer, Vb_layer = (
            self._aggregate_layer_branch_arrays_for_metrics(
                E_layer_data=E_layer_data,
                I_layer_data=I_layer_data,
                Vout_layer_data=Vout_layer_data,
                Vb_layer_data=Vb_layer_data,
                C=C,
            )
        )

        layer_results = self.compute_metrics(
            E_layer, I_layer, Vout_layer, C_for_layer, None, Vb_layer
        )
        self._filter_result_metrics_in_place(
            layer_results,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input,
        )
        self._try_add_label_shuffle_null_baselines_in_place(
            layer_results,
            E=E_layer,
            I_var=I_layer,
            Vout=Vout_layer,
            C=C_for_layer,
            Vb=Vb_layer,
            Vinf=None,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input and (Vb_layer is not None),
            seed_offset=seed_offset,
            failure_context=f"layer {layer_name}",
        )
        return layer_results

    def _compute_layer_branch_results(self, C: np.ndarray) -> dict[str, Any] | None:
        """Compute information metrics after aggregating branches within each layer."""
        self.logger.info("Computing MI at layer level...")

        all_results = []
        layer_names = []

        layers_to_process = [
            (name, data)
            for name, data in self.data_dict.items()
            if "excitation" in data
        ]

        for layer_name, layer_data in tqdm(
            layers_to_process,
            desc="Processing layers",
            leave=False,
            ncols=100,
        ):
            layer_results = self._compute_layer_branch_result(
                layer_name=layer_name,
                layer_data=layer_data,
                C=C,
                seed_offset=len(all_results),
            )
            all_results.append(layer_results)
            layer_names.append(layer_name)

        return self._finalize_layer_branch_results(
            all_results=all_results,
            layer_names=layer_names,
        )

    def _finalize_layer_branch_results(
        self,
        *,
        all_results: list[dict[str, Any]],
        layer_names: list[str],
    ) -> dict[str, Any] | None:
        """Average layer metrics and attach layer-branch analysis metadata."""
        if not all_results:
            self.logger.warning("No layer results to average")
            return None

        results = self._average_mi_results(all_results)
        results["layer_names"] = layer_names
        results["num_layers"] = len(all_results)
        results["computation_level"] = "layer_branch"

        layer_statistics = {}
        for layer_name, layer_result in zip(layer_names, all_results):
            layer_stats = self._compute_layer_statistics([layer_result])
            self._apply_layer_synapse_availability_in_place(layer_stats, layer_name)
            layer_statistics[layer_name] = layer_stats

        results["layer_statistics"] = layer_statistics
        layer_depths = self.collected_soma_relative_depths()
        if layer_depths:
            results["layer_soma_relative_depths"] = layer_depths
        self.logger.info(
            f"Generated layer statistics for {len(layer_statistics)} layers for plotting"
        )
        return results


__all__ = ["InformationBranchAnalysisMixin"]
