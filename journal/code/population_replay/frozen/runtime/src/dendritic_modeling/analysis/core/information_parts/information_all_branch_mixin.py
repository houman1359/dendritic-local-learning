"""Full-network all-branch information-analysis helpers."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from dendritic_modeling.models import BaseModel


@dataclass(frozen=True)
class _AllBranchMetricData:
    """Prepared full-network arrays for all-branch information metrics."""

    excitation: np.ndarray
    inhibition: np.ndarray
    output: np.ndarray
    labels: np.ndarray
    branch_input: np.ndarray | None


class InformationAllBranchMixin:
    """Compute full-network branch aggregation information metrics."""

    def _aggregate_all_branch_layer_tensors(
        self,
        layer_data: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
        """Prepare one layer's branch tensors for full-network aggregation."""
        missing_vb = False

        if self.branch_aggregation == "mean":
            layer_E = layer_data["excitation"].mean(dim=1, keepdim=True)
            layer_I = layer_data["inhibition"].mean(dim=1, keepdim=True)
            layer_Vb_raw = layer_data.get("branch_input")
            if layer_Vb_raw is not None:
                layer_Vb = layer_Vb_raw.mean(dim=1, keepdim=True)
            else:
                layer_Vb = None
                missing_vb = True
            layer_Vout = layer_data["output"].mean(dim=1, keepdim=True)
        elif self.branch_aggregation == "multivariate":
            layer_E = layer_data["excitation"]
            layer_I = layer_data["inhibition"]
            layer_Vb_raw = layer_data.get("branch_input")
            if layer_Vb_raw is not None:
                layer_Vb = layer_Vb_raw
            else:
                layer_Vb = None
                missing_vb = True
            layer_Vout = layer_data["output"]
        elif self.branch_aggregation == "sample":
            n_samples, n_branches = layer_data["excitation"].shape
            layer_E = layer_data["excitation"].view(n_samples * n_branches, 1)
            layer_I = layer_data["inhibition"].view(n_samples * n_branches, 1)
            layer_Vb_raw = layer_data.get("branch_input")
            if layer_Vb_raw is not None:
                layer_Vb = layer_Vb_raw.view(n_samples * n_branches, 1)
            else:
                layer_Vb = None
                missing_vb = True
            layer_Vout = layer_data["output"].view(n_samples * n_branches, 1)
        else:
            layer_E = layer_data["excitation"].mean(dim=1, keepdim=True)
            layer_I = layer_data["inhibition"].mean(dim=1, keepdim=True)
            layer_Vb_raw = layer_data.get("branch_input")
            if layer_Vb_raw is not None:
                layer_Vb = layer_Vb_raw.mean(dim=1, keepdim=True)
            else:
                layer_Vb = None
                missing_vb = True
            layer_Vout = layer_data["output"].mean(dim=1, keepdim=True)

        return layer_E, layer_I, layer_Vout, layer_Vb, missing_vb

    def _concatenate_all_branch_tensors_for_metrics(
        self,
        *,
        layer_excitation: list[torch.Tensor],
        layer_inhibition: list[torch.Tensor],
        layer_output: list[torch.Tensor],
        layer_branch_input: list[torch.Tensor],
        missing_vb_in_any_layer: bool,
        C: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Concatenate per-layer tensors for full-network information metrics."""
        C_for_metrics = C
        if self.branch_aggregation == "sample":
            E = torch.cat(layer_excitation, dim=0).detach().cpu().numpy()
            I_var = torch.cat(layer_inhibition, dim=0).detach().cpu().numpy()
            Vb = (
                torch.cat(layer_branch_input, dim=0).detach().cpu().numpy()
                if (layer_branch_input and not missing_vb_in_any_layer)
                else None
            )
            Vout = torch.cat(layer_output, dim=0).detach().cpu().numpy()
            total_samples = E.shape[0]
            original_samples = C.shape[0]
            expansion_factor = total_samples // original_samples
            C_for_metrics = np.repeat(C, expansion_factor, axis=0)
        else:
            E = torch.cat(layer_excitation, dim=1).detach().cpu().numpy()
            I_var = torch.cat(layer_inhibition, dim=1).detach().cpu().numpy()
            Vb = (
                torch.cat(layer_branch_input, dim=1).detach().cpu().numpy()
                if layer_branch_input
                else None
            )
            Vout = torch.cat(layer_output, dim=1).detach().cpu().numpy()

        return E, I_var, Vout, C_for_metrics, Vb

    def _compute_final_output_signal(
        self,
        *,
        model: BaseModel,
        inputs: torch.Tensor,
    ) -> np.ndarray | None:
        """Run the model and return final output only when the model exposes final_fc."""
        with torch.no_grad():
            final_output: torch.Tensor = model(inputs)
            if hasattr(model, "final_fc"):
                return final_output.detach().cpu().numpy()
        return None

    def _collect_all_branch_metric_data(
        self,
        C: np.ndarray,
    ) -> _AllBranchMetricData | None:
        """Collect and concatenate full-network branch tensors for metric computation."""
        layer_excitation = []
        layer_inhibition = []
        layer_branch_input = []
        missing_vb_in_any_layer = False
        layer_output = []

        for layer_data in self.data_dict.values():
            if "excitation" not in layer_data:
                continue

            layer_E, layer_I, layer_Vout, layer_Vb, missing_vb = (
                self._aggregate_all_branch_layer_tensors(layer_data)
            )
            missing_vb_in_any_layer = missing_vb_in_any_layer or missing_vb

            layer_excitation.append(layer_E)
            layer_inhibition.append(layer_I)
            if layer_Vb is not None:
                layer_branch_input.append(layer_Vb)
            layer_output.append(layer_Vout)

        if not layer_excitation:
            self.logger.warning(
                "No excitatory activations collected. Skipping information analysis."
            )
            return None

        E, I_var, Vout, C_for_metrics, Vb = (
            self._concatenate_all_branch_tensors_for_metrics(
                layer_excitation=layer_excitation,
                layer_inhibition=layer_inhibition,
                layer_output=layer_output,
                layer_branch_input=layer_branch_input,
                missing_vb_in_any_layer=missing_vb_in_any_layer,
                C=C,
            )
        )

        self.logger.info(
            "Aggregated data shapes - "
            f"E: {E.shape}, I: {I_var.shape}, Vb: {None if Vb is None else Vb.shape}, Vout: {Vout.shape}, C: {C_for_metrics.shape}"
        )

        return _AllBranchMetricData(
            excitation=E,
            inhibition=I_var,
            output=Vout,
            labels=C_for_metrics,
            branch_input=Vb,
        )

    def _finalize_all_branch_results(
        self,
        *,
        metric_data: _AllBranchMetricData,
        Vinf: np.ndarray | None,
    ) -> dict[str, Any]:
        """Compute all-branch metrics and attach aggregation metadata."""
        results = self.compute_metrics(
            metric_data.excitation,
            metric_data.inhibition,
            metric_data.output,
            metric_data.labels,
            Vinf,
            metric_data.branch_input,
        )
        self._try_add_label_shuffle_null_baselines_in_place(
            results,
            E=metric_data.excitation,
            I_var=metric_data.inhibition,
            Vout=metric_data.output,
            C=metric_data.labels,
            Vb=metric_data.branch_input,
            Vinf=Vinf,
            has_exc_synapses=True,
            has_inh_synapses=True,
            has_branch_input=metric_data.branch_input is not None,
            seed_offset=999_999,
            failure_context="all-branch analysis",
        )
        results["aggregation_method"] = self.branch_aggregation
        results["computation_level"] = "all_branch"
        layer_depths = self.collected_soma_relative_depths()
        if layer_depths:
            results["layer_soma_relative_depths"] = layer_depths
        return results

    def _compute_all_branch_results(
        self,
        *,
        model: BaseModel,
        inputs: torch.Tensor,
        C: np.ndarray,
    ) -> dict[str, Any] | None:
        """Compute full-network information metrics across all collected branches."""
        self.logger.info(
            f"Aggregating branches using method: {self.branch_aggregation} (full network aggregation)"
        )

        metric_data = self._collect_all_branch_metric_data(C)
        if metric_data is None:
            return None

        Vinf = self._compute_final_output_signal(model=model, inputs=inputs)

        return self._finalize_all_branch_results(
            metric_data=metric_data,
            Vinf=Vinf,
        )


__all__ = ["InformationAllBranchMixin", "_AllBranchMetricData"]
