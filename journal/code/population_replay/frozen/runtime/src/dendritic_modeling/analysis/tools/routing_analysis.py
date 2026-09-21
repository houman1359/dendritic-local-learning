"""
RoutingAnalyzer: per-level routing analysis for recurrent dendritic models.

Temporarily enables store_routing on the model, runs forward passes to collect
per-timestep routing info, then resets store_routing. Computes:
- Per-level raw-L2 and dimension-normalized RMS activity shares
- FF/REC decomposition (gamma_l): fraction of signal from feedforward compartments
- Temporal dynamics of these readouts within trials
"""

import logging
from typing import Any, Optional

import torch
import torch.utils.data

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.recurrent_introspection import (
    is_recurrent_core,
    iter_routing_entries,
)
from dendritic_modeling.analysis.utils.runtime import analysis_device_context
from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils.general import save_dict

logger = logging.getLogger(__name__)


def _extract_seq_lengths(batch) -> Optional[torch.Tensor]:
    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
        return batch[2]
    return None


def _valid_timestep_mask(
    seq_lengths: Optional[torch.Tensor],
    timestep_idx: int,
) -> tuple[Optional[torch.Tensor], bool]:
    if seq_lengths is None:
        return None, True

    valid_mask = (seq_lengths > timestep_idx).detach().cpu()
    return valid_mask, bool(valid_mask.any())


def _append_masked_routing_values(
    accumulator: dict[str, list[torch.Tensor]],
    key: str,
    values: torch.Tensor,
    valid_mask: Optional[torch.Tensor],
) -> None:
    values = values.detach().cpu()
    if valid_mask is not None:
        values = values[valid_mask]
    if values.numel() == 0:
        return
    accumulator.setdefault(key, []).append(values)


class RoutingAnalyzer(AbstractAnalyzer):
    """Analyzes per-level routing in recurrent dendritic E-I networks.

    Sets model.core_network.store_routing = True before evaluation passes,
    collects _routing_info from forward passes, then resets store_routing.

    Routing info format per timestep per layer:
        {
            "layer_0": {
                "excitatory": {
                    "level_contributions": Tensor[batch, n_levels],
                    "ff_rec_ratios": Tensor[batch, n_levels],
                },
                "inhibitory": {  # if present
                    "level_contributions": Tensor[batch, n_levels_i],
                    "ff_rec_ratios": Tensor[batch, n_levels_i],
                },
            }
        }
    """

    def __init__(self, params: Optional[Any] = None):
        super().__init__(logger_name="RoutingAnalyzer")
        self.params = params

    def analyze(
        self,
        model: BaseModel,
        data: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "routing",
        batch_size: int = 128,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform routing analysis on a recurrent dendritic model.

        Args:
            model: Model with a recurrent core network (e.g. EINetwork).
            data: Dataset returning (input_seq, target) tuples.
            device: Device for computation.
            save_path: Optional path to save results.
            filename: Filename for saved results.
            batch_size: Batch size for forward passes.

        Returns:
            Dictionary with per-level routing statistics, or empty dict if
            model is not a recurrent dendritic model.
        """
        self.log_analysis_start("routing_analysis")

        core = model.core_network
        if not is_recurrent_core(core):
            self.logger.info("Skipping routing analysis: core_network is not recurrent")
            return {}
        if not hasattr(core, "store_routing") or not hasattr(core, "_routing_info"):
            self.logger.info(
                "Skipping routing analysis: core_network has no routing support"
            )
            return {}

        original_store_routing = core.store_routing
        core.store_routing = True

        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

        try:
            level_contribs, level_activity_rms, ff_rec_ratios = (
                self._collect_routing_values(
                    model,
                    core,
                    loader,
                    device,
                )
            )
        finally:
            core.store_routing = original_store_routing

        if not level_contribs:
            self.logger.warning("No routing info collected")
            return {}

        results = self._summarize_routing_values(
            level_contribs, level_activity_rms, ff_rec_ratios
        )

        self.log_analysis_end("routing_analysis", num_results=len(results))

        if save_path is not None:
            save_dict(results, save_path, filename)

        return results

    def _collect_routing_values(
        self,
        model: BaseModel,
        core: Any,
        loader: torch.utils.data.DataLoader,
        device: str,
    ) -> tuple[
        dict[str, list[torch.Tensor]],
        dict[str, list[torch.Tensor]],
        dict[str, list[torch.Tensor]],
    ]:
        # Per key, each chunk is [n_valid, n_levels].
        level_contribs: dict[str, list[torch.Tensor]] = {}
        level_activity_rms: dict[str, list[torch.Tensor]] = {}
        ff_rec_ratios: dict[str, list[torch.Tensor]] = {}

        with analysis_device_context(model, device) as analysis_device:
            with torch.no_grad():
                for batch in loader:
                    x_batch = batch[0].to(analysis_device)
                    seq_lengths = _extract_seq_lengths(batch)
                    if seq_lengths is not None:
                        seq_lengths = seq_lengths.to(analysis_device)
                    _ = model(x_batch, seq_lengths=seq_lengths)
                    self._collect_batch_routing_values(
                        core._routing_info,
                        seq_lengths,
                        level_contribs,
                        level_activity_rms,
                        ff_rec_ratios,
                    )

        return level_contribs, level_activity_rms, ff_rec_ratios

    def _collect_batch_routing_values(
        self,
        routing_info: list[dict[str, Any]],
        seq_lengths: Optional[torch.Tensor],
        level_contribs: dict[str, list[torch.Tensor]],
        level_activity_rms: dict[str, list[torch.Tensor]],
        ff_rec_ratios: dict[str, list[torch.Tensor]],
    ) -> None:
        for timestep_idx, timestep_info in enumerate(routing_info):
            valid_mask, should_process = _valid_timestep_mask(seq_lengths, timestep_idx)
            if not should_process:
                continue
            for full_key, pop_info in iter_routing_entries(timestep_info):
                if "level_contributions" in pop_info:
                    _append_masked_routing_values(
                        level_contribs,
                        full_key,
                        pop_info["level_contributions"],
                        valid_mask,
                    )

                if "level_activity_rms" in pop_info:
                    _append_masked_routing_values(
                        level_activity_rms,
                        full_key,
                        pop_info["level_activity_rms"],
                        valid_mask,
                    )

                if "ff_rec_ratios" in pop_info:
                    _append_masked_routing_values(
                        ff_rec_ratios,
                        full_key,
                        pop_info["ff_rec_ratios"],
                        valid_mask,
                    )

    def _summarize_routing_values(
        self,
        level_contribs: dict[str, list[torch.Tensor]],
        level_activity_rms: dict[str, list[torch.Tensor]],
        ff_rec_ratios: dict[str, list[torch.Tensor]],
    ) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for key in sorted(level_contribs.keys()):
            results[key] = self._summarize_routing_entry(
                level_contribs[key],
                level_activity_rms.get(key),
                ff_rec_ratios.get(key),
            )
        return results

    def _summarize_routing_entry(
        self,
        level_contribs: list[torch.Tensor],
        level_activity_rms: Optional[list[torch.Tensor]],
        ff_rec_ratios: Optional[list[torch.Tensor]],
    ) -> dict[str, Any]:
        # Concatenate: [N, n_levels] where N = total_timesteps * batch_size.
        contribs = torch.cat(level_contribs, dim=0)
        # Historical raw-L2 share. It scales with sqrt(level width).
        contribs_sum = contribs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        l2_share = contribs / contribs_sum

        entry: dict[str, Any] = {
            "n_levels": contribs.shape[-1],
            "n_samples": contribs.shape[0],
            "mean_level_contribution": l2_share.mean(dim=0).tolist(),
            "std_level_contribution": l2_share.std(dim=0).tolist(),
            "mean_raw_voltage_norm": contribs.mean(dim=0).tolist(),
            "legacy_level_contribution_definition": "raw_l2_share",
        }

        if level_activity_rms:
            rms = torch.cat(level_activity_rms, dim=0)
            rms_sum = rms.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            rms_share = rms / rms_sum
            entry.update(
                {
                    "mean_level_activity_rms": rms.mean(dim=0).tolist(),
                    "mean_level_activity_share_rms": rms_share.mean(dim=0).tolist(),
                    "std_level_activity_share_rms": rms_share.std(dim=0).tolist(),
                    "activity_share_definition": (
                        "level_l2_div_sqrt_width_then_normalized"
                    ),
                }
            )

        if ff_rec_ratios:
            ratios = torch.cat(ff_rec_ratios, dim=0)
            entry["mean_ff_rec_ratio"] = ratios.mean(dim=0).tolist()
            entry["std_ff_rec_ratio"] = ratios.std(dim=0).tolist()

        return entry
