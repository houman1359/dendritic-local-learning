"""Spike-train analysis for opt-in LIF recurrent populations."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import run_no_grad_analysis_batches
from dendritic_modeling.config.analysis import (
    EvaluationRuntimeConfig,
    SpikeTrainAnalysisParams,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import LIFSoma
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    iter_named_modules_of_type,
    register_named_forward_hooks,
    run_with_forward_hooks,
)


def _binarize_spike_sequences(
    sequences: list[torch.Tensor],
    threshold: float,
) -> list[torch.Tensor]:
    return [(seq > threshold).float() for seq in sequences]


def _sequence_dimensions(sequences: list[torch.Tensor]) -> tuple[int, int, int]:
    n_samples = int(sum(seq.shape[0] for seq in sequences))
    n_timesteps = int(max((seq.shape[1] for seq in sequences), default=0))
    n_units = int(max((seq.shape[2] for seq in sequences), default=0))
    return n_samples, n_timesteps, n_units


def _total_spike_stats(
    sequences: list[torch.Tensor],
    dt: float,
) -> tuple[int, float, float, float]:
    total_bins = int(sum(seq.numel() for seq in sequences))
    total_spikes_tensor = sum(
        (seq.sum() for seq in sequences), torch.zeros((), dtype=torch.float32)
    )
    total_spikes = float(total_spikes_tensor.item())
    mean_rate_hz = total_spikes / max(total_bins * dt, 1e-12)
    spike_probability = total_spikes / max(total_bins, 1)
    return total_bins, total_spikes, mean_rate_hz, spike_probability


def _spike_count_stats(
    sequences: list[torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    counts = torch.cat([seq.sum(dim=1) for seq in sequences], dim=0)
    count_mean = counts.mean(dim=0)
    count_var = counts.var(dim=0, unbiased=False)
    fano = count_var / count_mean.clamp(min=1e-8)
    return counts, {
        "fraction_silent_units": float((counts.sum(dim=0) == 0).float().mean()),
        "fraction_silent_trials": float((counts.sum(dim=1) == 0).float().mean()),
        "spike_count_mean": float(counts.mean().item()),
        "spike_count_std": float(counts.std(unbiased=False).item()),
        "fano_factor_mean": float(fano.mean().item()),
        "fano_factor_max": float(fano.max().item()),
    }


def _isi_stats(isi_values: torch.Tensor, dt: float) -> dict[str, float]:
    if isi_values.numel() > 0:
        isi_mean = float(isi_values.float().mean().item() * dt)
        isi_std = float(isi_values.float().std(unbiased=False).item() * dt)
        isi_cv = isi_std / max(isi_mean, 1e-12)
    else:
        isi_mean = 0.0
        isi_std = 0.0
        isi_cv = 0.0
    return {
        "isi_mean": isi_mean,
        "isi_std": isi_std,
        "isi_cv": isi_cv,
    }


class SpikeTrainAnalyzer(ForwardHookRemovalMixin, AbstractAnalyzer):
    """Collect raster and population statistics from ``LIFSoma`` modules."""

    def __init__(self, params: SpikeTrainAnalysisParams | None = None):
        super().__init__("SpikeTrainAnalyzer")
        if params is None:
            params = SpikeTrainAnalysisParams()
        self.params = params
        self._current_batch: dict[str, list[torch.Tensor]] = {}
        self._spike_sequences: dict[str, list[torch.Tensor]] = {}

    def _attach_hooks(
        self, model: BaseModel
    ) -> list[torch.utils.hooks.RemovableHandle]:
        return register_named_forward_hooks(
            model,
            LIFSoma,
            self._forward_hook,
            prepare=self._prepare_spike_hook,
        )

    def _run_with_attached_hooks(
        self,
        handles: list[torch.utils.hooks.RemovableHandle],
        body: Callable[[], Any],
    ) -> Any:
        return run_with_forward_hooks(
            attach=lambda: handles,
            remove=self.remove_forward_hooks,
            body=body,
        )

    def _iter_lif_somas(self, model: BaseModel):
        yield from iter_named_modules_of_type(model, LIFSoma)

    @staticmethod
    def _prepare_spike_hook(name: str, module: LIFSoma) -> None:
        module._analysis_name = name

    def _register_spike_hook(
        self,
        name: str,
        module: LIFSoma,
    ) -> torch.utils.hooks.RemovableHandle:
        self._prepare_spike_hook(name, module)
        return module.register_forward_hook(self._forward_hook)

    def _forward_hook(
        self,
        module: LIFSoma,
        _inputs: tuple[Any, ...],
        output: tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ) -> None:
        name = getattr(module, "_analysis_name", module.__class__.__name__)
        spikes = output[4].detach()
        self._current_batch.setdefault(name, []).append(spikes)

    def _finalize_current_batch(self) -> None:
        for name, steps in self._current_batch.items():
            if not steps:
                continue
            sequence = torch.stack(steps, dim=1).cpu()  # [batch, time, units]
            self._spike_sequences.setdefault(name, []).append(sequence)
        self._current_batch = {}

    def analyze(
        self,
        model: BaseModel,
        data: torch.utils.data.Dataset | None = None,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "spike_trains",
        runtime: EvaluationRuntimeConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.log_analysis_start("spike_trains")
        self._spike_sequences = {}

        handles = self._attach_hooks(model)
        if not handles:
            self.logger.info("Skipping spike-train analysis: no LIFSoma modules found")
            return {}
        if data is None:

            def _skip_missing_data() -> dict[str, Any]:
                self.logger.info("Skipping spike-train analysis: no dataset provided")
                return {}

            return self._run_with_attached_hooks(handles, _skip_missing_data)

        sample_cap = getattr(self.params, "max_samples", None)

        def _collect_sequences() -> None:
            self._collect_spike_sequences(model, data, device, runtime, sample_cap)

        self._run_with_attached_hooks(handles, _collect_sequences)

        results = {
            "dt": float(self.params.dt),
            "threshold": float(self.params.threshold),
            "populations": {
                name: self._summarize_sequences(sequences)
                for name, sequences in self._spike_sequences.items()
            },
        }

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            save_dict(results, save_path, f"{filename}.json")

        self.log_analysis_end("spike_trains", num_results=len(results["populations"]))
        return results

    def _collect_spike_sequences(
        self,
        model: BaseModel,
        data: torch.utils.data.Dataset,
        device: str,
        runtime: EvaluationRuntimeConfig | None,
        sample_cap: int | None,
    ) -> None:
        run_no_grad_analysis_batches(
            model=model,
            dataset=data,
            device=device,
            runtime=runtime,
            explicit_max_samples=sample_cap,
            process_batch=lambda batch, analysis_device: self._collect_batch_spikes(
                model,
                batch,
                analysis_device,
            ),
        )

    def _collect_batch_spikes(
        self,
        model: BaseModel,
        batch: tuple[Any, ...],
        device: torch.device,
    ) -> None:
        self._current_batch = {}
        _ = model(batch[0].to(device))
        self._finalize_current_batch()

    def _summarize_sequences(self, sequences: list[torch.Tensor]) -> dict[str, Any]:
        threshold = float(self.params.threshold)
        dt = float(self.params.dt)
        binary = _binarize_spike_sequences(sequences, threshold)
        n_samples, n_timesteps, n_units = _sequence_dimensions(binary)
        _total_bins, total_spikes, mean_rate_hz, spike_probability = _total_spike_stats(
            binary, dt
        )
        _counts, count_stats = _spike_count_stats(binary)
        isi_summary = _isi_stats(self._collect_isis(binary), dt)

        summary: dict[str, Any] = {
            "n_samples": n_samples,
            "n_timesteps": n_timesteps,
            "n_units": n_units,
            "total_spikes": total_spikes,
            "mean_rate_hz": mean_rate_hz,
            "spike_probability": spike_probability,
            **count_stats,
            **isi_summary,
        }

        if self.params.include_per_timestep_rate:
            per_step = self._per_timestep_rate(binary)
            summary["per_timestep_rate_hz"] = per_step.tolist()
        if self.params.include_raster:
            summary["raster_events"] = self._compact_raster_events(binary)
        return summary

    @staticmethod
    def _collect_isis(sequences: list[torch.Tensor]) -> torch.Tensor:
        values = []
        for seq in sequences:
            _batch, time, _units = seq.shape
            flat = seq.permute(0, 2, 1).reshape(-1, time) > 0
            unit_indices, time_indices = torch.nonzero(flat, as_tuple=True)
            if time_indices.numel() <= 1:
                continue
            same_unit = unit_indices[1:] == unit_indices[:-1]
            diffs = time_indices[1:] - time_indices[:-1]
            diffs = diffs[same_unit]
            if diffs.numel() > 0:
                values.append(diffs)
        if not values:
            return torch.empty(0)
        return torch.cat(values)

    @staticmethod
    def _per_timestep_rate(sequences: list[torch.Tensor]) -> torch.Tensor:
        max_t = max(seq.shape[1] for seq in sequences)
        totals = torch.zeros(max_t, dtype=torch.float32)
        counts = torch.zeros(max_t, dtype=torch.float32)
        for seq in sequences:
            time = seq.shape[1]
            totals[:time] += seq.float().sum(dim=(0, 2))
            counts[:time] += seq.shape[0] * seq.shape[2]
        return totals / counts.clamp_min(1.0)

    def _compact_raster_events(
        self, sequences: list[torch.Tensor]
    ) -> list[dict[str, int]]:
        max_samples = int(self.params.max_raster_samples)
        events: list[dict[str, int]] = []
        sample_offset = 0
        for seq in sequences:
            samples_to_take = min(seq.shape[0], max_samples - sample_offset)
            if samples_to_take <= 0:
                return events
            nonzero = torch.nonzero(seq[:samples_to_take] > 0, as_tuple=False)
            for sample_idx, time_idx, unit_idx in nonzero.tolist():
                events.append(
                    {
                        "sample": int(sample_offset + sample_idx),
                        "time": int(time_idx),
                        "unit": int(unit_idx),
                    }
                )
            sample_offset += seq.shape[0]
        return events


__all__ = ["SpikeTrainAnalyzer"]
