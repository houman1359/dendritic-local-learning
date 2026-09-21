"""Frozen-checkpoint shared-gain and independent-load factorial analysis.

The analyzer separates deterministic input scaling from stochastic shared gain.
For every trial it samples one mean-one lognormal multiplier and applies that
same scalar to every input coordinate.  It then adds either independent
Gaussian input noise or a positive background to a declared inhibitory input
stream.  Condition-stable seeds provide common random numbers across matched
models without making perturbation draws statistical replicates.
"""

from __future__ import annotations

import hashlib
import math
from typing import Optional

import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.tools.branch_local_sensitivity import (
    BranchLocalSensitivityAnalyzer,
)
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    evaluation_kwargs_from_runtime,
    materialize_dataset,
)
from dendritic_modeling.config import (
    BranchLocalSensitivityAnalysisParams,
    GainLoadPerturbationAnalysisParams,
)
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel, Classifier
from dendritic_modeling.training.utils.evaluation import (
    evaluate_accuracy,
    evaluate_categorical_loglikelihood,
)
from dendritic_modeling.utils import save_dict

GAIN_LOAD_DRAW_SEED_RULE = (
    "condition key: sha256(base|gain|load|draw); component streams: "
    "sha256(base|component|level|draw); first 63 bits"
)


def _input_statistics(inputs: torch.Tensor) -> dict[str, float | int]:
    """Summarize the realized perturbed input distribution."""

    values = inputs.detach().to(dtype=torch.float64).reshape(-1)
    if values.numel() < 1 or not torch.isfinite(values).all():
        raise ValueError("gain-load input statistics require finite inputs")
    mean = float(values.mean().item())
    standard_deviation = float(values.std(unbiased=False).item())
    return {
        "scalar_count": int(values.numel()),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "zero_fraction": float((values == 0).to(torch.float64).mean().item()),
        "coefficient_of_variation": standard_deviation / max(abs(mean), 1e-12),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
    }


def gain_load_draw_seed(
    base_seed: int,
    gain_log_sd: float,
    load_level: float,
    draw: int,
) -> int:
    """Return a grid-order-independent seed for one condition and draw."""

    payload = (
        f"{int(base_seed)}|{float(gain_log_sd):.17g}|"
        f"{float(load_level):.17g}|{int(draw)}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _component_draw_seed(
    base_seed: int,
    component: str,
    level: float,
    draw: int,
) -> int:
    payload = (
        f"{int(base_seed)}|{component!s}|{float(level):.17g}|{int(draw)}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


class GainLoadPerturbationAnalyzer(AbstractAnalyzer):
    """Cross mean-one shared gain with an explicitly declared input load.

    ``gaussian_input`` adds independent zero-mean Gaussian noise to every input
    coordinate; ``positive_inhibitory`` adds positive lognormal background only
    to coordinates at or after ``inhibitory_start``.  The latter requires an
    ordered ``[E | I]`` input and is not inferred from model internals.
    """

    def __init__(self, params: GainLoadPerturbationAnalysisParams):
        super().__init__("GainLoadPerturbationAnalyzer")
        self.params = params

    @staticmethod
    def _validate_grid(values: list[float], name: str) -> tuple[float, ...]:
        normalized = tuple(float(value) for value in values)
        if not normalized:
            raise ValueError(f"{name} must contain at least one value")
        if any(not math.isfinite(value) or value < 0 for value in normalized):
            raise ValueError(f"{name} values must be finite and non-negative")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{name} values must be unique")
        return normalized

    def _perturb(
        self,
        inputs: torch.Tensor,
        *,
        gain_log_sd: float,
        load_level: float,
        gain_draw_seed: int,
        load_draw_seed: int,
    ) -> torch.Tensor:
        gain_generator = torch.Generator(device=inputs.device)
        gain_generator.manual_seed(int(gain_draw_seed))
        load_generator = torch.Generator(device=inputs.device)
        load_generator.manual_seed(int(load_draw_seed))
        trial_shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        if gain_log_sd == 0:
            gain = torch.ones(trial_shape, device=inputs.device, dtype=inputs.dtype)
        else:
            z_gain = torch.randn(
                trial_shape,
                generator=gain_generator,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            variance = float(gain_log_sd) ** 2
            gain = torch.exp(float(gain_log_sd) * z_gain - 0.5 * variance)
        perturbed = inputs * gain

        mode = str(self.params.load_mode).lower()
        if load_level > 0 and mode == "gaussian_input":
            load = torch.randn(
                inputs.shape,
                generator=load_generator,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            perturbed = perturbed + float(load_level) * load
        elif load_level > 0 and mode == "positive_inhibitory":
            inhibitory_start = self.params.inhibitory_start
            if inhibitory_start is None:
                raise ValueError("positive_inhibitory load requires inhibitory_start")
            inhibitory_start = int(inhibitory_start)
            if inputs.ndim != 2 or not 0 < inhibitory_start < inputs.shape[1]:
                raise ValueError(
                    "inhibitory_start must split a two-dimensional [E | I] input"
                )
            i_shape = (inputs.shape[0], inputs.shape[1] - inhibitory_start)
            if self.params.load_log_sd == 0:
                load = torch.full(
                    i_shape,
                    float(load_level),
                    device=inputs.device,
                    dtype=inputs.dtype,
                )
            else:
                z_load = torch.randn(
                    i_shape,
                    generator=load_generator,
                    device=inputs.device,
                    dtype=inputs.dtype,
                )
                load_sigma = float(self.params.load_log_sd)
                load = float(load_level) * torch.exp(
                    load_sigma * z_load - 0.5 * load_sigma**2
                )
            perturbed = perturbed.clone()
            perturbed[:, inhibitory_start:] += load
        elif mode not in {"gaussian_input", "positive_inhibitory"}:
            raise ValueError(f"Unknown load_mode={self.params.load_mode!r}")

        if self.params.clamp_min is not None:
            perturbed = perturbed.clamp_min(float(self.params.clamp_min))
        if self.params.clamp_max is not None:
            perturbed = perturbed.clamp_max(float(self.params.clamp_max))
        return perturbed

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ) -> dict[str, object]:
        """Evaluate every gain/load/draw cell on one frozen checkpoint."""

        if not isinstance(model, Classifier):
            raise TypeError("Gain-load accuracy analysis requires a classifier")
        gain_log_sds = self._validate_grid(self.params.gain_log_sds, "gain_log_sds")
        load_levels = self._validate_grid(self.params.load_levels, "load_levels")
        draws_per_condition = int(self.params.draws_per_condition)
        if draws_per_condition <= 0:
            raise ValueError("draws_per_condition must be positive")
        if self.params.load_log_sd < 0 or not math.isfinite(self.params.load_log_sd):
            raise ValueError("load_log_sd must be finite and non-negative")
        if (
            self.params.clamp_min is not None
            and self.params.clamp_max is not None
            and self.params.clamp_min > self.params.clamp_max
        ):
            raise ValueError("clamp_min must not exceed clamp_max")
        sensitivity_draws = {
            int(value) for value in self.params.branch_sensitivity_draws
        }
        if any(
            value < 0 or value >= draws_per_condition for value in sensitivity_draws
        ):
            raise ValueError(
                "branch_sensitivity_draws must index declared perturbation draws"
            )
        if (
            self.params.record_branch_local_sensitivity
            and not self.params.branch_sensitivity_module_name_prefix
        ):
            raise ValueError(
                "branch sensitivity requires branch_sensitivity_module_name_prefix"
            )
        eval_kwargs = evaluation_kwargs_from_runtime(runtime)
        records: list[dict[str, object]] = []

        with analysis_device_context(model, device) as analysis_device:
            items = materialize_dataset(
                test_dataset,
                runtime,
                explicit_max_samples=self.params.max_samples,
                device=analysis_device,
            )
            if len(items) < 2:
                raise ValueError("The test dataset must provide inputs and labels")
            inputs = items[0].to(analysis_device)
            labels = items[1].to(analysis_device)
            for gain_log_sd in gain_log_sds:
                for load_level in load_levels:
                    for draw in range(draws_per_condition):
                        draw_seed = gain_load_draw_seed(
                            self.params.base_draw_seed,
                            gain_log_sd,
                            load_level,
                            draw,
                        )
                        gain_draw_seed = _component_draw_seed(
                            self.params.base_draw_seed,
                            "shared_gain",
                            gain_log_sd,
                            draw,
                        )
                        load_draw_seed = _component_draw_seed(
                            self.params.base_draw_seed,
                            self.params.load_mode,
                            load_level,
                            draw,
                        )
                        perturbed = self._perturb(
                            inputs,
                            gain_log_sd=gain_log_sd,
                            load_level=load_level,
                            gain_draw_seed=gain_draw_seed,
                            load_draw_seed=load_draw_seed,
                        )
                        dataset = TensorDataset(perturbed, labels)
                        record: dict[str, object] = {
                            "gain_log_sd": gain_log_sd,
                            "load_level": load_level,
                            "draw": draw,
                            "draw_seed": draw_seed,
                            "gain_draw_seed": gain_draw_seed,
                            "load_draw_seed": load_draw_seed,
                            "n_samples": int(labels.numel()),
                        }
                        if self.params.record_input_statistics:
                            record["input_statistics"] = _input_statistics(perturbed)
                        if (
                            self.params.record_branch_local_sensitivity
                            and draw in sensitivity_draws
                        ):
                            sensitivity_limit = (
                                self.params.branch_sensitivity_max_samples
                            )
                            if sensitivity_limit is None:
                                sensitivity_limit = int(labels.numel())
                            sensitivity_limit = min(
                                int(sensitivity_limit), int(labels.numel())
                            )
                            sensitivity_dataset = TensorDataset(
                                perturbed[:sensitivity_limit].detach().cpu(),
                                labels[:sensitivity_limit].detach().cpu(),
                            )
                            sensitivity = BranchLocalSensitivityAnalyzer(
                                BranchLocalSensitivityAnalysisParams(
                                    module_name_prefix=(
                                        self.params.branch_sensitivity_module_name_prefix
                                    ),
                                    expected_n_somas=(
                                        self.params.branch_sensitivity_expected_n_somas
                                    ),
                                    max_samples=sensitivity_limit,
                                    include_soma=(
                                        self.params.branch_sensitivity_include_soma
                                    ),
                                )
                            ).analyze(
                                model=model,
                                test_dataset=sensitivity_dataset,
                                device=str(analysis_device),
                                runtime=EvaluationRuntimeConfig(
                                    mode="stream",
                                    batch_size=min(
                                        int(getattr(runtime, "batch_size", 256)),
                                        sensitivity_limit,
                                    ),
                                    num_workers=0,
                                    pin_memory=False,
                                    max_samples=sensitivity_limit,
                                ),
                            )
                            record["branch_local_sensitivity"] = sensitivity
                        if self.params.accuracy:
                            record["accuracy"] = float(
                                evaluate_accuracy(
                                    model,
                                    test_ds=dataset,
                                    move_device=False,
                                    **eval_kwargs,
                                )[-1]
                            )
                        if self.params.categorical_loglikelihood:
                            record["categorical_loglikelihood"] = float(
                                evaluate_categorical_loglikelihood(
                                    model,
                                    test_ds=dataset,
                                    move_device=False,
                                    **eval_kwargs,
                                )[-1]
                            )
                        records.append(record)

        result: dict[str, object] = {
            "gain_model": "one mean-one lognormal scalar per trial",
            "load_mode": self.params.load_mode,
            "load_definition": (
                "independent zero-mean Gaussian input noise; load_level is s.d."
                if self.params.load_mode == "gaussian_input"
                else "positive lognormal background on the declared I stream; "
                "load_level is its mean"
            ),
            "load_log_sd": float(self.params.load_log_sd),
            "inhibitory_start": self.params.inhibitory_start,
            "gain_log_sds": list(gain_log_sds),
            "load_levels": list(load_levels),
            "draws_per_condition": draws_per_condition,
            "base_draw_seed": int(self.params.base_draw_seed),
            "draw_seed_rule": GAIN_LOAD_DRAW_SEED_RULE,
            "common_random_numbers": True,
            "technical_draws_are_inference_units": False,
            "clamp_min": self.params.clamp_min,
            "clamp_max": self.params.clamp_max,
            "input_statistics_recorded": bool(self.params.record_input_statistics),
            "branch_local_sensitivity_recorded": bool(
                self.params.record_branch_local_sensitivity
            ),
            "branch_sensitivity_draws": sorted(sensitivity_draws),
            "records": records,
        }
        if save_path is not None:
            save_dict(result, save_path, f"{filename}.json")
        return result


__all__ = [
    "GAIN_LOAD_DRAW_SEED_RULE",
    "GainLoadPerturbationAnalyzer",
    "gain_load_draw_seed",
]
