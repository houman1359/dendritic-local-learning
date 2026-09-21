"""Code-faithful local sensitivities of fitted dendritic branch computations.

The analyzer evaluates derivatives of each branch's post-reactivation output
with respect to its local excitatory current ``E``, inhibitory current ``I``,
inherited child current ``C``, and inherited conductance ``G``.  It uses the
exact forward diagnostics emitted by :class:`DendriticBranchLayer` and the
fitted reactivation module, so learned reactivation gain is included.

These are local, within-branch derivatives at the observed operating points.
They are not end-to-end input gradients and they do not measure intervention
necessity or class information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from dendritic_modeling.analysis.tools.path_matched_intervention import (
    inventory_nested_tree,
)
from dendritic_modeling.analysis.utils.dendritic_depth import (
    SOMA_RELATIVE_DEPTH_REFERENCE,
)
from dendritic_modeling.analysis.utils.runtime import run_model_over_analysis_batches
from dendritic_modeling.config.analysis import (
    BranchLocalSensitivityAnalysisParams,
    EvaluationRuntimeConfig,
)
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.utils.general import save_dict


@dataclass
class _MomentAccumulator:
    count: int = 0
    total: float = 0.0
    total_square: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: torch.Tensor) -> None:
        finite = values.detach().to(dtype=torch.float64).reshape(-1)
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0:
            return
        self.count += int(finite.numel())
        self.total += float(finite.sum().item())
        self.total_square += float(finite.square().sum().item())
        self.minimum = min(self.minimum, float(finite.min().item()))
        self.maximum = max(self.maximum, float(finite.max().item()))

    def summary(self) -> dict[str, float | int]:
        if self.count < 1:
            raise RuntimeError("local-sensitivity accumulator is empty")
        mean = self.total / self.count
        variance = max(0.0, self.total_square / self.count - mean * mean)
        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
            "rms": float(math.sqrt(self.total_square / self.count)),
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
        }


def _local_voltage_partials(
    module: DendriticBranchLayer,
    diagnostics: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Return diagonal local derivatives of pre-reactivation voltage."""

    required = {"E", "I", "C", "G", "N", "T", "V", "mode"}
    missing = required.difference(diagnostics)
    if missing:
        raise RuntimeError(f"branch diagnostics are missing {sorted(missing)}")

    reference = diagnostics["V"]
    if not torch.is_tensor(reference):
        raise TypeError("branch diagnostic V must be a tensor")
    ones = torch.ones_like(reference)
    zeros = torch.zeros_like(reference)
    mode = str(diagnostics["mode"])

    if mode == "shunting":
        numerator = diagnostics["N"]
        denominator = 1 + diagnostics["T"] + float(module.epsilon)
        denominator_square = denominator.square()
        return {
            # E enters both the numerator and the conductance denominator.
            "E": (denominator - numerator) / denominator_square,
            "I": -numerator / denominator_square,
            # C is inherited current; its inherited conductance is represented by G.
            "C": ones / denominator,
            "G": -numerator / denominator_square,
        }

    if mode == "raw":
        if bool(getattr(module, "use_additive_normalization", False)):
            raise ValueError(
                "raw additive local sensitivities do not support the legacy "
                "across-branch z-score; use an unnormalized checkpoint or an "
                "explicit normalized additive control"
            )
        return {"E": ones, "I": -ones, "C": ones, "G": zeros}

    if mode == "conductance_normalized":
        numerator = diagnostics["E"] - diagnostics["I"] + diagnostics["C"]
        denominator = 1 + diagnostics["T"] + float(module.epsilon)
        denominator_square = denominator.square()
        return {
            "E": (denominator - numerator) / denominator_square,
            "I": (-denominator - numerator) / denominator_square,
            "C": ones / denominator,
            "G": -numerator / denominator_square,
        }

    if mode == "tangent_matched":
        operating_point = module.additive_operating_point
        if operating_point is None:
            raise RuntimeError("tangent-matched branch has no frozen operating point")
        n0, t0 = operating_point
        denominator = 1 + float(t0) + float(module.epsilon)
        v0 = float(n0) / denominator
        return {
            "E": ones * ((1.0 - v0) / denominator),
            "I": ones * (-v0 / denominator),
            "C": ones / denominator,
            "G": ones * (-v0 / denominator),
        }

    raise ValueError(f"unsupported branch integration mode {mode!r}")


def _reactivation_derivative(
    module: DendriticBranchLayer,
    voltage: torch.Tensor,
) -> torch.Tensor:
    """Differentiate the fitted elementwise reactivation at ``voltage``."""

    with torch.enable_grad():
        probe = voltage.detach().requires_grad_(True)
        output = module.reactivation(probe)
        if output.shape != probe.shape:
            raise ValueError(
                "branch local sensitivity requires a shape-preserving "
                f"reactivation, got {tuple(probe.shape)} -> {tuple(output.shape)}"
            )
        derivative = torch.autograd.grad(
            output,
            probe,
            grad_outputs=torch.ones_like(output),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
    return derivative.detach()


class BranchLocalSensitivityAnalyzer:
    """Summarize fitted within-branch sensitivities by dendritic depth."""

    def __init__(self, params: BranchLocalSensitivityAnalysisParams):
        self.params = params

    def analyze(
        self,
        model: torch.nn.Module,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "final",
        training: bool = False,
        runtime: EvaluationRuntimeConfig | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate local derivatives on a runtime-governed held-out split."""

        if training:
            raise ValueError("branch local sensitivity is a final-only analysis")

        modules, n_somas, branch_factors = inventory_nested_tree(
            model,
            module_name_prefix=self.params.module_name_prefix,
            expected_n_somas=self.params.expected_n_somas,
            expected_branch_factors=self.params.expected_branch_factors,
        )
        requested_depths = (
            set(self.params.depths)
            if self.params.depths
            else set(
                range(
                    0 if self.params.include_soma else 1,
                    len(branch_factors) + 1,
                )
            )
        )
        unavailable = sorted(requested_depths - set(modules))
        if unavailable:
            raise ValueError(
                f"requested dendritic depths are unavailable: {unavailable}"
            )
        selected = {
            depth: modules[depth]
            for depth in sorted(requested_depths)
            if depth >= (0 if self.params.include_soma else 1)
        }
        if not selected:
            raise ValueError("branch local sensitivity selected no available depths")

        accumulators: dict[tuple[str, str, str], _MomentAccumulator] = {}
        module_modes: dict[str, str] = {}
        previous_diagnostic_state: dict[str, bool] = {}

        def _accumulator(module_name: str, component: str, statistic: str):
            key = (module_name, component, statistic)
            return accumulators.setdefault(key, _MomentAccumulator())

        def _attach_hooks():
            handles = []
            for depth, (module_name, module) in selected.items():
                previous_diagnostic_state[module_name] = bool(
                    getattr(module, "_store_diagnostics", False)
                )
                module.set_branch_diagnostics(True)

                def _capture(
                    hooked_module,
                    _inputs,
                    output,
                    *,
                    name=module_name,
                    expected_depth=depth,
                ):
                    diagnostics = hooked_module.get_branch_diagnostics()
                    partials = _local_voltage_partials(hooked_module, diagnostics)
                    gain = _reactivation_derivative(hooked_module, diagnostics["V"])
                    mode = str(diagnostics["mode"])
                    previous_mode = module_modes.setdefault(name, mode)
                    if previous_mode != mode:
                        raise RuntimeError(
                            f"branch mode changed during analysis at {name}"
                        )
                    if expected_depth < 0 or (
                        expected_depth == 0 and not self.params.include_soma
                    ):
                        raise RuntimeError(
                            "soma branch entered dendritic sensitivity analysis"
                        )

                    _accumulator(name, "V", "value").update(diagnostics["V"])
                    _accumulator(name, "V", "absolute_sensitivity").update(gain.abs())
                    _accumulator(name, "V", "signed_sensitivity").update(gain)
                    _accumulator(name, "V", "absolute_contribution").update(
                        (gain * diagnostics["V"]).abs()
                    )
                    _accumulator(name, "output", "value").update(output)

                    for component in ("E", "I", "C", "G"):
                        component_values = diagnostics[component]
                        signed = gain * partials[component]
                        _accumulator(name, component, "value").update(component_values)
                        _accumulator(name, component, "signed_sensitivity").update(
                            signed
                        )
                        _accumulator(name, component, "absolute_sensitivity").update(
                            signed.abs()
                        )
                        _accumulator(name, component, "absolute_contribution").update(
                            (signed * component_values).abs()
                        )

                handles.append(module.register_forward_hook(_capture))
            return handles

        def _remove_hooks(handles) -> None:
            for handle in handles:
                handle.remove()
            for _depth, (module_name, module) in selected.items():
                module.set_branch_diagnostics(previous_diagnostic_state[module_name])
                module.clear_branch_diagnostics()

        run_model_over_analysis_batches(
            model=model,
            dataset=test_dataset,
            device=device,
            runtime=runtime,
            explicit_max_samples=self.params.max_samples,
            attach_hooks=_attach_hooks,
            remove_hooks=_remove_hooks,
        )

        records: list[dict[str, Any]] = []
        for depth, (module_name, module) in selected.items():
            mode = module_modes.get(module_name)
            if mode is None:
                raise RuntimeError(f"branch module {module_name} was never evaluated")
            for component in ("E", "I", "C", "G", "V"):
                value = _accumulator(module_name, component, "value").summary()
                signed = _accumulator(
                    module_name, component, "signed_sensitivity"
                ).summary()
                absolute = _accumulator(
                    module_name, component, "absolute_sensitivity"
                ).summary()
                contribution = _accumulator(
                    module_name, component, "absolute_contribution"
                ).summary()
                records.append(
                    {
                        "module_name": module_name,
                        "soma_relative_depth": int(depth),
                        "integration_mode": mode,
                        "component": component,
                        "coordinate_count": int(module.branch_config.output_dim),
                        "sample_coordinate_count": int(value["count"]),
                        "component_mean": float(value["mean"]),
                        "component_std": float(value["std"]),
                        "mean_signed_doutput_dcomponent": float(signed["mean"]),
                        "mean_abs_doutput_dcomponent": float(absolute["mean"]),
                        "rms_doutput_dcomponent": float(absolute["rms"]),
                        "mean_abs_first_order_contribution": float(
                            contribution["mean"]
                        ),
                    }
                )

        result = {
            "schema_version": 1,
            "analysis_type": "branch_local_sensitivity",
            "estimand": (
                "local diagonal derivative of fitted post-reactivation branch "
                "output with respect to E, I, C, or G at observed examples"
            ),
            "interpretation": (
                "descriptive local operating-point sensitivity; not class "
                "information, end-to-end input gradient, or intervention necessity"
            ),
            "evaluation_split": "test",
            "depth_reference": SOMA_RELATIVE_DEPTH_REFERENCE,
            "module_name_prefix": self.params.module_name_prefix,
            "tree": {
                "n_somas": int(n_somas),
                "branch_factors": branch_factors,
                "module_names_by_depth": {
                    str(depth): name for depth, (name, _module) in selected.items()
                },
            },
            "records": records,
        }
        if save_path is not None:
            save_dict(result, save_path, f"{filename}.json")
        return result


__all__ = ["BranchLocalSensitivityAnalyzer"]
