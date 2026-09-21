"""Forward-hook collection helpers for information analysis."""

from __future__ import annotations

import torch

from dendritic_modeling.analysis.utils.dendritic_depth import (
    soma_relative_dendritic_depth,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer, PopulationLayer
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    HookHandle,
    register_named_forward_hooks,
    remove_hook_handles,
)


def _has_no_branch_synapses(module: DendriticBranchLayer) -> bool:
    return module.branch_excitation is None and module.branch_inhibition is None


def _zero_branch_currents_for_output(output: torch.Tensor) -> torch.Tensor:
    batch_size = output.shape[0]
    n_branches = output.shape[1] if len(output.shape) > 1 else 1
    return torch.zeros(batch_size, n_branches, device=output.device)


def _resolve_synaptic_branch_currents(
    module: DendriticBranchLayer,
    excitatory_input: torch.Tensor | None,
    inhibitory_input: torch.Tensor | None,
    output: torch.Tensor,
    cached_currents: object,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    has_no_synapses = _has_no_branch_synapses(module)

    if has_no_synapses:
        return (
            _zero_branch_currents_for_output(output),
            _zero_branch_currents_for_output(output),
            has_no_synapses,
        )

    if isinstance(cached_currents, dict):
        excitation_per_branch = cached_currents.get(
            "pre_gate_excitation_current", cached_currents.get("excitation")
        )
        inhibition_per_branch = cached_currents.get(
            "pre_gate_inhibition_current", cached_currents.get("inhibition")
        )
        if excitation_per_branch is None:
            excitation_per_branch = torch.zeros_like(output)
        if inhibition_per_branch is None:
            inhibition_per_branch = torch.zeros_like(output)
        return excitation_per_branch, inhibition_per_branch, has_no_synapses

    excitation_per_branch = (
        module.branch_excitation(excitatory_input)
        if module.branch_excitation is not None and excitatory_input is not None
        else torch.zeros_like(output)
    )
    inhibition_per_branch = (
        module.branch_inhibition(inhibitory_input)
        if module.branch_inhibition is not None and inhibitory_input is not None
        else torch.zeros_like(output)
    )
    return excitation_per_branch, inhibition_per_branch, has_no_synapses


def _resolve_branch_input_current(
    module: DendriticBranchLayer,
    branch_input: torch.Tensor | None,
    cached_currents: object,
) -> torch.Tensor | None:
    if isinstance(cached_currents, dict):
        return cached_currents.get(
            "pre_gate_upstream_current", cached_currents.get("branch_input")
        )
    if branch_input is not None and module.input_branches:
        return module.branches_to_output(branch_input)
    return None


def _resolve_pre_gate_voltage(cached_currents: object) -> torch.Tensor | None:
    """Return the voltage immediately before branch reactivation, if captured."""
    if not isinstance(cached_currents, dict):
        return None
    return cached_currents.get("pre_gate_voltage")


def _has_analyzable_branch_signals(module: DendriticBranchLayer) -> bool:
    has_synapses = (
        module.branch_excitation is not None or module.branch_inhibition is not None
    )
    return has_synapses or module.input_branches


def _enable_analysis_current_capture(module: DendriticBranchLayer) -> object:
    previous_store = getattr(module, "_store_analysis_currents", False)
    module._store_analysis_currents = True
    return previous_store


def _restore_analysis_current_capture(
    module: DendriticBranchLayer,
    previous_store: object,
) -> None:
    module._store_analysis_currents = previous_store
    if hasattr(module, "_last_analysis_currents"):
        delattr(module, "_last_analysis_currents")


def _prepare_branch_hook_capture(
    data_dict: dict,
    hook_modules: list[tuple[DendriticBranchLayer, object]],
    name: str,
    module: DendriticBranchLayer,
) -> None:
    data_dict[name] = {}
    module._name = name
    previous_store = _enable_analysis_current_capture(module)
    hook_modules.append((module, previous_store))


def _restore_branch_hook_capture(
    hook_modules: list[tuple[DendriticBranchLayer, object]],
) -> None:
    for module, previous_store in hook_modules:
        _restore_analysis_current_capture(module, previous_store)
    hook_modules.clear()


def _attach_analyzable_branch_hooks(
    *,
    model: BaseModel,
    data_dict: dict,
    hook_modules: list[tuple[DendriticBranchLayer, object]],
    hook,
) -> list[HookHandle]:
    try:
        return register_named_forward_hooks(
            model,
            DendriticBranchLayer,
            hook,
            predicate=lambda _name, module: _has_analyzable_branch_signals(module),
            prepare=lambda name, module: _prepare_branch_hook_capture(
                data_dict,
                hook_modules,
                name,
                module,
            ),
            with_kwargs=True,
        )
    except Exception:
        _restore_branch_hook_capture(hook_modules)
        raise


def _store_branch_hook_data(
    data_dict: dict,
    module: DendriticBranchLayer,
    *,
    excitation_per_branch: torch.Tensor,
    inhibition_per_branch: torch.Tensor,
    branch_current: torch.Tensor | None,
    pre_gate_voltage: torch.Tensor | None,
    output: torch.Tensor,
    has_no_synapses: bool,
    excitatory_input: torch.Tensor | None,
    inhibitory_input: torch.Tensor | None,
    branch_input: torch.Tensor | None,
    compute_lda_weights: bool,
) -> None:
    layer_data = data_dict[module._name]

    # Canonical, stage-explicit names.  These distinguish currents/voltage
    # entering the branch nonlinearity from its post-reactivation output.
    layer_data["pre_gate_excitation_current"] = excitation_per_branch
    layer_data["pre_gate_inhibition_current"] = inhibition_per_branch
    layer_data["pre_gate_upstream_current"] = branch_current
    layer_data["pre_gate_voltage"] = pre_gate_voltage
    layer_data["post_gate_output"] = output

    # Compatibility aliases used by existing estimators and saved analyses.
    layer_data["excitation"] = excitation_per_branch
    layer_data["inhibition"] = inhibition_per_branch
    layer_data["branch_input"] = branch_current
    layer_data["output"] = output
    if hasattr(module, "layer_idx"):
        layer_data["soma_relative_depth"] = soma_relative_dendritic_depth(module)
    layer_data["has_synapses"] = not has_no_synapses
    layer_data["has_exc_synapses"] = module.branch_excitation is not None
    layer_data["has_inh_synapses"] = (
        module.branch_inhibition is not None and inhibitory_input is not None
    )
    layer_data["has_branch_input"] = branch_input is not None and module.input_branches

    if compute_lda_weights:
        layer_data["raw_excitatory_input"] = excitatory_input
        layer_data["raw_inhibitory_input"] = inhibitory_input
        layer_data["raw_branch_input"] = branch_input
        layer_data["module"] = module


def _prepare_network_layer_hook_capture(
    data_dict: dict,
    name_by_module_id: dict[int, str],
    name: str,
    module: PopulationLayer,
) -> None:
    """Initialize one population-network layer activation record."""
    name_by_module_id[id(module)] = name
    data_dict[name] = {
        "network_layer_index": len(data_dict),
        "network_layer_name": str(getattr(module.config, "name", name)),
        "readout_population": str(module.readout_population),
        "forward_call_count": 0,
    }


def _store_network_layer_hook_data(
    data_dict: dict,
    name_by_module_id: dict[int, str],
    module: PopulationLayer,
    output: object,
) -> None:
    """Store post-gate soma populations and the selected layer readout."""
    name = name_by_module_id[id(module)]
    layer_data = data_dict[name]
    readout = output[0] if isinstance(output, tuple) else output
    soma_outputs = {
        population_name: value
        for population_name, value in module._last_outputs.items()
        if isinstance(value, torch.Tensor)
    }
    layer_data["forward_call_count"] += 1
    layer_data["post_gate_soma_outputs"] = soma_outputs
    layer_data["post_gate_readout_output"] = readout
    layer_data["capture_reduction"] = (
        "single_forward_call"
        if layer_data["forward_call_count"] == 1
        else "last_forward_call"
    )


def _branch_population_coordinates(module_name: str) -> tuple[str, str] | None:
    """Return the containing network-layer path and population name."""
    if ".populations." not in module_name or ".branch_layers." not in module_name:
        return None
    network_layer_name, population_suffix = module_name.split(".populations.", 1)
    population_name = population_suffix.split(".branch_layers.", 1)[0]
    if not network_layer_name or not population_name:
        return None
    return network_layer_name, population_name


def _branch_to_parent_soma_indices(
    *,
    n_branches: int,
    n_somas: int,
) -> torch.Tensor:
    """Map contiguous regular-tree branch blocks to their parent somas."""
    if n_branches < 1 or n_somas < 1:
        raise ValueError("Branch-to-soma mapping requires positive dimensions")
    if n_branches % n_somas:
        raise ValueError(
            f"Branch count {n_branches} is not divisible by soma count {n_somas}"
        )
    branches_per_soma = n_branches // n_somas
    return torch.arange(n_branches, dtype=torch.int64) // branches_per_soma


class InformationHooksMixin(ForwardHookRemovalMixin):
    """Collects branch-layer currents and outputs during model evaluation."""

    def attach_forward_hooks(self, model: BaseModel):
        """Attach forward hooks to the model."""
        # Initialize data_dict attribute and store somatic_synapses config
        self.data_dict = {}
        self.network_layer_data = {}
        self._network_layer_name_by_module_id = {}
        self._analysis_hook_modules = []
        self.somatic_synapses_enabled = getattr(
            model.core_network, "somatic_synapses", True
        )

        branch_handles = _attach_analyzable_branch_hooks(
            model=model,
            data_dict=self.data_dict,
            hook_modules=self._analysis_hook_modules,
            hook=self.forward_hook,
        )
        try:
            network_layer_handles = register_named_forward_hooks(
                model,
                PopulationLayer,
                self.network_layer_forward_hook,
                prepare=lambda name, module: _prepare_network_layer_hook_capture(
                    self.network_layer_data,
                    self._network_layer_name_by_module_id,
                    name,
                    module,
                ),
            )
        except Exception:
            remove_hook_handles(branch_handles)
            _restore_branch_hook_capture(self._analysis_hook_modules)
            raise
        return [*branch_handles, *network_layer_handles]

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        *hook_args,
    ):
        """Forward hook to collect branch data."""
        if len(hook_args) == 3:
            input_args, input_kwargs, output = hook_args
        else:
            input_args, output = hook_args
            input_kwargs = {}

        excitatory_input = input_kwargs.get(
            "x", input_args[0] if len(input_args) > 0 else None
        )
        inhibitory_input = input_kwargs.get(
            "inhibitory_input", input_args[1] if len(input_args) > 1 else None
        )
        branch_input = input_kwargs.get(
            "branch_input", input_args[2] if len(input_args) > 2 else None
        )

        cached_currents = getattr(module, "_last_analysis_currents", None)
        (
            excitation_per_branch,
            inhibition_per_branch,
            has_no_synapses,
        ) = _resolve_synaptic_branch_currents(
            module,
            excitatory_input,
            inhibitory_input,
            output,
            cached_currents,
        )
        branch_current = _resolve_branch_input_current(
            module,
            branch_input,
            cached_currents,
        )
        pre_gate_voltage = _resolve_pre_gate_voltage(cached_currents)

        _store_branch_hook_data(
            self.data_dict,
            module,
            excitation_per_branch=excitation_per_branch,
            inhibition_per_branch=inhibition_per_branch,
            branch_current=branch_current,
            pre_gate_voltage=pre_gate_voltage,
            output=output,
            has_no_synapses=has_no_synapses,
            excitatory_input=excitatory_input,
            inhibitory_input=inhibitory_input,
            branch_input=branch_input,
            compute_lda_weights=bool(getattr(self, "compute_lda_weights", False)),
        )

    def network_layer_forward_hook(
        self,
        module: PopulationLayer,
        _inputs,
        output,
    ) -> None:
        """Collect actual soma/readout outputs for each population-network layer."""
        _store_network_layer_hook_data(
            self.network_layer_data,
            self._network_layer_name_by_module_id,
            module,
            output,
        )

    def attach_parent_soma_outputs_to_branch_records(self) -> None:
        """Bind every branch record to its corresponding post-gate soma output."""
        for module_name, layer_data in getattr(self, "data_dict", {}).items():
            coordinates = _branch_population_coordinates(module_name)
            if coordinates is None or "output" not in layer_data:
                continue
            network_layer_name, population_name = coordinates
            network_record = getattr(self, "network_layer_data", {}).get(
                network_layer_name
            )
            if not isinstance(network_record, dict):
                continue
            soma_outputs = network_record.get("post_gate_soma_outputs", {})
            parent_soma_output = soma_outputs.get(population_name)
            if not isinstance(parent_soma_output, torch.Tensor):
                continue
            if parent_soma_output.ndim != 2:
                self.logger.warning(
                    f"Skipping parent-soma binding for {module_name}: "
                    f"expected a two-dimensional soma output, got "
                    f"{tuple(parent_soma_output.shape)}"
                )
                continue

            n_branches = int(layer_data["output"].shape[1])
            n_somas = int(parent_soma_output.shape[1])
            try:
                branch_to_soma = _branch_to_parent_soma_indices(
                    n_branches=n_branches,
                    n_somas=n_somas,
                )
            except ValueError as exc:
                self.logger.warning(
                    f"Skipping parent-soma binding for {module_name}: {exc}"
                )
                continue

            layer_data["parent_soma_output"] = parent_soma_output
            layer_data["branch_to_parent_soma_index"] = branch_to_soma
            layer_data["parent_soma_population"] = population_name
            layer_data["branches_per_parent_soma"] = n_branches // n_somas

    def collected_soma_relative_depths(self) -> dict[str, int]:
        """Return canonical depth metadata for captured dendritic layers."""
        return {
            name: int(layer_data["soma_relative_depth"])
            for name, layer_data in getattr(self, "data_dict", {}).items()
            if "soma_relative_depth" in layer_data
        }

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        super().remove_forward_hooks(handles)
        _restore_branch_hook_capture(getattr(self, "_analysis_hook_modules", []))
        getattr(self, "_network_layer_name_by_module_id", {}).clear()


__all__ = [
    "InformationHooksMixin",
    "_branch_population_coordinates",
    "_branch_to_parent_soma_indices",
]
