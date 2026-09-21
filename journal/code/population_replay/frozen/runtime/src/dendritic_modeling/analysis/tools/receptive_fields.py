import math
import os
from copy import deepcopy
from math import prod
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import metrics
from sklearn.decomposition import NMF
from torch.distributions import Categorical

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.tools.epoch_aggregation import load_epoch_aggregation
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    materialize_dataset,
)
from dendritic_modeling.config import ReceptiveFieldAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriNet,
    DendriticBranchLayer,
    ExcitationInhibitionLayer,
    ExcitationInhibitionNetwork,
    PopulationNetwork,
    StatefulDendriNet,
)
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import (
    ForwardHookRemovalMixin,
    iter_named_modules_of_type,
    register_hook_groups,
    register_named_forward_hook_groups,
    run_with_forward_hooks,
)


class ReceptiveFieldAnalyzer(ForwardHookRemovalMixin, AbstractAnalyzer):
    """Analyzer for dendritic receptive fields and optional branch tuning."""

    _RAW_VERSION = 1

    def __init__(self, params: ReceptiveFieldAnalysisParams):
        super().__init__("ReceptiveFieldAnalyzer")

        self.n_rows: int = getattr(params, "n_rows", 2)
        self.n_cols: int = getattr(params, "n_cols", 2)

        rf_shape = self._as_shape(getattr(params, "rf_shape", None))
        exc_rf_shape = getattr(params, "exc_rf_shape", None)
        inh_rf_shape = getattr(params, "inh_rf_shape", None)
        self.exc_rf_shape = self._as_shape(
            rf_shape if exc_rf_shape is None else exc_rf_shape
        )
        self.inh_rf_shape = self._as_shape(inh_rf_shape)

        self.fig_save_format = str(getattr(params, "fig_save_format", "png")).lstrip(
            "."
        )
        if not self.fig_save_format:
            self.fig_save_format = "png"

        self.compute_activation_tuning = bool(
            getattr(params, "compute_activation_tuning", False)
        )
        self.activation_sample_signals = tuple(
            str(value) for value in getattr(params, "activation_sample_signals", [])
        )
        self.activation_sample_depths = tuple(
            int(value) for value in getattr(params, "activation_sample_depths", [])
        )
        self.activation_sample_branch_indices = tuple(
            int(value)
            for value in getattr(params, "activation_sample_branch_indices", [])
        )
        self.activation_sample_branch_indices_by_depth = {
            int(depth): tuple(int(value) for value in indices)
            for depth, indices in getattr(
                params,
                "activation_sample_branch_indices_by_depth",
                {},
            ).items()
        }
        self.activation_samples_per_class = int(
            getattr(params, "activation_samples_per_class", 0)
        )
        self.activation_sample_seed = int(getattr(params, "activation_sample_seed", 0))
        self.activation_sample_records: list[dict[str, object]] = []
        self.compute_component_rfs = bool(
            getattr(params, "compute_component_rfs", False)
        )
        self.plot_component_rfs = bool(getattr(params, "plot_component_rfs", True))
        self.compute_rf_tuning = bool(getattr(params, "compute_rf_tuning", False))
        self.plot_rf_tuning = bool(getattr(params, "plot_rf_tuning", True))
        self.rf_tuning_layer_index = int(getattr(params, "rf_tuning_layer_index", 0))
        self.n_class_examples = int(getattr(params, "n_class_examples", 5))
        self.save_raw_data = bool(getattr(params, "save_raw_data", True))
        self.load_raw_data = bool(getattr(params, "load_raw_data", False))
        self.raw_filename = getattr(params, "raw_filename", None)
        self.epsilon = float(getattr(params, "epsilon", 1e-8))

    def _infer_inh_rf_shape(
        self, core_network: ExcitationInhibitionNetwork
    ) -> Optional[tuple[int, int]]:
        """Pixel-space inhibition -> image shape; population-space -> raw vector."""
        if not isinstance(core_network, ExcitationInhibitionNetwork):
            if self.inh_rf_shape is not None:
                return self.inh_rf_shape
            return self.exc_rf_shape
        if isinstance(core_network.layers[0].inhibitory_cells, DendriNet):
            return None
        if self.inh_rf_shape is not None:
            return self.inh_rf_shape  # explicit override wins
        if self.exc_rf_shape is not None:
            return self.exc_rf_shape
        return None  # population-indexed vector, keep raw for the prod projection

    @staticmethod
    def _as_shape(value) -> Optional[tuple[int, int]]:
        if value is None:
            return None
        shape = tuple(int(v) for v in value)
        if len(shape) != 2:
            raise ValueError(f"receptive-field shape must be length 2, got {shape}")
        return shape

    @staticmethod
    def _reshape_receptive_field(
        weights: torch.Tensor, shape: Optional[tuple[int, int]]
    ) -> Optional[torch.Tensor]:
        """Reshape input-level receptive fields and skip incompatible weights."""
        weights = weights.detach().cpu()
        if shape is None:
            return weights if weights.ndim > 1 else weights.unsqueeze(0)
        if weights.numel() != prod(shape):
            return None
        return weights.reshape(shape)

    @staticmethod
    def _effective_weight(module) -> Optional[torch.Tensor]:
        if module is None:
            return None
        if hasattr(module, "pruned_weight"):
            return module.pruned_weight().detach().cpu()
        if hasattr(module, "weight"):
            weight = module.weight
            if callable(weight):
                weight = weight()
            return weight.detach().cpu()
        return None

    @staticmethod
    def _max_rq_optimization(
        cov: torch.Tensor,
        pc_1: torch.Tensor,
        lr: float = 1.0,
        max_iter: int = 1000,
        patience: int = 10,
    ) -> torch.Tensor:
        """Positive Rayleigh quotient optimizer kept for RF metric experiments."""
        with torch.enable_grad():
            _ = pc_1
            pre_v = torch.nn.Parameter(
                torch.zeros((cov.shape[0], 1), device=cov.device, dtype=cov.dtype),
                requires_grad=True,
            )
            optimizer = torch.optim.Adam([pre_v], lr=lr)

            best_pre_v = deepcopy(pre_v.data)
            best_rq = -float("inf")
            patience_counter = 0

            for _step in range(max_iter):
                optimizer.zero_grad()
                v = F.softplus(pre_v)
                rq = (v.T @ cov @ v) / (v.T @ v).clamp_min(1e-12)
                (-rq).backward()
                torch.nn.utils.clip_grad_value_(pre_v, 5.0)
                optimizer.step()

                rq_value = float(rq.detach().item())
                if rq_value > best_rq:
                    best_rq = rq_value
                    best_pre_v = deepcopy(pre_v.data)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            return F.softplus(best_pre_v).squeeze(1)

    @staticmethod
    def _max_rq_power_iteration(
        cov: torch.Tensor,
        tol: float = 1e-6,
        max_iter: int = 1000,
        patience: int = 10,
    ) -> torch.Tensor:
        """Positive power iteration approximation for the max Rayleigh vector."""
        w_prev = torch.randn(cov.shape[0], 1, device=cov.device, dtype=cov.dtype)
        v_prev = w_prev / w_prev.norm(p=2).clamp_min(1e-12)
        patience_counter = 0

        for _step in range(max_iter):
            w_next = torch.relu(cov @ v_prev)
            norm = w_next.norm(p=2)
            if norm <= 1e-12:
                break
            v_next = w_next / norm

            if (v_next - v_prev).abs().max() > tol:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            v_prev = v_next

        return v_prev.squeeze(1)

    @staticmethod
    def _nmf(
        inputs: torch.Tensor,
        n_components: int,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Extract non-negative matrix factorization components on the input data."""
        device = inputs.device
        dtype = inputs.dtype
        model = NMF(
            n_components=min(n_components, inputs.shape[0], inputs.shape[1]),
            init="nndsvda",
            max_iter=max_iter,
            tol=tol,
        )
        model.fit(inputs.detach().cpu().clamp_min(0).numpy())
        return torch.from_numpy(model.components_).to(device=device, dtype=dtype)

    def _compute_receptive_field_metrics(
        self,
        weights: torch.Tensor,
        weight_mask: torch.Tensor,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        branch_data: dict[str, object],
        prefix: str = "exc",
    ) -> dict[str, object]:
        """Compute class similarity metrics for one masked receptive field."""
        syn = weights[weight_mask]
        class_dict: dict[str, dict[str, float]] = {}

        for label in torch.unique(labels):
            input_mask = labels == label
            x = inputs[input_mask][:, weight_mask]
            if x.numel() == 0 or syn.numel() == 0:
                continue

            x_mean = x.mean(dim=0)
            x_sim_mean = F.cosine_similarity(x_mean, syn, dim=0).item()
            cov = torch.cov(x.T, correction=0)
            max_rq_pc = self._max_rq_power_iteration(cov)
            max_rq_sim = F.cosine_similarity(max_rq_pc, syn, dim=0).item()

            n_components = min(5, x.shape[0], x.shape[1])
            if n_components > 0:
                nmf_components = self._nmf(
                    x, n_components=n_components, max_iter=5000, tol=1e-3
                )
                nmf_sim = F.cosine_similarity(nmf_components, syn[None, :], dim=1)
                nmf_sim_mean = nmf_sim.mean().item()
                nmf_sim_max = nmf_sim.max().item()
                nmf_sum_sim = F.cosine_similarity(
                    nmf_components.sum(dim=0), syn, dim=0
                ).item()
            else:
                nmf_sim_mean = float("nan")
                nmf_sim_max = float("nan")
                nmf_sum_sim = float("nan")

            class_dict[f"class_{int(label.item())}"] = {
                "sim_mean": x_sim_mean,
                "sim_max_rq": max_rq_sim,
                "sim_nmf_mean": nmf_sim_mean,
                "sim_nmf_max": nmf_sim_max,
                "sim_sum_nmf": nmf_sum_sim,
                "active": (x_mean > 0).float().mean().item(),
            }

        branch_data[prefix] = class_dict
        return branch_data

    def _raw_checkpoint_path(self, save_path: str, filename: str) -> str:
        raw_filename = self.raw_filename or f"{filename}_raw_receptive_fields.pt"
        return os.path.join(save_path, raw_filename)

    def _save_raw_checkpoint(self, save_path: str, filename: str) -> None:
        if not self.save_raw_data:
            return
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            {
                "version": self._RAW_VERSION,
                "raw_dict": self.raw_dict,
                "exc_rf_shape": self.exc_rf_shape,
                "inh_rf_shape": self.inh_rf_shape,
            },
            self._raw_checkpoint_path(save_path, filename),
        )

    def _load_raw_checkpoint(self, save_path: str, filename: str) -> bool:
        if not self.load_raw_data:
            return False
        raw_path = self._raw_checkpoint_path(save_path, filename)
        if not os.path.exists(raw_path):
            return False
        try:
            payload = torch.load(raw_path, map_location="cpu")
        except TypeError:
            payload = torch.load(raw_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid receptive-field checkpoint at {raw_path}")
        raw_dict = payload.get("raw_dict", payload)
        if not isinstance(raw_dict, dict):
            raise ValueError(f"Invalid receptive-field raw_dict at {raw_path}")
        self.raw_dict = raw_dict
        return True

    def collect_raw_data(self, model: BaseModel):
        """Collect input-level receptive-field weights from all branch layers."""
        self.raw_dict = {}

        branch_counter = 0
        for module_name, module in iter_named_modules_of_type(
            model, DendriticBranchLayer
        ):
            exc_weights = self._effective_weight(module.branch_excitation)
            inh_weights = self._effective_weight(module.branch_inhibition)
            if exc_weights is None and inh_weights is None:
                continue

            for branch_idx in range(module.n_branches):
                item = {
                    "depth": module.layer_idx,
                    "branch_idx": branch_idx,
                    "module": module_name,
                    "exc": None,
                    "inh": None,
                }
                if exc_weights is not None:
                    item["exc"] = self._reshape_receptive_field(
                        exc_weights[branch_idx], self.exc_rf_shape
                    )
                if inh_weights is not None:
                    item["inh"] = self._reshape_receptive_field(
                        inh_weights[branch_idx], self.inh_rf_shape
                    )
                if item["exc"] is not None or item["inh"] is not None:
                    self.raw_dict[branch_counter] = item
                    branch_counter += 1

    def _activation_hook(
        self,
        output: torch.Tensor,
        layer_key: str,
        activation_type: str,
    ) -> None:
        if output is None:
            return
        output = output.detach().cpu()
        layer_data = self.activation_raw[layer_key]
        for branch_idx in range(output.shape[-1]):
            layer_data[branch_idx][activation_type] = output[..., branch_idx]

    def _collect_activation_raw(self, model: BaseModel, x: torch.Tensor) -> None:
        self.activation_raw = {}

        def _attach_hooks():
            return register_named_forward_hook_groups(
                model,
                DendriticBranchLayer,
                self._attach_activation_hooks,
                prepare=self._initialize_activation_records,
            )

        def _run_model():
            with torch.no_grad():
                _ = model(x)

        run_with_forward_hooks(
            attach=_attach_hooks,
            remove=self.remove_forward_hooks,
            body=_run_model,
        )

    def _initialize_activation_records(
        self, module_name: str, module: DendriticBranchLayer
    ) -> None:
        self.activation_raw[module_name] = {
            branch_idx: {"depth": module.layer_idx}
            for branch_idx in range(module.n_branches)
        }

    def _register_activation_hook(
        self,
        module: torch.nn.Module,
        layer_key: str,
        activation_type: str,
    ) -> torch.utils.hooks.RemovableHandle:
        return module.register_forward_hook(
            lambda module, inputs, output, key=layer_key: self._activation_hook(
                output, key, activation_type
            )
        )

    def _attach_activation_hooks(
        self, module_name: str, module: DendriticBranchLayer
    ) -> list[torch.utils.hooks.RemovableHandle]:
        def _hook_specs():
            if module.branch_excitation is not None:
                yield (
                    module.branch_excitation,
                    "exc",
                )
            if module.branch_inhibition is not None:
                yield (
                    module.branch_inhibition,
                    "inh",
                )
            if module.input_branches:
                yield (
                    module.branches_to_output,
                    "upstream",
                )
            yield module.reactivation, "vout"

        def _register_hook(spec):
            hook_module, activation_type = spec
            return [
                self._register_activation_hook(
                    hook_module,
                    module_name,
                    activation_type,
                )
            ]

        return register_hook_groups(_hook_specs(), _register_hook)

    def _summarize_signal(
        self,
        values: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, object]:
        values = values.reshape(values.shape[0], -1).mean(dim=-1)
        classes = torch.unique(labels)
        class_means = {}
        for label in classes:
            mask = labels == label
            if mask.any():
                class_means[str(int(label.item()))] = float(values[mask].mean().item())

        if not class_means:
            return {}

        preferred_class, preferred_mean = max(
            class_means.items(), key=lambda item: item[1]
        )
        unpreferred = [
            value for label, value in class_means.items() if label != preferred_class
        ]
        unpreferred_mean = (
            float(torch.tensor(unpreferred).mean().item()) if unpreferred else 0.0
        )

        return {
            "class_means": class_means,
            "preferred_class": int(preferred_class),
            "preferred_mean": float(preferred_mean),
            "unpreferred_mean": unpreferred_mean,
            "selectivity": float(preferred_mean - unpreferred_mean),
        }

    def _summarize_activation_tuning(self, labels: torch.Tensor) -> dict[str, object]:
        labels = labels.detach().cpu()
        summary = {}

        for layer_key, layer_data in self.activation_raw.items():
            layer_summary = {}
            for branch_idx, branch_data in layer_data.items():
                branch_summary = {"depth": branch_data["depth"], "signals": {}}
                for signal in ("exc", "inh", "upstream", "vout"):
                    if signal in branch_data:
                        branch_summary["signals"][signal] = self._summarize_signal(
                            branch_data[signal], labels
                        )

                if "exc" in branch_data and "upstream" in branch_data:
                    depolarizing_drive = branch_data["exc"] + branch_data["upstream"]
                    branch_summary["signals"]["depolarizing_drive"] = (
                        self._summarize_signal(depolarizing_drive, labels)
                    )

                if "exc" in branch_data and "inh" in branch_data:
                    ratio = branch_data["exc"] / (
                        branch_data["inh"].abs() + self.epsilon
                    )
                    branch_summary["signals"]["ei_ratio"] = self._summarize_signal(
                        ratio, labels
                    )

                layer_summary[str(branch_idx)] = branch_summary
            summary[layer_key] = layer_summary

        return summary

    def _sample_activation_tuning(
        self, labels: torch.Tensor
    ) -> list[dict[str, object]]:
        """Return label-balanced scalar activation records for selected depths.

        Dataset indices are selected once per class from labels alone, then
        reused for every requested module, branch, and signal. This makes the
        sampling independent of activation magnitude and preserves paired
        comparisons among signals.
        """

        if not self.activation_sample_signals:
            return []
        labels = labels.detach().cpu().reshape(-1)
        selected: list[tuple[int, int]] = []
        for class_label in sorted(int(value) for value in torch.unique(labels)):
            candidates = torch.nonzero(labels == class_label, as_tuple=False).reshape(
                -1
            )
            count = min(self.activation_samples_per_class, int(candidates.numel()))
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.activation_sample_seed + class_label)
            order = torch.randperm(int(candidates.numel()), generator=generator)[:count]
            selected.extend(
                (class_label, int(index)) for index in candidates[order].tolist()
            )

        records: list[dict[str, object]] = []
        selected_depths = set(self.activation_sample_depths)
        selected_branches = set(self.activation_sample_branch_indices)
        for module_name, layer_data in sorted(self.activation_raw.items()):
            for branch_index, branch_data in sorted(layer_data.items()):
                depth = int(branch_data["depth"])
                if depth not in selected_depths:
                    continue
                depth_branches = set(
                    self.activation_sample_branch_indices_by_depth.get(depth, ())
                )
                if depth_branches and branch_index not in depth_branches:
                    continue
                if (
                    not depth_branches
                    and selected_branches
                    and branch_index not in selected_branches
                ):
                    continue
                for signal in self.activation_sample_signals:
                    if signal == "ei_ratio":
                        if "exc" not in branch_data or "inh" not in branch_data:
                            continue
                        values = branch_data["exc"] / (
                            branch_data["inh"].abs() + self.epsilon
                        )
                    elif signal == "depolarizing_drive":
                        if "exc" not in branch_data or "upstream" not in branch_data:
                            continue
                        values = branch_data["exc"] + branch_data["upstream"]
                    else:
                        values = branch_data.get(signal)
                        if values is None:
                            continue
                    scalar = values.reshape(values.shape[0], -1).mean(dim=-1)
                    for class_label, dataset_index in selected:
                        records.append(
                            {
                                "module": module_name,
                                "depth": depth,
                                "branch_index": int(branch_index),
                                "signal": signal,
                                "class_label": class_label,
                                "dataset_index": dataset_index,
                                "value": float(scalar[dataset_index].item()),
                            }
                        )
        return records

    def _compute_activation_tuning(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str,
        runtime: Optional[EvaluationRuntimeConfig],
    ) -> dict[str, object]:
        with analysis_device_context(model, device) as analysis_device:
            items = materialize_dataset(
                test_dataset,
                runtime,
                device=analysis_device,
            )
            x: torch.Tensor = items[0].to(analysis_device)
            labels: torch.Tensor = items[1].cpu()
            self._collect_activation_raw(model, x)
            self.activation_sample_records = self._sample_activation_tuning(labels)
        return self._summarize_activation_tuning(labels)

    @staticmethod
    def _get_einet_core(model: BaseModel) -> Optional[ExcitationInhibitionNetwork]:
        core = getattr(model, "core_network", None)
        if isinstance(core, ExcitationInhibitionNetwork):
            return core
        wrapped = getattr(core, "einet", None)
        if isinstance(wrapped, ExcitationInhibitionNetwork):
            return wrapped
        return None

    def _get_rf_tuning_layer(
        self, model: BaseModel
    ) -> Optional[ExcitationInhibitionLayer]:
        core = self._get_einet_core(model)
        if core is None:
            return None
        layers = getattr(core, "layers", None)
        if layers is None:
            return None
        try:
            return layers[self.rf_tuning_layer_index]
        except IndexError:
            self.logger.warning(
                "RF tuning layer index %s is out of range",
                self.rf_tuning_layer_index,
            )
            return None

    def _get_component_dendrinet(
        self, model: BaseModel
    ) -> Optional[DendriNet | StatefulDendriNet]:
        """Select the dendritic readout population configured for one layer."""
        layer = self._get_rf_tuning_layer(model)
        if layer is not None and isinstance(layer.excitatory_cells, DendriNet):
            return layer.excitatory_cells

        core = getattr(model, "core_network", None)
        if not isinstance(core, PopulationNetwork):
            return None
        try:
            population_layer = core.layers[self.rf_tuning_layer_index]
        except IndexError:
            self.logger.warning(
                "Component RF layer index %s is out of range",
                self.rf_tuning_layer_index,
            )
            return None
        population_name = str(population_layer.readout_population)
        if population_name not in population_layer.populations:
            return None
        population = population_layer.populations[population_name]
        return population if isinstance(population, StatefulDendriNet) else None

    def _rf_tuning_forward_hook(
        self,
        module: DendriticBranchLayer,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        labels: torch.Tensor,
        data_dict: dict[str, dict[str, dict[str, list[float]]]],
        cell_type: str,
    ) -> None:
        """Collect class-conditioned branch currents and outputs."""
        excitatory_input = inputs[0]
        inhibitory_input = inputs[1] if len(inputs) > 1 else None

        if self.exc_inputs is None:
            self.exc_inputs = excitatory_input.detach()
        if self.inh_inputs is None and inhibitory_input is not None:
            self.inh_inputs = inhibitory_input.detach()

        excitation = None
        inhibition = None
        if module.branch_excitation is not None:
            excitation = module.branch_excitation(excitatory_input)
        if module.branch_inhibition is not None and inhibitory_input is not None:
            inhibition = module.branch_inhibition(inhibitory_input)

        n_soma = self.n_exc_soma if cell_type == "exc" else self.n_inh_soma
        if not n_soma:
            return
        branches_per_soma = max(int(module.n_branches // n_soma), 1)

        for branch_number in range(module.n_branches):
            soma_idx = int(branch_number // branches_per_soma)
            branch_idx = int(branch_number % branches_per_soma)
            location_key = (
                f"soma {soma_idx} - depth {module.layer_idx} - branch {branch_idx}"
            )
            branch_data: dict[str, dict[str, list[float]]] = {}

            for class_idx in range(self.n_classes):
                class_key = f"class_{class_idx}"
                class_mask = labels == class_idx
                class_data: dict[str, list[float]] = {}

                exc = None
                inh = None
                if excitation is not None:
                    exc = excitation[class_mask][:, branch_number]
                    class_data["exc"] = exc.detach().cpu().tolist()
                if inhibition is not None:
                    inh = inhibition[class_mask][:, branch_number]
                    class_data["inh"] = inh.detach().cpu().tolist()
                if exc is not None and inh is not None:
                    ratio = exc / (inh.abs() + self.epsilon)
                    class_data["ei_ratio"] = ratio.detach().cpu().tolist()

                vout = output[class_mask][:, branch_number]
                class_data["vout"] = vout.detach().cpu().tolist()
                branch_data[class_key] = class_data

            data_dict[location_key] = branch_data

    def _attach_rf_tuning_hooks(
        self,
        layer: ExcitationInhibitionLayer,
        labels: torch.Tensor,
        exc_cell_data: dict[str, dict[str, dict[str, list[float]]]],
        inh_cell_data: Optional[dict[str, dict[str, dict[str, list[float]]]]],
    ) -> list[torch.utils.hooks.RemovableHandle]:
        exc_dendrinet = layer.excitatory_cells
        if not isinstance(exc_dendrinet, DendriNet):
            return []

        self.n_exc_soma = exc_dendrinet.n_soma
        hook_specs: list[
            tuple[
                DendriticBranchLayer,
                str,
                dict[str, dict[str, dict[str, list[float]]]],
            ]
        ] = [
            (branch_layer, "exc", exc_cell_data)
            for branch_layer in exc_dendrinet.branch_layers
        ]

        inh_dendrinet = layer.inhibitory_cells
        self.n_inh_soma = None
        if isinstance(inh_dendrinet, DendriNet) and inh_cell_data is not None:
            self.n_inh_soma = inh_dendrinet.n_soma
            hook_specs.extend(
                (branch_layer, "inh", inh_cell_data)
                for branch_layer in inh_dendrinet.branch_layers
            )

        def _register_hook(
            spec: tuple[
                DendriticBranchLayer,
                str,
                dict[str, dict[str, dict[str, list[float]]]],
            ],
        ) -> list[torch.utils.hooks.RemovableHandle]:
            branch_layer, cell_type, data_dict = spec
            handle = branch_layer.register_forward_hook(
                lambda module, inputs, output, cell_type=cell_type, data_dict=data_dict: (
                    self._rf_tuning_forward_hook(
                        module,
                        inputs,
                        output,
                        labels=labels,
                        data_dict=data_dict,
                        cell_type=cell_type,
                    )
                )
            )
            return [handle]

        return register_hook_groups(hook_specs, _register_hook)

    def _collect_rf_tuning_activation_data(
        self,
        model: BaseModel,
        layer: ExcitationInhibitionLayer,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[
        dict[str, dict[str, dict[str, list[float]]]],
        Optional[dict[str, dict[str, dict[str, list[float]]]]],
    ]:
        exc_cell_data: dict[str, dict[str, dict[str, list[float]]]] = {}
        inh_cell_data: Optional[dict[str, dict[str, dict[str, list[float]]]]] = (
            {} if isinstance(layer.inhibitory_cells, DendriNet) else None
        )
        self.exc_inputs = None
        self.inh_inputs = None
        self.n_classes = len(torch.unique(labels))

        def _attach_hooks():
            return self._attach_rf_tuning_hooks(
                layer,
                labels,
                exc_cell_data,
                inh_cell_data,
            )

        def _run_model():
            with torch.no_grad():
                _ = model(x)

        run_with_forward_hooks(
            attach=_attach_hooks,
            remove=self.remove_forward_hooks,
            body=_run_model,
        )
        return exc_cell_data, inh_cell_data

    @staticmethod
    def _compute_point_biserial_correlation(
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        rpb_dict: dict[str, np.ndarray] = {}
        n_samples = inputs.shape[0]
        total_std = inputs.std(dim=0)

        for label in torch.unique(labels):
            class_idx = int(label.item())
            class_mask = labels == label
            pref_inputs = inputs[class_mask]
            unpref_inputs = inputs[~class_mask]
            if pref_inputs.numel() == 0 or unpref_inputs.numel() == 0:
                continue

            pref_prob = pref_inputs.shape[0] / n_samples
            unpref_prob = unpref_inputs.shape[0] / n_samples
            numer = (
                pref_prob
                * unpref_prob
                * (pref_inputs.mean(dim=0) - unpref_inputs.mean(dim=0))
            )
            rpb = numer / (total_std + 1e-9)
            rpb_dict[f"class_{class_idx}"] = rpb.detach().cpu().numpy()

        return rpb_dict

    @staticmethod
    def compute_confusion_entropy(
        model: BaseModel,
        x: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Compute row-normalized confusion and output-entropy by correctness."""
        with torch.no_grad():
            logits: torch.Tensor = model(x)

        y_pred = logits.argmax(dim=-1)
        labels_np = labels.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)
        confusion_matrix = metrics.confusion_matrix(
            y_true=labels_np,
            y_pred=y_pred_np,
            labels=unique_labels,
        ).astype(float)
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        confusion_matrix = np.divide(
            confusion_matrix,
            row_sums,
            out=np.zeros_like(confusion_matrix),
            where=row_sums != 0,
        )

        entropies = Categorical(logits=logits).entropy()
        correct_mask = labels == y_pred
        entropy_dict = {
            "correct": entropies[correct_mask].detach().cpu().numpy(),
            "incorrect": entropies[~correct_mask].detach().cpu().numpy(),
        }
        return confusion_matrix, entropy_dict

    def agg_rf_tuning(
        self,
        agg_dict: dict[str, dict[str, np.ndarray]],
        synapse_type: str,
        input_type: str,
    ) -> dict[str, dict[str, float]]:
        agg_tuning_dict: dict[str, dict[str, float]] = {}
        rpb_by_class = self.rpb_dict_by_input.get(input_type, {})

        for soma_key in sorted(agg_dict, key=_soma_sort_key):
            soma_idx = _soma_sort_key(soma_key)
            class_key = f"class_{soma_idx}"
            rf_key = f"{synapse_type}_rf"
            if class_key not in rpb_by_class or rf_key not in agg_dict[soma_key]:
                continue

            rpb = np.asarray(rpb_by_class[class_key], dtype=float).reshape(-1)
            w = np.asarray(agg_dict[soma_key][rf_key], dtype=float).reshape(-1)
            if rpb.shape != w.shape:
                self.logger.warning(
                    "Skipping RF tuning for %s %s: rpb shape %s != rf shape %s",
                    soma_key,
                    rf_key,
                    rpb.shape,
                    w.shape,
                )
                continue

            agg_tuning_dict[soma_key] = {
                "pos": _cosine_for_mask(np.abs(rpb), w, rpb > 0),
                "neg": _cosine_for_mask(np.abs(rpb), w, rpb < 0),
            }

        return agg_tuning_dict

    def collect_receptive_fields(self, ei_layer: ExcitationInhibitionLayer) -> tuple[
        dict[str, dict[str, np.ndarray]],
        Optional[dict[str, dict[str, np.ndarray]]],
    ]:
        """Collect location-keyed receptive fields from the selected E/I layer."""
        exc_rfs_dict = self._rfs_from_dendrinet(
            ei_layer.excitatory_cells,
            cell_type="exc",
        )

        inh_rfs_dict = None
        if isinstance(ei_layer.inhibitory_cells, DendriNet):
            inh_rfs_dict = self._rfs_from_dendrinet(
                ei_layer.inhibitory_cells,
                cell_type="inh",
            )

        return exc_rfs_dict, inh_rfs_dict

    def _rfs_from_dendrinet(
        self,
        dendrinet: DendriNet,
        cell_type: str,
    ) -> dict[str, dict[str, np.ndarray]]:
        rfs_dict: dict[str, dict[str, np.ndarray]] = {}
        n_soma = dendrinet.n_soma
        if not n_soma:
            return rfs_dict

        for branch_layer in dendrinet.branch_layers:
            exc_weight = self._effective_weight(branch_layer.branch_excitation)
            inh_weight = self._effective_weight(branch_layer.branch_inhibition)
            if exc_weight is None and inh_weight is None:
                continue

            branches_per_soma = max(int(branch_layer.n_branches // n_soma), 1)
            for branch_number in range(branch_layer.n_branches):
                soma_idx = int(branch_number // branches_per_soma)
                branch_idx = int(branch_number % branches_per_soma)
                location_key = (
                    f"soma {soma_idx} - depth {branch_layer.layer_idx} "
                    f"- branch {branch_idx}"
                )

                branch_rfs: dict[str, np.ndarray] = {}
                if exc_weight is not None:
                    exc_rf = self._reshape_receptive_field(
                        exc_weight[branch_number],
                        self.exc_rf_shape,
                    )
                    if exc_rf is not None:
                        branch_rfs["exc_rf"] = exc_rf.detach().cpu().numpy()
                if inh_weight is not None:
                    inh_shape = self.inh_rf_shape
                    if cell_type == "inh" and self.inh_rf_shape is None:
                        inh_shape = self.exc_rf_shape
                    inh_rf = self._reshape_receptive_field(
                        inh_weight[branch_number],
                        inh_shape,
                    )
                    if inh_rf is not None:
                        branch_rfs["inh_rf"] = inh_rf.detach().cpu().numpy()

                if branch_rfs:
                    rfs_dict[location_key] = branch_rfs

        return rfs_dict

    @staticmethod
    def _aggregate_rfs_by_soma(
        rfs_dict: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Average branch receptive fields within each soma."""
        agg_rfs_dict: dict[str, dict[str, np.ndarray]] = {}
        counts: dict[str, int] = {}

        for location_key, branch_data in rfs_dict.items():
            soma_idx = _location_key_to_tuple(location_key)[0]
            soma_key = f"soma {soma_idx}"
            counts[soma_key] = counts.get(soma_key, 0) + 1
            soma_data = agg_rfs_dict.setdefault(soma_key, {})

            for rf_key, rf_value in branch_data.items():
                if rf_key not in soma_data:
                    soma_data[rf_key] = deepcopy(rf_value)
                else:
                    soma_data[rf_key] = soma_data[rf_key] + rf_value

        for soma_key, soma_data in agg_rfs_dict.items():
            for rf_key, rf_value in soma_data.items():
                soma_data[rf_key] = rf_value / max(counts.get(soma_key, 1), 1)

        return agg_rfs_dict

    @staticmethod
    def _prod_agg_i_cell_e_rf_by_e_cell_i_ref(
        agg_inh_rfs_dict: dict[str, dict[str, np.ndarray]],
        exc_rfs_dict: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Project inhibitory-cell input RFs through E-cell inhibitory RF weights."""
        prod_rfs_dict: dict[str, dict[str, np.ndarray]] = {}

        for location_key, branch_rfs in exc_rfs_dict.items():
            e_cell_inh_rf = branch_rfs.get("inh_rf")
            if e_cell_inh_rf is None:
                continue
            i_cell_ixs = np.asarray(e_cell_inh_rf).reshape(-1).nonzero()[0].tolist()
            if not i_cell_ixs:
                continue

            prod_branch_rf: dict[str, np.ndarray] = {}
            for i_cell_idx in i_cell_ixs:
                i_cell_key = f"soma {i_cell_idx}"
                if i_cell_key not in agg_inh_rfs_dict:
                    continue
                i_cell_exc_rf = agg_inh_rfs_dict[i_cell_key].get("exc_rf")
                if i_cell_exc_rf is None:
                    continue
                if "inh_rf" not in prod_branch_rf:
                    prod_branch_rf["inh_rf"] = deepcopy(i_cell_exc_rf)
                else:
                    prod_branch_rf["inh_rf"] = prod_branch_rf["inh_rf"] + i_cell_exc_rf

            if "inh_rf" in prod_branch_rf:
                prod_branch_rf["inh_rf"] = prod_branch_rf["inh_rf"] / len(i_cell_ixs)
                prod_rfs_dict[location_key] = prod_branch_rf

        return prod_rfs_dict

    @staticmethod
    def _compute_mean_vector_dict(
        excitatory_inputs: torch.Tensor,
        inhibitory_inputs: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Compute total and class-conditioned mean vectors for input streams."""
        if inhibitory_inputs is None:
            inhibitory_inputs = excitatory_inputs

        mean_vector_dict: dict[str, dict[str, np.ndarray]] = {
            "total": {
                "exc": excitatory_inputs.mean(dim=0).detach().cpu().numpy(),
                "inh": inhibitory_inputs.mean(dim=0).detach().cpu().numpy(),
            }
        }

        for label in torch.unique(labels):
            class_idx = int(label.item())
            class_mask = labels == label
            mean_vector_dict[f"class_{class_idx}"] = {
                "exc": excitatory_inputs[class_mask].mean(dim=0).detach().cpu().numpy(),
                "inh": inhibitory_inputs[class_mask].mean(dim=0).detach().cpu().numpy(),
            }

        return mean_vector_dict

    @staticmethod
    def _smooth_map(arr: np.ndarray, n_iter: int = 1) -> np.ndarray:
        """Small separable smoothing used for compact RF summary plots."""
        kernel = np.asarray([1.0, 2.0, 1.0], dtype=float) / 4.0
        out = np.asarray(arr, dtype=float)
        for _ in range(max(0, int(n_iter))):
            out = np.apply_along_axis(
                lambda row: np.convolve(
                    np.pad(row, 1, mode="edge"),
                    kernel,
                    mode="valid",
                ),
                1,
                out,
            )
            out = np.apply_along_axis(
                lambda col: np.convolve(
                    np.pad(col, 1, mode="edge"),
                    kernel,
                    mode="valid",
                ),
                0,
                out,
            )
        return out

    @staticmethod
    def _normalize_map(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        finite = np.isfinite(arr)
        if not finite.any():
            return np.zeros_like(arr)
        lo = float(np.nanpercentile(arr[finite], 1))
        hi = float(np.nanpercentile(arr[finite], 99))
        if hi <= lo:
            lo = float(np.nanmin(arr[finite]))
            hi = float(np.nanmax(arr[finite]))
        if hi <= lo:
            return np.zeros_like(arr)
        return np.clip((arr - lo) / (hi - lo), 0, 1)

    @staticmethod
    def _weighted_image_mean(
        images: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        weights = np.clip(weights, 0, None)
        if weights.sum() <= 1e-12:
            return images.mean(axis=0)
        return (images * weights[:, None]).sum(axis=0) / weights.sum()

    def _as_component_map(
        self,
        value: Optional[np.ndarray],
        shape: tuple[int, int],
    ) -> np.ndarray:
        if value is None:
            return np.zeros(shape, dtype=float)
        arr = np.asarray(value, dtype=float)
        if arr.shape == shape:
            return arr
        if arr.size == prod(shape):
            return arr.reshape(shape)
        return np.zeros(shape, dtype=float)

    def _component_branch_hook(
        self,
        module: DendriticBranchLayer,
        inputs: tuple[torch.Tensor, ...],
        input_kwargs: dict[str, object],
        output: torch.Tensor,
        records: dict[int, dict[str, torch.Tensor]],
        branch_layer_idx: int,
    ) -> None:
        excitatory_input = input_kwargs.get("x", inputs[0] if len(inputs) > 0 else None)
        inhibitory_input = input_kwargs.get(
            "inhibitory_input", inputs[1] if len(inputs) > 1 else None
        )
        if not isinstance(excitatory_input, torch.Tensor):
            return
        if inhibitory_input is not None and not isinstance(
            inhibitory_input, torch.Tensor
        ):
            return

        excitation = torch.zeros_like(output)
        inhibition = torch.zeros_like(output)
        if module.branch_excitation is not None:
            excitation = module.branch_excitation(excitatory_input)
        if module.branch_inhibition is not None and inhibitory_input is not None:
            inhibition = module.branch_inhibition(inhibitory_input)

        records[branch_layer_idx] = {
            "x": excitatory_input.detach(),
            "E": excitation.detach(),
            "I": inhibition.detach(),
            "out": output.detach(),
        }

    def _collect_component_branch_records(
        self,
        model: BaseModel,
        dendrinet: DendriNet | StatefulDendriNet,
        x: torch.Tensor,
    ) -> list[dict[str, torch.Tensor]]:
        records: dict[int, dict[str, torch.Tensor]] = {}

        hook_specs = list(enumerate(dendrinet.branch_layers))

        def _register_hook(
            spec: tuple[int, DendriticBranchLayer],
        ) -> list[torch.utils.hooks.RemovableHandle]:
            branch_layer_idx, branch_layer = spec
            handle = branch_layer.register_forward_hook(
                lambda module, inputs, kwargs, output, branch_layer_idx=branch_layer_idx: (
                    self._component_branch_hook(
                        module,
                        inputs,
                        kwargs,
                        output,
                        records=records,
                        branch_layer_idx=branch_layer_idx,
                    )
                ),
                with_kwargs=True,
            )
            return [handle]

        def _attach_hooks():
            return register_hook_groups(hook_specs, _register_hook)

        def _run_model():
            with torch.no_grad():
                _ = model(x)

        run_with_forward_hooks(
            attach=_attach_hooks,
            remove=self.remove_forward_hooks,
            body=_run_model,
        )
        return [records[idx] for idx in sorted(records)]

    def _conditioned_soma_from_records(
        self,
        dendrinet: DendriNet | StatefulDendriNet,
        records: list[dict[str, torch.Tensor]],
        *,
        vary: str,
    ) -> torch.Tensor:
        output = None
        for branch_layer, record in zip(dendrinet.branch_layers, records):
            excitation = record["E"]
            inhibition = record["I"]

            if vary == "exc":
                excitation_used = excitation
                inhibition_used = inhibition.mean(dim=0, keepdim=True).expand_as(
                    inhibition
                )
            elif vary == "inh":
                excitation_used = excitation.mean(dim=0, keepdim=True).expand_as(
                    excitation
                )
                inhibition_used = inhibition
            elif vary == "mean":
                excitation_used = excitation.mean(dim=0, keepdim=True).expand_as(
                    excitation
                )
                inhibition_used = inhibition.mean(dim=0, keepdim=True).expand_as(
                    inhibition
                )
            else:
                raise ValueError(f"Unknown component RF conditioning mode: {vary}")

            if branch_layer.input_branches and output is not None:
                branch_drive = branch_layer.branches_to_output(output)
                branch_conductance = branch_layer.branches_to_output.sum_conductances()
                branch_conductance = branch_conductance.to(
                    device=excitation.device,
                    dtype=excitation.dtype,
                )
            else:
                branch_drive = torch.zeros_like(excitation)
                branch_conductance = torch.zeros(
                    (excitation.shape[-1],),
                    device=excitation.device,
                    dtype=excitation.dtype,
                )

            if branch_layer.use_shunting:
                voltage = (excitation_used + branch_drive) / (
                    1.0
                    + excitation_used
                    + inhibition_used
                    + branch_conductance
                    + branch_layer.epsilon
                )
            else:
                voltage = excitation_used + branch_drive - inhibition_used
                if branch_layer.use_additive_normalization:
                    voltage = (voltage - voltage.mean(dim=-1, keepdim=True)) / (
                        voltage.std(dim=-1, keepdim=True, unbiased=False)
                        + branch_layer.epsilon
                    )
            output = branch_layer.reactivation(voltage)

        if output is None:
            raise ValueError("No branch records were collected for component RFs")
        return output

    @staticmethod
    def _preferred_classes(
        soma_values: torch.Tensor,
        labels: torch.Tensor,
    ) -> np.ndarray:
        soma_values = soma_values.detach().cpu()
        labels = labels.detach().cpu()
        classes = torch.unique(labels)
        class_means = []
        for class_label in classes:
            class_mask = labels == class_label
            class_means.append(soma_values[class_mask].mean(dim=0))
        means = torch.stack(class_means, dim=0)
        preferred_ix = means.argmax(dim=0)
        return classes[preferred_ix].numpy().astype(int)

    def _component_rf_image_source(
        self,
        x: torch.Tensor,
        records: list[dict[str, torch.Tensor]],
    ) -> Optional[np.ndarray]:
        if self.exc_rf_shape is None:
            return None
        n_features = prod(self.exc_rf_shape)
        if x.shape[-1] == n_features:
            return x.detach().cpu().reshape(x.shape[0], n_features).numpy()
        if records and records[0]["x"].shape[-1] == n_features:
            source = records[0]["x"].detach().cpu()
            return source.reshape(source.shape[0], n_features).numpy()
        return None

    def _structural_component_maps(
        self,
        dendrinet: DendriNet | StatefulDendriNet,
        n_soma: int,
        shape: tuple[int, int],
        inhibitory_dendrinet: Optional[DendriNet] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        exc_rfs_dict = self._rfs_from_dendrinet(dendrinet, cell_type="exc")
        exc_agg_rfs_dict = self._aggregate_rfs_by_soma(exc_rfs_dict)

        agg_prod_rfs_dict = None
        if inhibitory_dendrinet is not None:
            inh_rfs_dict = self._rfs_from_dendrinet(
                inhibitory_dendrinet,
                cell_type="inh",
            )
            inh_agg_rfs_dict = self._aggregate_rfs_by_soma(inh_rfs_dict)
            agg_prod_rfs_dict = self._project_inhibitory_component_rfs_by_soma(
                exc_rfs_dict=exc_rfs_dict,
                inh_agg_rfs_dict=inh_agg_rfs_dict,
                shape=shape,
            )

        exc_maps = []
        inh_maps = []
        for soma_idx in range(n_soma):
            soma_key = f"soma {soma_idx}"
            soma_rfs = exc_agg_rfs_dict.get(soma_key, {})
            exc_maps.append(
                self._smooth_map(
                    self._as_component_map(soma_rfs.get("exc_rf"), shape),
                    n_iter=2,
                )
            )

            projected_inh = None
            if agg_prod_rfs_dict is not None:
                projected_inh = agg_prod_rfs_dict.get(soma_key, {}).get("inh_rf")
            if projected_inh is None:
                projected_inh = soma_rfs.get("inh_rf")
            inh_maps.append(
                self._smooth_map(
                    self._as_component_map(projected_inh, shape),
                    n_iter=2,
                )
            )

        return np.stack(exc_maps, axis=0), np.stack(inh_maps, axis=0)

    def _project_inhibitory_component_rfs_by_soma(
        self,
        *,
        exc_rfs_dict: dict[str, dict[str, np.ndarray]],
        inh_agg_rfs_dict: dict[str, dict[str, np.ndarray]],
        shape: tuple[int, int],
    ) -> dict[str, dict[str, np.ndarray]]:
        projected: dict[str, dict[str, np.ndarray]] = {}
        counts: dict[str, int] = {}

        for location_key, branch_rfs in exc_rfs_dict.items():
            inhibitory_weights = branch_rfs.get("inh_rf")
            if inhibitory_weights is None:
                continue

            branch_map = np.zeros(shape, dtype=float)
            any_projected = False
            for inh_idx, weight in enumerate(
                np.asarray(inhibitory_weights).reshape(-1)
            ):
                if abs(float(weight)) <= self.epsilon:
                    continue
                inh_rf = inh_agg_rfs_dict.get(f"soma {inh_idx}", {}).get("exc_rf")
                if inh_rf is None:
                    continue
                branch_map = branch_map + float(weight) * self._as_component_map(
                    inh_rf,
                    shape,
                )
                any_projected = True

            if not any_projected:
                continue

            soma_idx = _location_key_to_tuple(location_key)[0]
            soma_key = f"soma {soma_idx}"
            counts[soma_key] = counts.get(soma_key, 0) + 1
            soma_data = projected.setdefault(
                soma_key,
                {"inh_rf": np.zeros(shape, dtype=float)},
            )
            soma_data["inh_rf"] = soma_data["inh_rf"] + branch_map

        for soma_key, soma_data in projected.items():
            soma_data["inh_rf"] = soma_data["inh_rf"] / max(counts.get(soma_key, 1), 1)

        return projected

    def _compute_component_receptive_fields(
        self,
        model: BaseModel,
        x: torch.Tensor,
        labels: torch.Tensor,
        save_path: Optional[str],
        filename: str,
        training: bool,
    ) -> Optional[dict[str, object]]:
        if self.exc_rf_shape is None:
            self.logger.warning("Component RFs require rf_shape or exc_rf_shape")
            return None

        layer = self._get_rf_tuning_layer(model)
        dendrinet = self._get_component_dendrinet(model)
        if dendrinet is None:
            self.logger.warning(
                "Component RFs require a selectable dendritic readout population"
            )
            return None
        if any(
            getattr(branch_layer, "input_recurrent", False)
            or getattr(branch_layer, "input_rec_inhibitory", False)
            for branch_layer in dendrinet.branch_layers
        ):
            self.logger.warning(
                "Skipping component RFs: recurrent dendritic branches are not supported"
            )
            return None

        records = self._collect_component_branch_records(
            model,
            dendrinet,
            x,
        )
        if not records:
            self.logger.warning("Component RF hooks did not capture branch records")
            return None

        images = self._component_rf_image_source(x, records)
        if images is None:
            self.logger.warning(
                "Component RF inputs do not match requested RF shape %s",
                self.exc_rf_shape,
            )
            return None

        observed_soma = records[-1]["out"]
        preferred = self._preferred_classes(observed_soma, labels)
        e_conditioned = self._conditioned_soma_from_records(
            dendrinet,
            records,
            vary="exc",
        )
        i_conditioned = self._conditioned_soma_from_records(
            dendrinet,
            records,
            vary="inh",
        )
        mean_conditioned = self._conditioned_soma_from_records(
            dendrinet,
            records,
            vary="mean",
        )

        labels_np = labels.detach().cpu().numpy()
        n_soma = int(observed_soma.shape[-1])
        e_component_maps = []
        i_component_maps = []
        for soma_idx in range(n_soma):
            preferred_mask = labels_np == preferred[soma_idx]
            if not preferred_mask.any():
                preferred_mask = np.ones_like(labels_np, dtype=bool)

            e_component = e_conditioned[:, soma_idx] - mean_conditioned[:, soma_idx]
            i_component = mean_conditioned[:, soma_idx] - i_conditioned[:, soma_idx]
            e_map = self._weighted_image_mean(
                images[preferred_mask],
                e_component.detach().cpu().numpy()[preferred_mask],
            ).reshape(self.exc_rf_shape)
            i_map = self._weighted_image_mean(
                1.0 - images[preferred_mask],
                i_component.detach().cpu().numpy()[preferred_mask],
            ).reshape(self.exc_rf_shape)
            e_component_maps.append(self._smooth_map(e_map, n_iter=1))
            i_component_maps.append(self._smooth_map(i_map, n_iter=1))

        inhibitory_dendrinet = None
        if layer is not None and isinstance(layer.inhibitory_cells, DendriNet):
            inhibitory_dendrinet = layer.inhibitory_cells
        e_mean, i_mean = self._structural_component_maps(
            dendrinet,
            n_soma=n_soma,
            shape=self.exc_rf_shape,
            inhibitory_dendrinet=inhibitory_dendrinet,
        )
        maps = {
            "e_mean": e_mean,
            "e_comp": np.stack(e_component_maps, axis=0),
            "i_mean": i_mean,
            "i_comp": np.stack(i_component_maps, axis=0),
            "preferred": preferred,
        }

        out_path = None
        if save_path is not None:
            out_path = os.path.join(save_path, "epochs") if training else save_path
            self._save_component_receptive_fields(
                maps=maps,
                save_path=out_path,
                filename=filename,
                plot=bool(self.plot_component_rfs and not training),
            )

        summary: dict[str, object] = {
            "n_soma": n_soma,
            "preferred_classes": preferred.tolist(),
            "map_shape": list(self.exc_rf_shape),
            "maps": ["e_mean", "e_comp", "i_mean", "i_comp"],
        }
        if out_path is not None:
            summary["data_file"] = f"{filename}_component_receptive_fields.npz"
            if self.plot_component_rfs and not training:
                summary["figure_file"] = (
                    f"{filename}_component_receptive_fields.{self.fig_save_format}"
                )
        return summary

    def _save_component_receptive_fields(
        self,
        *,
        maps: dict[str, np.ndarray],
        save_path: str,
        filename: str,
        plot: bool,
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_path, f"{filename}_component_receptive_fields.npz"),
            **maps,
        )
        if plot:
            self._plot_component_receptive_fields(maps, save_path, filename)

    def _plot_component_receptive_fields(
        self,
        maps: dict[str, np.ndarray],
        save_path: str,
        filename: str,
    ) -> None:
        rows = [
            ("E mean", maps["e_mean"], "tab:orange"),
            ("E comp.", maps["e_comp"], "tab:orange"),
            ("I mean", maps["i_mean"], "tab:green"),
            ("I comp.", maps["i_comp"], "tab:green"),
        ]
        preferred = np.asarray(maps["preferred"], dtype=int)
        n_soma = int(preferred.shape[0])
        fig_width = max(5.5, 0.7 * n_soma)
        fig, axes = plt.subplots(
            4,
            n_soma,
            figsize=(fig_width, 3.0),
            squeeze=False,
            constrained_layout=True,
        )

        for row_idx, (label, row_maps, color) in enumerate(rows):
            for soma_idx in range(n_soma):
                ax = axes[row_idx, soma_idx]
                ax.imshow(
                    self._normalize_map(row_maps[soma_idx]),
                    cmap="magma",
                    vmin=0,
                    vmax=1,
                    interpolation="bilinear",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row_idx == 0:
                    ax.set_title(
                        f"a{soma_idx}\nclass {preferred[soma_idx]}",
                        fontsize=7,
                        pad=2,
                    )
                if soma_idx == 0:
                    ax.set_ylabel(
                        label,
                        color=color,
                        fontsize=8,
                        fontweight="bold",
                        rotation=0,
                        ha="right",
                        va="center",
                        labelpad=28,
                    )

        fig.savefig(
            os.path.join(
                save_path,
                f"{filename}_component_receptive_fields.{self.fig_save_format}",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    @staticmethod
    def _branch_data_to_preferred(
        data_dict: dict[str, dict[str, dict[str, list[float]]]],
    ) -> dict[str, dict[str, dict[str, list[float]]]]:
        """Collapse class-conditioned branch data into preferred/unpreferred bins."""
        pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]] = {}
        for location_key, branch_data in data_dict.items():
            metrics_for_branch = set()
            for class_data in branch_data.values():
                if isinstance(class_data, dict):
                    metrics_for_branch.update(class_data)

            pref_dict = {metric: [] for metric in metrics_for_branch}
            unpref_dict = {metric: [] for metric in metrics_for_branch}
            pref_class_key = f"class_{_location_key_to_tuple(location_key)[0]}"

            for class_key, class_data in branch_data.items():
                if not isinstance(class_data, dict):
                    continue
                target = pref_dict if class_key == pref_class_key else unpref_dict
                for metric, value_list in class_data.items():
                    target[metric].extend(value_list)

            pref_unpref_data[location_key] = {
                "preferred": pref_dict,
                "unpreferred": unpref_dict,
            }

        return pref_unpref_data

    def load_epochs_data(self, save_path: str):
        return load_epoch_aggregation(save_path, preserved_leaf_keys=())

    def _compute_rf_tuning_analysis(
        self,
        model: BaseModel,
        x: torch.Tensor,
        labels: torch.Tensor,
        save_path: Optional[str],
        training: bool,
    ) -> Optional[dict[str, object]]:
        layer = self._get_rf_tuning_layer(model)
        if layer is None:
            self.logger.warning(
                "RF tuning requested but the model does not expose a selectable "
                "ExcitationInhibitionLayer"
            )
            return None

        exc_cell_data, inh_cell_data = self._collect_rf_tuning_activation_data(
            model,
            layer,
            x,
            labels,
        )
        if self.exc_inputs is None:
            self.logger.warning("RF tuning hooks did not capture excitatory inputs")
            return None
        if self.inh_inputs is None:
            self.inh_inputs = self.exc_inputs

        self.rpb_dict_by_input = {
            "exc": self._compute_point_biserial_correlation(self.exc_inputs, labels),
            "inh": self._compute_point_biserial_correlation(self.inh_inputs, labels),
        }
        confusion_matrix, entropy_dict = self.compute_confusion_entropy(
            model, x, labels
        )
        pref_unpref_data = self._branch_data_to_preferred(exc_cell_data)
        exc_rfs_dict, inh_rfs_dict = self.collect_receptive_fields(layer)
        exc_agg_rfs_dict = self._aggregate_rfs_by_soma(exc_rfs_dict)

        exc_agg_tuning_dict = self.agg_rf_tuning(
            agg_dict=exc_agg_rfs_dict,
            synapse_type="exc",
            input_type="exc",
        )

        prod_rfs_dict = None
        agg_prod_rfs_dict = None
        inh_agg_rfs_dict = None
        if inh_rfs_dict is not None:
            inh_agg_rfs_dict = self._aggregate_rfs_by_soma(inh_rfs_dict)
            prod_rfs_dict = self._prod_agg_i_cell_e_rf_by_e_cell_i_ref(
                agg_inh_rfs_dict=inh_agg_rfs_dict,
                exc_rfs_dict=exc_rfs_dict,
            )
            agg_prod_rfs_dict = self._aggregate_rfs_by_soma(prod_rfs_dict)
            inh_agg_tuning_dict = self.agg_rf_tuning(
                agg_dict=agg_prod_rfs_dict,
                synapse_type="inh",
                input_type="exc",
            )
        else:
            inh_agg_tuning_dict = self.agg_rf_tuning(
                agg_dict=exc_agg_rfs_dict,
                synapse_type="inh",
                input_type="inh",
            )

        agg_tuning_dict = {
            "exc": exc_agg_tuning_dict,
            "inh": inh_agg_tuning_dict,
        }

        mean_vector_dict = self._compute_mean_vector_dict(
            self.exc_inputs,
            self.inh_inputs,
            labels,
        )

        if save_path is not None and not training and self.plot_rf_tuning:
            self._plot_rf_tuning_outputs(
                save_path=save_path,
                x=x,
                labels=labels,
                confusion_matrix=confusion_matrix,
                entropy_dict=entropy_dict,
                pref_unpref_data=pref_unpref_data,
                exc_cell_data=exc_cell_data,
                inh_cell_data=inh_cell_data,
                exc_rfs_dict=exc_rfs_dict,
                inh_rfs_dict=inh_rfs_dict,
                exc_agg_rfs_dict=exc_agg_rfs_dict,
                inh_agg_rfs_dict=inh_agg_rfs_dict,
                prod_rfs_dict=prod_rfs_dict,
                agg_prod_rfs_dict=agg_prod_rfs_dict,
                mean_vector_dict=mean_vector_dict,
                agg_tuning_dict=agg_tuning_dict,
            )

        return {
            "agg_tuning": agg_tuning_dict,
            "n_exc_receptive_fields": len(exc_rfs_dict),
            "n_inh_receptive_fields": len(inh_rfs_dict or {}),
            "confusion_matrix": confusion_matrix,
            "entropy": entropy_dict,
        }

    def _plot_rf_tuning_outputs(
        self,
        *,
        save_path: str,
        x: torch.Tensor,
        labels: torch.Tensor,
        confusion_matrix: np.ndarray,
        entropy_dict: dict[str, np.ndarray],
        pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]],
        exc_cell_data: dict[str, dict[str, dict[str, list[float]]]],
        inh_cell_data: Optional[dict[str, dict[str, dict[str, list[float]]]]],
        exc_rfs_dict: dict[str, dict[str, np.ndarray]],
        inh_rfs_dict: Optional[dict[str, dict[str, np.ndarray]]],
        exc_agg_rfs_dict: dict[str, dict[str, np.ndarray]],
        inh_agg_rfs_dict: Optional[dict[str, dict[str, np.ndarray]]],
        prod_rfs_dict: Optional[dict[str, dict[str, np.ndarray]]],
        agg_prod_rfs_dict: Optional[dict[str, dict[str, np.ndarray]]],
        mean_vector_dict: dict[str, dict[str, np.ndarray]],
        agg_tuning_dict: dict[str, dict[str, dict[str, float]]],
    ) -> None:
        discrim_inputs = self._discriminability_plot_inputs(x)
        _plot_class_discriminability_heatmaps(
            save_path=save_path,
            fig_format=self.fig_save_format,
            inputs=discrim_inputs,
            labels=labels,
            shape=self.exc_rf_shape,
        )
        if self.exc_rf_shape is not None:
            _plot_class_examples(
                save_path=save_path,
                fig_format=self.fig_save_format,
                inputs=discrim_inputs,
                labels=labels,
                shape=self.exc_rf_shape,
                n_examples=self.n_class_examples,
            )
        _plot_confusion_and_entropies(
            confusion_matrix=confusion_matrix,
            entropy_dict=entropy_dict,
            save_path=save_path,
            fig_format=self.fig_save_format,
        )
        _plot_agg_tuning_final(
            save_path=save_path,
            fig_format=self.fig_save_format,
            agg_tuning_dict=agg_tuning_dict,
        )
        epochs_data, epoch_numbers = self.load_epochs_data(save_path)
        if epochs_data and epoch_numbers:
            _plot_agg_tuning_vs_training(
                epochs_data=epochs_data,
                epoch_numbers=epoch_numbers,
                save_path=save_path,
                fig_format=self.fig_save_format,
            )

        activation_plots_path = os.path.join(save_path, "activation_plots")
        _plot_exc_vs_inh_by_compartment(
            pref_unpref_data=pref_unpref_data,
            save_path=activation_plots_path,
            fig_format=self.fig_save_format,
        )
        _plot_vout_vs_eiratio_by_compartment(
            pref_unpref_data=pref_unpref_data,
            save_path=activation_plots_path,
            fig_format=self.fig_save_format,
        )
        _plot_vout_activation_boxplots(
            data_dict=exc_cell_data,
            save_path=activation_plots_path,
            fig_format=self.fig_save_format,
            cell_type="exc",
        )
        if inh_cell_data is not None:
            _plot_vout_activation_boxplots(
                data_dict=inh_cell_data,
                save_path=activation_plots_path,
                fig_format=self.fig_save_format,
                cell_type="inh",
            )

        syn_act_rfs_path = os.path.join(save_path, "synapse_activation_rfs")
        _plot_receptive_fields(
            save_path=syn_act_rfs_path,
            fig_format=self.fig_save_format,
            title="exc_cell_exc_syn_pref_act_rfs",
            rfs_dict=exc_rfs_dict,
            synapse_type="exc",
            mean_vector_dict=mean_vector_dict,
            total_mean_vector=False,
            mean_vector_type="exc",
        )
        _plot_receptive_fields(
            save_path=syn_act_rfs_path,
            fig_format=self.fig_save_format,
            title="exc_cell_exc_syn_total_act_rfs",
            rfs_dict=exc_rfs_dict,
            synapse_type="exc",
            mean_vector_dict=mean_vector_dict,
            total_mean_vector=True,
            mean_vector_type="exc",
        )

        agg_syn_act_rfs_path = os.path.join(save_path, "agg_synapse_activation_rfs")
        _plot_receptive_fields(
            save_path=agg_syn_act_rfs_path,
            fig_format=self.fig_save_format,
            title="agg_exc_cell_exc_syn_pref_act_rfs",
            rfs_dict=exc_agg_rfs_dict,
            synapse_type="exc",
            mean_vector_dict=mean_vector_dict,
            total_mean_vector=False,
            mean_vector_type="exc",
            flatten_axes=True,
        )
        _plot_receptive_fields(
            save_path=agg_syn_act_rfs_path,
            fig_format=self.fig_save_format,
            title="agg_exc_cell_exc_syn_total_act_rfs",
            rfs_dict=exc_agg_rfs_dict,
            synapse_type="exc",
            mean_vector_dict=mean_vector_dict,
            total_mean_vector=True,
            mean_vector_type="exc",
            flatten_axes=True,
        )
        _plot_receptive_fields(
            save_path=agg_syn_act_rfs_path,
            fig_format=self.fig_save_format,
            title="agg_exc_cell_inh_syn_pref_act_rfs",
            rfs_dict=exc_agg_rfs_dict,
            synapse_type="inh",
            mean_vector_dict=mean_vector_dict,
            total_mean_vector=False,
            mean_vector_type="inh",
            flatten_axes=True,
        )
        _plot_receptive_fields(
            save_path=agg_syn_act_rfs_path,
            fig_format=self.fig_save_format,
            title="agg_exc_cell_inh_syn_total_act_rfs",
            rfs_dict=exc_agg_rfs_dict,
            synapse_type="inh",
            mean_vector_dict=mean_vector_dict,
            total_mean_vector=True,
            mean_vector_type="inh",
            flatten_axes=True,
        )

        if inh_rfs_dict is not None and inh_agg_rfs_dict is not None:
            _plot_receptive_fields(
                save_path=syn_act_rfs_path,
                fig_format=self.fig_save_format,
                title="inh_cell_exc_syn_total_act_rfs",
                rfs_dict=inh_rfs_dict,
                synapse_type="exc",
                mean_vector_dict=mean_vector_dict,
                total_mean_vector=True,
                mean_vector_type="exc",
            )
            _plot_receptive_fields(
                save_path=agg_syn_act_rfs_path,
                fig_format=self.fig_save_format,
                title="agg_inh_cell_exc_syn_total_act_rfs",
                rfs_dict=inh_agg_rfs_dict,
                synapse_type="exc",
                mean_vector_dict=mean_vector_dict,
                total_mean_vector=True,
                mean_vector_type="exc",
            )

        if prod_rfs_dict is not None and agg_prod_rfs_dict is not None:
            agg_prod_rfs_path = os.path.join(
                save_path, "inh_agg_prod_synapse_activation_rfs"
            )
            for title, rfs_dict, flatten_axes in (
                ("inh_agg_prod_ie_syn_pref_act_rfs", prod_rfs_dict, False),
                ("inh_agg_prod_ie_syn_total_act_rfs", prod_rfs_dict, False),
                ("agg_inh_agg_prod_ie_syn_pref_act_rfs", agg_prod_rfs_dict, True),
                ("agg_inh_agg_prod_ie_syn_total_act_rfs", agg_prod_rfs_dict, True),
            ):
                _plot_receptive_fields(
                    save_path=agg_prod_rfs_path,
                    fig_format=self.fig_save_format,
                    title=title,
                    rfs_dict=rfs_dict,
                    synapse_type="inh",
                    mean_vector_dict=mean_vector_dict,
                    total_mean_vector="total" in title,
                    mean_vector_type="exc",
                    flatten_axes=flatten_axes,
                )
        elif any("inh_rf" in value for value in exc_rfs_dict.values()):
            _plot_receptive_fields(
                save_path=syn_act_rfs_path,
                fig_format=self.fig_save_format,
                title="exc_cell_inh_syn_pref_act_rfs",
                rfs_dict=exc_rfs_dict,
                synapse_type="inh",
                mean_vector_dict=mean_vector_dict,
                total_mean_vector=False,
                mean_vector_type="inh",
            )
            _plot_receptive_fields(
                save_path=syn_act_rfs_path,
                fig_format=self.fig_save_format,
                title="exc_cell_inh_syn_total_act_rfs",
                rfs_dict=exc_rfs_dict,
                synapse_type="inh",
                mean_vector_dict=mean_vector_dict,
                total_mean_vector=True,
                mean_vector_type="inh",
            )

    def _discriminability_plot_inputs(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Prefer raw inputs for class-example/d-prime plots when they match RF shape."""
        if self.exc_rf_shape is None:
            return None
        n_features = prod(self.exc_rf_shape)
        if x.shape[-1] == n_features:
            return x
        if self.exc_inputs is not None and self.exc_inputs.shape[-1] == n_features:
            return self.exc_inputs
        return None

    def analyze(
        self,
        model: BaseModel,
        test_dataset: Optional[torch.utils.data.Dataset] = None,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ) -> Optional[dict[str, object]]:
        """Analyze receptive fields for models with dendritic branch layers."""
        if not any(iter_named_modules_of_type(model, DendriticBranchLayer)):
            self.logger.error(
                "Model must expose DendriticBranchLayer modules "
                "for receptive field analysis"
            )
            return None

        self.inh_rf_shape = self._infer_inh_rf_shape(model.core_network)

        loaded = self._prepare_raw_data(model, save_path, filename)
        if not self.raw_dict:
            self.logger.warning("No dendritic branch receptive fields were found")
            return None

        results = self._build_results(loaded)
        self._maybe_add_activation_tuning(
            results=results,
            model=model,
            test_dataset=test_dataset,
            device=device,
            runtime=runtime,
        )
        self._maybe_add_component_rfs(
            results=results,
            model=model,
            test_dataset=test_dataset,
            device=device,
            runtime=runtime,
            save_path=save_path,
            filename=filename,
            training=training,
        )
        self._maybe_add_rf_tuning(
            results=results,
            model=model,
            test_dataset=test_dataset,
            device=device,
            runtime=runtime,
            save_path=save_path,
            training=training,
        )

        if save_path is not None:
            self._save_results(results, save_path, filename, training)

        return results

    def _prepare_raw_data(
        self,
        model: BaseModel,
        save_path: Optional[str],
        filename: str,
    ) -> bool:
        loaded = save_path is not None and self._load_raw_checkpoint(
            save_path, filename
        )
        if loaded:
            return True
        self.collect_raw_data(model)
        if save_path is not None:
            self._save_raw_checkpoint(save_path, filename)
        return False

    def _build_results(self, raw_checkpoint_loaded: bool) -> dict[str, object]:
        return {
            "n_receptive_fields": len(self.raw_dict),
            "raw_checkpoint_loaded": raw_checkpoint_loaded,
        }

    def _maybe_add_activation_tuning(
        self,
        results: dict[str, object],
        model: BaseModel,
        test_dataset: Optional[torch.utils.data.Dataset],
        device: str,
        runtime: Optional[EvaluationRuntimeConfig],
    ) -> None:
        if not self.compute_activation_tuning:
            return
        if test_dataset is None:
            self.logger.warning(
                "Activation tuning requested but no test_dataset was provided"
            )
            return
        results["activation_tuning"] = self._compute_activation_tuning(
            model=model,
            test_dataset=test_dataset,
            device=device,
            runtime=runtime,
        )
        if self.activation_sample_records:
            results["activation_samples"] = {
                "selection_depends_on_activations": False,
                "selection_depends_on_labels": True,
                "selection_rule": "uniform_without_replacement_within_class",
                "signals": list(self.activation_sample_signals),
                "depths": list(self.activation_sample_depths),
                "branch_indices": list(self.activation_sample_branch_indices),
                "branch_indices_by_depth": {
                    str(depth): list(indices)
                    for depth, indices in sorted(
                        self.activation_sample_branch_indices_by_depth.items()
                    )
                },
                "requested_samples_per_class": self.activation_samples_per_class,
                "seed": self.activation_sample_seed,
                "records": self.activation_sample_records,
            }

    def _maybe_add_component_rfs(
        self,
        results: dict[str, object],
        model: BaseModel,
        test_dataset: Optional[torch.utils.data.Dataset],
        device: str,
        runtime: Optional[EvaluationRuntimeConfig],
        save_path: Optional[str],
        filename: str,
        training: bool,
    ) -> None:
        if not self.compute_component_rfs:
            return
        if test_dataset is None:
            self.logger.warning(
                "Component RFs requested but no test_dataset was provided"
            )
            return

        with analysis_device_context(model, device) as analysis_device:
            items = materialize_dataset(
                test_dataset,
                runtime,
                device=analysis_device,
            )
            x: torch.Tensor = items[0].to(analysis_device)
            labels: torch.Tensor = items[1].to(analysis_device)
            component_rfs = self._compute_component_receptive_fields(
                model=model,
                x=x,
                labels=labels,
                save_path=save_path,
                filename=filename,
                training=training,
            )

        if component_rfs is not None:
            results["component_receptive_fields"] = component_rfs

    def _maybe_add_rf_tuning(
        self,
        results: dict[str, object],
        model: BaseModel,
        test_dataset: Optional[torch.utils.data.Dataset],
        device: str,
        runtime: Optional[EvaluationRuntimeConfig],
        save_path: Optional[str],
        training: bool,
    ) -> None:
        if not self.compute_rf_tuning:
            return
        if test_dataset is None:
            self.logger.warning("RF tuning requested but no test_dataset was provided")
            return

        with analysis_device_context(model, device) as analysis_device:
            items = materialize_dataset(
                test_dataset,
                runtime,
                device=analysis_device,
            )
            x: torch.Tensor = items[0].to(analysis_device)
            labels: torch.Tensor = items[1].to(analysis_device)
            rf_tuning = self._compute_rf_tuning_analysis(
                model=model,
                x=x,
                labels=labels,
                save_path=save_path,
                training=training,
            )

        if rf_tuning is not None:
            results["rf_tuning"] = rf_tuning

    def _save_results(
        self,
        results: dict[str, object],
        save_path: str,
        filename: str,
        training: bool,
    ) -> None:
        out_path = os.path.join(save_path, "epochs") if training else save_path
        os.makedirs(out_path, exist_ok=True)
        save_dict(results, out_path, f"{filename}_summary.json")
        if "activation_tuning" in results:
            save_dict(
                results["activation_tuning"],
                out_path,
                f"{filename}_activation_tuning.json",
            )
        if "activation_samples" in results:
            save_dict(
                results["activation_samples"],
                out_path,
                f"{filename}_activation_samples.json",
            )
        if "rf_tuning" in results:
            rf_tuning = results["rf_tuning"]
            save_dict(
                rf_tuning,
                out_path,
                f"{filename}_rf_tuning.json",
            )
            # Preserve the training artifact consumed by RF-vs-training plots.
            if "agg_tuning" in rf_tuning:
                save_dict(rf_tuning["agg_tuning"], out_path, f"{filename}.json")
        if "component_receptive_fields" in results:
            save_dict(
                results["component_receptive_fields"],
                out_path,
                f"{filename}_component_receptive_fields.json",
            )
        if not training:
            self._plot_all_receptive_fields(save_path)

    def _plot_all_receptive_fields(self, save_path: str) -> None:
        for synapse_type in ("exc", "inh"):
            available = self._available_receptive_field_indices(synapse_type)
            if not available:
                self.logger.warning(
                    "No %s receptive fields matched the requested shape",
                    synapse_type,
                )
                continue
            ixs = self._sample_receptive_field_indices(available)
            self._plot_receptive_fields(ixs, synapse_type, save_path)

    def _available_receptive_field_indices(self, synapse_type: str) -> list[int]:
        return [
            ix
            for ix, item in self.raw_dict.items()
            if item.get(synapse_type) is not None
        ]

    def _sample_receptive_field_indices(self, available: list[int]) -> list[int]:
        n_branches = min(self.n_rows * self.n_cols, len(available))
        perm = torch.randperm(len(available))[:n_branches].sort()[0].tolist()
        return [available[ix] for ix in perm]

    def _plot_receptive_fields(self, ixs: list[int], synapse_type: str, save_path: str):
        """Plot receptive fields."""
        mins = []
        maxs = []
        for ix in ixs:
            data = self.raw_dict[ix][synapse_type]
            mins.append(float(torch.min(data)))
            maxs.append(float(torch.max(data)))
        vmin = min(mins)
        vmax = max(maxs)

        fig, axes = plt.subplots(
            self.n_rows, self.n_cols, figsize=(5 * self.n_cols, 5 * self.n_rows)
        )

        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])

        axes_flat = np.atleast_1d(axes).flatten()

        for ax in axes_flat:
            ax.set_axis_off()

        for ix, ax in zip(ixs, axes_flat):
            ax.set_axis_on()
            data = self.raw_dict[ix][synapse_type]
            sns.heatmap(
                data,
                ax=ax,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                cbar_ax=cax,
                square=True,
                xticklabels=False,
                yticklabels=False,
            )
            depth = self.raw_dict[ix]["depth"]
            branch_idx = self.raw_dict[ix]["branch_idx"]
            ax.set_title(f"Depth {depth}, Branch {branch_idx}")

        fig.suptitle(f"{synapse_type.title()} Receptive Fields")
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(
            os.path.join(
                save_path,
                f"{synapse_type}_receptive_fields.{self.fig_save_format}",
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def _location_key_to_tuple(location_key: str) -> tuple[int, int, int]:
    """Convert ``soma X - depth Y - branch Z`` keys to numeric tuples."""
    loc = location_key.replace("soma", "").replace("depth", "").replace("branch", "")
    loc = loc.replace(" ", "")
    return tuple(int(x) for x in loc.split("-"))


def _soma_sort_key(soma_key: str) -> int:
    return int(soma_key.replace("soma", "").strip())


def _cosine_for_mask(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    left_masked = np.asarray(left[mask], dtype=float)
    right_masked = np.asarray(right[mask], dtype=float)
    denom = np.linalg.norm(left_masked) * np.linalg.norm(right_masked)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(left_masked, right_masked) / denom)


def _plot_agg_tuning_final(
    save_path: str,
    fig_format: str,
    agg_tuning_dict: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Grouped bars of cosine similarity between aggregate RFs and class maps."""
    exc_data = agg_tuning_dict.get("exc") or {}
    inh_data = agg_tuning_dict.get("inh") or {}
    if not exc_data and not inh_data:
        return

    soma_keys = sorted(set(exc_data) | set(inh_data), key=_soma_sort_key)
    if not soma_keys:
        return

    x = np.arange(len(soma_keys), dtype=float)
    width = 0.18
    offsets = (np.arange(4) - 1.5) * width
    rpb_str = r"$\rho_{PB}$"
    specs = (
        ("exc", "pos", "#2E7FD9", None, f"exc pos {rpb_str}"),
        ("exc", "neg", "#D93E3E", None, f"exc neg {rpb_str}"),
        ("inh", "pos", "#2E7FD9", "///", f"inh pos {rpb_str}"),
        ("inh", "neg", "#D93E3E", "///", f"inh neg {rpb_str}"),
    )

    fig, ax = plt.subplots(1, 1, figsize=(max(7.0, 1.4 * len(soma_keys)), 5))
    hatch_line_color = (1.0, 1.0, 1.0, 0.72)
    with plt.rc_context({"hatch.linewidth": 3.0}):
        for idx, (syn_type, sign, color, hatch, label) in enumerate(specs):
            data = exc_data if syn_type == "exc" else inh_data
            vals = [float(data.get(sk, {}).get(sign, float("nan"))) for sk in soma_keys]
            ax.bar(
                x + offsets[idx],
                vals,
                width,
                label=label,
                facecolor=color,
                edgecolor=hatch_line_color if hatch else "none",
                hatch=hatch,
                linewidth=0.0,
            )

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor=hatch_line_color if hatch else "none",
            hatch=hatch,
            linewidth=0.0,
            label=label,
        )
        for _syn_type, _sign, color, hatch, label in specs
    ]
    ax.set_ylabel("Cosine Similarity")
    ax.set_xlabel("Soma")
    ax.set_xticks(x)
    ax.set_xticklabels([str(_soma_sort_key(sk)) for sk in soma_keys])
    ax.axhline(0.0, color="0.6", linewidth=0.8)
    ax.set_ylim(0.0, 1.0)
    ax.legend(handles=legend_handles, loc="best", framealpha=0.92)
    fig.suptitle(
        "Excitatory and Inhibitory Feature Tuning to Input Point-Biserial Correlation",
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"feature_tuning_final.{fig_format}"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_agg_tuning_vs_training(
    epochs_data: dict[str, dict[str, dict[str, list[float]]]],
    epoch_numbers: list[int],
    save_path: str,
    fig_format: str,
) -> None:
    exc_data = epochs_data.get("exc") or {}
    inh_data = epochs_data.get("inh") or {}
    soma_keys = sorted(set(exc_data) | set(inh_data), key=_soma_sort_key)
    if not soma_keys:
        return

    fig, axes = plt.subplots(
        1,
        len(soma_keys),
        figsize=(6 * len(soma_keys), 6),
        squeeze=False,
    )
    rpb_str = r"$\rho_{PB}$"
    for ax, soma_key in zip(axes.ravel(), soma_keys):
        for syn_type, sign in (
            ("exc", "pos"),
            ("exc", "neg"),
            ("inh", "pos"),
            ("inh", "neg"),
        ):
            data = exc_data if syn_type == "exc" else inh_data
            vals = data.get(soma_key, {}).get(sign)
            if not vals:
                continue
            ax.plot(
                epoch_numbers[: len(vals)],
                vals,
                linestyle="-" if syn_type == "exc" else "--",
                color="#2E7FD9" if sign == "pos" else "#D93E3E",
                label=f"{syn_type} {sign} {rpb_str}",
            )
        ax.set_ylim(0.0, 1.0)
        ax.set_title(soma_key.title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cosine Similarity")
        ax.legend()

    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"feature_tuning_vs_training_by_soma.{fig_format}"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_confusion_and_entropies(
    confusion_matrix: np.ndarray,
    entropy_dict: dict[str, np.ndarray],
    save_path: str,
    fig_format: str,
) -> None:
    """Plot confusion matrix and entropy by classification outcome."""
    fig, (ax_cm, ax_ent) = plt.subplots(
        1, 2, figsize=(10.5, 5), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    cm = np.asarray(confusion_matrix, dtype=float)
    if cm.size:
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 6},
            cmap="inferno",
            ax=ax_cm,
            vmin=0,
            cbar_kws={"label": "Ratio of Samples"},
        )
        ax_cm.set_xlabel("Predicted class")
        ax_cm.set_ylabel("True class")
    else:
        ax_cm.text(0.5, 0.5, "empty confusion matrix", ha="center", va="center")

    rows: list[dict[str, str | float]] = []
    for outcome in ("correct", "incorrect"):
        for value in np.asarray(entropy_dict.get(outcome, []), dtype=float).ravel():
            if np.isfinite(value):
                rows.append({"outcome": outcome, "entropy": float(value)})
    if rows:
        sns.boxplot(
            data=pd.DataFrame(rows),
            x="outcome",
            y="entropy",
            order=["correct", "incorrect"],
            ax=ax_ent,
            width=0.55,
        )
    else:
        ax_ent.text(0.5, 0.5, "no entropy data", ha="center", va="center")
    ax_ent.set_ylabel("Entropy")
    ax_ent.set_xlabel("Classification outcome")

    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"confusion_and_entropies.{fig_format}"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_class_examples(
    save_path: str,
    fig_format: str,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    shape: Optional[tuple[int, int]],
    n_examples: int = 5,
) -> None:
    if shape is None or inputs.shape[-1] != prod(shape):
        return

    unique_labels = torch.unique(labels)
    n_classes = len(unique_labels)
    if n_classes == 0:
        return
    n_examples = max(1, n_examples)

    fig, axes = plt.subplots(
        n_examples,
        n_classes,
        figsize=(4 * n_classes, 4 * n_examples),
        squeeze=False,
    )
    for class_col, label in enumerate(unique_labels):
        class_inputs = inputs[labels == label]
        if class_inputs.numel() == 0:
            continue
        ex_ixs = torch.randperm(class_inputs.shape[0])[:n_examples]
        ex_inputs = class_inputs[ex_ixs]
        for ex_idx in range(n_examples):
            ax = axes[ex_idx, class_col]
            source_idx = min(ex_idx, ex_inputs.shape[0] - 1)
            shaped_input = ex_inputs[source_idx].reshape(*shape).detach().cpu().numpy()
            sns.heatmap(
                shaped_input,
                ax=ax,
                cmap="Greys",
                vmin=0,
                vmax=1,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if ex_idx == 0:
                ax.set_title(f"Class {int(label.item())}")

    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"class_examples.{fig_format}"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _compute_dprime(
    inputs: torch.Tensor, labels: torch.Tensor
) -> dict[str, np.ndarray]:
    dprime_dict: dict[str, np.ndarray] = {}
    n_samples = inputs.shape[0]
    total_mean = inputs.mean(dim=0)
    total_var = inputs.var(dim=0)

    for label in torch.unique(labels):
        class_idx = int(label.item())
        class_mask = labels == label
        pref_inputs = inputs[class_mask]
        unpref_inputs = inputs[~class_mask]
        if pref_inputs.numel() == 0 or unpref_inputs.numel() == 0:
            continue

        pref_prob = pref_inputs.shape[0] / n_samples
        unpref_prob = unpref_inputs.shape[0] / n_samples
        numer = pref_prob * (pref_inputs.mean(dim=0) - total_mean) ** 2
        numer = numer + unpref_prob * (unpref_inputs.mean(dim=0) - total_mean) ** 2
        dprime = (numer / (total_var + 1e-9)).sqrt()
        dprime_dict[f"class_{class_idx}"] = dprime.detach().cpu().numpy()

    return dprime_dict


def _compute_point_biserial_correlation(
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, np.ndarray]:
    return ReceptiveFieldAnalyzer._compute_point_biserial_correlation(inputs, labels)


def _plot_discrim_metric_heatmaps(
    save_path: str,
    fig_format: str,
    metric_dict: dict[str, np.ndarray],
    metric_name: str,
    shape: Optional[tuple[int, int]],
    cmap: str,
    vmin: float,
    vmax: float,
    center: Optional[float] = None,
) -> None:
    if shape is None or not metric_dict:
        return
    values = list(metric_dict.values())
    if not values or values[0].size != prod(shape):
        return

    n_classes = len(metric_dict)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4), squeeze=False)
    axes_flat = axes.ravel()
    for ax, (class_key, value_vec) in zip(axes_flat, metric_dict.items()):
        shaped_vec = np.asarray(value_vec).reshape(*shape)
        sns.heatmap(
            shaped_vec,
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            cbar=ax is axes_flat[-1],
            xticklabels=False,
            yticklabels=False,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(class_key.replace("_", " ").title())

    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"class_{metric_name}_heatmaps.{fig_format}"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_class_discriminability_heatmaps(
    save_path: str,
    fig_format: str,
    inputs: Optional[torch.Tensor],
    labels: torch.Tensor,
    shape: Optional[tuple[int, int]],
) -> None:
    if inputs is None or shape is None or inputs.shape[-1] != prod(shape):
        return
    dprime_dict = _compute_dprime(inputs, labels)
    rpb_dict = _compute_point_biserial_correlation(inputs, labels)
    _plot_discrim_metric_heatmaps(
        save_path=save_path,
        fig_format=fig_format,
        metric_dict=dprime_dict,
        metric_name="dprime",
        shape=shape,
        cmap="binary",
        vmin=0,
        vmax=1,
    )
    _plot_discrim_metric_heatmaps(
        save_path=save_path,
        fig_format=fig_format,
        metric_dict=rpb_dict,
        metric_name="rpb",
        shape=shape,
        cmap="seismic_r",
        vmin=-1,
        vmax=1,
        center=0,
    )


def _plot_receptive_fields(
    save_path: str,
    fig_format: str,
    title: str,
    rfs_dict: dict[str, dict[str, np.ndarray]],
    synapse_type: str,
    mean_vector_dict: Optional[dict[str, dict[str, np.ndarray]]] = None,
    total_mean_vector: bool = False,
    mean_vector_type: Optional[str] = None,
    flatten_axes: bool = False,
) -> None:
    plot_items = [
        (location_key, branch_rfs[f"{synapse_type}_rf"])
        for location_key, branch_rfs in rfs_dict.items()
        if f"{synapse_type}_rf" in branch_rfs
    ]
    if not plot_items:
        return

    n_total = len(plot_items)
    if flatten_axes:
        n_rows = 1
        n_cols = n_total
    else:
        n_rows = 1
        n_cols = 1
        while n_rows * n_cols < n_total:
            if n_rows < n_cols:
                n_rows += 1
            else:
                n_cols += 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, (location_key, rf_value) in zip(axes_flat, plot_items):
        rf = np.asarray(rf_value, dtype=float)
        if mean_vector_dict is not None and mean_vector_type is not None:
            if "depth" in location_key and "branch" in location_key:
                soma_idx = _location_key_to_tuple(location_key)[0]
            else:
                soma_idx = _soma_sort_key(location_key)
            class_key = "total" if total_mean_vector else f"class_{soma_idx}"
            input_mean_vec = mean_vector_dict.get(class_key, {}).get(mean_vector_type)
            if (
                input_mean_vec is not None
                and np.asarray(input_mean_vec).size == rf.size
            ):
                rf = rf * np.asarray(input_mean_vec).reshape(rf.shape)

        sns.heatmap(rf, ax=ax, cmap="inferno", xticklabels=False, yticklabels=False)
        ax.set_title(location_key.title())

    for ax in axes_flat[n_total:]:
        ax.set_visible(False)

    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"{title}.{fig_format}"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_vout_activation_boxplots(
    data_dict: Optional[dict[str, dict[str, dict[str, list[float]]]]],
    save_path: str,
    fig_format: str,
    cell_type: str,
) -> None:
    if not data_dict:
        return

    branch_keys = list(data_dict.keys())
    class_order = sorted(
        {
            int(class_key.split("_")[1])
            for branch_data in data_dict.values()
            for class_key in branch_data
            if class_key.startswith("class_")
        }
    )
    if not class_order:
        return

    n_total = len(branch_keys)
    n_rows = 1
    n_cols = 1
    while n_rows * n_cols < n_total:
        if n_rows < n_cols:
            n_rows += 1
        else:
            n_cols += 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for ax, branch_key in zip(axes_flat, branch_keys):
        rows: list[dict[str, int | float]] = []
        for class_idx in class_order:
            class_data = data_dict[branch_key].get(f"class_{class_idx}", {})
            for value in class_data.get("vout", []):
                rows.append({"Class": class_idx, "vout": float(value)})
        if rows:
            sns.violinplot(
                data=pd.DataFrame(rows),
                x="Class",
                y="vout",
                ax=ax,
                order=class_order,
                color="0.88",
                cut=0,
                linewidth=0.8,
            )
        ax.set_xlabel("Class")
        ax.set_ylabel(r"$V_{\mathrm{out}}$")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(branch_key.title())

    for ax in axes_flat[n_total:]:
        ax.set_visible(False)

    cell_label = "excitatory" if cell_type == "exc" else "inhibitory"
    fig.suptitle(
        rf"$V_{{\mathrm{{out}}}}$ by input class ({cell_label} cell compartments)",
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(
            save_path, f"{cell_type}_vout_activation_by_class_violins.{fig_format}"
        ),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_pref_unpref_scatter_by_compartment(
    pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]],
    xvar: str,
    yvar: str,
    x_logscale: bool,
    save_path: str,
    fig_format: str,
    fig_basename: str,
    reference_line: Optional[str] = None,
) -> None:
    def _linear_pad(lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
            return 0.0, 1.0
        if hi == lo:
            margin = 0.05 * abs(lo if lo != 0 else 1.0)
            return lo - margin, hi + margin
        pad = (hi - lo) * frac
        return lo - pad, hi + pad

    def _log_pad(lo: float, hi: float, frac: float = 0.05) -> tuple[float, float]:
        if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0 or hi < lo:
            return 1e-6, 1.0
        if hi / lo < 1 + 1e-12:
            return lo * 0.9, hi * 1.1
        log_lo = math.log10(lo)
        log_hi = math.log10(hi)
        span = log_hi - log_lo
        return 10 ** (log_lo - frac * span), 10 ** (log_hi + frac * span)

    def _kde_curve_x(ax_marg: Any, xv: np.ndarray, color: str) -> None:
        if xv.size < 2:
            return
        try:
            sns.kdeplot(
                x=xv,
                ax=ax_marg,
                color=color,
                fill=True,
                linewidth=2,
                alpha=0.35,
                warn_singular=False,
            )
        except Exception:
            sns.histplot(x=xv, ax=ax_marg, color=color, stat="density", alpha=0.35)

    def _kde_curve_y(ax_marg: Any, yv: np.ndarray, color: str) -> None:
        if yv.size < 2:
            return
        try:
            sns.kdeplot(
                y=yv, ax=ax_marg, color=color, fill=True, linewidth=2, alpha=0.35
            )
        except Exception:
            sns.histplot(y=yv, ax=ax_marg, color=color, stat="density", alpha=0.35)

    branch_keys = [
        branch_key
        for branch_key, branch_data in pref_unpref_data.items()
        if xvar in branch_data["preferred"] and yvar in branch_data["preferred"]
    ]
    if not branch_keys:
        return

    n_total = len(branch_keys)
    n_rows = 1
    n_cols = 1
    while n_rows * n_cols < n_total:
        if n_rows < n_cols:
            n_rows += 1
        else:
            n_cols += 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 5.5 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for ax, branch_key in zip(axes_flat, branch_keys):
        branch_data = pref_unpref_data[branch_key]
        pref = branch_data["preferred"]
        unpref = branch_data["unpreferred"]
        xp = np.asarray(pref.get(xvar, []), dtype=float)
        yp = np.asarray(pref.get(yvar, []), dtype=float)
        xu = np.asarray(unpref.get(xvar, []), dtype=float)
        yu = np.asarray(unpref.get(yvar, []), dtype=float)

        mp = np.isfinite(xp) & np.isfinite(yp)
        mu = np.isfinite(xu) & np.isfinite(yu)
        if x_logscale:
            mp = mp & (xp > 0)
            mu = mu & (xu > 0)
        if not mp.any() and not mu.any():
            ax.set_visible(False)
            continue

        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="20%", pad=0.06, sharex=ax)
        ax_right = divider.append_axes("right", size="20%", pad=0.06, sharey=ax)
        ax_top.set_axis_off()
        ax_right.set_axis_off()

        xs = np.concatenate([arr for arr in (xp[mp], xu[mu]) if arr.size])
        ys = np.concatenate([arr for arr in (yp[mp], yu[mu]) if arr.size])
        if x_logscale:
            ax.set_xscale("log")
            x_lo, x_hi = _log_pad(float(xs.min()), float(xs.max()))
        else:
            x_lo, x_hi = _linear_pad(float(xs.min()), float(xs.max()))
        y_lo, y_hi = _linear_pad(float(ys.min()), float(ys.max()))

        if mu.any():
            ax.scatter(
                xu[mu],
                yu[mu],
                c="black",
                s=12,
                alpha=0.45,
                edgecolors="none",
                label="unpreferred",
            )
            _kde_curve_x(ax_top, xu[mu], "black")
            _kde_curve_y(ax_right, yu[mu], "black")
        if mp.any():
            ax.scatter(
                xp[mp],
                yp[mp],
                c="red",
                s=12,
                alpha=0.45,
                edgecolors="none",
                label="preferred",
            )
            _kde_curve_x(ax_top, xp[mp], "red")
            _kde_curve_y(ax_right, yp[mp], "red")

        ax.set_xlabel(xvar.replace("_", " ").title())
        ax.set_ylabel(yvar.replace("_", " ").title())
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ref_kw = {"color": "green", "linestyle": "--", "linewidth": 3}
        if reference_line == "y=x":
            lo = max(x_lo, y_lo)
            hi = min(x_hi, y_hi)
            if hi > lo:
                ax.plot([lo, hi], [lo, hi], **ref_kw)
        elif reference_line == "x=1":
            ax.axvline(1.0, **ref_kw)
        ax_top.set_title(branch_key.title(), fontsize=10, pad=4)
        ax.legend(loc="best", fontsize=8)

    for ax in axes_flat[n_total:]:
        ax.set_visible(False)

    fig.subplots_adjust(
        left=0.06, right=0.96, top=0.94, bottom=0.06, wspace=0.28, hspace=0.36
    )
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, f"{fig_basename}.{fig_format}"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _plot_exc_vs_inh_by_compartment(
    pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]],
    save_path: str,
    fig_format: str,
) -> None:
    _plot_pref_unpref_scatter_by_compartment(
        pref_unpref_data,
        xvar="inh",
        yvar="exc",
        x_logscale=False,
        save_path=save_path,
        fig_format=fig_format,
        fig_basename="exc_vs_inh_by_compartment_pref_unpref",
        reference_line="y=x",
    )


def _plot_vout_vs_eiratio_by_compartment(
    pref_unpref_data: dict[str, dict[str, dict[str, list[float]]]],
    save_path: str,
    fig_format: str,
) -> None:
    _plot_pref_unpref_scatter_by_compartment(
        pref_unpref_data,
        xvar="ei_ratio",
        yvar="vout",
        x_logscale=True,
        save_path=save_path,
        fig_format=fig_format,
        fig_basename="vout_vs_eiratio_by_compartment_pref_unpref",
        reference_line="x=1",
    )
