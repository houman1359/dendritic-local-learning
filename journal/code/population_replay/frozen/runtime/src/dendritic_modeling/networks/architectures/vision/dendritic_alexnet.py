"""AlexNet-shaped network built from hierarchical dendritic E/I layers."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
    HierarchicalDendriticConv,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
)
from dendritic_modeling.networks.architectures.replacement import (
    PopulationToFeatureReduction,
    SpatialNonNegativeInputAdapter,
    make_nonmixing_output_adapter,
)
from dendritic_modeling.networks.base import BaseNetwork


def _int_list(value: Any, *, name: str, length: int | None = None) -> list[int]:
    result = [int(item) for item in value]
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(result)}")
    return result


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = tuple(int(item) for item in value)
        if len(items) != 2:
            raise ValueError(f"Expected a scalar or pair, got {value!r}")
        return items
    scalar = int(value)
    return (scalar, scalar)


def _scale_list(value: Any, *, name: str, length: int) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = [float(item) for item in value]
        if len(result) != length:
            raise ValueError(f"{name} must contain {length} values, got {len(result)}")
        return result
    return [float(value)] * length


def _count_list(value: Any, *, name: str, length: int) -> list[int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
        if len(raw) != length:
            raise ValueError(f"{name} must contain {length} values, got {len(raw)}")
    else:
        raw = [value] * length
    result: list[int] = []
    for item in raw:
        count = int(item)
        if count < 1 or float(item) != count:
            raise ValueError(f"{name} values must be positive integers")
        result.append(count)
    return result


def _block_config(
    reference: dict[str, Any],
    *,
    excitatory_size: int,
    inhibitory_size: int,
    seed: int,
) -> dict[str, Any]:
    config = copy.deepcopy(reference)
    architecture = config.setdefault("architecture", {})
    architecture["excitatory_layer_sizes"] = [int(excitatory_size)]
    architecture["inhibitory_layer_sizes"] = [int(inhibitory_size)]
    config["initialization_seed"] = int(seed)
    indexed = config.setdefault("sparsity", {}).setdefault("indexed", {})
    indexed["seed"] = int(seed)
    structured = config.setdefault("connectivity", {}).get("structured")
    if isinstance(structured, dict):
        structured["seed"] = int(seed)
    return config


class DendriticAlexNet(BaseNetwork):
    """AlexNet tensor geometry with dendritic convolutional and FC layers.

    By default, each spatial block emits one excitatory soma per output
    channel. Configured soma groups can instead be reduced within channel by a
    fixed mean or a learned positive simplex. The final dendritic layer always
    emits one excitatory soma per class, so no dense signed classifier is
    hidden behind the dendritic network.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        input_channels: int = 3,
    ):
        super().__init__()
        vision = copy.deepcopy(dict(config.get("vision", {})))
        channels = _int_list(
            vision.get("channels", [64, 192, 384, 256, 256]),
            name="vision.channels",
            length=5,
        )
        kernels = _int_list(
            vision.get("kernels", [11, 5, 3, 3, 3]),
            name="vision.kernels",
            length=5,
        )
        strides = _int_list(
            vision.get("strides", [4, 1, 1, 1, 1]),
            name="vision.strides",
            length=5,
        )
        paddings = _int_list(
            vision.get("paddings", [2, 2, 1, 1, 1]),
            name="vision.paddings",
            length=5,
        )
        classifier_widths = _int_list(
            vision.get("classifier_widths", [4096, 4096]),
            name="vision.classifier_widths",
            length=2,
        )
        pool_after = {int(index) for index in vision.get("pool_after", [0, 1, 4])}
        if not pool_after.issubset(set(range(5))):
            raise ValueError("vision.pool_after indices must be between 0 and 4")
        pool_kernel = _pair(vision.get("pool_kernel", 3))
        pool_stride = _pair(vision.get("pool_stride", 2))
        adaptive_shape = _pair(vision.get("adaptive_pool_shape", [6, 6]))
        inhibitory_ratio = float(vision.get("inhibitory_ratio", 0.2))
        if inhibitory_ratio < 0:
            raise ValueError("vision.inhibitory_ratio must be non-negative")
        num_classes = int(vision.get("num_classes", 1000))
        if num_classes < 2:
            raise ValueError("vision.num_classes must be at least 2")
        dropout = float(vision.get("dropout", 0.5))
        if not 0 <= dropout < 1:
            raise ValueError("vision.dropout must be in [0, 1)")
        seed = int(vision.get("seed", config.get("initialization_seed", 0) or 0))
        output_adapter = dict(vision.get("output_adapter", {}))
        output_adapter_mode = str(output_adapter.get("mode", "identity"))
        spatial_initial_scales = _scale_list(
            output_adapter.get("spatial_initial_scale", 1.0),
            name="vision.output_adapter.spatial_initial_scale",
            length=5,
        )
        classifier_initial_scales = _scale_list(
            output_adapter.get("classifier_initial_scale", 1.0),
            name="vision.output_adapter.classifier_initial_scale",
            length=2,
        )
        spatial_initial_thresholds = _scale_list(
            output_adapter.get("spatial_initial_threshold", 0.25),
            name="vision.output_adapter.spatial_initial_threshold",
            length=5,
        )
        classifier_initial_thresholds = _scale_list(
            output_adapter.get("classifier_initial_threshold", 0.25),
            name="vision.output_adapter.classifier_initial_threshold",
            length=2,
        )
        output_organization = dict(vision.get("output_organization", {}))
        population_reduction_mode = str(output_organization.get("mode", "one_to_one"))
        spatial_somas_per_channel = _count_list(
            output_organization.get("spatial_somas_per_channel", 1),
            name="vision.output_organization.spatial_somas_per_channel",
            length=5,
        )
        classifier_somas_per_unit = _count_list(
            output_organization.get("classifier_somas_per_unit", 1),
            name="vision.output_organization.classifier_somas_per_unit",
            length=2,
        )

        self.input_adapter = SpatialNonNegativeInputAdapter(
            int(input_channels),
            str(vision.get("input_transform", "signed_split")),
        )
        self.input_channels = int(input_channels)
        self.num_classes = num_classes
        self.output_dim = num_classes
        self.channel_widths = tuple(channels)
        self.component_seeds = tuple(seed + index for index in range(8))

        feature_modules: list[nn.Module] = []
        current_channels = self.input_adapter.output_channels
        self.spatial_blocks = nn.ModuleList()
        self.spatial_population_reducers = nn.ModuleList()
        self.spatial_output_adapters = nn.ModuleList()
        for index, (
            out_channels,
            kernel,
            stride,
            padding,
            somas_per_channel,
        ) in enumerate(
            zip(
                channels,
                kernels,
                strides,
                paddings,
                spatial_somas_per_channel,
                strict=True,
            )
        ):
            population_channels = out_channels * somas_per_channel
            inhibitory_size = (
                0
                if inhibitory_ratio == 0
                else max(1, round(population_channels * inhibitory_ratio))
            )
            block = HierarchicalDendriticConv(
                config=_block_config(
                    config,
                    excitatory_size=population_channels,
                    inhibitory_size=inhibitory_size,
                    seed=self.component_seeds[index],
                ),
                in_channels=current_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
            self.spatial_blocks.append(block)
            reducer = PopulationToFeatureReduction(
                out_channels,
                somas_per_feature=somas_per_channel,
                feature_axis=1,
                mode=population_reduction_mode,
            )
            self.spatial_population_reducers.append(reducer)
            adapter = make_nonmixing_output_adapter(
                num_features=out_channels,
                feature_axis=1,
                mode=output_adapter_mode,
                initial_scale=spatial_initial_scales[index],
                initial_threshold=spatial_initial_thresholds[index],
            )
            self.spatial_output_adapters.append(adapter)
            feature_modules.append(block)
            if somas_per_channel != 1 or population_reduction_mode != "one_to_one":
                feature_modules.append(reducer)
            feature_modules.append(adapter)
            if index in pool_after:
                feature_modules.append(
                    nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
                )
            current_channels = out_channels
        self.features = nn.Sequential(*feature_modules)
        self.avgpool = nn.AdaptiveAvgPool2d(adaptive_shape)

        flat_dim = current_channels * adaptive_shape[0] * adaptive_shape[1]
        fc_sizes = [*classifier_widths, num_classes]
        fc_modules: list[nn.Module] = []
        self.feedforward_blocks = nn.ModuleList()
        self.feedforward_population_reducers = nn.ModuleList()
        self.feedforward_output_adapters = nn.ModuleList()
        current_dim = flat_dim
        for index, out_dim in enumerate(fc_sizes):
            somas_per_unit = (
                classifier_somas_per_unit[index]
                if index < len(classifier_widths)
                else 1
            )
            population_dim = out_dim * somas_per_unit
            inhibitory_size = (
                0
                if inhibitory_ratio == 0
                else max(1, round(population_dim * inhibitory_ratio))
            )
            block = ConfigurableEINetwork(
                config=_block_config(
                    config,
                    excitatory_size=population_dim,
                    inhibitory_size=inhibitory_size,
                    seed=self.component_seeds[5 + index],
                ),
                input_dim=current_dim,
            )
            self.feedforward_blocks.append(block)
            if index < len(classifier_widths):
                fc_modules.append(nn.Dropout(p=dropout))
            fc_modules.append(block)
            reducer = PopulationToFeatureReduction(
                out_dim,
                somas_per_feature=somas_per_unit,
                feature_axis=-1,
                mode=(
                    population_reduction_mode
                    if index < len(classifier_widths)
                    else "one_to_one"
                ),
            )
            self.feedforward_population_reducers.append(reducer)
            if somas_per_unit != 1 or (
                index < len(classifier_widths)
                and population_reduction_mode != "one_to_one"
            ):
                fc_modules.append(reducer)
            if index < len(classifier_widths):
                adapter = make_nonmixing_output_adapter(
                    num_features=out_dim,
                    feature_axis=-1,
                    mode=output_adapter_mode,
                    initial_scale=classifier_initial_scales[index],
                    initial_threshold=classifier_initial_thresholds[index],
                )
            else:
                adapter = nn.Identity()
            self.feedforward_output_adapters.append(adapter)
            fc_modules.append(adapter)
            current_dim = out_dim
        self.output_adapter_mode = output_adapter_mode
        self.population_reduction_mode = population_reduction_mode
        self.spatial_somas_per_channel = tuple(spatial_somas_per_channel)
        self.classifier_somas_per_unit = tuple(classifier_somas_per_unit)
        self.classifier = nn.Sequential(*fc_modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_adapter(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def architecture_contract(self) -> dict[str, Any]:
        return {
            "input_layout": "BCHW",
            "input_transform": self.input_adapter.transform,
            "input_channels": self.input_channels,
            "adapted_input_channels": self.input_adapter.output_channels,
            "spatial_channel_mapping": (
                "one_excitatory_soma_per_output_channel"
                if all(value == 1 for value in self.spatial_somas_per_channel)
                else "fixed_excitatory_soma_group_per_output_channel"
            ),
            "channel_widths": list(self.channel_widths),
            "class_mapping": "one_excitatory_soma_per_class",
            "num_classes": self.num_classes,
            "propagated_population": "excitatory",
            "inter_block_output_adapter": self.output_adapter_mode,
            "population_reduction": self.population_reduction_mode,
            "spatial_somas_per_channel": list(self.spatial_somas_per_channel),
            "classifier_somas_per_unit": list(self.classifier_somas_per_unit),
            "component_seeds": list(self.component_seeds),
        }

    def decay_weights(self, weight_decay: float, weight_boosting: bool = False):
        for block in (*self.spatial_blocks, *self.feedforward_blocks):
            block.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self):
        for block in (*self.spatial_blocks, *self.feedforward_blocks):
            block.apply_rewiring()

    def get_effective_params(self) -> int:
        block_parameters = sum(
            int(block.get_effective_params())
            for block in (*self.spatial_blocks, *self.feedforward_blocks)
        )
        interface_parameters = sum(
            parameter.numel()
            for modules in (
                self.spatial_population_reducers,
                self.spatial_output_adapters,
                self.feedforward_population_reducers,
                self.feedforward_output_adapters,
            )
            for module in modules
            for parameter in module.parameters()
        )
        return block_parameters + interface_parameters
