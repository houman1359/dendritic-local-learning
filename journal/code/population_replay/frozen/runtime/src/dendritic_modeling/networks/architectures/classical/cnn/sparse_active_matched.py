"""Fixed-index sparse point convolution matched to a dendritic reference.

The layer extracts shared local patches with :func:`torch.nn.functional.unfold`
and applies one :class:`SparseActiveMatchedPointAffine` to every spatial
position. It is the spatial analogue of ``sparse_active_matched_point``: all
output channels are active, the support is fixed during training, and the
learned-scalar target comes from the structured E/I reference in the config.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.networks.architectures.classical.mlp import (
    SparseActiveMatchedPointAffine,
)
from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.scripts.script_utils.config_utils import (
    prepare_ei_network_params,
)


def _pair(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("spatial convolution values must have length two")
        return int(value[0]), int(value[1])
    return int(value), int(value)


class SparseActiveMatchedPointConv(BaseNetwork):
    """Shared local sparse affine-plus-ReLU layer with exact active matching."""

    def __init__(
        self,
        config: dict[str, Any] | DictConfig,
        in_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        patch_dim = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

        raw_config = to_plain_dict(config)
        parameters = prepare_ei_network_params(raw_config, patch_dim)
        if "target_active_parameters" in raw_config:
            parameters["target_active_parameters"] = raw_config[
                "target_active_parameters"
            ]
        self.point = SparseActiveMatchedPointAffine(**parameters)
        self.out_channels = self.point.output_dim
        self.output_dim = self.out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "SparseActiveMatchedPointConv requires input shaped (B,C,H,W), "
                f"got {tuple(x.shape)}"
            )
        batch, _channels, height, width = x.shape
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        locations = patches.shape[-1]
        flat = patches.transpose(1, 2).reshape(batch * locations, -1)
        output = self.point(flat)
        out_height, out_width = self.compute_output_shape(height, width)
        return (
            output.reshape(batch, out_height, out_width, self.out_channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def compute_output_shape(self, height: int, width: int) -> tuple[int, int]:
        out_height = (
            height + 2 * self.padding[0] - self.kernel_size[0]
        ) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[
            1
        ] + 1
        return out_height, out_width

    def get_effective_params(self) -> int:
        return self.point.get_effective_params()

    def connectivity_resource_records(self) -> list[dict[str, Any]]:
        return self.point.connectivity_resource_records()

    def resource_accounting(self) -> dict[str, int | bool | str]:
        return self.point.resource_accounting()

    def connectivity_support_sha256(self) -> str:
        return self.point.connectivity_support_sha256()


__all__ = ["SparseActiveMatchedPointConv"]
