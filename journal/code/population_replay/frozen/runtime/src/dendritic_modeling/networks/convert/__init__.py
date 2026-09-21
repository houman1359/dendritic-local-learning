"""Conversion utilities (e.g. mapping classical conv layers to dendritic layers)."""

from dendritic_modeling.networks.convert.cnn_to_dendrite import conv2d_to_dendrite

__all__ = ["conv2d_to_dendrite"]
