"""Helpers to convert ordinary *linear* Conv2d kernels into dendritic push-pull
layers so that the forward pass is (almost) equivalent.
"""

from __future__ import annotations

import torch
from torch import nn

from dendritic_modeling.networks.layers.dendritic_conv2d import DendriteConv2d

__all__ = ["conv2d_to_dendrite"]


def conv2d_to_dendrite(
    conv: nn.Conv2d,
    *,
    baseline_inhib: float = 0.1,
    eps: float = 1e-6,
    copy_bias: bool = False,
    n_e_dendrites: int = 1,
    n_i_dendrites: int = 1,
    e_synapses_per_dendrite: int | None = None,
    i_synapses_per_dendrite: int | None = None,
    branch_aggregation: str = "sum",
    morphology_seed: int | None = None,
) -> DendriteConv2d:
    """Return a :class:`DendriteConv2d` whose E/I weights equal the push-pull
    decomposition of ``conv``.

    Parameters
    ----------
    conv
        Trained or un-trained Conv2d to mimic.
    baseline_inhib
        Constant added to inhibitory weights to keep denominator positive.
    eps
        Numerical epsilon added before log.
    copy_bias
        If ``True`` and ``conv.bias`` is not None, the bias term is copied into
        ``layer.bias_E`` while keeping inhibitory bias at zero.
    n_e_dendrites / n_i_dendrites
        Number of excitatory/inhibitory dendrites per output channel.
    e_synapses_per_dendrite / i_synapses_per_dendrite
        Active synapses per dendrite for E/I morphology masks.
    branch_aggregation
        Aggregation mode over dendrites ("sum", "mean", "max").
    morphology_seed
        Optional seed for deterministic morphology masks.
    """
    layer = DendriteConv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=copy_bias and (conv.bias is not None),
        n_e_dendrites=n_e_dendrites,
        n_i_dendrites=n_i_dendrites,
        e_synapses_per_dendrite=e_synapses_per_dendrite,
        i_synapses_per_dendrite=i_synapses_per_dendrite,
        branch_aggregation=branch_aggregation,
        morphology_seed=morphology_seed,
    )

    W = conv.weight.detach().cpu().numpy()
    E = W.clip(min=0)
    I_var = (-W).clip(min=0) + baseline_inhib

    E_t = torch.tensor(E, dtype=layer.weight_E_raw.dtype)
    I_t = torch.tensor(I_var, dtype=layer.weight_I_raw.dtype)

    def distribute_to_branches(
        target: torch.Tensor, branch_mask: torch.Tensor, aggregation: str
    ) -> torch.Tensor:
        """
        Distribute a single-kernel target onto branch kernels while respecting
        branch masks and aggregation mode.
        """
        # target shape: (O, I, Kh, Kw), branch_mask: (O, B, I, Kh, Kw)
        active_counts = branch_mask.sum(dim=1).clamp_min(1.0)  # (O, I, Kh, Kw)
        target_u = target.unsqueeze(1)
        if aggregation == "sum":
            return target_u * (branch_mask / active_counts.unsqueeze(1))
        if aggregation == "mean":
            scale = branch_mask.shape[1]
            return target_u * (branch_mask * scale / active_counts.unsqueeze(1))
        if aggregation == "max":
            return target_u * branch_mask
        raise ValueError(f"Unknown branch_aggregation: {aggregation}")

    E_branches = distribute_to_branches(E_t, layer.weight_E_mask, branch_aggregation)
    I_branches = distribute_to_branches(I_t, layer.weight_I_mask, branch_aggregation)

    with torch.no_grad():
        layer.weight_E_raw.copy_(torch.log(E_branches + eps))
        layer.weight_I_raw.copy_(torch.log(I_branches + eps))
        if copy_bias and conv.bias is not None:
            layer.bias_E.copy_(conv.bias)
            layer.bias_I.zero_()
    return layer
