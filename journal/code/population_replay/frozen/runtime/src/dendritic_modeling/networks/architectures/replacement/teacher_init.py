"""Teacher-weight initialization for vision replacement cores.

Two config-selectable routes copy a pretrained conv/linear teacher layer into
a dendritic replacement core so training starts at (or near) teacher parity
instead of from scratch — the vision counterpart of the transformer path's
teacher-TopK initializer:

- ``signed_topk``: the core must sit in the signed MLP corner (flat
  morphology, ``weight_transform: identity``, gates off, additive, no
  inhibitory contacts). The teacher kernel is flattened to patch space and
  its largest-magnitude entries per output channel are copied into the
  excitatory indexed synapses. Exact at full fan-in; energy-ranked
  approximation at sparse K.
- ``ei_sign_split``: the core must be raw additive with gates off and a
  DIRECT inhibitory bank (no inhibitory population), using a positive weight
  transform (``softplus``) or ``identity``. The teacher kernel is decomposed
  as ``W = W+ - W-``: the largest positive entries feed the excitatory
  contacts and the largest-magnitude negative entries feed the inhibitory
  contacts, both stored as positive magnitudes. Because raw additive computes
  ``E - I``, the result equals the TopK-truncated teacher map exactly — a
  positively-constrained E/I factorization of a signed teacher layer.

Teacher biases require the ``signed_per_channel_affine`` output adapter; both
routes fail closed with an explanatory error when the core's operator
configuration is incompatible.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.networks.utils.weight_transforms import inverse_softplus

logger = logging.getLogger(__name__)

TEACHER_INIT_MODES = ("none", "signed_topk", "ei_sign_split")


def flatten_teacher_weight(
    module: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return a teacher layer's weights as ``[out, in]`` plus optional bias."""

    if isinstance(module, nn.Conv2d):
        if module.groups != 1:
            raise ValueError("teacher_init supports groups=1 convolutions only")
        weight = module.weight.detach().reshape(module.out_channels, -1)
        bias = None if module.bias is None else module.bias.detach()
        return weight, bias
    if isinstance(module, nn.Linear):
        return module.weight.detach(), (
            None if module.bias is None else module.bias.detach()
        )
    raise TypeError(
        f"teacher_init requires a Conv2d or Linear teacher, got {type(module).__name__}"
    )


def find_teacher_layer(replaced_modules: list[nn.Module]) -> nn.Module:
    """Select the single parameterized teacher layer of a replaced span."""

    candidates = [
        module
        for module in replaced_modules
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    if len(candidates) != 1:
        raise ValueError(
            "teacher_init requires exactly one Conv2d/Linear in the replaced "
            f"span, found {len(candidates)}"
        )
    return candidates[0]


def _leaf_branch_layer(core: Any, *, mode: str) -> Any:
    einet = getattr(core, "einet", core)
    layers = getattr(einet, "layers", None)
    if layers is None or len(layers) != 1:
        raise ValueError(f"teacher_init={mode!r} requires a single EI layer core")
    cells = layers[0].excitatory_cells
    branch_layers = cells.branch_layers
    if len(branch_layers) != 2:
        raise ValueError(
            f"teacher_init={mode!r} requires flat morphology "
            "(excitatory_branch_factors=[1]: one synapse level plus the soma)"
        )
    leaf, soma = branch_layers[0], branch_layers[1]
    if not isinstance(leaf.reactivation, nn.Identity) or not isinstance(
        soma.reactivation, nn.Identity
    ):
        raise ValueError(f"teacher_init={mode!r} requires reactivation.enabled=false")
    if getattr(leaf, "use_shunting", True) or getattr(soma, "use_shunting", True):
        raise ValueError(f"teacher_init={mode!r} requires use_shunting=false")
    if layers[0].inhibitory_cells is not None:
        raise ValueError(
            f"teacher_init={mode!r} requires no inhibitory population "
            "(architecture.inhibitory_layer_sizes: [])"
        )
    return leaf, soma


def _write_indexed(
    synapse: Any, indices: torch.Tensor, raw_weights: torch.Tensor
) -> None:
    """Copy sorted teacher indices/weights into an indexed sparse synapse."""

    order = indices.argsort(dim=1)
    sorted_indices = indices.gather(1, order)
    sorted_weights = raw_weights.gather(1, order)
    with torch.no_grad():
        synapse.connection_indices.copy_(
            sorted_indices.to(synapse.connection_indices.dtype)
        )
        synapse.pre_w.copy_(sorted_weights.to(synapse.pre_w.dtype))


def copy_topk_weight_(
    weight: torch.Tensor,
    target: Any,
    *,
    path: str = "projection",
    input_scales: torch.Tensor | None = None,
    structured: bool = False,
) -> float:
    """Copy a signed dense weight into an indexed projection by scored TopK.

    This is the common support-selection primitive used by vision and
    transformer replacement.  ``input_scales`` changes only the selection
    score to ``|W_ij| * rms(x_j)``; the copied values remain the teacher's raw
    signed weights. ``structured`` selects four arbitrary positions within
    each retained 16-column block, matching the measured 4-of-16 layout.
    """

    required = ("connection_indices", "pre_w", "K", "weight_transform")
    if not all(hasattr(target, name) for name in required):
        raise TypeError(f"{path} target must be an indexed sparse projection")
    if str(target.weight_transform).lower() != "identity":
        raise ValueError(f"{path} TopK copy requires weight_transform='identity'")
    expected = (int(target.out_features), int(target.in_features))
    if tuple(weight.shape) != expected:
        raise ValueError(
            f"Teacher {path} weight has shape {tuple(weight.shape)}, expected {expected}"
        )

    detached = weight.detach()
    score = detached.abs()
    if input_scales is not None:
        if input_scales.numel() != detached.shape[1]:
            raise ValueError(
                f"input_scales for {path} has {input_scales.numel()} entries, "
                f"expected {detached.shape[1]}"
            )
        score = score * input_scales.detach().to(
            device=score.device,
            dtype=score.dtype,
        ).clamp_min(1e-12).unsqueeze(0)

    k = int(target.K)
    if structured:
        block, per = 16, 4
        input_dim = detached.shape[1]
        if k % per != 0 or input_dim % block != 0:
            raise ValueError(
                f"structured selection requires K%{per}==0 and D%{block}==0"
            )
        score_blocks = score.reshape(score.shape[0], input_dim // block, block)
        within_values, within_indices = torch.topk(
            score_blocks,
            k=per,
            dim=2,
            sorted=False,
        )
        block_mass = within_values.sum(dim=2)
        chosen_blocks = torch.topk(
            block_mass,
            k=k // per,
            dim=1,
            sorted=False,
        ).indices
        gathered = within_indices.gather(
            1,
            chosen_blocks.unsqueeze(-1).expand(-1, -1, per),
        )
        indices = (gathered + chosen_blocks.unsqueeze(-1) * block).reshape(
            score.shape[0],
            k,
        )
    else:
        indices = torch.topk(score, k=k, dim=1, sorted=False).indices
    selected = detached.gather(1, indices)
    _write_indexed(target, indices, selected)

    total_energy = detached.float().square().sum().clamp_min(1e-30)
    retained_energy = selected.float().square().sum() / total_energy
    return float(retained_energy.item())


def _soma_unit_coupling(soma: Any) -> None:
    """Set the branch-to-soma coupling to exactly one under its transform."""

    blocklinear = soma.branches_to_output
    transform = str(blocklinear.weight_transform).lower()
    with torch.no_grad():
        if transform == "identity":
            blocklinear.log_weight.fill_(1.0)
        elif transform == "softplus":
            blocklinear.log_weight.fill_(float(inverse_softplus(torch.tensor(1.0))))
        else:
            raise ValueError(
                f"teacher_init requires identity or softplus weight transform, "
                f"got {transform!r}"
            )


def _apply_bias(core: Any, bias: torch.Tensor | None) -> None:
    if bias is None:
        return
    adapter = getattr(core, "output_adapter", None)
    if adapter is None or not hasattr(adapter, "bias"):
        raise ValueError(
            "Teacher bias requires spatial.output_adapter_mode: "
            "signed_per_channel_affine on the replacement core"
        )
    with torch.no_grad():
        adapter.bias.copy_(bias.to(adapter.bias.dtype))


def signed_topk_teacher_init_(
    core: Any,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> dict[str, float]:
    """Copy a signed teacher map into the MLP-corner core via magnitude TopK."""

    leaf, soma = _leaf_branch_layer(core, mode="signed_topk")
    synapse = leaf.branch_excitation
    if str(synapse.weight_transform).lower() != "identity":
        raise ValueError(
            "teacher_init='signed_topk' requires morphology.weight_transform: "
            "identity (signed weights)"
        )
    if leaf.branch_inhibition is not None:
        raise ValueError(
            "teacher_init='signed_topk' requires zero inhibitory contacts "
            "(connectivity.ie_synapses_per_branch_per_layer: [0]); use "
            "'ei_sign_split' for an E/I-factorized start"
        )
    out_features, in_features = weight.shape
    if synapse.out_features != out_features or synapse.in_features != in_features:
        raise ValueError(
            f"Core synapse shape ({synapse.out_features}, {synapse.in_features}) "
            f"does not match teacher ({out_features}, {in_features})"
        )
    retained = copy_topk_weight_(weight, synapse, path="vision teacher")
    _soma_unit_coupling(soma)
    _apply_bias(core, bias)

    diagnostics = {
        "weight_energy_retained": retained,
        "k": int(synapse.K),
    }
    logger.info("signed_topk teacher init: %s", diagnostics)
    return diagnostics


def ei_sign_split_teacher_init_(
    core: Any,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> dict[str, float]:
    """Copy a signed teacher map as a positive E/I factorization (W+ vs W-)."""

    leaf, soma = _leaf_branch_layer(core, mode="ei_sign_split")
    excitatory = leaf.branch_excitation
    inhibitory = leaf.branch_inhibition
    if inhibitory is None:
        raise ValueError(
            "teacher_init='ei_sign_split' requires direct inhibitory contacts "
            "(connectivity.ie_synapses_per_branch_per_layer > 0 with no "
            "inhibitory population)"
        )
    transform = str(excitatory.weight_transform).lower()
    if transform not in {"identity", "softplus"}:
        raise ValueError(
            "teacher_init='ei_sign_split' requires weight_transform identity "
            f"or softplus, got {transform!r}"
        )

    out_features, in_features = weight.shape
    for synapse, name in ((excitatory, "excitatory"), (inhibitory, "inhibitory")):
        if synapse.out_features != out_features or synapse.in_features != in_features:
            raise ValueError(
                f"{name} synapse shape ({synapse.out_features}, "
                f"{synapse.in_features}) does not match teacher "
                f"({out_features}, {in_features})"
            )

    positive = weight.clamp_min(0.0)
    negative = (-weight).clamp_min(0.0)
    eps = 1e-12

    def _bank(magnitudes: torch.Tensor, synapse: Any) -> torch.Tensor:
        top = magnitudes.topk(synapse.K, dim=1).indices
        values = magnitudes.gather(1, top)
        if transform == "softplus":
            raw = inverse_softplus(values.clamp_min(1e-8))
        else:
            raw = values
        _write_indexed(synapse, top, raw)
        return values

    kept_pos = _bank(positive, excitatory)
    kept_neg = _bank(negative, inhibitory)
    _soma_unit_coupling(soma)
    _apply_bias(core, bias)

    total = weight.pow(2).sum().clamp_min(eps)
    diagnostics = {
        "weight_energy_retained": float(
            (kept_pos.pow(2).sum() + kept_neg.pow(2).sum()) / total
        ),
        "k_excitatory": int(excitatory.K),
        "k_inhibitory": int(inhibitory.K),
    }
    logger.info("ei_sign_split teacher init: %s", diagnostics)
    return diagnostics


def apply_teacher_init_(
    core: Any,
    replaced_modules: list[nn.Module],
    mode: str,
) -> dict[str, float] | None:
    """Dispatch a configured teacher initialization onto a replacement core."""

    normalized = str(mode or "none").strip().lower()
    if normalized not in TEACHER_INIT_MODES:
        raise ValueError(
            f"teacher_init mode must be one of {TEACHER_INIT_MODES}, got {mode!r}"
        )
    if normalized == "none":
        return None
    teacher = find_teacher_layer(replaced_modules)
    weight, bias = flatten_teacher_weight(teacher)
    if normalized == "signed_topk":
        return signed_topk_teacher_init_(core, weight, bias)
    return ei_sign_split_teacher_init_(core, weight, bias)


__all__ = [
    "TEACHER_INIT_MODES",
    "apply_teacher_init_",
    "copy_topk_weight_",
    "ei_sign_split_teacher_init_",
    "find_teacher_layer",
    "flatten_teacher_weight",
    "signed_topk_teacher_init_",
]
