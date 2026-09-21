"""Multi-region vision replacement: several named spans, one model.

The single-span splitter (``classical.pretrained``) supports exactly one
contiguous replaced span. Progressive replacement (REDUCTION_PROGRAM.md T3)
needs several independent spans — for example a dendritic conv5 *and* a
dendritic fc6+fc7 inside one AlexNet — each with its own operator
configuration.

Design constraints (deliberate, v1):

- Regions are named, contiguous, non-overlapping spans of the flattened
  backbone, in forward order.  A single region is allowed: the sequential
  replace-recover chain composes stage 1 (one region, dense teacher
  remainder as the suffix) in the same ``cores.<name>`` namespace as later
  stages so per-stage checkpoints carry forward without remapping.  The
  single-span ``target_modules`` path remains the default for one-off
  single-boundary runs.
- Retained modules *between* two regions ("bridges") must be parameter-free
  (pooling, dropout, flatten). The AlexNet ladder satisfies this, and the
  restriction keeps trainability and learning-rate semantics identical to the
  single-span path: the prefix is the encoder, the suffix is the decoder, and
  every learned parameter in between belongs to a replacement core.
- All boundary shapes come from teacher dummy-forwards, exactly like the
  single-span plan, so a region's core must reproduce its teacher span's
  output interface.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch.nn as nn

from dendritic_modeling.networks.architectures.classical.pretrained import (
    BackboneSegment,
    TensorSpec,
    _filter_omitted_layers,
    _flatten_model_layers,
    _load_backbone,
    _names_and_modules,
    _resolve_split_input_size,
    _resolve_target_module_span,
)
from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.utils.nvtx import nvtx_range


@dataclass
class RegionBoundary:
    """Teacher-derived interface of one replacement region.

    ``span_modules`` holds the replaced teacher modules themselves so that
    per-region teacher-weight initialization can read their weights; they are
    not part of the assembled network and are dropped with the plan.
    """

    name: str
    span_names: list[str]
    input_spec: TensorSpec | None
    input_dim: int
    output_spec: TensorSpec | None
    output_dim: int
    span_modules: list[nn.Module] = field(default_factory=list)


@dataclass
class MultiRegionSplitPlan:
    """Resolved multi-region split of one pretrained backbone."""

    backbone: str
    input_size: tuple[int, ...]
    prefix_segment: BackboneSegment
    regions: list[RegionBoundary]
    bridge_segments: list[BackboneSegment] = field(default_factory=list)
    suffix_segment: BackboneSegment | None = None

    def describe(self) -> str:
        lines = [f"MultiRegionSplitPlan(backbone={self.backbone!r})"]
        lines.append(f"  prefix: {self.prefix_segment._layer_names}")
        for index, region in enumerate(self.regions):
            lines.append(
                f"  region {region.name!r}: {region.span_names} "
                f"(in={region.input_spec or region.input_dim}, "
                f"out={region.output_spec or region.output_dim})"
            )
            if index < len(self.bridge_segments):
                lines.append(f"  bridge: {self.bridge_segments[index]._layer_names}")
        if self.suffix_segment is not None:
            lines.append(f"  suffix: {self.suffix_segment._layer_names}")
        return "\n".join(lines)


def _segment_input_size(spec: TensorSpec | None, flat_dim: int) -> tuple[int, ...]:
    if spec is not None:
        return (1, spec.channels, spec.height, spec.width)
    return (1, flat_dim)


def build_multi_region_split_plan(
    *,
    backbone: str,
    weights: str | None,
    regions: list[dict],
    omit_layers: list[str] | None = None,
    input_shape: list[int] | None = None,
    prefix_freeze: bool = True,
    suffix_freeze: bool = False,
) -> MultiRegionSplitPlan:
    """Resolve named replacement regions against one loaded backbone.

    ``regions`` entries need ``name`` and ``target_modules``; ordering in the
    list is normalized to backbone order. Raises on overlapping spans, on a
    parameterized bridge, and on unknown layers (via the shared span
    resolver).
    """

    if not regions:
        raise ValueError("Multi-region replacement needs at least one region")
    model = _load_backbone(backbone, weights)
    layers = _filter_omitted_layers(_flatten_model_layers(model), omit_layers)
    all_names = [name for name, _ in layers]

    resolved: list[tuple[int, int, str, list[str]]] = []
    seen_names: set[str] = set()
    for entry in regions:
        name = str(entry.get("name", "") or "")
        targets = list(entry.get("target_modules", []) or [])
        if not name:
            raise ValueError("Every replacement region needs a non-empty name")
        if name in seen_names:
            raise ValueError(f"Duplicate region name {name!r}")
        seen_names.add(name)
        span = _resolve_target_module_span(all_names, targets)
        if span is None:
            raise ValueError(f"Region {name!r} declares no target_modules")
        resolved.append((span[0], span[1], name, targets))
    resolved.sort(key=lambda item: item[0])
    for (_, end_a, name_a, _), (start_b, _, name_b, _) in itertools.pairwise(resolved):
        if start_b <= end_a:
            raise ValueError(
                f"Regions {name_a!r} and {name_b!r} overlap in backbone order"
            )

    input_size = _resolve_split_input_size(input_shape)
    prefix_layers = layers[: resolved[0][0]]
    prefix_names, prefix_modules = _names_and_modules(prefix_layers)
    prefix_segment = BackboneSegment(
        prefix_modules,
        prefix_names,
        freeze=prefix_freeze,
        input_size=input_size,
        auto_flatten=False,
    )

    current_size = _segment_input_size(
        prefix_segment.output_spec, prefix_segment.output_dim
    )
    region_boundaries: list[RegionBoundary] = []
    bridge_segments: list[BackboneSegment] = []
    for index, (start, end, name, _targets) in enumerate(resolved):
        span_layers = layers[start : end + 1]
        span_names, span_modules = _names_and_modules(span_layers)
        input_spec = TensorSpec(*current_size[1:]) if len(current_size) == 4 else None
        input_dim = (
            current_size[-1]
            if len(current_size) == 2
            else (current_size[1] * current_size[2] * current_size[3])
        )
        # Teacher span segment defines the region's required output interface.
        teacher_segment = BackboneSegment(
            span_modules,
            span_names,
            freeze=True,
            input_size=current_size,
            auto_flatten=len(current_size) == 2,
        )
        region_boundaries.append(
            RegionBoundary(
                name=name,
                span_names=span_names,
                input_spec=input_spec,
                input_dim=input_dim,
                output_spec=teacher_segment.output_spec,
                output_dim=teacher_segment.output_dim,
                span_modules=list(span_modules),
            )
        )
        current_size = _segment_input_size(
            teacher_segment.output_spec, teacher_segment.output_dim
        )

        if index < len(resolved) - 1:
            bridge_layers = layers[end + 1 : resolved[index + 1][0]]
            bridge_names, bridge_modules = _names_and_modules(bridge_layers)
            bridge_params = sum(
                parameter.numel()
                for module in bridge_modules
                for parameter in module.parameters()
            )
            if bridge_params:
                raise ValueError(
                    f"Retained modules between regions {name!r} and "
                    f"{resolved[index + 1][2]!r} carry {bridge_params} "
                    "parameters; multi-region v1 requires parameter-free "
                    "bridges (pooling/dropout/flatten). Extend one region to "
                    "cover the parameterized modules instead."
                )
            # Flatten inside the bridge only when the next region's teacher
            # span expects flat input; adjacent spatial spans (for example
            # conv4 -> conv5 with an empty bridge) must keep BCHW intact.
            next_start = resolved[index + 1][0]
            next_first_module = layers[next_start][1]
            next_expects_flat = isinstance(next_first_module, nn.Linear)
            bridge_segment = BackboneSegment(
                bridge_modules,
                bridge_names,
                freeze=True,
                input_size=current_size,
                auto_flatten=next_expects_flat,
            )
            bridge_segments.append(bridge_segment)
            current_size = _segment_input_size(
                bridge_segment.output_spec, bridge_segment.output_dim
            )

    suffix_layers = layers[resolved[-1][1] + 1 :]
    suffix_names, suffix_modules = _names_and_modules(suffix_layers)
    suffix_segment = BackboneSegment(
        suffix_modules,
        suffix_names,
        freeze=suffix_freeze,
        input_size=current_size,
        auto_flatten=True,
    )

    return MultiRegionSplitPlan(
        backbone=backbone,
        input_size=input_size,
        prefix_segment=prefix_segment,
        regions=region_boundaries,
        bridge_segments=bridge_segments,
        suffix_segment=suffix_segment,
    )


class MultiRegionCore(BaseNetwork):
    """Sequential composite of replacement cores and parameter-free bridges.

    State-dict layout: ``cores.<region_name>.*`` for each replacement core and
    ``bridges.<i>.*`` for the (parameter-free) retained segments, so per-region
    checkpoints remain individually addressable for warm starts and staged
    assembly.
    """

    def __init__(
        self,
        cores: dict[str, nn.Module],
        bridges: list[nn.Module],
    ):
        super().__init__()
        if len(bridges) != len(cores) - 1:
            raise ValueError(
                f"Expected {len(cores) - 1} bridges for {len(cores)} cores, "
                f"got {len(bridges)}"
            )
        self.cores = nn.ModuleDict(cores)
        self.bridges = nn.ModuleList(bridges)
        last = list(cores.values())[-1]
        self.output_dim = getattr(last, "output_dim", None)

    def forward(self, x):
        for index, (name, core) in enumerate(self.cores.items()):
            with nvtx_range(f"region:{name}"):
                x = core(x)
            if index < len(self.bridges):
                with nvtx_range(f"bridge:{index}"):
                    x = self.bridges[index](x)
        return x

    def get_effective_params(self) -> int:
        total = 0
        for core in self.cores.values():
            counter = getattr(core, "get_effective_params", None)
            if callable(counter):
                total += int(counter())
            else:
                total += sum(p.numel() for p in core.parameters())
        return total

    def decay_weights(self, weight_decay, weight_boosting=False):
        for core in self.cores.values():
            decay = getattr(core, "decay_weights", None)
            if callable(decay):
                decay(weight_decay, weight_boosting)

    def apply_rewiring(self):
        for core in self.cores.values():
            rewire = getattr(core, "apply_rewiring", None)
            if callable(rewire):
                rewire()


__all__ = [
    "MultiRegionCore",
    "MultiRegionSplitPlan",
    "RegionBoundary",
    "build_multi_region_split_plan",
]
