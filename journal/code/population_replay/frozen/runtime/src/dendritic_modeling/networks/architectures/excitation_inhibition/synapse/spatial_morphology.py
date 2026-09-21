"""Direct sparse indices for spatially constrained dendritic connectivity.

The sampler in this module maps a dendritic tree onto a ``(C, H, W)`` input
without constructing a dense ``[n_branches, C * H * W]`` mask.  Every soma
owns the full image.  A branch samples exactly ``K`` unique input features from
its assigned rectangle.  Rectangles can follow a nested morphology-aligned map
or be independently relocated while preserving the aligned map's window-size
distribution.

No weights are shared between somas or spatial locations.  This is therefore
an anatomical sparse-connectivity prior, not a convolution.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _resolve_index_dtype,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    INDEX_MANIFEST_METHODS,
    _resolve_pathway_config,
)
from dendritic_modeling.utils.stable_hash import stable_seed_offset

SPATIAL_MORPHOLOGY_METHODS = frozenset({"spatial_morphology", "morphology_spatial"})
SPATIAL_REGION_MODES = frozenset({"partition", "overlapping_grid", "random_windows"})
_AXIS_ALIASES = {
    "h": "height",
    "height": "height",
    "y": "height",
    "w": "width",
    "width": "width",
    "x": "width",
}


def uses_spatial_morphology(config, pathway: str) -> bool:
    """Return whether one configured pathway uses spatial image routing."""

    pathway_config = _resolve_pathway_config(config, pathway)
    method = str(pathway_config.get("method", "")).strip().lower()
    return method in SPATIAL_MORPHOLOGY_METHODS


def _positive_int_tuple(
    values: Sequence[int],
    *,
    name: str,
    expected_length: int | None = None,
) -> tuple[int, ...]:
    """Normalize a positive integer sequence."""

    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of positive integers")
    normalized = tuple(int(value) for value in values)
    if expected_length is not None and len(normalized) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values, got {len(normalized)}"
        )
    if not normalized or any(value < 1 for value in normalized):
        raise ValueError(f"{name} values must all be positive")
    return normalized


def _normalize_split_axes(
    values: Sequence[str] | None,
    *,
    depth: int,
) -> tuple[str, ...]:
    """Return root-to-leaf spatial split axes."""

    if values is None:
        return tuple("height" if idx % 2 == 0 else "width" for idx in range(depth))
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("spatial.split_axes must be a sequence")
    if len(values) != depth:
        raise ValueError(
            "spatial.split_axes must contain one root-to-leaf axis per "
            f"branch factor, got {len(values)} axes for depth {depth}"
        )
    axes = []
    for value in values:
        normalized = _AXIS_ALIASES.get(str(value).strip().lower())
        if normalized is None:
            choices = ", ".join(sorted(_AXIS_ALIASES))
            raise ValueError(
                f"Unknown spatial split axis {value!r}; choose from {choices}"
            )
        axes.append(normalized)
    return tuple(axes)


def _branch_path(
    local_branch_idx: int,
    branch_factors: tuple[int, ...],
) -> tuple[int, ...]:
    """Decode a root-to-branch path from a mixed-radix branch index."""

    digits = []
    remainder = int(local_branch_idx)
    for idx, factor in enumerate(branch_factors):
        suffix = math.prod(branch_factors[idx + 1 :])
        digit, remainder = divmod(remainder, suffix)
        if digit >= factor:
            raise RuntimeError("Decoded branch path exceeds its branch factor")
        digits.append(digit)
    return tuple(digits)


def _split_interval(
    lower: int,
    upper: int,
    *,
    factor: int,
    child: int,
) -> tuple[int, int]:
    """Select one integer child interval from an equal continuous partition."""

    length = upper - lower
    child_lower = lower + (length * child) // factor
    child_upper = lower + (length * (child + 1)) // factor
    if child_upper <= child_lower:
        raise ValueError(
            "Spatial morphology creates an empty region. Increase the input "
            "resolution or reduce branching along the corresponding axis."
        )
    return child_lower, child_upper


def _path_rectangle(
    *,
    height: int,
    width: int,
    path: tuple[int, ...],
    branch_factors: tuple[int, ...],
    split_axes: tuple[str, ...],
) -> tuple[int, int, int, int]:
    """Return ``(y0, y1, x0, x1)`` for one root-to-branch path."""

    y0, y1 = 0, height
    x0, x1 = 0, width
    for child, factor, axis in zip(path, branch_factors, split_axes):
        if axis == "height":
            y0, y1 = _split_interval(y0, y1, factor=factor, child=child)
        else:
            x0, x1 = _split_interval(x0, x1, factor=factor, child=child)
    return y0, y1, x0, x1


def _balanced_grid_shape(
    factor: int,
    *,
    region_height: float,
    region_width: float,
) -> tuple[int, int]:
    """Factor one branch fan-out into the most nearly square spatial grid."""

    candidates: list[tuple[float, int, int]] = []
    for rows in range(1, int(math.sqrt(factor)) + 1):
        if factor % rows:
            continue
        columns = factor // rows
        for grid_rows, grid_columns in ((rows, columns), (columns, rows)):
            cell_height = region_height / grid_rows
            cell_width = region_width / grid_columns
            aspect_penalty = abs(math.log(cell_height / cell_width))
            candidates.append((aspect_penalty, grid_rows, grid_columns))
    _, rows, columns = min(candidates)
    return rows, columns


def _overlapping_grid_rectangle(
    *,
    height: int,
    width: int,
    path: tuple[int, ...],
    branch_factors: tuple[int, ...],
    minimum_window_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return a nested, possibly overlapping image window for one path.

    Branch fan-out is placed on a two-dimensional grid inside the parent's
    continuous image region.  The final discrete window is expanded around
    the cell center when necessary.  This permits more terminal branches than
    disjoint image cells while keeping every branch spatially localized.
    """

    y0, y1 = 0.0, 1.0
    x0, x1 = 0.0, 1.0
    final_grid = (1, 1)
    final_child = (0, 0)
    final_parent = (y0, y1, x0, x1)
    for child, factor in zip(path, branch_factors):
        rows, columns = _balanced_grid_shape(
            factor,
            region_height=(y1 - y0) * height,
            region_width=(x1 - x0) * width,
        )
        row, column = divmod(child, columns)
        parent_y0, parent_y1 = y0, y1
        parent_x0, parent_x1 = x0, x1
        final_grid = (rows, columns)
        final_child = (row, column)
        final_parent = (parent_y0, parent_y1, parent_x0, parent_x1)
        y0 = parent_y0 + (parent_y1 - parent_y0) * row / rows
        y1 = parent_y0 + (parent_y1 - parent_y0) * (row + 1) / rows
        x0 = parent_x0 + (parent_x1 - parent_x0) * column / columns
        x1 = parent_x0 + (parent_x1 - parent_x0) * (column + 1) / columns

    center_y = 0.5 * (y0 + y1) * height
    center_x = 0.5 * (x0 + x1) * width
    base_height = max(1, math.ceil(y1 * height) - math.floor(y0 * height))
    base_width = max(1, math.ceil(x1 * width) - math.floor(x0 * width))
    window_height = min(height, max(base_height, minimum_window_shape[0]))
    window_width = min(width, max(base_width, minimum_window_shape[1]))

    parent_y0, parent_y1, parent_x0, parent_x1 = final_parent

    def _window_start(
        *,
        parent_lower: float,
        parent_upper: float,
        grid_count: int,
        child_index: int,
        window_size: int,
        full_size: int,
        center: float,
    ) -> int:
        parent_lower_px = max(0, math.floor(parent_lower * full_size))
        parent_upper_px = min(full_size, math.ceil(parent_upper * full_size))
        if parent_upper_px - parent_lower_px >= window_size:
            lowest = parent_lower_px
            highest = parent_upper_px - window_size
        else:
            centered = round(center - window_size / 2)
            radius = max(0, grid_count - 1)
            lowest = max(0, centered - radius // 2)
            highest = min(full_size - window_size, lowest + radius)
            lowest = max(0, highest - radius)
        if grid_count <= 1 or highest <= lowest:
            return min(max(0, round(center - window_size / 2)), full_size - window_size)
        return round(lowest + (highest - lowest) * child_index / (grid_count - 1))

    lower_y = _window_start(
        parent_lower=parent_y0,
        parent_upper=parent_y1,
        grid_count=final_grid[0],
        child_index=final_child[0],
        window_size=window_height,
        full_size=height,
        center=center_y,
    )
    lower_x = _window_start(
        parent_lower=parent_x0,
        parent_upper=parent_x1,
        grid_count=final_grid[1],
        child_index=final_child[1],
        window_size=window_width,
        full_size=width,
        center=center_x,
    )
    return (
        lower_y,
        lower_y + window_height,
        lower_x,
        lower_x + window_width,
    )


def _randomly_place_rectangle(
    rectangle: tuple[int, int, int, int],
    *,
    height: int,
    width: int,
    generator: torch.Generator,
) -> tuple[int, int, int, int]:
    """Place a reference-sized window uniformly without tree alignment."""

    y0, y1, x0, x1 = rectangle
    window_height = y1 - y0
    window_width = x1 - x0
    lower_y = int(
        torch.randint(
            height - window_height + 1,
            (1,),
            generator=generator,
        ).item()
    )
    lower_x = int(
        torch.randint(
            width - window_width + 1,
            (1,),
            generator=generator,
        ).item()
    )
    return (
        lower_y,
        lower_y + window_height,
        lower_x,
        lower_x + window_width,
    )


def spatial_morphology_rectangles(
    *,
    input_shape: Sequence[int],
    branch_factors: Sequence[int],
    level_idx: int,
    split_axes: Sequence[str] | None = None,
    region_mode: str = "partition",
    minimum_window_shape: Sequence[int] | None = None,
    seed: int = 0,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return the spatial window assigned to every local branch at one level.

    ``random_windows`` first obtains the exact window sizes of the aligned
    ``overlapping_grid`` construction, then places each window independently
    and uniformly over the image.  It therefore preserves the per-level window
    size distribution while removing parent--child spatial alignment.
    """

    _channels, height, width = _positive_int_tuple(
        input_shape,
        name="spatial.input_shape",
        expected_length=3,
    )
    factors = _positive_int_tuple(branch_factors, name="branch_factors")
    normalized_region_mode = str(region_mode).strip().lower()
    if normalized_region_mode not in SPATIAL_REGION_MODES:
        choices = ", ".join(sorted(SPATIAL_REGION_MODES))
        raise ValueError(
            f"Unknown spatial.region_mode {region_mode!r}; choose from {choices}"
        )
    axes = (
        _normalize_split_axes(split_axes, depth=len(factors))
        if normalized_region_mode == "partition"
        else ()
    )
    window_shape = (
        (1, 1)
        if minimum_window_shape is None
        else _positive_int_tuple(
            minimum_window_shape,
            name="spatial.minimum_window_shape",
            expected_length=2,
        )
    )
    if window_shape[0] > height or window_shape[1] > width:
        raise ValueError(
            "spatial.minimum_window_shape cannot exceed the input height or width"
        )
    if level_idx < 0 or level_idx > len(factors):
        raise ValueError(f"level_idx must be in [0, {len(factors)}], got {level_idx}")

    anatomical_depth = len(factors) - level_idx
    active_factors = factors[:anatomical_depth]
    active_axes = axes[:anatomical_depth]
    branches_per_owner = math.prod(active_factors)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % ((1 << 63) - 1))
    rectangles: list[tuple[int, int, int, int]] = []
    for local_branch_idx in range(branches_per_owner):
        path = _branch_path(local_branch_idx, active_factors)
        if normalized_region_mode == "partition":
            rectangle = _path_rectangle(
                height=height,
                width=width,
                path=path,
                branch_factors=active_factors,
                split_axes=active_axes,
            )
        else:
            rectangle = _overlapping_grid_rectangle(
                height=height,
                width=width,
                path=path,
                branch_factors=active_factors,
                minimum_window_shape=window_shape,
            )
            if normalized_region_mode == "random_windows":
                rectangle = _randomly_place_rectangle(
                    rectangle,
                    height=height,
                    width=width,
                    generator=generator,
                )
        rectangles.append(rectangle)
    return tuple(rectangles)


def _rectangle_candidates(
    *,
    channels: int,
    height: int,
    width: int,
    rectangle: tuple[int, int, int, int],
) -> torch.Tensor:
    """Return flattened CHW indices belonging to a rectangle."""

    y0, y1, x0, x1 = rectangle
    y = torch.arange(y0, y1, dtype=torch.long)
    x = torch.arange(x0, x1, dtype=torch.long)
    spatial = (y[:, None] * width + x[None, :]).reshape(-1)
    channel_offsets = torch.arange(channels, dtype=torch.long) * height * width
    return (channel_offsets[:, None] + spatial[None, :]).reshape(-1)


def _sample_unique_candidates(
    candidates: torch.Tensor,
    *,
    n_rows: int,
    synapses_per_branch: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample deterministic unique candidates without a dense row-wise mask."""

    population_size = int(candidates.numel())
    if synapses_per_branch > population_size:
        raise ValueError(
            f"Requested {synapses_per_branch} synapses from a spatial branch "
            f"region containing only {population_size} input features"
        )

    # Shuffle each region once, then give every row an independently seeded
    # affine traversal of that permutation. A step coprime to the population
    # size guarantees unique indices within each row for every valid K.
    permutation = torch.randperm(population_size, generator=generator)
    starts = torch.randint(
        population_size,
        (n_rows,),
        generator=generator,
        dtype=torch.long,
    )
    if population_size == 1:
        steps = torch.ones(n_rows, dtype=torch.long)
    else:
        steps = torch.randint(
            1,
            population_size,
            (n_rows,),
            generator=generator,
            dtype=torch.long,
        )
        invalid = torch.gcd(steps, torch.full_like(steps, population_size)) != 1
        while bool(invalid.any()):
            steps[invalid] = torch.randint(
                1,
                population_size,
                (int(invalid.sum().item()),),
                generator=generator,
                dtype=torch.long,
            )
            invalid = torch.gcd(steps, torch.full_like(steps, population_size)) != 1

    offsets = (
        starts[:, None]
        + steps[:, None] * torch.arange(synapses_per_branch, dtype=torch.long)
    ) % population_size
    return candidates[permutation[offsets]]


def _spatial_options(config: Mapping[str, object]) -> Mapping[str, object]:
    """Return the nested spatial options with a clear configuration error."""

    spatial = to_plain_dict(config.get("spatial", {}))
    if not spatial:
        raise ValueError(
            "method='spatial_morphology' requires a structured.spatial block"
        )
    return spatial


def sample_spatial_morphology_indices(
    *,
    input_shape: Sequence[int],
    out_features: int,
    in_features: int,
    synapses_per_branch: int,
    owner_count: int,
    branch_factors: Sequence[int],
    level_idx: int,
    split_axes: Sequence[str] | None = None,
    region_mode: str = "partition",
    minimum_window_shape: Sequence[int] | None = None,
    seed: int = 0,
    region_seed: int | None = None,
    index_dtype: str | torch.dtype = "int64",
) -> torch.Tensor:
    """Generate compact branch indices for a full-image dendritic map.

    ``level_idx`` follows DendriNet construction order: zero is the distal
    level and ``len(branch_factors)`` is the soma. ``split_axes`` and
    ``branch_factors`` are specified in the opposite, anatomical order:
    root-to-leaf.
    """

    channels, height, width = _positive_int_tuple(
        input_shape,
        name="spatial.input_shape",
        expected_length=3,
    )
    if channels * height * width != int(in_features):
        raise ValueError(
            "spatial.input_shape does not match the pathway input dimension: "
            f"{channels}*{height}*{width} != {in_features}"
        )
    factors = _positive_int_tuple(branch_factors, name="branch_factors")
    if owner_count < 1:
        raise ValueError("owner_count must be positive")
    if level_idx < 0 or level_idx > len(factors):
        raise ValueError(f"level_idx must be in [0, {len(factors)}], got {level_idx}")
    if synapses_per_branch < 1:
        raise ValueError("synapses_per_branch must be positive")

    anatomical_depth = len(factors) - level_idx
    active_factors = factors[:anatomical_depth]
    branches_per_owner = math.prod(active_factors)
    expected_out_features = owner_count * branches_per_owner
    if int(out_features) != expected_out_features:
        raise ValueError(
            "Spatial morphology does not match the DendriNet branch layout: "
            f"expected {expected_out_features} rows, got {out_features}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % ((1 << 63) - 1))
    rectangles = spatial_morphology_rectangles(
        input_shape=input_shape,
        branch_factors=factors,
        level_idx=level_idx,
        split_axes=split_axes,
        region_mode=region_mode,
        minimum_window_shape=minimum_window_shape,
        seed=seed if region_seed is None else region_seed,
    )
    indices = torch.empty(
        int(out_features),
        int(synapses_per_branch),
        dtype=torch.long,
    )
    owner_offsets = torch.arange(owner_count, dtype=torch.long) * branches_per_owner

    for local_branch_idx, rectangle in enumerate(rectangles):
        candidates = _rectangle_candidates(
            channels=channels,
            height=height,
            width=width,
            rectangle=rectangle,
        )
        rows = owner_offsets + local_branch_idx
        indices[rows] = _sample_unique_candidates(
            candidates,
            n_rows=owner_count,
            synapses_per_branch=synapses_per_branch,
            generator=generator,
        )
    return indices.to(
        dtype=_resolve_index_dtype(index_dtype, in_features=int(in_features))
    )


def sample_configured_indices(
    config,
    *,
    pathway: str,
    out_features: int,
    in_features: int,
    synapses_per_branch: int,
    owner_count: int,
    branch_factors: Sequence[int],
    layer_idx: int = 0,
    level_idx: int = 0,
    index_dtype: str | torch.dtype = "int64",
) -> torch.Tensor | None:
    """Sample direct sparse indices for a configured structured pathway."""

    pathway_config = _resolve_pathway_config(config, pathway)
    if not pathway_config:
        return None
    method = str(pathway_config.get("method", "bernoulli")).strip().lower()
    if method in INDEX_MANIFEST_METHODS:
        # Opt-in external index manifests (policy-computed supports).  Imported
        # lazily so the default random path never pays for, or depends on,
        # the manifest machinery.
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.index_manifest import (
            load_configured_manifest_indices,
        )

        return load_configured_manifest_indices(
            pathway_config,
            pathway=pathway,
            out_features=out_features,
            in_features=in_features,
            synapses_per_branch=synapses_per_branch,
            layer_idx=layer_idx,
            level_idx=level_idx,
            index_dtype=index_dtype,
        )
    if method not in SPATIAL_MORPHOLOGY_METHODS:
        return None

    spatial = _spatial_options(pathway_config)
    base_seed = int(pathway_config.get("seed", 0) or 0)
    seed = (
        base_seed
        + stable_seed_offset(
            "spatial_morphology",
            pathway,
            int(layer_idx),
            int(level_idx),
            int(out_features),
            int(in_features),
        )
    ) % ((1 << 63) - 1)
    configured_region_seed = spatial.get("region_seed")
    region_base_seed = (
        base_seed if configured_region_seed is None else int(configured_region_seed)
    )
    region_seed = (
        region_base_seed
        + stable_seed_offset(
            "spatial_morphology_regions",
            int(layer_idx),
            int(level_idx),
            int(in_features),
            tuple(int(value) for value in branch_factors),
        )
    ) % ((1 << 63) - 1)
    return sample_spatial_morphology_indices(
        input_shape=spatial.get("input_shape", ()),
        out_features=out_features,
        in_features=in_features,
        synapses_per_branch=synapses_per_branch,
        owner_count=owner_count,
        branch_factors=branch_factors,
        level_idx=level_idx,
        split_axes=spatial.get("split_axes"),
        region_mode=str(spatial.get("region_mode", "partition")),
        minimum_window_shape=spatial.get("minimum_window_shape"),
        seed=seed,
        region_seed=region_seed,
        index_dtype=index_dtype,
    )


__all__ = [
    "SPATIAL_MORPHOLOGY_METHODS",
    "SPATIAL_REGION_MODES",
    "sample_configured_indices",
    "sample_spatial_morphology_indices",
    "spatial_morphology_rectangles",
    "uses_spatial_morphology",
]
