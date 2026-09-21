"""Sparse heterogeneous-leak point RNN with an auditable E/I state budget."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.degree_grouped_indexed import (
    DegreeGroupedIndexedLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
    normalize_indexed_projection_backend,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    apply_recurrent_weight_cache,
)
from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.utils.weight_transforms import inverse_weight_transform
from dendritic_modeling.utils.stable_hash import stable_seed_offset


@dataclass
class TauOwnerPatternConfig:
    """Repeated within-output time-constant composition."""

    owner_count: int
    tau_counts: list[int]
    excitatory_owner_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("owner_count", self.owner_count),
            ("excitatory_owner_count", self.excitatory_owner_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.owner_count < 1:
            raise ValueError("owner_count must be positive")
        if self.excitatory_owner_count > self.owner_count:
            raise ValueError("excitatory_owner_count cannot exceed owner_count")
        if not self.tau_counts or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.tau_counts
        ):
            raise ValueError("tau_counts must contain non-negative integers")


@dataclass
class TauFeatureInputRoutingConfig:
    """Hard raw-input feature ranges assigned to each configured tau bank."""

    ranges_by_tau: list[list[tuple[int, int] | list[int]]]
    method: str = "tau_feature_blocks"
    require_source_coverage: bool = True

    def __post_init__(self) -> None:
        if self.method != "tau_feature_blocks":
            raise ValueError("input_routing method must be 'tau_feature_blocks'")
        if not isinstance(self.require_source_coverage, bool):
            raise TypeError("input_routing require_source_coverage must be boolean")
        if not isinstance(self.ranges_by_tau, (list, tuple)):
            raise TypeError("input_routing ranges_by_tau must be a sequence")

        normalized: list[list[tuple[int, int]]] = []
        for tau_ranges in self.ranges_by_tau:
            if not isinstance(tau_ranges, (list, tuple)) or not tau_ranges:
                raise ValueError(
                    "every input_routing ranges_by_tau entry must be nonempty"
                )
            normalized_tau_ranges = []
            for bounds in tau_ranges:
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError(
                        "input_routing feature ranges must be [start, stop] pairs"
                    )
                start, stop = bounds
                if any(
                    isinstance(value, bool) or not isinstance(value, Integral)
                    for value in (start, stop)
                ):
                    raise TypeError(
                        "input_routing feature-range bounds must be integers"
                    )
                normalized_tau_ranges.append((int(start), int(stop)))
            normalized.append(normalized_tau_ranges)
        self.ranges_by_tau = normalized


def _paper_tau_owner_patterns() -> list[TauOwnerPatternConfig]:
    return [
        TauOwnerPatternConfig(
            owner_count=48,
            tau_counts=[22, 10, 3],
            excitatory_owner_count=38,
        ),
        TauOwnerPatternConfig(
            owner_count=32,
            tau_counts=[21, 9, 5],
            excitatory_owner_count=26,
        ),
    ]


@dataclass
class HeterogeneousLeakCTRNNConfig:
    """Configuration for a flat multi-timescale point E/I recurrent core."""

    input_dim: int = 11
    input_projection_dims: list[int] = field(default_factory=lambda: [64])
    n_excitatory_outputs: int = 64
    n_inhibitory_outputs: int = 16
    channels_per_output: int = 35
    tau_values: list[float] = field(default_factory=lambda: [125.0, 25.0, 5.0])
    owner_tau_patterns: list[TauOwnerPatternConfig | dict[str, Any]] = field(
        default_factory=_paper_tau_owner_patterns
    )
    input_routing: TauFeatureInputRoutingConfig | dict[str, Any] | None = None
    dt: float = 1.0
    input_contacts: int = 6784
    recurrent_excitatory_contacts: int = 5056
    recurrent_inhibitory_contacts: int = 2528
    readout_contacts: int = 768
    input_scale: float = 1.0
    recurrent_excitatory_scale: float = 0.8
    recurrent_inhibitory_scale: float = 1.0
    readout_scale: float = 1.0
    excitatory_readout_mode: str = "positive_linear"
    activation_slope: float = 1.5
    activation_midpoint: float = 0.5
    topology_seed: int = 215058547
    initialization_seed: int = 1319312133
    output_mode: str = "last"
    indexed_output_chunk_size: int = 2048
    indexed_workspace_mb: float | None = None
    indexed_cache_transformed_weights: bool = False
    indexed_recompute_backward: bool = False
    indexed_persistent_indices: bool = True
    indexed_projection_backend: str = "eager"

    def __post_init__(self) -> None:
        self.owner_tau_patterns = [
            (
                pattern
                if isinstance(pattern, TauOwnerPatternConfig)
                else TauOwnerPatternConfig(**pattern)
            )
            for pattern in self.owner_tau_patterns
        ]
        if isinstance(self.input_routing, dict):
            self.input_routing = TauFeatureInputRoutingConfig(**self.input_routing)
        elif self.input_routing is not None and not isinstance(
            self.input_routing, TauFeatureInputRoutingConfig
        ):
            raise TypeError(
                "input_routing must be a TauFeatureInputRoutingConfig, mapping, or None"
            )
        integer_fields = (
            "input_dim",
            "n_excitatory_outputs",
            "n_inhibitory_outputs",
            "channels_per_output",
            "input_contacts",
            "recurrent_excitatory_contacts",
            "recurrent_inhibitory_contacts",
            "readout_contacts",
            "indexed_output_chunk_size",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("topology_seed", "initialization_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.input_projection_dims
        ):
            raise ValueError(
                "input_projection_dims must be empty or contain positive integers"
            )
        if not self.tau_values:
            raise ValueError("tau_values must contain positive values that are finite")
        resolved_tau_values: list[float] = []
        for value in self.tau_values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("tau_values must contain real numbers")
            resolved = float(value)
            if not math.isfinite(resolved) or resolved <= 0.0:
                raise ValueError(
                    "tau_values must contain positive values that are finite"
                )
            resolved_tau_values.append(resolved)
        if len(set(resolved_tau_values)) != len(resolved_tau_values):
            raise ValueError("tau_values must be unique")
        self.tau_values = resolved_tau_values
        if not math.isfinite(float(self.dt)) or float(self.dt) <= 0.0:
            raise ValueError("dt must be positive and finite")
        for name in (
            "input_scale",
            "recurrent_excitatory_scale",
            "recurrent_inhibitory_scale",
            "readout_scale",
            "activation_slope",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isfinite(float(self.activation_midpoint)):
            raise ValueError("activation_midpoint must be finite")
        if not isinstance(self.excitatory_readout_mode, str) or (
            self.excitatory_readout_mode not in {"positive_linear", "convex"}
        ):
            raise ValueError(
                "excitatory_readout_mode must be 'positive_linear' or 'convex'"
            )
        if (
            self.excitatory_readout_mode == "convex"
            and float(self.readout_scale) != 1.0
        ):
            raise ValueError(
                "readout_scale must equal 1.0 when excitatory_readout_mode is 'convex'"
            )
        if self.output_mode not in {"last", "mean", "all"}:
            raise ValueError("output_mode must be 'last', 'mean', or 'all'")
        if self.indexed_workspace_mb is not None and self.indexed_workspace_mb <= 0:
            raise ValueError("indexed_workspace_mb must be positive when provided")
        self.indexed_projection_backend = normalize_indexed_projection_backend(
            self.indexed_projection_backend
        )
        if self.indexed_recompute_backward and self.indexed_projection_backend not in {
            "eager",
            "recompute",
        }:
            raise ValueError(
                "indexed_recompute_backward cannot be combined with "
                f"indexed_projection_backend={self.indexed_projection_backend!r}"
            )

        n_outputs = self.n_outputs
        if sum(pattern.owner_count for pattern in self.owner_tau_patterns) != n_outputs:
            raise ValueError("owner_tau_patterns must account for every output owner")
        if (
            sum(pattern.excitatory_owner_count for pattern in self.owner_tau_patterns)
            != self.n_excitatory_outputs
        ):
            raise ValueError(
                "owner_tau_patterns excitatory counts must match n_excitatory_outputs"
            )
        for pattern in self.owner_tau_patterns:
            if len(pattern.tau_counts) != len(self.tau_values):
                raise ValueError(
                    "every owner tau pattern must match the number of tau_values"
                )
            if sum(pattern.tau_counts) != self.channels_per_output:
                raise ValueError(
                    "every owner tau pattern must sum to channels_per_output"
                )

        self._validate_input_routing()

        projected_dim = self.projected_input_dim
        self._validate_contact_budget(
            "input_contacts",
            self.input_contacts,
            sources=projected_dim,
            require_each_output=True,
        )
        self._validate_contact_budget(
            "recurrent_excitatory_contacts",
            self.recurrent_excitatory_contacts,
            sources=self.n_excitatory_outputs,
            require_each_output=True,
        )
        self._validate_contact_budget(
            "recurrent_inhibitory_contacts",
            self.recurrent_inhibitory_contacts,
            sources=self.n_inhibitory_outputs,
            require_each_output=False,
        )
        if self.recurrent_inhibitory_contacts < self.n_inhibitory_outputs:
            raise ValueError(
                "recurrent_inhibitory_contacts must cover every inhibitory source"
            )
        maximum_readout = self.n_excitatory_outputs * self.channels_per_output
        if not self.n_excitatory_outputs <= self.readout_contacts <= maximum_readout:
            raise ValueError(
                "readout_contacts must give every excitatory output at least one "
                "within-owner contact and cannot exceed the within-owner support"
            )
        if self.readout_contacts % self.n_excitatory_outputs:
            raise ValueError(
                "readout_contacts must be divisible by n_excitatory_outputs"
            )

    @property
    def n_outputs(self) -> int:
        return self.n_excitatory_outputs + self.n_inhibitory_outputs

    @property
    def hidden_dim(self) -> int:
        return self.n_outputs * self.channels_per_output

    @property
    def projected_input_dim(self) -> int:
        if self.input_projection_dims:
            return self.input_projection_dims[-1]
        return self.input_dim

    def _validate_input_routing(self) -> None:
        routing = self.input_routing
        if routing is None:
            return
        if self.input_projection_dims:
            raise ValueError(
                "tau_feature_blocks input_routing requires input_projection_dims=[] "
                "so hard routing applies to raw input features"
            )
        if len(routing.ranges_by_tau) != len(self.tau_values):
            raise ValueError(
                "input_routing ranges_by_tau must have one entry per tau_values entry"
            )

        maximum_row_degree = math.ceil(self.input_contacts / self.hidden_dim)
        all_allowed = torch.zeros(self.input_dim, dtype=torch.bool)
        for tau_index, tau_ranges in enumerate(routing.ranges_by_tau):
            allowed = torch.zeros(self.input_dim, dtype=torch.bool)
            for start, stop in tau_ranges:
                if start < 0 or stop > self.input_dim or stop <= start:
                    raise ValueError(
                        "input_routing feature ranges must satisfy "
                        "0 <= start < stop <= input_dim; "
                        f"got [{start}, {stop}] for tau index {tau_index}"
                    )
                if bool(allowed[start:stop].any()):
                    raise ValueError(
                        "input_routing feature ranges must not overlap within a "
                        f"tau entry; overlap at tau index {tau_index}"
                    )
                allowed[start:stop] = True
            if int(allowed.sum().item()) < maximum_row_degree:
                raise ValueError(
                    "input_routing must allow at least the maximum input row "
                    f"degree ({maximum_row_degree}) for tau index {tau_index}"
                )
            all_allowed |= allowed
        if routing.require_source_coverage and not bool(all_allowed.all()):
            missing = torch.nonzero(~all_allowed, as_tuple=False).flatten().tolist()
            raise ValueError(
                "input_routing excludes raw features required for source coverage: "
                f"{missing}"
            )

    def _validate_contact_budget(
        self,
        name: str,
        contacts: int,
        *,
        sources: int,
        require_each_output: bool,
    ) -> None:
        maximum = self.hidden_dim * sources
        minimum = self.hidden_dim if require_each_output else 1
        if not minimum <= contacts <= maximum:
            raise ValueError(
                f"{name} must be in [{minimum}, {maximum}], got {contacts}"
            )


@dataclass
class HeterogeneousLeakCTRNNState:
    """Leaky channel voltages and delayed point-population outputs."""

    voltage: torch.Tensor
    outputs: torch.Tensor

    def detach(self) -> HeterogeneousLeakCTRNNState:
        return HeterogeneousLeakCTRNNState(
            voltage=self.voltage.detach(),
            outputs=self.outputs.detach(),
        )

    def clone(self) -> HeterogeneousLeakCTRNNState:
        return HeterogeneousLeakCTRNNState(
            voltage=self.voltage.clone(),
            outputs=self.outputs.clone(),
        )

    def to(self, device: torch.device) -> HeterogeneousLeakCTRNNState:
        return HeterogeneousLeakCTRNNState(
            voltage=self.voltage.to(device),
            outputs=self.outputs.to(device),
        )


def _seed(base: int, *parts: object) -> int:
    return (int(base) + stable_seed_offset("heterogeneous_leak_ctrnn", *parts)) % (
        (1 << 63) - 1
    )


def _balanced_degree_vector(
    *,
    total_contacts: int,
    source_dim: int,
    categories: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Allocate floor/ceiling row degrees while stratifying category counts."""

    output_dim = int(categories.numel())
    base, remainder = divmod(int(total_contacts), output_dim)
    if base > source_dim or (base == source_dim and remainder):
        raise ValueError("contact budget exceeds available unique row sources")
    degrees = torch.full((output_dim,), base, dtype=torch.long)
    if remainder == 0:
        return degrees

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    unique_categories = torch.unique(categories, sorted=True)
    category_sizes = torch.tensor(
        [int((categories == category).sum().item()) for category in unique_categories],
        dtype=torch.long,
    )
    ideal = category_sizes.to(torch.float64) * (float(remainder) / output_dim)
    extras = torch.floor(ideal).to(torch.long)
    remaining = remainder - int(extras.sum().item())
    if remaining:
        fractions = ideal - extras.to(torch.float64)
        tie_break = torch.rand(fractions.shape, generator=generator) * 1e-9
        order = torch.argsort(fractions + tie_break, descending=True)
        extras[order[:remaining]] += 1

    for category, count in zip(
        unique_categories.tolist(), extras.tolist(), strict=True
    ):
        if count == 0:
            continue
        rows = torch.nonzero(categories == category, as_tuple=False).flatten()
        selected = rows[torch.randperm(rows.numel(), generator=generator)[:count]]
        degrees[selected] += 1
    if int(degrees.sum().item()) != total_contacts:
        raise RuntimeError("degree stratification changed the contact budget")
    return degrees


class HeterogeneousLeakCTRNN(BaseNetwork):
    """Flat point RNN with an exact heterogeneous-leak and contact ledger."""

    def __init__(self, config: HeterogeneousLeakCTRNNConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.n_excitatory_outputs
        self._output_mode = config.output_mode

        self.input_projection = self._build_input_projection()
        taus, tau_indices, owner_polarity = self._build_state_layout()
        self.register_buffer("taus", taus)
        self.register_buffer("tau_indices", tau_indices)
        self.register_buffer("owner_polarity", owner_polarity)
        self.register_buffer("decays", torch.exp(-float(config.dt) / taus))
        owner_indices = torch.arange(config.n_outputs).repeat_interleave(
            config.channels_per_output
        )
        self.register_buffer("owner_indices", owner_indices)

        categories = owner_polarity[self.owner_indices] * len(config.tau_values)
        categories = categories + tau_indices
        projection_dim = config.projected_input_dim
        input_connection_mask = self._build_input_connection_mask()
        common_sparse = {
            "weight_transform": "exp",
            "output_chunk_size": config.indexed_output_chunk_size,
            "workspace_mb": config.indexed_workspace_mb,
            "cache_transformed_weights": (config.indexed_cache_transformed_weights),
            "recompute_backward": config.indexed_recompute_backward,
            "projection_backend": config.indexed_projection_backend,
            "persistent_indices": config.indexed_persistent_indices,
        }
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed(config.initialization_seed, "input_drive"))
            self.input_drive = DegreeGroupedIndexedLinear(
                projection_dim,
                config.hidden_dim,
                _balanced_degree_vector(
                    total_contacts=config.input_contacts,
                    source_dim=projection_dim,
                    categories=categories,
                    seed=_seed(config.topology_seed, "input_degrees"),
                ),
                seed=_seed(config.topology_seed, "input_support"),
                require_source_coverage=(
                    True
                    if config.input_routing is None
                    else config.input_routing.require_source_coverage
                ),
                connection_mask=input_connection_mask,
                row_sum_init=config.input_scale,
                **common_sparse,
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed(config.initialization_seed, "recurrent_excitation"))
            self.recurrent_excitation = DegreeGroupedIndexedLinear(
                config.n_excitatory_outputs,
                config.hidden_dim,
                _balanced_degree_vector(
                    total_contacts=config.recurrent_excitatory_contacts,
                    source_dim=config.n_excitatory_outputs,
                    categories=categories,
                    seed=_seed(config.topology_seed, "rec_e_degrees"),
                ),
                seed=_seed(config.topology_seed, "rec_e_support"),
                require_source_coverage=True,
                row_sum_init=config.recurrent_excitatory_scale,
                **common_sparse,
            )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed(config.initialization_seed, "recurrent_inhibition"))
            self.recurrent_inhibition = DegreeGroupedIndexedLinear(
                config.n_inhibitory_outputs,
                config.hidden_dim,
                _balanced_degree_vector(
                    total_contacts=config.recurrent_inhibitory_contacts,
                    source_dim=config.n_inhibitory_outputs,
                    categories=categories,
                    seed=_seed(config.topology_seed, "rec_i_degrees"),
                ),
                seed=_seed(config.topology_seed, "rec_i_support"),
                require_source_coverage=True,
                row_sum_init=config.recurrent_inhibitory_scale,
                **common_sparse,
            )
        self.exc_readout = self._build_excitatory_readout(common_sparse)
        self.bias = nn.Parameter(torch.zeros(config.hidden_dim))

    @property
    def is_recurrent(self) -> bool:
        return True

    @property
    def current_taus(self) -> torch.Tensor:
        return self.taus

    def _build_input_projection(self) -> nn.Module:
        if not self.config.input_projection_dims:
            return nn.Identity()
        layers: list[nn.Module] = []
        previous = self.config.input_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(self.config.initialization_seed))
            for width in self.config.input_projection_dims:
                layers.append(nn.Linear(previous, width))
                layers.append(nn.ReLU())
                previous = width
        return nn.Sequential(*layers)

    def _build_input_connection_mask(self) -> torch.Tensor | None:
        routing = self.config.input_routing
        if routing is None:
            return None
        mask = torch.zeros(
            self.config.hidden_dim,
            self.config.input_dim,
            dtype=torch.bool,
        )
        for tau_index, tau_ranges in enumerate(routing.ranges_by_tau):
            allowed = torch.zeros(self.config.input_dim, dtype=torch.bool)
            for start, stop in tau_ranges:
                allowed[start:stop] = True
            mask[self.tau_indices == tau_index] = allowed
        return mask

    def _build_state_layout(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        excitatory_patterns: list[TauOwnerPatternConfig] = []
        inhibitory_patterns: list[TauOwnerPatternConfig] = []
        for pattern in self.config.owner_tau_patterns:
            excitatory_patterns.extend([pattern] * pattern.excitatory_owner_count)
            inhibitory_patterns.extend(
                [pattern] * (pattern.owner_count - pattern.excitatory_owner_count)
            )
        owner_patterns = [*excitatory_patterns, *inhibitory_patterns]
        if len(owner_patterns) != self.config.n_outputs:
            raise RuntimeError("owner pattern construction changed the owner count")

        tau_values: list[torch.Tensor] = []
        tau_indices: list[torch.Tensor] = []
        for owner_index, pattern in enumerate(owner_patterns):
            indices = torch.repeat_interleave(
                torch.arange(len(self.config.tau_values)),
                torch.tensor(pattern.tau_counts),
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _seed(self.config.topology_seed, "owner_tau_order", owner_index)
            )
            indices = indices[torch.randperm(indices.numel(), generator=generator)]
            tau_indices.append(indices)
            tau_values.append(
                torch.tensor(self.config.tau_values, dtype=torch.float32)[indices]
            )
        polarity = torch.cat(
            (
                torch.zeros(self.config.n_excitatory_outputs, dtype=torch.long),
                torch.ones(self.config.n_inhibitory_outputs, dtype=torch.long),
            )
        )
        return torch.cat(tau_values), torch.cat(tau_indices), polarity

    def _build_excitatory_readout(
        self, common_sparse: dict[str, Any]
    ) -> IndexedSparseLinear:
        contacts = self.config.readout_contacts
        n_outputs = self.config.n_excitatory_outputs
        degree, remainder = divmod(contacts, n_outputs)
        if remainder:
            raise ValueError(
                "readout_contacts must be divisible by n_excitatory_outputs "
                "for the within-owner readout"
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_seed(self.config.topology_seed, "readout_support"))
        rows = []
        connection_mask = torch.zeros(
            n_outputs,
            self.config.hidden_dim,
            dtype=torch.bool,
        )
        for owner in range(n_outputs):
            owner_start = owner * self.config.channels_per_output
            connection_mask[
                owner,
                owner_start : owner_start + self.config.channels_per_output,
            ] = True
            local = torch.randperm(
                self.config.channels_per_output,
                generator=generator,
            )[:degree]
            rows.append(local + owner_start)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                _seed(self.config.initialization_seed, "excitatory_readout")
            )
            projection = IndexedSparseLinear(
                in_features=self.config.hidden_dim,
                out_features=n_outputs,
                K=degree,
                connection_indices=torch.stack(rows),
                connection_mask=connection_mask,
                init_method="xavier_normal",
                weight_transform="exp",
                output_chunk_size=common_sparse["output_chunk_size"],
                workspace_mb=common_sparse["workspace_mb"],
                cache_transformed_weights=common_sparse["cache_transformed_weights"],
                recompute_backward=common_sparse["recompute_backward"],
                projection_backend=common_sparse["projection_backend"],
                persistent_indices=common_sparse["persistent_indices"],
            )
        per_contact = self.config.readout_scale / degree
        raw_value = inverse_weight_transform(
            torch.tensor(per_contact, dtype=projection.pre_w.dtype),
            "exp",
        )
        with torch.no_grad():
            projection.pre_w.fill_(float(raw_value.item()))
        return projection

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> HeterogeneousLeakCTRNNState:
        return HeterogeneousLeakCTRNNState(
            voltage=torch.zeros(
                batch_size,
                self.config.hidden_dim,
                device=device,
                dtype=dtype,
            ),
            outputs=torch.zeros(
                batch_size,
                self.config.n_outputs,
                device=device,
                dtype=dtype,
            ),
        )

    def _activation(self, voltage: torch.Tensor) -> torch.Tensor:
        slope = float(self.config.activation_slope)
        midpoint = float(self.config.activation_midpoint)
        return 0.5 * (torch.tanh(slope * (voltage - midpoint)) + 1.0)

    def _validate_state(
        self,
        state: HeterogeneousLeakCTRNNState,
        x: torch.Tensor,
    ) -> None:
        if not isinstance(state, HeterogeneousLeakCTRNNState):
            raise TypeError(
                f"state must be HeterogeneousLeakCTRNNState, got {type(state).__name__}"
            )
        expected_voltage = (x.shape[0], self.config.hidden_dim)
        expected_outputs = (x.shape[0], self.config.n_outputs)
        if tuple(state.voltage.shape) != expected_voltage:
            raise ValueError(
                f"state.voltage must have shape {expected_voltage}, got "
                f"{tuple(state.voltage.shape)}"
            )
        if tuple(state.outputs.shape) != expected_outputs:
            raise ValueError(
                f"state.outputs must have shape {expected_outputs}, got "
                f"{tuple(state.outputs.shape)}"
            )
        if (
            state.voltage.device != x.device
            or state.outputs.device != x.device
            or state.voltage.dtype != x.dtype
            or state.outputs.dtype != x.dtype
        ):
            raise ValueError("input and recurrent state must share device and dtype")

    def _convex_excitatory_readout(self, rates: torch.Tensor) -> torch.Tensor:
        """Project selected channels with stable row-normalized conductances."""

        projection = self.exc_readout
        indices = projection.connection_indices.to(
            device=rates.device,
            dtype=torch.long,
        )
        weights = torch.softmax(projection.pre_w, dim=-1)
        output_chunks = []
        chunk_size = projection._forward_output_chunk_size(rates)
        for start in range(0, projection.out_features, chunk_size):
            end = min(start + chunk_size, projection.out_features)
            chunk_indices = indices[start:end].reshape(-1)
            selected = rates.index_select(1, chunk_indices).reshape(
                rates.shape[0],
                end - start,
                projection.K,
            )
            output_chunks.append(
                (selected * weights[start:end].unsqueeze(0)).sum(dim=-1)
            )
        return torch.cat(output_chunks, dim=-1)

    def step(
        self,
        x: torch.Tensor,
        state: HeterogeneousLeakCTRNNState | None = None,
    ) -> tuple[torch.Tensor, HeterogeneousLeakCTRNNState]:
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("x must have shape [batch, input_dim]")
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"x must have input dimension {self.config.input_dim}, got {x.shape[-1]}"
            )
        if state is None:
            state = self.init_state(x.shape[0], x.device, x.dtype)
        self._validate_state(state, x)

        projected = self.input_projection(x)
        exc_previous = state.outputs[:, : self.config.n_excitatory_outputs]
        inh_previous = state.outputs[:, self.config.n_excitatory_outputs :]
        drive = (
            self.input_drive(projected)
            + self.recurrent_excitation(exc_previous)
            - self.recurrent_inhibition(inh_previous)
            + self.bias.to(device=x.device, dtype=x.dtype)
        )
        decay = self.decays.to(device=x.device, dtype=x.dtype)
        voltage = decay * state.voltage + (1.0 - decay) * drive
        rates = self._activation(voltage)
        owner_means = rates.reshape(
            x.shape[0],
            self.config.n_outputs,
            self.config.channels_per_output,
        ).mean(dim=-1)
        if self.config.excitatory_readout_mode == "convex":
            learned_exc = self._convex_excitatory_readout(rates)
        else:
            learned_exc = self.exc_readout(rates)
        exc_outputs = 0.5 * (
            owner_means[:, : self.config.n_excitatory_outputs] + learned_exc
        )
        outputs = torch.cat(
            (exc_outputs, owner_means[:, self.config.n_excitatory_outputs :]),
            dim=-1,
        )
        return exc_outputs, HeterogeneousLeakCTRNNState(voltage, outputs)

    def _reduce_outputs(
        self,
        outputs: torch.Tensor,
        seq_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        if seq_lengths is not None:
            if not isinstance(seq_lengths, torch.Tensor) or tuple(
                seq_lengths.shape
            ) != (outputs.shape[0],):
                raise ValueError("seq_lengths must have shape [batch]")
            integer_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if seq_lengths.dtype not in integer_dtypes:
                raise TypeError("seq_lengths must have an integer dtype")
            lengths = seq_lengths.to(device=outputs.device, dtype=torch.long)
            if bool((lengths < 0).any()) or bool((lengths > outputs.shape[1]).any()):
                raise ValueError("seq_lengths entries must be in [0, time]")
        else:
            lengths = None

        if self._output_mode == "last":
            if lengths is None:
                return outputs[:, -1]
            indices = (lengths - 1).clamp(0, outputs.shape[1] - 1)
            selected = outputs[
                torch.arange(outputs.shape[0], device=outputs.device),
                indices,
            ]
            return torch.where(
                lengths[:, None] > 0, selected, torch.zeros_like(selected)
            )
        if self._output_mode == "mean":
            if lengths is None:
                return outputs.mean(dim=1)
            mask = torch.arange(outputs.shape[1], device=outputs.device)[None, :]
            mask = mask < lengths[:, None]
            denominator = lengths.clamp_min(1).to(outputs.dtype)[:, None]
            return (outputs * mask[..., None].to(outputs.dtype)).sum(
                dim=1
            ) / denominator
        if self._output_mode == "all":
            if lengths is None:
                return outputs
            mask = torch.arange(outputs.shape[1], device=outputs.device)[None, :]
            return outputs * (mask < lengths[:, None])[..., None].to(outputs.dtype)
        raise RuntimeError(f"unsupported output mode {self._output_mode!r}")

    def forward(
        self,
        x: torch.Tensor,
        hidden: HeterogeneousLeakCTRNNState | None = None,
        return_hidden: bool = False,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, HeterogeneousLeakCTRNNState]:
        if not isinstance(x, torch.Tensor) or x.ndim != 3:
            raise ValueError("x must have shape [batch, time, input_dim]")
        if x.shape[1] < 1:
            raise ValueError("x must contain at least one timestep")
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"x must have input dimension {self.config.input_dim}, got {x.shape[-1]}"
            )
        state = hidden
        sequence_outputs = []
        apply_recurrent_weight_cache(self, True)
        try:
            for timestep in range(x.shape[1]):
                output, state = self.step(x[:, timestep], state)
                sequence_outputs.append(output)
        finally:
            apply_recurrent_weight_cache(self, False)
        assert state is not None
        reduced = self._reduce_outputs(
            torch.stack(sequence_outputs, dim=1),
            seq_lengths,
        )
        if return_hidden:
            return reduced, state
        return reduced

    def connectivity_resource_records(self) -> list[dict[str, Any]]:
        """Return generic records consumed by exact model-resource accounting."""

        modules = (
            ("input_drive", "ff_excitatory", self.input_drive),
            ("recurrent_excitation", "rec_excitatory", self.recurrent_excitation),
            ("recurrent_inhibition", "rec_inhibitory", self.recurrent_inhibition),
            ("exc_readout", "readout", self.exc_readout),
        )
        records = []
        for name, pathway, module in modules:
            counts = module.connectivity_resource_counts()
            candidate_parameters = int(
                counts.get(
                    "candidate_parameters",
                    sum(parameter.numel() for parameter in module.parameters()),
                )
            )
            records.append(
                {
                    "module": name,
                    "pathway": pathway,
                    "soma_relative_depth": -1,
                    "out_features": int(counts["out_features"]),
                    "in_features": int(counts["in_features"]),
                    "candidate_parameters": candidate_parameters,
                    "candidate_slots": int(counts["candidate_slots"]),
                    "active_synapses": int(counts["active_synapses"]),
                    "realized_k_min": int(counts["realized_k_min"]),
                    "realized_k_max": int(counts["realized_k_max"]),
                    "realized_k_mean": float(counts["realized_k_mean"]),
                    "selection_policy": str(counts["selection_policy"]),
                    "mask_source": str(counts["mask_source"]),
                }
            )
        return records

    def resource_ledger(self) -> dict[str, Any]:
        """Return the mechanism-matching quantities not captured by parameters."""

        tau_counts = {
            str(float(tau)): int((self.taus == float(tau)).sum().item())
            for tau in self.config.tau_values
        }
        records = self.connectivity_resource_records()
        return {
            "leaky_state_scalars_per_sample": self.config.hidden_dim,
            "delayed_output_scalars_per_sample": self.config.n_outputs,
            "reachable_carried_state_scalars_per_sample": (
                self.config.hidden_dim + self.config.n_outputs
            ),
            "tau_counts": tau_counts,
            "fixed_channel_to_owner_wires": self.config.hidden_dim,
            "input_projection_mode": (
                "mlp" if self.config.input_projection_dims else "identity"
            ),
            "projected_input_dim": self.config.projected_input_dim,
            "input_routing_method": (
                "unrestricted"
                if self.config.input_routing is None
                else self.config.input_routing.method
            ),
            "input_routing_hard": self.config.input_routing is not None,
            "input_routing_require_source_coverage": (
                True
                if self.config.input_routing is None
                else self.config.input_routing.require_source_coverage
            ),
            "input_routing_candidate_slots": int(
                self.input_drive.connectivity_resource_counts()["candidate_slots"]
            ),
            "input_source_ranges_by_tau": (
                None
                if self.config.input_routing is None
                else {
                    str(float(tau)): [list(bounds) for bounds in ranges]
                    for tau, ranges in zip(
                        self.config.tau_values,
                        self.config.input_routing.ranges_by_tau,
                        strict=True,
                    )
                }
            ),
            "input_contacts": self.config.input_contacts,
            "recurrent_excitatory_contacts": (
                self.config.recurrent_excitatory_contacts
            ),
            "recurrent_inhibitory_contacts": (
                self.config.recurrent_inhibitory_contacts
            ),
            "learned_readout_contacts": self.config.readout_contacts,
            "excitatory_readout_mode": self.config.excitatory_readout_mode,
            "excitatory_readout_bounded": (
                self.config.excitatory_readout_mode == "convex"
            ),
            "active_learned_contacts": sum(
                int(record["active_synapses"]) for record in records
            ),
        }


__all__ = [
    "HeterogeneousLeakCTRNN",
    "HeterogeneousLeakCTRNNConfig",
    "HeterogeneousLeakCTRNNState",
    "TauFeatureInputRoutingConfig",
    "TauOwnerPatternConfig",
]
