"""Typed branch-layer connectivity configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DendriticBranchConfig:
    """Connectivity and geometry options for one dendritic branch layer."""

    output_dim: int
    excitatory_input_dim: int | None = None
    excitatory_synapses_per_branch: int | None = None
    inhibitory_input_dim: int | None = None
    inhibitory_synapses_per_branch: int | None = None
    input_branch_factor: int | None = None
    recurrent_input_dim: int | None = None
    recurrent_synapses_per_branch: int | None = None
    recurrent_forbidden_input_index_per_output: Any | None = None
    rec_inhibitory_input_dim: int | None = None
    rec_inhibitory_synapses_per_branch: int | None = None
    rec_inhibitory_forbidden_input_index_per_output: Any | None = None
    layer_idx: int = 0
    excitatory_connection_indices: Any | None = None
    inhibitory_connection_indices: Any | None = None
    excitatory_connection_mask: Any | None = None
    inhibitory_connection_mask: Any | None = None
    recurrent_connection_mask: Any | None = None
    rec_inhibitory_connection_mask: Any | None = None

    @classmethod
    def from_config(
        cls,
        config: DendriticBranchConfig | Mapping[str, Any],
    ) -> DendriticBranchConfig:
        """Normalize an existing config object or mapping."""
        if isinstance(config, cls):
            return config
        if isinstance(config, Mapping):
            return cls.from_kwargs(config)
        raise TypeError("branch_config must be a DendriticBranchConfig or mapping")

    @classmethod
    def from_kwargs(
        cls,
        mapping: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> DendriticBranchConfig:
        """Build a typed config from legacy branch-layer kwargs."""
        source = dict(mapping or {})
        source.update(kwargs)
        if source.get("output_dim") is None:
            raise ValueError("DendriticBranchConfig requires output_dim")
        return cls(
            output_dim=source["output_dim"],
            excitatory_input_dim=source.get("excitatory_input_dim", None),
            excitatory_synapses_per_branch=source.get(
                "excitatory_synapses_per_branch", None
            ),
            inhibitory_input_dim=source.get("inhibitory_input_dim", None),
            inhibitory_synapses_per_branch=source.get(
                "inhibitory_synapses_per_branch", None
            ),
            input_branch_factor=source.get("input_branch_factor", None),
            recurrent_input_dim=source.get("recurrent_input_dim", None),
            recurrent_synapses_per_branch=source.get(
                "recurrent_synapses_per_branch", None
            ),
            recurrent_forbidden_input_index_per_output=source.get(
                "recurrent_forbidden_input_index_per_output", None
            ),
            rec_inhibitory_input_dim=source.get("rec_inhibitory_input_dim", None),
            rec_inhibitory_synapses_per_branch=source.get(
                "rec_inhibitory_synapses_per_branch", None
            ),
            rec_inhibitory_forbidden_input_index_per_output=source.get(
                "rec_inhibitory_forbidden_input_index_per_output", None
            ),
            layer_idx=source.get("layer_idx", 0),
            excitatory_connection_indices=source.get(
                "excitatory_connection_indices", None
            ),
            inhibitory_connection_indices=source.get(
                "inhibitory_connection_indices", None
            ),
            excitatory_connection_mask=source.get("excitatory_connection_mask", None),
            inhibitory_connection_mask=source.get("inhibitory_connection_mask", None),
            recurrent_connection_mask=source.get("recurrent_connection_mask", None),
            rec_inhibitory_connection_mask=source.get(
                "rec_inhibitory_connection_mask", None
            ),
        )

    def to_kwargs(self) -> dict[str, Any]:
        """Convert back to legacy branch-layer kwargs."""
        return {
            "output_dim": self.output_dim,
            "excitatory_input_dim": self.excitatory_input_dim,
            "excitatory_synapses_per_branch": self.excitatory_synapses_per_branch,
            "inhibitory_input_dim": self.inhibitory_input_dim,
            "inhibitory_synapses_per_branch": self.inhibitory_synapses_per_branch,
            "input_branch_factor": self.input_branch_factor,
            "recurrent_input_dim": self.recurrent_input_dim,
            "recurrent_synapses_per_branch": self.recurrent_synapses_per_branch,
            "recurrent_forbidden_input_index_per_output": (
                self.recurrent_forbidden_input_index_per_output
            ),
            "rec_inhibitory_input_dim": self.rec_inhibitory_input_dim,
            "rec_inhibitory_synapses_per_branch": (
                self.rec_inhibitory_synapses_per_branch
            ),
            "rec_inhibitory_forbidden_input_index_per_output": (
                self.rec_inhibitory_forbidden_input_index_per_output
            ),
            "layer_idx": self.layer_idx,
            "excitatory_connection_indices": self.excitatory_connection_indices,
            "inhibitory_connection_indices": self.inhibitory_connection_indices,
            "excitatory_connection_mask": self.excitatory_connection_mask,
            "inhibitory_connection_mask": self.inhibitory_connection_mask,
            "recurrent_connection_mask": self.recurrent_connection_mask,
            "rec_inhibitory_connection_mask": self.rec_inhibitory_connection_mask,
        }


__all__ = ["DendriticBranchConfig"]
