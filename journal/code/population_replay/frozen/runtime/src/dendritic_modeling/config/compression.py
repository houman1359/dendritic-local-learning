"""Typed deployment-compression configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf

TopologyEncoding = Literal["auto", "uint", "bitmask"]
DenseCompressionBackend = Literal[
    "none",
    "torchao_int8_weight_only",
    "torchao_int4_weight_only",
    "torchao_block_sparse",
]


@dataclass
class SparseTopologyCompressionConfig:
    """Losslessly freeze dynamic sparse synapses into fixed indexed layers."""

    enabled: bool = True
    topology_encoding: TopologyEncoding = "auto"
    require_final_dense_to_sparse_step: bool = True
    allow_indexed_dynamic: bool = True
    allow_indexed_rewire: bool = True
    # Variance TopK is deterministic after eval() freezes its activation
    # statistics. Stochastic TopK requires an explicit deployment decision:
    # reject, take deterministic TopK of learned scores, or sample once using
    # ``stochastic_seed`` and freeze that exact draw.
    allow_variance_topk: bool = True
    stochastic_topk_policy: str = "reject"
    stochastic_seed: int = 0
    # DeepST/credit-DeepST export snapshots the current learned mask. Ragged
    # row degrees are represented physically as CSR instead of padded weights.
    allow_deepst_snapshot: bool = True
    ragged_topology_format: str = "csr"

    def validate(self) -> None:
        if self.topology_encoding not in {"auto", "uint", "bitmask"}:
            raise ValueError("topology_encoding must be 'auto', 'uint', or 'bitmask'")
        if self.stochastic_topk_policy not in {
            "reject",
            "deterministic_topk",
            "sample_once",
        }:
            raise ValueError(
                "stochastic_topk_policy must be 'reject', "
                "'deterministic_topk', or 'sample_once'"
            )
        if self.ragged_topology_format not in {"reject", "csr"}:
            raise ValueError("ragged_topology_format must be 'reject' or 'csr'")


@dataclass
class DenseRuntimeCompressionConfig:
    """Optional TorchAO transform for conventional dense linear layers.

    These transforms are inference optimizations and are intentionally
    separate from native dendritic topology freezing. They may change model
    numerics and require the optional ``compression`` dependency.
    """

    backend: DenseCompressionBackend = "none"
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    min_in_features: int = 256
    min_out_features: int = 256
    int4_group_size: int = 128
    block_size: int = 64
    minimum_block_sparsity: float = 0.8

    def validate(self) -> None:
        supported = {
            "none",
            "torchao_int8_weight_only",
            "torchao_int4_weight_only",
            "torchao_block_sparse",
        }
        if self.backend not in supported:
            raise ValueError(
                f"Unsupported dense compression backend {self.backend!r}; "
                f"expected one of {sorted(supported)}"
            )
        if self.min_in_features < 1 or self.min_out_features < 1:
            raise ValueError("minimum dense layer dimensions must be positive")
        if self.int4_group_size not in {32, 64, 128, 256}:
            raise ValueError("int4_group_size must be one of 32, 64, 128, or 256")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if not 0.0 <= self.minimum_block_sparsity <= 1.0:
            raise ValueError("minimum_block_sparsity must be in [0, 1]")


@dataclass
class CompressionVerificationConfig:
    """Numerical verification policy for a compacted model."""

    enabled: bool = True
    # Indexed reductions can differ from masked dense reductions by a few
    # float32 ULPs because the summation order changes, even when support and
    # stored weights are identical.
    atol: float = 1e-5
    rtol: float = 1e-5

    def validate(self) -> None:
        if self.atol < 0 or self.rtol < 0:
            raise ValueError("verification tolerances must be non-negative")


@dataclass
class ModelCompressionConfig:
    """Complete model deployment-compression policy."""

    sparse_topology: SparseTopologyCompressionConfig = field(
        default_factory=SparseTopologyCompressionConfig
    )
    dense_runtime: DenseRuntimeCompressionConfig = field(
        default_factory=DenseRuntimeCompressionConfig
    )
    verification: CompressionVerificationConfig = field(
        default_factory=CompressionVerificationConfig
    )

    def __post_init__(self) -> None:
        if isinstance(self.sparse_topology, dict):
            self.sparse_topology = SparseTopologyCompressionConfig(
                **self.sparse_topology
            )
        if isinstance(self.dense_runtime, dict):
            self.dense_runtime = DenseRuntimeCompressionConfig(**self.dense_runtime)
        if isinstance(self.verification, dict):
            self.verification = CompressionVerificationConfig(**self.verification)
        self.validate()

    def validate(self) -> None:
        self.sparse_topology.validate()
        self.dense_runtime.validate()
        self.verification.validate()

    @classmethod
    def load(cls, path: str | Path) -> ModelCompressionConfig:
        """Load a standalone YAML compression policy."""

        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        payload: Any = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
        if not isinstance(payload, dict):
            raise TypeError("Compression config must contain a YAML mapping")
        unknown = set(payload) - {"sparse_topology", "dense_runtime", "verification"}
        if unknown:
            raise ValueError(
                "Unknown compression config field(s): " + ", ".join(sorted(unknown))
            )
        return cls(**payload)


__all__ = [
    "CompressionVerificationConfig",
    "DenseRuntimeCompressionConfig",
    "ModelCompressionConfig",
    "SparseTopologyCompressionConfig",
]
