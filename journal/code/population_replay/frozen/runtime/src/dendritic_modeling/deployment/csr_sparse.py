"""Physically compact CSR inference for ragged sparse projections."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CSRInferenceLinear(nn.Module):
    """Parameter-free sparse linear operator with variable contacts per row.

    CSR is the deployment fallback for trained methods such as global DeepST,
    whose final mask need not have a uniform K.  Only active values, column
    indices, and compressed row pointers are persistent.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        crow_indices: torch.Tensor,
        col_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        super().__init__()
        in_features = int(in_features)
        out_features = int(out_features)
        if in_features < 1 or out_features < 1:
            raise ValueError("CSR dimensions must be positive")
        if tuple(crow_indices.shape) != (out_features + 1,):
            raise ValueError(
                f"crow_indices must have shape ({out_features + 1},), "
                f"got {tuple(crow_indices.shape)}"
            )
        if col_indices.ndim != 1 or values.ndim != 1:
            raise ValueError("CSR col_indices and values must be one-dimensional")
        if col_indices.numel() != values.numel():
            raise ValueError("CSR col_indices and values must have equal lengths")
        if int(crow_indices[0]) != 0 or int(crow_indices[-1]) != values.numel():
            raise ValueError("CSR row pointers must span every stored value")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("crow_indices", crow_indices.to(dtype=torch.int32))
        self.register_buffer("col_indices", col_indices.to(dtype=torch.int32))
        self.register_buffer("values", values)
        self.register_buffer("_csr_weight", None, persistent=False)

    @property
    def active_weights(self) -> int:
        return int(self.values.numel())

    @property
    def row_degrees(self) -> torch.Tensor:
        return self.crow_indices[1:] - self.crow_indices[:-1]

    def _weight(self) -> torch.Tensor:
        cached = self._csr_weight
        if cached is None:
            cached = torch.sparse_csr_tensor(
                self.crow_indices,
                self.col_indices,
                self.values,
                size=(self.out_features, self.in_features),
                device=self.values.device,
                dtype=self.values.dtype,
            )
            self._csr_weight = cached
        return cached

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self._weight())

    def dense_weight(self) -> torch.Tensor:
        return self._weight().to_dense()

    def connectivity_resource_counts(self) -> dict[str, int | float | str]:
        degrees = self.row_degrees
        return {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "candidate_slots": self.out_features * self.in_features,
            "active_synapses": self.active_weights,
            "realized_k_min": int(degrees.min()),
            "realized_k_max": int(degrees.max()),
            "realized_k_mean": float(degrees.float().mean()),
            "selection_policy": "fixed_csr",
            "mask_source": "compressed_row_indices",
        }

    def parameter_estimate(self) -> dict[str, int]:
        dense = self.in_features * self.out_features
        return {
            "stored_total": self.active_weights,
            "active_total": self.active_weights,
            "dense_control": dense,
        }

    def _apply(self, fn):
        self._csr_weight = None
        return super()._apply(fn)

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        self._csr_weight = None
        super()._load_from_state_dict(*args, **kwargs)


def masked_weight_to_csr(
    weight: torch.Tensor,
    mask: torch.Tensor,
) -> CSRInferenceLinear:
    """Build exact CSR inference buffers from a dense effective weight and mask."""

    if weight.ndim != 2 or tuple(mask.shape) != tuple(weight.shape):
        raise ValueError("weight and mask must be matched two-dimensional tensors")
    active = mask.to(device=weight.device, dtype=torch.bool)
    row_degrees = active.sum(dim=1, dtype=torch.int32)
    crow = torch.zeros(
        weight.shape[0] + 1,
        dtype=torch.int32,
        device=weight.device,
    )
    crow[1:] = row_degrees.cumsum(dim=0)
    coordinates = torch.nonzero(active, as_tuple=False)
    columns = coordinates[:, 1].to(dtype=torch.int32)
    values = weight[active]
    return CSRInferenceLinear(
        weight.shape[1],
        weight.shape[0],
        crow_indices=crow,
        col_indices=columns,
        values=values,
    )


__all__ = ["CSRInferenceLinear", "masked_weight_to_csr"]
