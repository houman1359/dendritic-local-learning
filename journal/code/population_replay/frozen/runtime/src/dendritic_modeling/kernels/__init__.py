"""Dendritic-specific GPU kernels (Triton prototypes).

The indexed synapse layout is ELL format (padded-dense ``[out, K]`` indices
and weights), and profiling shows the eager gather path dominating GPU time
while materializing ``[batch, out, K]`` intermediates. These kernels fuse
gather, weight transform, and reduction — and, for the shunting kernel,
both E/I banks plus the divisive denominator — into single passes with no
intermediates. Forward/inference only in v0.
"""

from dendritic_modeling.kernels.triton_ell import (
    ell_gather_gemm,
    ell_glu_forward,
    ell_glu_train,
    ell_shunting_forward,
    reference_gather_gemm,
    reference_glu_forward,
    reference_shunting_forward,
    triton_available,
)

__all__ = [
    "ell_gather_gemm",
    "ell_glu_forward",
    "ell_glu_train",
    "ell_shunting_forward",
    "reference_gather_gemm",
    "reference_glu_forward",
    "reference_shunting_forward",
    "triton_available",
]
