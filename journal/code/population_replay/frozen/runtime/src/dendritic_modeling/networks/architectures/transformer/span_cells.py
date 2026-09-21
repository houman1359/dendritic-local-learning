"""Reusable span-replacement cells and key-driven checkpoint formats.

This is package architecture code. Experiment launchers may construct or load
these cells, but they must not redeclare their model definitions.

The module provides the rank_readout cell plus key-driven
checkpoint classification shared by every consumer of a span checkpoint.

WHY THIS MODULE EXISTS
----------------------
Until 2026-08-22 the span drivers supported exactly two FFN cells --
`SparseGLU` (the prescribed gated-sparse cell) and `DenseNarrowGLU` (the
distilled dense control) -- and each consumer (`olmo_span_replace.py`,
`span_recover.py`, `span_eval_reserved.py`, `llm_benchmark_suite.py`)
re-declared them locally, detecting the cell type with an ad-hoc
`"sparse" if "gate.core.connection_indices" in sd else "dense"` test. That
test is an ELSE-BRANCH FALLBACK: any checkpoint whose keys it does not
understand was silently treated as the dense control and would have been
scored as the wrong model. A third cell type makes that failure mode live,
so classification moves here and becomes an EXACT key-set match with a hard
abort (`SpanCellFormatError`) on anything unrecognized.

THE rank_readout CELL (promoted to production 2026-08-22)
---------------------------------------------------------
Provenance: `morphology_matrix.py` (transformer_replacement) morphology #6
and the identical `fmi_prospective/reference_grid.py` morphology #6, where
it won a BLIND pre-registered grid at OLMo-3-7B L12 (sub-teacher 0.9276
disjoint at 16x) and 2 of 3 layers of the 32x morphology matrix (L0 0.9873,
L31 0.9969). What it computes, verbatim from that code:

    gate = SPARSE(gate_proj, K_in)      # activation-weighted TopK, teacher-init
    up   = SPARSE(up_proj,   K_in)      # activation-weighted TopK, teacher-init
    h    = silu(gate(x)) * up(x)        # unchanged SwiGLU intermediate
    y    = down_b(down_a(h))            # DENSE rank-r readout, NOT sparse

i.e. the gate/up ("front") paths keep the standard sparse dendritic cell,
and the down projection -- the output side -- is replaced by a dense
low-rank factorization instead of a sparse one. The factors are initialized
from a BALANCED SVD of the teacher's down_proj:

    U, S, Vh = svd(W_down)              # full_matrices=False
    down_a.weight = diag(sqrt(S[:r])) @ Vh[:r]     # (r, inter)
    down_b.weight = U[:, :r] @ diag(sqrt(S[:r]))   # (d_model, r)

so down_b @ down_a is exactly the rank-r truncated SVD of W_down at init
(the singular values are split symmetrically between the two factors, which
keeps their scales comparable for a single shared learning rate). No
activation weighting is applied to the SVD: the registered grid used the
plain weight SVD and this promotion reproduces it bit-for-bit.

BUDGET-MATCHING RULE (stated here because comparisons depend on it)
-------------------------------------------------------------------
The cell must cost the same as the sparse cell at the same density, so the
rank is DERIVED, never chosen. The sparse cell's weight budget at density d
(with K_in = int(d * d_model), K_down = int(d * inter)) is

    budget = 2 * inter * K_in     (gate + up)  +  d_model * K_down  (down)

rank_readout keeps the gate/up terms identical (same K_in, same aw-TopK
init), so the ENTIRE remaining allowance is the sparse down projection's:

    remaining = budget - 2 * inter * K_in = d_model * K_down

The low-rank readout spends r * (inter + d_model) weights, hence

    r = max(8, floor(d_model * K_down / (inter + d_model)))

-- the largest rank that fits inside the sparse down projection's weight
count. The realized count is therefore never ABOVE the sparse budget and
below it by at most one rank quantum (inter + d_model weights), because
flooring discards the fractional rank. At OLMo-3-7B (d_model 4096,
inter 11008) that quantum is 15104 weights: 0.10% of the 32x budget and
0.10% of the 16x budget.

THE ONE EXCEPTION: the `max(8, ...)` floor is the registered grid's, and it
is the only way this cell can end up OVER budget -- when the allowance buys
fewer than 8 ranks the floored cell costs more than the sparse cell and the
comparison is NOT matched. That regime is unreachable at OLMo-3-7B for any
density >= 1/64 (the allowance there buys rank 46) and is reported as a
loud WARNING by `RankReadoutGLU.from_teacher` when it does occur, because a
budget overrun is invisible in a perplexity number.

Accounting basis: TRAINABLE WEIGHTS (`p.numel()` over parameters), the same
basis `DenseNarrowGLU` uses for its own budget assert. Sparse cells also
carry an int64 `connection_indices` BUFFER per projection, which the
rank_readout down path does not have; that buffer is excluded from both
sides of the comparison here exactly as it is for the dense control, and is
reported separately by the consumers' index-entry accounting.

CHECKPOINT FORMAT
-----------------
The persistent state dict is the module's own, so the three cell types are
distinguishable by an exact key-set match:

    sparse       gate/up/down . core . {pre_w, connection_indices}   (6 keys)
    dense        gate/up/down . weight                              (3 keys)
    rank_readout gate/up . core . {pre_w, connection_indices}
                 + down_a.weight + down_b.weight                    (6 keys)

`classify_span_cell` requires an EXACT match and raises otherwise, so an
unknown or partially-drifted checkpoint aborts instead of being scored.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedSparseLinear,
)

__all__ = [
    "SPAN_CELL_TYPES",
    "RankReadoutGLU",
    "SpanCellFormatError",
    "SparseLinearFP32",
    "classify_span_cell",
    "rank_readout_param_count",
    "rank_readout_rank",
    "sparse_cell_param_count",
]

SPAN_CELL_TYPES = ("sparse", "dense", "rank_readout")

# Topology seeds, shared with every driver's SparseGLU so a rank_readout
# gate/up core rebuilds identically to a sparse one. Irrelevant after a strict
# state-dict load (indices and weights both come from the checkpoint) but kept
# fixed for reproducible construction.
GATE_SEED, UP_SEED, DOWN_SEED = 31, 32, 33

# Registered minimum readout rank (morphology_matrix.py / reference_grid.py
# both wrote `max(8, ...)`). It binds only where the sparse down projection's
# weight budget buys fewer than 8 ranks -- impossible at OLMo-3-7B for any
# density >= 1/64, where the budget buys 46 -- and where it does bind the cell
# is OVER budget, which RankReadoutGLU.from_teacher reports loudly.
RANK_FLOOR = 8

_SPARSE_KEYS = frozenset(
    f"{p}.core.{k}"
    for p in ("gate", "up", "down")
    for k in ("pre_w", "connection_indices")
)
_DENSE_KEYS = frozenset(f"{p}.weight" for p in ("gate", "up", "down"))
_RANK_READOUT_KEYS = frozenset(
    [f"{p}.core.{k}" for p in ("gate", "up") for k in ("pre_w", "connection_indices")]
    + ["down_a.weight", "down_b.weight"]
)
_KEYSETS = {
    "sparse": _SPARSE_KEYS,
    "dense": _DENSE_KEYS,
    "rank_readout": _RANK_READOUT_KEYS,
}


class SpanCellFormatError(RuntimeError):
    """A span checkpoint carries cell keys this consumer does not recognize.

    Raised instead of falling back to a default cell type: an unrecognized
    checkpoint must abort the run, never be silently scored as some other
    model.
    """


def classify_span_cell(state_dict, where: str = "") -> str:
    """Return 'sparse' | 'dense' | 'rank_readout' for one layer state dict.

    EXACT key-set match, no fallback branch. Extra keys, missing keys, or an
    unknown layout raise SpanCellFormatError -- the hard abort that keeps a
    consumer from scoring a model it did not actually rebuild.
    """
    keys = frozenset(state_dict.keys())
    for name, expected in _KEYSETS.items():
        if keys == expected:
            return name
    site = f" ({where})" if where else ""
    # Report the nearest known layout so the failure is diagnosable.
    best, best_extra, best_missing = None, None, None
    for name, expected in _KEYSETS.items():
        extra, missing = sorted(keys - expected), sorted(expected - keys)
        if best is None or len(extra) + len(missing) < len(best_extra) + len(
            best_missing
        ):
            best, best_extra, best_missing = name, extra, missing
    raise SpanCellFormatError(
        f"unrecognized span cell checkpoint{site}: keys {sorted(keys)} match "
        f"none of {list(_KEYSETS)} exactly. Closest is {best!r} "
        f"(unexpected keys {best_extra}, missing keys {best_missing}). "
        "Refusing to install a cell this consumer does not understand -- "
        "scoring it would report numbers for the wrong model."
    )


def rank_readout_rank(d_model: int, inter: int, k_in: int, k_down: int) -> int:
    """Budget-matched bottleneck rank (see module docstring, BUDGET-MATCHING).

    r = max(8, floor(d_model * k_down / (inter + d_model))): the largest rank
    whose two dense factors (inter x r and d_model x r) fit inside the weight
    count the sparse cell would have spent on its down projection. Verbatim
    the morphology_matrix.py / reference_grid.py rule, restated in terms of
    `remaining = budget - 2 * inter * k_in = d_model * k_down`.
    """
    remaining = d_model * k_down
    r = max(RANK_FLOOR, int(remaining // (inter + d_model)))
    r_max = min(inter, d_model)
    assert r <= r_max, (
        f"rank {r} exceeds min(inter, d_model)={r_max}: the SVD init has no "
        f"such factors (d_model={d_model} inter={inter} k_down={k_down})"
    )
    return r


def sparse_cell_param_count(d_model: int, inter: int, k_in: int, k_down: int) -> int:
    """Trainable-weight budget of the sparse cell at a given density."""
    return 2 * inter * k_in + d_model * k_down


def rank_readout_param_count(d_model: int, inter: int, k_in: int, r: int) -> int:
    """Trainable-weight count of the rank_readout cell (gate/up + A/B)."""
    return 2 * inter * k_in + r * (inter + d_model)


class SparseLinearFP32(nn.Module):
    """IndexedSparseLinear wrapper, key- and forward-identical to the
    SparseLinearFP32 declared in olmo_span_replace.py / span_recover.py
    (module tree `.core`, state-dict keys core.pre_w + core.connection_indices,
    fp32 compute, flatten -> core -> unflatten forward).

    Shape-only constructor; `init_from_teacher` applies the activation-weighted
    TopK teacher init, which a strict state-dict load makes unnecessary.
    """

    def __init__(self, in_features, out_features, k, seed, workspace_mb=None):
        super().__init__()
        kwargs = {}
        if workspace_mb is not None:
            kwargs["workspace_mb"] = workspace_mb
        self.core = IndexedSparseLinear(
            in_features=in_features,
            out_features=out_features,
            K=k,
            seed=seed,
            weight_transform="identity",
            projection_backend="auto",
            output_chunk_size=128,
            **kwargs,
        )

    @torch.no_grad()
    def init_from_teacher(self, weight, scales):
        """Activation-weighted TopK init (olmo_span_replace.SparseLinearFP32,
        copied verbatim): score |W| by input RMS, keep the top-K columns per
        output row in index order, copy the teacher weights at those columns."""
        w = weight.detach().float()
        k = self.core.pre_w.shape[1]
        score = w.abs() * scales.to(w.device).clamp_min(1e-12).unsqueeze(0)
        idx = torch.topk(score, k=k, dim=1, sorted=False).indices.sort(dim=1).values
        self.core.connection_indices.copy_(
            idx.int().to(self.core.connection_indices.device)
        )
        self.core.pre_w.copy_(w.gather(1, idx).to(self.core.pre_w.dtype))
        return self

    def forward(self, x):
        shape = x.shape
        return self.core(x.reshape(-1, shape[-1]).float()).reshape(*shape[:-1], -1)


class RankReadoutGLU(nn.Module):
    """The rank_readout cell: sparse gate/up front + dense low-rank readout.

        y = down_b(down_a(silu(gate(x)) * up(x)))

    gate/up are the standard aw-TopK sparse cells at K_in; the down projection
    is a rank-r factorization (down_a: inter -> r, down_b: r -> d_model)
    initialized from a balanced SVD of the teacher down_proj. The rank is
    budget-derived, never chosen -- see the module docstring. fp32 params;
    output cast back to the input dtype, exactly as SparseGLU/DenseNarrowGLU.
    """

    def __init__(self, d_model, inter, k_gate, k_up, r, workspace_mb=None):
        super().__init__()
        self.gate = SparseLinearFP32(d_model, inter, k_gate, GATE_SEED, workspace_mb)
        self.up = SparseLinearFP32(d_model, inter, k_up, UP_SEED, workspace_mb)
        self.down_a = nn.Linear(inter, r, bias=False)
        self.down_b = nn.Linear(r, d_model, bias=False)

    # -- constructors ------------------------------------------------------
    @classmethod
    def from_teacher(
        cls, mlp, density, scales_in, device=None, workspace_mb=None, verbose=True
    ):
        """Build + initialize from a teacher SwiGLU MLP at a given density.

        `scales_in` is the per-input-channel RMS on the calibration batch (the
        same tensor SparseGLU uses for gate/up); the readout needs no
        activation statistics because it is initialized from the weight SVD.
        """
        d_model = int(mlp.gate_proj.weight.shape[1])
        inter = int(mlp.gate_proj.weight.shape[0])
        k_in = int(density * d_model)
        k_down = int(density * inter)
        assert k_in > 0 and k_down > 0, (
            f"density {density:g} rounds to K_in={k_in} K_down={k_down} at "
            f"d_model={d_model} inter={inter}: nothing to build"
        )
        r = rank_readout_rank(d_model, inter, k_in, k_down)
        self = cls(d_model, inter, k_in, k_in, r, workspace_mb=workspace_mb)
        self.gate.init_from_teacher(mlp.gate_proj.weight, scales_in)
        self.up.init_from_teacher(mlp.up_proj.weight, scales_in)
        with torch.no_grad():
            # Balanced SVD split (morphology_matrix.py #6, verbatim): each
            # factor carries sqrt(S), so down_b @ down_a is the rank-r
            # truncated SVD of W_down at init and both factors share a scale.
            u, s, vh = torch.linalg.svd(
                mlp.down_proj.weight.detach().float(), full_matrices=False
            )
            root = s[:r].sqrt()
            self.down_a.weight.copy_((torch.diag(root) @ vh[:r]).to(torch.float32))
            self.down_b.weight.copy_((u[:, :r] @ torch.diag(root)).to(torch.float32))
            del u, s, vh
        if device is not None:
            self.to(device)
        self.float()
        # Realized-vs-budget accounting, printed at install so a matched
        # comparison can be verified from the run log alone.
        realized = sum(int(p.numel()) for p in self.parameters())
        budget = sparse_cell_param_count(d_model, inter, k_in, k_down)
        quantum = inter + d_model
        assert realized == rank_readout_param_count(d_model, inter, k_in, r)
        floor_binds = (
            r == RANK_FLOOR and (d_model * k_down // (inter + d_model)) < RANK_FLOOR
        )
        if floor_binds:
            # The registered rule floors the rank at 8, so at densities low
            # enough for the budget to buy less than that the cell is OVER the
            # sparse budget and the comparison is NOT matched. Never silent:
            # this cannot be inferred from the ppl number alone. (It cannot
            # occur at OLMo-3-7B for any density >= 1/64 — r there is 46.)
            print(
                f"WARNING rank_readout cell: density={density:g} buys rank "
                f"{d_model * k_down // (inter + d_model)}, below the "
                f"registered floor of {RANK_FLOOR}; the floored cell is "
                f"{realized} params vs sparse budget {budget} "
                f"(+{100.0 * (realized - budget) / budget:.3f}% OVER) — "
                "this density is NOT budget-matched",
                flush=True,
            )
        else:
            assert 0 <= budget - realized < quantum, (
                f"rank_readout budget mismatch: realized {realized} vs sparse "
                f"budget {budget} (allowed deficit < one rank quantum "
                f"{quantum})"
            )
            if verbose:
                print(
                    f"rank_readout cell: density={density:g} K_in={k_in} "
                    f"K_down={k_down} rank={r} sparse_params={budget} "
                    f"rank_params={realized} "
                    f"(deficit {budget - realized} = "
                    f"{100.0 * (budget - realized) / budget:.3f}%, "
                    f"< one rank quantum {quantum})",
                    flush=True,
                )
        return self

    @classmethod
    def from_state_dict(cls, sd, device=None, workspace_mb=None):
        """Checkpoint-shaped skeleton: every shape is read off the state dict
        and the strict load supplies all weights, so no teacher, no density
        rounding, and no SVD are involved (span_recover.DenseNarrowGLUFromCkpt
        idiom)."""
        cell = classify_span_cell(sd, "RankReadoutGLU.from_state_dict")
        if cell != "rank_readout":
            raise SpanCellFormatError(
                f"RankReadoutGLU.from_state_dict was handed a {cell!r} "
                "checkpoint; installing it here would score the wrong cell"
            )
        inter = int(sd["gate.core.pre_w"].shape[0])
        d_model = int(sd["down_b.weight"].shape[0])
        r = int(sd["down_b.weight"].shape[1])
        assert tuple(sd["down_a.weight"].shape) == (r, inter), (
            f"down_a {tuple(sd['down_a.weight'].shape)} incompatible with "
            f"down_b {tuple(sd['down_b.weight'].shape)} / inter {inter}"
        )
        self = cls(
            d_model,
            inter,
            int(sd["gate.core.pre_w"].shape[1]),
            int(sd["up.core.pre_w"].shape[1]),
            r,
            workspace_mb=workspace_mb,
        )
        self.load_state_dict(sd)  # strict: keys are exactly the persistent set
        if device is not None:
            self.to(device)
        return self.float()

    def forward(self, x):
        mid = torch.nn.functional.silu(self.gate(x)) * self.up(x)
        return self.down_b(self.down_a(mid)).to(x.dtype)
