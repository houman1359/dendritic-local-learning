"""
MuonH: Muon with a hyperball constraint, plus an AdamW fallback group.
=====================================================================

Provenance
----------
Puro-2B (arXiv 2608.27370) reports a 1.19x cost-efficiency gain from
"MuonH": Muon (momentum + Newton-Schulz orthogonalization of the update
for 2D parameter matrices; Jordan et al., the standard quintic-iteration
Muon) combined with a *hyperball* constraint — after each optimizer step,
every Muon-managed matrix is projected back onto the ball whose radius is
that matrix's INITIAL Frobenius norm.  The hyperball plays the role of
weight decay (it bounds parameter-norm growth without shrinking matrices
that stay inside the ball), so no decoupled weight decay is applied to
the Muon group.  Non-matrix parameters (vectors, gates, scalars — and
embedding-like parameters in setups where they are trainable) stay on
AdamW, per Puro's split.  Both groups share the same base learning rate
(and therefore any external schedule applied to ``group["lr"]``); the
Muon group's effective LR is ``base_lr * muon_lr_multiplier`` (Puro used
a 10x Muon-vs-Adam ratio).

Application to the dendritic-replacement trainer
------------------------------------------------
In the joint-recovery path (``loops._train_joint_hf_text_source``) the
trainable set is the flat list of replacement-cell parameters:

- ``IndexedSparseLinear.pre_w`` — ``[out_features, K]`` matrices (the
  bulk of the parameters) -> Muon-managed.
- ``BlockLinear.log_weight`` — ``[out_features, block_size]`` matrices
  -> Muon-managed.
- biases, gains, gates, per-neuron scalars — 1D/0D -> AdamW fallback.
- degenerate ``[N, 1]`` / ``[1, N]`` matrices (single-contact synapse
  columns behave like gate vectors) -> AdamW fallback via ``min_dim``.

DDP compatibility (must be verified on the real run)
----------------------------------------------------
Muon's orthogonalization is a *nonlinear* function of the gradient, so it
is only rank-consistent because DDP all-reduces (averages) gradients
during ``loss.backward()`` — i.e. BEFORE ``optimizer.step()`` runs.
Every rank therefore orthogonalizes the identical averaged gradient with
identical deterministic math and produces the identical update; no
post-step parameter broadcast is needed.  This ordering is asserted at
runtime: for the first ``rank_consistency_check_steps`` steps in an
initialized process group, the optimizer all-reduces min/max digests of
the freshly updated Muon parameters and raises if any rank diverged.

Notes / caveats
---------------
- Sparse-topology rewiring (``apply_sparse_topology_updates_after_step``)
  zeroes optimizer-state tensors whose shape matches ``pre_w``; MuonH's
  ``momentum_buffer`` matches and is correctly cleared, while the scalar
  ``init_frobenius_norm`` (a Python float) is untouched by design.
- ``pre_w``/``log_weight`` live in log parameter space for exp-transform
  cells; Muon on log-space matrices is an empirical bet documented in the
  run header, not a proven default.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.optim import Optimizer

from dendritic_modeling.config.training import OptimizerConfig
from dendritic_modeling.training.optimizers.factory import (
    OPTIMIZER_REGISTRY,
    register_optimizer,
)

# Tuned quintic Newton-Schulz coefficients from the reference Muon
# implementation.  They drive the singular values of the (spectrally
# normalized) input into an approximately-unit band (~[0.68, 1.13] for
# well-conditioned matrices after 5 steps) rather than converging to
# exactly 1; that band is what Muon uses in practice.
_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def newton_schulz_orthogonalize(
    matrix: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Approximately orthogonalize a 2D matrix with the Muon quintic NS iteration.

    Returns a matrix whose singular values are approximately 1 and whose
    singular directions match the input's (i.e. an approximation of the
    polar factor ``U @ V^T``).  Computation is performed in float32 unless
    the input is already float32/float64, in which case the input dtype is
    preserved (float64 supports the exactness tests).
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"newton_schulz_orthogonalize requires a 2D matrix, got shape "
            f"{tuple(matrix.shape)}"
        )
    if int(steps) < 1:
        raise ValueError("steps must be >= 1")
    a, b, c = _NS_COEFFICIENTS
    X = matrix
    if X.dtype not in (torch.float32, torch.float64):
        X = X.float()
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.mT
    X = X / (X.norm() + eps)
    for _ in range(int(steps)):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def _frobenius_norm(tensor: torch.Tensor) -> float:
    """Frobenius norm as a Python float, without downcasting fp64 inputs."""
    data = tensor.detach()
    if data.dtype not in (torch.float32, torch.float64):
        data = data.float()
    return float(data.norm())


def hyperball_project_(param: torch.Tensor, radius: float) -> torch.Tensor:
    """Project ``param`` in place onto the Frobenius ball of ``radius``.

    A parameter inside the ball is left bitwise-untouched; one outside is
    rescaled so its Frobenius norm equals ``radius`` exactly (to floating
    point rounding).
    """
    radius = float(radius)
    if radius < 0:
        raise ValueError("hyperball radius must be non-negative")
    norm = _frobenius_norm(param)
    if norm > radius and norm > 0.0:
        param.mul_(radius / norm)
    return param


def _is_muon_parameter(param: torch.Tensor, min_dim: int) -> bool:
    """2D matrices with both dims >= min_dim are Muon-managed; rest is AdamW."""
    return param.ndim == 2 and min(param.shape) >= int(min_dim)


class MuonH(Optimizer):
    """Muon + hyperball for 2D matrices, decoupled AdamW for everything else.

    Parameters are split at construction: 2D tensors with
    ``min(shape) >= min_dim`` join the Muon group, every other parameter
    (vectors, gates, scalars, degenerate matrices) joins the AdamW group.
    Both groups carry the same base ``lr`` so an external scheduler that
    rescales ``group["lr"]`` acts on both; the Muon group's effective step
    size is ``lr * muon_lr_multiplier``.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        *,
        muon_lr_multiplier: float = 10.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_eps: float = 1e-7,
        hyperball: bool = True,
        min_dim: int = 2,
        weight_decay: float = 0.0,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
        rank_consistency_check_steps: int = 1,
    ):
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if muon_lr_multiplier <= 0:
            raise ValueError(
                f"muon_lr_multiplier must be positive, got {muon_lr_multiplier}"
            )
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        if int(ns_steps) < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")
        betas = tuple(float(b) for b in adamw_betas)
        if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"Invalid adamw_betas: {adamw_betas}")
        if adamw_eps <= 0:
            raise ValueError(f"adamw_eps must be positive, got {adamw_eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

        param_list = [p for p in params if isinstance(p, torch.Tensor)]
        if not param_list:
            raise ValueError("MuonH received no parameters")
        muon_params = [p for p in param_list if _is_muon_parameter(p, min_dim)]
        adamw_params = [p for p in param_list if not _is_muon_parameter(p, min_dim)]

        defaults = {
            "lr": float(lr),
            "muon_lr_multiplier": float(muon_lr_multiplier),
            "momentum": float(momentum),
            "nesterov": bool(nesterov),
            "ns_steps": int(ns_steps),
            "ns_eps": float(ns_eps),
            "hyperball": bool(hyperball),
            "betas": betas,
            "eps": float(adamw_eps),
            "weight_decay": float(weight_decay),
        }
        groups = []
        if muon_params:
            groups.append({"params": muon_params, "use_muon": True})
        if adamw_params:
            groups.append({"params": adamw_params, "use_muon": False})
        super().__init__(groups, defaults)
        self.rank_consistency_check_steps = int(rank_consistency_check_steps)
        self._completed_steps = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)

        self._completed_steps += 1
        # DDP contract: gradients were already averaged across ranks during
        # backward(), so the nonlinear Muon update just computed is identical
        # on every rank.  Assert that on the first step(s) of a distributed
        # run; a failure means step() ran on unsynchronized gradients.
        if (
            self._completed_steps <= self.rank_consistency_check_steps
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            self._assert_rank_consistent_parameters()
        return loss

    def _step_muon_group(self, group: dict) -> None:
        # Hyperball replaces weight decay for the Muon group (Puro-2B):
        # decay is intentionally NOT applied here.
        effective_lr = float(group["lr"]) * float(group["muon_lr_multiplier"])
        momentum = float(group["momentum"])
        nesterov = bool(group["nesterov"])
        ns_steps = int(group["ns_steps"])
        ns_eps = float(group["ns_eps"])
        hyperball = bool(group["hyperball"])
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("MuonH does not support sparse gradients")
            state = self.state[param]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            if hyperball and "init_frobenius_norm" not in state:
                # Captured at first step = the warm-start / initial matrix.
                state["init_frobenius_norm"] = _frobenius_norm(param)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)
            direction = grad.add(buf, alpha=momentum) if nesterov else buf
            update = newton_schulz_orthogonalize(direction, ns_steps, ns_eps)
            # Reference Muon shape-scaling keeps per-element update RMS
            # comparable across aspect ratios.
            scale = max(1.0, param.shape[0] / param.shape[1]) ** 0.5
            param.add_(update.to(param.dtype), alpha=-effective_lr * scale)
            if hyperball:
                hyperball_project_(param, state["init_frobenius_norm"])

    def _step_adamw_group(self, group: dict) -> None:
        lr = float(group["lr"])
        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("MuonH does not support sparse gradients")
            state = self.state[param]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1
            step_count = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            if weight_decay != 0.0:
                param.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step_count
            bias_correction2 = 1.0 - beta2**step_count
            denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
            param.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

    def _assert_rank_consistent_parameters(self) -> None:
        """All-reduce min/max digests of Muon parameters; raise on divergence."""
        digests = []
        device = None
        for group in self.param_groups:
            if not group.get("use_muon", False):
                continue
            for param in group["params"]:
                if device is None:
                    device = param.device
                data = param.detach().double()
                digests.append(data.sum())
                digests.append(data.norm())
        if not digests:
            return
        stacked = torch.stack([d.to(device) for d in digests])
        lo = stacked.clone()
        hi = stacked.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if not torch.equal(lo, hi):
            max_gap = float((hi - lo).abs().max())
            raise RuntimeError(
                "MuonH rank-consistency check failed: Muon-managed parameters "
                f"diverged across ranks (max digest gap {max_gap:.3e}) after "
                f"step {self._completed_steps}. Muon's orthogonalization is "
                "nonlinear, so step() must run AFTER the DDP gradient "
                "all-reduce (i.e. after loss.backward() on a DDP-wrapped "
                "module) with identical initial parameters on every rank."
            )

    def muon_parameter_count(self) -> int:
        """Number of Muon-managed parameter tensors (test/introspection aid)."""
        return sum(
            len(group["params"])
            for group in self.param_groups
            if group.get("use_muon", False)
        )

    def adamw_parameter_count(self) -> int:
        """Number of AdamW-fallback parameter tensors (test/introspection aid)."""
        return sum(
            len(group["params"])
            for group in self.param_groups
            if not group.get("use_muon", False)
        )


def _build_muonh(model_parameters, config: OptimizerConfig) -> MuonH:
    """Factory builder: construct MuonH from an ``OptimizerConfig``.

    Only reached when ``optimizer.name`` is ``muonh``; the default AdamW
    path never constructs or configures this class.
    """
    return MuonH(
        model_parameters,
        lr=float(config.lr),
        muon_lr_multiplier=float(getattr(config, "muon_lr_multiplier", 10.0)),
        momentum=float(getattr(config, "muon_momentum", 0.95)),
        nesterov=bool(getattr(config, "muon_nesterov", True)),
        ns_steps=int(getattr(config, "muon_ns_steps", 5)),
        hyperball=bool(getattr(config, "muon_hyperball", True)),
        min_dim=int(getattr(config, "muon_min_dim", 2)),
        weight_decay=float(config.weight_decay),
        adamw_betas=tuple(config.betas),
        adamw_eps=float(config.eps),
        rank_consistency_check_steps=int(
            getattr(config, "muon_rank_consistency_check_steps", 1)
        ),
    )


if "muonh" not in OPTIMIZER_REGISTRY:
    register_optimizer("muonh", _build_muonh, aliases=("muon_hyperball",))
