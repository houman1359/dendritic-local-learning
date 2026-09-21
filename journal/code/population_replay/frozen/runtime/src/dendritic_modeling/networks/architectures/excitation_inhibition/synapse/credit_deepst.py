"""Credit-gated branch-local structural plasticity.

The layer couples two timescales without introducing a second teaching signal:
Local Credit Assignment (LocalCA) updates active conductances quickly, while an
exponential trace of the same conductance-space gradient selects slower,
fixed-budget synaptic replacements.  Inactive inputs are observed only through
a small branch-local candidate pool, which represents silent or nascent
contacts rather than a dense trainable shadow matrix.
"""

from __future__ import annotations

import math

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
    DeepstLinear,
)


class CreditGatedDeepstLinear(DeepstLinear):
    """DeepST layer whose rewiring is selected by local credit traces.

    For an active contact ``i`` and candidate contact ``j`` on branch ``b``,
    transferring the active conductance ``w_bi`` has first-order predicted
    improvement

    ``benefit(i -> j) = w_bi * (mean_grad_bi - mean_grad_bj)``.

    A positive value means the LocalCA gradient predicts that moving the
    conductance to ``j`` lowers the loss.  Each accepted swap preserves the
    number of active synapses on its branch exactly.
    """

    def __init__(
        self,
        *args,
        credit_trace_decay: float = 0.95,
        credit_candidate_pool_size: int = 8,
        credit_turnover_fraction: float = 0.1,
        credit_weak_active_pool_fraction: float = 0.5,
        credit_min_observations: int = 5,
        credit_swap_margin: float = 0.0,
        credit_force_turnover: bool = False,
        credit_selection: str = "credit",
        credit_warmup_steps: int = 5,
        **kwargs,
    ):
        if kwargs.get("rewiring_mode", "global") != "constant_branch":
            raise ValueError(
                "CreditGatedDeepstLinear requires rewiring_mode='constant_branch'"
            )
        if not 0.0 <= float(credit_trace_decay) < 1.0:
            raise ValueError("credit_trace_decay must satisfy 0 <= decay < 1")
        if int(credit_candidate_pool_size) < 1:
            raise ValueError("credit_candidate_pool_size must be >= 1")
        if not 0.0 < float(credit_turnover_fraction) <= 1.0:
            raise ValueError("credit_turnover_fraction must satisfy 0 < f <= 1")
        if not 0.0 < float(credit_weak_active_pool_fraction) <= 1.0:
            raise ValueError("credit_weak_active_pool_fraction must satisfy 0 < f <= 1")
        if int(credit_min_observations) < 1:
            raise ValueError("credit_min_observations must be >= 1")
        if int(credit_warmup_steps) < 0:
            raise ValueError("credit_warmup_steps must be >= 0")
        if credit_selection not in {"credit", "random", "shuffled_credit"}:
            raise ValueError(
                "credit_selection must be 'credit', 'random', or 'shuffled_credit'"
            )

        super().__init__(*args, **kwargs)
        self.credit_trace_decay = float(credit_trace_decay)
        self.credit_candidate_pool_size = int(credit_candidate_pool_size)
        self.credit_turnover_fraction = float(credit_turnover_fraction)
        self.credit_weak_active_pool_fraction = float(credit_weak_active_pool_fraction)
        self.credit_min_observations = int(credit_min_observations)
        self.credit_swap_margin = float(credit_swap_margin)
        self.credit_force_turnover = bool(credit_force_turnover)
        self.credit_selection = str(credit_selection)
        self.credit_warmup_steps = int(credit_warmup_steps)

        self.register_buffer(
            "credit_trace", torch.zeros_like(self.pre_w), persistent=True
        )
        self.register_buffer(
            "credit_observations",
            torch.zeros_like(self.mask, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "credit_candidate_mask", torch.zeros_like(self.mask), persistent=True
        )
        self.register_buffer(
            "credit_record_steps", torch.zeros((), dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "credit_total_swaps", torch.zeros((), dtype=torch.long), persistent=True
        )
        self._sample_candidate_pool()

    @torch.no_grad()
    def _sample_candidate_pool(self) -> None:
        """Sample a bounded set of inactive, allowed contacts on each branch."""
        retired = self.credit_candidate_mask & ~self.mask
        self.credit_trace[retired] = 0
        self.credit_observations[retired] = 0
        self.credit_candidate_mask.zero_()
        for branch_idx in range(self.out_features):
            available = (~self.mask[branch_idx]) & (
                ~self._forbidden_connection_mask[branch_idx]
            )
            indices = torch.nonzero(available, as_tuple=False).squeeze(1)
            if indices.numel() == 0:
                continue
            count = min(self.credit_candidate_pool_size, int(indices.numel()))
            chosen = indices[
                torch.randperm(indices.numel(), device=indices.device)[:count]
            ]
            self.credit_candidate_mask[branch_idx, chosen] = True

    @torch.no_grad()
    def record_local_credit(self, local_gradient: torch.Tensor) -> None:
        """Update local gradient traces at active and sampled nascent contacts."""
        if local_gradient.shape != self.pre_w.shape:
            raise ValueError(
                "local_gradient must have shape "
                f"{tuple(self.pre_w.shape)}, got {tuple(local_gradient.shape)}"
            )
        if not torch.isfinite(local_gradient).all():
            raise ValueError("local_gradient contains non-finite values")

        gradient = local_gradient.to(
            device=self.credit_trace.device, dtype=self.credit_trace.dtype
        )
        observed = self.mask | self.credit_candidate_mask
        initialized = observed & (self.credit_observations == 0)
        continuing = observed & ~initialized
        self.credit_trace[initialized] = gradient[initialized]
        decay = self.credit_trace_decay
        self.credit_trace[continuing] = (
            decay * self.credit_trace[continuing] + (1.0 - decay) * gradient[continuing]
        )
        self.credit_observations[observed] += 1
        self.credit_record_steps += 1

    def _branch_swap_pairs(self, branch_idx: int) -> list[tuple[int, int]]:
        active = torch.nonzero(self.mask[branch_idx], as_tuple=False).squeeze(1)
        candidates = torch.nonzero(
            self.credit_candidate_mask[branch_idx]
            & (self.credit_observations[branch_idx] >= self.credit_min_observations),
            as_tuple=False,
        ).squeeze(1)
        if active.numel() == 0 or candidates.numel() == 0:
            return []

        n_active = int(active.numel())
        n_swaps = min(
            max(1, math.ceil(n_active * self.credit_turnover_fraction)),
            n_active,
            int(candidates.numel()),
        )
        active_weight = self.weight()[branch_idx, active]
        weak_pool_size = min(
            n_active,
            max(
                n_swaps,
                math.ceil(n_active * self.credit_weak_active_pool_fraction),
            ),
        )
        weak_order = torch.argsort(active_weight, descending=False, stable=True)
        active = active[weak_order[:weak_pool_size]]
        active_weight = active_weight[weak_order[:weak_pool_size]]
        if self.credit_selection == "random":
            active = active[
                torch.randperm(active.numel(), device=active.device)[:n_swaps]
            ]
            candidates = candidates[
                torch.randperm(candidates.numel(), device=candidates.device)[:n_swaps]
            ]
            return list(zip(active.tolist(), candidates.tolist(), strict=True))

        active_trace = self.credit_trace[branch_idx, active]
        candidate_trace = self.credit_trace[branch_idx, candidates]
        if self.credit_selection == "shuffled_credit":
            candidate_trace = candidate_trace[
                torch.randperm(candidate_trace.numel(), device=candidate_trace.device)
            ]
        benefits = active_weight[:, None] * (
            active_trace[:, None] - candidate_trace[None, :]
        )

        pairs: list[tuple[int, int]] = []
        used_active: set[int] = set()
        used_candidate: set[int] = set()
        order = torch.argsort(benefits.reshape(-1), descending=True)
        n_candidates = int(candidates.numel())
        for flat_idx in order.tolist():
            active_pos = flat_idx // n_candidates
            candidate_pos = flat_idx % n_candidates
            if active_pos in used_active or candidate_pos in used_candidate:
                continue
            benefit = float(benefits[active_pos, candidate_pos])
            if not self.credit_force_turnover and benefit <= self.credit_swap_margin:
                break
            pairs.append(
                (int(active[active_pos].item()), int(candidates[candidate_pos].item()))
            )
            used_active.add(active_pos)
            used_candidate.add(candidate_pos)
            if len(pairs) >= n_swaps:
                break
        return pairs

    @torch.no_grad()
    def apply_noise_and_enforce_constraints(self) -> None:
        """Apply local-credit swaps while preserving the per-branch budget."""
        self._last_rewired_mask.zero_()
        if self.param_space == "log":
            self.pre_w.data.clamp_(min=-10.0, max=10.0)
        elif self.param_space == "presigmoid":
            self.pre_w.data.clamp_(min=-5.0, max=5.0)
        if self.freeze_connectivity:
            return
        if int(self.credit_record_steps.item()) < self.credit_warmup_steps:
            return

        swaps = 0
        for branch_idx in range(self.out_features):
            for old_idx, new_idx in self._branch_swap_pairs(branch_idx):
                transferred_parameter = self.pre_w.data[branch_idx, old_idx].clone()
                self.mask[branch_idx, old_idx] = False
                self.mask[branch_idx, new_idx] = True
                self.pre_w.data[branch_idx, new_idx] = transferred_parameter
                self._last_rewired_mask[branch_idx, old_idx] = True
                self._last_rewired_mask[branch_idx, new_idx] = True
                self.credit_trace[branch_idx, old_idx] = 0
                self.credit_observations[branch_idx, old_idx] = 0
                swaps += 1

        if swaps:
            self.credit_total_swaps += swaps
        # Rotate the bounded nascent-contact sample even when no swap passes
        # the margin, so a branch can continue exploring without a dense shadow
        # parameter matrix.
        self._sample_candidate_pool()


__all__ = ["CreditGatedDeepstLinear"]
