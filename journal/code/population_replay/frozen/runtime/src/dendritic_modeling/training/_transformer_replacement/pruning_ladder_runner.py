"""Execute a structural prune-and-recover ladder inside layerwise training.

This is the consumer of ``training.pruning_ladder`` — the "start large,
learn, remove unused contacts, recover" arc.  After the main fit restores its
validation-best state, each rung:

1. resolves exact per-path contact targets against the CURRENT modules and
   applies them atomically (``training/replacement_pruning.py``);
2. rebuilds the optimizer, because fixed-contact pruning replaces
   ``Parameter`` objects and stale Adam moments would silently train the
   wrong slots;
3. re-evaluates, then RESETS the best-checkpoint tracker to the post-prune
   state.  This is the rule the experiment plan states for developmental
   pruning: a validation-best checkpoint from BEFORE the rung must never be
   restored afterwards, or the exported model silently reverts to the
   unpruned topology while its manifest claims the rung's contact counts;
4. runs the rung's recovery steps with the same step machinery as the main
   fit, restoring the best state seen WITHIN the rung.

Resume across a ladder is not supported: rung resolution validates the
current contact counts and fails closed on a model that was already pruned.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch.nn as nn

from dendritic_modeling.training.replacement_common import ReplacementTrainingHistory
from dendritic_modeling.training.replacement_pruning import (
    apply_replacement_pruning_rung_,
    resolve_replacement_pruning_rung,
)


def layerwise_pruning_units(units: Sequence[Any]) -> dict[str, nn.Module]:
    """Name each unit's replacement the way ``per_unit`` selectors address it."""
    named: dict[str, nn.Module] = {}
    for unit in units:
        name = f"layer_{int(unit.layer_index)}"
        if name in named:
            raise ValueError(f"duplicate pruning unit name {name!r}")
        named[name] = unit.replacement
    return named


def run_layerwise_pruning_ladder_(
    rungs: Sequence[Mapping[str, Any]],
    units: Sequence[Any],
    history: ReplacementTrainingHistory,
    *,
    start_step: int,
    evaluate: Callable[[], float],
    run_steps: Callable[[int, int], None],
    rebuild_optimizer: Callable[[], None],
    save_best: Callable[[int, float], None],
    restore_best: Callable[[], None],
) -> list[dict[str, Any]]:
    """Run every enabled rung and return one report entry per rung."""

    report: list[dict[str, Any]] = []
    step_base = int(start_step)
    for rung_index, rung_config in enumerate(rungs):
        named_units = layerwise_pruning_units(units)
        rung = resolve_replacement_pruning_rung(
            named_units,
            rung_config,
            rung_index=rung_index,
        )
        prune_reports = apply_replacement_pruning_rung_(named_units, rung)
        rebuild_optimizer()

        post_prune_valid = float(evaluate())
        # Reset the best tracker to the pruned state: from here on, "best"
        # can only ever name a checkpoint at this rung's topology.
        history.best_valid_loss = post_prune_valid
        history.best_step = step_base
        history.record_valid_loss(step_base, post_prune_valid)
        save_best(step_base, post_prune_valid)

        recovery_steps = int(rung.recovery_steps)
        if recovery_steps > 0:
            run_steps(step_base + 1, step_base + recovery_steps)
        restore_best()
        report.append(
            {
                "rung": rung.as_dict(),
                "prune_reports": prune_reports,
                "post_prune_valid_loss": post_prune_valid,
                "recovered_best_valid_loss": float(history.best_valid_loss),
                "recovered_best_step": int(history.best_step),
                "first_step": step_base + 1,
                "last_step": step_base + recovery_steps,
            }
        )
        step_base += recovery_steps
    return report


__all__ = [
    "layerwise_pruning_units",
    "run_layerwise_pruning_ladder_",
]
