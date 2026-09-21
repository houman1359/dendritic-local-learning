"""High-level same-checkpoint causal intervention analysis."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from dendritic_modeling.analysis.utils.causal_interventions import (
    FixedMeanCalibration,
    PathwayInterventionTarget,
    PopulationClassificationModel,
    evaluate_same_checkpoint_intervention,
)


class SameCheckpointCausalInterventionAnalyzer:
    """Evaluate a phase/pathway/depth intervention without changing weights."""

    def __init__(self, *, shuffle_seed: int = 0) -> None:
        self.shuffle_seed = int(shuffle_seed)

    def analyze(
        self,
        model: PopulationClassificationModel,
        *,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        targets: Sequence[PathwayInterventionTarget],
        phase_mask: torch.Tensor,
        mode: str,
        seq_lengths: torch.Tensor | None = None,
        fixed_mean_calibration: FixedMeanCalibration | None = None,
    ) -> dict[str, object]:
        result = evaluate_same_checkpoint_intervention(
            model,
            inputs,
            labels,
            targets=targets,
            phase_mask=phase_mask,
            mode=mode,
            seq_lengths=seq_lengths,
            shuffle_seed=self.shuffle_seed,
            fixed_mean_calibration=fixed_mean_calibration,
        )
        return result.to_dict()


__all__ = ["SameCheckpointCausalInterventionAnalyzer"]
