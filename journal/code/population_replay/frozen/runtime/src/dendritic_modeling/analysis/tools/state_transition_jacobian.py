"""High-level analysis entry point for the true recurrent-state Jacobian."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from dendritic_modeling.analysis.utils.state_transition import (
    StateTransitionLinearization,
    leading_state_transition_singular_value,
)
from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationNetwork,
)
from dendritic_modeling.networks.architectures.recurrent.population_state import (
    PopulationLayerState,
)


class StateTransitionJacobianAnalyzer:
    """Estimate the leading singular value of ``d q_t / d q_{t-1}``.

    Unlike :class:`JacobianSpectrumAnalyzer`, which summarizes individual
    sparse weight matrices, this analyzer differentiates the complete one-step
    state update at an observed state and input.  It uses JVP/VJP products and
    never materializes the dense Jacobian.
    """

    def __init__(
        self,
        *,
        max_iterations: int = 50,
        relative_tolerance: float = 1e-5,
        absolute_tolerance: float = 1e-7,
        seed: int = 0,
    ) -> None:
        self.max_iterations = int(max_iterations)
        self.relative_tolerance = float(relative_tolerance)
        self.absolute_tolerance = float(absolute_tolerance)
        self.seed = int(seed)

    def analyze(
        self,
        network: PopulationNetwork,
        *,
        state: Sequence[PopulationLayerState],
        input_t: torch.Tensor,
        initial_vector: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Return JSON-safe metadata and a matrix-free singular-value estimate."""

        linearization = StateTransitionLinearization(network, state, input_t)
        estimate = leading_state_transition_singular_value(
            linearization,
            max_iterations=self.max_iterations,
            relative_tolerance=self.relative_tolerance,
            absolute_tolerance=self.absolute_tolerance,
            seed=self.seed,
            initial_vector=initial_vector,
        )
        return {
            "analysis_type": "state_transition_jacobian",
            "is_state_transition_jacobian": True,
            "transition_definition": "q_t = F(q_{t-1}, x_t)",
            "differentiated_with_respect_to": "q_{t-1}",
            "operator_materialized": False,
            "method": "two_sided_power_iteration_with_jvp_vjp",
            "state_dimension": linearization.dimension,
            "batch_size": linearization.codec.batch_size,
            "input_shape": list(input_t.shape),
            "state_tensor_order": linearization.codec.metadata(),
            "leading_singular_value": estimate.singular_value,
            "power_iteration": estimate.metadata(),
        }


__all__ = ["StateTransitionJacobianAnalyzer"]
