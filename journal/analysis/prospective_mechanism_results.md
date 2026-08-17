# Prospective checkpoint-mechanism audit

> **Superseded publication analysis.** The complete historical diagnostic is
> retained for provenance. Publication inference uses the 120 input-valid
> checkpoints and 2,400 rows in
> `source_data/prospective_input_validity/mechanism_*_valid.csv`.

Collected 3200 rows from 160 backpropagation checkpoints. All expected conditions contain ten paired seeds.

## Geometry-to-learning link

At relative step $10^{-5}$, gradient cosine correlated with retained norm-matched one-step progress across scalar and neuron-indexed fields at Spearman rho=0.865 (checkpoint-clustered 95% bootstrap interval 0.823 to 0.899).
Eligibility-weighted gradient capture gave rho=0.841 (0.805 to 0.875).
Scalar feedback was a descent direction in 99/160 checkpoints, compared with 160/160 for neuron-indexed feedback.

## Ancestry value

- mnist, dendritic_additive, depth 1: neuron-indexed feedback retained 0.744 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- mnist, dendritic_additive, depth 2: neuron-indexed feedback retained 0.605 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- mnist, dendritic_additive, depth 3: neuron-indexed feedback retained 0.527 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- mnist, dendritic_additive, depth 4: neuron-indexed feedback retained 0.526 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- mnist, dendritic_shunting, depth 1: neuron-indexed feedback retained 0.819 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- mnist, dendritic_shunting, depth 2: neuron-indexed feedback retained 0.569 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- mnist, dendritic_shunting, depth 3: neuron-indexed feedback retained 0.514 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- mnist, dendritic_shunting, depth 4: neuron-indexed feedback retained 0.378 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- noise_resilience, dendritic_additive, depth 1: neuron-indexed feedback retained 0.706 more of the exact norm-matched one-step progress than scalar feedback (9/10 paired seeds).
- noise_resilience, dendritic_additive, depth 2: neuron-indexed feedback retained 0.680 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- noise_resilience, dendritic_additive, depth 3: neuron-indexed feedback retained 0.689 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- noise_resilience, dendritic_additive, depth 4: neuron-indexed feedback retained 0.723 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- noise_resilience, dendritic_shunting, depth 1: neuron-indexed feedback retained 0.824 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- noise_resilience, dendritic_shunting, depth 2: neuron-indexed feedback retained 0.585 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).
- noise_resilience, dendritic_shunting, depth 3: neuron-indexed feedback retained 0.409 more of the exact norm-matched one-step progress than scalar feedback (9/10 paired seeds).
- noise_resilience, dendritic_shunting, depth 4: neuron-indexed feedback retained 0.541 more of the exact norm-matched one-step progress than scalar feedback (10/10 paired seeds).

These are fixed-checkpoint diagnostics on matched backpropagation representations. They test how much of an exact update each feedback family can express; they do not substitute for the prospective trained-learning comparisons.
