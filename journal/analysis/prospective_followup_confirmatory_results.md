# Prospective shunting, topology, and fixed-budget results

This report is generated only after all 1,200 included frozen confirmatory runs pass the artifact audit. Seeds are paired and are the inferential unit. Canary runs are excluded.

## Matched-bandwidth ancestry routing

- task=mnist, core=dendritic_shunting, feedback=matched-bandwidth ancestry, depth=2: 0.57 percentage points (95% paired bootstrap CI 0.43 to 0.69; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.00293).
- task=mnist, core=dendritic_shunting, feedback=matched-bandwidth ancestry, depth=4: 0.38 percentage points (95% paired bootstrap CI 0.31 to 0.45; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.00293).
- task=mnist, core=dendritic_additive, feedback=matched-bandwidth ancestry, depth=2: 0.51 percentage points (95% paired bootstrap CI 0.37 to 0.62; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.00293).
- task=mnist, core=dendritic_additive, feedback=matched-bandwidth ancestry, depth=4: 0.39 percentage points (95% paired bootstrap CI 0.31 to 0.45; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.00293).
- task=noise_resilience, core=dendritic_shunting, feedback=matched-bandwidth ancestry, depth=2: 0.72 percentage points (95% paired bootstrap CI 0.62 to 0.82; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.00293).
- task=noise_resilience, core=dendritic_shunting, feedback=matched-bandwidth ancestry, depth=4: 0.59 percentage points (95% paired bootstrap CI 0.48 to 0.70; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.00293).
- task=noise_resilience, core=dendritic_additive, feedback=matched-bandwidth ancestry, depth=2: 0.70 percentage points (95% paired bootstrap CI 0.56 to 0.84; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.00293).
- task=noise_resilience, core=dendritic_additive, feedback=matched-bandwidth ancestry, depth=4: 0.15 percentage points (95% paired bootstrap CI 0.01 to 0.28; 8/10 positive pairs; exact Wilcoxon P=0.1055; within-study BH-adjusted P=0.1055).

## Task-aligned fixed topology

- task=mnist, core=dendritic_shunting, feedback=backprop: 0.57 percentage points (95% paired bootstrap CI 0.52 to 0.62; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=mnist, core=dendritic_shunting, feedback=per_soma: 1.24 percentage points (95% paired bootstrap CI 0.71 to 1.82; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=mnist, core=dendritic_shunting, feedback=per_soma_shared: 0.71 percentage points (95% paired bootstrap CI 0.61 to 0.79; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=mnist, core=dendritic_shunting, feedback=path_transport: 0.56 percentage points (95% paired bootstrap CI 0.52 to 0.62; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=mnist, core=dendritic_additive, feedback=backprop: 0.60 percentage points (95% paired bootstrap CI 0.48 to 0.72; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=mnist, core=dendritic_additive, feedback=per_soma: 1.60 percentage points (95% paired bootstrap CI 1.04 to 2.19; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=mnist, core=dendritic_additive, feedback=per_soma_shared: 0.36 percentage points (95% paired bootstrap CI 0.28 to 0.44; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=mnist, core=dendritic_additive, feedback=path_transport: 0.60 percentage points (95% paired bootstrap CI 0.48 to 0.72; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=noise_resilience, core=dendritic_shunting, feedback=backprop: 0.70 percentage points (95% paired bootstrap CI 0.56 to 0.83; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=noise_resilience, core=dendritic_shunting, feedback=per_soma: 3.04 percentage points (95% paired bootstrap CI 2.11 to 4.00; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=noise_resilience, core=dendritic_shunting, feedback=per_soma_shared: 1.15 percentage points (95% paired bootstrap CI 1.01 to 1.30; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=noise_resilience, core=dendritic_shunting, feedback=path_transport: 0.70 percentage points (95% paired bootstrap CI 0.56 to 0.82; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=noise_resilience, core=dendritic_additive, feedback=backprop: 1.00 percentage points (95% paired bootstrap CI 0.83 to 1.17; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=noise_resilience, core=dendritic_additive, feedback=per_soma: 2.57 percentage points (95% paired bootstrap CI 1.20 to 3.99; 9/10 positive pairs; exact Wilcoxon P=0.01367; within-study BH-adjusted P=0.01562).
- task=noise_resilience, core=dendritic_additive, feedback=per_soma_shared: 1.17 percentage points (95% paired bootstrap CI 0.95 to 1.35; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).
- task=noise_resilience, core=dendritic_additive, feedback=path_transport: 1.00 percentage points (95% paired bootstrap CI 0.83 to 1.16; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002757).

## Inhibitory dose

- task=noise_resilience, core=shunting - additive, feedback=backprop: -1.01 percentage points (95% paired bootstrap CI -1.22 to -0.74; 1/10 positive pairs; exact Wilcoxon P=0.003906; within-study BH-adjusted P=0.004297).
- task=noise_resilience, core=shunting - additive, feedback=per_soma: 22.11 percentage points (95% paired bootstrap CI 19.94 to 24.44; 10/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002387).
- task=noise_resilience, core=shunting - additive, feedback=per_soma_shared: -0.32 percentage points (95% paired bootstrap CI -0.57 to -0.08; 2/10 positive pairs; exact Wilcoxon P=0.04883; within-study BH-adjusted P=0.04883).
- task=noise_resilience, core=shunting - additive, feedback=path_transport: -1.10 percentage points (95% paired bootstrap CI -1.31 to -0.86; 0/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002387).

## Fixed-budget dendritic depth

- task=noise_resilience, core=dendritic_shunting, feedback=backprop: -0.55 percentage points (95% paired bootstrap CI -0.63 to -0.46; 0/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002772).
- task=noise_resilience, core=dendritic_shunting, feedback=per_soma: -7.57 percentage points (95% paired bootstrap CI -8.75 to -6.50; 0/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002772).
- task=noise_resilience, core=dendritic_shunting, feedback=per_soma_shared: -0.93 percentage points (95% paired bootstrap CI -1.09 to -0.79; 0/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002772).
- task=noise_resilience, core=dendritic_shunting, feedback=path_transport: -0.55 percentage points (95% paired bootstrap CI -0.63 to -0.46; 0/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002772).
- task=noise_resilience, core=dendritic_additive, feedback=backprop: -1.37 percentage points (95% paired bootstrap CI -1.56 to -1.18; 0/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002772).
- task=noise_resilience, core=dendritic_additive, feedback=per_soma: -7.92 percentage points (95% paired bootstrap CI -8.73 to -6.96; 0/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002772).
- task=noise_resilience, core=dendritic_additive, feedback=per_soma_shared: -1.06 percentage points (95% paired bootstrap CI -1.19 to -0.93; 0/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002772).
- task=noise_resilience, core=dendritic_additive, feedback=path_transport: -1.37 percentage points (95% paired bootstrap CI -1.56 to -1.18; 0/10 positive pairs; exact Wilcoxon P=0.001953; within-study BH-adjusted P=0.002772).

## Interpretation guardrails

A spatial-topology effect shared by backpropagation and local learning is a forward inductive-bias effect. A correct-versus-shuffled ancestry effect isolates feedback routing at matched bandwidth. An inhibition-dose effect changes the full operating point and is not, by itself, a backward-only causal intervention. The fixed-budget comparison holds distal leaves and active input contacts approximately constant, but deeper trees retain additional coupling and reactivation parameters.
