# Reconstructed-tree task learning

## Question

Does a sparse feedback basis taken from the same reconstructed dendritic tree support a local conductance update better than equal-rank site-shuffled or random-route feedback when the task is defined independently by measured visual responses?

## Design

The analysis used 7 MICrONS target cells across 13 eligible target--scan observations, with 10 held-out-stimulus splits per scan. Each complete compressed morphology was instantiated as a passive conductance network. Directly connected imaged partners supplied non-negative trial activity at mapped contact sites, and the measured postsynaptic response supplied the target. Route selection and one-time calibration used anatomy and fixed passive conductances only, never task responses. Restricted dictionaries had equal effective rank within each run.

Before learning, the baseline passive soma-to-site transfer vector was projected once onto each dictionary and frozen. Restricted updates then used scalar output error times this fixed vector and synapse-local eligibility; they did not use the exact dynamic adjoint. The one-time coefficients remain an anatomy-calibration oracle, and this is not evidence that the MICrONS neurons learned with this rule.

## Circularity audit

The positive structural-capacity fields elsewhere in the project are generated from the same route kernel being evaluated. This analysis does not reuse those fields. Its task gradients come from measured responses and an exact passive-tree inverse, while route selection never sees the task data. The exact gradients and route basis still share the reconstructed forward tree, as required by the topology-matching hypothesis, so this remains modeled evidence rather than an independent biological observation.

## Cell-level results

| method | normalized held-out MSE | exact-field capture | update cosine |
|---|---:|---:|---:|
| exact compartment error | 0.7877 | 1.0000 | 1.0000 |
| topology-matched routes | 0.8321 | 0.2728 | 0.4169 |
| site-shuffled routes | 0.8391 | 0.4504 | 0.5979 |
| random anatomical routes | 0.8333 | 0.3812 | 0.5261 |

Primary matched controls:

- topology-matched routes minus exact compartment error for heldout_normalized_mse: mean 0.0444, 95% cell-bootstrap CI [0.0126, 0.0800], positive/negative cells 5/2, two-sided Wilcoxon P=0.078125.
- topology-matched routes minus site-shuffled routes for heldout_normalized_mse: mean -0.0070, 95% cell-bootstrap CI [-0.0321, 0.0207], positive/negative cells 3/4, two-sided Wilcoxon P=0.8125.
- topology-matched routes minus random anatomical routes for heldout_normalized_mse: mean -0.0013, 95% cell-bootstrap CI [-0.0113, 0.0090], positive/negative cells 2/5, two-sided Wilcoxon P=0.578125.
- topology-matched routes minus site-shuffled routes for common_checkpoint_update_cosine: mean -0.1810, 95% cell-bootstrap CI [-0.3246, -0.0358], positive/negative cells 2/5, two-sided Wilcoxon P=0.109375.
- topology-matched routes minus random anatomical routes for common_checkpoint_update_cosine: mean -0.1092, 95% cell-bootstrap CI [-0.1902, -0.0041], positive/negative cells 1/6, two-sided Wilcoxon P=0.109375.

## Validation and interpretation

The largest relative finite-difference error for an exact conductance gradient was 3.050e-07. The route controls preserve channel count and effective rank; site shuffling additionally preserves every selected route column's values and nonzero count.

Topology-matched feedback has the lowest mean held-out error among the three restricted rules, but the cell-level intervals do not establish an advantage over either matched control. Its common-checkpoint gradient-direction diagnostic is also lower than both controls, showing that this descriptive metric and final learning need not order methods in the same way. The result places a boundary on anatomy-only routing in this seven-cell measured-response task.

## Reproduction

```bash
python scripts/run_reconstructed_tree_task_learning.py
```
