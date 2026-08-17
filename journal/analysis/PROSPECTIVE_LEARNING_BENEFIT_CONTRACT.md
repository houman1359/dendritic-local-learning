# Prospective learning-benefit experiment contract

Date frozen: 2026-08-02

## Question

When does a branching dendritic tree help local credit assignment, and which
part of any benefit is attributable to feedback routing, conductance-based
shunting, or forward computation?

## Primary factorial

The primary cohort crosses:

- task: MNIST and the frozen noise-resilience task (`sigma_task = 1.5`);
- core: conductance-based shunting and raw additive;
- within-neuron depth: `[2]`, `[2,2]`, `[2,2,2]`, and `[2,2,2,2]`;
- local feedback: scalar fallback, ancestry-shared soma coordinates, and exact
  path transport;
- seed: 42--51 in the confirmatory cohort.

All conditions use one feedforward population of 128 modeled neurons, a
linear decoder, the theorem-derived three-factor rule, the same optimizer
group rates, 40 excitatory and 20 inhibitory inputs per distal branch, and
architecture-specific reactivation policies fixed before training
(`analytical` for shunting and `occupancy_quantile` for additive). The
single-population design prevents equal neuron indices from separate
feedforward populations from sharing a feedback coordinate.

Matched end-to-end backpropagation controls use the same two cores, tasks,
depths, seeds, initialization policies, and training horizon. They are run once
per condition rather than duplicated across irrelevant feedback labels.

## Pre-specified contrasts

1. **Feedback value:** ancestry-shared minus scalar accuracy and exact
   transport minus ancestry-shared accuracy within core, depth, task, and seed.
2. **Depth sensitivity:** the paired slope of performance with depth and the
   depth-4 minus depth-1 change within each feedback/core/task condition.
3. **Shunting interaction:** the paired shunting-minus-additive contrast at
   each feedback level, plus the difference of that contrast between scalar
   and exact transport. A shunting advantage that disappears with exact
   transport supports a feedback-conditioning account; one that remains under
   exact transport supports a forward-computation or optimization account.
4. **Local-learning gap:** local three-factor minus matched backpropagation
   accuracy within core, depth, task, and seed.
5. **Noise interaction:** the change in each contrast from MNIST to the frozen
   noise task. This tests robustness, not generic image corruption.

## Outcomes and diagnostics

- Primary learning outcome: held-out test accuracy from the validation-selected
  checkpoint, paired by seed.
- Secondary outcomes: best validation loss, best epoch, convergence history,
  wall-clock duration, peak allocated GPU memory, and parameter count.
- Fixed-checkpoint mechanism diagnostics: stage-resolved exact-error capture,
  eligibility-weighted gradient capture, full-gradient cosine, norm-matched
  one-step loss change, path-gain log dispersion, `G_I/G_tot`, input
  resistance, and learned inter-compartment coupling.
- Dendritic and somatic stages are reported separately. Metrics are never
  pooled across an exact-by-construction soma stage for cross-core claims.

## Decision rules

- Artificial-network seeds are the inferential unit. Conditions are paired by
  seed; all ten seeds are retained irrespective of outcome.
- Effect sizes and paired bootstrap 95% intervals are primary. Exact Wilcoxon
  signed-rank tests are reported for the pre-specified paired contrasts. No
  t-test is used below ten pairs.
- Test data are not used to tune feedback, depth, learning rate, stopping, or
  model selection.
- The one-seed canary is a software and stability check only and will never be
  reported as scientific evidence.
- Archived results are not mixed with this cohort. Any failed condition is
  rerun from the frozen config and recorded, not silently replaced.

## Interpretation limits

The depth sweep increases the number of internal compartments and distal
branches. It tests scaling through successively deeper trees, not a
parameter-matched benefit of depth. A separate fixed-leaf and fixed-parameter
topology cohort is required before claiming morphology is beneficial at equal
capacity. Cross-core shunting/additive contrasts compare complete neuron
models; causal statements about the backward role of inhibition require the
fixed-forward counterfactual diagnostics described above.

## Execution gate

The confirmatory arrays may be submitted only after all canary conditions:

1. build and train without NaNs or transport-shape fallbacks;
2. emit a final checkpoint, performance result, resource report, and complete
   configuration;
3. show non-zero parameter updates for every trainable dendritic stage; and
4. pass the additive and conductance exact-transport regression tests.
