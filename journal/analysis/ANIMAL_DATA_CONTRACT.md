# Animal-learning data experiment contract

## Primary biological question

Does branch-routed local credit predict future synaptic or dendritic change
during learning better than synapse-only activity, generic Hebbian
coincidence, or a single neuron-wide teaching signal?

## Dataset priority

### Primary: longitudinal single-spine motor-learning data

Wright, Hedrick and Komiyama (2025) measured spine-level glutamatergic
activity, dendritic calcium, neuronal output, morphology, and subsequent spine
plasticity in apical and basal dendrites across learning. Raw data are
currently restricted at Zenodo record `14549129`; no numerical endpoint will
be selected before access.

### Supporting: vectorized dendritic instructive signals

Francioni et al. (2026) measured soma/dendrite activity, task-defined signed
neuron identity, error, reward, and learning across mice. Source data will be
used to estimate the dimensionality, reliability, and sign structure of
neuron-indexed teaching coordinates. It does not resolve individual branches
and therefore cannot by itself establish within-neuron routing.

### Supporting: branch-specific multi-task learning

Cichon and Gan (2015) reported task-specific apical branch calcium spikes,
plasticity of coactive spines, loss of branch segregation after SST
perturbation, and interference between learned motor tasks. Published
animal-level summaries will constrain a quantitative multi-task routing model.
Raw data will be used if publicly accessible or obtained from the authors.

## Pre-specified spine-plasticity models

For spine (j) on branch (b) of neuron (n), the response is subsequent
continuous spine-size change and, secondarily, categorical potentiation,
depression, or stability.

1. **Activity only:** the spine's own presynaptic activity.
2. **Local coactivity:** own activity and coactivity with nearby spines,
   including the published distance dependence.
3. **Neuron-wide coincidence:** own activity multiplied by somatic output or
   a neuron-wide teaching coordinate.
4. **Branch-routed rule:** synapse-local eligibility multiplied by a
   branch-indexed coordinate estimated without using the outcome spine.
5. **Compartment mixture:** branch-routed and soma-routed coordinates combined
   with an apical/basal interaction fixed from the theory.

The branch-routed coordinate may use only signals recorded before the measured
plasticity outcome. Oracle gradients are allowed only as an explicitly labeled
upper bound.

## Controls

1. branch-label permutation within neuron;
2. distance-preserving spine permutation;
3. activity- and baseline-size-matched permutation;
4. temporal permutation across trials or sessions;
5. soma-only and dendrite-only ablations;
6. apical/basal stratification;
7. held-out dendrites and held-out mice; and
8. analyses with and without spines used in the original publication's model
   selection.

## Primary endpoint and inferential unit

The primary endpoint is out-of-sample predictive log likelihood or squared
error for subsequent continuous spine change. The primary contrast is the
branch-routed rule minus the strongest non-routing baseline. Mouse is the
biological replication unit; dendrites and spines are nested observations.
Model selection occurs within training mice, and the final estimate is based
on leave-one-mouse-out or grouped held-out-mouse evaluation.

## Success criterion

The analysis is considered biologically supportive only if the branch-routed
model improves the primary held-out endpoint, the improvement is positive in
a clear majority of mice, and it survives branch, distance, activity, and
temporal controls. A compartment interaction must be reported even if it does
not support the predicted apical/basal distinction.

## Multi-task interference model

The model will represent each task by a gradient covariance over synapses.
Separate branches provide partially non-overlapping route subspaces. SST-like
inhibition controls which route is writable on each task. The primary
predictions are:

1. increasing task covariance overlap increases branch reuse;
2. removing branch-selective inhibition increases reuse further;
3. increased reuse predicts overwriting of previously potentiated synapses;
   and
4. interference in the old task is mediated by route overlap, not only by a
   change in total firing rate.

The model will be fit to one published outcome and evaluated against held-out
branch-overlap, spine-change, or behavioral outcomes where the reported data
permit.

## Reporting boundary

Reanalysis of published data is evidence that measurements are consistent
with, or predicted by, the theory. It is not a new causal manipulation. The
paper will distinguish reproduction of published findings from new tests
defined here.
