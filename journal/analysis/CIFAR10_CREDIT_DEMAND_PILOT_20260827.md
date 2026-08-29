# CIFAR-10 feedback-resolution pilot

Defined 27 August 2026 before execution of seeds 10600--10604.

## Question

MNIST and Fashion-MNIST show that selecting the correct neuron accounts for
most of the scalar-to-exact feedback gap.  An archived five-seed CIFAR-10
ladder showed a larger path-transport advantage, but its comparison condition
used the legacy matched-width/scalar-fallback implementation.  It therefore
cannot determine whether CIFAR-10 requires more information within a neuron or
merely benefits from receiving the correct neuron coordinate.

This exploratory pilot supplies the missing clean ladder on CIFAR-10:

1. strict scalar feedback;
2. one neuron-specific coordinate repeated over that neuron's descendants;
3. exact path-resolved transport;
4. matched backpropagation.

## Frozen pilot design

- Dataset: unaugmented, flattened CIFAR-10, using the established compact
  journal/NeurIPS configuration and official held-out test split.
- Architectures: matched shunting and additive `[3,3,3,3]` trees with 20
  somatic units and the same input contacts, decoder and initialization.
- Five paired exploratory seeds: 10600--10604.
- All four feedback conditions use the same 400-epoch limit, early stopping,
  optimizer groups, learning rates, weight decay, decoder update, batches and
  checkpoint policy.  Only the supplied feedback field or exact BP changes.
- W&B and all external experiment tracking are disabled.
- Configurations, checkpoints, logs and results are written only to the
  `kempner_project_b` filesystem.  The analysis cohort executes from a clean
  detached source checkout on `kempner_h100_priority`, using H100 GPUs for all
  conditions and seeds.

## Execution correction frozen before outcome inspection

An initial infrastructure cohort used `kempner_requeue`.  Eight artifact-only
canaries and eight additional additive scalar/neuron-specific tasks completed
there, on Blackwell GPUs.  Moving only the remaining tasks to H100 priority
would have confounded accelerator class with feedback condition.  That partial
migration was cancelled after 79 seconds, before any outcome was inspected.
The requeue outputs are retained for execution auditing but are excluded from
all scientific summaries.  The analysis cohort is therefore a new complete
40-run execution on H100 priority, with the design, seeds and decision gate
otherwise unchanged.

## Endpoints and decision gate

The independent training seed is the paired unit.  The primary contrasts are
neuron-specific minus strict scalar and exact path minus neuron-specific test
accuracy.  Exact path minus matched BP is an implementation/control contrast.

The pilot is eligible for a fresh ten-seed confirmation if, in either
architecture, exact path exceeds neuron-specific feedback by at least one
percentage point on average and in at least four of five paired seeds.  The
complete ladder is retained regardless of direction.  No pilot result enters
the manuscript as confirmatory evidence.

After training, fixed-checkpoint within-neuron diagnostics will measure
eligibility-weighted neuron-shared capture, gain-aware capture, nested-route
`K90`, the rank-matched ceiling and path-gain dispersion.  This separates task
difficulty from actual within-neuron credit demand.

## Interpretation boundary

A larger exact-path increment on CIFAR-10 would show that the importance of
within-neuron gain/address resolution is task dependent.  It would not show
that image difficulty alone causes high-rank credit, that shunting is uniquely
advantageous, or that dendritic learning outperforms backpropagation.
