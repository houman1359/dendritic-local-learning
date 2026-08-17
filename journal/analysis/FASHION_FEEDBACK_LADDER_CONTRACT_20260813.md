# Fashion-MNIST feedback-ladder contract

**Frozen before outcomes:** 13 August 2026

## Question

Does the ordering of scalar feedback, neuron-indexed feedback and exact path
transport replicate on a second image dataset under the same regular-tree
architecture and LocalCA training protocol used for the MNIST comparison?

## Design

- Dataset: Fashion-MNIST from the existing local torchvision cache.
- Architectures: dendritic shunting and width/connectivity-matched raw additive.
- Feedback: `per_soma` (scalar fallback), `per_soma_shared` (one coordinate per
  neuron), and `path_transport` (exact transported compartment error).
- Seeds: 10200--10209, paired across feedback conditions within architecture.
- Total: 60 fits (2 architectures x 3 feedback fields x 10 seeds).
- Training schedule, optimizer, decoder, sparsity and architecture are inherited
  unchanged from the validated 15-seed MNIST replacement cohort. Architecture-
  specific initialization policies remain analytical for shunting and
  occupancy-quantile for additive.

## Primary comparisons

Within each architecture:

1. neuron-indexed minus scalar-fallback held-out accuracy;
2. exact-path minus neuron-indexed held-out accuracy.

The neuronal-identity replication is supported when comparison 1 is positive
in at least 8/10 seeds and its paired bootstrap 95% interval excludes zero in
both architectures. Comparison 2 is reported as a bound on the remaining
within-tree transport gap, without requiring a positive outcome.

## Validity gates

All 60 resolved configurations, final metrics and final checkpoints must be
present; accuracies must be finite and in [0,1]; each architecture must contain
exactly the three feedback modes and the ten specified seeds; scientific
configuration signatures must agree within architecture after removing seed,
feedback mode and output path. No condition may be removed because of its
accuracy.

## Interpretation boundary

Fashion-MNIST is a second dataset but not an independent architecture or
biological test. A replicated identity effect strengthens generality beyond
MNIST; it does not show that dendritic topology itself is beneficial.
