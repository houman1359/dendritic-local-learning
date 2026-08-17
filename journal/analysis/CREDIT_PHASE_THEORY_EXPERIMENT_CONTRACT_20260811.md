# Credit-phase theory experiment contract

Frozen: 11 August 2026, before confirmatory outcomes.

## Question

Can one controlled family validate the predicted boundaries among retained
task signal, rejected gradient noise, task--tree alignment, hierarchy depth,
and branch-specific reliability?

## Fixed design

- Sixteen leaf coordinates and the orthonormal balanced-tree Haar basis.
- Fifty independent confirmatory seeds, distinct from two artifact-only canary
  seeds.
- Spectral screen: feedback rank `K = 1, 2, 4, 8, 16` crossed with alignment
  `rho = 0, .25, .5, .75, 1`.
- Stochastic quadratic training: task hierarchy and routed model depth each
  range from one to four while the sixteen-coordinate parameter budget remains
  fixed.
- Projection screen: independently controlled signal and gradient-noise
  retention.
- Reliability screen: eight branch blocks and five levels of branch-SNR
  heterogeneity.

## Prespecified controls

- exact full stochastic gradient;
- aligned tree-Haar projection;
- leaf-permuted rewired hierarchy;
- random rank-matched projection;
- dense PCA rank upper bound;
- best global backward gain;
- reliability-aligned, shuffled, and anti-aligned branch gains;
- an explicit multiplicative point gate receiving the identical gains.

## Primary predictions

1. Spectral ancestry advantage over random rank-matched routes grows with
   task--tree alignment, vanishes at full rank, and never exceeds the PCA
   upper bound at matched rank.
2. With hierarchical task signal and fine-scale gradient noise, aligned routed
   optimization is best near `model depth = task depth`; insufficient depth
   leaves signal bias and excess depth admits noise.
3. A projection improves the population-gradient estimate precisely when
   rejected noise exceeds discarded signal at the common step.
4. Reliability-aligned branch gains equal the best global gain when branch
   reliabilities are equal and improve on it as reliability heterogeneity
   grows. The explicit point gate must match exactly.

## Claim boundary

The experiment tests a fixed-state stochastic quadratic theory. It does not
instantiate a positive-conductance forward dendrite. Existing passive and
active cable experiments establish that physical shunts can produce selective
attenuation; joining the two results motivates, but does not substitute for, a
future trained positive-conductance reliability experiment.
