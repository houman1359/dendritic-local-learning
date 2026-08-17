# Same-span noisy coefficient-learning contract

Frozen: 11 August 2026, before outcomes for seeds 9098--9149.

## Question

Can route dictionaries with exactly the same address span learn at different
finite rates because their coordinate systems have different Gram spectra?

## Design

- Sixteen rate coordinates and the balanced-tree subspace through depth three
  (rank eight).
- Three exactly span-matched parameterizations: orthonormal tree-Haar
  coordinates, raw nested indicators, and statically rescaled nested
  indicators.
- Targets are unit-norm random combinations of the same eight Haar modes.
- Each local update observes the same noisy target estimate. Effective sample
  sizes are 4, 16, 64, and 256; observation noise scales as `1/sqrt(n)`.
- Eighty updates with checkpoints 1, 5, 20, and 80.
- Vanilla coefficient descent uses the same fraction of each dictionary's
  stability limit. A Gram-preconditioned upper control removes coordinate
  conditioning while leaving the route projector unchanged.
- Two artifact-only canary seeds precede 50 paired confirmatory seeds.

## Predictions

1. All three dictionaries have the same projector and zero target address
   residual to numerical precision.
2. Orthonormal Haar coordinates reduce finite-time estimation error relative
   to raw and strongly scaled nested coordinates under vanilla local descent.
3. Increasing effective sample size reduces the stochastic error floor but
   does not remove the deterministic conditioning penalty at finite time.
4. Gram preconditioning makes all three field trajectories identical; it is
   an oracle coordinate control, not a proposed local biological circuit.

## Claim boundary

The study separates address capacity from coefficient-estimation dynamics.
It can support a statement that static route gains and redundant nested
coordinates alter learnability without altering span. It cannot show that
tree-Haar coordinates are biologically present or that dendritic material is
necessary for the computation.
