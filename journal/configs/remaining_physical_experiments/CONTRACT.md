# Remaining physical experiments contract

Frozen: 13 August 2026, before the new outcome files were opened.

## Questions

1. Does a literal grouped-point/parallel-readout emulation reproduce the H=3
   serial-tree advantage when local branch modules, masks, trainable scalar
   count and axial initialization are matched?
2. Does the positive physical-depth effect replicate on an independently
   generated two-level task hierarchy?
3. Does shared- and path-transport LocalCA retain the H=2 effect, and is it
   removed when the two sensor tiers are reversed?

## Frozen design

- H=3 grouped-point: aligned and reversed placement, D1--D3, BP, ten paired
  seeds 10200--10209. These seeds match the already observed serial reference.
- H=2 crossover: aligned and reversed placement; exact-resource D1 `[8]` and
  D2 `[4,1]`; serial BP, grouped-point BP, and serial LocalCA with shared-soma
  or exact-path transport; ten fresh paired seeds 10300--10309.
- H=2 uses two 32-feature sensor tiers and a fixed 4/4 branch inventory. All
  other task, optimizer, stopping and selected operating-point settings are
  inherited without retuning from the H=3 confirmatory experiment.
- The training seed is the inferential unit. All primary comparisons are paired
  within seed.

## Primary estimands

1. H=3 aligned serial minus grouped-point accuracy at D3 and its
   aligned-minus-reversed interaction.
2. H=2 aligned D2-minus-D1 accuracy under serial BP and its
   aligned-minus-reversed interaction.
3. H=2 aligned serial-minus-grouped-point accuracy at D2 and its
   aligned-minus-reversed interaction.
4. H=2 D2-minus-D1 accuracy for shared and path LocalCA, plus their respective
   aligned-minus-reversed interactions.

Intervals are paired-seed bootstrap intervals. Exact two-sided sign-flip tests
and positive-pair counts are reported. A directional positive claim requires
an interval excluding zero and at least 8/10 positive paired seeds. All nulls
and reversals are retained. No task or optimizer parameter may be changed after
outcome inspection.

## Validity gates

- Every expected fit reaches a finite, stage-complete checkpoint without a
  fallback warning.
- D1 serial and grouped-point logits agree numerically at initialization.
- Within each hierarchy, the compared serial morphologies have identical
  branch-unit, active-contact, candidate-slot, trainable-parameter and
  persistent-state budgets.
- Each grouped-point direct projection contains the same number and initial
  values of raw conductances as the serial child aggregator it replaces.
