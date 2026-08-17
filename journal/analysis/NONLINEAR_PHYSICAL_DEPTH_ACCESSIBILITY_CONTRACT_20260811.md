# Exploratory nonlinear physical-depth accessibility contract

Frozen: 11 August 2026, after observing the severe-shift aligned canary and
before observing any result in this ladder.

## Why this is a separate pilot

The original canary trained at gain standard deviation 0.25 and tested at
1.2.  All three exact-resource physical morphologies fit the training
distribution, increasingly so with depth, but remained at chance on the test
distribution under backpropagation and both LocalCA transport modes.  Because
that result was already observed, any accessibility calibration is necessarily
exploratory.  It will not be described as preregistered or confirmatory.

## Frozen ladder

The task, positive-rate shunting equations, connectivity, signal delta 0.24,
training gain 0.25, optimizers, 180-epoch cap and exact-resource morphologies
remain unchanged.  Only test gain standard deviation is crossed:

- 0.25: same gain distribution as training;
- 0.50: moderate extrapolation;
- 0.80: larger extrapolation below the failed 1.20 canary.

The first pass uses exact backpropagation only, depths D1 `[8]`, D2 `[2,3]`
and D3 `[2,1,2]`, and two new seeds 10110 and 10111.  This is 18 fits.  The
same training examples and deterministic initialization are intentionally
repeated across the three test-gain values within seed and morphology, so the
ladder isolates evaluation accessibility rather than optimization variance.

## Decision rule

A gain level is accessible only if every depth has mean test accuracy between
0.60 and 0.98 and both individual seeds exceed 0.55.  Among accessible levels,
the largest shift is selected before any LocalCA result is observed.  If no
level is accessible, this task family is not advanced for a physical-depth
journal claim.

An accessible operating point is not yet a positive depth result.  Advancement
to sufficient-seed controls additionally requires a descriptive D2 or D3
advantage over D1 that is not driven by one seed.  That later experiment must
include aligned, zero-alignment, sensor-shuffled and tree-rewired controls,
both LocalCA transport modes, and matched grouped-point/projected-BP controls.

## Claim boundary

This ladder asks only where the existing production network can generalize.
It does not test or establish biological robustness, superior sample
efficiency, or a dendrite-specific computation.  Any result is reported
together with the failed 1.20-shift canary.
