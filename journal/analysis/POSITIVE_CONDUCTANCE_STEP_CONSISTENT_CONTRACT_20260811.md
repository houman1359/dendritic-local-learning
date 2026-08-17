# Step-consistent positive-conductance reliability correction

Frozen: 11 August 2026, before canary or confirmatory outcomes for seeds
8098--8149.

## Reason for the correction

The first mechanism experiment used the attenuation
`S/(S+N)`, which is optimal for a step `eta=1/L`, while its actual common
step was `eta=0.5/L`.  Those data remain archived in
`source_data/positive_conductance_reliability/`, but they are superseded for
inference by this prospectively specified correction.

For a fixed step `eta=c/L`, the branchwise smoothness guarantee is

`(c/L) [a S - (c/2) a^2 (S+N)]`,

so a physical attenuator constrained to `0 <= a <= 1` uses

`a* = min(1, S/[c(S+N)])`.

The best common gain is derived from pooled signal and noise by the same
formula.  Here `c=0.5`; all other task, noise and training settings are held
fixed from the original study.

## Prospective design

- Two artifact-only canary seeds: 8098--8099.
- Fifty untouched paired confirmatory seeds: 8100--8149.
- Five branch-SNR heterogeneity levels and nine methods.
- Forty common-noise updates at `eta=0.5/L`.
- Strictly nonnegative rates, excitatory conductances and shunt conductances.
- Shunts remain fixed from the initial oracle signal and noise energies.

## Independent implementation controls

The physical conductance update is evaluated from the shunted denominator
with the compensating current held fixed during differentiation.  The point
control is evaluated separately from the unshunted eligibility multiplied by
the realized sample-dependent gain `G/(G+kappa)`.  The state-clamped
unattenuated control constructs the matched forward state but deliberately
uses the unshunted denominator.  Equality of trajectories is an output gate,
not shared-array assignment.

An additional finite-difference gate perturbs an excitatory parameter while
holding the precomputed clamp current fixed.  This directly checks the
physical-shunt gradient used by the update.

## Predictions

1. At zero reliability heterogeneity, the aligned and best common fixed-step
   gains coincide.
2. At maximum heterogeneity, the aligned gain improves one-step loss decrease
   relative to the best common gain.
3. Shuffling or reversing branch assignments weakens the aligned advantage.
4. Independently computed point-gate and physical-shunt trajectories agree to
   numerical precision.
5. The aligned comparison with unshunted noisy learning is a two-sided
   boundary test, not a directional success criterion.

## Scope

The experiment tests a rate-based, single-compartment conductance reduction
under an oracle state clamp.  It does not test learned reliability,
multi-compartment transport, nonlinear dendritic depth, or a computation
unavailable to a point model supplied with the same gates.
