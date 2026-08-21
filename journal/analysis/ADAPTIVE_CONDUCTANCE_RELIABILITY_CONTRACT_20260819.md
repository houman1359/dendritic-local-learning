# Adaptive local conductance-reliability experiment

Frozen: 19 August 2026, before artifact-only canary or confirmatory outcomes
for seeds 8198--8249.

## Question

The completed positive-conductance experiment supplied each branch with its
initial oracle signal and noise energies. This extension asks whether the same
conditional benefit survives when those moments are estimated from noisy local
credit observations.

## Estimator

At the current parameter state, a branch observes two conditionally
independent noisy gradient samples, `g1_b` and `g2_b`, generated from the same
examples. It forms

`Sobs_b = <g1_b, g2_b>` and
`Nobs_b = ||g1_b - g2_b||^2 / 2`.

Their expectations are the squared clean branch gradient and the noise energy
of one gradient sample. Eight paired observations initialize the estimator;
an exponential moving average with decay 0.9 is then updated once per training
step. Negative finite-sample signal estimates and both moments below `1e-12`
are clipped only when a gain is constructed. For step `eta=c/L`, the adaptive
gain is `min(1, Shat/[c(Shat+Nhat)])`, bounded below by 0.03. The corresponding
nonnegative shunt uses the branch's current mean total conductance. The method
does not receive the oracle moment arrays or their branch ordering.

## Frozen design

- The task, positive inputs, conductance model, state clamp, step fraction and
  40-update horizon are unchanged from the corrected reliability experiment.
- Three heterogeneity levels: 0, 1 and 2.
- Seven paired methods: exact clean BP, noisy no-shunt, fixed initial-oracle
  shunt, adaptive local shunt, adaptive global shunt, fixed-permutation
  adaptive shunt and independently computed adaptive point gate.
- Two artifact-only canary seeds 8198--8199; 50 untouched confirmatory seeds
  8200--8249.
- Main update noise and estimator-probe noise are paired across methods.

## Primary and boundary comparisons

At heterogeneity 2, the two primary contrasts compare adaptive local with
adaptive global and shuffled shunts in final test loss. The comparisons with
no shunt and the fixed initial oracle are two-sided boundary results and are
retained regardless of sign. Immediate one-step results, estimator accuracy
and all lower-heterogeneity cells are secondary.

Inference uses the 50 independent seeds. Paired bootstrap intervals use 20,000
draws; Wilcoxon tests are descriptive two-sided tests. A positive mechanism
claim requires an interval excluding zero and at least 40/50 seed signs.

## Numerical gates

The canary exposes only non-outcome gates: positive inputs, exact state match,
finite-difference agreement, gains in [0.03, 1], and numerical identity of the
independent adaptive physical-shunt and point-gate trajectories. Confirmatory
analysis is forbidden unless the canary passes with unchanged script, base
implementation and configuration hashes.

## Claim boundary

The estimator removes access to oracle signal/noise energies, but it is not a
complete biological learning mechanism. It receives paired independent noisy
teaching observations, knows the optimizer's step fraction, and converts an
estimated gain into a shunt under an exact state-matching current clamp. The
experiment cannot establish kinetic inhibitory plasticity, a dendrite-only
computation or utility on a natural task.
