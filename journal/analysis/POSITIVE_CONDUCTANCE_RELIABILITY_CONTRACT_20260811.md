# State-matched positive-conductance reliability contract

Frozen: 11 August 2026, before confirmatory outcomes.

## Question

Does a strictly nonnegative focal conductance implement the branch-reliability
shrinkage predicted by the credit-operator theorem when forward branch voltage
is held identical by an oracle compensating current?

## Fixed design

- Eight branches with four nonnegative rate inputs per branch.
- A positive excitatory conductance and leak at every branch; inhibitory shunt
  conductance is nonnegative and has reversal 0.
- Exact state matching: the injected current `kappa * V_pre` cancels the
  shunt's forward voltage effect while retaining the lower physical input
  resistance in the local eligibility.
- Five levels of prescribed branch-SNR heterogeneity and 50 untouched paired
  confirmatory seeds.
- Forty stochastic full-data updates; the common step is one half of the
  numerically estimated initial smoothness limit.

## Controls

1. exact clean backpropagation;
2. noisy credit without shunting;
3. state-matched additive current without conductance attenuation;
4. the best global shunt gain;
5. reliability-aligned, shuffled and anti-aligned positive shunts;
6. an explicit point gate receiving the identical sample-dependent gains;
7. aligned conductance attenuation under clean exact credit.

## Frozen predictions

- At equal branch reliability, aligned and best-global shunts coincide.
- With heterogeneous reliability, aligned shunting improves one-step and final
  population loss relative to the best global gain.
- Shuffling weakens and anti-alignment reverses the benefit.
- The explicit point gate matches the state-matched conductance update exactly.
- With clean exact credit, the optimal reliability gain is one; exact clean
  backpropagation remains the optimization reference.

## Claim boundary

The compensating current is an oracle clamp that isolates backward gain from
forward feature suppression. This is a trained positive-conductance mechanism
test, not an autonomous biological circuit, learned shunt placement or a
unique dendritic advantage.
