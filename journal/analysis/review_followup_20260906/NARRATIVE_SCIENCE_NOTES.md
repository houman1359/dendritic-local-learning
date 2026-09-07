# Independent narrative science and flow notes

Read-only review; no edits to `main.tex`.

1. The image Results currently say the readout supplied the exact somatic gradient in all three conditions. This risks implying that the strict scalar retains the exact neuronal gradient. Distinguish calculation from delivery: “Each primary ladder computed neuronal errors through the exact readout derivative, then retained their neuronal coordinates or reduced them to the strict scalar; the random-feedback control is described below.” The neuron-specific condition is the exact somatic-gradient reference.

2. The conductance cancellation diagnostic compares raw gradient estimates before optimizer preconditioning, not the resulting Adam steps. Prefer: “At the same broadcast-trained weights, context-conditional gradient estimates cancelled more strongly under broadcast.” For the next sentence: “Before optimizer preconditioning, each example’s delivered gradient remained nonnegatively aligned with its exact gradient.” This preserves the correct distinction between per-example alignment and cancellation after averaging.

3. The compatible positive-conductance teacher described at the end of the interaction section (Supplementary Figs. S42–S44) and the initial monotonic conductance study opening the new section (S50) are distinct experiments. Name the former “the earlier compatible positive-conductance teacher” and the latter “an additional monotonic conductance teacher” so the reader does not infer repetition or reuse of one cohort.

4. The proposed shorter Discussion explicitly says the three supplied spatial patterns use oracle coefficients. Keep that qualification: the experiment establishes sufficiency of supplied profiles, not a learned local coefficient encoder or a need for three independent external errors.

5. The physical-depth accuracy reversal and persistent cross-entropy advantage should remain together. The proposed cuts leave that Results section intact and retain both facts in the Discussion.

6. The wider-bound opponent experiment correctly uses restarts from the same initial states and minibatches; describing it as continued optimization would be wrong. The present Results “restarts” wording is correct.

7. All Figure 1 claims and references need a final pass after the fresh six-arm experiment, especially strict scalar versus neuronal identity, projected K1 versus K3 spatial resolution, and the exact-path reference. The new oracle K1 projection isolates example-dependent mean amplitude from the additional distinctions preserved by K3.
