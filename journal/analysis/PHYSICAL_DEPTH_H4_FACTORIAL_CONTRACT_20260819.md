# Physical-depth by hierarchy extension: frozen H=4 contract

Frozen: 19 August 2026, before any H=4 training outcome was inspected.

## Question

Does the trained optimum in physical dendritic stage count track a fourth
level of task hierarchy, or did the earlier H=2 and H=3 results merely reward
the deepest architecture offered in each experiment?

This extension completes a trained H=2/H=3/H=4 series within the same
mechanism-matched task family. It is a controlled test of the predicted
depth--hierarchy crossover, not a natural-task or biological replication.

## Frozen design

- The task generator, optimizer, training horizon, conductance operating point,
  train/validation protocol and stopping rule are inherited unchanged from the
  H=3 confirmatory cohort.
- Only the task hierarchy is extended to four levels. The 64 input features
  are partitioned into four ordered 16-feature tiers.
- Physical architectures are D1 `[8]`, D2 `[2,3]`, D3 `[2,1,2]` and D4
  `[1,1,2,2]`. Each has exactly eight nonsomatic branch units per soma:
  `8`, `2+6`, `2+2+4`, and `1+1+2+4`.
- The common branch inventory is `[4,2,1,1]`. Under aligned placement its
  feature ranges are `[0:16]`, `[16:32]`, `[32:48]`, `[48:64]`; reversed
  placement assigns those ranges in the opposite physical order.
- Ten fresh paired seeds 10400--10409 are used. The training seed is the
  inferential unit.
- Five frozen arms are run:

  1. aligned serial shunting BP, D1--D4;
  2. reversed-placement serial shunting BP, D1--D4;
  3. aligned and reversed literal grouped-point BP, D1--D4;
  4. aligned and reversed serial LocalCA with shared-soma and exact-path
     transport, D1--D4;
  5. aligned serial raw-additive BP, D1--D4.

This is 360 fits: 40 + 40 + 80 + 160 + 40. No task, optimizer, initialization
or stopping parameter may be changed after outcomes are opened.

## Primary estimands

1. Aligned serial-BP D4-minus-D3 accuracy and D4-minus-D1 accuracy.
2. The aligned-minus-reversed interaction for serial-BP D4-minus-D3.
3. The aligned serial-minus-grouped-point contrast at D4 and its
   aligned-minus-reversed interaction.
4. D4-minus-D3 and aligned-minus-reversed interactions for shared-soma and
   exact-path LocalCA.
5. The shunting-minus-raw-additive interaction in D4-minus-D3 under aligned
   BP.

The H=2, H=3 and H=4 seed-level tables will then be combined without pooling
seeds across conditions as though they were repeated measurements. The
cross-hierarchy descriptive endpoint is whether the best mean serial depth is
D2, D3 and D4 for H=2, H=3 and H=4, respectively.

## Decision rules

- Paired-seed bootstrap intervals and exact two-sided sign-flip tests are
  reported for every primary contrast.
- A directional positive claim requires a 95% paired interval excluding zero
  and at least 8/10 paired seed signs in the predicted direction.
- All null, reversed and raw-additive outcomes are retained.
- Failure of D4 to exceed D3 is a falsification of a simple `D=H` trained
  crossover at H=4 and must be reported as such; it cannot be repaired by
  retuning the task or operating point.

## Validity gates

- All 360 fits reach finite, stage-complete checkpoints without fallback
  warnings.
- Within every architecture comparison, active contacts, candidate slots,
  trainable parameters and persistent state counts satisfy the same exact-
  resource rules used for H=2/H=3.
- D1 serial and literal grouped-point logits agree at initialization.
- Each grouped-point direct projection contains the same number and initial
  values of raw conductances as the serial child aggregator it replaces.
- The generated configuration count, seeds, hierarchy depth, inventories and
  feature ranges agree exactly with this contract.

