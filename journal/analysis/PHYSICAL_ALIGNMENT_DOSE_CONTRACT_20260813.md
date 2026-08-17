# Physical-depth alignment dose--response contract

Frozen: 13 August 2026, after the completed alpha-zero and alpha-one endpoint
cohorts were known, but before any intermediate-alignment outcome was viewed.
This is a prospective interpolation test of an established endpoint contrast,
not an independent replication or a blind selection of the operating point.

## Fixed design

The experiment reuses the positive-rate hierarchical gain--load task, signal
delta 0.80, matched train/test gain SD 0.25, child conductance 16, standard BP
optimizer, early stopping, D1 `[8]`, D2 `[2,3]`, D3 `[2,1,2]`, and paired seeds
10200--10209. No model, task or optimizer setting is retuned. The only new
factor is sensor alignment

`alpha in {0.25, 0.50, 0.75}`,

crossed with the three physical depths, for 90 new fits. The existing exact
same-seed alpha-zero and alpha-one BP cohorts provide the endpoints.

## Estimands and gates

The primary curve is the paired D3-minus-D1 test-accuracy effect as a function
of alpha. The directional prediction is that this depth effect increases with
alignment. We report all five alpha levels, the three adjacent paired
increments, alpha-0.75 minus alpha-0.25, and a within-seed linear slope without
selecting an alternative contrast after outcomes.

All 90 new configurations must complete with finite outcomes, no fallback
warning, exactly seeds 10200--10209 and the frozen resource counts (66,178
trainable parameters, 14,336 active contacts, 21,760 candidate slots and 2,944
persistent state scalars). A positive dose claim requires a positive mean
within-seed slope with a paired-bootstrap 95% interval excluding zero and at
least 8/10 positive seed slopes. Non-monotone intermediate means will be shown
and described rather than smoothed away.

## Scope

The experiment tests whether the already observed aligned physical-depth
effect appears as a graded task--tree correspondence rather than only at the
binary endpoints. It remains one calibrated H=3 task family. It cannot establish
prevalence in natural tasks, an independently selected crossover, a second
hierarchy depth or a dendrite-exclusive computation.
