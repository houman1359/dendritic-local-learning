# Final exploratory signal-accessibility contract

Frozen: 12 August 2026, after the coupling result and before observing any
result from this signal ladder.

## Design

The production positive-rate shunting model uses matched train/test
hierarchical gain SD 0.25 and mechanism-neutral child conductance 16.  The
coupling is the original canary value and the stable non-bimodal value from the
archived gain--load calibration; it is not chosen as a new optimum.  We cross:

- physical depth D1 `[8]`, D2 `[2,3]`, D3 `[2,1,2]`;
- excitatory signal delta 0.24, 0.36, 0.48 and 0.72;
- two new paired seeds 10130 and 10131.

All other task, resource and optimization coordinates remain fixed.  This is
24 exploratory exact-backpropagation fits.

## Frozen selector

Choose the smallest signal delta satisfying all of:

1. every morphology has mean test accuracy in [0.60, 0.95];
2. every individual seed exceeds 0.57;
3. D3 minus D1 is at least +0.02 in both seeds;
4. no morphology has mean accuracy above 0.95.

If no setting passes, the current task family is not advanced.  If a setting
passes, its value is frozen before any LocalCA or topology-control outcome is
observed.  A later sufficient-seed study must retain the failed severe-shift
and coupling canaries, use at least ten paired seeds, and include alignment,
sensor-shuffle, tree-rewiring, grouped-point and projected-BP controls.

Because this calibration follows multiple observed pilots, it is not evidence
for the final scientific claim.  It only chooses a measurable operating point.
