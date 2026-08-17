# Exploratory coupling-by-depth contract in the production rate model

Frozen: 12 August 2026, after the gain-accessibility result and before any
result from this coupling ladder.

## Question

Does physical depth become useful only when axial access is strong enough to
prevent serial attenuation, as suggested by the archived clean gain--load
calibration?  This asks whether the failed severe-shift canary reflects an
intrinsic limitation of depth or a specific coupling regime.

## Design

The production positive-rate shunting model is trained and tested on matched
hierarchical gain SD 0.25.  Signal delta remains 0.24.  We cross:

- physical depth D1 `[8]`, D2 `[2,3]`, D3 `[2,1,2]`;
- mechanism-neutral initial child conductance 1, 4, 16 and 64;
- two new paired seeds 10120 and 10121.

This is 24 exact-backpropagation fits.  Every morphology retains eight
non-somatic branch units, the same 4/2/2 sensor inventory, 16 excitatory and 12
inhibitory contacts per branch, and 66,178 trainable parameters.  Coupling is
set identically at every axial edge.  No optimizer or morphology-specific
hyperparameter is varied.

The coupling values were selected from an already observed external
calibration, which found a depth reversal between 1 and 64.  This ladder is
therefore exploratory and cannot provide a fresh confirmatory interaction.

## Gates and interpretation

An operating point is accessible only if every morphology has mean test
accuracy in [0.60, 0.98] and both individual seeds exceed 0.55.  The primary
descriptive interaction is

`(D3 - D1 at conductance 64) - (D3 - D1 at conductance 1)`.

A positive interaction supports a serial-access explanation, not a universal
advantage of depth.  Advancement requires an accessible coupling and a
D2/D3 advantage present in both seeds.  If it advances, the next frozen study
must test alignment, sensor shuffle, rewiring, LocalCA transport and grouped-
point/projected-BP controls at sufficient seed count.  If no coupling is
accessible, the nonlinear hierarchy-depth proposal remains unsupported by
the current production task.
