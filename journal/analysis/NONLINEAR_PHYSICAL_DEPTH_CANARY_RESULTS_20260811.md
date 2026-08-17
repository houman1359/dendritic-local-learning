# Nonlinear physical-depth canary: aligned backpropagation result

Run completed: 11 August 2026.  Status: diagnostic failure of the frozen
held-out learnability gate; not manuscript evidence for a positive depth
claim.

## Frozen run

- Run directory:
  `nonlinear_physical_depth_runs/journal_canary_physical_depth_aligned_bp_20260811231301`
- Frozen manifest SHA256:
  `5464dbc97f7ec8e2fd62c1d6d1be231308fa02f0b2f0317d0c0da135b908da26`
- Repository source identity recorded by the manifest:
  `661b6e62feb7f6becdd6963493520c9c6199ad24`
- Fresh diagnostic seeds: 10100 and 10101.
- Exact trainable parameter count in every morphology: 66,178.
- Morphologies: D1 `[8]`, D2 `[2,3]`, and D3 `[2,1,2]`, each with eight
  non-somatic branch units and the same 4/2/2 sensor inventory.

The run used the production positive-rate `PopulationNetwork`,
`StatefulDendriNet`, and divisive `DendriticBranchLayer` implementation.  It
therefore tests the original nonlinear rate code, not the reduced quadratic
credit-phase surrogate.

## Results

| Depth | Branch factors | Train accuracy, mean | Validation accuracy, mean | Test accuracy, mean | Test AUC, mean |
|---:|---:|---:|---:|---:|---:|
| 1 | `[8]` | 0.5368 | 0.5354 | 0.5056 | 0.5081 |
| 2 | `[2,3]` | 0.5627 | 0.5552 | 0.5049 | 0.5106 |
| 3 | `[2,1,2]` | 0.6560 | 0.6366 | 0.5102 | 0.5150 |

The primary descriptive depth contrast, best(D2,D3) minus D1, is only 0.0047
test-accuracy points in this two-seed screen.  More importantly, every cell is
below the preregistered 0.60 held-out-accuracy floor.  The aligned-BP gate
therefore fails.

Depth does improve fitting of the training distribution: D3 exceeds D1 by
0.1193 mean train-accuracy points and by 0.1012 mean validation-accuracy
points.  That improvement disappears under the frozen train-gain 0.25 to
test-gain 1.2 shift.  The defensible interpretation is that nonlinear physical
depth adds usable forward capacity in this operating point but does not, by
itself, produce gain-shift invariance.

## Decision

1. Do not run the zero-alignment, shuffled-sensor, or rewired-tree arms as if
   they were confirmatory; the prerequisite aligned learnability gate failed.
2. Complete the already frozen aligned LocalCA compatibility screen to test
   whether both production transport modes execute and descend.  Those runs
   remain diagnostic, regardless of outcome.
3. Freeze a separately labelled exploratory accessibility pilot with a
   moderate train--test gain shift.  Any setting selected after this outcome
   cannot be called preregistered or confirmatory.
4. Advance to sufficient-seed controls only if the new BP setting is learnable
   at every depth and not at ceiling.

## Aligned LocalCA compatibility screen

Both frozen production transport modes completed at all depths and seeds with
finite losses, no fallback transport and clean best-checkpoint reloads.  Mean
test accuracies were 0.5053, 0.5051 and 0.5093 for shared-soma LocalCA at
D1--D3, and 0.5051, 0.5036 and 0.5081 for exact path transport.  D3 again fit
the training distribution better (0.5779 shared; 0.5882 path transport) than
D1 (0.5340 and 0.5338), but the improvement did not survive the gain shift.

The LocalCA manifest SHA256 is
`b44658c52e1fdc69ba82e6806c9dcfe1de3f02c4a2c3cd5f06b599806f8bed9f`.
The row-level values are in
`source_data/nonlinear_physical_depth_canary/local_aligned_seed_rows.csv`.

## Runtime analysis issue

The generic `compartment_statistics` renderer raised a nonfatal post-training
`KeyError: 'exc_weights_mean'`.  The renderer expects the legacy single-cell
weight schema, whereas this experiment uses indexed sparse synapses inside a
population network.  All six training jobs exited successfully and wrote
their final performance files.  The generator now disables this incompatible
renderer; a schema-aware physical-depth diagnostic will compute the
prespecified branch-voltage, conductance, derivative, path-gain, and update
alignment endpoints.

The row-level values are frozen in
`source_data/nonlinear_physical_depth_canary/bp_aligned_seed_rows.csv`.  SHA256
digests of the six source `final.json` files, in configuration order 0--5,
are:

1. `162593c58f3e1640f58159a89ddfb12bf5f39a97c3321216ea57844a6f338b71`
2. `7f6ca5e757e411e3737d1ba23fe5823dd537c54604d215e46467ee645e1e7b11`
3. `dd34934c8e2395da653aab36926a72832c30fb4ace4ff231b3baa42060f1e75b`
4. `7111db84babf96f0d2e3fb2da4026dc32fdd20d7950e7b7eb65e7b2588f078c0`
5. `9191716d0e51107bd2e14941d8602054452315a3af52e2d5439c78edc3ef620b`
6. `0dd5b34b2d9a690457801243eaaf20dd6235fd6d8c2c3a542823ef33a011dc08`
