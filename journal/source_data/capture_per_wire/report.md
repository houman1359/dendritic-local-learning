# Wiring-normalized capture (capture per wire)

Secondary analysis of the frozen Figure 3 routing-capacity table
(`source_data/figure3/routing_capacity_curves.csv.gz`; 8 reconstructed
cells, Monte Carlo streams averaged within cell before any ratio).
Capture per wire divides a dictionary's credit capture by its wiring
density (fraction of dense-feedback nonzeros).  The oracle-relative
factor divides each cell's capture-per-wire by the dense PCA oracle's
capture-per-wire on the same cell at the same channel count; intervals
are paired cell bootstraps (20,000 resamples of the 8 cells,
`numpy.random.default_rng(20260812)`).

## Headline (8 feedback channels, mean over 8 cells)

| dictionary | capture | wiring density | capture per wire | vs dense oracle (95% CI) |
|---|---|---|---|---|
| morphology-aware paths | 0.487 | 6.92% | 7.67 | 14.18 (10.72-18.17) |
| random paths | 0.285 | 8.13% | 3.38 | 6.08 (5.56-6.65) |
| depth-only bins | 0.222 | 12.50% | 1.78 | 2.99 (2.50-3.46) |
| shuffled ancestry | 0.198 | 6.92% | 3.06 | 5.31 (3.93-6.59) |
| dense PCA oracle | 0.569 | 100.00% | 0.57 | 1.00 (1.00-1.00) |

- Morphology-aware paths reach 85.0% of the
  dense oracle's capture while using 6.92% of
  its wiring, a 14.2x advantage in
  capture per unit wiring (95% CI
  10.7-18.2).

## Verification against published 8-channel numbers (tolerance 0.005)

- morphology-aware paths: mean credit_capture: published 0.487, reproduced 0.4871
- morphology-aware paths: mean wiring_density: published 0.0692, reproduced 0.0692
- random paths: mean credit_capture: published 0.285, reproduced 0.2847
- depth-only bins: mean credit_capture: published 0.222, reproduced 0.2225
- shuffled ancestry: mean credit_capture: published 0.198, reproduced 0.1978
- morphology-aware paths: mean oracle_fraction: published 0.85, reproduced 0.8495

## Scope

Wiring density counts feedback nonzeros, not axonal path length or
conduction delay, so capture per wire is a connectivity-budget measure,
not a metabolic one.  The 16-channel rows present in the frozen curves
are outside the published 1-8 channel grid and are not reanalysed here.
The normalization rewards sparse dictionaries by construction; the
substantive comparison is against the equally sparse controls
(ancestry-shuffled paths share the morphology dictionary's exact wiring
density), not against the dense oracle alone.
