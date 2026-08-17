# Focal shunting and exact credit

Completed 235 focal perturbations across 45 real reconstructed cells.
Somatic current clamp held the output voltage—and therefore the scalar output error—fixed.

## Shuffle-eligible subset

The summary below is restricted to the 40 cells and 230 sites that also have a
within-cell depth-shuffled relation control. It is not the primary all-eligible
estimate. At dose 1, mean shunt localization was 0.1571 and matched-additive
localization was 0.0131. The cell-level shunt-minus-additive contrast was
0.1441, 95% bootstrap CI [0.1226, 0.1697], with 40/40 cells positive and
Wilcoxon p=1.819e-12.
True shunt localization exceeded the depth-shuffled relation control by 0.1176, 95% CI [0.0933, 0.1447], Wilcoxon p=1.819e-12.

The publication-facing primary estimate uses all 45 eligible cells and 235
sites: 0.1377, 95% interval [0.1165, 0.1621], with 45/45 cells positive. It is
stored in `../replication_summary.json`.

## Numerical validation

Maximum adjoint/finite-difference relative error: 7.169e-03.
Maximum residual state-matched somatic-voltage error: 1.735e-18.

## Boundary

This demonstrates a mechanistic signature in a calibrated passive model on real anatomy. It does not show that inhibition carried teaching signals in vivo.
