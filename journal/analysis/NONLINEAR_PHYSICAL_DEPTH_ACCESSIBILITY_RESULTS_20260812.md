# Exploratory nonlinear physical-depth accessibility result

Completed: 12 August 2026.  Status: no gain level passed the frozen all-depth
accessibility gate; threshold-like depth effect retained as exploratory.

## Result

| Test gain SD | D1 `[8]` | D2 `[2,3]` | D3 `[2,1,2]` | D3 minus D1 |
|---:|---:|---:|---:|---:|
| 0.25 | 0.5381 | 0.5604 | 0.6423 | +0.1043 |
| 0.50 | 0.5202 | 0.5291 | 0.5642 | +0.0440 |
| 0.80 | 0.5098 | 0.5159 | 0.5276 | +0.0178 |

Values are mean test accuracy over two fresh seeds.  At gain SD 0.25 the
D3-minus-D1 contrast is +0.1033 and +0.1052 in the two individual seeds.  This
is a reproducible threshold-like forward-depth effect in the small screen, not
a sufficient-seed estimate.

No gain level is accessible by the frozen rule because D1 and D2 remain below
0.60 even when train and test gain distributions match.  Accordingly, the
ladder does not authorize the planned sufficient-seed topology and LocalCA
factorial.

## Numerical isolation audit

Within every seed and morphology, training and validation accuracies are
bit-identical across all three test-gain settings.  Their maximum within-cell
range is exactly zero.  The only changing dataset coordinate is therefore
reflected only in the held-out test values, as intended.

## Why the earlier gain--load calibration was stronger

The archived gain--load child-conductance calibration used zero gain
variability in both training and test data.  This ladder uses hierarchical
gain SD 0.25 during training.  The prior conductance-16 shunting accuracies
0.7519, 0.7906 and 0.7427 therefore describe a clean signal task, not the
factorized multiscale-gain task run here.  The discrepancy is an experimental
regime difference, not a mismatch in the production dendritic equations.

The archived calibration also showed a strong coupling-by-depth interaction:
at coupling 1, D1/D2/D3 were 0.7997/0.6002/0.5057, whereas at coupling 64 they
were 0.7458/0.8194/0.9457.  A new-seed coupling ladder under hierarchical gain
is consequently the next diagnostic.  Because those older outcomes motivated
the ladder, it is explicitly exploratory.

## Provenance

- Run:
  `nonlinear_physical_depth_runs/journal_exploratory_physical_depth_accessibility_bp_20260811234745`
- Frozen manifest SHA256:
  `73ae9781ff0b07cbb460990f85c30363e78cdfd4a2e242657ed71cf6a23074c2`
- Row-level source:
  `source_data/nonlinear_physical_depth_accessibility/bp_seed_rows.csv`
