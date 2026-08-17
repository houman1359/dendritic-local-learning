# Two-factor Shapley attribution of focal gradient localization

## Question and estimand

This analysis allocates the focal-shunt localization effect between the two
factors of the modeled excitatory gradient: the soma-derived adjoint and the
local driving force. It uses the four stored per-site localization payoffs in
`source_data/focal_decomposition/site_decomposition.csv`.

The primary estimand is the full-shunt minus first-order-current-matched
additive contrast. For payoff values \(v_{00}\) (matched additive), \(v_{10}\)
(adjoint only), \(v_{01}\) (driving force only), and \(v_{11}\) (full shunt),
the two Shapley values are

\[
\phi_{q}=\tfrac12[(v_{10}-v_{00})+(v_{11}-v_{01})],\qquad
\phi_{D}=\tfrac12[(v_{01}-v_{00})+(v_{11}-v_{10})].
\]

They reconcile exactly to \(v_{11}-v_{00}\) at every site. The reported mean
weights each of the eight cells equally after averaging its focal sites.
Confidence intervals are from 20,000 hierarchical bootstrap draws: cells are
resampled with replacement, then sites are resampled within each sampled cell
occurrence.

## Primary result: full shunt versus matched additive

Across 101 focal sites in eight cells, the equal-cell mean localization
contrast was 0.06942 (95% hierarchical-bootstrap CI 0.04942 to 0.09132; all
8/8 cell contrasts positive, two-sided Wilcoxon \(P=0.0078125\)).
The four equal-cell mean payoffs were 0.03491 for matched additive, 0.12996
for adjoint only, 0.02571 for driving force only, and 0.10434 for full shunt.

- Adjoint Shapley attribution: 0.08684 (95% CI 0.06299 to 0.11089; 8/8 cells
  positive, \(P=0.0078125\)). This is 125.1% of the net contrast (95% CI
  113.3% to 146.5%).
- Driving-force Shapley attribution: -0.01741 (95% CI -0.02931 to -0.00914;
  8/8 cells negative, \(P=0.0078125\)). This is -25.1% of the net contrast
  (95% CI -46.5% to -13.3%).
- The two-factor interaction was -0.01643 (95% CI -0.02297 to -0.01081;
  8/8 cells negative, \(P=0.0078125\)). Shapley allocation divides this
  interaction equally between the two factors.

Thus, relative to the additive control, the adjoint contribution is larger
than the observed net benefit and the driving-force contribution partially
cancels it. A negative driving-force attribution does not mean that the
driving-force-only localization payoff is negative; it means that its average
marginal contribution to the full-versus-additive contrast is negative.

## Sensitivity: full shunt versus the unperturbed field

The matched additive perturbation is a control, not the literal factorial
state in which neither shunt factor changes. We therefore repeated the
allocation with unperturbed localization, which is zero by definition, as
\(v_{00}\).

The full-shunt localization was 0.10434 (95% CI 0.07455 to 0.13559). The
adjoint attribution was 0.10429 (95% CI 0.07449 to 0.13552), or 99.96% of the
net response. The driving-force attribution was 0.000043 (95% CI -0.000005 to
0.000174; 6/8 cells positive, \(P=0.148\)), or 0.04% of the net response. The
interaction was -0.05134 (95% CI -0.08073 to -0.02955).

The near-zero driving-force Shapley value reflects cancellation: its positive
standalone localization is offset by a negative interaction when it is added
after the adjoint change. It should not be read as evidence that dendritic
voltage changes are absent.

## Validation and boundaries

- Maximum absolute reconciliation residual was \(5.6\times10^{-17}\) at the
  site and bootstrap levels and \(6.9\times10^{-18}\) after cell averaging.
- Both allocations apply to the nonlinear localization-index payoff, not to
  an additive decomposition of raw gradient vectors.
- The adjoint-only and driving-force-only fields are algebraic substitutions
  in the passive model. They are not independent biological interventions and
  do not establish in-vivo plasticity.
- The matched-additive allocation is the relevant control contrast, but its
  four payoffs are not a literal two-by-two physical intervention. The
  unperturbed-reference analysis is the conventional factorial sensitivity.
- With eight cells, the cell-level sign consistency and effect sizes are more
  informative than treating the minimum attainable signed-rank value as
  high-powered inference.

## Reproducible artifacts

- Script: `scripts/analyze_focal_gradient_shapley.py`
- Site-level attributions: `source_data/focal_decomposition/site_shapley.csv`
- Cell-level summaries: `source_data/focal_decomposition/cell_shapley.csv`
- Bootstrap draws: `source_data/focal_decomposition/shapley_bootstrap_draws.csv.gz`
- Machine-readable summary: `source_data/focal_decomposition/shapley_summary.json`

Reproduce from the journal directory with:

```bash
python scripts/analyze_focal_gradient_shapley.py
```
