# Figure 5 within-target permutation correction

## What was wrong

The previous analysis computed both a pooled Spearman correlation and a
within-target centered correlation. Its permutation null shuffled learning
outcomes among methods within each target, but the two-sided tail probability
was evaluated against the absolute **pooled** correlation. The null and the
observed statistic therefore did not match. The accompanying bootstrap
interval also described the pooled rather than centered statistic.

## Correct test

For each target, the five non-oracle structural methods are retained. Credit
capture and normalized held-out MSE are each centered by their target mean.
The reported statistic is Spearman's correlation across the resulting 35
target-by-method observations.

The Monte Carlo null independently reassigns the five MSE values to method
labels within each target. It never exchanges observations between targets.
The centered correlation is recomputed after every reassignment, and the
two-sided value is

`(1 + number(|rho_null| >= |rho_observed|)) / (1 + valid permutations)`.

The interval resamples the seven target cells with replacement and recomputes
the same centered statistic. The pooled correlation is retained only as a
descriptive quantity and has no inferential result attached to it. All tests
use 20,000 bootstrap draws and 20,000 method-label permutations with the
pre-existing deterministic seed. No model was retrained.

## Corrected results

| cohort | targets | centered rho | target-bootstrap 95% CI | corrected permutation P | previous P |
|---|---:|---:|---:|---:|---:|
| 4 channels, primary | 7 | -0.150140 | [-0.448801, 0.144421] | 0.419479 | 0.291685 |
| 1 channel | 7 | -0.620636 | [-0.842009, -0.312863] | 0.000600 | 0.000300 |
| 2 channels | 7 | -0.288235 | [-0.621721, 0.046737] | 0.081546 | 0.182191 |
| 8 channels | 7 | -0.479759 | [-0.769631, 0.018028] | 0.010399 | 0.041398 |
| linear activation | 7 | 0.024930 | [-0.245851, 0.390021] | 0.888456 | 0.875756 |
| manual matches only | 6 | -0.134521 | [-0.543663, 0.317548] | 0.527424 | 0.293835 |
| reliability at least 0.1 | 7 | -0.236415 | [-0.497602, 0.100765] | 0.220389 | 0.068347 |
| reliability at least 0.2 | 7 | -0.130569 | [-0.347695, 0.210682] | 0.491675 | 0.432578 |

For the primary four-channel result, 8,389 of 20,000 null correlations were at
least as extreme as the observed centered correlation. The correction
therefore strengthens the stated boundary: the present seven-target cohort
does not show that methods with higher within-target credit capture learn
better. The channel-budget analyses remain sensitivities rather than separate
confirmatory tests and should not be presented as multiplicity-adjusted
discoveries.

## Changed files and outputs

- Corrected implementation:
  `drafts/dendritic-credit-routing/analysis/run_task_derived_credit_learning.py`
- Deterministic regression tests:
  `drafts/dendritic-credit-routing/tests/test_task_derived_credit_learning.py`
- Recomputed summaries, without retraining:
  `drafts/dendritic-credit-routing/results/microns_task_derived_credit_learning*/summary.json`
  and the corresponding `summary.md` files.
- Journal source summaries:
  `drafts/local-learning-journal/source_data/figure5/task_summary_ch{1,2,4,8}.json`
- Journal renderer:
  `drafts/local-learning-journal/scripts/build_journal_figures.py`
- Corrected figure:
  `drafts/local-learning-journal/figures/fig5_alignment_boundary.pdf` and
  `.png`.
- Updated source-data hashes:
  `drafts/local-learning-journal/source_data/provenance_manifest.tsv`.

The renderer now reads the four-channel centered correlation and permutation
value from the frozen JSON rather than hard-coding them.

## Validation

- Routing-analysis tests: 11 passed.
- Journal provenance and submission audit: 0 errors and 0 warnings.
- Figure 5 PDF uses embedded TrueType fonts and contains no Type 3 fonts.

Two prose locations still require integration by the manuscript owner because
they were explicitly outside this repair's edit scope:

- `drafts/local-learning-journal/main.tex`, line 426.
- `drafts/local-learning-journal/supplementary/supplementary.tex`, lines
  485--486.

Both currently state `P=0.292`; the corrected value is `P=0.419`.
