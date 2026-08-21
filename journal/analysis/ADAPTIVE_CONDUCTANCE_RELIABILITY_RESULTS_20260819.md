# Adaptive local conductance reliability: confirmatory results

Date: 19 August 2026

All 1,050 planned seed--condition outcomes completed for 50 independent seeds,
three reliability-heterogeneity levels and seven paired methods. Positive-input,
state-match, finite-difference, gain-range and independent point-gate gates all
passed. The maximum physical-shunt/point-gate trajectory difference was
`3.997e-15`.

At the frozen maximum heterogeneity (`h=2`):

- Adaptive local minus adaptive global one-step loss decrease was +0.001420
  (95% paired-seed bootstrap interval 0.001158 to 0.001684; 45/50 positive;
  two-sided Wilcoxon `P=4.49e-13`). Final loss was lower by 0.016217
  (0.014501 to 0.017913; 50/50 seeds favored local; `P=1.78e-15`).
- Adaptive local minus shuffled one-step loss decrease was +0.000736
  (0.000262 to 0.001255), but only 26/50 signs favored local and the frozen
  sign gate failed. Final loss was lower by 0.005528 (0.003766 to 0.007418;
  41/50 seeds favored local; `P=7.80e-8`), passing the frozen final-loss gate.
- Adaptive local final loss was higher (worse) than no shunt by 0.004250
  (0.002499 to 0.005975; 39/50) and higher than the fixed initial oracle by
  0.003449 (0.002577 to 0.004305; 43/50). The one-step adaptive-minus-no-shunt
  interval barely excluded zero, but only 24/50 signs favored adaptive and the
  Wilcoxon test was null; it is not treated as a positive result.
- Adaptive local and an independently evaluated adaptive point gate agreed to
  numerical precision at one step and final training.
- The final adaptive branch-gain ordering correlated with the fixed initial
  oracle ordering at mean within-seed Spearman rho 0.756; mean absolute gain
  error was 0.205. These are descriptive estimator diagnostics.

Interpretation: the online local estimator recovers enough branch ordering for
adaptive placement to outperform a single global gain and a shuffled map. It
does not recover the oracle magnitude well enough to improve final learning
over the unshunted noisy rule. This strengthens the paper's conditional
conductance result while preserving the point-emulation and state-clamp
boundaries. It does not establish autonomous inhibitory plasticity in a
biophysical neuron.
