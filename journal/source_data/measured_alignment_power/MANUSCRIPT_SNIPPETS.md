# Suggested integration text

These are proposed passages, not edits to the manuscript. Figure/section references should be assigned by the manuscript owner. The analysis is exploratory and conditional on its explicit signal/noise model.

## Main Results: compact version

A response-level sensitivity simulation preserved the observed partners, shared inputs, repeated stimuli and seven-target inference while calibrating noise to measured repeat reliabilities. Under the specified Gaussian ancestry-signal model, 80% detection corresponded to a mean simulated target-level partial-rank effect of approximately 0.25. This conditional sensitivity estimate does not exclude smaller effects or alignment involving unobserved partners.

## Main Results: numerical detail if space permits

The simulation used 4,000 complete datasets with the same 13 scans and seven targets, retaining shared presynaptic roots and optical recordings. At measured reliability, detection exceeded 80% when approximately 60% of reliable tuning variance followed the imposed ancestry covariance (grid bracket 55–60%; interpolation 58.9%), corresponding to a mean simulated target-level partial-rank effect of 0.249. At the first tested crossing, power was 80.85% (95% Monte Carlo interval, 79.60–82.04%); the lower interval first exceeded 80% at a 65% signal fraction. The zero-signal two-sided rejection rate was 4.95% (4.32–5.67%). These are model-conditional detection probabilities, not confidence bounds on biological absence.

## Supplementary Methods

To quantify sensitivity conditional on the observed sampling, we simulated joint stimulus-response matrices for the 102 unique presynaptic roots represented by 125 partner observations in 13 scans of seven targets. We retained the actual contact geometry, condition hashes, repeat counts and imaging identifiers. This preserved 20 target–partner combinations repeated across scans and two presynaptic roots shared between targets. Each scan contained 136 repeated conditions: 130 presented twice and six presented ten times; their hashes identified 316 repeated conditions across the dataset.

Ancestry-related tuning was generated from a positive-semidefinite covariance obtained from shared path lengths. Within each arbor, the kernel between two partners was the soma-to-common-ancestor path length divided by the geometric mean of their soma-to-contact path lengths. Zero-depth contacts contributed independent diagonal components. We embedded the seven kernels in the common presynaptic-root coordinates, summed them and normalized the diagonal. For each stimulus, reliable tuning combined independent Gaussian unit-specific and ancestry-correlated components with variance fractions 1−lambda and lambda. We tested lambda=0,0.05,...,1. A root's latent response was shared across its recordings; independent measurement noise was shared only for an identical optical recording and stimulus.

For each recording, its observed odd/even split-half Spearman reliability was clipped below at zero and converted to a Gaussian Pearson correlation, r_p=2 sin(pi r_s/6). The two negative estimates were retained in the source table and assigned zero reliable variance. With h the mean inverse half-repeat count, simulated trial responses had reliable variance r_p and noise variance (1−r_p)/h. Noise in each condition mean was divided by its actual repeat count. A prespecified perfect-reliability comparison removed measurement noise. An independent 1,000-dataset calibration audit found mean and maximum absolute discrepancies of 0.0020 and 0.0069 between the simulated mean and clipped observed Spearman reliabilities; no parameters were retuned.

We computed partner similarities from the joint response matrices and applied the original partial-rank analysis, controlling ranked Euclidean separation and contact-depth difference. Scan effects were averaged within target. For each of 4,000 datasets, the two-sided signed-rank P value was computed by enumerating the seven targets' 128 sign assignments; positive alignment required P<=0.05 and a positive mean effect. Paired Gaussian draws were reused across signal fractions and reliability scenarios. Wilson intervals quantify Monte Carlo uncertainty in detection probability. We report the first sustained 80% crossing, its tested grid bracket and descriptive interpolation. This calculation assumes the specified Gaussian stimulus-tuning and reliability model and known shared-input dependence; it cannot establish sensitivity to unobserved partners, arbitrary covariance patterns or unknown biological dependence between targets.

## Supplementary figure caption

**Conditional sensitivity of the measured ancestry-response test. A**, Detection of positive ancestry alignment as the fraction of reliable tuning variance carried by a shared-path covariance increases. Every simulated dataset preserves the 13 scans, seven targets, observed partners, stimulus identities and repeats. Green uses measured repeat reliabilities; gray removes measurement noise. **B**, The same detection probabilities plotted against the mean simulated seven-target partial-rank effect. This is a model-conditional effect axis, not a universal detectable correlation. The dashed line in A,B marks 80%; bands are Wilson 95% Monte Carlo intervals from 4,000 paired simulated datasets. Detection requires an exact two-sided seven-target signed-rank P<=0.05 and positive mean effect. **C**, Mean split-half Spearman reliability in 1,000 independent calibration datasets versus the observed value, for 125 partner records. The two negative estimates are assigned zero reliable variance (orange squares); the dotted segment shows that clipping. The identity line describes agreement, not a fitted regression. All correlations between partner pairs are calculated from joint response matrices; pairs are not independent replicates.

## Interpretation suitable for Discussion

The absence of a measured association remains a bounded result. The reliability-calibrated simulation suggests that an ancestry signal producing a mean partial-rank effect around 0.25 would usually have been detected within the observed sample. It does not establish sensitivity to weaker effects, different signal geometries or inputs missing from the reconstructions and recordings.

## Numerical table

| Quantity | Measured reliability | Perfect reliability |
|---|---:|---:|
| Smallest sustained tested signal fraction with at least 80% detection | 0.60 | 0.25 |
| Tested crossing bracket | 0.55–0.60 | 0.20–0.25 |
| Descriptive interpolated signal fraction | 0.589 | 0.206 |
| Mean simulated partial-rank effect at interpolation | 0.249 | 0.248 |
| Detection at maximal specified signal | 0.9623 [0.9559, 0.9677] | 0.9978 [0.9957, 0.9988] |
| Mean simulated partial-rank effect at maximal signal | 0.341 | 0.531 |
| Zero-signal two-sided rejection probability | 0.0495 [0.0432, 0.0567] | 0.0548 [0.0481, 0.0622] |

Brackets are 95% Monte Carlo intervals, except rows explicitly labeled as tested crossing brackets. The maximum tested signal is a ceiling only within this variance-mixture family. Seven targets give a discrete minimum two-sided P of 0.015625, not a universal maximum power below one.
