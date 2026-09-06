# Methods, Supplementary Information and response-baseline completion

## Ownership and edits

Edited `main.tex` only between `\section*{Methods}` and immediately before
`\section*{Ethics and data reuse}`, always splicing the latest file. Edited
`supplementary/supplementary.tex` throughout. Root owns all other main text and
main captions; figure agent owns existing canonical artwork. Native new S34
was explicitly assigned to this agent and is generated from the new response
baseline outputs.

Backups, intermediate source blocks, patches and temporary build outputs are
under `/tmp/dendritic_methods_consolidation/`. No historical numerical archive
was overwritten by this agent. Scientific-writing skill instructions were read
and applied. The paper is treated as a standalone Nature Communications work;
there is no dependency on a companion publication or claim of an official
NeurIPS decision in owned text.

Main Methods changed from 12,892 to 4,412 prose words (65.8% reduction),
including all newly completed experiments. The 28 former subsections became
nine thematic subsections and a compact cohort/reproducibility table. All four
original Methods equation labels and all scientific citation keys were
preserved across main+SI. Unique quadratic-screen settings and conceptual-map
coordinates were consolidated into their existing SI contexts; repeated
protocols were removed instead of copying the full old Methods into SI.

SI order now follows the main scientific progression: exact derivation,
learning fields, capacity, update moments/prospective selection, artificial
experiments, reconstruction, structural capacity, inhibitory anatomy, focal
transport, measured responses, imposed alignment, external animal/interference,
statistics. Existing S1–S31 figure numbers were retained. New S32–S34 are
inserted in order. Main-figure references were updated to the new 1–9 sequence
(feedback 2, moment theory 3, branch conflict 4, subtree factorial 5,
prospective selection 6). Physical-depth protocols are now SI S5.

## New experiment documentation integrated

- Prospective tree-constrained linear selection: separate development/held-out
  task seeds and rotation angles; exact reference-spectrum invariance; real
  minibatch covariance; explicit passive tree, supplied decoder and declared
  cable-cost limitations; source/selection seals; 12,800 primary fits and
  negative long-horizon comparison; positive rank-only baseline and posthoc
  one-step validation. No outcome-dependent retuning or biological law claimed.
- Explicit local-context coefficient encoder: complete separate calibration
  protocol, soft primary result and large oracle gap. Hard argmax readout is
  explicitly exploratory after soft outcomes, with unchanged encoder and paired
  seeds. Larger calibration sets also entail more updates at fixed epochs.
- Original 20-seed branch trajectories: six checkpoints, common-exact and
  own-state diagnostics, 28,800 rows, endpoint drift and cache-location-only
  recovery with official Fashion-MNIST checksums.
- Morphology audit: full direct/proxy contingency and nonrandom label
  availability, mapping/radius perturbations on fixed nominal probes, 1,056
  conditions, exact raw-edge series-resistance comparison. Chain compression
  preserves length but does not preserve heterogeneous axial resistance.
- Training-only nested response baselines and observed-input sensitivity,
  detailed below; matching new S34 and Methods/cohort pointers.

## New response-baseline analysis

Files:

- `configs/review_completion/measured_response_baselines.json`
- `scripts/analyze_review_response_baselines.py`
- `scripts/audit_review_response_baselines.py`
- `scripts/build_review_response_baselines_figure.py`
- `source_data/review_response_baselines/` (all outputs, manifests and audit)
- `figures/supplementary/figure_S34_panels_A-D.pdf`

The 3,380 new evaluations reuse 130 complete-tree outer splits across all
13 scans/seven targets. Three inner stimulus-identity folds select ridge
penalties and refit preprocessing and repeat-reliability filters. Manual-only
partners, three reliability thresholds, three nested retention fractions
(five draws), three Gaussian-noise doses and a raw-standardized ridge
reference are retained. Empty masks remain as intercept-only predictions.
Only linear models are fitted under these sensitivities; archived nonlinear
references are read unchanged. Mask draws average before splits, scans and
targets; 20,000 resamples of seven target means define intervals.

Primary normalized MSE: train mean 1; OLS 0.80698; nested ridge 0.80282;
archived exact nonlinear 0.78766. Paired exact-minus-ridge is −0.015153
[−0.027147, −0.004203], six of seven targets lower. Subtree-minus-ridge is
0.029262 [−0.013009, 0.073572]. Manual-only MSE is 0.91203, 25%-retention
MSE is 0.91689 and unit-noise MSE is 0.86875. These establish observed-input
sensitivity, not unknown-partner recovery, population power or a causal
explanation of the negative ancestry result.

Validation uses the unchanged full-tree split helper/seed rule and verifies
all 130 historical train/test counts and stimulus disjointness. Reconstructed
trial masks and identity sets are hashed. The original historical mask arrays
were not separately archived, so this limit is stated. A meaningful leakage
challenge changes every outer-test input and response to extreme values in
one full nested analysis per scan: all 338 condition-level selected penalties,
masks and training-only reliability values remain exactly unchanged.
`validation_report.json` records these checks and source hashes. All three
scripts pass Python compilation. The analysis completed in about 25 seconds;
the additional independent audit completed successfully. Native S34 and its
final compiled manuscript page were visually inspected.

## Caption and uncertainty audit

Corrections verified against retained plotting/analysis code and numerical
sources, with figure-agent coordination:

- S1C now MNIST only; D/E raw-additive only. S2B excludes noise shunting.
  S3 retains A–C only: A aggregate 3-seed mean±SD, B actual 3-seed values,
  C 5-seed paired accuracy differences with 20,000-draw bootstrap intervals.
- S7B/C are owner-summary SD; D paired-seed bootstrap.
- S8 displays only the 160 qualified additive runs. Its independently retained
  run-level audit excludes 160 signed-shunting runs.
- S9 is explicitly 120 trained checkpoints, norm-matched relative step1e−5;
  D reports within-rule associations rather than a pooled predictive claim.
- S11A now uses retained cell-bootstrap bars; B is descriptive site/regime
  scatter at unit input-conductance dose; C is a descriptive signed fraction.
- S12 D–F are 10,000-draw target/axon bootstrap; I/J are SEM over20 targets;
  H connects paired target values without an interval.
- S14 adds verified Gram-preconditioning equality, maximum1.151856e−15;
  figure/text/table round consistently to1.15e−15.
- S17B uses descriptive neuron SEM, with six-animal inference only in its
  separate paired contrasts/table.
- S18 retains old A/B/C/E/H/K as new A–F. S19 retains old G/I as new A/B.
  Redundant curves are removed while source sheets remain archived.
- S20 is cell-bootstrap. S21A/C/D are cell-bootstrap; B is descriptive site
  values and within-cell depth means without intervals.
- S22 retains A–H; duplicate I/J removed. All learning panels are the
  major-branch surrogate, not the complete-tree model. F is explicitly
  shuffle-minus-ancestry MSE, so positive favors ancestry.
- S23 retained reconstruction intervals are seed-bootstrap; S24B is SD and
  C/D are paired bootstrap; S25 uses cell-bootstrap source quantities.
- S26 is a same-seed implementation rerun, not new statistical replication.
- S27B uses cell SEM with budget-dependent eligible n; C uses50,000-draw
  cell-bootstrap intervals; new D exposes candidate count/rank/size limits.
- S28A spans D1–D3 and B is D3; S30 contrasts use percentage points;
  S31H explicitly distinguishes the two optimization recipes.
- S32/S33 visual intervals use5,000 resamples; primary encoder numerical
  contrasts use20,000. S34 uses20,000 target resamples.

Remaining provenance limitations are distinguished from plotted uncertainty:

1. Historical S1D/E and former S3D shunting executions cannot be established
   from retained aggregate tables. Corrective config intent is not execution
   evidence. The unresolved series are removed from current artwork and
   excluded from conductance-valid inference, while source rows are preserved.
2. Original individual S3A outcomes were not retained; only reported means and
   SD are available. The current panel therefore does not invent seed dots.
   For other inherited aggregate panels, declared uncertainty can be checked
   against retained tables/plot construction, but unavailable original
   executed configurations or raw outcomes cannot be independently recovered.
3. Exact package locks/checkpoints for some June/historical analyses remain
   unavailable. Current hashes do not retroactively certify those runs.
4. Reexecuted complete-tree oracle checkpoints differ slightly in three of
   130 fits; these discrepancies remain explicitly reported, with no exclusion.
5. Cell/target SEM or bootstrap never substitutes for independent-animal
   replication. New input/morphology sensitivities are descriptive, and no
   endogenous biological plasticity experiment or population-power claim has
   been introduced.

## Compile and layout checks

Temporary main and SI builds succeed. Latest SI build has101 pages with no
undefined references, missing citation keys, duplicate labels, overfull boxes
or oversized floats. All34 figure assets match their automatic numbering.
S29 caption was shortened to resolve its overflow, and the claim-boundary
table was consolidated to fit. Main cohort table, final S34 page and
claim-boundary table were visually inspected. Root was notified of a small
main6 float overflow and is reducing that figure width before its final build.

No new biological measurement is possible in this workspace. The complete
paper retains that inferential boundary rather than claiming the computational
experiments resolve it.
