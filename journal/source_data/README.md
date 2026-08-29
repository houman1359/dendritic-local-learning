# Source-data provenance

The active GitHub branch tracks this inventory and its provenance manifest but
does not track the bulky result tables, because the repository is synchronized
directly to Overleaf. Existing local tables remain in place as ignored files.
The complete tracked snapshot is permanently recoverable from
`archive/pre-overleaf-prune-20260820`; release and submission archives should
be built from that evidence snapshot rather than from Overleaf.

`provenance_manifest.tsv` is the machine-readable map from each registered
main or supplementary figure panel to its frozen source data, generating
analysis, and replication unit.
Paths are relative to the root of the `dendritic-modeling` repository. SHA256
digests pin every entry whose status is `ready`.

The manifest separates two record types:

- `panel_source` identifies the table or artifact from which a plotted panel
  and its reported statistics are derived.
- `figure_asset` tracks whether the submitted PDF itself is the final audited
  rendering. A source table can be ready while its journal-style rendering is
  still pending.
- `generator_snapshot` pins an unchanged inherited plotting script to the
  exact version used for the reproduced NeurIPS/arXiv panels.

Allowed statuses are `ready`, `pending`, `pending_analysis`, and
`pending_rerender`. Pending records remain warnings during active development
and become errors under the submission gate.

## Running the audit

From `drafts/dendritic-local-learning/journal`:

```bash
python scripts/audit_submission.py
python scripts/audit_submission.py --strict-pending
```

The audit checks:

1. manifest schema, unique entries, source existence, and SHA256 identity;
2. existence of every graphic included by `main.tex`;
3. panel-level provenance coverage and correct figure-asset assignment for
   every numbered main figure;
4. absence of the withdrawn pooled error-field values from manuscript text;
5. embedded fonts and the absence of Type 3 fonts when `pdffonts` is present.

The development audit can therefore report known unfinished work without
confusing it with missing or silently changed evidence. A release is ready
only when the strict audit exits successfully.

## Updating a frozen source

Do not silently replace a source file while retaining its old digest. Rerun the
analysis, inspect the resulting statistics and figure, then update the source
path and digest together. A digest can be obtained with:

```bash
sha256sum path/to/source.csv
```

For hierarchical datasets, the manifest names the highest independent unit
(cell, target cell, or training seed). Sites, stimulus splits, and Monte Carlo
streams remain nested observations and are not counted as independent
biological replicates.
`inferential_units.csv` provides the corresponding manuscript-wide audit of
headline endpoints, independent sample sizes, nested observations and the
uncertainty or paired test used for each evidence family.

The text-level source files include the separate five-seed path-gain diagnostic
cohort, with run/configuration identifiers and checkpoint hashes, and the
deterministic 10,000-trial validation of the projected-gradient capture bound.

## Prospective regular-tree evidence

`prospective_learning/seed_outcomes.csv` retains all 640 historical audited
training runs from the depth-by-feedback factorial. Publication panels and
inference instead use the 480 input-valid rows and companion summaries under
`prospective_input_validity/`. The same directory contains all 2,400 retained
diagnostic rows from 120 input-valid matched backpropagation checkpoints,
together with checkpoint-clustered association intervals and primary-step
feedback summaries for Supplementary Figure S9. Only non-somatic dendritic
stages and their trainable branch parameters enter the reported geometry.

`prospective_routing_control/` contains all 160 historical audited runs from
the separately frozen correct-versus-deranged routing cohort. The 120 input-
valid rows and six paired contrasts in `prospective_input_validity/` supply
Figure 3D and the matched-routing Supplementary table. Correct and deranged
conditions have the same teaching-coordinate count and per-example value
distribution; only the coordinate-to-tree map differs.

`prospective_followup/` contains the complete 1,200-run historical audit output
for matched-bandwidth routing, inhibitory dose, fixed spatial topology and
fixed-budget depth. It is a provenance archive, not the publication analysis.
The outcome-independent validity ledger excludes the full inhibitory-dose
family and supplies 120 routing, 240 spatial and 160 additive fixed-budget
runs. The fixed-budget rows supply Figure 3F and Supplementary Figure S8; they
hold 16 terminal branches and 960--968 active contacts per soma while varying
nominal depth. The spatial rows remain a boundary control because their effect
persists under backpropagation and on the randomly projected task.

`clean_exact_bp/` contains the detached tracked-clean 320-run comparison that
replaces the excluded dose panel. It supplies 160 seed-matched exact-transport
versus backpropagation pairs across two tasks, two cores and four depths. The
synthetic shunting condition uses an explicit non-negative transfer before
positive conductances are formed. All 320 artifacts and finite stage-complete
checkpoints passed. Every depth-averaged task--core interval includes zero;
the mean exact-minus-backpropagation difference across 160 raw pairs is
-0.0545 percentage points, with a 5.90-point maximum individual discrepancy.
The intended reading is empirical agreement, not formal equivalence.

`trained_subtree_address/` contains the completed frozen phase-1 trained
within-neuron address test shown in Figure 3G--I: 80 runs from ten paired
confirmatory seeds and eight feedback/resource conditions. The seed outcomes,
gradient audit, paired contrasts, condition summary and mechanism ledger are
all retained. The phase tests the decisive correct-versus-within-neuron-
deranged contrast at $K=2$ on one shallow forward architecture. It does not
by itself establish the broader bandwidth result.

`trained_subtree_address_full_factorial/` contains the completed 2,700-fit
confirmatory extension: 20 paired seeds crossing five parameter-matched
representations, four route budgets and the prespecified route and reference
families. It supplies the bandwidth-dependent structured-address result in the
main article. Ancestry loses to the best matched non-anatomical control at
$K=1,2$, wins modestly at $K=4$, and ties at full rank; gated-point and
dendritic implementations coincide when given identical fields.

`credit_phase_theory/` contains the frozen 50-seed, 10,800-row stochastic
quadratic phase experiment. Separate seed-level and summary tables record the
spectral task--tree alignment, hierarchy-depth, projection-denoising and
branch-reliability screens, together with the same-span diagnostic and primary
paired contrasts. `spectral_alignment_thresholds.csv` and
`spectral_bound_summary.csv` provide the deterministic Ky--Fan and affine
alignment-threshold reanalysis of those frozen spectral rows. The explicit point gate and reliability-aligned condition
are exactly equal. `credit_phase_existing/` is a secondary initialization
reanalysis of the already completed 2,700-fit factorial; it estimates gradient
noise from 64 minibatches per seed and reports the signal--noise--curvature
utility. Neither directory is evidence that a local rule uniformly exceeds
exact full-batch backpropagation or that a positive-conductance dendrite learns
oracle branch reliabilities.

`nonlinear_physical_depth_canary/`, `nonlinear_physical_depth_accessibility/`,
`nonlinear_physical_depth_coupling/`, `nonlinear_physical_depth_signal/` and
`nonlinear_physical_depth_boundary/` retain the complete calibration path from
the reviewer-requested bridge to the original positive-rate population
implementation. The first severe-shift canary is negative: deeper cells fit
training better, but every morphology remains at chance on the shifted test
distribution. The later ladders transparently locate the accessible,
non-ceiling operating point; none of their seeds enters inference.

`nonlinear_physical_depth_confirmatory/` contains the publication-facing
270-fit exact-resource cohort on fresh paired seeds 10200--10209. It includes
seed outcomes, condition summaries, paired contrasts, artifact/claim gates and
the six-panel main figure, together with a 30-checkpoint state and gradient
replay. Exact path LocalCA reproduces the selected autograd gradient direction;
shared-soma alignment falls with physical depth. The first cyclic placement
execution remains archived but is excluded because it changed candidate mask
slots. The width-preserving reversed-placement replacement is the only tree
control entering inference. This selected $H=3$ bridge is not the full
$D\times H\times K\times\rho$ programme and does not establish a uniquely
dendritic advantage over a grouped point model supplied with the same states.

`positive_conductance_reliability_step_consistent/` contains the prospectively
corrected state-matched conductance experiment: 2,250 seed--condition rows
across 50 fresh paired confirmatory seeds, five branch-SNR heterogeneity levels
and nine controls. Gains use the optimum for the actual half-smoothness step;
the earlier `positive_conductance_reliability/` directory remains a superseded
pilot archive. All rates and conductances are nonnegative. Aligned shunts
improve the immediate step and final loss relative to the step-consistent best
global shunt, but have no reliable final-loss difference from the unshunted
noisy rule. Independently computed point and conductance updates agree to
numerical precision. Shunts are fixed initial oracles and an oracle current
preserves voltage; this is not autonomous shunt learning.

`same_span_coefficient_learning/` contains 4,800 checkpoint rows from 50
paired seeds comparing orthonormal, raw nested and statically scaled nested
coordinates of the same rank-eight route span. It includes the preregistered
contrasts, exact finite-time bias--variance predictions and sample-size
crossover analysis. Slow ill-conditioned coordinates filter noise at the
smallest effective sample size, whereas orthonormal coordinates reduce bias
and win at high sample size. Gram-preconditioned controls are exactly
span-equivalent; they are an oracle coordinate control, not a proposed
biological mechanism.

`spatial_topology_audit/` quantifies the input-coverage difference between the
registered random and spatial fixed maps across ten topology seeds and 128
neurons per seed. It supplies Supplementary Figure S7 and the spatial-coverage
Supplementary table. Both maps
have 336 contacts per neuron; the spatial partition covers more unique input
coordinates and eliminates cross-branch collisions.

The source-equivalence record is in
`../analysis/prospective_source_equivalence.json`. It verifies the four exact
post-freeze source differences across 72 run families and 1,900 resolved
configurations and shows that all four additions were inactive for this
experiment at collection time. It is a dated attestation tied to the recorded
hashes; frozen manifests and archived source copies define these historical
runs after the development checkout advances.

## Expanded regular-tree evidence

`inherited_neurips/` contains the frozen numerical inputs for Supplementary
Figures S1--S3. Those figure assets and their archived generators are
byte-identical to the final NeurIPS/arXiv versions. The journal does not redraw
them. Supplementary Figure S4 is an expanded reorganization of the validated
regular-tree evidence and is explicitly distinguished from the literal
reproductions.

`figure2/feedback_learning_relevance_runs.csv` contains the earlier
ten-checkpoint audit linking an available error field to eligibility-weighted
gradient capture and norm-matched one-step loss decrease. It is retained as a
supporting provenance record and is superseded in the main argument by the
prospective 120-checkpoint input-valid analysis above.

`regular_tree_regimes/` contains the task, inhibition-dose, depth,
broadcast-noise, rule-family, error-source, mechanism-control, feedback-rank
and flattened CIFAR-10 source tables inherited from the original arXiv/NeurIPS
study. `scripts/export_regular_tree_source_data.py` verifies every upstream
file by SHA-256 before exporting it and removes machine-local run paths. These
analyses are presented as regime and mechanism controls, not as evidence of a
universal shunting or benchmark advantage.

## Focal-shunt physical and selectivity matrices

`physical_cable_sensitivity/` contains the earlier relative-dose calibration
across axial and membrane-resistance settings in the original and disjoint
v661 cohorts. `focal_selectivity_phase1/` contains the newly completed frozen
eight-cell passive matrix for Supplementary Figure S11. It crosses three
membrane resistances, three distributed-background levels, and relative-local,
fixed-absolute-nS, and input-conductance-normalized doses. The compressed site
table has 16,362 intervention rows; cell summaries, shunt-minus-additive
contrasts, signed attenuation/enhancement/sign-flip endpoints, local input
resistance, transport selectivity, source hashes and numerical gates are
provided separately. `focal_selectivity_active_ensemble/` contains all 512
accepted Na/K/Ca/HCN/NMDA steady-state equilibria, site-level outcomes and
eight cell-level contrasts used in Figure 8.

## Public-v661 MICrONS replication

`microns_v661_replication/` contains a frozen, 47-cell sensitivity analysis of the
structural routing and focal-perturbation analyses. The cells are disjoint from
the original eight-cell pilot by stable nucleus ID. They are not independent
animals: both cohorts come from the same MICrONS mouse. The replication uses
public version-661 SWCs, meshworks, and direct presynaptic coarse E/I calls; it
does not use a CAVE token or the spine/shaft classification proxy.

The publication-facing files are:

- `cohort_manifest.csv`: all 55 candidates, eight original-pilot exclusions,
  exact public URLs and file hashes, and focal eligibility;
- `cell_level_primary.csv`: eight-channel routing capture and cell-averaged
  focal contrasts;
- `replication_summary.json`: estimands, cell-level uncertainty, numerical
  validation, provenance, and limitations;
- `routing_capacity_20stream.csv.gz` and
  `routing_capacity_20stream_summary.json`: frozen cell-by-stream results and
  inferential summary from 20 independent perturbation streams;
- `supp_figure_routing_curves.csv`: cell-level plotting curves after averaging
  those streams within each cell and condition;
- `local_current_cache_inventory.csv`: why the pre-existing local cache could
  not provide a larger current-materialization cohort.

The associated audit is
`../analysis/microns_v661_replication_report.md`. Rebuild from the repository
root with:

```bash
python drafts/dendritic-local-learning/journal/scripts/build_static_microns_replication.py

cd drafts/dendritic-credit-routing
python analysis/analyze_microns_morphology_credit.py \
  --datadir ../local-learning-journal/data/microns_v661_replication \
  --outdir ../local-learning-journal/source_data/microns_v661_replication/routing \
  --classification-mode typed_only --n-shuffles 500 --seed 20260731
python analysis/run_focal_shunting_credit_perturbation.py \
  --segments ../local-learning-journal/source_data/microns_v661_replication/routing/segment_metrics.csv \
  --outdir ../local-learning-journal/source_data/microns_v661_replication/focal \
  --seed 20260731

cd ../..
python drafts/dendritic-local-learning/journal/scripts/summarize_static_microns_replication.py
```

The raw normalized SWC and per-cell synapse files are intentionally ignored by
Git. Their versioned URLs, byte sizes, and SHA-256 hashes are preserved in the
source-data manifest so an archive can be rebuilt exactly.

The main replication figure is generated with:

```bash
python drafts/dendritic-local-learning/journal/scripts/build_microns_v661_replication_figure.py
```

It reads the frozen 20-stream capacity files in this source-data directory and writes
`figures/fig_v661_robustness.{pdf,png}`, processed plotting
tables, a TeX-ready caption and Results paragraph, and
`prospective_endpoint_exclusion_log.csv`. The latter is a long-form,
machine-readable record of inclusion and exclusion for structural routing,
the shunt-versus-additive contrast, and the depth-shuffled relation control.

## Expanded inhibitory census

`microns_inhibitory_routes/` contains the complete stable-ID overlap between
the 47-cell sensitivity cohort and the published v795 inhibitory census. The
20 target cells are nested within the same MICrONS mouse. Publication-facing
files include:

- `cohort_overlap.csv`, the frozen stable-cell-ID join;
- `coordinate_transform.csv`, the audited raw-voxel to slanted-micrometre map;
- `mapped_inhibitory_contacts.csv`, all mapped known inhibitory contacts;
- `inhibitory_connection_topology.csv`, connection-level matched topology and
  descendant-domain outcomes;
- `inhibitory_axon_metrics.csv`, the same endpoints aggregated at the
  presynaptic-axon level;
- `inhibitory_connection_topology_3d_matched.csv` and
  `inhibitory_3d_matched_target_metrics.csv`, joint path-distance and 3D
  proximity controls;
- `target_metrics.csv`, the target-cell inferential table;
- `route_capture_curves.csv`, actual-site capacity and structural controls; and
- `summary.json`, frozen tests, intervals, mapping rates, and limitations.

Rebuild with `python scripts/analyze_microns_inhibitory_routes.py`. Raw census
files are held under ignored `external_data/`; their URLs and SHA-256 values
are recorded in `reproducibility/external_expansion_manifest.tsv`.

## External animal and interference analyses

`animal_learning_francioni/` contains extracted animal-level contrasts,
orthogonal common/signed modes, descriptive neuron-level distributions, and a
machine-readable statistical summary. The public workbook is checksum-pinned
and is not inferred from figure pixels. Rebuild with
`scripts/analyze_francioni_signed_credit.py`.

`branch_interference/` contains the exact 101 by 101 route-overlap and leakage
surface and its explicit numerical verification. It is a static quadratic
identity, not a temporal neural simulation. Rebuild with
`scripts/analyze_branch_credit_interference.py`.

## Point--dendrite and BP--local-credit controls

`point_dendrite_credit_controls/` contains the seed-level outcomes, condition
summary, paired contrasts and machine-readable audit for the same-task control
ladder added to the nonlinear physical-depth cohort. The primary extension
contains 140 fits: resource-identical serial-tree versus grouped-star models,
active- and total-parameter-matched point MLPs, and an autograd
soma-broadcast credit restriction. A transparently post-outcome amendment adds
60 soma-broadcast fits using the frozen LocalCA optimizer groups, because the
first broadcast arm inherited the standard-BP optimizer and therefore could
not isolate the update rule in a broadcast--LocalCA comparison. The dated
contract and amendment are in
`../analysis/POINT_DENDRITE_CREDIT_CONTROL_CONTRACT_20260813.md`.

Rebuild all summaries and `fig_point_dendrite_credit_controls.{pdf,png}` with:

```bash
python scripts/analyze_point_dendrite_credit_controls.py
```

The audit requires 200 finite new fits, no fallback warning, exact equality of
the serial/star resource counts, and precisely the paired seeds 10200--10209.

## Physical-depth alignment dose response

`physical_alignment_dose/` combines the unchanged $\alpha=0$ and $1$
physical-depth endpoints with 90 new fits at $\alpha=0.25,0.50,0.75$, crossed
with D1--D3 and ten paired seeds. It contains all new and combined seed
outcomes, condition summaries, paired depth-effect and dose contrasts, a
machine-readable audit and the concise result report. Rebuild the tables and
`fig_physical_alignment_dose.{pdf,png}` with:

```bash
python scripts/analyze_physical_alignment_dose.py
```

The two source manifests are intentional: a frozen-source guard stopped the
remaining first-array tasks after a concurrent edit, and only the stopped
conditions were rerun. The code difference was an optional rule override that
no dose-response configuration enabled. The exact audit is documented in
`../analysis/MID_ARRAY_SOURCE_EQUIVALENCE_20260813.md`.

## Literal grouped-point and second-hierarchy experiment

`remaining_physical_experiments/` contains the complete frozen 220-fit
extension: 60 H3 literal grouped-point fits on the existing paired seeds and
160 fresh H2 fits crossing serial BP, grouped-point BP, shared-soma LocalCA,
exact-path LocalCA, aligned placement and reversal. The combined table retains
the unchanged H3 serial and grouped-star references needed for paired
comparisons. The folder includes every run-level outcome, condition summary,
paired contrast, seed-level contrast vector, frozen-manifest hash and the
machine-readable completeness/resource audit.

Rebuild the tables and `fig_remaining_physical_crossovers.{pdf,png}` with:

```bash
python scripts/analyze_remaining_physical_experiments.py
```

The audit requires 220/220 finite fits, no fallback or nonfinite alert, the
frozen seed sets and exact resource equality. The dated contract is
`../configs/remaining_physical_experiments/CONTRACT.md`.

## Fashion-MNIST feedback ladder

`fashion_feedback_ladder/` contains the 60-fit second-dataset replication of
the scalar, neuron-indexed and exact-path feedback ladder in shunting and
additive regular trees. It includes seed-level outcomes, condition summaries,
paired contrasts, the preregistered decision and source-manifest audit. Rebuild
the tables and `fig_fashion_feedback_ladder.{pdf,png}` with:

```bash
python scripts/analyze_fashion_feedback_ladder.py
```

Neuron-indexed feedback improves over scalar fallback by 4.46 points in
shunting trees and 3.78 points in additive trees (10/10 positive seeds each).
Exact path transport adds no reliable benefit beyond neuron identity. The
multiple manifests reflect the same source-guard event documented for the
alignment dose response; no Fashion-MNIST configuration enabled the changed
optional rule override.

## Trained partition-residual reconstruction

`trained_partition_residual/` contains the hash-gated deterministic
reconstruction of all 2,700 trained route-factorial fits at initialization and
training. It reports address and coefficient residuals, endpoint agreement and
within-seed capture--accuracy associations. Rebuild with
`python scripts/analyze_trained_partition_residual.py`.

## Fashion-MNIST path-demand boundary

`path_necessity_fashion/` contains all 2,400 frozen confirmatory outcomes for
the branch-conflict experiment, the task and gradient audits, paired route
contrasts, seed-wise interaction slopes and analytic-versus-trained boundary
summaries. Rebuild the compact tables from the project-B raw outputs with
`python scripts/run_path_necessity_fashion.py --phase aggregate`, summarize the
discrete transition with `python scripts/analyze_path_necessity_boundary.py`,
and regenerate Supplementary Figure S29 with
`python scripts/build_path_necessity_fashion_figure.py`.

The frozen configuration labels conflict probability `alpha`; manuscript and
figure notation uses \(\chi\) for the same dose so that \(\alpha\) remains
available for the separate physical-depth alignment parameter. The executed
table's `simultaneously_driven_branches_per_example` column means that every
branch receives a nonzero input view; `active_forward_branches_per_example=1`
records that only the context-selected branch contributes to the logit.

The experiment shows when branch-selective information is useful. It does not
show that dendritic material is uniquely required: analytic backpropagation and
the matched gated-point implementation coincide with correct routing.

## Adaptive conductance reliability

`adaptive_conductance_reliability/` contains all 1,050 outcomes from the fresh
50-seed adaptive local reliability experiment, branch-level estimates,
condition summaries, paired contrasts and numerical gates. Rebuild the display
with `python scripts/build_adaptive_conductance_reliability_figure.py`.

## Irregular-tree wavelet scale analysis

`irregular_tree_wavelets/` contains the frozen weighted tree-Haar analysis for
the 47-cell public-v661 cohort and original eight cells. It includes every
mode's support and energy, cell- and cohort-level summaries, numerical audits
and the inferential summary. Rebuild tables with
`python scripts/analyze_irregular_tree_wavelets.py` and Supplementary Figure
S25 with `python scripts/build_irregular_tree_wavelet_figure.py`.

## H4 physical-depth factorial

`physical_depth_h4_factorial/` contains all 360 intended seed--condition
outcomes, summaries, seed-paired contrasts and the immutable-source/resource
audit for the H4 saturation test in Figure 6A--D. The audit explicitly records
the 90 D3 rows rerun after the original configurations failed before model
construction. Rebuild with:

```bash
python scripts/analyze_physical_depth_h4_factorial.py \
  --runs-root <original-H4-runs> \
  --d3-repair-runs-root <D3-repair-runs>
```

## Physical-depth clean-source replication

`physical_depth_clean_source_replication/` contains all 430 immutable-source
same-seed H2/H3 reruns, historical concordance pairs, recomputed seed-level
contrasts and the complete audit for Supplementary Figure S26. Rebuild with:

```bash
python scripts/analyze_physical_depth_clean_source_replication.py \
  --remaining-runs-root <clean-H2-and-point-runs> \
  --confirmatory-runs-root <clean-H3-runs>
```

The reruns test implementation/source concordance, not independent scientific
replication; no outlier is removed from the paired distribution.
