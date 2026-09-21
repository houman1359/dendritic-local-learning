# Experiment contract

> **Historical frozen contract.** Its executions remain auditable, but the
> 10 August 2026 outcome-independent input-validity audit excludes signed
> synthetic inputs driving the positive-conductance shunting denominator.
> Consequently, no inhibitory-dose interaction from this contract enters the
> paper. Current publication accounting is in
> `source_data/prospective_input_validity/summary.json`.

## Central hypothesis

A dendritic tree is a constrained decoder for learning signals. Its ancestry
defines which synapses can receive a shared teaching coordinate; conductance
sets the gain of those routes. Topology is useful only when this routing basis
aligns with task-derived credit.

## Evidence retained in the journal draft

### Exact theory and implementation checks

1. Exact gradients in a directed steady-state conductance tree factor into a
   synapse-local eligibility term and a compartment error transported from the
   soma along the unique dendritic path.
2. The additive comparison is a raw current-based architecture, not a
   conductance model. A standalone NumPy reference reproduces its parameter
   derivative and child-error transport with maximum finite-difference errors
   of `9.264e-11` and `5.623e-12`, respectively.

### Regular artificial trees

3. In regular `[3,3]` trees, replacing scalar fallback with one neuron-indexed
   coordinate shared across the 13 compartments of that neuron's tree raises
   three-factor MNIST accuracy in all 15 paired seeds in both shunting and
   additive architectures. The plotted cohort is a clean current-code rerun
   with complete configurations, checkpoints, logs, and output hashes for all
   60 runs.
4. Exact transported errors let the theorem-derived three-factor eligibility
   reach 0.97178 mean MNIST accuracy across five seeds, compared with 0.97026
   for matched backpropagation. This isolates feedback construction from the
   optional four- and five-factor empirical preconditioners; it is not a claim
   of general superiority to backpropagation.
5. Clean replacement arrays for the 15-seed feedback comparison completed on
   31 July 2026 as Slurm jobs 36616195 and 36616196. They passed the frozen
   completeness, scientific-configuration, checkpoint, and provenance checks
   and prospectively replaced the mixed archive irrespective of outcome.
6. A fixed-checkpoint audit connects the feedback hierarchy to actual local
   loss change. Scalar, neuron-indexed, and exact feedback produce ordered
   norm-matched one-step progress in all five checkpoints per architecture.
   The diagnostic averages frozen training, validation, and test batches within
   checkpoint and does not update or select parameters.
7. Archived task, depth, broadcast-noise, feedback-rank, rule, error-source and
   flattened CIFAR-10 controls are retained as regime evidence. They support a
   feedback-quality and robustness account, not a universal architecture or
   state-of-the-art performance claim.
8. A frozen bandwidth-matched control separates coordinate count from correct
   assignment. Across two tasks, two neuron models, and depths two and four,
   correct neuron-to-tree routing improves held-out accuracy by 0.15--0.72
   percentage points over a fixed derangement. Seventy-eight of 80 paired seeds
   favor the correct map. The result supports a smaller assignment-specific
   contribution; it does not attribute the much larger scalar-to-neuron-indexed
   gap entirely to topology.
8a. A frozen 400-run inhibitory-dose cohort tests the noise task at five
    inhibitory-contact counts, two neuron models, four teaching fields, and ten
    paired seeds. From zero to 40 inhibitory contacts, the
    shunting-minus-additive interaction is +22.11 percentage points under
    scalar feedback and between -0.33 and -1.10 points under neuron-indexed,
    exact, or backpropagated credit. Dose changes the complete trained operating
    point; this is a conditional regime result, not a backward-only intervention.
8b. A frozen 320-run fixed-topology cohort compares spatial and random sparse
    maps across two tasks, two neuron models and four teaching fields. The
    spatial map improves learning even under backpropagation and on the randomly
    projected task. At the same 336 contacts per neuron it covers 336 unique
    coordinates, versus 276.2 for independently sampled random branches. This
    is retained as a forward-connectivity boundary control and is not used as
    evidence of task-aligned credit routing.
8c. A separately frozen 320-run depth cohort holds 16 terminal branches and
    960--968 active input contacts per soma. Depth-four-minus-depth-one
    accuracy is negative in all ten paired seeds for every architecture and
    feedback field. The decrease is largest under scalar feedback ($-7.57$
    points in shunting and $-7.92$ in additive networks) and is $-0.55$ to
    $-1.37$ points with exact transport or backpropagation. This is a
    fixed-leaf/contact boundary, not a trainable-parameter match: deeper trees
    retain more compartments and coupling and reactivation parameters.

### Reconstructed anatomy and focal perturbation

9. Across the original eight reconstructed MICrONS trees, eight
   morphology-selected routes capture 0.487 of the weighted energy in modeled
   route-perturbation fields with 6.92% of dense-feedback wiring. They exceed
   random paths, depth bins, and ancestry shuffles in every
   cell. The tested columns are drawn from the generator of these fields, so
   this is explicitly a model-matched compression diagnostic rather than an
   independent routing test.
10. A frozen sensitivity cohort excludes the original eight cells by stable
   nucleus identifier and uses 47 public MICrONS minnie65 v661
   reconstructions. Averaging 20 independently generated perturbation streams
   within cell, morphology routes capture 0.756 of the modeled energy with
   8.08% of dense wiring, compared with 0.302 for random paths, 0.441 for depth
   bins, and 0.330 for ancestry shuffles. All 47 paired
   morphology contrasts are positive. This is a same-mouse, historical-release
   sensitivity analysis, not an independent-animal or population-random
   replication.
11. In an independent-generator control, ancestry dictionaries reconstruct
    exact reciprocal-cable focal-shunt response operators rather than fields
    sampled from the tested kernel. At eight channels, morphology capture is
    0.458 versus 0.363 for random real paths and 0.249 for row-shuffled paths,
    with both paired contrasts positive in 8/8 cells. Morphology is only 0.014
    above depth bins and 0.035 above degree- and depth-matched surrogate trees;
    neither paired ordering is reliable. This bounds the evidence for fine
    topology beyond coarse depth and degree.
12. Across 101 prospectively selected sites in the original eight trees, focal
   shunting in a reciprocal passive model produces more descendant-localized
   exact-gradient changes than a first-order-current-matched additive
   perturbation while somatic voltage and output error are held fixed. The
   mean cell-level localization difference is 0.0694 (95% cell-bootstrap
   interval 0.0533--0.0856; 8/8 cells positive).
13. In the v661 cohort, the shunt-minus-additive localization difference is
   0.1377 (95% interval 0.1165--0.1621; 45/45 cells positive; 235 sites
   nested within cells). True descendant relations exceed depth-matched
   shuffled relations by 0.1176 (95% interval 0.0935--0.1442; 39 positive and
   one numerical tie among 40 cells; 230 sites).
14. Holding the post-shunt voltage and driving-force field fixed, restoring
    the post-shunt adjoint raises localization by 0.0786, positive in 8/8
    cells. This exact factor-freeze contrast isolates transport within the
    modeled shunt. A two-factor Shapley allocation of the prespecified
    shunt-versus-additive contrast assigns 0.0868 (95% interval
    0.0630--0.1109; 8/8 cells positive) to the adjoint change and -0.0174
    (-0.0293 to -0.0091; 8/8 negative) to the driving-force change. The factor
    substitutions are algebraic controls, not independent biological
    interventions.
15. Physical passive-cable calibration shows that focal localization is
    electrotonic-regime dependent. At $R_a=150\,\Omega\,\mathrm{cm}$ and
    $R_m=15{,}000\,\Omega\,\mathrm{cm}^2$, the contrast is -0.0019 in the
    original eight and 0.0024 in 45 v661 cells; at $R_m=1{,}000$ and 300 it is
    0.0164 and 0.0674 in the original eight. Specific resistances are
    sensitivity settings and synaptic area is only relatively calibrated.

16. The inhibitory-census primary analysis is a target-level spatial
    association, not evidence of a teaching route. The tree-distance contrast
    survives aggregation within 92 presynaptic axons, while shared-path and
    descendant-domain overlap do not. No endpoint survives joint path-distance
    and 3D matching. The actual-site capacity endpoint reconstructs its own
    route dictionary and is interpreted only as internal compressibility.

### Measured responses and controlled task alignment

17. In the seven-target measured-response cohort, morphology-selected routes
    do not reliably outperform ancestry shuffling in held-out task-credit
    capture or projected learning. At four channels the mean capture difference
    is 0.0282 (95% interval -0.1464--0.2061), and the normalized-MSE difference
    is -0.0043 (95% interval -0.0273--0.0211). This is a boundary result, not
    evidence that topology is irrelevant.
18. The functional cohort contains seven targets, 69 mapped presynaptic
    partners, 27 manual matches, and 356 within-target partner pairs. Each
    target contributes 464 trials, 280 unique stimuli, and 136 repeated
    conditions. The eighth structural target is excluded because its only
    archived trial-manifest row lacks a DANDI asset identifier and path.
19. In a controlled quadratic experiment on the original eight reconstructed
    trees, rotating exact credit into or out of the fixed morphology-selected
    span reverses the route advantage. Across the non-morphology controls, the
    median within-cell Spearman relation between initial capture and learning
    progress is 0.990 (range 0.969--0.996). Alignment and projection
    coefficients are imposed using exact gradients, so this is a conditional
    sufficiency test, not independent biological evidence.

## Frozen selection and provenance rules

- The original eight anatomy cells are a hand-authored, deliberately small,
  layer-diverse pilot from an existing anatomy/functional-coregistration
  cohort. They are not represented as a population-random or coverage-ranked
  sample.
- The public-v661 selection universe contains 55 cells. The eight original
  pilot nuclei are excluded before endpoints are summarized, leaving 47
  disjoint cells. All candidates and endpoint-specific exclusions are recorded
  in machine-readable manifests.
- The v661 cohort comes from the same MICrONS mouse and is imbalanced by class
  (34 L2IT, 2 L3IT, 10 L4IT, and 1 L5ET). Direct presynaptic E/I calls cover
  4.71% of incoming synapses. These limitations travel with every v661 claim.
- For functional targets with several eligible scans, the archive does not
  contain a deterministic historical scan-selection rule. The seven selected
  target/scan pairs are therefore treated as frozen pilot assets, not as the
  output of a newly inferred rule.
- Source tables are replaced only after rerunning the frozen analysis,
  checking the intended inferential unit, and updating the source path and
  hash together. New scheduler submissions do not alter current evidence.

## Claims that are excluded

- Shunting is not claimed to improve local learning universally.
- Low path-gain dispersion is not treated as evidence of better gradient
  direction.
- Pooled error-field metrics that include an exact-by-construction somatic
  stage are not used for cross-architecture claims.
- MICrONS anatomy is not described as evidence that the mouse used the modeled
  rule during learning.
- Measured presynaptic visual responses are not described as teaching signals.
- The v661 cohort is not described as an independent biological replication.
- Algebraic focal-factor substitutions are not described as distinct
  biological perturbations.
- Oracle projections are not described as implementable local feedback rules.
- The method is not claimed to outperform backpropagation in accuracy, memory,
  latency, or large-scale benchmark performance.

## Statistical units and decision rules

- Reconstructed-cell analyses use the cell as the biological unit. Sites and
  Monte Carlo streams are nested samples, not independent cells.
- Measured-response analyses use the postsynaptic target cell as the unit.
- Artificial-network learning analyses use independently initialized training
  seeds as the unit.
- Exact identities and implementation checks report numerical error rather
  than inferential tests.
- Central paired cell comparisons report a paired mean effect, a 95%
  cell-bootstrap interval, positive-pair counts, and an exact two-sided
  Wilcoxon signed-rank test when applicable.
- Secondary parameter grids, channel sweeps, and reliability thresholds are
  labeled sensitivity analyses. No family-wise multiplicity claim is made.

The claim-level source paths, stable identifiers, and unresolved gates are in
`EVIDENCE_LEDGER.md`.
