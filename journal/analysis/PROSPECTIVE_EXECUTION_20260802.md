# Prospective execution record

> **Historical execution record, superseded for inference on 10 August
> 2026.** Artifact completion remains valid. Publication inference follows the
> later outcome-independent input-validity ledger: 480/640 central runs,
> 120/160 routing runs, 240/320 spatial runs, 160/320 fixed-budget runs and
> 120/160 checkpoints are retained; the full 400-run inhibitory-dose family is
> excluded.

Date: 2026-08-02

## Primary canary

All 32 frozen canary runs passed the artifact audit. There were no failed or
incomplete runs, non-finite values, transport-shape fallbacks, missing
checkpoints, missing dendritic stages, or degenerate learned couplings. Canary
results are a software and stability check only and are excluded from all
scientific summaries.

Slurm arrays:

- 36807995: MNIST, additive, backpropagation
- 36808011: MNIST, additive, local three-factor rule
- 36808025: MNIST, shunting, backpropagation
- 36808035: MNIST, shunting, local three-factor rule
- 36808046: noise resilience, additive, backpropagation
- 36808058: noise resilience, additive, local three-factor rule
- 36808075: noise resilience, shunting, backpropagation
- 36808082: noise resilience, shunting, local three-factor rule

The machine-readable audit is
`analysis/prospective_primary_canary_audit.csv`.

## Primary confirmatory cohort

The complete 640-run cohort was released only after the canary gate passed.
The arrays are capped at 12 simultaneously running GPU tasks in total: one per
backpropagation array and two per local-learning array.

- 36813018: MNIST, additive, backpropagation (40 runs)
- 36813021: MNIST, additive, local three-factor rule (120 runs)
- 36813033: MNIST, shunting, backpropagation (40 runs)
- 36813038: MNIST, shunting, local three-factor rule (120 runs)
- 36813040: noise resilience, additive, backpropagation (40 runs)
- 36813043: noise resilience, additive, local three-factor rule (120 runs)
- 36813046: noise resilience, shunting, backpropagation (40 runs)
- 36813050: noise resilience, shunting, local three-factor rule (120 runs)

Each array uses seeds 42--51. Results will enter the statistical analysis only
after all 640 records pass `audit_prospective_learning_runs.py`.

An audit on 3 August 2026 found 393 passing artifacts and 247 incomplete runs,
with no failed artifacts. It also found that the live repository had changed
after the frozen manifests were written. The differences add recurrent-model
registry entries, stricter type validation for sparse-layer constructor
arguments, and resource-accounting fields; they do not alter the resolved
dendritic configurations or update equations used here. Nevertheless, exact
byte-level source identity no longer holds. Partial results remain excluded,
and the final collector must use a pinned source snapshot or a separately
documented equivalence audit before this cohort can supply manuscript evidence.

CPU job 36829336 is queued after all eight arrays. It runs the strict artifact
and source-identity audit and, only if all 640 runs pass, generates the paired
confirmatory statistics and the depth-by-feedback figure. Partial runs cannot
enter this output.

## Dependent follow-up canaries

The following 12 arrays are queued with an `afterany` dependency on all eight
primary confirmatory arrays. Each is capped at one GPU. They test correct versus
deranged ancestry, inhibitory dose, and fixed random versus image-aligned
topology under local learning.

- 36813129, 36813141, 36813152, 36813153: ancestry-routing controls
- 36813154, 36813165, 36813166, 36813167: inhibition-dose controls
- 36813178, 36813179, 36813180, 36813192: spatial-topology local learning

The four matched spatial-topology backpropagation canaries are queued after the
first follow-up wave so that the study never requests more than 12 additional
GPUs: 36813223, 36813235, 36813246, and 36813247.

No follow-up confirmatory array will be released until all 36 follow-up canary
runs pass their frozen audit.

## Checkpoint-level learning relevance

Array 36813522 is queued after the two follow-up canary waves. It evaluates all
160 matched backpropagation checkpoints from the primary cohort (two tasks,
two neuron models, four depths, and ten seeds). At one frozen test batch per
checkpoint it measures, for the same five feedback families:

- dendritic exact-error capture after optimal rescaling;
- eligibility-weighted branch-gradient capture and cosine;
- whether a norm-matched update is a descent direction; and
- the fraction of exact one-step loss decrease retained at four step sizes.

The array is capped at eight GPUs. A completed MNIST additive checkpoint passed
the diagnostic preflight before submission. The diagnostic is descriptive of
the fixed trained state; it is connected to learning by the measured one-step
loss endpoint rather than by error geometry alone.

CPU job 36813788 is queued after array 36813522. It requires all 160 diagnostic
outputs, all five feedback families, four step sizes, and ten paired seeds per
condition before writing source-data tables or the mechanism report.

## Confirmatory follow-up gate

Before the confirmatory gate, a fixed-budget depth canary is run in two waves.
Its first wave contains eight local-learning arrays and four depth-1/depth-4
backpropagation controls (36814218, 36814219, 36814223, 36814236, 36814237,
36814241, 36814243, 36814246, 36814302, 36814318, 36814321, and 36814322). The
remaining four depth-2/depth-3 backpropagation controls are 36814326--36814329.
These 24 runs compare one- through four-stage trees at 16 leaves and 960--968
active input contacts per soma.

CPU job 36814408 is queued after those canaries. It requires both the 36-run
topology/dose canary and the 24-run fixed-budget-depth canary to pass their
frozen audits. It also verifies that every critical source file still matches
the common frozen source identity. Only then does it submit the 880-run
topology/dose confirmatory cohort followed by the 320-run fixed-budget-depth
cohort, always in waves of at most 12 one-GPU arrays. Final dependent CPU audits
are submitted automatically. If any canary or source check fails, the gate
exits before submitting confirmatory work. Earlier queued gates 36813567 and
36813909 were cancelled before execution as these safeguards were added.

The downstream analysis is implemented in
`scripts/analyze_prospective_followup_results.py`. It requires the complete
880-run topology/dose audit and complete 320-run fixed-budget audit together.
It then produces separate inhibition-dose, task-aligned-topology and
matched-bandwidth-routing, and fixed-budget-depth figures. The analysis fails
closed on missing, duplicated, failed, or unbalanced seed cohorts. The wrapper
`scripts/collect_prospective_followup_results.sh` reruns both strict audits
before invoking this analysis.

## Completion update, 3 August 2026

The primary cohort is complete. A fresh audit found 640 passing runs, no
failed runs, and no incomplete runs. The collector therefore generated the
paired condition summaries and the two prospective main figures. Neuron-indexed
feedback exceeds scalar feedback in all 160 matched seed pairs, and exact
three-factor transport remains within 0.14 percentage points of matched
backpropagation in every task--core--depth condition. The complete results are
in `analysis/prospective_primary_confirmatory_results.md` and
`source_data/prospective_learning/`.

The mechanism diagnostic is also complete: 160 independently trained
backpropagation checkpoints, five field labels, and four step sizes produced
all 3,200 expected rows. At the primary relative step of $10^{-5}$,
gradient cosine predicts retained one-step progress with Spearman
$\rho=0.865$ (checkpoint-clustered 95% bootstrap interval 0.823--0.899).
Neuron-indexed and exact feedback are descent directions in 160/160
checkpoints, compared with 99/160 for a global scalar.

The original gate job 36814408 stopped because live source files no longer
matched their frozen hashes. The follow-up audit
`scripts/audit_prospective_source_equivalence.py` establishes the exact scope
of those differences across the latest 72 run families and 1,900 resolved
configurations. Four exact additions are inactive for the prospective
configurations: recurrent-architecture registration and its spatial-morphology
helper, stricter integer validation whose accepted type is already used, and
an unselected scheduler profile. Removing only these inactive blocks
reproduces the frozen file hashes byte for byte. The machine-readable result is
`analysis/prospective_source_equivalence.json`.

All 36 topology/dose/routing canaries and all 24 fixed-budget-depth canaries
passed. Replacement gate 37048243 completed successfully and submitted the
880-run follow-up cohort, the dependent 320-run fixed-budget cohort, and final
strict audit jobs 37049234 and 37049238. These confirmatory arrays began
training on 3 August 2026. Their partial outputs remain excluded until both
dependent audits complete.

## Follow-up completion update, 4 August 2026

All 880 topology, dose, and routing runs completed and passed the strict
artifact audit. The audit found every expected seed and condition, finite
metrics, complete curves and checkpoints, all dendritic stages, nondegenerate
couplings, and no transport-shape fallback.

The bandwidth-matched routing result is reported separately because it changes
only the coordinate-to-tree assignment. Correct routing exceeded the fixed
derangement in 78/80 paired seeds, by 0.15--0.72 percentage points across the
eight task--core--depth conditions. Seven of eight condition contrasts survive
correction; the additive depth-four noise condition does not.

The 400-run inhibitory-dose family produced a large architecture-by-dose
interaction only under scalar feedback. From zero to 40 inhibitory contacts,
the shunting-minus-additive contrast changed by +22.11 percentage points under
scalar feedback, versus -0.33 with neuron-indexed feedback, -1.10 with exact
transport, and -1.01 under backpropagation. This is interpreted as a
feedback-dependent operating regime because dose changes the complete forward
state.

The 320-run fixed-topology family improved accuracy with the image-aligned map
under every feedback field, including backpropagation, and on both MNIST and
the randomly projected noise task. A subsequent connectivity audit showed that
the spatial partition covers 336 unique coordinates per neuron at 336 active
contacts, compared with 276.2 for independently sampled random branches, and
eliminates cross-branch collisions. It is therefore retained as a forward
sparse-connectivity boundary control rather than task-specific credit-routing
evidence.

## Fixed-budget completion update, 4 August 2026

The separate 320-run fixed-budget-depth cohort completed and passed its strict
artifact and current-source-equivalence audits: 320 passing, no failed or
incomplete runs. The design held 16 terminal branches and 960--968 active input
contacts per soma while varying nominal depth from one to four.

Depth four underperformed depth one in all ten paired seeds for every
architecture and feedback field. The prespecified depth-four-minus-depth-one
effects were $-7.57$ percentage points for shunting and $-7.92$ for additive
networks under scalar feedback; $-0.93$ and $-1.06$ under neuron-indexed
ancestry; and $-0.55$ to $-1.37$ under exact transport or backpropagation. All
eight exact Wilcoxon tests gave $P=0.001953$ and within-study Benjamini--
Hochberg adjusted $P=0.002772$. This supports a depth-stress interpretation:
scalar feedback amplifies the optimization cost of depth. It does not establish
a parameter-matched depth effect because deeper trees retain more compartments
and trainable coupling and reactivation parameters.
