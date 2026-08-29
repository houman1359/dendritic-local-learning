# CIFAR-10 raw-additive feedback ladder: frozen confirmatory contract

Frozen 28 August 2026, before generating or launching the confirmatory sweep and
before any confirmatory outcome exists.

## Question and rationale

The first five-seed CIFAR-10 ladder used the current raw-additive forward
operator but inherited a provisional reactivation and optimizer recipe.  Its
matched-backpropagation mean (38.98%) was therefore not an adequate reference
for deciding whether feedback resolution generalizes beyond MNIST and
Fashion-MNIST.  A completed, outcome-independent calibration screen compared
reactivation policies and two training recipes using matched BP only.  Selection
used validation accuracy, never test accuracy.  The selected current-code cell
was `additive_empirical_archived` (mean validation accuracy 49.96%, three seeds),
and it passed the frozen test-accuracy adequacy gate.
This selection is read from the corrected version-2 post-calibration audit,
whose analyzer SHA-256 is
`e938929556831cdf51323750cdf2e45588c2eaa36a1073552c47deafa27bb255`;
the correction changed gate provenance, not the selection algorithm or winner.

This experiment asks whether the feedback ladder persists when every arm uses
that selected raw-additive operating point.  It is not a comparison between raw
and normalized additive integration, and it is not a test of a generic
shunting advantage.

## Frozen design

- Dataset: CIFAR-10, flattened 3072-dimensional input, no input normalization,
  90:10 train--validation split.
- Model: one 20-cell dendritic population receiving the legacy
  `input_mode=1` direct excitatory and inhibitory input streams; excitatory
  branch factors `[3, 3, 3, 3]`; 25 direct input contacts of each sign per
  branch; identity encoder; MLP decoder `[32, 16, 10]`. The stored
  `inhibitory_layer_sizes`, EI and II fields are inert because
  `input_mode1_build_inhibitory_population` is omitted and therefore defaults
  to false; no explicit recurrent inhibitory-cell population is instantiated.
  The IE field remains active as the direct inhibitory-input contact count.
- Forward operator: current raw additive integration
  (`dendritic_additive`, `use_shunting=false`,
  `use_additive_normalization=false`, `additive_mode=raw`).
- Network initialization: adaptive initialization enabled with
  `preserve_shunting_center`, target conductance 5.0, and initial child
  conductance 1.0.  These settings are retained because they were active in
  the validation-selected BP screen cell.
- Reactivation: `param_tanh`; empirical, per-layer calibration from three
  training batches; requested initial `(m,b)=(1.5,0.5)`; invalid calibration
  must not be silently accepted.
- Optimization: Adam; 200 epochs maximum; batch size 256; early stopping with
  patience 40 and restoration of the best state; gradient clipping 5; built-in
  Adam weight decay zero; model-specific sparse active-weight maintenance rate
  0.01; one unsplit optimizer group with active learning rate 0.001. The stored
  block-linear and reactivation learning-rate fields remain
  0.0001 for exact configuration parity with the selected cell, but they are
  inert when `split_params=false` and are not described as separate optimizers.
- Four feedback conditions, paired within seed:
  1. strict scalar (`error_broadcast_mode=scalar`);
  2. neuron-specific (`per_soma_shared`);
  3. exact dendritic path transport (`path_transport`);
  4. matched backpropagation (`standard`).
- The local arms use the same five-factor eligibility rule, train the
  reactivation parameters, and update the decoder by backpropagation.  Thus the
  intended difference among the first three arms is feedback resolution.
- Twenty fresh paired seeds: 10800--10819.  A search of repository contracts,
  configurations, and project-B run metadata found no previous use of these
  seeds when this contract was frozen.

### Pre-outcome implementation clarification

At 11:54 ET on 28 August 2026, while the array was still incomplete and before
any confirmatory accuracy was inspected, a source-level comparison with the
empirical-paper implementation identified an imprecise phrase in the original
model bullet: the resolved YAML stores a 20-cell inhibitory size, but legacy
`input_mode=1` replaces that size by `None` unless
`input_mode1_build_inhibitory_population=true`. The condition is therefore the
direct signed-input architecture described above, not a two-population E/I
network. This clarification changes neither the generated configurations nor
the frozen comparisons. It is recorded before analysis because the distinction
is semantic and source-determined, not outcome-dependent.
The same audit clarified that the configuration field
`training.main.common.weight_decay_rate=0.01` controls the model's sparse
active-weight maintenance step; it is not Adam L2 weight decay. The resolved
Adam field is zero. References to the former as generic "weight decay" in the
preliminary recipe-screen ledger are corrected accordingly.

After the completed analysis, an editorial source audit corrected one word in
the clarification above: IE is the active direct inhibitory-input contact
field, whereas the explicit inhibitory-population size, EI and II fields are
inert in this input mode. This correction changes no configuration, comparison,
outcome or frozen decision rule.

The local sweep configuration is
`journal/configs/cifar10_additive_feedback_ladder_confirmatory.yaml`.  It is
ignored by Git by policy but remains the launch source.  The intended execution
identity is the clean detached worktree at source commit
`e516c7fec3169253ff8c14bc5f4ab1325469e4f5`.

Generation must invoke the sweep manager from that clean detached worktree while
passing the configuration by its external absolute path,
`/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/journal/configs/cifar10_additive_feedback_ladder_confirmatory.yaml`.
The configuration declares `sweep_contract.expected_config_count=80`; generation
must abort if the resolved count differs.  The ignored YAML must not be copied
into, or used to dirty, the frozen source worktree.
The four conditions use the sweep manager's native named-variant representation
over one complete inline base configuration; no external base-template lookup is
performed at generation or execution time.

## Execution constraints

- Partition: `kempner_h100_priority`; one H100 per run; at most eight concurrent
  jobs.
- Weights & Biases is disabled.
- Generated configurations, logs, checkpoints, results, manifests, and analysis
  outputs must reside below
  `/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/journal_extension_20260828/`.
  The existing read-only CIFAR-10 dataset may remain at its current
  `kempner_dev` location; it is an input, not a generated artifact.
- The generated launcher must pin the source commit and empty tracked diff,
  include the mount canary, and record a frozen manifest.  Any source-identity
  mismatch invalidates the cohort.
- No outcome may be inspected until all 80 expected runs have terminated and
  the analyzer's completeness, configuration, provenance, calibration, and
  convergence audits have run.

## Frozen endpoints and tests

The sole performance endpoint is test accuracy from the restored
validation-best checkpoint.  Validation accuracy selects checkpoints during
training but is not used to select a condition or alter this design.

For each condition, report the paired-seed mean, standard deviation, and 95%
Student-*t* confidence interval.  For each planned paired contrast, report the
mean difference, its 95% paired-*t* confidence interval, the directional paired
*t* test, an exact paired sign-flip/randomization sensitivity test, and the
number of positive seeds:

1. neuron-specific minus strict scalar (assay-validity positive control);
2. exact path minus neuron-specific (primary dendritic-resolution contrast);
3. exact path minus matched BP (equivalence contrast).

The two superiority hypotheses form one confirmatory family.  Directional
paired-*t* *P* values are Holm-adjusted.  They are also evaluated
hierarchically: the exact-path test is confirmatory only if the
neuron-specific positive control succeeds.  Exact path and BP are tested for
equivalence with two one-sided paired *t* tests (TOST, alpha 0.05) and a frozen
margin of +/-1 percentage point.  A difference merely failing to reach
significance is not evidence of equivalence.

## Configuration, calibration, and convergence audits

The analyzer must verify all of the following before evaluating promotion:

1. exactly four conditions by twenty seeds, with no duplicates or unexpected
   runs;
2. clean source commit `e516c7...`, empty tracked diff, frozen launcher and
   manifest, H100-priority launcher, W&B disabled, and project-B output paths;
   the analyzer must verify the frozen input-YAML hash, resolved-YAML hash, all
   80 generated-configuration hashes and indices, source-file hashes, launcher
   hash, and manifest hash rather than merely checking that these files exist;
3. the resolved data, architecture, raw-additive operator, adaptive network
   initialization, empirical reactivation calibration, optimizer grouping,
   learning rates, weight decay, epoch limit, patience, and decoder policy
   specified above;
4. within each seed, identical model/data/common-training configuration after
   removing only the prespecified strategy, feedback-mode, and output fields;
5. a converged, unreverted empirical calibration with finite layer parameters
   in every run, and identical calibration records across paired conditions;
6. finite train/validation trajectories, a validation-best checkpoint, a best
   loss consistent with the recorded validation minimum, and no evidence that
   training was right-censored at the epoch limit while still improving.

Missing or malformed data fail the audit.  A run that reaches epoch 200 is
flagged as right-censored if its best epoch is among the final five epochs or
the least-squares slope over the final ten validation losses is negative.
Right-censoring does not license post hoc extension of only a favorable arm; it
blocks main-text promotion and motivates a separately frozen convergence study.

## Separate adequacy and claim gates

No cross-dataset claim is eligible unless the audit passes and matched BP reaches
at least 45% mean test accuracy.  This adequacy threshold was frozen from the
completed validation-only screen and prevents a mutually low-performing ladder
from passing simply because its arms agree.

The neuron-specific-minus-scalar result can support a cross-dataset feedback-
bandwidth claim if the BP adequacy gate passes and the positive control has a
95% paired confidence interval above zero, Holm-adjusted directional *P* < 0.05,
and exact sign-flip *P* < 0.05.

The cohort can support a main-text within-arbor path-resolution claim only if
the audit and BP adequacy gates pass and all of the following are true:

1. the neuron-specific positive control passes the bandwidth gate above;
2. exact path exceeds neuron-specific feedback by at least 1 percentage point
   on average;
3. the 95% paired confidence interval for exact minus neuron-specific is wholly
   above zero;
4. exact path beats neuron-specific feedback in at least 16 of 20 paired seeds;
5. the exact-path superiority test remains significant after Holm correction
   and exact sign-flip sensitivity testing, in the prespecified hierarchical
   sequence.

Formal TOST equivalence of exact path and matched BP within +/-1 percentage
point is a separate gate for the stronger statement that exact transport
"recovers" matched BP.  Failure of equivalence does not erase valid evidence
that exact path improves over neuron-specific feedback; it only forbids the BP-
recovery statement.

Only claims whose corresponding audit, adequacy, bandwidth, path-resolution,
and equivalence gates pass may be promoted.  Otherwise the complete four-arm
cohort appears only as a compact Supplementary or talk-backup boundary result,
with the failed criterion named explicitly.  No condition may be dropped
because of its outcome.
