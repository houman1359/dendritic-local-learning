# Clean rerun and sensitivity contract

This contract covers reviewer requests that require new execution but do not
change the central trained-subtree hypothesis.

## Reproducibility release

Rerun, in order:

1. exact-transport three-/five-factor factorial and matched backpropagation;
2. measured-response branch model;
3. focal-shunt summary and finite-difference audits.

Every run must use a clean tagged commit, pinned lock file or container, strict
paper mode, resolved configuration, input/output checksums, and a generated
mechanism ledger. Transport-shape, decoder-to-soma, partner mapping, or missing-
artifact fallbacks must raise. The robust scalar reduction must record both the
number of examples and fraction of examples using its signed-maximum
replacement.

### Clean exact-transport/backpropagation audit

The implemented audit contains 320 training runs: exact path transport or
matched backpropagation, MNIST or the synthetic noise-resilience task,
additive or positive-conductance shunting cores, depths one through four, and
paired seeds 42--51. Training executes from detached tracked-clean source
commit `74792ca`; the nested synthetic-dataset repository is fixed at commit
`4f3612a`. Each accepted run must provide its resolved configuration, seed
record, complete learning summary, finite final metrics, stage-complete finite
checkpoint and scheduler/training logs. Exact-versus-backpropagation contrasts
are paired by seed within task, core and depth; depth-averaged inference first
averages the four depths within seed.

Launch gates exposed three non-outcome failures, which are retained outside
the inferential cohort: the detached checkout initially lacked its ignored
nested dataset repository; 48-GB host-memory requests were insufficient for
the synthetic task; and signed synthetic inputs are not valid positive
conductance drives for the shunting denominator. The final synthetic jobs
request 180 GB of host memory. The shunting synthetic condition rectifies the shared
transfer output for both exact and backpropagation runs; the additive condition
retains signed inputs. This correction occurred at the input-validity gate,
before an affected model could train, and prevents the clean audit from being
misrepresented as a byte-identical rerun of the earlier signed-shunting
configuration. When full-dataset metric materialization exceeds device memory,
the runtime records the event and recomputes the same metric in streaming
batches; the audit distinguishes this successful evaluation path from any
fatal OOM or non-finite endpoint.

The frozen sweep scripts were generated with an eight-task array throttle. On
10 August 2026 the four synthetic arrays were raised to a maximum throttle of
40 after submission because the valid shunting jobs require approximately two
hours each. This changes scheduling concurrency only; Slurm still applies
availability and fair-share admission, and no resolved scientific
configuration, seed, training path or acceptance gate changes.

The generic GPU request subsequently placed some synthetic jobs on 20-GB MIG
devices, which failed before completing training, and the depth-four Adam
state exceeded the 96-GB devices assigned to other attempts. Those incomplete
attempts were cancelled on 10 August 2026. Every configuration lacking an
accepted `performance/final.json` was resubmitted unchanged as Slurm jobs
38115959--38115962 with an H200 feature constraint and 180 GB host memory.
The requeue partition then preempted and restarted 60 array tasks before any
new endpoint completed. Those four arrays were cancelled, and only the still-
missing configurations were resubmitted to the non-preemptible
`kempner_dev` partition as jobs 38123106, 38123115, 38123120 and 38123122,
with the same H200 constraint, 180-GB host request and four-hour limit.
While the 67 additive configurations ran non-preemptibly, the two still-
pending shunting fallback arrays (38123120 and 38123122) were held. Identical
missing shunting configurations were submitted opportunistically to the H200
requeue partition as jobs 38126638 and 38126639. Both remained pending with
zero runtime and were cancelled; the non-preemptible shunting arrays were then
released. A second exact-shunting submission, job 38123221, had been created
19 seconds after 38123122 with the same missing-index list. It remained queued
with zero runtime and was cancelled as soon as the overlap was detected; the
earlier 38123122 array was retained. No simultaneous writer existed and no
result was changed.
Completed configurations were not rerun. The collector selects the newest
scheduler attempt per configuration and still rejects any fatal OOM in that
accepted attempt; hardware and host-memory corrections do not alter data,
model, optimizer, seed, epoch or analysis settings.

Two depth-four exact-transport tasks in job 38123122 subsequently encountered
CUDA allocator fragmentation during the first training forward pass: the
device had enough total H200 memory, but reserved non-contiguous blocks
prevented a 4.69-GB allocation. Their failed attempts produced no accepted
endpoint. The execution-only setting
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was then applied to every
missing shunting task. Configurations 31 and 34 completed under this setting
as jobs 38184487 and 38184755, demonstrating that the repair removed the
fragmentation failure without changing a resolved scientific configuration.
Full-dataset metric materialization can still exceed device memory at depth
four; the paired warning and successful streaming recomputation are retained
in the accepted logs and distinguished from a training failure by the strict
collector.

During the final queue audit, the zero-runtime pending backpropagation repair
38184826, the zero-runtime single-task exact repairs 38185459 and 38185494,
and the zero-runtime pending elements 38--39 of 38184755 were moved from the
development queue to the laboratory's dedicated H200 partition. The
replacement jobs are 38204572 (backpropagation configurations 5--39) and
38204573 (exact configurations 32--33 and 38--39). Active elements 36--37 of
38184755 were left untouched. The original dependent collector 38164312 was
canceled before runtime and replaced by collector 38204661, which depends
only on those active elements and the two replacement arrays. This was a
scheduling/account change only. All canceled targets had zero runtime, all
output directories have one active writer at most, and the source commits,
configurations, seeds, training code, allocator repair and acceptance criteria
are unchanged. After the first 12 backpropagation tasks launched, that array's
pending-task throttle was reduced to eight so four per-user GPU slots could be
used by the longer exact tasks; no running task was interrupted. After all
four exact replacement elements had acquired H200 devices, the
backpropagation throttle was restored to 12. As exact repairs completed, it
was then raised one slot at a time so freed devices could be reused while
preserving the same 16-job ceiling. These changes affected only the number of
concurrently executing frozen configurations.

At 04:07 on 11 August, backpropagation configurations 31--39 exited together
on nine different nodes when the laboratory filesystem reached its 4.0-TiB
group quota. Configuration 31 had completed all 180 epochs but its final
checkpoint write was truncated; configurations 32--39 stopped earlier. None
produced an accepted endpoint. This common-time, cross-node failure is treated
as a storage failure, not as a numerical outcome. Redundant
`main_network/standard_best_model.pt` intermediates were relocated to the
50-TiB laboratory scratch filesystem while final checkpoints, resolved
configurations and all logs were retained for the audit. Configurations
31--39 were rerun unchanged as job 38248969 after space reclamation; every
element completed with exit code zero. The dependent collector 38249039 then
audited all 320 runs and 160 method pairs. It found no missing artifact,
nonfinite endpoint or checkpoint tensor, incomplete dendritic stage, critical
accepted-log match, or transport-shape fallback. Forty-two runs used the
explicitly logged streaming terminal-metric path. The four depth-averaged
task--core intervals all included zero; mean exact-minus-backpropagation
accuracy across the 160 raw pairs was -0.0545 percentage points and the
largest absolute pair difference was 5.90 points. The result is an empirical
agreement control, not a formal equivalence test.

## Functional scan and response sensitivity

Primary scan rule: include all eligible scans and nest scan within target.
If resource limits prohibit this, freeze one deterministic rule before looking
at functional outcomes: most connected imaged partners, then highest median
split-half reliability, then earliest scan as a final tie-break.

For every retained scan, compute without outcome-dependent selection:

- the archived raw trial-window mean;
- baseline-subtracted response;
- prespecified early and lagged calcium windows;
- source-provided processed response where available;
- deconvolved activity/event rate where available;
- reliability-restricted and reliability-weighted variants.

Partners with multiple contacts receive a contact-size-weighted distribution
over segments in the segment-level model. The existing dominant-contact,
first-major-branch model remains a coarse control.

Execution resolution: all 13 automatically eligible scans were retained and
nested within seven targets. Inspection of the archived timestamps showed
only approximately 0.1-s intertrial gaps, so a pretrial baseline or post-trial
lag window would reuse the adjacent stimulus rather than define an independent
response. The release therefore retains the frozen full-trial raw-fluorescence
endpoint and the prespecified repeat-reliability thresholds. It does not label
an adjacent-stimulus subtraction as a baseline sensitivity. The complete-tree
extension below removes the major-branch state reduction; one activity feature
per imaged partner remains placed at its dominant size-weighted contact segment,
and this remaining multi-contact reduction is stated as a limitation rather
than silently treated as a distributed synapse measurement.

## Segment-level functional model

Place every mapped contact on the compressed reconstructed tree, retain
intermediate compartment states, compute exact segment-level adjoints, and
project the resulting fields into ancestry, depth, matched surrogate-tree,
random sparse, and dense rank-matched dictionaries. Split by stimulus identity,
nest repeated splits within scan and target, and keep target as the biological
inferential unit.

## Physical focal-shunt matrix

Cross:

- electrotonic axial-to-membrane regime;
- dose relative to local leak plus synaptic conductance;
- fixed absolute dose in nS;
- dose relative to local input conductance/input resistance;
- distributed background conductance.

Report absolute localization together with signed log-magnitude change,
attenuated/enhanced fractions, sign-flip probability, total descendant gradient
energy, and dependence on baseline adjoint sign and magnitude. Add a qualitative
active-compartment ensemble with sodium, potassium, calcium, NMDA, GABA-A, and
background-conductance variants. The objective is a regime boundary, not a
search for a universally positive effect.

## Release structure

The immutable paper release must provide one command per figure and fail on
missing runs, fallbacks, checksum changes, or unregistered panels. It must also
contain the MICRONS/DANDI join manifests, stable IDs, materializations,
resegmentation mappings, scan choices, exclusions with reasons, random seeds,
and proofreading/type status where available.
