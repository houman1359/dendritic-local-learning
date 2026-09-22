# Outcome-independent scheduler amendment for the CIFAR-10 shunting cohort

On 22 September 2026, the user explicitly requested `kempner_requeue` when GPUs
were unavailable. The H100-priority queue reported `QOSMaxGRESPerUser`. Before
any new cohort accuracy was inspected, waiting elements of array 47767554 were
moved in place to `kempner_requeue`, using QoS `normal`, the explicit feature
constraint `h100`, and the unchanged request for one GPU, eight CPUs and 64 GB.
Task 14 was moved first; tasks 15--79 followed. Tasks 0--13 had already completed
or started on H100-priority and were left there. The original eight-task array
throttle applies across both partitions.

This amendment supersedes only the original H100-priority scheduling restriction.
Seeds, source checkout, configurations, initialization, optimization, stopping
rules, comparisons and statistical decisions are unchanged. Every run still
uses an H100 GPU; the broader requeue pool's H200, A100 and Blackwell GPUs are
excluded by the explicit constraint. The original launcher, input YAML and
manifest remain byte-identical. Scheduler overrides and before/after states
are preserved separately rather than rewriting the frozen launch record.

Preempted tasks remain in the same Slurm array and requeue under the same
configuration and hardware constraint. They are not dropped from the cohort.
The completed-cohort audit additionally checks all 80 Slurm allocation records:
successful completion, exactly one H100, eight CPUs, 64 GB, and the explicit
H100 constraint on requeue tasks. It records restart counts. The frozen
scientific analyzer then checks the actual saved configurations, calibration,
checkpoints, convergence and seed completeness before computing results.

The original pending CPU analysis job is replaced by a dependent job that adds
this hardware audit before calling the unchanged, hash-pinned scientific
analyzer. The exact job IDs, hashes and command receipts are retained in
`cifar_shunting_revision_20260922/scheduler_migration_20260922/` and the execution
record. No outcome-based run selection or new scientific decision is introduced.

## Complete same-hardware replacement after immediate H100 preemptions

The H100 fallback started but tasks 14--17 were preempted within minutes.
The requeue partition has idle RTX PRO 6000 Blackwell Server Edition GPUs.
Before inspecting any scientific outcome, the execution plan was therefore
amended to rerun the entire 80-run cohort on that GPU type, constrained by
`rtx6000pro`. The partial H100 attempt (array 47767554) is retained for
infrastructure provenance and excluded in its entirety from scientific
summaries, whether a task had completed or not. It is not pooled with the new
execution. Remaining tasks and the dependent H100-only audit are cancelled.

The replacement uses the same source commit, all twenty seeds, all four rules,
and byte-equivalent scientific base configurations and variant overrides.
Only the output directory, scheduler account/partition/QoS, GPU constraint and wall-time
allowance change. The two-hour wall-time allowance accommodates a different
accelerator; the 400-epoch limit and early stopping are unchanged. The original
scientific analyzer is retained, and a revised copy changes only launch hashes,
scheduler checks and the output-path provenance. Its statistical decisions are
unchanged. A one-task infrastructure start is expanded to the original maximum
of eight after checking that the GPU runtime starts successfully, without
inspecting accuracy.

The completed replacement cohort must contain 80 successful allocations on
RTX PRO 6000 Blackwell Server Edition GPUs. The hardware checker requires the
explicit constraint and preserves any restart counts. This replaces the mixed-
partition H100 plan above. The original additive cohort used H100 GPUs; GPU type
and training recipe differ between architectures, so comparisons remain within
architecture. This change does not authorize an isolated forward-operator
comparison between architecture means.

The frozen sweep generator permits `kempner_dev` with `kempner_requeue`; this approved pair is used for the replacement.

The frozen generator's array writer omits the YAML GPU constraint, so the
replacement submission supplies `--constraint=rtx6000pro` explicitly. An initial
submission without that override (47787868) was cancelled while still pending;
it produced no training outputs. The accepted replacement is array 47787999.
Its first task is an infrastructure check within the complete scientific
cohort; after successful completion, CPU job 47788421 raises the concurrency
limit from one to eight without inspecting accuracy. CPU job 47788381 audits
all completed allocations and then runs the frozen scientific analysis.


## Move the entirely unstarted replacement to kempner_eng

At the user's explicit request on 22 September 2026, array 47787999 was moved
in place from `kempner_requeue` to `kempner_eng`. All 80 tasks were pending
before the move; no Blackwell training had started and no scientific outcomes
were inspected. The array was held briefly while its dependent checks were
replaced. This partition contains H200 GPUs, so the complete cohort now uses
`h200`, with one GPU, eight CPUs, 64 GB, QoS `normal`, and the same two-hour
wall-time allowance. Every scientific configuration, seed, frozen input,
launcher and statistical decision remains unchanged. The original launch
records retain their historical requeue settings; the scheduler override and
all-task before/after records document the actual execution.

The old Blackwell-specific ramp and analysis jobs (47788421 and 47788381)
were cancelled before running. Their replacements verify H200 allocations on
`kempner_eng`; the first successful run still raises concurrency from one to
eight without inspecting accuracy. The completed-cohort audit still requires
all 80 successful runs before scientific analysis. No hardware cohorts are
pooled. The excluded original H100 attempt remains excluded in its entirety.
Comparisons between additive and shunting architectures still differ in
training recipe and accelerator and do not isolate the forward operator.
The move receipt, updated checker and dependent-job receipts are retained in
`scheduler_migration_eng_20260922/` and `eng_submission.json` in the durable
revision directory.


## User-requested concurrency increase

At 19:23 UTC on 22 September 2026, the user requested a concurrency limit of
12. Array 47787999 was updated immediately to `ArrayTaskThrottle=12` on
`kempner_eng`, retaining the H200 constraint and all scientific settings.
The pending ramp job 47804988 was replaced by first-run check 47806131, which
retains the runtime and allocation check but does not modify the throttle.
Final completed-cohort audit and analysis job 47804995 is unchanged. The
scheduler receipt is preserved as `throttle12_receipt.json`; no scientific
outcomes were inspected for this scheduling decision.
