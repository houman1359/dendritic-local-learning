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
