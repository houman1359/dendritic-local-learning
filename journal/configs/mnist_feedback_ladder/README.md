# Matched MNIST exact-path extension

These four specifications create a current-source, fully matched 15-seed MNIST
feedback ladder for Figure 2A. They inherit the frozen shunting and additive
specifications in `../reruns/`, changing only the feedback field, run name,
scheduler placement and output location. The coordinate specifications rerun
the scalar and neuron-specific conditions because the earlier clean cohort was
executed on a July source snapshot; the exact-path and coordinate arms in this
extension therefore share one source commit and execution environment.

The full factorial is 15 seeds (42--56) x three feedback fields (`per_soma`,
`per_soma_shared`, and `path_transport`) for each architecture. W&B is
disabled. All generated configurations and results are stored on the
`kempner_project_b` filesystem under
`journal_extension_20260820/sweep_runs/mnist_feedback_ladder`.

The new outcomes must not enter Figure 2A until all 90 configurations pass the
same completeness, finite-metric, checkpoint and resolved-configuration audits
used for the existing scalar and neuron-specific conditions.

## Strict-scalar implementation control

The two `strict_scalar_*_15seed.yaml` specifications add a 30-run control that
forces `error_broadcast_mode: scalar` at every stage. This isolates the 640
near-somatic parameters that retained neuron coordinates in the legacy
`per_soma` implementation; every input synapse already received a strict
scalar in that legacy arm. The paired analysis and main-figure decision rule
were frozen before inspecting outcomes in
`strict_scalar_control_contract.yaml`. W&B remains disabled and all outputs use
the same `kempner_project_b` root and H100 scheduler profile as the matched
ladder.

The contract SHA256 is
`a29e92efb472398afc5ae6991a0f767cebf3a06a5961bf201b42e3a0a5842fe1`.
The arrays were submitted on 26 August 2026 as Slurm jobs `41913862`
(shunting) and `41913871` (additive). The executable source hashes in their
frozen manifests match the 25 August ladder; the outer repository commit
differs only because later commits did not modify those executable files.
All 30 tasks completed with exit code zero and passed the resolved-mode,
finite-metric, checkpoint, seed-balance and no-W&B gates. The strict arm was
not practically equivalent to the legacy fallback: strict-minus-legacy
accuracy was -4.859 percentage points in shunting networks (95% paired-seed
bootstrap interval, -5.669 to -3.994) and -2.274 points in additive networks
(-2.699 to -1.796). The strict arm therefore replaces the legacy scalar rung
in Figure 2; the legacy arm remains an explicit implementation audit.

## Execution record

The current-source arrays were submitted on 25 August 2026 from commit
`6c1aaa25abd056c417842e1c46378b65d036f6a7` with a clean tracked source
worktree:

- exact path, shunting: Slurm array `41840967`;
- exact path, additive: Slurm array `41840984`;
- scalar/neuron-specific, shunting: Slurm array `41842054`;
- scalar/neuron-specific, additive: Slurm array `41842076`.

Array `41842076` was moved while fully pending from
`kempner_h100_priority` to the permitted `kempner_eng` partition to avoid the
shared H100 allocation limit; its scheduler QOS consequently changed from
`kemp_gpu16_id38` to the partition-compatible `normal` QOS. These
scheduler-only changes did not alter a scientific configuration, source file
or output path.

After tasks 0--5 had started in the two coordinate arrays, their still-pending
tasks 6--29 were cancelled and resubmitted with an explicit `--gpus=1` request
to the `kempner_requeue` partition (`normal` QOS): Slurm arrays `41843834`
(shunting) and `41843847` (additive). The resubmitted tasks use the unchanged
generated scripts, configurations and output directories of arrays `41842054`
and `41842076`, respectively. No running or completed task was cancelled.
Additive task 9 was automatically requeued once by the preemptible partition;
its unchanged script, configuration and output location remain subject to the
same strict final-metric and checkpoint gates as every other task.
