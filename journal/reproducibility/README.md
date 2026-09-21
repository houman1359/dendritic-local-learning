# Reproducibility record

This directory distinguishes what is exactly archived from what can only be
reconstructed. Missing versions and hardware metadata are never inferred from
the current checkout.

## Evidence map

| Analysis | Frozen inputs in this package | Executable implementation | Provenance status |
|---|---|---|---|
| Regular-tree feedback | Portable sweep, representative resolved configs, 60 seed-level rows, July frozen manifest | Main repository plus `code/regular_tree/additive_reference.py` | Complete for seeds 47-56; partial for seeds 42-46 |
| Exact transport and backpropagation | Portable sweeps, representative resolved configs, 25 independent-seed results, and grouped summaries | Main repository | Configs and hardware survive; exact software environment does not |
| Reconstructed morphology and focal perturbation | Main source-data tables | Exact source copies in `code/reconstructed_tree/` | Source hashes preserved; raw MICrONS data are not duplicated |
| Public-v661 sensitivity cohort | Frozen 55-candidate manifest, 47-cell processed tables, endpoint exclusions, and 20-stream routing output | Exact source copies in `code/reconstructed_tree/` and journal scripts | Public static URLs and hashes preserved; cells come from the same mouse and a historical release |
| Measured-response branch model | Four-channel specification and seven-target manifest | Portable scripts in `code/task_derived/` | Analysis code and output survive; exact package lock and compute host do not |

`origin_manifest.tsv` gives the source path, source SHA-256 hash, destination,
and every portability change. `archived_hardware.csv` records only execution
metadata recoverable from Slurm accounting. `task_derived_primary_targets.csv`
contains the complete target, DANDI asset, NWB path, trial, site, and branch
manifest used by the four-channel result.

`original_cohort_functional_accounting.csv` accounts for all eight cells in
the structural pilot before the seven-target functional analysis. Source code
records the structural cohort as a hand-authored, deliberately small,
layer-diverse pilot from an existing anatomy/functional-coregistration cohort.
It was not a population-random or coverage-ranked sample; no more specific
deterministic sampling rule is recorded. The omitted cell, root
`864691136043380566` (nucleus `292648`), has one row in the archived trial
manifest, for session 8/scan 4, but that row has
`has_dandi_asset=False` and no DANDI asset identifier or path. This is the
verifiable exclusion reason; no additional biological or quality exclusion is
inferred. For targets with several DANDI-eligible scans, the archive does not
record a deterministic scan-selection rule, so the selected pairs are labeled
as frozen pilot assets.

## Regular-tree archive boundary

The reported feedback result has 15 seeds per architecture and condition.
It is a mixed archive:

- Seeds 47-56 are the 40-run sweep
  `rebuttal_feedback_definition_3f_mnist_extra10seed_20260724152657`.
  All resolved configurations, checkpoints, logs, and its frozen sweep
  manifest survive. The manifest records Python 3.10.13, Git commit
  `c39fa57987e5f5f702274482bdd9a7a6c9e3382b`, and a dirty tracked
  worktree, plus hashes for the sweep generator and selected model sources.
- Seeds 42-46 survive only as 20 source-data rows. Their original resolved
  configurations, checkpoints, exact software environment, job IDs, and GPU
  types were not found. The journal package does not manufacture them.

The exact-transport factorial and matched backpropagation reference retain
resolved configurations and complete run directories. Their exact Git commit,
Python version, and package lock were not archived. They should be described
as configuration-reproducible, not bitwise-reproducible.

## Archived hardware

The July feedback extension used 28 A100 3g.20GB MIG tasks (3.475 GPU-hours)
and 12 RTX PRO 6000 Blackwell Server Edition tasks (0.729722 GPU-hours), for
4.204722 GPU-hours over 40 tasks. The exact-transport factorial used 20
A100-SXM4-40GB tasks (2.952500 GPU-hours), and its backpropagation reference
used five (0.473889 GPU-hours). Times are summed GPU-task elapsed time; queue
delay and scheduler wall time are not included.

## Submitted full feedback rerun

`configs/reruns/feedback_definition_shunting_15seed.yaml` and
`feedback_definition_additive_15seed.yaml` replace the mixed archive with one
current-code cohort. Each file fixes seeds 42-56, both feedback conditions,
and the validated architecture-specific gate policy explicitly: analytical
for shunting and occupancy-quantile for additive. Generate-only validation on
31 July 2026 produced exactly 30 configurations and two Slurm array scripts
per architecture. The validated arrays were then submitted on 31 July 2026 as
Slurm jobs 36616195 (shunting) and 36616196 (additive), each with 30 tasks and
a six-task concurrency limit. Submission does not replace the mixed archive:
outputs must first pass completeness, configuration, provenance and numerical
checks. Details and commands are in `configs/reruns/README.md`.

The compute estimate is an extrapolation from the archived 40-task sweep:

`60 / 40 * 4.204722 = 6.307083 GPU-hours`.

With both arrays running at their configured limit of six tasks, the maximum
concurrency is 12 GPUs. The mean-throughput ideal is 31.5 minutes of active
wall time; 30-50 minutes is a more useful planning range given the archived
GPU-dependent task means. Queue time is excluded. This is a resource estimate,
not a completed experiment or a promised runtime. Scheduler state is not
frozen in this document; query Slurm using the job identifiers above.

## Additive-equation audit

The publication alias is raw additive: shunting is false, additive
normalization is false, and the forward voltage is excitation minus inhibition
plus child input. `code/regular_tree/additive_reference.py` re-expresses the
source equation, parametric-tanh derivative, softplus chain factor, and exact
child error transport in standalone NumPy. It is a readable reference, not
the training implementation used for the archived results.

Run its independent finite-difference check with:

```bash
python code/regular_tree/test_additive_reference.py
```

At package preparation it reported maximum raw-parameter derivative error
`9.264e-11` and child-transport error `5.623e-12`.

## External data

Raw MICrONS/CAVE and DANDI/NWB data are not redistributed. CAVE access may
require authentication, and DANDI extraction requires network access. Place
downloaded or service-derived inputs under `external_data/`; portable scripts
write intermediates and results under `reproduced_results/`. Both directories
are ignored by Git. The seven DANDI asset identifiers and NWB paths needed for
the reported task cohort are public metadata in the target manifest.
