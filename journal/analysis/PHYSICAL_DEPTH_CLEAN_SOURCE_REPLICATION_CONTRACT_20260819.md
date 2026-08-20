# Clean-source replication of the H=2 and H=3 physical-depth cohorts

Frozen: 19 August 2026, before any outcome from the clean-source reruns was
inspected.

## Reason

The completed H=2/H=3 jobs passed their original execution guards, but their
manifests identify the implementation as a base commit plus a tracked dirty
diff hash. The jobs are scientifically auditable, but the historical diff is
not a standalone immutable commit. Because the new H=4 extension must be
compared with H=2/H=3, the load-bearing earlier arms will be reproduced from
the same clean source commit as H=4.

## Frozen source and design

- Modeling source: `a99c3a777f99913e13dfe673a3f3a28bfe3566af`.
- No task, model, optimizer, stopping, seed or comparison definition changes.
- The H=2 and H=3 YAML files already frozen on 12--13 August 2026 are reused
  byte-for-byte.
- The original paired seeds are reused. This is a source-concordance
  replication, not an independent-seed replication.
- The 430 reruns comprise:

  1. all 220 fits in `configs/remaining_physical_experiments/`: H=3 literal
     grouped-point aligned/reversed; H=2 serial, grouped-point and shared/path
     LocalCA aligned/reversed;
  2. 210 load-bearing H=3 fits in
     `configs/nonlinear_physical_depth/confirmatory/`: aligned serial shunting
     BP, reversed serial shunting BP, aligned serial shunting shared/path
     LocalCA, reversed serial shunting shared/path LocalCA, and aligned serial
     raw-additive BP.

The independent- and shuffled-sensor H=3 controls are not rerun because they
do not enter the H=2--H=4 depth-matching series; their existing completed
outcomes remain boundary controls.

## Endpoints

1. Recompute every H=2/H=3 paired contrast named in the original contracts at
   the training-seed level.
2. Compare each rerun's held-out accuracy with its historical paired
   configuration and seed; report the full distribution and maximum absolute
   discrepancy rather than assuming exact hardware-level identity.
3. Recompute the best mean serial physical depth for H=2 and H=3 using only
   the clean-source reruns, before combining them with H=4.
4. Retain all null, reversed, point and raw-additive outcomes regardless of
   whether they reproduce the earlier headline.

## Validity and decision rules

- Every expected fit must reach a finite, stage-complete checkpoint without a
  fallback warning.
- Job scripts must export the frozen worktree's `src` directory and refuse a
  changed commit or tracked diff.
- Resource identities, seed sets, inventories, placement rules and generated
  configuration counts must match the original contracts.
- Paired-seed bootstrap intervals, exact two-sided sign-flip tests and positive
  pair counts are reported for scientific contrasts. The training seed is the
  inferential unit.
- No outcome-dependent retuning or selective replacement is allowed. If a
  load-bearing contrast changes sign or loses its directional gate, the paper
  must report that failure and use the clean-source result as the canonical
  endpoint.

## Submitted arrays

All arrays were pending when recorded; no rerun outcome had been inspected.

| Slurm array | Frozen arm | Fits |
|---|---|---:|
| `40490346` | H=3 aligned raw-additive BP | 30 |
| `40490349` | H=3 aligned shunting BP | 30 |
| `40490351` | H=3 aligned shunting LocalCA, shared and path | 60 |
| `40490354` | H=3 reversed shunting BP | 30 |
| `40490356` | H=3 reversed shunting LocalCA, shared and path | 60 |
| `40490358` | H=2 aligned grouped-point BP | 20 |
| `40490360` | H=2 aligned serial BP | 20 |
| `40490362` | H=2 aligned serial LocalCA, shared and path | 40 |
| `40490365` | H=2 reversed grouped-point BP | 20 |
| `40490371` | H=2 reversed serial BP | 20 |
| `40490374` | H=2 reversed serial LocalCA, shared and path | 40 |
| `40490377` | H=3 aligned grouped-point BP | 30 |
| `40490379` | H=3 reversed grouped-point BP | 30 |

Total: 430 clean-source reruns.

## Scheduler provenance

At 23:31 EDT on 19 August, all still-pending array elements were reassigned
in place from account `kempner_dev` to the authorized account
`kempner_bsabatini_lab` after remaining queued for priority. The job IDs,
partition, commands, immutable source checkout, configuration manifests,
seeds and output paths were unchanged. This was a scheduler-only change made
before any replication outcome was inspected.

## Completion and retained results

All 430 reruns completed from immutable commit `a99c3a7` and passed source,
completeness, finite-metric, seed, no-fallback and exact-resource gates. The
median absolute historical-to-clean accuracy change was 0.030 pp, the mean was
0.154 pp, and 357/430 pairs were within 0.1 pp. The maximum 15.01-pp change
occurred in the unstable raw-additive negative-control cohort and remains in
the concordance table.

- H2 aligned serial-BP D2 minus D1: +30.84 pp (30.46 to 31.23; 10/10).
- H2 architecture-by-placement interaction: +32.43 pp (32.13 to 32.76;
  10/10).
- H3 aligned serial-BP D3 minus D1: +30.86 pp (30.51 to 31.29; 10/10).
- H3 architecture-by-placement interaction: +31.27 pp (30.73 to 31.74;
  10/10).
- H3 shunting-minus-additive depth interaction: +37.66 pp (35.59 to 39.51;
  10/10).

Every load-bearing H2/H3 direction and claim gate was retained. Canonical
outputs are in `../source_data/physical_depth_clean_source_replication/` and
Supplementary Figure S26. This is same-seed source concordance, not an
independent scientific replication.
