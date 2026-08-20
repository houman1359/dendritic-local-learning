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
