# H=4 factorial execution ledger

## Frozen source

- Modeling commit: `a99c3a777f99913e13dfe673a3f3a28bfe3566af`
- Local source ref: `codex/h4-physical-depth-20260819`
- Runtime worktree: `/n/holylabs/kempner_dev/Users/hsafaai/Code/.dendritic-modeling-journal-runtimes/h4-a99c3a7`
- Every generated job exports that worktree's `src` directory on
  `PYTHONPATH`, checks the exact commit and refuses a tracked source diff.
- The relevant implementation suite passed 32/32 tests. Four additional
  manuscript-generator tests could not resolve their ignored external paper
  checkout from the isolated source worktree and were not source failures.
- Serial shunting, grouped-point shunting, raw-additive, shared-soma LocalCA
  and exact-path LocalCA D4 configurations passed `--validate-only`; every
  preflight reported 66,178 trainable parameters.
- The D1 serial/grouped-point equivalence gate passed bitwise on a paired
  32-example batch; see
  `PHYSICAL_DEPTH_H4_D1_INITIALIZATION_AUDIT_20260819.json`.

## Corrected submitted arrays

| Slurm array | Frozen arm | Fits |
|---|---|---:|
| `40489416` | aligned grouped-point shunting BP | 40 |
| `40489417` | aligned serial raw-additive BP | 40 |
| `40489418` | aligned serial shunting BP | 40 |
| `40489419` | aligned serial shunting LocalCA, shared and path | 80 |
| `40489420` | reversed grouped-point shunting BP | 40 |
| `40489421` | reversed serial shunting BP | 40 |
| `40489422` | reversed serial shunting LocalCA, shared and path | 80 |

Total: 360 fits, ten paired seeds 10400--10409.

## Invalidated submission attempt

Arrays `40484370`, `40484380`, `40484420`, `40484444`, `40484495`,
`40484539` and `40484560` were canceled while pending and produced no
training outcomes. Their preflight had unintentionally resolved the editable
parent checkout rather than the declared clean worktree. Repeating the check
with an explicit source path showed that the older pinned commit predated the
feedforward grouped-point implementation. No result from those arrays may be
used. The corrected arrays above were regenerated only after freezing and
testing the exact implementation source.
