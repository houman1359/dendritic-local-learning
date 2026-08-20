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

## Runtime events

Array `40489416` tasks 0--8 completed. Task 9 exited before training because
the allocated GPU on `holygpu8a19102` returned
`cudaErrorDevicesUnavailable`; model construction, data loading and the
initial gate audit had completed, but no checkpoint or outcome was produced.
The incomplete `results/config_9` directory and job logs were preserved under
`failed_attempts/`. A separate retry job, `40501760`, was canceled before it
started; frozen array element `40489416_9` was instead requeued in place with
that node excluded. This preserved the original array priority and released
the remaining array throttle. It is a hardware retry, not a scientific rerun
or configuration change.

At 23:31 EDT on 19 August, the still-pending elements of all H4 and
clean-source replication arrays were reassigned in place from account
`kempner_dev` to the authorized account `kempner_bsabatini_lab` after the
original account remained pending for scheduler priority. No partition,
command, configuration, seed, array index, output directory or scientific
parameter changed. The reassignment raised scheduler priority and tasks began
immediately. The nine completed elements of array `40489416` retain their
original account provenance; every subsequent task records the new account in
Slurm accounting. An attempted in-place change to
`kempner_h100_priority` was rejected by Slurm before mutation because the
existing generic-GPU request could not be revalidated for that partition; no
job ran under the rejected request.

## Invalidated submission attempt

Arrays `40484370`, `40484380`, `40484420`, `40484444`, `40484495`,
`40484539` and `40484560` were canceled while pending and produced no
training outcomes. Their preflight had unintentionally resolved the editable
parent checkout rather than the declared clean worktree. Repeating the check
with an explicit source path showed that the older pinned commit predated the
feedforward grouped-point implementation. No result from those arrays may be
used. The corrected arrays above were regenerated only after freezing and
testing the exact implementation source.

## D3 construction repair and completion

The original H4 arrays finished 270/360 configurations. All 90 failures were
D3 construction failures: automatic inventory rounding grouped the four tiers
as `[4], [2+1], [1]`, which did not match the intended D3 row inventory
`[4,2,2]`. No H4 scientific outcome was inspected before repair. Optional
explicit inventory grouping was added at modeling commit
`e90fb9896daa95df4aad4c0d92acc0cd65bcd750`, with legacy automatic behavior
unchanged; 33 targeted tests passed. The repaired D3 grouping is
`[[0], [1], [2,3]]`. Only the 90 failed conditions were rerun, retaining the
same seeds and all scientific settings.

The repaired arrays were `40513363` on H100 priority and `40514263`,
`40514269`, `40514274`, `40514280`, `40514295`, `40514306` on the RTX
partition. All 90 completed. The merged audit is `complete_pass`: 360/360
unique intended outcomes, no missing/nonfinite/fallback/resource failure, and
an explicit `result_origin` for every replacement row.

## Frozen results

- Aligned serial-BP D4 minus D3: -1.46 pp (95% interval -1.74 to -1.16;
  0/10 positive; exact sign-flip P=0.001953; BH q=0.003906). The simple
  one-stage-per-task-level prediction is falsified.
- Aligned serial-BP D4 minus D1: +25.63 pp (24.71 to 26.40; 10/10;
  q=0.003906). Depth remains useful relative to D1 but saturates at D3.
- D4 architecture-by-placement interaction: +25.91 pp (25.06 to 26.62;
  10/10; q=0.003906); grouped-point BP remains flat.
- Shared-LocalCA D4 minus D3: +5.27 pp (4.84 to 5.72; 10/10; q=0.003906).
- Path-LocalCA D4 minus D3: +0.60 pp (0.04 to 1.25; 7/10; P=0.085938),
  failing the frozen positive gate.
- The raw-additive D4-minus-D3 effect and shunting-minus-additive interaction
  are null.

Canonical outputs are in `../source_data/physical_depth_h4_factorial/` and
Figure 6A--D.
