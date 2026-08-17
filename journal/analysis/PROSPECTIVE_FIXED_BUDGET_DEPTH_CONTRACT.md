# Prospective fixed-budget dendritic-depth contract

Date frozen: 2026-08-02

## Question

Does arranging the same number of distal leaves and approximately the same
number of active input contacts into a deeper dendritic hierarchy change local
learning, after separating feedback routing from forward computation?

## Design

The cohort uses the frozen noise-resilience task, one population of 128 somas,
compact fixed-index input connectivity, and four trees with 16 distal leaves:

| Branch factors | Depth | Non-somatic compartments per soma | E/I contacts per compartment | Active input contacts per soma |
|---|---:|---:|---:|---:|
| `[16]` | 1 | 16 | 40 / 20 | 960 |
| `[4,4]` | 2 | 20 | 32 / 16 | 960 |
| `[2,2,4]` | 3 | 22 | 29 / 15 | 968 |
| `[2,2,2,2]` | 4 | 30 | 21 / 11 | 960 |

The eight-contact discrepancy at depth 3 is less than one percent and is
reported rather than hidden. The cohort crosses these shapes with shunting and
raw additive cores. Local learning uses scalar, ancestry-shared, or exact path
transport with the theorem-derived three-factor rule. Matched backpropagation
controls use the same shapes, contacts, core, task, initialization policy, and
training horizon. Confirmatory runs use seeds 42--51.

## Pre-specified contrasts

1. The depth-4 minus depth-1 change within core and feedback condition.
2. The linear depth trend and a shape-factor sensitivity analysis that uses
   the exact number of non-somatic compartments.
3. The depth-by-feedback interaction: a depth effect under scalar feedback
   that narrows under exact transport is a credit-routing effect.
4. The depth-by-core interaction: a shunting/additive difference that remains
   under exact transport and backpropagation is attributed to the complete
   forward/optimization model, not specifically to backward credit gating.
5. The local-rule minus matched-backpropagation gap at each shape.

## Interpretation limits

This is a fixed-leaf and approximately fixed-active-contact comparison, not an
exact trainable-parameter match. Deeper trees contain more coupling and
reactivation parameters. Exact parameter counts, active contacts, peak memory,
and runtime are therefore reported for every run. The random fixed-index map
does not test task-aligned morphology; that question is isolated in the
separate spatial-topology cohort.

Canaries are software checks only. Confirmatory arrays are released only if all
24 canary runs pass the same checkpoint, finite-value, stage, coupling, and
source-identity audit as the primary study.
