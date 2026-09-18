# Execution record — 18 September 2026

## Scope and status

Implemented a separate inhibition/credit experiment series. Existing Figure 5
data, manuscript figures and prior user edits are unchanged by this series.
The scientific protocol, equations, controls and limitations are in README.md.
These are development experiments; no positive learning result is assumed.

The production-transfer and analysis test suites pass **16 tests**. They check
the original Figure 5 voltage and 24-conductance gradients against genuine
DendriNet modules, finite-difference gradients in all three forward modes,
local graph detachment, context availability, gate-control normalization,
unchanged states during diagnostics and reciprocal-cable perturbation algebra.
A short training smoke test also completed with finite losses and gradients.

## Storage and jobs

Persistent root (not temporary storage):

`/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/inhibitory_credit_transfer_20260918`

- Training array **46950068**: 54 workers, at most 12 concurrently; 540
  development trajectories. All numerical settings are in `protocol.json`.
- Biological array **46950197**: original MICrONS, disjoint MICrONS, Pinky,
  and measured-response analyses. Its separate execution snapshot is
  `biology_revision_01/` under the root above.
- Original biological array **46950069** was cancelled while still pending,
  before analysis, to retain the two inherited Pinky QC exclusions. The
  replacement protocol links to the unchanged parent snapshot by hash.
- Collector **46950293** runs after the training array ends, including if an
  arm fails, and reports whether all 540 expected trajectories are present.

Workers request one CPU, use project_b outputs and disable W&B. The pending
jobs are eligible for both `serial_requeue` and `shared`; an attempted addition
of `test` was rejected because that partition cannot be combined with others.
Immutable
source/input manifests identify the working-tree code actually run, including
the production repository's pre-existing edits. No commit hash is substituted
for that snapshot. No experiment or source file has been committed or pushed.

`report.py` checks endpoint and checkpoint hashes, accounts for every expected
arm and explicitly labels an incomplete cohort. `development_report/` will
contain all endpoints, rate summaries, paired descriptive contrasts and
common-state diagnostics. It must not be mistaken for twenty-seed confirmation.

## Required next decision

Review the complete development outcomes, exact-BP feasibility, bound contacts
and convergence before launching the larger cohort. Choose each rate from
development validation, freeze the budget and contrasts, then run twenty
fresh paired seeds. Test results must not select the rates. Retain failed
controls and nulls. Manuscript integration follows validation, not submission
of the cluster jobs.

The anatomy analysis concerns modeled electrical responses on measured arbors;
the functional analysis concerns observed response segregation. Neither is a
measurement of an endogenous learning error or inhibition-dependent plasticity.
