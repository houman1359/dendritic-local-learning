# Reproducibility and completion record

The prospective follow-up is complete: twenty fresh seed blocks,
2026090801–2026090820, produce 1,080 initialized trajectories, 2,160 selected
endpoint views and 12,960 recorded curve rows. Every condition continues to
16,384 updates. No failed or unfavorable seed is removed. The primary budget
is 4,096 updates and the primary Adam rate is 0.03; the other two rates and
extended budget were frozen alongside the primary design.

`protocol.json` has SHA256
`c441998f6f91e1f8782b2918689806e119af567e479bcd5c94069dca837a2413`.
The protocol and numerical adapter were committed in `c6d3b2b`; the added
transitive test dependency was committed in `37dbfe9` before numerical jobs
started. The protocol explicitly records exposure to the panel's earlier
exploratory outcomes. Those outcomes are prior evidence, not follow-up results.
The excluded implementation seed is 2026090799, and the excluded historical
replay seed is 2101.

Ten mechanism tests passed in Slurm job 45289328. They check the independent
forward/gradient adapter against finite differences and the released model,
the local gate's preservation of proximal eligibility, the inhibitory-gain
failure caused by also gating proximal credit, and absence of exact-path
access in the local gradient branch. Job 45290256 replayed both tasks under
four existing rules through 4,096 updates. Exact, unit-broadcast and initially
calibrated trajectories reproduced parameter arrays, Adam moments and outcomes
bit for bit. Independent three-pattern projection differed by at most
3.73 × 10⁻¹⁴ in parameters and 1.56 × 10⁻¹⁷ in checkpoint NMSE. All eight
comparisons passed. These validation jobs used the cluster's test partition.

The fresh array, job 45289612 with twenty tasks and concurrency ten, ran on a
permitted production CPU partition. Scheduling changes affected only
partition, concurrency and wall-time limits; `execution_provenance.json`
records them. All scientific settings remained frozen. Per-seed audits retain
source, configuration, input, initialization and checkpoint hashes, software
versions, runtime records and all bound-contact diagnostics.

Portable replay job 45306543 reproduced opposed-task hard-gate and two-leaf
oracle fits for fresh seed 2026090801 through 4,096 updates. Both gave zero
maximum difference in canonical parameter arrays and selected endpoint NMSE.
Their complete outputs and reports are under `portable_validation/`. Replays
validate implementation and portability; they do not add independent
observations to the twenty-seed experiment. The portable entry point requires
no Git repository and writes to a separate output directory.

The analysis-helper snapshot was taken at 03:24:28 UTC after the first fresh
worker had started at 03:22:49 and completed at 03:24:24. Its initial wording
incorrectly called it a pre-execution snapshot. This was corrected at 03:25:14,
with the original wording and scheduler evidence retained in
`analysis_code_snapshot.json`. The snapshot preceded the agent's inspection
of fresh outcome values; it was not a pre-execution freeze. The actual
protocol, numerical adapter and statistical definitions were separately
frozen before execution.

The six numerical summary CSV files retain their initial hashes through all
presentation changes. `rendering_amendment.json` and
`final_rendering_record.json` distinguish visual changes from frozen science.
`report_initial_render.py` authenticates the earlier analysis-code snapshot.
The initial numerical `completeness_audit.json` retains historical figure
hashes. Current figures and their exact input sources are in
`figure_provenance.json`, and `completion_gate.json` verifies current figures,
all scientific input hashes, unchanged summaries and validation evidence.
Both native figures were visually inspected; normal text is at least 6.8 pt,
and no text lies outside either PDF page. Supplementary panel E is explicitly
an earlier cohort's diagnostic and is mapped to its separate original data.

Main Results, Methods, figure-caption and supplementary LaTeX snippets are
included for integration. Local-gate outcomes support an implementable credit
coefficient in this supplied-context circuit. They do not establish endogenous
discovery of the context, a general learned encoder, or convergence beyond the
tested rates, bounds and budgets. The two-leaf control retains oracle
coefficients and separate unit proximal credit. The proximal-gating failure
control deliberately freezes its two inhibitory gains through its interaction
with local eligibility.
