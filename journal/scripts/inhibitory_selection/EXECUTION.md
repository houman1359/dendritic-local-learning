# DendriNet selection follow-up — 18 September 2026

## Completed experiments

- Development: 180 fits, three paired seeds, 2,048 updates.
- Fresh cohort: 400 fits, twenty paired seeds, 4,096 updates.
- Common-rate sensitivity: 60 additional fits on the same fresh seeds.
- Earlier exploratory matrix: all 540 fits retained separately, including
  mixture failures and its conditional-additivity representation floor.

No W&B was used. Runtime snapshots, checkpoints, scheduler logs and raw
results are under the kempner_project_b `HS/LOCAL_LEARNING` directories
`inhibitory_selection_20260918` and `inhibitory_credit_transfer_20260918`.
Only compact publication tables and analysis/model code were added to the
paper repository. The study imports the immutable original production
DendriNet snapshot, rather than the changing parent worktree.

## Scheduler and verification

The pending development array 47028353 never executed; it was held and
replaced by single-allocation CPU batching because the test partition's
submission limit counts array elements. Completed training jobs:
47029329 (development), 47030262 (fresh), 47031187 (common-rate sensitivity).

Initial verification job 47030395 exhausted its 5 GB allocation. This was
analysis, not training: no endpoint was excluded or rerun. Forward-hook
closures retained network diagnostics between checkpoints. The revised
publisher explicitly collects each reconstructed network, and its separate
`reporting_revision_02` snapshot preserves the original failed verifier.
Job 47033019 reproduced all 400 ordinary-test and five-severity outcomes
exactly (maximum absolute discrepancy zero). Job 47033033 verified all
sixty sensitivity checkpoints to absolute tolerance 1e-10. Superseded
dependency-only collectors were canceled; their identities remain in logs.

## Scientific outcome and publication homes

At development-selected rates and distractor severity three, mean NMSE
was 0.010506 for broadcast, 0.009716 for uniform RMS, 0.0001213 for the
relative-resistance gate and 0.0001191 for exact BP. Both primary paired
gate advantages were positive in 20/20 seeds, with Holm-adjusted exact
sign-flip P = 3.8147e-6. Wrong-branch gating gave 0.06017.

Both primary advantages persisted in 20/20 seeds at each common Adam rate,
0.03 and 0.1, but effect magnitudes were rate-sensitive. Shunting also
outperformed tonic and reference-voltage-matched current forward controls.
These effects concern synthetic out-of-distribution distractors, not a
general natural-image or biological-learning advantage.

The nonlinear-parent interaction task is an important negative result:
exact BP reached ordinary-test NMSE 5.06e-5, whereas the resistance-only
gate gave 0.05109. The missing input-dependent parent activation gain
limits this approximation; failure is not a proof against all local rules.

The population curves and forward controls are Figure 5H,I; A–G retain
the original experiments. Abstract, Introduction, Results, Methods and
Discussion distinguish these levels. Supplementary Section S5 and Table
S14 give the complete conditions, equations, primary intervals, matched-rate
sensitivity, common-state diagnostics, bound contacts and the earlier pilot.
Other figures and all existing supplementary figure numbers are retained.

Publication tables are `source_data/curated_publication/inhibitory_selection_*`
and `inhibitory_transfer_pilot_*`. The figure renderer is
`scripts/conductance_local_gate/figure.py`, with the new native panels in
`figure_panels.py`. Data reproduction tests are in
`tests/test_inhibitory_selection.py`; model and inference tests are beside
the experiment scripts. No commit or push is performed by these scripts.

## Final paper validation

The combined PDF was rebuilt and the revised figure, Results paragraph,
supplementary task definition and Table S14 were visually inspected.
All 797 original Figure 5 A–G display-source rows are exactly unchanged.
The journal test suite passed 297 tests with one skip; six additional
model/inference tests passed beside the study scripts. LaTeX, panel-letter,
row-separation, lineage, reproducibility, citation, overlap, format and both
strict submission audits passed. The final LaTeX logs contain no undefined
references or overfull boxes. Figure 5's native strict canvas audit also
passed. These were checked as individual audit components, rather than
claiming that the earlier failed single `make audit` invocation passed.

Passing the format script does not make the manuscript submission-ready:
its narrative estimate is 9,362 words, above the author's 8,000-word working
target. This experiment-focused pass did not compress unrelated sections,
regenerate release ZIP archives, or publish the repository.
