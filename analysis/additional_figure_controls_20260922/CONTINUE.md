# Additional controls: continuation record

User authorized strict-scalar Fashion-MNIST and raw-additive physical-depth reversal.
No training implementation or optimizer code was changed. Recurrent work remains deferred.

Frozen protocol: PROTOCOL.md; manifest: manifest.json; hash list: frozen.sha256.
Archive commit: 832d4ca (unpublished protocol snapshot).
Submission: preflight 47813573; main array 47813575 (0–89, throttle12, afterok preflight);
final analysis 47813579 (afterany array). All GPU jobs are kempner_eng/H200.

80 Fashion fits: ten fresh paired seeds22600–22609, both architectures, four rules
(strict scalar, old matched-width fallback, neuron, exact paths), unchanged180epoch recipe.
140 physical fits: H3/D1–D3 seeds10200–10209; H4/D1–D4 seeds10400–10409;
paired aligned/reversed, original180epoch/patience30 BP recipe. Exact archived sources,
with e90fb989 construction for repaired H4/D3. Both ee and ie feature ranges reverse,
as in the existing shunting control. Aligned pairs are rerun to separate hardware replay
drift from the placement contrast. Analysis never combines old aligned with new reversed.

Preflight runs22 two-epoch smoke fits, excluded from cohort. All rules, architectures,
and depths represented. Validates resources (physical66178 parameters,14336 active,
21760 candidate,2944 states), paired data hashes, seeds, finite results and credit modes.
Full run directory claims are exclusive; do not rerun a partial job in place. Preserve
all failures and record any identical-config infrastructure retry in a separate attempt.

Pending full integration:
- Fashion fresh source prefix fashion_strict_scalar_confirmatory; preserve historical
  fashion_feedback_ladder. Main Fig1E should use neuron-minus-strict, F exact-minus-neuron;
  S6C fresh four-rule ladder includes old fallback. Update source registry, plotted exports,
  caption, S2 methods/results, cohort/source tables, protocol/source release allowlists.
- Physical fresh source prefix physical_depth_additive_reversal. Use paired new aligned
  and reversed values for raw-additive placement effects in Fig7/S24. Keep old aligned
  descriptive rows traceable; do not imply shunting replays were repeated. Report current
  hardware (H200) and inherited budgets. Remove disclosed omission only after completion.
- Coordinate with ongoing CIFAR array47787999, analysis47804995 and existing pending
  figure integration at ../cifar_shunting_revision_20260922/pending_journal. Do not overwrite
  those candidates or claim completed data before audits. Scalar CIFAR seed22008 has a
  frozen convergence flag; retain it and address at complete-cohort review.
- Rebuild all figure exports, provenance, PDFs and archives only after final scientific
  interpretation and validation. Main remains pushed d88f3d7, no new paper release yet.

Source checkouts are clean and pinned; another session's staged root changes are untouched.
Submission receipt and all inputs are copied to journal analysis/additional_figure_controls_20260922
and the archive checkout. Durable raw outputs stay here. Do not add raw checkpoints to main.

Additional CPU initialization-only check: Slurm47814022, four representative configs, no fitting or outcome-based decisions. See initialization_checks.json and logs/init_47814022.log. Protocol and submission commits832d4ca,c84cc5d are not pushed. Statistics sanity checks passed (analysis_function_checks.json).

Important for S24 integration: current builder scripts/build_supplementary_figure_physical_architecture_native.py hard-codes the old four reversed rows, prints not run, and assumes the largest reversed-depth lead is0.8pp. Update these outcome-specific assumptions for the added raw-additive row; derive its markings and range from data without requiring a favorable result. All previous source records remain traceable.
