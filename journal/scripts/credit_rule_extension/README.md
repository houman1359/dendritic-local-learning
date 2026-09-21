# Balanced extension of the algebraic credit-rule bridge

This follow-up reuses all twenty original seed blocks after their 1,024-update outcomes had been examined. Its protocol was frozen and committed before the longer trajectories were run. It is a protocol-led extension of a retrospectively selected cohort, not a fresh confirmatory sample.

All 720 unique trajectories are retained: three tasks, four rules, SGD and Adam, and the union of the original development-selected and common learning rates. Every trajectory is replayed from initialization and run to 16,384 updates. The inherited bounds, clipping, examples and minibatch streams remain fixed. Recorded original states and numerical metrics through 1,024 must reproduce before a run is accepted. The raw `phase=fresh` field is inherited from the original experiment; it does not claim fresh seed selection for this follow-up.

Run from the journal directory, with numerical-library thread counts set to one:

```
python -B scripts/credit_rule_extension/run.py check
python -B scripts/credit_rule_extension/run.py run --seed 211200 --protocol-sha256 HASH
python -B scripts/credit_rule_extension/analyze.py
python -B scripts/credit_rule_extension/audit.py
python -B scripts/credit_rule_extension/figure.py
```

The scientific runner checks that the protocol hash exists in the reachable paper commit at launch. Completed outcomes are immutable; a replay failure or partial job requires explicit investigation. `smoke.py` uses one historical development seed for a short execution benchmark and is excluded from every science table.

The primary estimand is terminal test NMSE at 1,024, 4,096, 8,192 and 16,384 updates. Secondary checkpoints minimize validation NMSE among declared observations within each budget, with ties assigned to the earliest observation. Rates and checkpoint choices never use test outcomes. Condition summaries report means, medians, descriptive paired bootstrap intervals and seedwise wins; full quartic distributions, clipping and bound contacts remain available. The independent audit recomputes test, validation and complete-domain population predictions directly from saved states for every selected outcome.

The standalone figures are supplementary candidates. They do not overwrite the released interaction figure or its original learned-credit capture analysis. Longer training and historical rate choices remain explicit constraints; neither unchanged nor reversed rankings establish an asymptotic winner.

For a restored reviewer archive without Git, use the post-experiment portability wrapper:

```
python scripts/credit_rule_extension/portable_run.py --seed 211200 --verify-only
python scripts/credit_rule_extension/portable_run.py --seed 211200 --output-root /tmp/credit_extension_seed_211200
```

Restore Source Data to its original article-relative paths first. The wrapper authenticates all original inventory entries and the selection record through `code/release_noise/release_hashes.py`; a changed release copy requires its declared original-to-released hash chain. It never rewrites canonical states or invents Git metadata. A full call retains all 36 conditions in the selected seed block and all 16,384 updates. Original checkpoint arrays, task metadata, numerical curves and diagnostics must reproduce within the unchanged absolute tolerance. Differences in numerical libraries may therefore cause an explicit failed replay, whose outputs and audit remain available for investigation.

The wrapper also authenticates `scripts/morphology_structure/constructive.py`, a transitive import omitted from the original top-level hash inventory. Its bytes were independently checked against the actual scientific launch commit; `portable_dependency_provenance.json` records this release-time addition without changing the earlier freeze. An explicitly excluded `--excluded-smoke-steps 64` execution checks loading and the first four historical checkpoints. The relocated validation used a temporary, Git-free copy, replayed all three tasks exactly, and rejected a deliberately modified model source. These checks do not add scientific observations.
