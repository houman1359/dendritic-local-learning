# Additional figure controls, 22 September 2026

These post-review extensions close two disclosed coverage gaps. They use existing
training implementations without performance or optimizer changes. Outcomes will
be reported regardless of direction. Historical results remain archived separately.

## Fashion-MNIST: strict scalar feedback

Run ten fresh paired seeds (22600–22609) in each of the shunting and raw-additive
architectures. Each seed receives four terminal-credit rules: strict scalar
(`scalar`), the historical matched-width fallback (`per_soma`), neuron-specific
feedback (`per_soma_shared`), and exact paths (`path_transport`). Exact paths here
retain the local learning recipe and decoder update; they are not a BP cohort.
All 80 runs use clean implementation e516c7fec3169253ff8c14bc5f4ab1325469e4f5.
Fresh controls avoid mixing new-source scalar results with the dirty historical
Fashion-MNIST checkout. These are an extension, not an exact historical replay.

Inherit the original architecture-specific recipe: one 128-neuron [3,3] layer,
40 excitatory and 20 inhibitory contacts per branch, linear ten-class readout,
180 epochs, batch 256, Adam, no early stopping, and validation-selected state.
Keep original learning rates and initialization/calibration policies. All random
streams are paired explicitly. Use the standard 48,000/12,000 train/validation
split and 10,000 official test examples, pixels divided by 255, without augmentation.
Cached IDX files must match the recorded official compressed MD5s and raw SHA-256s.

Primary outcome: test accuracy, in percentage points. Primary contrasts are
neuron minus strict scalar in each architecture; use 50,000 paired-seed bootstrap
resamples (percentile 95% intervals), exact two-sided sign-flip tests, and Holm
adjustment across these two tests. Report positive-seed counts. The earlier
directional criterion (at least 8/10 positive seeds and interval above zero in
both architectures) is a descriptive continuity check, not a replacement for
the adjusted tests. Exact paths minus neuron, fallback minus strict scalar,
and neuron minus fallback are descriptive contrasts. No rate search is added.

## Physical depth: reversed raw-additive placement

Run aligned and reversed placement together at every available depth: three-tier
tasks at D1–D3 (seeds 10200–10209) and four-tier tasks at D1–D4 (10400–10409).
There are 140 runs, or 70 aligned/reversed pairs. Preserve each archived aligned
configuration and its source: a99c3a777f99913e13dfe673a3f3a28bfe3566af, except repaired
four-tier D3, which uses e90fb9896daa95df4aad4c0d92acc0cd65bcd750. Preserve its explicit
tier grouping. Reverse feature ranges on both ee and ie pathways exactly as in
the corresponding existing shunting reversal recipe; retain all tier inventories,
branch factors, synapse counts, and initialization/training settings. This is the
paper's established placement intervention, not a new inhibitory-only intervention.

Each aligned/reversed pair differs only in those feature ranges and output names.
Keep the original 180-epoch maximum, patience 30, Adam/BP recipe, validation selection,
and ten paired seeds. Diagnostic dataset fingerprints and absolute output paths
are the only additional bookkeeping. Run both placements on the same allocated
H200. Report differences from archived aligned outcomes separately as replay drift;
do not substitute archived aligned outcomes into the new paired contrasts.

Report reversed minus aligned accuracy at each depth, with paired bootstrap
intervals and descriptive exact sign-flip P values. The two primary interaction
contrasts are (aligned deepest minus aligned D1) minus (reversed deepest minus
reversed D1), one for each hierarchy. Use exact sign-flip tests and Holm correction
across these two interaction tests. Seedwise resampling preserves all four cells.
These results describe performance under the inherited training budget, not a
convergence guarantee or a newly independent replication of the old task seeds.

## Execution and integrity

Freeze manifest, all 220 configs, protocol, launcher, validator, analyzer, and data
manifest by SHA-256 before submission. One job groups all four Fashion rules for a
seed, or both placements for a depth/seed: 90 jobs total. Use kempner_eng, account
kempner_dev, H200, one GPU, eight CPUs, 64 GB, four hours, array throttle 12.
Preflight every rule/architecture and every depth/placement at one representative
seed; initialize, train for two epochs only, and exclude these smoke outputs from
the cohort. Full jobs depend on successful preflight, including resource equality,
paired data fingerprints and resolved seeds. Smoke accuracy is not a selection
criterion. Preserve failures and logs; retry infrastructure failures only with the
same frozen config, recording attempts. Do not overwrite completed or partial runs.

Analysis requires complete seed groups, matching config hashes, source identities,
finite losses/metrics, expected credit mode, equal resources within comparisons,
and paired dataset fingerprints. Record per-epoch timing, best epoch, training
length, last-ten validation slope, cap contact and calibration/fallback messages.
Budget contact is reported and does not justify favorable-seed exclusion. A failed
integrity check blocks publication until explained; it does not silently delete a
run. Any substantive protocol amendment is dated and preserves the initial plan.

Integrate the new Fashion controls into Fig. 1 and the image-task supplement, and
the placement comparison into Fig. 7/S24 as space allows. Update cohort/source
indexes, captions, Methods, source data, and release archives after full analysis.
Do not prewrite an interpretation of the unobserved outcomes.
