# Measured-response branch-model specification

`primary_four_channel.json` is the machine-readable contract for the result in
Fig. 5 and Supplementary Table `functional_boundary`. It fixes the CAVE join,
DANDI/NWB extraction, stimulus grouping, standardization, branch model,
initialization, optimizer, feedback dictionaries, projection tolerance, and
inference unit.

The complete seven-target manifest is
`../../reproducibility/task_derived_primary_targets.csv`. Each target has 464
trials and 280 unique condition hashes, with 136 repeated identities. The
model holds out 20% of complete identities rather than individual trials.

The target universe was the eight-cell morphology pilot. The prospective
criterion retained every target with a DANDI scan and directly connected,
functionally imaged excitatory partners; seven targets qualified. The primary
pipeline additionally checks that the extraction has one target, matching
partner/contact counts, at least five partners for the functional-topology
screen, at least two sites after task filters, and a nonempty inhibitory route.
No reliability or manual-match filter was used in the primary analysis. The
archive does not contain a deterministic rule for selecting among multiple
eligible scans of a target, so the exact target/session/scan pairs in the CSV
are the frozen pilot assets. The eight-cell accounting and the evidenced
reason for the one target exclusion are in
`../../reproducibility/original_cohort_functional_accounting.csv`.

Partner response similarity is Pearson correlation across vectors of mean raw
fluorescence for the 136 identities with at least two trials. Repeat
reliability is a per-ROI Spearman correlation between condition means formed
from alternating occurrences. For the controlled ancestry statistic, response
similarity and shared-path fraction are ranked, as are `log1p` Euclidean
distance and `log1p` absolute soma-path-length difference. Each ranked focal
variable is residualized by ordinary least squares on an intercept and the two
ranked controls, and the residuals are correlated by Pearson correlation.

The end-to-end code expects service-derived inputs under `external_data/` and
writes intermediate and final products under `reproduced_results/`. The main
stages are documented in `../../code/task_derived/README.md`.

If one partner has contacts on multiple dendritic segments, all contacts
contribute to its reported total contact size and synapse count, but the model
still uses one feature and one site. The site is the segment with greatest
summed contact size, with synapse count as the tie-breaker.
