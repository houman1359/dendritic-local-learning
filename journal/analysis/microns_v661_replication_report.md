# Public-v661 MICrONS replication audit

## Decision

The local cache did not contain enough complete, non-overlapping current-materialization reconstructions for a larger replication. It contained 11 skeletons but only eight complete skeleton-plus-synapse pairs. One extra skeleton is the historical root of a cell already represented among those eight, and two extra functional targets have no cached whole-cell synapse file.

An official public static route does support a disjoint historical replication without CAVE authentication. The frozen 55-cell V1 L2-L5 IT/ET list maps uniquely to MICrONS version 661. Excluding the original eight by stable nucleus identifier leaves 47 cells. All 47 public SWC skeletons and postsynaptic meshworks were fetched successfully and analyzed with direct presynaptic coarse E/I labels only.

This is valid as a disjoint-cell robustness analysis of structural routing and the focal in-model mechanism. It is not an independent animal or population sample: all cells come from the same MICrONS mouse, the 55-cell universe descends from a convenience structural pilot, and v661 is a historical reconstruction rather than materialization 1822.

## Cohort provenance

- Frozen universe: 55 V1 excitatory L2-L5 IT/ET cells.
- Original pilot excluded by nucleus ID: 8.
- Public-v661 replication cells fetched and routed: 47.
- Incoming synapses in meshworks: 234,301.
- Direct E/I calls before spatial mapping: 11,024 (4.71%).
- Direct E/I synapses passing the spatial map: 10,516.
- Per cell, the direct-type set contained a median of 212 synapses (range 82-1372); every routed tree retained at least 14 excitatory-bearing and 25 inhibitory-bearing segments.
- Replication cell types: 34 L2IT, 2 L3IT, 10 L4IT, and 1 L5ET; the cohort is therefore layer/type imbalanced.
- Classification: matching v661 `baylor_log_reg_cell_type_coarse_v1` calls only; no spine/shaft proxy.
- Access: versioned public static URLs and SHA-256 hashes are recorded in `cohort_manifest.csv` and `replication_summary.json`. The parent MICrONS dataset is described by DOI 10.1038/s41586-025-08790-w. DANDI:000402 is functional imaging data and is not an accession for the structural files used here.

The original eight were a hand-authored, layer-diverse pilot selected from an existing anatomy/functional-coregistration cohort. The list was present before its recorded routing outcomes, but it was not an externally preregistered or population-random sample. The focal endpoint was specified after selection of the structural pilot and before the focal run.

The inherited 55-cell universe also is not population-wide. It consists of the V1 excitatory cells with stable nuclei in a 64-target export. Those 64 targets were the most represented targets in the first 50,000 rows returned from `vortex_compartment_targets` at materialization 1718. Accordingly, the journal should call the 47 cells a frozen, disjoint replication cohort, not a representative MICrONS sample.

## Structural routing replication

At eight feedback channels, morphology paths captured 75.6% of modeled field energy with 8.1% of dense feedback wiring. The dense PCA oracle captured 83.9%; the cellwise morphology-to-oracle capture ratio was 89.5% on average.

The directly typed reanalysis of the original eight cells captured 69.6% with morphology paths, compared with 78.2% for its dense oracle. Thus the disjoint v661 cohort reproduces the ordering under the same direct-type-first classification, although its effect size should not be pooled with the current-materialization cohort.

Morphology capture exceeded random paths by 45.3 percentage points (95% cell-bootstrap CI 41.9 to 48.8; 47/47 cells), depth bins by 31.5 points (CI 28.0 to 34.9; 47/47), and ancestry shuffles by 42.5 points (CI 40.0 to 44.9; 47/47).

The earlier exploratory association between maximum depth and credit-kernel participation rank does not replicate: Spearman rho = -0.093, p = 0.535, n = 47. It should not be promoted as a journal-level result.

## Focal perturbation replication

Forty-five of 47 cells had at least one eligible focal site, giving 235 sites. At dose 1, shunting localization exceeded the current-matched additive perturbation by 0.138 on average across cells (95% cell-bootstrap CI 0.116 to 0.162; 45/45 cells; two-sided Wilcoxon p = 5.68e-14). Mean localization was 0.151 for shunting and 0.013 for the additive control.

The depth-shuffled relation control is defined for 40 cells and 230 focal sites. True descendant relations exceeded shuffled relations by 0.118 (95% CI 0.094 to 0.144; 39/40 cells, with one numerical tie; p = 5.26e-08). Five additional one-site cells remain valid for the shunt-additive comparison but cannot supply an independent within-cell shuffled template. Two cells had no eligible focal site; all exclusions are explicit in `cohort_manifest.csv`.

The original focal summary silently reduced the shunt-additive comparison to the 40-cell subset used by the shuffle control. The packaged primary table corrects this: shunt versus additive uses all 45 eligible cells; only the topology-shuffle comparison uses 40.

## Numerical checks

Across 141 finite-difference checks, the maximum absolute gradient error was 1.22e-08. The maximum relative error among checks with absolute analytic gradient at least 1e-8 was 0.000461. The larger raw relative maximum (0.00717) is caused by division by an analytic gradient near numerical zero and should not be quoted without that qualification. The maximum soma-clamp error was 1.73e-18, and the minimum conductance-matrix eigenvalue was 0.260.

## Other local MICrONS caches examined

- `drafts/dendritic-credit-routing/data/microns_morphology`: 11 skeleton files, eight complete skeleton-plus-synapse pairs, and one coarse cell-type table. The exact cell-level disposition is in `local_current_cache_inventory.csv`.
- `drafts/dendritic-pop-draft/results/microns_cave_export_scaled` and its copied `dendritic-credit-routing/imported/population` version: 9,486 compartment-tagged rows across 64 targets. These provide soma/shaft/spine pools, not dendritic parent-child connectivity, so they cannot support ancestry or descendant perturbation tests. The key tables in the two locations are byte-identical.
- `drafts/dendritic-pop-draft/results/microns_hf_dataset/microns.h5`: a 20.6-GB functional stimulus/session cache with top-level groups `brain_areas`, `sessions`, `types`, and `videos`; it does not contain skeleton topology or whole-cell synapse locations.
- Existing routing result directories are alternative analyses of the same original eight cells, not additional biological observations.

No other complete local skeleton-plus-synapse cohort was found in the drafts tree.

## Recommended journal use

1. Present the current-materialization eight-cell cohort as the discovery pilot and this historical v661 cohort as a frozen, disjoint-cell replication.
2. Report routing at eight channels as captured energy, with all three matched controls and wiring density. Do not pool the two cohorts as 55 statistically independent observations.
3. Use 45 cells for the shunt-additive focal contrast and 40 for the depth-shuffle control.
4. State the 4.71% direct-type coverage and the historical-release/convenience-cohort limitations next to the result, not only in a general limitations section.
5. Do not retain the exploratory depth-rank claim, which was null in the larger cohort.
6. Describe the focal experiment as a mechanistic passive-network intervention on measured anatomy, not an in vivo perturbation or evidence of biological learning.
7. Before making the replication central, archive the normalized source tables or their exact static URLs and hashes with the submission. The source tables here already contain those identifiers.

## Files

- `cohort_manifest.csv`: all 55 frozen candidates, original-pilot exclusions, v661 roots, file hashes, routing inclusion, and focal eligibility.
- `local_current_cache_inventory.csv`: all locally cached current-pipeline skeleton roots and whether a matching synapse file exists.
- `cell_level_primary.csv`: the 47 replication cells and their eight-channel routing outcomes, plus focal contrasts when available.
- `replication_summary.json`: machine-readable estimands, uncertainty intervals, validation, sources, and limitations.
- Raw routing and focal outputs remain in their respective subdirectories.
