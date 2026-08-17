# Expanded MICRONS functional and inhibitory analysis contract

## Central questions

1. Do connected excitatory synapses requiring similar task updates occupy
   shared dendritic routes more often than expected from geometry and cell
   identity?
2. Are dendrite-targeting inhibitory synapses positioned where they can
   selectively regulate large or functionally distinct credit communities?

## Cohort construction

The analysis begins from all proofread excitatory target cells satisfying:

1. complete retained dendritic skeleton inside the analysis volume;
2. an unambiguous soma/root and valid synapse-to-skeleton mapping;
3. high-confidence functional coregistration or an eligible digital twin;
4. a minimum number of functionally characterized connected excitatory
   partners, frozen after a feasibility count but before endpoint evaluation;
5. sufficient dendritic inhibitory contacts for the placement endpoint; and
6. no overlap with any target used for exploratory threshold selection.

All candidates and every exclusion will be written to a manifest. MICrONS is
one mouse; cells are nested samples and do not create independent-animal
replication.

## Structural inhibitory-census analysis

Before evaluating functional endpoints, we will use the independent 47-cell
reconstruction cohort to define a structural overlap with the published
MICrONS inhibitory census. This analysis is a capacity and organization test,
not evidence of learning. Its cohort is the full stable-cell-ID intersection;
no cell is selected by an outcome.

Three endpoints are fixed before analysis:

1. For each presynaptic inhibitory cell to target-cell connection, retain one
   representative contact per axonal clump and measure pairwise tree distance
   and shared path. Compare with target-, compartment-, and path-distance-
   matched input locations. The target cell is the primary inferential unit.
2. Quantify the fraction and number of all mapped input synapses in the
   descendant domain of each inhibitory contact. Compare observed contacts to
   the same matched location null, and report DistTC and PeriTC contacts
   separately.
3. Construct the binary descendant-route dictionary exposed by actual
   inhibitory locations. Report capture at 1, 2, 4, 8, and 16 routes relative
   to random actual locations, depth bins, ancestry shuffles, and a dense SVD
   oracle. Eight routes is the primary fixed-count summary when available.

Known column-interneuron contacts are a sampled subset of all inhibition. All
remaining inputs are therefore called generic inputs, not excitatory inputs.
Connection-level observations remain nested within one reconstructed mouse.

## Functional signals

Two pre-specified views will be retained:

1. measured repeated responses for reliability-aware analyses; and
2. digital-twin natural-movie and Monet response vectors for broad coverage.

Measured and modeled responses will never be pooled without labeling. Stimulus
splits are made before constructing covariance or fitting a task.

## Credit definitions

1. **Encoding credit:** gradient of a held-out target-response prediction
   objective with respect to connected presynaptic synapse weights.
2. **Discrimination credit:** gradient for a fixed visual stimulus or context
   discrimination objective.
3. **Response covariance control:** covariance of presynaptic responses without
   a learning objective, reported separately from credit covariance.

The primary endpoint uses encoding credit. Other endpoints determine whether
any anatomy relation is specific to credit rather than ordinary response
similarity.

## Excitatory topology tests

For pairs of connected presynaptic partners or contacts, test whether credit
covariance predicts:

1. shared major-branch identity;
2. shared ancestry length;
3. lowest-common-ancestor depth;
4. route-overlap fraction; and
5. selection into the same sparse morphology dictionary atom.

Primary controls preserve target cell and match cable distance, soma path
length, branch order, cortical layer, synapse count, presynaptic functional
reliability, and marginal gradient variance. Statistical inference is grouped
by postsynaptic target, with target-level bootstrap intervals and
within-target quadratic-assignment permutations.

## Inhibitory credit-control centrality

For an inhibitory contact or candidate branch point (k), define a descendant
set (D(k)). Credit-control centrality is computed from the held-out credit
covariance among excitatory contacts in (D(k)), the contrast between
(D(k)) and nearby sister subtrees, and the modeled conductance required to
change descendant route gain. The exact formula and normalization will be
frozen before comparing observed inhibitory locations.

Primary comparisons:

1. observed dendrite-targeting inhibitory contacts versus depth-, distance-,
   compartment-, and target-cell-matched candidate locations;
2. dendrite-targeting versus perisomatic inhibitory motif groups;
3. branch points separating high-covariance communities versus other branch
   points; and
4. true descendant relations versus depth-preserving relation shuffles.

Synapse count, inhibitory axon identity, target cell, and spatial density are
explicit covariates. The inhibitory neuron or target cell, not individual
synapse, is the inferential unit depending on the endpoint.

## Required figures

1. dataset and cohort flow diagram;
2. at least six full reconstructed examples spanning layers and morphology;
3. synapse-level maps colored by functional credit community;
4. observed inhibitory locations overlaid on predicted control centrality;
5. held-out covariance versus ancestry with matched nulls;
6. per-cell effects and cohort estimates;
7. cell-class and reliability sensitivity panels; and
8. explicit examples of positive, null, and reversed alignment.

## Decision rules

Anatomical alignment is supported only by held-out credit statistics, not by
training-set optimization. Inhibitory placement is supported only if observed
contacts exceed matched location nulls and the result is not explained by
depth or generic synapse density. If only modeled route capacity replicates,
the claim remains capacity rather than biological organization.
