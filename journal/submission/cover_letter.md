Dear Editors,

We submit “A signal–noise phase theory of when dendritic structure helps local credit assignment” for consideration as an Article in *Nature Communications*.

Learning in the brain requires a neuronal teaching signal to be assigned to synapses distributed across a dendritic tree. We address a question left open by most biologically plausible learning theories: once limited feedback reaches a neuron, when does the tree improve its distribution within that neuron?

Our central contribution is a predictive credit-operator theory. We show that exact gradients in conductance-based neurons factor into synapse-local eligibility and a transported compartment error. A stochastic signal–gain–noise bound then predicts when restricted routes improve descent and when they must fail. Its prespecified utility predicts trained outcomes across independent route and depth experiments (Spearman ρ = 0.937 and 0.916), explaining positive effects, interior optima and nulls within one framework.

The experiments map the theory’s boundary. Per-neuron feedback recovers most of the scalar-feedback deficit, whereas correct ownership and within-tree address provide smaller conditional gains. Depth alone reduces accuracy at fixed budgets. On a calibrated mechanism-matched task, however, an aligned serial divisive tree beats a resource-identical nonserial control by 31 percentage points; reversing task–tree alignment abolishes the effect. Flexible active- and total-parameter-matched point networks perform about seven points better than the tree, showing that dendrites provide a structured inductive and wiring resource rather than a unique expressive capability. A matched credit-coordinate ladder further shows that soma-broadcast autograd is within 0.64 points of full backpropagation under a common optimizer; optimizer regime and path specificity account for the larger raw separation from the local rule.

Reconstructed MICrONS arbors supply sparse, mainly coarse ancestry addresses, retaining most dense-field capture with approximately 7% of the feedback wiring; the anatomy-specific efficiency is approximately 2.7-fold at matched density. These reconstructions all come from one mouse. Focal shunting localizes modeled credit only in permissive high-conductance regimes and is null at the textbook passive calibration. Measured visual responses show no reliable morphology-specific alignment. We regard these negative results as part of the contribution: they distinguish anatomical availability from task use and make the theory experimentally falsifiable. The paper closes with a quantitative branch-resolved focal-inhibition prediction and a six-animal reanalysis supporting the more basic requirement for signed neuron-specific coordinates.

An earlier manuscript, “Shunting inhibition and dendritic branching shape local credit assignment” (arXiv:2607.03556), contains the foundational factorization and idealized regular-tree experiments and was submitted to NeurIPS 2026. **Before journal submission, replace this sentence with the exact final NeurIPS status and confirmation that concurrent-consideration requirements are satisfied.** The present Article adds the credit-operator phase theory and validation, subtree-address factorial, physical-depth and point-network controls, reconstructed-anatomy and conductance boundary map, external animal analyses, and complete source-data and software package. The accompanying extension statement details retained and new material.

We also disclose the related preprint “When branch-local shunting helps: a gain-load-alignment principle for dendritic E/I networks” (arXiv:2607.24990). That work concerns forward population coding; this Article concerns backward learning-credit coordinates, addresses and transport. Both use public MICrONS resources from the same mouse, and the manuscript identifies the distinct cohorts and estimands.

Reviewer-ready source data, code, contracts and provenance records accompany the submission. **Before submission, replace this sentence with the public repository URL, immutable version, Zenodo DOI and MIT license.**

We believe the work will interest readers in cellular and systems neuroscience, learning theory and neuroAI because it turns a broad claim about dendrites into a quantitative phase theory with explicit positive and negative domains.

Thank you for considering our work.

Sincerely,

Houman Safaai, Maceo Richards and Bernardo L. Sabatini

on behalf of all authors
