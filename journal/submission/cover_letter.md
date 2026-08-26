Dear Editors,

We submit “When dendritic structure helps local credit assignment” for consideration as an Article in *Nature Communications*.

Learning in the brain requires a neuronal teaching signal to be assigned to synapses distributed across a dendritic tree. We address a question left open by most biologically plausible learning theories: once limited feedback reaches a neuron, when does the tree improve its distribution within that neuron?

Our central contribution is a predictive signal–noise theory of restricted credit routing. We show that exact gradients in conductance-based neurons factor into synapse-local eligibility and a transported compartment error. A stochastic signal–gain–noise bound then predicts when restricted routes improve descent and when they must fail. Its operator utility predicts trained outcomes across independent route and depth experiments (Spearman ρ = 0.937 and 0.916), explaining positive effects, interior optima and nulls within one framework.

The experiments map the theory’s boundary. Neuron-specific feedback recovers most of the scalar-feedback deficit, whereas correct coordinate-to-arbor assignment and within-tree address provide smaller conditional gains. Depth alone reduces accuracy at fixed budgets. On a calibrated mechanism-matched task, however, an aligned serial divisive tree beats a resource-identical nonserial control by 31 percentage points; reversing task–tree alignment abolishes the effect. A fixed-depth extension shows a graded advantage for nested and flat multiplicative factors, but no backpropagation advantage—and a reversal under local credit—when the required ratio is already locally available. Flexible active- and total-parameter-matched point networks perform about seven points better than the tree, showing that dendrites provide a structured inductive and wiring resource rather than a unique expressive capability. A matched credit-coordinate ladder further shows that soma-broadcast autograd is within 0.64 points of full backpropagation under a common optimizer; optimizer regime and path specificity account for the larger raw separation from the local rule.

Reconstructed MICrONS arbors supply sparse, mainly coarse nested-subtree addresses, retaining most dense-field capture with approximately 7% of the feedback wiring; the anatomy-specific efficiency is approximately 2.7-fold at matched density. An outcome-independent cohort from a second MICrONS mouse reproduces the model-matched structural routing direction, while remaining a two-animal capacity result rather than population inference. Focal shunting localizes modeled credit only in permissive high-conductance regimes and is null at the textbook passive calibration. Measured visual responses show no reliable morphology-specific alignment. We regard these negative results as part of the contribution: they distinguish anatomical availability from task use and make the theory experimentally falsifiable. The paper closes with a quantitative branch-resolved focal-inhibition prediction and a six-animal reanalysis of signed neuron-specific coordinates.

An earlier version, “Shunting inhibition and dendritic branching shape local credit assignment” (arXiv:2607.03556), is under review at NeurIPS 2026. We will not submit this Article while concurrent review remains active; its resolved status and the related manuscript will be disclosed to the editor at submission. The Article is the canonical standalone account: it integrates the complete factorization and regular-tree foundation with the signal–noise theory, subtree-address factorial, physical-depth and point-network controls, reconstructed-anatomy and conductance boundary map, external animal analyses, and complete source-data and software package. The accompanying related-work statement documents overlap transparently.

We also disclose the related preprint “When branch-local shunting helps: a gain-load-alignment principle for dendritic E/I networks” (arXiv:2607.24990). That work concerns forward population coding; this Article concerns backward learning-credit coordinates, addresses and transport. Both use public MICrONS resources from the same mouse, and the manuscript identifies the distinct cohorts and estimands.

Reviewer-ready source data, code, contracts and provenance records accompany the submission under the MIT License. The submitted software archive is byte-identified and will be released as the same public, versioned repository and permanent archival record upon publication.

We believe the work will interest readers in cellular and systems neuroscience, learning theory and neuroAI because it turns a broad claim about dendrites into a quantitative signal–noise theory with explicit positive and negative domains.

Thank you for considering our work.

Sincerely,

Houman Safaai, Maceo Richards and Bernardo L. Sabatini

on behalf of all authors
