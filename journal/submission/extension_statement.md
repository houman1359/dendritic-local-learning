# Relationship to the earlier preprint and conference submission

## Works being disclosed

- Earlier work: *Shunting inhibition and dendritic branching shape local credit assignment*, arXiv:2607.03556, submitted to NeurIPS 2026.
- Related but distinct work: *When branch-local shunting helps: a gain-load-alignment principle for dendritic E/I networks*, arXiv:2607.24990. This studies forward population readout rather than backward credit assignment.
- Proposed journal Article: *A signal--noise phase theory of when dendritic structure helps local credit assignment*.

As of 31 July 2026, the NeurIPS submission remains in discussion. This journal package is preparatory and will not be submitted while concurrent consideration continues. The status and citation will be updated before journal submission; if a proceedings version is published, it will be cited and supplied to the editor.

## Scientific relationship

The earlier work asks whether the local derivatives of a conductance-based dendritic tree give a biologically interpretable learning rule, and how restricted feedback performs in idealized regular trees. Its reusable core is the exact factorization

\[
\frac{\partial \mathcal L}{\partial g_i}
=x_iR_n^{\mathrm{tot}}(E_i-V_n)
\frac{\partial\mathcal L}{\partial V_n},
\]

together with the path recursion for the compartment error and controlled artificial-network experiments.

The journal Article asks a broader and materially different question: whether real dendritic topology supplies a sparse address map for credit, whether local conductance can selectively regulate that map, and when anatomical routes align with task-derived credit. It integrates the regular-tree mechanism with reconstructed anatomy, a frozen disjoint-cell robustness cohort, an algebraic test of the focal mechanism, independent functional observations and a formal topology–task matching criterion.

## Substantial additions in the journal Article

| Addition | Evidence or output | Why it is scientifically new relative to the earlier work |
|---|---|---|
| Topology-matched credit-routing theory | An ancestry dictionary, weighted field-capture measure and conditional one-step descent bound | Converts an ancestry-based route-sensitivity proxy into a quantitative communication-capacity hypothesis under fixed feedback channels and wiring; the descent interpretation applies when the field is an exact gradient in the stated coordinates. |
| Feedback-identity intervention | Fifteen paired seeds per architecture comparing scalar fallback with neuron-indexed sharing, plus same-checkpoint gradient diagnostics | Separates loss of neuron identity from local eligibility and exact within-tree transport. Accuracy improved in all 15 pairs in both architectures; diagnostic gradient cosine rose from 0.012 to 0.708 in shunting trees and from approximately zero to 0.625 in additive trees. |
| Prospective depth-by-feedback factorial | 640 current-code training runs crossing two tasks, two neuron models, four depths, three local feedback fields and matched backpropagation at ten paired seeds; 480 runs retained by an outcome-independent input rule | Establishes that the neuron-indexed result is not confined to one architecture or depth. Across the 12 retained task--model--depth cells, neuron-indexed feedback improves over a scalar teaching signal in all 120 paired comparisons, while exact three-factor local learning remains within 0.14 percentage points of matched backpropagation. |
| Bandwidth-matched routing control | 160 frozen runs; 120 retained after applying the same input-validity rule | Separates feedback dimension from anatomical assignment. Correct routing improves accuracy by 0.15--0.70 percentage points and is favored in 58/60 retained paired seeds, showing a smaller routing-map effect beyond the dominant coordinate-bandwidth gain. |
| Outcome-independent input-validity audit | Complete ledger for 1,840 historical artificial-tree training runs; 1,000 retained and 640 excluded without reference to outcomes | Prevents signed synthetic transfer outputs from being treated as valid positive-conductance shunting drives. The complete 400-run inhibitory-dose family is excluded because its prespecified cross-core endpoint depends on invalid cells; no dose interaction is carried forward. |
| Clean exact-transport/backpropagation audit | 320 detached tracked-clean runs crossing two learning methods, two tasks, two valid core/input conventions, four depths and ten paired seeds | Replaces the invalid historical shunting-noise implementation check. Every artifact and finite stage-complete checkpoint passed; all four depth-averaged intervals include zero, the mean difference across 160 pairs is -0.0545 percentage points, and the 5.90-point maximum individual discrepancy bounds this as agreement rather than formal equivalence. |
| Trained subtree-address factorial | 2,700 fits crossing 20 paired seeds, four coordinate budgets, five matched representations and six route families | Supplies the decisive within-neuron test. Correct ancestry beats the best matched non-anatomical route at $K=4$ and a degree/depth-matched rewired tree at $K=2,4$, while implementation-equivalent dendritic and point/flat controls coincide. The result is a structured-address resource claim, not unique dendritic superiority. |
| Fixed spatial-topology boundary control | 320 frozen learning runs (240 retained) plus a 2,560-map connectivity audit | Shows that spatial sparse connectivity can improve forward learning, while preventing a credit-routing overclaim: the benefit persists under backpropagation and a randomly projected task, and the spatial map covers 336 rather than 276.2 unique coordinates at the same contact count. |
| Fixed-contact depth boundary control | 320 frozen learning runs at 16 terminal branches and 960--968 active contacts per soma; 160 additive runs retained | Shows that greater depth is not intrinsically beneficial after matching leaf and contact budgets. Depth four underperforms depth one in all four retained feedback contrasts, with the largest loss under scalar feedback; deeper trees are not parameter matched. |
| Prospective credit-to-loss diagnostic | 2,400 retained rows from 120 independently trained backpropagation checkpoints, three field labels and four relative step sizes | Connects feedback geometry to an optimization endpoint at scale. At the primary step, neuron-indexed feedback raises mean dendritic-gradient cosine from 0.027 to 0.586, is a descent direction in 120/120 checkpoints, and gradient cosine predicts retained one-step progress with Spearman $\rho=0.893$. |
| Reconstructed morphology cohort | Eight MICrONS cells, 28,669 mapped and classified incoming synapses, real ancestry and cable geometry | Tests whether the mechanism has nontrivial sparse routing structure in real dendritic trees rather than only in regular synthetic trees. |
| Independent reciprocal-cable generator | Exact focal-shunt response operators, row-shuffled routes and 200 degree/depth-matched surrogate trees per cell | Removes the direct generator--dictionary correspondence in the primary compression analysis. Real ancestry exceeds random and row-shuffled routes, while the smaller differences from depth bins and degree/depth-matched surrogates bound the component specific to fine topology. |
| Frozen v661 robustness cohort | Forty-seven stable nuclei disjoint from the original eight, 10,516 mapped direct-type synapses, 20 perturbation streams, and 235 eligible focal sites in 45 cells | Tests the structural ordering and focal localization across additional reconstructed cells. Morphology paths captured 75.6% of modeled field energy with 8.1% of dense-feedback wiring and beat all three structural controls in 47/47 cells. The localization difference between shunting and its first-order-current-matched additive control was 0.138 (95% interval 0.116--0.162) and positive in 45/45 eligible cells. This is a same-mouse robustness analysis using historical v661 reconstructions and direct presynaptic E/I calls covering 4.71% of incoming synapses, not independent-animal replication. |
| Matched structural controls | Dense oracle, random real paths, depth bins and ancestry shuffles at fixed channel budgets | Controls channel count and compares ancestry with approximate sparsity, column-wise reassignment and scalar depth. |
| Synapse-resolved inhibitory census | Complete 20-cell overlap with 2,789 pre-mapping known inhibitory contacts, 402 multi-clump connections, 92 presynaptic-axon aggregates, joint 3D matching and actual-site compression controls | Establishes class-dependent dendritic scales and spatial co-targeting while showing that shared-path and descendant-domain effects do not persist under dependence and 3D controls. The capacity endpoint is identified as internal dictionary compressibility rather than independent routing evidence. |
| Prospective focal perturbation | 101 depth-stratified sites; conductance shunt versus first-order-current-matched additive perturbation with somatic state restored | Tests the central topology-specific mechanism directly by changing the adjoint transport operator while controlling output error. |
| Focal factor and cable analyses | Algebraic factor freezing at 101 sites plus physical axial/leak conductances across $R_a/R_m$ regimes | Holding the post-shunt driving force fixed, updating the adjoint raises localization by 0.079 in 8/8 cells. Physical calibration then shows that the effect is small in the specified high axial-to-leak regime and recovers as effective membrane resistance decreases. These are modeled operating-regime predictions, not in-vivo interventions or cell-specific fits. |
| Active steady-state focal ensemble | 512 accepted Na/K/Ca/HCN/NMDA equilibria across eight reconstructed cells | Extends the local-Jacobian sensitivity beyond a passive operator. The unit-dose shunt-minus-additive localization contrast is 0.382, positive in 8/8 cells, with descendant attenuation and no modeled sign reversals; channel kinetics and spiking remain outside scope. |
| Direct anatomy–function join | Sixty-nine functionally imaged connected partners across seven target cells | Asks whether observed visual response similarity is organized by the available ancestry routes. |
| Task-derived held-out credit and learning | All 13 eligible scans nested in seven targets, plus 520 fits on the complete compressed segment trees | Separates anatomical capacity from alignment with an independently defined objective and reports the absence of a reliable morphology-specific learning benefit under both coarse major-branch and full-tree analyses. |
| Alignment-controlled optimization | Six imposed alignment levels on all eight reconstructed trees, matched eight-channel dictionaries and norm-matched quadratic learning | Provides the controlled positive test: with anatomy and communication budget fixed, rotating task credit into or out of the anatomical span reverses capture and learning benefit. |
| External animal-data consistency test | Published source data from six animals with task-defined opposite causal signs for P+ and P- neurons | Shows the predicted signed dendritic contrast in all six animals and places 83.7% of two-coordinate contrast energy in the signed mode. This supports neuron-specific teaching coordinates but does not uniquely identify shunting or within-tree transport. |
| Route-overlap interference theory | Exact quadratic identity and a 101 by 101 numerical verification grid | Converts route overlap and off-route leakage into a quantitative one-step forgetting prediction connected cautiously to published branch-specific motor learning. |
| Integrated claim and limitations | Conditional mechanism, cell-level inference, oracle distinctions, explicit null result | Replaces any broad shunting or morphology advantage with a falsifiable statement: topology supplies routes, conductance regulates route gain, and utility requires task alignment. |
| Journal-grade reproducibility package | Panel-level source data, frozen analysis contracts, tests, supplement and code archive | Makes every figure and inferential unit auditable and distinguishes numerical sensitivity samples from biological replication. |

## Material retained from the earlier work

The following content is necessarily retained or restated because it defines the common mechanism:

- the steady-state conductance-tree equations;
- the exact local-eligibility and transported-error factorization;
- the definition of the three-factor local update;
- exact-gradient verification;
- regular-tree feedback and exact-transport controls; and
- a limited amount of introductory and methods language needed to state these elements consistently.

The Article restates the shared equations and only the limited explanatory and methods language needed to define them consistently. It contains eight numbered main figures, with continued multi-panel displays preserving the full evidence chain. Three unchanged earlier figures are identified explicitly as inherited Supplementary Figures S1--S3, with their generators and source tables preserved byte for byte. The earlier manuscript is disclosed and will be supplied to the editor together with this statement.

## Material not carried forward as a journal claim

- No universal accuracy advantage of shunting over a matched additive architecture.
- No claim that unweighted path-gain dispersion establishes improved gradient direction.
- No cross-architecture claim based on pooled error-field statistics that include an exact-by-construction somatic stage.
- No claim that MICrONS anatomy demonstrates endogenous learning in the animal.
- No claim that measured presynaptic responses are biological teaching signals.
- No claim of superior accuracy, memory or latency over backpropagation.

## Concise disclosure

> An earlier version focused on exact local credit factorization and learning in idealized regular trees is available as arXiv:2607.03556 and was submitted to NeurIPS 2026. The submitted Article retains that foundation but adds a prospective depth-by-feedback programme with an outcome-independent 1,840-run validity ledger, a bandwidth-matched routing control, a detached 320-run exact/backpropagation audit, a 2,700-fit subtree-address factorial, validity-qualified spatial-topology and fixed-contact depth boundaries, a 120-checkpoint credit-to-loss analysis, topology-matched capacity theory, independent reciprocal-cable and physical-unit controls, reconstructed MICRONS anatomy, an active-conductance ensemble, a frozen 47-cell same-mouse robustness cohort, a 20-cell synapse-resolved inhibitory-census analysis with axon and 3D controls, focal-conductance perturbations and factor decomposition, complete-tree and all-scan anatomy--function boundaries, held-out task-credit and controlled alignment analyses, a six-animal external consistency test, and a route-overlap interference prediction. The reconstructed cohorts come from the same MICRONS mouse and are not presented as independent-animal replication. These additions change the scientific question from whether the local rule works in regular artificial trees to when dendritic routes provide useful credit, how conductance regulates them, and where the mechanism fails.

## Authorship and related-work disclosure

All authors of the earlier credit-assignment manuscript are included in the
journal Article. Maceo Richards remains an author because the Article retains
theory and experiments to which he contributed. The related arXiv preprint
2607.24990 is also disclosed because it has overlapping authors, framework,
and use of public MICRONS anatomy and visual resources from the same mouse.
That work analyzes forward gain--load alignment in a larger topology--function
cohort; this Article analyzes backward credit routing, exact gradient
transport, focal conductance perturbations and feedback-constrained learning.
Any shared records and estimands will be identified explicitly. The authors
will provide both related manuscripts and a marked comparison if requested by
the editor.
