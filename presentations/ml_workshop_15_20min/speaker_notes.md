# When can dendritic structure help local credit assignment?

## 20-minute workshop speaker notes

Scripted time: 17:55. The remaining 2:05 is reserved for pauses, emphasis, and slide transitions.

## 1. When can dendritic structure help local credit assignment? [0:20]

**Purpose.** State the question and the conditional answer before introducing details.

**Say.** I will ask when dendritic structure can help a network learn from locally available signals. We will move from backpropagation to local learning and then to dendritic address and gain. The answer is conditional: useful routes require both sufficient feedback bandwidth and task-route alignment.

**How to read the slide.** The subtitle previews the argument: coordinate, address, route gain, then the alignment boundary.

**Likely question.** Do dendrites replace backpropagation?

**Answer.** No. Backpropagation is our exact reference. We test when restricted, dendrite-compatible feedback retains enough of that information to support learning.

**Claim guardrail.** Do not claim a biological implementation of backpropagation.

**Transition.** First, what information problem does any learning rule have to solve?

## 2. Learning changes network weights; credit assigns each change [0:50]

**Purpose.** Define credit assignment in language shared by machine learning and computational neuroscience.

**Say.** A neural network maps an input to an output through many trainable weights. Training begins with one global loss, but the optimizer needs a separate update for every weight. Credit assignment is this conversion: for each parameter, which parameter should change, in which direction, and by how much? The equation at the bottom is ordinary gradient descent. The difficult object is not the forward activity itself; it is the derivative of the global loss with respect to a parameter that may be many stages upstream. This is the common problem that backpropagation and proposed neuronal learning rules must address.

**How to read the slide.** Follow the forward arrows to the loss, then the highlighted reverse route to one weight. Read the equation as delta w i equals minus eta times the loss derivative.

**Likely question.** Is credit assignment just another name for gradient descent?

**Answer.** Gradient descent specifies how to use a derivative. Credit assignment is the problem of computing or approximating the parameter-specific information that derivative requires.

**Claim guardrail.** Keep the distinction between the optimization rule and the mechanism that supplies its derivative.

**Transition.** Backpropagation tells us exactly what that returned information would be.

## 3. Backpropagation defines the exact neuron-specific learning signal [1:00]

**Purpose.** Introduce the exact reference and situate the work among alternative credit-assignment approaches.

**Say.** For a point neuron u, the downstream network returns delta u, the derivative of the loss with respect to that neuron's output. The reverse recursion multiplies downstream errors by synaptic weights and activation derivatives, so delta u is specific to the neuron and generally nonlocal. We use this as an information reference, not as a claim that a brain literally runs backpropagation. Existing approaches replace or infer this signal in different ways: fixed feedback, state-based objectives, eligibility traces with modulatory factors, or dendritic teaching signals. Our question is complementary: once feedback is restricted, what spatial coordinates can a neuron and its dendrites provide?

**How to read the slide.** The recursion sums over neurons v downstream of u. Delta u belongs to a neuron, not yet to an individual incoming weight.

**Likely question.** Why is delta u not already weight-specific?

**Answer.** All incoming weights of neuron u share the same downstream loss derivative. Their different presynaptic activities and local derivatives make their final gradients distinct.

**Claim guardrail.** Do not imply that feedback alignment, equilibrium propagation, and three-factor rules are equivalent mechanisms.

**Transition.** That shared structure gives the standard local factorization.

## 4. Local learning preserves eligibility and approximates the returned signal [0:50]

**Purpose.** Define local learning and pre-empt the misconception that a shared scalar collapses neurons.

**Say.** For a point neuron, the exact weight gradient factorizes into a local eligibility and a neuron-specific loss signal. Eligibility uses quantities available at the connection: presynaptic activity and the postsynaptic derivative. A local rule can preserve this exact factor while replacing delta u by whatever learning signal is available. At one extreme every neuron receives the same scalar; at the other, feedback preserves one coordinate per neuron. A shared scalar does not make all weights equal, because every weight still has its own eligibility. It limits the rank of task-dependent learning information, so diversity remains but is underused.

**How to read the slide.** Read the factorization from left to right: local eligibility e i times exact neuron signal delta u; the available signal replaces only the second factor.

**Likely question.** Would one shared error make the hidden layer behave like one neuron?

**Answer.** No. Different inputs, activations, initial weights, and eligibilities keep units distinct. The limitation is feedback bandwidth: only one task-dependent direction is broadcast per example.

**Claim guardrail.** Call this a bandwidth bottleneck, not neuronal collapse.

**Transition.** A point neuron offers one destination; a dendritic neuron offers many internal destinations.

## 5. Dendrites turn one neuronal signal into four testable routing questions [0:55]

**Purpose.** Define the paper's organizing variables before any dendritic result is shown.

**Say.** We now replace one point unit by a neuron with N internal compartments. The available compartment field is A u times c u. The matrix A u specifies where each of K feedback channels is delivered; c u contains their signed values on the current example. This separates four questions that are often conflated. Coordinate asks whether feedback identifies the correct neuron. Ownership asks whether that coordinate reaches the correct arbor. Address asks which subtree within the arbor receives it. Gain asks how strongly the signal propagates. Finally, alignment asks whether these available routes span the credit pattern the task actually demands.

**How to read the slide.** A u has one row per compartment and one column per available channel. Multiplying by c u produces one available value at every compartment.

**Likely question.** Is A u the anatomical adjacency matrix?

**Answer.** No. It is a delivery or address matrix. Anatomy can generate its columns, but matched non-anatomical and deranged matrices let us separate route capacity from a specific topology.

**Claim guardrail.** Grouped-point controls can emulate the same routing matrix; an address need not uniquely require dendritic material.

**Transition.** To derive how route gain works, we need the conductance model.

## 6. Conductance makes dendritic voltage a normalized quotient [1:05]

**Purpose.** Define excitatory, inhibitory, additive, and shunting terms before deriving gradients.

**Say.** Each compartment receives nonnegative excitatory and inhibitory conductances. They are not distinguished by positive versus negative weights; they are distinguished by their reversal potentials. At steady state, voltage is the total reversal-weighted current divided by total conductance. Excitation and inhibition therefore affect the numerator, while both also increase the denominator and reduce local input resistance. Our additive control changes current without changing total conductance. A shunt can carry almost no net current when voltage is near the inhibitory reversal potential and still change the denominator. That denominator effect is what can regulate sensitivity and the gain of a credit route.

**How to read the slide.** G E and G I are sums of active synaptic conductances. R total is the reciprocal of leak, synaptic, and effective child conductance.

**Likely question.** How can inhibition matter when its instantaneous current is near zero?

**Answer.** Because conductance changes input resistance. Near the inhibitory reversal potential the numerator contribution can vanish while the denominator still increases.

**Claim guardrail.** The additive comparison is a current-based control, not a claim that all point-neuron inhibition is additive.

**Transition.** Differentiating this steady state reveals what a synapse can compute locally and what must be transported.

## 7. The exact dendritic gradient is local eligibility × transported compartment error [1:10]

**Purpose.** Present the central gradient factorization and distinguish directed-tree transport from the general adjoint.

**Say.** For conductance g i on compartment n, the exact gradient again has two factors. The local eligibility contains presynaptic activity, local input resistance, and the driving force between the synaptic reversal potential and local voltage. The second factor is delta V n u: how the loss changes if voltage at that compartment is perturbed. In the directed-tree approximation, this compartment error equals the somatic error times a path gain, alpha tilde n, which is a product of local derivatives, resistances, and coupling conductances along the unique path to the soma. For reciprocal cable models, the same error field is obtained by solving the steady-state adjoint.

**How to read the slide.** Read the first equation as local dendritic eligibility times transported compartment error. Alpha tilde n is dimensionless route gain along the directed path.

**Likely question.** Does the path-product formula apply to a fully reciprocal dendrite?

**Answer.** Not literally. It is exact for the directed-tree model. The more general result is the adjoint linear solve; the eligibility-times-compartment-error factorization remains valid.

**Claim guardrail.** Do not present one directed path as the general Green's function of a reciprocal cable.

**Transition.** We can now abstract any restricted feedback pathway as an operator on this exact field.

## 8. A feedback pathway selects, assigns, and scales stochastic credit [0:55]

**Purpose.** Introduce the common mathematical object used to compare feedback schemes.

**Say.** Let mu plus xi denote a stochastic exact gradient: mu is its mean task-aligned component and xi is zero-mean sampling noise. A feedback architecture applies an operator M before the update. Identity M gives unrestricted backpropagation. A scalar broadcast, one coordinate per neuron, subtree routes, deranged routes, and an exact compartment field are different choices of M. This abstraction separates the forward neuron from the information geometry of its learning signal. It also lets us ask one quantitative question across experiments: how much useful signal does a restricted route retain, and how much update energy and noise does it admit?

**How to read the slide.** M acts on the stochastic gradient field. It can restrict span, reassign coordinates, or rescale them.

**Likely question.** Is M learned by the model?

**Answer.** Not necessarily. The theory describes any fixed or state-dependent delivery operator. In our controlled comparisons, M is prescribed so its effects can be isolated.

**Claim guardrail.** This is a local linearization of delivered credit, not a claim that every biological pathway is globally linear.

**Transition.** The signal-noise utility follows from the smoothness bound for one update.

## 9. Operator utility balances retained signal, update cost, and noise [1:05]

**Purpose.** Make the phase theory the predictive centerpiece and state its scope honestly.

**Say.** The numerator measures squared alignment between the true mean gradient and the routed update. The denominator penalizes both the size of the retained signal and the routed noise, scaled by curvature. So a restricted operator can help when it preserves aligned signal while rejecting noisy or irrelevant dimensions. We evaluated this quantity before training across 540 conditions. Its rank correlation is 0.937 with observed norm-matched one-step progress and 0.916 with final accuracy. This is the paper's strongest quantitative result: one operator theory predicts major wins, nulls, and crossovers. It is not a guarantee for every small difference accumulated over a nonlinear trajectory.

**How to read the slide.** Read U of M as aligned signal squared divided by curvature times routed signal-plus-noise energy.

**Likely question.** Does a high U guarantee higher final test accuracy?

**Answer.** No. It is exact for an isotropic quadratic and otherwise a one-step smoothness guarantee. The strong final-accuracy association is empirical validation, not a theorem about all training trajectories.

**Claim guardrail.** Lead with prediction of large contrasts and phase boundaries; keep trajectory-scale exceptions explicit.

**Transition.** The first boundary appears on ordinary image tasks.

## 10. Neuron-specific feedback dominates standard image tasks [1:00]

**Purpose.** Establish that neuron identity, rather than exact dendritic address, is the main bottleneck on standard benchmarks.

**Say.** These ladders progressively increase feedback resolution. Moving from one strict layer-wide scalar to one coordinate per neuron improves accuracy by about eight to sixteen percentage points across MNIST and flattened CIFAR-10. Moving from one neuron-specific coordinate to the exact compartment field changes accuracy by less than one point. On CIFAR-10, the exact field is within the predefined one-point equivalence margin of backpropagation. The important conclusion is a boundary, not a universal dendritic benefit: on standard image tasks, preserving which neuron owns a learning signal is much more important than resolving the exact path inside that neuron.

**How to read the slide.** Compare adjacent rungs, not absolute axes across datasets. Green denotes shunting on MNIST; blue denotes the additive tree.

**Likely question.** Why does the scalar network still learn well?

**Answer.** Every weight retains a distinct eligibility and the forward network retains many hidden units. Scalar feedback restricts task-dependent credit but does not erase representational diversity.

**Claim guardrail.** Do not claim exact dendritic routing is generally needed from these benchmarks; they show the opposite.

**Transition.** To make a branch address necessary, the task must demand incompatible updates inside one neuron.

## 11. Credit conflict creates a demand for branch-specific signals [1:00]

**Purpose.** Define the controlled branch-conflict task before showing its outcome.

**Say.** Here every branch is active and therefore has nonzero eligibility, but a context selects which branch determines the label. The conflict parameter chi controls the distractors. At chi equals zero, selected and nonselected streams are class-compatible, so one shared learning signal can work. At chi equals one, the nonselected branches carry the opposite class, so simultaneously active branches require different or opposing updates. The branch-specific rule gates credit by context; the shared rule broadcasts the same value across branches. This is a deliberately constructed existence test of when within-neuron address resolution is required.

**How to read the slide.** B is the number of branches, c t selects the relevant branch, and d b is the mean update direction assigned to branch b.

**Likely question.** Was this task designed to favor branch-specific routing?

**Answer.** Yes, deliberately. It is a mechanism-matched necessity test: we vary the precise feature, simultaneous within-neuron credit conflict, that the theory says should create path demand.

**Claim guardrail.** Call it a controlled existence proof, not a naturalistic benchmark.

**Transition.** The theory predicts the exact conflict value at which shared credit changes sign.

## 12. Shared credit fails at the predicted conflict boundary [0:55]

**Purpose.** Show that the analytic crossover predicts trained failure and that route assignment is causal.

**Say.** The shared update has an analytic eigenvalue B minus two chi times B minus one. It crosses zero at chi c equals B over two times B minus one. The trained shared rule collapses near this predicted boundary for two, four, and eight branches. At full conflict, branch-specific routing gains thirty-five to fifty-eight percentage points. A cyclic derangement keeps the same rank and sparsity but delivers each channel to the wrong branch, and it fails. Analytic backpropagation, the correct routed field, and a gated-point implementation coincide because they are equivalent calculations of the same routing matrix.

**How to read the slide.** Lambda shared is the useful shared-mode coefficient. Its sign change defines chi c; the plotted collapse points track that boundary.

**Likely question.** Why does the critical conflict depend on B?

**Answer.** One selected stream contributes useful signal while B minus one distractors contribute conflicting signal. Increasing B changes that balance and shifts the zero crossing.

**Claim guardrail.** The result establishes the need for an address, not a unique material need for a dendritic tree.

**Transition.** A second task asks whether a small hierarchy of addresses can be efficient.

## 13. Nested tasks ask whether a few subtree addresses are efficient [0:50]

**Purpose.** Define the hierarchy task and make clear that K is feedback bandwidth.

**Say.** The next task has eight input streams organized by a nested context hierarchy. We expose only K independent feedback channels inside each neuron: one shared channel, two coarse subtrees, four intermediate subtrees, or eight leaf-specific channels. A K therefore has rank K. We compare true ancestry routes with derangements and matched non-anatomical low-rank bases while holding forward resources, rank, sparsity, and parameter count fixed. K is feedback bandwidth, not physical dendritic depth. The hypothesis is that a tree becomes an efficient basis only when the task's credit covariance has a compatible nested structure.

**How to read the slide.** The four route diagrams increase within-neuron bandwidth K from one to full rank eight.

**Likely question.** Is increasing K the same as adding dendritic depth?

**Answer.** No. K counts independent returned coordinates. Physical depth changes the forward serial computation and is a separate experiment kept in backup.

**Claim guardrail.** Do not let the audience interpret K as number of dendritic levels.

**Transition.** The result separates address ownership from the smaller contribution of this particular topology.

## 14. Subtree addresses help—but fine topology adds only a narrow gain [0:55]

**Purpose.** Separate the large value of correct address assignment from the small topology-specific effect.

**Say.** At intermediate bandwidth K equals four, assigning the channels to the correct subtrees beats a cyclic derangement by sixty-one percentage points. That is the large address-assignment effect. But against the strongest matched non-anatomical low-rank basis, the ancestry advantage is only 1.27 points. Ancestry loses at K equals one and two, wins at K equals four, and ties at full rank. Rewiring removes the intermediate gain. Dendritic, grouped-point, and gated-point implementations coincide when they receive the same routed field. So the robust conclusion is that correct addresses matter; the extra value of nested anatomy is narrow and task-matched.

**How to read the slide.** Compare correct ancestry first with derangement, then with the best rank- and sparsity-matched alternative. Those contrasts answer different questions.

**Likely question.** Does the sixty-one-point effect prove that dendritic topology is superior?

**Answer.** No. It proves that correct channel-to-subtree assignment matters. The topology-specific comparison is the much smaller 1.27-point advantage over the best matched alternative at K equals four.

**Claim guardrail.** Always pair the large assignment effect with the modest topology-specific effect.

**Transition.** If biological trees are candidate route dictionaries, how much field can real arbors represent per connection?

## 15. Real arbors provide sparse candidate routes, mostly through coarse geometry [0:55]

**Purpose.** Present anatomy as route capacity, not evidence of biological use.

**Say.** We map excitatory and inhibitory contacts onto reconstructed MICrONS arbors and use branch points to define nested route supports. Capture is the fraction of a target field's energy that lies in the route span. These sparse dictionaries retain about eighty-five percent of dense capture using about seven percent of the dense route-matrix connections. That is a 14.2-fold dense-normalized ratio, but the anatomy-specific advantage over a density-matched shuffled dictionary is closer to 2.7-fold. The ordering replicates in held-out cells and a small second-mouse cohort. Most of the capacity is explained by coarse branch geometry.

**How to read the slide.** P A projects a field q into the span of the route matrix A. Capture ranges from zero to one.

**Likely question.** Does high capture show that the animal uses these routes for learning?

**Answer.** No. It shows representational capacity of modeled route dictionaries. It does not measure endogenous task gradients or demonstrate plasticity through those routes.

**Claim guardrail.** Pair 14.2-fold with the density-matched approximately 2.7-fold control and call the second-mouse result descriptive.

**Transition.** Anatomy supplies addresses; conductance can in principle regulate their gain.

## 16. Focal shunting changes descendant credit only in permissive regimes [1:00]

**Purpose.** Explain the mechanistic contrast and foreground the standard-calibration null.

**Say.** We compare a focal inhibitory conductance with an additive current chosen to match the baseline first-order focal current, then restore somatic voltage. The local dendritic voltage is intentionally not matched, because the question is whether conductance changes sensitivity beyond current injection. In the directed tree, the log gain of a descendant route changes in proportion to minus the local input resistance when the shunt lies on that route. At the standard passive calibration the localization contrast is essentially zero. In high-conductance, electrotonically permissive regimes, descendant-localized changes emerge and survive active-channel extensions. Shunting is therefore a conditional gain mechanism, not a generic learning improvement.

**How to read the slide.** The indicator is one only for descendants of compartment k, and the effect scales with R total k.

**Likely question.** Is the main shunting result positive or null?

**Answer.** At the standard passive calibration it is null. The positive localization appears in a defined permissive high-conductance regime, which is why the claim is conditional.

**Claim guardrail.** Do not imply that shunting improves training generically or that local voltage was matched.

**Transition.** Capacity and conditional gain still do not show that measured cortical function uses these routes.

## 17. Measured visual responses show no morphology-specific alignment [0:50]

**Purpose.** State the biological null as a headline result and define its inferential scope.

**Say.** For seven reconstructed target cells from one MICrONS mouse, we map measured presynaptic visual responses through the conductance model and test held-out postsynaptic-response prediction. Nested subtrees do not outperform random or site-shuffled route dictionaries for held-out learning, field capture, or within-arbor structure-function similarity. The effects are centered near zero. This is an important boundary: measured anatomy offers possible addresses, but these visual-response data do not reveal preferential alignment between those addresses and the functional relation being modeled. The cohort is small and does not measure plasticity, so the result is a scoped null rather than a general rejection of dendritic learning.

**How to read the slide.** The pipeline moves from measured presynaptic responses to mapped conductances and then held-out postsynaptic prediction; the contrast plot compares topology with matched controls.

**Likely question.** Does this rule out morphology-specific credit assignment in cortex?

**Answer.** No. It rules out a detectable advantage in this seven-cell, one-mouse visual-response analysis. Other tasks, states, cell types, or direct plasticity measurements could differ.

**Claim guardrail.** Say seven cells and one mouse aloud; never generalize the null beyond this cohort and assay.

**Transition.** We therefore ask the narrower causal question: would the same routes help if the task field were aligned to them?

## 18. The same anatomical routes capture task credit aligned to their span [0:50]

**Purpose.** Show controlled sufficiency of alignment without implying endogenous biological use.

**Say.** We construct a fixed-energy field phi of a by rotating between a unit vector inside the anatomical route span and an orthogonal unit vector. Anatomy, field energy, curvature, and route count remain fixed; only alignment a changes. By construction, anatomical capture increases linearly with a, and the true subtree routes outperform matched controls as the field enters their span. This rescue shows that the anatomical dictionary is sufficient to represent a task field when the geometry is matched. It does not show that alignment was learned, that these are endogenous cortical error signals, or that the manipulation improves a trained network.

**How to read the slide.** u parallel lies in the column space of A and u perpendicular is orthogonal to it. The square roots keep total field energy fixed.

**Likely question.** Is this a trained-learning experiment?

**Answer.** No. It is a controlled representational analysis. Its purpose is to isolate alignment as the missing variable after the measured-response null.

**Claim guardrail.** Use the phrase conditional representational sufficiency.

**Transition.** The standard tasks, controlled positives, and biological nulls now occupy one common phase plane.

## 19. One alignment–bandwidth plane organizes the wins and nulls [0:45]

**Purpose.** Give the audience one unifying map rather than a list of disconnected results.

**Say.** The horizontal axis is task-route alignment: how much demanded credit lies in the available route span. The vertical axis is feedback bandwidth relative to the effective dimensionality of the task-credit field. Low bandwidth creates a coordinate or address bottleneck. At aligned intermediate bandwidth, restricted routes can preserve useful signal while rejecting irrelevant dimensions. At full rank, route capacity saturates; at weak alignment, extra structure cannot help. Each experiment estimates its coordinates separately. The background regimes and the bandwidth-equals-effective-rank boundary are theoretical, not fitted to the outcome points.

**How to read the slide.** Move right for stronger task-route alignment and up for more independent feedback coordinates relative to effective credit rank.

**Likely question.** Was the phase boundary fitted to these experiments?

**Answer.** No. The regime tint and K over effective-rank equals one line come from the operator theory. Experimental coordinates are estimated independently within each assay.

**Claim guardrail.** Do not treat approximate point placement as a universal calibrated phase diagram.

**Transition.** This leaves four conclusions, ordered by what the evidence establishes most strongly.

## 20. Coordinate → address → gain, all conditional on task–route alignment [0:45]

**Purpose.** End with four defensible statements and one memorable novelty claim.

**Say.** The first requirement is neuron-specific coordinate: it closes most of the scalar-to-exact gap on standard tasks. Within-neuron addresses become necessary when simultaneously active branches require conflicting updates. Nested topology adds a modest advantage only at matched intermediate bandwidth, while real arbors provide sparse candidate routes. Conductance can regulate route gain in permissive states, but measured endogenous use remains unestablished. The central contribution is therefore not that dendrites always improve learning. It is a predictive boundary map of when restricted dendritic routes should help, when they should not, and why.

**How to read the slide.** The final flow is coordinate, address, and gain, gated jointly by alignment and bandwidth.

**Likely question.** What is the one-sentence novelty?

**Answer.** A stochastic credit-operator theory quantitatively predicts when restricted dendritic routes preserve useful learning signal, and controlled experiments map both its positive regimes and its null boundaries.

**Claim guardrail.** Finish on the conditional theory; do not inflate anatomy or shunting into universal benefits.

**Transition.** Thank the audience and invite questions.
