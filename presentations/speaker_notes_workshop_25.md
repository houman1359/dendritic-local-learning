# Speaker notes — 26-slide workshop deck

Deck: `dendritic_credit_workshop_25.pdf` (26 core slides, then a Backup
divider and A1–A7). The target is 29–30 minutes, leaving time for a brief
interruption and transitions. Start each equation slide with its
**Intuition** sentence before reading any notation.

## Timing contract

| Arc | Slides | Cumulative target |
|---|---:|---:|
| Credit assignment and neuronal locality | 1–6 | 5:30 |
| From point neurons to exact dendritic credit | 7–12 | 12:15 |
| Predictive phase theory | 13–14 | 15:05 |
| Experiments and biological boundaries | 15–24 | 26:20 |
| Synthesis and close | 25–26 | 29:25 |

If more than 45 seconds late, compress slides 5, 12, 20, and 24 to their
takeaway lines. Do not rush slides 11, 13, 16–17, 21–23, or 25: they carry
the factorization, the theory, the path-demand crossover, and the biological
boundary test.

## Core slides

### 1. When dendritic structure helps local credit assignment — 0:30

The question is not whether dendrites can compute. It is whether their
structure changes the information and wiring needed to learn. The answer
will be conditional: coordinates select neurons, subtree routes address
synapses, conductance changes route gain, and alignment decides whether any
of this improves learning.

### 2. Learning is a credit-assignment problem — 0:50

One global consequence must be converted into many local causes. Move from
left to right: loss, network, neuron, branch, synapse. At each scale the
assignment asks the same three questions—where, which sign, and how much.
This is why a biological learning rule is incomplete until it identifies the
information that actually arrives at each synapse.

### 3. Backpropagation provides exact parameter credit — 1:00

**Intuition:** backpropagation gives every weight its own correctly signed
teaching coordinate.

The update is presynaptic activity times a parameter-specific error. The
reverse recursion constructs that error by multiplying downstream
Jacobians. Treat this as the optimization reference, not as the biological
mechanism we assume the brain implements literally.

### 4. Literal backpropagation is not a neuronal mechanism — 0:50

The biological problem is visible in four requirements: reuse of exact
weights, ordered reverse coordination, access to nonlocal state, and a unique
index for every parameter. Existing theories replace different pieces of
this reverse computation. Our comparison asks exactly what information a
synapse can access.

### 5. Local-learning theories differ in what replaces the exact error — 1:00

Read the four families without turning this into a literature survey:
modulatory three-factor rules, feedback alignment, inferred-state rules, and
dendritic error-compartment models. Their mechanisms differ, but most deliver
at most a scalar or one error coordinate per neuron. The missing question is
what happens spatially after that neuron-level coordinate arrives.

### 6. Point-neuron learning stops at one neuronal coordinate — 1:20

**Intuition:** a shared neuronal error does not make synapses identical;
their local eligibilities remain different, but the teaching signal cannot
explicitly name a branch.

In the equation, every synapse has its own activity-dependent eligibility
and all synapses in the neuron share \(\delta_u\). Different inputs,
receptive fields, states, and initialization therefore keep neurons and
weights distinct. The limitation is feedback bandwidth—not collapse into
one neuron. This distinction motivates the strict-scalar control, but the
spatial question begins inside the selected neuron.

### 7. A dendritic tree adds state, addresses, and route gain — 0:55

**Intuition:** the dendritic gradient adds a spatial factor after the
neuron-level coordinate arrives.

Preview the three factors: local eligibility, the neuronal coordinate, and a
tree-dependent transport factor. The four small trees turn this into the
talk's vocabulary: local state, which neuron, where in its arbor, and how
strongly the route transmits credit.

### 8. A dendritic neuron is a conductance tree — 1:15

**Intuition:** a compartment voltage is a conductance-weighted mixture, so a
local shunt changes both voltage and sensitivity.

At steady state, the numerator sums reversal-weighted inputs and the
denominator is total conductance. A shunting conductance raises that
denominator locally. It therefore changes the forward state and, because the
Jacobian depends on that state and conductance, the backward transport
operator. This is why shunting is derived here rather than introduced later
as an optional mechanism.

### 9. The conductance quotient yields an exact local eligibility — 1:00

**Intuition:** differentiating the conductance quotient automatically
produces a synapse-local driving-force term.

The exact factor is presynaptic activity times input resistance times
\(E_i-V_n\). All three are available locally. In the additive-current
control the weight is absent from the denominator, so the driving-force
factor disappears. This establishes what is truly local before discussing
how error reaches the compartment.

### 10. Each compartment receives a transported error field — 1:10

**Intuition:** a tree has one path from each compartment to the soma, so its
credit is the somatic error multiplied by the gains along that path.

Define \(\delta_n=\partial\mathcal L/\partial V_n\). The path product
contains local slopes, input resistances, and dendritic couplings. It is
state dependent and multiplicative; a focal shunt changes the terms on
descendant paths. The adjoint form and derivation are available in A1.

### 11. Exact dendritic credit factorizes at every synapse — 1:25

**Intuition:** exact synaptic credit is what the synapse measures locally
times what the tree transports to that location.

This is the central identity: eligibility \(e_i\) times transported error
\(\delta_{n(i)}\). The split is exact, not a proposed learning heuristic.
Reconstruction agreed with automatic differentiation across 100 diagnostic
runs at numerical precision. From here onward, every local rule can be
described by how it approximates the transported field.

### 12. Credit fields form a hierarchy of spatial resolution — 0:50

Name the four rungs once: global scalar, neuron coordinate, subtree address,
and exact compartment field. More spatial resolution costs more feedback
bandwidth. Route gain is orthogonal to this ladder: it rescales or conditions
an available coordinate but does not create another one. Keep neuron identity
separate from within-tree address; combining those concepts would obscure what
the experiments actually test.

### 13. A credit operator predicts when restricted routes help — 1:35

**Intuition:** a restricted route helps only when retained task signal is
worth more than discarded signal and admitted noise.

Represent a local rule by an operator \(M\) acting on a noisy teaching field.
The descent bound separates retained signal, curvature cost, and admitted
noise. Optimizing the step size gives \(U(M)\). Positivity is the feedback-
alignment condition. The bound is exact for the matched quadratic and
otherwise compares guarantees; it should not be sold as an exact prediction
of every training trajectory.

### 14. Alignment × bandwidth defines the phase boundary — 1:15

**Intuition:** low-rank routes help when they reject more noise than task
signal.

For an orthogonal projection \(P\), the displayed crossover is exact for the
identity-curvature quadratic at unit step. At a common step \(0<\eta<2\),
rejected noise must exceed \((2-\eta)/\eta\) times the discarded task signal.
This predicts an interior optimum: too few routes lose signal; too many admit
fine-scale noise. The background diagram
is explicitly a theory-guided schematic, not measured data. Across tested
conditions the operator prediction correlated with one-step progress at
\(\rho_s=0.937\) and final held-out accuracy at \(\rho_s=0.916\).

### 15. Neuron identity closes most of the ordinary classification gap — 1:05

On MNIST, moving from one strict layer-wide scalar to one signed coordinate
per neuron yields the large gain, 7.8–11.2 percentage points across the
shunting and raw-additive architectures. Correct ownership—assigning the same coordinates to the
correct neuron-to-tree map—adds only 0.15–0.70 points. The main bottleneck
here is which neuron, not where inside its arbor.

### 16. Credit conflict creates demand for a branch address — 1:05

**Intuition:** a branch address becomes useful when simultaneously active
branches carry eligibilities that ask for incompatible updates.

Each example drives \(B\in\{2,4,8\}\) branch groups with binary Fashion-MNIST
views. Context \(c\) selects the branch that determines the readout. A
nonselected view agrees with the selected label with probability \(1-\chi\) and
carries the opposite class with probability \(\chi\). The selector is
downstream of branch computation, so all branch-local eligibilities remain
nonzero; moving the selector inside eligibility would remove the intended
credit conflict.

Exact credit sends \(\delta\mathbf 1[b=c]\); the gain-matched shared rule sends
\(\delta/B\) to every branch. For centered binary evidence, the shared mode
has eigenvalue \(B-2\chi(B-1)\), predicting
\(\chi_c=B/[2(B-1)]\): \((1,2/3,4/7)\) for \(B=2,4,8\). This is a prediction
about branch-selective information. It is not a claim that dendritic material
is the only possible implementation.

### 17. The path benefit appears at the predicted conflict boundary — 1:10

At \(\chi=0\), the mean correct-path minus shared-credit differences are no
larger than 0.16 percentage points. At full conflict, the corresponding gains
are 34.9, 58.2, and 57.4 points for \(B=2,4,8\). The correct-path advantage
increases with conflict in 20 of 20 seeds at every branch count.

The transition moves in the predicted order: both the initial-utility zero and
the trained chance crossing satisfy
\(\chi^*_{B=8}<\chi^*_{B=4}<\chi^*_{B=2}\) in 20 of 20 seeds. Mean
trained crossings are approximately 0.59, 0.71, and 1.00, close to the
analytic \((4/7,2/3,1)\). Correct path, analytic backpropagation, and the
gated-point emulation have zero maximum endpoint difference. The experiment
therefore establishes a controlled need for an address, not a generic
Fashion-MNIST advantage or a dendrite-exclusive mechanism.

### 18. Ancestry routes help only at intermediate bandwidth — 1:00

The theory predicts a bias–variance crossover and the budget sweep locates
it. Nested ancestry routes beat the best rank-matched non-anatomical route by
1.27 points at \(K=4\), favored in 15 of 20 paired seeds
(raw \(P=0.0048\), BH-adjusted \(P=0.0064\)); \(K=1,2\) lose and \(K=8\) ties because all leaves are
independently addressed. Present this as a conditional interior win, not a
universal advantage of anatomy.

### 19. Physical depth helps only for task-matched serial computation — 1:05

Depth is useful only when its serial divisive stages match a constructed
multiplicative hierarchy. Under that alignment, D3 exceeds D1 by 31 points
at matched branch-unit and parameter budgets. Say the paired negative in the
same breath: at fixed budgets, depth per se reduced accuracy in 40 of 40
seeds, and shuffled, independent, or reversed sensors stay flat. This is a
controlled existence proof, not evidence that deeper trees generally learn
better.

### 20. Reconstructed arbors provide sparse candidate routes — 0:55

Real arbors offer an economical dictionary: about 85% of dense field capture
at roughly 7% of dense feedback wiring. Most of the 14.2-fold raw per-wire
gain is sparsity; relative to a density-matched shuffle the anatomy-specific
margin is about 2.7-fold. Morphology establishes route capacity, not
endogenous use.

### 21. Focal shunting changes descendant transport conditionally — 1:25

The matched perturbation isolates transport: local voltage, somatic state,
and output error are held fixed. Shunting produces a 0.069 localization
contrast; freezing the adjoint removes the contrast and restoring transport
recovers it. State the boundary immediately: the primary effect is null at
the textbook resting calibration and appears in the permissive high-
conductance regime. The exploratory adaptive-shunting study did not improve
final loss over no shunt. The supported claim is conditional local control of
route gain, not generic optimization benefit.

### 22. Measured responses show no morphology-specific alignment — 1:00

Apply the same route dictionaries to held-out visual-response objectives in
complete reconstructed trees. Exact compartment fields fit best, but
ancestry, rewired, random, and matched low-rank controls are statistically
similar. This honest null is central: the arbors can express addresses, yet
the measured responses do not show that the task uses those anatomical
coordinates.

### 23. Task alignment rescues the same anatomical dictionary — 1:20

**Intuition:** rotate the task gradient into the fixed anatomical span and
the same routes move from harmful to helpful.

Only alignment \(a\) changes; cells, route rank, field energy, and curvature
are fixed. Orthogonal fields lose in 8 of 8 cells, aligned fields win in 8 of
8, and field capture predicts 20-step progress at \(\rho_s=0.99\). This is
the causal bridge between the theory and the MICrONS null.

### 24. Animal learning is consistent with signed neuron coordinates — 0:55

The BCI mapping fixes causal signs: P+ and P− neurons should carry opposite
teaching coordinates. Across six animals, 83.7% of contrast energy lies in
that signed neuron-specific mode. This is consistent with the first spatial
rung—one signed coordinate per neuron—but is a retrospective re-expression,
not an independent test. It does not measure routing within a dendritic tree,
so stop the biological claim there.

### 25. One phase plane organizes the positive and null results — 1:40

Return to the prediction from slide 14. The background is theory guided; the
markers are measured or simulated outcomes. Walk through three cases only:
the intermediate-bandwidth ancestry win, the measured-response null, and the
controlled alignment rescue. Then connect the high-alignment corner to the
credit-conflict crossover on slides 16–17: an address pays only after task
eligibilities become incompatible with a shared mode. The red credit-reversal
marker is the earlier two-stream existence test retained as supporting
evidence. Do not relabel it as the new \(\chi\) sweep: conflict dose is not a
direct estimate of the plane's alignment coordinate. The point is not that
anatomy always helps, but that one signal–noise theory locates both its wins
and its failures.

### 26. Dendrites change the resource ledger, not the exact ceiling — 1:20

Close with three columns. First, a point implementation given the same
routed field reaches the same first-order ceiling. Second, a tree offers
local state, reusable within-neuron addresses, branch-local gain control, and
sparse feedback wiring. Third, the decisive experiments are branch-resolved:
focal inhibition should shift plasticity below the branch; the shift should
scale with dose and input resistance before saturating; useful routing should
track task–route alignment rather than morphology alone. Final sentence:
coordinates select neurons, trees address synapses, conductance tunes route
gain, and alignment decides whether any of it pays.

## Backup jump table

The Backup divider is PDF page 27. The seven backup frames are:

| Jump | PDF page | Use when asked about |
|---|---:|---|
| A1 | 28 | implicit differentiation, the adjoint, path-product derivation |
| A2 | 29 | derivation and exactness limits of \(U(M)\) |
| A3 | 30 | factorial controls and matched point implementations |
| A4 | 31 | input-validity exclusions and exact/BP agreement audit |
| A5 | 32 | reliability-optimal gain and same-span conditioning |
| A6 | 33 | fresh raw-additive CIFAR-10 feedback ladder |
| A7 | 34 | references and historical lineage |

Use A6 when asked whether the bandwidth result survives a harder dataset.
Lead with the adequate 50.12% matched-BP control, then the 16.39-point
strict-scalar-to-neuron gain. End with the boundary: exact path was 0.86 points
below neuron-specific feedback, so CIFAR-10 strengthens neuron identity but not
a generic within-arbor path-resolution claim.

For questions about the shunting null, answer from slide 21 before opening
A5: A5 tests a predicted one-step ordering and also reports that final loss
was null. For claims about exactness, distinguish exact gradient
factorization (slide 11/A1) from the smooth-loss utility bound (slide 13/A2).
