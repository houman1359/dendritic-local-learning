# Speaker notes — 20-slide ML workshop talk

The scripted material below totals about **17 minutes 20 seconds**. This leaves
roughly two minutes in a 20-minute slot for pauses, transitions, and emphasis.
Each slide begins with the intuition to say before discussing its equation or
plot, and ends with the intended transition.

## 1. When can dendritic structure help local credit assignment? — 0:20

**Opening intuition:** Dendrites may help learning not by replacing
backpropagation, but by structuring where a returned learning signal acts
inside one neuron.

Today I will separate three resources: a coordinate that identifies the
neuron, an address that selects locations within it, and a
conductance-dependent gain along that route. The conclusion is conditional:
these resources help only when they match distinctions required by the task.

**Transition:** Start before biology, with the learning problem shared by all
neural networks.

## 2. Learning changes network weights; credit assigns each change — 0:40

**Before the equation:** A global objective must be converted into a local
change for every trainable weight.

The network maps an input to an output and a loss. Credit for weight \(w_i\) is
the derivative \(\partial\mathcal L/\partial w_i\), and gradient descent applies
\(\Delta w_i=-\eta\,\partial\mathcal L/\partial w_i\). The derivative answers
three questions: which destination, which sign, and what magnitude? An
internal weight cannot infer its downstream consequence from local activity
alone.

**Transition:** Backpropagation gives the exact mathematical answer.

## 3. Backpropagation defines the exact neuron-specific learning signal — 0:55

**Before the recursion:** Backpropagation is our information reference, not an
assumption that neurons literally execute reverse-mode differentiation.

For neuron \(u\), \(\delta_u=\partial\mathcal L/\partial y_u\) is assembled by
the reverse chain rule. It is neuron-specific, not parameter-specific: all
incoming weights of the neuron share this downstream factor. Their gradients
become different only after \(\delta_u\) is multiplied by each weight's local
factor.

The literature strip places the work relative to fixed feedback,
state/target-inference methods, eligibility-plus-modulator rules, and
dendritic teaching-compartment models. These approaches mainly differ in how
the neuron-level signal is generated. Our additional question is how it is
resolved spatially after it reaches the neuron.

**Transition:** This gives the point-neuron factorization used by a local rule.

## 4. Local learning preserves eligibility and approximates the returned signal — 0:50

**Before the factorization:** A local rule preserves the part of the gradient
available at the connection and replaces how the task-dependent signal is
obtained.

For a point neuron,

\[
\frac{\partial\mathcal L}{\partial w_i}
=\underbrace{x_i f'_u(z_u)}_{e_i}\,\delta_u,
\qquad
\Delta w_i=-\eta e_i\delta_u^{\rm avail} .
\]

Eligibility \(e_i\) is connection-specific; the available learning signal
\(\delta_u^{\rm avail}\) may still depend on a global objective. This is the
slide-safe label for the manuscript notation \(\widehat\delta_u\). A shared scalar
does not make all weights equal because their inputs, states, initialization,
and eligibilities remain different. It removes task-specific feedback
coordinates.

**Transition:** A dendrite introduces spatial locations inside the selected
neuron.

## 5. Dendrites add within-neuron state, address, and route gain — 0:50

**Opening intuition:** Once feedback identifies a neuron, a branching arbor
creates a second assignment problem: where inside that neuron should the
signal act?

In the dendritic model, trainable network weights become conductances \(g_i\)
distributed across compartments \(n\). The routed field is
\(\boldsymbol\delta_u^{V,{\rm avail}}=A_u\boldsymbol\beta_u\). The \(K\) entries
of \(\boldsymbol\beta_u\) are independently communicated signals, and \(N_u\)
is the number of compartments in neuron \(u\), so
\(A_u\in\mathbb R^{N_u\times K}\). Each column of \(A_u\) defines one route's
spatial support and gain. This separates local state, subtree address, and
route gain.

A grouped-point model can be supplied with the same \(A_u\), so a positive
result establishes the value of the routed information before it establishes
a uniquely dendritic implementation.

**Transition:** Conductance is the physical ingredient that makes local state
and route gain different from an additive weight.

## 6. Conductance makes dendritic voltage a normalized quotient — 0:55

**Before the equation:** A conductance changes both the drive in the numerator
and the total conductance in the denominator.

The inverse denominator is the local input resistance
\(R_n^{\rm tot}=1/g_n^{\rm tot}\). An additive current changes the numerator
without changing \(g_n^{\rm tot}\). A shunting conductance raises total
conductance and lowers input resistance. Near its reversal potential, the
shunt's own net current can be small while its denominator effect remains.
Here \(\operatorname{syn}(n)\) denotes synapses on compartment \(n\),
\(\operatorname{child}(n)\) its children, and
\(a_c=f_c(V_c)\) the activation passed forward by child \(c\).

**Transition:** The same denominator appears in the derivative with respect to
a synaptic conductance.

## 7. Dendritic credit still factorizes into eligibility × learning signal — 0:50

**Before the gradient:** The exact dendritic rule has the same logical
structure as the point-neuron rule.

Define \(\delta_n^V=\partial\mathcal L/\partial V_n\). Differentiating the
steady-state voltage gives

\[
\frac{\partial\mathcal L}{\partial g_i}
=\underbrace{x_iR_n^{\rm tot}(E_i^{\rm rev}-V_n)}_{e_i^{\rm den}}
\underbrace{\delta_n^V}_{\text{compartment learning signal}} .
\]

The first factor is locally available from activity, input resistance, and
driving force. The second factor is task-dependent and must reach the correct
compartment.

**Transition:** The tree determines how a neuron-level signal becomes this
compartment field.

## 8. Tree transport turns one somatic signal into a compartment field — 0:55

**Before the path product:** Returned credit travels from the somatic signal
toward dendritic compartments.

For the directed-tree model,
\(\delta_n^V=\delta_0^V\gamma_n\), where \(\gamma_n\) is the product of local
derivatives, input resistances, and inter-compartment conductances along the
unique path from compartment \(n\) to the soma. The path product is exact for
this directed model. Reconstructed reciprocal cables use the steady-state
adjoint rather than a unique directed path.

The product is indexed in the forward child-to-parent direction \(n\to0\),
whereas the purple arrow shows the adjoint learning signal propagating over
the same path in reverse, \(0\to n\).

Dendrites do not generate the circuit-level error; they spatially transform it
after it reaches the neuron.

**Transition:** We can represent every restricted version of this returned
field with one mathematical object.

## 9. A feedback pathway selects, assigns, and scales stochastic credit — 0:45

**Before the operator:** A restricted pathway changes not only how many
signals are present, but also where they are assigned and how strongly they
act.

Write the stochastic backpropagation gradient as
\(\boldsymbol\mu_{\rm BP}\equiv\widehat{\boldsymbol\mu}
=\boldsymbol\mu+\boldsymbol\xi\), with zero-mean noise covariance \(\Sigma\).
The available update is
\(\boldsymbol\mu_{\rm route}\equiv\widetilde{\boldsymbol\mu}
=M\boldsymbol\mu_{\rm BP}\). \(M=I\) is
unrestricted backpropagation of the stochastic gradient—not a full-batch
gradient. Scalar, neuron-specific, and subtree feedback correspond to
structured operators of increasing bandwidth.

**Transition:** Restriction loses some signal, but can also exclude stochastic
components that would enter the update.

## 10. Operator utility balances retained signal, update cost, and noise — 1:05

**Before the utility:** A useful restricted route must preserve task-aligned
gradient signal while controlling update magnitude and admitted noise.

The numerator of \(U(M)\) is the squared alignment of the routed update with
the mean gradient. Its denominator contains the finite-step cost and the
noise passed by the same operator. The positivity condition
\(\boldsymbol\mu^{\mathsf T}M\boldsymbol\mu>0\) is required.

Across 540 trained-factorial conditions, utility has Spearman correlation
\(\rho_s=0.937\) with norm-matched one-step progress and \(\rho_s=0.916\)
with final accuracy. The result is exact for an isotropic quadratic with
Hessian \(L_{\rm sm}I\); otherwise it is a curvature bound and a one-step
smoothness guarantee. It predicts the large contrasts and within-family
progress, not every small difference accumulated through nonlinear training.

**Transition:** The first empirical question is whether standard tasks need
within-neuron resolution at all.

## 11. Neuron identity—not exact path resolution—dominates both standard tasks — 0:55

**Opening intuition:** On both standard tasks, selecting the correct neuron
matters much more than resolving the exact path within that neuron.

On MNIST, strict scalar to neuron-specific feedback adds 11.25 percentage
points in the shunting tree and 7.81 points in the additive tree. Replacing
the neuron signal by the exact compartment field then adds only 0.045 and
0.186 points. On flattened CIFAR-10, neuron-specific feedback adds 16.39
points over the scalar, while the exact compartment field is 0.86 points
below neuron-specific feedback and is equivalent to backpropagation within
the predefined one-point margin. “Exact path” on the source plot is the
implementation label for the exact compartment field.

The larger identity effect in the flattened CIFAR-10 cohort is descriptive;
this comparison does not identify task difficulty as its cause.

**Transition:** What task property actually creates a need for a branch
address?

## 12. Credit conflict creates a demand for branch-specific signals — 1:00

**Opening intuition:** Every branch is active, but context determines which
branch should control the output.

Here \(N\) is the number of trials, \(\delta_t\) is the downstream logit
gradient on trial \(t\), \(\mathbf 1[\cdot]\) is the indicator function, and
\(\mathbf d_b\) is the mean update direction for branch \(b\).

All \(B\) branches receive a Fashion-MNIST image and form nonzero eligibility.
On trial \(t\), context \(c_t\) selects the branch whose image defines the
target. The conflict dose \(\chi\) changes nonselected images from
class-compatible to opposite-class. The correct update contains the selector
\(\mathbf 1[c_t=b]\); the shared update contains only the scale-matching
factor \(1/B\) and therefore carries no branch identity.

At zero conflict, sharing is harmless. At high conflict, the same learning
signal multiplies eligibilities that call for opposite updates.

**Transition:** The compatibility matrix predicts where the shared mode stops
being a descent direction.

## 13. Shared credit fails at the predicted conflict boundary — 0:55

**Before the threshold:** The shared-mode eigenvalue is
\(\lambda_{\rm shared}=B-2\chi(B-1)\), so it crosses zero at
\(\chi_c=B/[2(B-1)]\).

The thresholds are \(1\), \(2/3\), and \(4/7\) for two, four, and eight
branches. Trained collapse points follow them. At full conflict,
branch-specific feedback gains 35–58 percentage points over shared feedback.
A cyclically deranged route fails at matched rank and sparsity. Analytic
backpropagation, the correct route, and the gated-point calculation are
algebraically equivalent forms of the same routing matrix; their agreement is
an implementation control, not an independent replication.

**Transition:** This compares one shared signal with complete branch
resolution; the next task tests intermediate bandwidth.

## 14. Nested tasks ask whether a few subtree addresses are efficient — 0:55

**Opening intuition:** A tree can be useful as a compressed basis only when
the task's credit hierarchy matches its partitions.

Eight streams are active at the leaves. Context selects one stream carrying
positive class evidence; distractors have opposite sign and strengthen with
tree distance. \(K\in\{1,2,4,8\}\) is the number of independently communicated
signals within one neuron. The route matrix \(A_K\) has rank \(K\).

Controls match rank, sparsity, forward resources, and parameter count while
changing assignment, basis, or topology.

**Transition:** The result separates the value of correct addresses from the
additional value of the tree's fine topology.

## 15. Subtree addresses help—but fine topology adds only a narrow gain — 0:55

**Opening intuition:** Correct assignment matters strongly; the advantage of
this particular anatomical basis is much smaller.

At the same \(K=4\) bandwidth, correct ancestry assignment exceeds cyclic
derangement by 61.1 percentage points. Against the seedwise strongest matched
non-anatomical low-rank control, the advantage is 1.27 points. Ancestry loses
to that matched control at \(K=1,2\), wins only at \(K=4\), and ties at full
rank \(K=8\). Degree- and depth-preserving rewiring removes the intermediate
gain. Dendritic, grouped-point, and gated-point implementations coincide when
they receive the same routed field.

**Transition:** Controlled tasks establish when an address can help; anatomy
asks whether real arbors provide candidate addresses.

## 16. Real arbors provide sparse candidate routes, mostly through coarse geometry — 0:55

**Before capture:** Capture is the fraction of modeled credit-field energy
contained in the route span,
\(C_A(\mathbf q)=\|P_A\mathbf q\|^2/\|\mathbf q\|^2\).

At eight channels, subtree routes retain 85% of the dense rank-matched capture
using about 7% of the dense route-matrix connections. This is 14.2 times the
dense capture per connection; the anatomy-specific density-matched comparison
is the more modest approximately 2.7-fold advantage over a shuffled
dictionary. “Connections” means nonzero route-matrix entries, not cable
length, energy, or reliability.

The full route ordering replicates in a disjoint 47-cell cohort. In a second
MICrONS mouse, the model-matched subtree advantage is positive in all ten
quality-controlled cells; this is descriptive cell-level replication, not an
independent animal-level inferential test. These are modeled fields on
measured anatomy, not observed task gradients.

**Transition:** A branch point supplies an address; can conductance regulate
its gain?

## 17. Focal shunting changes descendant credit only in permissive regimes — 1:00

**Opening intuition:** The key control compares conductance with an additive
input without claiming that local dendritic voltage is matched.

The additive input equals the shunt's baseline first-order focal current at
the same site. After either intervention, a separate somatic current restores
baseline somatic voltage. Local dendritic voltage is explicitly not clamped;
only the shunt changes the conductance matrix \(G\). The Green's-function
column \(G^{-1}\mathbf e_k\) determines the spatial spread of the adjoint
change.

At \(R_m=15{,}000\,\Omega\,\mathrm{cm}^2\), the shunt-minus-current
localization contrast is effectively null. Descendant localization emerges
only with lower membrane resistance or added background conductance and
persists in a steady-state active-channel extension. This is a conditional
route-gain mechanism, not a general learning advantage of inhibition.

**Transition:** Anatomical capacity and a possible gain mechanism still do not
show that biological activity uses these routes.

## 18. Measured visual responses show no morphology-specific alignment — 0:55

**Opening intuition:** This analysis tests whether a measured cortical
input–output relation is preferentially aligned with measured subtree routes.

Presynaptic visual responses enter the reconstructed arbor as conductances,
and the model predicts the target cell's response on held-out conditions.
Exact compartment errors give the lowest prediction error. Across seven
target cells from one MICrONS mouse, nested subtrees do not beat random or
site-shuffled routes for held-out prediction, response-derived field capture,
or within-arbor structure–function similarity.

This is the strongest boundary on the biological claim.

**Transition:** The theory predicts a causal counterfactual: the same routes
should help if the task field is rotated into their span.

## 19. The same anatomical routes capture task credit aligned to their span — 0:55

**Before the rotation:** Hold the cells, anatomy, route count, field energy,
and curvature fixed; change only alignment with the subtree span.

The field \(\boldsymbol\phi(a_{\rm route})\) rotates between a unit vector in
the route span and an orthogonal unit vector. Its subtree capture equals
\(a_{\rm route}\) by construction. The matched controls test whether the gain
is specific to the true subtree span across eight reconstructed cells.

This manipulation establishes conditional representational sufficiency.
Because the field direction and projection coefficients are imposed, it is
not a trained-learning comparison and does not demonstrate endogenous
biological alignment.

**Transition:** The wins, ties, and nulls now return to the same alignment–
bandwidth map predicted before the experiments.

## 20. Alignment and bandwidth determine which dendritic resources help — 0:50

**Closing answer:** The horizontal coordinate is task–route alignment. The
vertical coordinate is relative feedback bandwidth
\(K/r_{\rm eff}\), where \(r_{\rm eff}\) is the effective rank of task credit.
The \(\lambda_i\) in its definition are the eigenvalues of the task-credit
covariance spectrum.
Coordinates are operationalized separately within each experiment, and the
regime tint is theoretical rather than fitted.

The evidence ladder separates the claims. Neuron identity is dominant in the
model benchmarks. Branch-specific coordinates are required under controlled
conflict. Subtree routes are a conditional intermediate basis. Reconstructed
arbors provide sparse candidate routes. Shunting conditionally changes their
gain. Endogenous morphology-specific use is not established by the measured
visual-response analysis.

**Final sentence:** Dendritic structure is a conditional substrate for routing
local credit—not a general replacement for backpropagation.

## Optional backup transition — serial physical depth

Physical depth is intentionally outside the 20-slide core because it asks a
different forward-computation question. If discussion turns to depth, use the
backup distinction

\[
D_{\rm r}=\text{backward route resolution},
\qquad
D_{\rm p}=\text{forward serial physical depth}.
\]

The calibrated hierarchy experiment shows a task-matched compositional
benefit, not a universal dendritic advantage: aligned D3 exceeds D1 by 30.9
points under backpropagation, exact compartment feedback adds 11.0 points over
one shared somatic signal at D3, and flexible parameter-matched point models
remain an important ceiling.
