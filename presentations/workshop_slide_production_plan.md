# Canonical 15–20 minute workshop slide specification

## Executive decision

The canonical workshop talk is the 20-slide ML-audience deck in
`presentations/ml_workshop_15_20min/`. The 26-slide TeX deck is a longer
technical source and backup, not the presentation to use for this workshop.

The core story is one continuous argument:

1. learning changes network weights and therefore requires credit assignment;
2. backpropagation defines the exact neuron-specific learning signal;
3. a local rule preserves connection-specific eligibility and approximates
   that returned signal;
4. a dendritic neuron introduces within-neuron state, address, and route gain;
5. a credit operator predicts when restricting that field can help;
6. ordinary tasks, controlled conflict, and hierarchical routing map the
   computational boundary;
7. anatomy provides candidate routes, shunting conditionally regulates their
   gain, measured responses are null, and imposed alignment rescues the same
   routes.

Physical depth is intentionally outside the core. It asks whether serial
dendritic stages help *forward computation*, whereas the core talk asks how
task credit is routed *backward*. It belongs in backup or in an optional
two-slide module after slide 15.

## Authoritative files

- Editable source: `ml_workshop_15_20min/build_slides.py`
- Editable HTML/SVG master: `ml_workshop_15_20min/workshop_deck.html`
- Projection PDF: `ml_workshop_15_20min/dendritic_credit_ml_workshop.pdf`
- Slide PNGs: `ml_workshop_15_20min/png/slide_01.png`–`slide_20.png`
- Timed narration: `ml_workshop_15_20min/speaker_notes.md`
- Cut map: `ml_workshop_15_20min/slide_map.md`
- Visual audit: `ml_workshop_15_20min/contact_sheet.png`

Numerical results must come from the canonical manuscript figures or their
frozen source data. Canvas images are composition references only. Conceptual
schematics are native SVG so that arrows, labels, and colors remain editable.

## Exact visual-source map

This table is the build hand-off: it specifies which graphic is actually used,
which manuscript result it represents, and whether it should be redrawn or
cropped. Panel letters refer to the current nine-figure journal manuscript.

| Slide | Graphic placed on the slide | Canonical paper source / production rule |
|---|---|---|
| 1 | labeled dendritic arbor | native editable SVG using the Figure 1 color and geometry vocabulary |
| 2 | forward network plus returned-credit arrow | native editable SVG; conceptual precursor to Figure 1B |
| 3 | feedforward network, reverse chain rule, literature taxonomy | native editable SVG and text; equations match the point-neuron derivation |
| 4 | eligibility × returned-signal factorization | native equation cards; point-neuron part of Figure 1A,D |
| 5 | arbor with identity, subtree-address and route-gain overlays | native editable SVG; semantic decomposition from Figure 1C |
| 6 | directed-tree steady-state and shunt | native editable SVG plus the manuscript steady-state equations |
| 7 | exact conductance-gradient factorization | native editable SVG and equations from Figure 1D |
| 8 | soma-to-compartment transport path | native editable SVG; directed-tree path product and reciprocal-cable adjoint from Figure 1D |
| 9 | stochastic gradient → credit operator → routed update | native editable SVG; conceptual version of Figure 3A |
| 10 | predictive utility scatter | cropped and enlarged current Figure 3F |
| 11 | MNIST ladder; flattened CIFAR-10 ladder | current Figure 2B crop and Supplementary Figure S4D crop |
| 12 | compatible and conflicting branch-task examples | editable task asset generated from the same specification as Figure 4A,B and Supplementary Figure S29A |
| 13 | conflict-dose curves and analytic crossing | editable result asset generated from the frozen cohort used in Figure 4C--F |
| 14 | eight-context hierarchy and \(K=1,2,4,8\) partitions | cropped and enlarged current Figure 5A,B |
| 15 | accuracy across feedback bandwidth | cropped and enlarged current Figure 5E; the two callouts report the distinct Figure 5F,G contrasts |
| 16 | route dictionary plus wiring-normalized capture | native arbor and cropped current Figure 7F |
| 17 | matched additive-current/shunt schematic plus electrotonic boundary | native editable control schematic and current Figure 8F crop |
| 18 | held-out complete-tree learning and topology effects | cropped and enlarged current Figure 9B,C |
| 19 | imposed-alignment construction and dose response | cropped and enlarged current Figure 9D,E |
| 20 | alignment–bandwidth phase plane plus categorical evidence ladder | clean Figure 3G component plus a categorical redraw of Figure 9G; never use arbitrary bar lengths |

Do not paste a full manuscript sheet into the deck. Numerical panels are
cropped from the current vector masters at build time, whereas teaching
schematics remain native SVG so their labels and geometry can be revised
without altering the scientific source data.

## Talk-wide notation

| Symbol | Meaning |
|---|---|
| \(u\) | neuron |
| \(n\) | dendritic compartment |
| \(i\) | network connection or synaptic conductance |
| \(t\) | trial in the branch-conflict task |
| \(w_i\) | point-neuron network weight |
| \(g_i\) | trainable synaptic conductance |
| \(E_i^{\rm rev}\) | reversal potential |
| \(\delta_u=\partial\mathcal L/\partial y_u\) | exact neuron-specific learning signal |
| \(\delta_n^V=\partial\mathcal L/\partial V_n\) | exact compartment learning signal |
| \(\delta^{\rm avail}\) | available approximate learning signal; manuscript notation \(\widehat\delta\) |
| \(A_u\) | within-neuron route matrix |
| \(\boldsymbol\beta_u\) | communicated route coefficients |
| \(K\) | independently communicated signals within one neuron |
| \(\gamma_n\) | directed-tree path gain |
| \(M\) | parameter-space credit operator |
| \(\chi\) | branch-conflict probability |
| \(a_{\rm route}\) | imposed alignment with an anatomical route span |
| \(L_{\rm sm}\) | loss smoothness constant |

Use “learning signal” for the returned task-dependent factor. Use “error” only
when the quantity is explicitly a derivative of loss or a prediction error.
Use “exact compartment field” as the scientific condition name; “exact path”
is retained only where it is the frozen implementation label on a source plot.

## Visual contract

- 16:9, 2560 × 1440 export; white or very pale background.
- One dominant visual and at most one secondary plot per slide.
- Claim-style title; one interpretive sentence in the bottom ribbon.
- Blue: neuron-specific signal or additive control.
- Purple: subtree address or restricted route span.
- Green/teal: correct aligned route, anatomy, or supported result.
- Orange: shared/scalar signal or conditional result.
- Red: conflict, shunt, or exact/reference contrast when required.
- Gray: random, shuffled, deranged, rewired, or other matched controls.
- Forward activity arrows point toward output/soma. Returned-credit arrows
  point from loss/soma toward hidden units or dendritic compartments.
- No complete manuscript sheet is placed on a slide. Only the relevant panel
  is cropped and enlarged.

## Exact slide sequence

### Slide 1 — When can dendritic structure help local credit assignment?

**Time:** 0:20.

**Purpose:** State the question and conditional thesis.

**Visible content:** Title; subtitle “Feedback bandwidth, within-neuron
address, conductance-dependent route gain, and an alignment boundary”; one
clean arbor with labels `neuron identity`, `subtree address`, and `route gain`.

**Do not add:** Results, equations, or a claim that dendrites replace
backpropagation.

**Spoken logic:** Dendrites may help by structuring where a returned learning
signal acts, not by generating the global task error.

**Transition:** “Start before biology, with the learning problem shared by all
neural networks.”

### Slide 2 — Learning changes network weights; credit assigns each change

**Time:** 0:40.

**Purpose:** Define credit assignment using standard neural-network language
before introducing synapses or dendrites.

**Visible composition:** A left-to-right network computes a loss. One internal
weight \(w_i\) is highlighted. A dashed red arrow returns from the loss to the
highlighted weight. A side card asks `which neuron?`, `which location?`, and
`sign and magnitude?`.

**Equation:**

\[
\mathcal L=\mathcal L(f(\mathbf x;\mathbf w),y),
\qquad
\Delta w_i=-\eta\frac{\partial\mathcal L}{\partial w_i}.
\]

**Definitions to say:** \(w_i\) is one trainable weight; \(\eta\) is the
learning rate; the derivative is its exact credit.

**Takeaway:** A global objective must be converted into a destination, sign,
and magnitude for each weight.

### Slide 3 — Backpropagation defines the exact neuron-specific learning signal

**Time:** 0:55.

**Purpose:** Establish backpropagation as the information reference and place
the work relative to biological-learning approaches.

**Visible composition:** Network with reverse chain-rule arrow on the left;
equations and a compact four-row literature taxonomy on the right.

**Equations:**

\[
z_u=\sum_{i\in\operatorname{in}(u)}w_ix_i+b_u,
\qquad
y_u=f_u(z_u),
\]

\[
\delta_u\equiv\frac{\partial\mathcal L}{\partial y_u},
\qquad
\delta_u=\sum_{v:u\to v}w_{uv}f'_v(z_v)\delta_v.
\]

**Direct definition:** `δu is exact task information assigned to neuron u—not
yet to one weight.`

**Taxonomy:** fixed/random feedback; inferred targets or states;
eligibility-plus-learning-signal methods; dendritic teaching-signal models.

**Scientific boundary:** Do not call \(\delta_u\) parameter-specific. The
gradient becomes parameter-specific only on slide 4.

### Slide 4 — Local learning preserves eligibility and approximates the returned signal

**Time:** 0:50.

**Purpose:** Define local learning precisely and explain why a shared scalar
does not collapse the network into one neuron.

**Equations:**

\[
\frac{\partial\mathcal L}{\partial w_i}
=\underbrace{x_if'_u(z_u)}_{e_i\;:\;\text{local eligibility}}
\underbrace{\delta_u}_{\text{exact neuron signal}},
\]

\[
\Delta w_i=-\eta e_i\delta_u^{\rm avail}.
\]

**Visible comparison:** one layer-wide scalar \(m\) versus one signal
\(\delta_u^{\rm avail}\) per neuron. State once that the manuscript writes
the same available approximation as \(\widehat\delta_u\).

**Required note:** `A shared scalar does not equalize the weights: each update
still contains a different eligibility eᵢ.`

**Takeaway:** Locality determines where an update is computed; feedback
bandwidth determines what task information it can use.

### Slide 5 — Dendrites add within-neuron state, address, and route gain

**Time:** 0:50.

**Purpose:** Extend the point-neuron rule to a spatially structured neuron.

**Visible composition:** The recurring tree geometry occupies the left half.
The right half maps `network weight wᵢ → synaptic conductance gᵢ on compartment
n` and defines the route field.

**Equation:**

\[
\boldsymbol\delta_u^{V,{\rm avail}}=A_u\boldsymbol\beta_u,
\qquad
A_u\in\mathbb R^{N_u\times K}.
\]

**Definitions:** \(N_u\) is compartment count; \(K\) is independently
communicated signals; \(\boldsymbol\beta_u\) contains their coefficients; one
column of \(A_u\) defines a route's support and gain.

**Three cards:** local state; subtree address; route gain.

**Takeaway:** Selecting the neuron and selecting a location inside it are
distinct information problems.

### Slide 6 — Conductance makes dendritic voltage a normalized quotient

**Time:** 0:55.

**Purpose:** Derive the steady-state conductance logic before invoking
shunting in the gradient.

**Equations:**

\[
g_n^{\rm tot}=g_n^L+\sum_{i\in\operatorname{syn}(n)}g_ix_i
+\sum_{c\in\operatorname{child}(n)}g_{c\to n}^{\rm den},
\qquad
R_n^{\rm tot}=1/g_n^{\rm tot},
\]

\[
V_n=R_n^{\rm tot}
\left[g_n^LE_L+\sum_{i\in\operatorname{syn}(n)}g_ix_iE_i^{\rm rev}
+\sum_{c\in\operatorname{child}(n)}g_{c\to n}^{\rm den}a_c\right],
\qquad a_c=f_c(V_c).
\]

**Definitions:** \(\operatorname{syn}(n)\) is the set of synapses on
compartment \(n\); \(\operatorname{child}(n)\) is its child compartments; and
\(a_c\) is the forward activation passed by child \(c\).

**Visible comparison:** Additive current changes the numerator; shunting
conductance raises \(g_n^{\rm tot}\) and lowers \(R_n^{\rm tot}\).

**Shunting callout:** Near \(E_I\), net inhibitory current can be small while
the denominator still changes.

### Slide 7 — Dendritic credit still factorizes into eligibility × learning signal

**Time:** 0:50.

**Purpose:** Present the exact dendritic gradient identity without yet mixing
it with transport.

**Definitions and equations:**

\[
\delta_n^V\equiv\frac{\partial\mathcal L}{\partial V_n},
\qquad
\frac{\partial V_n}{\partial g_i}
=x_iR_n^{\rm tot}(E_i^{\rm rev}-V_n),
\]

\[
\frac{\partial\mathcal L}{\partial g_i}
=\underbrace{x_iR_n^{\rm tot}(E_i^{\rm rev}-V_n)}
_{e_i^{\rm den}:\text{ local dendritic eligibility}}
\underbrace{\delta_n^V}_{\text{compartment learning signal}}.
\]

**Takeaway:** The local factor changes with dendritic state; the task-dependent
factor still must reach the correct compartment.

### Slide 8 — Tree transport turns one somatic signal into a compartment field

**Time:** 0:55.

**Purpose:** Show where the compartment learning signal comes from.

**Visible composition:** Soma at the bottom, selected compartment at the top,
and a purple returned-credit arrow pointing from soma toward the compartment.

**Equations:**

\[
\delta_0^V=f'_0(V_0)\delta_u,
\qquad
\delta_n^V=\delta_0^V\gamma_n,
\]

\[
\gamma_n=
\prod_{(j\to k)\in\operatorname{path}(n\to0)}
f'_j(V_j)R_k^{\rm tot}g_{j\to k}^{\rm den}.
\]

**Direction note:** The product is indexed in the forward child-to-parent
direction \(n\to0\); the adjoint learning signal propagates over that same
path in reverse, \(0\to n\), as drawn.

**Two model statements:** Directed trees admit the exact path product;
reconstructed reciprocal cables use the steady-state adjoint
\(J_V^{\mathsf T}\mathbf q=\nabla_V\mathcal L\).

**Takeaway:** Dendrites transform a neuron-level signal; they do not generate
the circuit-level task error.

### Slide 9 — A feedback pathway selects, assigns, and scales stochastic credit

**Time:** 0:45.

**Purpose:** Define the credit operator before presenting its utility.

**Visible composition:** Stochastic BP gradient → operator \(M\) → available
local update. Label \(M\) with `select · assign · scale`.

**Equations:**

\[
\boldsymbol\mu_{\rm BP}\equiv\widehat{\boldsymbol\mu}
=\boldsymbol\mu+\boldsymbol\xi,
\quad
\mathbb E\boldsymbol\xi=0,
\quad
\operatorname{Cov}(\boldsymbol\xi)=\Sigma,
\]

\[
\boldsymbol\mu_{\rm route}\equiv\widetilde{\boldsymbol\mu}
=M\boldsymbol\mu_{\rm BP},
\qquad
\mathbf w^+=\mathbf w-\eta\boldsymbol\mu_{\rm route}.
\]

The slide uses the plain display labels \(\boldsymbol\mu_{\rm BP}\) and
\(\boldsymbol\mu_{\rm route}\) so the stochastic and routed gradients remain
visually distinct at presentation distance; the manuscript notation
\(\widehat{\boldsymbol\mu}\) and \(\widetilde{\boldsymbol\mu}\) is shown as
their formal equivalence.

**Required clarification:** \(M=I\) is unrestricted backpropagation of the
stochastic gradient, not the full-batch gradient.

### Slide 10 — Operator utility balances retained signal, update cost, and noise

**Time:** 1:05.

**Purpose:** Make the phase theory the predictive centerpiece.

**Equation:**

\[
U(M)=
\frac{[\boldsymbol\mu^{\mathsf T}M\boldsymbol\mu]^2}
{2L_{\rm sm}\left[
\|M\boldsymbol\mu\|^2+
\operatorname{tr}(M\Sigma M^{\mathsf T})\right]},
\qquad
\boldsymbol\mu^{\mathsf T}M\boldsymbol\mu>0.
\]

**Visible labels:** task-aligned signal; finite-step update cost; admitted
stochastic noise.

**Validation:** Enlarged predictive-utility plot; \(\rho_s=0.937\) with
norm-matched one-step progress and \(\rho_s=0.916\) with final accuracy.

**Scope strip:** Exact for an isotropic quadratic with Hessian
\(L_{\rm sm}I\); otherwise a curvature bound and one-step smoothness
guarantee. It does not predict every trajectory-accrued small difference.

### Slide 11 — Neuron identity—not exact path resolution—dominates both standard tasks

**Time:** 0:55.

**Purpose:** Establish the principal negative boundary before introducing
tasks deliberately requiring branch credit.

**Visible plots:** MNIST strict-scalar ladder; flattened CIFAR-10 additive
ladder without a convolutional front end.

**Required numeric table:**

- MNIST shunting: scalar → neuron \(+11.25\) pp; neuron → exact field
  \(+0.045\) pp;
- MNIST additive: \(+7.81\) pp and \(+0.186\) pp;
- CIFAR-10 additive: \(+16.39\) pp and \(-0.86\) pp.

**Plot-label clarification:** “Exact path” on the frozen CIFAR source plot is
the exact-compartment-field implementation. Exact field and backpropagation
are equivalent within the predefined ±1-point margin.

**Takeaway:** On both tasks, most of the gain comes from selecting the correct
neuron; exact within-tree resolution adds little. Do not claim that task
difficulty caused the larger CIFAR-10 identity effect.

### Slide 12 — Credit conflict creates a demand for branch-specific signals

**Time:** 1:00.

**Purpose:** Define the positive necessity task completely before showing its
result.

**Visible task:** Two \(B=4\) examples—compatible \(\chi=0\) and conflicting
\(\chi=1\). Every branch receives a Fashion-MNIST image and forms nonzero
eligibility. Context \(c_t\) selects the branch that determines the target.

**Equations:**

\[
\mathbf d_b^{\rm branch}
=\frac1N\sum_t\delta_t\mathbf1[c_t=b]\mathbf x_{t,b},
\]

\[
\mathbf d_b^{\rm shared}
=\frac1N\sum_t\delta_t\frac1B\mathbf x_{t,b}.
\]

**Symbol strip:** \(N\) is the number of trials; \(\delta_t\) is the
downstream logit gradient on trial \(t\); and \(\mathbf d_b\) is the mean
update direction for branch \(b\).

**Critical explanation:** \(1/B\) matches scale but carries no branch
identity. The selector \(\mathbf1[c_t=b]\) is the missing information.

### Slide 13 — Shared credit fails at the predicted conflict boundary

**Time:** 0:55.

**Purpose:** Connect the analytic shared-mode threshold to trained outcomes.

**Equation:**

\[
\lambda_{\rm shared}=B-2\chi(B-1),
\qquad
\chi_c(B)=\frac{B}{2(B-1)}.
\]

**Visible results:** Accuracy versus conflict dose; observed chance-crossing
dose versus analytic threshold; 35–58 pp branch-specific gain at full
conflict.

**Controls:** Cyclic derangement fails at matched rank and sparsity. Analytic
BP, correct routing, and gated-point calculation are equivalent forms of the
same routing matrix; call this an implementation control, not independent
replication.

**Takeaway:** The task proves a need for an address, not for dendritic material
alone.

### Slide 14 — Nested tasks ask whether a few subtree addresses are efficient

**Time:** 0:55.

**Purpose:** Define task hierarchy, feedback bandwidth, and matched controls.

**Visible task panel:** Eight streams are active. Context \(c=3\) selects
stream 3, which carries \(+y\); opposite-sign distractors strengthen with tree
distance. Labels explicitly distinguish same-half, sibling, and other-half
distractors.

**Visible bandwidth panel:** \(K=1,2,4,8\) route partitions within one neuron.

**Equation:**

\[
\boldsymbol\delta^{V,{\rm avail}}=A_K\boldsymbol\beta,
\qquad
\operatorname{rank}(A_K)=K.
\]

**Matched controls:** Rank, sparsity, forward resources, and parameter count
are fixed; assignment, basis, or topology changes.

### Slide 15 — Subtree addresses help—but fine topology adds only a narrow gain

**Time:** 0:55.

**Purpose:** Separate correct address assignment from topology-specific gain.

**Visible plot:** Directly labeled curves for correct ancestry, best
non-anatomical rank-matched control, and cyclic derangement across
\(K=1,2,4,8\).

**Two non-interchangeable comparisons:**

- \(+61.1\) pp: correct ancestry versus cyclic derangement at the same
  \(K=4\) bandwidth;
- \(+1.27\) pp: correct ancestry versus the strongest matched
  non-anatomical low-rank control at \(K=4\).

**Boundary against the strongest matched non-anatomical basis:** Ancestry
loses at \(K=1,2\), wins only at \(K=4\), and ties at \(K=8\). Rewiring
removes the intermediate benefit. This sentence does not describe cyclic
derangement, which is the separate assignment-error comparison.

**Implementation statement:** Dendritic, grouped-point, and gated-point models
coincide when supplied with the same routed field.

### Slide 16 — Real arbors provide sparse candidate routes, mostly through coarse geometry

**Time:** 0:55.

**Purpose:** Establish structural capacity without implying endogenous use.

**Equation and definition:**

\[
C_A(\mathbf q)=\frac{\|P_A\mathbf q\|^2}{\|\mathbf q\|^2},
\]

where capture is the fraction of modeled field energy contained in the route
span.

**Metrics shown together:** 85% of dense capture; approximately 7% of dense
route-matrix connections; 14.2× dense capture per connection; approximately
2.7× over a density-matched shuffled dictionary.

**Qualification:** Connections mean route-matrix nonzeros, not cable length,
energy, or reliability. These are modeled fields on measured anatomy.

**Replication:** Disjoint 47-cell cohort and ten QC-passing cells from a second
MICrONS mouse.

### Slide 17 — Focal shunting changes descendant credit only in permissive regimes

**Time:** 1:00.

**Purpose:** Present route gain and its primary null without overstating the
mechanism.

**Matched schematic:** Additive input equals the shunt's baseline first-order
current. A separate somatic current restores baseline somatic voltage. Local
dendritic voltage is not matched. Only the shunt changes \(G\).

**Exact reciprocal-cable relation:**

\[
\mathbf q'=\mathbf q-
\frac{\kappa_kq_k}{1+\kappa_k(G^{-1})_{kk}}
G^{-1}\mathbf e_k.
\]

**Definitions:** \(G\) is the passive conductance matrix; \(\kappa_k\) is the
focal shunt; \(G^{-1}\mathbf e_k\) is the Green's-function column controlling
spatial spread.

**Boundary plot:** Effectively null at
\(R_m=15{,}000\,\Omega\,\mathrm{cm}^2\); descendant localization emerges in
permissive high-conductance regimes and survives the steady-state active
extension.

**Takeaway:** State-dependent route gain, not a generic learning benefit of
inhibition.

### Slide 18 — Measured visual responses show no morphology-specific alignment

**Time:** 0:55.

**Purpose:** Present the measured biological null separately from the
controlled rescue.

**Pipeline:** Measured presynaptic responses → conductances at mapped arbor
sites → held-out target-cell response prediction.

**Visible plots:** Complete-tree held-out normalized MSE and standardized
topology-minus-control effects. Labels say that lower MSE is better and
positive standardized effects favor subtrees.

**Scope:** Seven target cells from one MICrONS mouse; exact compartment errors
are the reference.

**Result:** Nested subtrees do not beat random or site-shuffled routes for
held-out prediction, field capture, or within-arbor structure–function
similarity.

**Takeaway:** Anatomical availability does not establish preferential use by
the measured task.

### Slide 19 — The same anatomical routes capture task credit aligned to their span

**Time:** 0:55.

**Purpose:** Validate conditional representational sufficiency without
turning it into a trained-learning or endogenous biological claim.

**Equation:**

\[
\boldsymbol\phi(a_{\rm route})
=\sqrt{a_{\rm route}}\,\mathbf u_\parallel
+\sqrt{1-a_{\rm route}}\,\mathbf u_\perp,
\]

with \(\mathbf u_\parallel\in\operatorname{col}(A)\),
\(\mathbf u_\perp\perp\operatorname{col}(A)\), and both vectors normalized.

**Visible schematic:** The fixed-energy field rotates from orthogonal to
aligned while anatomy and route count remain fixed.

**Visible dose response:** Subtree capture increases with imposed alignment;
matched controls remain low. State explicitly
\(C_A[\boldsymbol\phi(a_{\rm route})]=a_{\rm route}\) by construction.

**Scope:** Eight reconstructed cells. Capture equals \(a_{\rm route}\) by
construction; the matched controls test route specificity. The manipulation
proves conditional representational sufficiency, not a trained-learning
benefit or endogenous biological use.

**Takeaway:** Alignment is sufficient for representational capture in the
model; endogenous morphology-specific alignment remains unestablished.

### Slide 20 — Alignment and bandwidth determine which dendritic resources help

**Time:** 0:50.

**Purpose:** Close the theory–experiment loop and separate evidence levels.

**Left panel:** Enlarged alignment–bandwidth phase plane.

\[
\text{relative bandwidth}=\frac{K}{r_{\rm eff}},
\qquad
r_{\rm eff}=\frac{(\sum_i\lambda_i)^2}{\sum_i\lambda_i^2}.
\]

Define \(\lambda_i\) as the eigenvalues of the task-credit covariance
spectrum. The presentation-specific x-axis label is `task–route alignment`;
it replaces the narrower `task–anatomy alignment` wording on the manuscript
source crop without changing any coordinates.

**Required footer:** Coordinates are estimated separately within each
experiment; the regime tint and \(K/r_{\rm eff}=1\) boundary are theoretical,
not fitted. Do not place physical depth on this backward-credit plane.

**Right panel:** Categorical evidence ladder—never quantitative bars.

- neuron identity: supported in the model benchmarks;
- subtree address: conditional;
- shunting route gain: conditional;
- anatomical route capacity: structurally supported;
- endogenous morphology-specific alignment: not established.

**Final spoken sentence:** “Dendritic structure is a conditional substrate for
routing local credit—not a general replacement for backpropagation.”

## Optional two-slide physical-depth backup

### Backup A — Forward physical depth is different from backward route resolution

Define

\[
D_{\rm r}=\text{backward route resolution},
\qquad
D_{\rm p}=\text{forward serial physical depth}.
\]

Show D1–D3 serial architectures and define hierarchy \(H\) and task–sensor
alignment \(a_{\rm sens}\). Do not call both alignment manipulations \(\alpha\).

### Backup B — Serial depth helps only when its operations match the task

Show the calibrated hierarchy result and grouped-point/flexible-point
controls: aligned D3 over D1 under BP \(+30.9\) pp; exact compartment feedback
over one shared somatic signal at D3 \(+11.0\) pp; sensor shuffling and reversal
remove the benefit; flexible point controls remain an important ceiling.

**Backup takeaway:** Serial depth is a task-matched compositional inductive
bias, not a universal point-versus-dendrite advantage.

## Final quality-control checklist

- Exactly 20 core pages, 16:9, 2560 × 1440 PNGs.
- Every task definition immediately precedes its result.
- No result panel is shown before its variables and controls are defined.
- Returned-credit arrows point away from loss/soma toward the recipient.
- \(E_i^{\rm rev}\), \(\gamma_n\), \(\chi\), and \(a_{\rm route}\) are used
  consistently.
- \(\rho_s\), not unqualified \(\rho\), labels Spearman correlations.
- The \(+61.1\) pp assignment effect is not confused with the \(+1.27\) pp
  topology effect.
- The 14.2× dense efficiency and approximately 2.7× density-matched anatomy
  contrast appear together.
- The focal control never claims matched local dendritic voltage.
- The measured null and imposed rescue remain separate slides.
- The final evidence display uses categories, not arbitrary bar lengths.
- Scripted speech remains near 17:20 so the talk has real transition time.
