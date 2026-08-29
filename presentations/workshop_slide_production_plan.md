# Workshop slide production plan

## Executive decision

The previous rendered deck should be treated as a storyboard, not as a visual
master. The scientific story is sound, but the paper-sized panels were made too
small, the equations were compressed into cards, and the slide compositions did
not have the visual hierarchy of an illustrator-built talk.

The new deck should be built from vector material throughout. Import the
canonical paper PDFs into Illustrator, Affinity Designer, Inkscape, Keynote or
PowerPoint; use clipping masks on individual panels; and export PNG only after
the complete slide is finished. Task diagrams, neuron schematics and theory
diagrams should be redrawn at slide scale. Do not put a complete journal figure
on a slide.

The recommended master sequence has 22 short slides and takes about 18--20
minutes. It is deliberately modular: slides 16 and 22 can each be split in two
for a longer talk, while several pairs can be merged for a 15-minute talk.

The story is:

1. backpropagation defines exact credit;
2. a local rule preserves synaptic eligibility but approximates the returned
   learning signal;
3. dendrites add within-neuron address and conductance-dependent route gain;
4. a credit-operator theory predicts when restricted routes should help;
5. controlled tasks establish the need for branch signals and the narrow
   benefit of tree-aligned routes;
6. forward physical depth is a separate, task-matched computation;
7. reconstructed arbors provide candidate routes, shunting can regulate their
   gain in permissive states, but measured activity does not establish their
   endogenous use.

## Authoritative sources

Use only the nine canonical main-figure PDFs as the authority for panel content
and lettering:

- `journal/figures/main/figure_01.pdf` through
  `journal/figures/main/figure_09.pdf`;
- `journal/scripts/build_main_figure_01.py` through
  `journal/scripts/build_main_figure_09.py` when a panel needs presentation-size
  fonts or direct labels;
- `journal/figures/supplementary/figure_S04_panels_A-D.pdf`, panel D, for the
  current raw-additive CIFAR-10 strict-scalar ladder;
- `presentations/credit_tree_lib.tex` for one consistent dendritic-tree geometry.

Do not use the older `figure_XX_panels_*.pdf` files as scientific sources. They
are compositor/provenance inputs and can carry older lettering or terminology.
Canvas images can be used as composition references, but not as sources of
equations, labels, tasks or numerical results.

The most useful Canvas composition references are `canvas_assets/slide_3.png`
for the global-to-local zoom, `slide_4.png` for the BP comparison,
`slide_8.png` and `slide_9.png` for address/eligibility, `slide_17.png` for the
branch-conflict composition, `slide_22.png` for focal shunting, and
`slide_24.png` for controlled alignment. Recreate their geometry and hierarchy;
do not copy their raster text.

## Visual production rules

### Canvas and typography

- Work at 16:9 in vector format. A suitable final raster export is
  2560 by 1440 pixels.
- Use a 12-column grid, outer margins of about 6% of slide width, and a title
  band occupying no more than 13% of slide height.
- Claim-style title: 46--58 pt. Equation: 34--44 pt. Axis labels: at least
  25--28 pt. Small citations: 15--17 pt. No paragraph text on the slide.
- Use one dominant visual region and at most one secondary result panel.
- Use direct labels rather than detached legends wherever possible.
- Put a qualification directly beside a result when it changes the scientific
  interpretation. Do not hide it in speaker notes.
- Use progressive builds only to reveal a derivation or comparison. Do not use
  decorative motion.

### Semantic colors

Keep the existing scientific palette, with one meaning per color:

- ink `#232323`: text and neutral equations;
- muted `#69707A`: scaffolding and secondary annotations;
- grid `#E3E7EC`: axes and separators;
- paper `#F7F8FA`: pale backgrounds;
- amber `#E2A23F`: one shared/scalar learning signal;
- blue `#20509E`: neuron-specific or additive comparison;
- green `#3FA26C`: correctly aligned subtree route or shunting condition;
- dark red `#932F1E`: exact backpropagation/reference ceiling;
- violet `#8F66CD`: oracle or unconstrained low-rank ceiling;
- gray: random, shuffled, rewired, reversed and other matched controls;
- excitatory blue `#2C6CB0`, inhibitory red `#B13138`, soma orange
  `#E8873C`, dendrite green `#3E8E63` in biological schematics.

When replotting a manuscript panel for the talk, preserve the numerical values
but use this condition-based color semantics consistently across slides.

### Repeated graphical language

- The same tree geometry should recur from slides 5 through 21. Change only the
  overlay: address, transported error, shunt, or anatomical route.
- Solid colored branches denote the support of a route. A gradient along the
  same branch denotes gain. Dots denote synapses. A halo at a branch point
  denotes a local conductance change.
- A backward signal arrow always points from loss/soma toward hidden neurons or
  dendritic compartments. Forward activity arrows point toward the soma/output.
- Every task slide must show the data-generating process before showing a result.
- Every result slide needs one bottom-line sentence, not a list of statistics.

## Notation contract for the talk

The manuscript contains a few symbols that collide when several sections are
shown in one talk. Use the following presentation notation and include the
mapping in the backup slides.

| Talk symbol | Meaning | Manuscript symbol |
|---|---|---|
| $u$ | neuron | $u$ |
| $n$ | dendritic compartment | $n$ |
| $i$ | synapse | $i$ |
| $x_i$ | presynaptic activity | $x_i$ |
| $w_i$ | point-neuron weight | $w_i$ |
| $g_i$ | synaptic conductance | $g_i$ |
| $E_i^{\rm rev}$ | synaptic reversal potential | $E_i$ |
| $V_n$ | compartment voltage | $V_n$ |
| $\delta_u=\partial\mathcal L/\partial y_u$ | exact neuron-specific learning signal | same |
| $\delta_{u,n}=\partial\mathcal L/\partial V_n$ | exact compartment learning signal | usually written explicitly |
| $\widehat\delta$ | available approximate learning signal | same |
| $K$ | independent feedback signals per neuron | $K$ |
| $A_u$ | within-neuron route matrix | $A_{u,nk}$ |
| $M$ | parameter-space credit operator | $M$ |
| $\gamma_n$ | directed dendritic path gain | $\widetilde\alpha_n$ |
| $D_{\rm r}$ | backward route resolution | same |
| $D_{\rm p}$ | forward serial physical depth; D1--D4 | same |
| $H$ | task hierarchy levels | same |
| $\chi$ | branch-conflict probability | same |
| $a_{\rm sens}$ | task--sensor alignment in the physical-depth task | $\alpha$ |
| $a_{\rm route}$ | imposed task-field alignment with a route span | $\alpha$ |
| $r_j$ | hierarchical task evidence | $E_j$ |
| $L_{\rm sm}$ | loss smoothness constant | $L$ |

The changes $E_i\rightarrow E_i^{\rm rev}$,
$\widetilde\alpha_n\rightarrow\gamma_n$, and
$E_j\rightarrow r_j$ prevent reversal potential, path gain and task evidence
from appearing to be the same object. The two alignment manipulations must not
both be called simply $\alpha$ in the talk.

## Slide-by-slide specification

### Slide 1 — When can dendritic structure help local credit assignment?

**Time:** 0:20.

**Purpose:** State the question and the conditional thesis without previewing
every result.

**Composition:** Use a clean network-to-neuron zoom. At left, a small network
ends in a task loss. One hidden neuron is highlighted. At right, that neuron is
enlarged into a branching dendritic tree with one highlighted synapse, one
colored subtree and one focal inhibitory contact. A thin backward arrow runs
from loss to neuron, then divides into subtree routes.

**On-slide text:** Title only, plus the subtitle
“feedback bandwidth, dendritic address and route gain.”

**Paper source:** Redraw the conceptual geometry of Fig. 1A--C. Do not use a
data panel or a full-figure crop.

**What to say:** “The question is not whether dendrites can participate in
learning. It is when their branching structure carries useful credit
information beyond a signal that already identifies the neuron.”

**Transition:** “To answer that, start from the information an individual
synapse needs.”

### Slide 2 — Credit assignment maps a global outcome to a local cause

**Time:** 0:45.

**Purpose:** Make the problem intuitive for an ML audience before introducing
biophysics.

**Composition:** Redraw Fig. 1B as a wide three-layer network. Highlight one
upstream synapse $w_i$, several downstream paths and the scalar loss
$\mathcal L$. Add a magnified inset at the synapse showing that it observes
only $x_i$ and the local postsynaptic state. Above the figure write three
questions: “which neuron?”, “which location?”, “how strongly?”. Keep the last
two muted until dendrites are introduced.

**Equation:** Show only the conceptual factorization

\[
\frac{\partial\mathcal L}{\partial w_i}
=\text{local sensitivity}\times\text{downstream consequence}.
\]

**Paper source:** Fig. 1B, redrawn at slide scale.

**What to say:** A synapse can measure its own activity, but the consequence of
changing it is computed elsewhere in the network. Credit assignment is the
return of that consequence to the parameters that caused it.

**Transition:** “Backpropagation tells us exactly what must be returned, even
if biology does not implement backpropagation literally.”

### Slide 3 — Backpropagation defines the exact reference signal

**Time:** 0:55.

**Purpose:** Establish BP as the mathematical reference and place the work in
the biological-learning literature.

**Composition:** Left 65%: a small network with the backward recursion traced
in dark red. Right 35%: a compact taxonomy, not a chronological timeline:

- fixed/random feedback: feedback alignment and direct feedback alignment;
- inferred targets or states: equilibrium propagation, predictive coding and
  prospective methods;
- local eligibility plus a learning signal: three-factor rules and e-prop;
- dendritic teaching/error compartments: somato-dendritic and apical-error
  models.

Put the citations in a single small footer: Rumelhart et al. 1986; Werfel et
al. 2005; Lillicrap et al. 2016; Schiess et al. 2016; Scellier and Bengio 2017;
Whittington and Bogacz 2017; Sacramento et al. 2018; Bellec et al. 2020.

**Equations, revealed in order:**

\[
z_u=\sum_{i\in\mathcal I_u}w_ix_i+b_u,
\qquad y_u=f_u(z_u),
\]

\[
\delta_u\equiv\frac{\partial\mathcal L}{\partial y_u},
\qquad
\delta_u=\sum_{v:u\to v}w_{uv}f_v'(z_v)\delta_v.
\]

**Graphic annotation:** Label $\delta_u$ “exact task information assigned to
neuron $u$.” Do not call it a biological error signal.

**What to say:** The different biological theories disagree about how the
signal is generated or approximated. They share an information question: how
many independently specified coordinates must the feedback pathway carry?

**Transition:** “Once $\delta_u$ reaches a point neuron, the remaining
gradient has a simple local form.”

### Slide 4 — A local rule preserves eligibility and approximates the learning signal

**Time:** 0:45.

**Purpose:** Define locality precisely and answer the concern that scalar
feedback would make all neurons identical.

**Composition:** Put a point neuron with three input synapses on the left.
Each synapse has a different $x_i$, state and eligibility color intensity.
On the right, show the exact factorization followed by the local rule.

**Equations:**

\[
\frac{\partial\mathcal L}{\partial w_i}
=\underbrace{x_i f_u'(z_u)}_{e_i:\ \text{local eligibility}}
\underbrace{\delta_u}_{\text{neuron-specific learning signal}},
\]

\[
\Delta w_i=-\eta e_i\widehat\delta_u.
\]

**Essential annotation:** “A shared scalar restricts task-information
bandwidth; it does not equalize weights because $e_i$ remains
synapse-specific.”

**Paper source:** Fig. 1A and Fig. 2A as references; redraw.

**Transition:** “A point-neuron model ends here. A biological neuron adds many
distinct locations inside the selected cell.”

### Slide 5 — A dendritic tree adds address and route gain

**Time:** 0:45.

**Purpose:** Introduce the paper's three central terms before any dendritic
result.

**Composition:** Use the same tree three times in a horizontal sequence.

1. **Neuron identity:** the whole cell receives $\widehat\delta_u$.
2. **Dendritic address:** one subtree is highlighted as the support of a route.
3. **Route gain:** the same support receives a graded intensity; place a focal
   inhibitory contact on its trunk.

Below the sequence show the restricted field equation:

\[
\widehat\delta_{u,n}
=\sum_{k=1}^{K}A_{u,nk}c_{u,k},
\qquad
\widehat{\boldsymbol\delta}_u=A_u\mathbf c_u.
\]

**Definitions on the graphic:** $K$ is the number of independently
communicated signals within one neuron; $c_{u,k}$ is a communicated
coefficient; column $k$ of $A_u$ specifies the route's support and gain.

**Paper source:** Redraw Fig. 1C. Use the geometry in
`presentations/credit_tree_lib.tex`.

**Transition:** “Address is an information structure. Conductance is what can
make the gain of that structure physically state dependent.”

### Slide 6 — Conductance changes both current and input resistance

**Time:** 1:00.

**Purpose:** Derive shunting from the forward steady state rather than
introducing it only as a verbal mechanism.

**Composition:** Use one enlarged dendritic compartment, not a complete tree.
Draw leak, excitatory synapses, an inhibitory synapse and one child-compartment
input. Color numerator currents and the total-conductance denominator
separately.

**Build 1 — exact current balance:**

\[
C_n\dot V_n
=g_n^L(E_L-V_n)
+\sum_{i\in\mathcal S_n}g_ix_i(E_i^{\rm rev}-V_n)
+\sum_{c\in\mathcal C_n}g_{c\to n}^{\rm den}(a_c-V_n),
\]

with $a_c=f_c(V_c)$.

**Build 2 — steady state:**

\[
g_n^{\rm tot}
=g_n^L+\sum_i g_ix_i+\sum_c g_{c\to n}^{\rm den},
\qquad
R_n^{\rm tot}=\frac{1}{g_n^{\rm tot}},
\]

\[
V_n=R_n^{\rm tot}
\left[g_n^LE_L+\sum_i g_ix_iE_i^{\rm rev}
+\sum_c g_{c\to n}^{\rm den}a_c\right].
\]

**Shunting graphic:** Put $V_n\approx E_I$ beside the inhibitory synapse.
Show that its net current can be small while $g_Ix_I$ still increases
$g_n^{\rm tot}$ and lowers $R_n^{\rm tot}$. Contrast this with a matched
additive negative current, which changes the numerator but not the denominator.

**Paper source:** Equations 5--6 in the Results; use Fig. 8A only as a later
visual reference, not as the source of this derivation.

**Transition:** “The same denominator that controls voltage also appears in
the exact derivative with respect to a synaptic conductance.”

### Slide 7 — The exact dendritic gradient still factorizes locally

**Time:** 0:55.

**Purpose:** Present the main conductance-gradient result cleanly.

**Composition:** Left 42%: one enlarged synapse on a highlighted compartment.
Place three callouts directly on it: presynaptic activity $x_i$, driving
force $E_i^{\rm rev}-V_n$, and input resistance $R_n^{\rm tot}$. Right 58%:
the derivative and gradient factorization.

**Equations:**

\[
\frac{\partial V_n}{\partial g_i}
=x_iR_n^{\rm tot}(E_i^{\rm rev}-V_n),
\]

\[
\frac{\partial\mathcal L}{\partial g_i}
=\underbrace{x_iR_n^{\rm tot}(E_i^{\rm rev}-V_n)}
_{\text{local dendritic eligibility}}
\underbrace{\frac{\partial\mathcal L}{\partial V_n}}
_{\text{compartment learning signal}}.
\]

For inhibition, the same equation holds with $E_i^{\rm rev}=E_I$. Put this
as a small annotation rather than another full equation.

**Paper source:** Redraw Fig. 1D. Do not paste its equation boxes.

**What to say:** The local factor changes, but the logic is the same as for a
point neuron: eligibility multiplied by task information. The new problem is
that the task information is now a field over compartments.

**Transition:** “How does a somatic error become a different value at each
location?”

### Slide 8 — Error transport creates a spatial credit field

**Time:** 0:50.

**Purpose:** Derive the directed-tree path gain and make route gain concrete.

**Composition:** Use one tree with a single soma-to-distal path illuminated.
Reveal one multiplicative factor at each edge. Place the vector
$(\delta_{u,1},\ldots,\delta_{u,N})$ beside the completed tree as a colored
credit field.

**Equations:**

\[
\delta_{0,u}\equiv\frac{\partial\mathcal L}{\partial V_0},
\]

\[
\delta_{u,n}
=\delta_{u,p(n)}f_n'(V_n)R_{p(n)}^{\rm tot}
g_{n\to p(n)}^{\rm den},
\]

\[
\delta_{u,n}=\delta_{0,u}\gamma_n,
\qquad
\gamma_n=
\prod_{(j\to k)\in{\rm path}(n\to0)}
f_j'(V_j)R_k^{\rm tot}g_{j\to k}^{\rm den}.
\]

**Shunt callout:** Place a shunt at compartment $k$, dim the descendant path
and show

\[
\frac{\partial\log\gamma_n^{\rm cond}}{\partial G_k^I}
=-R_k^{\rm tot}\,\mathbf 1[k\in{\rm path}(n\to0)].
\]

Label this “direct held-state effect,” not “learning improvement.” The general
adjoint belongs in backup.

**Paper source:** Fig. 1D and the path-gain equations in the Results, redrawn
using the transport mode of `credit_tree_lib.tex`.

**Transition:** “This gives a hierarchy of feedback fields that can now be
tested empirically.”

### Slide 9 — Feedback bandwidth forms an information ladder

**Time:** 0:45.

**Purpose:** Define the experimental conditions once and use the same icons
throughout the talk.

**Composition:** Four large aligned icons:

1. one scalar for the layer;
2. one signal per neuron, repeated across its tree;
3. $K$ signals assigned to subtrees;
4. the exact error at every compartment.

Under each icon show only the number of independently communicated values.
Use amber, blue, green and dark red, respectively. Use “exact compartment
field,” not “exact path,” in the formal label; “exact path transport” can be
used when referring to the code condition.

**Paper source:** Redraw Fig. 2A and Fig. 5B into one consistent ladder.

**Transition:** “The simplest test is whether standard image tasks need more
than neuron identity.”

### Slide 10 — On standard tasks, selecting the neuron provides almost all useful resolution

**Time:** 0:50.

**Purpose:** Establish the important negative boundary before presenting
positive constructed tasks.

**Composition:** Left 52%: presentation-sized replot of Fig. 2B, MNIST. Right
48%: presentation-sized replot of Supplementary Fig. S4D, flattened CIFAR-10.
Direct-label strict scalar, neuron-specific, exact compartment field and BP.
Do not include Fashion-MNIST in the core slide because that cohort uses the
legacy scalar-fallback condition.

**Numbers to place beside the plots:**

- MNIST: neuron-specific minus strict scalar is $+11.2$ percentage points
  for shunting and $+7.8$ for additive trees; exact compartment feedback then
  adds less than $0.2$ points.
- Flattened CIFAR-10: neuron-specific minus strict scalar is $+16.4$ points;
  exact compartment feedback is $0.86$ points below neuron-specific and is
  equivalent to matched BP within the predefined one-point margin.

**Visible qualification:** “Flattened CIFAR-10; no convolutional front end.
Harder classification strengthens neuron selection, not generic path demand.”

**Paper source:** Main Fig. 2B; Supplementary Fig. S4D. Rebuild the plots with
large labels rather than raster-cropping them.

**Transition:** “If harder standard tasks still do not create a generic need
for paths, what determines when restricting credit can help?”

### Slide 11 — A credit operator summarizes assignment, span and gain

**Time:** 0:45.

**Purpose:** Introduce the theoretical object before the bound.

**Composition:** Redraw Fig. 3A as a left-to-right pipeline: exact mean gradient
$\boldsymbol\mu$, stochastic perturbation $\boldsymbol\xi$, credit operator
$M$, routed update. Draw the column space of $M$ as a colored subspace.

**Equations:**

\[
\widehat{\boldsymbol\mu}=\boldsymbol\mu+\boldsymbol\xi,
\qquad
\mathbb E[\boldsymbol\xi]=0,
\qquad
{\rm Cov}(\boldsymbol\xi)=\Sigma,
\]

\[
\widetilde{\boldsymbol\mu}=M\widehat{\boldsymbol\mu},
\qquad
\mathbf w^+=\mathbf w-\eta\widetilde{\boldsymbol\mu}.
\]

**Definitions:** $\boldsymbol\mu=\nabla_{\mathbf w}\mathcal L$. $M=I$ is
exact BP. The column space of $M$ is the set of update patterns the pathway
can express. The matrix includes assignment and gain as well as support.

**Paper source:** Fig. 3A, redrawn rather than cropped.

**Transition:** “Restriction discards some signal, but it can also remove
stochastic components that would otherwise enter the update.”

### Slide 12 — Restriction helps only through a signal--noise tradeoff

**Time:** 1:00.

**Purpose:** Make the main theoretical result the centerpiece of the talk.

**Composition:** Put the bound across the upper half. Color the three terms:
green for retained task-aligned signal, muted blue for finite-step cost, amber
for admitted noise. In the lower left, reveal utility after optimizing the
step size. In the lower right, use Fig. 3D as a large phase-boundary plot.

**Bound:**

\[
\mathbb E\mathcal L(\mathbf w^+)
\leq \mathcal L(\mathbf w)
-\eta\underbrace{\boldsymbol\mu^{\mathsf T}M\boldsymbol\mu}
_{\text{task-aligned signal}}
+\frac{L_{\rm sm}\eta^2}{2}
\left[
\underbrace{\lVert M\boldsymbol\mu\rVert_2^2}_{\text{finite-step cost}}
+\underbrace{{\rm tr}(M\Sigma M^{\mathsf T})}_{\text{admitted noise}}
\right].
\]

**Utility:**

\[
U(M)=
\frac{[\boldsymbol\mu^{\mathsf T}M\boldsymbol\mu]^2}
{2L_{\rm sm}\left[
\lVert M\boldsymbol\mu\rVert_2^2+
{\rm tr}(M\Sigma M^{\mathsf T})\right]}.
\]

**Large intuitive callout for an orthogonal projector $P$ in the
identity-curvature quadratic at unit step:**

\[
{\rm tr}[(I-P)\Sigma]
>\lVert(I-P)\boldsymbol\mu\rVert_2^2.
\]

Caption it: “projection helps when rejected noise exceeds discarded signal.”
State verbally that the result is exact for a quadratic loss and otherwise a
one-step smoothness guarantee. It cannot beat exact full-batch BP at matched
first-order update norm.

**Paper source:** Fig. 3D and Eqs. 18--20 in the Results.

**Transition:** “That single tradeoff predicts several distinct boundaries.”

### Slide 13 — The theory predicts alignment, bandwidth and gain boundaries

**Time:** 0:50.

**Purpose:** Show that the theory predicts both positive and negative results.

**Composition:** Use three compact conceptual insets across the top, redrawn
from Fig. 3B--E:

1. rotation into the route span increases capture;
2. too-coarse routes discard signal and too-fine routes admit noise;
3. route-specific gains help only when matched to route reliability.

Use Fig. 3F as the dominant lower panel. Reserve the full alignment--bandwidth
plane in Fig. 3G for the final synthesis; at most show a small unlabeled
two-axis icon here.

**Numbers:** Operator utility versus norm-matched one-step progress across 540
trained-factorial conditions gives $\rho=0.937$. Label Fig. 3G's background
“theoretical regimes, not a fit.”

**Visible scope statement:** “Predicts large contrasts and within-family
one-step progress; not every small difference accumulated during nonlinear
training.”

**Paper source:** Fig. 3B--F. Replot F at large size and redraw B--E as icons
or very simple plots. Do not paste the entire seven-panel figure. Fig. 3G is
reserved for the conclusion.

**Transition:** “The first prediction is that shared feedback must fail when
active branches require incompatible changes.”

### Slide 14 — Branch conflict creates a genuine need for branch-specific credit

**Time:** 1:00.

**Purpose:** Define the positive task completely before showing the result.

**Composition:** Redraw Fig. 4A/B at large scale. Show $B\in\{2,4,8\}$
active branches, each receiving a Fashion-MNIST input. A context gate $c_i$
selects one branch for the forward logit. At $\chi=0$, nonselected branches
carry compatible class evidence; at $\chi=1$, they carry opposite-class
evidence. All branches remain locally active in both cases.

**Equations:**

\[
z_i=\mathbf w_{c_i}^{\mathsf T}\mathbf x_{i,c_i},
\qquad
\delta_i=\sigma(z_i)-y_i,
\]

\[
\mathbf d_b^{\rm branch}
=\frac1N\sum_i\delta_i\,\mathbf 1[c_i=b]\,\mathbf x_{i,b},
\]

\[
\mathbf d_b^{\rm shared}
=\frac1N\sum_i\delta_i\,\frac1B\,\mathbf x_{i,b}.
\]

Circle the selector $\mathbf 1[c_i=b]$ in the correct update and show its
absence from the shared update. The $1/B$ factor only matches scale; it does
not restore branch information.

**Paper source:** Fig. 4A/B, redrawn. Do not use an older two-stream task
schematic from the Canvas assets.

**Transition:** “The missing selector lets us predict exactly where the shared
mode stops being a descent direction.”

### Slide 15 — The analytic conflict boundary predicts trained failure

**Time:** 0:50.

**Purpose:** Connect a derived boundary to a large trained effect and its
implementation controls.

**Composition:** Left 35%: Fig. 4C or a clean redraw of the shared-mode curve.
Center 40%: presentation replot of Fig. 4D. Bottom or right 25%: a narrow
implementation-control strip from Fig. 4F.

**Equations:**

\[
C_\chi=(1-2\chi)\mathbf1\mathbf1^{\mathsf T}+2\chi I,
\]

\[
\lambda_{\rm shared}=B-2\chi(B-1),
\qquad
\chi_c=\frac{B}{2(B-1)}.
\]

Mark the predicted thresholds $1,2/3,4/7$ for $B=2,4,8$. State that the
correct-minus-shared effect is negligible at $\chi=0$ and reaches about
35--58 percentage points at full conflict.

**Essential control:** Correct dendritic routes, analytical BP and an
explicitly gated point implementation coincide; shared and deranged routes
fail.

**Visible conclusion:** “Branch-specific information is required; dendritic
material is not the only implementation.”

**Paper source:** Fig. 4C,D,F. Fig. 4E can be a backup paired-effect panel.

**Transition:** “That experiment compares one shared coordinate with full
branch resolution. The next question is whether a few subtree signals are an
efficient intermediate basis.”

### Slide 16 — Subtree routes test partial credit at fixed bandwidth

**Time:** 0:55.

**Purpose:** Define the hierarchy, bandwidth and matched controls before
interpreting the small topology effect.

**Composition:** Upper half: redraw Fig. 5A/B. Eight context streams occupy
the leaves of a binary hierarchy; the selected stream defines the target and
opposite-sign distractors increase with tree distance. Show $K=1,2,4,8$
as progressively finer subtree partitions.

Lower half: redraw the four controls from Fig. 5C/D:

- matched subtrees with correct signals;
- same supports with cyclically deranged signals;
- degree- and depth-matched rewired tree;
- non-anatomical rank-$K$ basis.

Use a small route-rank annotation:

\[
\widehat{\boldsymbol\delta}=A_K\mathbf c,
\qquad {\rm rank}(A_K)=K.
\]

**Paper source:** Fig. 5A--D, redrawn. If this slide becomes crowded, split
task/bandwidth and controls into two sequential builds rather than shrinking
the graphics.

**Transition:** “Correct assignment has a large effect; fine anatomical
topology has a much narrower one.”

### Slide 17 — Tree topology helps only at matched intermediate bandwidth

**Time:** 0:50.

**Purpose:** Present the hierarchy result honestly, including the strong
non-anatomical control.

**Composition:** Use Fig. 5E large on the left. Use Fig. 5F/G stacked on the
right. Direct-label task-matched subtrees, seedwise best non-anatomical control
and deranged assignment. Put a vertical highlight only at $K=4$.

**Visible findings:**

- correct assignment strongly outperforms derangement;
- matched subtrees are worse than the best non-anatomical low-rank control at
  small $K$;
- they exceed that strongest control by only about $1.3$ percentage points
  at $K=4$;
- rewiring removes the intermediate-bandwidth effect;
- all equal-rank routes converge at full resolution.

**Visible conclusion:** “A tree-aligned basis is useful only at matched
intermediate bandwidth; the topology-specific gain is conditional and modest.”

**Paper source:** Fig. 5E--G, rebuilt with large fonts. Never show only the
green accuracy curve without the best-control and rewiring comparisons.

**Transition:** “Backward route resolution is only one role of dendrites.
Serial forward computation is a separate question.”

### Slide 18 — Forward physical depth is distinct from backward route resolution

**Time:** 1:00.

**Purpose:** Define D1--D4 and the mechanism-matched task without confusing
physical depth with feedback bandwidth.

**Composition:** Begin with a two-column distinction:

- $D_{\rm r}$: how finely feedback is routed backward;
- $D_{\rm p}$: how many nonlinear stages are composed forward.

Then redraw Fig. 6A/B. Show three task families: nested factors, flat factors
and local ratios. Beside them show three architectures: serial dendrite,
resource-identical parallel grouped point and flexible parameter-matched point
network.

**Task equation:**

\[
r_j=s_y\prod_{\ell=1}^{H}h_{\ell,a_\ell(j)}.
\]

Define $s_y$ as the class signal and $h_{\ell,a_\ell(j)}>0$ as a nuisance
gain at hierarchy level $\ell$. Inhibitory sensors report these gains. A
shunting stage can divide out one factor when its sensor is placed at the
matching serial stage.

Show four sensor conditions as icons: aligned, independent, trial-shuffled and
reversed. Use $a_{\rm sens}$ for continuous sensor alignment.

**Paper source:** Fig. 6A/B, redrawn. Do not show results until this task and
the grouped-point controls are understood.

**Transition:** “If serial composition is the mechanism, depth should help
only for aligned tasks and ordered sensors.”

### Slide 19 — Serial depth is a task-matched inductive bias, not a universal advantage

**Time:** 0:55.

**Purpose:** Show the large positive depth effect together with the controls
that bound its interpretation.

**Composition:** Left 58%: Fig. 6C enlarged with the flexible point ceiling
retained. Right 42%: Fig. 6F/G as two aligned mini-plots, or one combined
serial-minus-grouped alignment plot with BP and local learning directly
labeled. Put Fig. 6E in backup.

**Numbers:**

- aligned D3 minus D1 under BP: $+30.9$ percentage points;
- exact compartment feedback minus shared somatic feedback at D3: $+11.0$
  points;
- independent, shuffled and reversed sensors remove the depth benefit;
- the resource-identical parallel grouped model does not reproduce ordered
  composition;
- flexible point networks can exceed the serial tree;
- depth is detrimental when the relevant signal--nuisance ratios are already
  local.

**Visible qualification:** “Calibrated mechanism-matched existence test; not
a general point-versus-dendrite expressivity result and no universal
$D_{\rm p}=H$ law.”

**Paper source:** Fig. 6C,F,G. Fig. 6D/E and Supplementary Fig. S18 are backup.

**Transition:** “Controlled tasks establish when structure can help. The next
question is whether real arbors provide the corresponding route resource.”

### Slide 20 — Reconstructed arbors provide sparse candidate address dictionaries

**Time:** 0:55.

**Purpose:** Separate anatomical capacity from evidence of biological use.

**Composition:** Left 38%: an enlarged vector crop of Fig. 7A with a single
branch point highlighted. Overlay a clean redraw of Fig. 7B showing that the
branch point defines a descendant support. Right 62%: use the data in Fig.
7E/F to build one presentation-specific comparison: two large capture/wiring
meters above one paired capture-per-connection plot. Fig. 7G can appear as a
small optional replication strip; do not paste all three paper panels.

**Equation:**

\[
C_A(\mathbf q)
=\frac{\lVert P_A\mathbf q\rVert_2^2}
{\lVert\mathbf q\rVert_2^2},
\]

where $P_A$ projects a modeled compartment credit field onto the route span.

**Numbers that must appear together:**

- at eight channels, subtrees retain about 85% of dense field capture with
  about 7% of the dense feedback connections;
- this is 14.2 times dense capture per connection, but the anatomy-specific
  comparison is only about 2.7-fold relative to a density-matched shuffled
  dictionary;
- the ordering appears in a disjoint 47-cell cohort and ten QC-passing cells
  from a second mouse.

**Visible qualification:** “Modeled fields on measured anatomy; connections
are not cable length, energy, reliability or observed teaching signals. Most
capacity follows coarse branching and depth.”

**Paper source:** Fig. 7A,B,E--G. Keep Fig. 7C/D for backup details.

**Transition:** “A branch point supplies an address. Can a local conductance
change regulate the gain of that address?”

### Slide 21 — Focal shunting regulates route gain only in permissive conductance states

**Time:** 0:55.

**Purpose:** Present the mechanism with its standard-passive null as the main
boundary, not as a footnote.

**Composition:** Left 45%: redraw Fig. 8A. Show a focal shunt and an additive
current matched for the same local voltage change. Only the shunt modifies the
conductance matrix. Right 55%: Fig. 8F large. Optionally place Fig. 8E or G as a
small high-conductance confirmation.

**Intuitive directed-path equation:**

\[
\frac{\partial\log\gamma_n^{\rm cond}}{\partial G_k^I}
=-R_k^{\rm tot}\mathbf1[k\in{\rm path}(n\to0)].
\]

**Exact reciprocal-cable perturbation, if the audience can absorb it:**

\[
\mathbf q'=\mathbf q-
\frac{\kappa_kq_k}{1+\kappa_k(G^{-1})_{kk}}
G^{-1}\mathbf e_k.
\]

Define $G^{-1}\mathbf e_k$ as the Green's-function column controlling the
spatial spread of the change. Do not show both equations at full size; use the
directed-path expression in the core talk and move the matrix identity to
backup unless the workshop is theory focused.

**Visible boundary:** “At the standard passive calibration
$R_m=15{,}000\,\Omega\,\mathrm{cm}^2$, the shunt-minus-current localization
contrast is effectively null. Descendant localization emerges only at lower
membrane resistance or with high background conductance.”

**Visible qualification:** “Modeled steady-state credit transport, not a
general learning benefit of inhibition.”

**Paper source:** Fig. 8A,F. Fig. 8B--E,G belong in backup or a longer talk.

**Transition:** “Route availability and a possible gain mechanism still do
not show that measured biological activity uses these routes.”

### Slide 22 — Measured activity is null; imposed alignment rescues the same routes

**Time:** 1:00.

**Purpose:** Close with the strongest biological boundary and the controlled
test of the theory's conditional prediction.

**Composition:** Use a clean two-stage build. If the final deck is a set of
static PNGs rather than an animated presentation file, make the measured null
and controlled rescue two consecutive slides instead of placing both states
on one PNG.

**Build 1 — measured-response test:** Redraw Fig. 9A as a short pipeline:
measured presynaptic visual responses, conductances mapped to the reconstructed
tree, predicted target response, held-out error. Place large vector crops or
presentation replots of Fig. 9B/C beside it. Headline: nested-subtree feedback
does not outperform random or site-shuffled routes across seven target cells.

**Build 2 — controlled rescue:** Fade the measured pipeline to 35% opacity.
Bring in a redraw of Fig. 9D and a large Fig. 9E:

\[
\boldsymbol\phi(a_{\rm route})
=\sqrt{a_{\rm route}}\,\mathbf u_{\parallel}
+\sqrt{1-a_{\rm route}}\,\mathbf u_{\perp},
\]

with $\mathbf u_{\parallel}\in{\rm col}(A)$ and
$\mathbf u_{\perp}\perp{\rm col}(A)$. Show that capture rises continuously
as the same fixed-energy field rotates into the subtree span.

**Visible conclusion:** “The routes are sufficient when the task is aligned,
but endogenous morphology-specific use is not established by the measured
visual-response task.”

**Visible qualification:** Measured-response learning is a modeled prediction
task on one MICrONS animal; the rescue imposes field direction and uses oracle
projection coefficients. It demonstrates conditional sufficiency, not
biological creation of alignment.

**Paper source:** Fig. 9A--E. Move Fig. 9F/G, the external P+/P- animal
reanalysis, to backup because it is neuron-level evidence and changes
inferential scale late in the talk.

**Transition:** “The wins, ties and nulls occupy different locations in one
alignment-by-bandwidth map.”

### Slide 23 if expanded, or final build of Slide 22 — Conditional synthesis

For the clearest 20-minute talk, make this a separate final slide. For a
strict 18-minute version, use it as the final build of Slide 22.

**Time:** 0:40.

**Composition:** Use a simplified, enlarged version of Fig. 3G on the left and
a redesigned Fig. 9H evidence ladder on the right. Do not reproduce the tiny
paper panel. Place four experimental anchors on the plane:

- standard tasks: neuron identity is the main bottleneck;
- branch conflict: branch coordinates are required;
- $K=4$ hierarchy: narrow aligned subtree benefit;
- measured MICrONS responses versus imposed alignment: null and rescue.

The evidence ladder should read:

1. **Supported:** neuron-specific feedback is useful.
2. **Supported as an information requirement:** branch-specific coordinates
   are needed under conflicting local updates.
3. **Conditional:** subtree routes can be an efficient intermediate basis.
4. **Conditional:** conductance can regulate route gain in permissive states.
5. **Structurally supported:** reconstructed arbors supply sparse candidate
   addresses.
6. **Not established:** endogenous morphology-specific learning-signal use.

**Final sentence:** “Dendrites are not a universal alternative to
backpropagation; they provide structured addresses and gains whose value is
set by task alignment, feedback bandwidth and noise.”

## Paper-panel index

| Paper source | Core use in the talk | Treatment |
|---|---|---|
| Fig. 1A--D | Slides 1--8 | Redraw all schematics and equations at slide scale |
| Fig. 1E | Narrative reference only | Replace with the simpler talk sequence; do not paste |
| Fig. 2A | Slide 9 | Redraw the feedback ladder |
| Fig. 2B | Slide 10 | Replot or retain as a large vector crop |
| Fig. 2C--F | Backup | Fashion uses legacy fallback; D--F are diagnostics |
| Supplementary Fig. S4D | Slide 10 | Replot the current strict-scalar CIFAR-10 ladder |
| Fig. 3A | Slide 11 | Redraw the operator pipeline |
| Fig. 3B--E | Slides 12--13 | Use D as the main plot; simplify B, C and E into prediction icons |
| Fig. 3F | Slide 13 | Large presentation replot |
| Fig. 3G | Final synthesis | Enlarge and redraw; background is theory, not a fit |
| Fig. 4A--C | Slide 14 and start of 15 | Redraw the task and analytic boundary |
| Fig. 4D,F | Slide 15 | Large result plus compact implementation control |
| Fig. 4E | Backup | Paired effect-size detail |
| Fig. 5A--D | Slide 16 | Redraw task, bandwidth and controls |
| Fig. 5E--G | Slide 17 | Keep all three together; they bound the topology claim |
| Fig. 6A,B | Slide 18 | Redraw task families and architecture controls |
| Fig. 6C,F,G | Slide 19 | Main result and alignment boundary |
| Fig. 6D,E | Backup | H4 matrix and saturation detail |
| Fig. 7A,B | Slide 20 | Vector reconstruction plus redrawn route definition |
| Fig. 7E--G | Slide 20 | Recompose into one capacity/wiring comparison and optional replication strip |
| Fig. 7C,D | Backup | Full route-capacity curves |
| Fig. 8A,F | Slide 21 | Redraw intervention; make electrotonic boundary the main plot |
| Fig. 8B--E,G | Backup | Relation, dose, factor-freeze and active-model details |
| Fig. 9A--E | Slide 22, or two static slides | Measured null followed by controlled rescue |
| Fig. 9F,G | Backup | External neuron-level signed-coordinate analysis |
| Fig. 9H | Final synthesis | Redesign as a large evidence ladder |

## Backup slides

The backup should remain technical and use the same notation. Recommended
order:

1. **Full point-neuron chain rule:** output loss derivative and hidden-node
   recursion.
2. **Full conductance quotient derivative:** derive
   $\partial V_n/\partial g_i=x_iR_n^{\rm tot}(E_i^{\rm rev}-V_n)$ line by
   line.
3. **Inhibitory-conductance gradient:**
   \[
   \frac{\partial\mathcal L}{\partial g_j^I}
   =x_jR_n^{\rm tot}(E_I-V_n)\frac{\partial\mathcal L}{\partial V_n}.
   \]
   Explain why the shunt can have a small gradient of its own while changing
   other gradients through $R_n^{\rm tot}$.
4. **General steady-state adjoint:**
   \[
   J_V^{\mathsf T}\mathbf q=\nabla_{\mathbf V}\mathcal L,
   \qquad
   \frac{\partial\mathcal L}{\partial g_i}
   =-\mathbf q^{\mathsf T}\frac{\partial\mathbf F}{\partial g_i}.
   \]
   Contrast a directed path product with reciprocal-cable Green's functions.
5. **Path-gain coefficient of variation:** restore Supplementary Fig. S1A,
   which contains the NeurIPS path-gain-dispersion result. Present it as a
   mechanistic regular-tree precursor, not a trained-learning advantage.
6. **Strict-scalar implementation audit:** current strict scalar versus the
   historical fallback; make clear why neurons do not collapse.
7. **Full standard-task ladder:** Fig. 2C/D and Supplementary Fig. S4D.
8. **Credit-operator proof:** descent lemma, optimum step size and the
   projector signal--noise boundary.
9. **Same-span conditioning:** Supplementary Fig. S14; address capacity versus
   coefficient-learning efficiency.
10. **Full branch-conflict eigendecomposition and paired interactions:** Fig.
    4E and Supplementary Fig. S29.
11. **Full subtree factorial:** Fig. 5A--G with all basis and implementation
    controls.
12. **Physical-depth replication and saturation:** Fig. 6D/E and
    Supplementary Figs. S18, S26 and S28.
13. **Anatomy route curves and second-animal cohort:** Fig. 7C/D/G and
    Supplementary Fig. S27.
14. **Focal-shunting factor freeze and active extension:** Fig. 8B--G and
    Supplementary Fig. S21.
15. **Detailed measured-response controls:** Supplementary Fig. S22.
16. **External P+/P- signed neuronal coordinates:** Fig. 9F/G and
    Supplementary Fig. S17; label as retrospective neuron-level consistency,
    not within-arbor evidence.

## Timing cuts

### Full 20-minute version

Use slides 1--22 and make the synthesis a separate slide 23. Split slide 16
into task/control and result slides if the audience needs more time to absorb
the hierarchy. Total: approximately 19--20 minutes.

### Recommended 18-minute version

Use slides 1--22, with the synthesis as the last build of slide 22. Keep all
task-definition slides. Spend less time on the anatomy wiring ratios and show
only the directed-path equation on the shunting slide.

### Tight 15-minute version

Make these combinations:

- combine slides 3 and 4: BP reference plus local three-factor rule;
- combine slides 5 and 9: coordinate, address, gain and feedback ladder;
- on slides 11--13, retain only the operator definition, utility, Fig. 3D and
  Fig. 3F;
- omit the partial-subtree hierarchy slides 16--17, or state their $K=4$
  result as one inset on the synthesis slide;
- combine physical-depth task and result into one slide;
- combine anatomy and focal shunting into one “available resource and
  conditional gain” slide;
- retain the branch-conflict task, its result, and the measured-null/rescue
  slide. They are the clearest positive requirement and biological boundary.

## What must not happen in the rebuilt deck

- Do not paste an entire main figure onto a slide.
- Do not rasterize paper panels before slide assembly.
- Do not put more than two numerical plots on a slide unless they share axes
  and form one comparison.
- Do not show a result before the task that generated it is defined.
- Do not call D1--D4 cell types; they are model physical depths.
- Do not reuse $\alpha$ for both sensor alignment and route alignment.
- Do not call the CIFAR-10 result a competitive vision benchmark.
- Do not call the branch-conflict result dendrite-exclusive.
- Do not show the $+30.9$-point depth result without the grouped-point and
  flexible-point controls.
- Do not show the 14.2-fold wiring number without the approximately 2.7-fold
  density-matched anatomical control.
- Do not state that shunting generically improves learning; lead with the
  standard-passive null.
- Do not state that MICrONS reveals morphology-specific learning signals; the
  measured-response result is null.

## One-sentence narrative test

Before final export, read only the slide titles. They should form this
continuous argument:

> Global outcomes require local credit; BP defines the exact neuronal signal;
> local learning preserves eligibility but restricts returned information;
> dendrites add address and conductance-dependent gain; standard tasks mainly
> need neuron identity; a signal--noise theory predicts when restriction can
> help; conflicting branches require separate coordinates; aligned subtrees
> help only at intermediate bandwidth; serial depth helps only when its
> operations match the task; real arbors offer sparse candidate routes and
> shunting can regulate them only in permissive states; measured activity does
> not establish endogenous use, but imposed alignment rescues the same routes.
