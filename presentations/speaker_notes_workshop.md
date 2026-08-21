# Speaker notes — workshop deck (30-minute technical delivery)

Deck: `dendritic_credit_workshop.pdf` — one 42-page build: 35 core frames
(footer `n / 35`) + Backup divider (p.36) + backup frames A1–A6 (pp.37–42).
Note numbers below are the core footer numbers. Register: spoken, "we".

Two conventions in these notes:

- **Intuition** — every hero-equation slide (3, 4, 7, 8, 10, 11, 12, 14, 15,
  16, 17) gets one plain-language sentence. Say it *before* the symbols,
  while gesturing at the panel; then let the equation confirm it.
- **Bold numbers** are the statistics moved *off* the slides in the one-deck
  revision. The slide no longer prints them — the audience only gets them if
  you say them. Non-bold numbers are visible on the slide; you may simply
  point.

## Cumulative time targets

| Act | Frames | Content | Cumulative target |
|-----|--------|---------|-------------------|
| I   | 1–4    | learning = credit assignment | 3:00 |
| II  | 5–8    | state of the art stops at the neuron | 6:30 |
| III | 9–17   | theory: factorization, operator, capture, gain | 16:00 |
| IV  | 18–29  | matched experiments | 26:30 |
| V   | 30–35  | synthesis + predictions | 30:00 |

Rule of thumb: dividers get ten seconds, equation slides get 70–95, result
slides get 40–75. If an act ends more than 45 seconds late, use its
"if pressed" line immediately — do not spread the debt forward.

---

## Act I — Learning as a credit-assignment problem (target 3:00)

### 1. Title — 0:30
We study what dendritic structure adds to *learning* — not to computation,
to learning — and we mean "adds" in a strict resource-accounting sense.
The plan: one exact factorization of synaptic credit, then a battery of
matched experiments that test each factor. The sentence to hold on to:
coordinates identify neurons, dendrites address synapses, conductance tunes
the gain — and alignment decides whether any of it pays.

### 2. One error arrives. Ten thousand synapses must decide. — 0:35
The hook, kept slow and simple: a cortical pyramidal neuron carries on the
order of ten thousand synapses (the journal says "thousands"; ~10^4 is the
order of magnitude, not a measurement), all hanging on one branching arbor —
and the task hands back exactly one output error, at the soma. Which
synapses change, in which direction, by how much? Everything that follows is
one question about the tree in between: is it in the way, or part of the
answer? No notation yet — the tree on the right is deliberately bare.

### 3. Learning dynamics reduce to who gets credit — 1:00
**Intuition:** learning dynamics are bookkeeping — every parameter must be
told which way to move, how strongly, and the activity-times-error product
is the address that finds it.
Under gradient flow the update to one weight is presynaptic activity times
an error term delta. That product carries exactly three things: a sign —
potentiate or depress; a magnitude — how much this weight matters right now;
and a destination — which parameter the credit belongs to. Backprop computes
delta recursively, layer by layer. So the whole question of biological
learning compresses to one line: at each synapse, what stands in for delta?

### 4. Backprop is the reference, not the mechanism — 0:55
**Intuition:** backprop is what the update looks like when the teaching
signal is allowed to know everything; a real synapse is allowed exactly
three local quantities.
The slide shows only the three keywords — speak the violations in full:
transport — the return path needs the transpose of the forward weights;
coordination — ordered Jacobian products across layers; nonlocal state —
delta depends on everything downstream. On the right is the local template:
presynaptic activity, compartment state, one delivered teaching signal. Our
discipline for the rest of the talk: keep the forward physics fixed and
change *only* the information carried by that teaching signal.

**If pressed (Act I):** compress slide 4 to its takeaway — "judge every rule
by what replaces delta" — in one sentence; never shorten slide 3.

---

## Act II — The state of the art, analytically (target 6:30)

### 5. Divider — 0:10
So where is the field? Point at the ladder in the corner — all four rungs
still gray except the first. That ladder fills in as we go.

### 6. Four families of answers — 1:10
**Intuition:** file every proposal by the error term it actually delivers to
the neuron — that one choice determines everything else about it.
Four families. Scalar modulators: eligibility times one broadcast scalar —
spatial precision is whatever the modulator has. Random feedback: a fixed
matrix B replaces the transpose, and it works because updates stay
correlated with the true gradient. State inference: settle the dynamics,
difference the states. Dendritic error compartments: apical–basal
segregation carries one error per neuron beside its activity. Different
hardware — one common denominator, which is the next slide.

### 7. They converge on one per-neuron coordinate — 1:00
**Intuition:** whatever route the error takes to get there, all synapses of
one neuron end up sharing a single number — so the arbor is invisible to the
update.
In the point-neuron gradient, every synapse shares delta-u and differs only
through its own eligibility. There is no variable in this equation that can
treat one subtree differently from another. That is the gap this talk fills.

### 8. This talk: what does the tree add? — 1:10
**Intuition:** the exact dendritic gradient adds a third factor to the
classic two — *where you sit in the tree* — an address and a gain.
Read the preview left to right: local state the synapse sees, the neuron
coordinate the field already delivers, and alpha-tilde — the transported
factor nobody has priced. Three candidate roles for the tree, three tree
pictures below. The rest of the talk derives this factor, prices it, and
tests it.

**If pressed (Act II):** cut slide 6 to a fifteen-second read of its four
tags; slide 7 carries the act and must be spoken in full.

---

## Act III — Theory of dendritic credit (target 16:00)

### 9. Divider — 0:10
Now the theory. Everything follows from taking a conductance-based tree
seriously — two rungs of the ladder are already colored because the field
earned them.

### 10. The forward model is a conductance quotient — 1:10
**Intuition:** a compartment's voltage is a conductance-weighted average of
the batteries feeding it — every input pushes the numerator *and* inflates
the denominator at once.
Steady-state Kirchhoff balance at every compartment; child activity
propagates rootward through the couplings. The point of the slide is the
denominator: normalization is not a bolt-on nonlinearity, it *is* the
membrane equation. The quotient is the mechanism everything else falls
out of.

### 11. The quotient rule yields an exact local eligibility — 1:10
**Intuition:** differentiate a ratio and the driving force falls out for
free — the exact eligibility is built from three quantities the synapse
already has.
Walk the algebra once: numerator sensitivity, denominator sensitivity,
combine — presynaptic activity, input resistance, driving force E minus V.
Additive-current models drop the denominator term and lose that factor.
So conductance normalization does not fight locality; it manufactures the
exact local factor.

### 12. Every exact gradient factors: eligibility × transported error — 1:35
**Intuition:** on a tree, the chain rule collapses to a single product of
junction gains along the one path from soma to synapse — credit travels the
same route as the current.
Top line: the exact gradient is eligibility times delta-n, the loss
sensitivity of this compartment's voltage — and delta-n itself is the
somatic coordinate times alpha-tilde. Bottom line: because the tree's
Jacobian is triangular, alpha-tilde is a path product of junction gains —
slope, input resistance, coupling, at every hop. This is exact, and we
checked it as software: reconstructing gradients through this factorization
against autodiff over **100 diagnostic runs**, the median relative gradient
error was **7.4 × 10⁻⁸**, the maximum **1.9 × 10⁻⁷** — machine noise, not
approximation. Framing sentence: every biological proposal in Act II is an
explicit approximation to this transported factor.

### 13. Dendritic credit is a four-level hierarchy — 0:40
No equations — this is the map. One tree, four decorations: a global scalar,
one coordinate per neuron, K subtree addresses, reliability-weighted route
gains. Rank zero, rank one, rank K, weighted. Each experiment in Act IV
tests one rung against the next.

### 14. A stochastic credit operator predicts when routes help — 1:25
**Intuition:** a restricted teaching signal helps exactly when the signal it
keeps outweighs the signal it throws away plus the noise it lets through.
Model any restricted rule as an operator M applied to a noisy gradient. The
descent lemma gives a guaranteed one-step utility U of M — retained signal
squared over bias-plus-noise. Positivity of g-transpose-M-g is precisely the
feedback-alignment condition. The three ways to die: misalignment, discarded
signal, admitted noise. One bound covers every teaching field in this talk.

### 15. Capture and the tree spectrum pick the best dictionary — 1:05
**Intuition:** capture asks one question — how much of the task's gradient
energy lands inside the span of routes the tree can actually express?
K ancestry routes span the K coarsest tree-Haar modes. So anatomy attains
the rank-K Ky–Fan optimum exactly when the task covariance is
coarse-dominant in the tree's own wavelet basis — that is the bar chart.
The headline is on the slide: at *full task–tree alignment* and K equals
four, ancestry captured 0.55 more task-gradient energy than a random
rank-matched route, fifty of fifty seeds — and near zero under random
alignment, which is the point.

### 16. Bandwidth has an interior optimum — 1:15
**Intuition:** each added route buys signal and noise together — when the
signal spectrum decays faster than the noise, the best bandwidth is
interior, not maximal.
The condition on the left: projected descent beats full bandwidth when
rejected noise exceeds discarded signal. The plane on the right is
*conceptual* — say that out loud; the measured version closes the talk.
The theory was checked in a controlled phase experiment before any
benchmark: the optimal routed depth equaled the generator's hierarchy depth
**for every H tested**; matched depth beat the best mismatched depth by
**0.0957 loss in 48 of 50 pairs**; and the one-step utility argmax picked
the trained minimum in **135 of 200 pairs**. Three anchors to preview on
the plane: too few channels loses, the aligned interior wins, and when
spans coincide routing can only tie.

### 17. Conductance implements reliability-weighted route gain — 1:00
**Intuition:** a shunt is a physical knob on the error path — one local
conductance edit rewrites the transport operator below that branch, and the
optimal setting is a signal-to-noise weight.
The slide keeps one equation: the finite-step optimal attenuation — S over
S-plus-N, a reliability weight. The operator statement is now spoken, not
printed: a shunt is a rank-one conductance edit of the adjoint —
Sherman–Morrison — **subtracting a multiple of a Green's-function column;
dose raises the amplitude as kappa-q over one-plus-kappa-r, monotonically,
and saturates at q over r**. Careful claim: that is a magnitude bound only —
the spatial-selectivity criterion, and the full identity, live in the cut
Sherman–Morrison backup (answer verbally if asked; see the backup section
below). So the fourth rung is physically implementable, branch-locally.

**If pressed (Act III):** skim slide 15 to its takeaway and let slide 22's
crossover argue it retroactively; slides 10–12 and 14 are the spine — never
cut them.

---

## Act IV — Matched experiments (target 26:30)

### 18. Divider — 0:10
Now we test it. Every slide from here: the manipulation, the predicted
quantity, the paired effect size.

### 19. Every comparison is matched-resource and paired by seed — 0:45
Four conventions, stated once: matched budgets — only credit information
changes; paired seeds — effects are within-seed differences; prespecified
designs — cohorts frozen before inference, everything stays in the ledger;
and point emulation — a point network given the same routed fields matches
by construction. The cohorts themselves are worth one breath, because none
of them is on a slide: **a 640-model prospective grid, a 2,700-fit address
factorial, a 10,800-row phase experiment, a 270-fit physical-depth cohort,
and a 512-draw active-conductance ensemble**. Every advantage you are about
to see is a *resource* claim, never dendrite-exclusive magic.

### 20. Per-neuron bandwidth dominates: +4.9 to +14 points — 1:00
Rung one: replace one broadcast scalar with one signed coordinate per
neuron. Split it by task, because the slide only prints the range: MNIST
**+4.9 to +7.0 points**; the noise task **+9.8 to +14.2** — a hundred
twenty of a hundred twenty valid pairs, the prospective 640-run cohort. The
panel shown is the 15-seed regular-tree replication (say so if someone
counts seed lines). And a replication that is now voice-only — its panel was
dropped from the slide: **Fashion-MNIST with no retuning at all, +4.5 and
+3.8 points, ten of ten seeds**. Attribution matters: this is bandwidth,
one to N — the classic Werfel–Xie–Seung bottleneck — not routing.

### 21. Correct ownership adds a small, reproducible margin — 0:40
**Intuition (spoken, no hero equation):** keep the exact same error values
and mail them to the wrong neurons — same bandwidth, wrong map.
A derangement of the neuron-to-tree assignment. Correct ownership wins by
0.15 to 0.7 points, fifty-eight of sixty pairs — on the slide. Say the
punchline: identity buys points; ownership buys fractions — but
reproducible fractions.

### 22. Subtree addresses win only at the predicted budget — 1:15
The factorial: 2,700 fits, twenty paired seeds, K from one to eight. Walk
the accuracy ladder out loud — **0.187, 0.449, 0.801, 0.810** at K equals
one, two, four, eight. Against the best bandwidth-matched control:
**minus 36 points at K one, minus 15 at K two**, then the win the slide
prints — plus 1.3 (**exactly +1.27**) at K four, fifteen of twenty seeds,
P 0.005 — and **a tie at K eight**. Against a rewired tree: **plus 23.1 at
K two, plus 5.0 at K four, twenty of twenty seeds each**. The crossover
lands where the capture spectrum put it on slide 15 — an interior optimum,
observed where predicted. Be explicit that this is conditional by design.

### 23. Depth pays when route scales match the task hierarchy — 0:55
Controlled validation: the generator puts class signal at levels up to H
and noise below. Best routed depth equals H for every H; matched depth
beats the best mismatch by **0.0957 loss, forty-eight of fifty**. The
alignment dose is the thing to say slowly, because only its final step is
printed: **0.02, 0.21, 0.82, 3.13, then 30.86 points** as alpha goes zero
to one — the final step is the +27.7 on the slide. Alignment is a steep
boundary the data cross, not a linear dose. One more voice-only item — its
panel was dropped: **an independent H-equals-two physical replication shows
the same signature**. The slide keeps only the depth-benefit dose curve;
the per-depth accuracy curves and dose contrasts say the same thing — quote
them if asked.

### 24. Aligned divisive depth helps; depth per se costs — 1:15
**Intuition (spoken, no hero equation):** every stage is itself a small
divisive normalizer — compose stages along the task's own hierarchy and
depth pays; compose them any other way and depth costs.
Construction first, because the matching is the argument: **the same eight
branch units and the same 66,178 parameters at every depth; stage widths
eight, then two–three, then two–one–two**. Aligned D3 minus D1: the slide
says +31 — **exactly +30.86, range 30.45 to 31.30, ten of ten seeds**.
Controls: **reversed placement minus 0.43; interaction variants all sit
near 31**; and **raw-additive stacking declines, 0.573 to 0.519**. Then the
paired negative, which buys us credibility: at fixed budgets depth four
*loses* to depth one — **minus 7.92 points under scalar feedback, minus
1.06 neuron-indexed, minus 1.37 under exact transport and backprop alike**
— and **flexible point MLPs beat serial D3 by about seven points**. This is
a constructed existence proof at a calibrated operating point — conditional
on alignment, never free.

### 25. Real arbors: most of dense capture at 7% of the wiring — 0:55
**Intuition (spoken):** an arbor is a sparse dictionary — each route is
just the set of synapses under one branch, weighted by one number.
Eight reconstructed MICrONS cells. Eight ancestry channels capture
**0.487** — 85 percent of the dense oracle — at **6.92 percent** of its
wiring: **14.2 times the capture per unit wiring, range 10.7 to 18.2**. A
density-matched shuffle already gets **5.3 times**, so the anatomy-specific
factor is the 2.7 on the slide. Two voice-only texture numbers: **the
median route addresses 2.7 percent of the mapped input**, and **the
participation rank of the route weights is 20.8 — against exactly 1 for a
scalar**. The honest boundary: **fine morphology does not beat depth- and
degree-matched controls, P 0.078**. Sparsity does most of the work; anatomy
adds a bounded factor on top.

### 26. A focal shunt rewrites transport below the branch — 1:00
The mechanism experiment. Somatic voltage and output error matched; only
the operator changes. Shunting localizes descendant credit at **0.104**
against **0.035** for the additive control — the 0.069 contrast on the
slide, **CI 0.053 to 0.086**, eight of eight cells, and dose-monotone from
**0.018 up to 0.128**. The factor decomposition is the argument, and it is
all voice now: **the adjoint factor contributes +0.087 and the
driving-force factor −0.017 — they partially oppose**; **freeze the adjoint
and the contrast collapses to 0.026; restore it and you recover 0.104**.
Shunting edits the transported error, not the local driving force.

### 27. The mechanism needs high-conductance states — 0:50
At textbook resting calibration — **R-m fifteen thousand ohm
centimeters-squared** — the effect is null: **minus 0.0019, two of eight
cells**. Raise membrane conductance toward in-vivo-like states — Destexhe
2003 — and it re-emerges: **0.0164 at a thousand**, then the slide's 0.067
(**0.0674**) at three hundred, eight of eight. And in **512 accepted active
steady states with sodium, potassium, calcium, HCN and NMDA conductances,
the contrast is 0.382, eight of eight — attenuation only, no sign flips**.
Frame it as strength: state-dependence is a *prediction* of the mechanism,
a testable boundary, not a failure.

### 28. Anatomy alone is not enough; alignment rescues it — 1:00
**Intuition (spoken):** we freeze everything about the dictionary —
channels, gradient energy, curvature — and rotate one thing: how much of
the task's gradient lies in the anatomical span.
First the honest null, in numbers the slide no longer prints: with measured
response statistics, **partial shared-path correlation −0.048, CI −0.224 to
+0.118; full-tree delta-MSE 0.007, CI −0.021 to +0.032 — thirteen scans,
520 fits**. Then the rescue: at alignment zero the same dictionary loses
eight of eight; at alignment one it wins eight of eight, and capture tracks
twenty-step learning at rho 0.99 — that scatter is the slide's one panel;
**the twenty-step learning-curve panel was dropped from the slide** — quote
it if asked. Caveat sentence, verbatim: **these are oracle projection
coefficients on quadratic objectives — sufficiency, not evidence the
circuit uses it endogenously**.

### 29. Animal teaching signals are signed and neuron-specific — 0:45
Six-animal BCI reanalysis; causal signs fixed by the experimental mapping.
The panel carries the 83.7 percent headline; the rest is voice:
**positive-mapped population +0.075, five of six animals, P 0.0313;
negative-mapped −0.150, six of six, P 0.0156; signed separation +0.225 in
all six**. The signed, neuron-specific mode carries the energy — **CI 60.6
to 95.7 percent — against 16.3 percent for a common scalar**. One honesty
clause: **with six animals, exact permutation tests sit at or one step
above their resolution floor** — the sign consistency is the evidence. So
the coordinate rung exists in vivo. Within-tree transport is the open
measurement — which is where our predictions land in Act V.

**If pressed (Act IV):** drop slides 21 and 27, giving each one sentence —
"ownership adds reproducible fractions of a point" and "the shunt effect
needs high-conductance states" — and point to backup; never drop 20, 22,
24, 26 or 28.

---

## Act V — What this teaches about brain learning (target 30:00)

### 30. Divider — 0:10
Ladder complete. What does it teach us?

### 31. Alignment × bandwidth decides when structure pays — 1:00
Every experiment on one measured plane — the fitted version of the
conceptual plane from slide 16. The tikz legend on the right names the
markers; the magnitudes stayed inside the figure, so say them as you point:
the trained factorial's interior optimum at K four; the spectral sweep
riding from **−0.01 to +0.55** as alignment rotates in; MICrONS controlled
fields from **−0.13 to +0.71** across a zero to one; the measured-response
null sitting exactly where the theory predicts a tie — **effective task
rank 1.07**, so every rank-matched span coincides; and credit reversal,
where opposed streams *force* routing — **+4.4 points**. Say it plainly:
the nulls land where the theory puts them — that is what makes this a
theory and not a collection of wins.

### 32. When a local rule wins is settled by four inequalities — 0:40
**Intuition (spoken):** when a local rule wins is not a matter of taste —
it is four inequalities, and we tested each one.
Ceiling: nothing restricted beats exact full-batch backprop at matched step
norm. Noise: projection beats stochastic backprop iff rejected noise
exceeds discarded signal. Span: anatomical routes beat bandwidth-matched
routes only under covariance alignment. Address: any within-tree address
beats the shared coordinate whenever transported credit varies across the
tree — the cosine formula makes that quantitative.

### 33. Why dendrites: one arbor multiplexes K channels — 0:35
**Intuition (spoken):** one arbor carries K teaching channels down a single
feedback wire; a point system buys the same channels with K times the
wiring.
The counting argument behind the whole program: a K-route dictionary
expresses up to rank-K teaching fields inside one neuron. In reconstructed
cells that was 85 percent of dense capture at seven percent of the wiring —
both on the slide. Dendrites relocate address resources; the ledger is
wiring.

### 34. The theory is testable with focal dendritic inhibition — 0:35
Four predictions, all branch-resolved — the slide carries only keywords,
speak them in full: focal inhibition during learning shifts later
plasticity preferentially *below* the perturbed branch, at matched soma and
output; the effect scales with dose and local input resistance and
saturates — **the bound is q-k over r-k, from the Sherman–Morrison
identity**, spoken here because the formula lives only in backup now; it
appears in high-conductance states and is null at rest; and learning
benefit tracks task-weighted capture, not morphological complexity.

### 35. Four questions, four answers — 0:30
Read the four answers as a landing, one per act, pointing at the completed
ladder: one signed coordinate per neuron; eligibility times transported,
routed error; yes — at the predicted budget, under alignment; one phase
plane organizes every result. Then close on the mantra in the takeaway bar,
slowly: coordinates name neurons; trees address synapses; conductance tunes
the gain; alignment decides the value. Thank you.

**If pressed (Act V):** drop slides 32 and 33; close on the phase plane,
the predictions and the answers.

---

## Backup quick reference (Q&A jumps; same PDF, divider p.36)

- **A1 (p.37) adjoint chain** — "isn't this just backprop?" Yes, made
  exact: implicit-function adjoint; Almeida–Pineda lineage; symmetric
  passive cable makes it a Green's-function solve.
- **A2 (p.38) U(M) derivation** — exactness caveat; U predicted one-step
  progress at rho 0.937 and final accuracy at 0.916 across 540 conditions;
  capture alone predicted one-step progress at 0.877.
- **A3 (p.39) factorial controls** — topology-alignment margins, and
  gated-point / flat / grouped implementations coincide *exactly* given the
  same routed fields — the "no dendritic magic" receipt.
- **A4 (p.40) validity audit** — outcome-independent exclusion of 840 of
  1,840 runs; 320-run clean audit; exact-vs-BP agreement on average.
- **A5 (p.41) reliability gain + same-span** — trained-model check of
  a-star: one-step ordering holds (49/50), final-loss difference null;
  same-span conditioning trades bias against variance, predicted crossovers
  observed. Quote if someone pushes on slide 17.
- **A6 (p.42) references.**

Five further backup frames (quotient-rule steps, Ky–Fan/weighted capture,
Sherman–Morrison selectivity, same-span bias–variance in full, full-tree
all-scan boundary) were cut from the compiled deck; they are preserved in
the `\iffalse` block of `dendritic_credit_workshop_appendix.tex` and can be
restored by moving a frame above that block. Two verbal answers drawn from
that block, since their numbers appear nowhere in the compiled deck:

- **Sherman–Morrison in full** (backs slides 17 and 34): the shunt updates
  the adjoint as q' = q − [κ q_k / (1 + κ (G⁻¹)_kk)] G⁻¹ e_k, with
  amplitude |a_k| = κ|q_k| / (1 + κ r_k), saturating at |q_k|/r_k; a
  current-matched additive perturbation changes the drive b but not G, so
  it cannot produce this update.
- **Full-tree all-scan boundary** (backs the measured-null discussion on
  slides 28 and 31): all 13 eligible scans, 520 fits — exact compartment
  error MSE 0.788; topology 0.832; random routes 0.833; site-shuffled
  0.839; topology-minus-shuffle 0.0070, CI −0.0207 to +0.0321 — no
  reliable advantage; a single coarse channel already captures 0.965 of
  the measured-response credit energy (effective rank 1.07), so every
  rank-matched dictionary spans it — the predicted-tie regime.
