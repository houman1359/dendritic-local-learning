# Canonical 20-minute workshop slide specification

## Executive decision

The workshop talk is the 20-slide ML-audience deck in
`presentations/ml_workshop_15_20min/`. It is a 17:55 scripted talk with 2:05
left for pauses and transitions. The older TeX decks, Canvas exports, and
longer workshop PDFs are source libraries and backup material; they are not
parallel final decks.

The story is:

1. define credit assignment for network weights;
2. use backpropagation as the exact information reference;
3. factor a local point-neuron update into eligibility and returned signal;
4. extend the destination space from one neuron to its dendritic compartments;
5. derive the conductance model and exact dendritic gradient;
6. use one credit operator to predict when restricted routes help;
7. map the boundary with ordinary tasks, controlled conflict, and nested tasks;
8. test anatomical capacity, conductance-dependent gain, measured functional
   alignment, and a controlled alignment rescue;
9. synthesize the evidence in an alignment–bandwidth phase plane.

Physical depth remains backup. It concerns forward serial computation rather
than the core talk's backward-credit question.

## Authoritative deliverables

- Projection PDF:
  `ml_workshop_15_20min/dendritic_credit_ml_workshop.pdf`
- Editable slide source: `ml_workshop_15_20min/build_slides.py`
- Editable HTML/SVG master: `ml_workshop_15_20min/workshop_deck.html`
- Slide PNGs: `ml_workshop_15_20min/png/slide_01.png` through
  `slide_20.png`
- Speaker/Q&A PDF:
  `ml_workshop_15_20min/dendritic_credit_ml_workshop_speaker_guide.pdf`
- Exact narration: `ml_workshop_15_20min/speaker_notes.md`
- Cut map: `ml_workshop_15_20min/slide_map.md`

Numerical results come from canonical manuscript vectors or frozen source
data. Canvas images are composition references only. Teaching schematics are
native HTML/SVG so labels, geometry, and semantic colors remain editable.

## Exact slide-by-slide content

| # | Visible content and equation | Scientific result or role | Audience question answered |
|---:|---|---|---|
| 1 | Clean labeled arbor; subtitle previews coordinate, address, gain, and alignment. No data or equation. | State the conditional thesis without overclaiming. | Do dendrites replace backpropagation? No. |
| 2 | Forward network, one highlighted weight, global loss, reverse credit arrow. \(\Delta w_i=-\eta\,\partial\mathcal L/\partial w_i\). | Define credit as destination, sign, and magnitude of each weight update. | What is credit assignment? |
| 3 | Feedforward network, reverse recursion, compact literature taxonomy. \(\delta_u\equiv\partial\mathcal L/\partial y_u\). | Backpropagation defines the exact neuron-specific information reference. | Why begin with backpropagation? |
| 4 | Eligibility × returned-signal factorization and scalar-versus-neuron cards. \(\partial\mathcal L/\partial w_i=e_i\delta_u\); \(\Delta w_i=-\eta e_i\delta_u^{\rm avail}\). | Define local learning and explain why scalar feedback limits bandwidth without collapsing neurons. | What is local, and what information is approximated? |
| 5 | Point-to-tree schematic and four cards: coordinate/ownership, address, gain, alignment. \(\widehat{\boldsymbol\delta}^{V}_u=A_uc_u\). | Define every organizing concept before results. | What additional resource does a dendrite provide? |
| 6 | Conductance tree with green E and red I contacts; additive versus shunting cards. Steady-state quotient and \(R_n^{\rm tot}=1/g_n^{\rm tot}\). | E and I are nonnegative conductances distinguished by reversal potentials; shunting changes the denominator. | How is shunting different from additive current? |
| 7 | One somatic error transported to a compartment plus the exact factorization. \(\partial\mathcal L/\partial g_i=x_iR_n^{\rm tot}(E_i^{\rm rev}-V_n)\delta^V_{n,u}\), \(\delta^V_{n,u}=\delta^V_{0,u}\widetilde\alpha_n\). | Separate exact local eligibility from transported compartment error; distinguish the directed path product from the reciprocal adjoint. | What information must feedback deliver to one dendritic synapse? |
| 8 | Scalar, neuron-specific, subtree, and exact fields passing through operator \(M\). \(\mu_{\rm route}=M(\mu+\xi)\). | Place all feedback architectures in one comparison space. | How can biologically different routes be compared mathematically? |
| 9 | Operator utility equation and enlarged prediction plot. \(U(M)=[\mu^TM\mu]^2/\{2L_{\rm sm}[\lVert M\mu\rVert^2+\operatorname{tr}(M\Sigma M^T)]\}\). | \(\rho_s=0.937\) for one-step progress and \(0.916\) for final accuracy across 540 conditions. | Can the theory predict trained outcomes? |
| 10 | MNIST and flattened-CIFAR ladders; three large contrast cards. | Scalar → neuron-specific: +7.8 to +16.4 points; neuron-specific → exact compartment field: −0.86 to +0.19 points; CIFAR exact field approximately matches BP. | Do standard tasks need exact dendritic paths? Mostly no. |
| 11 | Branch-conflict task with all branches active, context-selected target, and conflict dose \(\chi\). | Define the task feature that creates path demand: simultaneously active branches requiring opposing updates. | What task makes a branch address necessary? |
| 12 | Analytic crossing plus trained collapse curves. \(\lambda_{\rm shared}=B-2\chi(B-1)\), \(\chi_c=B/[2(B-1)]\). | Shared feedback fails at the predicted boundary; branch-specific gain is 35–58 points; derangement fails at matched rank/sparsity. | Does failure occur where theory predicts? |
| 13 | Eight-stream nested task and \(K=1,2,4,8\) route partitions. \(\operatorname{rank}(A_K)=K\). | Define K as within-neuron feedback bandwidth, not physical depth. | When can a tree be an efficient low-dimensional basis? |
| 14 | Accuracy across K with direct contrast callouts. | Correct address over derangement: +61.1 points; ancestry over best matched non-anatomical basis: +1.27 points only at K=4. | Is the benefit address assignment or topology? Mostly address assignment. |
| 15 | Measured MICrONS morphology, nested route supports, capture plot, and four metrics. \(C_A(q)=\lVert P_Aq\rVert^2/\lVert q\rVert^2\). | About 85% of dense capture with 7% of route-matrix connections; 14.2-fold dense-normalized and about 2.7-fold over density-matched shuffle. | Do real arbors supply sparse candidate routes? |
| 16 | Matched focal shunt versus additive-current schematic, intuitive gain derivative, electrotonic-boundary plot. | Standard passive calibration is null; descendant localization appears only in permissive high-conductance regimes. | Is shunting a generic learning advantage? No. |
| 17 | Measured-response pipeline and topology-minus-control effects centered near zero. | Seven cells from one mouse show no morphology-specific advantage in measured visual-response analyses. | Do measured cortical responses preferentially use these routes? Not in this cohort. |
| 18 | Fixed-energy rotation into the anatomical span and capture response. \(\phi(a)=\sqrt a\,u_\parallel+\sqrt{1-a}\,u_\perp\). | Imposed alignment rescues representational capture; it is not a trained-learning or endogenous-alignment result. | Is alignment sufficient under controlled geometry? |
| 19 | Large alignment–relative-bandwidth phase plane; three regime cards. | Low bandwidth bottlenecks coordinates; aligned intermediate bandwidth can help; full rank saturates and weak alignment remains null. | What single principle connects the wins and nulls? |
| 20 | Four large take-home cards linked as coordinate → address → gain, gated by alignment + bandwidth. No data plot. | End on the evidence-calibrated boundary map. | What should the audience remember? |

## Visual and terminology contract

- 16:9, 2560 × 1440 sRGB export.
- One dominant visual; at most one secondary result panel.
- Claim-style title and one interpretive bottom ribbon.
- Blue: neuron-specific coordinate or additive control.
- Purple: subtree address or restricted route span.
- Teal/green: correct aligned route, anatomy, or supported result.
- Orange: shared/scalar feedback or conditional result.
- Red: conflict or shunt.
- Gray: deranged, shuffled, rewired, or matched controls.
- Use “learning signal” for the returned task-dependent factor.
- Use “exact compartment field” as the scientific condition name; retain
  “exact path” only when a frozen source plot uses that implementation label.
- \(K\) is feedback bandwidth, \(\chi\) is branch conflict,
  \(\widetilde\alpha_n\) is directed-tree path gain, \(a\) is imposed
  task–route alignment, and \(M\) is the parameter-space credit operator.
- Forward arrows point toward output or soma; returned-credit arrows point
  from loss or soma toward upstream neurons and compartments.
- Never paste a complete manuscript figure sheet into a slide. Crop and enlarge
  only the panel required for the spoken claim.

## Claim guardrails

1. Backpropagation is the exact information reference, not a claimed neuronal
   implementation.
2. A shared scalar limits task-dependent bandwidth but does not equalize
   weights or collapse the hidden layer.
3. The branch-conflict task is a deliberately mechanism-matched existence
   proof.
4. The +61.1-point hierarchy contrast measures correct address assignment; the
   topology-specific matched contrast is +1.27 points.
5. Pair the 14.2-fold anatomy ratio with the approximately 2.7-fold
   density-matched control.
6. State the standard-passive shunting null before the permissive-regime gain.
7. State seven cells and one mouse for the measured-response null.
8. Call the alignment rescue conditional representational sufficiency, not
   trained or endogenous biological use.
9. The operator utility is a one-step guarantee or curvature bound; its
   association with final accuracy is empirical validation.
