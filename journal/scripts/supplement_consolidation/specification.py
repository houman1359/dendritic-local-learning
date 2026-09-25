"""Editorial panel selection; all plotted observations remain unchanged.

The source panel names refer to the pre-consolidation published-artwork
registry. These identifiers are provenance coordinates, not live citations.

2026-09-10: rebuilt against the frozen supplement table
``analysis/figure_overhaul_20260908/v2/SI_NUMBERING.md`` (36 figures, S1--S36),
which executes AMENDMENTS B1--B8 and DECISIONS Q1--Q7.  ``FIGURES`` is the
single source of truth for the S-number: it is the index of the entry in this
list, enumerated in module order by ``build.py``.
"""
from __future__ import annotations

# (semantic file/label id, prose module, source panel groups, title, caption)
# A group ('Sx', '*') retains a complete native source sheet.  Groups are
# consumed in order, so the order of the groups is the printed panel order.
FIGURES = [
# ---------------------------------------------------------------- si_01 ----
('mechanistic_chain', 'si_01_exact', [('S1', 'ABC')], 'Path gains, error alignment and forward inhibition in regular-tree image models.', r'''DendriNet steady-state image models. \textbf{A--C}, MNIST in two-stage $[4,4]$ directed trees. \textbf{A}, Coefficient of variation of conductance-stage path gains (standard deviation divided by mean), in shunting and normalized-additive models with five inhibitory synapses per branch: dots, five paired seeds; means and bars, one sample standard deviation. The schematic uses the balanced tree of main Fig.~1B, rather than the experimental $[4,4]$ geometry; $\alpha_1,\alpha_2,\alpha_3$ label serial gains and $\delta_0$ the somatic error; darker sites have larger path gains. This loading measure omits branch derivatives. \textbf{B}, Stage-resolved cosine of matched-width scalar-fallback and exact errors; the width-matched somatic stage is excluded because its cosine is one by construction. Points are checkpoints; bars show their standard deviation. \textbf{C}, Post-training inhibition interventions: zero activity, sample shuffling, per-branch batch mean and uniform matched mean. Points are retained runs; bars show their standard deviation. The dashed line marks ten-class chance. These interventions also change forward computation and do not isolate backward shunting. Historical noise-task panels with unresolved execution or normalization settings are archived and excluded from the displayed evidence (Section~\ref{note:credit_images}).'''),
('credit_validation','si_01_exact',[('S2','AD'),('S3','C')],
 'Local eligibility and exact transport reproduce the learning gradient.',
 r'''DendriNet image models with shunting or additive integration. \textbf{A}, Test accuracy on MNIST, Fashion-MNIST (F-MNIST) and figure-ground MNIST (FG-MNIST) for the shunting architecture trained by backpropagation (red-brown) or by the local rule (green), and for the normalized-additive architecture trained by the local rule (blue); bars are means $\pm$ 1 s.d. across five seeds, and the backpropagation intervals are narrower than the line weight. \textbf{B}, Transport, activation and additive-model implementation controls on MNIST; points are means $\pm$ 1 s.d. across five seeds, and the backpropagation interval is smaller than its marker. Colour names the architecture as in \textbf{A} (red-brown backpropagation, green shunting, blue additive, violet exact path transport); within a family the reference configuration (circle for shunting, square for additive) is the configuration drawn in \textbf{A} and a triangle is the ablated variant (identity activation without tanh reactivation; raw additive without normalization). The additive group draws the complete normalized/raw $\times$ gain-mode factorial, one gain mode per row. The MNIST backpropagation and normalized-additive comparisons in \textbf{A} and \textbf{B} use separate cohorts; their means are not paired outcomes. \textbf{C}, Exact-transport-minus-backpropagation accuracy across three-/five-factor and local/backpropagated decoder choices; open circles are five paired seeds, diamonds are condition means and bars are 95\% paired-seed bootstrap intervals; dark violet denotes a backpropagated decoder and lighter violet a local decoder. Exact transport supplies an oracle compartment error. The local three-factor eligibility is sufficient when this error is supplied. Normalized- and raw-additive models are distinct controls; no noise-task shunting aggregate is used here.'''),
# 2026-09-11: the six S3 caption requests applied, then condensed to 18 caption
# lines: the full requested text overflowed the page by 128.6 pt under the
# 503.7 pt sheet (make supplement, 'Float too large').
('utility_signal_noise','si_01_exact',[('N3','*')],
 'Signal retention and noise jointly constrain one-step progress.',
 r'''Quadratic/operator analyses (\textbf{A--D,G}); DendriNet checkpoints (\textbf{E}); quadratic learning on anatomical routes (\textbf{F}). Dots, individual units; symbols, means and 95\% bootstrap intervals (\textbf{C}, seeds; \textbf{E}, checkpoints; \textbf{F}, cells, one Monte Carlo stream each). Intervals in \textbf{C,E} lie under markers. \textbf{B,D} colour printed cell values by sign (red positive, blue negative) and magnitude; \textbf{B}'s $-0.01$ controls are near-neutral. \textbf{A}, Fixed-operator one-step guarantee: $U(M)=\max(\bm\mu^{\top}M\bm\mu,0)^{2}/(2L\,\mathrm{E}\|M(\bm\mu+\bm\xi)\|^{2})$ ($\bm\mu$, exact gradient; $M$, linear update operator; $\bm\xi$, zero-mean noise; $L$, loss smoothness) lower-bounds the expected decrease at the bound-optimal nonnegative step $\eta^*$, subject to smoothness along the update segments (Eqs.~\ref{eq:s_operator_bound},\ref{eq:s_operator_utility}); zero updates have $U=0$. \textbf{B}, Subtree-minus-random spectral capture over route budget $K$ and covariance mixture $\rho$, means of 50 paired seeds per cell; $K=16$ (full rank) and $\rho=0$ are by-construction controls, only the endpoints isospectral. \textbf{C}, Final loss against route resolution $D_{\rm r}$ for hierarchies $H_{\rm c}=1$--$4$ (teal, light to dark; circle, square, triangle, diamond); $D_{\rm r}=4$ resolves every leaf ($M=I$); 50 paired seeds per point. \textbf{D}, Exact projection boundary: retaining signal fraction $f_{\rm sig}$ and noise fraction $f_{\rm noise}$ changes normalized loss by $(f_{\rm noise}-f_{\rm sig})/2$; the dashed boundary is $f_{\rm noise}=f_{\rm sig}$, and cells extend to the midpoints between sampled fractions. \textbf{E}, Gradient cosine and norm-matched progress at 120 trained checkpoints (one held-out batch; relative step $10^{-5}$): strict scalar amber, per neuron salmon. Two below-range scalar values are labelled at the floor. Boxes show medians, quartiles and 1.5-interquartile-range whiskers; open diamonds, means. Dashed references: exact path (dark red, 1 by construction), zero (grey). \textbf{F}, Eight-arbor positive control: twenty-step projected loss reduction relative to the full gradient (dashed, 1), with route-aligned credit (green circles, morphology-selected; grey squares, random paths; blue triangles, depth bins; pink diamonds, ancestry-shuffled). \textbf{G}, Analytic rank--noise trade-off, $2L$ times the optimized bound $q^2/(q+K\sigma^2)$ where $q$ is retained gradient energy and $\sigma^2$ the noise variance per retained direction, for $(K,q)=(1,0.8)$ (solid) and $(2,1)$ (dashed), crossing at $\sigma^2=4/7$.  \textbf{F} imposes route alignment and uses oracle projections. These checks do not test biological credit or long-horizon selection; endpoint selection fails prospectively (Fig.~\ref{fig:si_original_selector}).'''),
('reliability_gain','si_01_exact',[('S13','AF'),('S24','CD')],
 'Fixed and estimated reliability gains have bounded learning benefits.',
 r'''Standalone NumPy parallel-conductance model with logistic readout. All panels use positive-rate synthetic tasks with eight parallel, nonserial branch blocks. \textbf{A}, A supplied compensating current preserves branch voltage while a positive shunt reduces input resistance and eligibility: $G$ is baseline total conductance, $\kappa_b\geq0$ the added shunt on branch $b$, and $V'=V$ the imposed voltage match. \textbf{B}, Paired control-minus-aligned final-loss contrasts at maximal heterogeneity for the best global, shuffled, anti-aligned and unshunted controls; fixed aligned shunting has no reliable final advantage over unshunted noisy learning. \textbf{C,D}, Adaptive-rule final loss and paired control-minus-adaptive contrasts. Adaptive local gain beats global and shuffled attenuation but loses to no shunt and the fixed oracle. In \textbf{B,D} the dashed line is the aligned or adaptive-local reference, translucent dots are the 50 individual paired seeds, and diamonds with bars are means and 95\% paired-seed bootstrap intervals; an interval narrower than its diamond is hidden by it. The point implementation given identical gains reproduces the conductance final loss exactly in all 50 seeds of each study (paired difference 0, 50/50 ties). Curves and shading in \textbf{C} are means and 95\% paired-seed bootstrap intervals across the same 50 seeds. The supplied compensating current in \textbf{A} and the paired gradient observations from which the adaptive rule in \textbf{C,D} estimates its gains are explicit information resources.'''),
('same_span_conditioning','si_01_exact',[('S14','ABEF')],
 'Identical route spans can learn differently through conditioning.',
 r'''\textbf{A}, Haar, raw nested (redundant) and scaled nested (statically scaled) coordinates parameterize the same rank-eight span in sixteen abstract coefficients; they share one projector but not one Gram matrix. \textbf{B}, Positive eigenvalues of the dictionary Gram matrix $\Phi^{\mathsf T}\Phi$; condition numbers are 1, 15 and 388.52. \textbf{C}, Population loss after eighty updates across effective sample size on a logarithmic axis; curves and bands are means and 95\% seed-bootstrap intervals across fifty seeds. The strip above the axis shows, for every seed, the predicted effective sample size at which the exact finite-time risks of the nested and Haar coordinates are equal (raw nested: median 38.5, interquartile range 31.1--63.7; scaled nested: median 8.2, interquartile range 6.3--12.6); large symbols and dotted guides mark the medians. \textbf{D}, Paired nested-minus-Haar contrasts; negative values favor nested coordinates. Points and bars are means and 95\% paired-seed bootstrap intervals across fifty seeds (n = 50 pairs); dashed curves are the exact finite-time risk. Slow modes reduce variance at low data and retain bias at high data. Gram-preconditioned controls agree to $1.15\times10^{-15}$. This is a coefficient-learning comparison within one span, not a simulation of forward dendritic morphology or a biological implementation of the preconditioner.'''),
# ---------------------------------------------------------------- si_02 ----
('image_generalization','si_02_credit_rules_images',[('N4C','CDE'),('S30','CD')],
 'Feedback resolution across image and feedback controls.',
 r'''DendriNet image-classification networks. \textbf{A,B}, Flattened CIFAR-10 in shunting and raw-additive architectures, respectively. Each compares strict scalar, per-neuron, exact-path and backpropagation feedback over twenty paired seeds, using a separately validation-selected training recipe held fixed across rules. Architecture means do not isolate forward operators because training recipes and GPU types differ. Panels share one accuracy axis; bars show 95\% Student-$t$ intervals. Grey bands mark $\pm1$ percentage point around each backpropagation mean; the band alone does not establish equivalence. Shunting uses an additional stopping extension on the original twenty seeds; all runs satisfy the stopping checks, with unchanged selected accuracies. \textbf{C}, Fashion-MNIST in shunting and raw-additive trees (ten paired seeds; 95\% bootstrap intervals). Four rules include both strict scalar and the less restrictive matched-width fallback; this fresh cohort replaces the earlier fallback-only comparison. In \textbf{A--C}, filled symbols are means and small open symbols individual seeds. \textbf{D,E}, MNIST accuracy contrasts under direct feedback alignment (DFA), which supplies fixed random soma feedback: larger contrasts in \textbf{D} and smaller contrasts in \textbf{E}, whose axis is expanded about twentyfold. Open symbols are fifteen paired seed differences, filled symbols their means and bars 95\% paired-seed bootstrap intervals; blue, raw additive; green, shunting. The top rows of \textbf{D,E} show the MNIST-DFA estimates of main Fig.~1E,F at supplement scale. DFA changes the feedback source; it does not add a dendritic layer or increase the dimensionality of the ten-class readout error.'''),
('mnist_dictionary_geometry', 'si_02_credit_rules_images', [('N7', '*')], 'Intermediate dictionaries capture more image-task credit without improving accuracy.', r'''DendriNet image-classification networks. \textbf{A,B}, Six-rule MNIST comparison in shunting (green) and raw-additive (blue) networks, with ten fresh paired seeds per architecture. Small dots are seeds; filled circles show means at development-selected rates and open squares means at the original common rate. A ringed filled circle denotes a single fit used for both rate policies. Projected $K=1$ and $K=3$ rules obtain their coefficients from the exact activation-error field; per-neuron feedback instead broadcasts the readout-derived neuronal error. Decoder-only learning freezes the initialized core. The shared accuracy axis makes small differences among the four dendritic rules difficult to see; \textbf{C} resolves the within-tree contrasts. \textbf{C}, Paired accuracy differences for three projected subtree profiles versus one projected common profile, and for exact paths versus three profiles. Within each group, marks show shunting selected/common rates, then additive selected/common rates. Dots are ten paired seed differences; symbols and bars are means and descriptive 95\% seed-bootstrap intervals. The dashed line marks zero; counts give the number of positive differences among ten seeds. These small effects contrast with the large benefit of preserving neuronal identity. Activation-error capture from these same fits is shown once, in main Fig.~1G; voltage-space and aggregate-energy capture remain in Source Data. The MNIST comparisons in this figure use one dendritic layer and a linear readout.'''),
('error_field_geometry','si_02_credit_rules_images',[('N8','*')],
 'Error-field geometry separates direction, amplitude and spatial variation.',
 r'''DendriNet checkpoint analysis. Independent fifteen-seed flattened-MNIST cohort of 128 directed $[3,3]$ trees per network; green circles denote shunting and blue squares raw-additive trees, offset at each category. \textbf{A}, Branch-gradient cosine with the exact gradient at common fixed checkpoints trained with per-neuron feedback; matched-width scalar-fallback and per-neuron fields are evaluated at identical weights and states, and the exact-path cosine is one by construction and is not drawn. \textbf{B}, Exact voltage-error magnitude by depth at exact-path checkpoints: the batch root-mean-square ratio $\operatorname{RMS}(\delta^V_{n,u})/\operatorname{RMS}(\delta^V_{0,u})$, averaged over neurons and paths, on a logarithmic axis, with somatic value one by normalization, marked by the dashed grey rule at one from which every per-seed profile starts; no symbol is drawn at the soma. \textbf{C}, Within-depth path-specific residual error energy at the exact-path checkpoints of \textbf{B}, after subtracting the mean across paths within depth $d$: distal shunting attenuates magnitude but retains the larger path-specific fraction. In every panel the thin lines with small marks are the fifteen paired seeds per architecture and the large symbols joined by the heavy line are seed means. In \textbf{A} and \textbf{C}, whiskers in the series colour are 95\% seed-bootstrap intervals of the mean (10,000 whole-seed draws), visible where they protrude beyond the symbol and otherwise within it; in \textbf{B} every interval is at most 0.08 ratio units, within the symbol, and none is drawn. The dashed grey rule in \textbf{A} marks cosine zero; the key names the seed-mean and single-seed glyphs. No additional models are trained here; greater path-specific variation did not yield an image accuracy benefit, and voltage- and activation-space capture are different quantities.'''),
('input_coverage_depth', 'si_02_credit_rules_images', [('S7', 'ABC'), ('N9', 'A')], 'Spatial input maps improve coverage and image classification across credit rules.', r'''DendriNet image models and their input-contact maps. Depth-four $[2,2,2,2]$ trees compare spatially partitioned and randomly assigned contacts. \textbf{A}, Sixteen branches of an example neuron on the $28\times28$ input plane: spatial map, left; random map with the same seed, right. Shading identifies branches; pale pixels are uncontacted. \textbf{B,C}, Unique input coverage and cross-branch Jaccard overlap (intersection divided by union of contacted pixels), across 1,280 maps per condition (128 neurons in each of ten seeds; mean $\pm$ standard deviation). Contact counts are matched; spatial maps reach all 336 contacted inputs without cross-branch collisions. \textbf{D}, MNIST spatial-minus-random accuracy under backpropagation (BP), matched-width scalar fallback (MW scalar), neuron-specific sharing (Neuron) and exact transport (Exact path). Each seed averages shunting and raw-additive models. Small points are ten paired seeds; larger symbols and bars are means and 95\% paired-seed bootstrap intervals. The benefit under BP identifies a forward input-coverage effect. Historical noise-task comparisons are archived because the executed generator is unresolved; they do not support an ordinary-task depth boundary here.'''),
# ---------------------------------------------------------------- si_03 ----
('branch_conflict_controls','si_03_conflict_ancestry',[('S19','A'),('S29','DEF')],
 'Branch-selective credit limits interference and preserves update direction.',
 r'''Logistic classifiers with context-selected input blocks and prescribed feedback supports. \textbf{A}, Two-branch learning in ten paired seeds: held-out accuracy on the balanced binary task, with chance at 0.5 (dashed rule) and the axis broken over the two spans that hold no seed. Points show seeds; open diamonds and bars are means and 95\% paired-seed bootstrap intervals, which are narrower than the diamond where no bar is visible. The exact-path, correct-ancestry and gated-point rules (and backpropagation) give bitwise-identical accuracy in every seed and share one row. The random rank-2 control is bimodal across seeds (seven at chance, three recovered), so its row shows the median as a vertical bar and the two counts instead of a mean. The context-0 forgetting of the same cohort is shown in main Fig.~2E. \textbf{B--D}, Shared-versus-exact gradient cosine in the separate branch-conflict experiment with two (\textbf{B}), four (\textbf{C}) or eight (\textbf{D}) branches, drawn against the eight sampled conflict doses at equal spacing (tick labels give $\chi$; the odd doses $4/7$ and $2/3$ bracket the boundaries). Solid curves evaluate common exact-learning states at epochs 0, 50 and 250; the dashed amber curve evaluates the shared learner's own final state. The violet dashed rule is the mean-field boundary $\chi_c = B/[2(B-1)]$, at 1, $2/3$ and $4/7$. Curves use twenty paired seeds; the 95\% bootstrap bands are narrower than the line stroke except near the boundary at epoch 0. Per-seed zero-alignment doses are tabulated in Source Data. An exact quadratic interference calculation is given in the accompanying derivation. Learned-state geometry can depart from the initialization mean-field approximation.'''),
('ancestry_coefficients','si_03_conflict_ancestry',[('S23','A'),('S32','CD')],
 'Available route span and learned route coefficients are different constraints.',
 r'''Logistic input-block classifiers on a feedback-only tree, with a separate supervised cue encoder. \textbf{A}, Trained field capture by route budget in the eight-context factorial, with $K$ on an ordinal axis. The ancestry, depth-bin and random-sparse partitions have coincident capture at matched bandwidth (within 0.005 at every budget; one stroke, drawn at the ancestry values), and derangement removes the target coordinate. The inset resolves the three partitions at $K=2$ and $K=4$ as capture minus the ancestry mean, in percentage points of capture, with each family's own 95\% interval; $n=20$ seeds throughout, and on the main axes the intervals are narrower than the symbols. Capture correlates with held-out accuracy across conditions (within-seed Spearman correlation about 0.82 over all budgets and 0.64 for $K<8$; Source Data), but does not explain the small ancestry-specific $K=4$ advantage over controls with similar capture. \textbf{B}, Held-out accuracy versus calibration set size (16, 64 and 256 examples, ordinal axis) at cue-noise standard deviation 0.5 and zero delay; 30 epochs are fixed, so both data and computation increase. \textbf{C}, Cue delay (0, 1 and 4 trials, ordinal axis, sharing the accuracy axis of \textbf{B}) at 256 calibration trials and noise SD 0.5; delays mismatch independent contexts and are measured in trials, not physiological time. In \textbf{B,C} the oracle-context and frozen-profile controls do not depend on the panel factor and are drawn as labelled reference lines (dash-dot, 80.3\%; dotted, 18.7\%) with their 95\% seed intervals as bands; the mismatched-encoder control coincides with the frozen profile in \textbf{B}, and the soft, hard and mismatched conditions coincide within one percentage point at delays 1 and 4 in \textbf{C}, where the three series are offset slightly in $x$. The dashed, square-marked curves marked with an asterisk use maximum-probability route selection by the same frozen encoder, an exploratory paired sensitivity specified after the soft outcomes with no encoder refit. Small symbols are the twenty fresh analysis seeds; curves, circles, squares and bars are means and 95\% paired-seed bootstrap intervals, and a bar narrower than its symbol is hidden by it. The cue-noise sensitivity of the same encoder is shown in main Fig.~3F. The calibration cue and activation targets are supplied resources. No serial forward dendritic computation is present.'''),
# ---------------------------------------------------------------- si_04 ----
('scalar_tree_capacity', 'si_04_interactions_boolean', [('N12', '*')], 'Interaction structure constrains scalar trees at matched input spectra and resources.', r'''Algebraic multi-affine scalar trees on binary inputs. \textbf{A}, Exhaustive families on eight independent binary inputs: 105 pairwise matchings (green circles), 35 two-quartic targets (purple triangles), and 24 nested-prefix controls (amber squares), which appear only in \textbf{E,F}. The nested controls share an anisotropic input spectrum, $(1/4,1/4,1/2,1/2,3/4,3/4,1,1)$; matching and quartic targets share the input-gradient second moment $I_8/4$ (rank eight). In the nested formula, $\pi$ is a seeded input permutation. Every tree has seven multi-affine nodes, 28 coefficients and 14 edges, with a root readout. \textbf{B}, Full rank-two and centered rank-one cut-tail lower bounds versus achieved population NMSE across 1,680 fits (140 targets $\times$ 12 candidates); labels give fit counts and the dashed line marks equality; mark area scales with the fit count. The centered constraint is stronger. \textbf{C}, Excess NMSE above the best fitted candidate among twelve fixed trees. Small dots are individual targets; larger marks are family means; counts identify targets with zero excess. Bound and pilot selectors appear above the gap; fixed balanced-shape, retrospectively best-fixed and uniform-random references appear below it. The centered-cut selector has zero excess for 101 of 105 matchings and all 35 quartic targets. These are exhaustive finite-family calculations, not independent training-seed replications. \textbf{D,E}, A minimum-depth tree for the first matching (depth three) and the first seeded nested control (depth four). Open squares are input leaves; filled circles are multi-affine units; the root is read out below. \textbf{F}, Minimum possible maximum centered-cut bound over all labeled binary trees at depth limits of three, four or five edges. The bound is zero at every limit for matching and quartic targets; nested controls have bound 0.146 at depth three and zero at depths four and five. Dashed lines in \textbf{C,F} mark zero.'''),
('oracle_profile_credit','si_04_interactions_boolean',[('N13','*')],
 'Task-dependent credit requirements in bounded multi-affine trees.',
 r'''\textbf{A,B}, Final test normalized mean squared error (NMSE) on oracle-compatible trees under Adam and stochastic gradient descent (SGD), respectively. Exact path (red-brown, circle, solid line), root broadcast (amber, square, dashed), one oracle profile (grey, triangle, dash-dot), two subtree profiles (violet, diamond, dotted) and two shuffled profiles (blue, cross, long dash). The NMSE axis is logarithmic; small translucent points are the twenty individual seeds behind each mean, and dotted rules spanning only the family they apply to mark the known label-noise floors: 0.0225 for matching and nested tasks and 0.045 for quartic tasks. \textbf{C}, Paired exact-credit NMSE increase after reassigning the leaf inputs (not the shuffled-profiles rule) while preserving tree shape and parameter count, for Adam and SGD (axis labels); every mark is the exact-path red-brown, the twenty paired seed differences are drawn behind each mean, the number of positive differences out of twenty is printed above each strip, the dashed rule is zero, and the Adam quartic interval (0.507--0.517) is narrower than its marker. \textbf{D}, Adam gradient cosine on oracle-compatible trees between the exact and delivered aggregate clean-population updates, evaluated at each rule's own trained state after 0, 1, 16, 64, 256 and 1,024 updates (equally spaced ordinal axis), each seed averaging the three task families before aggregation; the exact-path cosine is 1 by definition and is the dashed red-brown reference, not a series; the fan behind each mean is the twenty per-seed values. Points and whiskers in \textbf{A--F} are means and pointwise 95\% intervals from 10,000 whole-seed bootstrap draws, twenty fresh seeds per family. \textbf{E,F}, Final NMSE at the three tested learning rates 0.003, 0.01 and 0.03 (categorical positions), averaged over both compatible and reassigned input assignments and all three families, on one common logarithmic NMSE axis; each point carries its twenty per-seed means and whisker, and thin connectors in each rule's dash pattern join the three rates. Filled markers, the development-selected rate of each rule and optimizer (used in \textbf{A--D}); open markers, the other tested rates. The Adam strips of \textbf{C} give the leaf-assignment contrast reported in the main text (``matching'' is ``Pairwise'' there).'''),
('boolean_capacity','si_04_interactions_boolean',[('N14','*')],
 'Boolean interactions distinguish structure, depth and canonical credit.',
 r'''Algebraic multi-affine representations of Boolean functions. \textbf{A}, Truth tables of seven four-input templates on the sixteen equally weighted patterns (white, 0; slate, 1); columns are $abcd$ in numeric order, $a$ the most significant bit, hairlines between groups of four sharing $a$ and $b$. Mixed targets: \((a\land b)\lor(c\land d)\), \((a\land b)\mathbin{\mathrm{XOR}}(c\land d)\), \((a\mathbin{\mathrm{XOR}}b)\land(c\mathbin{\mathrm{XOR}}d)\) and nested \(a\land[b\lor(c\land d)]\). \textbf{B}, Centered-cut normalized-mean-squared-error (NMSE) regression lower bounds for all fifteen labeled binary trees (twelve coefficients, six edges); the rule separates the balanced (depth-two) trees T10--T12 from the twelve comb trees. The colour ramp uses exponent 0.6, with zero palest; the nested row is positive on every tree but T15 (0.105--0.199); row maxima are printed at right. Open rings mark the 49 exactly representable family--tree pairs, certified with coefficients in $[-2,2]$ (largest $4/\sqrt{15}$; T indices and cut sets in Source Data). \textbf{C}, OR-of-AND and nested each use two AND gates and one OR gate (inputs above, output below) but need exact depths two and three. \textbf{D}, Minimum exact depth (ordinal axis, 2 or 3); the columns count exactly compatible trees among all fifteen (the ring counts of \textbf{B}) and among the three balanced: 15/15 and 3/3 for AND, OR and parity (associative-gate structure controls), 1/15 and 1/3 per mixed-gate target, 1/15 and 0/3 for nested. \textbf{E}, Post hoc target-projection energy $\|\mathbb E[z\mid x_S]\|^2$ of all seven families on the six two-input subsets, normalized target $z$: dot area is proportional to the energy, exact zeros are open rings, and the rule separates aligned pairs ($ab$, $cd$) from crossed. XOR-of-AND has $1/5$ on every pair; parity has zero on every proper subset (all fourteen per family in Source Data). \textbf{F}, Canonical derivatives \(\partial F/\partial u\) for left-branch output \(u\) against right-branch output \(v\): AND, \(v\); OR, \(1-v\); XOR, \(1-2v\); dashed rule, zero. Green, amber and violet in \textbf{C,F} name the gates AND, OR and XOR, not bound values. Panels are exhaustive analytic diagnostics (no sampling error bars): positive bounds concern truth-value regression, not threshold classification; \textbf{E} is local target information, predicting neither grouping compatibility nor broadcast-learning failure; \textbf{F}'s raw derivatives need not survive training and imply no learning outcome; assumptions: Supplementary Note~\ref{note:boolean_morphology}.'''),
('boolean_learning','si_04_interactions_boolean',[('N15','*')],
 'Boolean grouping benefit and a small primary credit effect under frozen recipes.',
 r'''Multi-affine scalar-tree learners on Boolean targets. \textbf{A--D}, Clean population normalized mean squared error (NMSE), twenty-seed means, for seven target families (rows) and four trees (three balanced groupings and comb a|(b|cd)) under Adam (\textbf{A,B}) and stochastic gradient descent (SGD; \textbf{C,D}) with exact (\textbf{A,C}) or broadcast (\textbf{B,D}) credit, on a shared logarithmic colour scale. Rates (0.003, 0.01, 0.03 and 0.01 in \textbf{A--D}) were selected per optimizer and rule on five development seeds; per-cell 95\% intervals are in Source Data. \textbf{E}, Prespecified Adam XOR-of-AND contrasts: grouping, the crossed-tree mean (ac|bd, ad|bc) minus the aligned ab|cd under exact credit; credit, broadcast minus exact on ab|cd. Green dots, twenty paired seed differences; rules and whiskers, means and Bonferroni-adjusted 97.5\% intervals (grouping 0.549--0.583); amber dotted, the 0.01-NMSE mean-effect margin. The grouping axis is broken, zero and the margin in its foot strip; the credit subpanel has its own y range and a solid zero rule. Grouping passes both criteria, credit does not, its mean lying below the margin; both rules classify all sixteen patterns perfectly in every seed. \textbf{F}, Accuracy (gray circles) and balanced accuracy (blue crosses) versus NMSE for the 112 conditions at raw-output threshold 0.5; sixty coincide at 1.00 on both. Amber stars, the AND/OR constant-majority references (NMSE one, accuracy $15/16$, balanced accuracy $1/2$). \textbf{G}, Same-rate broadcast-minus-exact contrasts on the aligned XOR-of-AND tree, Adam (left) and SGD (right): dots, twenty paired seed differences per common rate (logarithmic axis); rules and whiskers, joined across rates, means with descriptive 95\% intervals; amber dotted, the 0.01 margin; dark solid, zero. The SGD symmetric-log axis is linear within $\pm 0.002$; all seed differences are shown. \textbf{H}, Broadcast gradient cosine on the aligned balanced tree for XOR-of-AND (green) and parity (violet); solid lines, filled markers Adam; dashed lines, open markers SGD; on all sixteen clean patterns at each learner's own state before clipping or optimizer transformation; checkpoint zero (shared initialization) precedes the logarithmic update axis; amber dotted, cosine one. Bands and whiskers, pointwise 95\% intervals. The large parity credit effect is descriptive.'''),
('fixed_profile_budget', 'si_04_interactions_boolean', [('N16', '*')], 'Longer matched budgets retain the task-dependent fixed-profile deficit.', r'''Seven-unit multi-affine scalar-tree learners. \textbf{A,B}, Pairwise-target learning under Adam at development-selected and common rates. Selected rates are 0.003 for exact path and unit broadcast and 0.01 for initial profile and initial sign; the common rate is 0.003. Colors identify exact path (red-brown), unit broadcast (amber), initial profile (blue) and initial sign (black dotted). Curves show means at fixed checkpoints, bands descriptive 95\% whole-seed bootstrap intervals and faint lines twenty seed trajectories. Initial-profile and initial-sign curves overlap when their rates match. Grey references mark 1,024 updates and the label-noise NMSE floor (0.0225 pairwise, 0.045 quartic). \textbf{C}, Quartic learning at selected (thick) and common (thin) rates; exact-path and unit-broadcast rates coincide across policies and are drawn once. \textbf{D}, Pairwise NMSE at 16,384 updates: dots, twenty seeds; short lines and whiskers, means and 95\% seed-bootstrap intervals. \textbf{E}, Quartic-minus-pairwise difference in initial-profile-minus-exact NMSE across four budgets. Circles/solid lines use terminal states; squares/dashed lines use validation-selected states. Violet denotes selected rates and green the common rate. Whiskers are 95\% paired seed-bootstrap intervals; every difference is positive in all twenty seeds. \textbf{F}, Quartic errors at 1,024 versus 16,384 updates for all twenty exact-path and twenty initial-profile seeds. The diagonal marks unchanged error; grey guides mark the noise floor. The right view (linear axes, magnified about $23\times$; open circles, the sixteen exact seeds within 0.043--0.050) resolves fits near that floor. Exact seeds with high early error (three of twenty, above 0.5 NMSE at 1,024 updates) reach the floor by the longer budget. These continuations reuse the original seeds and do not add an independent cohort.'''),
('credit_optimizer_controls','si_04_interactions_boolean',[('N17','*')],
 'Nested targets and SGD controls in the balanced budget extension.',
 r'''Seven-unit multi-affine scalar-tree learners. \textbf{A}, Nested-target control from the main text, under Adam at selected rates: 0.003 for exact and unit-broadcast credit, 0.01 for initial-profile and initial-sign credit. \textbf{B}, Pairwise targets under SGD at common rate 0.03. \textbf{C,D}, Quartic targets under SGD at selected rates (\textbf{C}: 0.03 for exact and initial-profile credit; 0.01 for unit and initial-sign broadcast) and common rate 0.03 (\textbf{D}). Exact and initial-profile runs are identical across these views and appear only in \textbf{C}. Dark red solid, exact path; amber dashed, unit broadcast; blue solid, initial-profile (calibrated) broadcast; black dotted, initial-sign broadcast. Thick curves and bands are means and descriptive 95\% whole-seed bootstrap intervals (10,000 draws); thin curves show all twenty seeds per rule. Vertical dashed lines mark the original 1,024-update budget; grey horizontal lines mark noise-only NMSE (0.0225 for nested/pairwise, 0.045 for quartic). Both axes are logarithmic, with different vertical limits retaining every divergent seed. In \textbf{B}, two unit-broadcast seeds diverge (final NMSE above 1) while eighteen finish near the noise floor, raising the mean; five initial-profile seeds also retain larger errors (final NMSE above 0.05). In \textbf{D}, eighteen unit-broadcast and fifteen initial-sign seeds exceed NMSE one at 16,384 updates. By contrast, the three broadcast curves nearly coincide in \textbf{A} (means within 6\% of one another after 1,024 updates) and remain near one in \textbf{C} (means 0.94--1.09; no seed exceeds NMSE 2); line styles distinguish overlapping curves. \textbf{A,B} are different tasks and optimizers, not a paired comparison. Only \textbf{C,D} compare rate policies at fixed task and optimizer. These controls separate extended training from rate-dependent instability.'''),
# ---------------------------------------------------------------- si_05 ----
('conductance_grouping','si_05_conductance_learning',[('N18','*')],
 'Input grouping and restricted feedback in positive-conductance trees.',
 r'''Standalone NumPy directed conductance model. \textbf{A}, Seven-compartment directed shunting model (one physical shape; sixteen positive conductances: four excitatory and six inhibitory contacts, six couplings; soma-only readout) under the groupings $01|23$, $02|13$ and $03|12$. The first tree marks excitatory/inhibitory contacts (blue/carmine), two proximal subtrees (greys) and the soma (yellow) with error \(\delta_0\). Each leaf has one contact of each type; each parent has one inhibitory contact. Teacher-task permutations preserve the input-gradient spectrum. \textbf{B}, Analytic population lower bound from interactions crossing the student's two proximal input blocks (converged quadrature, not a certified bound): small pale dots, the 120 incompatible task/grouping pairs (six per seed, two values each); larger dots, the twenty seed means; mean bound 0.013222 population normalized mean squared error (NMSE); compatible groupings are exactly zero in all twenty seeds (superimposed). \textbf{C,D}, Held-out sample NMSE at five checkpoints (ordinal update axis, straight segments) under Adam and stochastic gradient descent (SGD) for exact path (dark red circles), fixed calibrated broadcast (blue squares), one oracle profile (grey triangles, dashed) and two subtree profiles (purple diamonds, dashed): coloured curves, compatible means over three task permutations within seed, with 95\% whiskers offset per checkpoint; grey band, the union of the four rules' 95\% intervals for the incompatible groupings (two per task; final means 0.029--0.032 NMSE under both optimizers); all rules share the initialization error. \textbf{E}, Paired compatible-tree NMSE differences from exact path at 1,000 updates for the three other rules under Adam and SGD: dots, the twenty independent seed means; dashed rule, zero; the Adam fixed-broadcast interval includes zero; the small SGD effect implies no universal need for precise credit. \textbf{F}, Aggregate gradient cosine on 256 calibration examples, compatible Adam models, same checkpoints (common exact-trained states): dots, the twenty seed means; symbols and whiskers, their mean and 95\% interval (individual records reach 0.79); exact path is 1 by construction (dashed reference); fixed broadcast and one oracle profile differ by at most 0.0064. Means and whiskers in \textbf{B--F}: 10,000 whole-seed bootstrap draws, pointwise 95\% intervals.'''),
('conductance_precision','si_05_conductance_learning',[('N19','*')],
 'A first conductance task produces a small precision benefit from exact credit.',
 r'''Standalone NumPy seven-compartment directed conductance model. \textbf{A,B}, Test NMSE under Adam for ungated independent inputs and gated conflicting inputs in the original 16-conductance model. Exact credit is dark red (solid circles), unit broadcast amber (dashed squares), and fixed initial-profile credit blue (dotted triangles). Means and 95\% paired-seed bootstrap intervals summarize twenty fresh seeds; thin lines show every trajectory. The shared vertical axis plots $\log_{10}$ NMSE; update zero appears before the horizontal-axis break. The dotted vertical line marks the original 4,096-update budget. Fits continue to 16,384 updates with unchanged optimizer and minibatch state. All rules share each seed's initialization. The small precision advantage of exact credit occurs on the gated task; ungated endpoints are similar. Global gradient clipping can make the unit and initial-profile rules differ despite Adam's approximate invariance to fixed positive coordinate scaling. \textbf{C}, Gated-task NMSE differences, initial profile minus exact credit, at validation-selected checkpoints within the extended window. Adam and separately tuned SGD use different vertical scales, both in thousandths of NMSE. Dots are twenty paired differences; triangles and whiskers show means and 95\% bootstrap intervals. Every difference is positive, but the mean Adam gap remains below $6\times10^{-4}$ NMSE. The dashed line marks zero. \textbf{D}, At exact-rule validation-selected states, best rank-one capture of the six-compartment path field (dark red) and eligibility-weighted capture by the student's fixed initial profile (blue). Both are squared-energy projection ratios with one fitted amplitude per example. Dots are twenty seeds; symbols and bars show means and 95\% bootstrap intervals, mostly within the symbols (three of the four). Ungated path capture is one by construction. High eligibility-weighted capture helps explain the small learning difference. The two tasks change both inhibition and distractor dependence, so they do not isolate a gating intervention. Source Data retains all development and fresh fits, continuations and endpoint estimates.'''),
('conductance_optimization','si_05_conductance_learning',[('N20','*')],
 'Parameter-range, development and rate controls retain the opposed-task broadcast deficit.',
 r'''Standalone NumPy seven-compartment directed conductance model. \textbf{A}, Task-by-credit interaction (opposed minus aligned difference in rule-minus-exact-path test NMSE) for the calibrated broadcast, unit broadcast and three-profile oracle rules, each under original $[-7,7]$ and wider $[-20,20]$ log-conductance bounds with Adam and SGD. Small dots are the twenty paired seeds of each row, the large symbol their mean and the bar its 95\% paired bootstrap interval; the dashed rule is zero, and each row prints its positive-seed count out of twenty. All 160 broadcast seed values are positive; the oracle means and intervals lie within $\pm5\times10^{-6}$ of zero (too narrow to draw), with 10/20 (Adam) and 17/20 (SGD) seeds positive; unit-broadcast means are within 13\% of the calibrated-broadcast means. \textbf{B}, The calibrated-broadcast gap at each seed's last saved checkpoint before its fit first reaches an original bound, against the update of that contact (Adam; logarithmic axis; one dot per seed, $n=20$, no interval): all twenty gaps are positive, the dashed rule is zero, and the seeds with first contact at 1,039 and 1,041 updates nearly coincide. \textbf{C}, All four development regimes (aligned strong; opposed strong, moderate and ungated) under Adam at the original 4,096-update budget, each rule at its best of three rates (0.03 for all); small dots are the three development seeds, symbols their means; exact and oracle means agree within 13\%, and the two broadcast means coincide except when ungated. \textbf{D,E}, Six-rate opposed-tuning sweeps under Adam and SGD at a 16,384-update budget; the dotted vertical line marks the originally selected rate (Adam 0.03, SGD 0.3). Small dots are the three development seeds behind every point and lines join the three-seed means; seed ranges (up to 25-fold) stay far below the three-decade rule separation at rates up to 1. Unit and calibrated broadcast coincide within 0.03\% under Adam and 7.4\% under SGD, so the open amber triangle sits on the filled blue square wherever they agree. One key serves \textbf{A} and \textbf{C--E}: dark red filled circles, solid lines, exact path; blue filled squares, solid lines, calibrated broadcast; amber open triangles, dashed lines, unit broadcast; purple open diamonds, dashed lines, three-profile oracle; in \textbf{A} the hue names the rule whose deficit is drawn, and the dots of \textbf{B} are the calibrated-broadcast blue.'''),
('local_gate_controls','si_05_conductance_learning',[('N21','*')],
 'Local conductance gating is robust to rate choice but depends on where it is applied.',
 r'''The standalone NumPy model of main Fig.~5. \textbf{A,B}, Aligned tuning within 4,096 and 16,384 updates. \textbf{C,D}, Opposed tuning within the same budgets. Each rule has three strips, from top to bottom: Adam rates 0.01 (upward triangle), 0.03 (diamond; primary) and 0.1 (downward triangle). Small dots are test NMSE at each seed's validation-selected checkpoint; larger markers and whiskers are arithmetic means and 95\% percentile intervals from 20,000 whole-seed bootstrap draws ($n=20$ fresh paired seeds). All panels share a logarithmic NMSE axis. Thirteen of the 96 intervals lie within their markers and are not drawn. Colors follow main Fig.~5: dark red, exact; purple, oracle projections; green, hard and continuous distal gates; amber, broadcast; rose, gating also at proximal compartments; grey, wrong branch. The two-profile oracle supplies two terminal patterns with unit proximal credit. The continuous distal gate replaces binary selection with attenuation based on parent inhibitory conductance. A horizontal separator distinguishes rules accurate on both tasks from those failing on at least one. The unit-broadcast row also represents the calibrated rule's mean to two significant figures in every task, budget and rate condition; their individual seeds can differ by up to 3.1-fold. In \textbf{B}, one or two outlier seeds raise the gate-also-proximal means 17--25-fold above their medians, although most seeds match the accurate rules. Its position below the separator reflects its opposed-task failure. Outliers similarly raise the two-profile-oracle mean at rate 0.03 and continuous-gate mean at 0.1 in \textbf{D}. Showing all seeds distinguishes these outlier effects from consistent learning deficits.'''),
# ---------------------------------------------------------------- si_06 ----
('physical_architecture','si_06_physical_depth',[('N22','*')],
 'Task-matched serial computation differs from grouped and flexible point controls.',
 r'''DendriNet serial and grouped-point models, with a separate point MLP reference. Small dots show ten seeds per condition; large symbols and bars are means and 95\% seed-bootstrap intervals, paired for differences. \textbf{A}, Serial D3 tree ($[2,1,2]$, eight modules in three stages), resource-identical grouped point model (one stage), and flexible point multilayer perceptron (MLP). Blue dots, excitatory class-bearing contacts; open rings, modules; yellow disc, soma; arrow, somatic error $\delta_0$. \textbf{B}, Four-tier task under aligned sensors or reversed tier placement. Cells show mean accuracy (\%); outlines identify aligned-task depths whose lower confidence limit exceeds the runner-up's upper limit. Reversing placement removes the shunting depth advantage. The additive rows use a paired same-seed follow-up; neither shows a comparable depth benefit. \textbf{C}, Serial-minus-grouped-point accuracy at D3 under exact-path LocalCA for nested-factor (dark teal), flat-factor (teal) and local-ratio (light teal) tasks. Main Fig.~7C gives the distinct exact-BP comparison. \textbf{D}, Serial-minus-star accuracy at D1--D3 under BP. The star is the resource-identical all-active grouped control; the open symbol shows the aligned-minus-reversed D3 difference. The broken axis omits an empty interval. \textbf{E}, At the original 180-epoch budget, flexible point networks matched to active or total parameter count (grey), compared with serial D3 exact BP (black). The stopping extension is shown in main Fig.~7E,F. \textbf{F}, Serial-BP accuracy against task--sensor alignment $\alpha$ for D1 (dotted circles), D2 (dashed squares) and D3 (solid triangles); intervals are mostly narrower than their symbols. \textbf{G}, Paired D3-minus-D1 advantage at each alignment level, with an axis break. Between $\alpha=0.25$ and 0.75 the advantage grows by 2.92 points (95\% interval 2.65--3.17), a secant of 5.84 points per unit $\alpha$. Within-seed linear fits over all five levels give 25.8 (25.4--26.3), dominated by the final jump. Main Fig.~7D shows the original three-tier comparison. The two-tier control is Supplementary Fig.~S26B.'''),
('physical_calibration','si_06_physical_depth',[('S15','ABCD')],
 'Transparent calibration of the nonlinear physical-depth operating point.',
 r'''DendriNet steady-state conductance networks. All panels use the three-level hierarchical gain-load task and serial shunting morphologies D1 \([8]\), D2 \([2,3]\) and D3 \([2,1,2]\), each with eight nonsomatic compartments per soma. They test the prerequisite for the main hypothesis: whether the mechanism-matched divisive task is accessible at every depth before comparing depth benefits. \textbf{A}, The original aligned shunting-backpropagation (BP) pilot at signal contrast 0.24 fit the training distribution increasingly with depth but remained at chance under the prespecified severe gain shift (test gain standard deviation (SD) 1.2). Bars grow from chance (0.5) and show two-seed means; colour encodes depth as in \textbf{B--D}, filled bars are the training split, open bars the severe-shift test split, and open circles the individual exploratory seeds. \textbf{B}, With training gain SD fixed at 0.25, increasing unseen test gain variation eroded accessibility at every depth. The dotted line marks the matched train/test condition. \textbf{C}, At matched gain and signal contrast 0.24, stronger child coupling selectively made serial D3 computation accessible; the dotted line marks child conductance 16 used subsequently. \textbf{D}, At matched gain and child conductance 16, increasing excitatory signal contrast moved D1--D3 away from chance without saturating D1 or D2. The shaded window marks the accessibility criterion frozen before the ladder was run: every depth's two-seed mean test accuracy in \([0.60, 0.95]\), each seed above 0.57 and D3 minus D1 at least 0.02 in both seeds. Contrast 0.72 failed it by 0.003 (D1 mean 0.597); the separately run 0.80 boundary (dotted line; D1 mean 0.609) met it and selected the operating point used for the analysis seeds. Lines and filled symbols show two-seed means and small open circles the two exploratory seeds, dodged slightly in \(x\); no confidence interval is drawn. Panels \textbf{A--C} share an expanded ordinate (0.48--0.70); \textbf{D} uses the full range. All pilot, ladder and boundary seeds were excluded from the confirmatory contrasts.'''),
('physical_optimizer','si_06_physical_depth',[('S18','EF'),('S28','B')],
 'Physical-depth conclusions depend on sensor fidelity, task hierarchy and optimization.',
 r'''DendriNet conductance networks and grouped-point controls. \textbf{A}, Serial-minus-grouped-point accuracy at D3 under matched (aligned) and reversed sensor placement, the derived aligned-minus-reversed difference (open marker) and the star-minus-point architecture control, in percentage points. \textbf{B}, Separate independently seeded two-level task with D1 $[8]$ and D2 $[4,1]$: the depth-by-placement interaction (aligned D2-minus-D1 gain minus the reversed gain) for serial and grouped-point backpropagation and for shared or exact-path LocalCA; arm colours follow main Fig.~7D. \textbf{A,B} share one broken axis with 5 pp ticks (the empty 1.6--18 pp span is removed) and draw the ten paired seeds behind every mean. \textbf{C}, Original-budget D3 optimizer/credit-coordinate contrasts in the serial conductance morphology $[2,1,2]$, in drawn order: BP minus soma broadcast under the BP optimizer, 0.64 points (95\% interval 0.15--1.17); soma broadcast, BP minus LocalCA optimizer, 13.47 (12.89--14.05); soma-broadcast autograd minus shared-soma LocalCA, $-0.06$ ($-0.23$ to 0.09); exact-path minus shared-soma LocalCA, 10.96 (10.34--11.58); BP minus exact-path LocalCA, 3.08 (2.48--3.79). Means and intervals use ten paired seeds and 95\% bootstrap intervals; panel~\textbf{C} is generated from \texttt{source\_data/point\_dendrite\_credit\_controls/} and is registered in the Source Data manifest. These original-budget effects depend on training duration: the same-seed stopping extension shows a temporary LocalCA accuracy reversal, followed by a small exact-path advantage and lower cross-entropy at validation-based early stopping (main Fig.~7E--H). All sixty extended fits reached the unchanged validation-stopping criterion.'''),
# ---------------------------------------------------------------- si_07 ----
('anatomy_capacity_controls','si_07_anatomy',[('S10','B'),('S20','B'),('S27','BD')],
 'Cohort, label and arbor-size controls bound modeled anatomical capacity.',
 r'''Constructed route fields and geometric controls on reconstructed arbors. \textbf{A}, Route-generated capture of constructed target fields versus budget in 47 disjoint same-mouse cells (46 at 16 channels); morphology routes have 8.1\% nonzero coefficient density at eight channels (density per cell and arm is tabulated in Source Data). The key below the sheet applies to \textbf{A} and \textbf{C}. Twenty modeled streams are averaged within each cell; bands are 95\% cell-bootstrap intervals of the mean, several times narrower than the cell-to-cell spread, and the two null arms (random paths, shuffled ancestry) nearly coincide. \textbf{B}, Initial eight-cell cohort: squared-energy capture at $K=8$ restricted to directly typed presynaptic contacts, every cell drawn, with the dense PCA oracle as the ceiling; diamonds and whiskers show means and 95\% cell-bootstrap intervals. \textbf{C,D}, Second-mouse Pinky cohort: relative residual norm versus route budget against the dense principal-component ceiling, every qualifying cell drawn faintly behind its arm mean, and the number of inhibitory-bearing segments (candidate routes) versus effective field rank for the ten of twelve selected cells that passed the typed-input criterion (numeric labels identify cells; open markers: the two excluded cells; 78--114 reconstructed segments per tree). Eligibility changes from ten cells at $K\leq4$ to eight at $K=8$. Panel \textbf{C} uses mean $\pm$ s.e.m., and its residual norm is not squared-energy capture. All fields are model generated. Main Fig.~8 equalizes a common broadcast and leads with actual-tree surrogate controls; these route-generated comparisons are structural positive controls, not measured biological teaching signals.'''),
('inhibitory_spatial_controls', 'si_07_anatomy', [('S12', 'BDEGH')], 'Spatial matching limits apparent inhibitory targeting of ancestry domains.', '\\textbf{A}, Six layer-diverse reconstructions with mapped inhibitory contacts in $x$--$y$ projection, pia up. Grey lines join compressed skeleton segments; cells are scaled independently, with 50~$\\mu$m bars. Mapped inhibitory contacts per arbor: L2a 232, L2c 180, L3a 240, L4a 119, L4c 91 and L5ET 259. \\textbf{B,C}, Observed-minus-matched tree distance and shared-path fraction in twenty targets, after aggregation within 92 presynaptic axons, and after joint path-distance/Euclidean matching (3D). Points are the displayed inferential units; diamonds and whiskers are means and 95\\% unit-bootstrap intervals. Labels give one-sided Wilcoxon signed-rank $p$ values (closer in \\textbf{B}, larger in \\textbf{C}). Two axon distances beyond $\\pm105~\\mu$m are shown at the frame with their values and retained in every statistic. Only tree distance remains significant after axon aggregation ($p=0.0008$), although its mean bootstrap interval spans zero; neither measure remains significant after joint three-dimensional matching. The weighted domain-overlap measure, strongly correlated with shared-path fraction, is retained with its definition and separate inference in Source Data. \\textbf{D}, Cumulative descendant-input fraction for distal-dendrite-targeting (DTC, 1,142) and perisomatic-targeting (PTC, 840) contacts, retaining one contact per axonal clump. Faint curves are twenty targets; bold curves pool contacts. Target-median DTC-minus-PTC differences are negative in all twenty targets. \\textbf{E}, Observed and matched descendant-input fractions for twenty targets. The right axis shows paired differences, their mean and 95\\% target-bootstrap interval; the mean is near zero and the two-sided Wilcoxon test gives $p=0.55$. Cells derive from one mouse. These structural comparisons do not identify contacts carrying teaching signals.'),
('coarse_energy','si_07_anatomy',[('S25','ABCD')],
 'Irregular reconstructed trees concentrate modeled route-field energy coarsely, but the anatomy-specific excess does not replicate.',
 r'''\textbf{A}, Weighted unbalanced tree-Haar contrasts recursively partition each reconstructed arbor into coarse, intermediate and fine supports, defined by descendant excitatory-weight fractions \(>1/4\), \(1/16\)--\(1/4\) and \(\leq1/16\), respectively; the schematic colours one example subtree per bin (key), with the grey trunk leading to them and the soma drawn as the yellow ellipse. Green, blue and rose mark scale bins throughout the figure. \textbf{B}, Non-scalar conductance-weighted ancestry-route energy fraction in the disjoint 47-cell cohort (teal, filled circles), compared with column-permuted ancestry (grey, open squares) and dimension-matched isotropic noise (grey, open triangles). Small points are individual cells; large marks are cell means, and the black whiskers drawn over them are 95\% cell-bootstrap intervals for the actual-route series and for both nulls. \textbf{C}, Signal-to-isotropic-noise power by scale for all 47 cells (points); diamonds show means, with 95\% cell-bootstrap intervals drawn over them as black whiskers (the intermediate-scale interval is only slightly wider than the diamond), and the dashed line marks equality with isotropic noise. \textbf{D}, Actual-minus-permuted coarse energy per cell (points) in the original 8-cell cohort (light teal) and the disjoint 47-cell cohort (teal); diamonds show cohort means with 95\% cell-bootstrap intervals as black whiskers, and the dashed line marks zero. The positive original-cohort contrast is not reliable in the 47-cell cohort. Cells are the inferential units; all reconstructions derive from one mouse. Modeled route fields are not measured teaching signals.'''),
('anatomy_preprocessing','si_07_anatomy',[('S33','*')],
 'Label coverage and geometric preprocessing affect anatomical route dictionaries.',
 r'''All analyses use eight initial reconstructed cells from one mouse. \textbf{A}, Direct-presynaptic versus target-proxy excitatory/inhibitory (E/I) labels for 2,012 jointly labeled mapped contacts; cells print counts and row percentages and are shaded by row percentage (colour bar). \textbf{B}, Direct-label fraction by compartment: individual cells and cell means with 95\% cell-bootstrap intervals. Denominators differ (22--66 contacts per cell at the soma, 342 pooled; 157--1,541 internal, 5,568 pooled; 1,148--5,697 terminal, 25,580 pooled), but summary estimates and bootstrap intervals weight cells equally. Direct-label coverage falls from the soma (cell mean 20.8\%) to internal (12.4\%) and terminal (7.0\%) compartments. \textbf{C}, Jaccard overlap (intersection divided by union) of the four selected routes with the nominal hybrid-label, $5\,\mu$m dictionary across mapping thresholds (ordinal, equally spaced) and label choices; the open symbol at hybrid, $5\,\mu$m is the reference compared with itself (Jaccard $=1$). With hybrid labels, 5 of 8 cells change at least one route at $2\,\mu$m (mean 0.72) and 3 of 8 at $10\,\mu$m (mean 0.82); with direct-only labels all eight differ at every threshold (means 0.21--0.24). \textbf{D}, Capture of the fixed nominal field under heterogeneous log-radius perturbations indexed by their standard deviation (SD), for the hybrid and direct-only dictionaries as in \textbf{C}; thin lines join the eight cells across SD. The paired within-cell change from SD 0 to 0.5 is $-0.034$ with hybrid labels ($-0.050$ to $-0.018$; 7 of 8 cells decline) and $-0.011$ with direct-only labels ($-0.053$ to $0.019$). Large symbols and bars in \textbf{C,D} show cell means and 95\% cell-bootstrap intervals; small symbols are individual cells, tied values spread side by side. \textbf{E}, Distribution of the mean-radius compressed axial resistance divided by the exact series sum over raw edges (0.1-log-unit bins). No segment exceeds its series sum; 72 of 608 segments (8 cells) fall below 0.89 (0.05 log units), the worst to 0.003 (2.5 log units). \textbf{F}, Series-resistance minus mean-radius capture of the nominal field against the mean of the two, at the nominal condition (hybrid labels, $5\,\mu$m, SD 0), one point per cell; the dashed line marks equality. Six of eight cells are identical and two differ by $+0.049$ and $+0.017$.'''),
# ---------------------------------------------------------------- si_08 ----
('shunt_sensitivity','si_08_shunting',[('N29','*')],
 'Dose, cable parameters and input mapping determine focal-shunt selectivity.',
 r'''Reciprocal passive steady-state cable calculations on reconstructed arbors. \textbf{A}, Shunt-minus-current-injection localization across fixed absolute doses at three membrane resistances ($R_m$ of 300, 1,000 and 15,000~$\Omega\,\mathrm{cm}^2$: one lightness step, marker and dash pattern each) and three background-conductance multipliers (column groups), on a symmetric-log axis that is linear within $\pm0.001$. The contrast is close to proportional to dose at every setting, and the $R_m$ ordering reverses with background: with no background the $R_m=15{,}000$ contrast stays near zero (slightly negative at 0.05 and 0.5~nS, an interval spanning zero at 5~nS), whereas at background multiplier four it is the largest of the three at every dose, with 8/8 cells positive. \textbf{B}, Shunt and matched-injection localization (grey lines pair cells) and the true relation against a reassigned foreign template (unpaired); positive favors the shunt or the true relation. \textbf{C}, Transport selectivity $S_k$ against full synaptic-gradient localization at unit input-conductance-normalized dose, 101 sites; descriptive site--regime points, $R_m$ colours as in \textbf{A}, background multipliers 0, 1 and 4 as filled, open and plus markers; dotted line, $S_k=1$. \textbf{D}, Signed census of the same sites at $R_m=1{,}000\,\Omega\,\mathrm{cm}^2$, background multiplier one, unit dose: fraction of descendant gradients attenuated, enhanced or sign reversed by shunt (attenuates only) and matched injection (enhances only), as in all 81 passive conditions. \textbf{E}, All mapped versus directly typed contacts (grey lines pair cells); the contrast increases in 8/8 cells. \textbf{F}, Joint excitatory and inhibitory conductance-scale and inhibitory-reversal sensitivity by row; ``E/I'' denotes two equal scale factors, not their ratio; open marker and band, the reference condition (E/I 0.35, reversal $-0.2$) and its interval; right column, cells with a positive contrast. \textbf{A,B,E,F}, means and 95\% cell-bootstrap intervals over eight cells, every cell drawn: in \textbf{A} the eight per-cell contrasts are the small dots behind each mean, and the three $R_m$ series are offset within each dose so their fans and intervals stay apart.'''),
('shunt_replication','si_08_shunting',[('N30','*')],
 'Focal-shunt controls replicate structurally and under weak-channel linearization.',
 r'''Reciprocal cable calculations in \textbf{A,B,D,E}; reconstructed contact coverage in \textbf{C}. \textbf{A}, Focal shunting (green) versus baseline-current-matched injection (blue) in 45 disjoint cells from the same mouse, totaling 235 sites. Small dots are cell means; lines pair each cell. Open diamonds and bars are means and 95\% cell-bootstrap intervals (20,000 draws). The injection mean lies near the zero rule; its interval is narrower than the diamond. \textbf{B}, True (green) versus reassigned (rose) descendant relations in the forty cells with valid paired templates (230 sites). The true-relation values are the corresponding subset of \textbf{A}. Counts in \textbf{A,B} give cells with positive paired contrasts; the tie in \textbf{B} is a contrast within $10^{-12}$ of zero. \textbf{C}, Direct excitation/inhibition label coverage and selected focal-site counts, capped at sixteen per cell. Markers distinguish L2IT (34 cells), L3IT (2), L4IT (10) and L5ET (1); the L5ET cell had 47 eligible sites. \textbf{D,E}, Separate eight-cell weak-channel ensemble, linearized at each solved steady state. Sixty-four channel draws and sites are averaged within each cell. \textbf{D}, Localization versus dose relative to local input conductance: focal shunt (green circles) and matched injection (blue squares). Dots, means and intervals are as in \textbf{A}; intervals are often within symbols. Localization measures absolute gradient changes. Signed effects show descendant attenuation by shunting and enhancement by injection in every cell at every dose. Inset: active-minus-passive localization per cell (dots; open symbols, means) at matched calibration ($R_m=1{,}000\,\Omega\,\mathrm{cm}^2$, background conductance equal to leak). Mean changes are at most 0.010 for shunting and 0.023 for injection. \textbf{E}, Paired shunt-minus-injection contrasts, positive in all eight cells at each dose (two-sided Wilcoxon $P=0.0078$). Dashed lines mark zero. This weak-channel check tests local linearization near passive responses, not regenerative dynamics or channel kinetics.'''),
# ---------------------------------------------------------------- si_09 ----
('measured_transfer_geometry','si_09_measured',[('N31','*')],
 'The measured-response learning comparison is limited by mapped coverage and transfer geometry.',
 r'''Recorded-response and anatomy summaries (\textbf{A,B}), fitted reciprocal-cable update geometry (\textbf{C}), and Gaussian response-simulation calibration (\textbf{D}). \textbf{A}, Seven targets from one mouse, one selected scan per target: mapped functional partners (filled circles), manually curated subset (open circles), and partner split-half repeat reliability (Spearman $r$; 69 dots), with target medians. \textbf{B}, Input coordinates per selected route across all thirteen scans, ordered by mapped-input count. The ring identifies the representative scan of main Fig.~10E. Six scans have one coordinate per route; the mean is 1.31. \textbf{C}, Common-checkpoint update reconstruction, $\mathcal M_D$ (Eq.~\ref{eq:micronsupdatematch}; 1 = exact), for the unrestricted fixed transfer profile (blue), ancestry-restricted fixed profile (green circles), and trialwise oracle amplitudes on ancestry (green), random-anatomical and site-shuffled routes (grey diamonds). Small dots are seven target means; larger symbols and bars show their mean and 95\% target-bootstrap interval (20,000 draws). Solid bars denote fixed profiles; dashed bars denote oracle amplitudes. The unrestricted fixed profile nearly reconstructs the update (95\% interval 0.964--0.986, narrower than its symbol); restricting its support to selected ancestry routes removes useful coverage. Response-prediction comparisons are in Supplementary Fig.~\ref{fig:si_measured_predictor_controls}A and Table~\ref{tab:functional_full_tree}. \textbf{D}, Reliability calibration: mean simulated split-half Spearman reliability minus the nonnegative target $r_+=\max(r_{\rm measured},0)$, from 1,000 independent calibration datasets. Dots represent 125 partner--scan records; whiskers show $\pm1.96$ Monte Carlo standard errors. Open squares mark two negative measured reliabilities ($-0.098$ and $-0.188$) assigned zero reliable variance. The dashed line marks zero discrepancy. Five of 125 simulated means differ from the calibration target by more than 1.96 standard errors; the largest discrepancy is 0.0069. \textbf{A,B} are descriptive; targets are the inferential units in \textbf{C}, and records assess simulation calibration in \textbf{D}. No panel measures endogenous learning signals.'''),
('measured_predictor_controls','si_09_measured',[('S34','*')],
 'Linear baselines explain most measured-response prediction, and degrading observed inputs worsens performance.',
 r'''Reciprocal-cable fits and separate linear regressions predict recorded responses. Thirteen scans retain the complete-tree analysis's 130 outer stimulus-identity splits. All preprocessing, reliability masks and ridge tuning use training identities only. \textbf{A}, Ordinary least-squares (OLS) and nested ridge-regression baselines against the archived exact compartment-error fit; the dotted line is the training-mean predictor (normalized MSE $=1$ by construction). Lower normalized MSE is better. Below, paired differences from nested ridge (dotted line, zero difference) for each comparator: OLS, ridge on untransformed inputs, the exact compartment-error fit, and the archived route-restricted fits with topology-matched (ancestry), random anatomical and site-shuffled routes, whose absolute errors are retained in Table~\ref{tab:functional_full_tree}. The exact fit gives a small additional nonlinear benefit, mean $-0.0152$ (95\% interval $-0.0271$ to $-0.0042$), whereas topology-matched routes ($0.029$; $-0.013$ to $0.074$) are matched by random routes ($0.031$; $-0.009$ to $0.068$). \textbf{B}, Ridge prediction when retaining nested random fractions of observed presynaptic inputs; five masks per split are averaged before inference. Manual-only partners, right of the separator, are the same ridge model under a provenance restriction, not a random retention dose. Empty masks use the training mean and remain included. \textbf{C}, Added Gaussian predictor noise in both training and test data, in units of training input standard deviation (SD) after the quantile transform. \textbf{D}, Reliability filtering based only on training repeats; parentheses give mean retained partner counts. In \textbf{B--D} the dotted horizontal line marks the training-mean predictor (normalized MSE $=1$). Thin traces and small dots show seven target means after averaging splits within scan and scans within target. Larger symbols and bars are means and 95\% target-bootstrap intervals (20,000 draws). All targets come from one mouse. Only ridge is refitted under the sensitivity conditions. Removing observed inputs does not recover unknown partners, establish population power or explain away the absence of preferential ancestry alignment.'''),
# ---------------------------------------------------------------- si_10 ----
('original_selector','si_10_selection_statistics',[('S35','*')],
 'The original scalar initialization score predicts a first step but fails at prospective endpoint selection.',
 r'''Contextual linear learners with identity or passive tree-transfer matrices. \textbf{A}, Four balanced leaf assignments (balanced 1--4; the number under each leaf is the input index at that leaf slot) and one comb, each at route budgets $K=1,2,4,8$: twenty candidates with prescribed cost. Filled nodes mark the $K$ feedback channels, one subtree each. \textbf{B}, Protocol: development tasks fit the score-to-loss scale; independent calibration samples seal all candidate scores and choices; held-out tasks on 20 new seeds then train every candidate and assess regret. \textbf{C}, Final test half-mean-squared-error (MSE) plus cost, expressed as excess over the best trained candidate, for the feedback-only (green) and joint passive-transfer (blue) arms; the dashed line marks zero regret. Small points are the twenty whole-seed means; open diamonds and whiskers are means and 95\% seed-bootstrap intervals (10,000 draws), which are narrower than the diamond in the rank-only, fixed / max-budget and random-expectation rows. The fixed / max-budget baseline (the development choice, which is also the maximum budget) and the privileged rank-only baseline outperform the moment selector; random expectation averages all candidates uniformly. \textbf{D}, Route budget of the candidate chosen by the moment selector versus imposed rank; small points are the twenty seed-block means, open circles their mean, and the gray dashed line the retrospectively best candidate's budget, which equals the rank in every task of both arms. Mean selected budgets differ from it at every rank in both arms. \textbf{E}, Mean within-task Spearman correlation of initialization utility with the actual first-step test-loss decrease and with the final test-loss ranking; small points are the twenty seed-block means, open circles and whiskers their mean and 95\% interval resampling whole seeds ($n=20$ seed blocks). Agreement is stronger for the first step (means 0.98 and 0.97) than for the final ranking (0.62 in both arms). Colours in \textbf{D,E} as in \textbf{C}. All 12,800 confirmation candidate fits and 38,400 checkpoints remain included. Every model has a freely trained dense weight matrix; passive transfer changes conditioning and cost but does not necessarily restrict representation. This experiment is distinct from the locally connected multi-affine construction and the separately frozen finite-horizon follow-up.'''),
('finite_horizon','si_10_selection_statistics',[('N35','*')],
 'Finite-horizon prediction improves selection but strong simple baselines remain.',
 r'''Contextual linear learners with identity or passive tree-transfer matrices. \textbf{A,B}, Final test loss + cost regret ($\times 1{,}000$) pooled over generating ranks in the feedback-only and joint-transfer fixed-cache learners; regret is excess above the best trained candidate, and the axis is linear from zero to 0.01 and logarithmic above, with the dashed rule at zero regret. Small points in \textbf{A--C} are the twenty whole-seed means, markers and whiskers their mean and 95\% interval. The top three grey rows are the baselines of Supplementary Fig.~\ref{fig:si_original_selector}C,D (moment selector there, original scalar here) on independent fresh seeds; in \textbf{A} the Gaussian full-batch and context-count rows coincide because they chose the same candidate in all 320 tasks. \textbf{C}, The four strong selectors at each generating rank in the joint arm, on the regret scale of \textbf{A,B}: Gaussian SGD (dark red circles, the primary method), Gaussian full-batch (rose squares), observed context count (violet diamonds) and the privileged population oracle (hollow circles, dashed). \textbf{D}, Median elapsed time on one central processing unit per twenty-candidate decision, rows sorted by cost on a logarithmic time axis; `empirical full-batch' is the split-calibration forecast, and the dashed rule with its grey band is the median and interquartile range of `all 256-update fits', the cost of training every candidate. Timings include calibration-model preparation but not shared calibration generation. \textbf{E}, Original scalar forecast versus final test half-mean-squared-error in the feedback-only arm. \textbf{F}, The calibration-derived Gaussian SGD forecast on the same cases and arm. Hexagons (one $21\times10$ grid, identical limits and scale in both panels) bin the 6,400 fits per panel on the logarithmic colour bar beside \textbf{F}, shared by \textbf{E,F}; the dashed grey line marks equality and each panel prints its root-mean-square forecast error. Source Data retain the joint arm, which reproduces \textbf{E,F}, and the population-oracle forecast bias (exact only under fresh examples). Markers and whiskers in \textbf{A--C} are means and pointwise 95\% intervals from 10,000 whole-seed bootstrap draws over twenty seed blocks, hidden by the marker in the fixed / max-budget and rank-only rows; \textbf{D} shows medians and interquartile ranges of 320 (context count), 640 (forecasts and original scalar) or 1,280 (pilot and full training) timing draws, hidden by the marker where narrower than it. Marker shapes and colours are shared across \textbf{A--D}.'''),
('morphology_estimation','si_10_selection_statistics',[('N36','*')],
 'Finite calibration, adaptive construction and direct gradient learning.',
 r'''Noisy-example estimation and learning in multi-affine scalar trees. \textbf{A}, Sealed calibration protocol (Section~\ref{note:finite_calibration_morphology}) at the primary condition (256 labels, noise SD 0.5; 64 held out): the estimator (green; Lasso over 255 Walsh terms) scores 12 candidate cuts, and two-sweep pilot fits (orange) are discarded before final training. \textbf{B}, Pooled fixed-baseline-minus-estimated and pilot-minus-estimated NMSE regret: small dots, 80 per-task paired differences; open squares, family means; short rule, pooled mean; whisker, Bonferroni-adjusted 97.5\% paired interval; solid rule, zero; the 0.01-NMSE margin lies below both lower confidence bounds (0.083, 0.027). \textbf{C}, Pooled menu regret (excess clean-test NMSE above the best trained candidate) against calibration labels: solid curves, filled markers, noise SD 0.5; dashed, open markers, no noise; the fixed / estimated-rank baseline is one grey curve for both noise levels, and the estimated-cut noise contrast lies within the seed interval (1,024 labels: 0.022 [0.010, 0.035] noiseless, 0.033 [0.015, 0.056] at SD 0.5). \textbf{D}, Family-specific primary regret (green, estimated interactions; grey, fixed/estimated-rank baseline; orange, pilot): small dots, 20 per-task regrets; short rule, mean; the quartic estimated-cut interval (0.0027 [0.0014, 0.0046]) is narrower than the rule. \textbf{E}, Secondary adaptive trees (green; ALS fits) against two retrospective oracles, the best trained candidate (grey) and the true-target tree (purple); marks as in \textbf{D}. \textbf{F}, One-CPU selection time for eighty primary tasks: dots, task times; rules, medians (0.046 s and 0.066 s; 1.4-fold); whiskers, interquartile ranges. \textbf{G}, Fresh Adam cohort: estimated (green), development-fixed (grey) and target-informed (purple; oracle) trees under exact credit (filled; learning rate 0.01) and root broadcast (open; 0.003): small dots, 20 per-seed endpoints; dashed rule, label-noise floor 0.0225. \textbf{H}, Pooled paired noisy-test differences: small dots, 20 seed differences; open squares, family means (matching, quartic, nested, random, left to right); short rule, pooled mean; whisker, Bonferroni-adjusted 97.5\% interval for the two primary contrasts and pointwise 95\% for estimated-minus-target-informed; solid rule, zero. Target-informed construction is not a performance ceiling. Bands (\textbf{C}) and whiskers (\textbf{D,E,G}): pointwise 95\% whole-seed bootstrap intervals, $n=20$.'''),
]

# Sentences appended to a caption that is otherwise the frozen original.
CAPTION_APPEND = {
}

# Curated labels that must survive a merge as aliases (SI_NUMBERING 3.3).
ALIAS_EXTRA = {
 # 2026-09-13: the native renders N3, N22, N30 and N31 replaced the last four
 # sheets that pasted crops of frozen or training-script renders; the retired
 # sheets' labels stay as aliases of the figures that now carry their panels.
 # Old S31's label is main Fig. 6's (ALIAS_BLACKLIST) and old S18's stays with
 # physical_optimizer, which still pastes its E and F.
 'utility_signal_noise': ['fig:si_checkpoint_geometry','fig:supp_utility','fig:supp_alignment'],
 'measured_transfer_geometry': ['fig:si_measured_topology','fig:supp_measured_detail',
                                'fig:supp_measured_alignment_power'],
 'shunt_replication': ['fig:supp_weak_channel_linearization'],
 'shunt_sensitivity': ['fig:supp_focal_matrix','fig:supp_focal_detail'],
 # Native renders N14, N15 and N18 replaced the frozen sheets S43, S44 and S42
 # (2026-09-12); the frozen sheets' labels stay as aliases of the same figures.
 'boolean_capacity': ['fig:supp_boolean_theory'],
 'boolean_learning': ['fig:supp_boolean_learning'],
 'conductance_grouping': ['fig:supp_morphology_conductance'],
 # Native renders N7, N8, N12, N13, N16, N17, N19, N20, N21, N35 and N36
 # replaced the frozen sheets S49, S45, S36+S37, S40, S54, S55, S50, S51+S52,
 # S53, S38+S39 and S41 (2026-09-12); their labels stay as aliases.
 'mnist_dictionary_geometry': ['fig:supp-image-ladder-controls'],
 'error_field_geometry': ['fig:supp_image_diagnostics'],
 'scalar_tree_capacity': ['fig:supp_interaction_structure','fig:supp_constructive_morphology'],
 'oracle_profile_credit': ['fig:supp_morphology_credit'],
 'fixed_profile_budget': ['fig:supp_credit_rule_extension'],
 'credit_optimizer_controls': ['fig:supp_credit_rule_extension_controls'],
 'conductance_precision': ['fig:supp-conductance-small-effect'],
 'conductance_optimization': ['fig:supp-conductance-robustness','fig:supp-conductance-expanded-rates'],
 'local_gate_controls': ['fig:supp_local_gate_controls'],
 'finite_horizon': ['fig:supp_finite_horizon_selection','fig:supp_finite_horizon_prediction'],
 'morphology_estimation': ['fig:supp_morphology_calibration'],
}
# fig:physicaldepth is main Fig. 6's label; it must not be re-declared here.
ALIAS_BLACKLIST = {'fig:physicaldepth'}

# Panels that are drawn natively for the supplement because no crop of a
# frozen sheet can produce them (SI_PLAN M1 and AMENDMENTS B3).
EXTRA_ASSETS = {
 'X1': {'path': 'figures/supplementary/components/si_checkpoint_merged.pdf',
        'old_label': 'fig:si_checkpoint_merged_panel', 'letters': [],
        'builder': 'scripts/build_si_restored_panels.py::checkpoint_merged',
        'note': 'SI_PLAN M1: old S9A and old S9C on one axis.'},
 'X2': {'path': 'figures/supplementary/components/si_utility_bound.pdf',
        'old_label': 'fig:si_utility_bound_panel', 'letters': [],
        'builder': 'scripts/build_si_restored_panels.py::utility_bound',
        'note': 'AMENDMENTS B3: the one-step bound demoted out of main Fig. 1C.'},
}

# The Fig. 9 builder rewrites figures/components/credit_first_figure_08.pdf in
# place, so the frozen M9 provenance sheet (commit 55bf9b8, the SI_PLAN
# baseline) is kept under figures/supplementary/frozen/ and read from there.
# The sha256 in original_assets.json is unchanged and still verifies.
ASSET_PATH_OVERRIDES = {
 'M9': 'figures/supplementary/frozen/credit_first_figure_08_M9.pdf',
}

# Shared figure-level keys are removed from individual crops and placed once.
# Coordinates are points in the immutable input PDF, not data coordinates.
REMOVED_SHARED_REGIONS = {
 'S16': [[0,384,518.4,404.4]],
 'S32': [[0,0,518.4,37]],
 # S36, S37, S38 and S49 strips removed 2026-09-12: their sheets are pasted
 # whole from the native renders N12, N35 and N7.
}
SHARED_LEGENDS = {
 'ancestry_coefficients': [('S32',[120,0,440,36])],
 # The old S49 strip of mnist_dictionary_geometry and the DECISIONS Q6 legend
 # strip of scalar_tree_capacity (old S37D) were removed 2026-09-12: the
 # native renders N7 and N12 carry their own keys.
}
# The old S53 whole-sheet crop of local_gate_controls (to y 452, which cut the
# deregistered panel E) was removed 2026-09-12: the native render N21 is
# pasted whole.
WHOLE_CROPS = {}

# Audited decorated-panel bounds. Adjacent panels sometimes share margins;
# their axis fragments must not be imported with the selected panel.
PANEL_BOUNDS = {
 ('S27','B'):[266,10.2,515,188.6], ('S27','D'):[266,208.7,515,380.4],
 ('S12','H'):[135.8,334.8,288.5,466.9],
}
# 2026-09-13: the crops of old S46, S5 E, X1, X2 (sheet S3), old S31 and S18 A--D
# (S22), old S10 F--H and S47 (S30) and old S22 A, M9 B/D/E and S56 C (S31) were
# retired: those four sheets are pasted whole from the native renders N3, N22,
# N30 and N31, so their paste-layer boxes, redactions, patches and re-set text
# are gone with them.
PANEL_REDACTIONS = {}
PANEL_PATCHES = {}
PANEL_TEXT = {}
# Review pass 2026-09-23: declared text edits on frozen source panels, keyed
# like PANEL_REDACTIONS -- ('remove', line) or ('replace', line, new line).
PANEL_TEXT_EDITS = {}
REMOVED_SHARED_REGIONS['S27']=[[178.5,188.6,391.5,206.7]]
SHARED_LEGENDS['anatomy_capacity_controls']=[('S27',[178.5,188.6,391.5,206.7])]
PANEL_BOUNDS[('S20','B')]=[265.8,14.2,514.8,201.3]
REMOVED_SHARED_REGIONS['S19']=[[217.9,232.3,374.1,241]]
NATIVE_LEGENDS={}

# 2026-09-11: paste layer re-measured on the fifteen rebuilt upstream renders
# (analysis/figure_visual_review_20260910/supplement_wave1_reports.json).
# Stale hand-coded regions were deleted above: the S13 between-row key, the
# S24 between-row key (now inside S24 C), both S14 figure-level keys (each
# panel carries its own), the S46 D/E boxes and E 'analytic' tag of the old
# 4-module grid, the S19 A x-label redaction and restoration (now native),
# and the marker-less native legend of same_span_conditioning.
# Old S19 A: the ink union runs to x 297 only through markers clipped away by
# the broken axis; trim to the visible panel (x label bottom 233.8).
PANEL_BOUNDS[('S19','A')]=[13.5,10.8,276.5,236.0]
# Old S14: both rows on one column geometry (x 249-253 and 259-264.6 empty).
PANEL_BOUNDS[('S14','A')]=[9.99,4.76,253.0,131.28]
PANEL_BOUNDS[('S14','E')]=[9.99,302.97,253.0,487.47]
PANEL_BOUNDS[('S14','B')]=[259.0,4.76,503.86,156.16]
PANEL_BOUNDS[('S14','F')]=[259.0,302.97,503.86,487.47]
# Old S7/S8/S23: letter-excluded crops (ink union + 2 pt without the source
# letter), so the pasted letter sits about 5 pt above each title.
PANEL_BOUNDS[('S7','A')]=[34.0,25.2,168.1,112.7]
PANEL_BOUNDS[('S7','B')]=[183.5,25.2,321.2,157.9]
PANEL_BOUNDS[('S7','C')]=[349.7,25.2,493.2,157.9]
PANEL_BOUNDS[('S7','D')]=[15.2,187.2,168.8,327.5]
PANEL_BOUNDS[('S8','E')]=[177.5,190.4,341.4,330.7]
PANEL_BOUNDS[('S8','G')]=[13.5,355.6,172.6,488.2]
PANEL_BOUNDS[('S23','A')]=[15.5,23.2,380.1,179.8]
# Old S2 A and D (sheet S2 A, B) and old S3 C: letter-excluded tops (A's key
# top 20.0, D's axes top 208.2, C's title top 189.5), one crop height for A
# and D with both x-axis rules (y 166.0 and 350.8) 148.0 pt below the crop
# top, and every crop of the sheet starting 60.4 pt left of its own y spine
# (S02 x 62, S03 x 66.4), which is what old S3 C's row labels need; the
# extensions to x 1.6, 196 and 275 cover empty render margin only (old S2 B's
# y label starts at 198.6, old S2 E's at 278.0).
PANEL_BOUNDS[('S2','A')]=[1.6,18.0,196.0,189.5]
PANEL_BOUNDS[('S2','D')]=[1.6,202.8,275.0,374.3]
PANEL_BOUNDS[('S3','C')]=[6.0,187.5,503.9,344.2]
# Old S1 is pasted whole; these are its provenance rects only.  The letter
# partition hands old S1 B's y label to A and C's to B (letter 3 pt inside the
# label column), so the auto rects overlapped by 4-12 pt; split at x 185/349.
PANEL_BOUNDS[('S1','A')]=[26.0,5.8,174.3,188.3]
PANEL_BOUNDS[('S1','B')]=[184.0,5.8,326.9,200.6]
PANEL_BOUNDS[('S1','C')]=[348.2,5.8,505.1,205.5]


# 2026-09-11 (wave two): paste layer re-measured on the sixteen rebuilt renders
# (analysis/figure_visual_review_20260910/supplement_wave2_reports.json).
# Deleted above: the old S10 G/H tick-label patch (removed region, 'true' /
# 'descendants' PANEL_TEXT; G's tick is natively 'true relation' and the old
# region now sat on H's x label), the two S12 G/H redactions of an orphan label
# that no longer exists, and the old S12 H box (it carried D's x label and cut
# H's right axis and bottom).  Every box below is ink + 2 pt unless stated.
# Old S10 (legacy 12-module grid: row 0 A 3, B 5, C 4 modules; rows 1-2 three
# equal 4-module panels).  The legacy letter partition cuts G at x 160.8 while
# G's 'true relation' column and tick label run to x 183.7, where H's y label
# starts (183.7); G and H are cut at 185.2 / 181.5 and each paste redacts the
# neighbour's touching text in its own copy.  F, G and H are letter-excluded on
# the left (F's y label 342.6, G's 21.1) so the three fit one row at scale 1.0
# (490.8 pt of 498.4); tops keep the letter box like old S47 in the same sheet.
# Old S10 B (sheet S25 A): its left neighbour's bar runs to 134.1 and its tick
# label to 127.6, so the auto box carried them; letter-excluded left (y label
# 141.7) and title-top 33.1.  The box is padded right to the 249 pt column of
# old S27 B/D (S25's two-column grid, x 6.2 and 263.2) and the paste redacts
# old S10 C's territory (its y label starts 342.6) in its own copy.
PANEL_BOUNDS[('S10','B')]=[139.7,31.1,388.7,142.6]
PANEL_REDACTIONS[('S10','B')]=[[340.0,0.0,518.4,160.0]]
# Old S20 B: five two-line tick labels (bottom 199.3) and the x-axis spine to
# 501.4 (zero-height lines are invisible to make_bounds); title-top 16.2;
# padded right to the 249 pt column.  Old S27 B/D: title tops 12.2 / 210.7,
# B's x label bottom 186.6 and D's 378.4; the between-row key (text 189.6-205.7,
# handles from 180.4) is the shared legend.
# Old S12 (sheet S26): letter-excluded tops (title tops 21.0, 177.7, 336.8) so
# the sheet letter sits 5 pt above each title; the render still draws the
# deregistered I and J, so H is cut at 288.5 (its right-axis labels end 286.5,
# I's letter is at 288.9).
# B is padded into its blank margins (old A's ink ends 114.0; page edge) to
# 394.4 pt so that [B, D] no longer fits one row at scale 1.0 and the layout
# takes [1,3,2]: the B--D triplet on one row instead of D beside the schematic.
PANEL_BOUNDS[('S12','B')]=[124.0,19.0,518.4,127.8]
# The final x-axis text ends at y=323.26; the next row's I/J glyphs begin
# immediately above their text boxes at y=325.33. Exclude those neighbouring
# glyph tips without clipping the complete Wilcoxon labels or plotted marks.
PANEL_BOUNDS[('S12','D')]=[138.0,175.7,249.1,324.0]
PANEL_BOUNDS[('S12','E')]=[270.3,175.7,380.8,324.0]
PANEL_BOUNDS[('S12','F')]=[402.0,175.7,512.5,324.0]
PANEL_BOUNDS[('S12','G')]=[4.9,334.8,122.7,476.7]
# Old S18: every axes box is 147.7 x 107 pt (x 84-231.7 and 339.7-487.4) and
# the auto boxes cut the axes right edges (A/C/E at 223-228, D at 476.7) or
# ran to the page edge (F); A and B take title-baseline tops (22.4-9 = 13.4)
# so their titles sit 9 pt below the crop top like the re-set title of old
# S31 G in the same S22 row; C/D likewise (179.4-9).  Old S18 F is padded
# right (blank page margin) so the S24 rows share one left edge (x 9.5).
PANEL_BOUNDS[('S18','E')]=[2.0,322.8,234.1,472.7]
PANEL_BOUNDS[('S18','F')]=[257.7,322.8,516.9,472.7]
# Old S28 B (S24 C): row labels from x 8.4, x-axis spine to 503.6.
PANEL_BOUNDS[('S28','B')]=[6.2,209.5,505.6,389.5]
# Old S31 G (S22 D): the re-set 8 pt title centred over its axes (363.6-483.4;
# text length 132.1 pt in the paste font).  Old S31 D (S22 C): the horizontal
# colourbar title 'accuracy' (474.4-505.1 x 148.9-156.8) collided with the
# 'reversed tier placement' subtitle; its characters are removed through the
# 2.5 pt band between the panel title (bottom 150.7) and the subtitle (top
# 153.6), because MuPDF drops every character whose box touches a redaction,
# and the title is re-set rotated beside the bar (tick labels end 511.0).
# Old S11 and S21 (sheet S29): the auto boxes cut the grid and zero lines that
# run to the axes right edges (S11 B 384.8, S11 C 508.7, S21 A 249.0, S21 D
# 496.7).
PANEL_BOUNDS[('S11','B')]=[225.4,5.8,386.8,178.1]
PANEL_BOUNDS[('S11','C')]=[390.5,5.8,510.7,176.0]
PANEL_BOUNDS[('S21','A')]=[32.0,4.8,251.0,171.8]
PANEL_BOUNDS[('S21','D')]=[321.0,201.3,498.7,361.6]
# Old S25 (sheet S27, pasted whole): provenance rects to the axes right edges
# (B 510.6, C 239.9, D 510.6), which make_bounds misses.
PANEL_BOUNDS[('S25','B')]=[293.5,16.2,512.6,161.4]
PANEL_BOUNDS[('S25','C')]=[18.8,180.0,241.9,308.0]
PANEL_BOUNDS[('S25','D')]=[288.9,180.0,512.6,308.4]

# 2026-09-12: the fourteen sheets that rested on frozen renders with no
# generator (analysis/figure_visual_review_20260910/frozen_drafts.json) were
# first given a measured paste layer (crops, redactions, re-set labels, shared
# legends and patches keyed on old S45, S49, S36, S37, S40, S43, S44, S54,
# S55, S42, S50, S51, S52, S38 and S39) and have since all been replaced by
# native renders pasted whole: N14, N15 and N18 (S14, S15, S18) and then N7,
# N8, N12, N13, N16, N17, N19, N20, N21, N35 and N36 (S7, S8, S12, S13, S16,
# S17, S19, S20, S21, S35, S36).  Every paste-layer entry measured on those
# frozen renders was removed with them (the removed entries are listed in the
# commit that retired each key); the frozen sheets stay in the registry for
# provenance and their labels survive as aliases (ALIAS_EXTRA).

# This source ledger originally named only the rendered asset. The scientific
# builder explicitly reads these two tables; preserve that numerical closure.
EXPLICIT_NUMERICAL_SOURCES = {
 ('S25','B'):['source_data/irregular_tree_wavelets/cohort_scale_summary.csv'],
 ('S25','C'):['source_data/irregular_tree_wavelets/cell_scale_summary.csv',
              'source_data/irregular_tree_wavelets/cohort_scale_summary.csv'],
 ('S25','D'):['source_data/irregular_tree_wavelets/cohort_scale_summary.csv'],
}

# Per-figure Source Data directories (SI_PLAN 5.6), by new S-number index.
SOURCE_DATA_DIRS = {
 'credit_validation':['figure2'],
 'utility_signal_noise':['credit_phase_theory','review_evidence_reanalysis','alignment_controlled','prospective_input_validity'],
 'reliability_gain':['positive_conductance_reliability_step_consistent'],
 'same_span_conditioning':['same_span_coefficient_learning'],
 'image_generalization':['cifar10_shunting_stopping_extension','cifar10_additive_feedback_ladder_confirmatory','fashion_strict_scalar_control','mnist_between_within_factorial','regular_tree_regimes'],
 'mnist_dictionary_geometry':['image_ladder_controls','mnist_feedback_ladder'],
 'error_field_geometry':['figure2','prospective_input_validity'],
 'input_coverage_depth':['spatial_topology_audit','prospective_input_validity'],
 'branch_conflict_controls':['path_necessity_fashion','review_branch_trajectories','trained_subtree_address','clean_exact_bp'],
 'ancestry_coefficients':['review_coefficient_encoder','review_coefficient_hard_readout','trained_subtree_address','trained_partition_residual'],
 'scalar_tree_capacity':['morphology_structure'],
 'oracle_profile_credit':['morphology_credit'],
 'boolean_capacity':['boolean_theory'],
 'boolean_learning':['boolean_morphology'],
 'fixed_profile_budget':['credit_rule_extension'],
 'credit_optimizer_controls':['credit_rule_extension'],
 'conductance_grouping':['morphology_conductance'],
 'conductance_precision':['conductance_credit_demand'],
 'conductance_optimization':['conductance_credit_demand'],
 'local_gate_controls':['conductance_local_gate'],
 'physical_architecture':['additional_figure_controls','physical_alignment_dose','remaining_physical_experiments','nonlinear_physical_depth_confirmatory','physical_depth_h4_factorial','point_dendrite_credit_controls','task_family_alignment'],
 'physical_calibration':['nonlinear_physical_depth_accessibility','nonlinear_physical_depth_boundary','nonlinear_physical_depth_canary','nonlinear_physical_depth_coupling','nonlinear_physical_depth_signal'],
 'physical_optimizer':['remaining_physical_experiments','point_dendrite_credit_controls'],
 'anatomy_capacity_controls':['figure3','microns_v661_replication','pinky_v185_replication'],
 'inhibitory_spatial_controls':['microns_inhibitory_routes'],
 'coarse_energy':['irregular_tree_wavelets'],
 'anatomy_preprocessing':['review_morphology_uncertainty'],
 'shunt_sensitivity':['figure4','focal_selectivity_phase1','review_focal_depth'],
 'shunt_replication':['focal_selectivity_active_ensemble','focal_selectivity_phase1','microns_v661_replication'],
 'measured_transfer_geometry':['figure5','measured_alignment_power','fulltree_boundary','fulltree_within_span_oracle','review_response_baselines','credit_first_figures'],
 'measured_predictor_controls':['review_response_baselines'],
 'animal_credit_reanalysis':['alignment_controlled'],
 'original_selector':['prospective_morphology_selection'],
 'finite_horizon':['morphology_finite_horizon','prospective_morphology_selection'],
 'morphology_estimation':['morphology_calibration'],
}

# Why a dropped panel is not printed, and where its numbers still are
# (SI_PLAN 5.5).  Any (sheet, letter) without an override falls back to the
# generated record: its own sentence from original_captions.json, the sheet's
# generic reason and its registered numerical sources.
PANEL_REASONS = {
 'S17':'The six-animal inference is reported in Table S7 and the protocol in Section S11. The descriptive figure is retained in the source archive; it adds no within-tree evidence.',
 'S1':'Restored in full as Supplementary Fig. S1.',
 'S43':'Replaced by the native render N14 (scripts/build_supplementary_figure_boolean_capacity_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S14; no crop of this frozen render is used.',
 'S44':'Replaced by the native render N15 (scripts/build_supplementary_figure_boolean_learning_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S15; no crop of this frozen render is used.',
 'S42':'Replaced by the native render N18 (scripts/build_supplementary_figure_conductance_grouping_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S18; no crop of this frozen render is used.',
 'S49':'Replaced by the native render N7 (scripts/build_supplementary_figure_mnist_dictionary_geometry_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S7; no crop of this frozen render is used.',
 'S45':'Replaced by the native render N8 (scripts/build_supplementary_figure_error_field_geometry_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S8; no crop of this frozen render is used.',
 'S36':'Replaced by the native render N12 (scripts/build_supplementary_figure_scalar_tree_capacity_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S12; no crop of this frozen render is used.',
 'S37':'Replaced by the native render N12 (scripts/build_supplementary_figure_scalar_tree_capacity_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S12; no crop of this frozen render is used.',
 'S40':'Replaced by the native render N13 (scripts/build_supplementary_figure_oracle_profile_credit_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S13; no crop of this frozen render is used.',
 'S54':'Replaced by the native render N16 (scripts/build_supplementary_figure_fixed_profile_budget_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S16; no crop of this frozen render is used.',
 'S55':'Replaced by the native render N17 (scripts/build_supplementary_figure_credit_optimizer_controls_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S17; no crop of this frozen render is used.',
 'S50':'Replaced by the native render N19 (scripts/build_supplementary_figure_conductance_precision_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S19; no crop of this frozen render is used.',
 'S51':'Replaced by the native render N20 (scripts/build_supplementary_figure_conductance_optimization_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S20; no crop of this frozen render is used.',
 'S52':'Replaced by the native render N20 (scripts/build_supplementary_figure_conductance_optimization_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S20; no crop of this frozen render is used.',
 'S53':'Replaced by the native render N21 (scripts/build_supplementary_figure_local_gate_controls_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S21; no crop of this frozen render is used.',
 'S38':'Replaced by the native render N35 (scripts/build_supplementary_figure_finite_horizon_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S36; no crop of this frozen render is used.',
 'S39':'Replaced by the native render N35 (scripts/build_supplementary_figure_finite_horizon_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S36; no crop of this frozen render is used.',
 'S41':'Replaced by the native render N36 (scripts/build_supplementary_figure_morphology_estimation_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S37; no crop of this frozen render is used.',
 'S6':'Whole figure omitted; the exact static quadratic result is derived in Supplementary Section S3 and drawn in main Fig. 2C.',
 'S16':'Whole figure omitted; Supplementary Section S1 retains the 135/200 seed-task optima and the 13/20 and 8/20 seed-level counts in prose.',
 'S26':'Whole figure omitted; the same-seed rerun outcomes and the two overflow pairs remain in Table S10 (physical-depth reproducibility).',
 'S48':'Promoted to main Fig. 8D; no copy is kept in the supplement.',
}
PANEL_CONTENT = {
 ('S9','A'): dict(content='Candidate/exact gradient cosine at 120 valid trained checkpoints, by delivery rule.',
                  reason='Re-rendered natively, together with old S9C, as the single merged axis S3E (SI_PLAN M1); no crop of this panel is pasted.',
                  numbers_at='source_data/prospective_input_validity/mechanism_checkpoint_rows_valid.csv'),
 ('S9','B'): dict(content='Scaled gradient capture at the same 120 checkpoints.',
                  reason='Omitted; direction and one-step progress are the two quantities the merged S3E carries, and capture is retained in Source Data.',
                  numbers_at='source_data/prospective_input_validity/mechanism_checkpoint_rows_valid.csv'),
 ('S9','C'): dict(content='Norm-matched one-step loss decrease as a fraction of the exact step, at the same 120 checkpoints.',
                  reason='Re-rendered natively, together with old S9A, as the single merged axis S3E (SI_PLAN M1).',
                  numbers_at='source_data/prospective_input_validity/mechanism_checkpoint_rows_valid.csv'),
 ('S9','D'): dict(content='Within-rule association between candidate/exact gradient cosine and norm-matched one-step progress, as a scatter over the same 120 checkpoints.',
                  reason='Deleted in the M1 merge: the two box panels state the association, and this panel printed the placeholder string "legend: within-rule association" inside its axes.',
                  numbers_at='source_data/prospective_input_validity/mechanism_checkpoint_rows_valid.csv'),
 ('S19','B'): dict(content='Context-0 forgetting after training on context 1, ten paired seeds, for the six delivery rules.',
                   reason='Promoted to main Fig. 2E; removed from the supplement, no copy kept.',
                   numbers_at='source_data/review_branch_trajectories/'),
 ('S22','B'): dict(content='Shared-ancestry/response rank correlation after controlling Euclidean separation and soma-to-contact depth differences.',
                   reason='Deleted in the M2 merge: it duplicates main Fig. 9A,B.',
                   numbers_at='source_data/fulltree_boundary/'),
 ('S22','C'): dict(content='Target-specific associations for shared-path, negative-tree-distance, partial-correlation and same-major-branch measures.',
                   reason='Promoted to main Fig. 9C.',
                   numbers_at='source_data/fulltree_boundary/'),
 ('M9','C'): dict(content='Input coverage in all thirteen scans, ordered by input count (48.0% of mapped inputs reached).',
                  reason='Promoted to main Fig. 9F.',
                  numbers_at='source_data/credit_first_figures/'),
 ('S32','A'): dict(content='Protocol box: separate calibration cues and local route-activation targets train a four-output softmax encoder, frozen during task learning.',
                   reason='Deleted (AMENDMENTS B4): the protocol is stated in Supplementary Section S3 and in Methods.',
                   numbers_at='source_data/review_coefficient_encoder/'),
 ('S32','B'): dict(content='Held-out accuracy versus cue-noise standard deviation at 256 calibration trials and zero delay.',
                   reason='Promoted to main Fig. 3F; removed from the supplement, no copy kept.',
                   numbers_at='source_data/review_coefficient_encoder/condition_summary.csv'),
 ('S32','E'): dict(content='Task-learning trajectories in the primary cue condition.',
                   reason='Omitted; the endpoint contrasts it summarizes are drawn in main Fig. 3F and Supplementary Fig. S11B,C.',
                   numbers_at='source_data/review_coefficient_encoder/'),
 ('S32','F'): dict(content='Primary soft-encoder-minus-control accuracy differences relative to oracle, frozen and mismatched coefficients, twenty paired seeds.',
                   reason='Deleted (AMENDMENTS B4): it duplicates the contrast drawn in main Fig. 3D and carried a truncated y-axis label.',
                   numbers_at='source_data/review_coefficient_encoder/'),
 ('S53','E'): dict(content='Historical twenty-seed cancellation cohort (2101-2120): the norm of mixed context-conditional distal gradients divided by the mixture of their norms.',
                   reason='Promoted to main Fig. 5G and re-rendered natively there (DECISIONS Q7); it leaves the supplement, and neither the earlier curated crop nor the native render N21 contains it.',
                   numbers_at='source_data/conductance_credit_demand/opponent/summaries/context_gradient_summary.csv'),
 ('S23','B'): dict(content='Trained address capture versus held-out accuracy for the six restricted route families at K<8.',
                   reason='Omitted; capture alone does not predict utility and the caption of S11A now states the within-seed Spearman correlations (about 0.82 over all budgets, 0.64 for K<8).',
                   numbers_at='source_data/trained_subtree_address/'),
 ('S23','C'): dict(content='Within-seed Spearman correlations between capture and accuracy for all route conditions and after excluding K=8.',
                   reason='Omitted; the same two coefficients are stated in the caption of S11A.',
                   numbers_at='source_data/trained_subtree_address/'),
 ('S39','B'): dict(content='Original scalar forecast versus final test half-MSE in the joint-transfer arm.',
                   reason='Not restored; the caption of S35 states that the joint arm reproduces the feedback-only hexbins, whose rows are retained in Source Data.',
                   numbers_at='source_data/morphology_finite_horizon/runs/*/seed_*_predictions.csv'),
 ('S39','D'): dict(content='Calibration-derived Gaussian SGD forecast versus final test half-MSE in the joint-transfer arm.',
                   reason='Not restored; the caption of S35 states that the joint arm reproduces the feedback-only hexbins.',
                   numbers_at='source_data/morphology_finite_horizon/runs/*/seed_*_predictions.csv'),
 ('S39','E'): dict(content='Population-oracle predicted minus exact population loss under fresh examples, at label-noise SD 0.75 and update 256.',
                   reason='Not printed; the forecast-bias result is stated in the caption of S35 and retained in Source Data.',
                   numbers_at='source_data/morphology_finite_horizon/'),
 ('S39','F'): dict(content='Population-oracle predicted minus exact population loss under reused finite-cache minibatches.',
                   reason='Not printed as a panel: the native render N35 draws S35F without the SI_NUMBERING inset; the result is stated in the S35 caption and retained in Source Data.',
                   numbers_at='source_data/morphology_finite_horizon/'),
 ('S37','D'): dict(content='Counts, minimum exact depths, reconstruction error and coefficient magnitude of the 105 matching, 35 quartic and 24 nested targets.',
                   reason='No longer pasted (it was the DECISIONS Q6 legend strip of S12 until 2026-09-12): the native render N12 fixes the three-family key in its panel A and the target counts are in the S12 caption.',
                   numbers_at='source_data/morphology_structure/'),
 ('S52','C'): dict(content='Aligned-tuning control of the wider six-rate development sweep under Adam.',
                   reason='Omitted; all 288 outcomes are retained in the condition table and Source Data, and the S20 caption says so.',
                   numbers_at='source_data/conductance_credit_demand/'),
 ('S52','D'): dict(content='Aligned-tuning control of the wider six-rate development sweep under SGD.',
                   reason='Omitted; all 288 outcomes are retained in the condition table and Source Data.',
                   numbers_at='source_data/conductance_credit_demand/'),
 ('S45','G'): dict(content='Per-neuron-minus-strict-scalar and exact-path-minus-per-neuron effects, the DFA neuron-identity contrast and the ownership derangement strips.',
                   reason='Omitted from the split figure; the same contrasts are drawn in main Fig. 1E,F and in S6D,E.',
                   numbers_at='source_data/figure2/'),
 ('S31','E'): dict(content='Descriptive depth optima across independent two-, three- and four-level cohorts.',
                   reason='Omitted; the underlying grids are covered by S22B,C and S24B and all cell values remain in Source Data.',
                   numbers_at='source_data/nonlinear_physical_depth_confirmatory/'),
 ('S31','H'): dict(content='Credit-resolution effects: BP minus soma-broadcast, and exact-path minus shared-soma under LocalCA.',
                   reason='Omitted; the same two contrasts are drawn in S24C with their intervals.',
                   numbers_at='source_data/remaining_physical_experiments/'),
 ('S11','A'): dict(content='Shunt-minus-current-injection localization across fixed absolute doses at three membrane resistances and three background multipliers.',
                   reason='Replaced by the native render N29 (scripts/build_supplementary_figure_shunt_sensitivity_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S31; no crop of this render is used.',
                   numbers_at='source_data/focal_selectivity_phase1/paired_contrasts.csv'),
 ('S11','B'): dict(content='Transport selectivity S_k, the descendant-to-control ratio of median absolute Green-function entries, against full synaptic-gradient localization at unit input-conductance-normalized dose.',
                   reason='Replaced by the native render N29 (scripts/build_supplementary_figure_shunt_sensitivity_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S31; no crop of this render is used.',
                   numbers_at='source_data/focal_selectivity_phase1/cell_condition_metrics.csv'),
 ('S11','C'): dict(content='Fraction of descendant gradients attenuated, enhanced or sign reversed at R_m = 1,000 ohm cm^2, background multiplier one and unit dose.',
                   reason='Replaced by the native render N29 (scripts/build_supplementary_figure_shunt_sensitivity_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S31; no crop of this render is used.',
                   numbers_at='source_data/focal_selectivity_phase1/'),
 ('S21','A'): dict(content='Within-cell controls: shunt and baseline-current-matched injection localization, and the true relation against a reassigned foreign template.',
                   reason='Replaced by the native render N29 (scripts/build_supplementary_figure_shunt_sensitivity_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S31; no crop of this render is used.',
                   numbers_at='source_data/figure4/cell_primary_contrasts.csv'),
 ('S21','C'): dict(content='Dimensionless excitatory/inhibitory conductance-scale and normalized inhibitory-reversal sensitivity.',
                   reason='Replaced by the native render N29 (scripts/build_supplementary_figure_shunt_sensitivity_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S31; no crop of this render is used.',
                   numbers_at='source_data/figure4/'),
 ('S21','D'): dict(content='All mapped contacts versus directly typed presynaptic contacts, summarized by shunt-minus-current localization in eight reconstructed cells.',
                   reason='Replaced by the native render N29 (scripts/build_supplementary_figure_shunt_sensitivity_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S31; no crop of this render is used.',
                   numbers_at='source_data/figure4/direct_typed_cell_primary_contrasts.csv'),
 ('S12','A'): dict(content='Schematic of the inhibitory-route matching design.',
                   reason='Omitted; the matched-null design is described in Supplementary Section S9.',
                   numbers_at='source_data/microns_inhibitory_routes/'),
 ('S12','C'): dict(content='Contact counts per target for the twenty-target inhibitory analysis.',
                   reason='Omitted; the counts are tabulated in Supplementary Table S6.',
                   numbers_at='source_data/microns_inhibitory_routes/'),
 ('S8','H'): dict(content='Runtime in minutes for the historical noise-resilience depth cohort.',
                  reason='Omitted; the resource controls that bear on the depth claim are the contact and parameter counts printed as S9F.',
                  numbers_at='source_data/prospective_input_validity/'),
 ('S8','I'): dict(content='Peak memory in mebibytes for the same cohort.',
                  reason='Omitted; retained with the runtime record in Source Data.',
                  numbers_at='source_data/prospective_input_validity/'),
 ('S12','I'): dict(content='Weighted modeled-field capture versus site budget for the dense oracle, anatomically selected inhibitory routes, random sites, depth bins and shuffled routes.',
                   reason='Omitted; the same capacity comparison is drawn for the route cohorts in main Fig. 7C,D and in S25.',
                   numbers_at='source_data/microns_inhibitory_routes/'),
 ('S12','J'): dict(content='Eight-site capture within each of twenty targets.',
                   reason='Omitted; the per-target values are retained in Source Data and summarized by S12I\'s omitted record.',
                   numbers_at='source_data/microns_inhibitory_routes/'),
 ('S48','*'): dict(content='Focal-shunt localization versus input-conductance-normalized dose, against baseline-current-matched injection, 101 sites in eight cells.',
                   reason='Promoted to main Fig. 8D; no copy is kept in the supplement.',
                   numbers_at='source_data/focal_selectivity_phase1/'),
 ('M9','A'): dict(content='Schematic of the measured-response prediction model on a reconstructed target.',
                  reason='Omitted; the same construction is shown in main Fig. 9.',
                  numbers_at='source_data/credit_first_figures/'),
 ('S40','C'): dict(content='Paired exact-credit NMSE increase after shuffling leaf input assignments at fixed tree shape and parameter count.',
                   reason='Replaced by panel C of the native render N13, drawn from the same paired-contrast table and printed as S13C; the Adam strips support the leaf-assignment contrast discussed in the main text.',
                   numbers_at='source_data/morphology_credit/'),
}

# Figures whose frozen panel inventory cannot be pasted at PASTE_SCALE = 1.0
# inside the 540 pt sheet cap (SI_PLAN 5.1/5.2, DECISIONS Q5).  Each entry is
# a work queue item, not a waiver: it names the native re-render or legacy
# port that removes the exemption, and the reason the crop cannot.  build.py
# asserts that every below-target figure appears here and writes the table
# into configs/supplement_consolidation/audit_report.json.

# 2026-09-13: the four sheets that still pasted crops of frozen or
# training-script renders are now pasted whole from the native renders N3
# (S3), N22 (S22), N30 (S30) and N31 (S31).  Each crop they retire is
# recorded here with what it showed and where its numbers are.
PANEL_CONTENT.update({
 ('S46','A'): dict(content='Fixed-operator special case: the one-step utility bound U(M) for the update M(mu + xi), with mean alignment and mean squared update named on the schematic.',
                  reason='Replaced by the native render N3 (scripts/build_supplementary_figure_utility_signal_noise_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S3; no crop of this render is used.',
                  numbers_at='source_data/credit_phase_theory/'),
 ('S46','C'): dict(content='Subtree-minus-random spectral capture over route budget K and covariance mixture rho; the endpoints are isospectral and the full-rank contrast is zero.',
                  reason='Replaced by the native render N3 (scripts/build_supplementary_figure_utility_signal_noise_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S3; no crop of this render is used.',
                  numbers_at='source_data/credit_phase_theory/spectral_phase_summary.csv'),
 ('S46','D'): dict(content='Final quadratic loss against route resolution for four hierarchy widths, where amplified fine-scale noise makes an interior resolution the designed optimum.',
                  reason='Replaced by the native render N3 (scripts/build_supplementary_figure_utility_signal_noise_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S3; no crop of this render is used.',
                  numbers_at='source_data/credit_phase_theory/depth_training_summary.csv'),
 ('S46','E'): dict(content='Analytic projected-minus-unprojected one-step loss, (f_noise - f_sig)/2, over the sampled retained signal and noise fractions.',
                  reason='Replaced by the native render N3 (scripts/build_supplementary_figure_utility_signal_noise_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S3; no crop of this render is used.',
                  numbers_at='source_data/credit_phase_theory/projection_phase_summary.csv'),
 ('S5','E'): dict(content='Loss reduction after twenty projected steps, relative to the corresponding full-gradient sequence, in the eight constructed arbors.',
                  reason='Replaced by the native render N3 (scripts/build_supplementary_figure_utility_signal_noise_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S3; no crop of this render is used.',
                  numbers_at='source_data/alignment_controlled/cell_alignment_metrics.csv'),
 ('X1','*'): dict(content='Candidate/exact gradient cosine and norm-matched one-step loss decrease at the same 120 valid trained checkpoints (the SI_PLAN M1 merge of old S9A and S9C).',
                  reason='Replaced by the native render N3 (scripts/build_supplementary_figure_utility_signal_noise_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S3; no crop of this render is used.',
                  numbers_at='source_data/prospective_input_validity/mechanism_checkpoint_rows_valid.csv'),
 ('X2','*'): dict(content='Illustrative rank-noise trade-off: 2L times the optimized one-step bound for two (K, q) pairs (AMENDMENTS B3, demoted out of main Fig. 1C). Analytic, no data.',
                  reason='Replaced by the native render N3 (scripts/build_supplementary_figure_utility_signal_noise_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S3; no crop of this render is used.',
                  numbers_at='analytic; no table'),
 ('S31','B'): dict(content='Serial D3, resource-identical grouped-point and approximately parameter-matched dense multilayer-perceptron controls; only the first two share modules and contacts.',
                  reason='Replaced by the native render N22 (scripts/build_supplementary_figure_physical_architecture_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S24; no crop of this render is used.',
                  numbers_at='source_data/nonlinear_physical_depth_confirmatory/'),
 ('S31','C'): dict(content='Three-level aligned task across D1--D3 and the five feedback conditions.',
                  reason='Replaced by the native render N22 (scripts/build_supplementary_figure_physical_architecture_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S24; no crop of this render is used.',
                  numbers_at='source_data/nonlinear_physical_depth_confirmatory/condition_summary.csv'),
 ('S31','D'): dict(content='Four-level task, including D4, with aligned or reversed sensor placement.',
                  reason='Replaced by the native render N22 (scripts/build_supplementary_figure_physical_architecture_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S24; no crop of this render is used.',
                  numbers_at='source_data/physical_depth_h4_factorial/condition_summary.csv'),
 ('S31','G'): dict(content='Serial-minus-grouped accuracy at fixed D3 across task family and alignment under exact-path local credit assignment.',
                  reason='Replaced by the native render N22 (scripts/build_supplementary_figure_physical_architecture_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S24; no crop of this render is used.',
                  numbers_at='source_data/task_family_alignment/paired_contrasts.csv'),
 ('S18','A'): dict(content='Serial-minus-grouped-star accuracy effects and the aligned-minus-reversed change in that D3 effect, in percentage points.',
                  reason='Replaced by the native render N22 (scripts/build_supplementary_figure_physical_architecture_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S24; no crop of this render is used.',
                  numbers_at='source_data/point_dendrite_credit_controls/paired_contrasts.csv'),
 ('S18','B'): dict(content='Flexible point networks matched to active or total parameter count versus serial D3.',
                  reason='Replaced by the native render N22 (scripts/build_supplementary_figure_physical_architecture_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S24; no crop of this render is used.',
                  numbers_at='source_data/point_dendrite_credit_controls/combined_seed_outcomes.csv'),
 ('S18','C'): dict(content='Accuracy across sensor--task alignment alpha at D1, D2 and D3.',
                  reason='Replaced by the native render N22 (scripts/build_supplementary_figure_physical_architecture_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S24; no crop of this render is used.',
                  numbers_at='source_data/physical_alignment_dose/condition_summary.csv'),
 ('S18','D'): dict(content='Change in the D3-minus-D1 advantage from alpha = 0.25 to 0.75 and its within-seed fitted slope.',
                  reason='Replaced by the native render N22 (scripts/build_supplementary_figure_physical_architecture_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S24; no crop of this render is used.',
                  numbers_at='source_data/physical_alignment_dose/paired_contrasts.csv'),
 ('S10','F'): dict(content='Localization of exact-gradient changes after a unit-dose focal shunt and baseline-current-matched current injection (45 cells, 235 sites).',
                  reason='Replaced by the native render N30 (scripts/build_supplementary_figure_shunt_replication_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S32; no crop of this render is used.',
                  numbers_at='source_data/microns_v661_replication/supp_figure_focal_cells.csv'),
 ('S10','G'): dict(content='Localization for true descendant relations and relation templates reassigned to other focal sites (40 cells, 230 sites).',
                  reason='Replaced by the native render N30 (scripts/build_supplementary_figure_shunt_replication_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S32; no crop of this render is used.',
                  numbers_at='source_data/microns_v661_replication/supp_figure_focal_cells.csv'),
 ('S10','H'): dict(content='Direct presynaptic excitatory/inhibitory label coverage and eligible focal sites per cell.',
                  reason='Replaced by the native render N30 (scripts/build_supplementary_figure_shunt_replication_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S32; no crop of this render is used.',
                  numbers_at='source_data/microns_v661_replication/cohort_manifest.csv'),
 ('S47','A'): dict(content='Focal-shunt and baseline-current-matched injection localization across doses relative to local input conductance, for the fixed Jacobian and the passive model at the same calibration.',
                  reason='Replaced by the native render N30 (scripts/build_supplementary_figure_shunt_replication_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S32; no crop of this render is used.',
                  numbers_at='source_data/focal_selectivity_active_ensemble/condition_summary.csv'),
 ('S47','B'): dict(content='Paired shunt-minus-current localization contrasts at the three doses in the eight-cell weak-channel ensemble.',
                  reason='Replaced by the native render N30 (scripts/build_supplementary_figure_shunt_replication_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S32; no crop of this render is used.',
                  numbers_at='source_data/focal_selectivity_active_ensemble/paired_contrasts.csv'),
 ('S22','A'): dict(content='Number of mapped functional presynaptic partners for each of the seven targets.',
                  reason='Replaced by the native render N31 (scripts/build_supplementary_figure_measured_transfer_geometry_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S33; no crop of this render is used.',
                  numbers_at='source_data/figure5/functional_target_metrics.csv'),
 ('M9','B'): dict(content='Four-route support for the scan selected by median mapped-input count: rows are input coordinates and columns selected routes.',
                  reason='Replaced by the native render N31 (scripts/build_supplementary_figure_measured_transfer_geometry_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S33; no crop of this render is used.',
                  numbers_at='source_data/credit_first_figures/figure_08_support.csv'),
 ('M9','D'): dict(content='Response-prediction normalized MSE for exact learning, treeless ridge and the restricted dictionaries, with the random and shuffled surrogate rows.',
                  reason='Replaced by the native render N31 (scripts/build_supplementary_figure_measured_transfer_geometry_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S33; no crop of this render is used.',
                  numbers_at='source_data/credit_first_figures/figure_08_prediction_summary.csv'),
 ('M9','E'): dict(content='Common-checkpoint update reconstruction by the unrestricted fixed transfer profile, the restricted ancestry profile and oracle trialwise amplitudes.',
                  reason='Replaced by the native render N31 (scripts/build_supplementary_figure_measured_transfer_geometry_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S33; no crop of this render is used.',
                  numbers_at='source_data/fulltree_within_span_oracle/condition_summary.csv'),
 ('S56','C'): dict(content='Mean split-half Spearman reliability in 1,000 independent calibration datasets against the measured value, for 125 partner records.',
                  reason='Replaced by the native render N31 (scripts/build_supplementary_figure_measured_transfer_geometry_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S33; no crop of this render is used.',
                  numbers_at='source_data/measured_alignment_power/reliability_calibration_audit.csv'),
})

SCALE_EXEMPTIONS = {
 # 2026-09-12: the scalar_tree_capacity (S12), conductance_optimization (S20)
 # and finite_horizon (S35) exemptions were cleared by the native renders
 # N12, N20 and N35, pasted whole at scale 1.0; 2026-09-13 cleared
 # physical_architecture (S22) and measured_transfer_geometry (S31) the same
 # way, with N22 and N31.
 # 2026-09-13: the last exemption, shunt_sensitivity (S29), was cleared by the
 # native render N29 -- the legacy old-S11 panels were ported to the canvas --
 # so every one of the 36 sheets is now pasted whole at scale 1.0.
}

# Maintained native displays have a separate current-render registry.
# Keep the frozen original artwork identities available for source history.
import json as _json
from pathlib import Path as _Path
EXTRA_ASSETS.update(_json.loads((_Path(__file__).resolve().parents[2] /
    'configs/supplement_consolidation/current_native_assets.json').read_text()))

for _letter,_content,_source in [
 ('A','Schematic of the published positive and negative causal decoder weights.','animal_signed_contrasts.csv'),
 ('B','Descriptive neuron-level residual distributions from the published source workbook.','neuron_sd_residual_distributions.csv'),
 ('C','Six paired animal contrasts and their bootstrap intervals.','animal_signed_contrasts.csv')]:
 PANEL_CONTENT[('S17',_letter)]=dict(content=_content,
  reason='Redundant external display omitted; the inferential results remain in Table S7 and the protocol in Section S11.',
  numbers_at='source_data/animal_learning_francioni/'+_source)

# Current homes of displays removed in the September 21 presentation pass.
PANEL_CONTENT[('S31','C')]['reason']='Displayed once in main Figure 7D; the supplementary duplicate was removed.'
PANEL_CONTENT[('S31','G')]['reason']='The distinct exact-path LocalCA comparison remains in Supplementary Figure S24C; main Figure 7C uses exact BP.'
PANEL_CONTENT[('M9','B')]['reason']='The representative support matrix is displayed once in main Figure 10E; S31B retains the distinct all-scan occupancy summary.'
PANEL_CONTENT[('M9','D')]['reason']='Prediction outcomes remain in Table S6 and are compared with linear baselines in Figure S34A; the repeated absolute-error display was removed.'

# Two separately rendered mechanistic figures occupy S22 and S23.
CONSOLIDATED_FIGURE_NUMBERS = tuple(range(1, 22)) + tuple(range(24, 38))

EXPLICIT_NUMERICAL_SOURCES.update({
('S1', 'A'): ['source_data/inherited_neurips/theory_diag_by_condition.csv', 'source_data/inherited_neurips/path_gain_cv_mnist_ni5_seed.csv'],
('S1', 'B'): ['source_data/inherited_neurips/error_field_decomposition_runs.csv'],
('S1', 'C'): ['source_data/inherited_neurips/inhibition_causality_runs.csv'],
('N9', 'A'): ['source_data/spatial_topology_audit/task_feedback_effects.csv', 'source_data/prospective_input_validity/followup_publication_seed_outcomes.csv', 'source_data/curated_publication/si_mnist_coverage_plotted.csv'],
})

PANEL_REASONS.update({
 "S1": "The historical noise-task panels are archival only: normalization and aggregate execution identity are unresolved.",
 "S7": "The mixed-task learning panel is replaced by a verified MNIST-only panel; the unresolved noise-task series remains archival.",
 "S8": "The historical noise-task generator cannot be identified from retained execution records; these outcomes are archived and excluded from current task or depth claims.",
 "S12": "Weighted domain overlap repeats shared-path fraction; its definition and separate inference remain in Source Data. Other historical overview panels retain their original archival status.",
})

# Complete panel mappings for the fixed-budget CIFAR extension and the
# retained Fashion-MNIST panel; frozen S4 remains unchanged.
NUMERICAL_SOURCE_OVERRIDES = {}
NUMERICAL_SOURCE_OVERRIDES[('N4C', 'C')] = ['source_data/cifar10_shunting_stopping_extension/condition_summary.csv', 'source_data/cifar10_shunting_stopping_extension/paired_contrasts.csv', 'source_data/cifar10_shunting_stopping_extension/seed_outcomes.csv', 'source_data/cifar10_shunting_stopping_extension/summary.json']
NUMERICAL_SOURCE_OVERRIDES[('N4C', 'D')] = ['source_data/cifar10_additive_feedback_ladder_confirmatory/condition_summary.csv', 'source_data/cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv', 'source_data/cifar10_additive_feedback_ladder_confirmatory/seed_outcomes.csv', 'source_data/cifar10_additive_feedback_ladder_confirmatory/summary.json']
NUMERICAL_SOURCE_OVERRIDES[('S4', 'E')] = ['source_data/fashion_feedback_ladder/condition_summary.csv', 'source_data/fashion_feedback_ladder/paired_contrasts.csv', 'source_data/fashion_feedback_ladder/seed_outcomes.csv', 'source_data/fashion_feedback_ladder/audit.json']
PANEL_CONTENT[('N4C', 'A')] = dict(content='Historical regular-tree depth comparison.', reason='Unchanged supporting panel; only the fresh CIFAR panels C,D of this sheet are selected. Historical artwork remains separately registered.', numbers_at='source_data/regular_tree_regimes/' if 'A' != 'E' else 'source_data/fashion_feedback_ladder/')
PANEL_CONTENT[('N4C', 'B')] = dict(content='Historical regular-tree noise comparison.', reason='Unchanged supporting panel; only the fresh CIFAR panels C,D of this sheet are selected. Historical artwork remains separately registered.', numbers_at='source_data/regular_tree_regimes/' if 'B' != 'E' else 'source_data/fashion_feedback_ladder/')
PANEL_CONTENT[('N4C', 'E')] = dict(content='Historical Fashion-MNIST feedback ladder.', reason='Unchanged supporting panel; only the fresh CIFAR panels C,D of this sheet are selected. Historical artwork remains separately registered.', numbers_at='source_data/regular_tree_regimes/' if 'E' != 'E' else 'source_data/fashion_feedback_ladder/')
PANEL_CONTENT[('S4', 'C')] = dict(content='Historical five-seed shunting CIFAR comparison with matched-width fallback and random rank-four feedback.', reason='Current CIFAR ladders are drawn together from the complete source tables in N4C and displayed in S6A,B; historical inputs remain archived.', numbers_at='source_data/regular_tree_regimes/' if 'C' == 'C' else 'source_data/cifar10_additive_feedback_ladder_confirmatory/')
PANEL_CONTENT[('S4', 'D')] = dict(content='Twenty-seed raw-additive CIFAR feedback ladder.', reason='Current CIFAR ladders are drawn together from the complete source tables in N4C and displayed in S6A,B; historical inputs remain archived.', numbers_at='source_data/regular_tree_regimes/' if 'D' == 'C' else 'source_data/cifar10_additive_feedback_ladder_confirmatory/')

# Review pass 2026-09-23: frozen panels lose their headline titles, method notes
# and zero-line names (each fact is in the caption), and axis titles become
# sentence case.  Short condition labels, keys and effect annotations stay.
PANEL_TEXT_EDITS.update({
 ('S1','A'): [('remove','Path-gain field'), ('remove','darker site = larger path gain')],
 ('S1','B'): [('remove','Stage-resolved cosine'),
              ('replace','matched-width-field cosine','Matched-width-field cosine')],
 ('S1','C'): [('remove','Inhibition interventions'), ('replace','accuracy (%)','Accuracy (%)')],
 ('S2','A'): [('remove','mean ± s.d., 5 seeds'), ('replace','test accuracy (%)','Test accuracy (%)')],
 ('S2','D'): [('remove','mean ± s.d., 5 seeds')],
 ('S3','C'): [('remove','Exact-path-transport factorial (MNIST)'),
              ('remove','5 paired seeds / 95% bootstrap CI'),
              ('replace','exact transport − matched BP accuracy (pp)',
               'Exact transport − matched BP accuracy (pp)')],
 ('S13','A'): [('remove','Supplied state clamp')],
 ('S13','F'): [('remove','Paired endpoint effects'), ('remove','aligned'),
               ('replace','control loss − aligned loss','Control loss − aligned loss')],
 ('S24','C'): [('remove','No endpoint gain over no shunt'), ('remove','mean / 95% CI'),
               ('replace','credit-reliability heterogeneity','Credit-reliability heterogeneity'),
               ('replace','test loss after 40 updates','Test loss after 40 updates')],
 ('S24','D'): [('remove','Final-loss boundary at high heterogeneity'),
               ('remove','adaptive'), ('remove','local'),
               ('replace','control loss − adaptive-local loss','Control loss − adaptive-local loss')],
 ('S14','A'): [('remove','Same address span'), ('remove','same projector'),
               ('remove','different Gram matrix')],
 ('S14','B'): [('remove','Different conditioning'),
               ('replace','positive Gram mode','Positive Gram mode'),
               ('replace','eigenvalue / largest eigenvalue','Eigenvalue / largest eigenvalue')],
 ('S14','E'): [('remove','Final-loss crossover'),
               ('replace','effective sample size','Effective sample size'),
               ('replace','loss after 80 updates','Loss after 80 updates')],
 ('S14','F'): [('remove','Risk reversal'),
               ('replace','effective sample size','Effective sample size'),
               ('replace','nested − Haar loss','Nested − Haar loss')],
 ('S4','E'): [('remove','Fashion-MNIST')],
 ('S30','C'): [('remove','MNIST, DFA: large contrasts'),
               ('replace','paired accuracy difference (pp)','Paired accuracy difference (pp)')],
 ('S30','D'): [('remove','MNIST, DFA: contrasts below 0.5 pp'),
               ('replace','paired accuracy difference (pp; expanded scale)',
                'Paired accuracy difference (pp)')],
 ('S7','A'): [('remove','Branch input maps')],
 ('S7','B'): [('remove','Unique input coverage'), ('replace','contact map','Contact map'),
              ('replace','features per neuron','Features per neuron')],
 ('S7','C'): [('remove','Cross-branch collision'), ('replace','contact map','Contact map'),
              ('replace','mean branch Jaccard','Mean branch Jaccard')],
 ('S19','A'): [('remove','Two-branch learning'),
               ('remove','Small dots: seeds; diamonds: mean ± 95% CI')],
 ('S29','D'): [('replace','Update alignment, 2 branches','2 branches'),
               ('replace','credit conflict χ (sampled doses)','Credit conflict χ (sampled doses)'),
               ('replace','cosine with exact update','Cosine with exact update')],
 ('S29','E'): [('replace','Update alignment, 4 branches','4 branches'),
               ('replace','credit conflict χ (sampled doses)','Credit conflict χ (sampled doses)')],
 ('S29','F'): [('replace','Update alignment, 8 branches','8 branches'),
               ('replace','credit conflict χ (sampled doses)','Credit conflict χ (sampled doses)')],
 ('S23','A'): [('remove','Capture by route budget'),
               ('replace','teaching-route budget K (ordinal axis)','Teaching-route budget K (ordinal axis)'),
               ('replace','exact-field capture','Exact-field capture'),
               ('replace','capture − ancestry (pp)','Capture − ancestry (pp)')],
 ('S18','E'): [('remove','Aligned and reversed placement'),
               ('replace','paired difference (pp)','Paired difference (pp)')],
 ('S18','F'): [('remove','Depth-by-placement interaction'),
               ('replace','paired difference (pp)','Paired difference (pp)')],
 ('S28','B'): [('remove','Paired D3 optimizer and credit contrasts'),
               ('replace','paired difference (pp)','Paired difference (pp)')],
 ('S10','B'): [('remove','Route capacity'), ('replace','feedback channels','Feedback channels'),
               ('replace','modeled field capture','Modeled field capture')],
 ('S20','B'): [('remove','Direct-type capture, K = 8 (8 cells)'),
               ('replace','field capture','Field capture')],
 ('S27','B'): [('remove','Residual versus route budget'), ('remove','(n=10)'), ('remove','(n=8)'),
               ('replace','route budget K; qualifying cells','Route budget K','center',(0,-8.8)),
               ('replace','weighted relative residual norm','Weighted relative residual norm')],
 ('S27','D'): [('remove','Candidate routes and field rank'),
               ('replace','inhibitory-bearing segments (candidate routes)',
                'Inhibitory-bearing segments (candidate routes)'),
               ('replace','kernel participation rank','Kernel participation rank')],
 ('S12','B'): [('remove','Inhibitory contacts on reconstructed arbors'),
               ('replace','L2a n=232','L2a'), ('replace','L2c n=180','L2c'),
               ('replace','L3a n=240','L3a'), ('replace','L4a n=119','L4a'),
               ('replace','L4c n=91','L4c'), ('replace','L5ET n=259','L5ET')],
 ('S12','D'): [('replace','observed − matched (µm)','Observed − matched (µm)'),
               ('replace','one-sided Wilcoxon p','One-sided Wilcoxon p')],
 ('S12','E'): [('replace','observed − matched','Observed − matched'),
               ('replace','one-sided Wilcoxon p','One-sided Wilcoxon p')],
 ('S12','G'): [('remove','Class-specific domains'),
               ('replace','descendant-input fraction','Descendant-input fraction'),
               ('replace','cumulative fraction','Cumulative fraction')],
 ('S12','H'): [('remove','Placement control'),
               ('replace','descendant-input fraction','Descendant-input fraction')],
 ('S15','A'): [('remove','Severe gain shift at test time'),
               ('replace','physical depth','Physical depth'), ('replace','accuracy','Accuracy')],
 ('S15','B'): [('remove','Unseen gain shift erodes access'),
               ('replace','test gain SD','Test gain SD'), ('replace','test accuracy','Test accuracy'),
               ('replace','seed (n = 2)','seed','left')],
 ('S15','C'): [('remove','Coupling unlocks serial depth'),
               ('replace','initial child conductance','Initial child conductance'),
               ('replace','test accuracy','Test accuracy')],
 ('S15','D'): [('remove','Accessible non-ceiling boundary'), ('remove','criterion window:'),
               ('remove','depth means in [0.60, 0.95]'),
               ('replace','excitatory signal contrast','Excitatory signal contrast'),
               ('replace','test accuracy','Test accuracy')],
 ('S25','A'): [('remove','Irregular-tree scale basis')],
 ('S25','B'): [('remove','Disjoint same-animal cohort (47 cells)'),
               ('replace','non-scalar route-energy fraction','Non-scalar route-energy fraction')],
 ('S25','C'): [('remove','Coarse modes carry excess power'),
               ('replace','signal / isotropic-noise power','Signal / isotropic-noise power')],
 ('S25','D'): [('remove','Coarse excess: original vs disjoint cohort'), ('remove','(n=8)'),
               ('remove','(n=47)'),
               ('replace','actual − permuted coarse energy','Actual − permuted coarse energy')],
})

# the (n=...) tick line is gone; the crop stops above the legend row's top sliver
PANEL_BOUNDS[('S27','B')] = [266.0, 10.2, 515.0, 186.0]

# Review pass 2026-09-23: ancestry_coefficients B,C (S32 C,D) and its shared key.
PANEL_TEXT_EDITS.update({
 ('S32','C'): [('remove','Calibration data and computation'),
               ('replace','oracle 80.3%','oracle'), ('replace','frozen 18.7%','frozen')],
 ('S32','D'): [('remove','Cue / eligibility timing mismatch'),
               ('remove','soft, hard and mismatched'), ('remove','coincide at delays 1 and 4'),
               ('remove','(within 1 pp)'),
               ('replace','oracle 80.3%','oracle'), ('replace','frozen 18.7%','frozen')],
})
# the key crop stops above the asterisk footnote, which the caption states
SHARED_LEGENDS['ancestry_coefficients'] = [('S32', [120, 0, 440, 25.5])]

NUMERICAL_SOURCE_OVERRIDES[('N4C', 'E')] = ['source_data/fashion_strict_scalar_control/' + name for name in ('seed_outcomes.csv', 'condition_summary.csv', 'paired_contrasts.csv', 'summary.json')]
PANEL_CONTENT[('N4C', 'E')] = dict(content='Fresh four-rule Fashion-MNIST strict-scalar control.', reason='Complete 80-fit paired control replacing the earlier fallback-only comparison.', numbers_at='source_data/fashion_strict_scalar_control/')

NUMERICAL_SOURCE_OVERRIDES[('N22', 'B')] = [
    'source_data/physical_depth_h4_factorial/condition_summary.csv',
    'source_data/physical_depth_h4_factorial/seed_outcomes.csv',
    'source_data/additional_figure_controls/physical_placement_summary.csv',
    'source_data/additional_figure_controls/seed_outcomes.csv',
    'source_data/additional_figure_controls/paired_contrasts.csv',
]

NUMERICAL_SOURCE_OVERRIDES[('N22', 'C')] = ['source_data/task_family_alignment/' + name for name in ('architecture_effects.csv', 'seed_outcomes.csv')]
for _panel in ('D', 'E'):
    NUMERICAL_SOURCE_OVERRIDES[('N22', _panel)] = ['source_data/point_dendrite_credit_controls/' + name for name in ('condition_summary.csv', 'paired_contrasts.csv', 'combined_seed_outcomes.csv')]
for _panel in ('F', 'G'):
    NUMERICAL_SOURCE_OVERRIDES[('N22', _panel)] = ['source_data/physical_alignment_dose/' + name for name in ('condition_summary.csv', 'paired_contrasts.csv', 'combined_seed_outcomes.csv')]

PANEL_REASONS['S4'] = 'Current CIFAR and four-rule Fashion comparisons are rendered in N4C and displayed in S6; the earlier cohorts remain archived.'
PANEL_CONTENT[('S4', 'E')] = dict(content='Historical fallback-only Fashion-MNIST comparison.', reason='Replaced in the display by the fresh four-rule strict-scalar cohort in N4C E; original numerical outcomes remain available.', numbers_at='source_data/fashion_feedback_ladder/')
# Native Fashion's ylabel begins at x=349.88 pt. The automatic midpoint
# crop at 334.41 pt included its neighbour's right spine, forcing a 0.3%
# shrink of the sheet. Retain every Fashion mark and its full axis labels.
PANEL_BOUNDS[('N4C', 'E')] = (347.8, 191.735, 505.6357, 352.1729)
PANEL_TEXT_EDITS[('N4C', 'E')] = []
# Architecture identity is given by S6's caption; removing the two headings
# also keeps all three first-row axes aligned at their unchanged print size.
PANEL_TEXT_EDITS[('N4C', 'C')] = [('replace', 'Shunting CIFAR-10', 'Shunting')]
PANEL_TEXT_EDITS[('N4C', 'D')] = [('replace', 'Raw-additive CIFAR-10', 'Raw additive')]

# Separate architecture keys that nearly touched after source-panel locking.
for _panel in ('C', 'D'):
    PANEL_TEXT_EDITS[('S30', _panel)].append(('replace', 'additive', 'additive', 'left', (-6.0, 0.0)))
