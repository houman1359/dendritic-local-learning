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
('mechanistic_chain','si_01_exact',[('S1','*')],
 'Mechanistic chain from path gains to learning in regular-tree models.',
 r'''MNIST panels \textbf{A--C} use two-stage $[4,4]$ directed trees; the historical noise-resilience panels \textbf{D,E} use two-stage $[3,3]$ additive trees. \textbf{A}, Conductance-stage path-gain field in shunting and normalized-additive models at five inhibitory synapses per branch; the inset shows paired-seed coefficients of variation (CV; standard deviation divided by mean). The schematic is drawn on the shared balanced tree of main Fig.~1B rather than the $[4,4]$ geometry: one root-to-leaf route carries the serial path gains \(\alpha_1\), \(\alpha_2\), \(\alpha_3\), distal sites are tinted by their illustrative path gain (darker, larger) in each model's colour, and \(\delta_0\) is the somatic error. This loading quantity omits branch derivatives and is distinct from the trained exact-error profiles in Supplementary Fig.~S45D,E. \textbf{B}, Stage-resolved cosine of matched-width (MW) scalar-fallback versus exact MNIST errors; the width-matched somatic stage is excluded because its cosine is one by construction. \textbf{C}, MNIST post-training inhibition interventions: zero, sample shuffle, per-branch batch mean and a uniform matched mean. The dashed rule marks ten-class chance. These also change forward computation and do not isolate backward shunting. \textbf{D,E}, Feedback fidelity and learning on the additive noise task; solid and dashed lines denote matched-width scalar-fallback and exact transport. Bars show standard deviations across checkpoints in \textbf{B} and retained runs in \textbf{C}; shaded bands in \textbf{D,E} show standard deviations across retained runs (0.6--0.9 percentage points at doses 5--40 in \textbf{E}, and below 0.02 for the exact-path series in \textbf{D}, narrower than the marker); white points identify checkpoints in \textbf{B} and runs in \textbf{C}. Historical noise-task shunting aggregates with unresolved execution lineage are excluded and retained in Source Data only. The retained additive aggregates do not resolve their normalization setting. The state-matched focal analysis in main Fig.~8 provides the conductance-specific transport comparison.'''),
('credit_validation','si_01_exact',[('S2','AD'),('S3','C')],
 'Local eligibility and exact transport reproduce the learning gradient.',
 r'''\textbf{A}, Test accuracy on MNIST, Fashion-MNIST (F-MNIST) and figure-ground MNIST (FG-MNIST) for the shunting architecture trained by backpropagation (red-brown) or by the local rule (green), and for the normalized-additive architecture trained by the local rule (blue); bars are means $\pm$ 1 s.d. across five seeds, and the backpropagation intervals are narrower than the line weight. \textbf{B}, Transport, activation and additive-model implementation controls on MNIST; points are means $\pm$ 1 s.d. across five seeds, and the backpropagation interval is smaller than its marker. Colour names the architecture as in \textbf{A} (red-brown backpropagation, green shunting, blue additive, violet exact path transport); within a family the house marker (circle for shunting, square for additive) is the configuration drawn in \textbf{A} and a triangle is the ablated variant (identity activation without tanh reactivation; raw additive without normalization). The additive group draws the complete normalized/raw $\times$ gain-mode factorial, one gain mode per row. The MNIST backpropagation and normalized-additive conditions in \textbf{A} and \textbf{B} come from different run cohorts (the matched ceiling-refresh and local-rule tuning cohorts in \textbf{A}; the exact-transport and gain-normalization revision cohorts in \textbf{B}): the backpropagation means differ by 0.09 percentage points, and the normalized-additive bar in \textbf{A} lies within the range of the four normalized rows in \textbf{B}. \textbf{C}, Exact-transport-minus-backpropagation accuracy across three-/five-factor and local/backpropagated decoder choices; open circles are five paired seeds, diamonds are condition means and bars are 95\% paired-seed bootstrap intervals; dark violet denotes a backpropagated decoder and lighter violet a local decoder. Exact transport supplies an oracle compartment error. The local three-factor eligibility is sufficient when this error is supplied. Normalized- and raw-additive models are distinct controls; no noise-task shunting aggregate is used here.'''),
# 2026-09-11: the six S3 caption requests applied, then condensed to 18 caption
# lines: the full requested text overflowed the page by 128.6 pt under the
# 503.7 pt sheet (make supplement, 'Float too large').
('utility_signal_noise','si_01_exact',[('S46','ACDE'),('X1','*'),('S5','E'),('X2','*')],
 'Restricted routes trade signal retention against admitted noise, and the one-step quantity is measurable at trained states.',
 r'''\textbf{A}, Fixed-operator one-step utility bound for the update $M(\bm\mu+\bm\xi)$ (zero-mean noise $\bm\xi$): the boxed $U(M)$, the squared positive part of $\bm\mu^{\top}M\bm\mu$ over $2L\,\mathrm{E}\|M(\bm\mu+\bm\xi)\|^{2}$ ($L$, loss smoothness), lower-bounds the expected one-step loss decrease that \textbf{B--D} sweep. \textbf{B}, Subtree-minus-random spectral capture across route budget $K$ and covariance mixture $\rho$, printed per cell (red positive, blue negative, tint by magnitude); $K=16$ and $\rho=0$ are by-construction controls; only the endpoints are isospectral. \textbf{C}, Final quadratic loss against route resolution $D_{\rm r}$ for hierarchies $H_{\rm c}=1$--$4$, constructed so that fine-scale noise favours an interior resolution; $D_{\rm r}=4$ resolves every leaf ($M=I$); bands, 95\% bootstrap intervals, 50 paired seeds. \textbf{D}, Exact one-step projection boundary: retaining signal fraction $f_{\rm sig}$ and noise fraction $f_{\rm noise}$ changes normalized loss by $(f_{\rm noise}-f_{\rm sig})/2$, printed as in \textbf{B}; unequally spaced fractions form equal cells, so the dashed line is the cell-wise sign boundary, not the diagonal. \textbf{E}, Candidate/exact gradient cosine and norm-matched one-step loss decrease at the same 120 valid trained checkpoints (points; two off-scale strict-scalar values are printed at the floor); exact path is 1 by construction. Boxes, medians, quartiles and 1.5-interquartile-range whiskers; diamonds, means with 95\% checkpoint-bootstrap intervals (one held-out batch, relative step $10^{-5}$) narrower than the marker. \textbf{F}, Separate constructed positive control, eight reconstructed arbors: loss reduction after twenty projected steps versus the full-gradient sequence (green, morphology-selected routes; gray, random paths; blue, depth bins; pink, ancestry-shuffled routes); symbols, cell means with hierarchical 95\% bootstrap intervals. \textbf{G}, Illustrative rank--noise trade-off under isotropic noise, $2L$ times the optimized bound $q^2/(q+K\sigma^2)$ for $(K,q)=(1,0.8)$ (solid) and $(2,1)$ (dashed), crossing at $\sigma^2=4/7$ (shaded between); analytic, no data, not an endpoint-selection prediction. Imposed alignment and oracle projections explain the conditional relationship in \textbf{E,F}, which measures neither biological credit nor a long-horizon selector; the bound is local, and endpoint selection fails prospectively (Fig.~\ref{fig:si_original_selector}).'''),
('reliability_gain','si_01_exact',[('S13','AF'),('S24','CD')],
 'Fixed and estimated reliability gains have bounded learning benefits.',
 r'''All panels use positive-rate synthetic tasks with eight parallel, nonserial branch blocks. \textbf{A}, A supplied compensating current preserves branch voltage while a positive shunt reduces input resistance and eligibility. \textbf{B}, Paired control-minus-aligned final-loss contrasts at maximal heterogeneity for the best global, shuffled, anti-aligned and unshunted controls; fixed aligned shunting has no reliable final advantage over unshunted noisy learning. \textbf{C,D}, Adaptive-rule final loss and paired control-minus-adaptive contrasts. Adaptive local gain beats global and shuffled attenuation but loses to no shunt and the fixed oracle. In \textbf{B,D} the dashed line is the aligned or adaptive-local reference, translucent dots are the 50 individual paired seeds, and diamonds with bars are means and 95\% paired-seed bootstrap intervals; an interval narrower than its diamond is hidden by it. The point implementation given identical gains reproduces the conductance final loss exactly in all 50 seeds of each study (paired difference 0, 50/50 ties). Curves and shading in \textbf{C} are means and 95\% paired-seed bootstrap intervals across the same 50 seeds. The supplied compensating current in \textbf{A} and the paired gradient observations from which the adaptive rule in \textbf{C,D} estimates its gains are explicit information resources.'''),
('same_span_conditioning','si_01_exact',[('S14','ABEF')],
 'Identical route spans can learn differently through conditioning.',
 r'''\textbf{A}, Haar, raw nested (redundant) and scaled nested (statically scaled) coordinates parameterize the same rank-eight span in sixteen abstract coefficients. \textbf{B}, Positive Gram eigenvalues; condition numbers are 1, 15 and 388.52. \textbf{C}, Population loss after eighty updates across effective sample size on a logarithmic axis; curves and bands are means and 95\% seed-bootstrap intervals across fifty seeds. The strip above the axis shows, for every seed, the predicted effective sample size at which the exact finite-time risks of the nested and Haar coordinates are equal (raw nested: median 38.5, interquartile range 31.1--63.7; scaled nested: median 8.2, interquartile range 6.3--12.6); large symbols and dotted guides mark the medians. \textbf{D}, Paired nested-minus-Haar contrasts; negative values favor nested coordinates. Points and bars are means and 95\% paired-seed bootstrap intervals across fifty seeds (n = 50 pairs); dashed curves are the exact finite-time risk. Slow modes reduce variance at low data and retain bias at high data. Gram-preconditioned controls agree to $1.15\times10^{-15}$. This is a coefficient-learning comparison within one span, not a simulation of forward dendritic morphology or a biological implementation of the preconditioner.'''),
# ---------------------------------------------------------------- si_02 ----
('image_generalization','si_02_credit_rules_images',[('S4','CDE'),('S30','CD')],
 'The neuron-identity bottleneck generalizes across image and feedback controls.',
 r'''\textbf{A}, Flattened CIFAR-10 in the shunting architecture, with matched-width fallback, random rank-four, exact-path and backpropagation rules (five seeds; bars are $\pm$1 s.d.). \textbf{A} and \textbf{B} share one accuracy axis. \textbf{B}, Raw-additive CIFAR-10 with strict scalar, neuron-specific, exact-path and backpropagation feedback (twenty paired seeds; 95\% Student-$t$ intervals, at most 0.5 percentage points and mostly hidden by the mean symbols; grey band, the prespecified $\pm$1 percentage-point backpropagation equivalence margin). The strict scalar of \textbf{B}, \textbf{D} and \textbf{E} is a separately trained control distinct from the matched-width fallback of \textbf{A} and \textbf{C}. Exact path is 0.859 percentage points below neuron-specific feedback and equivalent to backpropagation within the margin. \textbf{C}, Fashion-MNIST in shunting and raw-additive trees (ten paired seeds; 95\% bootstrap intervals); path resolution adds no reliable benefit. In \textbf{A--C}, filled symbols are cohort means and small open symbols individual seeds. \textbf{D,E}, Paired accuracy contrasts in the MNIST between-by-within factorial under fixed random soma feedback (DFA): contrasts of order one percentage point or more in \textbf{D} and contrasts below half a point in \textbf{E}, whose axis is expanded about twentyfold. Open symbols are the fifteen paired seed differences, filled symbols their means and bars 95\% paired-seed bootstrap intervals; blue, raw additive; green, shunting. The top rows of \textbf{D} and \textbf{E} are the MNIST-DFA estimates of main Fig.~1E,F at supplement scale. The neuron-specific-minus-strict-scalar contrast of about eleven percentage points is drawn in \textbf{D}. DFA preserves the neuron-versus-scalar benefit, but does not add a second dendritic layer or increase the dimensionality of the ten-class readout error.'''),
('mnist_dictionary_geometry','si_02_credit_rules_images',[('S49','CDEF')],
 'Intermediate dictionaries capture more image-task credit without improving accuracy.',
 r'''\textbf{A,B}, Six-rule MNIST ladder in shunting and raw-additive networks; ten fresh paired seeds per architecture. Filled/open symbols use development-selected/original common rates. In \textbf{A} and \textbf{B} colour names the rule at each tick (amber, strict scalar; salmon, per neuron; blue, projected $K=1$; violet, $K=3$ subtrees; dark red, exact path; grey, decoder only); in \textbf{C} and \textbf{D} colour names the architecture (green, shunting; blue, raw additive). Where the selected multiplier equals the original common rate (shunting per neuron, projected $K=1$ and exact path; additive exact path and decoder only) the filled and open symbols are the same ten runs drawn twice. The four dendritic arms lie within 0.25 percentage points of one another in each architecture, below the symbol size at this scale; their paired within-tree contrasts are resolved in \textbf{C}. Decoder-only learning freezes the initialized core. \textbf{C}, Paired accuracy differences for three profiles versus a projected common profile, and exact versus three-profile credit; filled and open as in \textbf{A,B}. Note the scale: the full $y$ range of \textbf{C} is 0.35 percentage points, about one eightieth of the 30-point range of \textbf{A} and \textbf{B}. The number of seeds (of ten) with a positive difference is 7, 5, 5 and 3 for the four $K=3$ minus projected $K=1$ marks and 3, 4, 4 and 7 for the four exact minus $K=3$ marks, in the order drawn (shunting selected, shunting common, additive selected, additive common). \textbf{D}, Initial and trained mean activation-error capture (fraction) by uniform $K=1$ (triangles, solid lines) or subtree $K=3$ (diamonds, dashed lines) profiles, averaged over 2,048 held-out images within each seed; at shunting initialization the two profiles coincide (0.532 for both) and the triangle lies under the diamond. Projection coefficients use the exact activation-error field of this ten-seed cohort, so \textbf{D} reproduces the $K=1$ and $K=3$ points of main Fig.~1G for this cohort; voltage-space capture and the aggregate energy capture of the same fits are retained in Source Data and are not the plotted quantity. \textbf{A} and \textbf{B} show the ten per-seed values with the mean and no interval bars; \textbf{C} draws means with descriptive 95\% seed-bootstrap intervals; the intervals in \textbf{D} are at most 0.007 capture units, below the symbol size, and are not drawn. Greater capture did not yield an image accuracy benefit. All image cohorts have one dendritic layer and a linear readout.'''),
('error_field_geometry','si_02_credit_rules_images',[('S45','CDE')],
 'Error-field geometry separates direction, amplitude and spatial variation.',
 r'''Independent fifteen-seed flattened-MNIST cohort of 128 directed $[3,3]$ trees; green circles denote shunting and blue squares raw-additive trees. \textbf{A}, Branch-gradient cosine with the exact gradient at common fixed checkpoints trained with per-neuron feedback; matched-width scalar-fallback and per-neuron fields are evaluated at identical weights and states, and the exact-path cosine is one by construction and is not drawn. \textbf{B}, Exact voltage-error magnitude by depth at exact-path checkpoints: the batch root-mean-square ratio $\operatorname{RMS}(\delta^V_{n,u})/\operatorname{RMS}(\delta^V_{0,u})$, averaged over neurons and paths, with somatic value one by normalization; the open dashed diamond at soma (def.) marks this normalization constant, not a measurement. \textbf{C}, Within-depth path-specific residual error energy after subtracting the mean across paths within depth $d$: distal shunting attenuates magnitude but retains the larger path-specific fraction, rising from 17.9\% (shunting) and 18.1\% (raw additive), where the two mid symbols overlap, to 53\% and 32\%. Thin lines in \textbf{A} and \textbf{C} are the fifteen individual seeds per architecture and symbols are seed means; \textbf{B} draws seed means only. The 95\% seed-bootstrap intervals are drawn as bars but are hidden by the symbols (half-widths at most 0.03 in \textbf{A}, 0.08 in \textbf{B} and 1.2 percentage points in \textbf{C}). No additional models are trained here; greater path-specific variation did not yield an image accuracy benefit, and voltage- and activation-space capture are different quantities.'''),
('input_coverage_depth','si_02_credit_rules_images',[('S7','ABCD'),('S8','EG')],
 'Input coverage and ordinary-task depth are distinct from credit resolution.',
 r'''\textbf{A--D}, Spatial versus randomly assigned contacts in depth-four $[2,2,2,2]$ trees. \textbf{A}, The sixteen branches of one example neuron on the $28\times28$ input plane under the spatial sampler (left; grid lines mark its sixteen disjoint regions) and under the matched random sampler of the same seed (right); shade identifies the branch and pale pixels are uncontacted (57\%). \textbf{B,C}, Unique input coverage and cross-branch overlap for all 1,280 owner maps per condition (128 neurons $\times$ ten seeds; mean $\pm$ s.d.): contact count is matched, but spatial maps reach all 336 contacted inputs (336/336) and avoid collisions. \textbf{D}, Paired spatial-minus-random test accuracy under backpropagation (BP), the matched-width scalar field (MW scalar), the shared per-soma field (Neuron) and exact path transport (Exact path). Small symbols are the ten paired seeds; large symbols and bars are means and 95\% paired-seed bootstrap intervals, and a bar narrower than its symbol is hidden by it. Black circles are MNIST, each seed averaging the shunting and raw-additive cores; blue squares are the randomly projected noise task, raw-additive core only, in which BP and exact path coincide by construction. The accuracy benefit persists under backpropagation and therefore reflects a forward sparsity prior. \textbf{E,F}, Separate raw-additive historical noise cohort with sixteen terminals (blue squares; small symbols are the ten paired seeds): all four D4-minus-D1 accuracy contrasts are negative in 10/10 paired seeds, with BP and exact path identical in every seed because exact path transport reproduces backpropagation in an additive tree; active contact counts stay within 122.9--123.9 thousand across D1--D4 while parameter counts rise from 130.6 to 135.9 thousand. The noise-task executed generator remains unresolved and its shunting runs fail the positive-conductance input criterion. These retained additive outcomes are a depth boundary, not evidence for shunting; exclusions are enumerated in Table~\ref{tab:input_validity}.'''),
# ---------------------------------------------------------------- si_03 ----
('branch_conflict_controls','si_03_conflict_ancestry',[('S19','A'),('S29','DEF')],
 'Branch-selective credit limits interference and preserves update direction.',
 r'''\textbf{A}, Two-branch learning in ten paired seeds: held-out accuracy on the balanced binary task, with chance at 0.5 (dashed rule) and the axis broken over the two spans that hold no seed. Points show seeds; open diamonds and bars are means and 95\% paired-seed bootstrap intervals, which are narrower than the diamond where no bar is visible. The exact-path, correct-ancestry and gated-point rules (and backpropagation) give bitwise-identical accuracy in every seed and share one row. The random rank-2 control is bimodal across seeds (seven at chance, three recovered), so its row shows the median as a vertical bar and the two counts instead of a mean. The context-0 forgetting of the same cohort is drawn at the main-text type scale in main Fig.~2E. \textbf{B--D}, Shared-versus-exact gradient cosine in the separate branch-conflict experiment with two (\textbf{B}), four (\textbf{C}) or eight (\textbf{D}) branches, drawn against the eight sampled conflict doses at equal spacing (tick labels give $\chi$; the odd doses $4/7$ and $2/3$ bracket the boundaries). Solid curves evaluate common exact-learning states at epochs 0, 50 and 250; the dashed amber curve evaluates the shared learner's own final state. The violet dashed rule is the mean-field boundary $\chi_c = B/[2(B-1)]$, at 1, $2/3$ and $4/7$. Curves use twenty paired seeds; the 95\% bootstrap bands are narrower than the line stroke except near the boundary at epoch 0. Per-seed zero-alignment doses are tabulated in Source Data. An exact quadratic interference calculation is given in the accompanying derivation. Learned-state geometry can depart from the initialization mean-field approximation.'''),
('ancestry_coefficients','si_03_conflict_ancestry',[('S23','A'),('S32','CD')],
 'Available route span and learned route coefficients are different constraints.',
 r'''\textbf{A}, Trained field capture by route budget in the eight-context factorial, with $K$ on an ordinal axis. The ancestry, depth-bin and random-sparse partitions have coincident capture at matched bandwidth (within 0.005 at every budget; one stroke, drawn at the ancestry values), and derangement removes the target coordinate. The inset resolves the three partitions at $K=2$ and $K=4$ as capture minus the ancestry mean, in percentage points of capture, with each family's own 95\% interval; $n=20$ seeds throughout, and on the main axes the intervals are narrower than the symbols. Capture therefore does not explain the small ancestry-specific advantage, and capture alone does not predict trained utility: the within-seed Spearman correlation between capture and utility is about 0.82 over all budgets and 0.64 for $K<8$, and is retained numerically in Source Data. \textbf{B}, Held-out accuracy versus calibration set size (16, 64 and 256 examples, ordinal axis) at cue-noise standard deviation 0.5 and zero delay; 30 epochs are fixed, so both data and computation increase. \textbf{C}, Cue delay (0, 1 and 4 trials, ordinal axis, sharing the accuracy axis of \textbf{B}) at 256 calibration trials and noise SD 0.5; delays mismatch independent contexts and are measured in trials, not physiological time. In \textbf{B,C} the oracle-context and frozen-profile controls do not depend on the panel factor and are drawn as labelled reference lines (dash-dot and dotted) with their 95\% seed intervals as bands; the mismatched-encoder control coincides with the frozen profile in \textbf{B}, and the soft, hard and mismatched conditions coincide within one percentage point at delays 1 and 4 in \textbf{C}, where the three series are offset slightly in $x$. The dashed, square-marked curves marked with an asterisk use maximum-probability route selection by the same frozen encoder, an exploratory paired sensitivity specified after the soft outcomes with no encoder refit. Small symbols are the twenty fresh analysis seeds; curves, circles, squares and bars are means and 95\% paired-seed bootstrap intervals, and a bar narrower than its symbol is hidden by it. The cue-noise sensitivity of the same encoder is drawn at the main-text type scale in main Fig.~3F. The calibration cue and activation targets are supplied resources. No serial forward dendritic computation is present.'''),
# ---------------------------------------------------------------- si_04 ----
('scalar_tree_capacity','si_04_interactions_boolean',[('S36','ABD'),('S37','ABC')],
 'Interaction structure constrains scalar trees at matched input spectra and resources.',
 r'''\textbf{A}, Exhaustive pairwise matching and two-quartic target families on eight independent binary inputs; both have input-gradient second moment $I_8/4$, of rank eight. Each candidate has seven scalar multi-affine nodes, 28 coefficients and fourteen edges, and is read out at the root only. \textbf{B}, Full rank-two and centered rank-one cut-tail lower bounds versus achieved population NMSE in four-restart alternating least squares; matching targets are filled green circles and quartic targets open purple triangles, and mark area is proportional to the number of candidate fits at that point (1,680 fits, 140 targets $\times$ 12 candidates, which coincide on eight distinct points under the full bound and nine under the centered bound, with between 4 and 725 fits per point); where the two families share a point their marks overlap. \textbf{C}, Mean excess NMSE above the best fitted candidate in the fixed pool of twelve; score ties use tolerance $10^{-10}$ and lexicographic candidate identifier, and best-fixed selection uses family outcomes in hindsight. Each mark is a family mean over the 105 matching (green circles) or 35 quartic (purple triangles) targets; the per-target distributions are strongly skewed, since under the centered-cut bound 101 of the 105 matching targets and all 35 quartic targets have exactly zero excess, so the matching mean of 0.0095 is carried by four targets; the 980 per-target values are in Source Data. The fixed balanced-shape policy (mean excess 0.181 for matching and 0.043 for quartic targets) is not drawn. \textbf{D}, Actual tree returned for the first enumerated matching, a depth-three parallel construction. \textbf{E}, First seeded nested-prefix control tree, of minimum depth four. Leaves access the indicated input only. \textbf{F}, Minimum possible maximum centered-cut bound over all labeled binary trees at each depth limit of 3, 4 or 5 edges (the connecting segments are guides between the three limits, not bounds at fractional depth), calculated separately for every target and identical within each family: matching and quartic targets have bound zero at every limit and so admit exact depth-three trees, whereas nested controls have bound 0.146 at depth limit 3 and zero at 4 and 5, so they require depth four, and they do not share the $I_8/4$ input spectrum. A zero bound is equivalent to an exact fit in this multi-affine class. The matching series (green circles) lies exactly under the quartic series (purple squares) at zero and is therefore not visible; nested controls are orange triangles. The key below the panels gives the target counts of the three families and fixes the colours used on this sheet: green for the 105 matching targets, purple for the 35 quartic targets and orange for the 24 nested controls; marker shapes are not shared between panels and are stated above for each panel. All 164 targets construct with error below $7\times10^{-30}$ and coefficients of magnitude at most one. These exhaustive capacity results have no sampling error bars. The theorem concerns scalar multi-affine composition, not arbitrary conductance dendrites.'''),
('oracle_profile_credit','si_04_interactions_boolean',[('S40','ABCDEF')],
 'Task-dependent credit requirements in bounded multi-affine trees.',
 r'''\textbf{A,B}, Final test normalized mean squared error (NMSE) on oracle-compatible trees under Adam and stochastic gradient descent (SGD), respectively. Exact path (green), root broadcast (amber), one oracle profile (slate blue), two subtree profiles (violet) and two shuffled profiles (dark grey) share the forward model, initialization, examples and minibatches; this key is specific to this figure (exact path is red-brown in the main figures). Dotted segments mark known label-noise floors: 0.0225 for matching/nested tasks and 0.045 for quartic tasks. Projection coefficients access the exact current path field and are oracle diagnostics; root broadcast supplies the same scalar error to each internal node. \textbf{C}, Paired exact-credit NMSE increases after shuffling leaf input assignments (not the shuffled-profiles rule) while preserving tree shape and parameter count; green and amber here mean Adam and SGD, both under exact credit, and the Adam quartic interval (0.507--0.517) is hidden by its marker. \textbf{D}, Adam gradient cosine on oracle-compatible trees between the exact and delivered aggregate clean-population updates, evaluated at each rule's own trained state after 0, 1, 16, 64, 256 and 1,024 updates (0 drawn left of the log axis); the exact-path cosine is 1 by definition. Each seed averages the three task families before aggregation; these are not common-state or mean per-example cosines. Points and whiskers in \textbf{A--C} and bands in \textbf{D} are means and pointwise 95\% intervals from 10,000 whole-seed bootstrap draws, twenty fresh seeds per family. \textbf{E,F}, Final NMSE at learning rates 0.003, 0.01 and 0.03, averaged over both compatible and shuffled assignments and all three families; points are means over the same twenty seeds, without intervals, on different NMSE axes in \textbf{E} and \textbf{F}. Stars mark rates selected separately for each rule and optimizer using only development seeds. Matching and quartic tasks share the input-gradient second moment $I_8/4$; nested tasks have a different spectrum. The Adam series of \textbf{C} is the contrast of main Fig.~4H (drawn there horizontally with the paired seeds and a pooled row; ``matching'' is ``Pairwise'' there); \textbf{C} adds the SGD arm.'''),
('boolean_capacity','si_04_interactions_boolean',[('N14','*')],
 'Boolean interactions distinguish structure, depth and canonical credit.',
 r'''\textbf{A}, Truth tables of seven four-input templates on the sixteen equally weighted patterns (white, 0; slate, 1); columns are $abcd$ in numeric order, $a$ the most significant bit, hairlines between groups of four sharing $a$ and $b$. Mixed targets: \((a\land b)\lor(c\land d)\), \((a\land b)\mathbin{\mathrm{XOR}}(c\land d)\), \((a\mathbin{\mathrm{XOR}}b)\land(c\mathbin{\mathrm{XOR}}d)\) and nested \(a\land[b\lor(c\land d)]\). \textbf{B}, Centered-cut normalized-mean-squared-error (NMSE) regression lower bounds for all fifteen labeled binary trees (twelve coefficients, six edges); the rule separates the balanced (depth-two) trees T10--T12 from the twelve comb trees. The colour ramp is a power mapping (exponent 0.6) of the bound from a faint tint (zero cells palest); the nested row is positive on every tree but T15 (0.105--0.199); row maxima are printed at right. Open rings (a marker, not a bound) mark the 49 exactly representable family--tree pairs, certified with coefficients in $[-2,2]$ (largest $4/\sqrt{15}$; T indices and cut sets in Source Data). \textbf{C}, OR-of-AND and nested each use two AND gates and one OR gate (node colours as in \textbf{F}; inputs above, output below) but need minimum exact depths two and three. \textbf{D}, Minimum exact depth (ordinal axis, 2 or 3); the columns count exactly compatible trees among all fifteen (the ring counts of \textbf{B}) and among the three balanced: 15/15 and 3/3 for AND, OR and parity, 1/15 and 1/3 per mixed-gate target, 1/15 and 0/3 for nested. \textbf{E}, Post hoc target-projection energy $\|\mathbb E[z\mid x_S]\|^2$ of all seven families on the six two-input subsets, normalized target $z$: dot area is proportional to the energy, values printed beside the dots, exact zeros are open rings, and the rule separates aligned pairs ($ab$, $cd$) from crossed. XOR-of-AND has $1/5$ on every pair; parity has zero on every proper subset (all fourteen per family in Source Data). \textbf{F}, Canonical derivatives \(\partial F/\partial u\) for left-branch output \(u\) against right-branch output \(v\): AND, \(v\); OR, \(1-v\); XOR, \(1-2v\); dashed rule, zero. Green, amber and violet in \textbf{C,F} name the gates AND, OR and XOR, not bound values; \textbf{D,E} are drawn in ink only. Panels are exhaustive analytic diagnostics (no sampling error bars): positive bounds concern truth-value regression, not threshold classification; \textbf{E} is local target information, predicting neither grouping compatibility nor broadcast-learning failure; \textbf{F}'s raw derivatives need not survive training and imply no learning outcome; assumptions: Supplementary Note~\ref{note:boolean_morphology}.'''),
('boolean_learning','si_04_interactions_boolean',[('N15','*')],
 'Boolean grouping benefit and a small primary credit effect under frozen recipes.',
 r'''\textbf{A--D}, Clean population normalized mean squared error (NMSE), twenty-seed means, for seven target families (rows, labelled at \textbf{A,C}) and four trees (three balanced groupings and the comb a|(b|cd)) under Adam (\textbf{A,B}) and stochastic gradient descent (SGD; \textbf{C,D}) with exact (\textbf{A,C}) or broadcast (\textbf{B,D}) credit, on one logarithmic colour scale (key at right). Rates (0.003, 0.01, 0.03 and 0.01 in \textbf{A--D}) were selected per optimizer and rule on five development seeds; per-cell 95\% intervals are in Source Data. \textbf{E}, Prespecified Adam XOR-of-AND contrasts: grouping, the crossed-tree mean (ac|bd, ad|bc) minus the aligned ab|cd under exact credit; credit, broadcast minus exact on ab|cd. Dots (green, as XOR-of-AND in \textbf{G,H}), twenty paired seed differences; rules and whiskers, means and Bonferroni-adjusted 97.5\% intervals (grouping 0.549--0.583); amber dotted, the 0.01-NMSE mean-effect margin. The grouping axis is broken, zero and the margin in its foot strip; the credit subpanel has its own y range and a solid zero rule. Grouping passes both criteria, credit does not; both rules classify all sixteen patterns perfectly in every seed. \textbf{F}, Accuracy (gray circles) and balanced accuracy (blue crosses) versus NMSE for the 112 conditions at raw-output threshold 0.5; sixty overprint at 1.00 on both. Amber stars, the AND/OR constant-majority references (NMSE one, accuracy $15/16$, balanced accuracy $1/2$). \textbf{G}, Same-rate broadcast-minus-exact contrasts on the aligned XOR-of-AND tree, Adam (left) and SGD (right): dots, twenty paired seed differences per common rate (logarithmic axis); rules and whiskers, joined across rates, means with descriptive 95\% intervals; amber dotted, the 0.01 margin; dark solid, zero. The SGD y axis is symmetric-log (linear within $\pm 0.002$) to keep the rate-0.003 seed differences ($-0.94$ to $0.61$) in view. \textbf{H}, Broadcast gradient cosine on the aligned balanced tree for XOR-of-AND (green) and parity (violet); solid lines, filled markers Adam; dashed lines, open markers SGD; on all sixteen clean patterns at each learner's own state before clipping or optimizer transformation; eight checkpoint markers on a logarithmic update axis from 1, and update 0 (shared initial state, one whisker per optimizer) left of an axis break; amber dotted, cosine one. Bands and whiskers, pointwise 95\% intervals. The large parity credit effect is descriptive.'''),
('fixed_profile_budget','si_04_interactions_boolean',[('S54','ABCDEF')],
 'Longer matched budgets retain the task-dependent fixed-profile deficit.',
 r'''\textbf{A,B}, Pairwise targets under Adam at the separately selected rates and at the common rate (0.003), with exact, unit-broadcast, initial-profile and initial-sign credit; initial profile is the calibrated-broadcast rule of main Fig.~4 and initial sign keeps only its signs. In \textbf{A--D} and \textbf{F} colour names the credit rule: dark red, exact path; amber, unit broadcast; blue, initial profile; black dotted, initial sign. \textbf{C,D}, Quartic targets at the same two rate choices. Dashed verticals mark 1,024 updates; dotted horizontals the noise-only NMSE (0.0225 pairwise, 0.045 quartic). Drawn checkpoints are 64, 256, 512, 1,024 and every 256 updates thereafter; curves are straight segments between them. Exact and unit-broadcast selected rates equal the common rate, so those curves are the same runs in \textbf{A,B} and in \textbf{C,D}; initial sign traces initial profile in \textbf{A}, \textbf{B} and \textbf{D} (within 0.03 in \textbf{C}) because positive profile scaling largely cancels in Adam. At 16,384 updates the pairwise means are 0.0232 (exact), 0.0233 (unit) and 0.0247 (profile and sign) in \textbf{A} and 0.0232, 0.0233 and 0.0233 in \textbf{B}, within the line width. \textbf{E}, Quartic-minus-pairwise difference in initial-profile-minus-exact NMSE at budgets 1,024, 4,096, 8,192 and 16,384: circles/solid, terminal states; squares/dashed, validation-selected states. Here colour names the rate view, not the credit rule: dark red, rates selected per rule (the series of main Fig.~4F); blue, the common rate. The dotted line marks zero; the four 95\% bands overlap heavily, together spanning 0.65--1.01. \textbf{F}, All twenty exact-path and twenty initial-profile quartic seeds at 1,024 and 16,384 updates; the dashed diagonal marks unchanged error, the dotted guides the quartic noise-only NMSE; sixteen exact seeds lie within 0.044--0.050 on both axes and overplot. \textbf{F} resolves the three stalled seeds of main Fig.~4D; \textbf{A} and \textbf{C} repeat the selected-rate curves of main Fig.~4C,D. Curves show means and descriptive 95\% whole-seed bootstrap intervals. All 720 trajectories (twenty seed blocks, three tasks, four rules, two optimizers, both rate choices) are retained and the original endpoints replay exactly. Rates remain fixed and coefficients bounded; this is a retrospective-cohort extension under a prospective protocol, not fresh confirmation.'''),
('credit_optimizer_controls','si_04_interactions_boolean',[('S55','ABCD')],
 'Nested targets and SGD controls in the balanced budget extension.',
 r'''\textbf{A}, The same condition as main Fig.~4E, repeated as the reference for the SGD panels: nested targets under Adam at the selected rates, 0.003 for exact-path and unit-broadcast credit and 0.01 for initial-profile and initial-sign credit. \textbf{B}, Pairwise targets under SGD at the common rate 0.03 for all four rules. \textbf{C,D}, Quartic targets under SGD at the selected rates (\textbf{C}: 0.03 for exact-path and initial-profile credit, 0.01 for unit and initial-sign broadcast) and at the common rate 0.03 (\textbf{D}). Exact-path and initial-profile credit use rate 0.03 in both \textbf{C} and \textbf{D}, so those two curves are the same runs drawn twice; only the unit and initial-sign broadcast arms differ. Colours follow the main figures: exact path dark red, unit broadcast amber, initial-profile broadcast (the calibrated broadcast of main Fig.~4) blue and initial-sign broadcast dotted black. The same previously observed twenty seeds, fixed rules and rate choices are retained. Curves show mean test NMSE from update 64 onward; bands give descriptive 95\% whole-seed bootstrap intervals. A mean can sit where no seed does: in \textbf{B} the unit-broadcast mean rises to 0.34 at 16,384 updates because 2 of 20 seeds diverge (to 2.3 and 4.2) while the other 18 end at the floor (median 0.023), and 5 of 20 initial-profile seeds end at 0.050--0.125; in \textbf{D}, 18 of 20 unit-broadcast and 15 of 20 initial-sign seeds exceed NMSE 1 at 16,384 updates. In \textbf{A} the three broadcast controls coincide to within 6\% after 1,024 updates, and in \textbf{C} all three stay between 0.94 and 1.09 without diverging, so their curves overprint. Vertical dashed lines mark the original 1,024-update cap; horizontal dotted rules mark the noise-only NMSE, printed at the left end of each rule (0.023 in \textbf{A},\textbf{B}; 0.046 in \textbf{C},\textbf{D}). In \textbf{B} the initial-sign and unit-broadcast means reach this rule by update 512 and the exact-path mean by 2,304, so those three curves overprint on the rule thereafter apart from the unit-broadcast excursions after 9,472 updates. Panels state the task, optimizer and rate view. \textbf{A} and \textbf{B} are single conditions, not a matched pair; only \textbf{C} and \textbf{D} contrast the two rate views at fixed task and optimizer. Of the twelve task $\times$ optimizer $\times$ rate-view cells computed, this figure and Supplementary Fig.~\ref{fig:si_fixed_profile_budget} draw eight; the nested/Adam common-rate, pairwise/SGD selected-rate and both nested/SGD cells are in Source Data but not drawn. No outcome is excluded or used to retune rates. The vertical axis of \textbf{D} extends to about 37, against about 2 in \textbf{A}--\textbf{C}, to retain the divergent broadcast arms and their intervals; the two curves shared with \textbf{C} are therefore drawn at 1.6 times smaller vertical gain. These controls retain optimizer dependence and bounded-parameter scope.'''),
# ---------------------------------------------------------------- si_05 ----
('conductance_grouping','si_05_conductance_learning',[('N18','*')],
 'Input grouping and restricted feedback in positive-conductance trees.',
 r'''\textbf{A}, Seven-compartment directed shunting model (one physical shape; sixteen positive conductances: four excitatory and six inhibitory contacts, six couplings; soma-only readout) under the groupings $01|23$, $02|13$ and $03|12$. The first tree marks the contacts (per leaf one excitatory, blue, and one inhibitory, carmine; one inhibitory per proximal compartment); its six edges are the couplings, the proximal subtrees two greys, the soma yellow with the somatic-error arrow \(\delta_0\). Teacher-task permutations preserve the input-gradient spectrum. \textbf{B}, Analytic population lower bound from interactions crossing the student's two proximal input blocks (converged quadrature, not a certified bound): small pale dots, the 120 incompatible task/grouping pairs (six per seed, two values each); larger dots, the twenty seed means; mean bound 0.013222 population normalized mean squared error (NMSE); compatible groupings are exactly zero in all twenty seeds (superimposed). \textbf{C,D}, Held-out sample NMSE at checkpoints 0, 10, 100, 300 and 1,000 updates (ordinal axis, straight segments) under Adam and stochastic gradient descent (SGD) for exact path (dark red circles), fixed calibrated broadcast (blue squares), one oracle profile (grey triangles, dashed) and two subtree profiles (purple diamonds, dashed; main-text colours): coloured curves, compatible means over three task permutations within seed, with 95\% whiskers offset per checkpoint; grey band, the union of the four rules' 95\% intervals for the incompatible groupings (two per task; final means 0.029--0.032 NMSE under both optimizers); all rules share the initialization error. \textbf{E}, Paired compatible-tree NMSE differences from exact path at 1,000 updates for the three other rules under Adam and SGD: dots, the twenty independent seed means; dashed rule, zero; the Adam fixed-broadcast interval includes zero; the small SGD effect implies no universal need for precise credit. \textbf{F}, Aggregate gradient cosine on 256 calibration examples, compatible Adam models, same checkpoints: dots, the twenty seed means; symbols and whiskers, their mean and 95\% interval (individual records reach 0.79); exact path is 1 by construction, drawn as the dashed reference; fixed broadcast and one oracle profile differ by at most 0.0064. Means and whiskers in \textbf{B--F}: 10,000 whole-seed bootstrap draws, pointwise 95\% intervals.'''),
('conductance_precision','si_05_conductance_learning',[('S50','ABCD')],
 'A first conductance task produces a small precision benefit from exact credit.',
 r'''\textbf{A,B}, Test normalized mean squared error (NMSE) under Adam at the nine saved checkpoints (0, 64, 256, 1,024, 2,048, 4,096, 8,192, 12,288 and 16,384 updates, joined by straight segments) for ungated independent inputs and gated conflicting inputs in the original 16-conductance model, with exact-path credit, unit broadcast and the initial profile (the fixed calibrated broadcast). Curves show means and 95\% paired-seed bootstrap intervals across 20 fresh seeds; the vertical dotted line marks the original 4,096-update budget. All three rules share the initialization error (24.8 in \textbf{A}, 19.2 in \textbf{B}), above the drawn range; curves enter the frame before 64 updates. All fits continue to 16,384 updates with unchanged optimizer and minibatch state. On the ungated task the three rules are indistinguishable at 16,384 updates (exact $5.9\times10^{-6}$ [$1.5\times10^{-6}$, $1.2\times10^{-5}$], unit broadcast $1.6\times10^{-6}$ [$7.0\times10^{-7}$, $2.7\times10^{-6}$]); the precision benefit of exact credit is a gated-task effect. Unit-broadcast and initial-profile curves differ here because global gradient clipping can break Adam's invariance to fixed positive coordinate scaling. \textbf{C}, Gated-task NMSE differences, initial profile minus exact credit, at each fit's best validation checkpoint in the extended 16,384-update window, under Adam and under SGD, the latter at its separately selected rate with the same seeds and window. Points are the 20 paired seed differences; symbols and bars show their means and 95\% bootstrap intervals. The Adam gap is $0.000502$ [$0.000267$, $0.000772$] and the SGD gap is $0.000114$ [$0.000076$, $0.000157$], each positive in all 20 seeds; under Adam the mean gap thus stays below $6\times10^{-4}$ NMSE. \textbf{D}, Diagnostics at the exact-rule best-validation states distinguish the best rank-one capture of the six-compartment path field from capture by each student's fixed initial profile after weighting by local parameter eligibility. Both are squared-energy projection ratios with one freely fitted amplitude per example. Marks are means over 20 seeds with 95\% bootstrap intervals of the mean, narrower than the marker for three of the four points; ungated path capture is 1 by construction (effective rank 1.000 in every seed). High eligibility-weighted capture helps explain the small learning difference. Colours on this sheet: dark red is exact-path credit (\textbf{A,B}), the initial-profile-minus-exact gap (\textbf{C}) and rank-one path capture (\textbf{D}); orange is unit broadcast; blue is the initial profile (\textbf{A,B}) and eligibility-weighted profile capture (\textbf{D}); line style also separates the three rules in \textbf{A,B} (solid, dashed and dotted, in that order). The gated and ungated tasks change both inhibition and distractor dependence, so this first study is a task-family comparison. All 216 development fits, 240 fresh fits and their continuations are retained in Source Data.'''),
('conductance_optimization','si_05_conductance_learning',[('S51','BCD'),('S52','AB')],
 'Parameter-range, development and rate controls retain the opposed-task broadcast deficit.',
 r'''\textbf{A}, Task-by-credit interaction (opposed minus aligned calibrated-broadcast-minus-exact NMSE) under original $[-7,7]$ (open circles) and wider $[-20,20]$ (filled squares) log-conductance bounds with Adam and SGD; fits run to 16,384 updates. Symbols are twenty-seed means, bars 95\% paired bootstrap intervals; every seed is positive. Unit-broadcast rows are within 13\% and oracle rows below $4\times10^{-6}$. \textbf{B}, Initial-profile-minus-exact NMSE at each seed's last saved checkpoint before its fit first reaches an original bound: one dot per seed, no interval; all twenty are positive, and eleven share the 256 column, leaving seventeen visible marks. \textbf{C}, Four development regimes under Adam at the original 4,096-update budget (each rule's best of three rates; three development seeds): aligned strong, opposed strong, opposed moderate (parent inhibition one) and opposed ungated (zero); exact and oracle marks agree within 13\%, and the broadcast pair coincides except when ungated. \textbf{D,E}, Six-rate opposed-tuning sweeps under Adam and SGD at a 16,384-update budget, so exact-path values are lower than in \textbf{C} ($1.0\times10^{-7}$ against $2.0\times10^{-5}$ at Adam rate 0.03); dotted lines mark the originally selected rates. Points are three-seed means without error bars; seed ranges span up to a factor of 25, far below the three-decade rule separation at rates up to 1, and unit broadcast and initial profile coincide within 7.4\%. The improvement trigger was not reached; aligned sweeps and all 288 outcomes are in Source Data. Colours: dark red exact path (as in main Fig.~4), purple three-profile oracle, orange unit broadcast, blue open squares initial profile (main Fig.~4's calibrated broadcast); the dark red of \textbf{A} and lighter blue of \textbf{B} are not rule colours. Some broadcast fits reach the wider bounds; these are finite-budget controls, not a convergence proof.'''),
('local_gate_controls','si_05_conductance_learning',[('S53','ABCD')],
 'Local conductance gating is robust to rate choice but depends on where it is applied.',
 r'''\textbf{A,B}, Aligned tuning at 4,096 and 16,384 updates. \textbf{C,D}, Opposed tuning at the same budgets. Every rule receives each of the three Adam rates on the x axis, fixed (`frozen') before the confirmatory seeds were run; the middle column, 0.03, is the primary rate and the outer columns are robustness conditions. Each cell prints the arithmetic mean test NMSE at validation-selected checkpoints over $n=20$ new paired seeds and is filled by the base-ten logarithm of that same number on the bar at right (0 is NMSE 1; $-8$ is $10^{-8}$). No interval is drawn: per-cell medians and 95\% whole-seed bootstrap intervals are tabulated in Source Data (\texttt{condition\_means.csv}), and the within-cell seed spread of $\log_{10}$ NMSE (median s.d. 0.54 decades) means that differences of less than about a decade between rows are not resolved by this table; the fill resolves only the failed-versus-accurate split, so finer comparisons are read from the printed digits. Row labels are those of the frozen render; in the vocabulary of main Fig.~5, `Initial profile' is the calibrated broadcast (profile fixed from the initial exact path factors), `Local distal gate' is the distal gate, `Gate also proximal' is the also-proximal placement and `Two leaf patterns + unit proximal' is the two-profile oracle. Unit broadcast and calibrated broadcast (`Initial profile') are two rules whose means agree to two significant figures in every cell, so the two rows are drawn identically and are read as one broadcast row; individual seeds differ by up to 3.1-fold between them. Exact path, the three-pattern oracle, the local distal gate and the shunt-proportional gate learn accurately. Swapping the gate fails; extending the gate to proximal compartments leaves a substantial opposed-task deficit. The two leaf patterns with unit proximal feedback isolate this placement effect. On the aligned task at 16,384 updates the gate-also-proximal means ($1.4$--$3.3\times10^{-6}$) are raised 17--25-fold above their medians ($0.8$--$2.0\times10^{-7}$) by one or two outlier seeds, and those medians match the accurate rules' ($0.5$--$1.9\times10^{-7}$), so the darker fill of that row in \textbf{B} is an outlier effect, not an aligned-task deficit; smaller outlier inflation (about tenfold) affects the two-leaf oracle at 0.03 and the shunt-proportional gate at 0.1 in \textbf{D}. These are supplied-context, bounded-conductance experiments; the local gate does not estimate arbitrary route coefficients. Both budgets and all rates and seeds remain included; the historical cancellation cohort of the same study is drawn at the main-text type scale in main Fig.~5G.'''),
# ---------------------------------------------------------------- si_06 ----
('physical_architecture','si_06_physical_depth',[('S31','BCDG'),('S18','ABCD')],
 'Task-matched serial computation differs from grouped and flexible point controls.',
 r'''\textbf{A}, Serial D3, resource-identical grouped-point and approximately parameter-matched flexible point networks; only serial and grouped models share modules and contacts. \textbf{B}, Original-budget three-level aligned task across D1--D3 and the five-arm feedback ladder. \textbf{C}, Four-level task including D4 under aligned or reversed sensor placement. \textbf{D}, Serial-minus-grouped accuracy at fixed D3 across nested, flat and local-ratio task families under exact-path LocalCA (backpropagation counterpart, main Fig.~6C). \textbf{E}, Serial-minus-star accuracy at D1--D3, where the star is the resource-identical all-active grouped control; the open marker is the derived aligned-minus-reversed difference of the D3 contrast, and the axis break removes the empty 9--28 pp span. \textbf{F}, Flexible point networks matched to active or to total parameter count, ten seeds per condition drawn (y axis 0.90--1.00); the serial D3 mark is the exact-BP run of \textbf{B}. \textbf{G}, Accuracy across task--sensor alignment $\alpha$ (depth colours ordinal within this panel). \textbf{H}, The D3-minus-D1 advantage at each $\alpha$ with the ten paired seeds behind each mean, on an axis broken between 4.5 and 28.5 pp; from $\alpha=0.25$ to 0.75 the advantage grows by 2.92 points (95\% interval 2.65--3.17). Curves and differences use ten paired seeds per cohort and 95\% paired-seed bootstrap intervals; in \textbf{B} and \textbf{G} most intervals are narrower than the marker. In \textbf{C} the raw-additive arm ran under aligned placement only, so its reversed row is empty, and an outlined cell marks the best depth of a row when its 95\% interval clears the runner-up's. Main Fig.~6A,D re-render the architectures and the three-tier ladder from the same seeds; the two-tier task is in Supplementary Fig.~S24B.'''),
('physical_calibration','si_06_physical_depth',[('S15','*')],
 'Transparent calibration of the nonlinear physical-depth operating point.',
 r'''All panels use the three-level hierarchical gain-load task and serial shunting morphologies D1 \([8]\), D2 \([2,3]\) and D3 \([2,1,2]\), each with eight nonsomatic compartments per soma. They test the prerequisite for the main hypothesis: whether the mechanism-matched divisive task is accessible at every depth before comparing depth benefits. \textbf{A}, The original aligned shunting-backpropagation (BP) pilot at signal contrast 0.24 fit the training distribution increasingly with depth but remained at chance under the prespecified severe gain shift (test gain standard deviation (SD) 1.2). Bars grow from chance (0.5) and show two-seed means; colour encodes depth as in \textbf{B--D}, filled bars are the training split, open bars the severe-shift test split, and open circles the individual exploratory seeds. \textbf{B}, With training gain SD fixed at 0.25, increasing unseen test gain variation eroded accessibility at every depth. The dotted line marks the matched train/test condition. \textbf{C}, At matched gain and signal contrast 0.24, stronger child coupling selectively made serial D3 computation accessible; the dotted line marks child conductance 16 used subsequently. \textbf{D}, At matched gain and child conductance 16, increasing excitatory signal contrast moved D1--D3 away from chance without saturating D1 or D2. The shaded window marks the accessibility criterion frozen before the ladder was run: every depth's two-seed mean test accuracy in \([0.60, 0.95]\), each seed above 0.57 and D3 minus D1 at least 0.02 in both seeds. Contrast 0.72 failed it by 0.003 (D1 mean 0.597); the separately run 0.80 boundary (dotted line; D1 mean 0.609) met it and selected the operating point used for the analysis seeds. Lines and filled symbols show two-seed means and small open circles the two exploratory seeds, dodged slightly in \(x\); no confidence interval is drawn. Panels \textbf{A--C} share an expanded ordinate (0.48--0.70); \textbf{D} uses the full range. All pilot, ladder and boundary seeds were excluded from the confirmatory contrasts.'''),
('physical_optimizer','si_06_physical_depth',[('S18','EF'),('S28','B')],
 'Physical-depth conclusions depend on sensor fidelity, task hierarchy and optimization.',
 r'''\textbf{A}, Serial-minus-grouped-point accuracy at D3 under matched (aligned) and reversed sensor placement, the derived aligned-minus-reversed difference (open marker) and the star-minus-point architecture control, in percentage points. \textbf{B}, Separate independently seeded two-level task with D1 $[8]$ and D2 $[4,1]$: the depth-by-placement interaction (aligned D2-minus-D1 gain minus the reversed gain) for serial and grouped-point backpropagation and for shared or exact-path LocalCA; arm colours follow Supplementary Fig.~S22B. \textbf{A,B} share one broken axis with 5 pp ticks (the empty 1.6--18 pp span is removed) and draw the ten paired seeds behind every mean. \textbf{C}, Original-budget D3 optimizer/credit-coordinate contrasts in the serial conductance morphology $[2,1,2]$, in drawn order: BP minus soma broadcast under the BP optimizer, 0.64 points (95\% interval 0.15--1.17); soma broadcast, BP minus LocalCA optimizer, 13.47 (12.89--14.05); soma-broadcast autograd minus shared-soma LocalCA, $-0.06$ ($-0.23$ to 0.09); exact-path minus shared-soma LocalCA, 10.96 (10.34--11.58); BP minus exact-path LocalCA, 3.08 (2.48--3.79). Means and intervals use ten paired seeds and 95\% bootstrap intervals; panel~\textbf{C} is generated from \texttt{source\_data/point\_dendrite\_credit\_controls/} and is registered in the Source Data manifest. These earlier recipe-specific effects do not imply a converged credit advantage: at 600 epochs the LocalCA accuracy difference reverses, while exact credit retains lower cross-entropy (main Fig.~6). All D3 continuations hit the cap; D1 usually stops early.'''),
# ---------------------------------------------------------------- si_07 ----
('anatomy_capacity_controls','si_07_anatomy',[('S10','B'),('S20','B'),('S27','BD')],
 'Cohort, label and arbor-size controls bound modeled anatomical capacity.',
 r'''\textbf{A}, Route-generated capture of constructed target fields versus budget in 47 disjoint same-mouse cells (46 at 16 channels); morphology routes use 8.1\% of dense-feedback wiring at eight channels (wiring density per cell and arm is tabulated in Source Data). The key below the sheet applies to \textbf{A} and \textbf{C}. Twenty modeled streams are averaged within each cell; bands are 95\% cell-bootstrap intervals of the mean, several times narrower than the cell-to-cell spread, and the two null arms (random paths, shuffled ancestry) nearly coincide. \textbf{B}, Initial eight-cell cohort: squared-energy capture at $K=8$ restricted to directly typed presynaptic contacts, every cell drawn, with the dense PCA oracle as the ceiling. \textbf{C,D}, Second-mouse Pinky cohort: relative residual norm versus route budget against the dense principal-component ceiling, every qualifying cell drawn faintly behind its arm mean, and the number of inhibitory-bearing segments (candidate routes) versus effective field rank for the ten of twelve selected cells that passed the typed-input criterion (open markers: the two excluded cells; 78--114 reconstructed segments per tree). Eligibility changes from ten cells at $K\leq4$ to eight at $K=8$. Panel \textbf{C} uses mean $\pm$ s.e.m., and its residual norm is not squared-energy capture. All fields are model generated. Main Fig.~7 equalizes a common broadcast and leads with actual-tree surrogate controls; these route-generated comparisons are structural positive controls, not measured biological teaching signals.'''),
('inhibitory_spatial_controls','si_07_anatomy',[('S12','BDEFGH')],
 'Spatial matching limits apparent inhibitory targeting of ancestry domains.',
 r'''\textbf{A}, Six layer-diverse reconstructions with mapped inhibitory contacts in $x$--$y$ projection, pia up. Grey lines are the compressed dendritic skeleton (segment end points joined by straight edges); each cell is scaled independently and its own bar marks 50~$\mu$m. \textbf{B--D}, Observed-minus-matched tree distance, shared-path fraction and weighted descendant-domain overlap (``domain overlap'') in the twenty-target analysis, after aggregation within 92 presynaptic axons, and after joint path-distance/Euclidean matching (``3D''). Points are the individual units; diamonds and whiskers are means and 95\% bootstrap intervals over the displayed inferential unit; the value beneath each unit is the one-sided Wilcoxon signed-rank $p$ (alternative: closer in \textbf{B}, larger in \textbf{C,D}). In \textbf{B}, two axon values beyond $\pm 105~\mu$m are drawn as open triangles at the frame with their values; they remain in every statistic. Persistence is judged by the Wilcoxon test: only tree distance persists after axon aggregation ($p = 0.0008$, 62 of 92 axons closer, versus $p = 0.10$ and $0.39$ for the two overlap measures), although its axon-level bootstrap interval of the mean spans zero because of the two off-scale axons; no endpoint of the \textbf{B--D} triplet persists after joint three-dimensional matching ($p \geq 0.11$). Shared-path fraction and domain overlap are nearly collinear (Pearson $r = 0.98$, $0.83$ and $0.99$ over the target, axon and 3D units), so \textbf{D} is a weighted restatement of \textbf{C}. \textbf{E}, Cumulative descendant-input fraction of contacts from distal-dendrite-targeting (DTC, $n = 1{,}142$) versus perisomatic-targeting (PTC, $n = 840$) interneurons, retaining one contact per axonal clump; faint curves are the twenty individual targets and bold curves the pooled contacts. The per-target median DTC-minus-PTC difference is $-0.016$ (95\% bootstrap interval $-0.023$ to $-0.011$; negative in 20 of 20 targets). \textbf{F}, Descendant-input fraction at observed inhibitory contacts versus matched locations in each of the twenty targets; the difference column shows the per-target observed-minus-matched values on the right axis with their mean and 95\% bootstrap interval ($+0.002$, $-0.002$ to $+0.006$; two-sided Wilcoxon $p = 0.55$). No reliable overall localization advantage is detected. Cells derive from one mouse. These structural comparisons do not show that contacts carried teaching signals.'''),
('coarse_energy','si_07_anatomy',[('S25','*')],
 'Irregular reconstructed trees concentrate modeled route-field energy coarsely, but the anatomy-specific excess does not replicate.',
 r'''\textbf{A}, Weighted unbalanced tree-Haar contrasts recursively partition each reconstructed arbor into coarse, intermediate and fine supports, defined by descendant excitatory-weight fractions \(>1/4\), \(1/16\)--\(1/4\) and \(\leq1/16\), respectively; the schematic colours one example subtree per bin (key), with the grey trunk leading to them and the soma drawn as the yellow ellipse. Green, blue and rose mark scale bins throughout the figure. \textbf{B}, Non-scalar conductance-weighted ancestry-route energy fraction in the disjoint 47-cell cohort (teal, filled circles), compared with column-permuted ancestry (grey, open squares) and dimension-matched isotropic noise (grey, open triangles). Small points are individual cells; large marks are cell means, and the black whiskers drawn over them are 95\% cell-bootstrap intervals for the actual-route series and for both nulls. \textbf{C}, Signal-to-isotropic-noise power by scale for all 47 cells (points); diamonds show means, with 95\% cell-bootstrap intervals drawn over them as black whiskers (the intermediate-scale interval is only slightly wider than the diamond), and the dashed line marks equality with isotropic noise. \textbf{D}, Actual-minus-permuted coarse energy per cell (points) in the original 8-cell cohort (light teal) and the disjoint 47-cell cohort (teal); diamonds show cohort means with 95\% cell-bootstrap intervals as black whiskers, and the dashed line marks zero. The positive original-cohort contrast is not reliable in the 47-cell cohort. Cells are the inferential units; all reconstructions derive from one mouse. Modeled route fields are not measured teaching signals.'''),
('anatomy_preprocessing','si_07_anatomy',[('S33','*')],
 'Label coverage and geometric preprocessing affect anatomical route dictionaries.',
 r'''All analyses use eight initial reconstructed cells from one mouse. \textbf{A}, Direct-presynaptic versus target-proxy excitatory/inhibitory (E/I) labels for 2,012 jointly labeled mapped contacts; cells print counts and row percentages and are shaded by row percentage (colour bar). \textbf{B}, Direct-label fraction by compartment: individual cells and cell means with 95\% cell-bootstrap intervals. Denominators differ (22--66 contacts per cell at the soma, 342 pooled; 157--1,541 internal, 5,568 pooled; 1,148--5,697 terminal, 25,580 pooled), so per-cell points carry unequal weight. \textbf{C}, Jaccard overlap of the four selected routes with the nominal hybrid-label, $5\,\mu$m dictionary across mapping thresholds (ordinal, equally spaced) and label choices; the open symbol at hybrid, $5\,\mu$m is the reference compared with itself (Jaccard $=1$). With hybrid labels, 5 of 8 cells change at least one route at $2\,\mu$m (mean 0.72) and 3 of 8 at $10\,\mu$m (mean 0.82); with direct-only labels all eight differ at every threshold (means 0.21--0.24). \textbf{D}, Capture of the fixed nominal field under heterogeneous log-radius perturbations indexed by their standard deviation (SD), for the hybrid and direct-only dictionaries as in \textbf{C}; thin lines join the eight cells across SD. The paired within-cell change from SD 0 to 0.5 is $-0.034$ with hybrid labels ($-0.050$ to $-0.018$; 7 of 8 cells decline) and $-0.011$ with direct-only labels ($-0.053$ to $0.019$). Large symbols and bars in \textbf{C,D} show cell means and 95\% cell-bootstrap intervals; small symbols are individual cells, tied values spread side by side. \textbf{E}, Distribution of the mean-radius compressed axial resistance divided by the exact series sum over raw edges (0.1-log-unit bins). No segment exceeds its series sum; 72 of 608 segments (8 cells) fall below 0.89 (0.05 log units), the worst to 0.003 (2.5 log units). \textbf{F}, Series-resistance minus mean-radius capture of the nominal field against the mean of the two, at the nominal condition (hybrid labels, $5\,\mu$m, SD 0), one point per cell; the dashed line marks equality. Six of eight cells are identical and two differ by $+0.049$ and $+0.017$.'''),
# ---------------------------------------------------------------- si_08 ----
('shunt_sensitivity','si_08_shunting',[('S11','A'),('S21','A'),('S11','BC'),('S21','DC')],
 'Dose, cable parameters and input mapping determine focal-shunt selectivity.',
 r'''\textbf{A}, Shunt-minus-current-injection localization across fixed absolute doses at three membrane resistances ($R_m$ of 300, 1,000 and 15,000~$\Omega\,\mathrm{cm}^2$: one lightness step, marker and dash pattern each) and three background-conductance multipliers (column groups), on a symmetric-log axis that is linear within $\pm0.001$. The contrast is close to proportional to dose at every setting, and the $R_m$ ordering reverses with background: with no background the $R_m=15{,}000$ contrast stays near zero (slightly negative at 0.05 and 0.5~nS, an interval spanning zero at 5~nS), whereas at background multiplier four it is the largest of the three at every dose, with 8/8 cells positive. \textbf{B}, Shunt and matched-injection localization (grey lines pair cells) and the true relation against a reassigned foreign template (unpaired); positive favors the shunt or the true relation. \textbf{C}, Transport selectivity $S_k$ against full synaptic-gradient localization at unit input-conductance-normalized dose, 101 sites; descriptive site--regime points, $R_m$ colours as in \textbf{A}, background multipliers 0, 1 and 4 as filled, open and plus markers; dotted line, $S_k=1$. \textbf{D}, Signed census of the same sites at $R_m=1{,}000\,\Omega\,\mathrm{cm}^2$, background multiplier one, unit dose: fraction of descendant gradients attenuated, enhanced or sign reversed by shunt (attenuates only) and matched injection (enhances only), as in all 81 passive conditions. \textbf{E}, All mapped versus directly typed contacts (grey lines pair cells); the contrast increases in 8/8 cells. \textbf{F}, E/I scale and inhibitory-reversal sensitivity by row; open marker and band, the reference condition (E/I 0.35, reversal $-0.2$) and its interval; right column, cells with a positive contrast. \textbf{A,B,E,F}, means and 95\% cell-bootstrap intervals, eight cells.'''),
('shunt_replication','si_08_shunting',[('S10','FGH'),('S47','AB')],
 'Focal-shunt controls replicate structurally and under weak-channel linearization.',
 r'''\textbf{A}, Focal shunt versus baseline-current-matched injection in 45 disjoint same-mouse cells (235 sites). \textbf{B}, True versus reassigned descendant relation templates in forty cells (230 sites); the true-relation column repeats the focal-shunt values of \textbf{A} for the forty cells that have a reassigned-relation partner. \textbf{C}, Direct E/I label coverage and the number of selected focal sites per cell, capped at 16 (one L5ET cell had 47 eligible sites); cell classes are ordinal colours within this panel. \textbf{D,E}, Separate initial eight-cell weak-channel ensemble: localization versus dose (dose divided by local input conductance) and paired shunt-minus-current contrasts. A fixed steady-state Jacobian (solid) is compared with the passive model (dashed); the two means differ by at most 0.010 localization units for the shunt and 0.023 for current injection at every dose, so the passive curve lies under the active one. Shunting attenuates descendants whereas matched current injection enhances them. Sixty-four channel draws and sites are averaged within each cell; intervals bootstrap cells and are narrower than the marker except at dose 4, and the contrast is positive in 8/8 cells at every dose (Wilcoxon $p=0.0078$). Weak conductances test local linearization near the passive response and do not establish robustness to regenerative dynamics or channel kinetics.'''),
# ---------------------------------------------------------------- si_09 ----
('measured_transfer_geometry','si_09_measured',[('S22','A'),('M9','BDE'),('S56','C')],
 'The measured-response learning comparison is limited by mapped coverage and transfer geometry.',
 r'''\textbf{A}, The seven targets (one mouse; one selected scan per target): mapped functional presynaptic partners (filled circles), the manually curated subset (open circles) and the split-half repeat reliability of each partner's responses (odd/even repeats, Spearman $r$; small dots, 69 records) with the target median (bar). \textbf{B}, Four-route support for the scan selected by median mapped-input count (target 1, session 4, scan 10; nine mapped inputs); rows are input coordinates and columns selected routes. \textbf{C}, Response-prediction NMSE for exact learning, treeless ridge and restricted dictionaries, with random and shuffled surrogate rows; dots are seven target means, diamonds and bars their means and 95\% target-bootstrap intervals, and the dotted line marks ridge (0.803). \textbf{D}, Common-checkpoint update reconstruction by the unrestricted fixed transfer profile (0.976, interval 0.964--0.986, narrower than the marker), the restricted ancestry profile (0.232) and oracle trialwise ancestry amplitudes (0.244). \textbf{E}, Mean split-half (odd/even repeat) Spearman reliability in 1,000 independent calibration datasets versus the measured value for 125 partner records; the two records with negative measured reliability (orange squares) are assigned zero reliable variance, and the identity line is not fitted. $n$ is seven targets in \textbf{C,D} and 125 partner--scan records in \textbf{E}; intervals are 95\% target-bootstrap intervals in \textbf{C,D}; in \textbf{E} the Monte Carlo standard error over the 1,000 datasets (0.0020--0.0028) is narrower than the marker, and 5 of 125 simulated means differ from the measured value by more than 1.96 standard errors (largest 0.0069); \textbf{A} and \textbf{B} are descriptive. \textbf{B} is the support matrix of main Fig.~9E.'''),
('measured_predictor_controls','si_09_measured',[('S34','*')],
 'Linear baselines explain most measured-response prediction, and degrading observed inputs worsens performance.',
 r'''Thirteen scans retain the complete-tree analysis's 130 outer stimulus-identity splits. All preprocessing, reliability masks and ridge tuning use training identities only. \textbf{A}, Ordinary least-squares (OLS) and nested ridge-regression baselines against the archived exact compartment-error fit; the dotted line is the training-mean predictor (normalized MSE $=1$ by construction). Lower normalized MSE is better. Below, paired differences from nested ridge (dotted line, zero difference) for each comparator: OLS, ridge on untransformed inputs, the exact compartment-error fit, and the archived route-restricted fits with topology-matched (ancestry), random anatomical and site-shuffled routes, whose absolute errors are in Supplementary Fig.~\ref{fig:si_measured_transfer_geometry}C. The exact fit gives a small additional nonlinear benefit, mean $-0.0152$ (95\% interval $-0.0271$ to $-0.0042$), whereas topology-matched routes ($0.029$; $-0.013$ to $0.074$) are matched by random routes ($0.031$; $-0.009$ to $0.068$). \textbf{B}, Ridge prediction when retaining nested random fractions of observed presynaptic inputs; five masks per split are averaged before inference. Manual-only partners, right of the separator, are the same ridge model under a provenance restriction, not a random retention dose. Empty masks use the training mean and remain included. \textbf{C}, Added Gaussian predictor noise in both training and test data, in units of training input standard deviation (SD) after the quantile transform. \textbf{D}, Reliability filtering based only on training repeats; parentheses give mean retained partner counts. In \textbf{B--D} the dotted horizontal line marks the training-mean predictor (normalized MSE $=1$). Thin traces and small dots show seven target means after averaging splits within scan and scans within target. Larger symbols and bars are means and 95\% target-bootstrap intervals (20,000 draws). All targets come from one mouse. Only ridge is refitted under the sensitivity conditions. Removing observed inputs does not recover unknown partners, establish population power or explain away the absence of preferential ancestry alignment.'''),
('animal_credit_reanalysis','si_09_measured',[('S17','*')],
 'Published animal data are consistent with signed neuron-specific teaching coordinates.',
 r'''This reanalysis contains no reconstructed dendritic morphology and measures no within-tree learning signal; it tests only the sign of two neuron-level coordinates. \textbf{A}, In the source brain--computer-interface (BCI) experiment, P+ and P- populations had positive and negative causal decoder weights, predicting opposite dendritic residual contrasts. \textbf{B}, Descriptive neuron-level residual distributions from published source sheet ``5E'' for error-increasing and error-reducing trials, in the source z-score normalization; dots show means, and the neuron-level standard errors (0.015--0.024 z) are smaller than the marker. The axis is clipped to $-1.7$ to $1.5$ z; the two neurons beyond it are counted at the arrow-capped ticks. These are descriptive and are not animal-bootstrap intervals. \textbf{C}, The six paired animal contrasts (reduction minus increase; lines join the two populations of one animal), with means and animal-bootstrap 95\% intervals from Source Data \texttt{summary.json}: P+ 0.075 (0.030--0.121), five of six animals positive; P$-$ $-0.150$ ($-0.205$ to $-0.087$), six of six negative. The mode-energy decomposition uses the same $n=6$ animals and is reported in Table~\ref{tab:animal_credit}; animals are the inferential units for those tests. This retrospective external comparison is consistent with signed neuron-specific information but does not resolve within-tree transport or uniquely identify a conductance mechanism.'''),
# ---------------------------------------------------------------- si_10 ----
('original_selector','si_10_selection_statistics',[('S35','*')],
 'The original scalar initialization score predicts a first step but fails at prospective endpoint selection.',
 r'''\textbf{A}, Four balanced leaf assignments (balanced 1--4; the number under each leaf is the input index at that leaf slot) and one comb, each at route budgets $K=1,2,4,8$: twenty candidates with prescribed cost. Filled nodes mark the $K$ feedback channels, one subtree each. \textbf{B}, Development, independent calibration, sealed choices and confirmation training. \textbf{C}, Final test half-mean-squared-error (MSE) plus cost, expressed as excess over the best trained candidate, for the feedback-only (green) and joint passive-transfer (blue) arms; the dashed line marks zero regret. Small points are the twenty whole-seed means; open diamonds and whiskers are means and 95\% seed-bootstrap intervals (10,000 draws), which are narrower than the diamond in the rank-only, fixed / max-budget and random-expectation rows. The fixed / max-budget baseline (the development choice, which is also the maximum budget) and the privileged rank-only baseline outperform the moment selector; random expectation averages all candidates uniformly. \textbf{D}, Route budget of the candidate chosen by the moment selector versus imposed rank; small points are the twenty seed-block means, open circles their mean, and the gray dashed line the retrospectively best candidate's budget, which equals the rank in every task of both arms. \textbf{E}, Mean within-task Spearman correlation of initialization utility with the actual first-step test-loss decrease and with the final test-loss ranking; small points are the twenty seed-block means, open circles and whiskers their mean and 95\% interval resampling whole seeds ($n=20$ seed blocks). Colours in \textbf{D,E} as in \textbf{C}. All 12,800 confirmation candidate fits and 38,400 checkpoints remain included. Every model has a freely trained dense weight matrix; passive transfer changes conditioning and cost but does not necessarily restrict representation. This experiment is distinct from the locally connected multi-affine construction and the separately frozen finite-horizon follow-up.'''),
('finite_horizon','si_10_selection_statistics',[('S38','ABCD'),('S39','AC')],
 'Finite-horizon prediction improves selection but strong simple baselines remain.',
 r'''\textbf{A,B}, Final costed-test-loss regret pooled over generating ranks in the feedback-only and joint-transfer fixed-cache learners; regret is excess above the best trained candidate. The top three rows are the baselines of Supplementary Fig.~\ref{fig:si_original_selector}C,D (moment selector there, original scalar here) on independent fresh seeds. \textbf{C}, Differences among strong selectors at each rank in the joint arm. The independently calibrated Gaussian SGD forecast improves over the original scalar utility; the full-batch and observed-context-count baselines remain competitive. All finite-horizon policies choose $K=r$ in this family, so improvement does not establish fine morphology selection. \textbf{D}, Median elapsed time on one central processing unit per twenty-candidate decision; `empirical full-batch' is the split-calibration forecast and `all 256-update fits' the cost of training every candidate. Timings include calibration-model preparation but not shared calibration generation. \textbf{E}, Original scalar forecast versus final test half-mean-squared-error in the feedback-only arm. \textbf{F}, The calibration-derived Gaussian SGD forecast on the same cases and arm. Hexagons bin the 6,400 fits per panel on the logarithmic colour bar beside \textbf{F}, shared by \textbf{E,F}; the dashed red line marks equality. Source Data retain the joint arm, which reproduces \textbf{E,F}, and the population-oracle forecast bias (exact only under fresh examples). Points and whiskers in \textbf{A--C} are means and pointwise 95\% intervals from 10,000 whole-seed bootstrap draws over twenty seed blocks, not visible where narrower than the marker; \textbf{D} shows medians only, of 320--1,280 timing draws per method. All 25,600 candidate fits and both regimes are retained; these are linear-learning forecast tests, not a general dendritic morphology law.'''),
('morphology_estimation','si_10_selection_statistics',[('S41','*')],
 'Finite calibration, adaptive construction and direct gradient learning.',
 r'''\textbf{A}, Sealed primary calibration: 256 labels at noise standard deviation (SD) 0.5, 192 for estimator or pilot fitting and 64 for the pilot gate. \textbf{B}, Prespecified pooled pilot-minus-estimated (upper) and fixed-baseline-minus-estimated (lower) normalized mean squared error (NMSE) differences. Whiskers are Bonferroni-adjusted 97.5\% paired bootstrap intervals; the dotted line marks the 0.01-NMSE improvement margin, not zero, which lies off the drawn axis. \textbf{C}, Excess clean-test NMSE above the best trained candidate across calibration sizes; solid curves use noise SD 0.5 and dashed curves no noise; no intervals are drawn; the estimated-cut noise contrast lies within the seed interval (at 1,024 labels, 0.022 [0.010, 0.035] noiseless against 0.033 [0.015, 0.056] at SD 0.5; 95\% seed-bootstrap intervals in brackets). \textbf{D}, Family-specific primary regret for estimated interactions, the development-fixed/estimated-rank baseline and the pilot; the quartic estimated-cut interval (0.0027 [0.0014, 0.0046]) is narrower than the line width. \textbf{E}, Secondary adaptive trees fitted by alternating least squares (ALS) against the best trained twelve-candidate menu and a true-target construction oracle. \textbf{F}, Elapsed time (median and interquartile range; medians 0.046 s and 0.066 s, 1.4-fold, on an axis that excludes zero) on one CPU for eighty tasks, before final training. \textbf{G,H}, Independent twenty-seed Adam cohort. \textbf{G}, Three structures and two credit rules in four families: filled symbols, exact credit (Adam 0.01); open, root broadcast (0.003). Label-noise floor: 0.0225; \textbf{E} uses clean targets. \textbf{H}, The fixed-minus-estimated (exact credit) and broadcast-minus-exact (estimated trees) rows use Bonferroni-adjusted 97.5\% intervals; the estimated-minus-target-informed row (exact credit) a pointwise 95\% interval. Target-informed construction is not a performance ceiling. The pooled broadcast-minus-exact contrast hides a family reversal (\textbf{G}): $-0.004$ for matching against 0.52--0.97 elsewhere. Whiskers in \textbf{D,E,G}: pointwise 95\% intervals, 10,000 whole-seed bootstrap draws ($n=20$). Both cohorts are independent of Supplementary Fig.~\ref{fig:si_oracle_profile_credit}. Colours on this sheet: green, the estimated tree; grey, the fixed/estimated-rank baseline (\textbf{B--D}), best trained menu (\textbf{E}), development-fixed tree (\textbf{G}) and fixed-minus-estimated row (\textbf{H}); orange, the two-sweep pilot (\textbf{B--D}) and, in \textbf{H} only, the broadcast-minus-exact row, which involves no pilot; purple, the target-informed construction (\textbf{E}, \textbf{G} and the estimated-minus-target-informed row of \textbf{H}), absent from the top key.'''),
]

# Sentences appended to a caption that is otherwise the frozen original.
CAPTION_APPEND = {
 'mechanistic_chain': r'This chain is the figure support for Supplementary Section~S1; the exact-path field it feeds is drawn at the main-text type scale in main Fig.~1.',
 'animal_credit_reanalysis': r'This reanalysis is restored as a figure so that the six-animal signed-credit comparison described in Methods has drawn support in \textbf{C}; the animal-level contrasts also remain tabulated.',
}

# Curated labels that must survive a merge as aliases (SI_NUMBERING 3.3).
ALIAS_EXTRA = {
 'utility_signal_noise': ['fig:si_checkpoint_geometry'],
 'measured_transfer_geometry': ['fig:si_measured_topology'],
 # Native renders N14, N15 and N18 replaced the frozen sheets S43, S44 and S42
 # (2026-09-12); the frozen sheets' labels stay as aliases of the same figures.
 'boolean_capacity': ['fig:supp_boolean_theory'],
 'boolean_learning': ['fig:supp_boolean_learning'],
 'conductance_grouping': ['fig:supp_morphology_conductance'],
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
 'S36': [[0,0,518.4,16]],
 'S37': [[0,0,518.4,18]],
 'S38': [[0,0,518.4,16]],
 'S49': [[107,484,514,504]],
}
SHARED_LEGENDS = {
 'ancestry_coefficients': [('S32',[120,0,440,36])],
 'mnist_dictionary_geometry': [('S49',[117.9,484,433.2,504])],
 # DECISIONS Q6: old S37D is registered as a legend strip, not a panel. Only
 # the three-family key is kept; its prose is already in the caption.
 'scalar_tree_capacity': [('S37',[312.9,248.6,394.8,304.9])],
}
WHOLE_CROPS = {'local_gate_controls': [0,0,518.4,452]}

# Audited decorated-panel bounds. Adjacent panels sometimes share margins;
# their axis fragments must not be imported with the selected panel.
PANEL_BOUNDS = {
 ('S5','E'):[180.2,157.5,346.95,294.7],
 ('S54','B'):[264.8,15.8,508.0,158.0], ('S54','D'):[264.8,173.2,508.0,315.4],
 ('S54','F'):[264.8,330.6,508.0,478.6],
 ('S51','B'):[276.5,12.3,506.3,203.1],
 ('S31','D'):[242,134.5,516,278], ('S31','G'):[327,283,501,408],
 ('S27','B'):[266,10.2,515,188.6], ('S27','D'):[266,208.7,515,380.4],
 ('S12','H'):[135.8,334.8,288.5,466.9], ('S10','F'):[340.6,163.4,504.8,284.2],
 ('M9','D'):[4,285,263,456],
}
PANEL_REDACTIONS = {
 ('S31','D'):[[240,129,517,134.3]],
 ('S31','G'):[[315,300,343,378]],
 ('M9','D'):[[219,415,230,446]],
}
# Reuse the actual shared-axis ticks and their original typography. Only the
# labels are copied; the plotted observations and axes remain those of panel G.
PANEL_PATCHES = {
 ('S31','G'):[{'source':'S31','bbox':[184,301,205,387],
              'target_bbox':[340.4,301,361.4,387], 'role':'shared y tick labels'}],
}
PANEL_TEXT = {('S31','G'):[{'text':'Serial minus grouped point (pp)',
 'origin':[336,385], 'fontsize':8.0, 'rotate':90, 'role':'shared y-axis label'}]}
REMOVED_SHARED_REGIONS['S27']=[[178.5,188.6,391.5,206.7]]
SHARED_LEGENDS['anatomy_capacity_controls']=[('S27',[178.5,188.6,391.5,206.7])]
PANEL_BOUNDS[('S31','B')]=[310,4.5,516,136]
PANEL_BOUNDS[('S20','B')]=[265.8,14.2,514.8,201.3]
REMOVED_SHARED_REGIONS['S19']=[[217.9,232.3,374.1,241]]
PANEL_BOUNDS[('M9','E')]=[275.5,283.5,502,455]
PANEL_REDACTIONS.pop(('M9','D'),None)
PANEL_BOUNDS[('S31','G')]=[327,283,516,408]
PANEL_TEXT[('S31','G')][0]['text']='Serial minus grouped (pp)'
NATIVE_LEGENDS={}
PANEL_BOUNDS[('S10','G')]=[19.1,307.1,185.2,427.9]
PANEL_BOUNDS[('S10','H')]=[181.5,307.1,342.0,430.1]
PANEL_BOUNDS[('S52','A')]=[24.9,12.8,259.0,200.1]
PANEL_REDACTIONS[('S31','G')].append([343,282,518,296])
PANEL_TEXT[('S31','G')]=[{'text':'Exact LocalCA: serial advantage (pp)',
 'origin':[329,292], 'fontsize':8.0, 'role':'quantity and optimizer in panel title'}]
PANEL_BOUNDS[('S31','G')]=[342,283,516,408]
PANEL_TEXT[('S31','G')][0]['origin']=[344,292]

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
# Old S46 A, C, D, E (sheet S3): letter-excluded tops (title tops 14.2, 165.2,
# 165.2 and 336.2), which also takes 10.8 pt off the sheet so its caption fits.
PANEL_BOUNDS[('S46','A')]=[15.0,12.2,262.4,122.8]
PANEL_BOUNDS[('S46','C')]=[15.0,163.2,261.8,310.5]
PANEL_BOUNDS[('S46','D')]=[261.7,163.2,494.3,313.4]
PANEL_BOUNDS[('S46','E')]=[15.0,334.2,261.8,481.3]
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
PANEL_REDACTIONS[('S10','G')]=[[183.5,316.0,193.0,418.6]]   # H's y label
PANEL_REDACTIONS[('S10','H')]=[[155.0,419.7,183.6,427.0]]   # G's 'relation'
# Old S47 (frozen): padded on the inner and right blank margins so the S30
# rows share one left edge (row 1 x 7.8, row 2 x 7.9); ink is 15.8-231.1 and
# 285.0-497.5.
PANEL_BOUNDS[('S47','A')]=[13.8,0.0,260.0,177.5]
PANEL_BOUNDS[('S47','B')]=[270.0,0.0,518.4,177.5]
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
PANEL_BOUNDS[('S12','D')]=[138.0,175.7,249.1,325.3]
PANEL_BOUNDS[('S12','E')]=[270.3,175.7,380.8,325.3]
PANEL_BOUNDS[('S12','F')]=[402.0,175.7,512.5,325.3]
PANEL_BOUNDS[('S12','G')]=[4.9,334.8,122.7,476.7]
# Old S18: every axes box is 147.7 x 107 pt (x 84-231.7 and 339.7-487.4) and
# the auto boxes cut the axes right edges (A/C/E at 223-228, D at 476.7) or
# ran to the page edge (F); A and B take title-baseline tops (22.4-9 = 13.4)
# so their titles sit 9 pt below the crop top like the re-set title of old
# S31 G in the same S22 row; C/D likewise (179.4-9).  Old S18 F is padded
# right (blank page margin) so the S24 rows share one left edge (x 9.5).
PANEL_BOUNDS[('S18','A')]=[2.0,13.4,234.1,158.6]
PANEL_BOUNDS[('S18','B')]=[257.7,13.4,489.8,151.6]
PANEL_BOUNDS[('S18','C')]=[2.0,170.4,234.1,312.2]
PANEL_BOUNDS[('S18','D')]=[257.7,170.4,489.9,315.7]
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
PANEL_TEXT[('S31','G')][0]['origin']=[357.5,292]
PANEL_REDACTIONS[('S31','D')].append([474.0,150.9,505.6,153.4])
PANEL_TEXT[('S31','D')]=[{'text':'accuracy','origin':[516.5,220.7],'fontsize':6.8,'rotate':90,'role':'colourbar title, re-set beside the bar'}]
PANEL_BOUNDS[('S31','D')]=[242,134.5,518.4,278]
# Old S11 and S21 (sheet S29): the auto boxes cut the grid and zero lines that
# run to the axes right edges (S11 B 384.8, S11 C 508.7, S21 A 249.0, S21 D
# 496.7).
PANEL_BOUNDS[('S11','B')]=[225.4,5.8,386.8,178.1]
PANEL_BOUNDS[('S11','C')]=[390.5,5.8,510.7,176.0]
PANEL_BOUNDS[('S21','A')]=[32.0,4.8,251.0,171.8]
PANEL_BOUNDS[('S21','D')]=[321.0,201.3,498.7,361.6]
# Old M9 (frozen, sheet S31): B without the 77 pt blank left margin and D from
# its 'Shuffled' label; E to its x-axis spine (504.4).  Old S56 C: the in-plot
# sentence '125 partner records; ...' (text box 59.9-294.6 x 240.6-248.5) is
# removed; nothing else touches the box.
PANEL_BOUNDS[('M9','B')]=[80.7,126.6,206.2,264.4]
PANEL_BOUNDS[('M9','D')]=[33.2,283.5,262.2,454.6]
PANEL_BOUNDS[('M9','E')]=[275.5,283.5,506.4,455]
PANEL_REDACTIONS[('S56','C')]=[[58.5,239.5,296.0,249.5]]
# Old S25 (sheet S27, pasted whole): provenance rects to the axes right edges
# (B 510.6, C 239.9, D 510.6), which make_bounds misses.
PANEL_BOUNDS[('S25','B')]=[293.5,16.2,512.6,161.4]
PANEL_BOUNDS[('S25','C')]=[18.8,180.0,241.9,308.0]
PANEL_BOUNDS[('S25','D')]=[288.9,180.0,512.6,308.4]

# 2026-09-12: paste layer for the fourteen sheets that rest on frozen renders
# with no generator (analysis/figure_visual_review_20260910/frozen_drafts.json).
# Seven of them left the whole-sheet branch (old S40, S43, S44, S54, S55, S42
# and S50 pasted by letter) so that crops, redactions and re-set labels
# apply; S43, S44 and S42 have since been replaced by the native renders N14,
# N15 and N18, pasted whole, and their entries are gone (2026-09-12).  Every
# box is measured on the render in source points.  Replaced in place above:
# old S54 B/D/F and S51 B bounds, the S49 and S37 shared-legend boxes (ink
# union + 2 pt, so build.py's centring lands on the ink) and the old S52 A
# bounds; the stale S52 A redaction box (it sat on D's x spine) is deleted.

# S8 error_field_geometry
PANEL_BOUNDS[('S45','C')]=[149.7,91.2,232.0,233.1]
PANEL_BOUNDS[('S45','D')]=[270.3,91.2,382.5,233.1]
PANEL_BOUNDS[('S45','E')]=[400.4,91.2,507.4,233.1]
PANEL_REDACTIONS[('S45','C')]=[[183.0,114.6,254.5,126.5],[0,0,518.4,90.9],[0,233.4,518.4,428],[0,90.9,149.4,233.4],[232.3,90.9,518.4,233.4]]
PANEL_REDACTIONS[('S45','D')]=[[304.4,126.9,361.5,135.55],[304.4,135.3,343.0,143.9],[302.0,144.3,315.1,152.7],[0,0,518.4,90.9],[0,233.4,518.4,428],[0,90.9,270.0,233.4],[382.8,90.9,518.4,233.4]]
PANEL_REDACTIONS[('S45','E')]=[[419.9,114.6,514.4,126.4],[0,0,518.4,90.9],[0,233.4,518.4,428],[0,90.9,400.1,233.4],[507.7,90.9,518.4,233.4]]
PANEL_TEXT[('S45','D')]=[{'text':'def.','origin':[313.6,152.6],'fontsize':7.0,'rotate':0,'role':'definitional-marker label re-set beside the soma diamond'}]
# S7 mnist_dictionary_geometry
PANEL_BOUNDS[('S49','C')]=[9.06,180.08,247.82,322.01]
PANEL_BOUNDS[('S49','D')]=[273.06,180.08,511.82,322.01]
PANEL_BOUNDS[('S49','E')]=[9.06,344.0,247.82,484.0]
PANEL_BOUNDS[('S49','F')]=[273.06,344.0,511.82,484.0]
PANEL_REDACTIONS[('S49','F')]=[[288.6,356.7,300.3,469.5]]
PANEL_TEXT[('S49','F')]=[{'text':'Mean activation-error capture (fraction)','origin':[297.4,482.4],'fontsize':8.0,'rotate':90,'role':'y-axis label re-set to the plotted quantity, as in main Fig. 1G'}]
# S12 scalar_tree_capacity
PANEL_BOUNDS[('S36','A')]=[49.8,20.5,228.3,132.7]
PANEL_BOUNDS[('S36','B')]=[288.8,10.3,508.4,183.7]
PANEL_BOUNDS[('S36','D')]=[312.6,213.1,504.9,384.0]
PANEL_BOUNDS[('S37','A')]=[49.8,26.5,231.0,139.2]
PANEL_BOUNDS[('S37','B')]=[312.9,26.5,494.1,160.9]
PANEL_BOUNDS[('S37','C')]=[13.8,202.4,241.8,362.3]
PANEL_REDACTIONS[('S37','C')]=[[118.5,269.9,201.6,283.4],[118.5,284.4,201.6,287.6]]
PANEL_REDACTIONS[('S37','B')]=[[386.9,38.9,431.2,46.9]]
# Old S37 B: the orange 'root readout' label re-centred on the root marker (see the
# redaction above); a patch keeps its own colour and face.
PANEL_PATCHES[('S37','B')]=[{'source':'S37','bbox':[386.9,38.9,431.2,46.9],'target_bbox':[415.4,38.9,459.7,46.9],'role':'root readout label re-centred on the root node'}]
# S13 oracle_profile_credit
REMOVED_SHARED_REGIONS['S40']=[[0,0,518.4,28.0]]
SHARED_LEGENDS['oracle_profile_credit']=[('S40',[150.0,4.0,400.0,27.0])]
PANEL_BOUNDS[('S40','A')]=[18.1,45.4,233.6,171.1]
PANEL_BOUNDS[('S40','B')]=[282.8,45.4,491.1,171.1]
PANEL_REDACTIONS[('S40','E')]=[[51.8,448.2,68.9,456.7],[142.5,448.2,155.3,456.7],[223.4,448.2,236.2,456.7]]
PANEL_REDACTIONS[('S40','F')]=[[316.4,448.2,333.6,456.7],[407.2,448.2,420.0,456.7],[488.0,448.2,500.8,456.7]]
PANEL_TEXT[('S40','E')]=[{'text':'0.003','origin':[51.55,455.0],'fontsize':7.0,'role':'x tick label with leading zero'},{'text':'0.01','origin':[142.09,455.0],'fontsize':7.0,'role':'x tick label with leading zero'},{'text':'0.03','origin':[222.93,455.0],'fontsize':7.0,'role':'x tick label with leading zero'}]
PANEL_TEXT[('S40','F')]=[{'text':'0.003','origin':[316.19,455.0],'fontsize':7.0,'role':'x tick label with leading zero'},{'text':'0.01','origin':[406.73,455.0],'fontsize':7.0,'role':'x tick label with leading zero'},{'text':'0.03','origin':[487.57,455.0],'fontsize':7.0,'role':'x tick label with leading zero'}]
# S14 boolean_capacity and S15 boolean_learning: pasted whole from the native
# renders N14 and N15 (2026-09-12); the old S43/S44 crop, redaction and re-set
# label entries were removed with the frozen renders they were measured on.
# S16 fixed_profile_budget
PANEL_BOUNDS[('S54','A')]=[23.1,15.8,266.3,158.0]
PANEL_BOUNDS[('S54','C')]=[23.1,173.2,266.3,315.4]
PANEL_BOUNDS[('S54','E')]=[27.9,330.6,266.3,478.6]
PANEL_REDACTIONS[('S54','F')]=[[331.5,331.6,465.4,343.3]]
PANEL_TEXT[('S54','F')]=[{'text':'Quartic seeds: 1,024 vs 16,384 updates','origin':[327.3,340.8],'fontsize':8.0,'role':'descriptive title replacing the assertion title'},{'text':'n = 20 seeds per rule','origin':[405.0,425.0],'fontsize':7.0,'role':'seed count printed inside the empty band'}]
PANEL_TEXT[('S54','E')]=[{'text':'8,192','origin':[191.5,464.7],'fontsize':8.0,'role':'tick label for the unlabelled fourth budget column'}]
# S17 credit_optimizer_controls
PANEL_BOUNDS[('S55','A')]=[23.0,15.5,265.0,182.2]
PANEL_BOUNDS[('S55','B')]=[265.5,15.5,508.0,182.2]
PANEL_BOUNDS[('S55','C')]=[23.0,197.0,265.0,363.7]
PANEL_BOUNDS[('S55','D')]=[265.5,197.0,508.0,363.7]
PANEL_TEXT[('S55','A')]=[{'text':'floor 0.023','origin':[61.0,149.5],'fontsize':7.0,'rotate':0,'role':'noise-floor value'}]
PANEL_TEXT[('S55','B')]=[{'text':'floor 0.023','origin':[303.0,149.5],'fontsize':7.0,'rotate':0,'role':'noise-floor value'}]
PANEL_TEXT[('S55','C')]=[{'text':'floor 0.046','origin':[61.0,312.1],'fontsize':7.0,'rotate':0,'role':'noise-floor value'}]
PANEL_TEXT[('S55','D')]=[{'text':'floor 0.046','origin':[303.0,321.6],'fontsize':7.0,'rotate':0,'role':'noise-floor value'}]
# S18 conductance_grouping: pasted whole from the native render N18
# (2026-09-12); the old S42 crop, redaction, re-set label and shared-legend
# entries were removed with the frozen render they were measured on.
# S19 conductance_precision
PANEL_BOUNDS[('S50','A')]=[27.9,10.8,270.05,200.61]
PANEL_BOUNDS[('S50','B')]=[273.0,10.8,510.86,200.61]
PANEL_BOUNDS[('S50','C')]=[27.9,217.8,270.05,407.61]
PANEL_BOUNDS[('S50','D')]=[273.0,217.8,510.86,407.61]
PANEL_REDACTIONS[('S50','C')]=[[171.4,230.9,256.2,249.9],[87.4,218.8,236.2,230.6]]
PANEL_REDACTIONS[('S50','D')]=[[320.2,218.8,489.3,230.6]]
PANEL_TEXT[('S50','C')]=[{'text':'Gated-task endpoint gaps','origin':[109.5,228.0],'fontsize':8.0,'role':'descriptive panel title replacing the redacted claim title'}]
PANEL_TEXT[('S50','D')]=[{'text':'Path-field capture at exact-rule states','origin':[328.3,228.0],'fontsize':8.0,'role':'descriptive panel title replacing the redacted claim title'},{'text':'= 1 by construction','origin':[347.5,241.8],'fontsize':7.0,'role':'value note beside the ungated rank-one path-capture mark'}]
# S20 conductance_optimization
# The 1 pt strip through 'Log-conductance bounds' is split around the two
# gridlines at x 434.64 and 472.22 (0.55 pt wide) so they stay unbroken; the
# narrowest glyph ('-', 2.6 pt) still touches a strip.
PANEL_REDACTIONS[('S51','B')]=[[404.8,30.5,434.2,31.5],[435.1,30.5,471.8,31.5],[472.7,30.5,497.2,31.5]]
PANEL_BOUNDS[('S51','C')]=[32.4,223.3,266.5,414.6]
PANEL_BOUNDS[('S51','D')]=[285.5,223.3,518.4,414.6]
PANEL_BOUNDS[('S52','B')]=[262.4,12.8,500.4,200.1]
PANEL_TEXT[('S51','D')]=[{'text':'4,096-update budget','origin':[326.0,258.0],'fontsize':7.0,'rotate':0,'role':'development budget of this panel'}]
PANEL_TEXT[('S52','A')]=[{'text':'16,384-update budget','origin':[63.0,47.0],'fontsize':7.0,'rotate':0,'role':'development budget of this panel'}]
PANEL_TEXT[('S52','B')]=[{'text':'16,384-update budget','origin':[304.5,47.0],'fontsize':7.0,'rotate':0,'role':'development budget of this panel'}]
# S35 finite_horizon
PANEL_BOUNDS[('S38','A')]=[16.30,24.46,247.06,226.53]
PANEL_BOUNDS[('S38','B')]=[289.55,24.46,518.40,226.53]
PANEL_BOUNDS[('S38','C')]=[16.30,244.31,247.06,425.79]
PANEL_BOUNDS[('S38','D')]=[289.55,244.31,518.40,427.02]
PANEL_BOUNDS[('S39','A')]=[21.48,18.99,252.24,161.39]
PANEL_BOUNDS[('S39','C')]=[29.32,174.96,258.17,317.50]
PANEL_REDACTIONS[('S39','A')]=[[56.0,19.5,153.0,32.5]]
PANEL_REDACTIONS[('S39','C')]=[[56.0,175.5,185.0,188.5]]
PANEL_TEXT[('S39','A')]=[{'text':'Feedback only: original scalar forecast','origin':[57.02,29.2],'fontsize':8.0,'role':'panel title re-set with the series name of A--D'}]
PANEL_TEXT[('S39','C')]=[{'text':'Feedback only: Gaussian SGD forecast','origin':[57.02,185.1],'fontsize':8.0,'role':'panel title re-set with the series name of A--D'},{'text':'count','origin':[229.4,184.6],'fontsize':7.0,'role':'colour-bar title re-set above the re-placed bar (source title at y 95-103 lies outside the patch)'}]
# Old S39: the shared hexbin colour bar (image at 494-501 x 108-230 of the render)
# sat outside every crop; re-placed beside F, inside the widened F bounds.
PANEL_PATCHES[('S39','C')]=[{'source':'S39','bbox':[493.0,106.0,517.0,236.0],'target_bbox':[233.5,187.5,257.5,317.5],'role':'shared hexbin colour bar of the source sheet, re-placed beside F'}]

# This source ledger originally named only the rendered asset. The scientific
# builder explicitly reads these two tables; preserve that numerical closure.
EXPLICIT_NUMERICAL_SOURCES = {
 ('S25','B'):['source_data/irregular_tree_wavelets/cohort_scale_summary.csv'],
 ('S25','C'):['source_data/irregular_tree_wavelets/cell_scale_summary.csv',
              'source_data/irregular_tree_wavelets/cohort_scale_summary.csv'],
 ('S25','D'):['source_data/irregular_tree_wavelets/cohort_scale_summary.csv'],
 ('X1','*'):['source_data/prospective_input_validity/mechanism_checkpoint_rows_valid.csv'],
}

# Per-figure Source Data directories (SI_PLAN 5.6), by new S-number index.
SOURCE_DATA_DIRS = {
 'mechanistic_chain':['inherited_neurips','theory'],
 'credit_validation':['figure2'],
 'utility_signal_noise':['credit_phase_theory','review_evidence_reanalysis','alignment_controlled','prospective_input_validity'],
 'reliability_gain':['positive_conductance_reliability_step_consistent'],
 'same_span_conditioning':['same_span_coefficient_learning'],
 'image_generalization':['cifar10_additive_feedback_ladder_confirmatory','fashion_feedback_ladder','mnist_between_within_factorial','regular_tree_regimes'],
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
 'physical_architecture':['physical_alignment_dose','remaining_physical_experiments','nonlinear_physical_depth_confirmatory','point_dendrite_credit_controls','task_family_alignment'],
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
 'S1':'Restored in full as Supplementary Fig. S1.',
 'S43':'Replaced by the native render N14 (scripts/build_supplementary_figure_boolean_capacity_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S14; no crop of this frozen render is used.',
 'S44':'Replaced by the native render N15 (scripts/build_supplementary_figure_boolean_learning_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S15; no crop of this frozen render is used.',
 'S42':'Replaced by the native render N18 (scripts/build_supplementary_figure_conductance_grouping_native.py), drawn from the same source tables and pasted whole as Supplementary Fig. S18; no crop of this frozen render is used.',
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
                   reason='Promoted to main Fig. 5G and re-rendered natively there (DECISIONS Q7); it leaves the supplement, and the curated crop never contained it.',
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
                   reason='Not printed as a panel: SI_NUMBERING keeps it as an inset of S35F, which is not constructible from a crop at paste scale 1.0; the result is stated in the S35 caption and retained in Source Data.',
                   numbers_at='source_data/morphology_finite_horizon/'),
 ('S37','D'): dict(content='Counts, minimum exact depths, reconstruction error and coefficient magnitude of the 105 matching, 35 quartic and 24 nested targets.',
                   reason='Registered as the shared legend strip of S12 (DECISIONS Q6): the three-family key is printed below the panels and its prose is in the caption.',
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
 ('S11','A'): dict(content='Shunt-minus-current-injection localization across fixed absolute doses at three membrane resistances.',
                   reason='RESTORED as S29A.', numbers_at='source_data/focal_selectivity_phase1/'),
 ('S12','A'): dict(content='Schematic of the inhibitory-route matching design.',
                   reason='Omitted; the matched-null design is described in Supplementary Section S7.',
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
                  reason='Omitted; the same construction is drawn at the main-text type scale in main Fig. 9.',
                  numbers_at='source_data/credit_first_figures/'),
 ('S40','C'): dict(content='Paired exact-credit NMSE increase after shuffling leaf input assignments at fixed tree shape and parameter count.',
                   reason='Copy kept: printed here as S13C and reproduced at the main-text type scale as Fig. 4H.',
                   numbers_at='source_data/morphology_credit/'),
}

# Figures whose frozen panel inventory cannot be pasted at PASTE_SCALE = 1.0
# inside the 540 pt sheet cap (SI_PLAN 5.1/5.2, DECISIONS Q5).  Each entry is
# a work queue item, not a waiver: it names the native re-render or legacy
# port that removes the exemption, and the reason the crop cannot.  build.py
# asserts that every below-target figure appears here and writes the table
# into configs/supplement_consolidation/audit_report.json.
SCALE_EXEMPTIONS = {
 'scalar_tree_capacity': dict(
   figure='S12', native_order='legacy port',
   builder='scripts/build_morphology_followup_figures.py',
   reason=('Six frozen panels plus the Q6 legend strip need 573.7 pt of panel band at 1.0; '
           'the widest row (old S36A,B,D) is 606.1 pt against 498.4 pt of usable width. '
           'The R6/R8 restorations are cleaner re-rendered than cropped.')),
 'conductance_optimization': dict(
   figure='S20', native_order='N4',
   builder=('scripts/conductance_credit_demand/build_supplementary.py + '
            'scripts/conductance_credit_demand/report_expanded_rates.py'),
   reason=('Five panels of about 196 pt each need three rows: 587.1 pt of panel band at 1.0 '
           'against 471 pt available under the cap. Both sources are native.')),
 'physical_architecture': dict(
   figure='S22', native_order='N6 (mandatory, SI_NUMBERING S22 height note)',
   builder=('scripts/build_supplementary_figures_s17_s20_native.py + '
            'scripts/build_main_figure_06.py'),
   reason=('AMENDMENTS B6 keeps eight panels (three copies kept plus R16, R17); four rows of '
           'about 149 pt need 595 pt of panel band at 1.0 against 448 pt available.')),
 'shunt_sensitivity': dict(
   figure='S29', native_order='N7',
   builder=('scripts/build_supplementary_figure_s21_native.py + '
            'scripts/build_focal_selectivity_figure.py (legacy port for old S11)'),
   reason=('The frozen six-panel order of AMENDMENTS B8 with Q3 needs 558.9 pt of panel band '
           'at 1.0. Old S21 is native; old S11A--C is legacy and must be ported.')),
 'measured_transfer_geometry': dict(
   figure='S31', native_order='N8',
   builder=('scripts/build_journal_figures.py + scripts/credit_first_figures/build_measured.py '
            '+ scripts/measured_alignment_power/report.py'),
   reason=('Closest to target: 562.8 pt at 1.0, 22.8 pt over the cap, because old S56C is a '
           '496.4 pt full-width panel that must hold its own row. All three sources are native.')),
 'finite_horizon': dict(
   figure='S35', native_order='legacy port',
   builder='scripts/build_morphology_followup_figures.py',
   reason=('Q4 fallback puts R10, R11 and R13 on this sheet; three rows of tall dot plots need '
           '546.2 pt of panel band at 1.0. This is also the port that would make the DECISIONS '
           'Q4 merge feasible.')),
}
