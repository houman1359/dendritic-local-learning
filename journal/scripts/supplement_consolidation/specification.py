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
 r'''\textbf{A,B}, Six-rule MNIST ladder in shunting and raw-additive networks; ten fresh paired seeds per architecture. Filled/open symbols use development-selected/original common rates. Decoder-only learning freezes the initialized core. \textbf{C}, Paired accuracy differences for three profiles versus a projected common profile, and exact versus three-profile credit. \textbf{D}, Initial and trained field-energy capture by uniform $K=1$ or subtree $K=3$ profiles, averaged over 2,048 held-out images within each seed. Projection coefficients use the exact activation-error field of this ten-seed cohort, so \textbf{D} reproduces the $K=1$ and $K=3$ points of main Fig.~1G for this cohort; voltage-space capture for the same fits is retained in Source Data. Means and bars in A--D have descriptive 95\% seed-bootstrap intervals. Greater capture did not yield an image accuracy benefit. All image cohorts have one dendritic layer and a linear readout.'''),
('error_field_geometry','si_02_credit_rules_images',[('S45','CDE')],
 'Error-field geometry separates direction, amplitude and spatial variation.',
 r'''Independent fifteen-seed flattened-MNIST cohort of 128 directed $[3,3]$ trees; green circles denote shunting and blue squares raw-additive trees. \textbf{A}, Branch-gradient cosine with the exact gradient at common fixed checkpoints trained with per-neuron feedback; matched-width scalar-fallback and per-neuron fields are evaluated at identical weights and states, and the exact-path cosine of one is definitional. \textbf{B}, Exact voltage-error magnitude by depth at exact-path checkpoints: the batch root-mean-square ratio $\operatorname{RMS}(\delta^V_{n,u})/\operatorname{RMS}(\delta^V_{0,u})$, averaged over neurons and paths, with somatic value one by normalization. \textbf{C}, Within-depth path-specific residual error energy after subtracting the mean across paths within depth $d$: distal shunting attenuates magnitude but retains the larger path-specific fraction, rising from 18\% to 53\% (shunting) and 32\% (raw additive). Symbols and bars are means and 95\% seed-bootstrap intervals. No additional models are trained here; greater path-specific variation did not yield an image accuracy benefit, and voltage- and activation-space capture are different quantities.'''),
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
 r'''\textbf{A}, Exhaustive pairwise matching and two-quartic target families on eight independent binary inputs; both have input-gradient second moment $I_8/4$. Each candidate has seven scalar multi-affine nodes, 28 coefficients and fourteen edges. \textbf{B}, Full rank-two and centered rank-one cut-tail lower bounds versus achieved population NMSE in four-restart alternating least squares. \textbf{C}, Mean excess NMSE above the best fitted candidate in the fixed pool of twelve; score ties use tolerance $10^{-10}$ and lexicographic candidate identifier, and best-fixed selection uses family outcomes in hindsight. \textbf{D}, Actual tree returned for the first enumerated matching, a depth-three parallel construction. \textbf{E}, First seeded nested-prefix control tree, of minimum depth four. Leaves access the indicated input only. \textbf{F}, Minimum possible maximum centered-cut bound over all labeled binary trees at each depth limit, calculated separately for every target: matching and quartic targets admit exact depth-three trees, whereas nested controls require depth four and do not share the $I_8/4$ input spectrum. The key below the panels gives the target counts and minimum exact depths of the three families. All 164 targets construct with error below $7\times10^{-30}$ and coefficients of magnitude at most one. These exhaustive capacity results have no sampling error bars. The theorem concerns scalar multi-affine composition, not arbitrary conductance dendrites.'''),
('oracle_profile_credit','si_04_interactions_boolean',[('S40','*')],None,None),
('boolean_capacity','si_04_interactions_boolean',[('S43','*')],None,None),
('boolean_learning','si_04_interactions_boolean',[('S44','*')],None,None),
('fixed_profile_budget','si_04_interactions_boolean',[('S54','*')],
 'Longer matched budgets retain the task-dependent fixed-profile deficit.',
 r'''\textbf{A,B}, Pairwise targets under Adam at the separately selected rates and at the common rate, with exact, unit-broadcast, initial-profile and initial-sign credit. \textbf{C,D}, Quartic targets at the same two rate choices. Initial-sign controls use dotted curves; vertical lines mark 1,024 updates and horizontal lines mark noise-only NMSE. \textbf{E}, Quartic-minus-pairwise difference in calibrated-minus-exact NMSE across four budgets: circles use terminal states and squares validation-selected states. \textbf{F}, Every exact and calibrated quartic seed at the original 1,024-update and extended 16,384-update endpoints; the diagonal marks unchanged error. Panel \textbf{F} resolves the three stalled seeds that main Fig.~4D draws as individual trajectories. Curves show means and descriptive 95\% whole-seed bootstrap intervals. All 720 trajectories retain the previously observed twenty seed blocks, three tasks, four rules, two optimizers and the union of the original selected and common rates. The complete original endpoints replay exactly. Slow exact quartic fits resolve with more updates; restricted profiles retain a large deficit. Rates remain fixed and coefficients bounded; this is a retrospective-cohort extension under a prospective protocol, not fresh confirmation.'''),
('credit_optimizer_controls','si_04_interactions_boolean',[('S55','*')],None,None),
# ---------------------------------------------------------------- si_05 ----
('conductance_grouping','si_05_conductance_learning',[('S42','*')],None,None),
('conductance_precision','si_05_conductance_learning',[('S50','*')],None,None),
('conductance_optimization','si_05_conductance_learning',[('S51','BCD'),('S52','AB')],
 'Parameter-range, development and rate controls retain the opposed-task broadcast deficit.',
 r'''\textbf{A}, Task-by-credit interaction under original $[-7,7]$ and wider $[-20,20]$ log-conductance bounds, with Adam and SGD. All fits retain their original initialization/minibatch streams and extend to 16,384 updates; endpoints use validation-selected checkpoints. Wider-bound interactions are positive in all seeds. \textbf{B}, Initial-profile-minus-exact NMSE at each seed's last saved checkpoint before that seed's calibrated-profile fit first reaches an original bound; all twenty differences are already positive. Dots are twenty paired seeds and intervals are 95\% paired bootstrap intervals in \textbf{A,B}. \textbf{C}, All four retained development regimes under Adam, using each rule's lowest mean validation NMSE among its three equally budgeted rates: aligned strong gating, opposed strong gating, opposed moderate gating (inactive-parent inhibitory activity one) and opposed ungated (activity zero). These are development summaries over three seeds, not additional confirmatory seed blocks. \textbf{D,E}, Equally budgeted six-rate development sweeps for opposed tuning under Adam and SGD; corresponding aligned-task outcomes are retained in the full condition table and in Source Data rather than drawn here. Each point averages three development seeds; all 288 outcomes are retained. The predefined improvement trigger was not reached, so rates were not retuned on fresh test outcomes. Some broadcast fits also reach wider bounds; these are finite-budget controls, not an asymptotic convergence proof.'''),
('local_gate_controls','si_05_conductance_learning',[('S53','ABCD')],
 'Local conductance gating is robust to rate choice but depends on where it is applied.',
 r'''\textbf{A,B}, Aligned tuning at 4,096 and 16,384 updates. \textbf{C,D}, Opposed tuning at the same budgets. Every rule receives every displayed Adam rate; 0.03 is primary. Entries are mean test NMSE at validation-selected checkpoints across twenty new paired seeds; the color scale shows base-ten logarithm. Unit and calibrated broadcast agree to two significant figures in every cell and are read as one broadcast row. Exact credit, the three-pattern oracle, the local distal gate and the shunt-proportional gate learn accurately. Swapping the gate fails; extending the gate to proximal compartments leaves a substantial opposed-task deficit. The two leaf patterns with unit proximal feedback isolate this placement effect. These are supplied-context, bounded-conductance experiments; the local gate does not estimate arbitrary route coefficients. Both budgets and all rates and seeds remain included; the historical cancellation cohort of the same study is drawn at the main-text type scale in main Fig.~5G.'''),
# ---------------------------------------------------------------- si_06 ----
('physical_architecture','si_06_physical_depth',[('S31','BCDG'),('S18','ABCD')],
 'Task-matched serial computation differs from grouped and flexible point controls.',
 r'''\textbf{A}, Serial D3, resource-identical grouped-point and approximately parameter-matched flexible point networks. Only serial and grouped models share modules and contacts. \textbf{B}, Original-budget three-level aligned task across D1--D3 and the five-arm feedback ladder. \textbf{C}, Four-level task including D4 under aligned or reversed sensor placement. \textbf{D}, Serial-minus-grouped accuracy at fixed D3 across nested, flat and local-ratio task families under exact-path LocalCA, the counterpart of the backpropagation contrast in main Fig.~6C; its 95\% intervals are also carried onto that panel. \textbf{E}, Serial-minus-star accuracy at D1--D3, where the star is the resource-identical all-active grouped control (Source Data \texttt{all\_active\_star}); the open marker is the derived aligned-minus-reversed difference of the D3 contrast, and the axis break removes the empty 9--28 pp span. \textbf{F}, Flexible point networks matched to active or to total parameter count, ten seeds per condition drawn (y axis 0.90--1.00); the serial D3 mark is the exact-BP run of \textbf{B}. \textbf{G}, Accuracy across task--sensor alignment $\alpha$ (depth colours ordinal within this panel). \textbf{H}, The D3-minus-D1 advantage at each $\alpha$ with the ten paired seeds behind each mean, on an axis broken between 4.5 and 28.5 pp; from $\alpha=0.25$ to 0.75 the advantage grows by 2.92 points (95\% interval 2.65--3.17) and its within-seed linear slope is 25.8 points per unit $\alpha$ (25.4--26.3). Intermediate doses were specified after the endpoint results and constitute interpolation. Curves and differences use ten paired seeds per cohort and 95\% paired-seed bootstrap intervals; in \textbf{B} and \textbf{G} most intervals are narrower than the marker. In \textbf{C} the raw-additive arm was run under aligned placement only, so its reversed row is empty, and an outlined cell marks the best depth of its row where that cell's 95\% interval clears the runner-up's; the reversed block is flat by design. Main Fig.~6A,D re-render the architectures and the three-tier depth ladder at the main-text type scale from the same ten seeds. The three- and four-tier depth grids that earlier appeared in main Fig.~6 are covered by \textbf{B} and \textbf{C} here, and the two-tier task by its interaction contrasts in Supplementary Fig.~S24B; all cell values remain in Source Data. These task-calibrated, original-budget comparisons establish forward-computation dependence on task and placement, not an optimizer-independent requirement for resolved credit.'''),
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
 r'''All analyses use eight initial reconstructed cells from one mouse. \textbf{A}, Direct-presynaptic versus target-proxy excitatory/inhibitory (E/I) labels for 2,012 jointly labeled mapped contacts; counts and row percentages expose the asymmetric disagreement, and the shading encodes the row percentage (colour bar). Contacts are nested observations. \textbf{B}, Direct-label fraction by compartment: individual cells and cell means with 95\% cell-bootstrap intervals. Denominators differ widely (22--66 contacts per cell at the soma, 342 pooled; 157--1,541 internal, 5,568 pooled; 1,148--5,697 terminal, 25,580 pooled), so per-cell points carry unequal weight. Label availability is not assumed random. \textbf{C}, Jaccard overlap of the four selected routes with the nominal hybrid-label, $5\,\mu$m dictionary across mapping thresholds (an ordinal factor, drawn at equal spacing) and label choices. The open symbol at hybrid, $5\,\mu$m is the reference dictionary compared with itself (Jaccard $=1$ by definition). With hybrid labels, 5 of 8 cells change at least one selected route at $2\,\mu$m (mean 0.72) and 3 of 8 at $10\,\mu$m (mean 0.82); with direct-only labels all eight cells differ at every threshold (means 0.21--0.24). \textbf{D}, Capture of the fixed nominal model-generated field under heterogeneous log-radius perturbations, indexed by their standard deviation (SD), for the hybrid and direct-only label dictionaries as in \textbf{C}; five draws per nonzero dose are averaged within cell, and thin lines join the eight cells across SD. The paired within-cell change from SD 0 to SD 0.5 is $-0.034$ with hybrid labels (95\% cell-bootstrap interval $-0.050$ to $-0.018$; 7 of 8 cells decline) and $-0.011$ with direct-only labels ($-0.053$ to $0.019$). Large symbols and bars in \textbf{C,D} show cell means and 95\% cell-bootstrap intervals (5,000 draws); small symbols are individual cells, with tied values spread side by side. \textbf{E}, Distribution of the mean-radius compressed axial resistance divided by the exact series sum over raw edges, on logarithmic ratio and count axes (0.1-log-unit bins). No segment exceeds its series sum; 72 of 608 segments (8 cells) fall below 0.89 (0.05 log units), the worst to 0.003 (2.5 log units). Length preservation does not imply axial-resistance preservation. \textbf{F}, Series-resistance minus mean-radius capture of the nominal field against the mean of the two, at the nominal condition (hybrid labels, $5\,\mu$m, SD 0), one point per cell, with leak terms held fixed; the dashed line marks equality. Six of eight cells are identical and two differ by $+0.049$ and $+0.017$. Across 1,056 retained sensitivity conditions, target fields, probe sites and weights remain those of the nominal model. These are preprocessing sensitivities, not calibrated geometric uncertainties or tests of endogenous task alignment.'''),
# ---------------------------------------------------------------- si_08 ----
('shunt_sensitivity','si_08_shunting',[('S11','A'),('S21','A'),('S11','BC'),('S21','DC')],
 'Dose, cable parameters and input mapping determine focal-shunt selectivity.',
 r'''All panels use passive reciprocal-cable models of the initial eight reconstructed cells; membrane resistance $R_m$ and background conductance are sensitivity parameters, not cell-specific fits. \textbf{A}, Shunt-minus-current-injection localization across fixed absolute doses at three membrane resistances ($R_m$ of 300, 1,000 and 15,000~$\Omega\,\mathrm{cm}^2$: one lightness step, marker and dash pattern each) and three background-conductance multipliers (column groups), on a symmetric-log axis that is linear within $\pm0.001$. The contrast is close to proportional to dose at every setting, and the $R_m$ ordering reverses with background: with no background the $R_m=15{,}000$ contrast stays near zero (slightly negative at 0.05 and 0.5~nS, an interval spanning zero at 5~nS), whereas at background multiplier four it is the largest of the three at every dose, with 8/8 cells positive. \textbf{B}, Within-cell controls in the same eight-cell cohort: shunt and baseline-current-matched injection localization indices (descendant minus depth-matched unrelated median absolute log-gradient change) and the relation-reassignment comparison of the true relation with a matched foreign relation template; positive values favor descendant localization for the physical shunt or the true relation. Grey lines pair the shunt and current-injection values of each cell; the reassignment control is a different comparison and stands apart, unpaired. \textbf{C}, Cable and transport selectivity $S_k$, the descendant-to-control ratio of median absolute Green-function entries for focal site $k$, versus full synaptic-gradient localization at unit input-conductance-normalized dose across 101 sites, three membrane resistances and three background levels; points are site--regime values, coloured by $R_m$ as in \textbf{A} with background multipliers 0, 1 and 4 drawn as filled, open and plus markers; they are descriptive because sites and regimes repeat within cell, and the dotted line marks $S_k=1$. \textbf{D}, Signed census of the same 101 sites: the fraction of descendant gradients attenuated, enhanced or sign reversed at $R_m=1{,}000\,\Omega\,\mathrm{cm}^2$, background multiplier one and unit dose, for the focal shunt and its baseline-current-matched injection. In all 81 tested passive conditions the shunt attenuated every descendant gradient without enhancement or sign reversal, and the matched injection enhanced every one without attenuation or sign reversal. \textbf{E}, All mapped versus directly typed presynaptic contacts, paired within the same eight cells by grey lines; the contrast increases in 8/8 cells. \textbf{F}, Dimensionless excitatory/inhibitory conductance-scale and normalized inhibitory-reversal sensitivity; each row names the E/I scale and the inhibitory reversal it holds, the open marker and the shaded band mark the reference condition (E/I 0.35, reversal $-0.2$) and its interval, and the right-hand column counts cells with a positive contrast. Sites are averaged within the eight cells; means and 95\% cell-bootstrap intervals are shown in \textbf{A,B,E,F}. Positive localization is a larger descendant change, not enhancement. The normalized-dose comparison is main Fig.~8D.'''),
('shunt_replication','si_08_shunting',[('S10','FGH'),('S47','AB')],
 'Focal-shunt controls replicate structurally and under weak-channel linearization.',
 r'''\textbf{A}, Focal shunt versus baseline-current-matched injection in 45 disjoint same-mouse cells (235 sites). \textbf{B}, True versus reassigned descendant relation templates in forty cells (230 sites); the true-relation column repeats the focal-shunt values of \textbf{A} for the forty cells that have a reassigned-relation partner. \textbf{C}, Direct E/I label coverage and the number of selected focal sites per cell, capped at 16 (one L5ET cell had 47 eligible sites); cell classes are ordinal colours within this panel. \textbf{D,E}, Separate initial eight-cell weak-channel ensemble: localization versus dose (dose divided by local input conductance) and paired shunt-minus-current contrasts. A fixed steady-state Jacobian (solid) is compared with the passive model (dashed); the two means differ by at most 0.010 localization units for the shunt and 0.023 for current injection at every dose, so the passive curve lies under the active one. Shunting attenuates descendants whereas matched current injection enhances them. Sixty-four channel draws and sites are averaged within each cell; intervals bootstrap cells and are narrower than the marker except at dose 4, and the contrast is positive in 8/8 cells at every dose (Wilcoxon $p=0.0078$). Weak conductances test local linearization near the passive response and do not establish robustness to regenerative dynamics or channel kinetics.'''),
# ---------------------------------------------------------------- si_09 ----
('measured_transfer_geometry','si_09_measured',[('S22','A'),('M9','BDE'),('S56','C')],
 'The measured-response learning comparison is limited by mapped coverage and transfer geometry.',
 r'''\textbf{A}, The seven targets of the measured cohort (one mouse; one selected scan per target): mapped functional presynaptic partners (filled circles) and the manually curated subset (open circles), and the split-half repeat reliability of each partner's responses (odd/even repeats, Spearman $r$; small dots, 69 records) with the target median (bar). \textbf{B}, Actual four-route support for the scan selected by median mapped-input count with identifier tie breaking (target 1, session 4, scan 10; nine mapped inputs); rows are input coordinates and columns selected routes. Across the thirteen scans, 6 of 13 supports place one site per route (mean 1.31 sites per route). \textbf{C}, Response-prediction NMSE for exact learning, treeless ridge and restricted dictionaries, with random and shuffled surrogate rows; dots are seven target means, diamonds and bars their means and 95\% target-bootstrap intervals, and the dotted line marks ridge (0.803). \textbf{D}, Common-checkpoint update reconstruction by the unrestricted fixed transfer profile (0.976, interval 0.964--0.986, narrower than the marker), the restricted ancestry profile (0.232) and oracle trialwise ancestry amplitudes (0.244). The unrestricted fixed profile is nearly sufficient because somatic error mainly scales a stable transfer vector; the random-anatomical (0.385) and site-shuffled (0.468) surrogates with oracle amplitudes, not drawn, reconstruct the update better than the ancestry restriction. \textbf{E}, Mean split-half (odd/even repeat) Spearman reliability in 1,000 independent calibration datasets versus the measured value, for 125 partner records; the two records with negative measured reliability (orange squares) are assigned zero reliable variance, and the identity line is not fitted. Scans and splits are averaged within targets before inference; $n$ is seven targets in \textbf{C,D} and 125 partner--scan records in \textbf{E}. Intervals are 95\% target-bootstrap intervals in \textbf{C,D}; in \textbf{E} the Monte Carlo standard error over the 1,000 datasets (0.0020--0.0028) is narrower than the marker, and 5 of 125 simulated means differ from the measured value by more than 1.96 standard errors (largest difference 0.0069); \textbf{A} and \textbf{B} are descriptive. Target-level associations for all four topology measures are in main Fig.~9C, coverage across all thirteen scans is in main Fig.~9F, and the support matrix in \textbf{B} is the one in main Fig.~9E. This is an offline passive-tree geometry diagnostic, distinct from the empirical ancestry-response association.'''),
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
 r'''\textbf{A,B}, Final costed-test-loss regret by imposed rank in the feedback-only and joint-transfer fixed-cache learners; regret is excess above the best trained candidate. \textbf{C}, Differences among strong selectors at each rank in the joint arm. The independently calibrated Gaussian SGD forecast improves over the original scalar utility; the full-batch and observed-context-count baselines remain competitive. All finite-horizon policies choose $K=r$ in this family, so improvement does not establish fine morphology selection. \textbf{D}, Median measured elapsed time on one central processing unit per twenty-candidate decision. Plug-in timings include calibration-model preparation, count-only time excludes shared calibration generation, and the pilot includes sixteen candidate updates; times measure these implementations, not hardware-independent complexity. \textbf{E}, Original scalar forecast versus final test half-mean-squared-error in the feedback-only arm. \textbf{F}, The calibration-derived Gaussian SGD forecast on the same cases and arm. Hexagons encode counts on a logarithmic color scale and dashed lines mark equality; the joint arm reproduces \textbf{E,F} and is retained in Source Data. The population-oracle forecast bias --- predicted minus exact population loss of the realized trained weights under fresh examples or finite-cache reuse --- is retained in Source Data: the fresh-example recurrence is exact under the stated Gaussian and conditional-linear assumptions, whereas cache reuse violates independence. Curves and whiskers in \textbf{A--D} are means and pointwise 95\% intervals from 10,000 whole-seed bootstrap draws over twenty seed blocks. All 25,600 candidate fits and both regimes are retained. These are linear-learning forecast tests, not a general nonlinear dendritic morphology law.'''),
('morphology_estimation','si_10_selection_statistics',[('S41','*')],None,None),
]

# Sentences appended to a caption that is otherwise the frozen original.
CAPTION_APPEND = {
 'mechanistic_chain': r'This chain is the figure support for Supplementary Section~S1; the exact-path field it feeds is drawn at the main-text type scale in main Fig.~1.',
 'oracle_profile_credit': r'Panel \textbf{C} is reproduced at the main-text type scale as main Fig.~4H.',
 'animal_credit_reanalysis': r'This reanalysis is restored as a figure so that the six-animal signed-credit comparison described in Methods has drawn support in \textbf{C}; the animal-level contrasts also remain tabulated.',
}

# Curated labels that must survive a merge as aliases (SI_NUMBERING 3.3).
ALIAS_EXTRA = {
 'utility_signal_noise': ['fig:si_checkpoint_geometry'],
 'measured_transfer_geometry': ['fig:si_measured_topology'],
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
 'mnist_dictionary_geometry': [('S49',[107,484,514,504])],
 # DECISIONS Q6: old S37D is registered as a legend strip, not a panel. Only
 # the three-family key is kept; its prose is already in the caption.
 'scalar_tree_capacity': [('S37',[310.0,244.0,500.0,307.0])],
}
WHOLE_CROPS = {'local_gate_controls': [0,0,518.4,452]}

# Audited decorated-panel bounds. Adjacent panels sometimes share margins;
# their axis fragments must not be imported with the selected panel.
PANEL_BOUNDS = {
 ('S5','E'):[180.2,157.5,346.95,294.7],
 ('S54','B'):[266.5,11.5,516,158], ('S54','D'):[266.5,169,516,316],
 ('S54','F'):[266.5,326,516,486],
 ('S51','B'):[276.5,7.5,516,203.5],
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
PANEL_BOUNDS[('S52','A')]=[4,6.6,251.1,200.2]
PANEL_REDACTIONS[('S52','A')]=[[251,174,261,191]]
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
