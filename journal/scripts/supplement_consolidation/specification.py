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
('mechanistic_chain','si_01_exact',[('S1','*')],None,None),
('credit_validation','si_01_exact',[('S2','AD'),('S3','C')],
 'Local eligibility and exact transport reproduce the learning gradient.',
 r'''\textbf{A}, Matched shunting backpropagation and local learning on MNIST, Fashion-MNIST and figure-ground MNIST; blue denotes normalized-additive learning. \textbf{B}, Transport, activation and additive-model implementation controls on MNIST. \textbf{C}, Exact-transport-minus-backpropagation accuracy across three-/five-factor and local/backpropagated decoder choices; points are five paired seeds and bars are 95\% paired-seed bootstrap intervals. Exact transport supplies an oracle compartment error. The local three-factor eligibility is sufficient when this error is supplied. Normalized- and raw-additive models are distinct controls; no noise-task shunting aggregate is used here.'''),
('utility_signal_noise','si_01_exact',[('S46','ACDE'),('X1','*'),('S5','E'),('X2','*')],
 'Restricted routes trade signal retention against admitted noise, and the one-step quantity is measurable at trained states.',
 r'''\textbf{A}, Fixed-operator version of the one-step utility bound: the delivered update is $M(\bm\mu+\bm\xi)$, with mean gradient $\bm\mu$ and zero-mean noise $\bm\xi$. \textbf{B}, Subtree-minus-random capture across route budget and covariance-mixture coefficient; only the endpoints are isospectral. \textbf{C}, Final quadratic loss when the signal hierarchy and amplified fine-scale noise make an interior route resolution favorable by construction. \textbf{D}, Exact one-step projection boundary: retaining signal fraction $f_{\rm sig}$ and noise fraction $f_{\rm noise}$ changes normalized loss by $(f_{\rm noise}-f_{\rm sig})/2$. \textbf{E}, Candidate/exact gradient cosine and norm-matched one-step loss decrease at the same 120 valid trained checkpoints, drawn on one dimensionless axis; boxes summarize checkpoints and diamonds with bars are means and 95\% checkpoint-bootstrap intervals from a single held-out batch at relative step $10^{-5}$. \textbf{F}, A separate constructed positive control on eight reconstructed arbors: loss reduction after twenty projected steps against the full-gradient sequence. Green denotes morphology-selected routes, gray random paths, blue depth bins and pink ancestry-shuffled routes; symbols are cell means with hierarchical 95\% bootstrap intervals. \textbf{G}, Illustrative rank--noise trade-off under isotropic noise: twice the smoothness constant $L$ times the optimized one-step bound is $q^2/(q+K\sigma^2)$, comparing $(K,q)=(1,0.8)$ and $(2,1)$, with the sign change shaded; analytic, no data, and not an endpoint-selection prediction. The related initialization-utility diagnostic matches 135 of 200 seed--task optima; the remaining cases are retained numerically. Panels B--D use abstract Haar coordinates, not a forward dendritic neuron. The 50 paired seeds per signal--noise screen and all four task depths are retained. The imposed alignment and oracle projections explain the conditional relationship in \textbf{E,F}; neither diagnostic measures biological credit or validates a long-horizon initialization selector. The bound concerns local progress; the prospective endpoint-selection failure is reported in Fig.~\ref{fig:si_original_selector}.'''),
('reliability_gain','si_01_exact',[('S13','AF'),('S24','CD')],
 'Fixed and estimated reliability gains have bounded learning benefits.',
 r'''All panels use positive-rate synthetic tasks with eight parallel, nonserial branch blocks. \textbf{A}, A supplied compensating current preserves branch voltage while a positive shunt reduces input resistance and eligibility. \textbf{B}, Paired final-loss contrasts at maximal heterogeneity; fixed aligned shunting has no reliable final advantage over unshunted noisy learning. \textbf{C,D}, Adaptive-rule final loss and paired control-minus-adaptive contrasts. Adaptive local gain beats global/shuffled attenuation but loses to no shunt and the fixed oracle. The point implementation given identical gains agrees numerically. Curves and contrast intervals are means and 95\% paired-seed bootstrap intervals across 50 independent seeds in each study. Supplied state compensation and gradient-moment observations are explicit information resources.'''),
('same_span_conditioning','si_01_exact',[('S14','ABEF')],
 'Identical route spans can learn differently through conditioning.',
 r'''\textbf{A}, Haar, redundant nested and statically scaled nested coordinates parameterize the same rank-eight span in sixteen abstract coefficients. \textbf{B}, Positive Gram eigenvalues; condition numbers are 1, 15 and 388.52. \textbf{C}, Population loss after eighty updates across effective sample size. \textbf{D}, Paired nested-minus-Haar contrasts; negative values favor nested coordinates. Points and bars are means and 95\% paired-seed bootstrap intervals across fifty seeds; dashed curves are the exact finite-time risk. Slow modes reduce variance at low data and retain bias at high data. Gram-preconditioned controls agree to $1.15\times10^{-15}$. This is a coefficient-learning comparison within one span, not a simulation of forward dendritic morphology or a biological implementation of the preconditioner.'''),
# ---------------------------------------------------------------- si_02 ----
('image_generalization','si_02_credit_rules_images',[('S4','CDE'),('S30','CD')],
 'The neuron-identity bottleneck generalizes across image and feedback controls.',
 r'''\textbf{A}, Flattened CIFAR-10 in the shunting architecture, with matched-width scalar fallback, random rank-four, exact-path and backpropagation rules (five seeds). \textbf{B}, Raw-additive CIFAR-10 with strict scalar, per-neuron, exact-path and backpropagation feedback (twenty paired seeds; 95\% Student-$t$ intervals). Exact path is 0.859 percentage points below per-neuron feedback and equivalent to backpropagation within the prespecified one-point margin. \textbf{C}, Fashion-MNIST in shunting and raw-additive trees (ten paired seeds; 95\% bootstrap intervals); path resolution adds no reliable benefit. \textbf{D,E}, Dominant and sub-point contrasts under fixed random soma feedback (DFA), with fifteen paired MNIST seeds and 95\% bootstrap intervals. The per-neuron-minus-strict-scalar contrast of about eleven percentage points is drawn in \textbf{D}. DFA preserves the neuron-versus-scalar benefit, but does not add a second dendritic layer or increase the dimensionality of the ten-class readout error. Percentage-point scales differ.'''),
('mnist_dictionary_geometry','si_02_credit_rules_images',[('S49','CDEF')],
 'Intermediate dictionaries capture more image-task credit without improving accuracy.',
 r'''\textbf{A,B}, Six-rule MNIST ladder in shunting and raw-additive networks; ten fresh paired seeds per architecture. Filled/open symbols use development-selected/original common rates. Decoder-only learning freezes the initialized core. \textbf{C}, Paired accuracy differences for three profiles versus a projected common profile, and exact versus three-profile credit. \textbf{D}, Initial and trained field-energy capture by uniform $K=1$ or subtree $K=3$ profiles, averaged over 2,048 held-out images within each seed. Projection coefficients use the exact activation-error field of this ten-seed cohort, so \textbf{D} is not a duplicate of the main-text capture panel; voltage-space capture for the same fits is retained in Source Data. Means and bars in A--D have descriptive 95\% seed-bootstrap intervals. Greater capture did not yield an image accuracy benefit. All image cohorts have one dendritic layer and a linear readout.'''),
('error_field_geometry','si_02_credit_rules_images',[('S45','CDE')],
 'Error-field geometry separates direction, amplitude and spatial variation.',
 r'''Independent fifteen-seed flattened-MNIST cohort of 128 directed $[3,3]$ trees; green circles denote shunting and blue squares raw-additive trees. \textbf{A}, Branch-gradient cosine with the exact gradient at common fixed checkpoints trained with per-neuron feedback; matched-width scalar-fallback and per-neuron fields are evaluated at identical weights and states, and the exact-path cosine of one is definitional. \textbf{B}, Exact voltage-error magnitude by depth at exact-path checkpoints: the batch root-mean-square ratio $\operatorname{RMS}(\delta^V_{n,u})/\operatorname{RMS}(\delta^V_{0,u})$, averaged over neurons and paths, with somatic value one by normalization. \textbf{C}, Within-depth path-specific residual error energy after subtracting the mean across paths within depth $d$: distal shunting attenuates magnitude but retains the larger path-specific fraction, rising from 18\% to 53\% (shunting) and 32\% (raw additive). Symbols and bars are means and 95\% seed-bootstrap intervals. No additional models are trained here; greater path-specific variation did not yield an image accuracy benefit, and voltage- and activation-space capture are different quantities.'''),
('input_coverage_depth','si_02_credit_rules_images',[('S7','ABCD'),('S8','EG')],
 'Input coverage and ordinary-task depth are distinct from credit resolution.',
 r'''\textbf{A--D}, Spatial versus randomly assigned contacts in depth-four $[2,2,2,2]$ trees: example map, unique input coverage, cross-branch overlap and learning benefit under backpropagation and three local fields. Contact count is matched, but spatial maps reach more unique inputs and avoid collisions. The accuracy benefit persists under backpropagation and therefore reflects a forward sparsity prior. \textbf{E,F}, Separate raw-additive historical noise cohort with sixteen terminals: all four D4-minus-D1 accuracy contrasts are negative in 10/10 paired seeds; active contact counts are approximately fixed while parameter counts vary. Learning contrasts use means and 95\% paired-seed bootstrap intervals; map summaries use mean $\pm$ s.d. The noise-task executed generator remains unresolved and its shunting runs fail the positive-conductance input criterion. These retained additive outcomes are a depth boundary, not evidence for shunting; exclusions are enumerated in Table~\ref{tab:input_validity}.'''),
# ---------------------------------------------------------------- si_03 ----
('branch_conflict_controls','si_03_conflict_ancestry',[('S19','A'),('S29','DEF')],
 'Branch-selective credit limits interference and preserves update direction.',
 r'''\textbf{A}, Two-branch learning in ten paired seeds. Points show seeds; large symbols and bars are means and 95\% paired-seed bootstrap intervals. Exact path and the equivalent gated-point rule supply branch selection; the context-0 forgetting of the same cohort is drawn at the main-text type scale in main Fig.~2E. \textbf{B--D}, Shared-versus-exact gradient cosine in the separate branch-conflict experiment with two, four or eight branches. Solid curves evaluate common exact-learning states at epochs 0, 50 and 250; dashed curves evaluate the shared learner's own final state. Curves and shading use twenty paired seeds and 95\% bootstrap intervals. An exact quadratic interference calculation is given in the accompanying derivation. Learned-state geometry can depart from the initialization mean-field approximation.'''),
('ancestry_coefficients','si_03_conflict_ancestry',[('S23','A'),('S32','CD')],
 'Available route span and learned route coefficients are different constraints.',
 r'''\textbf{A}, Trained field capture by route budget in the eight-context factorial. Ancestry, depth and random partitions have coincident capture at matched bandwidth; derangement removes the target coordinate. Capture therefore does not explain the small ancestry-specific advantage, and capture alone does not predict trained utility: the within-seed Spearman correlation between capture and utility is about 0.82 over all budgets and 0.64 for $K<8$, and is retained numerically in Source Data. \textbf{B}, Held-out accuracy versus calibration set size (16, 64 and 256 examples) at cue-noise standard deviation 0.5 and zero delay; 30 epochs are fixed, so both data and computation increase. \textbf{C}, Cue delay at 256 calibration trials and noise SD 0.5; delays mismatch independent contexts and are measured in trials, not physiological time. In \textbf{B,C} the dashed curves marked with an asterisk use maximum-probability route selection by the same frozen encoder, an exploratory paired sensitivity specified after the soft outcomes with no encoder refit. Curves, diamonds and bars are means and 95\% paired-seed bootstrap intervals over twenty fresh analysis seeds. The cue-noise sensitivity of the same encoder is drawn at the main-text type scale in main Fig.~3F. The calibration cue and activation targets are supplied resources. No serial forward dendritic computation is present.'''),
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
 r'''\textbf{A}, Serial D3, resource-identical grouped-point and approximately parameter-matched flexible point networks. Only serial and grouped models share modules and contacts. \textbf{B}, Original-budget three-level aligned task across D1--D3 and the five-arm feedback ladder. \textbf{C}, Four-level task including D4 under aligned or reversed sensor placement. \textbf{D}, Serial-minus-grouped accuracy at fixed D3 across nested, flat and local-ratio task families under exact-path LocalCA, the counterpart of the backpropagation contrast in main Fig.~6C; its 95\% intervals are also carried onto that panel. \textbf{E,F}, Flexible point networks matched to active or to total parameter count. \textbf{G}, Accuracy across sensor--task alignment $\alpha$. \textbf{H}, Change in the D3-minus-D1 advantage from $\alpha=0.25$ to 0.75 and its fitted slope across doses; intermediate doses were specified after the endpoint results and constitute interpolation. Curves and differences use ten paired seeds per cohort and 95\% paired-seed bootstrap intervals. Main Fig.~6A,D re-render the architectures and the three-tier depth ladder at the main-text type scale from the same ten seeds. The two-, three- and four-tier depth grids that earlier appeared in main Fig.~6 are covered by \textbf{B} and \textbf{C} here and by Supplementary Fig.~S24B; all cell values remain in Source Data. These task-calibrated, original-budget comparisons establish forward-computation dependence on task and placement, not an optimizer-independent requirement for resolved credit.'''),
('physical_calibration','si_06_physical_depth',[('S15','*')],None,None),
('physical_optimizer','si_06_physical_depth',[('S18','EF'),('S28','B')],
 'Physical-depth conclusions depend on alignment, task hierarchy and optimization.',
 r'''\textbf{A}, Serial/point contrasts under aligned and reversed placement. \textbf{B}, Separate independently seeded two-level task with D1 $[8]$ and D2 $[4,1]$: depth-by-alignment interactions for serial and grouped-point backpropagation and for shared or exact-path LocalCA. \textbf{C}, Original-budget D3 optimizer/credit-coordinate contrasts in the serial conductance morphology $[2,1,2]$. Under the BP optimizer, full versus soma-broadcast differentiation differs by 0.64 points; under the split LocalCA optimizer, exact versus shared differs by 10.96 points; changing the broadcast optimizer costs 13.47 points. Means and intervals use ten paired seeds and 95\% bootstrap intervals. These earlier recipe-specific effects do not imply a converged credit advantage: at 600 epochs the LocalCA accuracy difference reverses, while exact credit retains lower cross-entropy (main Fig.~6). All D3 continuations hit the cap; D1 usually stops early.'''),
# ---------------------------------------------------------------- si_07 ----
('anatomy_capacity_controls','si_07_anatomy',[('S10','B'),('S20','B'),('S27','BD')],
 'Cohort, label and arbor-size controls bound modeled anatomical capacity.',
 r'''\textbf{A}, Route-generated field capture versus budget in 47 disjoint same-mouse cells; morphology routes use 8.1\% of dense-feedback wiring at eight channels. Twenty modeled streams are averaged within each cell; bands and bars are 95\% cell-bootstrap intervals. \textbf{B}, Initial eight-cell cohort: capture restricted to directly typed presynaptic contacts. \textbf{C,D}, Second-mouse Pinky cohort: relative residual norm versus route budget against the dense principal-component ceiling, and input-bearing candidate-site count versus effective field rank. Eligibility changes from ten cells at $K\leq4$ to eight at $K=8$; small arbors approach the rank ceiling. Panel \textbf{C} uses mean $\pm$ s.e.m., and its residual norm is not squared-energy capture. All fields are model generated. Main Fig.~7 equalizes a common broadcast and leads with actual-tree surrogate controls; these route-generated comparisons are structural positive controls, not measured biological teaching signals.'''),
('inhibitory_spatial_controls','si_07_anatomy',[('S12','BDEFGH')],
 'Spatial matching limits apparent inhibitory targeting of ancestry domains.',
 r'''\textbf{A}, Six layer-diverse reconstructions with mapped inhibitory contacts in $x$--$y$ projection. \textbf{B--D}, Observed-minus-matched tree distance, shared-path fraction and weighted descendant-domain overlap in the twenty-target analysis, after aggregation within 92 presynaptic axons, and after joint path-distance/Euclidean matching. Symbols and intervals are means and 95\% bootstrap intervals over the displayed inferential unit. Only tree distance persists after axon aggregation; no endpoint of the \textbf{B--D} triplet persists after joint three-dimensional matching. \textbf{E}, Cumulative descendant-input fraction by distal-dendrite-targeting versus perisomatic-targeting class, retaining one contact per axonal clump. \textbf{F}, Inhibitory-contact domain size versus matched locations; no reliable overall localization advantage is detected. Cells derive from one mouse. These structural comparisons do not show that contacts carried teaching signals.'''),
('coarse_energy','si_07_anatomy',[('S25','*')],None,None),
('anatomy_preprocessing','si_07_anatomy',[('S33','*')],None,None),
# ---------------------------------------------------------------- si_08 ----
('shunt_sensitivity','si_08_shunting',[('S11','A'),('S21','A'),('S11','BC'),('S21','DC')],
 'Dose, cable parameters and input mapping determine focal-shunt selectivity.',
 r'''All panels use passive reciprocal-cable models of the initial eight reconstructed cells; membrane resistance $R_m$ and background conductance are sensitivity parameters, not cell-specific fits. \textbf{A}, Shunt-minus-current-injection localization across fixed absolute doses at three membrane resistances. \textbf{B}, Within-cell controls in the same eight-cell cohort: shunt and baseline-current-matched injection localization indices (descendant minus depth-matched unrelated median absolute log-gradient change) and the relation-reassignment comparison of the true relation with a matched foreign relation template; positive values favor descendant localization for the physical shunt or the true relation. \textbf{C}, Cable and transport selectivity $S_k$, the descendant-to-control ratio of median absolute Green-function entries for focal site $k$, versus full synaptic-gradient localization at unit input-conductance-normalized dose across 101 sites, three membrane resistances and three background levels; points are site--regime values and are descriptive because sites and regimes repeat within cell, and the dotted line marks $S_k=1$. \textbf{D}, Signed census of the same 101 sites: the fraction of descendant gradients attenuated, enhanced or sign reversed at $R_m=1{,}000\,\Omega\,\mathrm{cm}^2$, background multiplier one and unit dose. Every tested passive condition attenuated without enhancement or sign reversal. \textbf{E}, All mapped versus directly typed presynaptic contacts. \textbf{F}, Dimensionless excitatory/inhibitory conductance-scale and normalized inhibitory-reversal sensitivity. Sites are averaged within the eight cells; means and 95\% cell-bootstrap intervals are shown in \textbf{A,B,E,F}. Positive localization is a larger descendant change, not a sign of enhancement. Absolute dose, input-conductance dose and local nonaxial normalized dose are distinct; the normalized-dose comparison is now main Fig.~8D.'''),
('shunt_replication','si_08_shunting',[('S10','FGH'),('S47','AB')],
 'Focal-shunt controls replicate structurally and under weak-channel linearization.',
 r'''\textbf{A}, Focal shunt versus baseline-current-matched injection in 45 disjoint same-mouse cells (235 sites). \textbf{B}, True versus reassigned descendant relation templates in forty cells (230 sites). \textbf{C}, Direct E/I label coverage and focal-site eligibility per cell. \textbf{D,E}, Separate initial eight-cell weak-channel ensemble: localization versus dose and paired shunt-minus-current contrasts. A fixed steady-state Jacobian is compared with the passive model; shunting attenuates descendants whereas matched current injection enhances them. Sixty-four channel draws and sites are averaged within each cell. All intervals bootstrap cells, not sites or channel draws. Weak conductances test local linearization near the passive response; they do not establish robustness to strongly regenerative dynamics, channel kinetics or a newly solved nonlinear equilibrium at each intervention dose.'''),
# ---------------------------------------------------------------- si_09 ----
('measured_transfer_geometry','si_09_measured',[('S22','A'),('M9','BDE'),('S56','C')],
 'The measured-response learning comparison is limited by mapped coverage and transfer geometry.',
 r'''\textbf{A}, Number of mapped functional presynaptic partners for each of seven targets in one mouse. \textbf{B}, Actual four-route support for the scan selected by median mapped-input count, with identifier tie breaking; rows are input coordinates and columns selected routes. The selected supports are mostly single distal sites, rather than illustrative nested bands. \textbf{C}, Response-prediction NMSE for exact learning, treeless ridge and restricted dictionaries, with random and shuffled surrogate rows; dots are seven target means, diamonds and bars their means and 95\% target-bootstrap intervals, and the dotted line marks ridge. \textbf{D}, Common-checkpoint update reconstruction by the unrestricted fixed transfer profile, the restricted ancestry profile and oracle trialwise ancestry amplitudes. The unrestricted fixed profile is nearly sufficient because somatic error mainly scales a stable transfer vector; restricted rules underperform ridge. \textbf{E}, Mean split-half Spearman reliability in 1,000 independent calibration datasets versus the observed value, for 125 partner records; the two negative estimates are assigned zero reliable variance and the identity line is not fitted. Scans and splits are averaged within targets before inference; $n$ is seven targets in \textbf{C,D} and 125 partner--scan records in \textbf{E}, with 95\% bootstrap intervals throughout. Target-level associations for all four topology measures are in main Fig.~9C, and coverage across all thirteen scans is in main Fig.~9F. This is an offline passive-tree geometry diagnostic, distinct from the empirical ancestry-response association.'''),
('measured_predictor_controls','si_09_measured',[('S34','*')],None,None),
('animal_credit_reanalysis','si_09_measured',[('S17','*')],None,None),
# ---------------------------------------------------------------- si_10 ----
('original_selector','si_10_selection_statistics',[('S35','*')],None,None),
('finite_horizon','si_10_selection_statistics',[('S38','ABCD'),('S39','AC')],
 'Finite-horizon prediction improves selection but strong simple baselines remain.',
 r'''\textbf{A,B}, Final costed-test-loss regret by imposed rank in the feedback-only and joint-transfer fixed-cache learners; regret is excess above the best trained candidate. \textbf{C}, Differences among strong selectors at each rank in the joint arm. The independently calibrated Gaussian SGD forecast improves over the original scalar utility; the full-batch and observed-context-count baselines remain competitive. All finite-horizon policies choose $K=r$ in this family, so improvement does not establish fine morphology selection. \textbf{D}, Median measured elapsed time on one central processing unit per twenty-candidate decision. Plug-in timings include calibration-model preparation, count-only time excludes shared calibration generation, and the pilot includes sixteen candidate updates; times measure these implementations, not hardware-independent complexity. \textbf{E}, Original scalar forecast versus final test half-mean-squared-error in the feedback-only arm. \textbf{F}, The calibration-derived Gaussian SGD forecast on the same cases and arm. Hexagons encode counts on a logarithmic color scale and dashed lines mark equality; the joint arm reproduces \textbf{E,F} and is retained in Source Data. The population-oracle forecast bias --- predicted minus exact population loss of the realized trained weights under fresh examples or finite-cache reuse --- is retained in Source Data: the fresh-example recurrence is exact under the stated Gaussian and conditional-linear assumptions, whereas cache reuse violates independence. Curves and whiskers in \textbf{A--D} are means and pointwise 95\% intervals from 10,000 whole-seed bootstrap draws over twenty seed blocks. All 25,600 candidate fits and both regimes are retained. These are linear-learning forecast tests, not a general nonlinear dendritic morphology law.'''),
('morphology_estimation','si_10_selection_statistics',[('S41','*')],None,None),
]

# Sentences appended to a caption that is otherwise the frozen original.
CAPTION_APPEND = {
 'mechanistic_chain': r'This chain is the figure support for Supplementary Section~S1; the exact-path field it feeds is drawn at the main-text type scale in main Fig.~1.',
 'oracle_profile_credit': r'Panel \textbf{C} is reproduced at the main-text type scale as main Fig.~4H.',
 'animal_credit_reanalysis': r'This reanalysis is restored as a figure so that the six-animal signed-credit comparison described in Methods has drawn support; the animal-level contrasts remain tabulated.',
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
 'S13': [[105,168,438,182]],
 'S14': [[155,168,371,184]],
 'S16': [[0,384,518.4,404.4]],
 'S24': [[90,151,423,164]],
 'S32': [[0,0,518.4,35]],
 'S36': [[0,0,518.4,16]],
 'S37': [[0,0,518.4,18]],
 'S38': [[0,0,518.4,16]],
 'S49': [[107,484,514,504]],
}
SHARED_LEGENDS = {
 'reliability_gain': [('S24',[90,151,423,164])],
 'ancestry_coefficients': [('S32',[100,0,422,34])],
 'mnist_dictionary_geometry': [('S49',[107,484,514,504])],
 # DECISIONS Q6: old S37D is registered as a legend strip, not a panel. Only
 # the three-family key is kept; its prose is already in the caption.
 'scalar_tree_capacity': [('S37',[310.0,244.0,500.0,307.0])],
}
WHOLE_CROPS = {'local_gate_controls': [0,0,518.4,452]}

# Audited decorated-panel bounds. Adjacent panels sometimes share margins;
# their axis fragments must not be imported with the selected panel.
PANEL_BOUNDS = {
 ('S46','D'):[4,148,177.7,296.5], ('S46','E'):[183,148,337,296],
 ('S5','E'):[180.2,157.5,346.95,294.7],
 ('S54','B'):[266.5,11.5,516,158], ('S54','D'):[266.5,169,516,316],
 ('S54','F'):[266.5,326,516,486],
 ('S51','B'):[276.5,7.5,516,203.5],
 ('S31','D'):[242,134.5,516,278], ('S31','G'):[327,283,501,408],
 ('S27','B'):[266,3.5,515,188], ('S27','D'):[266,206.5,515,394],
 ('S12','H'):[136.5,303,267,447], ('S10','F'):[345.8,158.8,514,288],
 ('M9','D'):[4,285,263,456],
}
PANEL_REDACTIONS = {
 ('S31','D'):[[240,129,517,134.3]],
 ('S31','G'):[[315,300,343,378]],
 ('S12','H'):[[130,316,140,329]],
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
REMOVED_SHARED_REGIONS['S27']=[[170,188.2,408,206.4]]
SHARED_LEGENDS['anatomy_capacity_controls']=[('S27',[170,188.2,408,206.4])]
PANEL_BOUNDS[('S31','B')]=[310,4.5,516,136]
PANEL_BOUNDS[('S20','B')]=[256,7.5,502,191]
PANEL_REDACTIONS[('S46','E')]=[[314.8,168.8,343,177.6]]
REMOVED_SHARED_REGIONS['S19']=[[217.9,232.3,374.1,241]]
PANEL_BOUNDS[('M9','E')]=[275.5,283.5,502,455]
PANEL_REDACTIONS.pop(('M9','D'),None)
PANEL_BOUNDS[('S31','G')]=[327,283,516,408]
PANEL_TEXT[('S31','G')][0]['text']='Serial minus grouped (pp)'
REMOVED_SHARED_REGIONS['S14'].append([346,334,475,350])
NATIVE_LEGENDS={'same_span_conditioning':[
 {'label':'orthonormal Haar','color':[0.20,0.65,0.42]},
 {'label':'raw nested','color':[0.88,0.49,0.40]},
 {'label':'scaled nested','color':[0.12,0.32,0.68]}]}
PANEL_REDACTIONS[('S19','A')]=[[146.5,224.6,223,234.8]]
PANEL_TEXT[('S19','A')]=[{'text':'Held-out accuracy','origin':[146.8,232.2], 'fontsize':8.0,'role':'restored x-axis label'}]
REMOVED_SHARED_REGIONS['S10']=[[210.8,410.1,274.4,428.4]]
PANEL_BOUNDS[('S10','G')]=[13.6,302.6,267,431]
PANEL_BOUNDS[('S10','H')]=[274.5,302.6,515,432]
PANEL_TEXT[('S10','G')]=[
 {'text':'true','origin':[235,417], 'fontsize':7.0,'role':'compact x-axis label, line 1'},
 {'text':'descendants','origin':[219,426], 'fontsize':7.0,'role':'compact x-axis label, line 2'}]
PANEL_BOUNDS[('S52','A')]=[4,6.6,251.1,200.2]
PANEL_REDACTIONS[('S52','A')]=[[251,174,261,191]]
PANEL_REDACTIONS[('S31','G')].append([343,282,518,296])
PANEL_TEXT[('S31','G')]=[{'text':'Exact LocalCA: serial advantage (pp)',
 'origin':[329,292], 'fontsize':8.0, 'role':'quantity and optimizer in panel title'}]
PANEL_BOUNDS[('S10','H')]=[265,302.6,515,432]
PANEL_BOUNDS[('S31','G')]=[342,283,516,408]
PANEL_TEXT[('S31','G')][0]['origin']=[344,292]
# 2026-09-10: the orphan clipped label between old S12 G and H (review 5.4).
PANEL_REDACTIONS[('S12','G')]=[[136.0,334.0,141.5,412.0]]

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
 'ancestry_coefficients':['review_coefficient_encoder','review_coefficient_hard_readout','trained_subtree_address'],
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
