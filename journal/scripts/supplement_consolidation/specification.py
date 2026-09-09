"""Editorial panel selection; all plotted observations remain unchanged.

The source panel names refer to the pre-consolidation published-artwork
registry. These identifiers are provenance coordinates, not live citations.
"""
from __future__ import annotations

# (semantic file/label id, prose module, source panel groups, title, caption)
# A group ('Sx', '*') retains a complete native source sheet.
FIGURES = [
('credit_validation','si_01_exact',[('S2','AD'),('S3','C')],
 'Local eligibility and exact transport reproduce the learning gradient.',
 r'''\textbf{A}, Matched shunting backpropagation and local learning on MNIST, Fashion-MNIST and figure-ground MNIST; blue denotes normalized-additive learning. \textbf{B}, Transport, activation and additive-model implementation controls on MNIST. \textbf{C}, Exact-transport-minus-backpropagation accuracy across three-/five-factor and local/backpropagated decoder choices; points are five paired seeds and bars are 95\% paired-seed bootstrap intervals. Exact transport supplies an oracle compartment error. The local three-factor eligibility is sufficient when this error is supplied. Normalized- and raw-additive models are distinct controls; no noise-task shunting aggregate is used here.'''),
('utility_signal_noise','si_01_exact',[('S46','ACDE')],
 'Restricted routes trade signal retention against admitted noise.',
 r'''\textbf{A}, Fixed-operator version of the one-step utility bound: the delivered update is $M(\bm\mu+\bm\xi)$, with mean gradient $\bm\mu$ and zero-mean noise $\bm\xi$. \textbf{B}, Subtree-minus-random capture across route budget and covariance-mixture coefficient; only the endpoints are isospectral. \textbf{C}, Final quadratic loss when the signal hierarchy and amplified fine-scale noise make an interior route resolution favorable by construction. \textbf{D}, Exact one-step projection boundary: retaining signal fraction $f_{\rm sig}$ and noise fraction $f_{\rm noise}$ changes normalized loss by $(f_{\rm noise}-f_{\rm sig})/2$. The related initialization-utility diagnostic matches 135 of 200 seed--task optima; the remaining cases are retained numerically. Panels B--D use abstract Haar coordinates, not a forward dendritic neuron. The 50 paired seeds per signal--noise screen and all four task depths are retained. The bound concerns local progress; the prospective endpoint-selection failure is reported in Fig.~\ref{fig:si_original_selector}.'''),
('checkpoint_geometry','si_01_exact',[('S9','ACD'),('S5','E')],
 'Gradient geometry predicts a small matched step under stated conditions.',
 r'''\textbf{A--C}, Candidate/exact gradient cosine, norm-matched one-step loss decrease and their within-rule association at 120 valid trained checkpoints. Boxes summarize checkpoints; diamonds and intervals show means and 95\% checkpoint-bootstrap intervals. A single held-out batch and relative step $10^{-5}$ are used. \textbf{D}, A separate constructed positive control on eight reconstructed arbors: a field interpolates between the route span and its orthogonal complement, then undergoes twenty projected quadratic steps. Green denotes morphology-selected routes, gray random paths, blue depth bins and pink ancestry-shuffled routes. Symbols show cell means and hierarchical 95\% bootstrap intervals. The imposed alignment and oracle projections explain this conditional relationship; it does not measure biological credit. Neither diagnostic validates a long-horizon initialization selector.'''),
('reliability_gain','si_01_exact',[('S13','AF'),('S24','CD')],
 'Fixed and estimated reliability gains have bounded learning benefits.',
 r'''All panels use positive-rate synthetic tasks with eight parallel, nonserial branch blocks. \textbf{A}, A supplied compensating current preserves branch voltage while a positive shunt reduces input resistance and eligibility. \textbf{B}, Paired final-loss contrasts at maximal heterogeneity; fixed aligned shunting has no reliable final advantage over unshunted noisy learning. \textbf{C,D}, Adaptive-rule final loss and paired control-minus-adaptive contrasts. Adaptive local gain beats global/shuffled attenuation but loses to no shunt and the fixed oracle. The point implementation given identical gains agrees numerically. Curves and contrast intervals are means and 95\% paired-seed bootstrap intervals across 50 independent seeds in each study. Supplied state compensation and gradient-moment observations are explicit information resources.'''),
('same_span_conditioning','si_01_exact',[('S14','ABEF')],
 'Identical route spans can learn differently through conditioning.',
 r'''\textbf{A}, Haar, redundant nested and statically scaled nested coordinates parameterize the same rank-eight span in sixteen abstract coefficients. \textbf{B}, Positive Gram eigenvalues; condition numbers are 1, 15 and 388.52. \textbf{C}, Population loss after eighty updates across effective sample size. \textbf{D}, Paired nested-minus-Haar contrasts; negative values favor nested coordinates. Points and bars are means and 95\% paired-seed bootstrap intervals across fifty seeds; dashed curves are the exact finite-time risk. Slow modes reduce variance at low data and retain bias at high data. Gram-preconditioned controls agree to $1.15\times10^{-15}$. This is a coefficient-learning comparison within one span, not a simulation of forward dendritic morphology or a biological implementation of the preconditioner.'''),
('image_generalization','si_02_credit_rules_images',[('S4','CDE'),('S30','CD')],
 'The neuron-identity bottleneck generalizes across image and feedback controls.',
 r'''\textbf{A}, Flattened CIFAR-10 in the shunting architecture, with matched-width scalar fallback, random rank-four, exact-path and backpropagation rules (five seeds). \textbf{B}, Raw-additive CIFAR-10 with strict scalar, neuron-specific, exact-path and backpropagation feedback (twenty paired seeds; 95\% Student-$t$ intervals). Exact path is 0.859 percentage points below neuron-specific feedback and equivalent to backpropagation within the prespecified one-point margin. \textbf{C}, Fashion-MNIST in shunting and raw-additive trees (ten paired seeds; 95\% bootstrap intervals); path resolution adds no reliable benefit. \textbf{D,E}, Dominant and sub-point contrasts under fixed random soma feedback (DFA), with fifteen paired MNIST seeds and 95\% bootstrap intervals. DFA preserves the neuron-versus-scalar benefit, but does not add a second dendritic layer or increase the dimensionality of the ten-class readout error. Percentage-point scales differ.'''),
('mnist_dictionary_geometry','si_02_credit_rules_images',[('S49','CDEF'),('S45','CDE')],
 'Intermediate dictionaries capture more image-task credit without improving accuracy.',
 r'''\textbf{A,B}, Six-rule MNIST ladder in shunting and raw-additive networks; ten fresh paired seeds per architecture. Filled/open symbols use development-selected/original common rates. Decoder-only learning freezes the initialized core. \textbf{C}, Paired accuracy differences for three profiles versus a projected common profile, and exact versus three-profile credit. \textbf{D}, Initial and trained activation-error capture by one or three profiles, averaged over 2,048 images within each seed. Projection coefficients use the exact field. Means and bars in A--D have descriptive 95\% seed-bootstrap intervals. \textbf{E--G}, Independent fifteen-seed cohort: common-state branch-gradient cosine, soma-normalized exact voltage-error magnitude by depth, and within-depth path-specific error energy. These distinguish direction, amplitude and spatial variation; exact-path cosine one is definitional. Greater path-specific variation did not yield an image accuracy benefit. Voltage- and activation-space capture are different quantities. All image cohorts have one dendritic layer and a linear readout.'''),
('input_coverage_depth','si_02_credit_rules_images',[('S7','ABCD'),('S8','EG')],
 'Input coverage and ordinary-task depth are distinct from credit resolution.',
 r'''\textbf{A--D}, Spatial versus randomly assigned contacts in depth-four $[2,2,2,2]$ trees: example map, unique input coverage, cross-branch overlap and learning benefit under backpropagation and three local fields. Contact count is matched, but spatial maps reach more unique inputs and avoid collisions. The accuracy benefit persists under backpropagation and therefore reflects a forward sparsity prior. \textbf{E,F}, Separate raw-additive historical noise cohort with sixteen terminals: all four D4-minus-D1 accuracy contrasts are negative in 10/10 paired seeds; active contact counts are approximately fixed while parameter counts vary. Learning contrasts use means and 95\% paired-seed bootstrap intervals; map summaries use mean $\pm$ s.d. The noise-task executed generator remains unresolved and its shunting runs fail the positive-conductance input criterion. These retained additive outcomes are a depth boundary, not evidence for shunting; exclusions are enumerated in Table~\ref{tab:input_validity}.'''),
('branch_conflict_controls','si_03_conflict_ancestry',[('S19','AB'),('S29','DEF')],
 'Branch-selective credit limits interference and preserves update direction.',
 r'''\textbf{A,B}, Two-branch learning and context-0 forgetting in ten paired seeds. Small forgetting after deranged feedback reflects its poor initial solution. Points show seeds; large symbols and bars are means and 95\% paired-seed bootstrap intervals. Exact transport and the equivalent gated-point rule supply branch selection. \textbf{C--E}, Shared-versus-exact gradient cosine in the separate branch-conflict experiment with two, four or eight branches. Solid curves evaluate common exact-learning states at epochs 0, 50 and 250; dashed curves evaluate the shared learner's own final state. Curves and shading use twenty paired seeds and 95\% bootstrap intervals. An exact quadratic interference calculation is given in the accompanying derivation. Learned-state geometry can depart from the initialization mean-field approximation.'''),
('ancestry_coefficients','si_03_conflict_ancestry',[('S23','A'),('S32','ABF')],
 'Available route span and learned route coefficients are different constraints.',
 r'''\textbf{A}, Trained field capture by route budget in the eight-context factorial. Ancestry, depth and random partitions have coincident capture at matched bandwidth; derangement removes the target coordinate. Thus capture does not explain the small ancestry-specific advantage. \textbf{B}, An explicit context cue trains a four-output encoder, frozen before task learning. \textbf{C}, Cue-noise sensitivity; dashed starred curves use a post hoc hard readout of the same encoder. \textbf{D}, Primary soft-encoder-minus-control accuracy contrasts. The primary encoder is about 54.47 points below the oracle. With noisy calibration cues, hard selection narrows but does not eliminate this gap; it matches the oracle in the noiseless condition. All curves/differences retain twenty paired seeds and 95\% bootstrap intervals. The calibration cue and activation targets are supplied resources. No serial forward dendritic computation is present.'''),
('scalar_tree_capacity','si_04_interactions_boolean',[('S36','AB'),('S37','BC')],
 'Interaction structure constrains scalar trees at matched input spectra and resources.',
 r'''\textbf{A}, Exhaustive pairwise matching and two-quartic target families on eight independent binary inputs; both have input-gradient second moment $I_8/4$. Each candidate has seven scalar multi-affine nodes, 28 coefficients and fourteen edges. \textbf{B}, Full rank-two and centered rank-one cut-tail lower bounds versus achieved population NMSE in four-restart alternating least squares. \textbf{C}, Exact compatible construction for the first enumerated nested target. \textbf{D}, Best centered-cut bound over all labeled trees at each depth limit. Matching/quartic targets admit exact depth-three trees; nested controls require depth four and have a different input spectrum. All 164 targets construct with error below $7\times10^{-30}$ and coefficients of magnitude at most one. These exhaustive capacity results have no sampling error bars. The theorem concerns scalar multi-affine composition, not arbitrary conductance dendrites.'''),
('oracle_profile_credit','si_04_interactions_boolean',[('S40','*')],None,None),
('boolean_capacity','si_04_interactions_boolean',[('S43','*')],None,None),
('boolean_learning','si_04_interactions_boolean',[('S44','*')],None,None),
('fixed_profile_budget','si_04_interactions_boolean',[('S54','BDEF')],
 'Longer matched budgets retain the task-dependent fixed-profile deficit.',
 r'''\textbf{A,B}, Pairwise and quartic targets under the common Adam rate, with exact, unit-broadcast, initial-profile and initial-sign credit. \textbf{C}, Quartic-minus-pairwise difference in calibrated-minus-exact NMSE across four budgets: circles use terminal states and squares validation-selected states, at selected and common rates. \textbf{D}, Every exact and calibrated quartic seed at the original 1,024-update and extended 16,384-update endpoints. Curves show means and descriptive 95\% whole-seed bootstrap intervals. All 720 trajectories retain the previously observed twenty seed blocks, three tasks, four rules, two optimizers and original rate choices. The complete original endpoints replay exactly. Slow exact quartic fits resolve with more updates; restricted profiles retain a large deficit. Rates remain fixed and coefficients bounded; this is a retrospective-cohort extension under a prospective protocol, not fresh confirmation.'''),
('credit_optimizer_controls','si_04_interactions_boolean',[('S55','*')],None,None),
('conductance_grouping','si_05_conductance_learning',[('S42','*')],None,None),
('conductance_precision','si_05_conductance_learning',[('S50','*')],None,None),
('conductance_optimization','si_05_conductance_learning',[('S51','BC'),('S52','AB')],
 'Parameter-range and rate controls retain the opposed-task broadcast deficit.',
 r'''\textbf{A}, Task-by-credit interaction under original $[-7,7]$ and wider $[-20,20]$ log-conductance bounds, with Adam and SGD. All fits retain their original initialization/minibatch streams and extend to 16,384 updates; endpoints use validation-selected checkpoints. Dots are twenty paired seeds and intervals are 95\% paired bootstrap intervals. Wider-bound interactions are positive in all seeds. \textbf{B}, Initial-profile-minus-exact NMSE before each calibrated-profile fit's first original bound contact; all twenty differences are already positive. \textbf{C,D}, Equally budgeted six-rate development sweeps for opposed tuning under Adam/SGD; corresponding aligned-task outcomes are included in the full condition table. Each point averages three development seeds; all 288 outcomes are retained. The predefined improvement trigger was not reached, so rates were not retuned on fresh test outcomes. Some broadcast fits also reach wider bounds; these are finite-budget controls, not an asymptotic convergence proof.'''),
('local_gate_controls','si_05_conductance_learning',[('S53','ABCD')],
 'Local conductance gating is robust to rate choice but depends on where it is applied.',
 r'''\textbf{A,B}, Aligned tuning at 4,096 and 16,384 updates. \textbf{C,D}, Opposed tuning at the same budgets. Every rule receives every displayed Adam rate; 0.03 is primary. Entries are mean test NMSE at validation-selected checkpoints across twenty new paired seeds; the color scale shows base-ten logarithm. Exact credit, the three-pattern oracle, the local distal gate and the shunt-proportional gate learn accurately. Swapping the gate fails; extending the gate to proximal compartments leaves a substantial opposed-task deficit. The two leaf patterns with unit proximal feedback isolate this placement effect. These are supplied-context, bounded-conductance experiments; the local gate does not estimate arbitrary route coefficients. All rates, seeds and budgets remain included.'''),
('physical_architecture','si_06_physical_depth',[('S31','BCD G'.replace(' ','')),('S18','AB')],
 'Task-matched serial computation differs from grouped and flexible point controls.',
 r'''\textbf{A}, Serial D3, resource-identical grouped-point and approximately parameter-matched flexible point networks. Only serial/grouped models share modules and contacts. \textbf{B}, Original-budget three-level aligned task across D1--D3 and feedback rules. \textbf{C}, Four-level task across D1--D4 under aligned/reversed sensor placement. \textbf{D}, Serial-minus-grouped accuracy at fixed D3 across nested, flat and local-ratio task families, under exact-path LocalCA. \textbf{E}, Serial-minus-grouped-star effects and their alignment interaction. \textbf{F}, Flexible point controls matched to active or total parameter count. Curves and differences use ten paired seeds per cohort and 95\% paired-seed bootstrap intervals. These task-calibrated, original-budget comparisons establish forward-computation dependence on task/placement, not an optimizer-independent requirement for resolved credit. Main Fig.~6 shows the longer-budget accuracy reversal and cross-entropy trajectory.'''),
('physical_calibration','si_06_physical_depth',[('S15','*')],None,None),
('physical_optimizer','si_06_physical_depth',[('S18','EF'),('S28','B')],
 'Physical-depth conclusions depend on alignment, task hierarchy and optimization.',
 r'''\textbf{A}, Serial/point contrasts under aligned and reversed placement. \textbf{B}, Separate two-level task: depth-by-alignment interactions for serial/grouped-point backpropagation and shared/exact LocalCA. \textbf{C}, Original-budget D3 optimizer/credit-coordinate contrasts. Under the BP optimizer, full versus soma-broadcast differentiation differs by 0.64 points; under the split LocalCA optimizer, exact versus shared differs by 10.96 points; changing the broadcast optimizer costs 13.47 points. Means and intervals use ten paired seeds and 95\% bootstrap intervals. These earlier recipe-specific effects do not imply a converged credit advantage: at 600 epochs the LocalCA accuracy difference reverses, while exact credit retains lower cross-entropy (main Fig.~6). All D3 continuations hit the cap; D1 usually stops early.'''),
('anatomy_capacity_controls','si_07_anatomy',[('S10','B'),('S20','B'),('S27','BD')],
 'Cohort, label and arbor-size controls bound modeled anatomical capacity.',
 r'''\textbf{A}, Route-generated field capture versus budget in 47 disjoint same-mouse cells; morphology routes use 8.1\% of dense-feedback wiring at eight channels. Twenty modeled streams are averaged within each cell; bands/bars are 95\% cell-bootstrap intervals. \textbf{B}, Initial eight-cell cohort: capture restricted to directly typed presynaptic contacts. \textbf{C,D}, Second-mouse Pinky cohort: relative residual norm versus route budget, and candidate-site count versus effective field rank. Eligibility changes from ten cells at $K\leq4$ to eight at $K=8$; small arbors approach the rank ceiling. Panel C uses mean $\pm$ s.e.m., and its residual norm is not squared-energy capture. All fields are model generated. Main Fig.~7 equalizes a common broadcast and leads with actual-tree surrogate controls; these route-generated comparisons are structural positive controls, not measured biological teaching signals.'''),
('inhibitory_spatial_controls','si_07_anatomy',[('S12','BDEFGH')],
 'Spatial matching limits apparent inhibitory targeting of ancestry domains.',
 r'''\textbf{A}, Six layer-diverse reconstructions with mapped inhibitory contacts and physical scale bars. \textbf{B--D}, Observed-minus-matched tree distance, shared-path fraction and descendant-domain overlap in the twenty-target analysis, after aggregation within 92 presynaptic axons, and after joint path-distance/Euclidean matching. Symbols and intervals are means and 95\% bootstrap intervals over the displayed inferential unit. Only tree distance persists after axon aggregation; no endpoint persists after joint 3D matching. \textbf{E}, Descendant-input fraction by distal-dendrite- versus perisomatic-targeting class, retaining one contact per axonal clump. \textbf{F}, Inhibitory-contact domain size versus matched locations; no reliable overall localization advantage is detected. Cells derive from one mouse. These structural comparisons do not show that contacts carried teaching signals.'''),
('coarse_energy','si_07_anatomy',[('S25','*')],None,None),
('anatomy_preprocessing','si_07_anatomy',[('S33','*')],None,None),
('shunt_sensitivity','si_08_shunting',[('S11','A'),('S21','DC'),('S48','*')],
 'Dose, cable parameters and input mapping determine focal-shunt selectivity.',
 r'''\textbf{A}, Shunt-minus-current localization across absolute doses at three membrane resistances. \textbf{B}, All mapped versus directly typed presynaptic contacts. \textbf{C}, Excitatory/inhibitory scale and inhibitory-reversal sensitivity. \textbf{D}, Normalized-dose comparison of focal shunting with baseline-current-matched injection. Sites are averaged within the initial eight reconstructed cells; means and 95\% cell-bootstrap intervals are shown in all panels. Positive localization is a larger descendant change, not a sign of enhancement; passive shunts attenuate descendant adjoints. Absolute dose, input-conductance dose and local nonaxial normalized dose are distinct. Electrical parameters are sensitivity settings, not cell-specific fits.'''),
('shunt_replication','si_08_shunting',[('S10','FGH'),('S47','AB')],
 'Focal-shunt controls replicate structurally and under weak-channel linearization.',
 r'''\textbf{A}, Focal shunt versus baseline-current-matched injection in 45 disjoint same-mouse cells (235 sites). \textbf{B}, True versus reassigned descendant relations in forty cells (230 sites). \textbf{C}, Direct E/I label coverage and focal-site eligibility. \textbf{D,E}, Separate initial eight-cell weak-channel ensemble: localization versus dose and paired shunt-minus-current contrasts. A fixed steady-state Jacobian is compared with the passive model; shunting attenuates descendants whereas matched current injection enhances them. Sixty-four channel draws and sites are averaged within each cell. All intervals bootstrap cells, not sites or channel draws. Weak conductances test local linearization near the passive response; they do not establish robustness to strongly regenerative dynamics, channel kinetics or a newly solved nonlinear equilibrium at each intervention dose.'''),
('measured_topology','si_09_measured',[('S22','ABC'),('S56','C')],
 'Measured topology associations have limited target and response reliability.',
 r'''\textbf{A}, Mapped presynaptic partner counts for seven targets. \textbf{B,C}, Target-level shared-ancestry/response association and alternative path, tree-distance, partial-correlation and same-major-branch measures. Partial correlations control Euclidean separation and soma-to-contact depth differences. The target, not the partner pair, is the inferential unit. \textbf{D}, Calibration of the sensitivity model against measured odd/even-repeat Spearman reliability for 125 partner records. Two negative estimates are assigned zero reliable variance; the identity line is not fitted. One thousand independent calibration datasets estimate model mean reliability. Main Fig.~9 reports alignment and detection sensitivity using all thirteen scans from these seven targets. The detected-effect scale is conditional on this covariance/noise model and sampling, not a universal detectable correlation.'''),
('measured_transfer_geometry','si_09_measured',[('M9','BCDE')],
 'The measured-response learning comparison is limited by transfer geometry.',
 r'''\textbf{A}, Actual four-route support for the scan selected by median mapped-input count, with identifier tie breaking; rows are input coordinates and columns selected routes. \textbf{B}, Input coverage in all thirteen scans, ordered by input count. The selected supports are mostly single distal sites, rather than illustrative nested bands. \textbf{C}, Response-prediction NMSE for exact learning, treeless ridge and restricted dictionaries; dots are seven target means, diamonds and bars their means and 95\% target-bootstrap intervals. The dotted line marks ridge. \textbf{D}, Common-checkpoint update reconstruction by the unrestricted fixed transfer profile, the restricted ancestry profile and oracle trialwise ancestry amplitudes. The unrestricted fixed profile is nearly sufficient because somatic error mainly scales a stable transfer vector; restricted rules underperform ridge. Scans and splits are averaged within targets before inference. This is an offline passive-tree geometry diagnostic, distinct from the empirical ancestry-response association.'''),
('measured_predictor_controls','si_09_measured',[('S34','*')],None,None),
('original_selector','si_10_selection_statistics',[('S35','*')],None,None),
('finite_horizon','si_10_selection_statistics',[('S38','ABC'),('S39','F')],
 'Finite-horizon prediction improves selection but strong simple baselines remain.',
 r'''\textbf{A,B}, Final costed-loss regret in feedback-only and joint-transfer fixed-cache learners. \textbf{C}, Strong selectors in the joint arm. The independently calibrated Gaussian SGD forecast improves over the original scalar utility; the full-batch and observed-context-count baselines remain competitive. All finite-horizon policies choose $K=r$ in this family, so improvement does not establish fine morphology selection. \textbf{D}, Population-oracle forecast minus realized population loss in the joint arm under fresh examples or finite-cache reuse. Curves and intervals resample twenty independent seed blocks with pointwise 95\% bootstrap intervals. The fresh-example recurrence is exact under the stated Gaussian/conditional-linear assumptions; cache reuse violates independence. All outcomes and baseline definitions remain included. These are linear-learning forecast tests, not a general nonlinear dendritic morphology law.'''),
('morphology_estimation','si_10_selection_statistics',[('S41','*')],None,None),
]

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
 'same_span_conditioning': [('S14',[155,168,371,184])],
 'ancestry_coefficients': [('S32',[100,0,422,34])],
 'mnist_dictionary_geometry': [('S49',[107,484,514,504])],
}
WHOLE_CROPS = {'local_gate_controls': [0,0,518.4,443]}

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
 'origin':[336,385], 'fontsize':7.6, 'rotate':90, 'role':'shared y-axis label'}]}
REMOVED_SHARED_REGIONS['S27']=[[170,188.2,408,206.4]]
SHARED_LEGENDS['anatomy_capacity_controls']=[('S27',[170,188.2,408,206.4])]
SHARED_LEGENDS['fixed_profile_budget']=[('S54',[175,32,253,66])]
WHOLE_CROPS['local_gate_controls']=[0,0,518.4,452]
PANEL_BOUNDS[('S31','B')]=[310,4.5,516,136]
PANEL_BOUNDS[('S20','B')]=[256,7.5,502,191]
PANEL_REDACTIONS[('S46','E')]=[[314.8,168.8,343,177.6]]
REMOVED_SHARED_REGIONS['S19']=[[217.9,232.3,374.1,241]]
PANEL_BOUNDS[('M9','E')]=[275.5,283.5,502,455]
PANEL_REDACTIONS.pop(('M9','D'),None)
PANEL_BOUNDS[('S31','G')]=[327,283,516,408]
PANEL_TEXT[('S31','G')][0]['text']='Serial minus grouped (pp)'
REMOVED_SHARED_REGIONS['S14'].append([346,334,475,350])
SHARED_LEGENDS.pop('same_span_conditioning',None)
NATIVE_LEGENDS={'same_span_conditioning':[
 {'label':'orthonormal Haar','color':[0.20,0.65,0.42]},
 {'label':'raw nested','color':[0.88,0.49,0.40]},
 {'label':'scaled nested','color':[0.12,0.32,0.68]}]}
PANEL_REDACTIONS[('S19','A')]=[[146.5,224.6,223,234.8]]
PANEL_TEXT[('S19','A')]=[{'text':'Held-out accuracy','origin':[146.8,232.2], 'fontsize':8.4,'role':'restored x-axis label'}]
REMOVED_SHARED_REGIONS['S10']=[[210.8,410.1,274.4,428.4]]
PANEL_BOUNDS[('S10','G')]=[13.6,302.6,267,431]
PANEL_BOUNDS[('S10','H')]=[274.5,302.6,515,432]
PANEL_TEXT[('S10','G')]=[
 {'text':'true','origin':[235,417], 'fontsize':7.6,'role':'compact x-axis label, line 1'},
 {'text':'descendants','origin':[219,426], 'fontsize':7.6,'role':'compact x-axis label, line 2'}]
PANEL_BOUNDS[('S52','A')]=[4,6.6,251.1,200.2]
PANEL_REDACTIONS[('S52','A')]=[[251,174,261,191]]
PANEL_REDACTIONS[('S31','G')].append([343,282,518,296])
PANEL_TEXT[('S31','G')]=[{'text':'Exact LocalCA: serial advantage (pp)',
 'origin':[329,292], 'fontsize':8.0, 'role':'quantity and optimizer in panel title'}]
PANEL_BOUNDS[('S10','H')]=[265,302.6,515,432]
PANEL_BOUNDS[('S31','G')]=[342,283,516,408]
PANEL_TEXT[('S31','G')][0]['origin']=[344,292]
PANEL_TEXT[('S31','G')][0]['fontsize']=7.6

# This source ledger originally named only the rendered asset. The scientific
# builder explicitly reads these two tables; preserve that numerical closure.
EXPLICIT_NUMERICAL_SOURCES = {
 ('S25','B'):['source_data/irregular_tree_wavelets/cohort_scale_summary.csv'],
 ('S25','C'):['source_data/irregular_tree_wavelets/cell_scale_summary.csv',
              'source_data/irregular_tree_wavelets/cohort_scale_summary.csv'],
 ('S25','D'):['source_data/irregular_tree_wavelets/cohort_scale_summary.csv'],
}
