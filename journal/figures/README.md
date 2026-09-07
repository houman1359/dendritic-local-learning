# Figure directory

Article: *Dendritic morphology as a dictionary for local credit assignment*.

The main manuscript includes exactly nine publication assets, `main/figure_01.pdf` through `main/figure_09.pdf`. The Supplementary Information includes 52 figures, S1–S52. Native components, generated intermediates and historical assets are inputs or retained outputs, not additional numbered figures.

`../scripts/rebuild_final_publication_figures.py` defines the current main mapping and renders the supplementary sequence. Historical builder numbers differ from publication numbers. Use this entry point for the complete figure build; the older compositor mapping is not the current main sequence.

## Main figures

| Figure | Content | Native builder |
|---|---|---|
| 1 | Credit framework, neuronal teaching coordinates and image-task reference | `credit_first_figures/build_framework.py` |
| 2 | Context-gated branch conflict and the branch-selection boundary | `build_main_figure_04.py` |
| 3 | Ancestry at matched bandwidth, generator coefficient prediction and paired controls | `credit_first_figures/build_ancestry.py` |
| 4 | Interaction order, fixed-profile learning and learned credit geometry on matched trees | `credit_rule_bridge/build_figure.py` |
| 5 | Context-dependent feature tuning and credit in conductance trees | `conductance_credit_demand/build_opponent_figure.py` |
| 6 | Physical depth, optimizer dependence and extended training budgets | `physical_depth_budget/build_main_figure5.py` |
| 7 | Anatomical dictionaries beyond a shared broadcast | `credit_first_figures/build_anatomy.py` |
| 8 | Focal shunting as ancestry-partition gain and the electrotonic boundary | `shunt_ancestry_gain/build_figure.py` |
| 9 | Empirical ancestry–response similarity and a separate transfer-geometry diagnostic | `credit_first_figures/build_measured.py` |

All builder paths are relative to `../scripts/`. Figure 3 makes the coefficient-sign prediction explicit rather than presenting its bandwidth profile as an independently discovered optimum. Figure 4 compares pairwise and quartic targets with matched tree, initialization and input streams; it distinguishes a fixed initial profile from a fitted rank-one oracle. Figure 7 gives every dictionary access to the same broadcast component. Figure 8 reserves weak-channel linearization for S47. Figure 9 plots actual selected route support.

## Supplementary figures

S1–S17 retain the foundations and initial mechanism/context controls. S18 contains expanded physical controls; S19 within-neuron addressing; S20 anatomy; S21 focal shunting; S22 measured responses; S23 partition residuals; S24 adaptive reliability; S25 irregular-tree wavelets; S26 the 430 H2/H3 physical-depth reruns; S27 second-mouse structural capacity; S28 credit-coordinate comparisons; S29 branch-conflict trajectories; S30 between-neuron by within-arbor feedback; and S31 the earlier physical-depth/optimizer controls.

S32 covers local coefficient estimators; S33 label/morphology sensitivity; S34 response baselines; S35 the unsuccessful prospective initialization-based selector; S36–S37 interaction capacity and constructive morphology; S38–S39 finite-horizon forecasts; S40 credit learning; S41 finite calibration and separate end-to-end learning; S42 positive-conductance grouping; S43 Boolean capacities and gate derivatives; and S44 Boolean learning. The separate 320-fit clean exact-transport/BP control is Supplementary Table S18; its noise cohort is the three-class noisy-line task.

S45 contains expanded image-task/gradient diagnostics (`figure_S45_image_diagnostics.pdf`); S46 contains the one-step utility analysis and designed signal/noise screens (`figure_S46_utility.pdf`); and S47 contains the weak-channel local-linearization check (`figure_shunt_weak_channels.pdf`). S48 retains the normalized shunt dose curve; S49 contains the six-arm MNIST dictionary, rate-selection and decoder controls; S50 retains the first conductance task's small precision effect; S51 reports the opponent-tuning optimization and parameter-range controls; S52 shows the equally budgeted expanded learning-rate grid. Captions and methods fragments are included in the Supplementary Information.

Several older filenames retain panel suffixes from previous layouts. The current TeX legends and actual PDF panel letters define the display, not those historical suffixes. In particular, `figure_S31_panels_A-G.pdf` contains A–H. Rebuilding the final assets does not rename these retained files.

## Numerical sources and figure checks

The current panel inventory is `../source_data/credit_first_provenance/source_inventory.tsv`. New main evidence is retained in `credit_rule_bridge`, `conductance_credit_demand`, `image_ladder_controls`, `anatomy_commonmode`, `physical_depth_budget` and `shunt_ancestry_gain`, with earlier cohorts preserved in their existing Source Data folders. Development, fresh-test, replay and descriptive scopes are identified by each protocol and caption.

The Source Data package manifest assigns files to current displays while preserving `original_source`. Some former main-only evidence is retained under `Methods/retained_evidence/`; restore it to its original analysis path through the manifest. Display-specific row filters remain explicit. A display subset must not silently replace a complete source table during reconstruction.

Final presentation checks should use the actual compiled assets: panel order and callouts, embedded fonts, text size at print scale, alignment, overlaps and clipping. Final PDF page counts come from the compiled manuscripts. Neither an older audit record nor a successfully generated asset establishes that the current complete bundle has passed these checks.

Supplementary Figure S48 is a single-panel dose curve printed at its native 3.6-inch width. This preserves the shared text and line sizes; the other supplementary sheets use the full manuscript width. The style audit checks this explicit native-width placement.
