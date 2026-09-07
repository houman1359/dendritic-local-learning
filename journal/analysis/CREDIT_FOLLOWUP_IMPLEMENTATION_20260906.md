# Follow-up to the credit-first panel review

This revision preserves **Dendritic morphology as a dictionary for local credit assignment** and the task-to-credit sequence of the reviewed `cf9fda8` release. The new work tests the review's remaining scientific concern in a directed conductance tree and fills the missing MNIST dictionary, rate and decoder comparisons. It does not reinstate the failed initialization-based morphology selector or the reversed physical-depth accuracy effect.

This is an internal implementation and verification record, excluded from reviewer archives. All prespecified scientific fits and checkpoint captures are complete. Final integration and release identities are recorded separately in `review_followup_20260906/final_release_validation.json` after packaging.

## Conductance credit: what the new result establishes

The new main Figure 5 compares aligned and opposed feature tuning in the same seven-compartment directed E/I conductance architecture. Teacher and student use the same equations, establishing representability. All 24 positive student conductances are plastic. Paired tasks share input samples, student initialization and minibatch streams. Both subtrees receive the same positive features; binary context opens inhibitory conductance in one proximal compartment. The opposed teacher gives the two subtrees opposing feature preferences.

Twenty fresh paired seed blocks compare exact credit, unit broadcast, an initially calibrated fixed profile, and a three-pattern dictionary whose coefficients are projections of the exact field. All local synaptic and coupling eligibilities are retained. The spatial patterns distinguish the two terminal groups and the proximal pair. This is an oracle test of the dictionary's delivery capacity; it does not supply a biological encoder or imply a need for six independent errors. Binary context generates at most two path profiles at fixed parameters.

Under Adam, the opposed-minus-aligned difference in calibrated-minus-exact test NMSE is **0.663817 [0.510621, 0.804260]** at validation-selected states within 4,096 updates and **0.627764 [0.463911, 0.774494]** within 16,384 updates. Both contrasts are positive in all 20 paired blocks. The longer-budget opposed-task means are approximately **0.000000248 exact**, **0.000000903 three-pattern oracle** and **0.628226 calibrated broadcast**. These are validation-selected endpoints; the plotted learning curves use their labeled fixed checkpoints.

The result also holds under SGD. A separate sensitivity restarts all conditions with log-conductance bounds widened from `[-7,7]` to `[-20,20]`; its Adam task-by-credit contrast is **0.626180 [0.462459, 0.773066]**. All 320 original-bound replay trajectories reproduce the final parameters exactly. Broadcast deficits already exist before first original-bound contact. Some broadcast parameters also reach the wider bounds, so this is a finite-budget, finite-range result rather than a convergence theorem.

At common broadcast-trained weights, exact and approximate gradients evaluated on the same examples show stronger cancellation across contexts under broadcast. These are gradients before optimizer transformation. Positive per-example gradient alignment can coexist with a poor gradient averaged over conflicting examples.

The initial 16-conductance study remains visible in S50. It produces a small positive extended-budget credit contrast, **0.000502 [0.000267, 0.000772]** NMSE, while mean broadcast error under Adam is below 0.0006. This motivates the opponent-tuning construction without presenting the first study as a literal null. A retained moderate-gating development regime also produces broadcast failure despite roughly 0.98 rank-one teacher path capture. Raw path-energy capture alone therefore does not determine learning success.

Because every original selected learning rate lay at the upper tested boundary, a separately frozen expanded-rate check reruns all four rules, both primary tasks, both optimizers and six rates at the same 16,384-update budget. All 288 development outcomes are retained in S52. Larger rates do not rescue broadcast learning; neither declared trigger for a fresh follow-up is met. The matched-budget comparison and the original fresh results remain separate analyses.

Independent verification recomputes eight main conductance contrast means and 20,000-draw paired bootstrap intervals directly from retained endpoint rows, without importing the report aggregator. Every recomputed mean and interval equals its source summary. The expanded-rate audit independently reconstructs all 96 rate-group means from 288 endpoint rows (maximum discrepancy 2.22e-16) and confirms that both follow-up triggers are false. See `review_followup_20260906/conductance_opponent_recomputation.json` and `conductance_expanded_rate_recomputation.json`.

## MNIST: the trained dictionary comparison preserves the image result

All **108 development and 190 unique fresh fits** completed their 180 epochs. The 120 selected-rate and 120 common-rate outcomes reuse 50 identical fits; twelve short canaries remain excluded. Initial model hashes match across each seed's conditions, and every decoder-only core remains unchanged.

At development-selected rates, neuron-specific feedback improves on the strict scalar by **9.230 percentage points in shunting trees and 7.456 in additive trees**, reaching 97.124% and 97.114% accuracy. Decoder-only accuracies are 80.956% and 89.568%. Thus the network comparison includes a measured finite-budget reference for learning with the core frozen.

The direct spatial-bandwidth comparison, **three projected subtree profiles minus one projected profile**, is **+0.049 points [−0.048, 0.141] in shunting trees** and **+0.036 [−0.056, 0.122] in additive trees**. Common-original-rate differences are +0.006 [−0.035, 0.046] and −0.013 [−0.069, 0.047]. These intervals do not establish a universal equivalence result; they delimit the small accuracy differences in this design.

Exact transport minus the simpler neuron-shared rule is +0.010 [−0.062, 0.075] points in shunting trees and **+0.182 [0.114, 0.252] in additive trees**, positive in all ten additive seeds. That additive contrast is not a null. However, the single projected profile already attains 97.304% accuracy, compared with 97.296% exact and 97.340% with three profiles. The small improvement over the simpler shared signal therefore does not establish a benefit from separating subtrees.

All forty initial/trained checkpoints were analyzed on 2,048 held-out images. At exact-trained states, mean activation-field capture increases from **0.621653 to 0.654900** with one versus three profiles in shunting trees and from **0.763997 to 0.858719** in additive trees. Voltage-space capture, retained separately, increases from 0.486411 to 0.626754 and from 0.515771 to 0.714054. The historical fifteen-seed voltage-space atlas remains explicitly separate. New capture and learning use the same declared projection geometry; oracle coefficient access, differing field norms and exact local eligibilities are stated.

The optional forty-state checkpoint archive has 20 portable resolved configurations and unchanged binary weights. Full capture replay is a separate release check, not a new scientific cohort. A failed wrapper check exposed JSON exponents being parsed as strings through YAML; the correction is confined to the portable parser, with the original scientific training and capture code unchanged.

## Corrections to the remaining ledger

| Review item | Implemented correction |
|---|---|
| Conductance morphology never required spatial credit | New main Figure 5 tests task-dependent credit demand in a faithful directed E/I conductance balance, with a trained intermediate dictionary and the controls above. |
| Physical calibration described as reducing the effect | Main text now distinguishes spatial selectivity from overall gradient change. Main Figure 8D plots signed descendant and off-route changes; Figure 8E gives the electrotonic boundary. |
| No trained intermediate MNIST dictionary, rate sweep or decoder reference | Separately frozen six-rule comparison: strict scalar, neuron-shared, projected K1, subtree K3, exact and decoder-only. All 298 scientific fits complete; the spatial contrasts and decoder reference are reported above. |
| Exact image readout errors not stated | Results and Methods specify one dendritic layer and a linear readout computing exact neuronal output errors before the primary feedback intervention. Random-feedback controls are explicitly separate. |
| Deranged branch credit described as failing only at high conflict | Text now reports its 12–25 percentage-point deficit even at zero conflict. |
| Calibrated profile beats exact on pairwise targets | Adam's approximate invariance to fixed positive coordinate scaling explains calibrated/sign-only agreement. The text does not incorrectly attribute slower exact-path learning to that invariance. |
| Invisible physical-depth legend curve | Both nearly overlapping trajectories remain plotted with a joint legend entry and a caption explanation. |
| Missing historical capture values | Results restore the historical voltage-space K1/K3 captures: 0.480/0.625 for shunting and 0.519/0.718 for additive models. New activation-space dictionary capture is reported separately. |
| Missing software versions and MNIST parameters | Main Methods state the full benchmark budget, batch size, optimizer and parameter-group rates; actual scientific and clean reviewer environments are distinguished. |
| Fluorescence processing unspecified | Methods identify interval means of the supplied `RoiResponseSeries` traces, with no deconvolution or reliability correction performed in our analysis. Direct upstream metadata inspection identifies processed two-photon data but does not establish raw fluorescence or dF/F; the unsupported word “raw” was removed throughout. Evidence: `review_followup_20260906/dandi_fluorescence_metadata.json`. |
| Missing citations to S18, S19, S21 and S26 | Added alongside the relevant task and diagnostic descriptions. A further reference audit corrects the S29 branch-selection pointer, S46 theory-section pointer, S9 adjoint-panel pointer and the optional-preconditioner section pointer. |
| Unverified 0.838 algebraic interaction contrast | Independently recovered from `credit_rule_bridge/summaries/paired_seed_contrasts.csv`: 0.837817 [0.719044, 0.930753], positive in all 20 blocks. The capture directory was not the endpoint source. |
| Shared checkout behind reviewed release | The paper `main` now has the reachable reviewed release as its parent, preserving working files. Tags retain both the reviewed state and previous main state. The final integration commit and rebuilt archive identities are recorded separately after packaging. |

## Scope retained throughout

The main narrative remains a progression in useful credit distinctions: neuronal identity, contextual selection, ancestry grouping, interaction-dependent profiles and conductance-mediated branch credit. Anatomy supplies candidate spatial patterns; shunts can change gains on the appropriately weighted ancestry partition. The measured ancestry-response test remains null, and the measured learning comparison remains a transfer-geometry diagnostic.

The 600-epoch physical-depth result remains a reversal of the earlier large accuracy advantage for exact LocalCA credit. All D3 trajectories still reach that budget, and accuracy and cross-entropy give different comparisons. The initialization-based selector still loses to the fixed tree in all 20 seeds. Neither result is removed or reframed as a success.

The final paper is intended as the Nature Communications journal manuscript. Submission, author declarations, third-party status changes and any repository upload are not represented as completed actions.

## Final integration and release checks

The draft software preview passes archive policy, membership and checksum checks. Separate clean-CPU fixtures verify actual sanitized software and partial Source Data restoration for both new studies, preserving every preview byte. The conductance fixture accepts declared path relocation and rejects a changed scientific protocol even with a valid declared hash chain. The image fixture verifies 502 historical Python sources and 310 configurations; K3 and projected-K1 canaries reproduce the corresponding unsanitized CPU trajectories. An independent adapter audit verifies projection geometry, gate and synaptic updates, somatic handling and decoder-only freezing. K1 and K3 preserve the same common-mode mean at identical parameters, not necessarily the same field norm. Final full-package validation remains distinct from these bounded preview checks.

All scientific fits, checkpoint capture and publication rendering are complete. The main manuscript has 31 pages, approximately 6,400 narrative words and nine main figures; Supplementary Information has 145 pages and 52 figures. The full manuscript suite passes 182 tests with one skip. Main and supplementary source audits report no errors or warnings; layout, citations, overlaps, style lineage, working length target and reproducibility checks pass. The independent MNIST audit reproduces all 298 canonical endpoints and derived contrasts, and all forty checkpoint states replay with zero capture or forward difference.

The final render amendment preserves the historical conductance inventory and explicitly records four presentation-identity changes. S52 changes only renderer metadata. The main Figure 5 plots, source tables and scientific inputs are unchanged; its current provenance reproduces exactly, although the historical provenance file is unavailable for a field-level comparison. The current inventory verifies all 1,055 entries.

The release procedure commits the integrated scientific inputs, builds all four archives from a clean named worktree, and executes restoration and portable checks only in disposable copies. Final full-package validation and archive identities are recorded separately in `review_followup_20260906/final_release_validation.json`. Earlier archives at `cf9fda8` remain identifiable and must not be mistaken for this revision.
