# Credit-first revision, 6 September 2026

This record documents the historical reviewed release `cf9fda8`. The subsequent implementation and current release checks are documented in `CREDIT_FOLLOWUP_IMPLEMENTATION_20260906.md`.

The paper keeps its original question: which spatial distinctions must useful dendritic credit preserve for a task? The retained title is **Dendritic morphology as a dictionary for local credit assignment**. Forward representability is a prerequisite for the learning comparisons; the noisy-query structure estimator remains supplementary evidence within this same Article. It does not replace the task-to-credit argument.

The new evidence supports that decision. It does not restore the failed initialization selector, and it does not establish a universal law from scalar task rank to optimal morphology.

## What the new experiments establish

| Comparison | Evidence | Interpretation in the revised paper |
|---|---|---|
| Matched pairwise versus quartic tasks | One compatible tree, equal input-sensitivity second moments, paired weights/data/minibatches; 432 development and 1,200 fresh fits. Exact versus initially calibrated Adam NMSE: 0.0317 versus 0.0236 for pairwise, 0.1516 versus 0.9813 for quartic. The task-by-credit contrast is 0.838 [95% paired interval 0.719, 0.931], positive in all twenty seeds; a common rate gives 0.843 [0.727, 0.935]. | A fixed spatial profile can support pairwise learning but leaves a large higher-order error gap in this controlled model and training budget. |
| Credit geometry of those same learners | At exact-trained states, best rank-one path capture is 99.7% for pairwise, 46.1% for quartic and 44.8% for the separate nested target. Loss-weighted, eligibility-weighted and site-normalized checks retain the separation. Independent replay of 600 earlier fits reproduces their recorded metrics. | Identical input spectra can accompany different learned credit spectra. Uniform-broadcast capture does **not** separate the tasks; the relevant profile can be signed and nonuniform. This is learned-state geometry, not an initialization selector or proof of the minimum number of external error signals. |
| Positive-conductance boundary | Every tested fixed-profile rule learns the compatible positive-conductance teacher accurately with Adam. It is a separate teacher, not an implementation of the algebraic quartic target. | The higher-order result is not promoted to a universal conductance-neuron law. |
| Anatomical routes beyond broadcast | Every family spends one of its K columns on the same constant component. At K=8 in 47 disjoint v661 cells, ancestry/depth-bin total capture is 0.619/0.511 and residual capture is 0.527/0.392. The total-capture difference is 0.108 [0.077, 0.135], positive in 45/47 cells. Ancestry also exceeds degree–depth surrogates by 0.041 [0.020, 0.062]. | Anatomy supplies additional spatial capacity after the original common-mode mismatch is repaired. Actual rank and wiring costs remain explicit. Cohorts and mice are not pooled. All 65 operators and 149,331 dictionary evaluations are retained. |
| Focal shunt identity | Exact passive-tree algebra and numerical checks show a common adjoint factor within each block of the shunt site's ancestry partition. | A focal shunt implements gains on the **baseline-weighted ancestry partition**. This unifies conductance with the dictionary framework; it is not a diagonal action on an arbitrary overlapping dictionary, and local driving forces can vary within a block. |
| Physical-depth budget extension | Sixty paired restarts preserve the original recipes and stopping rule, extending the maximum to 600 epochs. The exact-BP D3–D1 accuracy advantage grows from 30.82 to 38.17 points. The LocalCA exact-minus-shared accuracy contrast changes from +10.86 to −1.52 points, reversing in all ten seeds; the BP contrast is +0.48 points. | Forward-depth benefit persists. The original LocalCA credit advantage is a finite-budget accuracy result. A secondary comparison retains lower exact-credit cross-entropy (0.254 versus 0.276), so metric and budget both matter. All fifty D3 fits still reach the 600-epoch cap with declining validation loss; no convergence claim is made. |

At fixed weights and inputs, path derivatives do not depend on the labels. Eighty relabeling controls confirm this. The capture analysis therefore explicitly separates the initial state, each rule's own trajectory, and common reference states. An oracle rank-one profile measures capacity; it is not a newly trained encoder.

## The resulting main sequence

1. Framework and image tasks: preserving neuronal identity accounts for most of the feedback benefit; exact path resolution adds little and slightly reduces flattened-CIFAR-10 accuracy.
2. Context conflict: branch-selective information prevents harmful off-route updates; the analytic threshold and equivalent eligibility-gating interpretation are explicit.
3. Ancestry at matched bandwidth: the coefficient-derived K profile precedes the 1.27-point K=4 result. The location follows from the generator and is not presented as a discovered optimum.
4. Interaction order and credit: compatible same-tree tasks, paired learning curves, common-rate contrast and learned-state capture form the central new evidence.
5. Serial physical depth and learning budget: promoted from S31, with optimizer and longer-training comparisons.
6. Anatomical dictionaries beyond a common broadcast: original, disjoint-cell and second-mouse cohorts shown separately, with rank and cost.
7. Shunting as ancestry gain: the electrotonic boundary receives a wide quantitative axis; the weak-channel ensemble is labeled a linearization check and moved to S47.
8. Measured responses: empirical ancestry–response similarity is separated from the offline transfer-geometry comparison. The support matrix is reconstructed from actual saved routes; coverage counts mapped partner inputs, some sharing physical segments.

The old complete image diagnostic sheet is S45; the local utility/noise screens are S46. The conceptual evidence map and final checklist panel are removed. The substantive forward-construction, noisy-query selection and Boolean analyses remain in the supplement.

## Writing and corrections

The abstract has five sentences and leads with credit. The final narrative is approximately **6,433 words**, with **2,690 words of Methods** and a **140-word abstract**. This meets the agreed 6,000–6,500-word working target; the journal’s approximately 5,000-word narrative guidance remains an advisory. The Discussion synthesizes the task-to-credit evidence rather than walking through an equal-weight inventory of sections.

Variables, task generators, panel citations and equation introductions were checked. The matched interaction experiment now defines pair/quartet notation and the nested target in the main text. The shunt adjoint is defined physically. Repeated limitations are consolidated around the corresponding claim.

The main K=4 statistic consistently uses the Holm-adjusted paired Wilcoxon value, P=0.0101; the alternative paired mean-sign-flip test remains explicitly identified in SI. S8F uses correctly registered numeric depths. The caption typo and Greedy DOI are corrected. The predecessor credit preprint is cited in the main text and the inherited foundations are disclosed.

The noise-task audit found a genuine historical naming collision. The clean 320-fit exact/BP control in **Supplementary Table S18** used a three-class noisy-line generator, with the frozen dispatcher's unusually large 784-pixel side length verified against checkpoint dimensions. It is not corrupted MNIST. Other historical cohorts retain an intended projected-noise MNIST protocol but lack the executed nested-generator hash; inherited S1/S3 aggregates have incomplete joins. The paper states these limits and the released loader requires explicit generator names. S26 is the separate 430-fit H2/H3 physical reproduction and must not be confused with this control.

## Preservation and release verification

The preceding complete paper snapshot is reachable as `credit-paper-baseline-20260906` at `5483817839c16a9d48c343062dee4f6340ed84cd`. An additional revision checkpoint is reachable as `credit-first-revision-checkpoint-20260906` at `e30b10b114170e654a78558ec83060253dec980b`. The shared working checkout and its index were left intact.

The reviewer software is rebuilt from explicit committed allowlists, excluding the unrelated presentation, experiment runners and internal revision logs. A clean CPU virtual environment without system site-packages installed the pinned dependencies, passed pip check and passed installation/dispatch tests. Source Data retains complete outcome ledgers as well as display-specific subsets. Original and portable-copy hashes are recorded separately; restoration checks the full chain and refuses incomplete reconstruction.

The full depth extension is complete: 60 paired restarts, 34,159 observed epochs, and 208 compact export files, including complete histories and all 120 configuration copies. The historical runtime was preserved at the reachable a99c3a7 commit. Two excluded one-epoch release smokes passed for exact BP and LocalCA.

Final integration commit, test counts, archive names and checksums are recorded in the release-validation section below. Author-specific declarations, official conference status and immutable public deposition identifiers remain matters for the actual submission; this work does not assert a rejection, withdrawal or journal submission.


## Final release validation

The main PDF has 29 pages and the supplement 138 pages (167-page combined reading copy). All eight main figures and 47 supplementary figures are included. Both TeX builds finish without warnings or box/float overflows; the combined PDF preserves all page text, links and bookmarks. The format, citation, overlap, figure-lineage, main/SI provenance, additive-reference and reproducibility checks pass. The final provenance inventory covers 2,785 records and 2,713 distinct files.

The final committed build and archive validation record is written separately to `analysis/CREDIT_FIRST_RELEASE_VALIDATION_20260906.json` so the frozen source does not contain a circular reference to its own commit or archive hashes. Generated ZIPs are excluded from version control and built from the clean, reachable release branch.

The scientific integration is preserved at `3aa6d72b3592e64e439419a86ffb0e10cfcf4036`. Subsequent release checks corrected the figure-count metadata and added ten committed historical generators plus one frozen scientific protocol to the software allowlist. A new preflight checks all 117 registered generator paths and the protocol against that allowlist. The final source is preserved on `credit-first-release-20260906` at `cf9fda8ad31453a7eed0c1e45c8bf420e5b19f7a`.


The final release checks are complete. **182 journal tests pass, with one optional historical-run archive check skipped**. The exact exported core installs in a clean environment and passes 16 targeted tests. Both strict restored-manuscript audits verify all 2,785 provenance records with zero errors or warnings. All 3,378 restored software/data hash links verify, and both historical depth launcher conditions verify all 476 runtime files.

The four final archives are installed under `submission/`: `Nature_Communications_Submission.zip`, `Overleaf_Project.zip`, `Source_Data.zip`, and `Dendritic_credit_assignment_software.zip`, each with a SHA-256 file. The cleaned software is 12.31 MB. All archives identify the same clean `cf9fda8` release, and the extracted Overleaf project reproduces every main/SI page in text and rendered appearance. Earlier generated packages were preserved separately under `analysis/previous_generated_release_*`; they are excluded from the submission.

The revision, experiments, figures and local release work are complete. Final author declarations, official-form confirmation, actual concurrent-submission status, immutable deposition identifiers and journal upload remain author submission steps.
