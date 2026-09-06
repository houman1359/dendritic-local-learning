# Draft answers for the official reporting forms

Article: **Dendritic morphology as a dictionary for local credit assignment**.
Prepared from the manuscript, executable protocols and frozen numerical record
on 6 September 2026. These are field-by-field technical answers for author
review. They do not confirm authorship, funding, conflicts, ethics approval,
exclusive consideration or public release identifiers.

Current display inventory: eight main figures and 47 supplementary figures. The credit-first revision adds 432 development and 1,200 fresh matched-credit fits, independent saved-state capture analysis, budget-matched anatomical common-mode comparisons across 65 modeled operators, and the exact passive ancestry-partition gain identity. Historical noise-task generators are distinguished by cohort; unresolved execution lineage remains explicit.

The original official PDFs are unchanged. The Machine Learning Checklist has
a populated AcroForm draft, `DRAFT_machine-learning-checklist_technical.pdf`.
Saved widget values were reopened in PyMuPDF and all three rendered pages
were visually inspected; author-controlled fields remain pending. The
validation JSON records the answer-document version used when filling it.
The Software Checklist and Reporting Summary are XFA forms that render only
a “Please wait” page in this environment; their field labels were read from
the embedded templates and answer transfer remains pending. None of the
forms has been verified in Adobe Acrobat Reader. The corresponding author
must review and approve the completed forms and verify the saved versions
in Reader before upload.

## Shared author fields

| Field | Draft response |
|---|---|
| Corresponding author(s) | Author confirmation required. Use the approved names and contact details in the final manuscript/portal. |
| Last updated by author(s) | Leave pending until the corresponding author reviews the completed form; do not substitute the automated preparation date for author approval. |
| Current related-manuscript status | The author intends this standalone Nature Communications Article as the sole publication from the project. The prior NeurIPS submission is expected not to be accepted but has not been officially rejected. Confirm its final decision/withdrawal and applicable exclusivity rules before submission. |
| Immutable code/data identifier | Final DOI/version/reviewer-access URL remains to be supplied. Local Source Data and software archives are prepared from explicit allowlists and hashes. |

## Machine Learning Checklist v1.1

| Item | Draft answer and supporting scope |
|---|---|
| 1. CodeOcean capsule | No capsule is established in the current record. Do not select this option unless one is created and approved. |
| 1. Source code | Yes: reviewer software archive containing the modeling implementation, article analyses, configurations, tests and provenance. Insert the final archive/repository identifier after the final build. |
| 1. Compiled standalone software | Not applicable: this is research source code, not a compiled standalone application. |
| 1. Test dataset and replication instructions | Yes: deterministic synthetic generators, numerical tests, source tables, public benchmark identifiers and stage-specific instructions. Restricted/raw upstream biological caches are not redistributed; their access conditions and identifiers are documented. |
| 1. README/install/run instructions | Yes: repository and article READMEs, reproducibility instructions, environment records and experiment-specific READMEs. A clean CPU virtual environment without system site-packages installed the pinned release dependencies, passed pip check and passed the focused installation/dispatch tests. Final archive hashes identify the committed source version. |
| 1. Reviewer access | Local review archives are prepared; confirm the actual uploaded archive or reviewer-access link at submission. |
| 1. Pretrained models | No externally pretrained foundation model is used. Archived experiment checkpoints are outputs of the reported models, not external pretraining. |
| 1. Post-publication code/data access | Availability statements are drafted. Final immutable access identifiers require author completion. |
| 2A. All data sources listed | Yes: synthetic generators; MNIST, Fashion-MNIST and CIFAR-10; MICrONS anatomy/synapses; DANDI response assets; and cited publisher-supplied external source data. |
| 2B. Public train/test/validation sets and links | Qualified. Benchmark sources and synthetic generators are public/reproducible; biological sources and asset IDs are specified. Some live/proxy resources and historical execution artifacts have explicit access/provenance limitations. Do not answer an unqualified “all public” until final access is verified. |
| 2C. Dataset biases | Yes: selected biological cohorts, one-mouse functional/focal scope, sparse observed partners, label missingness and proxy uncertainty, synthetic task construction, finite candidate sets and the restricted multi-affine model class are discussed. The Boolean extension uses seven fixed logical templates, not newly sampled functional families. See Methods, SI S4/S6–S10 and Figs. S33–S44. |
| 2D. Cleaning/preprocessing | Yes: model inputs, clipping/gain order, morphology mapping/compression, type joins, response reliability and stimulus splitting are executable and described. Boolean labels are centered and variance-normalized using the full sixteen-row truth table; accuracy uses the corresponding raw-0.5 threshold. Historical unresolved input-validity lineages are retained explicitly. |
| 2E. Combining sources | Yes: CAVE morphology/synapse joins and DANDI responses use explicit root/target/scan identifiers and eligibility rules. Scans and trial splits are nested within targets; external datasets are not pooled as interchangeable biological replications. |
| 3A. Architecture | Additive and conductance dendritic networks, matched point/grouped controls, explicit tree-constrained linear learners, scalar multi-affine trees, fixed-shape directed-conductance grouping models, local context-coefficient encoders and response-regression baselines. Individual models, forward equations and learning rules are given in Methods/SI and configurations. |
| 3B. Model Card | No separate deployment Model Card is supplied. This is a mechanistic research study with model specifications, assumptions and limitations in the manuscript and code. |
| 3C. Training/validation/test separation | Yes where models or hyperparameters are fitted. The original linear selector and new finite-calibration/end-to-end cohorts have separate development, calibration, training and test streams. The new cohorts use independent random target functions; final candidate restart choice uses a separate validation stream where applicable. Response ridge tuning is nested inside outer stimulus-identity splits. Boolean development and fresh seeds separately vary initialization, sampled noisy data and common leaf permutations within the same seven templates; rates are sealed before fresh runs. Full-domain capacity calculations are separately labeled and do not claim sampled generalization. |
| 3D. Split method | Explicit seeded benchmark splits and synthetic samples; held-out task seeds and rotation angles for the original linear selector; twenty fresh seed blocks per new interaction-based cohort, with independent random coefficients/signs across four task families; a separate Boolean cohort of five development and twenty fresh seeds within seven fixed templates; complete stimulus-identity outer splits and nested tuning for response baselines. |
| 3E. Splits mimic real-world applications | No general deployment claim. The original linear split tests constructed task seeds and rotations. New interaction-based cohorts test fresh random functions within their declared four-family mixture, not arbitrary new task families. Boolean seeds test new initialization and noisy data within seven fixed templates, not new logical families. Biological response splits test held-out stimuli under the observed-input restriction. |
| 3F. Leakage avoidance | Yes: candidate choices sealed before confirmatory training; independent calibration/training/test draws; training-only reliability and preprocessing/tuning inside response-training folds. For the finite 256-pattern interaction domain and sixteen-pattern Boolean domain, input patterns may recur across independent draws with fresh label noise. True target coefficients are reserved for explicitly privileged reference selectors and evaluation, not supplied to the finite-label selector. Freezes, source hashes and stream definitions are supplied. |
| 3G. Interpretability | Yes for the stated mechanistic quantities: analytic gradient reconstruction, explicit dictionaries/projectors, controlled routing/optimizer interventions and numerical invariants. This does not establish that biological neurons use the modeled mechanism. |
| 4A. Metrics | Yes: held-out accuracy, loss/NMSE, gradient alignment, projected energy, update match, costed loss/regret and spatial selectivity have explicit definitions and coordinates. Boolean primary NMSE uses all sixteen clean patterns after exact centering and variance normalization; threshold accuracy and balanced accuracy are secondary. Native units and uncertainty are preserved. |
| 4B. Cross-validation | Yes for the nested response-regression baselines. Other prespecified simulations use paired seed replication and fixed held-out splits; do not describe every experiment as cross-validation. |
| 4C. Community benchmarks | Yes: MNIST, Fashion-MNIST and CIFAR-10, alongside deliberately constructed mechanistic tasks. |
| 4D. Simple baselines | Yes: scalar/shared feedback, fixed and maximum-budget candidates, development-fitted rank-only and random selectors, two-sweep training pilots, constant-mean and random-candidate Boolean controls, mean and ridge response predictors, and matched point/grouped controls. True-target cut/DP selectors and retrospective trained minima are clearly labeled privileged references. |
| 4E. Current state-of-the-art benchmarks | No. The study makes no state-of-the-art accuracy, efficiency or deployment claim; its comparisons test mechanisms and information/resource constraints. |
| 4F. Ablations | Yes: feedback resolution/ownership, route topology/rank, cue quality/readout, optimizer coordinates, shunt/current/parameter substitutions, mapping/label sensitivity, input removal/noise, interaction grouping, label-independent initialization, exact versus broadcast credit and reliability filtering. |
| 4G. Fully independent dataset | Qualified: image-task replications, the original linear task/seed holdout and the separate fresh interaction-based cohorts are independent within their stated scope. Functional data remain seven targets in one mouse; the second MICrONS mouse is a structural check only. The Boolean fresh seeds do not constitute independent functional families. This is not independent biological validation of dendritic learning. |
| 5A. Hardware | Yes: archived or newly captured software/Slurm records identify available hardware/resources. Missing historical metadata is not inferred. |
| 5B. Compute cost | Yes where recorded: runtime, allocated CPU/GPU task time and parallelism. The original S35 development/confirmation/diagnostic arrays used 951 allocated CPU-seconds over 65 one-CPU tasks; this is not the total for the subsequent morphology bridges. New studies have their own Slurm accounting and runtime records. The Boolean smoke/development/fresh jobs used 342 allocated CPU-seconds over 26 one-CPU tasks, with fresh concurrency capped at eight; this is allocated time, not measured utilization. No carbon-footprint estimate is claimed. Refer to the final per-study resource audit for totals. |

## Code and Software Submission Checklist

| Field | Draft response |
|---|---|
| Source code / compiled software | Supply the final reviewer source archive. There is no compiled standalone application. The archive includes the implementation and article analyses/configurations/tests, with source snapshots and checksums. |
| Small demonstration dataset | Synthetic task generators and finite-difference/projector tests supply small self-contained demonstrations. The data packages provide derived numerical examples; biological reanalysis requires the stated upstream assets. |
| README: system requirements | Python environment and package versions are recorded in the release metadata. Core analyses use NumPy, SciPy, pandas, Matplotlib and PyTorch where applicable; figure/PDF workflows also use PyMuPDF and LaTeX tooling. Full network training uses documented GPUs; new linear, multi-affine and small conductance experiments run on CPUs. Check the final release environment file rather than copying stale version numbers. |
| README: tested versions | The original S35 prospective numerical tests used Python 3.10.13 and NumPy 2.2.6; other analyses have their own frozen records. Do not imply all historical runs used this environment. |
| Non-standard hardware | No special hardware for the small NumPy tests. Large network sweeps use HPC/GPU resources identified by their run records. |
| Installation instructions | Follow the reviewer-archive README: create the documented environment and install the included modeling package with its test dependencies. Article-specific scripts are included separately. |
| Typical installation time | Not measured on a normal desktop. Author/release verification should record an actual clean-install time; do not supply an invented estimate. |
| Demo instructions | Run `pytest -q tests/test_prospective_morphology_selection.py` from the journal environment, the regular-tree finite-difference check listed in the reproducibility README, and `pytest -q scripts/boolean_morphology/test_model.py` for the Boolean reference. |
| Expected demo output | The original S35 example has three passing invariant tests: disjoint split identities, tree/projector/spectrum/gradient/covariance checks, and zero-coupling training equivalence. The recorded original check passed all three in 7.09 seconds in its available environment; new bridge validation has separate records. The three Boolean tests cover truth-table normalization, forty-eight exact-gradient finite differences, root-credit equality and permutation invariance; complete numerical replay is documented separately. |
| Expected desktop runtime | The 7.09-second value is an observed environment-specific run, not a desktop guarantee. Record a desktop result if the form requires it. |
| Instructions for own data | The readout/task and biological-data entry points specify array shapes, model coordinates, required identifiers, preprocessing and splits. The code is a research implementation; applying it to other data requires matching those documented assumptions. |
| Full reproduction | Use the archived configurations, source/data hashes and stage order. Prospective selectors must be sealed before confirmatory training; reproductions should use a fresh output directory and preserve the original record. |
| Open-source link | Insert the final approved repository/archive URL, immutable version and license. Local preparation does not establish public deposition. |
| Detailed functionality/pseudocode location | Main Methods and SI give equations/protocols; executable scripts and experiment READMEs specify implementation and numerical verification. |

## Nature Portfolio Reporting Summary — applicable fields

| Field | Draft response |
|---|---|
| Statistical reporting | Units, exact sample counts, repeated/nested observations, effect sizes, interval methods and paired test families are specified in Methods/SI and source tables. `inferential_units.csv` and panel provenance identify the applicable unit. Confirm final figure/line references after compilation. |
| Distinct or repeated measurements | Training/task seeds are independent; candidates, conditions and checkpoints are paired within seed. Biological sites, contacts, scans and stimulus splits are nested within cells/targets. Two MICrONS mice support only a structural directional comparison. |
| Covariates | Task rank, rotation/alignment, noise, route budget, candidate geometry, optimizer, depth, cue sample size/noise/delay, shunt dose, mapping threshold, label availability and response reliability are explicitly defined per experiment. |
| Assumptions and corrections | Analytic assumptions are separated from empirical tests. Paired bootstrap, Wilcoxon, exact sign-flip, Holm or BH families are individually named; no universal normality or equivalence assumption is asserted. |
| Summary/uncertainty | Means, paired differences and 95% intervals where appropriate; SD and neuron SEM are labeled where descriptive. New seed-based inference uses 10,000 whole-seed bootstrap draws over 20 seeds. The finite-calibration, separate end-to-end and Boolean cohorts each use two Bonferroni-adjusted 97.5% intervals for their own primary comparisons; credit/conductance intervals are pointwise descriptive. |
| Bayesian analysis | Not applicable; no Bayesian posterior or MCMC inference is used. |
| Hierarchical designs | Biological inference resamples the highest applicable cell/target/animal unit; nested scans/splits/sites are not counted as independent animals. Prospective candidate conditions are averaged within task seed. |
| Data collection software | No new animal or human data were collected. Public data were retrieved through documented MICrONS/CAVE, DANDI and benchmark interfaces; analysis manifests record available source identifiers and hashes. |
| Data analysis software | See the final software/environment release record and experiment-specific scripts. Do not infer missing historical versions from the current environment. |
| Human participants, data or material | Not applicable: no human participants or human biological material were studied. |
| Human sex/gender, race/ethnicity, recruitment, population | Not applicable. |
| Field-specific reporting | Life sciences, with computational/modeling and public neuroscience-data reuse. |
| Sample size | Prespecified or frozen computational seed counts are reported per study. Original linear selection: 20 confirmatory task seeds and 320 paired task conditions. Finite calibration: 20 new seed blocks, four families (80 functions), six calibration conditions (480 choices), and 2,240 final candidate fits. Separate end-to-end confirmation: a further 20 seed blocks, four families and six conditions (480 fits). Credit: 3,600 fits; conductance: 1,440 fits, each in its own 20-seed cohort. These four earlier bridges total 7,760 fresh fits, not independent biological samples. The separate Boolean cohort adds 6,720 fresh fits from twenty seed blocks, 1,680 development fits from five seeds and a separately retained 336-fit implementation smoke; seven templates, candidates, optimizers, rates and checkpoints are nested conditions. Coefficient studies: 20 seeds; functional reanalysis: seven targets/13 scans; morphology uncertainty: eight cells/one mouse. Other cohorts remain separately tabulated. No prospective biological power calculation is claimed. |
| Data exclusions | Rule-based input/mapping/reliability/availability exclusions and unresolved historical lineages are documented. No fresh confirmatory tree-selection outcomes were excluded. Failed attempts before fitting and explicitly excluded smoke records are retained separately. The Boolean study has no failed fresh fits; its 336-fit complete smoke is excluded from fresh confirmation. Three complete-tree checkpoint replays with numerical drift are retained. |
| Replication | Independent simulation seeds, image-task replications and a second-mouse structural check; no independent-animal replication of measured dendritic learning is claimed. The original negative linear selection result, hard random-interaction outcomes, task-dependent credit effects and coefficient/response limits are retained. In the Boolean primary task, grouping improves NMSE by 0.563174 [adjusted 97.5% interval 0.549144–0.582749]; the compatible exact-credit improvement is 0.002507 [0.001780–0.003318], below the predefined 0.01 practical margin despite a positive interval. Both rules classify the compatible XOR-of-AND target perfectly. Larger parity credit effects remain descriptive. |
| Randomization | Seeded initializations, contexts, datasets, route controls, stimulus splits and perturbations are specified. Morphology/site selection is described by its actual frozen or outcome-free rule, without inventing population-random sampling. |
| Blinding | No blinding is claimed for these computational analyses. Frozen protocols, sealed selections, source hashes and outcome-free selection rules address the relevant analysis-bias risks. |
| Antibodies, cell lines, palaeontology, plants, clinical trials | Not applicable to the present analyses. Any such procedures in upstream studies should not be reported as new work performed here. |
| Animals and other organisms | Relevant as reuse of public mouse neuroscience data; no new animals were used. Identify the source studies and distinguish minnie65, Pinky and the external six-animal reanalysis. |
| Laboratory-animal strain, sex, age and housing | Verify and cite the upstream source-study records. These details are not consistently represented in the derived analysis metadata; do not infer them from dataset names. |
| Wild/field-collected organisms | Not applicable. |
| Ethics oversight | No new animal or human experiments. Cite upstream approvals as appropriate and have the corresponding author confirm the institutional/public-data-reuse statement. Do not invent a new approval number. |
| Dual-use experiments of concern | Not applicable to this computational dendritic-learning study. |
| ChIP-seq, flow cytometry, MRI | Not used in the present analyses. The optical/connectomic source studies are identified separately; do not mark MRI because neural data are analyzed. |

## Final transfer and verification

Transfer the applicable answers into fresh official forms, resolve all
“qualified” or author-only fields, save and reopen the XFA forms in Adobe
Acrobat Reader, and confirm that every response is visible. The corresponding
author must approve the exact final files. These steps remain pending; this
answer document does not claim they have been performed.
