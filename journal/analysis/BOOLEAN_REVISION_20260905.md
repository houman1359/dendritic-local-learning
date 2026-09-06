# Boolean structure and credit revision — 5 September 2026

This revision implements the requested AND/OR/XOR extension of the Nature Communications credit paper. It follows, and does not replace, the earlier [morphology revision](MORPHOLOGY_REVISION_20260905.md). All seven logical templates, candidate trees, learning rules, optimizers, rates, seeds, endpoints and the two primary comparisons were fixed before the fresh experiment. No failed primary comparison was replaced. The existing prospective-selection failure, optimizer controls and conductance results remain visible.

## What the extension establishes

The new examples separate three questions: whether a tree can represent a computation, how much depth it requires, and whether a specified credit rule learns its parameters. These questions have different answers even for four-bit functions.

1. **Input grouping has a provable and measured effect.** For XOR of two AND branches, the aligned multi-affine tree admits exact representation. Each crossed balanced tree has normalized truth-value MSE at least 8/15. Across twenty fresh seed blocks, the mean of the two crossed groupings increased exact-credit Adam NMSE by 0.5632 relative to the aligned tree (adjusted 97.5% bootstrap interval 0.5491–0.5827). This passed both the positive-interval and 0.01 mean-improvement criteria.
2. **Composition can determine minimum depth at fixed gate count.** OR of two AND branches requires depth two; a AND [b OR (c AND d)] requires depth three. Both use four inputs, three gates, two ANDs and one OR. Their input-gradient ranks are four, but their full spectra differ. The earlier 140-target isospectral census supplies the stronger spectrum-matched control.
3. **Detailed credit helps some tasks and recipes, while other tasks learn well with broadcast.** The primary aligned XOR-of-AND comparison gave exact NMSE 0.00095 versus broadcast 0.00345. Its 0.00251 improvement (adjusted 97.5% interval 0.00178–0.00332) failed the predefined 0.01 practical margin. Both rules classified all sixteen patterns correctly in every fresh seed. Under the frozen Adam recipes, parity showed a large descriptive difference across all four tested trees: exact NMSE about 0.0008–0.0010 versus broadcast 0.89–0.98. Parity was a prespecified task; this contrast was not a primary test and is not presented as one.

## Exact capacity and mechanism controls

The independent exact audit enumerated all fifteen rooted, unordered binary trees on four labeled leaves: three balanced trees of depth two and twelve comb trees of depth three. Each has three scalar multi-affine units, twelve coefficients, six edges and root-only output. The 105 task–tree pairs contain 49 exact-compatible cases. AND4, OR4 and parity4 each admit all fifteen trees; each mixed task admits one tree.

Every compatible case has an exact rational raw-output construction and a normalized construction inside the learning box ±2. The largest squared coefficient is exactly 16/15, giving maximum magnitude 4/sqrt(15), about 1.033. All normalized certificate outputs replay to within 2.22e−16. The centered-cut criterion concerns real-valued regression on the uniform truth table; its positive lower bounds do not imply classification error after thresholding. Overlapping cut bounds are combined by a maximum, not added.

Canonical AND, OR and XOR derivatives illustrate conditional credit, including the XOR multiplier 1−2v. They describe one choice of internal coordinates. All local coefficients remain free during training, and a sign-changing canonical derivative alone does not prove that broadcast learning fails. These signed algebraic units are separate from the positive-conductance model in main Figure 6F and Supplementary Figure S42.

An explicitly **post hoc** exact diagnostic was added after the fresh outcomes were known. The projection energy ||E[z|x_S]||² vanishes for every proper input subset in parity, whereas it is 1/5 for each two-input subset of XOR-of-AND. All 98 target/subset values were checked against direct conditional means. For a fixed subtree-local eligibility, a zero projection removes the direct target term from its population broadcast update. Root-mediated effects and finite-sample training remain possible. The diagnostic does not predict grouping compatibility: aligned and crossed XOR-of-AND pairs have the same projection energy. It supplies a mechanistic clue, not a replacement primary experiment or a proof of impossibility.

## Fresh learning experiment

Seven fixed four-bit templates were tested: AND4, OR4, parity4, OR of ANDs, XOR of ANDs, AND of XORs, and the nested AND/OR task. Four prespecified trees crossed two credit rules, SGD/Adam and rates 0.003, 0.01 and 0.03. Five development seeds generated 1,680 fits. A separate excluded implementation smoke seed generated 336 fits. Twenty fresh seed blocks generated **6,720 fits**, with zero failures. These are additional to the earlier 7,760 fresh morphology/credit/conductance bridge fits; the cohorts and hypotheses are kept separate.

Per-rule rates were selected from pooled development performance and sealed before any fresh fitting: Adam exact/broadcast 0.003/0.01; SGD exact/broadcast 0.03/0.01. Every rate remains in Source Data and the same-rate comparisons appear in S44. These are comparisons of bounded, frozen recipes, not a claim that a full optimizer search has found the best possible broadcast learner. All three same-rate Adam XOR-of-AND credit differences were below 0.003 NMSE.

Each task uses all sixteen equally likely input patterns to define exact centering and population-variance normalization. Training uses 256 draws with replacement, normalized-label Gaussian noise SD 0.15, batch 32 and 2,048 updates. All coefficients learn from label-independent initialization; paired conditions share input permutation, initialization, data and minibatches. Parameter bounds are ±2 and gradient-norm clipping is ten. The primary endpoint is clean NMSE over the complete truth table. Independent noisy-test NMSE on 1,024 examples, accuracy and balanced accuracy are secondary. Uniform product input sampling is not class-balanced sampling.

There are two prespecified primary Adam contrasts on XOR-of-AND. Each uses twenty paired seed blocks and 10,000 bootstrap draws, with two-sided 97.5% intervals for Bonferroni adjustment. Success requires a positive lower interval limit and a mean improvement of at least 0.01, jointly. A lower confidence limit above 0.01 was not the rule. Only the grouping contrast passed. New seeds vary initialization, data, noise, permutations and sampling; they do not introduce new logical template families.

## Manuscript and figures

- Main Figure 6A now introduces a four-input aligned/crossed XOR-of-AND example with the exact 8/15 crossed-grouping bound and a clearly labeled canonical gate derivative. Panels B–F preserve the larger structural, noisy-estimation, parameter-learning and conductance results.
- Supplementary Figure S43 presents exhaustive capacity, minimum depth, post hoc projection energy and canonical gate credit.
- Supplementary Figure S44 presents all seven learning tasks, primary uncertainty, trajectories, rate/optimizer controls, classification and learning diagnostics.
- Results, Discussion, Methods, the cohort table and supplementary methods distinguish representation, learning recipe, classification and primary versus descriptive inference. The weak primary XOR credit result remains explicit.
- The release now contains nine main figures, forty-four supplementary figures and the additional Boolean methods fragment. Current source maps, reporting answers and software inventories were updated.

The final narrative contains approximately 8,183 words, the abstract 185 words, and all nine main captions remain below 350 words (Figure 6: 295). Nine figures plus the Methods cohort table use ten main display items. The combined PDF contains 39 main and 125 supplementary pages. This retains the author’s approximately 8,000-word working target; it does not change the journal’s approximately 5,000-word guidance.

## Reproducibility and verification

Numerical evidence is in [Boolean learning Source Data](../source_data/boolean_morphology/README.md) and [exact Boolean Source Data](../source_data/boolean_theory/README.md). Implementations are under `scripts/boolean_morphology/` and `scripts/boolean_theory/`. Scientific figures are generated from saved tables by `scripts/build_boolean_morphology_figures.py` and the main-figure builder. No training is required for a frozen-data figure rebuild.

The learning validator reconstructed all 8,400 development/fresh endpoints and frozen streams, and replayed all 336 fits and 2,048 steps in one fresh seed with bit-identical final weights and nontiming trajectories. An independent implementation separately reproduced all 8,400 endpoints and initial states, checked all 67,200 recorded checkpoint bounds, verified permutations and development-only rate selection, and reproduced both primary intervals. All 3,600 incompatible endpoints respected the theorem; the smallest margin above a bound was 0.000321828. The independent audit does not claim to replay every full training trajectory.

The pre-experiment protocol SHA-256 is `4373c421dd8ad60de8275d8384f441fc7c19d000232081bbb5be2eb7df0597df`; the pre-fresh selection seal is `965160e7d9126f974da817735dfb0ec1df35b6f09cc2056fabdab5a9f3c5a8a1`. Runtime environment details, source hashes, scheduler records, all rates, weights and outcomes accompany the source tables. Scheduler accounting reports 342 allocated CPU-seconds over 26 jobs; it does not measure actual CPU consumption or peak memory.

The combined test run passed 164 tests with one skip. Main/SI provenance, citation, format, figure-style and LaTeX layout checks passed. Fifty existing publication assets are byte-identical to the previous release; rendered Figure 6B–F are pixel-identical. Both new compiled supplementary figure pages and the updated main figure/cohort table were visually inspected. All internal PDF links and page text were preserved in the combined document.

Final PDF, provenance, packaging and isolated released-code checks are recorded under [boolean_revision_20260905](boolean_revision_20260905/). The previous release was copied and hashed before changes in `before/manifest.json`. Final archives use a separate clean local paper snapshot and the recorded production commit; the active working repositories are not committed by this workflow. No manuscript was submitted or externally published, and author-specific form fields and related-submission declarations still require accurate author completion.

## Relevance to dendritic replacement

The useful transferable test separates information available within each candidate subtree from interactions that must cross its boundary. Scalar-subtree cut constraints can identify unavoidable approximation error before optimization, while the parity control tests whether a credit rule uses higher-order information under a specified recipe. Representation bounds, finite-data estimation, learning dynamics and biological conductance constraints need separate checks in the replacement project. This revision does not modify that project's implementation or establish a general morphology-selection theorem for its architectures.
