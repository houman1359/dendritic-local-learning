# Local Credit Assignment — Comprehensive Paper Review (2026-04-16)

This review covers: text, figures/schematics, analysis, arxiv version, supplementary material, and overall paper logic. Each section lists concrete suggestions ordered from highest to lowest priority.

---

## 0. Overall verdict and framing

The paper has a **genuinely novel mechanistic claim** (shunting concentrates conductance-stage path gains → improves broadcast fidelity → enables low-bandwidth local learning), a **clean theoretical foundation** (Theorem 1, Corollary 1), a **compelling oracle experiment**, and now a **strong set of extensions** (soma-on, ablation, b/m policy, weight distributions). The story is real — the issue is packaging.

**The single biggest thing missing**: a headline "mechanism figure" that tells the story in one glance, plus a much sharper Figure 1 schematic. Readers of AI papers judge novelty and execution on Figure 1; yours is currently the weakest figure.

Secondary but important: the paper buries the **connection to the broader dendritic-analysis codebase** the user mentioned. That framing needs to appear in the intro, the contributions list, and the availability/impact statement.

---

## 1. Text — prose, flow, transitions

### High priority

1. **Sharpen the first paragraph of the abstract**. Current version front-loads mathematical detail ("factorize into a synapse-local eligibility term and a single non-local compartment error"). Rewrite as:
   - Sentence 1: What question? (when can local credit assignment work in conductance-based dendritic networks?)
   - Sentence 2: What mechanism? (shunting concentrates path gains)
   - Sentence 3: What evidence? (oracle experiment + gradient-fidelity diagnostic)
   - Sentence 4: What practical impact? (near-BP performance on MNIST/FMNIST, near-ceiling on cue routing with soma, CIFAR-10 gap largely explained by broadcast quality)

2. **Rewrite Introduction to have a clean contributions bullet list** at the end. Currently the intro flows well but readers scanning for novelty lose it in prose. Add:
   ```
   Our contributions are:
   (1) an exact factorization theorem for conductance-based dendritic trees (Thm 1);
   (2) a direct gradient-fidelity diagnostic that explains the shunting advantage;
   (3) a transported-error oracle that isolates broadcast quality as the dominant bottleneck;
   (4) a morphology-aware rank-K extension and soma-on rescue of structured tasks;
   (5) an open dendritic-modeling framework supporting this and related analyses.
   ```

3. **Add a codebase-framing paragraph** (end of intro, or beginning of Sec 4). One sentence in the abstract too. The user wants this emphasized, and it genuinely raises the paper's contribution to the community.

4. **Section 3.3 (4F/5F) narrative is a tension point**. The paper says 3F is theorem-facing and 4F/5F are "practical heuristic wrappers," but *all* headline numbers use 5F. Reviewers will call this a motte-and-bailey. Fix by:
   - Adding a paragraph that explicitly frames 4F/5F as "empirical denoising layers that don't change the underlying credit signal but reduce variance." 
   - Back this up with the 5F sensitivity appendix already present (fig_s3).
   - Optionally add a proposition: "Under i.i.d. noise on $e_n$, the 5F update is an unbiased estimator of the 3F update with lower variance." (a 3-line argument, not a theorem).

5. **Section 4.2 (Mechanistic Chain) is well-written** but the interpretive punchline at the end of panel C is buried. Add a one-sentence mini-conclusion:
   > "Reading across panels A–C: the same inhibitory regimes where path gains concentrate (A) are exactly where per-soma broadcast becomes a faithful proxy for the compartment error (B), which is exactly where LocalCA approaches the transported-oracle ceiling (C). This three-link chain is the paper's central empirical claim."

6. **Cue routing + soma result should be promoted to a one-line callout in Sec 4.8** rather than a parenthetical aside. Current phrasing risks being missed. Example:
   > "A striking companion observation: enabling direct somatic synapses alongside quantile init turns this hard task from 81 ± 13% (soma-off LocalCA) into 100 ± 0% (shunting fixed-router LocalCA + soma). See Appendix Fig. S_cue_routing_soma and Section C."

7. **Discussion paragraph "Limitations"** lists limitations in a single wall of text. Split into bullets for readability.

### Medium priority

8. **Citation hygiene**: 44 `\bibitem` entries, only 24 `\cite{}` calls in the text. **Roughly 20 references are defined but never cited**. Either (a) add citations where they belong, or (b) remove them. Run a quick `latex-check` to identify unused ones. Prime candidates to remove: `gretton2005hsic` (HSIC is in appendix only; if the HSIC section is reached, cite it there), `welford1962note`, `bellec2020eprop` (if eprop isn't discussed), `turrigiano2008homeostatic`, `buzsaki2014logdynamic`, `megias2001total` (if the MICrONS section doesn't reference it).

9. **Bibliography formatting inconsistency**. Some entries use full author names (`Kao, C.-H., \& Hariharan, B.`), others use `et al.` with no clear rule. Standardize: either "first author et al." for >4 authors, or spell out up to 6. 

10. **"CIFAR-10 soma-on" caption currently says "Additive-core soma-on runs could not be cleanly compared here because the legacy base config required a manual fix"**. This is honest but reads as ad-hoc. Better: either launch that run and include it (it's already fast once you fixed the config), or drop the disclaimer and just say "We report only the shunting-soma configurations here."

11. **The final sentence "We also do not claim the same empirical story under externally signed inputs; those legacy stress tests remain outside the corrected headline result set"** belongs in the methods/setup section, not the limitations paragraph. It's a scope statement, not a limitation.

12. **Move "Additional Stress Tests" (Sec 4.6)** earlier, merging into the "regime dependence" section. It currently sits between Morphology and CIFAR-10 and breaks the flow.

### Low priority

13. Consider dropping the `info_shunting` mentions from the main text entirely — you already say it's a legacy signed-input benchmark that violates the nonneg-drive assumption. Relegate to appendix.

14. Consistent tense: most of the paper is present tense ("we show", "shunting improves"), but a few sentences slip into past ("we started from exact gradients…"). Standardize to present-tense active voice.

15. Paragraphs in Section 3.4 are short and choppy. Merge the pathway-vector and router-role paragraphs into one or two longer paragraphs with clearer transitions.

---

## 2. Figures — quality and publication readiness

### High priority

1. **Figure 1 (model + credit) needs a complete redesign**. It is the weakest figure and it's the first thing reviewers see. Specific problems:
   - Panel A (compartmental neuron): too small, text overlaps diagram, the "shunting: V enters denominator" callout is illegible.
   - Panel B (rule hierarchy): looks like a slide, not a paper figure. Boxes have no consistent alignment, labels feel placeholder.
   - Panel C (broadcast modes): The four rows are visually crowded, the soma circle isn't clearly distinguishable from pre-synaptic circles, and the "shared field" vs "per-soma" vs "low-rank" vs "structured pathways" concept is not visually clear at first glance.
   - Panel D (learning dynamics): tiny overlapping legend, "91%" percentages sit on top of curves.

   **Recommendation**: I'll rebuild this as a 4-panel figure with:
   - (A) A clean compartmental diagram with three labeled synapses (one excitatory, one shunting, one dendritic), a soma, a single large arrow showing broadcast $e_n$, and a zoom-in callout showing $\Delta g = x \cdot R^{tot} \cdot (E-V) \cdot e_n$. Drop the multi-compartment $V^{(1)}, V^{(2)}$ boxes — they're distracting.
   - (B) A 2-column comparison: "Additive" on left, "Shunting" on right, each showing the voltage equation. Remove the 3F/4F/5F hierarchy from here entirely; put it in a small inline algorithm or as a single-panel appendix figure.
   - (C) Broadcast modes as a vertical stack of 4 pictograms, each with one clear teaching signal arrow and one dendritic tree. Use icons, not prose.
   - (D) Single clean learning curve: Shunting BP, Shunting LocalCA, Additive LocalCA on MNIST, 5 seeds with shaded SD. Drop 3F/4F/5F — too many lines.

2. **Figure 6 (cue routing) Panel A schematic is difficult to read**. Too much text, arrows are unclear, the "shared field → dendritic layer" flow gets lost. Redesign with:
   - Big top-left: task icon (context + noisy cue A + noisy cue B → prediction)
   - Middle: two dendritic branches (path 1, path 2), each getting its own vectorized feedback $e_{n,k}$
   - Bottom strip caption: "Rank-1 feedback collapses pathway identity; higher-rank or soma-channel restores it."
   - Shorten the existing 3-line caption inside the panel.

3. **Panel label consistency**. Currently mixed: some figures use `(A)` with period, some just `A`, some `A.`. Pick one (recommend `(A)` with a tab). All figures should match.

4. **Figure 3 Panel D (exact factorization sanity)**: the "max rel L2 < 1.9e-07" text box sits awkwardly inside the plot area. Move to caption or a small annotation box with border.

5. **Figure 5 Panel D**: "Effective bits per neuron" x-axis is logarithmic but uses 1, 2, 4, 8, 32 with linear spacing — inconsistent. Either use a log scale or drop the numerical x-axis and use categorical labels ("sign", "top-30", "1-bit", "4-bit", "8-bit", "full").

### Medium priority

6. **Figure 2 Panel B**: MNIST solid and Noise dashed lines overlap heavily at low $N_I$. Consider separating these into two sub-panels or using distinct marker shapes.

7. **Figure 4 Panel C**: Looks fine but the hatched pattern on BP bars doesn't clearly differentiate from the LocalCA bars. Use a darker edge color or lighter fill.

8. **Color accessibility**: The green/blue pair for shunting/additive is colorblind-safe (deuteranopia OK), but the shunting-vs-BP green pair are quite close. Add pattern (solid vs hatched) or different saturation.

9. **Supplementary figure sizes**: some appendix figures (fig_s1, fig_s2) use `\textwidth` while others use 0.95. Standardize to 0.95 to give the figures breathing room.

10. **Fonts in figures**: Check that all figures use serif (Times) to match the paper. Some saved figures may still use DejaVu Sans.

### Low priority

11. Combined-panel schematic "flow diagrams" (like fig_s_ablation dotted line) look good; keep that style consistent.

---

## 3. Schematics — specific requests

**Figure 1 reference designs**: Look at the Figure 1 in the *Nature Neuroscience 2021 Payeur et al. "Burst-dependent synaptic plasticity"* paper for the style you should aim for — large clean compartments, color-coded synapses, teaching signals shown as unambiguous dashed arrows, panel letters in large sans-serif caps.

For **Figure 6 (cue routing)**, the reference design would be the *DFA paper (Nokland 2016)* Figure 1 — clean task schematic on the left, network architecture in the middle, feedback path as a distinct arrow.

---

## 4. Additional analyses that would strengthen the paper

### High-value additions (each ~1-2 days of compute + plotting)

1. **Time-to-accuracy comparison**. Plot epochs-to-90%-of-BP for LocalCA and BP across all main benchmarks. This directly addresses "is LocalCA trainable at practical cost?" — a common reviewer concern.

2. **A "map" summary figure in the main text**. One 2-panel figure that plots the mechanistic chain: left panel = scatter of (path-gain CV) vs (cosine alignment of per-soma to exact error) across all conditions; right panel = scatter of (cosine alignment) vs (final test accuracy). This would show the causal chain in a single figure and strongly support the mechanism.

3. **Morphology × inhibition with matched theory diagnostics**. Currently `fig_s8_morphology_ie_regime` shows accuracy heatmaps. Complement with a heatmap of path-gain CV across the same grid. This makes the "morphology shifts the regime" claim quantitative rather than suggestive.

4. **Direct comparison to one or two modern local-learning methods on the same benchmark**. Currently the related-work section cites AugLocal/CCL/DLL/EBD as higher-performance but doesn't put them on the same plot. Even a narrow apples-to-apples comparison (same MNIST architecture, no HSIC) would help reviewers contextualize the 91% vs 97% gap.

5. **Weight distribution × somatic inputs**. Currently we have weight distributions at multiple depths (new) and soma-on accuracy (new), but not weight distributions under soma-on. One small sweep (12 configs × 5 seeds = 60 jobs) would close this loop.

### Lower priority

6. **Biological testable predictions quantified**. Section 5 "Testable predictions" lists 4 predictions but doesn't quantify them. At least one should have a figure: e.g., predicted $\rho_n$ trajectory during learning or a specific effect size for GABA-A blockade.

7. **Sparse connectivity check**. Real cortex has much sparser connections than our 40 E + 20 I per branch. One sweep varying connectivity density would address biological plausibility.

8. **LocalCA + FA combination**. You already have FA and LocalCA separately. Does combining them (FA-style random projection + local plasticity rule) recover anything? This is a one-line variant and could cleanly position the paper as complementary to FA.

---

## 5. Supplementary material — completeness audit

### Missing or thin

1. **Dataset details**: Appendix `app:synthetic_tasks` describes context gating, noise resilience, and cue integration, but does not give:
   - Train/valid/test split sizes for each synthetic task
   - Example inputs (a small visualization would help)
   - Noise generator seed protocol
   - Difference between "fixed-router" and "learned-router" cue routing variants

2. **LocalCA algorithm box**. The algorithm appears (algorithm env at line 1193-ish) but should be more prominent. Put the pseudocode for the 3F update right below Theorem 1 in the main text.

3. **Assumptions list**: Currently "Biological Plausibility Assumptions" is one paragraph. Format as an explicit bullet list:
   - A1: Steady-state voltage (no temporal dynamics).
   - A2: Tree topology (no recurrent connections within a dendrite).
   - A3: Nonnegative conductances (softplus parameterization).
   - A4: Nonnegative presynaptic drive (ReLU transfer).
   - A5: Pre-reactivation voltage passes to parent (for Theorem 1); post-reactivation handled via activation-derivative factor.
   - A6: Single perisomatic broadcast delivered to all compartments.
   Reviewers will look for this.

4. **Full hyperparameter table per experiment**: Representative table exists (`tab:representative_architectures`) but no single table lists LR, weight decay, batch size, epochs, patience for every experiment. Add one.

5. **Random low-rank broadcast**: Describe explicitly how $\Gamma_K$ is sampled and whether it's frozen or learned during training. Currently one line only.

6. **Path-transport oracle**: Make explicit that this is an oracle (requires access to the full dendritic tree's $\alpha_n$ during training), and is therefore not a biologically plausible rule. The paper mostly does this but should be crisper.

7. **Statistical significance**: Headline numbers are reported with SD. Consider adding confidence intervals or paired-seed tests (signed-rank) for the most important comparisons to strengthen claims like "shunting outperforms additive by 50 pp at $N_I=10$."

8. **Seed protocol**: "5 seeds for headlines, 3 seeds for support, 1-2 seeds for exploratory" is stated. Make a small table mapping each figure/claim to its seed count.

### Already present and good

- Units and parameterization table ✓
- Decoder update modes ✓
- Model and mode inventory ✓
- Compute resources statement ✓
- NeurIPS checklist ✓
- MICrONS connectome analysis ✓
- HSIC auxiliary objective description ✓

---

## 6. Arxiv-style .tex — preparation

NeurIPS 2026 format forbids double column and is 10pt Computer Modern. Arxiv preprint typically uses:
- `\documentclass[11pt]{article}` with 1-inch margins
- Author/affiliation visible (no anonymization)
- Full contact info
- Arxiv license statement

I'll prepare a parallel `local_credit_assignment_arxiv.tex` that:
1. Uses a standard preamble (no `neurips_2026.sty`)
2. Includes the real author block
3. Removes the NeurIPS Paper Checklist section
4. Adds an arxiv-style title page with affiliations and corresponding author email
5. Keeps the same figure includes and bibliography

This is mechanical enough to do in one edit once the content edits above are agreed.

---

## 7. Paper logic and novelty framing — biology ↔ ML connection

**The paper's novelty claim as currently written**: "exact factorization for conductance-based dendritic trees + mechanistic account of shunting as path-gain concentration + direct fidelity and oracle experiments + structured rank-K viewpoint."

**What's missing from that claim**:

1. **The biology→ML bridge isn't explicit enough**. The paper has the ingredients (MICrONS weight distribution validation, GABA predictions, testable predictions) but doesn't weave them into the headline contribution. Add a subsection *"From biophysics to inductive bias"* that explicitly states: "Our exact factorization shows that a conductance-based dendritic branch is not just a biological detail — it's an inductive bias that makes a single broadcast signal sufficient for learning. This is a fundamentally different approach from DFA/FA, which modify feedback pathways, and from DTP/DFC, which modify inference dynamics."

2. **The codebase as contribution**. One subsection in the intro + one paragraph at the end of Sec 1 stating:
   > "This work is part of an open dendritic-analysis framework that supports arbitrary tree morphologies, conductance-based and additive integration, and a wide family of local learning rules. We release all configs, analysis scripts, and figure code to support reproducible dendritic analysis research."

3. **Testable predictions** list has 4 items but feels like an afterthought. Expand Section 5 into its own subsection (not just a paragraph) and for each prediction state:
   - The experimental handle (what you'd measure)
   - The expected effect size (quantified from our simulations)
   - The closest existing experiment (citation)

4. **Limitations** currently mixes "we didn't test X" with "our approach can't handle X". Separate:
   - Scope limitations (not tested): CIFAR >10 classes, longer recurrent dynamics, multi-task
   - Structural limitations (method issue): steady-state assumption, single-compartment plasticity, per-neuron broadcast

---

## 8. Suggested priority order for changes (if you have one week)

**Day 1-2** — Text pass:
- Rewrite abstract (1 hr)
- Add contributions list to intro (30 min)
- Add codebase framing (30 min)
- Clean 4F/5F framing (2 hr)
- Trim unused bibliography entries (1 hr)
- Fix 20 smaller text issues (3 hr)

**Day 3-4** — Figure redesign:
- Rebuild Figure 1 schematic (4 hr)
- Rebuild Figure 6 Panel A cue-routing schematic (2 hr)
- Standardize panel labels across all figures (1 hr)
- Polish color palette (1 hr)

**Day 5** — Additional analyses:
- Launch and analyze time-to-accuracy comparison (compute overnight)
- Add "map" summary figure (morning)
- Add morphology × path-gain CV heatmap (afternoon)

**Day 6** — Supplementary audit:
- Assumptions bullet list
- Full hyperparameter table
- Dataset visualizations
- Seed/claim mapping table

**Day 7** — Arxiv prep + final polish:
- Prepare `local_credit_assignment_arxiv.tex`
- Final proof-read
- Compile PDF and check all references resolve

---

## 9a. Claim hygiene and reproducibility

This section is added on the 2nd pass after the user's feedback that "make the corrected result set explicit" should be near the very top of the priority list. Two sub-items:

### 9a.1 Scope/claim taxonomy

Add a small paragraph (and a compact table) to the main paper that separates:

| Tier | What it is | Where it lives | Seed count |
|---|---|---|---|
| **Headline** | Corrected, positive-input, nonneg-conductance, 5 seeds, matched architecture. Used to support the main mechanistic claim. | Figs 2, 3, 4, 5 (main text), Table in Sec 4.3 | 5 |
| **Supportive appendix** | Stable 5-seed (or 3-seed, clearly labeled) results that corroborate the headline claim but are not needed to carry it. | Appendix Figs S1–S5, the b/m policy figure, the soma-extension figure, the ablation figure, the weight-distribution figure | 3–5 |
| **Exploratory** | 1–2-seed screens, signed-input legacy benchmarks (info-shunting), or incomplete sweeps. | Appendix tables marked "exploratory", Sec on morphology×inhibition | 1–3 |
| **Provisional** | New sweeps not yet at full seed/completion. Either flagged or held back. | Currently none (weight-dist completed 60/60 on 2026-04-16 22:00). | — |

Explicit list of what is corrected:
- positive-input enforcement for all headline runs
- corrected `noise_resilience` (nonneg-drive, `relu` transfer)
- decoder-aware soma mapping for nonlinear decoders (CIFAR-10)
- occupancy-quantile reactivation init with safeguarded calibration
- consistent learned-local $(b,m)$ under LocalCA as the default

Explicit list of what is supportive extension, not headline:
- soma-on LocalCA (all 4 phase families) — stable 5 seeds
- b/m policy comparison (learned vs quantile-maintained) — stable 4 seeds per cell
- component ablation — stable 5 seeds, 12 groups out of 14 (two `relu_reactivation` groups failed to initialize; documented)
- CIFAR-10 depth-4 soma extension — stable 3 seeds for additive; shunting currently resubmitted after config fix; results will be reported once complete
- cue-routing soma — stable for the subset that ran (additive learned-router LocalCA; shunting fixed-router LocalCA; both BP variants). Shunting learned-router variants failed due to signed-input mismatch in the cue-integration task; omitted from the stable subset.

Explicit list of exploratory / not-final:
- morphology × inhibition regime map (3 seeds; no matched theory diagnostics across the grid yet)
- info-shunting legacy benchmark (signed inputs; not part of the corrected set)

### 9a.2 Reproducibility audit

**Source-of-truth for figure data.** Each supplementary figure should pull its data from a CSV under `drafts/dendritic-local-learning/analysis/<figure_topic>_summary_YYYYMMDD/`, written by a script that walks the sweep's `results/config_*/performance/final.json` and `results/config_*/config.json`. Current status:

- `fig_s_soma_extension`: **data is hard-coded** in the figure function (from a markdown report). Fix: write `summarize_soma_extension.py` that reads from the four paper-facing soma sweeps under `/n/holylfs06/.../sweep_runs/sweep_neurips_phase*_*_soma_rerun_*/` and produces a grouped CSV.
- `fig_s_bm_policy`: uses `bm_classification_grouped_summary.csv` → ✓ OK
- `fig_s_ablation`: uses `component_ablation_summary_20260416/ablation_grouped_summary.csv` → ✓ OK
- `fig_s_cue_routing_soma`: uses `cue_routing_soma_summary_20260416/cue_routing_soma_grouped.csv` → ✓ OK but reports numbers from prior-paper prose for the soma-off baseline; those should move to a tiny CSV too.
- `fig_s_weight_dist_by_depth`: uses `weight_dist_by_depth_summary_20260416/weight_dist_grouped_summary.csv` → ✓ OK, sweep now complete.

**Hard-coded-numbers policy.** If any figure uses numbers that are not in a versioned CSV under `analysis/*_summary_*/`, flag it in that figure's caption with "(numbers carried over from prose)". Plan to remove all such flags before submission.

**Latest-directory mechanism.** `_latest_analysis_dir()` returns the most recent `<prefix>_<date>` directory; this is convenient but fragile if more than one summarizer exists for the same topic. Recommend: add an explicit `ANALYSIS_DATE` constant at the top of `generate_neurips_figures.py` that pins the exact summary directories used in the submission.

### 9a.3 Proposed short edit list (immediate, this session)

1. Add a small claim-hygiene paragraph + Table to the main paper (in Sec 4 "Experiments"→ Setup, or as a new "Claim tiers" subsection at the start of Experiments).
2. Reframe the cue-routing soma section to state explicitly that learned-router shunting is omitted due to signed-input incompatibility, and that the reported result is on the stable successful subset.
3. Revise the abstract to open with the question, not the factorization.
4. Add a contributions bullet list to the intro.
5. Add the codebase / framework contribution sentence in the intro and a dedicated appendix subsection.
6. Add an assumptions bullet list (A1–A6) to Appendix B.
7. Add a seed-to-claim mapping table.
8. Pin `ANALYSIS_DATE="20260416"` in `generate_neurips_figures.py`.

---

## 9. What I'll do right now

Rather than try to complete all of the above in one pass (which would produce mediocre edits across too many files), I'll:

1. Prepare a clean `local_credit_assignment_arxiv.tex` parallel version as requested.
2. Rewrite the abstract and introduction along the lines above.
3. Generate a list of unused `\bibitem` entries for cleanup.
4. Scaffold the Figure 1 redesign (structure and layout, ready for you to review before I commit to the full rebuild).
5. Add the codebase-framing paragraph to the intro.

Larger figure rebuilds (Fig 1 and Fig 6) need your input on stylistic direction first — do you want a minimalist Bengio-style schematic, a more biologically detailed Payeur/Sacramento-style one, or a compact infographic-style (EBD paper style)?

After your feedback on (5), I'll do the actual rebuild in the next iteration.
