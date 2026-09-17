# Figure review, 2026-09-08 — synthesis of the nine-reviewer panel

Scope: HEAD main figures 1–9 and curated SI S1–S35 in `drafts/dendritic-local-learning/journal` (read-only), compared with the rebuilt set in `.dendritic-figure-overhaul-20260908/journal/figures/main`. Every claim is anchored to a panel and a file a reviewer or the verification pass opened. Line numbers are `main.tex` at HEAD 55bf9b8. Source Data paths are relative to `journal/source_data/`.

## 1. Headline verdict

The message — morphology supplies a fixed dictionary A of delivery profiles, task structure decides which spatial distinctions are useful, a teacher must still supply c, conductance sets diagonal gains Γ — is carried by the nine-figure spine in the right order, and the section headings are tempered claims the data support (verified: Fig 3E +1.27 pp [0.59, 1.95], 15/20, Holm P = 0.0101; Fig 6E +10.86 → −1.52 pp; Fig 9A −0.069 [−0.251, 0.106]). Today's moves were mostly right: promoting S7D → 1F, S10 → 3D, S20 → 5F and S23 → 6D–F answered the four deepest referee objections and demoting the transfer-geometry diagnostic to S31/S32 was correct; two were wrong (restoring the one-step bound as Fig 1C; rebuilding Fig 6A without the architectures while deleting the validation-loss panel), and the 56 → 35 SI consolidation dropped six panel groups that `main.tex` still cites. The panel set is not yet right: the six credit rules, the exact-minus-per-neuron contrast, the deranged-route dose curve, the rewiring control, the broadcast/spatial split and the background-conductance rescue are all quoted in Results and drawn nowhere, while seven lone full-width strips (Fig 1G at 441 × 49 pt, Fig 3E at 466 × 65 pt) carry three numbers each. Three text/panel mismatches are confirmed (L132 vs Fig 1E; L136 vs Fig 1G; `inferential_units.csv` n = 15 vs plotted n = 10) and one is refuted (Fig 3E's 1.95 is correct). Graphically HEAD is far from Nature standard — DejaVu Sans throughout, 4.76–5.32 pt type in Figs 3 and 5, 23 strict-audit violations (not one, as the brief says), three tree orientations, anatomy hues within ΔE 3–7 of series hues, δ₀ drawn in one schematic of nine — whereas the overhaul set passes the audit cleanly and should be the production base once its panel lists are corrected to today's text.

## 2. Allocation

### 2a. Moves made today

| Item | From → To | Verdict | Reason |
|---|---|---|---|
| One-step bound | S2 → Fig 1C | **wrong** | No data, n or interval; axis "(illustration)"; L124 disowns it as a selector (20/20 losses, S33); legend hides that q differs (0.8 vs 1.0); 117 × 115 pt beside the primary experiment. Return to S2 or a ~90 pt inset anchored to 1F's capture values. |
| Trained capture K=1 vs 3 | S7D → Fig 1F | correct | Resolves the D/E paradox (0.622→0.655, 0.764→0.859; `curated_publication/figure_01_plotted.csv`). Fix title and legend strokes. |
| Coefficient-learning cohort | S10B–D → Fig 3D | right idea, **wrong rendering** | Joins two grid corners differing in two factors; drops the frozen floor (18.7%); hides that at 256 noise-free cues soft is −0.31 pp from oracle with coefficient accuracy 1.0 (`review_coefficient_encoder/condition_summary.csv`, verified). Use the S10C form on the 3 × 3 grid. |
| Continuous gate + placements | S20 → Fig 5F | correct | 2.18e-5 / 2.33e-5 / 2.32e-5 / 0.949 / 0.123 verified (`conductance_local_gate/summaries/condition_means.csv`). |
| Long-budget trajectories | S23 → Fig 6D–F | correct | Values verified in `physical_depth_followup/`. But the −2.94 marker and validation-loss panel were dropped. |
| Fig 6A architecture trees | old 6A → deleted | **wrong** | Fig 6 now compares architectures it never draws; "serial − grouped point" undefined on the page (git db902e1). |
| Validation-loss panel (six arms) | old 6D → deleted | **wrong** | Sole evidence for "all fifty D3 fits reach the cap with falling validation loss" and the 0.48 pp exact-minus-broadcast claim (L295; 0.99367−0.98889). |
| Transfer-geometry diagnostic | old 9B–E → S31/S32 | correct | Tests passive-tree geometry (one profile reconstructs 98.4%), not the biological hypothesis. L381 (seven numbers) was not shortened; caption cites "SI" with no number. |
| Fig 3B coefficient sums | retained | partially | Four verified constants in a 172 × 124 pt line plot. Keep content as the K = 1,2,4,8 dictionary strip (overhaul B). |
| Fig 7E capture vs wiring | retained | partially | Real control, but 5/6 points in a 9 × 28 pt box, two markers fuse. Fold as printed columns into 7D. |
| SI 56 → 35 | | partially | 114/277 old panels have no destination (`reference_map.json`); six dropped groups still cited (§4c); S1, S3, S4, S5, S12, S30 uncited from main. |
| Old S53E cancellation ratio | deleted | **wrong** | Only figure for the L269 mechanism; Methods L482 still promises "cancellation diagnostics" in S18–S20. |

### 2b. Moves the panel recommends

| Item | Source → destination | Decision | Reason |
|---|---|---|---|
| Six-rule mechanism card | overhaul 1B → Fig 1 | **make** | Rules exist only as x-tick strings; Figs 1–7 depend on them. |
| Exact − per-neuron (+0.010 / +0.182 pp) | `image_ladder_controls/summaries/paired_contrasts_six_rules.csv` → Fig 1E | **make** | Headline within-tree number has no mark (verified). |
| Cohort forest (per-neuron−scalar; exact−per-neuron; ±1 pp band) | Fig 1G + S6B/C/D → one Fig 1 panel | merge | Repairs the L136 citation, gives Fashion-MNIST a mark, shows the equivalence band. |
| Deranged curve, 8 doses × 3 B | `path_necessity_fashion/condition_summary.csv` → Fig 2D | **make** | 12–25 pp zero-conflict claim (L178) undrawn. |
| Measured initial utility (24 cells) | same → Fig 2C | **make** | Equation becomes a validation; deviations are systematic (max 0.0175 at B=8, χ=1; CI excludes prediction in ≥5/24 cells) — say so. |
| Fig 2E | → fold into 2D | merge | E is D's χ=1 column; print "BP = gated = branch-specific, 20/20 ties" in-panel. |
| S9B context-0 forgetting | S9 → Fig 2 | **promote** | Only place misrouted credit destroys prior knowledge (0.56 vs 0.00); both Fig 2 and SI reviewers choose it. S9A stays (duplicates 2E). |
| Rewiring control (+23.15 pp K=2, +5.02 K=4) | `trained_subtree_address_full_factorial/condition_summary.csv` → Fig 3 | **make** | Largest alignment-specific effect; in no figure (L199 cites "Source Data"). |
| Fig 3E + 3F | → one forest | merge | Saves ~100 pt; note "derangement +61.1 off scale". |
| Credit-delivery card (six sites) | overhaul 4B → Fig 4 | **make** | The only credit-rule figure that draws no credit. |
| Fig 4F | scalar → cumulative energy vs k + uniform | rebuild | Gives L238 a mark (uniform 0.122 vs 0.169, `credit_rule_extension/summaries/all_diagnostics.csv`). |
| Fig 4E | → paired-seed strip | rebuild | "20/20" undrawn; two same-colour bands. |
| S12C leaf shuffle | S12 → Fig 4 | **promote** | Only morphology manipulation in the section; S12 uncited. S12A/B stay (Fig 4 reviewer: dilutes the two-rule contrast). |
| Cancellation ratio (0.636 vs 0.330→0.172) | `conductance_credit_demand/opponent/summaries/context_gradient_summary.csv` → Fig 5 | **make** | Mechanism for L269; repairs L482. |
| Delivery card gate/exact/broadcast/oracle | overhaul 5B → Fig 5 | **make** | Four rules defined only by a legend. |
| Fig 5B, 5E | → inset of A; third width | shrink | B is two sigmoids; E equals D's difference to 3 s.f. |
| S20C rate strip | S20 → Fig 5 | promote (compact) | L267 asserts rate robustness; not the 9 × 3 heat map. |
| S21A architectures; S21B five-arm ladder | S21 → Fig 6 | **promote** | Restores the deleted drawing; S21B carries grouped-point, raw-additive, point-MLP controls with intervals and can absorb 6B. |
| Validation-loss panel | git db902e1 → Fig 6 | **restore** | See §2a. |
| Broadcast/spatial/rest bars | derivable from `figure_07_plotted.csv` E → Fig 7 | **make** | Title turns on "beyond a shared broadcast"; the share (≈0.196; 31.4% at L321) is drawn nowhere and in no CSV. |
| Per-cell scatter (11 tie cells) | `figure_07_plotted.csv` D → Fig 7 | **make** | The L323 caveat is prose only. |
| Mechanism strip t → shunt → z, δ₀ → Ac | overhaul 7C → Fig 7B | **make** | No panel says what the target fields are. |
| Fig 7F | → all families × cohorts | rebuild | L327 "ordering held" cites a panel with no controls. |
| S25B–D joint-3D null | → name at L327 | cite | SI reviewer wants a main panel; Fig 7 already gains three. Promote only if space remains. |
| Background-conductance rescue (−0.0004 → +0.121 → +0.259, 8/8) | `focal_selectivity_phase1/paired_contrasts.csv` → Fig 8 | **make** | Resolves 8E; in no figure (L360 cites "Source Data"). |
| Matched-injection arm (opposite sign) | `shunt_ancestry_gain/signed_calibration/signed_cohort_summary.csv` → Fig 8D | **make** | Shunt −0.0148 vs injection +0.0099; visible only in S29's caption. |
| S28D dose | S28 → Fig 8 | promote | L358 dose claim rests on SI. S29A (45/45) optional. |
| Fig 8E | → half width + |y| < 0.005 inset | shrink | 457 × 121 pt for nine points. |
| Measurement schematic | overhaul 9B → Fig 9A | **make** | Only main figure with no schematic. |
| Fig 9B | λ axis → observed-effect axis with the −0.069 interval | rebuild | The one comparison is delegated to 6.8 pt text. |
| S31B coverage (48.0%) | S31 → Fig 9 | **promote** | L381's limitation has no display item and no Source Data table. |
| S30C four-measure strip | S30 → Fig 9 | promote (compact) | S30 uncited; delete S30B (duplicates 9A). |
| Fig 9C | → inset of 9B | shrink | Calibration input; "n = 125" reads as study n. |
| S31C NMSE + controls | S31 → Fig 9 | conditional | Only if L381 keeps its numbers; random-route 0.0305 ≈ ancestry 0.0293 (`review_response_baselines/paired_ridge_contrasts.csv`). |
| S7G energy by depth | S7 → Fig 1 | **do not promote** | SI reviewer for, Fig 1 reviewer against (axes not commensurable; F shows structure; Fig 1 full). Cite beside 1F; split S7. |
| S17E, S18B, S13C | | optional | No space after mandatory additions; cite by panel. |

## 3. Figure by figure

Layouts use the graphics reviewer's grid (12 modules at 518.4 pt, 40 pt gutters, heights 340/415/490 pt), adopting overhaul panels where judged better.

### Fig 1 — Identity, not finer spatial address, carries the image benefit
Identity buys 9.23 / 7.46 pp (verified); K=1→3→exact buys ≤0.2 pp although capture rises.

| Panel | Decision | Issues | Fix |
|---|---|---|---|
| A boxes | redesign | Three text boxes, no morphology, no E/I contact (L86 cites A for E/I), no δ₀; box colours reuse data hues; 6.8 pt | Overhaul A + one inhibitory glyph; drop "Credit enters at the soma"; write ε consistently with Eq. 4 |
| B dictionaries | redesign | "Delivered field = A c" drops Γ; K=12 diagonal all green; tree→row order only in 6.8 pt grey text | Colour K=12 by subtree; 12-row strip; add "K=1: c = δ_u; K=12: c = ε" |
| C bound | → S2 | §2a | Inset only if tied to 1F |
| D six rules | keep | Ticks vs text names; intervals drawn for 4/12 means; grey band unexplained; y 70–100 | Unify names; y 78–100; +9.2/+7.5 tag; "intervals < symbol"; label band |
| E contrasts | redesign | Plots Exact−K=3 (−0.093/−0.044 pp, never stated) while L132 gives exact−per-neuron — **confirmed mismatch** | Three contrasts on one axis; name references |
| F capture | keep | Title contradicted at Initial (0.5322 vs 0.5323); no intervals; no n; legend keys not solid/dashed | SI twin's title; "n = 10"; text clause on additive capture falling |
| G controls | merge | 441 × 49 pt for 3 numbers all in S6; L136 eleven-point citation wrong — **confirmed** | Cohort forest with ±1 pp band |

Layout: three rows, 488 pt, 7 panels — A (4) + six-rule cards (8), 124 pt; dictionary (4) + ladder (8), 116 pt; E / F / G at 4 sharing y, 108 pt. Overhaul E's labels sit 13 pt above their rows and it drops the seed fan; cap schematic area at 30% (overhaul 37.6%).

### Fig 2 — Credit reaching an unselected branch is harmless until its input conflicts
Shared ties branch-specific at χ=0 (+0.0008, p = 0.12) and inverts at χ_c = B/[2(B−1)].

| Panel | Decision | Issues | Fix |
|---|---|---|---|
| A task | redesign | Same tree twice; no δ₀; context overloads junction ring; 6.8 pt | One tree + χ state; red context input; δ₀; glyph key |
| B delivery | redesign | Derangement missing; bus from a floating dot, not the soma; no gate glyph | Overhaul B (three ghost trees, × gate) |
| C boundary | redesign | Prediction only; 17,357 pt² > D's three facets (15,424); B=8 line clipped | Overlay 24 cells with CIs; y to −0.8; "initial signed utility s(χ)" |
| D dose | redesign | Deranged absent; green occluded under orange χ 0–0.75 at B=2 (draw order); bands α 0.11 invisible; 3.8 pt markers at 2.0 pt pitch | Add deranged; markers, draw last; α 0.22; tick sampled doses; swarm at χ=1 |
| E full conflict | merge → D | Legend on seed dots (x 78–80 pt); composite marker unexplained; grey = deranged here, gated point in S9A | Print the two 20/20 statistics |
| F crossings | keep | Five non-crossing seeds at χ ≈ 1.05; no n; 29.5 pt misaligned with D | Broken axis; n = 20; align |
| new S9B | promote | | own n (10 seeds) |

Layout: three rows, 486 pt, 8 panels on the overhaul grid (columns x 51–173 / 214–336 / 383–505). Call `align_letters()` — overhaul letters drift 28 pt in column 8.

### Fig 3 — Ancestry wins only where its partition matches the task (K = 4)
Design fixes the useful budget (−3.05/−0.05/+0.85/+1.00, verified); rewiring destroys it; cue-supplied coefficients are the binding limit.

| Panel | Decision | Issues | Fix |
|---|---|---|---|
| A tree | redesign | Leaves c_i collide with Methods' context index (L453); no z, δ₀, soma; nodes 9.7–14 pt off midpoints so siblings ambiguous; 4.76 pt subscripts | Overhaul A: soma, z, δ₀, "junctions define supports only", four-swatch key, b_1..b_8 |
| B sums | rebuild | Four constants in 172 × 124 pt; normalized values undrawn | Dictionary strip K = 1,2,4,8 |
| C ladder | keep | Purple "best control" at K=1,2 is only the dense rank-K field; ancestry = deranged = 0.18665 at K=1 (below chance) unexplained; fraction vs pp elsewhere | Percent; annotate ties and sub-chance |
| D cue cohort | rebuild | Two corners differing in two factors; floor 0.18669 and mismatched +7.10 absent; leakage is a small-calibration effect (−60.90 / −5.23 / −0.31 pp at 16/64/256 noise-free cues, **confirmed**) | S10C form, 3 × 3 grid; qualify L205, L401 |
| E + F | merge | 466 × 65 and 99 pt strips; left edges 39 vs 119 pt; fourth control (+61.06) omitted silently; conclusion title for 1.27 pp | One forest, neutral title |
| new rewiring; control card | make | | Overhaul D, F (key F's grey levels) |

Fig 3E is **correct**: +1.27 [0.59, 1.95], 15/20, Holm P = 0.0101 reproduce `review_evidence_reanalysis/ancestry_k4_control_contrasts.csv`; the 1.9653 in `trained_subtree_address_full_factorial/paired_contrasts.csv` is a second bootstrap. Add a per-panel Source Data index and a Holm column (only `bh_adjusted_p` ships). Layout: three rows, 490 pt, 7 panels; re-letter so the cue cohort is cited last (L199 → L201 → L205).

### Fig 4 — Matched input spectra, different learned credit geometry
Pairwise stays rank-one (99.7%), quartic and nested do not (46.1 / 44.8%); a fixed profile fails for the latter out to 16,384 updates.

| Panel | Decision | Issues | Fix |
|---|---|---|---|
| A targets | redesign | Two different trees under "Same eight inputs and tree"; no soma/f/δ₀; no ⊕/⊗ key; 38% of canvas | Overhaul A: one tree, badges only; key; ±0.5 shown |
| B pairwise | keep | Legend crosses the 1,024 line; five names for "fixed profile"; no n | Rename; move legend; minor ticks |
| C quartic | redesign | Staircase 1,024–4,096 is the 3 stalled seeds resolving (n > 0.1: 3, 2, 1, 0 — verified); y unlabelled, floor 0.045 vs B's 0.0225 | Per-seed traces or "3 stalled / 17 at floor"; label floor |
| D nested | redesign | Depth-four, different spectrum (S11/S12 captions) styled like B/C. (A's banner is correctly scoped — verified, A shows two trees only) | Badge "Nested control (depth 4, unmatched spectrum)"; caption clause |
| E deficit | rebuild | Two same-colour α 0.10 bands; 8 numbers in 206 × 102 pt; 8,192 unlabelled | Paired strip, four budgets (overhaul G, centre labels) |
| F rank-one | rebuild | Uniform 0.122/0.169/0.176 and rank 1.006/3.238/3.417 exist but are absent; 0.9973 vs 0.9999 "distinct" per caption yet indistinguishable; all black | Overhaul H, drop in-plot legend |
| new delivery card; S12C | make | | Overhaul B (fixed profile in blue) |

Layout: three rows, 490 pt, 8 panels.

### Fig 5 — A local inhibitory gate delivers branch-selective credit matching exact path credit
Gate = exact (2.33e-5 vs 2.18e-5), broadcast fails (0.487) only on the opposed task (20/20), misplacement destroys it, a graded gate survives.

| Panel | Decision | Issues | Fix |
|---|---|---|---|
| A tree | redesign | No inputs, output, δ₀; inhibition = grey word "Inhibited" at 6.8 pt on a line; two text/line collisions | Overhaul A: exp(±z) inputs, E/I glyphs, a_p = 0/10, attenuated subtree, V_s, δ₀, gate as multiplier; B as inset |
| B teacher | → inset | Two logistics; its legend reads as C/D's | |
| C aligned | redesign | Rule legend 200 pt away; "16,384" label 13.4 pt off its tick; 5.32 pt exponents; no rate/n | Legend in C; 64 tick + break; 7 pt; "Adam 0.03, n = 20" |
| D opposed | redesign | **Readout mismatch**: L265 cites C,D for 0.487, panel plots fixed-checkpoint 0.5219 (`condition_means.csv`; `curve_band_source.csv` step 4096); oracle unflagged | Re-plot or re-cite; direct labels |
| E interaction | shrink | Equals D's difference to 3 s.f.; no paired lines; top point on frame; Holm P = 3.81e-6 unprinted | Third width; connect seeds; print stats |
| F placements | redesign | "Gate also proximal" 0.123 is early-stopped (best step ≤ 256; endpoint 0.5532); swapped best_step = 0 in 4/20; no broadcast reference; nine hues | Two groups; broadcast dashed at 0.487; annotate; five hues |
| new cancellation; delivery card; rate strip | make | | |

Add a `conductance_local_gate` row to `inferential_units.csv`. Layout: overhaul grid, 490 pt, 7–8 panels; replace the 13 pt capsules with tinted patches.

### Fig 6 — Task organization sets the forward benefit; budget sets the credit ranking
Serial pays only when factor information is distributed and sensed (30.32 / 22.20 / 0.00 pp, verified); ranking flips with budget (+10.86 → −1.52), cross-entropy does not (0.254 vs 0.276).

| Panel | Decision | Issues | Fix |
|---|---|---|---|
| A cards | redesign | Task-only; no tree/soma/δ₀; 24% of canvas; outlines in shunting green / additive blue | 60% width: generative model + D1/D3 trees (S21A) |
| B heat map | replace | Cells are aligned/serial/shunting/BP only (unstated); no chance line (D1 0.60 reads mid-range); D3 > D4 outline rests on −1.46 pp, 0/10, undrawn; three encodings | S21B ladder, or % + colorbar + hatched untested + printed contrast |
| C alignment | keep | Local-ratio α=1 is an exact 0 (10 identical pairs) drawn as a null; x to 1.35; SI's 30.41/22.26/1.30 is a different estimand | Clip x; open marker; name estimand |
| D trajectories | shrink | 431 × 84 pt; broadcast arms missing; restarts undisclosed (explains 30.82 vs 30.86); title "persists" vs text "budget-specific"; no chance | Half width; add broadcast; caption "restarts"; retitle |
| E accuracy diff | keep | −2.94 and "epoch 315" cited, unmarked; "−1.52" floats 17 pt off its point; no tick < 0; crossing unresolved (CI straddles 0, epochs ~300–330). Do not carry the overhaul's "312: first negative" (single dip) | Mark minimum; −5 tick; leader; shade |
| F CE diff | keep | Note crosses the trough (y 488.6–496.5 vs 488.9); no "(nats)" | Move note; unit; endpoint tag |
| restore validation loss; promote S21A/B | make | | |

Resolve the duplicate `\label{fig:physicaldepth}` (main.tex:306; `si_06_physical_depth_figures.tex:7`). Layout: three rows, 492 pt, 8 panels; recolour E/F off the purple meaning "D3 exact LocalCA" in D.

### Fig 7 — Reconstructed arbors supply spatial routes beyond a shared broadcast
Ancestry routes compress a cell's own fields better than four controls at 18.8% of dense wiring; the advantage is modest (4.1 pp total / 5.1 residual), heterogeneous (9 negative cells, 11 overlapping the surrogate null), and is capacity, not teaching.

| Panel | Decision | Issues | Fix |
|---|---|---|---|
| A arbor | redesign | Cited (L313) for inhibitory-bearing routes, shows none; 8 red vs ~120 blue strokes under a diverging bar; orange soma unexplained; no root_id or Source Data | Half size; I contacts; one route subtree in green; root_id |
| B toy | redesign | K=3 toy in a K=8 figure; 1-bit raster matrix; "Common" spans two columns | Overhaul B (real 70 × 8, ≥6 pt/column — overhaul's is 3.75) + overhaul C strip |
| C capture vs K | keep | K=1 = 1.37e-32 by construction; unpaired bands; C/E at 109.85 vs 103.89 pt per unit | Drop K=1, extend to 16; shape-code controls; "paired test in D" |
| D paired | keep | Residual scale (5.08/13.51 pp) vs L323 total (0.041/0.108) unreconciled; per-cell points for surrogate only; random rank 5.66 unflagged | "residual" in caption; both scales in text; wiring % and rank as columns (absorbs E) |
| E wiring | → D | §2a | |
| F cohorts | redesign | No control family (L327 unsupported); axis label wrong for residual symbols; four names for one cohort; Pinky 0.981 with no ceiling; one mouse per cohort absent | Overhaul G; 1.0 line; site count |
| new decomposition; per-cell scatter | make | | Overhaul E, F |

Reconcile wiring density 18.8% (7E) / 8.1% (S24A, same cohort) / 6.92% (`capture_per_wire/report.md`). Layout: three rows, 488 pt, 8 panels; move overhaul G's Pinky footer off the interval it crosses.

### Fig 8 — In a passive model, focal shunts apply ancestry-defined gains whose selectivity is set by electrotonic state
A shunt is exactly a diagonal gain on an ancestry partition; selectivity is 34× larger at R_m = 300 than at 15,000 (−0.0019 / +0.0024, verified) and is restored by background conductance — a result in no figure.

| Panel | Decision | Issues | Fix |
|---|---|---|---|
| A partition | redesign | Three-block vocabulary vs five in B; no inputs/output/δ₀; injection and soma-restoring current undrawn; shunt = bare red bar | Overhaul A+B+C: two cards, real-arbor partition with 50 µm bar, q′ = diag(h)Bη |
| B normalized | keep | No n/interval (101 sites in 8 cells); "depth control" ≠ CSV "depth-matched unrelated"; blue = injection collides with A | Grey injection; caption stats; rename |
| C adjoint | redesign | "Decomposition" contradicts SI; 2 of 4 corners (adjoint-only 0.130 absent); no y label; 0.079, 8/8 unprinted | "Adjoint replacement"; 2 × 2; label; print |
| D signed | redesign | Injection arm omitted; labels far from marks; cohorts marked by 6.8 pt text 114 pt above ticks; circle/square mean relation here, perturbation in B | Add injection; leaders; bracket |
| E state | shrink | 457 × 121 pt for 9 points; disjoint abscissa medians 47 cells vs 45 (`build_focused_main.py:50`); R_m printed twice; points at ratio 136.0/137.1 overlap; bracket misses its point | Half width; recompute; secondary axis; inset; key |
| new rescue; S28D | make | | |

Soften L342 to add "selectivity set by electrotonic state"; convert 19 literal "S28B,C"-style citations to `\ref`; L355's within-cell control is S29B. Layout: three rows, 490 pt, 7–8 panels.

### Fig 9 — Recorded partners show no ancestry alignment; the sample detects only r ≳ 0.25
Null (−0.069 / −0.048; five of seven target values identical across the two rows) plus the honest bound (80% at partial r ≈ 0.249; MC-respecting λ = 0.65).

| Panel | Decision | Issues | Fix |
|---|---|---|---|
| A null | redesign | Full width for 14 points, 40% empty; labels 6.8 < 7.6 pt ticks; rows share targets, unstated | Half width; ±0.5; scan circles; "n = 7; 20,000 draws"; add S30C strip |
| B detection | rebuild | λ axis shares no scale with A; MC band ~1.5 pt though captioned; 0.025 intercept unlabelled; λ = 0.65 unquoted | Overhaul D + perfect-reliability curve |
| C reliability | → inset | x to 1.0 for max 0.57; grey unlinked from B; `partner_index` cannot reproduce 102 partners | Inset; green; global partner id |
| new schematic; S31B | make | | Overhaul B with leaders; coverage table to Source Data |

Do not adopt overhaul G/H (they re-import S31/S32). Layout: 460–485 pt or two rows at 340 pt. Reconcile `si_09_measured.tex` "−0.255 to 0.108" with plotted −0.2508/0.1055.

## 4. Supplement

### 4a. S1–S35

| Fig | Q | Cited | Decision | Note |
|---|---|---|---|---|
| S1 | 3 | no | restyle + cite L431 | No n/interval in A,B; "MN/FMN/FG" undefined; two paste scales |
| S2 | 3 | yes | restyle | H_c undefined; D's boundary unkeyed; 5.44 pt |
| S3 | 2 | no | merge → S2 | Placeholder "legend: within-rule association" printed in C; D no legend |
| S4 | 3 | no | restyle + cite L433 | Negative result on fixed shunting; blue = "anti"; 5.29 pt |
| S5 | 3 | no | restyle + cite L433 | Only evidence for "AΓ preserves span, changes conditioning"; 5.59 pt; 1.2e-15 vs 1.15e-15 |
| S6 | 3 | by panel | restyle | B's ±1 pp band ~3 px on 0–60%; two colour grammars; 0.859 pp verified |
| S7 | 2 | by panel | split | A/B axis hides 0.01–0.18 pp; two cohorts (10 vs 15 seeds); 559 pt |
| S8 | 3 | range | restyle + panel cites | A has no axes; green/blue = tasks; F in "millions" |
| S9 | 4 | by panel | promote B | Grey = deranged in 2E, gated point here; 5% scale mismatch |
| S10 | 2 | S10B–D | rebuild | y label truncated "(pp"; hard/soft colours swapped vs 3D; D duplicates 3D; old S32C/D dropped though L205 relies on them |
| S11 | 2 | L470 | restyle | 5.0–6.17 pt; D unlabelled series (old S37D dropped); C lost partner (S37A) |
| S12 | 3 | **no** | restyle + cite L230–242 | 3,600 fits uncited; exact green here, dark red in S15/16 |
| S13 | 3 | range | restyle | Diverging map on a bound; E+F six numbers on half a page; C candidate for 4A |
| S14 | 3 | range | restyle | 4.76 pt colourbar; E/G y ranges differ 60× |
| S15 | 4 | range | keep | Selected-rate Adam (old S54A,C) dropped though L234 reports them |
| S16 | 4 | range | keep | D's y to 10 vs 1 |
| S17 | 3 | yes | restyle | 4.76 pt; F truncated with tautological exact = 1; 0.013222 verified |
| S18 | 3 | yes | restyle | Linear x vs S15/16 log x; 0.000502 verified |
| S19 | 3 | yes | restyle | Caption "dots" on a panel with none; nine type sizes |
| S20 | 3 | yes | merge rows | Unit and calibrated broadcast identical to 2 s.f. in all 12 cells |
| S21 | 4 | range | restyle | Duplicate label; D duplicates 6C (with intervals 6C lost); F candidate |
| S22 | 4 | range | keep + cite Methods | Fixes the Fig 6 operating point; 5.88 pt; n = 2 |
| S23 | 3 | range | restyle | C (0.64/10.96/13.47 pp) has no registered source in `manifest.json` |
| S24 | 3 | range | restyle | A five unlabelled curves; s.e.m. vs bootstrap; 0.262 pt strokes; 8.1% verified |
| S25 | 2 | range | rebuild | Clouds without skeletons; orphaned clipped label; 5.58 pt; null deserves naming at L327 |
| S26 | 4 | range | keep | All numbers verified; floating dashes as a series |
| S27 | 4 | range | keep | Best SI figure; all numbers verified; E empty histogram |
| S28 | 3 | yes | restyle | L355 promises a dropped within-cell control; 5.14 pt |
| S29 | 3 | D,E | restyle | 153 strokes < 0.27 pt; A (45/45) buried |
| S30 | 2 | **no** | merge → S31 | B duplicates 9A; D calibration input |
| S31 | 4 | range | promote B (C conditional) | Numbers verified |
| S32 | 4 | range | keep | Only SI figure with n/unit/interval on-figure; ridge blue then green |
| S33 | 4 | yes | merge with S34 | Conclusion titles; selector green in C, blue in D |
| S34 | 3 | range | merge with S33 | 5.0 pt; cost panel (S38D) and hexbins (S39A–D) dropped under a "prediction" headline |
| S35 | 4 | yes | restyle | 4.76 pt; grey means three things |

### 4b. Promotions (final)
S9B → Fig 2; S12C → Fig 4; S21A, S21B → Fig 6; S28D → Fig 8; S31B → Fig 9; S30C strip → Fig 9; S20C rate strip → Fig 5; old S53E → Fig 5. Not promoted (reasons in §2b): S7E–G, S12A/B, S17E, S18B, S13C, S25B–D, S28A/C, S33C.

### 4c. Lost in consolidation and still referenced
| Dropped | Text that relies on it |
|---|---|
| Old S32C, S32D (calibration size; cue delay) | L205 "(S10B–D)" — S10 has neither axis |
| Old S54A, S54C (selected-rate Adam trajectories) | L234–236 primary numbers cited to S15–S16 |
| Old S36D (twelve-candidate selection) | Methods L470 |
| Old S37A, S37D (depth-3 partner; 105/35/24 key) | S11C, S11D |
| Old S53E (cancellation) | L269; Methods L482 |
| Old S21A (within-cell control, 8 cells) | L355 "S28B,C" |
| Old S38D (CPU cost); S39A–D (forecast hexbins) | L461 "computational costs"; S34 headline |
| Old S1 A–E, S6, S16, S26 (whole figures); S17 (six-animal reanalysis) | L97/L431 "Section S1"; Methods L516 |
| Old S51D (development regimes); S18C,D (α dose) | SI §S5 protocol; L289 |

`omitted_panel_status` repeats one sentence for all 114 dropped panels.

### 4d. SI style
Every curated SI figure is a collage of crops at mixed scales (`manifest.json`: S1 1.05/1.0068; S3 1.0272/1.05; S6 1.0039/1.0261; S7 1.027/1.05; S9 1.0294/0.9797; S19 0.981/1.006; S24 1.05/0.985; S28 1.05/0.999; S29 1.05/0.974), giving S6 six type sizes and S24 eleven. Sub-6 pt type in S2, S4, S5, S11, S14, S17, S22, S25, S28, S34, S35; hairlines < 0.25 pt in S24, S29. Three rule palettes (S12/S17 exact green; S15/16/18 dark red; S14 viridis); S10C swaps hard/soft vs 3D. On-figure n/interval only in S1C, S4C, S6B, S24C, S25A, S26D, S32. No per-SI Source Data in `curated_publication/`, and no `figure_02_plotted.csv`. Re-render natively on the main canvas.

## 5. Graphics standard for the next build

Adopt the overhaul set (0 strict violations, manifests on all nine) as production base; HEAD has 23 violations across eight figures (`scripts/figure_canvas.py --audit --strict`).

| Element | Spec | Gap today |
|---|---|---|
| Typeface | Helvetica/Arial class (Nimbus Sans), `pdf.fonttype 42`; assert no DejaVu | All 18 PDFs embed DejaVuSans |
| Type | Three sizes: 7.0 (ticks, annotations, keys), 8.0 (labels, titles), 9.0 bold letters; floor 7.0 | Six sizes + letter; 4.76 pt (Fig 3), 5.32 pt (Fig 5); 7.2/7.4/7.6 indistinguishable |
| Strokes | 0.55/0.7/0.85/0.95/1.25 pt, nothing above; `DECORATIVE_LW_PT` 2.5 → 1.35; capsules become 16% tint patches | Overhaul 13.0 and 8.68 pt capsules (Fig 5), 5.3–6.2 (7B), 4.2–5.6 (3); HEAD Fig 5 off-system |
| Palette | Saturated hues mean rule/architecture only (shunting green, additive blue, exact dark, broadcast amber, oracle violet); anatomy warm grey (L* ≈ 55, chroma < 0.02), one soma accent, E/I contacts the only saturated anatomy marks; gate build on ΔE ≥ 15 normal / ≥ 10 CVD over series × anatomy | OKLab ΔE×100: shunting/dend 5.9 (4.8 deutan), bp/inh 6.9 (4.1), scalar/soma 6.5, additive/exc 8.3, point_mlp/mute 3.1; overhaul Fig 1 has 193 paths #3FA26C beside 92 #3E8E63 |
| Ordinal / K | Four-hue K-cycle and a non-grey ordinal ramp, never a series | Overhaul Fig 2 B greys collide with deranged grey |
| Letters | 9 pt bold, x locked to module column, `align_letters()` unconditional; audit shared x to 0.5 pt | Drift: HEAD 7 28.7 pt, 8 17.1; overhaul 2 28.0, 6 16.5, 8 13.5 |
| Legends | Direct labels; one rule-key strip in the schematic row; no boxes inside data axes; no figure-level legends | HEAD 5 has two floating legends; boxes over data in 1, 2, 4, 6, 7; overhaul 6C, 4H |
| Forest idiom | One helper: label centred on row, 0.55 pt tick, seed fan always, n + interval tag, reserved gutter | Overhaul 1E/F, 4G, 9C/G/H labels above rows; HEAD 1G, 3E/F, 7D/F, 9A |
| Schematic glyphs | Soma lowest node, filled accent, δ₀ arrow entering every schematic, output leaving right; tapered grey tree with white junction rings; E filled blue, I filled red, inactive I open red; gate/shunt = inhibitory family + badge, never a bar; exactly four delivery glyphs (external amber bus = layer scalar; bus at soma = per-neuron; tinted capsule + arrow = subtree; α-tagged chain = exact); assert in `native_schematics.py` | HEAD: soma at bottom (2A), top (3A, 4A, 5A, 7B, 8A), left-right (1B); δ₀ in one of nine; 5A soma labelled "Somatic error" at top; 1A no morphology |
| Matrices | ≥ 6 pt per row/column asserted; header ≤ 1.5× column; vector | Overhaul 7B 3.75 pt/column; HEAD 7B 1-bit raster |
| Secondary axes | Real spine with unit | HEAD 8E, overhaul 8G loose R_m numerals; overhaul 6F doubled clusters |
| Heights | 340/415/490 pt justified by panel count; schematics ≤ 30% | HEAD 365–572 (three below aspect 1.05); overhaul nine near-full pages; Fig 1 37.6% |
| Full-width data panels | None unless data need the aspect | HEAD seven (1G, 3E, 3F, 6A, 6D, 8E, 9A) |
| Captions | Bold finding clause; per-panel sentence ending in n, unit, interval, endpoint; 220–320 words; final sentence names the Source Data file; `README_panel_columns.md` | Blanket interval sentences false per panel (Fig 1 "D–G", Fig 5 "bars in C–F") |
| Audit additions | Letter grid; in-axis text vs data artists; same-figure ΔE < 15 with different roles; raster ≥ 300 dpi | Would catch overhaul 7G footer over the Pinky interval |

## 6. Confirmed mismatches and inconsistencies

Re-verified this session:

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | Fig 1E plots Exact − K=3, never stated; L132's +0.010/+0.182 is unplotted | **confirmed** | `figure_01_plotted.csv` E: exact_path_minus_subtree_k3 −0.00093 (3/10), −0.00044 (4/10); `paired_contrasts_six_rules.csv` exact_path_minus_neuron_shared 0.00010, 0.00182 (10/10) in no panel |
| 2 | L136 cites Fig 1G for "about eleven points" | **confirmed** | 1G's only axis is "Exact − neuron-specific (pp)", −1.0..0.0; neuron − scalar = 11.03/10.80 pp, n = 15 (`mnist_between_within_factorial/paired_contrasts.csv`), drawn in S6D |
| 3 | `inferential_units.csv` n = 15 for the Fig 1D/E endpoint | **confirmed** | Row `regular_tree_feedback … 15`; all D/E/F rows n = 10; `provenance_manifest.tsv` "ten paired fresh seeds"; no fresh-cohort row |
| 4 | Fig 2C is prediction-only; measurement exists | correct, numbers corrected | Max deviation 0.0175 at B=8, χ=1 (−0.7325 [−0.7397, −0.7251] vs −0.75); systematic; CI excludes prediction at B=8 χ ∈ {0, .25, .75, 1}, B=4 χ ∈ {0, 1} |
| 5 | Fig 3E 1.95 vs 1.9653 | **refuted** | Panel reads `review_evidence_reanalysis/ancestry_k4_control_contrasts.csv` (1.9458); 1.9653 is another bootstrap (four architectures 1.953–1.965 for one mean) |
| 6 | Fig 3D / L401 over-generalize leakage | **confirmed** | Noise-free soft vs oracle 0.80291: 0.19392 / 0.75061 / 0.79978 at 16/64/256 cues, coefficient accuracy 1.0; S10C already draws the 256-cue point on the oracle line |
| 7 | Fig 4A banner covers the nested target | **refuted** | A shows only "Pairwise target" and "Quartic target"; D titled "Nested (separate tree)"; L226/232 scope the claim |
| 8 | L238 "uniform capture did not" vs 4F | text correct, no mark | `all_diagnostics.csv` uniform 0.1224/0.1686/0.1757; 4F plots rank-one only |

Reviewer-verified against files: L265 cites 5C,D for 0.487 while the panel plots 0.5219; Fig 7D residual vs L323 total; wiring density 18.8/8.1/6.92%; Fig 8E abscissa 47 vs 45 cells; duplicate `fig:physicaldepth`; S10D truncated label; S3C placeholder; S10C/3D colour swap; S19 caption "dots"; `si_09_measured.tex` interval; S23C unsourced; §4c references.

Vocabulary to unify: five names for the fixed-profile arm (Fig 4); four for the endpoint; four for the initial 8-cell cohort; five for the projected rules (Fig 1); ε = AΓc (L145) vs δ̂ = Ac (Eq. 4), ε undefined in the body; "alignment" with six meanings (L61; 180, 269; 193, 453; 289–303; 52, 373–405; 516); Fig 6D "7/7/2 of 10" mixes predicates.

## 7. Disagreements with the external Deep Review

| Deep Review | Panel position |
|---|---|
| Promote S7 route capture to Fig 1 | Done (1F), correct; no further S7 promotion — S7G stays in SI (Fig 1 reviewer over SI reviewer). |
| Demote Fig 3B | Disagree: it is the a-priori prediction and explains the below-chance K=1 point; rebuild as a dictionary strip. |
| Demote Fig 7E | Agree on the panel, not the content: fold into 7D. |
| Fig 9 to SI or boundary test | Keep in main — a null on the only measured dataset is a credibility asset; recast with schematic, observed-effect axis, coverage panel. |
| Keep S31 in main if Fig 9 stays | Disagree: passive-tree transfer geometry is not the biological hypothesis; promote S31B only, S31C conditionally. |
| Promote S20 to Fig 5 | Done (5F); do not add the heat map, only a rate strip. |
| Fig 8 as operating-regime map | Done and agreed; incomplete without the background-conductance rescue, which the Deep Review misses. |
| Captions < 350 words | Already true everywhere (154–305); the defect is blanket interval sentences false per panel. |
| Cut prose 30–45% | Results ≈ 5,842 words; trim toward 5,000 via audit-trail sentences (L197, 203, 240, 325, 327, 353) and L381, not numbers. |
| Eight-figure option (9 into 8) | Reject: merging makes the null read as a failed shunt test. |

Silent in the Deep Review, found by the panel: the Fig 6 architecture and validation-loss deletions, six dropped SI groups still cited, six uncited SI figures, the palette ΔE failures, the L132/L136 mismatches.

## 8. Work plan

1. **Text/number repairs (hours).** L132/Fig 1E; L136 → S6D; `inferential_units.csv` (scope n = 15 row, add n = 10 and `conductance_local_gate` rows); L265 readout; Fig 7D "residual" + both scales at L323; qualify L401; duplicate label; S10D label; S3C placeholder; `si_09_measured.tex` interval; SI literals → `\ref`; L355 → S29B.
2. **Restore what consolidation broke**: old S53E (to Fig 5), S32C/D (S10), S54A/C (S15), S37A/D (S11), S21A (S28), S38D and S39A–D (S33/S34), or rewrite the citing sentences; name each dropped panel in `omitted_panel_status`.
3. **Adopt the overhaul build system** with §5 changes: typeface, 7/8/9 pt, palette registers with ΔE gate, `align_letters()` unconditional, `DECORATIVE_LW_PT` 1.35, forest helper, matrix assertion, new audits.
4. **Rebuild Figs 1, 5, 6** (largest content gaps): mechanism card + cohort forest + exact−per-neuron; delivery card + cancellation + rate strip; architectures + S21B + validation loss.
5. **Rebuild Figs 2, 3, 8**: deranged curve + utility overlay + S9B; rewiring + 3 × 3 cue grid + merged forest + dictionary strip; rescue + injection arm + half-width E.
6. **Rebuild Figs 4, 7, 9**: delivery card + energy-vs-k + seed strip + S12C; real dictionary + decomposition + per-cell scatter + full-cohort F; schematic + observed-effect axis + S31B + 9C inset (correct overhaul 9's panel list first).
7. **SI re-render** natively at one scale; unify rule palette; merge S3→S2, S30→S31, S33+S34; split S7; cite S1, S3–S5, S8, S12, S22, S30; add per-SI Source Data and `figure_02_plotted.csv`.
8. **Caption and vocabulary pass**: one name per arm/endpoint/cohort/rule across panels, captions, text, CSV; ε/δ̂; disambiguate "alignment"; per-panel n/unit/interval/endpoint; Source Data pointer; soften L342.
9. **Prose trim** toward 5,000 words, L381 to two sentences; `make combined`; strict-audit all 18 PDFs before the next review.
