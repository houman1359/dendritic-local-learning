# utility_signal_noise — SI text pass ledger (2026-09-23)

- Source key: N3 (whole sheet, `('N3','*')`)
- Builder: `scripts/build_supplementary_figure_utility_signal_noise_native.py` (run as `python3 <builder>`, no arguments)
- Output: `figures/supplementary/figure_utility_signal_noise_native.pdf`
- Reproducibility gate: PASS. The unchanged builder run with `python3` (Mambaforge, matplotlib 3.10.6) gave a byte-identical PDF (sha256 d18b9c71…f1508). Original kept at `si_pass/renders/utility_signal_noise_orig.pdf`, before/after renders at `si_pass/renders/utility_signal_noise_{before,after}.png`, builder backup at `si_pass/builder_backup/`.
- Edited: yes. New sha256 2a2a928c…1914c3, 518.4 × 458.4 pt. Rebuild with the same `python3` (matplotlib 3.10.6), the version that wrote the original.
- Verification: `figure_canvas.py --audit --strict` 0 violations (the original had 3 row-alignment violations); `letter_ink_audit.py` 0 problems (smallest gaps: top 8.9 pt, left 4.5 pt).
- Data check: the builder's printed plotted values (every mean, interval and cell value) match the original builder's output line for line. Vector paths per panel are identical except for G's removed annotation leader.

## Removed

| # | Panel | String removed (verbatim) | Caption status |
|---|---|---|---|
| 1 | A | `Fixed-operator special case` (title) | Already in the caption: "\textbf{A}, Fixed-operator one-step guarantee: …" |
| 2 | B | `Spectral capture advantage: subtree − random` (title) | Already in the caption: "\textbf{B}, Subtree-minus-random spectral capture over route budget $K$ and covariance mixture $\rho$" |
| 3 | B | `mean of 50 paired seeds per cell` (tag) | The caption has "50 paired seeds per cell" but does not say the values are means. **ADD TO CAPTION** (addition 1) |
| 4 | B | `   (16 = full rank)` (from the x label `route budget K   (16 = full rank)`) | The caption has "$K=16$ and $\rho=0$ are by-construction controls" but not "full rank". **ADD TO CAPTION** (addition 2) |
| 5 | C | `Hierarchy × resolution` (title) | Already in the caption: "\textbf{C}, Final loss against route resolution $D_{\rm r}$ for hierarchies $H_{\rm c}=1$--$4$" |
| 6 | C | `50 paired seeds per point; mean, 95 % CI within marker` (tag) | Already in the caption: "symbols, means and 95\% bootstrap intervals (\textbf{C}, seeds; …). Intervals in \textbf{C,E} lie under markers." and "50 paired seeds per point" |
| 7 | C | ` (4 = full rank)` (from the x label `route resolution (4 = full rank)`) | Already in the caption: "$D_{\rm r}=4$ resolves every leaf ($M=I$)" |
| 8 | D | `Projection boundary: Δ loss` (title) | Already in the caption: "\textbf{D}, Exact projection boundary: retaining signal fraction $f_{\rm sig}$ and noise fraction $f_{\rm noise}$ changes normalized loss by $(f_{\rm noise}-f_{\rm sig})/2$" |
| 9 | D | `cells span midpoints between the sampled fractions` (tag) | Not in the caption. **ADD TO CAPTION** (addition 3) |
| 10 | E | `Trained checkpoints,` / `one held-out batch` (two-line title) | Already in the caption: "at 120 trained checkpoints (one held-out batch; relative step $10^{-5}$)" |
| 11 | E | `n = 120; box: median, quartiles, 1.5 IQR` (tag) | Already in the caption: "120 trained checkpoints" and "Boxes show medians, quartiles and 1.5-interquartile-range whiskers" |
| 12 | E | `diamond: mean, 95 % CI narrower than marker` (tag) | Already in the caption: "open diamonds, means", "symbols, means and 95\% bootstrap intervals (… \textbf{E}, checkpoints …)" and "Intervals in \textbf{C,E} lie under markers." |
| 13 | E | `exact path = 1 by construction` (label on the dark-red dashed rule; the rule stays) | The caption has "Dashed references: exact path (dark red, 1), zero (grey)." but not "by construction". **ADD TO CAPTION** (addition 4) |
| 14 | F | `Iterative learning, eight arbors` (title) | Already in the caption: "\textbf{F}, Eight-arbor positive control: twenty-step projected loss reduction relative to the full gradient" |
| 15 | F | `n = 8 arbors per point; mean, 95 % CI` (tag) | Already in the caption: "Eight-arbor" and "symbols, means and 95\% bootstrap intervals (… \textbf{F}, cells, one Monte Carlo stream each)" |
| 16 | F | `full gradient = 1` (label on the grey dashed rule; the rule stays) | Already in the caption: "relative to the full gradient (dashed, 1)" |
| 17 | G | `Rank–noise trade-off (analytic)` (title) | Already in the caption: "\textbf{G}, Analytic rank--noise trade-off" |
| 18 | G | `q² / (q + Kσ²), no data` (tag) | Already in the caption: "$2L$ times the optimized bound $q^2/(q+K\sigma^2)$" and "Quadratic/operator analyses (\textbf{A--D,G})" |
| 19 | G | `sign change at` / `σ² = 4/7 ≈ 0.571` (callout and its leader line) | Already in the caption: "crossing at $\sigma^2=4/7$" |
| 20 | G | ` one-step` (from the y label `2L × optimized one-step bound`, now `2L × optimized bound`) | Already in the caption: "$2L$ times the optimized bound $q^2/(q+K\sigma^2)$". The 110 pt label was longer than the 95 pt axes and rose above G's letter. |

## Caption additions

Edits to the `utility_signal_noise` caption (5th element of its `FIGURES` tuple). Net effect is about +14 words. The sheet is also 31.6 pt shorter, which leaves room on the page; this caption previously overflowed with "Float too large".

1. In the \textbf{B} sentence, replace `covariance mixture $\rho$, 50 paired seeds per cell;` with `covariance mixture $\rho$, means of 50 paired seeds per cell;`
2. In the same \textbf{B} sentence, replace `$K=16$ and $\rho=0$ are by-construction controls` with `$K=16$ (full rank) and $\rho=0$ are by-construction controls`
3. In the \textbf{D} sentence, replace `the dashed boundary is $f_{\rm noise}=f_{\rm sig}$.` with `the dashed boundary is $f_{\rm noise}=f_{\rm sig}$, and cells extend to the midpoints between sampled fractions.`
4. In the \textbf{E} sentence, replace `Dashed references: exact path (dark red, 1), zero (grey).` with `Dashed references: exact path (dark red, 1 by construction), zero (grey).`

## Text re-cased (sentence case, no content change)

`covariance mixture ρ` → `Covariance mixture ρ`; `route budget K …` → `Route budget K`; `final loss` → `Final loss`; `route resolution …` → `Route resolution`; `retained signal fraction` → `Retained signal fraction`; `retained noise fraction` → `Retained noise fraction`; `dimensionless value` → `Dimensionless value`; E group labels `gradient cosine` / `one-step progress` → `Gradient cosine` / `One-step progress`; `credit aligned to routes (%)` → `Credit aligned to routes (%)`; `relative 20-step progress` → `Relative 20-step progress`; `noise variance σ² (arbitrary units)` → `Noise variance σ² (arbitrary units)`.

## Kept

- A schematic labels `update` / `−η M(μ + ξ)`, `routes M`, `mean` / `gradient μ` and the card `U(M) = max(mean alignment, 0)² /` / `(2L × mean squared update)`. These label the schematic's parts and its defining ratio, and the caption gives the exact definition. They are drawn by the shared helper `build_utility_supplement.operator_schematic`, which is not an assigned file.
- E `−1.48, −1.34`: the values of the two strict-scalar checkpoints drawn at the axis floor. This is data the caption points to ("Two below-range scalar values are labelled at the floor").
- E group labels `Gradient cosine`, `One-step progress`: category axis labels for the two halves of the shared axis.
- C legend `hierarchy 1`–`hierarchy 4`, F legend `morphology-selected`, `random paths`, `depth bins`, `ancestry-shuffled`, G legend `K = 1, q = 0.8`, `K = 2, q = 1`: series keys.
- The B and D printed cell values: data.

## Layout changes

- Canvas 518.4 × 490 → 518.4 × 458.4 pt (aspect 1.06 → 1.13). Top margin 26 → 21, vertical gutter 44 → 40, row weights [104, 120, 116] → [104, 120, 97.4]. The lock still carves 2.2 pt above row 2.
- Axes heights are unchanged: A/B 104, C/D 120, E–G 95.2 pt. The left margin went from 40 to 41 pt so that column 0 takes the same 6 pt declared reserve as columns 4, 6 and 8. Widths are now A–D 206.7 pt (was 206.3/207.2) and E/F/G 123.7/123.6/123.7 pt (was 123.1/123.9/124.0). This removes the three pre-existing strict row-alignment violations.
- Letters sit 2.5 pt above their axes tops, with baselines 18.5 (A, B), 162.5 (C, D) and 324.7 (E–G) pt from the top. Their x positions are unchanged at 4.1, 260.2, 174.9 and 339.2 pt. Row and column alignment is exact.
- Consolidation: the letter positions changed, so the N3 entry in `original_assets.json` must be refreshed by the coordinator. I did not run `refresh_native_assets.py`.
