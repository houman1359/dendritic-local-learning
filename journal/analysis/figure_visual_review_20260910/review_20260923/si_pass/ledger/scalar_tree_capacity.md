# scalar_tree_capacity (N12) — SI text pass, group B

- Source key: N12
- Builder: `scripts/build_supplementary_figure_scalar_tree_capacity_native.py` (no argparse; run as `python3 scripts/build_supplementary_figure_scalar_tree_capacity_native.py`)
- Output: `figures/supplementary/figure_scalar_tree_capacity_native.pdf`
- Reproducibility gate: PASS. The unchanged builder rebuilt the output byte-identically (sha256 `a6f10ed7c3e250a2bfc6c89f7f03b54f81ecfd3cc9a7aed370c235853df80c41` for both the original and the rebuild). Original saved as `si_pass/renders/scalar_tree_capacity_orig.pdf`; renders `scalar_tree_capacity_before.png` / `scalar_tree_capacity_after.png`.
- Baseline note: the unchanged sheet already FAILED the strict audit with 2 violations: `edge-clearance` (the letters A and B sat 1.9 pt from the top edge, lifted onto the super-title) and `panel-emphasis` (1.74x, because B's footprint included the super-title spanning both B panels). Both are gone after this pass.
- New PDF: `figure_canvas.py --audit --strict` gives 0 violations; `letter_ink_audit.py` gives total problems 0; the render was inspected (no overlaps, clipping or odd whitespace). New sha256 prefix `1f53135a29733294`.
- Data unchanged: the builder's data printouts ([A]–[F] lines) are identical to the gate run, and the vector-path census matches the original except for the three removed grey key discs in B.

## Removed

| # | Panel | String (verbatim) | Status |
|---|---|---|---|
| 1 | A | `Fixed spectrum; different interactions` (title) | caption already states it (caption title): 'Interaction structure constrains scalar trees at matched input spectra and resources.' |
| 2 | A | `105` (count prefix of `105 perfect matchings M of x1 … x8`, which now reads `perfect matchings M of x1 … x8`) | caption already states it: '105 pairwise matchings (green circles)' |
| 3 | A | `35` (count prefix of `35 four-plus-four partitions A \| A′`) | caption already states it: '35 two-quartic targets (purple triangles)' |
| 4 | A | `24` (count prefix of `24 nested-prefix controls (D–F)`) | caption already states it: 'and 24 nested-prefix controls (amber squares)' |
| 5 | A | `(D–F)` (panel scope of the nested controls) | ADD TO CAPTION. Nested controls appear only in E and F; B and C use only the 140 matching and quartic targets. The addition says "E,F" because D shows the first matching. |
| 6 | A | `½ × products of the first 2, 4, 6, 8` / `inputs of a seeded permutation` (definitional sentence) | Replaced in the artwork by the equivalent formula `fπ = ½ Σ(k ∈ {2, 4, 6, 8}) Π(j ≤ k) xπ(j)`, the Supplementary Note equation for the nested target and in the same chained-subscript form as the other two formulas. The meaning of π ("seeded permutation"): ADD TO CAPTION. |
| 7 | A | `matching, quartic: input-gradient` / `second moment I8 / 4, rank 8` | ADD TO CAPTION |
| 8 | A | `nested: anisotropic spectrum` | caption already states it: 'The nested controls share an anisotropic input spectrum.' |
| 9 | A | `(¼, ¼, ½, ½, ¾, ¾, 1, 1)` | ADD TO CAPTION (the spectrum values) |
| 10 | A | `every tree: 7 multi-affine nodes,` / `28 coefficients, 14 edges, root readout` | The readout is in the caption: 'filled circles are multi-affine units; the root is read out below'. The budget (7 nodes, 28 coefficients, 14 edges): ADD TO CAPTION |
| 11 | B | `The centered constraint is stronger` (super-title over both B panels) | ADD TO CAPTION |
| 12 | B | `bound = NMSE` (label on the dashed equality rule) | caption already states it: 'the dashed line marks equality' |
| 13 | B | `1,680 fits:` / `140 targets` / `× 12 candidates` | '1,680 fits': caption already states it: 'across 1,680 fits'. '140 targets × 12 candidates': ADD TO CAPTION |
| 14 | B | `fits per mark`, plus the mark-area key it headed (three grey discs labelled `700`, `100`, `30`) | The heading is a method note, and the key is redundant because every mark prints its fit count (caption already states it: 'labels give fit counts'). Without its heading, the grey key read as data points. The area scaling: ADD TO CAPTION |
| 15 | C | `Selection within the twelve fixed candidates` (title) | caption already states it: 'Excess NMSE above the best fitted candidate among twelve fixed trees.' |
| 16 | D | `Parallel: minimum depth 3` (title) | caption already states it: 'A minimum-depth tree for the first matching (depth three)'. The depth ruler in D also shows depth 3. |
| 17 | E | `Nested: minimum depth 4` (title) | caption already states it: 'and the first seeded nested control (depth four)'. The depth ruler in E also shows depth 4. |
| 18 | F | `Depth constraint over all labeled trees` (title) | caption already states it: 'Minimum possible maximum centered-cut bound over all labeled binary trees at depth limits of three, four or five edges.' |
| 19 | F | `0.146` (value printed beside the nested mark) | caption already states it: 'nested controls have bound 0.146 at depth three' |
| 20 | F | `matching, n = 105` / `quartic, n = 35` / `nested, n = 24`. The key now reads `matching` / `quartic` / `nested`. | caption already states it: '105 pairwise matchings (green circles), 35 two-quartic targets (purple triangles), and 24 nested-prefix controls (amber squares)' |

## Caption additions

1. In the **A** sentence, after `and 24 nested-prefix controls (amber squares)`, insert:
   `, which appear only in \textbf{E,F}`
2. In the **A** sentence, change `The nested controls share an anisotropic input spectrum.` by inserting before its final period:
   `, $(1/4,1/4,1/2,1/2,3/4,3/4,1,1)$; matching and quartic targets share the input-gradient second moment $I_8/4$ (rank eight)`
   Then append after that sentence:
   ` In the nested formula, $\pi$ is a seeded input permutation. Every tree has seven multi-affine nodes, 28 coefficients and 14 edges, with a root readout.`
3. In the **B** sentence, after `across 1,680 fits`, insert:
   ` (140 targets $\times$ 12 candidates)`
4. In the **B** sentence, after `the dashed line marks equality`, insert:
   `; mark area scales with the fit count`
   Then append after the B sentence (after its period):
   ` The centered constraint is stronger.`

With these additions the caption grows from 223 to about 280 words.

## Kept

- `Full-cut bound, rank ≤ 2` and `Centered-cut bound, rank ≤ 1`: condition labels that tell the two otherwise identical B panels apart (≤5 words, no verb).
- `perfect matchings M of x1 … x8`, `four-plus-four partitions A | A′` and `nested-prefix controls`, with the three formulas `fM = ½ Σ(i, j) ∈ M xi xj`, `fA = ½ (Πi ∈ A xi + Πi ∉ A xi)` and `fπ = …`: the schematic content of A. This is the sheet's family key: marker, a family name that defines the formula's symbol, and the defining formula.
- `4 + 4`, `355 + 416` and the per-mark counts in B: data labels (fit counts), as the caption says ('labels give fit counts').
- `at zero excess`: the three-word header of C's per-row count column (the counts are a plotted quantity).
- C row labels (`centered-cut bound`, `full-cut bound`, `sum of centered tails`, `two-sweep fitting pilot`, `fixed balanced candidate`, `best fixed in hindsight`, `random expectation`): categorical tick labels.
- `101/105, 35/35` … `0/105, 31/35`: compact per-row counts.
- `root readout` (D, E): schematic label of the readout arrow.
- `depth` with its ruler ticks 0–4 (D, E): the schematic's axis.
- `matching`, `quartic`, `nested` (F): compact symbol key.

## Layout changes

- Removed the titles of A, C, D, E and F, and B's super-title together with its manual letter lift (`B_SUPER_TITLE`, `B_SUPER_DY_PT`). Letter B is now placed by the canvas like every other letter.
- A: the three key entries (name + formula) are spread over the full cell height (new constant `A_FOOT_PT = 3.0`), so the schematic still fills at least 85% of its cell. The six-line paragraph at the foot of A is gone.
- B: removed the mark-area key (grey discs + numbers + heading).
- Row 0 axes grew from 124.7 to 128.7 pt, because the top reserve shrank once the super-title was gone. Rows 1–2, all margins, the gutters and the canvas (518.4 × 492 pt) are unchanged.
- Letters: row 0 now at 6.8 pt from the top edge (was 2.0 pt, which failed edge clearance); C at 185.5 pt (unchanged); D–F at 342.6 pt (was 337.5 pt, now closer to their panels).
- Axis labels changed to sentence case: `Best fitted population NMSE`, `Cut-tail lower bound / target variance` (×2), `Excess population NMSE above the best of the twelve candidates`, `Depth limit (root-to-leaf edges)`, `Minimum cut bound (NMSE)`.
- Updated the module docstring to match. Removed the unused `B_KEY_N` and the unused imports `PT_EMPH` and `PT_TITLE`.
