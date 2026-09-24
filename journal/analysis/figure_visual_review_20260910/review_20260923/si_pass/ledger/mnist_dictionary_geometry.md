# mnist_dictionary_geometry — SI text pass ledger (2026-09-23)

- Source key: N7 (whole sheet, `('N7','*')`)
- Builder: `scripts/build_supplementary_figure_mnist_dictionary_geometry_native.py` (no arguments)
- Output: `figures/supplementary/figure_mnist_dictionary_geometry_native.pdf`
- Reproducibility gate: PASS with the right interpreter.
  - `python3` (Mambaforge, matplotlib 3.10.6) differed from the saved original by exactly one byte, the PDF Producer string "v3.10.6" against "v3.10.9". The original was restored at once.
  - `/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/.venv/bin/python` (matplotlib 3.10.9, the version that wrote the original) gave a byte-identical PDF (sha256 9e39f04c…667a).
  - Files: original at `si_pass/renders/mnist_dictionary_geometry_orig.pdf`, renders at `si_pass/renders/mnist_dictionary_geometry_{before,after}.png`, builder backup in `si_pass/builder_backup/`.
- Edited: yes. New sha256 c4cb8f9c…c5c43f, 518.4 × 382 pt. It was built with the `.venv` python above; rebuild with the same interpreter so the Producer string stays the same.
- Verification: `figure_canvas.py --audit --strict` 0 violations; `letter_ink_audit.py` 0 problems (smallest gaps: top 5.4 pt, left 4.4 pt).
- Data check: the builder's printed plotted values match the original builder's output line for line. Vector paths per panel are identical; the only difference is the page background rectangle.

## Removed

| # | Panel | String removed (verbatim) | Caption status |
|---|---|---|---|
| 1 | A | `: ten fresh paired seeds` (title `Shunting: ten fresh paired seeds` → `Shunting`) | Already in the caption: "Six-rule MNIST comparison in shunting (green) and raw-additive (blue) networks, with ten fresh paired seeds per architecture." |
| 2 | B | `: ten fresh paired seeds` (title `Raw additive: ten fresh paired seeds` → `Raw additive`) | Same caption text as 1. |
| 3 | C | `Added within-tree resolution` (title) | Already in the caption: "\textbf{C} resolves the within-tree contrasts. \textbf{C}, Paired accuracy differences for three projected subtree profiles versus one projected common profile, and for exact paths versus three profiles." |

## Caption additions

None needed.

## Text re-cased (sentence case, no content change)

`test accuracy (%)` → `Test accuracy (%)` (A, B); `paired accuracy difference (pp)` → `Paired accuracy difference (pp)` (C).

## Kept

- Titles `Shunting` and `Raw additive`: short condition labels that tell the two otherwise identical ladders apart.
- C `seeds > 0` with the counts `7/10`, `5/10`, `5/10`, `3/10`, `3/10`, `4/10`, `4/10`, `7/10`: compact per-row sign counts. The caption says "counts give the number of positive differences among ten seeds".
- C group labels `K = 3 subtrees minus` / `projected K = 1` and `exact path minus` / `K = 3 subtrees`: category labels for the two contrast groups.
- A/B tick labels `Strict scalar`, `Per neuron`, `Proj. K = 1`, `K = 3 subtrees`, `Exact path`, `Decoder only`: categories.
- Bottom key `development-selected rate`, `original common rate`, `one fit for both rates`, `one seed`, `shunting`, `raw additive`: compact symbol keys, each a marker or patch plus at most four words.

## Layout changes

- Canvas 518.4 × 392 → 518.4 × 382 pt (aspect 1.32 → 1.36). The vertical gutter went from 50 to 40 pt: the band C's title used is gone, and the gutter now holds A/B's two-line tick labels plus C's letter band and the lock's pad.
- Axes boxes: A/B 200.2 × 128 pt, unchanged; C 449.4 × 130.9 pt (was 131.0).
- Letters: A/B unchanged (x 15.8/271.1, baseline 13.3 pt from the top). C has x 15.8 and baseline 186.6 pt (was 191.3), 2.5 pt above its axes.
- Consolidation: the N7 letter C moved, so the N7 entry in `original_assets.json` must be refreshed by the coordinator. I did not run `refresh_native_assets.py`.
