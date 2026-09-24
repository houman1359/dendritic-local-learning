# shunt_sensitivity — clarity-pass ledger

- **Source key:** N29 (whole sheet)
- **Builder:** `scripts/build_supplementary_figure_shunt_sensitivity_native.py` (gitignored; pre-edit copy at `si_pass/builder_backup/build_supplementary_figure_shunt_sensitivity_native.py.orig`)
- **Output:** `figures/supplementary/figure_shunt_sensitivity_native.pdf`
- **Reproducibility gate:** PASS. The unchanged builder under `python3` (Mambaforge, matplotlib 3.10.6) wrote a bitwise-identical PDF (sha256 `84a4e4a8ed1e2f0e3c2503d4453ec4a75a839e931d0cdc4ff06b115b763c5afc`, saved as `si_pass/renders/shunt_sensitivity_orig.pdf`; before render `shunt_sensitivity_before.png`).
- **After edit:** sha256 `40620b74ab2b756ba43559b51eb50f8583def9e27648fb6f2c1f5f5965c2d6db`. Strict canvas audit: 0 violations. The original had 1 (`panel-emphasis`, 1.35x), fixed here, see Layout changes. `letter_ink_audit.py`: total problems 0. Vector-mark census unchanged: 1455 paths with the same type, colour, width and item count.

## Removed

| # | Panel | Verbatim string | Caption status |
|---|---|---|---|
| 1 | A | `Dose, cable and background` (panel title) | caption already states it: '\textbf{A}, Shunt-minus-current-injection localization across fixed absolute doses at three membrane resistances (…) and three background-conductance multipliers (column groups)' |
| 2 | B | `Within-cell controls` (panel title) | caption already states it: '\textbf{B}, Shunt and matched-injection localization (grey lines pair cells) and the true relation against a reassigned foreign template (unpaired)' |
| 3 | C | `Cable selectivity` (panel title) | caption already states it: '\textbf{C}, Transport selectivity $S_k$ against full synaptic-gradient localization' |
| 4 | D | `Signed outcomes` (panel title) | caption already states it: '\textbf{D}, Signed census of the same sites' |
| 5 | E | `Direct presynaptic types` (panel title) | caption already states it: '\textbf{E}, All mapped versus directly typed contacts' |
| 6 | F | `Synaptic scales and reversal` (panel title) | caption already states it: '\textbf{F}, Joint excitatory and inhibitory conductance-scale and inhibitory-reversal sensitivity by row' |
| 7 | C | `colour: Rm as in A` (legend title) | caption already states it: '$R_m$ colours as in \textbf{A}' |
| 8 | C | `Sk = 1` (reference-rule label, drawn as S + subscript k + " = 1") | caption already states it: 'dotted line, $S_k=1$' |
| 9 | F | `reference interval` (band label) | caption already states it: 'open marker and band, the reference condition (E/I 0.35, reversal $-0.2$) and its interval' |

Shortened (count kept, wording compacted):

| Panel | Before | After | Caption status |
|---|---|---|---|
| E | `8/8 cells increase` | `8/8 increase` | caption already states it: 'the contrast increases in 8/8 cells' |
| F | `8/8 cells` (×3), `7/8 cells` (×2) | `8/8 > 0` (×3), `7/8 > 0` (×2) | caption already states it: 'right column, cells with a positive contrast' |

Axis labels changed to sentence case (text only): `shunt − current-injection localization` → `Shunt − …` (A, E y; F x); `fixed shunt conductance (nS)` → `Fixed …`; `localization index` → `Localization index` (B, C); `transport selectivity S`+k → `Transport selectivity S`+k; `descendant-gradient fraction` → `Descendant-gradient fraction`.

## Caption additions

None. Every removed string is already in the current caption.

## Kept

- `background ×0`, `background ×1`, `background ×4` (A, under the dose ticks): condition labels that separate the three column groups of A.
- `Rm 300`, `Rm 1,000`, `Rm 15,000` (A key): compact symbol key (colour, marker and dash per $R_m$).
- `background ×0`, `×1`, `×4` (C key): compact marker key (filled, open, plus).
- `shunt`, `current injection` (D key): compact colour key.
- `1.00` and `0` value labels on the D bars: a zero bar draws nothing, so each `0` marks a census value that would otherwise be invisible. The census fractions are the plotted quantity.
- `8/8 increase` (E) and `8/8 > 0` / `7/8 > 0` (F): compact sign counts.
- F row labels (`E/I 0.10` / `reversal −0.2`, …): category labels of the forest rows.

## Layout changes

- The six panel titles were removed. Each row's top reserve was already set by the 13 pt letter band, so no axes box moved. The letters dropped 5.2 pt to their panels' new topmost ink.
- F: the top y-limit of the categorical row axis changed from −1.30 to −0.62, matching the 0.62 bottom pad. That headroom only held the removed `reference interval` label. Row positions, the x scale and the band are unchanged.
- F: the declared right reserve changed from 33 pt to 26 pt, because the note column shrank from `8/8 cells` to `8/8 > 0` (21.6 pt + 3 pt offset). The reserve is locked on the shared right column edge, so B, D and F each gain 7 pt of axes width (x-limits unchanged). This also clears the original's strict-audit `panel-emphasis` violation: D's slot fill goes from 0.745 to 0.779, and the ratio from 1.35x to 1.29x.
- Unused `token_subscript` import removed.
