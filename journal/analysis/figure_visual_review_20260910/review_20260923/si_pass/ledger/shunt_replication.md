# shunt_replication — clarity-pass ledger

- **Source key:** N30 (whole sheet)
- **Builder:** `scripts/build_supplementary_figure_shunt_replication_native.py` (gitignored; pre-edit copy at `si_pass/builder_backup/build_supplementary_figure_shunt_replication_native.py.orig`)
- **Output:** `figures/supplementary/figure_shunt_replication_native.pdf`
- **Reproducibility gate:** PASS. The unchanged builder under `python3` (Mambaforge, matplotlib 3.10.6) wrote a bitwise-identical PDF (sha256 `99a77f980e0ad8aff168fe199023c5864e16dd684ace7ce81e02f0e7bd6d72c3`, saved as `si_pass/renders/shunt_replication_orig.pdf`; before render `shunt_replication_before.png`).
- **After edit:** sha256 `c6ad28410a72c088f9a92e1b0cd4a365c84270e5cd47e6af096bfd4cea953b25`; canvas 518.4 × 398.2 pt (was 414.0). Strict canvas audit: 0 violations. `letter_ink_audit.py`: total problems 0. Vector-mark census unchanged: 551 paths with the same type, colour, width and item count. Every axes box keeps its size, and the builder's whisker-versus-marker report is identical.

## Removed

| # | Panel | Verbatim string | Caption status |
|---|---|---|---|
| 1 | A | `Matched perturbations` (panel title) | caption already states it: '\textbf{A}, Focal shunting (green) versus baseline-current-matched injection (blue)' |
| 2 | B | `True versus reassigned relation` (panel title) | caption already states it: '\textbf{B}, True (green) versus reassigned (rose) descendant relations' |
| 3 | C | `Direct-label coverage` (panel title) | caption already states it: '\textbf{C}, Direct excitation/inhibition label coverage and selected focal-site counts' |
| 4 | D | `Weak-channel linearization versus passive` (panel title) | caption already states it: '\textbf{D,E}, Separate eight-cell weak-channel ensemble, linearized at each solved steady state' and 'Inset: active-minus-passive localization at matched calibration' |
| 5 | E | `Weak-channel localization contrasts` (panel title) | caption already states it: '\textbf{E}, Paired shunt-minus-injection contrasts' |
| 6 | D | `attenuates descendants` (under the focal-shunt label) | caption already states it: 'Signed effects show descendant attenuation by shunting and enhancement by injection in every cell at every dose.' |
| 7 | D | `enhances descendants` (under the current-injection label) | caption already states it (same sentence as #6) |
| 8 | D inset | `per cell` (second line of the inset label `active − passive localization,` / `per cell`) | ADD TO CAPTION (see below) |
| 9 | E | `no effect` (zero-rule label) | caption already states it: 'Dashed lines mark zero.' |
| 10 | C | class counts in the key: `L2IT (34)`, `L3IT (2)`, `L4IT (10)`, `L5ET (1)` → `L2IT`, `L3IT`, `L4IT`, `L5ET` | caption already states it: 'Markers distinguish L2IT (34 cells), L3IT (2), L4IT (10) and L5ET (1)' |
| 11 | C | `(max 16)` from the y label `selected focal sites (max 16)` | caption already states it: 'selected focal-site counts, capped at sixteen per cell' |

Shortened (count kept, wording compacted):

| Panel | Before | After | Caption status |
|---|---|---|---|
| A | `45/45 cells positive` | `45/45 > 0` | the count is not defined for A in the caption; see the caption change below |
| B | `39 positive, 1 tie / 40 cells` | `39/40 > 0, 1 tie` | caption: 'Counts give cells with positive paired contrasts.' The tie is not explained; see below |
| E | `8/8 cells positive at every dose` | `8/8 > 0 at each dose` | caption already states it: 'positive in all eight cells at each dose' |

Axis labels changed to sentence case (text only): `localization index` → `Localization index` (A, B, D); `dose / local input conductance` → `Dose / …` (D x, E y); `shunt − current-injection localization` → `Shunt − …` (E x); `direct E/I labels (% inputs)` → `Direct …` (C x); `selected focal sites (max 16)` → `Selected focal sites` (C y); inset `dose` → `Dose`; inset label `active − passive localization,` → `Active − passive localization` (one line).

## Caption additions

1. **D sentence:** after 'Inset: active-minus-passive localization', insert ` per cell (dots; open symbols, means)`. The sentence then reads: 'Inset: active-minus-passive localization per cell (dots; open symbols, means) at matched calibration ($R_m=1{,}000\,\Omega\,\mathrm{cm}^2$, background conductance equal to leak).'
2. **B sentence:** replace 'Counts give cells with positive paired contrasts.' with:
   `Counts in \textbf{A,B} give cells with positive paired contrasts; the tie in \textbf{B} is a contrast within $10^{-12}$ of zero.`
   This defines the compact `45/45 > 0` in A (the caption defined the count only for B) and the `1 tie` in B (`zero_tolerance` = 1e-12, `cells_tied_within_tolerance` = 1 in `replication_summary.json`). The `1 tie` was also unexplained before this pass.

## Kept

- `45/45 > 0`, `39/40 > 0, 1 tie`, `8/8 > 0 at each dose`: compact sign counts (the A/B/E results). "at each dose" is needed because one line serves E's three rows.
- `focal shunt`, `current injection` (D): direct series labels.
- `L2IT`, `L3IT`, `L4IT`, `L5ET` (C key): compact symbol key.
- `Active − passive localization` (D inset): the inset's y-axis title.
- Category ticks `matched current` / `injection`, `focal` / `shunt`, `reassigned` / `relation`, `true` / `relation`: category labels.

## Layout changes

- The five panel titles were removed, along with their title pads (`TITLE_PAD_KEYED` 24 pt and `TITLE_PAD_BAND` 14 pt) and the now-unused `PT_EMPH` import.
- Row 0 declares an 8 pt top reserve (`ROW0_TOP_PT`) for C's two-row class key and the A/B count lines. `ROW_PT[0]` changed from 150 to 140.2 pt (132.2 + 8), so the row-0 axes boxes keep their 116.1 × 132.2 pt size.
- `VGUTTER_PT` changed from 50 to 44 pt: the row-1 titles no longer sit in the gutter. The row-1 axes boxes are unchanged (240.7 × 158.0 and 157.7 × 158.0 pt).
- Canvas height changed from 414.0 to 398.2 pt (aspect 1.25 → 1.30). All letters stay on the module grid, and the letters within each row are level.
