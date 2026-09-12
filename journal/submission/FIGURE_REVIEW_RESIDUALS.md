# Figure review of 2026-09-10 to 12: what was fixed and what remains

Every panel of all 45 figures was reviewed from its rendered image against ten
criteria, each finding was put to an independent skeptic who re-read the same
image, and only findings that survived were acted on. This file is the record
of the outcome for the author. The full per-figure lists, agent reports and
deferral reasons are under `analysis/figure_visual_review_20260910/`.

## Totals

| | verified | fixed | deferred with a reason | cannot be fixed |
|---|---:|---:|---:|---:|
| Nine main figures | 157 | 114 | 67 | 0 |
| Supplement, 22 rebuildable sheets | 320 | 230 | 77 | 0 |
| Supplement, 14 frozen sheets, rebuilt natively | 197 | 201 paste and caption changes, then 195 in-panel fixes | 38 | 0 |
| Cross-figure | 35 | absorbed into captions as colour keys | | |

The 114 main-figure fixes include 18 repairs of regressions that the first
fix pass introduced and an independent checker caught. Twelve cosmetic
residuals remain on the main figures, of the kind a 1.7 pt gap where an editor
would want 3 pt; they are listed in `residuals_round2.json`.

## Corrections of substance

These changed what the paper claims, not how it looks.

- **Figure 1F** painted a CIFAR-only backpropagation equivalence margin behind
  all four cohort rows, certifying three cohorts against a test never run on
  them. It is confined to the CIFAR-10 row with its own datum.
- **Figure 1G** drew the K = 12 identity column, exactly 1.000 by construction
  in all forty seeds, as the panel's largest movement. It is a labelled rule.
- **Figure 2E** printed "after 20 switch epochs"; the released data's config
  is the twelve-epoch one. Fixed in the artwork and the text.
- **Figure 2C**'s caption named diamonds the panel no longer draws.
- **Figure 7** text cited panel E for numbers that are panel D's K = 8 column.
- **Figure 8** attributed 0.079 log units to the post-shunt adjoint alone; that
  is the paired difference, and the adjoint alone is 0.130.
- **Figure 1E** and the Results described the Fashion-MNIST contrast as
  per-neuron minus strict scalar; that cohort has no strict-scalar arm, and
  the baseline is the matched-width fallback, which the supplement measures
  as a further 4.86 points lower.
- **Supplementary S26B**'s caption claimed a persistence the drawn intervals
  did not show; the panels now print the one-sided Wilcoxon p under each tick.
- **Supplementary S29A** silently drew only the background-leak = 0 slice, in
  which the R_m = 15,000 curve is flat. All three backgrounds are drawn, and
  the caption states that the ordering reverses at background 4.
- **Supplementary S33** gained the six-animal contrast panel its finding asked
  for; **S34**'s two regret panels became one forest with both arms.
- The projection screen's retained-noise grid, a restart-snapshot epoch, a
  finite-difference maximum, a source-directory attribution and a panel
  letter in the supplement prose were wrong and are corrected.

## The fourteen frozen sheets, rebuilt

Fourteen sheets (S7, S8, S12 to S21, S35, S36) were pasted from twenty upstream
renders that had no generator anywhere in the repository. On 2026-09-12 each
was given a native builder, `scripts/build_supplementary_figure_<ident>_native.py`,
that reads only the study's frozen tables under `source_data/`, draws the sheet
on the paper's canvas, and asserts every plotted number against the table it
comes from. Each new render was checked by an independent agent that recomputed
every value from the tables without trusting the builder, read the old and new
sheets side by side, and ran the strict audit. Three were rejected on first
check and repaired; all fourteen were then accepted.

| | count |
|---|---:|
| Value mismatches against the frozen tables, across all 14 | 0 |
| In-panel findings fixed that the paste layer could not reach | 195 |
| Findings the builders declined, with a reason each | 38 |

The old frozen renders remain in the registry for provenance and every old
cross-reference label survives as an alias. The 38 declined items are in
`analysis/figure_visual_review_20260910/native_not_fixed.json`. Most are
deliberate: the builders were told to keep each sheet's panel letters and
plotted content, so findings that asked to delete a panel, merge two panels,
or drop a series were not taken; a few ask for per-seed values that no table
records; several are cross-figure restructures beyond one sheet. Two are
genuine limits of the data: S15G and S35A,B carry intervals narrower than
their markers at any scale that keeps the fans visible, and their captions
say so.

## Deferred with a reason

144 findings across the main figures and rebuildable sheets were deferred by
the fixing agents. Reading them, they are of three kinds: fixes that need a
change to a shared library and were filed as requests (a forest helper with an
aligned statistics column, a sanctioned residual strip, a break-at option for
an axis spanning orders of magnitude, an address ramp adopted by every builder);
fixes that would reverse a decision the builder's own QA history records with a
reason; and schematic redraws, which the agents were told to avoid. The
requests and the deferrals with their reasons are in the `fix_reports.json`
and `supplement_wave*_reports.json` files.

## Verification state

`make audit` passes end to end on the committed tree: 215 tests, the LaTeX
layout, figure-lineage, citation, format and submission audits, and provenance.
Every main figure and every native supplementary render passes the strict
canvas audit with zero violations, and the main figures pass the three layout
audits. All nine main-figure legends are at or under the
350-word guidance. The combined PDF is rebuilt from this state.
