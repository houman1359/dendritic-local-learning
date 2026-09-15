# Figure review of 2026-09-10 to 13: what was fixed and what remains

**Later panel-placement pass (15 September 2026).** The remaining letter-clearance
and row-spacing defects were repaired across the main paper and supplement.
The current verification record and PDF fingerprints are in
`analysis/reviews/panel_letter_followup_20260914/COMPLETED.md`. The counts and
remaining-item descriptions below belong to their dated review snapshots.

Every panel of all 45 figures was reviewed from its rendered image against ten
criteria, each finding was put to an independent skeptic who re-read the same
image, and only findings that survived were acted on. This file is the record
of the outcome for the author. The full per-figure lists, agent reports and
deferral reasons are under `analysis/figure_visual_review_20260910/`.

**Current inventory correction (2026-09-14).** The supplement manifest contains
26 whole-source sheets and 10 panel compositions: eight combine multiple source
assets, and S5 and S26 reflow panels from one source each. All sheet and panel
paste scales are 1.0. The review counts and verification statements below
describe their dated snapshots; they do not establish the outcome of a later
build or replace the final-version audit logs.

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

## Native-builder expansion (review of 2026-09-13)

Four sheets still pasted crops of renders that no builder could produce -- S3
took two panels from a script that trains models, S22 and S31 from frozen
renders, S30 from a frozen weak-channel render -- and S29 pasted a legacy
non-canvas render, which is why it was the one sheet that could not be pasted
at full scale. All five were given native builders on the pattern of the
fourteen rebuilt the day before: each reads only the frozen tables under
`source_data/`, asserts every plotted number against the table it comes from,
and draws the whole sheet on the paper's canvas.

| | count |
|---|---:|
| Value mismatches against the frozen tables, across all five | 0 |
| In-panel findings fixed | 94 |
| Findings the builders declined, with a reason each | 16 |
| Caption claims re-checked against the drawn sheet | 232 |
| Caption claims that were wrong, and are now corrected | 13 |

Each render was checked by an independent agent that recomputed every value
from the tables without trusting the builder, read the old and new sheets side
by side and ran the strict audit; three were rejected on first check and
repaired. A second pass then read each new caption against its own render,
claim by claim -- every colour, marker, line style, reference rule, count and
interval -- and found thirteen small false statements (a rule described as grey
that is drawn dark, an axis break given as 9--28 pp where it is 9--28.5, an
interval called hidden that is merely narrow, a surrogate comparison stated as
fact where the intervals overlap). All thirteen are fixed, and two summary
markers their own seed clouds had filled are drawn open again.

**Current production inventory (checked 2026-09-14).** Of the 36 sheets,
26 are pasted whole and 10 compose selected panels (eight from multiple source
assets and two single-source reflows). All sheet and panel paste scales are
1.0, so none is reduced by the paste layer, but the ten compositions retain
explicit crop and placement geometry. The historical review reported
byte-identical native rebuilds; that does not mean a composed sheet is
identical to any one source render. Reproducibility of a later revision must
be checked against that revision's build and audit records.

**Source Data.** The released inventory named the retired render for every
sheet that had been rebuilt. Ninety-five rows now name the native render and
the builder that draws it, and 327 rows were added for every table those
builders actually read -- found by instrumenting the builders' file reads, not
by trusting the configuration. One builder had been asserting its own values
against the published plotted-values table it helps generate; it now asserts
against the frozen study tables instead.

**House tokens.** The paste layer was stroking every redaction box at the PDF
default width, putting one stray 1.0 pt path into each cropped sheet; it no
longer does. The upstream renders that still used their own type sizes, stroke
weights and role colours were brought onto the canvas ladder (38 fixes across
nine builders, no plotted value moved, each checked against the previous render
path by path). The strict audit of the 36 sheets went from 54 notes to 7, and
not one of them is about type, stroke weight or colour any more: three sheets
whose centred rows fill 87--91 % of the page width instead of 92 %, and four
short sheets whose page is wider than the 1.55 aspect the canvas prefers.
These were the residuals recorded at that stage; their counts are not the
current composition inventory or a current audit outcome.

## The nine main legends, read four times (2026-09-13)

The legends of the nine main figures had never been checked claim by claim
against what the figures draw. Four independent readings were run, each
recomputing every number from the frozen tables and reading the rendered
figure rather than trusting the builder:

| reading | claims checked | issues found | blocking | major |
|---|---:|---:|---:|---:|
| first | 620 | 61 | 2 | 13 |
| second, of the corrections | -- | 32 | 0 | 2 |
| third, cold | 541 | 46 | 0 | 17 |
| fourth, cold | 556 | 4 | 0 | 0 |

All 143 are fixed. The count rose at the third reading because fresh readers
found long-standing defects the first pass had missed, and fell to four at the
fourth, every one of them a small regression the third round's own edits had
introduced. That is the convergence this record claims: not that the legends
are perfect, but that two independent cold readings in a row now find nothing
false in them.

**What a reader would have been told wrongly.** Figure 2's caption gave the
sign of its grey tags backwards, so a reader following the caption would have
concluded that the deranged route is 25 points *better* than branch-specific
selection. Figure 4 named its pooled marker a black diamond where the panel
draws a plus, and described panel H as an NMSE when it plots a
shuffled-minus-compatible difference. Figure 1 told the reader that every
interval in two panels is a seed bootstrap; the CIFAR-10 rows are paired
Student-$t$ intervals, which the tables reproduce to one part in $10^{10}$.
Figure 8 listed the shunt conductance $\kappa=0.390$ among the per-block gains.
Figure 5 called its inset the teacher voltage when it draws the teacher's
terminal voltage, and left the green of its addressed subtree undefined.
Eleven body sentences pointed at panels that do not hold what they cite.

**Three corrections changed the artwork, because the figure itself made the
false claim.** Figure 1's panel titles asserted bounds its own points violate
("Resolution: $\le$ 0.2 pp" against a drawn $-0.86$); they now state the drawn
range. Figure 7's panel H printed "one mouse per cohort" beside a caption that
had been corrected to two mice. Figure 6's panel drew and labelled its
unresolved-crossing window as epochs 300--330, five epochs past the last epoch
whose interval straddles zero; the band and its label now end at 325.

**Source Data.** Figure 7's released plotted table was a stale 114-row
truncation of the builder's own 620-row record: it held neither panel G's 188
per-cell differences nor panel H's random-route series, both of which the
figure draws. The builder now writes the released copy itself, so the caption's
pointer resolves to a file that contains what the panels show.

**A reproducibility trap found on the way.** Of the five figure entry points in
`build_restored_main.py`, three (Figures 1, 7 and 9) were superseded by their
own builders during the overhaul and crashed with an `AttributeError` on a
helper that had moved. They now exit with the name of the builder that draws
each figure.

## The design pass (2026-09-14)

The author's verdict after the four readings was that the figures were true
but did not look like a journal's: every figure was the same 518 x 490 pt,
panel area followed the grid rather than the content (Fig. 1G gave eight
points the area Fig. 1D gave sixty; Fig. 3E and 3F filled a row for a
four-row forest and six points), data panels carried claim titles and
sentences of statistics, and 40-pt gutters left white bands. All nine were
re-laid out on 2026-09-14 (`analysis/figure_visual_review_20260910/
DESIGN_AUDIT_20260914.md` has the per-figure table and the outcome):

- panel widths and row heights follow the marks (heights now 444-472 pt);
- no data panel carries a title; schematics keep short noun titles;
- every sentence inside a plot box is gone, after checking that the caption
  or the Results sentence carries the fact -- two captions were edited to
  carry what nothing else did (Fig. 2E's paired contrast; Fig. 3E and 3F for
  the redrawn panels), and every printed value that left a panel is now
  asserted against its source table inside the builder instead;
- Fig. 3F is three facets (soft, hard, paired accuracy minus oracle), Fig. 3E
  a five-row forest with its wins and Holm P in the label column, Fig. 2B's
  and Fig. 5B's delivery cards are drawn a quarter larger;
- shared-axis pairs and strips (Fig. 2 C-E and F-H, Fig. 3 C/D and the F
  facets, Fig. 4 C-E) have equal widths and equal gaps by declared reserves.

Every figure passes the strict canvas audit, the three layout audits, the
caption-sync and cross-reference tests and the provenance refresh.

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

## Verification state recorded after the design pass (historical)

The design-pass record reported that `make audit` passed end to end on its
committed tree: the tests, LaTeX layout, figure-lineage, citation, format and
submission audits, and provenance. It also reported zero strict-canvas
violations for every main figure and all nineteen native supplementary
renders, passing main-figure layout audits, nine main legends at or under
350 words, no supplementary page overflows, and a rebuilt combined PDF.
These statements apply to that snapshot. Later edits require fresh checks
and a combined PDF built from the same source version.
