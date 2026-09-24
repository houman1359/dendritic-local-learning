#!/usr/bin/env python3
"""Main figure 9 (``fig:boundary``): measured responses bound the anatomical
alignment proposal.

Six lettered panels on one native canvas, three rows of 5 + 7 modules:

    A  schematic  Shared path vs similarity          (r0 c0-4, 124 pt)
    B  data       No ancestry alignment in either cohort (r0 c5-11)
    C  data       All four measures null              (r1 c0-4, 114 pt)
    D  data       Observed effect below the detection point (r1 c5-11)
    E  schematic  One input per route                 (r2 c0-4, 108 pt)
    F  data       Routes reach half the mapped inputs (r2 c5-11)

Canvas 518.4 x 490.0 pt (aspect 1.058, on the sanctioned 340/415/490 ladder,
CF-1).  Schematic area on the B12 / CF-10 formula
``sum(schematic slot w_pt * h_pt) / (live_w_pt * live_h_pt)``
= (176.8*124 + 176.8*108) / (464.4 * 434) = 41,017.6 / 201,549.6 = **20.4 %**
(cap 30 %; no waiver -- the three G4 waivers are Figs 1, 3 and 5).  The v1
draft's 16.6 % used the full canvas as the denominator and is withdrawn.

Glyph-rule notes carried into the manifest:

* Panel A declares the set-wide CF-4 delta-0 exemption with the reason
  ``stimulus recording; no credit is delivered in this experiment``; panel E
  carries the figure's one delta-0, dashed because the field is imposed.
* Panel B's thirteen descriptive scan circles ride 0.45 row-units below the
  second row.  That is NOT the CF-6 ``+0.22`` second-arm offset -- it is a
  within-row distribution rug (AMENDMENTS rejudge item 1), and it is recorded
  here so QA does not read 0.45 as a violation.
* B and C are two of the set's nine ``figure_canvas.forest()`` panels (CF-6).
* Fig 9 uses no in-axes key (CF-5: the set's only sanctioned key is Fig 5C),
  no ``ORDINAL_RAMP`` entry, and neither ``bp`` nor ``additive`` (B14).
  ``shunting`` is the ancestry statistic under test in B, C, D and F.

Private helpers beyond the shared library (DECISIONS G5, reported in
IMPLEMENTATION_NOTES.md under "library follow-ups"):
``_delta_hat`` (a hatted delta token: Nimbus Sans has no U+0302 and no
precomposed delta-with-circumflex and mathtext is forbidden, so the hat is
DRAWN as a two-segment polyline centred on the delta's measured advance
width, apex 0.5 pt above its measured cap height -- the v1 second ``^`` glyph
read as a stray superscript at 800 dpi), ``_split_duplicates`` (panel F's
duplicate separation, below), ``_seat_arrows`` (panel E's delivery-arrow
re-seating, below), ``_data_inset`` (``Frame.axes_inset`` needs a Frame, and D
is a data panel), and ``_wrapped`` (pre-broken footer lines: ``Frame.footer``
wraps to the frame width, and the two schematic footers must break at the
phrase, not the word).

Two further QA repairs (2026-09-09) that are geometry, not content:

* **Panel F duplicate separation.** Two of the thirteen scans share an exact
  ``(n_sites, coverage)`` pair -- ``(10, 0.40)`` once with and once without
  four single-input routes, and ``(9, 0.4444)`` twice -- so the raw scatter
  drew only eleven countable markers.  ``_split_duplicates`` spreads the
  members of each duplicate group symmetrically on x in steps of
  ``DUP_OFFSET`` = 0.40 mapped inputs (+/-0.20 for a pair, just over one
  ``MARKER_MS`` 4.6 diameter on this axis), ordered by
  ``(target_root_id, session, scan_idx)`` so the displacement is
  deterministic; y (the claim) is never moved, and the filled series is now
  drawn LAST so no marker can be covered.
* **Panels B, C and D reference rules.**  The y limits of B and C are padded
  to carry their annotation blocks, so ``forest()``'s full-height reference
  ``axvline`` ran through seven lines of 7 pt text and read as a
  strike-through.  Both call ``forest(..., reference=None)`` and draw the
  CF-7 dashed ``mute`` ``LW_REF`` zero rule here, bounded to the row bands
  plus (in B) the scan rug and skipping the inter-row band that carries B's
  cohort-bracket label; D's ``x = 0`` rule is a bounded segment that stops at
  ``y = 0.80``, below BOTH the lambda annotation and the two-line
  ``0.249 measured`` / ``0.248 perfect`` readout -- at ``y = 1.0`` the dashes
  still ran through the decimal point of ``0.249`` (QA 2026-09-09).  The
  right-aligned ``no alignment`` label stays on each line.

Deviations from ``v2/fig9/PLAN.md`` §2, and why (all recorded in TEXT.md too):
row weights and the vertical gutter are **[124, 114, 108] with vgutter 44**, not
[126, 116, 112] with 40.  At 40 pt the 41 pt band between rows holds one row's
x tick labels plus x label and the next row's letter band, and
``audit_row_separation.py`` measured 6.5 / 6.7 pt against its 8.5 pt (3 mm)
floor; 44 pt gives 9.6 / 9.6 pt.  Canvas height, margins, module split and
letter columns are unchanged (22 + 124 + 44 + 114 + 44 + 108 + 34 = 490), so
``live_h_pt`` is still 434 and the CF-10 schematic fraction is 20.4 %, not the
plan's 20.9 %.

Two further QA repairs (2026-09-09, second pass):

* **Panel A's pair-1 leader** starts from the MEASURED right edge of the
  green ``shared path`` label plus 3 pt, not from a nominal ``+17 pt``
  offset, which put the hairline on the final ``h``.
* **Panel D's false-positive rule** ends at ``x = 0.205``, not 0.235, so a
  clear ~4 pt gap separates its last dash from the inset's ``-0.2`` tick
  label; and D's ``x = 0`` rule stops at ``y = 0.80`` (see above).

**Deviation: the plan's §2 panel table lists B's two means and C's four means
with their intervals as printed on-panel; they are NOT drawn.** Both forests
run to the canvas right margin (deviation 4: the library's outside-spine note
placement already ran 4.7 pt off-canvas in B), so there is no right gutter to
print into; set inside the axes at 7 pt, ``-0.069 [-0.251, 0.106]`` is ~79 pt
wide = 0.45 of B's 1.0 data range and would be right-aligned across the
positive dots and the upper interval whisker in both panels.  The numbers are
carried by the body sentence at L377 and, since 2026-09-10, by B's own
caption sentence (the earlier claim that "the caption carries them" was
false: the caption printed no estimate for B or C at all); the graphical
diamond + interval + the on-panel ``3/7 +`` / ``4/7 +`` counts remain.

Four QA repairs (2026-09-10, third pass -- the residual-review round):

* **Panel E's route capsules leave ``K_CYCLE``** (deviation 23).  The library
  colours a ``mode='subtree'`` delivery from ``journal_style.K_CYCLE`` =
  (shunting, additive, local, oracle), so the card printed a 16 % tint of
  ``additive`` #20509E and one of the ``local``/``scalar`` amber #E2A23F --
  the two hues PLAN section 4 (B14) bars from this figure, and the reason its
  own acceptance check 6 failed.  Following the AMENDMENTS item-27 precedent
  for Fig 7B, ``_retint_capsules`` re-tints the four patches the library just
  added with ``CAPSULE_CYCLE`` = (shunting, oracle, shunting, oracle): the two
  hues check 6 admits, alternating in target order so the sibling capsules on
  ``T7`` and ``T8`` (whose entry stubs overlap at their shared junction) never
  share a tint.  The library is NOT edited (SPEC_ERRATA #7); geometry, count,
  16 % strength and the 0.55 pt edge are unchanged.
* **Panel D names lambda and lengthens the 80 % drop** (deviation 24).  The
  annotation is two ``PT_BASE`` lines, ``ancestry variance lambda = 0.65`` /
  ``respecting the MC band`` -- lambda is the simulated fraction of reliable
  tuning variance assigned to the ancestry kernel (``PROTOCOL.md``), and the
  symbol was previously undefined anywhere on the figure.  The single drop at
  the crossing now runs 0.80 -> 0.56 instead of 0.80 -> 0.72; it stops at the
  top of the reliability inset because the corridor below is occupied by that
  inset's own ``records`` axis label and 0 / 20 / 40 tick labels, and below
  those by the observed rug and its in-axis label.  The inset title moves to
  x = 0.285 so it clears the drop by ~8 pt, and the ``measured reliability``
  leader starts at x = 0.288 so the drop crosses its middle rather than
  landing on its endpoint.
* **Panel F duplicate separation raised to 0.40** (deviation 14).  At 0.26
  mapped inputs the pair members were ~3.3 pt apart against ``MARKER_MS``
  4.6, so the ``(10, 40.0 %)`` ring and disc still fused at print scale.
* **Panel A's shared inset axis labels** (deviation 25).  ``response`` is one
  rotated ``PT_BASE`` label shared by both tuning insets, now centred between
  their two centres, and each leader lands on the far half of its inset's
  left edge (85 % for pair 1, 15 % for pair 2) so neither terminates inside
  the label's ~28 pt box.

Every printed number is read here from the Source Data files recorded in
``figure_08_sources.json``; nothing is typed in.
"""
from pathlib import Path
import argparse
import sys
import json
import hashlib

import numpy as np
import pandas as pd

J = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J / 'scripts'))
sys.path.insert(0, str(J / 'code/reconstructed_tree'))

from figure_canvas import (NativeCanvas, Margins, COLORS, PT_BASE, PT_EMPH,
                           LW_HAIR, LW_ERR, LW_REF, LW_DATA,
                           MARKER_MS, SEED_MS, style_panel)
from journal_style import (label_color, strengthen,
                           style_direct_color_labels, tint_pct)
from native_schematics import Frame
from analyze_microns_morphology_credit import ancestry_matrix, parent_map

S = J / 'source_data'
OUT = J / 'figures/components/credit_first_figure_08.pdf'
MAIN = J / 'figures/main/figure_10.pdf'
REC = S / 'credit_first_figures'

ROUTE = COLORS['shunting']    # the ancestry statistic under test (B14 role)
CEIL = COLORS['oracle']       # the perfect-reliability ceiling comparator
GRAY = COLORS['mute']         # scaffolding and non-ancestry comparators
INK = COLORS['ink']
REP_TARGET = 864691135810666525   # representative scan, fixed by median rule
DELTA0_REASON = ('stimulus recording; no credit is delivered in '
                 'this experiment')
DUP_OFFSET = 0.40     # mapped inputs; +/-0.20 for a coincident pair (panel F)
# Builder-local address cycle for panel E's four route capsules (deviation 23).
# ``journal_style.K_CYCLE`` is (shunting, additive, local, oracle) and PLAN
# section 4 / AMENDMENTS B14 bar BOTH ``additive`` and ``local`` from this
# figure; SPEC_ERRATA #7 forbids editing the library, so the capsules
# ``credit_delivery`` draws are re-tinted here with the only two hues the
# plan's acceptance check 6 admits, alternating in target order so the two
# sibling capsules (T7, T8) never share a tint.
CAPSULE_CYCLE = ('shunting', 'oracle', 'shunting', 'oracle')
CAPSULE_PCT = 16
_WORDS = ('no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
          'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen')


# ── private helpers (DECISIONS G5) ───────────────────────────────────────
def _delta_hat(f, xy, tail=' = A c', *, size=PT_BASE, color=INK):
    """``δ̂`` + ``tail`` anchored at ``xy`` (left edge of the delta).

    Nimbus Sans carries neither U+0302 (combining circumflex) nor a
    precomposed delta-with-hat, and mathtext is forbidden (CF-2).  The v1
    build set a second ``^`` glyph above the delta; at 800 dpi that caret was
    a detached mark a full x-height clear of the letter and offset left of
    its centre, so it read as a stray superscript rather than a hat (QA
    2026-09-09).  The hat is therefore DRAWN: a two-segment polyline at the
    stroke weight of the type, centred on the delta's measured advance width
    and with its apex 0.5 pt above the delta's measured cap height.
    """
    base = f.text(xy, 'δ', size=size, color=color, ha='left', va='center')
    f.ax.figure.canvas.draw()
    r = f.ax.figure.canvas.get_renderer()
    bb = base.get_window_extent(renderer=r).transformed(
        f.ax.transData.inverted())
    w_pt = (bb.x1 - bb.x0) * f.w_pt
    x_mid = (bb.x0 + bb.x1) / 2.0
    apex = bb.y1 + f.fy(0.5)
    half = f.fx(min(2.1, 0.42 * w_pt))
    f.ax.plot([x_mid - half, x_mid, x_mid + half],
              [apex - f.fy(1.5), apex, apex - f.fy(1.5)],
              color=color, lw=f.lw(LW_HAIR), solid_capstyle='round',
              solid_joinstyle='miter', zorder=6, clip_on=False)
    f.text((bb.x1 + f.fx(0.8), xy[1]), tail.lstrip(), size=size, color=color,
           ha='left', va='center')


def _data_inset(ax, rect, *, grid='none'):
    """A styled inset on a DATA axes (``Frame.axes_inset`` needs a Frame)."""
    inner = ax.inset_axes(rect, transform=ax.transData)
    inner.set_facecolor('white')
    style_panel(inner, grid=grid)
    return inner


def _wrapped(lines):
    """Join pre-broken footer lines (the break points are chosen by phrase)."""
    return '\n'.join(lines)


# ── panel A: the statistic, on one arbor ─────────────────────────────────
def panel_statistic(f, subtitle_lines):
    """Two contact pairs on one arbor; one shares a soma-to-ancestor path."""
    # Review pass 2026-09-23: the title, the two count lines and the
    # `Stimulus-response recordings; no learning assay` footer are legend A's
    # sentences; the schematic takes the whole slot.
    band = 0.0
    sub_pt = 10.5 * len(subtitle_lines)
    top = 1.0 - f.fy(sub_pt)
    for i, line in enumerate(subtitle_lines):
        f.text((0.5, 1.0 - f.fy(10.5 * i + 5.2)), line, size=PT_BASE,
               color=GRAY, ha='center', va='center')

    core_y0, core_h = f.fy(band), top - f.fy(band)
    core_pt = core_h * f.h_pt
    foot_pt, head_pt = 11.0, 9.0        # tag strip under the soma / over the canopy
    tree_w_pt = min(92.0, 0.60 * f.w_pt)   # 2026-09-23: 80 -> 92, fills the freed slot
    nodes = f.balanced_tree((-f.fx(3.0), core_y0 + f.fy(foot_pt), f.fx(tree_w_pt),
                             f.fy(core_pt - foot_pt - head_pt)),
                            depth=3, mode='forward', labels=True, trunk=True,
                            output='y')
    canopy = max(nodes[t][1] for t in nodes.terminals)

    # pair 1 -- shares a soma-to-ancestor path (both contacts inside one patch)
    f.partition(nodes, [['S', 'J1', 'JL', 'JLL', 'T1', 'T2']],
                colors=['shunting'], labels=None, pct=16)
    for name in ('T1', 'T2'):
        f.contact(nodes[name], kind='exc')
    pair1_x = (nodes['T1'][0] + nodes['T2'][0]) / 2.0
    # 2026-09-23: the name sits just above the capsule it names, not at the
    # top of the slot 40 pt away
    tag1 = (pair1_x, max(nodes['T1'][1], nodes['T2'][1]) + f.fy(9.0))
    lab1 = f.text(tag1, 'shared path', size=PT_BASE, color=label_color(ROUTE),
                  ha='center', va='center')

    # pair 2 -- one contact in each half-tree, no shared path above the soma
    for name in ('T3', 'T7'):
        f.contact(nodes[name], kind='exc')
        _annotation_leader(f, nodes[name], nodes.soma)
    f.text((nodes.soma[0], core_y0 + f.fy(1.5)), 'no shared path',
           size=PT_BASE, color=GRAY, ha='center', va='bottom')

    # The recorded presynaptic partners in this functional cohort are
    # excitatory. Inhibitory-bearing routes belong to the separate anatomy
    # analyses, not to this measured-partner schematic.
    f.contact(nodes['T5'], kind='exc')

    # tuning sketches, one per pair, with a leader back to the pair
    col_x = f.fx(tree_w_pt + 10.0)
    w = 1.0 - col_x
    h_pt = min(19.0, max(9.0, (core_pt - 34.0) / 2.0))
    bots = (core_y0 + f.fy(core_pt - h_pt - 11.0),
            core_y0 + f.fy(foot_pt + 0.5))
    # the pair-1 leader starts clear of the 'shared path' label's own box:
    # a nominal +17 pt offset put the hairline on the final 'h' (QA
    # 2026-09-09), so the box is measured and the leader starts 3 pt right of it
    f.ax.figure.canvas.draw()
    _bb1 = lab1.get_window_extent(
        renderer=f.ax.figure.canvas.get_renderer()).transformed(
        f.ax.transData.inverted())
    anchors = ((_bb1.x1 + f.fx(3.0), tag1[1]),
               f._off(nodes['T7'], 2.6, 0.0))
    for i, (title, phase) in enumerate((('pair 1', 0.06), ('pair 2', 0.46))):
        y0 = bots[i]
        f.text((col_x, y0 + f.fy(h_pt + 1.5)), title, size=PT_BASE, color=INK,
               ha='left', va='bottom')
        inner = f.axes_inset((col_x, y0, w, f.fy(h_pt)))
        t = np.linspace(0.0, 1.0, 80)
        for shift, colour in ((0.0, INK), (phase, GRAY)):
            inner.plot(t, 0.5 + 0.40 * np.sin(2 * np.pi * (t + shift)),
                       color=colour, lw=LW_REF, solid_capstyle='round')
        inner.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(-0.05, 1.05))
        for name, spine in inner.spines.items():
            spine.set_visible(name in ('left', 'bottom'))
            spine.set_linewidth(LW_HAIR)
            spine.set_color(COLORS['edge'])
        # the leaders land on the FAR half of each inset's left edge, away
        # from the shared rotated 'response' label that sits between them
        # (deviation 25): the label is ~28 pt long at PT_BASE and the two
        # inset centres are only ~29 pt apart, so a terminus on either centre
        # sits inside it.
        f.leader(anchors[i],
                 (col_x - f.fx(2.0),
                  y0 + f.fy(h_pt * (0.85 if i == 0 else 0.15))))
    f.text((col_x - f.fx(4.5), (bots[0] + bots[1]) / 2.0 + f.fy(h_pt / 2.0)),
           'response', size=PT_BASE, color=GRAY, ha='center', va='center',
           rotation=90)
    f.text((col_x + w / 2.0, bots[1] - f.fy(1.5)), 'condition',
           size=PT_BASE, color=GRAY, ha='center', va='top')

    f.require_soma_lowest()
    f.require_delta0(allow_no_delta0=True, reason=DELTA0_REASON)
    return nodes


# ── panel E: the realized route dictionary ───────────────────────────────
def panel_routes(f, matrix, groups, n_routes, subtitle_lines):
    """Nine mapped inputs placed by ancestry beside the realized 9 x 4
    support; the delivered field is imposed, not observed."""
    n = matrix.shape[0]
    # Review pass 2026-09-23: title, count line, footer and `local rule`
    # badge moved to legend E; arbor and matrix take the freed slot.
    band = 0.0
    sub_pt = 10.5 * len(subtitle_lines)
    for i, line in enumerate(subtitle_lines):
        f.text((0.5, 1.0 - f.fy(10.5 * i + 5.2)), line, size=PT_BASE,
               color=GRAY, ha='center', va='center')
    core_y0 = f.fy(band)
    core_h = 1.0 - f.fy(band + sub_pt)

    # E's arbor keeps its width: the 'local rule' badge is anchored at
    # ``tree_w_pt + 4`` and a wider arbor walks it into the matrix row
    # labels (1.2 pt clear at 72 pt against 7.4 pt here).  E's freed height
    # goes to the realized-support matrix instead.
    tree_w_pt = min(86.0, 0.54 * f.w_pt)
    nodes = f.balanced_tree((0.0, core_y0 + f.fy(13.0), f.fx(tree_w_pt),
                             core_h - f.fy(13.0)),
                            depth=3, mode='forward', labels=True,
                            output='ŷ')
    shared, singles = groups
    seats = {shared[0]: ('T1', 0.0), shared[1]: ('T2', 0.0),
             shared[2]: ('T2', 5.2)}
    for i, label in enumerate(singles):
        seats[label] = (f'T{i + 3}', 0.0)
    reached = [i + 1 for i in range(n) if matrix[i].any()]
    placed = {}
    for label, (seat, drop) in seats.items():
        par = nodes.parent[seat]
        placed[label] = (nodes[seat] if drop == 0.0
                         else _toward(f, nodes[seat], nodes[par], drop))
    before = len(f.ax.patches)
    f.credit_delivery(nodes, mode='subtree',
                      targets=[seats[i][0] for i in reached],
                      rule_color='shunting', alpha_tags=False)
    _retint_capsules(f, f.ax.patches[before:])
    _seat_arrows(f, f.ax.patches[before:], list(placed.values()))
    for label in seats:
        f.contact(placed[label], kind='exc')
    f.error_in(nodes.soma, side='right', dashed=True)

    mat_w = 44.0
    mat_h = min(86.0, core_h * f.h_pt - 20.0)
    mat_x = 1.0 - f.fx(mat_w)
    mat_y = core_y0 + f.fy(13.0)
    f.dictionary_matrix((mat_x, mat_y, f.fx(mat_w), f.fy(mat_h)), matrix,
                        measured=True, yticks=list(range(1, n + 1)),
                        col_labels=[f'r{i + 1}' for i in range(n_routes)],
                        label='A', min_cell_pt=6.0)
    _delta_hat(f, (f.fx(6.0), core_y0 + f.fy(4.5)), tail=' = A c')

    f.require_soma_lowest()
    f.require_delta0()
    return nodes


def _dot_column(values, min_sep, step):
    """Level index per value so a rug reads as a countable dot column.

    Panel B's rug carries the thirteen per-scan values, whose adjacent gaps
    fall to 0.0028 -- 0.5 pt on this axis against a ``SEED_MS`` 2.9 pt ring,
    so the flat rug printed about eleven countable marks where the panel's
    own tag says thirteen (visual review 2026-09-10).  Values closer than
    ``min_sep`` are stacked on successive levels ``step`` apart instead.
    The VALUE is never moved: only the rug's meaningless y is assigned.
    """
    values = np.asarray(values, dtype=float)
    level = np.zeros(len(values), dtype=float)
    taken = []
    for i in np.argsort(values, kind='stable'):
        k = 0
        while any(abs(values[i] - vx) < min_sep and kx == k for vx, kx in taken):
            k += 1
        taken.append((float(values[i]), k))
        level[i] = k * step
    return level


def _split_duplicates(support, x, offset_pt=DUP_OFFSET):
    """Separate scans that share an exact ``(n_sites, coverage)`` pair.

    Two of the thirteen scans are exactly coincident in panel F --
    ``(10, 0.40)`` once with and once without four single-input routes, and
    ``(9, 0.4444)`` twice -- so an unjittered scatter drew eleven countable
    markers where the panel, its tag and the caption all say thirteen (QA
    2026-09-09).  Members of a duplicate group are spread symmetrically on
    the x axis in steps of ``offset_pt`` mapped inputs (+/-0.13 for a pair),
    ordered by ``(target_root_id, session, scan_idx)`` so the displacement is
    deterministic and reproducible.  y is never moved.
    """
    x = np.asarray(x, dtype=float).copy()
    order = np.lexsort((support.scan_idx.to_numpy(),
                        support.session.to_numpy(),
                        support.target_root_id.to_numpy()))
    rank = np.empty(len(support), dtype=int)
    rank[order] = np.arange(len(support))
    keys = list(zip(support.n_sites.to_numpy(),
                    np.round(support.coverage.to_numpy(float), 9)))
    for key in sorted(set(keys)):
        idx = sorted((i for i, k in enumerate(keys) if k == key),
                     key=lambda i: rank[i])
        if len(idx) < 2:
            continue
        for j, i in enumerate(idx):
            x[i] += (j - (len(idx) - 1) / 2.0) * offset_pt
    return x


def _retint_capsules(f, patches, cycle=CAPSULE_CYCLE, pct=CAPSULE_PCT):
    """Re-tint the four route capsules ``credit_delivery`` just drew.

    ``Frame.credit_delivery(mode='subtree')`` colours its capsules from
    ``journal_style.K_CYCLE`` = (shunting, additive, local, oracle).  Two of
    those four hues -- ``additive`` #20509E and the ``local``/``scalar`` amber
    #E2A23F -- are barred from this figure by PLAN section 4 (B14) and by the
    plan's own acceptance check 6, which admits only ``shunting``, ``oracle``
    and the anatomy register as saturated hues here.  SPEC_ERRATA #7 forbids
    editing the library, so this follows the AMENDMENTS item-27 precedent set
    for Fig 7B: the patches the library just added are re-tinted in place with
    a builder-local address cycle drawn from the admissible hues, alternating
    in target order so the two sibling capsules (T7 and T8, whose entry stubs
    overlap at their shared junction) never carry the same tint.  Geometry,
    count, tint strength (16 %) and the 0.55 pt edge are untouched; the
    delivery arrows (``FancyArrowPatch``) are left alone.
    """
    caps = [q for q in patches if getattr(q, '_posA_posB', None) is None]
    for i, patch in enumerate(caps):
        face = tint_pct(COLORS[cycle[i % len(cycle)]], pct)
        patch.set_facecolor(face)
        patch.set_edgecolor(strengthen(face, 2.4))
    return [cycle[i % len(cycle)] for i in range(len(caps))]


def _seat_arrows(f, patches, contacts, clearance_pt=3.4):
    """Pull each subtree-delivery arrow tip clear of the contact it addresses.

    ``Frame.credit_delivery(mode='subtree')`` lands its arrow at 80 % of the
    parent->terminal segment.  In this 5-module card the terminal pitch is
    small enough that the head crossed the ``exc`` disc it points at and the
    contact stopped being countable at print scale (QA 2026-09-09).  The
    library is not edited: the arrows it just added are re-seated here so the
    tip stops ``clearance_pt`` points short of the nearest contact centre,
    along the same parent direction.
    """
    C = [f._to_pt(c) for c in contacts]
    for patch in patches:
        pos = getattr(patch, '_posA_posB', None)
        if pos is None:
            continue
        A, B = f._to_pt(pos[0]), f._to_pt(pos[1])
        u = B - A
        L = float(np.linalg.norm(u))
        if L < 1e-9:
            continue
        u = u / L
        d = min(float(np.linalg.norm(B - c)) for c in C)
        pull = max(0.0, clearance_pt - d)
        pull = min(pull, max(0.0, L - 3.0))
        if pull <= 0.0:
            continue
        patch.set_positions(f._from_pt(A), f._from_pt(B - u * pull))


def _annotation_leader(f, a, b, *, start_pt=2.6, stop_pt=5.5, color=GRAY):
    """A DASHED annotation leader from ``a`` toward ``b`` (panel A).

    ``Frame.leader`` draws a SOLID ``mute`` hairline with no dash option.
    Panel A's two pair-2 connectors are annotation, not anatomy: drawn solid
    they cut straight across the grey arbor and converged on the yellow soma
    at the same point as the real trunk, so at a glance they read as two
    extra branches rather than as the link between a contact pair and the
    soma it does not share a path above (visual review 2026-09-10).  They are
    drawn here instead -- same hue and hairline weight, dashed, starting
    clear of the contact disc and stopping ``stop_pt`` short of the soma
    centre so no annotation touches the soma marker.  The library is NOT
    edited (SPEC_ERRATA #7).
    """
    A, B = f._to_pt(a), f._to_pt(b)
    u = B - A
    L = float(np.linalg.norm(u))
    if L < start_pt + stop_pt + 2.0:
        return
    u = u / L
    p0, p1 = f._from_pt(A + u * start_pt), f._from_pt(B - u * stop_pt)
    f.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=f.lw(LW_HAIR),
              dashes=(2.0, 1.7), zorder=1.5, solid_capstyle='butt')


def _toward(f, a, b, pt):
    """Point ``pt`` points from ``a`` toward ``b``."""
    A, B = f._to_pt(a), f._to_pt(b)
    d = B - A
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    return f._from_pt(A + d * pt)


# ── build ────────────────────────────────────────────────────────────────
TOPOLOGY_MEASURES = (
    ('shared_path_partial_r', 'Partial\nrank r', 'shunting'),
    ('shared_path_spearman_r', 'Shared\npath r', 'mute'),
    ('negative_tree_distance_spearman_r', 'Tree\ndistance r', 'mute'),
    ('same_major_branch_delta', 'Same major\nbranch Δ', 'mute'),
)


def current_plotted_table(effect_summary, effects, scan_metrics, all_scans,
                          cell_metrics, power, reliability, support, matrix,
                          metadata):
    """Export the current B--F panels, not the superseded A--C layout.

    Summary estimates and the individual targets/scans remain separate record
    types.  D-inset includes every finite reliability estimate, including negatives.
    """
    from source_data_export import exact_id_table

    rows = []
    selected = effect_summary[
        effect_summary.endpoint.eq('structure_function_partial_r')
        & effect_summary.comparison.eq('selected_scans')].iloc[0]
    rows.append(dict(panel='B', record='target_mean', **selected.to_dict()))
    target_values = effects[effects.endpoint.eq('structure_function_partial_r')
                            & effects.comparison.eq('selected_scans')]
    rows.extend(dict(panel='B', record='target_value', **r)
                for r in target_values.to_dict('records'))
    for r in scan_metrics.to_dict('records'):
        rows.append(dict(panel='B', record='scan_value',
                         target_root_id=r['target_root_id'],
                         session=r['session'], scan_idx=r['scan_idx'],
                         endpoint='shared_path_partial_r',
                         effect=r['shared_path_partial_r']))
    for column, _label, _colour in TOPOLOGY_MEASURES:
        summary = all_scans['metrics'][column]
        rows.append(dict(panel='C', record='target_mean', endpoint=column,
                         comparison='scan_complete', n_targets=len(cell_metrics),
                         mean=summary['mean'],
                         ci95_low=summary['target_bootstrap_ci95'][0],
                         ci95_high=summary['target_bootstrap_ci95'][1],
                         positive_targets=summary['positive_targets']))
        rows.extend(dict(panel='C', record='target_value', endpoint=column,
                         comparison='scan_complete',
                         target_root_id=r['target_root_id'], effect=r[column])
                    for r in cell_metrics.to_dict('records'))
    rows.extend(dict(panel='D', record='detection_probability', **r)
                for r in power.to_dict('records'))
    rows.append(dict(panel='D', record='observed_estimate', **selected.to_dict()))
    rows.extend(dict(panel='D-inset', record='repeat_reliability',
                     drawn=bool(np.isfinite(r['measured_split_half_spearman'])), **r)
                for r in reliability.to_dict('records'))
    for i, site in enumerate(metadata['site_segment_ids']):
        for j, route in enumerate(metadata['selected_route_segments']):
            rows.append(dict(panel='E', record='support_entry',
                             target_root_id=metadata['target_root_id'],
                             session=metadata['session'], scan_idx=metadata['scan_idx'],
                             input_row=i + 1, route_column=j + 1,
                             site_segment_id=int(site), route_segment_id=int(route),
                             value=float(matrix[i, j])))
    rows.extend(dict(panel='F', record='scan_support', **r)
                for r in support.to_dict('records'))
    return exact_id_table(rows)


def label_scan_rug(ax, rug_y):
    """Distinguish descriptive scan values from target-level inference."""
    return ax.annotate('All scans\n(descriptive)',
                       xy=(0.0, float(np.mean([min(rug_y), max(rug_y)]))),
                       xycoords=('axes fraction', 'data'),
                       xytext=(-4.0, 0.0), textcoords='offset points',
                       fontsize=PT_BASE, color=GRAY, ha='right', va='center',
                       annotation_clip=False, zorder=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-main', action='store_true',
                        help='also write figures/main/figure_10.pdf')
    parser.add_argument('--no-emit-main', action='store_true')
    args, _ = parser.parse_known_args()
    emit_main = not args.no_emit_main
    REC.mkdir(exist_ok=True)

    # -- realized route supports, all 13 scans (panels E, F) --------------
    seg = pd.read_csv(S / 'figure3/segment_metrics.csv')
    metadata = [json.loads(x) for x in
                (S / 'fulltree_boundary/output/dictionary_and_validation_metadata.jsonl'
                 ).read_text().splitlines()]
    support, matrices = [], {}
    for r in metadata:
        if r['replicate'] != 0:
            continue
        _, parents, _ = parent_map(seg[seg.root_id.eq(r['target_root_id'])])
        a = ancestry_matrix(r['site_segment_ids'], r['selected_route_segments'],
                            parents)
        assert np.count_nonzero(a) == r['dictionary_nonzeros']['topology-matched routes']
        key = (r['target_root_id'], r['session'], r['scan_idx'])
        matrices[key] = (a, r)
        support.append(dict(
            target_root_id=key[0], session=key[1], scan_idx=key[2],
            n_sites=len(a), n_routes=a.shape[1], coverage=a.any(1).mean(),
            sites_per_route=a.sum() / a.shape[1],
            one_site_routes=bool((a.sum(0) == 1).all())))
    support = pd.DataFrame(support).sort_values(
        ['n_sites', 'target_root_id', 'session', 'scan_idx']).reset_index(drop=True)
    assert len(support) == 13 and support.target_root_id.nunique() == 7
    rep = support.iloc[len(support) // 2]
    key = tuple(int(rep[k]) for k in ['target_root_id', 'session', 'scan_idx'])
    a, meta = matrices[key]
    assert key[0] == REP_TARGET, key

    _, parents, _ = parent_map(seg[seg.root_id.eq(key[0])])
    sites = list(meta['site_segment_ids'])

    def route_up(node):
        out = [node]
        while parents.get(node) not in (None, node):
            node = parents[node]
            out.append(node)
        return out

    routes_up = {i + 1: route_up(s) for i, s in enumerate(sites)}
    depth1 = {}
    for label, chain in routes_up.items():
        depth1.setdefault(chain[-2], []).append(label)
    shared = sorted(max(depth1.values(), key=len))
    singles = sorted(l for l in routes_up if l not in shared)
    assert len(shared) == 3 and len(singles) == 6, (shared, singles)

    # -- cohort accounting (panel A sub-title) ----------------------------
    effect_summary = pd.read_csv(S / 'review_evidence_reanalysis/'
                                      'functional_native_contrasts.csv')
    effects = pd.read_csv(S / 'review_evidence_reanalysis/'
                              'functional_native_target_effects.csv')
    scan_metrics = pd.read_csv(S / 'functional_topology_all_scans/scan_metrics.csv')
    all_scans = json.loads((S / 'functional_topology_all_scans/summary.json'
                            ).read_text())
    cell_metrics = pd.read_csv(S / 'functional_topology_all_scans/cell_metrics.csv')
    selected = pd.read_csv(S / 'figure5/functional_target_metrics.csv')
    n_targets = int(all_scans['n_target_cells'])
    n_scans = int(all_scans['n_scans'])
    n_partners = int(selected.n_partners.sum())
    p_lo = int(scan_metrics.n_presynaptic_partners.min())
    p_hi = int(scan_metrics.n_presynaptic_partners.max())
    subtitle_a = [f'Selected: {n_targets} targets, {n_partners} partners',
                  f'All: {n_scans} scans, 102 partners']

    # -- B: the two cohorts ------------------------------------------------
    pa = effect_summary[effect_summary.endpoint.eq(
        'structure_function_partial_r')].set_index('comparison')
    b_rows = []
    for mode, label in (('selected_scans', 'Selected scans'),
                        ('scan_complete', f'All {n_scans} scans')):
        r = pa.loc[mode]
        v = effects[effects.endpoint.eq('structure_function_partial_r')
                    & effects.comparison.eq(mode)].effect.to_numpy()
        assert len(v) == 7, mode
        b_rows.append(dict(label=label, mean=float(r['mean']),
                           lo=float(r.ci95_low), hi=float(r.ci95_high),
                           seeds=list(map(float, v)), color='shunting',
                           marker='D', n=int(r.n_targets),
                           note=f'{int(r.positive_targets)}/{len(v)} +'))
    scan_values = scan_metrics.shared_path_partial_r.to_numpy()
    assert len(scan_values) == n_scans

    # -- C: four topology measures ----------------------------------------
    c_rows = []
    for column, label, colour in TOPOLOGY_MEASURES:
        m = all_scans['metrics'][column]
        v = cell_metrics[column].to_numpy()
        assert len(v) == 7, column
        c_rows.append(dict(label=label, mean=float(m['mean']),
                           lo=float(m['target_bootstrap_ci95'][0]),
                           hi=float(m['target_bootstrap_ci95'][1]),
                           seeds=list(map(float, v)), color=colour,
                           marker='D', n=len(v),
                           note=f"{int(m['positive_targets'])}/{len(v)} +"))

    # -- D: detection curves ----------------------------------------------
    power = pd.read_csv(S / 'measured_alignment_power/power_summary.csv')
    curves = {rel: power[power.reliability.eq(rel)].sort_values('mean_target_effect')
              for rel in ('measured', 'perfect')}
    thresholds = json.loads((S / 'measured_alignment_power/RESULTS.json').read_text())
    cut_m = float(thresholds['thresholds']['measured']
                  ['descriptive_mean_partial_rank_at_interpolation'])
    cut_p = float(thresholds['thresholds']['perfect']
                  ['descriptive_mean_partial_rank_at_interpolation'])
    lam = float(thresholds['thresholds']['measured']
                ['smallest_lambda_lower_mc_bound_ge_80'])
    floor = float(thresholds['null_calibration']['measured']
                  ['positive_direction_false_detection'])
    mc_hw = float(max(((d.power_ci95_high - d.power_ci95_low) / 2.0).max()
                      for d in curves.values()))
    obs = pa.loc['selected_scans']
    reliability = pd.read_csv(S / 'measured_alignment_power/'
                                 'reliability_calibration_audit.csv')
    reliab = reliability.measured_split_half_spearman.to_numpy()
    partner_index = pd.read_csv(S / 'measured_alignment_power/inputs/partner_index.csv')
    n_records = int(len(reliab))
    n_partners_unique = int(partner_index.pre_pt_root_id.nunique())
    assert n_records == len(partner_index) == 125
    hist_edges = np.linspace(-0.2, 0.6, 9)
    hist_counts, _ = np.histogram(reliab, bins=hist_edges)
    assert int(hist_counts.sum()) == n_records
    # Negative reliabilities are displayed and retained in the audit.
    n_negative = int(hist_counts[hist_edges[1:] <= 0.0].sum())
    keep = hist_edges[1:] > 0.0
    assert n_negative == 2 and int(hist_counts[keep].sum()) == n_records - 2

    # -- F: coverage across the thirteen scans ----------------------------
    n_in = support.n_sites.to_numpy(float)
    cov = support.coverage.to_numpy(float) * 100.0
    single = support.one_site_routes.to_numpy(bool)
    n_in = _split_duplicates(support, n_in)
    k_routes = int(support.n_routes.max())
    mean_cov = float(support.coverage.mean()) * 100.0
    below = support[support.coverage < support.n_routes / support.n_sites - 1e-9]
    above = support[support.coverage > support.n_routes / support.n_sites + 1e-9]

    # -- canvas -----------------------------------------------------------
    # Design pass 2026-09-14: 22 + 112 + 30 + 108 + 30 + 112 + 34 = 448 pt at
    # 30 pt gutters (the lock pass carves ~10 pt off the top of each data row
    # for the x labels above it and the letters).  The data panels carry no
    # titles, and every line of caption prose they held -- B's rug and sign
    # notes, C's cohort line and sign key, D's lambda note, F's three key
    # lines -- is gone; the caption states each of them.
    canvas = NativeCanvas(490 / 72, 3, row_weights=[132, 130, 112],
                          hgutter_pt=30, vgutter_pt=30,
                          margins=Margins(left=40, right=14, top=22, bottom=34))
    # 2026-09-23: no titles, and the two schematics are unlocked: locked,
    # they inherited C's 54 pt row-label reserve and sat 54 pt right of
    # their letters in a 117 pt box
    ax_a = canvas.panel('A', 0, 0, 5, schematic=True, lock=False,
                        inset_pt=(0.0, 0.0, 14.0, 0.0))   # = B's top reserve
    ax_b = canvas.panel('B', 0, 5, 7)
    ax_c = canvas.panel('C', 1, 0, 5)
    ax_d = canvas.panel('D', 1, 5, 7)
    ax_e = canvas.panel('E', 2, 0, 5, schematic=True, lock=False,
                        inset_pt=(0.0, 0.0, 9.0, 2.0))    # = F's reserves
    ax_f = canvas.panel('F', 2, 5, 7)
    # row 2 carries E's title above its axes as well as the letters, and
    # row 1's x labels hang 20 pt into the gutter: a 14 pt declared top
    # reserve keeps the two rows 3 mm apart (audit_row_separation)
    # 2026-09-23: A and E no longer carry titles, but the 14 pt top reserve
    # stays on B and F: it keeps the data panels' slot fill within the
    # 1.35x panel-emphasis band (B reached 1.01 of its slot without it).
    canvas.declare_reserve('F', top=9.0)     # 14 -> 9: no blank band over row 2
    canvas.declare_reserve('B', top=14.0)

    # -- B ----------------------------------------------------------------
    # Only the SELECTED-SCAN cohort is drawn here.  The all-thirteen-scan
    # partial rank correlation was drawn twice in this one figure -- as B's
    # second forest row and again as C's green top row, the same estimate on
    # two axes 1.62x apart in width (visual review 2026-09-10) -- so B keeps
    # the selected-scan estimate and the thirteen per-scan values, and C's
    # green row is now the figure's single drawing of the all-scan estimate.
    # ``b_rows`` still carries both cohorts: the second is recorded in the
    # provenance payload, it is simply no longer redrawn.
    b_plot = b_rows[:1]
    out_b = canvas.forest(
        ax_b, [dict(r, note=None) for r in b_plot],
        value_label='Ancestry–response partial rank correlation',
        xlim=(-0.5, 0.5), reference=None, reference_label='', tag='')
    ax_b.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    ax_b.set_xticklabels(['−0.5', '−0.25', '0', '0.25', '0.5'])
    # the rug is a dot column, not a flat row: four of the thirteen adjacent
    # gaps are under one ring diameter on this axis and the flat rug printed
    # about eleven countable marks against its own '13 scan values' tag.
    y_rug = out_b['ypos'][-1] + 0.62
    rug_y = y_rug + _dot_column(scan_values, 0.024, 0.155)
    ax_b.plot(scan_values, rug_y, linestyle='none',
              marker='o', markersize=SEED_MS, markerfacecolor='white',
              markeredgecolor=GRAY, markeredgewidth=LW_HAIR, zorder=3.5)
    label_scan_rug(ax_b, rug_y)
    rug_bot = float(rug_y.max())
    ax_b.set_ylim(rug_bot + 0.35, -0.70)
    # CF-7 zero rule, CLIPPED to the row band plus the scan rug.  forest()'s
    # own reference is a full-height axvline; with the y limits padded to
    # carry the annotation block it ran down through the annotation text and
    # read as a strike-through (QA 2026-09-09), so the rule is drawn here
    # instead and stops above the annotation block.  The rug rings are filled
    # white so the rule cannot print through a mark.
    ax_b.plot([0.0, 0.0], [-0.45, rug_bot + 0.17], color=GRAY, lw=LW_REF,
              dashes=(2.6, 2.0), zorder=1.0, solid_capstyle='butt')
    # (2026-09-23: the zero rule is unlabelled; its meaning follows from the axis)
    # The concise rug label distinguishes its thirteen scan values from the
    # seven-target estimate above; it does not introduce a second inference.

    # C separates correlations from the difference in mean similarity.
    ax_c.set_axis_off()
    upper = ax_c.inset_axes([0, .49, 1, .51])
    lower = ax_c.inset_axes([0, -.02, 1, .18])
    for axis, rows, label in ((upper, [dict(row, label=row['label'].replace('\n', ' ')) for row in c_rows[:3]], 'Rank correlation'),
                              (lower, c_rows[3:], 'Mean similarity difference')):
        canvas.forest(axis, [dict(row, note=None) for row in rows], value_label=label, xlim=(-.5, .5),
                      reference=0., reference_label='', tag='', band=False)
        axis.set_xticks([-.5, 0, .5], ['−0.5', '0', '0.5'])
        axis.tick_params(labelsize=PT_BASE, pad=1, length=2)
        axis.xaxis.labelpad = 1
    canvas.declare_reserve('C', left=54, bottom=22)

    # -- D ----------------------------------------------------------------
    for rel, colour, name in (('perfect', CEIL, 'perfect reliability'),
                              ('measured', ROUTE, 'measured reliability')):
        d = curves[rel]
        ax_d.fill_between(d.mean_target_effect, d.power_ci95_low,
                          d.power_ci95_high, color=tint_pct(colour, 16),
                          lw=0, zorder=2 if rel == 'perfect' else 2.4)
        ax_d.plot(d.mean_target_effect, d.power, color=colour, lw=LW_DATA,
                  solid_capstyle='round', zorder=3 if rel == 'perfect' else 3.4)
    ax_d.axhline(0.80, color=GRAY, lw=LW_REF, dashes=(2.6, 2.0), zorder=1)
    # the false-positive floor is drawn only left of the inset, so the inset's
    # own tick labels never sit on a reference rule (CF-7 keeps the label
    # right-aligned ON the line)
    ax_d.plot([-0.30, 0.205], [floor, floor], color=GRAY, lw=LW_REF,
              dashes=(2.6, 2.0), zorder=1)
    # bounded, not an axvline: the padded y limits carry the lambda annotation
    ax_d.plot([0.0, 0.0], [-0.32, 0.80], color=GRAY, lw=LW_HAIR,
              dashes=(2.6, 2.0), zorder=1, solid_capstyle='butt')
    # one drop from the 80 % crossing (deviation 6), run down to the top of
    # the inset -- as far as the corridor at this x is free (deviation 24)
    ax_d.plot([cut_m, cut_m], [0.56, 0.80], color=GRAY, lw=LW_HAIR,
              dashes=(2.6, 2.0), zorder=1.5)
    ax_d.set(xlim=(-0.30, 0.55), ylim=(-0.32, 1.12))
    ax_d.set_xticks([-0.25, 0.0, 0.25, 0.50])
    ax_d.set_xticklabels(['−0.25', '0', '0.25', '0.50'])
    ax_d.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax_d.set_xlabel('Mean simulated partial rank correlation', fontsize=PT_EMPH)
    ax_d.set_ylabel('Detection probability', fontsize=PT_EMPH)
    style_panel(ax_d, grid='none')
    ax_d.spines['left'].set_bounds(0.0, 1.0)
    ax_d.spines['bottom'].set_bounds(-0.30, 0.55)
    ax_d.text(0.545, 0.815, '80 % detection', fontsize=PT_BASE, color=GRAY,
              ha='right', va='bottom', zorder=6)
    # The false-detection label is set INSIDE the axes on the same left edge
    # as the lambda note.  Right-aligned at x = -0.020 it overhung the left
    # spine by ~14 pt and the spine printed straight through the word
    # (visual review 2026-09-10); a single line long enough to carry the
    # phrase would instead cross the x = 0 rule, so it breaks after the noun.
    # one line: the caption gives the two false-detection rates, so the rule
    # is named and not numbered on the panel (design pass 2026-09-14)
    # Review pass 2026-09-23: `false detection`, `no alignment` and the two
    # 80 % crossing values are legend D's; the values are held here.
    assert f'{cut_m:.3f}' == '0.249' and f'{cut_p:.3f}' == '0.248', (cut_m, cut_p)
    # (design pass 2026-09-14: the two-line `ancestry variance lambda = 0.65
    # / respecting the MC band` note is gone -- the running text carries the
    # sentence -- and the value is held to the source here instead)
    assert abs(lam - 0.65) < 5e-3, lam
    ax_d.text(0.545, 1.07, 'perfect reliability', fontsize=PT_BASE,
              color=label_color(CEIL), ha='right', va='center', zorder=6)
    ax_d.plot([0.42, 0.42], [1.035, 1.00], color=GRAY, lw=LW_HAIR, zorder=2)
    ax_d.text(0.545, 0.72, 'measured reliability', fontsize=PT_BASE,
              color=label_color(ROUTE), ha='right', va='center', zorder=6)
    ax_d.plot([0.288, 0.223], [0.712, 0.742], color=GRAY, lw=LW_HAIR, zorder=2)
    # observed rug -- the estimate located on this axis, not a power estimate
    y_obs = -0.14
    ax_d.plot([-0.30, 0.55], [y_obs, y_obs], color=GRAY, lw=LW_HAIR, zorder=1)
    ax_d.plot([float(obs.ci95_low), float(obs.ci95_high)], [y_obs, y_obs],
              color=ROUTE, lw=LW_ERR, solid_capstyle='butt', zorder=4)
    ax_d.plot([float(obs['mean'])], [y_obs], marker='D', markersize=MARKER_MS,
              markerfacecolor='white', markeredgecolor=ROUTE,
              markeredgewidth=LW_ERR, linestyle='none', zorder=5)
    ax_d.text(0.135, y_obs - 0.10, 'observed',
              fontsize=PT_BASE, color=label_color(ROUTE), ha='left',
              va='center', zorder=6)
    # inset: measured split-half reliability, the calibration of the curves
    # the inset title starts 8 pt right of the 0.249 drop (deviation 24)
    # 0.29, not 0.255: on the shorter 2026-09-14 row the inset's rotated
    # `records` label reached the measured-reliability curve
    inset = _data_inset(ax_d, (0.290, 0.17, 0.255, 0.34))
    for lo, hi, count in zip(hist_edges[:-1], hist_edges[1:], hist_counts):
        inset.bar(lo, count, width=hi - lo, align='edge',
                  facecolor=tint_pct(ROUTE, 40), edgecolor=ROUTE,
                  linewidth=LW_HAIR, zorder=3)
    inset.set(xlim=(-0.2, 0.6), ylim=(0, 46))
    inset.set_xticks([-0.2, 0.0, 0.3, 0.6])
    inset.set_xticklabels(['−0.2', '0', '0.3', '0.6'])
    inset.set_yticks([0, 20, 40])
    inset.set_xlabel('split-half r', fontsize=PT_BASE, labelpad=1.0)
    inset.set_ylabel('records', fontsize=PT_BASE, labelpad=1.0)

    # -- F ----------------------------------------------------------------
    grid_n = np.linspace(4.0, 18.6, 200)
    ax_f.plot(grid_n, 100.0 * k_routes / grid_n, color=GRAY, lw=LW_REF,
              dashes=(2.6, 2.0), zorder=2)
    ax_f.text(6.1, 100.0 * k_routes / 6.1 + 4.0, f'{k_routes}/n',
              fontsize=PT_BASE, color=GRAY, ha='left', va='bottom', zorder=6)
    # the two black references meant different things in one dash pattern:
    # 'all inputs' is dotted, the 4/n curve stays dashed.  The mean rule is
    # grey, not green -- drawn in the data hue at the data's own weight it
    # ran through five of the thirteen markers and they read as ornaments on
    # the reference rather than as measurements (visual review 2026-09-10).
    ax_f.axhline(100.0, color=GRAY, lw=LW_REF, dashes=(0.9, 1.7), zorder=1)
    ax_f.axhline(mean_cov, color=GRAY, lw=LW_REF, dashes=(5.0, 1.6, 1.2, 1.6),
                 zorder=1.5)
    assert f'{mean_cov:.1f}' == '48.0', mean_cov      # value: legend F
    ax_f.text(18.5, mean_cov + 2.5, 'mean', fontsize=PT_BASE,
              color=GRAY, ha='right', va='bottom', zorder=6)
    # open first, filled last: nothing of the thirteen may be covered
    ax_f.plot(n_in[single], cov[single], linestyle='none', marker='o',
              markersize=MARKER_MS, markerfacecolor='white',
              markeredgecolor=ROUTE, markeredgewidth=LW_ERR, zorder=4)
    ax_f.plot(n_in[~single], cov[~single], linestyle='none', marker='o',
              markersize=MARKER_MS, markerfacecolor=ROUTE,
              markeredgecolor=ROUTE, markeredgewidth=0, zorder=5)
    ax_f.set(xlim=(4.0, 18.6), ylim=(-6.0, 112.0))
    ax_f.set_xticks([5, 8, 11, 14, 17])
    ax_f.set_yticks([0, 25, 50, 75, 100])
    ax_f.set_xlabel('Mapped inputs in the scan', fontsize=PT_EMPH)
    ax_f.set_ylabel('Mapped inputs reached (%)', fontsize=PT_EMPH)
    style_panel(ax_f, grid='y')
    ax_f.spines['left'].set_bounds(0.0, 100.0)
    # (design pass 2026-09-14: the `open: one input per route (6 of 13
    # scans)`, `dashed: 4/n, four distinct inputs` and `coincident scans
    # offset +-0.2 on x` lines are gone -- caption F carries all three -- and
    # the counts are held to the source here instead)
    assert int(single.sum()) == 6 and n_scans == 13 and k_routes == 4
    # 4/n is NOT a ceiling: three of the thirteen scans plot above it because
    # a route can reach more than one input (``sites_per_route`` runs to 2.0),
    # and the old note explained only the downward direction (visual review
    # 2026-09-10).  The curve is relabelled for what it actually is -- the
    # level a scan reaches when the four routes cover four DISTINCT inputs --
    # so a mark above it reads as a route that reached more than one input and
    # a mark below it as routes that repeated one.  The six open scans, whose
    # routes each reach a single input, are exactly the ones 4/n bounds; the
    # caption names both directions.
    # Two pairs of scans are exactly coincident in (inputs, coverage) and are
    # separated on x by ``_split_duplicates``; neither the panel nor the
    # caption said so, and an undisclosed offset on an integer count reads as
    # a fractional input (visual review 2026-09-10).  The claim (y) is never
    # moved, so the disclosure names the x offset only.

    # Tighten the tick / label pads on every data axes: the 40 pt vertical
    # gutter has to hold one row's x labels and the next row's letter band,
    # and the default pads leave the row-separation audit under its 8.5 pt
    # floor.  The type size is untouched (CF-2).
    for ax in (ax_b, ax_c, ax_d, ax_f):
        ax.tick_params(axis='both', pad=1.6)
        ax.xaxis.labelpad = 1.2
        ax.yaxis.labelpad = 1.2

    # -- lock the grid, then draw the schematics at their final sizes -----
    style_direct_color_labels(canvas.fig)
    canvas.lock_reserves()

    panel_statistic(Frame(ax_a), [])
    panel_routes(Frame(ax_e), a, (shared, singles), int(a.shape[1]), [])

    problems = canvas.save(OUT, name='credit_first_figure_08', dpi=180)
    if emit_main:
        MAIN.parent.mkdir(parents=True, exist_ok=True)
        MAIN.write_bytes(OUT.read_bytes())

    # -- render-time Source Data and provenance ---------------------------
    support.to_csv(REC / 'figure_08_support.csv', index=False)
    np.savez_compressed(REC / 'figure_08_actual_support.npz', matrix=a,
                        site_ids=np.array(meta['site_segment_ids']),
                        route_ids=np.array(meta['selected_route_segments']))
    reliability_rows = reliability.copy()
    reliability_rows = reliability_rows.merge(
        partner_index[['scan', 'partner_index', 'pre_pt_root_id',
                       'repeat_reliability']],
        on=['scan', 'partner_index'], how='left')
    assert len(reliability_rows) == n_records
    assert reliability_rows.pre_pt_root_id.nunique() == n_partners_unique
    reliability_rows.to_csv(REC / 'figure_09_reliability_source.csv',
                            index=False)
    plotted = current_plotted_table(
        effect_summary, effects, scan_metrics, all_scans, cell_metrics, power,
        reliability_rows, support, a, meta)
    plotted.to_csv(S / 'curated_publication/figure_10_plotted.csv', index=False)
    plotted.to_csv(REC / 'figure_10_plotted.csv', index=False)

    files = [Path(__file__),
             S / 'figure3/segment_metrics.csv',
             S / 'fulltree_boundary/output/dictionary_and_validation_metadata.jsonl',
             S / 'curated_publication/figure_10_plotted.csv',
             S / 'review_evidence_reanalysis/functional_native_contrasts.csv',
             S / 'review_evidence_reanalysis/functional_native_target_effects.csv',
             S / 'functional_topology_all_scans/scan_metrics.csv',
             S / 'functional_topology_all_scans/cell_metrics.csv',
             S / 'functional_topology_all_scans/summary.json',
             S / 'figure5/functional_target_metrics.csv',
             S / 'measured_alignment_power/power_summary.csv',
             S / 'measured_alignment_power/reliability_calibration_audit.csv',
             S / 'measured_alignment_power/RESULTS.json',
             S / 'measured_alignment_power/inputs/partner_index.csv',
             Path(__file__).with_name('source_data_export.py')]
    payload = dict(
        figure='fig9', label='fig:boundary', panels='a-f',
        plotted_panel_mapping={
            'B': 'selected-target estimate and values; descriptive all-scan values',
            'C': 'four all-scan target-level topology measures',
            'D': 'detection probability and the observed-estimate rug',
            'D-inset': 'all repeat reliabilities, with histogram inclusion flagged',
            'E': 'representative mapped-input by route support matrix',
            'F': 'mapped-input coverage in every eligible scan'},
        canvas=dict(width_pt=518.4, height_pt=448.0,
                    schematic_fraction=round(
                        sum(w * h for _, _, w, h in (canvas.slot_pt(0, 0, 5),
                                                     canvas.slot_pt(2, 0, 5)))
                        / (464.4 * (448.0 - 22 - 34)), 4),
                    schematic_formula=('sum(schematic slot w_pt * h_pt) / '
                                       '(live_w_pt * live_h_pt)')),
        delta0_exemption=dict(panel='A', reason=DELTA0_REASON),
        forest_panels=['B', 'C'],
        scan_rug_offset_rows=0.45,
        scan_rug_note=('descriptive within-row distribution rug, not the '
                       'CF-6 +0.22 second-arm offset'),
        representative=dict(zip(['target_root_id', 'session', 'scan_idx'], key)),
        selection='Median mapped-input count, identifier ties; no outcome selection',
        coordinate_definition=(
            'Rows are mapped partner inputs; multiple inputs can share a '
            'physical segment. Legacy n_sites and sites_per_route keys count '
            'these input coordinates.'),
        ancestry_placement=dict(shared_subtree=shared, own_branch=singles,
                                rule='deepest shared ancestor of the mapped '
                                     'input segments; schematic placement only'),
        n_targets=n_targets, n_scans=n_scans,
        selected_scan_partners=n_partners,
        partners_per_scan=[p_lo, p_hi],
        panel_b={r['label']: dict(mean=r['mean'], ci95=[r['lo'], r['hi']],
                                  note=r['note']) for r in b_rows},
        panel_b_drawn=[b_rows[0]['label']],
        panel_b_note=('the all-scan cohort is computed and recorded here but '
                      'no longer redrawn in B: the same estimate is drawn '
                      'once, as C\'s green top row'),
        panel_c={r['label'].replace('\n', ' '):
                 dict(mean=r['mean'], ci95=[r['lo'], r['hi']], note=r['note'])
                 for r in c_rows},
        panel_d=dict(threshold_measured=cut_m, threshold_perfect=cut_p,
                     lambda_mc_respecting=lam, false_detection=floor,
                     mc_halfwidth_max=mc_hw,
                     n_replicates=int(curves['measured'].n_replicates.max()),
                     inset_bins=[int(v) for v in hist_counts],
                     inset_bin_edges=[float(v) for v in hist_edges],
                     inset_records=n_records,
                     inset_records_drawn=n_records,
                     inset_negative_records=n_negative,
                     inset_partners=n_partners_unique),
        panel_f=dict(mean_coverage_pct=mean_cov,
                     all_one_site_scans=int(single.sum()),
                     scans_below_k_over_n=int(len(below)),
                     scans_above_k_over_n=int(len(above)),
                     above_rows=[dict(target_root_id=int(r.target_root_id),
                                      session=int(r.session),
                                      scan_idx=int(r.scan_idx),
                                      n_sites=int(r.n_sites),
                                      sites_per_route=round(float(r.sites_per_route), 4),
                                      coverage_pct=round(float(r.coverage) * 100, 1))
                                 for r in above.itertuples()],
                     below_rows=[dict(target_root_id=int(r.target_root_id),
                                      session=int(r.session),
                                      scan_idx=int(r.scan_idx),
                                      n_sites=int(r.n_sites),
                                      coverage_pct=round(float(r.coverage) * 100, 1))
                                 for r in below.itertuples()]),
        observed_selected_scan=dict(mean=float(obs['mean']),
                                    ci95_low=float(obs.ci95_low),
                                    ci95_high=float(obs.ci95_high)),
        source_sha256={str(p.relative_to(J)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in files},
        layout_findings=list(problems))
    (REC / 'figure_08_sources.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ('source_sha256', 'ancestry_placement')},
                     indent=2))


if __name__ == '__main__':
    main()
