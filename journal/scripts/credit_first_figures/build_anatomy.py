#!/usr/bin/env python3
"""Main Figure 7 (``fig:topology``) -- ancestry routes on reconstructed arbors.

Production builder for main Figure 7 under the v2 overhaul plan
``analysis/figure_overhaul_20260908/v2/fig7/PLAN.md`` as amended by
``v2/AMENDMENTS.md`` and ruled by ``v2/DECISIONS.md`` (DECISIONS names this
file the production builder and retires ``build_restored_main.py::figure7``
unedited; its positional-``METHODS`` defect is recorded in
``IMPLEMENTATION_NOTES.md``).  Entry point: :func:`figure7`.

Eight lettered panels on one native canvas, 518.4 x 490.0 pt (aspect 1.058),
three rows 122 / 119 / 109 pt, 38 pt horizontal and 40 pt vertical gutters,
margins 52 / 14 / 24 / 36::

    row 0   A routes on an arbor   B seven routes, one dictionary   C the field
    row 1   D capture vs budget    E where the energy goes          F cell by cell
    row 2   G paired advantage and its wiring cost      H three cohorts

Every mean, interval, matrix entry and field value is read from the frozen
tables under ``source_data/anatomy_commonmode`` (and ``source_data/figure3``
for the skeleton); nothing under ``source_data/`` is written.

Cross-figure rules CF-1 .. CF-12 (AMENDMENTS section 3), restated:
CF-1 canvas 518.4 x 490 pt, on the 340/415/490 ladder, aspect 1.058 >= 1.05.
CF-2 exactly three type sizes 7.0 / 8.0 / 9.0-bold; no DejaVu; subscripts via
     ``Frame.subscript`` / ``token_subscript``, never mathtext.
CF-3 strokes only 0.55 / 0.70 / 0.85 / 0.95 / 1.25 pt; every area mark is a
     16 % ``tint_patch`` with a 0.55 pt edge; no open stroke >= 1.35 pt.
CF-4 the glyph family: filled soma disc with an ink rim, tapered ``dend``
     strokes, open white junction rings, filled ``exc`` / ``inh`` contacts,
     shunt = contact + 7 pt badge, attenuation at FADE_PCT 38, ghosts at
     GHOST_PCT 45, one ink delta-0 arrow into the soma of panel C.  Panels A
     and B are named in CF-4's closed exemption list and declare their
     ``DELTA0_EXEMPTIONS`` reason.  No delivery glyph is drawn in this figure
     (the dictionary is A and its coefficients c; no credit is delivered), so
     the four-delivery-mode rule is not exercised.
CF-5 zero legends and zero keys inside any data axes (the set's one sanctioned
     key is Fig 5C): D end labels, E direct segment labels and named bars,
     G gutter row labels, H Pinky-side family names.
CF-6 the forest idiom in G through ``figure_canvas.forest``; second arm at
     +0.22 rows with an open marker; the cell fan at -0.22 rows.
CF-7 one dashed ``mute`` reference per reference, label right-aligned on the
     line, zero drawn once (D's broadcast floor, G's no-advantage line,
     H's ceiling).
CF-8 caption rules (the caption lives in v2/fig7/TEXT.md and is mirrored to
     ``figures/provenance/structure_restoration_20260908/figure_07_caption.md``).
CF-9 titles: sentence case, no terminal period, <= 26 characters at <= 4
     modules and <= 42 above.  ``Cell by cell`` (F) is one of the set's two
     recorded method-naming titles.
CF-10 schematic area on the single B12 formula
     ``3 x (125.5 x 122) / (452.4 x 430) = 23.61 %`` <= 30 %, no waiver.
     The v1 row-height/full-canvas ratio 24.9 % is retired and appears nowhere.
CF-11 matrices: ``check_matrix_cells`` >= 6.0 pt per row and column, headers
     <= 1.5 x the column; ZERO image XObjects in this figure.
CF-12 letters 9 pt bold through ``canvas.align_letters()``, <= 0.5 pt spread.

Waivers and declared deviations, in one place (all reported in the build note
and written into the canvas manifest / provenance record):

* **D3 waiver** -- row 1 (D budget curve / E decomposition / F per-cell
  scatter) is three 4-module panels that share no axis; each is a different
  estimand and the row is column-locked.
* **H idiom waiver** (plan decision 0.7) -- H is a grouped vertical dot plot,
  not a forest: at 5 modules a twelve-row horizontal forest cannot carry
  twelve labelled rows, and the drawn 1.0 ceiling needs a horizontal
  reference.  Section 5's forest rule governs one-category-vs-value panels;
  H is a two-factor panel.
* **Palette waiver** (plan section 4.0) -- the six family hues are
  ``shunting / point_mlp / ink / highlight / low_rank / oracle``; worst normal
  OKLab dE*100 = 17.39 (shunting/point_mlp), worst CVD = 7.20
  (shunting/low_rank, protan; deutan 9.35), below the 10.0 floor.  Accepted
  because (i) it is forced -- under B14's constraints ``bp`` and ``additive``
  are banned from this figure and amber is reserved for the broadcast, and
  every subset containing ``per_soma`` fails harder (shunting/per_soma protan
  1.67); (ii) the pair is separated by marker shape ``o`` vs ``X``, by the
  band in D and by direct labels; (iii) the two are never adjacent in E's bar
  order or G's row order.  AMENDMENTS' stated fallback
  (``random = mix('point_mlp', 55)``) is NOT taken: a grey tint beside the
  grey ``point_mlp`` surrogate series in the same panel is a worse failure.
* **B's address cycle** -- ``journal_style.K_CYCLE``'s second entry is
  ``additive``, which B14 bans from this figure, and SPEC_ERRATA #7 forbids
  editing the library, so B uses a builder-local three-hue address cycle
  (``shunting``, ``local``, ``oracle``) at two tint levels.  The plan's
  62 %-of-ink mixes are replaced by WHITE tints: an ink mix is neither a
  registered role colour nor a white tint of one, and the strict audit's
  role-colour check fails it (measured this session: ``mix('shunting', 62,
  'ink')`` is dE 6.2 from ``ordinal4``, ``mix('local', 62, 'ink')`` dE 10.0
  from ``low_rank``, ``mix('oracle', 62, 'ink')`` dE 9.5 from ``additive``).
  White tints of the same three hues are recognised as weakened role colours
  and pass.  Amber in B is the address register of a schematic; the broadcast
  tag ``s`` is the only amber in B's matrix and the footer says so.
* **Soma position on a measured arbor** -- CF-4's "soma is the lowest node"
  cannot hold for a reconstruction: this cell has a basal skirt, and no
  rotation of the principal plane puts segment 0 at the bottom (measured:
  the best rotation still leaves 27 % of the drawing's height below the
  soma; the pia-up orientation used here leaves 36 of 78 segments below it).
  A/B/C therefore draw the real arbor in the anatomical pia-up orientation --
  the principal plane rotated so the projected pia axis points up -- with the
  library's own filled soma disc and ink rim; ``Frame.require_soma_lowest()``
  is called in all three panels (it passes: the arbor is drawn through Frame
  primitives and registers no library ``Nodes`` tree) and the measured fact is
  recorded as a ``soma-below`` schematic note in the manifest.
* **Vector matrices** -- ``Frame.dictionary_matrix`` and
  ``Frame.dictionary_product`` draw through ``imshow``, i.e. an image XObject,
  which CF-11 forbids in this figure (an 8 x 8 image over 52 pt resolves at
  11 dpi, far below RASTER_DPI_MIN).  B's dictionary and C's field strip are
  drawn as vector cells by the private ``_matrix_cells`` helper, which calls
  the library's own ``check_matrix_cells`` so the 6.0 pt floor and the
  1.5 x header rule are enforced by the library, not by the builder.
* **``Frame.arbor`` does not exist** (SPEC_ERRATA #7 / DECISIONS G5): the
  private ``_arbor`` helper below reuses ``build_main_figure_07``'s projection
  and draws through ``Frame.dendrite`` / ``junction`` / ``soma`` / ``contact``
  / ``shunt`` / ``fade`` / ``error_in`` and ``journal_style.tint_patch``.
  Reported as a library follow-up.
* **Axis-label and annotation wording** is wrapped to the panel width at
  7.0 / 8.0 pt; where the plan's verbatim string cannot be set inside a
  4-module panel at the type floor it is set on two lines or shortened, and
  the full phrase is carried by the caption.  Every such case is listed in the
  build note.
* **D and E share one fraction axis** (plan check 8): both are 0 -> 1.34 over
  the same row, so their points-per-unit are identical to well under the
  0.25 pt tolerance.  The plan's separate limits (D 0.15-1.0, E 0-1) could not
  both hold; the headroom above 1.0 carries each panel's on-panel notes.

Further declared deviations from the v2 plan, every one forced by a binding
rule or by the 7.0 pt type floor inside a 4-module (125.5 pt) panel:

1. **Titles A and B shortened to CF-9's 26-character limit** (the plan's table
   measured them against the 42-character limit, which applies above 4
   modules): `Routes on a reconstructed arbor` (31) -> `Routes on a real arbor`
   (22); `Seven routes, one dictionary` (28) -> `One arbor, seven routes` (23).
   C-H are unchanged and already inside their limits.
2. **Row-0 titles are set as axes titles**, not inside the drawing cell, so the
   three schematics keep 13 pt more drawing height and their letters, titles
   and baselines match rows 1 and 2.
3. **One label gutter of 37 pt is declared for D, E, F and G.**  ``forest()``
   measures the gutter its row labels need (36.9 pt); the audit requires every
   4-module panel of a row to share one axes width and every panel of a grid
   column to share one x0, so the same reserve is declared on D, E and F.
4. **E prints no per-family numbers.**  Six values of ~17 pt on a 14.8 pt bar
   pitch collide at the type floor; the family names are staggered on two
   baselines with their marker glyphs, the three shares are named once in the
   reserved gutter at the ancestry bar's band heights (which is also why E
   carries no separate y label), and the broadcast value 0.203 is printed in
   the gutter and again on D's reference line.  The six spatial values are in
   the caption's ancestry/oracle pair and in ``figure_07_plotted.csv``.
5. **G's axis runs to +88 with the fan clipped at +55.**  The 188 within-cell
   differences span -39.8 to +76.5 pp; drawing all of them would compress the
   four paired means and their intervals to a few points.  Four of 188 fall
   outside and the panel says so.  The per-row n is ``forest()``'s own
   right-hand note; wiring and rank are printed inside the axes beyond the fan
   window; the zero reference is drawn once, across the rows only.
6. **F is not ``set_aspect('equal')``**: an aspect-adjusted axes changes its
   ACTIVE box, which breaks the row-height and row-alignment locks.  Both axes
   keep the same 0.25-1.0 range instead, and the headroom above 1.0 carries the
   two annotation lines and the ``equal`` end label.
7. **Axis labels and annotations wrapped or shortened to the panel width**:
   D's `Profiles K (log 2)` (the `one broadcast + K-1 spatial` gloss moves to
   the caption), F's `200-surrogate mean at K = 8`, H's Pinky/cohort footnotes,
   and G's footer set on four lines.  Every string that lost words is listed in
   ``analysis/figure_overhaul_20260908/v2/fig7/TEXT.md``.
8. **G's 6 % row band is trimmed to the fan window** (it ends at the column
   rule, +56.5 pp) instead of spanning the whole x range as ``forest()`` draws
   it.  A band is a row cue, not a datum, and with the full-width band the
   three printed columns sat on an area mark -- eight TEXT-ON-DATA findings
   from ``audit_text_over_data``.  CF-6's band is still drawn on every row.
9. **E carries no `oracle` badge** (plan section 4 E).  The only free paper in
   E is the 21 pt strip above the bars, and it carries the initial-cohort
   annotation plus this panel's ``n = 47 cells at K = 8``, which a data panel
   needs more than a badge.  The sixth bar is named ``SVD`` under its diamond
   marker and the badge is drawn in H.
10. **The right-hand overhang is deliberate.**  ``FILL_W_MIN`` asks for ink
   across >= 92 % of 518.4 pt, and this plan's margins (52 / 14) leave a live
   area of 452.4 pt = 87.3 %: 16 pt come free from the panel letters on the
   left, so ~9 pt must overhang on the right.  H's four family names carry it
   (right-aligned 10 pt beyond the spine, beside their leaders); every other
   label stays inside its axes.
11. **Single bars in C's capture formula.**  Nimbus Sans has no U+2016 DOUBLE
   VERTICAL LINE (checked: matplotlib falls back to DejaVu, which CF-2 bans),
   so the norms are set as ``|P t|`` and ``|t|`` with W subscripts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))
sys.path.insert(0, str(JOURNAL / "code/reconstructed_tree"))

import build_main_figure_07 as anatomy                        # noqa: E402
from analyze_microns_morphology_credit import (                # noqa: E402
    ancestry_matrix, parent_map)
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR,  # noqa
                           LW_REF, MARKER_MS, SEED_MS, Margins,
                           NativeCanvas, PT_BASE, PT_EMPH, style_panel)
from journal_style import (DIV_CMAP, label_color,               # noqa: E402
                           palette_report, tint_patch)
from credit_tree_schematics import mix                          # noqa: E402
from native_schematics import (BADGE_STYLE, Frame,               # noqa: E402
                               check_matrix_cells, reference_line)

SOURCE = JOURNAL / "source_data"
COMMON = SOURCE / "anatomy_commonmode"
COMPONENT = JOURNAL / "figures/components/credit_first_figure_07.pdf"
MAIN = JOURNAL / "figures/main/figure_07.pdf"
RECORDS = JOURNAL / "figures/provenance/structure_restoration_20260908"
LEGACY_RECORDS = SOURCE / "credit_first_figures"

# ── families: an explicit map keyed by the literal method strings ────────
# Plan decision 0.2 (mandatory).  METHODS is NEVER indexed positionally: the
# list was reordered by the 2026-09-08 library commit while the old
# ``figure7()`` still read ``METHODS[1]`` as "Surrogate tree", which silently
# plotted the SVD oracle under the surrogate's name.
FAMILIES = {
    "common + ancestry": dict(
        label="Ancestry", short="Ancestry", color="shunting", marker="o"),
    "common + surrogate ancestry": dict(
        label="Surrogate tree", short="Surrogate", color="point_mlp",
        marker="s"),
    "common + depth bins": dict(
        label="Depth bins", short="Depth", color="ink", marker="^"),
    "common + shuffled routes": dict(
        label="Shuffled routes", short="Shuffled", color="highlight",
        marker="v"),
    "common + random routes": dict(
        label="Random routes", short="Random", color="low_rank", marker="X"),
    "common-constrained SVD": dict(
        label="SVD oracle", short="SVD", color="oracle", marker="D"),
}
ANCESTRY = "common + ancestry"
SURROGATE = "common + surrogate ancestry"
DEPTH = "common + depth bins"
SHUFFLED = "common + shuffled routes"
RANDOM = "common + random routes"
ORACLE = "common-constrained SVD"
#: Drawing order: ancestry, surrogate, depth, shuffled, random, oracle.
ORDER = [ANCESTRY, SURROGATE, DEPTH, SHUFFLED, RANDOM, ORACLE]
#: The four families H draws in all three cohorts (plan decision 0.8).
COHORT_FAMILIES = [ANCESTRY, SURROGATE, DEPTH, ORACLE]
#: The four paired controls of G, top row first.
CONTROLS = [SURROGATE, DEPTH, SHUFFLED, RANDOM]

# Legacy names kept so ``build_restored_main.py`` still imports (that builder
# is retired for Figure 7 and is left byte-identical).
METHODS = list(ORDER)
LABELS = [FAMILIES[m]["label"] for m in ORDER]

COHORTS = ["original8", "v661", "pinky"]
COHORT_PANEL = {"original8": "Initial,\n8 cells", "v661": "Disjoint,\n47 cells",
                "pinky": "Pinky,\n8 cells"}
BOOT_SEED = 202609061
FAN_SEED = 26090847
N_BOOT = 20_000
ROOT_B = 864691135409937097
ROUTE_SITE = 4396           # route 3: the inhibitory origin C shunts
BROADCAST = "local"         # amber, reserved in this figure for the broadcast

# Canvas geometry (CF-1 / CF-10).
CANVAS_H_PT = 490.0
ROW_H = [122.0, 119.0, 109.0]
MARGINS = dict(left=52.0, right=14.0, top=24.0, bottom=36.0)
HGUT, VGUT = 38.0, 40.0
LIVE_W = 518.4 - MARGINS["left"] - MARGINS["right"]          # 452.4
LIVE_H = CANVAS_H_PT - MARGINS["top"] - MARGINS["bottom"]    # 430.0
MODULE_PITCH = (LIVE_W + HGUT) / 12.0                        # 40.867
SLOT4_W = 4 * MODULE_PITCH - HGUT                            # 125.47
SCHEMATIC_FRACTION = 3 * (SLOT4_W * ROW_H[0]) / (LIVE_W * LIVE_H)
SCHEMATIC_INSET = (7.0, 7.0, 7.0, 13.0)  # left, right, top, bottom (points)
GUTTER_PT = 37.0          # one label gutter for every 4+ module panel
BOTTOM_R1 = 5.0           # room under row 1 for D's footnote line
BOTTOM_R2 = 20.0          # room under row 2 for G's and H's footnotes


CAPTION = r"""\caption{\textbf{Ancestry routes on reconstructed arbors compress a cell's own focal-shunt response fields better than four matched controls, at a fifth of dense wiring, and the advantage is modest and heterogeneous.}
\textbf{A}, Median-sized arbor of the initial eight-cell cohort (root 864691135409937097, 78 segments): its 76 inhibitory-bearing segments and route 3 over the five sites it addresses; schematic, no data; scale bar, 50~$\mu$m.
\textbf{B}, The same arbor's seven $K=8$ routes and the matrix $A_8$, collapsed to eight tree-ordered site blocks; supports nest and 32 of 70 sites lie on a route; schematic, no data (entries measured).
\textbf{C}, A focal shunt on route 3 attenuates its descendants and makes the field $\bm t$ (block means); capture is the $W$-weighted energy of $P_{A_K,W}\bm t$; schematic, no data (field column measured).
\textbf{D}, Total captured energy against budget $K$; cell means with unpaired 95\% cell-bootstrap bands (20,000 draws) for ancestry and surrogates; dashed floor, the shared broadcast 0.203; $n=47$ cells, 46 at $K=16$.
\textbf{E}, That energy at $K=8$ as broadcast, spatial and unexplained shares; cell means, no interval; $n=47$ cells.
\textbf{F}, Each cell's ancestry capture against its own 200-surrogate mean; 38 of 47 lie above equality, open symbols the 11 cells whose surrogates match it at least half the time; cohort mean with 95\% cell-bootstrap intervals; $n=47$ cells at $K=8$.
\textbf{G}, Paired ancestry-minus-control advantages at $K=8$ on post-broadcast (filled) and total (open) scales, with all 47 within-cell differences; means and 95\% cell-bootstrap intervals; right: wiring, rank, positive cells.
\textbf{H}, Residual capture after the broadcast in three cohorts, one mouse each; means with 95\% cell-bootstrap intervals at $K=8$; $n=8$, 47 and 8 cells.
Randomized controls average 200 draws; the SVD is an oracle ceiling; typed-contact and joint 3D matching controls are in Supplementary Figs.~S25B and~S26B--D. Fields are modeled passive responses: compression capacity, not observed teaching. Source Data: \texttt{source\_data/curated\_publication/figure\_07\_plotted.csv}.}"""


# ── small helpers ────────────────────────────────────────────────────────
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_ci(values, seed):
    """Descriptive cell-bootstrap mean and 95 % interval (20,000 draws)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(N_BOOT, values.size),
                       replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def text_w_pt(ax, text, size):
    """Rendered width of one line, in points."""
    fig = ax.get_figure()
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:                                     # pragma: no cover
        return 0.6 * size * len(text)
    art = ax.text(0, 0, text, fontsize=size)
    w = art.get_window_extent(renderer=renderer).width * 72.0 / fig.dpi
    art.remove()
    return float(w)


def wrap_pt(ax, text, size, width_pt):
    """Greedy word wrap of ``text`` to ``width_pt`` at ``size``."""
    words, lines, cur = str(text).split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if cur and text_w_pt(ax, trial, size) > width_pt:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# ── data loaders (frozen tables only) ────────────────────────────────────
def cohort_tables():
    return {c: pd.read_csv(COMMON / c / "cell_method_summary.csv")
            for c in COHORTS}


def cohort_summaries():
    return {c: pd.read_csv(COMMON / c / "cohort_method_summary.csv")
            for c in COHORTS}


def reports():
    return {c: json.loads((COMMON / c / "summary.json").read_text())
            for c in COHORTS}


def surrogate_pairs():
    """Per v661 cell at K = 8: ancestry total capture, the mean of its 200
    surrogate trees and the fraction of those trees that reach or beat it."""
    rows = []
    for path in sorted((COMMON / "v661" / "cells").glob("rows_*.csv.gz")):
        table = pd.read_csv(path)
        eight = table[table.channels.eq(8)]
        tree = float(eight[eight.method.eq(ANCESTRY)].total_capture.iloc[0])
        draws = eight[eight.method.eq(SURROGATE)].total_capture.to_numpy(float)
        rows.append(dict(root_id=int(table.root_id.iloc[0]), tree=tree,
                         surrogate_mean=float(draws.mean()),
                         n_replicates=int(draws.size),
                         fraction_ge=float((draws >= tree).mean())))
    return pd.DataFrame(rows)


def arbor_routes(n_routes=7):
    """The seven K = 8 ancestry routes of ROOT_B, reproduced from
    ``scripts/anatomy_commonmode/run.py`` lines 105-112, with the eight
    tree-ordered site blocks B and C share."""
    segments = pd.read_csv(SOURCE / "figure3" / "segment_metrics.csv")
    cell = segments[segments.root_id.eq(ROOT_B)].copy()
    _, parent, children = parent_map(cell)
    npz = np.load(COMMON / "original8" / "cells" / f"operator_{ROOT_B}.npz")
    e_sites = [int(v) for v in npz["e_sites"]]
    i_sites = [int(v) for v in npz["i_sites"]]
    beta = np.asarray(npz["beta"], float)
    weights = np.asarray(npz["weights"], float)
    ancestry_A = ancestry_matrix(e_sites, i_sites, parent)
    order = np.argsort(-(beta * (ancestry_A.T @ weights) / weights.sum()))
    chosen = [int(j) for j in order[:n_routes]]
    origins = [i_sites[j] for j in chosen]
    supports = [ancestry_A[:, j] > 0 for j in chosen]
    union = np.zeros(len(e_sites), bool)
    for support in supports:
        union |= support
    coverage = int(union.sum())
    # Plan section 3 B.5, mandatory:
    assert origins == [4784, 4621, 4396, 4458, 4975, 4209, 4516], origins
    assert [int(s.sum()) for s in supports] == [1, 1, 5, 1, 1, 29, 1]
    assert coverage == 32 and len(e_sites) == 70
    depth = {int(r.segment_id): int(r.topological_depth)
             for r in cell.itertuples(index=False)}
    # the eight disjoint site blocks: one per distinct route membership,
    # ordered proximal to distal by the depth of the deepest origin on them
    blocks = {}
    for index, site in enumerate(e_sites):
        key = tuple(k for k in range(n_routes) if supports[k][index])
        blocks.setdefault(key, []).append(index)
    keyed = sorted((k for k in blocks if k),
                   key=lambda k: (max(depth[origins[j]] for j in k), min(k)))
    block_keys = keyed + [()]
    block_sizes = [len(blocks[k]) for k in block_keys]
    assert block_sizes == [23, 1, 1, 1, 4, 1, 1, 38], block_sizes
    # nesting, asserted on the supports themselves (4458 in 4396 in 4209)
    idx = {o: k for k, o in enumerate(origins)}
    assert supports[idx[4396]][supports[idx[4458]]].all()
    assert supports[idx[4209]][supports[idx[4396]]].all()
    return dict(cell=cell, parent=parent, children=children, npz=npz,
                e_sites=e_sites, i_sites=i_sites, weights=weights,
                origins=origins, supports=supports, coverage=coverage,
                depth=depth, blocks=blocks, block_keys=block_keys,
                block_sizes=block_sizes)


def block_field(arb):
    """C's field: the measured operator column of i-site 4396, reduced to the
    eight tree-ordered site blocks (block means) and scaled by max |t|."""
    npz = arb["npz"]
    column = arb["i_sites"].index(ROUTE_SITE)
    response = np.asarray(npz["weighted_response"], float)[:, column]
    scale = float(np.abs(response).max())
    values = [float(response[arb["blocks"][k]].mean()) / scale
              for k in arb["block_keys"]]
    return np.asarray(values), scale


# ── private glyph helpers (missing from the shared library) ──────────────
def _arbor_geometry(cell):
    """Isotropic principal-plane projection, rotated so pia points up.

    ``build_main_figure_07.morphology_geometry`` fixes the plane (the first
    two principal axes of the segment cloud, isotropic, normalised by the
    larger span); this adds the one rotation inside that plane that puts the
    projected cortical (pia) direction on +y, so the arbor fans up and the
    ``pia`` arrow is a measured direction rather than a convention.
    """
    xyz = cell[["x_um", "y_um", "z_um"]].to_numpy(float)
    centred = xyz - xyz.mean(axis=0, keepdims=True)
    _, _, basis = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ basis[:2].T
    span_um = max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]))
    projected = projected / span_um
    pia = np.array([0.0, -1.0, 0.0]) @ basis[:2].T      # MICrONS y grows down
    pia = pia / max(float(np.linalg.norm(pia)), 1e-12)
    theta = np.arctan2(pia[0], pia[1])
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    projected = projected @ rot.T
    positions = {int(seg): point for seg, point
                 in zip(cell.segment_id.to_numpy(int), projected, strict=True)}
    rows = {int(row.segment_id): row for row in cell.itertuples(index=False)}
    parent = {int(row.segment_id): int(row.parent_segment_id)
              for row in cell.itertuples(index=False)}
    return positions, rows, parent, span_um


def _fit_iso(f, rect, xy, pad_pt=1.0):
    """Isotropic map of projected coordinates into ``rect`` (frame fractions)."""
    xy = np.asarray(xy, float)
    x0, y0, w, h = rect
    avail_x = w * f.w_pt - 2 * pad_pt
    avail_y = h * f.h_pt - 2 * pad_pt
    span_x = max(np.ptp(xy[:, 0]), 1e-9)
    span_y = max(np.ptp(xy[:, 1]), 1e-9)
    scale = min(avail_x / span_x, avail_y / span_y)
    ox = x0 + (w - f.fx(span_x * scale)) / 2.0
    oy = y0 + (h - f.fy(span_y * scale)) / 2.0
    xmin, ymin = xy[:, 0].min(), xy[:, 1].min()

    def place(point):
        return (ox + f.fx((float(point[0]) - xmin) * scale),
                oy + f.fy((float(point[1]) - ymin) * scale))

    return place, scale


def _arbor(f, rect, arb, *, mode="plain"):
    """The one real-arbor drawing of row 0, through Frame primitives.

    ``mode='plain'`` draws the tapered warm-grey tree at full strength;
    ``mode='ghost'`` draws it at GHOST_PCT so routes or a shunted subtree can
    be stroked over it.  Returns ``(place, xy, soma_xy, scale, span_um,
    n_below)`` where ``n_below`` is how many segments sit below the soma --
    the measured fact behind this builder's ``soma-below`` note.
    """
    cell = arb["cell"]
    positions, rows, parent, span_um = _arbor_geometry(cell)
    place, scale = _fit_iso(f, rect, np.asarray(list(positions.values()),
                                                float))
    xy = {seg: place(p) for seg, p in positions.items()}
    depth = arb["depth"]
    top = max(int(d) for d in depth.values()) or 1
    ghost = mode == "ghost"
    for seg, par in parent.items():
        if par not in rows:
            continue
        level = min(3, int(round(3.0 * depth[seg] / top)))
        f.dendrite(xy[seg], xy[par], level=level, ghost=ghost, zorder=2)
    kids = {}
    for seg, par in parent.items():
        if par in rows:
            kids.setdefault(par, []).append(seg)
    for seg, children in kids.items():
        if len(children) > 1 and depth[seg] > 0:
            f.junction(xy[seg], ghost=ghost, zorder=3)
    soma_id = min(rows, key=lambda key: depth[key])
    n_below = int(sum(1 for seg, p in xy.items()
                      if p[1] < xy[soma_id][1] - 1e-9))
    return place, xy, xy[soma_id], scale, span_um, n_below


def _scale_bar(f, rect, scale, span_um, *, um=50.0, x0=None, y0=None):
    """50 micron bar in ink at LW_DATA with its label above it."""
    length = f.fx(um / span_um * scale)
    bx = rect[0] + f.fx(2.0) if x0 is None else x0
    by = rect[1] + f.fy(2.0) if y0 is None else y0
    f.ax.plot([bx, bx + length], [by, by], color=COLORS["ink"], lw=LW_DATA,
              solid_capstyle="butt", zorder=6)
    f.text((bx + length / 2.0, by + f.fy(1.6)), f"{um:.0f} µm",
           size=PT_BASE, color=COLORS["ink"], va="bottom")
    return length


def _matrix_cells(f, rect, values, colors, *, n, k, where, headers=None):
    """A matrix drawn as VECTOR cells (CF-11: no image XObject in Fig 7).

    Stands in for ``Frame.dictionary_matrix`` / ``dictionary_product``, both
    of which go through ``imshow``.  The library's own ``check_matrix_cells``
    is called so the 6.0 pt floor and the 1.5 x header rule are enforced by
    the library rather than restated here.
    """
    w_pt, h_pt = rect[2] * f.w_pt, rect[3] * f.h_pt
    check_matrix_cells(w_pt, h_pt, n, k, where=where, headers=headers,
                       ax=f.ax)
    cw, ch = rect[2] / k, rect[3] / n
    for i in range(n):
        for j in range(k):
            f.ax.add_patch(Rectangle(
                (rect[0] + j * cw, rect[1] + rect[3] - (i + 1) * ch), cw, ch,
                facecolor=colors[i][j], edgecolor=COLORS["edge"],
                lw=LW_HAIR, zorder=3))
    _ = values
    return cw, ch


# ── row 0: the schematics ────────────────────────────────────────────────
def _foot(f, core, lines, *, size=PT_BASE, lead=8.6):
    """Wrapped footer lines at the foot of a schematic cell; returns points."""
    text = [wrap_pt(f.ax, line, size, core[2] * f.w_pt) for line in lines]
    total = sum(1 + t.count("\n") for t in text) * lead
    y = core[1] + f.fy(total - lead * 0.25)
    for block in text:
        f.text((core[0], y), block, size=size, color=COLORS["mute"],
               ha="left", va="top", linespacing=1.18)
        y -= f.fy(lead * (1 + block.count("\n")))
    return total


def panel_a(ax, arb):
    """A: the reconstruction, its inhibitory-bearing segments and route 3."""
    f = Frame(ax)
    core = (0.0, 0.0, 1.0, 1.0)
    foot_pt = _foot(f, core, [f"root {ROOT_B}"])
    key_pt = 18.0
    draw = (core[0], core[1] + f.fy(foot_pt + key_pt), core[2],
            core[3] - f.fy(foot_pt + key_pt))
    place, xy, soma, scale, span_um, n_below = _arbor(f, draw, arb,
                                                      mode="plain")
    cell = arb["cell"]
    i_bearing = [int(r.segment_id) for r in cell.itertuples(index=False)
                 if float(r.I_size) > 0]
    for seg in i_bearing:
        f.contact(xy[seg], kind="inh", dia_pt=2.4, zorder=4.4)
    # route 3: its inhibitory origin, its five input-bearing descendants and
    # a 16 % shunting tint capsule over the support they form
    idx = arb["origins"].index(ROUTE_SITE)
    support = [site for site, on in zip(arb["e_sites"], arb["supports"][idx])
               if on]
    chain, parent = [], arb["parent"]
    for site in support:
        cursor = int(site)
        while cursor in xy:
            chain.append(cursor)
            if cursor == ROUTE_SITE:
                break
            cursor = parent.get(cursor, -1)
    tint_patch(f.ax, ("ribbon", [[xy[s] for s in sorted(set(chain))]], 7.0),
               color="shunting", pct=16, radius_pt=2.0, zorder=1.0,
               clip_on=False)
    f.contact(xy[ROUTE_SITE], kind="inh", dia_pt=3.6, zorder=5)
    for site in support:
        f.contact(xy[site], kind="exc", dia_pt=3.0, zorder=5)
    f.soma(soma, zorder=6)
    # pia arrow: the projected cortical axis, which this projection puts on +y
    px = draw[0] + f.fx(3.0)
    py = draw[1] + draw[3]
    f.arrow((px, py - f.fy(12.0)), (px, py - f.fy(1.0)), color=COLORS["mute"],
            lw=LW_EDGE, head=4.0, zorder=5)
    f.text((px + f.fx(2.0), py - f.fy(6.0)), "pia", size=PT_BASE,
           color=COLORS["mute"], ha="left")
    tag = (xy[ROUTE_SITE][0] + f.fx(9.0), xy[ROUTE_SITE][1] + f.fy(10.0))
    f.leader(xy[ROUTE_SITE], tag)
    f.text((tag[0] + f.fx(1.0), tag[1]), "route 3", size=PT_BASE,
           color=COLORS["shunting"], ha="left")
    _scale_bar(f, draw, scale, span_um,
               x0=draw[0] + draw[2] - f.fx(50.0 / span_um * scale + 1.0),
               y0=draw[1] + f.fy(3.0))
    for row, (kind, text) in enumerate((
            ("inh", f"inhibitory-bearing ({len(i_bearing)} of {len(cell)})"),
            ("exc", f"route-3 sites ({len(support)} of "
                    f"{len(arb['e_sites'])})"))):
        y = core[1] + f.fy(foot_pt + key_pt - 5.0 - row * 8.6)
        f.contact((core[0] + f.fx(2.0), y), kind=kind, dia_pt=3.0, zorder=5)
        f.text((core[0] + f.fx(6.0), y), text, size=PT_BASE,
               color=COLORS["ink"], ha="left")
    f.note("soma-below", panel="A", below=n_below, segments=len(xy),
           reason=("a measured pyramidal arbor has a basal skirt: no rotation "
                   "of the principal plane makes the soma the lowest node, so "
                   "CF-4's soma-lowest rule is declared, not asserted"))
    f.require_soma_lowest()
    f.require_delta0(allow_no_delta0=True, reason=(
        "panel A/B are the anatomical construction; the somatic error enters "
        "in panel C of the same row"))
    return dict(inhibitory=len(i_bearing), route3=len(support),
                below_soma=n_below)


def _route_hues():
    """Builder-local three-hue address cycle for B's seven routes.

    ``journal_style.K_CYCLE`` is (shunting, additive, local, oracle) and B14
    bans ``additive`` from this figure; SPEC_ERRATA #7 forbids editing the
    library.  Three admissible hues at two tint levels, WHITE tints (an ink
    mix is neither a role colour nor a recognised tint of one and fails the
    strict audit's role-colour check).
    """
    base = ["shunting", BROADCAST, "oracle"]
    hues = [COLORS[base[k]] for k in range(3)]
    hues += [mix(base[k], 72, "white") for k in range(3)]
    hues += [mix(base[0], 48, "white")]
    return hues


def panel_b(ax, arb):
    """B: the same arbor's seven routes and the collapsed dictionary A8."""
    f = Frame(ax)
    core = (0.0, 0.0, 1.0, 1.0)
    foot_pt = _foot(f, core, [
        "rows: eight tree-ordered site blocks",
        f"{arb['coverage']} of {len(arb['e_sites'])} sites lie on a route; "
        "supports nest (4 ⊂ 3 ⊂ 6)"])
    sub_pt = 10.0
    f.text((core[0] + core[2] / 2.0, core[1] + core[3] - f.fy(sub_pt * 0.4)),
           "A8 = [ s | r1 ... r7 ]", size=PT_BASE, color=COLORS["ink"])
    body = (core[0], core[1] + f.fy(foot_pt + 4.0), core[2],
            core[3] - f.fy(foot_pt + 4.0 + sub_pt))
    cell_pt = 6.4
    matrix_w = f.fx(8 * cell_pt)
    counts_w = f.fx(11.0)
    arbor_rect = (body[0], body[1], body[2] - matrix_w - counts_w - f.fx(4.0),
                  body[3])
    place, xy, soma, scale, span_um, n_below = _arbor(f, arbor_rect, arb,
                                                      mode="ghost")
    parent, origins = arb["parent"], arb["origins"]
    hues = _route_hues()
    for k, origin in enumerate(origins):
        cursor = int(origin)
        while cursor in xy:
            par = parent.get(cursor, -1)
            if par not in xy:
                break
            f.ax.plot([xy[cursor][0], xy[par][0]], [xy[cursor][1], xy[par][1]],
                      color=hues[k], lw=LW_EDGE, solid_capstyle="round",
                      zorder=3.0 + 0.01 * k)
            cursor = par
    f.soma(soma, zorder=6)
    taken = []
    floor_pt = arbor_rect[1] * f.h_pt
    ceiling_pt = (arbor_rect[1] + arbor_rect[3]) * f.h_pt - 3.0
    right_pt = (arbor_rect[0] + arbor_rect[2]) * f.w_pt - 1.0
    left_pt = arbor_rect[0] * f.w_pt + 1.0
    def _place(point, reach=1.0):
        """Freest of six offsets around ``point`` that stays in the cell."""
        best = fallback = None
        for dx, dy, ha, va in ((3.4, 2.8, "left", "bottom"),
                               (3.4, -2.8, "left", "top"),
                               (-3.4, 2.8, "right", "bottom"),
                               (-3.4, -2.8, "right", "top"),
                               (6.6, 0.0, "left", "center"),
                               (-6.6, 0.0, "right", "center")):
            px = point[0] * f.w_pt + reach * dx
            py = point[1] * f.h_pt + reach * dy
            room = min(((px - qx) ** 2 + (py - qy) ** 2
                        for qx, qy in taken), default=1e9)
            item = (room, px, py, ha, va)
            if fallback is None or room > fallback[0]:
                fallback = item
            top = py + (8.0 if va == "bottom" else 4.0)
            bottom = py - (8.0 if va == "top" else 4.0)
            edge = px + (4.5 if ha == "left" else -4.5)
            if top > ceiling_pt or bottom < floor_pt:
                continue
            if edge > right_pt or edge < left_pt:
                continue
            if best is None or room > best[0]:
                best = item
        _, px, py, ha, va = best or fallback
        taken.append((px, py))
        return px / f.w_pt, py / f.h_pt, ha, va

    # the broadcast tag sits under the soma disc, on the basal side where no
    # route leaves it (every route climbs to the canopy); it is reserved
    # before the numerals so none of them lands on it
    taken.append((soma[0] * f.w_pt, soma[1] * f.h_pt - 6.0))
    f.text((soma[0], soma[1] - f.fy(6.0)), "s", size=PT_BASE,
           color=COLORS[BROADCAST], ha="center", va="top")
    for k, origin in enumerate(origins):
        point = xy[int(origin)]
        f.contact(point, kind="inh", dia_pt=2.9, zorder=5)
        px, py, ha, va = _place(point)
        f.text((px, py), str(k + 1), size=PT_BASE, color=label_color(hues[k]),
               ha=ha, va=va)
    # the collapsed dictionary, drawn as vector cells (CF-11)
    keys, sizes = arb["block_keys"], arb["block_sizes"]
    colors = [[COLORS[BROADCAST]]
              + [COLORS["shunting"] if k in key else COLORS["panel_bg"]
                 for k in range(7)] for key in keys]
    m_h = f.fy(8 * cell_pt)
    m_rect = (body[0] + body[2] - matrix_w - counts_w,
              body[1] + body[3] - m_h - f.fy(9.5), matrix_w, m_h)
    headers = ["s"] + [str(k + 1) for k in range(7)]
    _matrix_cells(f, m_rect, None, colors, n=8, k=8,
                  where="figure 7B dictionary", headers=headers)
    cw, ch = m_rect[2] / 8.0, m_rect[3] / 8.0
    for j, head in enumerate(headers):
        f.text((m_rect[0] + (j + 0.5) * cw,
                m_rect[1] + m_rect[3] + f.fy(1.5)), head, size=PT_BASE,
               color=COLORS["ink"] if j else COLORS[BROADCAST], va="bottom")
    for i, size in enumerate(sizes):
        f.text((m_rect[0] + m_rect[2] + f.fx(1.5),
                m_rect[1] + m_rect[3] - (i + 0.5) * ch), str(size),
               size=PT_BASE, color=COLORS["mute"], ha="left")
    f.note("address-cycle", panel="B",
           cycle="shunting / local / oracle at two white-tint levels",
           reason="K_CYCLE's second entry is `additive`, which AMENDMENTS B14 "
                  "bans from Figure 7")
    f.note("soma-below", panel="B", below=n_below, segments=len(xy),
           reason="see panel A")
    f.require_soma_lowest()
    f.require_delta0(allow_no_delta0=True, reason=(
        "panel A/B are the anatomical construction; the somatic error enters "
        "in panel C of the same row"))
    return dict(block_sizes=sizes, coverage=arb["coverage"])


def panel_c(ax, arb, field, scale):
    """C: a focal shunt on route 3 makes the field t; capture is its energy."""
    f = Frame(ax)
    core = (0.0, 0.0, 1.0, 1.0)
    foot_pt = _foot(f, core, [
        "W = excitatory contact area",
        f"measured field column for route 3, scaled by max |t| = {scale:.3f}"])
    strip_pt, form_pt = 22.0, 12.0
    top = (core[0], core[1] + f.fy(foot_pt + strip_pt + form_pt), core[2],
           core[3] - f.fy(foot_pt + strip_pt + form_pt))
    place, xy, soma, iso, span_um, n_below = _arbor(f, top, arb, mode="ghost")
    parent = arb["parent"]
    idx = arb["origins"].index(ROUTE_SITE)
    support = [site for site, flag in zip(arb["e_sites"], arb["supports"][idx])
               if flag]
    faded = set()
    for site in support:
        cursor = int(site)
        while cursor in xy:
            par = parent.get(cursor, -1)
            if par in xy:
                faded.add((xy[cursor], xy[par]))
            if cursor == ROUTE_SITE or par not in xy:
                break
            cursor = par
    f.fade(list(faded))
    # the addressed subtree, marked with the same 16 % shunting capsule A uses
    # (the FADE tint is LIGHTER than the ghost tree, so attenuation alone
    # leaves the reader no way to see WHICH descendants the shunt reaches)
    chain = []
    for site in support:
        cursor = int(site)
        while cursor in xy:
            chain.append(cursor)
            if cursor == ROUTE_SITE:
                break
            cursor = parent.get(cursor, -1)
    tint_patch(f.ax, ("ribbon", [[xy[s] for s in sorted(set(chain))]], 6.0),
               color="shunting", pct=16, radius_pt=2.0, zorder=1.0,
               clip_on=False)
    shunt_xy = xy[ROUTE_SITE]
    f.shunt(shunt_xy, label=None)
    badge_w = text_w_pt(f.ax, "g", PT_BASE) + text_w_pt(f.ax, "shunt", PT_BASE)
    badge_xy = (max(core[0] + f.fx(1.0), shunt_xy[0] - f.fx(badge_w + 20.0)),
                shunt_xy[1] + f.fy(13.5))
    f.leader((badge_xy[0] + f.fx(badge_w + 2.5), badge_xy[1] - f.fy(0.5)),
             (shunt_xy[0] - f.fx(1.8), shunt_xy[1] + f.fy(1.8)))
    f.subscript((badge_xy[0], badge_xy[1]), "g", "shunt", size=PT_BASE,
                color=COLORS["inh"], ha="left")
    f.soma(soma, output=8.0, label="z", zorder=6)
    f.error_in(soma, label="δ0", side="left")
    # the measured field, one signed cell per site block, aligned with B
    cell_pt = 8.0
    strip_w = f.fx(8 * cell_pt)
    strip_h = f.fy(cell_pt)
    sx = core[0] + f.fx(10.0)
    sy = core[1] + f.fy(foot_pt + 9.0)
    norm = mpl.colors.Normalize(-1.0, 1.0)
    _matrix_cells(f, (sx, sy, strip_w, strip_h), None,
                  [[DIV_CMAP(norm(v)) for v in field]], n=1, k=8,
                  where="figure 7C field strip")
    f.text((sx - f.fx(2.5), sy + strip_h / 2.0), "t", size=PT_EMPH,
           color=COLORS["ink"], ha="right")
    f.text((sx + strip_w + f.fx(3.0), sy + strip_h / 2.0), "→ A c",
           size=PT_EMPH, color=COLORS["ink"], ha="left")
    sister = 6                       # row 7: the sister block on the path
    f.leader((sx + (sister + 0.5) * strip_w / 8.0, sy),
             (sx + (sister + 0.5) * strip_w / 8.0, sy - f.fy(3.5)))
    f.text((sx + strip_w, sy - f.fy(4.5)), "sister block", size=PT_BASE,
           color=COLORS["mute"], ha="right", va="top")
    # C = |P t|^2_W / |t|^2_W -- BOTH norms are W-weighted, so the equation
    # is set as two chained subscript tokens (Frame.subscript carries one
    # subscript per call and mathtext is banned by CF-2)
    eq_y = core[1] + f.fy(foot_pt + strip_pt + form_pt * 0.45)
    lead, tail = "C = |P t|²", "|t|²"
    w_lead = (text_w_pt(f.ax, lead, PT_EMPH) + 0.4
              + text_w_pt(f.ax, "W", PT_BASE) + 0.6
              + text_w_pt(f.ax, " ÷ ", PT_EMPH))
    w_tail = text_w_pt(f.ax, tail, PT_EMPH) + 0.4 + text_w_pt(f.ax, "W",
                                                              PT_BASE)
    eq_x = core[0] + core[2] / 2.0 - f.fx((w_lead + w_tail) / 2.0)
    f.subscript((eq_x, eq_y), lead, "W", " ÷ ", size=PT_EMPH, ha="left")
    f.subscript((eq_x + f.fx(w_lead), eq_y), tail, "W", size=PT_EMPH,
                ha="left")
    f.note("soma-below", panel="C", below=n_below, segments=len(xy),
           reason="see panel A")
    f.require_soma_lowest()
    f.require_delta0()
    return dict(field=list(map(float, field)), scale=float(scale))


# ── row 1 ────────────────────────────────────────────────────────────────
#: D and E share one fraction axis: identical points-per-unit (plan check 8).
Y_TOP = 1.34
Y_TICKS = [0.0, 0.25, 0.5, 0.75, 1.0]
D_XMAX = 108.0             # 16 -> 87 reserves the direct-label band at the right


def _badge(ax, x, y, kind, *, ha="left", va="bottom"):
    """``Frame.badge`` for a DATA axes (constructing a Frame resets limits)."""
    key, face, edge = BADGE_STYLE[kind]
    return ax.text(x, y, kind, fontsize=PT_BASE, color=COLORS[key], ha=ha,
                   va=va, zorder=7,
                   bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.28",
                             facecolor=face, edgecolor=edge, linewidth=LW_HAIR))


def _leader(ax, p0, p1, color=None):
    """A hairline leader drawn as data, so no text sits on its own arrow."""
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
            color=COLORS["mute"] if color is None else color, lw=LW_HAIR,
            solid_capstyle="butt", zorder=1.6, clip_on=False)


def panel_d(ax, summaries, tables, floor):
    """D: total capture against the column budget K, six families."""
    table = summaries["v661"]
    cells = tables["v661"]
    rows = []
    for index, method in enumerate(ORDER):
        spec = FAMILIES[method]
        colour = COLORS[spec["color"]]
        part = table[table.method.eq(method)].sort_values("channels")
        x = part.channels.to_numpy(float)
        mean = part.total_capture_mean.to_numpy(float)
        lo = hi = [np.nan] * len(x)
        if method in (ANCESTRY, SURROGATE):
            lo, hi = [], []
            for k in x:
                subset = cells[cells.channels.eq(int(k))
                               & cells.method.eq(method)]
                _, a, b = mean_ci(subset.total_capture.to_numpy(float),
                                  BOOT_SEED + 10 * index + int(k))
                lo.append(a)
                hi.append(b)
            ax.fill_between(x, lo, hi, color=colour, alpha=0.11, linewidth=0,
                            zorder=1.2)
        ax.plot(x, mean, color=colour, lw=LW_DATA, marker=spec["marker"],
                ms=MARKER_MS - 0.8, mfc=colour, mec="white", mew=LW_HAIR,
                zorder=2 + 0.01 * index, solid_capstyle="round")
        for k, m, a, b in zip(x, mean, lo, hi):
            rows.append(dict(panel="D", method=method, series=spec["label"],
                             channels=int(k), total_capture_mean=float(m),
                             ci95_low=float(a), ci95_high=float(b),
                             n_cells=int(part[part.channels.eq(k)]
                                         .total_capture_count.iloc[0])))
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.85, D_XMAX)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(0.0, Y_TOP)
    ax.set_yticks(Y_TICKS)
    ax.set_xlabel("Profiles K (log 2)")
    ax.set_ylabel("Total field energy\ncaptured")
    style_panel(ax, grid="y")
    # direct labels, de-collided, in the reserved band at the right (CF-5)
    ends = sorted(((float(table[table.method.eq(m)
                                & table.channels.eq(16)]
                          .total_capture_mean.iloc[0]), m) for m in ORDER),
                  reverse=True)
    step = 0.098
    placed, last = [], None
    for value, method in ends:
        y = value if last is None else min(value, last - step)
        placed.append((y, value, method))
        last = y
    for y, value, method in placed:
        spec = FAMILIES[method]
        _leader(ax, (16.6, value), (19.8, y), COLORS["mute"])
        ax.text(21.0, y, spec["short"], fontsize=PT_BASE,
                color=COLORS[spec["color"]], ha="left", va="center",
                zorder=5)
    reference_line(ax, floor, axis="y", label="broadcast",
                   span=(0.85, D_XMAX))
    ax.annotate(f"{floor:.3f}", xy=(D_XMAX, floor), xytext=(0.0, -1.6),
                textcoords="offset points", fontsize=PT_BASE,
                color=COLORS["mute"], ha="right", va="top")
    ax.plot([8, 8], [0.0, 1.0], color=COLORS["edge"], lw=LW_REF,
            dashes=(2.2, 1.8), zorder=0.6, solid_capstyle="butt")
    ax.text(8.0, 1.010, "analysed budget", fontsize=PT_BASE,
            color=COLORS["mute"], ha="center", va="bottom")
    ax.text(0.02, 0.995, "bands: unpaired 95 % cell\nbootstrap; paired test in G",
            transform=ax.transAxes, fontsize=PT_BASE, color=COLORS["mute"],
            ha="left", va="top", linespacing=1.2)
    ax.text(0.02, 0.012, "47 disjoint cells (46 at K = 16)",
            transform=ax.transAxes, fontsize=PT_BASE, color=COLORS["mute"],
            ha="left", va="bottom")
    return pd.DataFrame(rows)


def panel_e(ax, summaries, initial, floor):
    """E: where the energy goes at K = 8 -- broadcast, spatial, unexplained."""
    eight = summaries["v661"][summaries["v661"].channels.eq(8)] \
        .set_index("method")
    rows = []
    for index, method in enumerate(ORDER):
        spec = FAMILIES[method]
        spatial = float(eight.loc[method, "incremental_total_capture_mean"])
        total = float(eight.loc[method, "total_capture_mean"])
        common = total - spatial
        ax.bar(index, common, width=0.62, color=COLORS[BROADCAST],
               edgecolor="white", lw=LW_HAIR, zorder=2)
        ax.bar(index, spatial, bottom=common, width=0.62,
               color=COLORS[spec["color"]], edgecolor="white", lw=LW_HAIR,
               zorder=2)
        ax.bar(index, 1.0 - total, bottom=total, width=0.62,
               color=COLORS["grid"], edgecolor="white", lw=LW_HAIR, zorder=2)
        rows.append(dict(panel="E", method=method, series=spec["label"],
                         broadcast=common, spatial=spatial,
                         unexplained=1.0 - total, n_cells=47))
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(0.0, Y_TOP)
    ax.set_yticks(Y_TICKS)
    style_panel(ax, grid="y")
    # family names on two levels, each with its marker glyph (CF-5): six
    # names cannot share one baseline in a 4-module panel at the 7 pt floor
    evens = [i for i in range(6) if i % 2 == 0]
    odds = [i for i in range(6) if i % 2 == 1]
    ax.set_xticks(evens, [FAMILIES[ORDER[i]]["short"] for i in evens])
    ax.set_xticks(odds, [FAMILIES[ORDER[i]]["short"] for i in odds],
                  minor=True)
    ax.tick_params(axis="x", which="major", length=0, pad=6.0,
                   labelsize=PT_BASE)
    ax.tick_params(axis="x", which="minor", length=0, pad=19.0,
                   labelsize=PT_BASE)
    for index, method in enumerate(ORDER):
        spec = FAMILIES[method]
        drop = -0.035 if index % 2 == 0 else -0.188
        ax.plot([index], [drop], marker=spec["marker"], ms=MARKER_MS - 0.8,
                mfc=COLORS[spec["color"]], mec="white", mew=LW_HAIR,
                ls="none", clip_on=False, zorder=3)
    # the three shares, named once, in the reserved gutter at the band heights
    # of the ancestry bar; these names carry the axis meaning, so E has no
    # separate y label (CF-5: a name, not a key).
    total_a = float(eight.loc[ANCESTRY, "total_capture_mean"])
    spatial_a = float(eight.loc[ANCESTRY, "incremental_total_capture_mean"])
    marks = [(f"broadcast\n{floor:.3f}", (total_a - spatial_a) / 2.0),
             ("spatial", total_a - spatial_a / 2.0),
             ("unexplained", (1.0 + total_a) / 2.0)]
    for name, y in marks:
        ax.annotate(name, xy=(0.0, y), xycoords=("axes fraction", "data"),
                    xytext=(-22.0, 0.0), textcoords="offset points",
                    fontsize=PT_BASE, color=COLORS["mute"], ha="right",
                    va="center", annotation_clip=False, linespacing=1.2)
    # No `oracle` badge here (plan section 4 E): the only free paper in E is
    # the 21 pt strip above the bars, and it carries the initial-cohort
    # annotation and this panel's n / endpoint line, which a data panel needs
    # more than a badge.  The sixth bar is named `SVD` under its diamond and
    # the badge is drawn in H.  Declared deviation.
    width = ax.get_position().width * 518.4 - 2.0
    ax.text(0.02, 0.995, wrap_pt(
        ax, "initial eight-cell cohort: broadcast "
            f"{initial['common']:.3f}, ceiling {initial['rank_one']:.3f}; "
            f"n = {int(rows[0]['n_cells'])} cells at K = 8",
        PT_BASE, width),
        transform=ax.transAxes, fontsize=PT_BASE, color=COLORS["mute"],
        ha="left", va="top", linespacing=1.2)
    return pd.DataFrame(rows)


def panel_f(ax, pairs):
    """F: each cell's ancestry capture against its own 200 surrogate trees."""
    tie = pairs.fraction_ge >= 0.5
    above = int((pairs.tree > pairs.surrogate_mean).sum())
    width = ax.get_position().width * 518.4 - 4.0
    block = wrap_pt(ax, f"{above} of {len(pairs)} cells above equality; open, "
                    f"the {int(tie.sum())} cells where at least half of the "
                    "200 surrogates reach the tree", PT_BASE, width)
    lines = 1 + block.count("\n") + 1          # + the 'equal' line
    h_pt = ax.get_position().height * 490.0
    head_pt = 8.8 * lines
    ymax = 1.0 + 0.75 * head_pt / max(h_pt - head_pt, 1.0)
    ax.plot([0.25, 1.0], [0.25, 1.0], color=COLORS["mute"], lw=LW_REF,
            dashes=(2.2, 1.8), zorder=1, solid_capstyle="butt")
    ax.text(1.0, 1.008, "equal", fontsize=PT_BASE, color=COLORS["mute"],
            ha="right", va="bottom")
    ax.plot(pairs.surrogate_mean[~tie], pairs.tree[~tie], marker="o",
            ls="none", ms=4.2, mfc=COLORS["shunting"], mec="none",
            alpha=0.75, zorder=2)
    ax.plot(pairs.surrogate_mean[tie], pairs.tree[tie], marker="o", ls="none",
            ms=4.2, mfc="white", mec=COLORS["shunting"], mew=LW_EDGE,
            zorder=3)
    mx, mlo, mhi = mean_ci(pairs.surrogate_mean.to_numpy(float), BOOT_SEED + 1)
    my, ylo, yhi = mean_ci(pairs.tree.to_numpy(float), BOOT_SEED + 2)
    ax.errorbar([mx], [my], xerr=[[mx - mlo], [mhi - mx]],
                yerr=[[my - ylo], [yhi - my]], fmt="D",
                color=COLORS["shunting"], mfc="white",
                mec=COLORS["shunting"], mew=LW_ERR, ms=MARKER_MS,
                elinewidth=LW_ERR, capsize=2.0, zorder=4)
    _leader(ax, (mx + 0.020, my - 0.020), (0.628, 0.425))
    ax.text(0.998, 0.425, "cohort mean,\n95 % CI", fontsize=PT_BASE,
            color=COLORS["ink"], ha="right", va="center", linespacing=1.2)
    ax.text(0.25, ymax * 0.999, block, fontsize=PT_BASE, color=COLORS["mute"],
            ha="left", va="top", linespacing=1.2)
    ax.text(0.995, 0.262, "cell is the unit; n = 47",
            fontsize=PT_BASE, color=COLORS["mute"], ha="right", va="bottom")
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0.25, ymax)
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("200-surrogate mean at K = 8")
    ax.set_ylabel("Ancestry capture")
    style_panel(ax)
    return pd.DataFrame(dict(panel="F", root_id=pairs.root_id,
                             ancestry_total_capture=pairs.tree,
                             surrogate_mean=pairs.surrogate_mean,
                             fraction_surrogates_ge=pairs.fraction_ge,
                             n_replicates=pairs.n_replicates))


# ── row 2 ────────────────────────────────────────────────────────────────
G_XLIM = (-25.0, 88.0)      # data to +55, then the three printed columns
G_FAN_MAX = 55.0
G_COLS = ((74.0, "wiring"), (87.0, "rank"))
G_RIGHT = 20.0            # keeps the 7-module panel inside the aspect band


def panel_g(canvas, ax, report, tables, summaries):
    """G: the paired advantage on both scales, with its wiring cost."""
    cells = tables["v661"][tables["v661"].channels.eq(8)]
    eight = summaries["v661"][summaries["v661"].channels.eq(8)] \
        .set_index("method")
    pivot = cells.pivot_table(index="root_id", columns="method",
                              values="residual_capture")
    rows, second, extra = [], [], []
    for method in CONTROLS:
        spec = FAMILIES[method]
        res = next(c for c in report["comparisons"]
                   if c["metric"] == "residual_capture"
                   and c["control"] == method)
        tot = next(c for c in report["comparisons"]
                   if c["metric"] == "total_capture"
                   and c["control"] == method)
        seeds = 100.0 * (pivot[ANCESTRY] - pivot[method]).to_numpy(float)
        rows.append(dict(label=spec["label"].replace(" ", "\n"),
                         mean=100.0 * res["mean_difference"],
                         lo=100.0 * res["ci95"][0], hi=100.0 * res["ci95"][1],
                         color="shunting", marker="o", n=int(res["n_cells"]),
                         note=f"{int(res['cells_positive'])}/"
                              f"{int(res['n_cells'])}"))
        second.append((100.0 * tot["mean_difference"],
                       100.0 * tot["ci95"][0], 100.0 * tot["ci95"][1]))
        extra.append(dict(
            seeds=seeds, positive=int(res["cells_positive"]),
            n=int(res["n_cells"]),
            wiring=100.0 * float(eight.loc[method, "wiring_density_mean"]),
            rank=float(eight.loc[method, "dictionary_rank_mean"])))
    out = canvas.forest(
        ax, rows, value_label="Ancestry advantage (percentage points)",
        reference=None, band=True, tick=True,
        tag="", xlim=G_XLIM, color="shunting", marker_size=MARKER_MS,
        gutter_pt=GUTTER_PT)
    ypos = out["ypos"]
    ax.set_ylim(3.62, -1.42)            # strips for the sub-title and headers
    # forest() spans its 6 % row band across the whole x range; the three
    # printed columns then sit ON an area mark, which the overlap audit
    # reports as TEXT-ON-DATA (eight findings).  The band is a row cue, not a
    # datum, so it is trimmed to the fan window and the columns keep white
    # paper.  Declared deviation from CF-6's band, recorded in the docstring.
    for patch in ax.patches:
        if abs(patch.get_width() - (G_XLIM[1] - G_XLIM[0])) < 1e-6:
            patch.set_bounds(patch.get_x(), patch.get_y(),
                             G_FAN_MAX + 1.5 - patch.get_x(),
                             patch.get_height())
    rng = np.random.default_rng(FAN_SEED)
    beyond = 0
    for i, item in enumerate(extra):
        jitter = rng.uniform(-0.14, 0.14, size=item["seeds"].size)
        inside = (item["seeds"] >= G_XLIM[0]) & (item["seeds"] <= G_FAN_MAX)
        beyond += int((~inside).sum())
        ax.plot(item["seeds"][inside], ypos[i] - 0.22 + jitter[inside],
                ls="none", marker="o", ms=SEED_MS, mfc=COLORS["shunting"],
                mec="none", alpha=0.30, zorder=2.0, clip_on=True)
    for i, (mean, lo, hi) in enumerate(second):
        y = ypos[i] + 0.22
        ax.plot([lo, hi], [y, y], color=COLORS["shunting"], lw=LW_ERR,
                zorder=3.0, solid_capstyle="butt")
        for bound in (lo, hi):
            ax.plot([bound, bound], [y - 0.09, y + 0.09],
                    color=COLORS["shunting"], lw=LW_ERR, zorder=3.0,
                    solid_capstyle="butt")
        ax.plot([mean], [y], ls="none", marker="o", ms=MARKER_MS,
                mfc="white", mec=COLORS["shunting"], mew=LW_ERR, zorder=4.0)
    zero, = ax.plot([0.0, 0.0], [-0.46, 3.46], color=COLORS["mute"],
                    lw=LW_REF, zorder=1.0, solid_capstyle="butt")
    zero.set_dashes((2.6, 2.0))
    ax.text(1.6, -0.70, "no advantage", fontsize=PT_BASE,
            color=COLORS["mute"], ha="left", va="center", zorder=5)
    # the three printed columns, inside the axes, clear of every interval
    ax.plot([G_FAN_MAX + 1.5, G_FAN_MAX + 1.5], [-0.60, 3.45],
            color=COLORS["grid"], lw=LW_HAIR, zorder=0.5)
    for x, head in G_COLS:
        ax.text(x, -0.70, head, fontsize=PT_BASE, color=COLORS["ink"],
                ha="right", va="center", zorder=5)
    for i, item in enumerate(extra):
        dagger = "\u2020" if CONTROLS[i] == RANDOM else ""
        for x, value in zip([c[0] for c in G_COLS],
                            (f"{item['wiring']:.1f} %",
                             f"{item['rank']:.2f}{dagger}")):
            ax.text(x, ypos[i], value, fontsize=PT_BASE, color=COLORS["ink"],
                    ha="right", va="center", zorder=5)
    ax.set_xticks([-20, 0, 20, 40])
    ancestry = eight.loc[ANCESTRY]
    ax.text(G_XLIM[0] + 1.0, -1.20,
            f"ancestry: {100 * float(ancestry['wiring_density_mean']):.1f} % "
            f"of dense wiring, rank "
            f"{float(ancestry['dictionary_rank_mean']):.2f}",
            fontsize=PT_BASE, color=COLORS["mute"], ha="left", va="center",
            zorder=5)
    nonzero = float(eight.loc[ANCESTRY, "nonzero_coefficients_mean"])
    assert abs(nonzero
               - float(eight.loc[SHUFFLED, "nonzero_coefficients_mean"])) < 1e-6
    below = [
        "filled: post-broadcast scale; open: total-energy scale",
        f"ancestry and shuffled routes: identical {nonzero:.1f} nonzero "
        "entries",
        f"† rank-limited (5.66 of 8); {beyond} of "
        f"{sum(len(i['seeds']) for i in extra)} differences beyond the axis",
    ]
    for row, line in enumerate(below):
        ax.annotate(line, xy=(0.0, 0.0), xycoords="axes fraction",
                    xytext=(0.0, -19.4 - 8.2 * row), textcoords="offset points",
                    fontsize=PT_BASE, color=COLORS["mute"], ha="left",
                    va="top")
    ax.annotate("n = 47 cells per row; mean [95 % cell bootstrap, "
                "20,000 draws]; K = 8", xy=(1.0, 0.0),
                xycoords="axes fraction", xytext=(0.0, -44.0),
                textcoords="offset points", fontsize=PT_BASE,
                color=COLORS["mute"], ha="right", va="top")
    frame = []
    for i, method in enumerate(CONTROLS):
        frame.append(dict(panel="G", control=method,
                          series=FAMILIES[method]["label"],
                          residual_pp=rows[i]["mean"],
                          residual_lo=rows[i]["lo"], residual_hi=rows[i]["hi"],
                          total_pp=second[i][0], total_lo=second[i][1],
                          total_hi=second[i][2],
                          cells_positive=extra[i]["positive"],
                          n_cells=extra[i]["n"], wiring_pct=extra[i]["wiring"],
                          dictionary_rank=extra[i]["rank"]))
    return pd.DataFrame(frame), out


def panel_h(ax, tables, inclusion):
    """H: residual capture at K = 8, four families in three cohorts."""
    offsets = [-0.24, -0.08, 0.08, 0.24]
    rows, pinky = [], {}
    for group, cohort in enumerate(COHORTS):
        eight = tables[cohort][tables[cohort].channels.eq(8)]
        for index, method in enumerate(COHORT_FAMILIES):
            spec = FAMILIES[method]
            values = eight.loc[eight.method.eq(method),
                               "residual_capture"].to_numpy(float)
            mean, lo, hi = mean_ci(values, BOOT_SEED + 100 * group + index)
            x = group + offsets[index]
            ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]],
                        fmt=spec["marker"], color=COLORS[spec["color"]],
                        mfc=COLORS[spec["color"]], mec="white", mew=LW_HAIR,
                        ms=MARKER_MS - 0.8, elinewidth=LW_ERR, capsize=1.8,
                        zorder=3)
            rows.append(dict(panel="H", cohort=cohort, method=method,
                             series=spec["label"], residual_capture=mean,
                             ci95_low=lo, ci95_high=hi,
                             n_cells=int(values.size)))
            if cohort == "pinky":
                pinky[method] = (x, mean)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0.0, 1.16)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticks(range(len(COHORTS)), [COHORT_PANEL[c] for c in COHORTS])
    ax.set_ylabel("Residual capture\nafter the broadcast")
    style_panel(ax, grid="y")
    ax.plot([-0.55, 3.55], [1.0, 1.0], color=COLORS["mute"], lw=LW_REF,
            zorder=1.0, solid_capstyle="butt")
    ax.text(3.53, 1.012, "ceiling", fontsize=PT_BASE, color=COLORS["mute"],
            ha="right", va="bottom")
    ax.text(-0.50, 1.155, "mean [95 % cell bootstrap]; K = 8", fontsize=PT_BASE,
            color=COLORS["mute"], ha="left", va="top")
    # the four family names, once, beside the Pinky group (CF-5)
    order = sorted(pinky, key=lambda m: -pinky[m][1])
    step, last = 0.125, 0.930
    for method in order:
        x, mean = pinky[method]
        y = min(mean, last)
        last = y - step
        _leader(ax, (x + 0.06, mean), (2.62, y))
        ax.annotate(FAMILIES[method]["short"], xy=(1.0, y),
                    xycoords=("axes fraction", "data"), xytext=(10.0, 0.0),
                    textcoords="offset points", fontsize=PT_BASE,
                    color=COLORS[FAMILIES[method]["color"]], ha="right",
                    va="center", annotation_clip=False)
    eligible = inclusion[inclusion.focus_budget_eligible]
    lines = [
        f"Pinky: {int(eligible.n_e_sites.min())}–"
        f"{int(eligible.n_e_sites.max())} sites per cell, "
        f"{len(eligible)} of {int(inclusion.inherited_qc_included.sum())} "
        "cells",
        "one mouse per cohort; n = 8, 47, 8 cells",
        "random routes outrank depth bins in the initial and Pinky "
        "cohorts (D)",
    ]
    width = ax.get_position().width * 518.4
    block = "\n".join(wrap_pt(ax, line, PT_BASE, width) for line in lines)
    ax.annotate(block, xy=(0.0, 0.0), xycoords="axes fraction",
                xytext=(0.0, -21.0), textcoords="offset points",
                fontsize=PT_BASE, color=COLORS["mute"], ha="left", va="top",
                linespacing=1.2)
    oracle_mean = float(np.mean(
        tables["original8"].loc[tables["original8"].channels.eq(8)
                                & tables["original8"].method.eq(ORACLE),
                                "residual_capture"]))
    _badge(ax, 0.24, oracle_mean + 0.13, "oracle", ha="center")
    return pd.DataFrame(rows)


# ── the canvas ───────────────────────────────────────────────────────────
def figure7(*, out=COMPONENT, png=True, dpi=200, quiet=False):
    """Build main Figure 7 and write its component, main copy and record."""
    tables = cohort_tables()
    summaries = cohort_summaries()
    report = reports()
    inclusion = pd.read_csv(COMMON / "pinky" / "cohort_inclusion.csv")
    arb = arbor_routes()
    field, field_scale = block_field(arb)
    pairs = surrogate_pairs()
    assert int(pairs.n_replicates.min()) == 200 and len(pairs) == 47
    # decision 0.2: the family map is keyed by the literal method strings
    assert set(FAMILIES) == set(tables["v661"].method.unique())
    assert FAMILIES[DEPTH]["label"] == "Depth bins"
    assert FAMILIES[ORACLE]["label"] == "SVD oracle"
    floor = float(summaries["v661"]
                  .loc[summaries["v661"].channels.eq(1),
                       "total_capture_mean"].unique()[0])
    initial = dict(common=float(report["original8"]["common_capture_mean"]),
                   rank_one=float(report["original8"]["rank_one_capture_mean"]))

    canvas = NativeCanvas(
        CANVAS_H_PT / 72.0, 3, row_weights=ROW_H, hgutter_pt=HGUT,
        vgutter_pt=VGUT, margins=Margins(**MARGINS))
    a = canvas.panel("A", 0, 0, 4, schematic=True, lock=False,
                     inset_pt=SCHEMATIC_INSET, title="Routes on a real arbor")
    b = canvas.panel("B", 0, 4, 4, schematic=True, lock=False,
                     inset_pt=SCHEMATIC_INSET, title="One arbor, seven routes")
    c = canvas.panel("C", 0, 8, 4, schematic=True, lock=False,
                     inset_pt=SCHEMATIC_INSET, title="A shunt makes the field")
    d = canvas.panel("D", 1, 0, 4, title="Capture rises with budget")
    e = canvas.panel("E", 1, 4, 4, title="Where the energy goes")
    fx = canvas.panel("F", 1, 8, 4, title="Cell by cell")
    g = canvas.panel("G", 2, 0, 7,
                     title="Paired advantage and its wiring cost")
    h = canvas.panel("H", 2, 7, 5, title="Three cohorts, one ordering")
    for name in ("D", "E", "F"):
        canvas.declare_reserve(name, left=GUTTER_PT, bottom=BOTTOM_R1)
    canvas.declare_reserve("G", left=GUTTER_PT, right=G_RIGHT,
                           bottom=BOTTOM_R2)
    canvas.declare_reserve("H", left=20.0, bottom=BOTTOM_R2)
    # place the panels once before drawing: every wrap width and every Frame
    # geometry below is measured from the FINAL axes box
    canvas.lock_reserves()

    schem_a = panel_a(a, arb)
    schem_b = panel_b(b, arb)
    schem_c = panel_c(c, arb, field, field_scale)
    rows_d = panel_d(d, summaries, tables, floor)
    rows_e = panel_e(e, summaries, initial, floor)
    rows_f = panel_f(fx, pairs)
    rows_g, forest_out = panel_g(canvas, g, report["v661"], tables, summaries)
    rows_h = panel_h(h, tables, inclusion)

    canvas.lock_reserves()
    findings = canvas.align_letters()
    problems = canvas.save(Path(out), name="credit_first_figure_07", png=png,
                           dpi=dpi, quiet=quiet, lock=False)
    layout = list(findings) + list(problems)
    MAIN.parent.mkdir(parents=True, exist_ok=True)
    MAIN.write_bytes(Path(out).read_bytes())

    # ── the plotted table and the provenance record ─────────────────────
    RECORDS.mkdir(parents=True, exist_ok=True)
    rows_b = pd.DataFrame([
        dict(panel="B", block=i + 1, route_columns="+".join(
            str(k + 1) for k in key) or "broadcast only",
             site_count=int(size), field_t=float(value))
        for i, (key, size, value) in enumerate(
            zip(arb["block_keys"], arb["block_sizes"], field))])
    rows_c = pd.DataFrame([
        dict(panel="C", route=k + 1, origin_segment=int(origin),
             support_sites=int(support.sum()))
        for k, (origin, support) in enumerate(zip(arb["origins"],
                                                  arb["supports"]))])
    plotted = pd.concat([rows_b, rows_c, rows_d, rows_e, rows_f, rows_g,
                         rows_h], ignore_index=True)
    plotted.to_csv(RECORDS / "figure_07_plotted.csv", index=False)
    palette = palette_report(
        series={FAMILIES[m]["label"]: COLORS[FAMILIES[m]["color"]]
                for m in ORDER},
        anatomy={FAMILIES[m]["label"]: COLORS[FAMILIES[m]["color"]]
                 for m in ORDER})
    worst_normal = min((r for r in palette["rows"]
                        if r["series"] != r["anatomy"]),
                       key=lambda r: r["normal"])
    worst_cvd = min((r for r in palette["rows"]
                     if r["series"] != r["anatomy"]),
                    key=lambda r: r["cvd"])
    files = [Path(__file__), Path(anatomy.__file__),
             JOURNAL / "scripts/figure_canvas.py",
             JOURNAL / "scripts/journal_style.py",
             JOURNAL / "scripts/native_schematics.py",
             JOURNAL / "scripts/anatomy_commonmode/run.py",
             JOURNAL / "scripts/anatomy_commonmode/protocol.json",
             JOURNAL / "scripts/analyze_reciprocal_routing_controls.py",
             SOURCE / "figure3/segment_metrics.csv",
             COMMON / "protocol_freeze.json",
             COMMON / "original8/cells" / f"operator_{ROOT_B}.npz",
             COMMON / "pinky/cohort_inclusion.csv"]
    files += [COMMON / cohort / name for cohort in COHORTS
              for name in ("cell_method_summary.csv",
                           "cohort_method_summary.csv", "summary.json",
                           "operator_audit.csv")]
    panels = {
        "a": ("source_data/figure3/segment_metrics.csv filtered to root "
              f"{ROOT_B} (78 of 616 segments): the pia-up principal-plane "
              "projection, all "
              f"{schem_a['inhibitory']} inhibitory-bearing segments as inh "
              "contacts, route 3's origin 4396 and its five input-bearing "
              "descendants under a 16 % shunting tint; schematic, no data"),
        "b": ("the same skeleton; the seven K = 8 routes reproduced in the "
              "builder from scripts/anatomy_commonmode/run.py:105-112 over "
              f"source_data/anatomy_commonmode/original8/cells/operator_{ROOT_B}"
              ".npz with scripts/anatomy_commonmode/protocol.json (origins "
              "4784, 4621, 4396, 4458, 4975, 4209, 4516; supports 1, 1, 5, 1, "
              "1, 29, 1; coverage 32 of 70); the matrix is the eight "
              f"tree-ordered site blocks {schem_b['block_sizes']}"),
        "c": ("weighted_response column of i-site 4396 in the same operator "
              "npz, reduced to the eight site blocks of B by a block mean and "
              f"scaled by max |t| = {schem_c['scale']:.5f}: "
              f"{[round(v, 3) for v in schem_c['field']]}"),
        "d": ("v661 cohort_method_summary.csv total_capture_mean at K = 1, 2, "
              "4, 8, 16 for the six families; ancestry and surrogate bands "
              "are 20,000-draw cell bootstraps of cell_method_summary.csv "
              f"(seed {BOOT_SEED} + 10i + K); broadcast floor {floor:.5f}"),
        "e": ("v661 cohort_method_summary.csv at K = 8: broadcast = "
              "total_capture_mean - incremental_total_capture_mean, spatial = "
              "incremental_total_capture_mean, unexplained = 1 - "
              "total_capture_mean; the annotation is original8/summary.json "
              f"common_capture_mean {initial['common']:.5f} and "
              f"rank_one_capture_mean {initial['rank_one']:.5f}"),
        "f": ("per-cell K = 8 total capture of common + ancestry against the "
              "mean of the cell's 200 common + surrogate ancestry replicates "
              "from v661/cells/rows_<root_id>.csv.gz; open symbols are the "
              "cells whose surrogate fraction >= 0.5; cohort mean with "
              "20,000-draw cell-bootstrap intervals on both coordinates"),
        "g": ("v661/summary.json comparisons for BOTH metrics (residual "
              "filled, total open) with their retained 95 % intervals and "
              "cells_positive; the fan is the 47 within-cell residual "
              "differences from cell_method_summary.csv at K = 8; the right "
              "columns are wiring_density_mean and dictionary_rank_mean from "
              "cohort_method_summary.csv"),
        "h": ("residual_capture at K = 8 per cohort from each cohort's "
              "cell_method_summary.csv (initial 8, disjoint 47, Pinky 8) for "
              "ancestry, surrogate tree, depth bins and the "
              f"common-constrained SVD; 20,000-draw cell bootstrap, seed "
              f"{BOOT_SEED} + 100 g + i; Pinky site counts and eligibility "
              "from pinky/cohort_inclusion.csv"),
    }
    payload = dict(
        figure="Figure 7",
        label="fig:topology",
        output=str(Path(out).relative_to(JOURNAL)),
        output_sha256=sha(out),
        main_copy=str(MAIN.relative_to(JOURNAL)),
        panel_letters="abcdefgh",
        replication_unit=("reconstructed cell; initial eight-cell, disjoint "
                          "47-cell and Pinky second-mouse cohorts kept "
                          "separate"),
        panel_sources=panels,
        source_sha256={str(p.relative_to(JOURNAL)): sha(p) for p in files},
        schematic_fraction=round(SCHEMATIC_FRACTION, 5),
        schematic_fraction_formula=("sum(schematic slot w x h) / (live_w x "
                                    "live_h) = 3 x (125.5 x 122) / (452.4 x "
                                    "430)"),
        palette=dict(
            families={FAMILIES[m]["label"]: COLORS[FAMILIES[m]["color"]]
                      for m in ORDER},
            worst_normal=dict(pair=[worst_normal["series"],
                                    worst_normal["anatomy"]],
                              delta_e=round(worst_normal["normal"], 2)),
            worst_cvd=dict(pair=[worst_cvd["series"], worst_cvd["anatomy"]],
                           delta_e=round(worst_cvd["cvd"], 2)),
            waiver=("shunting/low_rank protan 7.20 is accepted: forced once "
                    "bp, additive, amber and per_soma are excluded, and the "
                    "pair is separated by marker (o vs X), by D's band and by "
                    "direct labels; the AMENDMENTS fallback "
                    "random = mix('point_mlp', 55) is rejected because it "
                    "puts a grey tint beside the grey point_mlp series"),
            forbidden_present=[key for key in ("bp", "additive")
                               if key in [FAMILIES[m]["color"]
                                          for m in ORDER]]),
        waivers=[
            "D3: row 1 is three 4-module panels that share no axis; each is a "
            "different estimand and the row is column-locked",
            "H idiom (decision 0.7): grouped vertical dot plot, not a forest; "
            "a two-factor panel at 5 modules",
            "B address cycle: builder-local shunting/local/oracle plus white "
            "tints, because K_CYCLE's second entry is the banned `additive`",
            "soma-lowest: declared, not asserted, for the measured arbor of "
            f"A/B/C ({schem_a['below_soma']} of {len(arb['cell'])} segments "
            "sit below the soma in any projection of this reconstruction)",
            "G band: forest()'s 6 % row band is trimmed to the fan window so "
            "the printed wiring / rank columns sit on paper, not on an area "
            "mark (eight TEXT-ON-DATA findings otherwise)",
            "E badge: no `oracle` badge; the strip above E's bars carries the "
            "initial-cohort annotation and the panel's n and endpoint, and H "
            "carries the badge",
            "right overhang: H's four family names are set 10 pt beyond their "
            "spine because a 452.4 pt live area cannot meet FILL_W_MIN 92 % "
            "of 518.4 pt on its own",
        ],
        layout_findings=layout,
        scope=("Modeled passive response capacity on measured anatomy; no "
               "evidence of endogenous biological route usage. The initial "
               "eight-cell and disjoint 47-cell cohorts are the same animal; "
               "Pinky is one second animal."))
    (RECORDS / "figure_07.json").write_text(json.dumps(payload, indent=2)
                                            + "\n")
    # the caption travels with the artwork so main.tex and the record cannot
    # drift; the authoritative copy is v2/fig7/TEXT.md
    (RECORDS / "figure_07_caption.md").write_text(CAPTION + "\n")
    if not quiet:
        print(json.dumps({"layout": layout,
                          "schematic_fraction": round(SCHEMATIC_FRACTION, 4),
                          "worst_normal": round(worst_normal["normal"], 2),
                          "worst_cvd": round(worst_cvd["cvd"], 2)}, indent=2))
    return layout


def build(out=COMPONENT, **kwargs):
    """Legacy entry point; Figure 7 is built by :func:`figure7`."""
    return figure7(out=out, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-main", action="store_true",
                        help="also copy the component to figures/main "
                             "(always done)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    figure7(quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
