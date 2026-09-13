#!/usr/bin/env python3
"""Supplementary sheet S31 (ident ``measured_transfer_geometry``) -- the
measured-response learning comparison is limited by mapped coverage and
transfer geometry -- rebuilt as ONE native full-width
:class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/measured_transfer_geometry.pdf``)
is a paste of three upstream renders: S22 panel A (``build_journal_figures.py``),
the frozen M9 snapshot ``figures/supplementary/frozen/credit_first_figure_08_M9.pdf``
(panels B, D, E; no generator draws them any more) and the frozen S56 render
(panel C; builder null).  This builder reads ONLY the frozen tables under
``source_data/`` and redraws the same five panels with the same plotted
quantities.  Nothing about the numbers changes; every printed or plotted value
is asserted against the table it comes from.  The means and 95 % intervals of
C and D are DRAWN from the frozen summary rows (``review_response_baselines/
condition_summary.csv`` and ``figure_08_prediction_summary.csv`` panel G;
``fulltree_within_span_oracle/condition_summary.csv`` and panel H); the
study's own target bootstrap is re-run only inside an assertion that those
rows are what the seven per-target rows give.  Its two procedural constants
are literals here (``BOOT_DRAWS = 20000``, ``SEED_C = 20260907`` -- the
``bootstrap_draws`` and ``analysis_seed`` of
``configs/review_completion/measured_response_baselines.json``, which this
builder does not read; ``SEED_D = 20260906``, the base seed of
``analyze_fulltree_within_span_oracle.py:summarize``): C uses one
``default_rng(SEED_C)`` stream over the methods in sorted order, D a fresh
``default_rng(SEED_D + index)`` per (method, mode) in sorted order, 20,000
draws each.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
mixed_S31.json), panel by panel:

* A keeps the earlier pass's two-column dot chart (mapped partners and the
  manually curated subset on one target axis; the 69 per-partner split-half
  reliabilities of the seven selected scans as a fan behind each target
  median), now in the neutral ink of a descriptive panel so that green means
  one thing on the sheet (the ancestry route dictionary).
* B is labelled with its scan (target 1, session 4, scan 10; nine mapped
  inputs, four routes), drawn as vector cells, and carries beside the matrix
  the 13-scan occupancy strip the review asked for (input coordinates per
  route from ``figure_08_support.csv``), so the sheet itself shows that 6 of
  13 supports place one coordinate per route and which scan the matrix is.
* C and D are one five-row ladder each on the same rule palette: exact
  compartment error dark red (``bp``), the ancestry (topology-matched) routes
  green (``shunting``, the local rule), the unrestricted fixed transfer
  profile blue (``additive``, the paper's fixed-profile hue), ridge / random /
  site-shuffled the one grey control series (``point_mlp``).  D now draws the
  two surrogate rows its own table holds (random anatomical 0.385, site-
  shuffled 0.468) that the frozen panel omitted; oracle-amplitude rows are
  diamonds with a dashed interval, fixed-profile rows circles with a solid
  one.  The unrestricted row's interval (0.964-0.986), narrower than the
  marker, is printed on its row instead of drawn as a stub.
* E is the residual calibration panel the review asked for: simulated mean
  minus measured split-half r against the measured value (x from 0), every
  record with its ±1.96 Monte Carlo SE whisker, the two negative records at
  their clipped value 0 as keyed hollow squares, one dashed zero rule, no
  in-plot caption sentence, and the 5-of-125 exceedance count printed with
  its denominator.
* One 12-module grid, one gutter, letters on the module columns.

Second pass (checker report mixed_wave_reports.json, S31):

* C, D draw the frozen summary rows; the bootstrap lives in an assertion
  with literal constants (no read outside ``source_data/``).
* E stacks the two clipped records at x = 0 exactly (no dodge; their
  residuals differ), and the key states that each record's mean and SE come
  from 1,000 simulated calibration datasets and prints the two raw measured
  values (-0.098 and -0.188).
* B's strip key hangs its glyphs in the gap left of the strip (x < 0.85,
  outside every data column and left of the x = 1 rule); gridlines of the A
  and B strips stop at the first data row so no key text sits on a rule.
* A's count axis starts at -0.9 and its split-half axis at -0.035 so the
  n = 1 marker and the 0.011 record clear the spine.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_BASE,
    PT_EMPH,
    SEED_ALPHA,
    SEED_MS,
    Margins,
    NativeCanvas,
    _text_width_pt,
    style_panel,
    tint_pct,
)

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data"
FIG5 = SOURCE / "figure5"
POWER = SOURCE / "measured_alignment_power"
CFF = SOURCE / "credit_first_figures"
BASELINES = SOURCE / "review_response_baselines"
ORACLE = SOURCE / "fulltree_within_span_oracle"
OUT = ROOT / "figures" / "supplementary" / "figure_measured_transfer_geometry_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]
ROUTE = COLORS["shunting"]      # the ancestry route dictionary (local rule)
EXACT = COLORS["bp"]            # exact compartment error
PROFILE = COLORS["additive"]    # the unrestricted fixed transfer profile
CONTROL = COLORS["point_mlp"]   # ridge, random anatomical, site-shuffled
DASHES = (2.6, 2.0)
N_TARGETS = 7
N_SCANS = 13
N_RECORDS = 125
N_SIMULATED = 1000
Z95 = 1.96
# The study bootstraps' procedural constants, as literals (see the docstring).
BOOT_DRAWS = 20000
SEED_C = 20260907       # measured_response_baselines.json: analysis_seed
SEED_D = 20260906       # analyze_fulltree_within_span_oracle.py:summarize base seed

# C: (table key, row label, colour)
ROWS_C = (
    ("archived_exact compartment error", "exact", EXACT),
    ("ridge_all", "ridge", CONTROL),
    ("archived_topology-matched routes", "ancestry", ROUTE),
    ("archived_random anatomical routes", "random", CONTROL),
    ("archived_site-shuffled routes", "shuffled", CONTROL),
)
# D: (method, mode, row label, colour, oracle amplitudes?)
ROWS_D = (
    ("unprojected baseline transfer", "frozen_baseline", "unrestricted\nfixed profile", PROFILE, False),
    ("topology-matched routes", "frozen_baseline", "ancestry\nfixed profile", ROUTE, False),
    ("topology-matched routes", "trialwise_update_oracle", "ancestry\noracle", ROUTE, True),
    ("random anatomical routes", "trialwise_update_oracle", "random\noracle", CONTROL, True),
    ("site-shuffled routes", "trialwise_update_oracle", "shuffled\noracle", CONTROL, True),
)
REPRESENTATIVE = dict(target_root_id=864691135810666525, session=4, scan_idx=10)


def csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def jitter(n, seed, scale):
    """``build_journal_figures.jitter`` verbatim (presentation only)."""
    return np.random.default_rng(seed).normal(0.0, scale, int(n))


# ── the two study bootstraps, verbatim (used ONLY inside assertions) ─────
def boot_draws(values, rng, draws=BOOT_DRAWS):
    values = np.asarray(values, float)
    m = values[rng.integers(0, len(values), (draws, len(values)))].mean(1)
    lo, hi = np.quantile(m, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def baselines_intervals(cell):
    """``analyze_review_response_baselines.py`` lines 111-113: one stream,
    methods in ``groupby`` (sorted) order."""
    rng = np.random.default_rng(SEED_C)
    out = {}
    for method, g in cell.groupby("method"):
        out[method] = boot_draws(g.nmse, rng, BOOT_DRAWS)
    return out


def oracle_intervals(cell):
    """``analyze_fulltree_within_span_oracle.py:summarize``: a fresh stream
    per (method, mode) in sorted order."""
    out = {}
    for index, ((method, mode), g) in enumerate(cell.groupby(["method", "mode"])):
        rng = np.random.default_rng(SEED_D + index)
        out[(method, mode)] = boot_draws(g.update_match, rng, BOOT_DRAWS)
    return out


def assert_bootstrap_reproduces(recomputed, drawn, key):
    """The frozen row that is drawn is what the study bootstrap gives."""
    np.testing.assert_allclose(recomputed[key], drawn, rtol=0, atol=1e-9,
                               err_msg=f"bootstrap does not reproduce {key}")


# ── drawing helpers (the forest vocabulary of the finite-horizon sheet) ──
def fan_h(ax, y, values, color, *, half=0.16, zorder=2.0):
    values = np.asarray(values, float)
    jit = np.linspace(-half, half, len(values)) if len(values) > 1 else np.zeros(1)
    ax.plot(values, y + jit, linestyle="none", marker="o", markersize=SEED_MS,
            markerfacecolor=color, markeredgecolor="none", alpha=SEED_ALPHA,
            zorder=zorder, clip_on=True)


def whisker_h(ax, y, lo, hi, color, *, dashed=False, zorder=3.0):
    kw = dict(color=color, lw=LW_ERR, zorder=zorder, solid_capstyle="butt")
    if dashed:
        ax.plot([lo, hi], [y, y], dashes=DASHES, **kw)
    else:
        ax.plot([lo, hi], [y, y], **kw)
    for xb in (lo, hi):
        ax.plot([xb, xb], [y - 0.13, y + 0.13], **kw)


def mean_marker(ax, x, y, color, marker, *, zorder=4.0, ms=MARKER_MS):
    ax.plot([x], [y], linestyle="none", marker=marker, markersize=ms,
            markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=LW_HAIR, zorder=zorder)


def row_label(ax, y, label, color, marker, *, glyph_dx=-4.5, text_dx=-9.5):
    """Row label in the left gutter, ink, with the series glyph beside it."""
    tr = transforms.offset_copy(ax.get_yaxis_transform(), fig=ax.figure,
                                x=glyph_dx, y=0.0, units="points")
    ax.plot([0.0], [y], transform=tr, linestyle="none", marker=marker,
            markersize=MARKER_MS * 0.8, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=LW_HAIR, clip_on=False,
            zorder=5.0)
    return ax.annotate(label, xy=(0.0, y), xycoords=("axes fraction", "data"),
                       xytext=(text_dx, 0.0), textcoords="offset points",
                       ha="right", va="center", fontsize=PT_BASE, color=INK,
                       linespacing=0.95, annotation_clip=False)


def row_bands(ax, colors, x0, x1):
    for i, color in enumerate(colors):
        ax.fill_between([x0, x1], i - 0.42, i + 0.42, facecolor=tint_pct(color, 6),
                        edgecolor="none", linewidth=0.0, zorder=0.2, clip_on=True)


def tag(ax, text, *, ha="right", x=1.0, dy=2.0):
    return ax.annotate(text, xy=(x, 1.0), xycoords="axes fraction", xytext=(0.0, dy),
                       textcoords="offset points", ha=ha, va="bottom", fontsize=PT_BASE,
                       color=MUTE, annotation_clip=False)


def forest_frame(ax, n_rows, *, headroom=0.0):
    ax.set_ylim(n_rows - 0.4, -0.6 - headroom)
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)


def data_grid(ax, xs, y_lo, y_hi):
    """Vertical gridlines that span only the data rows, so the keys in the
    headroom sit on no rule."""
    ax.vlines(list(xs), y_lo, y_hi, colors=COLORS["grid"], lw=LW_HAIR, alpha=0.9,
              zorder=0, capstyle="butt", clip_on=True)


def sub_axes(cv, host, rect_fracs):
    """Satellite axes inside a host panel's box; the host only carries the
    title and the letter.  ``rect_fracs`` are (x0, w) fractions of the box."""
    box = host.get_position()
    host.set_axis_off()
    host.patch.set_visible(False)
    out = []
    for fx0, fw in rect_fracs:
        ax = cv.fig.add_axes([box.x0 + fx0 * box.width, box.y0, fw * box.width, box.height])
        style_panel(ax)
        cv.bind_satellite(ax, host)
        out.append(ax)
    return out


# ── A: the measured cohort ───────────────────────────────────────────────
def load_cohort():
    functional = csv(FIG5 / "functional_target_metrics.csv")
    audit = csv(POWER / "reliability_calibration_audit.csv")
    assert len(functional) == N_TARGETS and functional.target_root_id.is_unique
    assert len(audit) == N_RECORDS and audit.scan.nunique() == N_SCANS
    records = []
    for row in functional.itertuples(index=False):
        key = (f"target{row.target_nucleus_id}_ses{row.session}"
               f"_scan{row.scan_idx}_automatic_conservative")
        vals = audit.loc[audit.scan.eq(key), "measured_split_half_spearman"].to_numpy(float)
        assert vals.size == int(row.n_partners), (key, vals.size, row.n_partners)
        assert abs(float(np.median(vals)) - float(row.median_repeat_reliability)) < 1e-9
        assert 0 < int(row.n_manual_subset) <= int(row.n_partners)
        records.append(vals)
    assert sum(len(v) for v in records) == 69
    return functional, audit, records


def panel_cohort(ax1, ax2, functional, records):
    n_mapped = functional.n_partners.to_numpy(float)
    n_manual = functional.n_manual_subset.to_numpy(float)
    medians = functional.median_repeat_reliability.to_numpy(float)
    assert n_mapped.tolist() == [10, 13, 17, 6, 9, 5, 9]
    y = np.arange(N_TARGETS)
    ylim = (N_TARGETS - 0.4, -2.7)          # two rows of headroom hold the keys
    for yi, (lo, hi) in enumerate(zip(n_manual, n_mapped)):
        ax1.plot([lo, hi], [yi, yi], color=MUTE, lw=LW_HAIR, zorder=2)
    ax1.plot(n_mapped, y, linestyle="none", marker="o", markersize=MARKER_MS,
             markerfacecolor=INK, markeredgecolor="white", markeredgewidth=LW_HAIR,
             zorder=4, label="mapped partners")
    ax1.plot(n_manual, y, linestyle="none", marker="o", markersize=MARKER_MS,
             markerfacecolor="white", markeredgecolor=INK, markeredgewidth=LW_ERR,
             zorder=4, label="manual subset")
    ax1.set_yticks(y, [f"target {i + 1}" for i in y])
    ax1.set_ylim(*ylim)
    ax1.set_xlim(-0.9, 20)                   # the n = 1 marker clears the spine (~3 pt)
    ax1.set_xticks([0, 10, 20], ["0", "10", "20"])
    ax1.set_xlabel("partners")
    data_grid(ax1, [10, 20], N_TARGETS - 0.4, -0.5)
    ax1.legend(loc="upper left", fontsize=PT_BASE, frameon=False, handlelength=1.0,
               handletextpad=0.4, borderaxespad=0.2, labelspacing=0.25, borderpad=0.0)
    for yi, vals in enumerate(records):
        ax2.plot(vals, yi + jitter(vals.size, 700 + yi, 0.10), linestyle="none",
                 marker="o", markersize=SEED_MS * 0.9, markerfacecolor=INK,
                 markeredgecolor="none", alpha=0.45, zorder=3,
                 label="partner record" if yi == 0 else None)
    ax2.plot(medians, y, linestyle="none", marker="|", markersize=6.5, color=INK,
             markeredgewidth=LW_ERR + 0.25, zorder=5, label="target median")
    ax2.set_yticks(y, [""] * N_TARGETS)
    ax2.tick_params(axis="y", length=0.0)
    ax2.set_ylim(*ylim)
    ax2.set_xlim(-0.035, 0.63)               # the smallest (0.011) and largest (0.570) records clear the frame (~3 pt)
    ax2.set_xticks([0, 0.3, 0.6], ["0", "0.3", "0.6"])
    ax2.set_xlabel("split-half r")
    data_grid(ax2, [0.3, 0.6], N_TARGETS - 0.4, -0.5)
    ax2.legend(loc="upper left", fontsize=PT_BASE, frameon=False, handlelength=1.0,
               handletextpad=0.4, borderaxespad=0.2, labelspacing=0.25, borderpad=0.0)
    print(f"[A] partners {n_mapped.astype(int).tolist()}, manual "
          f"{n_manual.astype(int).tolist()}, medians {np.round(medians, 4).tolist()}; "
          f"{sum(len(v) for v in records)} records")


# ── B: the four-route support of the representative scan, and all 13 scans
def load_support():
    support = csv(CFF / "figure_08_support.csv")
    sources = json.loads((CFF / "figure_08_sources.json").read_text())
    npz = np.load(CFF / "figure_08_actual_support.npz")
    matrix = np.asarray(npz["matrix"], float)
    site_ids = np.asarray(npz["site_ids"])
    route_ids = np.asarray(npz["route_ids"])
    assert len(support) == N_SCANS == sources["n_scans"]
    assert support.target_root_id.nunique() == N_TARGETS == sources["n_targets"]
    assert sources["representative"] == REPRESENTATIVE
    rep = support[(support.target_root_id.eq(REPRESENTATIVE["target_root_id"]))
                  & support.session.eq(REPRESENTATIVE["session"])
                  & support.scan_idx.eq(REPRESENTATIVE["scan_idx"])]
    assert len(rep) == 1
    rep = rep.iloc[0]
    n, k = matrix.shape
    assert (n, k) == (9, 4) == (int(rep.n_sites), int(rep.n_routes))
    assert site_ids.size == n and route_ids.size == k
    assert set(np.unique(matrix)) == {0.0, 1.0} and int(matrix.sum()) == 4
    occupied = np.flatnonzero(matrix.any(axis=1))
    assert (matrix.sum(axis=1)[occupied] == 1).all() and (matrix.sum(axis=0) == 1).all()
    # the route ids are the site ids of the occupied rows: one site per route
    for r in range(k):
        i = int(np.flatnonzero(matrix[:, r])[0])
        assert site_ids[i] == route_ids[r]
    assert bool(rep.one_site_routes) and float(rep.sites_per_route) == 1.0
    np.testing.assert_allclose(rep.coverage, len(occupied) / n, rtol=0, atol=1e-12)
    # the 13-scan occupancy the caption talks about
    ones = int(support.one_site_routes.sum())
    assert ones == 6 == sources["panel_f"]["all_one_site_scans"]
    assert (support.one_site_routes == support.sites_per_route.eq(1.0)).all()
    assert (support.n_routes == 4).all()
    np.testing.assert_allclose(support.sites_per_route * 4, np.round(support.sites_per_route * 4),
                               rtol=0, atol=1e-9)
    assert abs(float(support.sites_per_route.mean()) - 1.31) < 0.005   # the caption's 1.31
    assert support.n_sites.is_monotonic_increasing
    assert (support.n_sites.min(), support.n_sites.max()) == tuple(sources["partners_per_scan"])
    # the representative is target 1 of panel A (nucleus 256456)
    functional = csv(FIG5 / "functional_target_metrics.csv")
    assert int(functional.iloc[0].target_root_id) == REPRESENTATIVE["target_root_id"]
    assert int(functional.iloc[0].target_nucleus_id) == 256456
    print(f"[B] matrix {n}x{k}, {int(matrix.sum())} cells at rows "
          f"{(occupied + 1).tolist()} routes {(matrix[occupied].argmax(axis=1) + 1).tolist()}; "
          f"13 scans: one coordinate per route in {ones}/{N_SCANS}, mean "
          f"{support.sites_per_route.mean():.4f} per route")
    return support, matrix, rep


def panel_support(ax_m, ax_s, support, matrix):
    """The representative scan's 9 x 4 support on the left (its row labels
    hang into the gutter), the 13-scan occupancy strip on the right (its row
    labels hang into the gutter on the other side), so no text of either
    satellite lies inside the host box."""
    n, k = matrix.shape
    # the matrix as vector cells: route colour on the panel ground
    for i in range(n):
        for j in range(k):
            face = ROUTE if matrix[i, j] else COLORS["panel_bg"]
            ax_m.add_patch(Rectangle((j, i), 1, 1, facecolor=face, edgecolor="none",
                                     linewidth=0.0, zorder=2 if matrix[i, j] else 1))
    ax_m.set_xlim(0, k)
    ax_m.set_ylim(n, 0)
    ax_m.set_xticks(np.arange(k) + 0.5, [str(j + 1) for j in range(k)])
    ax_m.set_yticks(np.arange(n) + 0.5, [str(i + 1) for i in range(n)])
    ax_m.tick_params(axis="both", length=0.0, pad=2.0)
    for name, spine in ax_m.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(LW_HAIR)
        spine.set_color(EDGE)
    ax_m.set_xlabel("route")
    ax_m.set_ylabel("mapped input")
    # the 13 scans: input coordinates per route, rows ordered by mapped inputs
    ys = np.arange(N_SCANS)
    spr = support.sites_per_route.to_numpy(float)
    rep_i = int(np.flatnonzero(support.target_root_id.eq(REPRESENTATIVE["target_root_id"])
                               & support.session.eq(REPRESENTATIVE["session"])
                               & support.scan_idx.eq(REPRESENTATIVE["scan_idx"]))[0])
    ax_s.plot(spr, ys, linestyle="none", marker="o", markersize=MARKER_MS * 0.8,
              markerfacecolor=ROUTE, markeredgecolor="white", markeredgewidth=LW_HAIR,
              zorder=4, label="scan")
    ax_s.plot([spr[rep_i]], [rep_i], linestyle="none", marker="o", markersize=MARKER_MS * 1.6,
              markerfacecolor="none", markeredgecolor=ROUTE, markeredgewidth=LW_ERR,
              zorder=3.5, label="matrix scan")
    ax_s.set_ylim(N_SCANS - 0.4, -2.4)     # headroom for the two-entry key
    ax_s.set_yticks(ys, [str(int(v)) for v in support.n_sites])
    ax_s.yaxis.tick_right()
    ax_s.yaxis.set_label_position("right")
    ax_s.tick_params(axis="y", length=0.0, pad=2.0)
    ax_s.spines["left"].set_visible(False)
    ax_s.set_xlim(0.85, 2.15)
    ax_s.set_xticks([1.0, 1.5, 2.0], ["1", "1.5", "2"])
    data_grid(ax_s, [1.0, 1.5, 2.0], N_SCANS - 0.4, -0.5)
    ones = int(support.one_site_routes.sum())
    ax_s.set_xlabel(f"inputs per route\n(1 in {ones} of {N_SCANS} scans)")
    ax_s.set_ylabel(f"mapped inputs, {N_SCANS} scans", rotation=270, labelpad=9.0)
    # the key's glyphs hang 4.5 pt left of the strip (x < 0.85: no data column,
    # left of the x = 1 rule, clear of the matrix frame), its text over the headroom
    anchor = transforms.offset_copy(ax_s.transAxes, fig=ax_s.figure, x=-8.0, y=-1.4,
                                    units="points")
    ax_s.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), bbox_transform=anchor,
                fontsize=PT_BASE, frameon=False, handlelength=1.0, handletextpad=0.4,
                borderaxespad=0.0, labelspacing=0.25, borderpad=0.0)


# ── C: response prediction ───────────────────────────────────────────────
def panel_prediction(ax):
    cell = csv(BASELINES / "target_metrics.csv")
    summary = csv(BASELINES / "condition_summary.csv").set_index("method")
    figure = csv(CFF / "figure_08_prediction_summary.csv")
    recomputed = baselines_intervals(cell)          # assertion only
    g = figure[figure.panel.eq("G")].set_index("method")
    assert len(g) == 5
    printed = []
    rows = []
    for key, label, color in ROWS_C:
        z = cell[cell.method.eq(key)].sort_values("target_root_id")
        assert len(z) == N_TARGETS and z.target_root_id.is_unique
        seeds = z.nmse.to_numpy(float)
        r = summary.loc[key]
        assert int(r.n_targets) == N_TARGETS
        m, lo, hi = float(r.mean_nmse), float(r.ci95_low), float(r.ci95_high)   # drawn
        np.testing.assert_allclose(m, seeds.mean(), rtol=0, atol=1e-9)
        np.testing.assert_allclose([m, lo, hi], [g.loc[key, "mean"], g.loc[key, "ci95_low"],
                                                 g.loc[key, "ci95_high"]], rtol=0, atol=1e-9)
        assert_bootstrap_reproduces(recomputed, [m, lo, hi], key)
        rows.append((label, color, seeds, m, lo, hi))
        printed.append(f"{label} {m:.4f} [{lo:.4f}, {hi:.4f}] targets "
                       f"{seeds.min():.3f}-{seeds.max():.3f}")
    ridge = float(summary.loc["ridge_all"].mean_nmse)
    assert abs(ridge - 0.803) < 5e-4                     # the caption's 0.803
    print("[C] " + "; ".join(printed))
    lo_all = min(min(r[2].min(), r[4]) for r in rows)
    hi_all = max(max(r[2].max(), r[5]) for r in rows)
    x0, x1 = 0.60, 0.98
    assert x0 < lo_all and hi_all < x1 - 0.02, (lo_all, hi_all)
    forest_frame(ax, len(rows))
    ax.set_xlim(x0, x1)
    row_bands(ax, [r[1] for r in rows], x0, x1)
    ax.axvline(ridge, color=MUTE, lw=LW_REF, zorder=1.0, dashes=DASHES)
    ax.annotate(f"ridge {ridge:.3f}", xy=(ridge, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-2.5, 2.0), textcoords="offset points", ha="right", va="bottom",
                fontsize=PT_BASE, color=MUTE, annotation_clip=False)
    labels = []
    for y, (label, color, seeds, m, lo, hi) in enumerate(rows):
        fan_h(ax, y, seeds, color)
        whisker_h(ax, y, lo, hi, color)
        mean_marker(ax, m, y, color, "o")
        labels.append(row_label(ax, y, label, color, "o"))
    ax.set_xticks([0.6, 0.7, 0.8, 0.9], ["0.6", "0.7", "0.8", "0.9"])
    ax.set_xlabel("normalized MSE, held out")
    tag(ax, f"n = {N_TARGETS} targets; mean [95 % CI]", dy=11.0)
    return labels


# ── D: common-checkpoint update reconstruction ───────────────────────────
def panel_reconstruction(ax):
    cell = csv(ORACLE / "cell_metrics.csv")
    summary = csv(ORACLE / "condition_summary.csv").set_index(["method", "mode"])
    figure = csv(CFF / "figure_08_prediction_summary.csv")
    report = json.loads((ORACLE / "report.json").read_text())
    assert report["n_targets"] == N_TARGETS and report["n_exact_replays"] == 130
    recomputed = oracle_intervals(cell)             # assertion only
    h = figure[figure.panel.eq("H")].copy()
    h["key"] = h.method.str.split(" | ", regex=False)
    assert len(h) == 5
    printed = []
    rows = []
    for method, mode, label, color, oracle in ROWS_D:
        z = cell[cell.method.eq(method) & cell["mode"].eq(mode)].sort_values("target_root_id")
        assert len(z) == N_TARGETS and z.target_root_id.is_unique
        seeds = z.update_match.to_numpy(float)
        r = summary.loc[(method, mode)]
        assert int(r.n_targets) == N_TARGETS
        m, lo, hi = float(r.mean_update_match), float(r.ci95_low), float(r.ci95_high)  # drawn
        np.testing.assert_allclose(m, seeds.mean(), rtol=0, atol=1e-9)
        fr = h[h.key.map(lambda k: k == [method, mode])]
        assert len(fr) == 1
        fr = fr.iloc[0]
        np.testing.assert_allclose([m, lo, hi], [fr["mean"], fr.ci95_low, fr.ci95_high],
                                   rtol=0, atol=1e-9)
        assert_bootstrap_reproduces(recomputed, [m, lo, hi], (method, mode))
        rows.append((label, color, seeds, m, lo, hi, oracle))
        printed.append(f"{label.replace(chr(10), ' ')} {m:.4f} [{lo:.4f}, {hi:.4f}] targets "
                       f"{seeds.min():.3f}-{seeds.max():.3f}")
    top = rows[0]
    assert abs(top[3] - 0.976) < 5e-4 and abs(top[4] - 0.964) < 5e-4 and abs(top[5] - 0.986) < 5e-4
    assert abs(rows[1][3] - 0.232) < 5e-4 and abs(rows[2][3] - 0.244) < 5e-4
    assert abs(rows[3][3] - 0.385) < 5e-4 and abs(rows[4][3] - 0.468) < 5e-4
    assert rows[3][3] > rows[2][5] or rows[3][3] > rows[2][3]   # surrogates above ancestry
    print("[D] " + "; ".join(printed))
    x0, x1 = 0.0, 1.02
    assert all(0 < r[2].min() and r[2].max() < 1.0 for r in rows)
    forest_frame(ax, len(rows), headroom=0.55)
    ax.set_xlim(x0, x1)
    row_bands(ax, [r[1] for r in rows], x0, x1)
    labels = []
    for y, (label, color, seeds, m, lo, hi, oracle) in enumerate(rows):
        marker = "D" if oracle else "o"
        fan_h(ax, y, seeds, color)
        whisker_h(ax, y, lo, hi, color, dashed=oracle)
        mean_marker(ax, m, y, color, marker, ms=MARKER_MS * (0.85 if oracle else 1.0))
        labels.append(row_label(ax, y, label, color, marker))
    # the top row's interval is narrower than its marker: say so on the row
    ax.annotate(f"CI {top[4]:.3f}–{top[5]:.3f} hidden by marker",
                xy=(top[2].min(), 0), xycoords="data", xytext=(-8.0, 0.0),
                textcoords="offset points", ha="right", va="center", fontsize=PT_BASE,
                color=MUTE)
    handles = [
        Line2D([], [], color=INK, lw=LW_ERR, marker="o", ms=MARKER_MS * 0.8,
               markerfacecolor=INK, markeredgecolor="white", markeredgewidth=LW_HAIR,
               label="fixed transfer profile"),
        Line2D([], [], color=INK, lw=LW_ERR, dashes=DASHES, marker="D", ms=MARKER_MS * 0.7,
               markerfacecolor=INK, markeredgecolor="white", markeredgewidth=LW_HAIR,
               label="oracle trialwise amplitudes"),
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=PT_BASE,
              handlelength=2.2, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.15,
              borderpad=0.0, labelspacing=0.25)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_xlabel("update reconstruction (1 = exact)")
    tag(ax, f"n = {N_TARGETS} targets; mean [95 % CI]", dy=2.0)
    return labels


# ── E: reliability calibration residuals ─────────────────────────────────
def panel_calibration(ax, audit):
    meta = json.loads((POWER / "reliability_calibration_audit.json").read_text())
    assert meta["n_replicates"] == N_SIMULATED
    assert len(audit) == N_RECORDS and (audit.n_simulated == N_SIMULATED).all()
    measured = audit.measured_split_half_spearman.to_numpy(float)
    clipped = audit.clipped_measured_spearman.to_numpy(float)
    np.testing.assert_allclose(clipped, np.maximum(measured, 0.0), rtol=0, atol=1e-15)
    negative = measured < 0
    assert int(negative.sum()) == 2 and (clipped[negative] == 0.0).all()
    sim = audit.simulated_mean_split_half_spearman.to_numpy(float)
    se = audit.simulated_se.to_numpy(float)
    assert 0.0020 <= se.min() and se.max() <= 0.0029, (se.min(), se.max())
    resid = sim - clipped
    beyond = int((np.abs(resid) > Z95 * se).sum())
    assert beyond == 5
    largest = float(np.abs(resid).max())
    assert abs(largest - 0.0069) < 5e-5
    assert 0.011 < clipped[~negative].min() and clipped.max() < 0.58
    raw_negative = sorted(measured[negative].tolist(), reverse=True)
    assert [round(v, 3) for v in raw_negative] == [-0.098, -0.188]
    print(f"[E] {N_RECORDS} records, {int(negative.sum())} negative (raw "
          f"{np.round(measured[negative], 4).tolist()}); residual "
          f"{resid.min():.4f}..{resid.max():.4f}, SE {se.min():.4f}-{se.max():.4f}; "
          f"{beyond}/{N_RECORDS} beyond {Z95} SE; largest |residual| {largest:.4f}")
    pos = ~negative
    n_pos = int(pos.sum())
    records = ax.errorbar(clipped[pos], resid[pos], yerr=Z95 * se[pos], fmt="o",
                          ms=SEED_MS, color=INK, markeredgecolor="none", ecolor=INK,
                          elinewidth=LW_HAIR, alpha=0.5, capsize=0.0, zorder=2.5,
                          label=f"record: mean and ±{Z95} SE whisker from\n"
                                f"{N_SIMULATED:,} simulated datasets (n = {n_pos} of {N_RECORDS})")
    # the two clipped records sit at x = 0 exactly (no dodge); their residuals
    # differ, so the squares stack
    x_neg = clipped[negative]
    assert (x_neg == 0.0).all() and resid[negative][0] != resid[negative][1]
    ax.errorbar(x_neg, resid[negative], yerr=Z95 * se[negative], fmt="none",
                ecolor=INK, elinewidth=LW_HAIR, alpha=0.35, capsize=0.0, zorder=2.0)
    ax.plot(x_neg, resid[negative], linestyle="none", marker="s",
            markersize=SEED_MS + 0.6, markerfacecolor="white", markeredgecolor=INK,
            markeredgewidth=LW_ERR, zorder=4.0)
    ax.axhline(0.0, color=MUTE, lw=LW_REF, zorder=1.0, dashes=DASHES)
    lim = float(np.max(np.abs(resid) + Z95 * se)) * 1.12
    ax.set_ylim(-lim, lim)                   # headroom for the key is added after a draw
    ax.set_yticks([-0.01, -0.005, 0.0, 0.005, 0.01], ["−0.01", "−0.005", "0", "0.005", "0.01"])
    ax.set_xlim(-0.02, 0.60)
    ax.set_xticks([0, 0.2, 0.4, 0.6], ["0", "0.2", "0.4", "0.6"])
    ax.set_xlabel("measured split-half r (odd/even repeats)")
    ax.set_ylabel("simulated − measured\nsplit-half r")
    ax.grid(True, axis="y", zorder=0, linewidth=LW_HAIR, alpha=0.9, color=COLORS["grid"])
    handles = [
        records,
        Line2D([], [], linestyle="none", marker="s", ms=SEED_MS + 0.6, markerfacecolor="white",
               markeredgecolor=INK, markeredgewidth=LW_ERR,
               label=f"negative measured r, set to 0 (n = {int(negative.sum())} of {N_RECORDS};\n"
                     f"raw −{abs(raw_negative[0]):.3f} and −{abs(raw_negative[1]):.3f})"),
        Line2D([], [], color=MUTE, lw=LW_REF, dashes=DASHES,
               label="simulated mean = measured"),
    ]
    key = ax.legend(handles=handles, loc="upper left", ncol=1, frameon=False,
                    fontsize=PT_BASE, handlelength=1.8, handletextpad=0.5,
                    borderaxespad=0.2, borderpad=0.0, labelspacing=0.25)
    tag(ax, f"{beyond} of {N_RECORDS} beyond ±{Z95} SE; largest |difference| {largest:.4f}",
        dy=2.0)
    return dict(beyond=beyond, largest=largest, se=(float(se.min()), float(se.max())),
                n_pos=n_pos, key=key, lim=lim)


def calibration_headroom(ax, key, lim, *, clearance_pt=4.0):
    """Extend E's y-range upward by exactly the key's height (+ clearance)
    so the data band keeps its symmetric ±lim range below the key."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    key_pt = key.get_window_extent(renderer).height / fig.dpi * 72.0 + 0.2 * PT_BASE
    ax_pt = ax.get_position().height * fig.get_figheight() * 72.0
    head_pt = key_pt + clearance_pt
    per_pt = 2.0 * lim / (ax_pt - head_pt)
    ax.set_ylim(-lim, lim + head_pt * per_pt)
    return head_pt


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 403.0
ROW_PT = [140.0, 165.0]
HGUTTER_PT = 24.0
VGUTTER_PT = 52.0
MARGINS = Margins(left=6.0, right=8.0, top=16.0, bottom=30.0)
TITLE_PAD = 11.0


def build(path: Path = OUT):
    functional, audit, records = load_cohort()
    support, matrix, rep = load_support()
    # the seven targets of A are the seven targets of C and D
    roots = set(int(v) for v in functional.target_root_id)
    assert roots == set(int(v) for v in csv(BASELINES / "target_metrics.csv").target_root_id)
    assert roots == set(int(v) for v in csv(ORACLE / "cell_metrics.csv").target_root_id)
    assert roots == set(int(v) for v in support.target_root_id)

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 2, row_weights=ROW_PT, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS)
    ax_a = cv.panel("A", 0, 0, 5, title="Measured cohort")
    ax_b = cv.panel("B", 0, 5, 3, title="Four-route support\ntarget 1, session 4, scan 10")
    ax_c = cv.panel("C", 0, 8, 4, title="Response prediction")
    ax_d = cv.panel("D", 1, 0, 6, title="Fixed-profile fidelity: update reconstruction")
    ax_e = cv.panel("E", 1, 6, 6, title="Reliability calibration: simulated − measured")
    for ax in (ax_a, ax_b, ax_d, ax_e):
        ax.set_title(ax.get_title(), fontsize=PT_EMPH, color=INK, pad=TITLE_PAD, fontweight="normal")
    ax_c.set_title(ax_c.get_title(), fontsize=PT_EMPH, color=INK, pad=TITLE_PAD + 9.0, fontweight="normal")

    # A: two sub-axes on one target axis (counts | reliability records)
    ax_a1, ax_a2 = sub_axes(cv, ax_a, [(0.0, 0.47), (0.55, 0.45)])
    panel_cohort(ax_a1, ax_a2, functional, records)
    # B: the 9 x 4 matrix | the 13-scan occupancy strip
    ax_bm, ax_bs = sub_axes(cv, ax_b, [(0.0, 0.40), (0.54, 0.46)])
    panel_support(ax_bm, ax_bs, support, matrix)
    labels = panel_prediction(ax_c)
    labels_d = panel_reconstruction(ax_d)
    stats = panel_calibration(ax_e, audit)
    head_e = calibration_headroom(ax_e, stats.pop("key"), stats.pop("lim"))
    print(f"[E] key headroom {head_e:.1f} pt")

    # reserves the lock cannot measure: the satellites' tick columns and the
    # forests' label gutters (row labels are annotations, not tick labels)
    cv.fig.canvas.draw()
    renderer = cv.fig.canvas.get_renderer()
    gutter_c = max(_text_width_pt(t, renderer) for t in labels) + 9.5 + 4.0
    gutter_d = max(_text_width_pt(t, renderer) for t in labels_d) + 9.5 + 4.0
    ticks_a = max(_text_width_pt(t, renderer) for t in ax_a1.get_yticklabels()) + 4.6 + 6.0
    cv.declare_reserve("A", left=ticks_a, right=4.0)
    cv.declare_reserve("B", left=8.0, right=8.0)      # emphasis parity with A and C
    cv.declare_reserve("C", left=gutter_c, right=4.0)
    cv.declare_reserve("D", left=gutter_d, right=12.0)
    cv.declare_reserve("E", left=gutter_d, right=12.0)   # one width for the D/E pair
    print(f"[reserves] A {ticks_a:.1f}, C {gutter_c:.1f}, D {gutter_d:.1f} pt")
    problems = cv.save(path, name="figure_measured_transfer_geometry_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    box = {n: cv.axes[n].get_position() for n in "ABCDE"}
    print("[boxes] " + "; ".join(
        f"{n} {b.width * cv.width_pt:.1f}x{b.height * cv.height_pt:.1f}" for n, b in box.items()))
    print(f"[E] tag {stats}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)
