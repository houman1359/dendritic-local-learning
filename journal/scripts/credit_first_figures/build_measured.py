#!/usr/bin/env python3
"""Main figure 9: measured responses and the realized-route boundary.

Frozen-data reanalysis only.  Eight lettered panels on one native canvas:
three schematics (A the passive-tree learner and its two credit deliveries,
B the structure-function question, E the realized route dictionary of the
representative scan) and five data panels (C observed alignment, D the
simulated detection curve, F route coverage, G held-out prediction,
H update reconstruction).

The representative route matrix is selected by median mapped-input count,
with target/session/scan identifiers as deterministic ties.  All 13 scan
supports and all seven target outcomes are retained as Source Data.

Waivers against the shared layout rule (D3, three 4-module panels per row):
row 1 is C (data) + D (data) + E (schematic) and row 2 is F + G + H; C and D
do not share an axis, and F does not share G/H's forest idiom.  Both rows are
column-locked so the letters align, and every panel in a row keeps one axes
height, which is what the rule protects.

Every printed number is derived here from the Source Data files recorded in
``figure_08_sources.json``; nothing is typed in.
"""
from pathlib import Path
import sys
import json
import hashlib

import numpy as np
import pandas as pd

J = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J / 'scripts'))
sys.path.insert(0, str(J / 'code/reconstructed_tree'))

from figure_canvas import (NativeCanvas, Margins, COLORS, PT_SMALL, PT_ANNOT,
                           PT_LABEL, PT_TICK, LW_HAIR, LW_EDGE, LW_ERR, LW_REF,
                           LW_DATA, MARKER_MS, SEED_MS, SEED_ALPHA, style_panel)
from journal_style import label_color, style_direct_color_labels
from native_schematics import BADGE_STYLE, Frame, mix
from analyze_microns_morphology_credit import ancestry_matrix, parent_map

S = J / 'source_data'
OUT = J / 'figures/components/credit_first_figure_08.pdf'
MAIN = J / 'figures/main/figure_09.pdf'
REC = S / 'credit_first_figures'

ROUTE = COLORS['shunting']       # morphology-matched implementable rule
EXACT = COLORS['bp']             # exact-path credit
GRAY = COLORS['mute']            # random / shuffled / ridge controls
FIXED = COLORS['per_soma']       # unrestricted fixed calibrated profile
ORACLE = COLORS['oracle']        # oracle comparator
INK = COLORS['ink']
REP_TARGET = 864691135810666525  # representative scan, fixed by median rule
FAN_LANES = (-0.075, 0.0, 0.075)  # deterministic lanes for scan-level circles


# ── shared data-panel idiom ───────────────────────────────────────────────
def forest(ax, rows, xlabel, xlim, *, xticks=None, label_pad_pt=3.0,
           lift=0.24):
    """Row-per-condition dot-and-interval forest with IN-AXIS row labels.

    The row label is set inside the axes above its own marker, so no ink
    enters the 28-pt panel-letter gutter (spec 0.4.4) and no column has to
    carve a category-label reserve out of its slot.
    """
    n = len(rows)
    span = xlim[1] - xlim[0]
    for k, row in enumerate(rows):
        y = n - 1 - k
        values = np.asarray(row['values'], float)
        ax.scatter(values, y + np.linspace(-.13, .13, len(values)),
                   s=SEED_MS ** 2, color=row['color'], alpha=SEED_ALPHA,
                   edgecolors='none', zorder=3)
        ax.errorbar(row['mean'], y,
                    xerr=[[row['mean'] - row['lo']], [row['hi'] - row['mean']]],
                    fmt=row.get('marker', 'o'), ms=MARKER_MS, color=row['color'],
                    mfc='white', mew=LW_ERR, lw=LW_ERR, capsize=2, zorder=5)
        if row.get('open') is not None:
            extra = np.asarray(row['open'], float)
            # Scan-level values ride just under their own row (not as a third
            # row) on a deterministic three-lane fan: consecutive values in
            # sorted order never share a lane, so near-duplicates separate.
            order = np.argsort(extra, kind='stable')
            lane = np.empty(len(extra))
            lane[order] = np.take(FAN_LANES, np.arange(len(extra)) % len(FAN_LANES))
            ax.plot(extra, y - 0.24 + lane, 'o', ms=SEED_MS - 0.4,
                    mfc='none', mec=row.get('open_color', GRAY), mew=LW_HAIR,
                    zorder=4)
            tag = row.get('open_label')
            if tag:
                ax.text(float(extra.max()) + span * 0.035, y - 0.24, tag,
                        fontsize=PT_SMALL, color=row.get('open_color', GRAY),
                        ha='left', va='center', zorder=6)
        ax.text(xlim[0] + span * (label_pad_pt / 100.0), y + lift, row['label'],
                fontsize=PT_SMALL, color=label_color(row['color']),
                ha='left', va='bottom', zorder=6)
    ax.set_yticks([])
    ax.set_ylim(-0.62, n - 0.36)
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_xlabel(xlabel)
    style_panel(ax, grid='x')
    ax.spines['left'].set_visible(False)
    return ax


def _badge_on(ax, x, y, kind, *, ha='left'):
    """Frame.badge on a DATA axes: the library helper needs a Frame, which
    rescales its host to 0-1, so the same rounded PT_SMALL tag is drawn here
    from BADGE_STYLE (reported as a private helper)."""
    key, face, edge = BADGE_STYLE[kind]
    colour = COLORS[key]
    try:
        colour = label_color(colour, background=face)
    except ValueError:
        pass
    return ax.text(x, y, kind, fontsize=PT_SMALL, color=colour, ha=ha,
                   va='center', zorder=7,
                   bbox=dict(boxstyle='round,pad=0.28,rounding_size=0.294',
                             facecolor=face, edgecolor=edge,
                             linewidth=LW_HAIR))


def cohort_note(ax, text, *, x=0.99, y=0.015):
    """PT_SMALL mute in-panel cohort declaration (spec 0.4.11)."""
    return ax.text(x, y, text, transform=ax.transAxes, fontsize=PT_SMALL,
                   color=GRAY, ha='right', va='bottom', zorder=6)


def mini_tuning(f, rect_pt, phase):
    """A 24 x 15 pt pair of condition-mean response curves (schematic)."""
    inner = f.axes_inset(rect_pt)
    t = np.linspace(0, 1, 60)
    for shift, lw in ((0.0, LW_EDGE), (phase, LW_EDGE)):
        inner.plot(t, 0.5 + 0.42 * np.sin(2 * np.pi * (t + shift)),
                   color=COLORS['exc'], lw=lw, solid_capstyle='round')
    inner.set_xticks([])
    inner.set_yticks([])
    inner.set_ylim(-0.05, 1.05)
    for name, spine in inner.spines.items():
        spine.set_visible(name in ('left', 'bottom'))
        spine.set_linewidth(LW_HAIR)
        spine.set_color(COLORS['edge'])
    return inner


# ── panel A: the learner and its two credit deliveries ────────────────────
def panel_learner(f, header, footer):
    """Two stations: the passive-tree learner, then its two credit deliveries."""
    band = f.footer(footer, band_pt=11.0)
    top = 1.0 - f.fy(1.0)
    gap = f.fx(14.0)
    left = (0.0, f.fy(band), 0.46 * (1.0 - gap), top - f.fy(band))
    right = (left[0] + left[2] + gap, left[1], (1.0 - gap) - left[2], left[3])

    # station 1: condition-mean partner responses drive a passive tree
    core = f.task_card(left, title=header)
    tree_w, tree_h = 0.54 * core[2], f.fy(56.0)
    tree_rect = (core[0], core[1] + f.fy(11.0), tree_w, tree_h)
    nodes = f.balanced_tree(tree_rect, depth=3, mode='plain', labels=False)
    for name in ('T1', 'T3', 'T4', 'T6', 'T8'):
        f.contact(nodes[name], kind='exc')
    canopy = max(nodes[t][1] for t in nodes.terminals)
    f.text((core[0] + core[2] / 2.0, core[1] + core[3] - f.fy(1.0)),
           'condition-mean partner responses', size=PT_SMALL, color=INK,
           ha='center', va='top')
    tag = (core[0] + tree_w + f.fx(5.0), canopy)
    f.leader(f._off(nodes['T8'], 1.6, 1.6), f._off(tag, -1.5, 0.0))
    f.text(tag, 'g ≥ 0', size=PT_SMALL, color=COLORS['exc'], ha='left',
           va='center')

    card_w, card_h = 50.0, 26.0
    sx, sy = nodes.soma
    read = (core[0] + tree_w + f.fx(8.0), sy - f.fy(card_h / 2.0),
            f.fx(card_w), f.fy(card_h))
    f.soma(nodes.soma, output=(read[0] - sx) * f.w_pt - 4.5)
    f.error_in(nodes.soma, side='right')
    f.group(read, tint='white', edge=COLORS['grid'], lw=LW_HAIR)
    f.text((read[0] + read[2] / 2.0, sy), 'linear\nreadout ŷ', size=PT_SMALL,
           linespacing=1.15)
    cx, y_top = read[0] + read[2] / 2.0, read[1] + read[3]
    f.arrow((cx, y_top + f.fy(12.0)), (cx, y_top + f.fy(1.5)), color=INK,
            lw=LW_EDGE, head=3.4).set_linestyle((0, (2.2, 1.8)))
    f.text((cx + f.fx(2.5), y_top + f.fy(12.5)), 'y', size=PT_SMALL,
           color=INK, ha='left', va='bottom')
    f.subscript((read[0] + f.fx(1.5), read[1] + read[3] + f.fy(2.0)),
                'δ', 'out', size=PT_ANNOT, color=INK, ha='left', va='bottom')

    # station 2: exact transport versus the realized-route restriction
    core = f.task_card(right, title='credit delivery into the tree')
    for i, (cell, mode, kind) in enumerate(
            zip(_split_rect(f, core, 2, gap_pt=8.0),
                ('exact', 'subtree'), ('exact', 'local rule'))):
        cell = Frame.inset(cell, left=0.02, right=0.02)
        rect = (cell[0], cell[1] + f.fy(31.0), cell[2], f.fy(56.0))
        gn = f.balanced_tree(rect, depth=3, mode='plain', ghost=True,
                             labels=False)
        if mode == 'exact':
            f.credit_delivery(gn, mode='exact', targets=['T6'])
        else:
            f.credit_delivery(gn, mode='subtree',
                              targets=['T1', 'T4', 'T6', 'T8'],
                              rule_color='shunting')
        f.error_in(gn.soma, side='right', label=None)
        f.badge((cell[0] + cell[2] / 2.0, cell[1] + f.fy(15.0)), kind,
                ha='center', va='bottom')
        if mode == 'subtree':
            f.text((cell[0] + cell[2] / 2.0, cell[1] + f.fy(1.0)), 'δ = A c',
                   size=PT_ANNOT, color=INK, ha='center', va='bottom')
    return nodes


def _split_rect(f, rect, n, *, gap_pt=8.0):
    """``n`` equal columns inside ``rect`` (Frame.split works on the frame)."""
    gap = f.fx(gap_pt)
    w = (rect[2] - (n - 1) * gap) / n
    return [(rect[0] + i * (w + gap), rect[1], w, rect[3]) for i in range(n)]


# ── panel B: the structure-function question ──────────────────────────────
def panel_question(f):
    band = f.footer('partial rank r | distance, depth\n'
                    'recorded during stimuli', band_pt=11.0)
    core = (0.0, f.fy(band), 1.0, 1.0 - f.fy(band))
    tree_rect = (core[0], core[1] + f.fy(11.0), core[2] * 0.58,
                 core[3] * 0.74)
    nodes = f.balanced_tree(tree_rect, depth=3, mode='plain', labels=False)
    f.partition(nodes, [['S', 'J1', 'JL', 'JLL']], colors=('shunting',),
                labels=None)
    for name in ('T1', 'T2', 'T5', 'T8'):
        f.contact(nodes[name], kind='exc')
    f.soma(nodes.soma, output=9.0)
    f.error_in(nodes.soma, side='right', label=None, dashed=True)
    f.text(f._off(nodes['JL'], -4.0, 0.0), 'shared\npath', size=PT_SMALL,
           color=label_color(ROUTE), ha='right', va='center', linespacing=1.15)
    mid = ((nodes['T5'][0] + nodes['T8'][0]) / 2.0,
           max(nodes['T5'][1], nodes['T8'][1]))
    f.text(f._off(mid, 0.0, 4.5), 'none', size=PT_SMALL, color=GRAY,
           ha='center', va='bottom')

    # the two tuning comparisons span the cell's full height so the panel's
    # ink reaches its own frame (the strict audit's cell-fill check)
    col_x = core[0] + core[2] * 0.63
    w_pt, h_pt = 26.0, 16.0
    top = core[1] + core[3]
    for i, (phase, tag, tone) in enumerate(
            ((0.10, 'shared path', label_color(ROUTE)),
             (0.42, 'none', GRAY))):
        y = (top - f.fy(h_pt + 9.0)) if i == 0 else (core[1] + f.fy(3.0))
        f.text((col_x, y + f.fy(h_pt + 1.5)), tag, size=PT_SMALL, color=tone,
               ha='left', va='bottom')
        mini_tuning(f, (col_x, y, f.fx(w_pt), f.fy(h_pt)), phase)
    return nodes


# ── panel E: the realized route dictionary ────────────────────────────────
def panel_routes(f, matrix, groups, n_routes):
    """Balanced-tree icon with the mapped inputs placed by ancestry, beside
    the realized input-by-route support matrix (rows in input order)."""
    n = matrix.shape[0]
    tree_rect = (0.0, f.fy(10.0), 0.56, 0.62)
    nodes = f.balanced_tree(tree_rect, depth=3, mode='plain', labels=False)
    # ancestry placement: the shared-subtree group on one depth-2 subtree,
    # every other mapped input on its own branch.
    shared, singles = groups
    seats = {shared[0]: ('T1', 0.0), shared[1]: ('T2', 0.0),
             shared[2]: ('T2', 5.2)}
    for i, label in enumerate(singles):
        seats[label] = (f'T{i + 3}', 0.0)
    reached = [i + 1 for i in range(n) if matrix[i].any()]
    f.credit_delivery(nodes, mode='subtree',
                      targets=[seats[i][0] for i in reached],
                      rule_color='shunting')
    # numbers alternate between two lanes above the canopy so neighbouring
    # single digits keep their own air at a 7-pt terminal pitch
    for label, (seat, drop) in seats.items():
        par = nodes.parent[seat]
        xy = nodes[seat] if drop == 0.0 else _toward(f, nodes[seat],
                                                     nodes[par], drop)
        f.contact(xy, kind='exc')
        if drop:
            f.text(f._off(xy, -3.0, 0.0), str(label), size=PT_SMALL,
                   color=INK, ha='right', va='center')
            continue
        lane = 3.0 if int(seat[1:]) % 2 else 10.0
        f.text(f._off(xy, 0.0, lane), str(label), size=PT_SMALL, color=INK,
               ha='center', va='bottom')
    f.soma(nodes.soma, output=8.0)
    f.error_in(nodes.soma, side='right', label=None, dashed=True)

    mat = (0.645, f.fy(17.0), 0.335, 1.0 - f.fy(30.0))
    f.dictionary_matrix(mat, matrix, measured=True,
                        yticks=list(range(1, n + 1)), label='A')
    f.text((mat[0] + mat[2] / 2.0, mat[1] + mat[3] + f.fy(2.5)),
           f'routes 1–{n_routes}', size=PT_SMALL, color=INK, ha='center',
           va='bottom')
    return nodes


def _toward(f, a, b, pt):
    """Point ``pt`` points from ``a`` toward ``b``."""
    A, B = f._to_pt(a), f._to_pt(b)
    d = B - A
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    return f._from_pt(A + d * pt)


# ── build ─────────────────────────────────────────────────────────────────
def main():
    REC.mkdir(exist_ok=True)

    # -- realized route supports, all 13 scans ----------------------------
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
    # Ancestor supports contain actual zeros; no decorative nested bands.

    # -- ancestry groups of the representative scan's mapped inputs -------
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

    # -- observed structure-function alignment (panel C) ------------------
    contrasts = pd.read_csv(S / 'review_evidence_reanalysis/functional_native_contrasts.csv')
    effects = pd.read_csv(S / 'review_evidence_reanalysis/functional_native_target_effects.csv')
    scan_metrics = pd.read_csv(S / 'functional_topology_all_scans/scan_metrics.csv')
    all_scans = json.loads((S / 'functional_topology_all_scans/summary.json').read_text())
    selected = pd.read_csv(S / 'figure5/functional_target_metrics.csv')
    align_rows = []
    for mode, label in (('selected_scans', 'Selected scans'),
                        ('scan_complete', f"All {all_scans['n_scans']} scans")):
        r = contrasts[contrasts.endpoint.eq('structure_function_partial_r')
                      & contrasts.comparison.eq(mode)].iloc[0]
        v = effects[effects.endpoint.eq('structure_function_partial_r')
                    & effects.comparison.eq(mode)].effect.to_numpy()
        assert len(v) == 7
        align_rows.append(dict(label=label, values=v, mean=r['mean'],
                               lo=r.ci95_low, hi=r.ci95_high, color=ROUTE,
                               marker='o',
                               open=(scan_metrics.shared_path_partial_r.to_numpy()
                                     if mode == 'scan_complete' else None)))

    # -- simulated detection curve (panel D) ------------------------------
    power = pd.read_csv(S / 'measured_alignment_power/power_summary.csv')
    power = power[power.reliability.eq('measured')].sort_values('mean_target_effect')
    thresholds = json.loads((S / 'measured_alignment_power/RESULTS.json').read_text())
    cut = float(thresholds['thresholds']['measured']
                ['descriptive_mean_partial_rank_at_interpolation'])
    obs = contrasts[contrasts.endpoint.eq('structure_function_partial_r')
                    & contrasts.comparison.eq('selected_scans')].iloc[0]

    # -- held-out response prediction (panel G) ---------------------------
    target = pd.read_csv(S / 'review_response_baselines/target_metrics.csv')
    csum = pd.read_csv(S / 'review_response_baselines/condition_summary.csv')
    ridge_contrasts = pd.read_csv(S / 'review_response_baselines/paired_ridge_contrasts.csv')
    predict_rows, estimates = [], []
    for name, label, colour, marker in (
            ('archived_exact compartment error', 'Exact', EXACT, 'o'),
            ('ridge_all', 'Ridge', GRAY, '^'),
            ('archived_topology-matched routes', 'Ancestry', ROUTE, 'o'),
            ('archived_random anatomical routes', 'Random', GRAY, 'v'),
            ('archived_site-shuffled routes', 'Shuffled', GRAY, 'v')):
        v = target[target.method.eq(name)].sort_values('target_root_id').nmse.to_numpy()
        assert len(v) == 7
        r = csum[csum.method.eq(name)].iloc[0]
        predict_rows.append(dict(label=label, values=v, mean=r.mean_nmse,
                                 lo=r.ci95_low, hi=r.ci95_high, color=colour,
                                 marker=marker))
        estimates.append(dict(panel='G', method=name, mean=r.mean_nmse,
                              ci95_low=r.ci95_low, ci95_high=r.ci95_high))
    ridge_mean = float(csum[csum.method.eq('ridge_all')].mean_nmse.iloc[0])

    # -- update reconstruction (panel H) ----------------------------------
    cells = pd.read_csv(S / 'fulltree_within_span_oracle/cell_metrics.csv')
    osum = pd.read_csv(S / 'fulltree_within_span_oracle/condition_summary.csv')
    span_rows = []
    for name, mode, label, colour, marker in (
            ('unprojected baseline transfer', 'frozen_baseline',
             'Unrestricted / fixed', FIXED, 'o'),
            ('topology-matched routes', 'frozen_baseline',
             'Ancestry / fixed', ROUTE, 'o'),
            ('topology-matched routes', 'trialwise_update_oracle',
             'Ancestry / oracle', ROUTE, 'D'),
            ('random anatomical routes', 'trialwise_update_oracle',
             'Random / oracle', GRAY, 'D'),
            ('site-shuffled routes', 'trialwise_update_oracle',
             'Shuffled / oracle', GRAY, 'D')):
        r = osum[osum.method.eq(name) & osum['mode'].eq(mode)].iloc[0]
        v = cells[cells.method.eq(name) & cells['mode'].eq(mode)].sort_values(
            'target_root_id').update_match.to_numpy()
        assert len(v) == 7
        span_rows.append(dict(label=label, values=v, mean=r.mean_update_match,
                              lo=r.ci95_low, hi=r.ci95_high, color=colour,
                              marker=marker))
        estimates.append(dict(panel='H', method=f'{name} | {mode}',
                              mean=r.mean_update_match, ci95_low=r.ci95_low,
                              ci95_high=r.ci95_high))

    # -- canvas -----------------------------------------------------------
    canvas = NativeCanvas(485 / 72, 3, row_weights=[122, 100, 104],
                          hgutter_pt=40, vgutter_pt=48,
                          margins=Margins(left=52, right=14, top=25, bottom=38))
    ax_a = canvas.panel('A', 0, 0, 8, schematic=True,
                        title='Partner responses drive a passive tree')
    ax_b = canvas.panel('B', 0, 8, 4, schematic=True,
                        title='Shared path vs similarity')
    ax_c = canvas.panel('C', 1, 0, 4, title='Alignment near zero')
    ax_d = canvas.panel('D', 1, 4, 4, title='Below the detection point')
    ax_e = canvas.panel('E', 1, 8, 4, schematic=True,
                        title='One input per route')
    ax_f = canvas.panel('F', 2, 0, 4, title='Coverage falls with inputs')
    ax_g = canvas.panel('G', 2, 4, 4, title='No gain over ridge')
    ax_h = canvas.panel('H', 2, 8, 4, title='Span, not c, is the limit')

    # -- C: observed alignment --------------------------------------------
    forest(ax_c, align_rows, 'Partial rank correlation', (-0.74, 0.62),
           xticks=[-0.5, -0.25, 0, 0.25, 0.5])
    ax_c.axvline(0, color=GRAY, lw=LW_REF, ls='--', zorder=1)
    ax_c.text(0.03, 1.42, 'chance', fontsize=PT_SMALL, color=GRAY, ha='left',
              va='center')
    ax_c.text(-0.72, -0.33, 'scans', fontsize=PT_SMALL, color=GRAY,
              ha='left', va='center')

    # -- D: simulated detection curve -------------------------------------
    x = power.mean_target_effect.to_numpy()
    ax_d.fill_between(x, power.power_ci95_low, power.power_ci95_high,
                      color=mix('shunting', 16), lw=0, zorder=2)
    ax_d.plot(x, power.power, color=ROUTE, lw=LW_DATA, zorder=3,
              solid_capstyle='round')
    ax_d.axhline(0.80, color=GRAY, lw=LW_REF, ls='--', zorder=1)
    ax_d.plot([cut, cut], [0, 0.80], color=GRAY, lw=LW_HAIR, ls='--', zorder=1)
    ax_d.text(cut + 0.012, 0.80, f'80 %\nat {cut:.3f}', fontsize=PT_SMALL,
              color=GRAY, ha='left', va='top', linespacing=1.15)
    ax_d.errorbar(obs['mean'], 0.055,
                  xerr=[[obs['mean'] - obs.ci95_low],
                        [obs.ci95_high - obs['mean']]],
                  fmt='D', ms=MARKER_MS, color=INK, mfc='white', mew=LW_ERR,
                  lw=LW_ERR, capsize=2, zorder=6)
    ax_d.text(obs.ci95_high + 0.015, 0.055, 'observed', fontsize=PT_SMALL,
              color=INK, ha='left', va='center')
    ax_d.set(xlim=(-0.30, 0.375), ylim=(-0.02, 1.03),
             yticks=[0, 0.5, 1.0],
             xlabel='Mean simulated effect', ylabel='Detection probability')
    ax_d.set_xticks([-0.2, 0, 0.2])
    style_panel(ax_d, grid='y')
    # keep D's y decoration narrow: its panel letter shares the 40-pt gutter
    # with C's right edge (canvas.align_letters measures the collision).
    ax_d.yaxis.labelpad = 1.0

    # -- F: route coverage ------------------------------------------------
    single = support.one_site_routes.to_numpy()
    n_in = support.n_sites.to_numpy()
    cov = support.coverage.to_numpy() * 100.0
    grid_n = np.arange(n_in.min(), n_in.max() + 1)
    ax_f.plot(grid_n, 100.0 * support.n_routes.max() / grid_n, color=GRAY,
              lw=LW_REF, ls='--', zorder=2)
    ax_f.text(grid_n[-1] - 0.3, 100.0 * support.n_routes.max() / grid_n[-1] + 4,
              '4/n', fontsize=PT_SMALL, color=GRAY, ha='right', va='bottom')
    ax_f.plot(n_in[~single], cov[~single], 'o', ms=MARKER_MS - 0.6, color=ROUTE,
              mfc=ROUTE, mew=0, zorder=4)
    ax_f.plot(n_in[single], cov[single], 'o', ms=MARKER_MS - 0.6, color=ROUTE,
              mfc='white', mew=LW_ERR, zorder=5)
    ax_f.axhline(100, color=GRAY, lw=LW_REF, ls='--', zorder=1)
    ax_f.text(n_in.max(), 100.5, 'all inputs', fontsize=PT_SMALL, color=GRAY,
              ha='right', va='bottom')
    ax_f.set(xlim=(3.6, 18.4), ylim=(0, 108), xticks=[5, 9, 13, 17],
             yticks=[0, 25, 50, 75, 100],
             xlabel='Mapped inputs per scan', ylabel='Inputs reached (%)')
    style_panel(ax_f, grid='y')
    ax_f.text(0.03, 0.03, f'open: all routes single-input '
                           f'({int(single.sum())}/13)',
              transform=ax_f.transAxes, fontsize=PT_SMALL, color=GRAY,
              ha='left', va='bottom', zorder=6)

    # -- G: held-out prediction -------------------------------------------
    forest(ax_g, predict_rows, 'Held-out NMSE', (0.58, 1.02),
           xticks=[0.6, 0.7, 0.8, 0.9])
    ax_g.axvline(ridge_mean, color=GRAY, lw=LW_REF, ls=':', zorder=1)
    _badge_on(ax_g, 1.01, 4.28, 'exact', ha='right')

    # -- H: update reconstruction -----------------------------------------
    forest(ax_h, span_rows, 'Update reconstruction', (-0.06, 1.20),
           xticks=[0, 0.5, 1.0])
    ax_h.axvline(1.0, color=GRAY, lw=LW_REF, ls='--', zorder=1)
    ax_h.text(1.02, 3.66, 'exact', fontsize=PT_SMALL, color=GRAY, ha='left',
              va='center')
    _badge_on(ax_h, 0.62, 2.28, 'oracle')

    # -- lock the grid, then draw the schematics at their final sizes -----
    style_direct_color_labels(canvas.fig)
    canvas.lock_reserves()

    header = (f"{all_scans['n_target_cells']} targets, {all_scans['n_scans']} "
              f"scans, {int(scan_metrics.n_presynaptic_partners.min())}–"
              f"{int(scan_metrics.n_presynaptic_partners.max())} partners per "
              f"scan ({int(selected.n_partners.sum())} selected)")
    panel_learner(Frame(ax_a), header,
                  'A = realized routes; c fixed or oracle; '
                  'readout and bias exact, stimuli held out')
    panel_question(Frame(ax_b))
    panel_routes(Frame(ax_e), a, (shared, singles), int(a.shape[1]))

    findings = list(canvas.align_letters())
    problems = canvas.save(OUT, name='credit_first_figure_08', dpi=180,
                           lock=False)
    MAIN.parent.mkdir(parents=True, exist_ok=True)
    MAIN.write_bytes(OUT.read_bytes())

    # -- render-time Source Data and provenance ---------------------------
    support.to_csv(REC / 'figure_08_support.csv', index=False)
    pd.DataFrame(estimates).to_csv(REC / 'figure_08_prediction_summary.csv',
                                   index=False)
    np.savez_compressed(REC / 'figure_08_actual_support.npz', matrix=a,
                        site_ids=np.array(meta['site_segment_ids']),
                        route_ids=np.array(meta['selected_route_segments']))
    files = [Path(__file__),
             S / 'figure3/segment_metrics.csv',
             S / 'fulltree_boundary/output/dictionary_and_validation_metadata.jsonl',
             S / 'review_evidence_reanalysis/functional_native_contrasts.csv',
             S / 'review_evidence_reanalysis/functional_native_target_effects.csv',
             S / 'functional_topology_all_scans/scan_metrics.csv',
             S / 'functional_topology_all_scans/summary.json',
             S / 'figure5/functional_target_metrics.csv',
             S / 'measured_alignment_power/power_summary.csv',
             S / 'measured_alignment_power/RESULTS.json',
             S / 'review_response_baselines/target_metrics.csv',
             S / 'review_response_baselines/condition_summary.csv',
             S / 'review_response_baselines/paired_ridge_contrasts.csv',
             S / 'fulltree_within_span_oracle/cell_metrics.csv',
             S / 'fulltree_within_span_oracle/condition_summary.csv']
    payload = dict(
        representative=dict(zip(['target_root_id', 'session', 'scan_idx'], key)),
        selection='Median mapped-input count, identifier ties; no outcome selection',
        coordinate_definition=(
            'Rows are mapped partner inputs; multiple inputs can share a '
            'physical segment. Legacy n_sites and sites_per_route keys count '
            'these input coordinates.'),
        ancestry_placement=dict(shared_subtree=shared, own_branch=singles,
                                rule='deepest shared ancestor of the mapped '
                                     'input segments; schematic placement only'),
        mean_scan_coverage=float(support.coverage.mean()),
        mean_sites_per_route=float(support.sites_per_route.mean()),
        all_one_site_scans=int(support.one_site_routes.sum()),
        scans_below_four_over_n=int((support.coverage
                                     < support.n_routes / support.n_sites
                                     - 1e-9).sum()),
        n_scans=13, n_targets=7,
        selected_scan_partners=int(selected.n_partners.sum()),
        partners_per_scan=[int(scan_metrics.n_presynaptic_partners.min()),
                           int(scan_metrics.n_presynaptic_partners.max())],
        detection_threshold_effect=cut,
        observed_selected_scan=dict(mean=float(obs['mean']),
                                    ci95_low=float(obs.ci95_low),
                                    ci95_high=float(obs.ci95_high)),
        ridge_mean_nmse=ridge_mean,
        paired_ridge_contrasts={r.comparator: float(r.other_minus_allridge_mean_nmse)
                                for r in ridge_contrasts.itertuples()},
        source_sha256={str(p.relative_to(J)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in files},
        layout_findings=list(problems) + findings)
    (REC / 'figure_08_sources.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({k: v for k, v in payload.items() if k != 'source_sha256'},
                     indent=2))


if __name__ == '__main__':
    main()
