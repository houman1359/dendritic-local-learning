#!/usr/bin/env python3
"""Main Figure 8: focal shunts and ancestry-defined gains.

Seven panels on one 12-module NativeCanvas at 490 pt (three rows,
122/116/116): A/B/C are the schematic family (intervention pair, the
partition on a real arbor, the diagonal-gain dictionary), D is the
tree-relation result, E/F/G are the localization row.

Waiver (specification D3): row 2 carries three 4-module data panels.  E and
F share one localization-index axis by construction (E keeps the ticks and
the label, F drops the duplicate column); G is the third 4-module panel and
carries its own y (the shunt-minus-injection contrast), which the shared rule
allows only as a recorded exception -- it is the electrical-state claim of
the figure and belongs beside the two localization panels it explains.

Private helpers (the shared library is frozen for per-figure work, errata 7):
``_arbor_blocks`` (Frame.arbor of specification D10 does not exist in
scripts/native_schematics.py), ``ORDINAL_RAMP``/``K_CYCLE`` (D7/D8 lists are
not registered in scripts/journal_style.py) and ``_formula`` (a two-subscript
chain; Frame.subscript carries one subscript per call).
"""
from pathlib import Path
import sys
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
OUT = JOURNAL / 'source_data/shunt_ancestry_gain'
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(JOURNAL / 'code/reconstructed_tree'))

from figure_canvas import (NativeCanvas, Margins, COLORS, PT_TITLE, PT_LABEL,
                           PT_TICK, PT_ANNOT, PT_SMALL, LW_HAIR, LW_EDGE,
                           LW_REF, LW_ERR, LW_DATA, MARKER_MS, ERR_CAPSIZE,
                           SEED_MS, SEED_ALPHA, token_subscript)
from journal_style import (style_direct_color_labels, apply_neurips_style,
                           label_color)
from credit_tree_schematics import mix
from native_schematics import (Frame, reference_line, BADGE_STYLE,
                               _text_w_pt)
from build_journal_figures import mean_ci
import build_main_figure_07 as f7
import build_main_figure_08 as old

INK = COLORS['ink']
MUTE = COLORS['mute']
SHUNT = COLORS['shunting']          # the focal shunting conductance
INJECT = COLORS['point_mlp']        # matched-injection control (never additive)
GHOST = mix('mute', 45)

# D7: subtree / block identity hues.  Never a data series.  The partition of
# a real arbor has five blocks, so the fourth takes D7's K = 5-8 extension
# (the same hue at 62 % ink) rather than plain mute, which the ghosted
# soma-side block already owns.
K_CYCLE = ('dend', 'soma', 'exc', 'mute')
ARBOR_BLOCKS = (COLORS['dend'], COLORS['soma'], COLORS['exc'],
                mix('mute', 62, 'ink'))
# D8: one ordinal ramp for cohort level (8 cells, 45 cells).
ORDINAL_RAMP = (mix('edge', 45), mix('edge', 75), COLORS['edge'])

M_SHUNT, M_INJECT, M_CONTROL, M_CONTRAST = 'o', 's', '^', 'D'
MEAN_MS = MARKER_MS + 1.2

LOCAL_YLIM = (-0.010, 0.248)
LOCAL_YTICKS = (0.0, 0.05, 0.10, 0.15, 0.20)
LOCAL_TICKLABELS = ('0', '0.05', '0.10', '0.15', '0.20')
LOCAL_LABEL = 'localization index'

MEDIAN_ROOT = 864691135409937097    # the median-sized cell, as in Fig. 7A
FOCAL_SEGMENT = 4227                # depth 4, 16 descendants
CATEGORIES = ('descendant', 'sister', 'ancestor', 'depth-matched unrelated',
              'unrelated')
# One line each: a second line of x tick labels under row 1 eats the
# vertical gutter the row-separation audit measures.
CATEGORY_LABELS = ('descendant', 'sister', 'ancestor', 'depth control',
                   'unrelated')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _num(fmt, value):
    return fmt.format(value).replace('-', '−')


# ── exact ancestry gains of the median cell ──────────────────────────────
def ancestry_gains():
    """The exact block gains of the median cell shunted at segment 4227.

    Reads the frozen segment table and the frozen focal dose, rebuilds the
    same passive conductance system the Source Data generator used, and
    returns the ancestry path, the block of every segment and the exact gain
    of every block.  The identity q' = q * (B eta) is asserted here, so the
    numbers printed in panel C are derived, never typed.
    """
    from run_focal_shunting_credit_perturbation import conductance_system
    segments = pd.read_csv(JOURNAL / 'source_data/figure3/segment_metrics.csv')
    cell = segments[segments.root_id.eq(MEDIAN_ROOT)].copy()
    electrical, matrix, _rhs, soma_index, parents, _children = \
        conductance_system(cell, 0.35, 0.35, 1.0, -0.2)
    ids = electrical.segment_id.astype(int).tolist()
    index = {seg: i for i, seg in enumerate(ids)}
    focal = index[FOCAL_SEGMENT]
    # the frozen unit dose of this focal site (figure4/focal_localization.csv)
    table = pd.read_csv(JOURNAL / 'source_data/figure4/focal_localization.csv')
    row = table[table.root_id.eq(MEDIAN_ROOT)
                & table.focal_segment_id.eq(FOCAL_SEGMENT)
                & np.isclose(table.dose, 1.0)
                & table.perturbation.eq('focal shunt')]
    kappa = float(row.delta_conductance.iloc[0])

    green = np.linalg.inv(matrix)
    ancestors = {}

    def route(segment):
        if segment in ancestors:
            return ancestors[segment]
        chain, node = [], int(segment)
        while node != -1:
            chain.append(node)
            node = int(parents.get(node, -1))
        ancestors[segment] = chain[::-1]
        return ancestors[segment]

    path = route(FOCAL_SEGMENT)
    block = {seg: max(set(route(seg)) & set(path), key=path.index)
             for seg in ids}
    gain = {}
    for node in path:
        a = index[node]
        gain[node] = float(1.0 - kappa * green[focal, soma_index]
                           * green[a, focal]
                           / ((1.0 + kappa * green[focal, focal])
                              * green[a, soma_index]))
    # exactness: the shunted adjoint is the baseline adjoint times its block
    shunted = np.linalg.inv(matrix + kappa * np.outer(
        np.eye(len(ids))[focal], np.eye(len(ids))[focal]))
    before = green[:, soma_index]
    after = shunted[:, soma_index]
    predicted = before * np.asarray([gain[block[s]] for s in ids])
    residual = float(np.abs(predicted - after).max())
    assert residual < 1e-12, f'ancestry gain identity broke: {residual}'
    descendants = [s for s in ids if block[s] == FOCAL_SEGMENT
                   and s != FOCAL_SEGMENT]
    assert len(descendants) == 16, len(descendants)
    assert len(path) == 5, path
    return dict(path=path, block=block, gain=gain, kappa=kappa,
                residual=residual, ids=ids,
                n_descendants=len(descendants))


# ── private glyph helpers ────────────────────────────────────────────────
def _formula(f, xy, items, *, size=PT_ANNOT, color=None, ha='left'):
    """Left-to-right chain of plain and (base, sub) tokens; no mathtext."""
    color = INK if color is None else color
    widths = []
    for item in items:
        if isinstance(item, str):
            widths.append(_text_w_pt(f.ax, item, size))
        else:
            widths.append(_text_w_pt(f.ax, item[0], size) + 0.4
                          + _text_w_pt(f.ax, item[1], PT_SMALL))
    total = sum(widths)
    x, y = xy
    if ha == 'center':
        x -= f.fx(total / 2.0)
    elif ha == 'right':
        x -= f.fx(total)
    for item, width in zip(items, widths):
        if isinstance(item, str):
            f.ax.text(x, y, item, fontsize=size, color=color, ha='left',
                      va='center', zorder=6, clip_on=False)
        else:
            token_subscript(f.ax, x, y, item[0], item[1], size=size,
                            sub_size=PT_SMALL, color=color, ha='left',
                            va='center', zorder=6, clip_on=False)
        x += f.fx(width)
    return total


def _badge(ax, xy, kind, *, ha='center', va='center'):
    """Frame.badge on a data axes (Frame would blank the panel's axis)."""
    color, face, edge = BADGE_STYLE[kind]
    return ax.text(xy[0], xy[1], kind, fontsize=PT_SMALL,
                   color=COLORS.get(color, color), ha=ha, va=va, zorder=6,
                   bbox=dict(boxstyle='round,pad=0.28', facecolor=face,
                             edgecolor=edge, linewidth=LW_HAIR))


def _lerp(a, b, t):
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def _arbor_blocks(f, rect, gains, *, tags=True, scale_bar=True):
    """Private Frame.arbor(mode='blocks'): the ancestry partition on the cell.

    Specification D10 asks for a shared ``Frame.arbor``; the helper does not
    exist in the frozen glyph library, so the projection, taper, soma glyph
    and 50 um bar of build_main_figure_07.panel_arbor are reproduced here
    with the K-cycle block strokes of D7 and the shared Frame primitives.
    """
    cell, positions, rows, parent, span_um = f7.morphology_geometry()
    assert int(cell.root_id.iloc[0]) == MEDIAN_ROOT, 'median cell moved'
    block, path = gains['block'], gains['path']
    tag_pt = 11.0 if tags else 0.0
    field = (rect[0], rect[1] + f.fy(tag_pt), rect[2],
             rect[3] - f.fy(tag_pt))
    xy = np.asarray(list(positions.values()), dtype=float)
    to_axes = f7.fit_isotropic(xy, field, f.w_pt, f.h_pt, pad_pt=2.0)
    place = {key: tuple(to_axes(point)[0]) for key, point in positions.items()}
    # block -> stroke colour: descendants, then the three sister blocks
    # outward along the route; the soma-side block is the ghost.
    order = list(reversed(path))               # focal .. soma
    colour = {node: ARBOR_BLOCKS[i] for i, node in enumerate(order[:4])}
    weight = {order[0]: LW_DATA, order[1]: LW_ERR, order[2]: LW_ERR,
              order[3]: LW_ERR}
    for segment in rows:
        p = parent[segment]
        if p not in rows:
            continue
        home = block[segment]
        start, end = place[segment], place[p]
        ghost = home not in colour
        f.ax.plot([start[0], end[0]], [start[1], end[1]],
                  color=GHOST if ghost else colour[home],
                  lw=LW_HAIR if ghost else weight[home],
                  solid_capstyle='round', zorder=1.6 if ghost else 2.4)
    soma_id = min(rows, key=lambda key: rows[key].topological_depth)
    f.soma(place[soma_id])
    f.error_in(place[soma_id], side='left')
    site = place[FOCAL_SEGMENT]
    f.contact(site, kind='inh')
    tag = (0.50, 0.955)
    f.leader(site, (tag[0] - f.fx(1.0), tag[1] - f.fy(2.0)), color=MUTE)
    f.subscript(tag, 'g', 'shunt', size=PT_SMALL, color=COLORS['inh'],
                ha='left', va='center')
    if tags:
        # one tag row under the drawing, left to right in block order, each
        # led back into its own block: the blocks interleave in projection,
        # so a tag placed inside the arbor lands on another block's strokes.
        entries = ((order[0], f"descendants ({gains['n_descendants']})",
                    0.86, ARBOR_BLOCKS[0]),
                   (order[1], 'sister blocks', 0.48, ARBOR_BLOCKS[1]),
                   (path[0], 'soma side', 0.10, MUTE))
        for node, text, x_tag, hue in entries:
            members = [s for s in rows if block[s] == node]
            pts = np.asarray([place[s] for s in members], dtype=float)
            anchor = (float(np.median(pts[:, 0])), float(np.min(pts[:, 1])))
            y_tag = rect[1] + f.fy(tag_pt - 3.0)
            f.leader((x_tag, y_tag + f.fy(2.5)), anchor, color=MUTE)
            f.text((x_tag, y_tag), text, size=PT_SMALL,
                   color=label_color(hue), ha='center', va='top')
    if scale_bar:
        anchor = np.asarray([xy[:, 0].min(), xy[:, 1].min()])
        bar = to_axes(np.vstack([anchor, anchor + [50.0 / span_um, 0.0]]))
        width = float(bar[1, 0] - bar[0, 0])
        x0 = field[0] + f.fx(1.0)
        y0 = field[1] + f.fy(4.0)
        f.ax.plot([x0, x0 + width], [y0, y0], color=INK, lw=LW_DATA,
                  solid_capstyle='butt', zorder=7)
        f.text((x0 + width / 2.0, y0 + f.fy(2.0)), '50 µm', size=PT_SMALL,
               color=INK, va='bottom')


# ── panel A: the two interventions ───────────────────────────────────────
def panel_interventions(ax):
    """Focal shunt versus the current injection that matches its drive."""
    f = Frame(ax)
    bracket_pt = 15.0
    cards = f.split(2, axis='x', gap_pt=12.0,
                    pad_pt=(0.0, 0.0, 0.0, bracket_pt))
    for cell, title, hero in ((cards[0], 'focal shunt', True),
                              (cards[1], 'matched current injection', False)):
        core = f.task_card(cell, title=title, footer='soma V restored',
                           emphasis=hero, tone=None if hero else 'control')
        body = Frame.inset(core, left=0.08, right=0.08, top=0.02)
        # 9 pt under the soma for the two arrows, 12 pt over the canopy for
        # the intervention tag: both cards keep the identical tree geometry.
        body = (body[0], body[1] + f.fy(9.0), body[2],
                body[3] - f.fy(9.0 + 12.0))
        # the same tree and the same shunt site as panel C (schematic brief)
        nodes = f.balanced_tree(body, depth=3, trunk=False, mode='forward',
                                output='z')
        site = _lerp(nodes['JL'], nodes['JLL'], 0.80)
        tag_y = body[1] + body[3] + f.fy(6.0)
        if hero:
            f.contact(site, kind='inh')
            f.subscript((site[0] - f.fx(2.5), site[1] - f.fy(4.0)),
                        'g', 'shunt', size=PT_SMALL, color=COLORS['inh'],
                        ha='right', va='top')
            f.fade(['JLL'], nodes=nodes)
            f.fade([(nodes['JL'], nodes['JLL'])])
            f.text((body[0] + f.fx(1.0), tag_y), 'descendants attenuated',
                   size=PT_SMALL, color=label_color(COLORS['inh']),
                   ha='left', va='center')
        else:
            f.contact(site, kind='inh', active=False)
            width = _formula(f, (body[0] + f.fx(1.0), tag_y),
                             ['κ(', ('E', 'I'), ' − ', ('V', 'k'), ')'],
                             size=PT_SMALL, color=MUTE, ha='left')
            f.arrow((body[0] + f.fx(width + 4.0), tag_y),
                    (site[0] - f.fx(1.6), site[1] + f.fy(2.2)),
                    color=MUTE, lw=LW_EDGE, head=3.4, rad=-0.18)
        f.error_in(nodes.soma, side='right')
        drive = (nodes.soma[0] - f.fx(16.0), nodes.soma[1] - f.fy(3.5))
        f.arrow(drive, (nodes.soma[0] - f.fx(3.0), nodes.soma[1]),
                color=MUTE, lw=LW_EDGE, head=3.4)
        f.subscript((drive[0] - f.fx(1.5), drive[1]), 'I', 'soma',
                    size=PT_SMALL, color=MUTE, ha='right', va='center')
    y = f.fy(bracket_pt - 3.5)
    x0 = cards[0][0] + f.fx(6.0)
    x1 = cards[1][0] + cards[1][2] - f.fx(6.0)
    f.ax.plot([x0, x1], [y, y], color=MUTE, lw=LW_HAIR, zorder=2)
    for x in (x0, x1):
        f.ax.plot([x, x], [y, y + f.fy(2.4)], color=MUTE, lw=LW_HAIR,
                  zorder=2)
    f.text(((x0 + x1) / 2.0, y - f.fy(1.5)), 'baseline focal current matched',
           size=PT_SMALL, color=MUTE, va='top')
    return ax


# ── panel B: the partition on the median reconstruction ──────────────────
def panel_arbor_partition(ax, gains):
    """One gain per ancestry block, drawn on the median reconstructed cell."""
    f = Frame(ax)
    band = f.footer('one gain per block')
    _arbor_blocks(f, (0.0, f.fy(band), 1.0, 1.0 - f.fy(band)), gains)
    return ax


# ── panel C: the diagonal-gain dictionary ────────────────────────────────
def panel_gain_dictionary(ax, gains):
    """B eta: the block indicators carry one exact gain each."""
    f = Frame(ax)
    band = f.footer('driving forces may vary within a block')
    top = 1.0 - f.fy(11.0)
    _formula(f, (f.fx(2.0), top), ['q′ = diag(h) B η'], size=PT_ANNOT,
             ha='left')
    f.ax.text(1.0 - f.fx(1.0), top, 'h = baseline transfer', ha='right',
              va='center', fontsize=PT_SMALL, color=MUTE, zorder=6,
              clip_on=False)
    key_y = top - f.fy(11.0)
    names = ('descendants', 'sister', 'soma side')
    stops = (0.0, 0.40, 0.71)
    for i, text in enumerate(names):
        colour = COLORS[K_CYCLE[i]]
        f.disc((stops[i] + f.fx(2.0), key_y), 2.0, fill=colour)
        f.ax.text(stops[i] + f.fx(5.5), key_y, text, fontsize=PT_SMALL,
                  color=label_color(colour), ha='left', va='center',
                  zorder=6, clip_on=False)

    core_h = key_y - f.fy(6.0) - f.fy(band)
    core = (0.0, f.fy(band), 1.0, core_h)
    tree_rect = (core[0], core[1] + f.fy(11.0), 0.42,
                 core[3] - f.fy(11.0))
    nodes = f.balanced_tree(tree_rect, depth=3, trunk=False, mode='plain')
    blocks = [nodes.subtree('JLL'),
              nodes.subtree('JLR') + ['JL'],
              nodes.subtree('JR') + ['S']]
    f.partition(nodes, blocks, colors=K_CYCLE[:3], labels=[None] * 3)
    site = _lerp(nodes['JL'], nodes['JLL'], 0.80)
    f.contact(site, kind='inh')
    f.subscript((site[0] - f.fx(2.5), site[1] - f.fy(4.0)), 'g', 'shunt',
                size=PT_SMALL, color=COLORS['inh'], ha='right', va='top')
    f.error_in(nodes.soma, side='right')

    # B (8 terminal rows) x eta (one gain per block) = q'/q
    terminals = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
    member = {t: i for i, block in enumerate(blocks) for t in block}
    B = np.zeros((8, 3))
    for r, t in enumerate(terminals):
        B[r, member[t]] = 1.0
    order = list(reversed(gains['path']))
    eta = np.asarray([gains['gain'][order[0]], gains['gain'][order[1]],
                      gains['gain'][gains['path'][0]]])
    groups = [int(B[:, j].sum()) for j in range(3)]
    prod_rect = (core[0] + 0.455, core[1], 0.40, core[3])
    axes = f.dictionary_product(prod_rect, B, eta, cell_pt=8.0,
                                col_colors=K_CYCLE[:3], row_groups=groups,
                                captions=('B', 'η', 'q′/q'),
                                numbers=False)
    for edge in (0.5, 1.5):
        axes[1].axhline(edge, color=COLORS['edge'], lw=LW_HAIR, zorder=4)
    # one printed gain per row block, in clear whitespace with a leader
    field = B @ eta
    edges = np.cumsum([0] + groups)
    host, box = ax.get_position(), axes[2].get_position()
    x_right = (box.x1 - host.x0) / host.width
    y_top = (box.y1 - host.y0) / host.height
    y_bottom = (box.y0 - host.y0) / host.height
    for j in range(3):
        centre = (edges[j] + edges[j + 1]) / 2.0
        y = y_top - (centre / 8.0) * (y_top - y_bottom)
        f.leader((x_right, y), (x_right + f.fx(5.0), y), color=MUTE)
        f.ax.text(x_right + f.fx(6.5), y, f'{field[int(edges[j])]:.3f}',
                  fontsize=PT_ANNOT, color=INK, ha='left', va='center',
                  zorder=6, clip_on=False)
    return ax


# ── panel D: tree-relation selectivity ───────────────────────────────────
def panel_tree_relation(ax):
    """Median |Delta log |gamma|| by relation to the site, at unit dose."""
    category = pd.read_csv(JOURNAL / 'source_data/figure4/category_effects.csv')
    category = category[np.isclose(category.dose, 1.0)]
    values = {}
    for perturbation, color, marker, offset in (
            ('matched additive', INJECT, M_INJECT, -0.15),
            ('focal shunt', SHUNT, M_SHUNT, 0.15)):
        cells = category[category.perturbation.eq(perturbation)].groupby(
            ['root_id', 'category'],
            as_index=False).median_abs_log_gradient_change.mean()
        for index, relation in enumerate(CATEGORIES):
            v = cells[cells.category.eq(relation)] \
                .median_abs_log_gradient_change.to_numpy(float)
            values[(perturbation, relation)] = v
            ax.plot(np.full(v.size, index + offset), v, marker='o', ls='none',
                    ms=SEED_MS, mfc=color, mec='none', alpha=SEED_ALPHA,
                    zorder=2.4)
            m, lo, hi = mean_ci(v, seed=1610 + index)
            ax.errorbar(index + offset, m, yerr=[[m - lo], [hi - m]],
                        marker=marker, ms=MEAN_MS, color=color,
                        markerfacecolor='white', markeredgecolor=color,
                        markeredgewidth=LW_ERR, lw=LW_ERR,
                        capsize=ERR_CAPSIZE, zorder=5)
    unrelated = category[category.perturbation.eq('matched additive')
                         & category.category.eq('unrelated')] \
        .median_abs_log_gradient_change.to_numpy(float)
    assert unrelated.size and np.all(np.abs(unrelated) < 1e-9), \
        "injection 'unrelated' changes are no longer structurally zero"
    ax.set_xticks(range(5), CATEGORY_LABELS)
    ax.tick_params(axis='x', length=0, pad=2.5, labelsize=PT_TICK)
    ax.set_xlim(-0.62, 4.62)
    ax.set_ylim(-0.007, 0.172)
    ax.set_yticks((0.0, 0.05, 0.10, 0.15), ('0', '0.05', '0.10', '0.15'))
    ax.tick_params(axis='y', labelsize=PT_TICK)
    ax.set_ylabel('median |Δ log |γ||', fontsize=PT_LABEL, color=INK)
    ax.text(0.52, 0.152, 'focal shunt', color=SHUNT, fontsize=PT_ANNOT,
            ha='left', va='center', zorder=6)
    ax.text(0.52, 0.126, 'current injection', color=INJECT, fontsize=PT_ANNOT,
            ha='left', va='center', zorder=6)
    ax.text(3.35, 0.030, 'injection off-route:\nzero by construction',
            ha='center', va='bottom', fontsize=PT_SMALL, color=MUTE,
            linespacing=1.15, zorder=6)
    ax.text(0.985, 0.96, 'unit dose, 8 cells', transform=ax.transAxes,
            ha='right', va='top', fontsize=PT_SMALL, color=MUTE)
    return values


# ── panel E: which factor carries the localization ───────────────────────
def panel_factor_freeze(ax):
    """Baseline-adjoint control versus the full shunt, paired within cells."""
    shapley = pd.read_csv(JOURNAL / 'source_data/focal_decomposition'
                          / 'cell_shapley.csv')
    shapley = shapley[shapley.estimand.eq(
        'full_shunt_minus_matched_additive')].sort_values('root_id')
    columns = ('driving_force_only_localization', 'full_shunt_localization')
    colors, markers = (MUTE, SHUNT), (M_CONTROL, M_SHUNT)
    for _, row in shapley.iterrows():
        ax.plot((0, 1), row[list(columns)].to_numpy(float), color=MUTE,
                lw=LW_HAIR, alpha=0.5, zorder=1.6)
    out = {}
    for index, column in enumerate(columns):
        v = shapley[column].to_numpy(float)
        out[column] = v
        ax.plot(np.full(v.size, index), v, marker='o', ls='none', ms=SEED_MS,
                mfc=colors[index], mec='none', alpha=SEED_ALPHA, zorder=2.4)
        m, lo, hi = mean_ci(v, seed=1710 + index)
        ax.errorbar(index, m, yerr=[[m - lo], [hi - m]], marker=markers[index],
                    ms=MEAN_MS, color=colors[index], markerfacecolor='white',
                    markeredgecolor=colors[index], markeredgewidth=LW_ERR,
                    lw=LW_ERR, capsize=ERR_CAPSIZE, zorder=5)
    ax.axhline(0.0, color=MUTE, ls='--', lw=LW_REF, zorder=1.0)
    ax.set_xticks((0, 1), ('baseline\nadjoint', 'post-shunt\nadjoint'))
    ax.tick_params(axis='x', length=0, pad=2.5, labelsize=PT_TICK)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(*LOCAL_YLIM)
    ax.set_yticks(LOCAL_YTICKS, LOCAL_TICKLABELS)
    ax.tick_params(axis='y', labelsize=PT_TICK)
    ax.set_ylabel(LOCAL_LABEL, fontsize=PT_LABEL, color=INK)
    ax.text(0.0, 0.215, 'driving force\nonly', color=label_color(MUTE),
            fontsize=PT_SMALL, ha='center', va='center', linespacing=1.15,
            zorder=6)
    _badge(ax, (0.0, 0.172), 'control')
    ax.text(1.0, 0.215, 'full shunt', color=SHUNT, fontsize=PT_SMALL,
            ha='center', va='center', zorder=6)
    ax.text(0.985, 0.985, '8 cells', transform=ax.transAxes, ha='right',
            va='top', fontsize=PT_SMALL, color=MUTE)
    return out


# ── panel F: dose response ───────────────────────────────────────────────
def panel_passive_dose(ax):
    """Localization against normalized dose, cell-equal-weighted means."""
    focal = pd.read_csv(JOURNAL / 'source_data/figure4/focal_localization.csv')
    per_cell = focal.groupby(['root_id', 'dose', 'perturbation'],
                             as_index=False).localization_index.mean()
    out = {}
    for pindex, (perturbation, color, marker) in enumerate((
            ('matched additive', INJECT, M_INJECT),
            ('focal shunt', SHUNT, M_SHUNT))):
        subset = per_cell[per_cell.perturbation.eq(perturbation)]
        doses, means, lows, highs = [], [], [], []
        for dose, group in subset.groupby('dose'):
            m, lo, hi = mean_ci(group.localization_index.to_numpy(float),
                                seed=1660 + 10 * pindex + int(dose * 4))
            doses.append(float(dose))
            means.append(m)
            lows.append(lo)
            highs.append(hi)
        order = np.argsort(doses)
        doses, means = np.asarray(doses)[order], np.asarray(means)[order]
        lows, highs = np.asarray(lows)[order], np.asarray(highs)[order]
        out[perturbation] = (doses, means)
        ax.errorbar(doses, means, yerr=[means - lows, highs - means],
                    marker=marker, ms=MEAN_MS, lw=LW_DATA, color=color,
                    markerfacecolor='white', markeredgecolor=color,
                    markeredgewidth=LW_ERR, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE, zorder=3)
    ax.set_xscale('log', base=2)
    ax.set_xticks((0.25, 0.5, 1, 2), ('0.25', '0.5', '1', '2'))
    ax.tick_params(axis='x', labelsize=PT_TICK)
    ax.set_xlim(0.215, 2.35)
    ax.set_ylim(*LOCAL_YLIM)
    ax.set_yticks(LOCAL_YTICKS, [''] * len(LOCAL_YTICKS))
    ax.tick_params(axis='y', labelsize=PT_TICK)
    ax.set_xlabel('perturbation dose', fontsize=PT_LABEL, color=INK)
    ax.text(0.27, 0.213, 'focal shunt', color=SHUNT, fontsize=PT_ANNOT,
            ha='left', va='center', zorder=6)
    ax.text(0.27, 0.150, 'current injection', color=INJECT, fontsize=PT_ANNOT,
            ha='left', va='center', zorder=6)
    ax.text(0.985, 0.03, 'shared y with E', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=PT_SMALL, color=MUTE)
    return out


# ── panel G: the electrical boundary ─────────────────────────────────────
COHORTS = (('original_eight', 'initial\n8 cells', ORDINAL_RAMP[1]),
           ('v661_disjoint', 'disjoint 45 cells', ORDINAL_RAMP[2]))
RM_PRIORITY = ('Ra150_Rm300', 'Ra150_Rm15000', 'Ra150_Rm3000',
               'Ra150_Rm1000', 'Ra150_Rm5000', 'Ra150_Rm30000')


def _rm_label(regime):
    value = int(regime.split('Rm')[1])
    return f'{value:,}' if value >= 10_000 else str(value)


def panel_electrotonic(ax):
    """Shunt-minus-injection localization against the electrotonic ratio."""
    physical = pd.read_csv(JOURNAL / 'source_data/physical_cable_sensitivity'
                           / 'cell_primary_contrasts.csv')
    ratio = pd.read_csv(JOURNAL / 'source_data/physical_cable_sensitivity'
                        / 'cell_electrotonic_ratios.csv')
    eligible = physical[['cohort', 'regime', 'root_id']].drop_duplicates()
    ratio = ratio.merge(eligible, on=['cohort', 'regime', 'root_id'],
                        how='inner', validate='one_to_one')
    ratio_mean = ratio.groupby(['cohort', 'regime'], as_index=False) \
        .median_axial_to_leak_ratio.median()
    physical = physical.merge(ratio_mean, on=['cohort', 'regime'],
                              validate='many_to_one')
    ax.set_xscale('log')
    ax.set_xlim(0.95, 190.0)
    ax.set_ylim(-0.0175, 0.106)
    curves, standard = {}, {}
    for cohort, _label, color in COHORTS:
        subset = physical[physical.cohort.eq(cohort)]
        if cohort == 'original_eight':
            subset = subset[subset.regime.str.startswith('Ra150_')]
        points = []
        for (regime, x_value), group in subset.groupby(
                ['regime', 'median_axial_to_leak_ratio']):
            m, lo, hi = mean_ci(group.difference.to_numpy(float),
                                seed=1740 + len(points))
            points.append((float(x_value), m, lo, hi, regime))
        points.sort()
        x = np.asarray([p[0] for p in points])
        means = np.asarray([p[1] for p in points])
        lows = np.asarray([p[2] for p in points])
        highs = np.asarray([p[3] for p in points])
        curves[cohort] = points
        ax.errorbar(x, means, yerr=[means - lows, highs - means],
                    marker=M_CONTRAST, ms=MEAN_MS, lw=LW_DATA,
                    capsize=ERR_CAPSIZE, color=color, markerfacecolor='white',
                    markeredgecolor=color, markeredgewidth=LW_ERR,
                    elinewidth=LW_ERR, zorder=3)
        for px, pm, _lo, _hi, regime in points:
            if regime == 'Ra150_Rm15000':
                standard[cohort] = (px, pm)
        # Rm key: one row per cohort, greedy so no two labels collide.
        keep, taken = [], []
        span = np.log10(190.0 / 0.95)
        axes_w_pt = (ax.get_position().width
                     * ax.get_figure().get_size_inches()[0] * 72.0)
        for regime in RM_PRIORITY:
            match = [p for p in points if p[4] == regime]
            if not match:
                continue
            px = match[0][0]
            width = (_text_w_pt(ax, _rm_label(regime), PT_SMALL)
                     / max(axes_w_pt, 1.0) * span)
            centre = np.log10(px / 0.95)
            if any(abs(centre - c) < (width + w) / 2.0 + 0.035
                   for c, w in taken):
                continue
            taken.append((centre, width))
            keep.append(match[0])
        for px, _pm, _lo, hi, regime in keep:
            label = _rm_label(regime)
            if cohort == 'original_eight':
                ax.text(px, -0.0075, label, ha='center', va='top',
                        fontsize=PT_SMALL, color=MUTE, zorder=6)
            else:
                ax.text(px, hi + 0.0035, label, ha='center', va='bottom',
                        fontsize=PT_SMALL, color=MUTE, zorder=6)
    # The zero reference carries no label here: at four modules the only
    # clear pocket at its right end is already held by the R_m key, and the
    # y = 0 tick names the value.  Caption: dashed line = no contrast.
    reference_line(ax, 0.0, axis='y', label=None)
    ax.set_xticks((1, 10, 100), ('1', '10', '100'))
    ax.set_yticks((0.0, 0.02, 0.04, 0.06, 0.08),
                  ('0', '0.02', '0.04', '0.06', '0.08'))
    ax.tick_params(labelsize=PT_TICK)
    ax.set_xlabel('median axial / leak ratio', fontsize=PT_LABEL, color=INK)
    ax.set_ylabel('localization contrast', fontsize=PT_LABEL, color=INK)
    ax.text(1.02, 0.0135, COHORTS[0][1], color=label_color(COHORTS[0][2]),
            fontsize=PT_SMALL, ha='left', va='center', linespacing=1.15,
            zorder=6)
    ax.text(3.4, 0.0905, COHORTS[1][1], color=label_color(COHORTS[1][2]),
            fontsize=PT_SMALL, ha='left', va='center', zorder=6)
    xs = sorted(value[0] for value in standard.values())
    if len(xs) == 2:
        y_bar = 0.026
        ax.plot(xs, [y_bar, y_bar], color=MUTE, lw=LW_HAIR,
                solid_capstyle='butt', zorder=2)
        for x_value in xs:
            ax.plot([x_value, x_value], [y_bar - 0.002, y_bar + 0.002],
                    color=MUTE, lw=LW_HAIR, zorder=2)
        ax.text(185.0, y_bar + 0.004, 'standard Rₘ', ha='right',
                va='bottom', fontsize=PT_SMALL, color=MUTE, zorder=6)
    return curves


# ── supplementary assets kept by this builder ────────────────────────────
def signed_calibration(ax):
    """Signed shunt responses: broad attenuation versus localization."""
    summary = pd.read_csv(OUT / 'signed_calibration/signed_cohort_summary.csv')
    conditions = [('original_eight', 'Ra150_Rm300', 0),
                  ('original_eight', 'Ra150_Rm15000', 1),
                  ('v661_disjoint', 'Ra150_Rm300', 2.4),
                  ('v661_disjoint', 'Ra150_Rm15000', 3.4)]
    for cohort, regime, x in conditions:
        for category, offset, color, marker in (
                ('descendant', -0.12, SHUNT, M_SHUNT),
                ('depth-matched unrelated', 0.12, MUTE, M_INJECT)):
            row = summary[summary.cohort.eq(cohort) & summary.regime.eq(regime)
                          & summary.perturbation.eq('focal shunt')
                          & summary.category.eq(category)].iloc[0]
            mean = row.mean_signed_log_change
            ax.errorbar(x + offset, mean,
                        yerr=[[mean - row.ci95_low], [row.ci95_high - mean]],
                        color=color, marker=marker, ms=MARKER_MS, mfc='white',
                        lw=LW_ERR, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                        zorder=3)
    ax.axhline(0, color=MUTE, lw=LW_REF, ls=':', zorder=1)
    ax.set_xlim(-0.48, 3.88)
    ax.set_ylim(-0.17, 0.03)
    ax.set_xticks([0, 1, 2.4, 3.4], ['300', '15,000', '300', '15,000'])
    ax.set_yticks([-.15, -.10, -.05, 0],
                  ['−0.15', '−0.10', '−0.05', '0'])
    ax.set_xlabel('Membrane resistance (Ω cm²)', fontsize=PT_LABEL)
    ax.set_ylabel('median Δ log |γ|\nnegative = attenuation',
                  fontsize=PT_LABEL)
    ax.text(.5, .017, 'initial 8 cells', ha='center', fontsize=PT_SMALL)
    ax.text(2.9, .017, 'disjoint 45 cells', ha='center', fontsize=PT_SMALL)
    ax.text(-.12, -.162, 'descendants', color=SHUNT, fontsize=PT_SMALL,
            va='bottom')
    ax.text(2.4, -.162, 'off-route', color=MUTE, fontsize=PT_SMALL,
            va='bottom')
    ax.set_title('Signed gradient change')


def build_signed_calibration():
    """The demoted signed-calibration panel, as a standalone SI asset."""
    dest = OUT / 'figures'
    dest.mkdir(parents=True, exist_ok=True)
    apply_neurips_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    fig.subplots_adjust(left=.22, right=.95, bottom=.24, top=.87)
    signed_calibration(ax)
    style_direct_color_labels(fig)
    fig.savefig(dest / 'signed_calibration_panel.pdf')
    fig.savefig(dest / 'signed_calibration_panel.png', dpi=200)
    plt.close(fig)
    source = OUT / 'signed_calibration/signed_cohort_summary.csv'
    record = dict(
        output_sha256=sha(dest / 'signed_calibration_panel.pdf'),
        source_sha256={str(source.relative_to(JOURNAL)): sha(source)},
        builders_sha256={str(Path(__file__).relative_to(JOURNAL)):
                         sha(Path(__file__))},
        panel='demoted main Figure 8 signed-calibration panel',
        native_width_in=3.6, native_height_in=2.7,
        scientific_data='Unchanged signed category effects at the two frozen '
                        'physical-calibration endpoints; no new outcomes.',
        note='Demoted out of main Figure 8 by the 2026-09-08 figure overhaul; '
             'the manuscript integrator wires it into the supplement.')
    (dest / 'signed_calibration_provenance.json').write_text(
        json.dumps(record, indent=2) + '\n')


def build_normalized_dose():
    """Preserve the original normalized dose curve as Supplementary Fig. S48."""
    dest = OUT / 'figures'
    dest.mkdir(parents=True, exist_ok=True)
    apply_neurips_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    fig.subplots_adjust(left=.20, right=.95, bottom=.24, top=.87)
    old.panel_passive_dose(ax)
    style_direct_color_labels(fig)
    fig.savefig(dest / 'normalized_passive_dose.pdf')
    fig.savefig(dest / 'normalized_passive_dose.png', dpi=200)
    plt.close(fig)
    target = JOURNAL / 'figures/supplementary/figure_S48_normalized_shunt_dose.pdf'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(dest / 'normalized_passive_dose.pdf', target)
    source = JOURNAL / 'source_data/figure4/focal_localization.csv'
    table = pd.read_csv(source)
    counts = table.groupby(['dose', 'perturbation']).agg(
        cells=('root_id', 'nunique'), focal_sites=('focal_segment_id', 'size'))
    assert counts.cells.eq(8).all() and counts.focal_sites.eq(101).all()
    record = dict(
        output_sha256=sha(target),
        source_sha256={str(source.relative_to(JOURNAL)): sha(source)},
        builders_sha256={str(path.relative_to(JOURNAL)): sha(path) for path in
                         [Path(__file__), HERE.parent / 'build_main_figure_08.py',
                          HERE.parent / 'build_journal_figures.py']},
        panel='Supplementary Figure S48', native_width_in=3.6,
        native_height_in=2.7,
        scientific_data='Unchanged previous main Figure 7C normalized '
                        'passive-dose curve; no new outcomes or filtering.',
        doses=[.25, .5, 1., 2.],
        dose_normalization='Added shunt conductance / baseline local (leak + '
                           'excitatory + inhibitory) conductance, excluding '
                           'axial coupling',
        summary_unit='Focal-site localization averaged within each of eight '
                     'cells, then equal-weighted cell means',
        n_focal_sites=101, n_cells=8, n_bootstrap=20000,
        bootstrap_seed='1660 + 10 * perturbation_index + int(dose * 4); '
                       'matched additive index 0, focal shunt index 1')
    (dest / 'normalized_passive_dose_provenance.json').write_text(
        json.dumps(record, indent=2) + '\n')


def build_weak_channel_check():
    """Retain the weak-channel/passive comparison as a supplementary asset."""
    dest = OUT / 'figures'
    active = JOURNAL / 'source_data/focal_selectivity_active_ensemble'
    apply_neurips_style()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.75))
    fig.subplots_adjust(left=.10, right=.985, bottom=.23, top=.85, wspace=.42)
    old.panel_active_dose(axs[0], pd.read_csv(active / 'condition_summary.csv'),
                          old.load_passive_reference())
    old.panel_contrast_forest(axs[1],
                              pd.read_csv(active / 'paired_contrasts.csv'),
                              pd.read_csv(active / 'cell_condition_metrics.csv'))
    axs[0].set_title('Weak-channel linearization vs passive')
    axs[1].set_title('Weak-channel localization contrasts')
    for ax, letter in zip(axs, 'AB'):
        ax.text(-.19, 1.13, letter, transform=ax.transAxes,
                fontweight='bold', fontsize=10.5, va='bottom')
    style_direct_color_labels(fig)
    fig.savefig(dest / 'weak_channel_linearization_check.pdf')
    fig.savefig(dest / 'weak_channel_linearization_check.png', dpi=200)
    plt.close(fig)


# ── the figure ───────────────────────────────────────────────────────────
LETTER_CLEAR_PT = 4.0   # extra clearance a row-mate leaves for the next letter


def _equalize_row(canvas, names, sides, *, clear_pt=0.0):
    """One axes-box width for every same-span panel of a row.

    The reserve lock spends the gutter on whichever panel STARTS at a
    boundary, so a row of three equal spans can still end up with three
    widths (the audit's row-alignment check).  Declaring the shortfall on the
    side that faces the gutter puts them all on the narrowest width, plus
    ``clear_pt`` so the next panel's letter is not pushed onto its neighbour.
    """
    widths = {name: canvas.axes[name].get_position().width * canvas.width_pt
              for name in names}
    target = min(widths.values()) - float(clear_pt)
    index = {'left': 0, 'right': 1}
    for name, side in zip(names, sides):
        held = canvas._locks.get(name, (0.0, 0.0, 0.0, 0.0))[index[side]]
        canvas.declare_reserve(name, **{side: held + widths[name] - target})
    return target


def build():
    dest = OUT / 'figures'
    dest.mkdir(parents=True, exist_ok=True)
    gains = ancestry_gains()
    canvas = NativeCanvas(490 / 72, 3, row_weights=[122, 116, 116],
                          hgutter_pt=37, vgutter_pt=40,
                          margins=Margins(left=49, right=12, top=22,
                                          bottom=34))
    a = canvas.panel('A', 0, 0, 7, schematic=True,
                     title='Focal shunt vs matched current injection')
    b = canvas.panel('B', 0, 7, 5, schematic=True,
                     title='Partition on a real arbor')
    c = canvas.panel('C', 1, 0, 5, schematic=True,
                     title='Diagonal gains per block')
    d = canvas.panel('D', 1, 5, 7, grid='y',
                     title='Shunts change descendant gradients most')
    e = canvas.panel('E', 2, 0, 4, grid='y', title='Transport term dominates')
    g_f = canvas.panel('F', 2, 4, 4, grid='y', title='Contrast grows with dose')
    g = canvas.panel('G', 2, 8, 4, grid='y',
                     title='Cable state sets contrast')
    canvas.fig.canvas.draw()
    # The data panels first: they set the tick and axis labels the reserve
    # lock measures, so the schematic frames below are sized once and final.
    relation = panel_tree_relation(d)
    freeze = panel_factor_freeze(e)
    dose = panel_passive_dose(g_f)
    electro = panel_electrotonic(g)
    canvas.lock_reserves()
    _equalize_row(canvas, ('E', 'F', 'G'), ('left', 'right', 'right'),
                  clear_pt=LETTER_CLEAR_PT)
    # C is a schematic: its tight box IS its axes box, so it has to end far
    # enough left of D's letter, which D's own y label pushes into the gutter.
    canvas.declare_reserve('C', right=11.0)
    canvas.lock_reserves()
    canvas.fig.canvas.draw()

    panel_interventions(a)
    panel_arbor_partition(b, gains)
    panel_gain_dictionary(c, gains)

    style_direct_color_labels(canvas.fig)
    findings = canvas.align_letters()
    if findings:
        print('letter alignment: ' + '; '.join(findings))
    output = dest / 'shunt_ancestry_gain_native.pdf'
    canvas.save(output, name='shunt_ancestry_gain_native', dpi=200)
    plt.close(canvas.fig)
    main = JOURNAL / 'figures/main/figure_08.pdf'
    main.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(output, main)

    sources = [JOURNAL / 'source_data' / folder / name for folder, name in [
        ('figure3', 'segment_metrics.csv'),
        ('figure4', 'category_effects.csv'),
        ('figure4', 'focal_localization.csv'),
        ('focal_decomposition', 'cell_shapley.csv'),
        ('physical_cable_sensitivity', 'cell_primary_contrasts.csv'),
        ('physical_cable_sensitivity', 'cell_electrotonic_ratios.csv'),
        ('shunt_ancestry_gain/signed_calibration', 'signed_cohort_summary.csv'),
        ('shunt_ancestry_gain/signed_calibration', 'signed_cell_effects.csv'),
        ('shunt_ancestry_gain/signed_calibration', 'signed_category_effects.csv'),
        ('shunt_ancestry_gain/signed_calibration', 'validation.json')]]
    numbers = dict(
        ancestry_route=[int(v) for v in gains['path']],
        focal_segment=FOCAL_SEGMENT, median_root=str(MEDIAN_ROOT),
        n_descendants=gains['n_descendants'],
        kappa_unit_dose=gains['kappa'],
        block_gains={str(k): float(v) for k, v in gains['gain'].items()},
        identity_residual=gains['residual'],
        relation_descendant_shunt=float(np.mean(
            relation[('focal shunt', 'descendant')])),
        relation_descendant_injection=float(np.mean(
            relation[('matched additive', 'descendant')])),
        baseline_adjoint_range=[float(freeze[
            'driving_force_only_localization'].min()),
            float(freeze['driving_force_only_localization'].max())],
        full_shunt_range=[float(freeze['full_shunt_localization'].min()),
                          float(freeze['full_shunt_localization'].max())],
        dose_shunt=[float(v) for v in dose['focal shunt'][1]],
        dose_injection=[float(v) for v in dose['matched additive'][1]],
        electrotonic={cohort: [[p[4], p[0], p[1]] for p in points]
                      for cohort, points in electro.items()})
    record = dict(
        output_sha256=sha(output),
        main_figure_sha256=sha(main),
        builders_sha256={str(p.relative_to(JOURNAL)): sha(p) for p in
                         [Path(__file__),
                          HERE.parent / 'build_main_figure_07.py',
                          HERE.parent / 'build_main_figure_08.py',
                          HERE.parent / 'native_schematics.py']},
        source_sha256={str(p.relative_to(JOURNAL)): sha(p) for p in sources},
        panels={
            'A': 'schematic; no data (main.tex:319-324)',
            'B': 'source_data/figure3/segment_metrics.csv (median cell '
                 '864691135409937097, ancestry partition of segment 4227)',
            'C': 'source_data/figure3/segment_metrics.csv + '
                 'source_data/figure4/focal_localization.csv (unit dose kappa)',
            'D': 'source_data/figure4/category_effects.csv',
            'E': 'source_data/focal_decomposition/cell_shapley.csv',
            'F': 'source_data/figure4/focal_localization.csv',
            'G': 'source_data/physical_cable_sensitivity/'
                 'cell_primary_contrasts.csv + cell_electrotonic_ratios.csv'},
        plotted_numbers=numbers,
        scientific_data='D, E, F and G call the same frozen source rows, '
                        'estimators and bootstrap seeds as the archived panel '
                        'functions. B and C are exact consequences of the '
                        'frozen segment table and the frozen unit dose: the '
                        'ancestry gains printed in C are computed from the '
                        'same passive conductance system the Source Data '
                        'generator builds and the partition identity is '
                        'asserted at build time (residual '
                        f"{gains['residual']:.2e}).",
        changes='Rebuilt as seven panels on three rows per the 2026-09-08 '
                'design specification: A adds the intervention pair, B the '
                'partition on the median reconstruction, C the block-gain '
                'dictionary; the signed-calibration panel is demoted to a '
                'standalone supplementary asset. Matched current injection '
                'is drawn in the point-neuron control hue, cohorts on the '
                'ordinal ramp.')
    (dest / 'figure_provenance.json').write_text(
        json.dumps(record, indent=2) + '\n')

    build_signed_calibration()
    build_normalized_dose()
    build_weak_channel_check()


if __name__ == '__main__':
    build()
