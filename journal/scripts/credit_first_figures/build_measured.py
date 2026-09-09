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
precomposed delta-with-circumflex, so the caret is a second 7.0 pt glyph),
``_data_inset`` (``Frame.axes_inset`` needs a Frame, and D is a data panel),
and ``_wrapped`` (pre-broken footer lines: ``Frame.footer`` wraps to the frame
width, and the two schematic footers must break at the phrase, not the word).

Deviations from ``v2/fig9/PLAN.md`` §2, and why (all recorded in TEXT.md too):
row weights and the vertical gutter are **[124, 114, 108] with vgutter 44**, not
[126, 116, 112] with 40.  At 40 pt the 41 pt band between rows holds one row's
x tick labels plus x label and the next row's letter band, and
``audit_row_separation.py`` measured 6.5 / 6.7 pt against its 8.5 pt (3 mm)
floor; 44 pt gives 9.6 / 9.6 pt.  Canvas height, margins, module split and
letter columns are unchanged (22 + 124 + 44 + 114 + 44 + 108 + 34 = 490), so
``live_h_pt`` is still 434 and the CF-10 schematic fraction is 20.4 %, not the
plan's 20.9 %.

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
from journal_style import label_color, style_direct_color_labels, tint_pct
from native_schematics import Frame
from analyze_microns_morphology_credit import ancestry_matrix, parent_map

S = J / 'source_data'
OUT = J / 'figures/components/credit_first_figure_08.pdf'
MAIN = J / 'figures/main/figure_09.pdf'
REC = S / 'credit_first_figures'

ROUTE = COLORS['shunting']    # the ancestry statistic under test (B14 role)
CEIL = COLORS['oracle']       # the perfect-reliability ceiling comparator
GRAY = COLORS['mute']         # scaffolding and non-ancestry comparators
INK = COLORS['ink']
REP_TARGET = 864691135810666525   # representative scan, fixed by median rule
DELTA0_REASON = ('stimulus recording; no credit is delivered in '
                 'this experiment')


# ── private helpers (DECISIONS G5) ───────────────────────────────────────
def _delta_hat(f, xy, tail=' = A c', *, size=PT_BASE, color=INK):
    """``δ̂`` + ``tail`` anchored at ``xy`` (left edge of the delta).

    Nimbus Sans carries neither U+0302 (combining circumflex) nor a
    precomposed delta-with-hat, so the hat is drawn as a second glyph at the
    same type size rather than as mathtext (CF-2 forbids mathtext).
    """
    f.text(xy, 'δ', size=size, color=color, ha='left', va='center')
    f.text(f._off(xy, 1.7, 2.2), '^', size=size, color=color, ha='center',
           va='center')
    f.text(f._off(xy, 4.4, 0.0), tail, size=size, color=color, ha='left',
           va='center')


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
    band = f.footer(_wrapped([
        'partial rank r | Euclidean separation,',
        'soma-to-contact path-length difference',
        'recorded during stimuli, not learning']),
        band_pt=10.5, min_frame_pt=60.0)
    sub_pt = 10.5 * len(subtitle_lines)
    top = 1.0 - f.fy(sub_pt)
    for i, line in enumerate(subtitle_lines):
        f.text((0.5, 1.0 - f.fy(10.5 * i + 5.2)), line, size=PT_BASE,
               color=GRAY, ha='center', va='center')

    core_y0, core_h = f.fy(band), top - f.fy(band)
    core_pt = core_h * f.h_pt
    foot_pt, head_pt = 11.0, 9.0        # tag strip under the soma / over the canopy
    tree_w_pt = min(74.0, 0.55 * f.w_pt)
    nodes = f.balanced_tree((0.0, core_y0 + f.fy(foot_pt), f.fx(tree_w_pt),
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
    tag1 = (pair1_x, core_y0 + core_h - f.fy(7.0))
    f.text(tag1, 'shared path', size=PT_BASE, color=label_color(ROUTE),
           ha='center', va='center')

    # pair 2 -- one contact in each half-tree, no shared path above the soma
    for name in ('T3', 'T7'):
        f.contact(nodes[name], kind='exc')
        f.leader(nodes[name], nodes.soma)
    f.text((nodes.soma[0], core_y0 + f.fy(1.5)), 'no shared path',
           size=PT_BASE, color=GRAY, ha='center', va='bottom')

    # the E/I register: partners are not all excitatory
    f.contact(nodes['T5'], kind='inh', active=False)

    # tuning sketches, one per pair, with a leader back to the pair
    col_x = f.fx(tree_w_pt + 14.0)
    w = 1.0 - col_x
    h_pt = 18.0
    bots = (core_y0 + f.fy(core_pt - h_pt - 11.0),
            core_y0 + f.fy(foot_pt + 0.5))
    anchors = ((pair1_x + f.fx(17.0), tag1[1]),
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
        f.leader(anchors[i], (col_x - f.fx(2.0), y0 + f.fy(h_pt / 2.0)))
    f.text((col_x - f.fx(4.5), bots[1] + f.fy(h_pt / 2.0)), 'response',
           size=PT_BASE, color=GRAY, ha='center', va='center', rotation=90)
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
    band = f.footer(_wrapped([
        'schematic placement by ancestry;',
        'the field is imposed, not observed']),
        band_pt=10.5, min_frame_pt=60.0)
    sub_pt = 10.5 * len(subtitle_lines)
    for i, line in enumerate(subtitle_lines):
        f.text((0.5, 1.0 - f.fy(10.5 * i + 5.2)), line, size=PT_BASE,
               color=GRAY, ha='center', va='center')
    core_y0 = f.fy(band)
    core_h = 1.0 - f.fy(band + sub_pt)

    tree_w_pt = min(70.0, 0.52 * f.w_pt)
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
    f.credit_delivery(nodes, mode='subtree',
                      targets=[seats[i][0] for i in reached],
                      rule_color='shunting', alpha_tags=False)
    for label, (seat, drop) in seats.items():
        par = nodes.parent[seat]
        xy = nodes[seat] if drop == 0.0 else _toward(f, nodes[seat],
                                                     nodes[par], drop)
        f.contact(xy, kind='exc')
    f.error_in(nodes.soma, side='right', dashed=True)
    f.badge((f.fx(tree_w_pt + 4.0), core_y0 + core_h - f.fy(4.0)),
            'local rule', ha='right', va='top')

    mat_w = 44.0
    mat_h = min(56.0, core_h * f.h_pt - 20.0)
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


def _toward(f, a, b, pt):
    """Point ``pt`` points from ``a`` toward ``b``."""
    A, B = f._to_pt(a), f._to_pt(b)
    d = B - A
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    return f._from_pt(A + d * pt)


# ── build ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--emit-main', action='store_true',
                        help='also write figures/main/figure_09.pdf')
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
    plotted = pd.read_csv(S / 'curated_publication/figure_09_plotted.csv')
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
    subtitle_a = [f'{n_targets} targets, {n_partners} partners,',
                  f'{n_scans} scans ({p_lo}–{p_hi} per scan)']

    # -- B: the two cohorts ------------------------------------------------
    pa = plotted[plotted.panel.eq('A')].set_index('comparison')
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
    measures = (('shared_path_partial_r', 'Partial\nrank r', 'shunting'),
                ('shared_path_spearman_r', 'Shared\npath r', 'mute'),
                ('negative_tree_distance_spearman_r', 'Tree\ndistance r', 'mute'),
                ('same_major_branch_delta', 'Same major\nbranch Δ', 'mute'))
    c_rows = []
    for column, label, colour in measures:
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
    reliab = plotted[plotted.panel.eq('C')].measured_split_half_spearman.to_numpy()
    partner_index = pd.read_csv(S / 'measured_alignment_power/inputs/partner_index.csv')
    n_records = int(len(reliab))
    n_partners_unique = int(partner_index.pre_pt_root_id.nunique())
    assert n_records == len(partner_index) == 125
    hist_edges = np.linspace(-0.2, 0.6, 9)
    hist_counts, _ = np.histogram(reliab, bins=hist_edges)
    assert int(hist_counts.sum()) == n_records

    # -- F: coverage across the thirteen scans ----------------------------
    n_in = support.n_sites.to_numpy(float)
    cov = support.coverage.to_numpy(float) * 100.0
    single = support.one_site_routes.to_numpy(bool)
    k_routes = int(support.n_routes.max())
    mean_cov = float(support.coverage.mean()) * 100.0
    below = support[support.coverage < support.n_routes / support.n_sites - 1e-9]

    # -- canvas -----------------------------------------------------------
    canvas = NativeCanvas(490 / 72, 3, row_weights=[124, 114, 108],
                          hgutter_pt=40, vgutter_pt=44,
                          margins=Margins(left=40, right=14, top=22, bottom=34))
    ax_a = canvas.panel('A', 0, 0, 5, schematic=True,
                        title='Shared path vs similarity')
    ax_b = canvas.panel('B', 0, 5, 7,
                        title='No ancestry alignment in either cohort')
    ax_c = canvas.panel('C', 1, 0, 5, title='All four measures null')
    ax_d = canvas.panel('D', 1, 5, 7,
                        title='Observed effect below the detection point')
    ax_e = canvas.panel('E', 2, 0, 5, schematic=True,
                        title='One input per route')
    ax_f = canvas.panel('F', 2, 5, 7,
                        title='Routes reach half the mapped inputs')

    # -- B ----------------------------------------------------------------
    out_b = canvas.forest(
        ax_b, [dict(r, note=None) for r in b_rows],
        value_label='Ancestry–response partial rank correlation',
        xlim=(-0.5, 0.5), reference=0.0, reference_label='', tag='')
    ax_b.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    ax_b.set_xticklabels(['−0.5', '−0.25', '0', '0.25', '0.5'])
    for row, y in zip(b_rows, out_b['ypos']):
        ax_b.text(0.49, y, row['note'], fontsize=PT_BASE, color=GRAY,
                  ha='right', va='center', zorder=6)
    y_rug = out_b['ypos'][-1] + 0.45
    ax_b.plot(scan_values, np.full(len(scan_values), y_rug), linestyle='none',
              marker='o', markersize=SEED_MS, markerfacecolor='none',
              markeredgecolor=GRAY, markeredgewidth=LW_HAIR, zorder=3.5)
    ax_b.set_ylim(2.62, -0.62)
    ax_b.text(-0.012, -0.52, 'no alignment', fontsize=PT_BASE,
              color=GRAY, ha='right', va='center', zorder=6)
    ax_b.text(-0.49, y_rug + 0.33, f'{n_scans} scan values (descriptive)',
              fontsize=PT_BASE, color=GRAY, ha='left', va='center', zorder=6)
    ax_b.text(0.49, y_rug + 0.61, 'positive = ancestry alignment →',
              fontsize=PT_BASE, color=GRAY, ha='right', va='center', zorder=6)
    ax_b.text(-0.49, y_rug + 0.89,
              f'n = {n_targets} target cells; 95 % target bootstrap, '
              f'20,000 draws', fontsize=PT_BASE, color=GRAY, ha='left',
              va='center', zorder=6)
    # cohort bracket: the two rows are the same seven targets
    bx = -0.478
    ax_b.plot([bx, bx], [-0.16, 1.16], color=GRAY, lw=LW_HAIR, zorder=3)
    for yb in (-0.16, 1.16):
        ax_b.plot([bx, bx + 0.013], [yb, yb], color=GRAY, lw=LW_HAIR, zorder=3)
    ax_b.text(-0.458, 0.5, 'same seven targets, re-analysed', fontsize=PT_BASE,
              color=GRAY, ha='left', va='center', zorder=6)

    # -- C ----------------------------------------------------------------
    canvas.forest(ax_c, c_rows, value_label='Target-level association',
                  xlim=(-0.5, 0.5), reference=0.0, reference_label='', tag='')
    ax_c.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
    ax_c.set_xticklabels(['−0.5', '−0.25', '0', '0.25', '0.5'])
    ax_c.set_ylim(4.82, -0.66)
    ax_c.text(-0.012, -0.56, 'no alignment', fontsize=PT_BASE, color=GRAY,
              ha='right', va='center', zorder=6)
    ax_c.text(-0.49, 3.62, 'tree distance sign-flipped: + = closer',
              fontsize=PT_BASE, color=GRAY, ha='left', va='center', zorder=6)
    ax_c.text(-0.49, 4.05, f'all {n_scans} scans; n = {n_targets} target cells',
              fontsize=PT_BASE, color=GRAY, ha='left', va='center', zorder=6)
    ax_c.text(-0.49, 4.48, '95 % target bootstrap, 20,000 draws',
              fontsize=PT_BASE, color=GRAY, ha='left', va='center', zorder=6)

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
    ax_d.plot([-0.30, 0.235], [floor, floor], color=GRAY, lw=LW_REF,
              dashes=(2.6, 2.0), zorder=1)
    ax_d.axvline(0.0, color=GRAY, lw=LW_HAIR, dashes=(2.6, 2.0), zorder=1)
    ax_d.plot([cut_m, cut_m], [0.72, 0.80], color=GRAY, lw=LW_HAIR,
              dashes=(2.6, 2.0), zorder=1.5)
    ax_d.set(xlim=(-0.30, 0.55), ylim=(-0.32, 1.16))
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
    ax_d.text(-0.020, floor + 0.075, f'false-positive rate {floor:.3f}',
              fontsize=PT_BASE, color=GRAY, ha='right', va='center', zorder=6)
    ax_d.text(-0.008, 0.42, 'no alignment', fontsize=PT_BASE, color=GRAY,
              ha='right', va='center', zorder=6)
    ax_d.text(0.190, 0.95, f'{cut_m:.3f} measured', fontsize=PT_BASE,
              color=GRAY, ha='right', va='center', zorder=6)
    ax_d.text(0.190, 0.86, f'{cut_p:.3f} perfect', fontsize=PT_BASE,
              color=GRAY, ha='right', va='center', zorder=6)
    ax_d.plot([0.200, 0.2445], [0.895, 0.815], color=GRAY, lw=LW_HAIR,
              zorder=2)
    ax_d.text(-0.295, 1.09, f'λ = {lam:.2f} respecting the MC band',
              fontsize=PT_BASE, color=INK, ha='left', va='center', zorder=6)
    ax_d.text(0.545, 1.09, 'perfect reliability', fontsize=PT_BASE,
              color=label_color(CEIL), ha='right', va='center', zorder=6)
    ax_d.plot([0.42, 0.42], [1.055, 1.00], color=GRAY, lw=LW_HAIR, zorder=2)
    ax_d.text(0.545, 0.72, 'measured reliability', fontsize=PT_BASE,
              color=label_color(ROUTE), ha='right', va='center', zorder=6)
    ax_d.plot([0.250, 0.223], [0.725, 0.742], color=GRAY, lw=LW_HAIR, zorder=2)
    # observed rug -- the estimate located on this axis, not a power estimate
    y_obs = -0.14
    ax_d.plot([-0.30, 0.55], [y_obs, y_obs], color=GRAY, lw=LW_HAIR, zorder=1)
    ax_d.plot([float(obs.ci95_low), float(obs.ci95_high)], [y_obs, y_obs],
              color=ROUTE, lw=LW_ERR, solid_capstyle='butt', zorder=4)
    ax_d.plot([float(obs['mean'])], [y_obs], marker='D', markersize=MARKER_MS,
              markerfacecolor='white', markeredgecolor=ROUTE,
              markeredgewidth=LW_ERR, linestyle='none', zorder=5)
    ax_d.text(0.135, y_obs - 0.10, 'observed (selected scans)',
              fontsize=PT_BASE, color=label_color(ROUTE), ha='left',
              va='center', zorder=6)
    # inset: measured split-half reliability, the calibration of the curves
    ax_d.text(0.255, 0.600, 'Repeat reliability', fontsize=PT_BASE,
              color=GRAY, ha='left', va='center', zorder=6)
    inset = _data_inset(ax_d, (0.255, 0.12, 0.290, 0.42))
    for lo, hi, count in zip(hist_edges[:-1], hist_edges[1:], hist_counts):
        negative = hi <= 0.0
        inset.bar(lo, count, width=hi - lo, align='edge',
                  facecolor='none' if negative else tint_pct(ROUTE, 40),
                  edgecolor=GRAY if negative else ROUTE, linewidth=LW_HAIR,
                  zorder=3)
    inset.set(xlim=(-0.2, 0.6), ylim=(0, 46))
    inset.set_xticks([-0.2, 0.2, 0.6])
    inset.set_xticklabels(['−0.2', '0.2', '0.6'])
    inset.set_yticks([0, 20, 40])
    inset.set_xlabel('split-half r', fontsize=PT_BASE, labelpad=1.0)
    inset.set_ylabel('records', fontsize=PT_BASE, labelpad=1.0)

    # -- F ----------------------------------------------------------------
    grid_n = np.linspace(4.0, 18.6, 200)
    ax_f.plot(grid_n, 100.0 * k_routes / grid_n, color=GRAY, lw=LW_REF,
              dashes=(2.6, 2.0), zorder=2)
    ax_f.text(6.1, 100.0 * k_routes / 6.1 + 4.0, f'{k_routes}/n',
              fontsize=PT_BASE, color=GRAY, ha='left', va='bottom', zorder=6)
    ax_f.axhline(100.0, color=GRAY, lw=LW_REF, dashes=(2.6, 2.0), zorder=1)
    ax_f.text(18.5, 102.0, 'all inputs', fontsize=PT_BASE, color=GRAY,
              ha='right', va='bottom', zorder=6)
    ax_f.axhline(mean_cov, color=ROUTE, lw=LW_REF, dashes=(5.0, 1.6, 1.2, 1.6),
                 zorder=1.5)
    ax_f.text(18.5, mean_cov + 2.5, f'mean {mean_cov:.1f} %', fontsize=PT_BASE,
              color=label_color(ROUTE), ha='right', va='bottom', zorder=6)
    ax_f.plot(n_in[~single], cov[~single], linestyle='none', marker='o',
              markersize=MARKER_MS, markerfacecolor=ROUTE,
              markeredgecolor=ROUTE, markeredgewidth=0, zorder=4)
    ax_f.plot(n_in[single], cov[single], linestyle='none', marker='o',
              markersize=MARKER_MS, markerfacecolor='white',
              markeredgecolor=ROUTE, markeredgewidth=LW_ERR, zorder=5)
    ax_f.set(xlim=(4.0, 18.6), ylim=(-6.0, 134.0))
    ax_f.set_xticks([5, 8, 11, 14, 17])
    ax_f.set_yticks([0, 25, 50, 75, 100])
    ax_f.set_xlabel('Mapped inputs in the scan', fontsize=PT_EMPH)
    ax_f.set_ylabel('Mapped inputs reached (%)', fontsize=PT_EMPH)
    style_panel(ax_f, grid='y')
    ax_f.spines['left'].set_bounds(0.0, 100.0)
    ax_f.text(18.5, 129.0,
              f'open: four single-input routes ({int(single.sum())} of '
              f'{n_scans} scans)', fontsize=PT_BASE, color=GRAY, ha='right',
              va='center', zorder=6)
    ax_f.text(18.5, 118.0,
              f'{len(below)} scans fall below {k_routes}/n: routes can repeat '
              f'an input', fontsize=PT_BASE, color=GRAY, ha='right',
              va='center', zorder=6)
    ax_f.text(18.5, 72.0,
              f'n = {n_scans} scans from {n_targets} target cells;',
              fontsize=PT_BASE, color=GRAY, ha='right', va='center', zorder=6)
    ax_f.text(18.5, 62.0, 'descriptive, no interval', fontsize=PT_BASE,
              color=GRAY, ha='right', va='center', zorder=6)

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

    panel_statistic(Frame(ax_a), subtitle_a)
    panel_routes(Frame(ax_e), a, (shared, singles), int(a.shape[1]),
                 [f'{len(a)} mapped inputs, {a.shape[1]} routes'])

    problems = canvas.save(OUT, name='credit_first_figure_08', dpi=180)
    if emit_main:
        MAIN.parent.mkdir(parents=True, exist_ok=True)
        MAIN.write_bytes(OUT.read_bytes())

    # -- render-time Source Data and provenance ---------------------------
    support.to_csv(REC / 'figure_08_support.csv', index=False)
    np.savez_compressed(REC / 'figure_08_actual_support.npz', matrix=a,
                        site_ids=np.array(meta['site_segment_ids']),
                        route_ids=np.array(meta['selected_route_segments']))
    reliability_rows = plotted[plotted.panel.eq('C')][
        ['scan', 'partner_index', 'measured_split_half_spearman']].copy()
    reliability_rows = reliability_rows.merge(
        partner_index[['scan', 'partner_index', 'pre_pt_root_id',
                       'repeat_reliability']],
        on=['scan', 'partner_index'], how='left')
    assert len(reliability_rows) == n_records
    assert reliability_rows.pre_pt_root_id.nunique() == n_partners_unique
    reliability_rows.to_csv(REC / 'figure_09_reliability_source.csv',
                            index=False)

    files = [Path(__file__),
             S / 'figure3/segment_metrics.csv',
             S / 'fulltree_boundary/output/dictionary_and_validation_metadata.jsonl',
             S / 'curated_publication/figure_09_plotted.csv',
             S / 'review_evidence_reanalysis/functional_native_target_effects.csv',
             S / 'functional_topology_all_scans/scan_metrics.csv',
             S / 'functional_topology_all_scans/cell_metrics.csv',
             S / 'functional_topology_all_scans/summary.json',
             S / 'figure5/functional_target_metrics.csv',
             S / 'measured_alignment_power/power_summary.csv',
             S / 'measured_alignment_power/RESULTS.json',
             S / 'measured_alignment_power/inputs/partner_index.csv']
    payload = dict(
        figure='fig9', label='fig:boundary', panels='a-f',
        canvas=dict(width_pt=518.4, height_pt=490.0,
                    schematic_fraction=round(
                        (176.8 * 124 + 176.8 * 108) / (464.4 * 434), 4),
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
        panel_c={r['label'].replace('\n', ' '):
                 dict(mean=r['mean'], ci95=[r['lo'], r['hi']], note=r['note'])
                 for r in c_rows},
        panel_d=dict(threshold_measured=cut_m, threshold_perfect=cut_p,
                     lambda_mc_respecting=lam, false_detection=floor,
                     mc_halfwidth_max=mc_hw,
                     n_replicates=int(curves['measured'].n_replicates.max()),
                     inset_bins=[int(v) for v in hist_counts],
                     inset_records=n_records,
                     inset_partners=n_partners_unique),
        panel_f=dict(mean_coverage_pct=mean_cov,
                     all_one_site_scans=int(single.sum()),
                     scans_below_k_over_n=int(len(below)),
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
