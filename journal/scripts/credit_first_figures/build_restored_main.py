#!/usr/bin/env python3
"""Render the restored task-to-credit main figures from immutable evidence.

Only figures/components, figures/provenance and (with --emit-main) figures/main
are written. No experiment, fit, endpoint selection or Source Data file is
changed. Whole-seed intervals are descriptive redraws of existing trajectories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, FixedFormatter
import numpy as np
import pandas as pd

J = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(J/'scripts'))
import build_main_figure_06 as depth
import build_main_figure_07 as anatomy
import build_framework as framework
import build_anatomy as commonmode
import build_measured as measured
import focused_main_evidence as focused
from figure_canvas import (NativeCanvas, Margins, COLORS, PT_LABEL, PT_ANNOT,
                           PT_SMALL, PT_LEGEND, LW_DATA, LW_REF, LW_EDGE,
                           LW_ERR, LW_HAIR, MARKER_MS, style_panel)
from journal_style import style_direct_color_labels
from journal_style import SEED_MS  # QA 2026-09-09: seed-fan marker size in panel F

S=J/'source_data'
OUT=J/'figures/components'
REC=J/'figures/provenance/structure_restoration_20260908'
RULES=('exact','unit_broadcast','calibrated_broadcast')
RC={'exact':COLORS['bp'],'unit_broadcast':COLORS['scalar'],
    'calibrated_broadcast':COLORS['additive']}
RN={'exact':'exact path','unit_broadcast':'unit broadcast',
    'calibrated_broadcast':'calibrated broadcast'}
TN={'matching':'Pairwise','quartet':'Quartic','nested':'Nested'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(S/path, float_precision='round_trip')


def boot(values, seed=210999):
    values=np.asarray(values,float)
    ix=np.random.default_rng(seed).integers(len(values),size=(10000,len(values)))
    means=values[ix].mean(axis=1)
    lo,hi=np.quantile(means,[.025,.975],axis=0)
    return values.mean(axis=0),lo,hi


def save(canvas, number, sources, panels, caption, rows=(), extra=None,
         equalize=True):
    OUT.mkdir(exist_ok=True);REC.mkdir(parents=True,exist_ok=True)
    path=OUT/f'restored_main_{number:02d}.pdf'
    style_direct_color_labels(canvas.fig)
    # Equal module spans keep equal plotting widths despite long forest labels.
    # ``equalize=False`` (design pass 2026-09-14) leaves the builder's own
    # per-column reserves alone: a builder that has already given every
    # same-span panel the same TOTAL reserve, split per column so that the
    # gaps of a shared-axis strip come out equal, must not have the maximum
    # left and the maximum right re-applied to every panel.
    locks=canvas.lock_reserves()
    groups={}
    for rec in canvas._records:
        if not rec.get('schematic'):
            groups.setdefault(rec['colspan'],[]).append(rec['name'])
    for group in groups.values():
        if len(group)<2 or not equalize:continue
        left=max(locks[name][0]for name in group)
        right=max(locks[name][1]for name in group)
        for name in group:canvas.declare_reserve(name,left=left,right=right)
    findings=canvas.save(path,name=f'restored_main_{number:02d}',dpi=180)
    plt.close(canvas.fig)
    pd.DataFrame(rows).to_csv(REC/f'figure_{number:02d}_plotted.csv',index=False)
    helpers=[Path(__file__),Path(focused.__file__),J/'scripts/figure_canvas.py',J/'scripts/journal_style.py']
    if number == 4: helpers.append(J/'scripts/review_completion/noise_panels.py')
    provenance=dict(figure=f'Figure {number}',output=str(path.relative_to(J)),
        output_sha256=sha(path),panel_sources=panels,
        source_sha256={str((S/p).relative_to(J)):sha(S/p) for p in sources},
        builder_sha256={str(p.relative_to(J)):sha(p) for p in helpers},
        source_data_immutable=True,new_experiments=0,layout_findings=findings,
        rendering_scope='Existing outcomes only; no scientific endpoint or rate selection. '
        'Any redraw bootstrap uses whole existing seeds/cells, not independent pairs.')
    if extra:provenance.update(extra)
    (REC/f'figure_{number:02d}.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (REC/f'figure_{number:02d}_caption.md').write_text(caption+'\n')
    print(path,flush=True)


def source_address_gain(ax):
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_axis_off()
    for x,w,label,color in [(.015,.20,'Readout\nloss',COLORS['ink']),
                            (.32,.29,'Neuron error\nδ₀',COLORS['bp']),
                            (.705,.27,'Dendritic\ncredit ε',COLORS['shunting'])]:
        ax.add_patch(FancyBboxPatch((x,.68),w,.245,boxstyle='round,pad=.012',
                     fc=COLORS['panel_bg'],ec=color,lw=LW_EDGE))
        ax.text(x+w/2,.80,label,ha='center',va='center',fontsize=PT_LABEL,color=color)
    for lo,hi in [(.225,.307),(.62,.698)]:
        ax.annotate('',(hi,.80),(lo,.80),arrowprops=dict(arrowstyle='->',lw=LW_DATA,color=COLORS['edge']))
    ax.text(.49,.57,'ε = A Γ c',ha='center',va='center',fontsize=PT_LABEL)
    for x,title,sub in [(.14,'A: address','which sites'),(.50,'Γ: gain','how strongly'),(.84,'c: coefficient','which signal')]:
        ax.text(x,.43,title,ha='center',fontsize=PT_ANNOT)
        ax.text(x,.31,sub,ha='center',fontsize=PT_SMALL,color=COLORS['mute'])
    ax.text(.50,.13,'Update = −learning rate × local eligibility × credit',
            ha='center',fontsize=PT_SMALL)
    ax.text(.50,.015,'Error source and spatial delivery are separate choices',
            ha='center',fontsize=PT_SMALL,color=COLORS['mute'])


def utility(ax,rows):
    variance=np.linspace(0,2,201)
    for k,q,name,color in [(1,.8,'One profile',COLORS['shunting']),
                            (2,1.,'Two profiles',COLORS['bp'])]:
        score=q*q/(q+k*variance)
        ax.plot(variance,score,label=name,color=color,lw=LW_DATA)
        rows.extend(dict(panel='C',noise_variance=float(v),rank=k,captured_energy=q,
                         twice_L_times_bound=float(y)) for v,y in zip(variance,score))
    ax.set(xlim=(0,2),ylim=(0,1.05),xticks=[0,1,2],yticks=[0,.5,1],
           xlabel='Noise variance (illustration)',ylabel='Optimized one-step bound')
    ax.legend(frameon=False,fontsize=PT_SMALL,loc='upper right',handlelength=1.25,
              handletextpad=.4,borderaxespad=.15)


def figure1():
    # Superseded 2026-09-08: main Figure 1 is drawn by
    # scripts/credit_first_figures/build_framework.py.  The body below predates
    # that move and calls framework.read_fresh() with the old three-value
    # contract, so it cannot run; it is kept for the record of the earlier
    # layout, not as a live path.
    raise SystemExit('main Figure 1 is built by scripts/credit_first_figures/'
                     'build_framework.py; this entry point is superseded')
    conditions,seeds,paired=framework.read_fresh();rows=[]
    c=NativeCanvas(535/72,4,row_weights=[113,127,108,55],hgutter_pt=39,vgutter_pt=39,
                   margins=Margins(left=43,right=16,top=24,bottom=35))
    source_address_gain(c.panel('A',0,0,6,schematic=True,title='From a task error to a local update',lock=False))
    framework.dictionaries(c.panel('B',0,6,6,schematic=True,title='Morphology supplies spatial profiles',lock=False))
    utility(c.panel('C',1,0,4,title='Noise and resolution',grid='y'),rows)
    d=c.panel('D',1,4,8,title='MNIST: learning with six credit rules')
    framework.accuracy(d,conditions,seeds)
    for label in d.get_legend().get_texts(): label.set_fontsize(PT_SMALL)
    framework.resolution(c.panel('E',2,0,6,title='Additional resolution within a neuron'),paired,seeds)
    capture,individual_capture=focused.activation_capture(
        read('image_ladder_controls/summaries/delivery_coordinate_capture_summary.csv'),
        read('image_ladder_controls/summaries/delivery_coordinate_capture.csv'),seeds)
    f=c.panel('F',2,6,6,title='More profiles capture more credit',grid='y')
    for architecture,offset in [('shunting',-.035),('additive',.035)]:
        for basis,marker,ls in [('broadcast_k1','^','-'),('subtrees_k3','D','--')]:
            values=capture[capture.architecture.eq(architecture)&capture.basis.eq(basis)]
            values=values.set_index('checkpoint').loc[['initial','trained']]
            mean=values['mean'].to_numpy();low=values.ci_low.to_numpy();high=values.ci_high.to_numpy()
            f.errorbar(np.array([0,1])+offset,mean,yerr=[mean-low,high-mean],
                       color=COLORS[architecture],marker=marker,ls=ls,lw=LW_DATA,
                       ms=MARKER_MS-1,elinewidth=LW_ERR,capsize=2)
    f.set(xlim=(-.17,1.17),ylim=(0,1.025),xticks=[0,1],xticklabels=['Initial','Trained'],
          yticks=[0,.5,1],ylabel='Mean activation-error capture')
    f.legend(handles=[Line2D([],[],color=COLORS['mute'],marker=marker,ls=ls,
                            lw=LW_DATA,ms=MARKER_MS-1,label=label)
                      for marker,ls,label in [('^','-','K = 1'),('D','--','K = 3')]],
             loc='lower left',ncol=2,frameon=False,fontsize=PT_SMALL,
             handlelength=1.5,columnspacing=1)
    legacy=framework.legacy_contrasts()
    framework.legacy_forest(c.panel('G',3,0,12,title='Separate image controls'),legacy)
    rows.extend(dict(panel='D',**r)for r in conditions.to_dict('records'))
    rows.extend(dict(panel='E',**r)for r in paired.to_dict('records'))
    rows.extend(dict(panel='F',record_type='archived summary',**r)for r in capture.to_dict('records'))
    rows.extend(dict(panel='F',record_type='underlying seed',**r)for r in individual_capture.to_dict('records'))
    rows.extend(dict(panel='G',**r)for r in legacy.to_dict('records'))
    sources=['image_ladder_controls/summaries/'+n for n in ('condition_summary_six_rules.csv','fresh_analysis_rows_six_rules.csv','paired_contrasts_six_rules.csv')]
    sources+=['mnist_between_within_factorial/paired_contrasts.csv','cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv']
    sources+=['image_ladder_controls/summaries/'+n for n in
              ('delivery_coordinate_capture_summary.csv','delivery_coordinate_capture.csv','capture_audit.json')]
    panels={'A':'Conceptual factorization, not experimental data. Exact readout errors in the image cohort; alternative source is tested separately.',
      'B':'Actual three-proximal/nine-distal projection dictionaries from build_framework.dictionaries; K1/K3 oracle projection.',
      'C':'Illustrative isotropic-noise projection bound q²/(q+Kσ²), equal smoothness L, fixed orthogonal projections; q=.8,K=1 versus q=1,K=2. Plotted value is 2L times the optimized lower bound. No endpoint-selection prediction.',
      'D':'Six-arm fresh MNIST selected-rate cohort; 10 paired seeds per architecture,180 epochs; existing confidence intervals.',
      'E':'Same-cohort paired K3−projectedK1 and exact−K3 test-accuracy differences.',
      'F':'Fresh10seed/architecture exact-rule selected-rate checkpoints, initial and validation-selected trained states; activation-space mean captured-energy ratio over2048images×128neurons withinseed, excluding only zero fields. Same cohort/coordinates as D/E. K1/K3 oracle projection; archived whole-seed intervals unchanged.',
      'G':'Separate existing MNIST DFA and flattened CIFAR10 exact−neuron-specific contrasts.'}
    caption='''**Task-derived credit separates neuronal identity, spatial address and gain. A,** A readout loss supplies a neuron-specific error; a spatial dictionary A, route gains Γ and coefficients c determine the delivered field ε. A synaptic update multiplies this field by local eligibility and the negative learning rate. **B,** The actual twelve nonsomatic sites of the image model: K=1 broadcasts, K=3 groups each proximal site with its three children, and K=12 spans arbitrary site fields. Projected K=1 and K=3 use oracle coefficients and retain exact somatic errors. **C,** An explicitly illustrative one-step tradeoff: a one-dimensional projection captures q=0.8 of unit gradient energy, whereas a two-dimensional projection captures q=1. Under isotropic projected noise of variance σ², optimizing a smoothness-bound learning rate gives a bound proportional to q²/(q+Kσ²); the ordinate is twice the smoothness constant times that bound. This conditional local statement does not predict the best final trained tree. **D,** MNIST in one 128-neuron dendritic layer with a linear readout. The per-neuron condition broadcasts each neuron's exact somatic activation error. Decoder-only freezes dendritic parameters. Thin lines pair ten fresh seeds per architecture; symbols and bars show means and existing 95% intervals. Rates were selected on three separate development seeds; fits use 180 epochs and validation-selected checkpoints. **E,** Additional within-neuron resolution in the same cohort, with individual paired differences and 95% intervals. **F,** Mean activation-error capture by one or three profiles at the initial and validation-selected exact-rule checkpoints of the same ten fresh seeds per architecture. Each seed averages captured-energy ratios over nonzero fields from 2,048 images and 128 neurons; source intervals are retained. Colors match D; triangles/solid lines denote one profile and diamonds/dashes three profiles. These oracle projections use the same activation coordinates as the tested delivery rules. **G,** Separate historical direct-feedback-alignment (DFA) MNIST cohorts (15 seeds) and flattened CIFAR-10 (20 seeds); colors/shapes match D. pp, percentage points. Detailed one-step derivations, image fields and learning-rate controls remain in the Supplementary Information.'''
    save(c,1,sources,panels,caption,rows,{'helper_sha256':{str(Path(framework.__file__).relative_to(J)):sha(framework.__file__)}})


# ── Figure 4 (fig:prospective) — v2 overhaul build, 2026-09-09 ────────────
# Binding documents: analysis/figure_overhaul_20260908/v2/fig4/PLAN.md and
# AMENDMENTS.md (which overrides the plan where they conflict), DECISIONS.md.
FIG4_NAME = {'matching': 'Pairwise', 'quartet': 'Quartic', 'nested': 'Nested'}
FIG4_FLOOR = {'matching': .0225, 'quartet': .045, 'nested': .0225}
FIG4_SITES = ('JL', 'JR', 'JLL', 'JLR', 'JRL', 'JRR')
FIG4_BADGES = {
    'pairwise': {'JLL': '×', 'JLR': '×', 'JRL': '×',
                 'JRR': '×', 'JL': '+', 'JR': '+', 'J1': '+'},
    'quartic': {'JLL': '×', 'JLR': '×', 'JRL': '×',
                'JRR': '×', 'JL': '×', 'JR': '×', 'J1': '+'}}
FIG4_XLIM = (64, 18000)
FIG4_YLIM = (.012, 2)


def f4_card(f, cell, title, footer, *, title_lines=1, badge=None,
            lift_pt=12.0):
    """Private helper (DECISIONS G5): a task_card whose header is PT_EMPH.

    ``native_schematics.Frame.task_card`` sets its header at PT_ANNOT (7.0);
    CF-2 puts card headers at PT_EMPH (8.0), so the card is composed from the
    library's own ``group`` / ``text`` primitives instead.  ``lift_pt`` raises
    the drawing core above the footer band, because ``Frame.error_in`` puts
    the delta-0 tail 10 pt below the soma and ``balanced_tree`` reserves only
    the soma radius there.  ``badge`` is drawn on the header line, right
    aligned.  ``lift_pt`` is 12.0 because the tag of ``error_in`` sits 7 pt
    below the soma centre while the tree pads only the soma radius under it:
    at 7 pt of lift the five delta-0 tags of A and B overlapped their own card
    footers by 2.6 pt (measured on the first render).  Recommended upstream as
    ``task_card(title_size=, badge=, lift_pt=)``.
    """
    from journal_style import PT_BASE, PT_EMPH
    x0, y0, w, h = cell
    f.group(cell, tint='white', edge=COLORS['grid'], lw=LW_HAIR, radius_pt=3.0)
    top_pt = 10.5 * title_lines + 2.5
    bot_pt = 10.5 * (footer.count('\n') + 1) + 1.5 if footer else 0.0
    f.text((x0 + f.fx(4.5), y0 + h - f.fy(2.5)), title, size=PT_EMPH,
           color=COLORS['ink'], ha='left', va='top', linespacing=1.15)
    if badge:
        f.badge((x0 + w - f.fx(3.5), y0 + h - f.fy(3.0)), badge, ha='right',
                va='top')
    if footer:
        f.text((x0 + w / 2.0, y0 + f.fy(bot_pt * .5)), footer, size=PT_BASE,
               color=COLORS['mute'], linespacing=1.15)
    return (x0, y0 + f.fy(bot_pt + lift_pt), w,
            h - f.fy(top_pt + bot_pt + lift_pt))


def f4_operator_badges(f, nodes, badges):
    """Private helper (DECISIONS G5): operator badges at every junction.

    ``Frame.balanced_tree(badges=...)`` drops the badges whenever the widest
    sibling x gap is under 9.5 pt, which at a 72 pt card is always; that
    check is a single x-gap proxy that ignores the junctions' very different
    heights (the four depth-2 junctions are 12 pt apart here and J1 sits 15 pt
    below its nearest neighbour).  The glyph itself is the library's own
    ``Frame._junction_badge``.  Recommended upstream as a
    ``badge_min_pitch_pt`` keyword on ``balanced_tree``.
    """
    for name, text in badges.items():
        if name in nodes:
            f._junction_badge(nodes[name], text, r_pt=3.3)


def f4_targets_panel(ax):
    """Panel A — two targets, one shared tree."""
    from journal_style import PT_BASE
    from native_schematics import Frame
    f = Frame(ax)
    f.text((.5, 1.0 - f.fy(4.0)), 'same tree, weights and examples',
           size=PT_BASE, color=COLORS['mute'], ha='center', va='top')
    f.text((.5, .0), '× product     + sum     root sums at the soma',
           size=PT_BASE, color=COLORS['mute'], ha='center', va='bottom')
    cells = f.split(2, axis='x', gap_pt=10.0)
    y0, hh = f.fy(11.5), 1.0 - f.fy(23.0)
    spec = [('Pairwise', 'four pair products', 'pairwise'),
            ('Quartic', 'two quartet products', 'quartic')]
    for cell, (title, foot, kind) in zip(cells, spec):
        cell = (cell[0], y0, cell[2], hh)
        core = f4_card(f, cell, title, foot, badge='teacher')
        nodes = f.balanced_tree(
            core, depth=3, mode='forward', trunk=True, output='y',
            input_labels=[('x', str(i + 1)) for i in range(8)])
        f4_operator_badges(f, nodes, FIG4_BADGES[kind])
        # balanced_tree already labels the input span above the canopy.
        # Do not repeat x1/x8 beside the same contacts at this small scale.
        f.error_in(nodes.soma, side='right', label='δ0')
    f.require_soma_lowest()
    f.require_delta0()
    return f


def f4_bus(f, nodes, targets, colour, radii):
    """Private helper (DECISIONS G5): the broadcast bus, clear of the contacts.

    ``Frame.credit_delivery(mode='neuron')`` pins its rail 8 pt above the
    highest TARGET junction (unit y 2.32 here) while the terminals of the
    same tree reach unit y 3.06, so on a 74 pt card the amber rail lands
    INSIDE the excitatory contact row -- it bisected four of the eight
    contacts -- and its riser, dropped at the leftmost target's abscissa,
    crossed the outermost leaf branch.  The rule register may not overprint
    the anatomy register (CF-4), so this helper redraws the identical glyph
    (one rail, a hairline drop into every target, the source dot at this
    soma) with three corrections: the rail sits above the whole canopy
    (highest contact + one contact radius + 1.6 pt), the riser sits outside
    the leftmost contact, and rail, riser and drops are drawn BEHIND the
    anatomy (zorder 1.55, under ``dendrite`` at 2 and ``contact`` at 4.5),
    so the six drops pass behind the contact discs instead of through them.
    ``radii`` maps each target to its drop-disc radius in points (equal on
    the unit-broadcast card, the mean absolute calibrated weight on the
    calibrated-broadcast card).  Recommended upstream as ``bus_lift_pt`` and
    ``drop_radii`` keywords on ``credit_delivery``.
    """
    from native_schematics import CONTACT_DIA_PT
    fp = f._to_pt
    targets = list(targets)
    term = [fp(nodes[t]) for t in nodes.terminals]
    site = [fp(nodes[t]) for t in targets]
    soma = fp(nodes.soma)
    c_r = CONTACT_DIA_PT * .5 * f.scale
    bus_y = max(p[1] for p in term) + c_r + 1.6
    x_lo = min([p[0] for p in term] + [p[0] for p in site]) - c_r - 1.6
    x_hi = max(p[0] for p in site)
    zb = 1.55
    p0, p1 = f._from_pt((x_lo, bus_y)), f._from_pt((x_hi, bus_y))
    f.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=COLORS[colour],
              lw=f.lw(LW_HAIR), solid_capstyle='round', zorder=zb)
    for (x, y) in site:
        f.arrow(f._from_pt((x, bus_y)), f._from_pt((x, y + 2.8)),
                color=COLORS[colour], lw=LW_HAIR, head=2.6, zorder=zb)
    foot = (soma[0] - (nodes.soma_r_pt * f.scale + 2.2), soma[1])
    chain = [f._from_pt(foot), f._from_pt((x_lo, foot[1])),
             f._from_pt((x_lo, bus_y))]
    f.ax.plot([q[0] for q in chain], [q[1] for q in chain],
              color=COLORS[colour], lw=f.lw(LW_HAIR), solid_capstyle='round',
              solid_joinstyle='round', zorder=zb)
    f.disc(f._from_pt(foot), 1.6, fill=COLORS[colour], zorder=6.0)
    for name, r_pt in radii.items():
        # a 0.55 pt white keyline, so the drop disc separates from the open
        # junction ring it lands on
        f.disc(nodes[name], r_pt, fill=COLORS[colour], edge='white',
               lw=LW_HAIR, zorder=6.2)


def f4_delivery_panel(ax, radii):
    """Panel B — where each rule delivers credit."""
    from native_schematics import Frame
    f = Frame(ax)
    cells = f.split(3, axis='x', gap_pt=9.0)
    y0, hh = f.fy(13.0), 1.0 - f.fy(13.0)
    spec = [('Exact\npath', 'per-site q(x)'),
            ('Unit\nbroadcast', 'one shared error at six sites'),
            ('Calibrated\nbroadcast', 'step 0, 256 examples')]
    for i, (cell, (title, foot)) in enumerate(zip(cells, spec)):
        cell = (cell[0], y0, cell[2], hh)
        core = f4_card(f, cell, title, foot, title_lines=2)
        nodes = f.balanced_tree(core, depth=3, mode='forward', trunk=True,
                                output='y', labels=False)
        if i == 0:
            f.credit_delivery(nodes, mode='exact', targets=list(FIG4_SITES),
                              alpha_tags=True, rule_color='bp')
        elif i == 1:
            # QA 2026-09-10 (major): the same 'neuron' bus glyph, drawn by
            # f4_bus so the rail clears the excitatory contact row.  An equal
            # drop disc on every site makes the six recipients countable and
            # leaves card 3 differing only by its weighted radii.
            f4_bus(f, nodes, FIG4_SITES, 'scalar',
                   {site: 1.3 for site in FIG4_SITES})
        else:
            f4_bus(f, nodes, FIG4_SITES, 'additive', radii)
        f.error_in(nodes.soma, side='right', label='δ0')
    # No badge column: with the three canonical names the badges push the
    # strip onto a second line, which the 104 pt card row cannot spare.
    f.rule_key((0.0, 0.0, 1.0, f.fy(12.0)),
               [('Exact path', 'bp', 'exact'),
                ('Unit broadcast', 'scalar', 'neuron'),
                ('Calibrated broadcast', 'additive', 'neuron')], size=7.0)
    f.require_soma_lowest()
    f.require_delta0()
    return f


def f4_badge(ax, x, y, kind, *, ha='right', va='top'):
    """The library badge glyph on a DATA axes, in axes coordinates.

    ``native_schematics.Frame`` cannot be used here: constructing a Frame
    rewrites the host axes' limits to 0-1, which a data panel cannot survive.
    The style tuple is the library's own ``BADGE_STYLE`` entry, so the badge
    is identical to the ones panels A and B draw.
    """
    from journal_style import PT_BASE, label_color
    from native_schematics import BADGE_STYLE
    key, face, edge = BADGE_STYLE[kind]
    colour = COLORS[key]
    try:
        colour = label_color(colour, background=face)
    except ValueError:
        pass
    return ax.text(x, y, kind, transform=ax.transAxes, fontsize=PT_BASE,
                   color=colour, ha=ha, va=va, zorder=7,
                   bbox=dict(boxstyle='round,pad=0.28,rounding_size=0.286',
                             facecolor=face, edgecolor=edge, linewidth=LW_HAIR))


def f4_title(ax, text, *, accent=None, sub=(), right_sub=None, pad=13.5):
    """Panel title at PT_EMPH, its task-colour accent rule and sub-title.

    The sub-title sits above the axes rather than inside it: at four modules
    the only in-axes whitespace in C--E is where the reference labels and the
    direct labels already are (checked on the first render).
    """
    from journal_style import PT_BASE, PT_EMPH
    ax.set_title(text, fontsize=PT_EMPH, color=COLORS['ink'], loc='left',
                 pad=pad, fontweight='normal')
    if accent is not None:
        # offset_copy keeps the accent a fixed number of POINTS above the axes
        # top: the reserve lock resizes the panel after this call, so an axes
        # fraction computed here would drift onto the sub-title.
        from matplotlib.transforms import offset_copy
        tr = offset_copy(ax.transAxes, fig=ax.figure, x=0.0, y=pad - 2.5,
                         units='points')
        ax.plot([0.0, 0.13], [1.0, 1.0], transform=tr, color=accent,
                lw=LW_DATA, clip_on=False, zorder=6, solid_capstyle='butt')
    base = 3.0 + (9.5 if right_sub else 0.0)
    for i, line in enumerate(sub):
        ax.annotate(line, xy=(0.0, 1.0), xycoords='axes fraction',
                    xytext=(0.0, base + 9.5 * (len(sub) - 1 - i)),
                    textcoords='offset points', ha='left', va='bottom',
                    fontsize=PT_BASE, color=COLORS['mute'],
                    annotation_clip=False)
    if right_sub:
        ax.annotate(right_sub[0], xy=(1.0, 1.0), xycoords='axes fraction',
                    xytext=(0.0, 3.0), textcoords='offset points',
                    ha='right', va='bottom', fontsize=PT_BASE,
                    color=right_sub[1], annotation_clip=False)
    return ax


def f4_band(ax, x, lo, hi, color):
    """A 16 % tint area mark with a 0.55 pt same-tint edge (CF-3)."""
    from journal_style import tint_patch
    poly = list(zip(x, hi)) + list(zip(x[::-1], lo[::-1]))
    return tint_patch(ax, ('poly', poly), color=color, pct=16, edge=True,
                      lw=LW_HAIR, zorder=1.2, clip_on=True)


def f4_reference(ax, value, *, axis, label, label_x=None):
    """CF-7 reference line: dashed COLORS['mute'] at LW_REF, labelled on it.

    QA 2026-09-10 (minor): the label is now set RIGHT-ALIGNED ON the line, as
    CF-7 requires set-wide, and the figure-local waiver that put it at the
    free left end is withdrawn.  It goes BELOW the rule: in C, D and E every
    curve converges onto the noise floor from above and nothing is ever drawn
    under it (the floor is the smallest value in the panel), and in G the
    three cumulative curves reach 1.00 above the 0.95 rule.  ``label_x``
    right-aligns the label at a data abscissa short of the line's right end
    when that end is taken.
    """
    from journal_style import PT_BASE
    if axis == 'y':
        lo, hi = ax.get_xlim()
        line, = ax.plot([lo, hi], [value, value], color=COLORS['mute'],
                        lw=LW_REF, zorder=1.0, solid_capstyle='butt')
        if label:
            ax.annotate(label, xy=(hi if label_x is None else label_x, value),
                        xytext=(-2.0, -1.6),
                        textcoords='offset points', fontsize=PT_BASE,
                        color=COLORS['mute'], ha='right', va='top', zorder=5)
    else:
        lo, hi = ax.get_ylim()
        line, = ax.plot([value, value], [lo, hi], color=COLORS['mute'],
                        lw=LW_REF, zorder=1.0, solid_capstyle='butt')
        if label:
            ax.annotate(label, xy=(value, 0.0),
                        xycoords=('data', 'axes fraction'),
                        xytext=(2.0, 1.5), textcoords='offset points',
                        fontsize=PT_BASE, color=COLORS['mute'], ha='left',
                        va='bottom', zorder=5)
    line.set_dashes((2.2, 1.8))
    return line


def f4_curve(ax, data, task, panel, rows, *, seed_layer=False):
    """C / D / E — held-out NMSE against updates, one task per panel."""
    from matplotlib.ticker import FixedLocator, NullFormatter
    for rule in RULES:
        p = data[data.task.eq(task) & data.model.eq('algebraic')
                 & data.optimizer.eq('adam') & data.selected_rate
                 & data.rule.eq(rule) & data.step.ge(64)]
        w = p.pivot(index='seed', columns='step',
                    values='test_nmse').sort_index()
        assert w.shape == (20, 64) and not w.isna().any().any()
        steps = w.columns.to_numpy(float)
        if seed_layer and rule == 'exact':
            for seed, trace in w.iterrows():
                ax.plot(steps, trace.to_numpy(), color=COLORS['bp'],
                        lw=LW_HAIR, alpha=.28, zorder=2.0,
                        solid_capstyle='round')
                rows.extend(dict(panel=panel, record='seed trajectory',
                                 task=FIG4_NAME[task], series=RN[rule],
                                 seed=int(seed), x_name='training updates',
                                 x=int(s), value=float(v), unit='test NMSE',
                                 endpoint='fixed checkpoint')
                            for s, v in trace.items())
        mean, lo, hi = boot(w.to_numpy())
        f4_band(ax, steps, lo, hi, RC[rule])
        line, = ax.plot(steps, mean, color=RC[rule], lw=LW_DATA, zorder=3.0,
                        solid_capstyle='round')
        if rule == 'unit_broadcast':
            line.set_dashes((2.6, 1.6))
        rows.extend(dict(panel=panel, record='curve summary',
                         task=FIG4_NAME[task], series=RN[rule],
                         x_name='training updates', x=int(s), unit='test NMSE',
                         mean=float(m), ci95_low=float(l), ci95_high=float(h),
                         n_seeds=20, interval='95 % percentile bootstrap',
                         endpoint='fixed checkpoint')
                    for s, m, l, h in zip(steps, mean, lo, hi))
    ax.set(xscale='log', yscale='log', xlim=FIG4_XLIM, ylim=FIG4_YLIM)
    ax.set_xlabel('Training updates', fontsize=PT_LABEL, color=COLORS['ink'])
    ax.xaxis.set_major_locator(FixedLocator([64, 1024, 16384]))
    ax.set_xticklabels(['64', '1,024', '16,384'])
    ax.xaxis.set_minor_locator(
        FixedLocator([2 ** k for k in range(7, 15) if 2 ** k != 1024]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([.02, .1, 1]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    floor = FIG4_FLOOR[task]
    f4_reference(ax, 1024, axis='x', label=None)   # the x tick names it
    f4_reference(ax, floor, axis='y', label=f'noise {floor:g}')
    rows.append(dict(panel=panel, task=FIG4_NAME[task], series='noise floor',
                     x_name='expected label-noise NMSE', unit='test NMSE',
                     mean=float(floor), n_seeds=20, interval='none',
                     endpoint='analytic'))
    return ax


def f4_forest_strip(ax, *, extra_rows, n_rows, key=()):
    """A blank strip below the last forest row, holding F's endpoint key.

    CF-6's footer tag and CF-6's per-row n both have to fit a 105 pt axes, so
    the tag is set above the axes (right-aligned, with the panel title) and
    this strip carries only the two-token endpoint key: two markers with their
    names, one per line -- direct labels, not a legend artist (CF-5).
    """
    from journal_style import PT_BASE
    ax.set_ylim(n_rows - .4 + extra_rows, -.6)
    if not key:
        return
    pitch = extra_rows / (len(key) + .7)
    x0, x1 = ax.get_xlim()
    for i, (mfc, name) in enumerate(key):
        y = n_rows - .4 + pitch * (.85 + i)
        # QA 2026-09-11 (minor): at x0 + 0.02 the ink key diamond abutted the
        # dashed 0.0 reference rule (0.3 pt clear) and read as a bullet on
        # the rule.  The tokens now sit on the rule's right, ~4 pt clear.
        x = x0 + .12 * (x1 - x0)
        ax.plot([x], [y], marker='D', ms=MARKER_MS - .8, mfc=mfc,
                mec=COLORS['ink'] if mfc == 'white' else 'white',
                mew=LW_HAIR, ls='none', zorder=4.0, clip_on=False)
        ax.annotate(name, xy=(x, y), xycoords=('data', 'data'),
                    xytext=(4.5, 0.0), textcoords='offset points',
                    ha='left', va='center', fontsize=PT_BASE,
                    color=COLORS['mute'], annotation_clip=False)


def f4_deficit(ax, contrast, seeds, rows):
    """Panel F — the predefined quartic-minus-pairwise deficit, four budgets."""
    from figure_canvas import forest as forest_rows
    key = 'calibrated_broadcast minus exact interaction'
    base = contrast[contrast.task.eq('quartet_minus_matching')
                    & contrast.optimizer.eq('adam')
                    & contrast.metric.eq('test_nmse')
                    & contrast.rate_view.eq('selected_rate')
                    & contrast.contrast.eq(key)]
    fan = seeds[seeds.task.eq('quartet_minus_matching')
                & seeds.optimizer.eq('adam') & seeds.metric.eq('test_nmse')
                & seeds.rate_view.eq('selected_rate') & seeds.contrast.eq(key)]
    budgets = [1024, 4096, 8192, 16384]
    entries, second = [], []
    for budget in budgets:
        end = base[base.budget.eq(budget) & base.endpoint.eq('terminal')]
        val = base[base.budget.eq(budget)
                   & base.endpoint.eq('validation_selected')]
        assert len(end) == 1 and len(val) == 1
        end, val = end.iloc[0], val.iloc[0]
        assert int(end.positive_seeds) == 20 and int(val.positive_seeds) == 20
        draw = fan[fan.budget.eq(budget)
                   & fan.endpoint.eq('terminal')].difference.to_numpy()
        assert len(draw) == 20
        rows.extend(dict(panel='F', record='paired seed difference',
                         task='Quartic minus pairwise',
                         series='calibrated broadcast minus exact path',
                         seed=int(r.seed), x_name='budget (training updates)',
                         x=int(budget), value=float(r.difference),
                         unit='test NMSE difference', endpoint='endpoint state')
                    for r in fan[fan.budget.eq(budget)
                                 & fan.endpoint.eq('terminal')].itertuples())
        entries.append(dict(label=f'{budget:,}', mean=float(end['mean']),
                            lo=float(end.ci95_low), hi=float(end.ci95_high),
                            n=20, marker='D', fan=list(map(float, draw))))
        second.append((float(val['mean']), float(val.ci95_low),
                       float(val.ci95_high)))
        for state, rec in (('endpoint state', end),
                           ('validation-selected state', val)):
            rows.append(dict(panel='F', task='Quartic minus pairwise',
                             series='calibrated broadcast minus exact path',
                             x_name='budget (training updates)', x=int(budget),
                             unit='test NMSE difference', endpoint=state,
                             mean=float(rec['mean']),
                             ci95_low=float(rec.ci95_low),
                             ci95_high=float(rec.ci95_high), n_seeds=20,
                             positive_seeds=int(rec.positive_seeds),
                             interval='95 % paired percentile bootstrap'))
    # QA 2026-09-10 (minor): carmine is the figure's 'exact path' rule colour
    # (panel B's key); F's marks are an endpoint STATE, not a rule, so they
    # are drawn in neutral ink and the three rule hues keep one meaning.
    # The upper bound drops 1.56 -> 1.22: the widest drawn value is the
    # 1.116 seed, so a third of the old axis carried nothing.
    out = forest_rows(ax, entries, value_label='Interaction-dependent deficit D', reference=0.0,
                      reference_label='no deficit', color='ink',
                      xlim=(-.06, 1.22), tag='')
    # Keep the subscript at the 7 pt floor used by the other figure labels.
    ax.annotate('int', xy=(1, 0), xycoords=ax.xaxis.label,
                xytext=(.3, -1.6), textcoords='offset points', fontsize=7,
                ha='left', va='bottom', annotation_clip=False)
    # QA 2026-09-09: the twenty-seed fan sits 0.22 rows BELOW the row (the
    # validation-selected arm is 0.22 above), and the endpoint interval gets
    # a white casing over the fan, so both intervals stay readable.
    rng = np.random.default_rng(4)
    for y, entry in zip(out['ypos'], entries):
        fan = np.asarray(entry['fan'])
        ax.plot(fan, y - .22 + rng.uniform(-.05, .05, len(fan)), ls='none',
                marker='o', ms=SEED_MS, mfc=COLORS['mute'], mec='none',
                alpha=.55, zorder=2.0)
        ax.plot([entry['lo'], entry['hi']], [y, y], color='white',
                lw=LW_ERR + 1.6, zorder=3.3, solid_capstyle='butt')
        ax.plot([entry['lo'], entry['hi']], [y, y], color=COLORS['ink'],
                lw=LW_ERR, zorder=3.4, solid_capstyle='butt')
    for y, (m, lo, hi) in zip(out['ypos'], second):
        yy = y + .22
        ax.plot([lo, hi], [yy, yy], color=COLORS['ink'], lw=LW_ERR, zorder=3.0,
                solid_capstyle='butt')
        for xb in (lo, hi):
            ax.plot([xb, xb], [yy - .09, yy + .09], color=COLORS['ink'],
                    lw=LW_ERR, zorder=3.0, solid_capstyle='butt')
        ax.plot([m], [yy], marker='D', ms=MARKER_MS - .8, mfc='white',
                mec=COLORS['ink'], mew=LW_HAIR, ls='none', zorder=4.0)
    # Design pass 2026-09-14: no in-plot key strip -- the caption names the
    # filled (endpoint) and open (validation-selected) diamonds -- so the four
    # rows take the whole box; the per-row 'n = 20' tags went on 2026-09-10.
    f4_forest_strip(ax, extra_rows=.22, n_rows=len(entries))
    return out


F4_G_YTOP = 1.24   # panel G ylim top; see the QA 2026-09-11 note in f4_energy


def f4_energy(ax, eigen, diag, rows, ramp):
    """Panel G — cumulative captured path-field energy against k."""
    from journal_style import PT_BASE
    from matplotlib.ticker import FixedLocator
    field = eigen[eigen.field.eq('path_q') & eigen.rule.eq('exact')
                  & eigen.optimizer.eq('adam') & eigen.selected_rate]
    kk = np.arange(1, 7, dtype=float)
    style = {'matching': ('o', True), 'quartet': ('s', False),
             'nested': ('^', True)}
    uni_x = -0.92                 # the 'uniform' tick is 30 pt wide: at
    # QA 2026-09-11 (minor): xlim left -1.50 -> -1.58 and the category tick
    # -0.95 -> -0.92 so the 'uniform' tick label keeps ~2.8 pt from the '0'
    # y-tick label (it had 1.2 pt; the leftmost uniform cap stays ~3 pt
    # inside the spine).  ylim top 1.16 -> 1.24 so the opaque 'oracle' badge
    # on its top-right seat clears the dodged k = 5 and k = 6 marks instead
    # of covering four of them.
    ax.set(xlim=(-1.58, 6.55), ylim=(0, F4_G_YTOP))   # x = 0 it ran into '1'
    # QA 2026-09-10 (major): quartic and nested cumulative means differ by at
    # most 0.030 at every k, so at a shared abscissa the two markers, their
    # bands and their curves print as one ribbon.  Every series is dodged by
    # a fifth of the k spacing -- inside the uniform category too, where the
    # three collapsed onto one blob -- so all three are separately countable.
    dodge = {'matching': -0.20, 'quartet': 0.0, 'nested': 0.20}
    uni_dodge = {'matching': -0.30, 'quartet': 0.0, 'nested': 0.30}
    for task, colour in ramp.items():
        cut = field[field.family.eq(task) & field.step.eq(1024)]
        w = cut.pivot_table(index='seed', columns='index', values='fraction')
        assert len(w) == 20
        cum = w.cumsum(axis=1).to_numpy()[:, :6]
        mean, lo, hi = boot(cum)
        kx = kk + dodge[task]
        f4_band(ax, kx, lo, hi, colour)
        marker, filled = style[task]
        curve, = ax.plot(kx, mean, color=colour, lw=LW_DATA, marker=marker,
                         ms=MARKER_MS - 1.2, mfc=colour if filled else 'white',
                         mec=colour, mew=LW_HAIR, zorder=3.0)
        if task == 'nested':
            # QA 2026-09-09 (minor): quartic #10908C and nested #0E6C70 are
            # adjacent ORDINAL_RAMP entries and their curves differ by under
            # 0.03 at every k, so nested is separated by STROKE as well as by
            # marker -- the same dash idiom unit broadcast uses in C-E.
            curve.set_dashes((2.6, 1.6))
        uni = diag[diag.task.eq(task) & diag.step.eq(1024)]
        assert len(uni) == 20
        um, ul, uh = boot(uni.path_uniform_oracle_capture.to_numpy(), 771009)
        ax.errorbar([uni_x + uni_dodge[task]], [um],
                    yerr=[[um - ul], [uh - um]],
                    color=colour,
                    marker=marker, ms=MARKER_MS - 1.2, lw=0.0,
                    elinewidth=LW_ERR, capsize=2.0, capthick=LW_ERR,
                    mfc=colour if filled else 'white', mec=colour,
                    mew=LW_HAIR, zorder=3.0)
        assert abs(mean[0]
                   - float(uni.path_best_rank_one_capture.mean())) < 1e-9
        rows.append(dict(panel='G', task=FIG4_NAME[task],
                         series='uniform profile capture', x_name='category',
                         x=0, x_label='uniform (not fitted)',
                         unit='captured path-field energy',
                         mean=float(um), ci95_low=float(ul),
                         ci95_high=float(uh), n_seeds=20,
                         interval='95 % percentile bootstrap',
                         endpoint='fixed checkpoint (1,024 updates)'))
        rows.extend(dict(panel='G', task=FIG4_NAME[task],
                         series='cumulative oracle capture',
                         x_name='best fitted directions k', x=int(k),
                         unit='captured path-field energy', mean=float(m),
                         ci95_low=float(l), ci95_high=float(h), n_seeds=20,
                         interval='95 % percentile bootstrap',
                         endpoint='fixed checkpoint (1,024 updates)')
                    for k, m, l, h in zip(kk, mean, lo, hi))
    init = field[field.family.eq('matching') & field.step.eq(0)].pivot_table(
        index='seed', columns='index', values='fraction').cumsum(axis=1)
    other = field[field.family.eq('quartet') & field.step.eq(0)].pivot_table(
        index='seed', columns='index', values='fraction').cumsum(axis=1)
    assert np.max(np.abs(init.to_numpy() - other.to_numpy())) < 1e-12
    ref = init.mean(axis=0).to_numpy()[:6]
    line, = ax.plot(kk, ref, color=COLORS['mute'], lw=LW_REF, zorder=1.6)
    line.set_dashes((2.2, 1.8))
    rows.extend(dict(panel='G', task='Pairwise and quartic',
                     series='initial spectrum (shared)',
                     x_name='best fitted directions k', x=int(k),
                     unit='captured path-field energy', mean=float(m),
                     n_seeds=20, interval='none', endpoint='step 0')
                for k, m in zip(kk, ref))
    ax.set_xlabel('Best fitted directions k', fontsize=PT_LABEL,
                  color=COLORS['ink'])
    ax.set_ylabel('Path-field energy captured (fraction)', fontsize=PT_LABEL,
                  color=COLORS['ink'])
    ax.xaxis.set_major_locator(FixedLocator([uni_x, 1, 2, 3, 4, 5, 6]))
    ax.set_xticklabels(['uniform\nnot fitted', '1', '2', '3', '4', '5', '6'])
    ax.yaxis.set_major_locator(FixedLocator([0, .5, 1]))
    ax.set_yticklabels(['0', '0.5', '1.0'])
    ax.plot([0.0, 0.0], [0, F4_G_YTOP], color=COLORS['grid'], lw=LW_HAIR,
            zorder=.9, solid_capstyle='butt')
    f4_reference(ax, .95, axis='y', label='0.95')
    # QA 2026-09-09 (minor, ratification requested): these three are direct
    # labels (annotation artists; no legend frame, no handles, no legend
    # artist) set at the right end of the curve bundle, each in its own
    # series colour, with its marker glyph on the curve side.  They are not
    # placed ON their own curves because the quartic and nested cumulative
    # means differ by at most 0.03 at every k (0.013 at k = 1), so no
    # position and no leader can separate them: identity is carried by the
    # marker shape and, since this build, by the nested dash pattern.  Under
    # CF-5 this is the figure's only key-shaped object outside B's sanctioned
    # footer strip and is recorded as such in TEXT.md section 4.
    # Pairwise is a true on-curve direct label: its mean is flat at 1.00 and
    # the band above it is empty, so the word sits over its own curve with no
    # marker token.  Only the two overlapping teals keep a marker.
    ax.annotate(FIG4_NAME['matching'], xy=(1.12, 1.035), ha='left',
                va='bottom', fontsize=PT_BASE, color=ramp['matching'])
    for task, y in (('quartet', .62), ('nested', .50)):
        ax.annotate(FIG4_NAME[task], xy=(5.95, y), ha='right', va='center',
                    fontsize=PT_BASE, color=ramp[task])
        marker, filled = style[task]
        ax.plot([6.25], [y], marker=marker, ms=MARKER_MS - 1.2, ls='none',
                mfc=ramp[task] if filled else 'white', mec=ramp[task],
                mew=LW_HAIR, zorder=4.0, clip_on=False)
    # QA 2026-09-10 (minor): the two lines that printed the 16,384-update
    # k = 1 captures (1.00 / 0.40 / 0.40) are deleted -- they were plotted
    # VALUES set as prose inside the plot box, and they are carried by the
    # caption and by figure_04_plotted.csv instead.  Only the identification
    # of the grey dashed reference stays on the panel.
    ax.annotate('dashed grey: initial (shared)',
                xy=(.55, .33), ha='left', va='top', fontsize=PT_BASE,
                color=COLORS['mute'])
    for task in ('matching', 'quartet', 'nested'):
        late = diag[diag.task.eq(task) & diag.step.eq(16384)]
        rows.append(dict(panel='G', task=FIG4_NAME[task],
                         series='best rank-one profile capture',
                         x_name='training updates', x=16384,
                         unit='captured path-field energy',
                         mean=float(late.path_best_rank_one_capture.mean()),
                         n_seeds=20, interval='none',
                         endpoint='fixed checkpoint (16,384 updates)'))
    return ax


def f4_shuffle(ax, contrast, endpoints, rows, ramp):
    """Panel H — leaf-assignment controls and their within-seed family average."""
    from figure_canvas import forest as forest_rows
    key = 'shuffled minus compatible: exact'
    picked = endpoints[endpoints.rule.eq('exact')
                       & endpoints.optimizer.eq('adam')]
    fans = {}
    for family in ('matching', 'quartet', 'nested'):
        w = picked[picked.family.eq(family)].pivot_table(
            index='seed', columns='structure', values='test_nmse')
        assert len(w) == 20
        fans[family] = (w['assignment_shuffled'] - w['compatible'])
    fans['all'] = sum(fans.values()) / 3.0
    entries = []
    for family, label in (('matching', 'Pairwise'), ('quartet', 'Quartic'),
                          ('nested', 'Nested'), ('all', 'Family average')):
        rec = contrast[contrast.optimizer.eq('adam')
                       & contrast.structure.eq('paired')
                       & contrast.contrast.eq(key)
                       & contrast.family.eq(family)]
        assert len(rec) == 1
        rec = rec.iloc[0]
        assert int(rec.positive_seeds) == 20 and int(rec.n_seeds) == 20
        assert abs(float(rec['mean']) - float(fans[family].mean())) < 1e-9
        rows.extend(dict(panel='H', record='paired seed difference',
                         task=label, series='shuffled minus compatible, exact path',
                         seed=int(seed), x_name='task family', value=float(value),
                         unit='test NMSE difference',
                         endpoint='endpoint state (1,024 updates)')
                    for seed, value in fans[family].items())
        # QA 2026-09-09 (major): the fan is NOT handed to forest(), which
        # would draw it in the row's own ORDINAL_RAMP hue at nearly the mean
        # marker's size and hide the estimand (the Quartic interval is only
        # 0.010 wide).  It is drawn below, in COLORS['mute'], exactly as
        # panel F draws its own fan.
        entries.append(dict(label='Family\naverage' if family == 'all' else label,
                            mean=float(rec['mean']),
                            lo=float(rec.ci95_low), hi=float(rec.ci95_high),
                            fan=list(map(float, fans[family])), n=20,
                            # QA 2026-09-11 (minor): the pooled row was a
                            # filled dark diamond, the same glyph as F's
                            # (now ink) 'endpoint' marker one panel to the
                            # left; a hollow diamond would repeat F's open
                            # 'validation-selected' token instead.  The pool
                            # is a filled plus, a glyph no other panel uses.
                            marker='P' if family == 'all' else 'o',
                            color=ramp.get(family, COLORS['mute'])))
        rows.append(dict(panel='H', task=label,
                         series='shuffled minus compatible, exact path',
                         x_name='task family', unit='test NMSE difference',
                         mean=float(rec['mean']), ci95_low=float(rec.ci95_low),
                         ci95_high=float(rec.ci95_high), n_seeds=20,
                         positive_seeds=20,
                         interval='95 % paired percentile bootstrap',
                         endpoint='endpoint state (1,024 updates)'))
    out = forest_rows(ax, entries,
                      value_label='Shuffled − compatible\nexact NMSE',
                      reference=0.0, reference_label='no cost',
                      xlim=(-.04, 1.05), tag='')
    # QA 2026-09-09 (major): the twenty-seed fan in COLORS['mute'] 0.22 rows
    # BELOW its row and the interval bar re-drawn over a white casing, so the
    # row-coloured mean marker is the only saturated mark on the row -- the
    # same construction panel F uses.
    from journal_style import PT_BASE
    rng = np.random.default_rng(11)
    for y, entry in zip(out['ypos'], entries):
        fan = np.asarray(entry['fan'])
        # QA 2026-09-10 (major): +-0.04 rows of random jitter left the twenty
        # paired seeds printing as about five resolvable disks.  The seeds are
        # now ranked and dealt onto five fixed levels spanning 0.32 rows, so
        # value-neighbours never share a level and all twenty are countable;
        # the small random component only breaks ties within a level.
        order = np.argsort(fan, kind='stable')
        cycle = np.array([2.0, 0.0, 4.0, 1.0, 3.0])   # not 0,1,2,3,4: a
        level = np.empty(len(fan))                    # monotone cycle draws
        level[order] = cycle[np.arange(len(fan)) % 5]  # a false diagonal
        ax.plot(fan, y - .25 + (level - 2.0) * .08
                + rng.uniform(-.012, .012, len(fan)), ls='none',
                marker='o', ms=SEED_MS, mfc=COLORS['mute'], mec='none',
                alpha=.45, zorder=2.0, clip_on=True)
        col = entry['color']
        ax.plot([entry['lo'], entry['hi']], [y, y], color='white',
                lw=LW_ERR + 1.6, zorder=3.3, solid_capstyle='butt')
        ax.plot([entry['lo'], entry['hi']], [y, y], color=col, lw=LW_ERR,
                zorder=3.4, solid_capstyle='butt')
        for xb in (entry['lo'], entry['hi']):
            ax.plot([xb, xb], [y - .13, y + .13], color=col, lw=LW_ERR,
                    zorder=3.4, solid_capstyle='butt')
        # QA 2026-09-11: the pooled plus is drawn 1.6 pt larger and without
        # the white casing edge -- at 4.6 pt a plus's arms are 1.5 pt wide
        # and a white hairline edge left them printing as a broken cross.
        plus = entry['marker'] == 'P'
        ax.plot([entry['mean']], [y], ls='none', marker=entry['marker'],
                ms=MARKER_MS + (1.6 if plus else 0.0), mfc=col,
                mec=col if plus else 'white', mew=0.0 if plus else LW_HAIR,
                zorder=4.2)
        # QA 2026-09-10 (minor): the Quartic interval is 0.010 wide, which at
        # this axes is narrower than the mean marker, so that row would show a
        # bare dot while the other three show bars and could be misread as
        # carrying no interval.  Any row whose bar is shorter than the marker
        # prints its interval directly under the marker instead.
        # QA 2026-09-10 (minor): the Quartic interval is 0.010 wide, narrower
        # than its own mean marker, so that row used to be the ONLY one
        # carrying a printed interval and it sat at its own height.  Every
        # row now prints its interval on the same seat, 0.30 rows under its
        # marker, so the four rows read as one column and the four redundant
        # 'n = 20' tags (the count is in the panel's right sub-title) go.
        ax.annotate(f"[{entry['lo']:.3f}, {entry['hi']:.3f}]",
                    xy=(entry['mean'], y + .32), ha='center', va='center',
                    fontsize=PT_BASE, color=COLORS['mute'], zorder=4.2)
    ax.plot([-.04, 1.05], [2.5, 2.5], color=COLORS['grid'], lw=LW_HAIR,
            zorder=1.2, solid_capstyle='butt')
    f4_forest_strip(ax, extra_rows=.22, n_rows=len(entries))
    return out


FIG4_CAPTION = r'''\caption{\textbf{Higher-order interactions expose limits of fixed credit profiles despite matched input sensitivities.}
\textbf{A}, Pairwise and quartic targets combine products of two or four inputs on compatible seven-unit scalar trees. Junction badges mark interactions; $y,\delta_0$ denote student output and error. Units are multi-affine, not conductance-based; both tasks use squared-error loss.
\textbf{B}, Credit over six nonsomatic sites: exact path $\bm q(x)$, unit broadcast (amber) and calibrated broadcast (blue). Marker size represents mean absolute calibrated weights, 0.133--0.386; signs are mixed. Other elements are schematic.
\textbf{C,D}, Held-out normalized mean squared error (NMSE) against updates for pairwise and quartic targets at rule-specific rates; \textbf{D} includes all twenty exact trajectories. Dashed lines indicate label-noise floors and the 1,024-update primary checkpoint.
\textbf{E}, Noise-control comparison: quartic-minus-pairwise difference in calibrated-minus-exact clean-domain NMSE at 16,384 updates for fixed-absolute noise, no noise and variance-matched relative noise, which share initialization, inputs and standardized noise within seeds. Diamonds, inherited selected rates; circles, inherited common Adam rate 0.003; dots, paired seed differences; whiskers, 95\% bootstrap intervals. Rates were inherited from the original comparison, not retuned for each noise condition.
\textbf{F}, Quartic-minus-pairwise difference in calibrated-broadcast-minus-exact NMSE across training budgets. Dots, paired seeds; filled diamonds, endpoints; open diamonds, validation-selected states.
\textbf{G}, Exact-trained path-field energy captured by the best $k$ oracle directions; uniform broadcast is a separate unfitted category. Grey dashed line, shared pairwise/quartic initialization spectrum; horizontal dashed line, 95\% capture. The nested mean is dashed; series are offset horizontally for visibility. States are from 1,024 updates.
Each cohort has twenty paired seeds. \textbf{C,D,F,G} reuse the original cohort; \textbf{E} uses twenty new seeds. Bands/whiskers are 95\% seed-bootstrap intervals, paired for differences. Curves use fixed checkpoints; \textbf{F} adds validation selection. Nested and leaf-shuffle displays remain in Supplementary Figs.~S17A and S13C.
Source Data: \texttt{source\_data/curated\_publication/figure\_04\_plotted.csv}.}'''


def figure4():
    """Main Figure 4, fig:prospective — seven panels A-G on one native canvas.

    CF-1 canvas 518.4 x 490.0 pt, aspect 1.058, height on the 340/415/490
    ladder.  CF-10 schematic_fraction = 25.3 % on the AMENDMENTS B12 formula
    (A 171.5 pt + B 254.9 pt over a 118 pt row against the live area
    463.4 x 430 pt); schematic waiver: none.  The left margin is 40 pt, not
    the plan's 58 pt: at 58 pt the strict audit fails `fill-width` (content
    fills 90.7 % of the canvas, gate 92 %).  The fraction is unchanged to a
    tenth of a point by that substitution (25.2 % at 58 pt).  waiver D3: row 2 (F seed strip / G energy curve)
    is two 6-module panels that share no axis (2026-09-21: the former H, the
    common-rate noise panel, merged into E); each is a different estimand
    and the row is column-locked.  CF-5: zero legend
    artists -- the only sanctioned key in the nine-figure set is the frameless
    rule key inside Fig 5C, and B's rule-key strip is a schematic panel's
    footer, not an in-axes key.  CF-4 DELTA0_EXEMPTIONS: one, panel E's
    morphology icon ("morphology icon in a data panel; credit delivery is
    drawn in panel B").
    CF-7 (QA 2026-09-10): no waiver.  Every horizontal reference label --
    'noise 0.0225' (C, E), 'noise 0.045' (D) and '0.95' (G) -- is right
    aligned ON its own rule, set below it, where nothing else is ever drawn
    (the floor is the smallest value in C-E and the three cumulative curves
    reach 1.00 above G's 0.95 rule); D's 'Exact path' direct label moved from
    below the red curve to the empty band above it to free that end.  F's
    'no deficit' and H's 'no cost' are placed by figure_canvas.forest itself,
    right-aligned at the top of the zero rule.  The vertical 1,024 rule
    carries no label because the major x tick already prints 1,024.
    CF-5 PENDING (not approved): G's two-token direct-label block (the words
    'Quartic' and 'Nested', each with its own series marker; annotation
    artists only, no legend artist, no frame, no handles) is submitted for
    ratification and stands as pending until a judge rules.  Measured
    justification: the quartic and nested cumulative means differ by at most
    0.030 at every k (k = 5; 0.013 at k = 1), which is 2.2 pt on this axes,
    so no on-curve position and no leader can separate the two curves;
    identity is carried by marker shape plus the nested dash pattern.
    Panel B rail clearance (QA 2026-09-10, fixed here; the library keyword is
    still recommended upstream): the 'neuron' bus of Frame.credit_delivery
    sits 8 pt above the highest TARGET junction (unit y 2.32) while the
    terminals reach unit y 3.06, so on a 74 pt card the rail landed inside
    the excitatory contact row and bisected four of the eight contacts, and
    its riser crossed the outermost leaf branch.  Cards 2 and 3 are therefore
    drawn by the private f4_bus, which puts the rail above the whole canopy,
    the riser outside the leftmost contact, and rail, riser and drops behind
    the anatomy (zorder 1.55, under dendrite 2 and contact 4.5), so the rule
    register never overprints the anatomy register.  bus_lift_pt and
    drop_radii keywords on credit_delivery remain the upstream follow-up.
    Private helpers under DECISIONS G5 (library
    follow-ups): f4_card, f4_operator_badges, f4_bus, f4_badge,
    f4_title, f4_reference, f4_forest_tag.
    """
    from journal_style import PT_BASE, ORDINAL_RAMP, tint_patch
    from native_schematics import Frame
    from matplotlib.ticker import FixedLocator

    data = read('credit_rule_extension/summaries/all_curves.csv')
    contrast = read('credit_rule_extension/summaries/paired_contrasts.csv')
    seedfan = read('credit_rule_extension/summaries/paired_seed_contrasts.csv')
    profile = read('credit_rule_bridge/figures/initial_profile_source.csv')
    floors = read('credit_rule_bridge/summaries/'
                  'task_variance_and_noise_floor.csv')
    eigen = read('credit_resolution_bridge/capture/'
                 'matched_bridge_eigenvalues.csv')
    spectra = read('credit_resolution_bridge/capture/'
                   'matched_bridge_spectra.csv')
    morph = read('morphology_credit/summaries/fresh/paired_contrasts.csv')
    ends = read('morphology_credit/summaries/fresh/selected_endpoints.csv')
    diag = read('credit_rule_extension/summaries/all_diagnostics.csv')
    diag = diag[diag.model.eq('algebraic') & diag.optimizer.eq('adam')
                & diag.selected_rate & diag.rule.eq('exact')
                & diag.state.eq('own_checkpoint')]
    for task, floor in FIG4_FLOOR.items():
        row = floors[floors.model.eq('algebraic') & floors.task.eq(task)]
        assert abs(float(row.expected_label_noise_nmse.iloc[0]) - floor) < 1e-12
    original = read('credit_rule_bridge/summaries/all_diagnostics.csv')
    original = original[original.model.eq('algebraic')
                        & original.optimizer.eq('adam')
                        & original.selected_rate & original.rule.eq('exact')
                        & original.state.eq('own_checkpoint')
                        & original.step.isin([0, 1024])]
    joined = original.merge(diag, on=['seed', 'task', 'step', 'model',
                                      'optimizer', 'rule', 'rate', 'state'],
                            suffixes=('_original', '_extension'),
                            validate='one_to_one')
    assert len(joined) == 120
    replay = float(np.max(np.abs(
        joined.path_best_rank_one_capture_original
        - joined.path_best_rank_one_capture_extension)))
    assert replay < 5e-16
    rank95 = spectra[spectra.field.eq('path_q') & spectra.rule.eq('exact')
                     & spectra.optimizer.eq('adam') & spectra.selected_rate
                     & spectra.step.eq(1024)].groupby('family').rank95.mean()
    assert round(float(rank95['matching']), 2) == 1.00

    ramp = {'matching': ORDINAL_RAMP[1], 'quartet': ORDINAL_RAMP[2],
            'nested': ORDINAL_RAMP[3]}
    lo_p = float(profile.mean_abs_profile.min())
    hi_p = float(profile.mean_abs_profile.max())
    radii = {row.junction: 1.10 + 1.30 * (row.mean_abs_profile - lo_p)
             / (hi_p - lo_p) for row in profile.itertuples()}
    assert set(radii) == set(FIG4_SITES)

    rows = [dict(panel='B', task='both targets',
                 series='calibrated broadcast profile',
                 x_name='nonsomatic site', x_label=row.junction,
                 unit='mean |calibrated weight| over seeds',
                 mean=float(row.mean_abs_profile),
                 signed_mean=float(row.mean_profile), n_seeds=20,
                 interval='none', endpoint='step 0 calibration',
                 drop_radius_pt=float(radii[row.junction]))
            for row in profile.itertuples()]

    # Design pass 2026-09-14: 24 + 122 + 30 + 100 + 30 + 114 + 36 = 456 pt.
    # The schematic row keeps 122 pt (B's 7-module card row sits at aspect
    # 2.34 under the 2.40 cap); the two data rows are sized to
    # their marks (the lock pass carves ~14 pt off the top of row 2 for
    # row 1's x labels and row 2's letters, so its axes are ~96 pt like
    # row 1's).  Only neutral task names remain above C-E; the
    # caption states the frozen rates, the three stalled quartic
    # seeds, the depth-four control, the 20/20 positive deficits and the
    # 1,024-update oracle checkpoint, and every series is direct-labelled.
    c = NativeCanvas(468 / 72, 3, row_weights=[122, 100, 114], hgutter_pt=30,
                     vgutter_pt=36,
                     margins=Margins(left=40, right=5, top=24, bottom=36))
    a = c.panel('A', 0, 0, 5, title=None, schematic=True, lock=False,
                inset_pt=(10, 10, 8, 10))
    f4_title(a, 'Two targets on one scalar tree')
    f4_targets_panel(a)
    b = c.panel('B', 0, 5, 7, title=None, schematic=True, lock=False,
                inset_pt=(10, 10, 8, 10))
    f4_title(b, 'Where each rule delivers credit')
    f4_delivery_panel(b, radii)

    # Neutral task names identify the three conditions without reinstating
    # claim-style headlines or the removed statistics paragraphs.
    task_labels = {'matching': 'Pairwise', 'quartet': 'Quartic',
                   'nested': 'Nested control'}
    curve_axes = {}
    for letter, col, task in (('C', 0, 'matching'), ('D', 4, 'quartet')):
        ax = c.panel(letter, 1, col, 4, title=task_labels[task], grid='none')
        curve_axes[task] = ax
        if task == 'nested':
            tint_patch(ax, ('rect', FIG4_XLIM[0], FIG4_YLIM[0],
                            FIG4_XLIM[1] - FIG4_XLIM[0],
                            FIG4_YLIM[1] - FIG4_YLIM[0]), color='mute', pct=16,
                       edge=True, lw=LW_HAIR, radius_pt=2.0, zorder=0.0,
                       clip_on=True, transform=ax.transData)
        f4_curve(ax, data, task, letter, rows, seed_layer=(task == 'quartet'))
        # QA 2026-09-10 (major): C, D and E share one log NMSE axis, so D and
        # E now carry the same three tick labels; without them no value could
        # be read off either panel.  The rotated 'Held-out NMSE' title stays on C
        # alone (the axis is shared and the row is column-locked).
        ax.set_yticklabels(['0.02', '0.1', '1'])
        if letter == 'C':
            ax.set_ylabel('Held-out NMSE', fontsize=PT_LABEL, color=COLORS['ink'])
    d = curve_axes['quartet']
    d.yaxis.set_minor_locator(FixedLocator([.045]))
    # QA 2026-09-10 (minor, CF-7): the right end of the floor rule now
    # carries its own 'noise 0.045' label, so the direct label slides left
    # along the same clear strip under the rule -- where the exact bundle it
    # names has been sitting since 256 updates (the seeds dip at most 0.66 pt
    # below the rule) -- instead of taking that end for itself.
    d.annotate('Exact path', xy=(1400, FIG4_FLOOR['quartet']),
               xytext=(-2.0, -1.6), textcoords='offset points', ha='right',
               va='top', fontsize=PT_BASE, color=COLORS['bp'])
    from review_completion.noise_panels import panel as noise_panel
    e = c.panel('E', 1, 8, 4, title=None, grid='none')
    noise_panel(e, 'E', rows)

    f_ax = c.panel('F', 2, 0, 6, title=None, grid='none')
    f4_deficit(f_ax, contrast, seedfan, rows)
    g_ax = c.panel('G', 2, 6, 6, title=None, grid='none')
    f4_energy(g_ax, eigen, diag, rows, ramp)
    f4_badge(g_ax, .965, .975, 'oracle')
    # One 25 pt reserve per 4-module panel so the six data panels share one
    # axes width, split per module column so the two gaps of each row come
    # out equal and the row ends flush with B's ink: column 0 takes it all
    # on the left (C's and F's y title and tick labels must clear the
    # panel-letter column, audit_letter_alignment.py), column 4 splits
    # 20 + 5, column 8 splits 15 + 10 -- C->D and D->E are then both 50 pt
    # axes to axes, with D's and E's tick labels in that gap, and E's and
    # H's right edge sits 10 pt inside the slot, exactly where B's 10 pt
    # inset puts its ink (the 5 pt right margin makes that 503.4 pt).
    # save(equalize=False) keeps the split.  Row 2 declares an 18 pt top
    # reserve: F's and H's 'no deficit' / 'no cost' labels stand 8.5 pt
    # above their axes, and audit_row_separation wants 8.5 pt of clear
    # gutter under row 1's x labels.
    for name in ('C', 'F'):
        c.declare_reserve(name, left=25.0)
    c.declare_reserve('F', top=18.0)
    # the lock pass pads column 0's right edge by whatever F's x tick labels
    # hang past its axes (2.2 pt here); the other two columns take the same
    # amount on their right so the six axes widths stay equal
    locks = c.lock_reserves()
    # One right pad for every data panel: the larger of C's and F's measured
    # right overhang, so the three 4-module panels (25 + pad each) and the
    # two 6-module panels of row 2 (2026-09-21: F 25 + pad, G 15 + 10 + pad)
    # all keep equal axes widths.
    pad_right = max(locks['C'][1], locks['F'][1])
    c.declare_reserve('C', right=pad_right)
    c.declare_reserve('D', left=20.0, right=4.2 + pad_right)
    c.declare_reserve('E', left=15.0, right=10.0 + pad_right)
    c.declare_reserve('F', right=pad_right)
    c.declare_reserve('G', left=15.0, right=10.0 + pad_right)
    # Match total measured reserves, including the new two-line noise label.
    for _ in range(2):
        measured = c.lock_reserves()
        for group in (('C', 'D', 'E'), ('F', 'G')):
            total = max(measured[name][0] + measured[name][1] for name in group)
            for name in group:
                c.declare_reserve(name, left=measured[name][0],
                                  right=total - measured[name][0])

    sources = ['credit_rule_bridge/protocol_freeze.json',
               'credit_rule_bridge/figures/initial_profile_source.csv',
               'credit_rule_bridge/figures/figure_provenance.json',
               'credit_rule_extension/summaries/all_curves.csv',
               'credit_rule_bridge/summaries/task_variance_and_noise_floor.csv',
               'credit_rule_extension/summaries/paired_contrasts.csv',
               'credit_rule_extension/summaries/paired_seed_contrasts.csv',
               'credit_resolution_bridge/capture/matched_bridge_eigenvalues.csv',
               'credit_rule_extension/summaries/all_diagnostics.csv',
               'credit_resolution_bridge/capture/matched_bridge_spectra.csv',
               'morphology_credit/summaries/fresh/paired_contrasts.csv',
               'morphology_credit/summaries/fresh/selected_endpoints.csv',
               'morphology_credit/protocol_freeze.json',
               'credit_rule_extension/protocol_freeze.json',
               'credit_rule_bridge/selection_freeze.json',
               'credit_rule_bridge/summaries/all_diagnostics.csv']
    panels = {
        'A': 'Schematic. credit_rule_bridge/protocol_freeze.json '
             '(protocol.algebraic, protocol.fresh_seeds): the two targets, the '
             'shared balanced tree and the twenty fresh paired seeds. No '
             'measured series.',
        'B': 'Schematic. credit_rule_bridge/figures/initial_profile_source.csv '
             '(site, junction, mean_profile, mean_abs_profile, n_seeds) sets '
             'the six drop-dot radii; figures/figure_provenance.json and '
             'protocol_freeze.json fix the calibration protocol (256 unlabeled '
             'examples at step 0).',
        **{p: 'credit_rule_extension/summaries/all_curves.csv, algebraic Adam '
              'at the frozen selected rates, all twenty fresh seeds, every '
              'saved checkpoint from 64 to 16,384 updates; noise floor from '
              'credit_rule_bridge/summaries/task_variance_and_noise_floor.csv.'
           for p in ('C', 'D')},
        'F': 'credit_rule_extension/summaries/paired_contrasts.csv '
             '(quartet_minus_matching, calibrated_broadcast minus exact '
             'interaction, selected-rate Adam, endpoint and '
             'validation-selected states at 1,024/4,096/8,192/16,384) with the '
             'twenty paired seeds from paired_seed_contrasts.csv.',
        'G': 'credit_resolution_bridge/capture/matched_bridge_eigenvalues.csv '
             '(path_q, exact, selected-rate Adam, steps 0 and 1,024) as a '
             'per-seed cumulative spectrum; uniform-profile and rank-one '
             'capture from credit_rule_extension/summaries/all_diagnostics.csv '
             'at 1,024 and 16,384 updates; rank95 means (1.00/4.60/5.05) from '
             'matched_bridge_spectra.csv.',
        'E': 'New twenty-seed noise sensitivity, clean full-domain interaction deficit at 16384 updates, at inherited selected rates (diamonds) and inherited common Adam rate 0.003 (circles); fixed-absolute, noise-free and matched relative noise.'}
    sources += ['curated_publication/noise_controls_curves.csv',
                'curated_publication/noise_controls_contrasts.csv',
                'curated_publication/noise_controls_provenance.json']
    live_w = 518.4 - 40 - 5
    live_h = 468.0 - 24 - 36
    frac = sum(w * h for _, _, w, h in (c.slot_pt(0, 0, 5), c.slot_pt(0, 5, 7))) \
        / (live_w * live_h)
    contract = {'revision': '2026-09-20 review completion',
        'noise_panels': 'E replaces the duplicated nested and shuffled-leaf displays (Supplementary Figs S17A and S13C) and shows both inherited rate policies; the former H merged into it.',
        'noise_scope': 'New paired seeds, inherited rates, clean full-domain outcomes; common-rate panel reuses seeds.',
        'rank95_at_1024': {k: round(float(v), 2) for k, v in rank95.items()}}
    for row in rows:
        row.setdefault('record', 'summary')
    save(c, 4, sources, panels, FIG4_CAPTION, rows,
         {'original_capture_rows_joined': len(joined),
          'original_capture_replay_max_abs_difference': replay,
          'csv_parse_mode': 'round_trip; replay equality checked to '
                            'floating-point precision',
          'canvas_contract': contract}, equalize=False)


def f6_formula(f, xy, parts, *, size=None, color=None):
    """Private helper (errata #7): an expression as baseline-shifted runs.

    `build_main_figure_06.formula` measures its advance widths in points and
    lays them out in DATA units, which only works on an axes whose data
    coordinates are points; a `Frame` is a 0-1 frame, so the same idea is
    re-expressed here with annotation chaining (x from the previous run, y
    always from the frame ordinate of the first run) so no baseline drifts
    and no mathtext is emitted.  Recommended upstream as a `Frame.formula`
    method.

    Fix round 2026-09-09: the y anchor used to be the *bounding box* of the
    previous run, so every run after the first subscript sat 1.5 pt below the
    opening baseline and the squaring exponent, raised 2.2 pt from that drifted
    baseline, printed only 0.7 pt above the true one and read as a full-size
    factor.  The ordinate is now taken from the axes fraction of the first
    run, which is fixed, and the superscript offset is 3.0 pt against the
    subscript's -1.7 pt.  Type stays at the one size the token path allows
    (CF-2: 7.0 / 8.0 / 9.0 only); no mathtext.
    """
    from journal_style import COLORS, PT_BASE
    size = PT_BASE if size is None else size
    color = COLORS['ink'] if color is None else color
    ax = f.ax
    base = ax.text(xy[0], xy[1], parts[0][0], fontsize=size, color=color,
                   ha='left', va='baseline', zorder=6)
    prev = base
    for text, dy in parts[1:]:
        prev = ax.annotate(text, xy=(1.0, xy[1]),
                           xycoords=(prev, 'axes fraction'),
                           xytext=(0.2, dy), textcoords='offset points',
                           fontsize=size, color=color, ha='left',
                           va='baseline', zorder=6, annotation_clip=False)
    return base


def f6_stage_card(f, cell, depth, stages, title, foot, tiers, hero):
    """Private helper (errata #7): one architecture card of Fig 6A.

    `native_schematics.draw_stage_pair` draws the same two cards, but it also
    emits a two-entry rule key (CF-5 forbids keys outside Fig 5C) and it can
    carry neither the plan's card footers, nor the excitatory class contacts,
    nor the forward output arrow.  The card is therefore recomposed here from
    the same library glyphs (task_card / stage_tree / contact / error_in) and
    nothing is added to `native_schematics`.
    """
    from native_schematics import Frame, SOMA_R_PT, MUTE
    from journal_style import COLORS, PT_BASE, LW_EDGE
    core = f.task_card(cell, title=title, footer=foot, emphasis=hero)
    core = Frame.inset(core, left=0.05, right=0.05)
    base = (core[0] + core[2] * 0.38, core[1] + f.fy(14.0))
    height = max(4.0 * f.fy(1.0), core[3] - f.fy(14.0 + 5.0))
    tree = f.stage_tree(base, height, depth, branching=stages,
                        width=core[2] * 0.62, rings=True)
    lowest = min(p[1] for stage in tree for p in stage)
    assert lowest > base[1], 'glyph rule (a): the soma is not the lowest node'
    # QA 2026-09-10: eight class-bearing contacts on EVERY card.  One
    # contact per last-stage compartment gave D1 eight and D3 four, which
    # contradicts the card footer, the title and panel B's eight-slot rail;
    # the eight inputs are now distributed over the distal compartments
    # (two per leaf on D3) at the same 4.4 pt pitch the D1 row uses.
    leaves = tree[-1]
    per = max(1, 8 // max(len(leaves), 1))
    for px, py in leaves:
        offs = [0.0] if per == 1 else [(k - (per - 1) / 2.0) * 4.4
                                       for k in range(per)]
        for o in offs:
            f.contact((px + f.fx(o), py + f.fy(3.2)), kind='exc')
    right = core[0] + core[2]
    if depth == 1:
        px, py = tree[0][1]
        f.contact((base[0] + (px - base[0]) * 0.5,
                   base[1] + (py - base[1]) * 0.5), kind='inh')
        # QA 2026-09-10: +9.0 pt, not +5.0: at +5.0 the tag's box overlapped
        # the output arrow's `y` label by 1.9 x 2.0 pt.
        f.text((right, base[1] + f.fy(9.0)), 'all tiers', size=PT_BASE,
               color=COLORS['inh'], ha='right')
    else:
        for s, comps in enumerate(tree):
            parent = base if s == 0 else tree[s - 1][0]
            px, py = comps[0]
            f.contact((parent[0] + (px - parent[0]) * 0.5,
                       parent[1] + (py - parent[1]) * 0.5), kind='inh')
            f.text((right, py), list(reversed(tiers))[s], size=PT_BASE,
                   color=COLORS['inh'], ha='right', va='center')
    x0 = base[0] + f.fx(SOMA_R_PT + 1.4)
    f.arrow((x0, base[1]), (x0 + f.fx(6.0), base[1]), color=MUTE, lw=LW_EDGE,
            head=3.0)
    f.text((x0 + f.fx(7.6), base[1]), 'y', size=PT_BASE, ha='left')
    f.error_in(base, side='left', label='δ0')
    return tree


def f6_grouped_icon(f, rect):
    """Private helper: the resource-identical grouped-point control swatch.

    Eight point modules on one row, each wired straight to a single soma with
    no dendrite between them.  The soma is a declared SWATCH
    (`soma(delta0=False)`), so glyph rule (b) is satisfied without inventing a
    somatic error for a control that is only named, never trained, on A.
    """
    from journal_style import COLORS, LW_HAIR, PT_BASE
    from matplotlib.patches import Rectangle
    x0, y0, w, h = rect
    grey = COLORS['point_mlp']
    top = y0 + h - f.fy(7.0)
    cx = x0 + w / 2.0
    soma_y = y0 + f.fy(9.0)
    for k in range(8):
        mx = x0 + w * (k + 0.5) / 8.0
        f.ax.add_patch(Rectangle((mx - f.fx(1.9), top), 2 * f.fx(1.9),
                                 f.fy(3.2), facecolor='white',
                                 edgecolor=grey, lw=LW_HAIR, zorder=3))
        f.ax.plot([mx, cx], [top, soma_y], color=grey, lw=LW_HAIR, zorder=2)
    f.soma((cx, soma_y), r_pt=2.4, delta0=False)
    f.note('delta0-swatch', panel='a',
           reason='grouped-point control drawn as a glyph swatch')
    f.text((cx, y0 + f.fy(0.5)), 'grouped point', size=PT_BASE,
           color=COLORS['mute'], va='bottom')


def f6_architectures(ax):
    """Fig 6A: the same eight compartments in one stage or three."""
    from native_schematics import Frame
    from journal_style import COLORS, PT_BASE
    f = Frame(ax)
    foot_pt = 32.0
    cells = f.split(2, axis='x', gap_pt=6.0, pad_pt=(0, 0, 0, foot_pt))
    tiers = ('fine', 'coarse', 'global')
    f6_stage_card(f, cells[0], 1, [8], 'D1  [8]',
                  'all three tiers in one stage', tiers, False)
    f6_stage_card(f, cells[1], 3, [2, 1, 2], 'D3  [2,1,2]',
                  'one tier per stage', tiers, True)
    f6_grouped_icon(f, (0.0, 0.0, f.fx(40.0), f.fy(foot_pt - 2.0)))
    lines = ('Eight compartments per soma', 'Matched contacts',
             'and parameters')
    for i, line in enumerate(lines):
        f.text((f.fx(44.0), f.fy(foot_pt - 5.0 - i * 7.4)), line,
               size=PT_BASE, color=COLORS['mute'], ha='left')
    f.require_soma_lowest()
    f.require_delta0()
    return f


def f6_task_model(ax):
    """Fig 6B: where the nuisance gain factors act on the input stream."""
    from native_schematics import Frame, check_matrix_cells
    from journal_style import (COLORS, ORDINAL_RAMP, LW_HAIR, PT_BASE,
                               strengthen, tint_patch)
    f = Frame(ax)
    rail_x0, rail_x1 = f.fx(36.0), 1.0 - f.fx(7.0)
    n = 8
    pitch = (rail_x1 - rail_x0) / n
    rail_y = 1.0 - f.fy(32.0)
    slot_h = f.fy(6.5)
    for k in range(n):
        tint_patch(ax, ('rect', rail_x0 + k * pitch + f.fx(0.6), rail_y,
                        pitch - f.fx(1.2), slot_h), color='panel_bg', pct=100,
                   edge_color=COLORS['grid'], radius_pt=1.0, zorder=1.0)
    f.contact((rail_x0 + pitch * 0.5, rail_y + slot_h / 2.0), kind='exc')
    f.text((rail_x0 - f.fx(2.0), rail_y + slot_h / 2.0), 'class signal',
           size=PT_BASE, color=COLORS['exc'], ha='right')
    for i, (tag, blocks) in enumerate((('global ×1', 1), ('coarse ×2', 2),
                                       ('fine ×4', 4))):
        y = 1.0 - f.fy(5.0 + i * 8.4)
        per = n // blocks
        for b in range(blocks):
            a = rail_x0 + b * per * pitch + f.fx(0.8)
            z = rail_x0 + (b + 1) * per * pitch - f.fx(0.8)
            f.rule(y, a, z, color=COLORS['mute'], lw=LW_HAIR)
            for x in (a, z):
                f.ax.plot([x, x], [y, y - f.fy(2.2)], color=COLORS['mute'],
                          lw=LW_HAIR, zorder=1.2)
        f.contact((rail_x1 + f.fx(3.6), y), kind='inh', label='α')
        f.text((rail_x0 - f.fx(2.0), y), tag, size=PT_BASE,
               color=COLORS['mute'], ha='right')
    f.text((f.fx(1.0), rail_y - f.fy(5.0)), 'inhibitory sensors; fidelity α',
           size=PT_BASE, color=COLORS['mute'], ha='left',
           va='top')
    # The full positive-rate generator, clipping floor and parameter values
    # are specified in Results. The panel shows the task's spatial structure.
    f.text((.5, rail_y - f.fy(19.0)), 'Class signal × nuisance gains',
           size=PT_BASE, color=COLORS['ink'], ha='center')
    fam = (('nested', ORDINAL_RAMP[3], 'nested'),
           ('flat', ORDINAL_RAMP[2], 'flat'),
           ('local ratio', ORDINAL_RAMP[1], 'ratio'))
    strip_top = rail_y - f.fy(29.0)
    row_pt = 12.0
    check_matrix_cells((rail_x1 - rail_x0) * f.w_pt, 3 * row_pt, 3, n,
                       where='fig 6B family strip')
    for r, (name, colour, kind) in enumerate(fam):
        y = strip_top - f.fy(r * row_pt)
        h = f.fy(row_pt - 2.6)
        tint_patch(ax, ('rect', rail_x0, y - h, rail_x1 - rail_x0, h),
                   color=colour, pct=16, radius_pt=1.0, zorder=1.0)
        f.text((rail_x0 - f.fx(2.0), y - h / 2.0), name, size=PT_BASE,
               color=colour, ha='right')
        cut = strengthen(colour, 2.2)
        if kind == 'ratio':
            for k in range(n):
                f.text((rail_x0 + (k + 0.5) * pitch, y - h / 2.0), '÷',
                       size=PT_BASE, color=colour)
        elif kind == 'flat':
            for b in range(1, n):
                x = rail_x0 + b * pitch
                f.ax.plot([x, x], [y - h, y], color=cut, lw=LW_HAIR,
                          zorder=2.0)
        else:
            # 4 / 2 / 1: the deeper the cut, the coarser the tier it bounds.
            for b, frac in ((2, 0.5), (4, 1.0), (6, 0.5)):
                x = rail_x0 + b * pitch
                f.ax.plot([x, x], [y - h, y - h + h * frac], color=cut,
                          lw=LW_HAIR, zorder=2.0)
    f.text((f.fx(1.0), strip_top - f.fy(3 * row_pt + 2.0)),
           'same product, different supports', size=PT_BASE,
           color=COLORS['mute'], ha='left', va='top')
    f.text((f.fx(1.0), strip_top - f.fy(3 * row_pt + 10.0)),
           '÷ : local E/I ratio', size=PT_BASE,
           color=ORDINAL_RAMP[1], ha='left', va='top')
    f.require_soma_lowest()          # no tree in this card: a no-op assertion
    f.require_delta0(allow_no_delta0=True,
                     reason='input-side generative model; the tree, its soma '
                            'and δ0 are drawn in panel A')
    return f


def f6_style(ax, title=None):
    """Tighten the label bands so three rows clear 3 mm (audit_row_separation).

    ``title`` is None for every data panel since the 2026-09-14 design pass.
    """
    from journal_style import COLORS, PT_EMPH
    if title:
        ax.set_title(title, fontsize=PT_EMPH, color=COLORS['ink'], pad=1.0,
                     fontweight='normal')
    ax.tick_params(axis='both', which='major', pad=0.8, length=2.0)
    ax.tick_params(axis='both', which='minor', length=1.4)
    ax.xaxis.labelpad = 0.8
    ax.yaxis.labelpad = 1.2
    return ax


def f6_note_data(ax, x, y, lines, *, color=None, ha='left', dy_pt=7.4,
                 size=None):
    """The same note anchored in DATA coordinates.

    Every note in this figure is placed against the curve it explains, so it
    is written where the data is, not where the axes fraction happens to fall
    after the lock pass moves the box.
    """
    from journal_style import COLORS, PT_BASE
    color = COLORS['mute'] if color is None else color
    size = PT_BASE if size is None else size
    h_pt = ax.get_window_extent().height * 72.0 / ax.figure.dpi
    lo, hi = ax.get_ylim()
    step = dy_pt / h_pt * (hi - lo)
    return [ax.text(x, y - i * step, line, fontsize=size, color=color, ha=ha,
                    va='top', zorder=6, clip_on=False)
            for i, line in enumerate(lines)]


def f6_sign_legend(ax, top, bottom, *, top_color, bottom_color, top_frac=1.0,
                   top_x=1.0, top_ha='right', bottom_x=1.0, bottom_ha='right'):
    """The paired panels' sign key, at the two ends of the y range.

    Plan §4 G: the sign of the difference is named on the panel, not by a
    second data artist.  The plan's rotated placement beside the y axis was
    measured against the tick column and the y label and collides with both
    (three text-collision violations), so the two readings sit at the top and
    the bottom of the panel's own edge, outside every annotation block.

    QA 2026-09-10: which reading belongs at which end is a property of the
    ordinate, not of the idiom, so the caller supplies both the words and
    their hues.  On a loss ordinate (panel H, nats) the arm that is AHEAD is
    the one with the LOWER value, so `exact path ahead` seats at the BOTTOM;
    plan §4 H's `exact path ahead top` is written for a gain ordinate and is
    wrong for H (recorded as a deviation).
    """
    from journal_style import PT_BASE
    ax.text(top_x, top_frac, top, transform=ax.transAxes, fontsize=PT_BASE,
            color=top_color, ha=top_ha, va='top', zorder=6, clip_on=False)
    ax.text(bottom_x, 0.0, bottom, transform=ax.transAxes, fontsize=PT_BASE,
            color=bottom_color, ha=bottom_ha, va='bottom', zorder=6,
            clip_on=False)


def f6_ref(ax, value, label, *, at=None, span=None, dy=0.0, ha='right',
           va='bottom'):
    """Dashed mute LW_REF reference with its label right-aligned on the line.

    `native_schematics.reference_line` fixes the label to the right END of the
    span, which in four of this figure's panels lands on a datum; the line is
    therefore drawn by the library and only the label anchor is chosen here,
    keeping the CF-7 idiom (dashed mute LW_REF, PT_BASE label sitting on the
    line) with a collision-free anchor.
    """
    from journal_style import COLORS, PT_BASE
    from native_schematics import reference_line
    reference_line(ax, value, axis='y', label=None, span=span)
    if label:
        ax.text(at, value + dy, label, fontsize=PT_BASE, color=COLORS['mute'],
                ha=ha, va=va, zorder=6, clip_on=False)


def f6_budget_rule(ax, span):
    """The 180-epoch budget rule in E, F, G and H.

    CF-7 asks every reference line to carry a right-aligned label on the
    line.  `reference_line(axis='x')` sets that label rotated at the TOP of
    the rule, which in all four panels lands on a curve, a band or a note; and
    all four panels already carry a major x tick at exactly 180 directly under
    the rule, which names it.  The line is therefore drawn with the library's
    own dashed-mute-LW_REF idiom and the tick is its label.
    """
    from native_schematics import reference_line
    return reference_line(ax, 180.0, axis='x', label=None, span=span)


def f6_trim(ax, *, x=None, y=None):
    """Hold each spine to the range its ticks name.

    Three data panels carry a direct-label column inside their own axes (a
    declared reserve would have to come out of a 124 pt slot and would then
    collide with the next panel's tick column), and four carry an annotation
    band outside the tick range, so the data limits run past the last tick.
    Bounding the spine keeps the drawn axis equal to the measured range, which
    is what a reader reads.
    """
    if x is not None:
        ax.spines['bottom'].set_bounds(*x)
    if y is not None:
        ax.spines['left'].set_bounds(*y)
    return ax


def f6_family_dose(ax, effects, seeds, rows):
    """Fig 6C: serial minus grouped point, by task family and sensor fidelity."""
    import numpy as np
    from journal_style import (ORDINAL_RAMP, SEED_MS, SEED_ALPHA, PT_BASE,
                               MARKER_MS, LW_ERR, LW_HAIR, LW_EDGE)
    fam = (('nested_factor', 'nested', ORDINAL_RAMP[3], 'o'),
           ('flat_factor', 'flat', ORDINAL_RAMP[2], 's'),
           ('local_ratio', 'local ratio', ORDINAL_RAMP[1], '^'))
    alphas = (0.0, 0.5, 1.0)
    xs = (0, 1, 2)
    ax.set(xlim=(-0.22, 2.92), ylim=(-8.5, 37.0), xticks=list(xs),
           yticks=[0, 10, 20, 30], xlabel='Sensor fidelity α',
           ylabel='Serial − grouped point (pp)')
    ax.set_xticklabels(['0', '0.5', '1'])
    ax.set_yticks([-5], minor=True)
    f6_trim(ax, x=(0, 2), y=(-5, 30))
    f6_ref(ax, 0.0, 'no advantage', at=-0.18, dy=2.8, ha='left')
    rng = np.random.default_rng(60310)
    for key, name, colour, marker in fam:
        sub = (effects[effects.family.eq(key) & effects.credit.eq('bp')
                       & effects.estimand.eq('serial_minus_grouped')]
               .set_index('alignment_alpha'))
        mean = [100 * float(sub.loc[a, 'mean_difference']) for a in alphas]
        low = [100 * float(sub.loc[a, 'ci95_low']) for a in alphas]
        high = [100 * float(sub.loc[a, 'ci95_high']) for a in alphas]
        ax.plot(xs, mean, color=colour, lw=LW_HAIR, zorder=2.0,
                solid_capstyle='round')
        for i, x in enumerate(xs):
            tie = key == 'local_ratio' and i == 2
            ax.errorbar([x], [mean[i]],
                        yerr=[[mean[i] - low[i]], [high[i] - mean[i]]],
                        fmt=marker, ms=MARKER_MS,
                        mfc='white' if tie else colour,
                        mec=colour if tie else 'white', mew=LW_EDGE,
                        ecolor=colour, elinewidth=LW_ERR, capsize=2.0,
                        zorder=3.0)
            rows.append(dict(panel='C', family=key, alignment_alpha=alphas[i],
                             credit='bp', estimand='serial_minus_grouped',
                             mean_pp=mean[i], ci95_low_pp=low[i],
                             ci95_high_pp=high[i], n_seeds=10))
        pair = (seeds[seeds.credit.eq('bp') & seeds.alignment_alpha.eq(1.0)
                      & seeds.family.eq(key)]
                .pivot_table(index='seed', columns='architecture',
                             values='test_accuracy'))
        fan = 100 * (pair['serial'] - pair['grouped_point'])
        jitter = rng.uniform(-0.10, 0.10, len(fan))
        ax.plot(2 + jitter, fan.to_numpy(), marker='o', ls='none',
                ms=SEED_MS, mfc=colour, mec='none', alpha=SEED_ALPHA,
                zorder=2.6)
        for seed, value in fan.items():
            rows.append(dict(panel='C', record='paired seed difference',
                             family=key, alignment_alpha=1.0,
                             credit='bp', estimand='serial_minus_grouped',
                             seed=int(seed), value_pp=float(value),
                             mean_pp=float(value)))
    for name, y, colour in (('nested', 30.32, ORDINAL_RAMP[3]),
                            ('flat', 22.20, ORDINAL_RAMP[2]),
                            ('local ratio', 9.0, ORDINAL_RAMP[1])):
        ax.text(2.13, y, name, fontsize=PT_BASE, color=colour, va='center',
                ha='left', zorder=6, clip_on=False)
    ax.plot([2.055, 2.105], [1.3, 7.6], color=ORDINAL_RAMP[1], lw=LW_HAIR,
            zorder=1.6)
    # Design pass 2026-09-14: the two-line tie note is gone (caption C states
    # that the alpha = 1 local-ratio value is an exact tie in all ten pairs;
    # the open marker is its mark).  Only the panel-specific caveat stays.
    f6_note_data(ax, -0.15, 36.5, ('intervals < markers',))
    return ax


# QA 2026-09-10 (major): 18.0 -> 43.0.  The 32-point band under the 50 %
# chance rule existed only to seat D's and E's multi-line statistics blocks,
# which together took 39 % of both panels' height, detached the x axis from
# the data and duplicated the caption.  Both blocks are deleted and the band
# with them; the drawn axis is still 50-100 (`f6_trim`) and the remaining
# 7 points hold the `chance` reference label.
F6_ACC_YLIM = (43.0, 105.0)   # 2026-09-14: 110 -> 105, E's sub-title band went


def f6_num(value, places=2, *, signed=False):
    """A frozen table value as on-panel text, with a typographic minus."""
    text = '%.*f' % (places, float(value))
    if signed and not text.startswith('-'):
        text = '+' + text
    return text.replace('-', '−')


def f6_depth_ladder(ax, conf, remaining, ceiling, contrast, rows):
    """Fig 6D: the three-tier depth ladder with its two controls.

    Palette deviation (recorded, AMENDMENTS section 5 role table): the
    reversed-placement control is drawn in `highlight` and the point-network
    ceiling shares `point_mlp` with the grouped-point arm.  Plan section 4 D
    names `mute` for reversed placement, which the role table bans as a data
    series; `highlight` is the table's declared "second control in a figure
    that has already spent point_mlp", and both grey marks are point-network
    controls, so the hues are semantically right.  The role table's `highlight`
    and `point_mlp` rows do not yet list Fig 6 -- an integrator edit, not a
    build change.
    """
    from journal_style import (COLORS, PT_BASE, MARKER_MS, LW_ERR, LW_EDGE,
                               LW_DATA, LW_REF, LW_HAIR, tint_patch)
    from figure_canvas import token_subscript
    arms = (
        ('exact BP', COLORS['ink'], 'o', True, (0, (4.2, 2.0)),
         dict(regime='aligned', mechanism='shunting', method='bp')),
        ('exact path', COLORS['bp'], '^', True, '-',
         dict(regime='aligned', mechanism='shunting', method='local3f',
              transport='path_transport')),
        ('shared soma', COLORS['scalar'], 's', True, '-',
         dict(regime='aligned', mechanism='shunting', method='local3f',
              transport='per_soma_shared')),
        ('additive', COLORS['additive'], 'X', False, (0, (1.2, 1.6)),
         dict(regime='aligned', mechanism='additive', method='bp')),
    )
    ax.set(xlim=(0.80, 4.42), ylim=F6_ACC_YLIM, xticks=[1, 2, 3],
           yticks=[50, 60, 70, 80, 90, 100], ylabel='Test accuracy (%)')
    ax.set_xticklabels(['D1', 'D2', 'D3'])
    f6_trim(ax, x=(1, 3), y=(50, 100))
    label_y = {}
    for name, colour, marker, filled, dash, sel in arms:
        sub = conf.copy()
        for key, value in sel.items():
            sub = sub[sub[key].eq(value)]
        sub = sub.sort_values('depth')
        x = sub.depth.to_numpy(float)
        m = 100 * sub.mean_test_accuracy.to_numpy(float)
        lo = 100 * sub.ci95_low_test_accuracy.to_numpy(float)
        hi = 100 * sub.ci95_high_test_accuracy.to_numpy(float)
        ax.plot(x, m, color=colour, ls=dash, zorder=2.4, solid_capstyle='round',
                lw=LW_REF if name == 'exact BP' else LW_DATA)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt=marker, ms=MARKER_MS,
                    mfc=colour if filled else 'white',
                    mec='white' if filled else colour, mew=LW_EDGE,
                    ecolor=colour, elinewidth=LW_ERR, capsize=2.0, zorder=3.0)
        label_y[name] = (float(m[-1]), colour)
        rows.extend(dict(panel='D', arm=name, depth=int(d),
                         mean_test_accuracy_pp=float(a), ci95_low_pp=float(b),
                         ci95_high_pp=float(c), n_seeds=10)
                    for d, a, b, c in zip(x, m, lo, hi))
    # QA 2026-09-10 (minor): the two resource-matched controls agree within
    # 0.4 points at every depth, so at a shared abscissa they printed as one
    # series carrying two labels.  Each is dodged 0.08 depth units off the
    # tick (about 5 pt apart on the page) so both marker sets are visible;
    # the plotted values are unchanged.
    for name, colour, marker, dash, regime, arch, dx in (
            ('grouped', COLORS['point_mlp'], 'D', (0, (4.2, 2.0)), 'aligned',
             'grouped_point', -0.08),
            ('reversed', COLORS['highlight'], 'v', (0, (1.2, 1.6)),
             'rewired_tree', 'serial_tree', 0.08)):
        sub = remaining[remaining.hierarchy.eq(3) & remaining.regime.eq(regime)
                        & remaining.architecture.eq(arch)
                        & remaining.credit.eq('full_bp')].sort_values('depth')
        x = sub.depth.to_numpy(float) + dx
        m = 100 * sub.mean_test_accuracy.to_numpy(float)
        lo = 100 * sub.ci_low.to_numpy(float)
        hi = 100 * sub.ci_high.to_numpy(float)
        ax.plot(x, m, color=colour, lw=LW_DATA, ls=dash, zorder=2.2,
                solid_capstyle='round')
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt=marker, ms=MARKER_MS,
                    mfc='white', mec=colour, mew=LW_EDGE, ecolor=colour,
                    elinewidth=LW_ERR, capsize=2.0, zorder=3.0)
        label_y[name] = (float(m[-1]), colour)
        rows.extend(dict(panel='D', arm=name, depth=int(round(d - dx)),
                         mean_test_accuracy_pp=float(a), ci95_low_pp=float(b),
                         ci95_high_pp=float(c), n_seeds=10)
                    for d, a, b, c in zip(x, m, lo, hi))
    f6_ref(ax, 50.0, 'chance', at=4.40, dy=-0.7, va='top')
    top = ceiling[ceiling.architecture.eq('point_mlp_total')
                  & ceiling.regime.eq('aligned')].iloc[0]
    lo = 100 * float(top.ci95_low_test_accuracy)
    hi = 100 * float(top.ci95_high_test_accuracy)
    tint_patch(ax, ('rect', 0.80, lo, 3.62, hi - lo), color='point_mlp',
               pct=8, radius_pt=0.8, zorder=0.6)
    ax.plot([0.80, 4.42], [100 * float(top.mean_test_accuracy)] * 2,
            color=COLORS['point_mlp'], lw=LW_REF, ls=(0, (5, 2.2)),
            zorder=1.4, solid_capstyle='butt')
    ax.text(4.40, 100 * float(top.mean_test_accuracy) + 0.8,
            'point network 99.0 %', fontsize=PT_BASE,
            color=COLORS['point_mlp'], ha='right', va='bottom', zorder=6,
            clip_on=False)
    rows.append(dict(panel='D', arm='point network reference', depth=0,
                     mean_test_accuracy_pp=100 * float(top.mean_test_accuracy),
                     ci95_low_pp=lo, ci95_high_pp=hi, n_seeds=10))
    # QA 2026-09-10: the lower three seats are re-pitched to an even
    # 7.2 accuracy points (9.2 pt on the page, one label band plus 2.2 pt)
    # and `reversed` sits on its own series level (60.67 %) with `additive`
    # 1.2 points above its own (51.90 %), clear of the 50 % chance rule; the
    # old 3.0-point gap between the two overlapped by 3.2 pt on the page.
    for name, y in (('exact BP', 91.0), ('exact path', 85.6),
                    ('shared soma', 77.8), ('grouped', 67.6),
                    ('reversed', 60.4), ('additive', 53.1)):
        value, colour = label_y[name]
        ax.text(3.16, y, name, fontsize=PT_BASE, color=colour, ha='left',
                va='center', zorder=6, clip_on=False)
        if abs(y - value) > 1.4:
            ax.plot([3.03, 3.13], [value, y], color=colour, lw=LW_HAIR,
                    zorder=1.8)
    row = contrast[contrast.contrast.eq(
        'depth__serial_bp__aligned__d4_d3')].iloc[0]
    # QA 2026-09-10 (major/minor): the five-line four-tier block is DELETED.
    # It was the largest text element in the panel and it reported a D4-minus-
    # D3 contrast that is drawn nowhere in the figure (there is no D4 tick),
    # while two of its lines repeated the caption footer.  The contrast stays
    # in figure_06_plotted.csv and moves to the caption; the space it held
    # goes back to the data (see F6_ACC_YLIM).
    rows.append(dict(panel='D', arm='four-tier D4 − D3', depth=4,
                     mean_test_accuracy_pp=float(row.mean_pp),
                     ci95_low_pp=float(row.ci_low_pp),
                     ci95_high_pp=float(row.ci_high_pp), n_seeds=10))
    token_subscript(ax, 0.09, -0.085, 'Serial physical depth D', 'p',
                    size=8.0, sub_size=7.0, color=COLORS['ink'], ha='left',
                    va='top', transform=ax.transAxes, clip_on=False)
    return ax


F6_ARMS = (
    ('exact_autograd_bp_recipe', 3, 'ink', 'dashed', 'exact BP (D3)', True),
    ('broadcast_autograd_bp_recipe', 3, 'ink', 'dotted', 'broadcast (BP)',
     False),
    ('path_transport', 3, 'bp', 'solid', 'exact path', True),
    ('per_soma_shared', 3, 'scalar', 'solid', 'shared soma', True),
    ('broadcast_autograd_localca_recipe', 3, 'scalar', 'dotted',
     'broadcast (local)', False),
    # QA 2026-09-10 (major): the D1 reference was DASHED in F and, because of
    # E's stopping split, SOLID-then-DOTTED in E, so a dotted curve in E was
    # not a broadcast arm and the same arm changed style between the two
    # panels.  It now has one dedicated style in both -- thin solid grey
    # ('thin': LW_REF, ls '-') -- and dotted is reserved for the two
    # autograd-broadcast variants, exactly as the caption key states.
    ('exact_autograd_bp_recipe', 1, 'ink', 'thin', 'exact BP (D1)', True),
)
F6_DASH = {'solid': '-', 'dashed': (0, (4.2, 2.0)), 'dotted': (0, (1.2, 1.6)),
           'thin': '-'}
F6_XLIM = (0.0, 1071.0)
F6_XLAB = 712.0     # E's label column (QA 2026-09-10: room for the elbow)
F6_XLAB_F = 655.0   # F's column: the right margin caps the run at 5.9 pt


def f6_trajectories(ax, curves, metric, scale, panel, rows, *, split=None):
    """Fig 6E,F: the six 600-epoch restart arms, one hue per credit coordinate.

    Dotted is the autograd-broadcast variant of the hue's coordinate, so the
    six arms need three hues and no key (CF-5).
    """
    from journal_style import COLORS, LW_DATA, LW_REF
    from credit_tree_schematics import mix
    ends = {}
    for arm, depth, cname, style, label, band in F6_ARMS:
        p = curves[curves.arm.eq(arm) & curves.depth.eq(depth)
                   & curves.metric.eq(metric)].sort_values('epoch')
        colour = mix(cname, 45) if (arm.startswith('exact') and depth == 1) \
            else COLORS[cname]
        lw = LW_DATA if style == 'solid' else LW_REF
        if band:
            ax.fill_between(p.epoch, scale * p.ci95_low, scale * p.ci95_high,
                            color=colour, alpha=0.12, lw=0, zorder=1.5)
        x = p.epoch.to_numpy(float)
        y = scale * p['mean'].to_numpy(float)
        if split is not None and depth == 1:
            # QA 2026-09-10: the carried-forward segment (past the epoch where
            # fewer than eight of the ten D1 fits are still running) is marked
            # by a lighter alpha, NOT by dots -- dotted means the
            # autograd-broadcast variant everywhere else in E and F.
            keep = x <= split
            ax.plot(x[keep], y[keep], color=colour, lw=lw, ls='-', zorder=2.4,
                    solid_capstyle='round')
            ax.plot(x[~keep], y[~keep], color=colour, lw=lw, ls='-',
                    alpha=0.45, zorder=2.4, solid_capstyle='round')
        else:
            ax.plot(x, y, color=colour, lw=lw, ls=F6_DASH[style], zorder=2.4,
                    solid_capstyle='round')
        ends[label] = (float(y[-1]), colour)
        rows.extend(dict(panel=panel, record='curve summary',
                         arm=arm, depth=depth, metric=metric,
                         epoch=int(e), mean=float(m), ci95_low=float(a),
                         ci95_high=float(b), n_seeds=10,
                         plotted_mean=scale * float(m),
                         plotted_ci95_low=scale * float(a),
                         plotted_ci95_high=scale * float(b),
                         plotted_unit='accuracy (%)' if metric == 'test_accuracy'
                                      else 'cross-entropy (nats)',
                         band_drawn=bool(band))
                    for e, m, a, b in zip(p.epoch, p['mean'], p.ci95_low,
                                          p.ci95_high))
    return ends


def f6_direct_ends(ax, ends, places, x, *, lead_from=None):
    """Direct labels in the panel's own right-hand label column, with leaders.

    QA 2026-09-10: the leaders used to start exactly on each curve's last
    datum and were drawn in a tint of the series hue, so at print scale they
    were continuous with the stroke and read as four trajectories plunging at
    the last epoch.  A leader now (i) is mute, never a series hue, (ii) starts
    a clear 2.2 pt to the right of the endpoint marker, and (iii) is an elbow
    -- a short horizontal jog at the datum's own level before it turns toward
    the label -- so it cannot be mistaken for data.  A seat within 1.2 pt of
    its own endpoint gets no leader at all.
    """
    from journal_style import COLORS, PT_BASE, LW_HAIR
    x0, x1 = ax.get_xlim()
    w_pt = ax.get_window_extent().width * 72.0 / ax.figure.dpi
    per_pt = (x1 - x0) / w_pt                      # data units per point
    lo, hi = ax.get_ylim()
    h_pt = ax.get_window_extent().height * 72.0 / ax.figure.dpi
    y_per_pt = (hi - lo) / h_pt
    for key, y in places.items():
        # QA 2026-09-10: a key may be a TUPLE of arm labels whose endpoints
        # coincide.  Two seats 0.02 points apart cannot be told apart by their
        # leaders (they start from the same spot in the same hue), so the pair
        # shares one seat: both names are stacked, a mute hairline bracket
        # ties them together, and a single elbow runs to the shared endpoint.
        names = (key,) if isinstance(key, str) else tuple(key)
        value, colour = ends[names[0]]
        step = 8.3 * y_per_pt
        seats = [y - i * step for i in range(len(names))]
        for name, seat in zip(names, seats):
            ax.text(x, seat, name, fontsize=PT_BASE, color=ends[name][1],
                    ha='left', va='center', zorder=6, clip_on=False)
        mid = 0.5 * (seats[0] + seats[-1])
        if len(names) > 1:
            bx = x - 1.6 * per_pt
            ax.plot([bx, bx], [seats[0], seats[-1]], color=COLORS['mute'],
                    lw=LW_HAIR, zorder=1.6, solid_capstyle='butt')
        if lead_from is None or abs(mid - value) <= 1.2 * y_per_pt:
            continue
        start = lead_from + 2.2 * per_pt
        jog = lead_from + 3.6 * per_pt
        end = x - (2.6 if len(names) > 1 else 1.8) * per_pt
        ax.plot([start, jog, end], [value, value, mid], color=COLORS['mute'],
                lw=LW_HAIR, zorder=1.6, solid_capstyle='round')


def f6_accuracy_budget(ax, curves, stopping, budget, rows):
    """Fig 6E: validation-selected accuracy over the 600-epoch restarts."""
    # QA 2026-09-10 (major): E used to carry neither y tick labels nor a y
    # title, so no accuracy in it could be read; the shared axis is now
    # labelled on E as it is on D.
    ax.set(xlim=F6_XLIM, ylim=F6_ACC_YLIM, xticks=[0, 180, 400, 600],
           yticks=[50, 60, 70, 80, 90, 100], xlabel='Epoch',
           ylabel='Test accuracy (%)')
    f6_trim(ax, x=(0, 600), y=(50, 100))
    d1 = stopping[stopping.arm.eq('exact_autograd_bp_recipe')
                  & stopping.depth.eq(1)]
    runs = sorted(int(v) for v in d1.epochs_run)
    split = max(e for e in runs if sum(r >= e for r in runs) >= 8)
    ends = f6_trajectories(ax, curves, 'test_accuracy', 100.0, 'E', rows,
                           split=split)
    # CF-7 keeps the reference label right-aligned on its rule; it moves to
    # the upper side of the 50 % rule so that no annotation line shares a
    # baseline with it (the previous build read as
    # '[0.40, 0.55], 10 of 10 seeds  chance').
    f6_ref(ax, 50.0, 'chance', at=1065.0, dy=0.7, va='bottom', span=F6_XLIM)
    f6_budget_rule(ax, (50.0, 100.0))
    # QA 2026-09-10: the two coincident pairs share one seat each.  At 600 the
    # local broadcast arm is 0.02 points from shared soma and the BP broadcast
    # arm 0.48 from exact BP, so five separate leaders left from what is one
    # spot on the page; E now names four distinguishable ends, as F does.
    f6_direct_ends(ax, ends, {('exact BP (D3)', 'broadcast (BP)'): 99.4,
                              ('shared soma', 'broadcast (local)'): 86.4,
                              'exact path': 73.4, 'exact BP (D1)': 61.2},
                   F6_XLAB, lead_from=600.0)
    gain180 = budget[budget.budget.eq(180)
                     & budget.contrast.eq('depth_gain_exact_bp')].iloc[0]
    gain600 = budget[budget.budget.eq(600)
                     & budget.contrast.eq('depth_gain_exact_bp')].iloc[0]
    bcast = budget[budget.budget.eq(600)
                   & budget.contrast.eq('bp_exact_minus_broadcast')].iloc[0]
    counts = '/'.join(str(sum(r >= e for r in runs)) for e in (180, 400, 600))
    # The tag block leaves the swept region entirely: the 180-epoch rule now
    # stops at the 100 % gridline and every tag line sits in the detached
    # gutter below the 50 % floor, where no rule, curve, direct label or
    # reference label crosses it.  The condition sub-title takes the empty
    # band above the data instead.
    # QA 2026-09-10: shortened from `600-epoch restarts of the same seeds`.
    # At 141 pt the sub-title pushed panel E's tight bounding box under
    # panel F's letter (the canvas letter check flagged it); the caption
    # carries the full sentence.
    # (design pass 2026-09-14: the `600-epoch restarts, same seeds` note is
    # gone; caption E carries the sentence)
    # QA 2026-09-10 (major): the four-line contrast block is DELETED.  It sat
    # between the curves and the x axis, took 39 % of the panel's height and
    # detached the axis from the data; every number in it is a paired
    # contrast that belongs in the caption and in figure_06_plotted.csv,
    # where all four now live.
    for tag, row in (('depth_gain_exact_bp 180', gain180),
                     ('depth_gain_exact_bp 600', gain600),
                     ('bp_exact_minus_broadcast 600', bcast)):
        rows.append(dict(panel='E', arm=tag, metric='paired_contrast_pp',
                         mean=float(row['mean']), ci95_low=float(row.ci95_low),
                         ci95_high=float(row.ci95_high),
                         positive_seeds=int(row.positive_seeds), n_seeds=10))
    rows.append(dict(panel='E', arm='D1 fits still training',
                     metric='epochs_run >= epoch', mean=float(split),
                     n_seeds=10, note=counts))
    return ax


def f6_validation_loss(ax, curves, stopping, rows):
    """Fig 6F: best validation loss for the same six arms."""
    from journal_style import COLORS, MARKER_MS, LW_EDGE, LW_HAIR
    ax.set(xlim=F6_XLIM, ylim=(0.055, 0.765), xticks=[0, 180, 400, 600],
           yticks=[0.1, 0.3, 0.5, 0.7], xlabel='Epoch',
           ylabel='Best validation loss')
    f6_trim(ax, x=(0, 600), y=(0.1, 0.7))
    ends = f6_trajectories(ax, curves, 'best_validation_loss', 1.0, 'F', rows)
    f6_budget_rule(ax, (0.100, 0.678))
    d1 = stopping[stopping.arm.eq('exact_autograd_bp_recipe')
                  & stopping.depth.eq(1) & stopping.stopped_before600]
    stops = sorted(int(v) for v in d1.epochs_run)
    track = curves[curves.arm.eq('exact_autograd_bp_recipe')
                   & curves.depth.eq(1)
                   & curves.metric.eq('best_validation_loss')
                   ].set_index('epoch')['mean']
    ax.plot(stops, [float(track.loc[e]) for e in stops], marker='o',
            ls='none', ms=MARKER_MS * 0.8, mfc='white',
            mec=COLORS['point_mlp'], mew=LW_EDGE, zorder=3.4)
    rows.extend(dict(panel='F', record='stopping marker',
                     arm='exact_autograd_bp_recipe', depth=1, seed=int(r.seed),
                     metric='stopping_epoch', epoch=int(r.epochs_run),
                     mean=float(track.loc[r.epochs_run]), n_seeds=10)
                for r in d1.sort_values(['epochs_run', 'seed']).itertuples())
    f6_direct_ends(ax, ends, {'exact BP (D3)': 0.120, 'exact path': 0.214,
                              'shared soma': 0.320}, F6_XLAB_F, lead_from=600.0)
    # Design pass 2026-09-14: the three-line stopping note, its leader and
    # the `convergence not established` caveat are gone -- caption F names
    # the open markers, and the running text carries the 103-590 range, the
    # fifty capped D3 fits and the maturity caveat.  F draws all six arms but
    # seats only three direct labels (the other three end inside the bundle),
    # so it still says once that the arm-to-style mapping is E's.
    assert stops[0] == 103 and stops[-1] == 590, (stops[0], stops[-1])
    f6_note_data(ax, 250.0, 0.600, ('arms and styles as in E',))
    return ax


def f6_paired(ax, paired, metric, rows, *, ylim, yticks, ylabel, marks,
              notes, sign, rule_span, band=None, zero_dy=0.0, zero_at=612.0,
              zero_va='bottom', zero_ha='right', sign_top=(1.0, 1.0, 'right'),
              sign_bottom=(1.0, 'right'), sign_hues=('bp', 'amber')):
    """Fig 6G,H: one paired exact-path-minus-shared-soma difference curve."""
    from journal_style import (COLORS, LW_DATA, LW_HAIR, MARKER_MS, LW_EDGE,
                               tint_patch)
    from credit_tree_schematics import AMBER_TEXT
    p = paired[paired.metric.eq(metric)].sort_values('epoch')
    ax.set(xlim=(0, 615), ylim=ylim, xticks=[0, 180, 400, 600], yticks=yticks,
           xlabel='Epoch', ylabel=ylabel)
    f6_trim(ax, x=(0, 600), y=(yticks[0], yticks[-1]))
    if band is not None:
        lo, hi = band
        tint_patch(ax, ('rect', lo, ylim[0], hi - lo, ylim[1] - ylim[0]),
                   color='mute', pct=6, radius_pt=0.0, zorder=0.5, edge=False)
    ax.fill_between(p.epoch, p.ci95_low, p.ci95_high, color=COLORS['mute'],
                    alpha=0.12, lw=0, zorder=1.4)
    ax.plot(p.epoch, p['mean'], color=COLORS['bp'], lw=LW_DATA, zorder=2.6,
            solid_capstyle='round')
    f6_ref(ax, 0.0, 'no difference', at=zero_at, span=(0, 615), dy=zero_dy,
           va=zero_va, ha=zero_ha)
    f6_budget_rule(ax, rule_span)
    hue = {'bp': COLORS['bp'], 'amber': AMBER_TEXT}
    f6_sign_legend(ax, sign[0], sign[1], top_color=hue[sign_hues[0]],
                   bottom_color=hue[sign_hues[1]], top_x=sign_top[0],
                   top_frac=sign_top[1], top_ha=sign_top[2],
                   bottom_x=sign_bottom[0], bottom_ha=sign_bottom[1])
    at = p.set_index('epoch')
    for epoch, tx, ty, ha, lines, leader in marks:
        value = float(at.loc[epoch, 'mean'])
        ax.plot([epoch], [value], marker='o', ms=MARKER_MS, mfc=COLORS['bp'],
                mec='white', mew=LW_EDGE, ls='none', zorder=3.4)
        if leader is not None:
            # QA 2026-09-10: mute, not the series hue -- a full-saturation
            # vertical carmine hairline at epoch 600 crossing the zero rule
            # read as an error bar or a spike in the difference curve.
            (lx0, ly0), (lx1, ly1) = leader
            ax.plot([lx0, lx1], [ly0, ly1], color=COLORS['mute'], lw=LW_HAIR,
                    zorder=1.8)
        f6_note_data(ax, tx, ty, lines, color=COLORS['ink'], ha=ha)
    for x, y, ha, lines in notes:
        f6_note_data(ax, x, y, lines, ha=ha)
    rows.extend(dict(panel='G' if metric == 'test_accuracy' else 'H',
                     record='paired curve summary',
                     metric=metric, epoch=int(e), mean=float(m),
                     ci95_low=float(a), ci95_high=float(b),
                     positive_seeds=int(ps), negative_seeds=int(ns),
                     n_seeds=10)
                for e, m, a, b, ps, ns in zip(p.epoch, p['mean'], p.ci95_low,
                                              p.ci95_high, p.positive_seeds,
                                              p.negative_seeds))
    return ax


F6_CAPTION = r'''\caption{\textbf{Serial computation benefits distributed gain correction, while credit-rule accuracy rankings depend on training duration.}
\textbf{A}, Equal-compartment DendriNet morphologies: one stage (D1, $[8]$) or three (D3, $[2,1,2]$). Blue, excitatory class-bearing contacts; carmine, inhibitory gain sensors; $\delta_0$, somatic error. D1 receives all gain tiers together; D3 separates them by stage. Footer, grouped-point control.
\textbf{B}, Gain supports: nested fine/coarse/global blocks, flat equal-resolution blocks, or local ratios with an excitation-matched inhibitory sensor in each module. $\alpha$ denotes sensor fidelity.
\textbf{C}, Serial-minus-grouped-point accuracy at fixed D3 under exact BP. Colors distinguish gain-support families; pale dots show ten paired differences at full fidelity. Open marker, analytic local-ratio tie, not an empirical null. LocalCA counterpart: Supplementary Fig.~S24C.
\textbf{D}, Three-tier task across serial depths: exact BP (black), exact-path LocalCA (red-brown), shared-soma LocalCA (amber), additive integration, grouped/reversed resource controls and a point-network reference. Exact BP and exact-path LocalCA use different optimizer recipes. Resource controls are offset horizontally. Most intervals are smaller than symbols; the separate four-tier cohort is not shown.
\textbf{E}, Validation-selected accuracy for six restarted conditions, up to 600 epochs. Dotted lines, autograd-broadcast variants, one coinciding with shared-soma LocalCA. Thin grey, D1 reference, lightened below eight active fits. Labels identify paired endpoint comparisons.
\textbf{F}, Best validation loss for the same conditions; open markers indicate D1 stopping epochs.
\textbf{G,H}, Paired exact-path-minus-shared-soma accuracy and cross-entropy. Grey span in \textbf{G}, epochs 300--325, where the interval intermittently straddles zero; dashed lines in \textbf{E--H}, original 180-epoch budget.
Means use ten paired seeds. Bars/bands are descriptive 95\% seed-bootstrap intervals: per-condition in \textbf{D--F} where drawn, paired differences elsewhere, pointwise in \textbf{E--H}. Accuracy endpoints are validation-selected. Source Data: \texttt{source\_data/curated\_publication/figure\_07\_plotted.csv}.}'''


def figure6():
    """Figure 6 (`fig:physicaldepth`) -- eight panels, letters abcdefgh.

    CF-1 canvas 518.4 x 490.0 pt, aspect 1.058, height on the 340/415/490
    ladder.  CF-2 exactly three type sizes 7.0 / 8.0 / 9.0-bold, subscripts
    through `figure_canvas.token_subscript` and `Frame.subscript`, never
    mathtext.  CF-3 strokes only 0.55 / 0.70 / 0.85 / 0.95 / 1.25 pt and every
    area mark a 16 % `journal_style.tint_patch` with a 0.55 pt edge.  CF-4 one
    ink delta-0 arrow into every soma, with panel B the declared
    DELTA0_EXEMPTIONS entry (input-side generative model) and A's
    grouped-point icon a declared soma swatch.  CF-5 zero legend artists --
    the only sanctioned key in the nine-figure set is Fig 5C's frameless
    four-entry rule key.  CF-6 Fig 6 uses no forest, recorded (Figs 5 and 6
    are the two no-forest figures).  CF-7 every reference line dashed mute at
    LW_REF with its label right-aligned on the line; the 180-epoch rule in
    E--H is named by the major x tick that sits under it (see
    `f6_budget_rule`).  CF-8 caption rules and the exact final Source Data
    sentence.  CF-9 titles sentence case, no terminal period, <= 26 characters
    at <= 4 modules and <= 42 above; Fig 6 has no recorded title exception.
    CF-10 schematic area on the B12 formula:
    sum(schematic slot w_pt x h_pt) / (live_w_pt x live_h_pt) =
    2 x 134.47 x 116 / (477.4 x 438) = 31,197 / 209,101 = 14.9 % <= 30 %, no
    waiver claimed.  Margins and vertical gutter depart from the plan's
    56/15/24/36 + 40 pt: at left 56 no left reserve is locked, the y-label
    column then runs 7 pt left of the panel letter (audit_letter_alignment)
    and the page-ink box fills only 90.9 % of the width (strict fill-width
    floor 92 %); and at vgutter 40 the ink-to-ink row gaps measure 8.4 and
    5.8 pt against the 8.5 pt (3 mm) floor of audit_row_separation.  Left 26
    with a declared 20 pt left reserve, top 22 / bottom 30 and vgutter 44 fix
    all three; the height stays 490 pt (22+116+44+118+44+116+30) and the row
    weights stay the plan's 116/118/116.  CF-11 `check_matrix_cells` >= 6.0 pt on panel B's family
    strip.  CF-12 `align_letters()` runs unconditionally inside `save()`; no
    hand-placed letters.

    Waiver D3: row 1 is three 4-module data panels; D and E share
    y = test accuracy (%) and E and F share x = epoch, and all three are
    column-locked.

    Gamma (B10): Fig 6 is not one of the four Gamma-thread figures (1, 2, 5,
    8); no Gamma token appears in the artwork or the caption.

    Deviations from the plan, recorded (fix round 2026-09-09):

    1. Panel C's `no advantage` label stays at the LEFT end of the zero rule
       instead of the CF-7 right end.  Measured: the alpha = 1 end of that
       rule carries the open exact-tie marker at exactly 0.00, its hairline
       leader at x 2.05-2.11, the ten-seed fan at x 1.90-2.10 and the two-line
       tie note below; the nested and flat connectors cross y 2.8-5.5 at
       x 1.07-1.21, so no 42 pt label fits above or below the rule right of
       x = 1 without striking one of them.  Every other reference label in the
       figure is right-aligned on its rule.
    2. WITHDRAWN by the 2026-09-10 visual review (major).  D and E shared
       y = (18, 110); the 32-point band under the 50 % chance rule existed
       only to seat D's five-line and E's four-line statistics blocks, which
       took 39 % of both panels' height and detached the x axis from the
       data.  Both blocks are deleted and the shared range is now
       (43, 110): the drawn axis is still the plan's 50-100 (`f6_trim`) and
       the remaining 7 points hold the `chance` reference label.  Every fact
       the blocks carried is in the caption and in figure_06_plotted.csv.
    3. WITHDRAWN with deviation 2.  Panel D's four-tier D4-minus-D3 contrast
       was the largest text element in the panel and reported a contrast
       drawn nowhere in the figure (there is no D4 abscissa); it is deleted
       from the artwork, stays in figure_06_plotted.csv, and moves to the
       caption.
    4. The `180-epoch budget` label asked for on E's dotted rule is OMITTED.
       Seats tried and rejected by measurement: E's top band is taken by the
       sub-title and the label cannot be right-aligned on a rule 19 pt from
       the axis origin; G's top band is crossed by the paired curve's 13.15 pp
       maximum at epoch 147; F's top band carries the stopping-epoch note; H's
       upper band is fully dealt to four one-per-baseline items.  The rule is
       named by the major x tick at 180 that sits under it in all four panels
       and by the caption's closing clause.
    5. Panel H prints the endpoint tag on one line
       (`... 9 of 10; decays 3.4x`); the plan's second line
       `ordering does not reverse` is dropped because it restates the panel
       title verbatim, and the band above the curve holds one line only (the
       curve rises to -0.0157 at epoch ~510).
    6. Panel D palette: see `f6_depth_ladder` -- `highlight` for reversed
       placement and a second `point_mlp` mark for the point-network reference
       are outside the AMENDMENTS section 5 role table's listed figures; the
       hues are kept and the table rows need Fig 6 added (integrator).
    7. Panel B's squaring exponent is set through the same baseline-shifted
       token path as the subscripts, at the one permitted size, raised 2.2 pt
       over a 10.4 pt equation pitch (`f6_formula`); no mathtext, no fourth
       type size.  The 7.0 pt text floor forbids a smaller glyph, so the
       clearance is bought with pitch, not with type size.
    8. Panels E and F seat their direct labels in a right-hand label column
       and connect each seat to its endpoint with a MUTE LW_HAIR elbow that
       starts 2.2 pt clear of the endpoint marker (`f6_direct_ends`); the
       plan's plain leader from the datum in a tint of the series hue read as
       the curve itself plunging at the last epoch.  E's column moves from
       2.4 to 10.6 pt right of epoch 600 so the elbow has a horizontal run;
       F's column stays at 5.9 pt because the right margin caps it there.
    9. The 600-epoch marks in G and H carry NO leader: any hairline from the
       note down to the endpoint marker is a 28 pt near-vertical at x = 600
       that crosses the zero rule and reads as an error bar.  Both notes name
       their own epoch (`at 600: ...`) and 600 carries the panel's only mark
       there, so the leader is redundant.
   10. Panel F's `convergence not established` caveat sits in the lower-left
       gutter under the 0.1 spine bound, not in the right-hand label column:
       it applies to all six arms, and on the column's baseline band it read
       as a second line of the `exact BP (D3)` label.
   11. Panel E's sub-title reads `600-epoch restarts, same seeds`: the
       longer wording pushed E's tight bounding box under panel F's letter
       (canvas letter check), and the `n = 10` clause was dropped on
       2026-09-10 with every other per-panel repeat of the caption footer.
       The caption keeps the full sentence.

    Visual-review round 2026-09-10 (main06.json):

   17. Panel E carries y tick labels and the rotated `Test accuracy (%)`
       title.  It previously carried neither, so no accuracy in it could be
       read; D and E share the axis and both now name it.
   18. The D1 reference has ONE dedicated style in E and F: thin solid grey
       (`'thin'`, LW_REF at ls '-'), the carried-forward segment past the
       stopping split marked by alpha 0.45.  It used to be dashed in F and
       solid-then-DOTTED in E, so a dotted curve in E was not a broadcast
       arm and the same arm changed style between panels; dotted now means
       the autograd-broadcast variant everywhere, as the caption key says.
   19. Panel G's three-line cohort footer (`n = 10 paired seeds; pointwise
       95 % bootstrap; validation-selected states`) is deleted: it repeated
       the caption footer verbatim and was the largest of the eight
       annotation elements filling a third of a data-free plot box.  The
       same duplicated clauses are stripped from C (four of five lines), F
       (`; n = 10`) and H (a whole note).  C keeps `intervals < markers`.
   20. Panel D's two resource-matched controls are dodged +-0.08 depth units
       off the tick: they agree within 0.4 points at every depth and printed
       as one series carrying two labels.  Plotted values are unchanged.
   21. Panel F states `arms and styles as in E` in the free band between the
       D1 reference and the falling bundle: F draws six arms but can seat
       only three direct labels.

    Residual-QA round 2026-09-10 (TEXT.md section 4, deviations 16-21):

   12. Panel H's sign key is INVERTED with respect to plan section 4 H.  H's
       ordinate is a loss difference in nats, so the arm that is ahead is the
       one BELOW zero: `shared soma ahead` seats at the top, `exact path
       ahead` at the bottom.  The plan copies G's gain-ordinate key, which
       contradicts H's own `below zero: lower loss with exact-path credit`
       note, both marked points and the title.  G is unchanged.
   13. Panel E seats FOUR direct labels, not six: the two broadcast arms end
       0.48 (BP) and 0.02 (local) points from their partners at 600, so each
       pair shares one seat, tied by a mute hairline bracket to one elbow.
       The caption names the coincidence so it reads as the result.
   14. Panel G's 600-epoch note moves from +6.5 pp (seven points above and
       200 epochs left of the marker it names) into the gutter under the
       trough, with the same short shallow mute diagonal the 486 mark uses:
       from there the leader runs below the curve and never crosses the zero
       rule, so the error-bar reading behind deviation 9 does not return
       (deviation 9 still holds for H).  The note states its sign.  The
       cohort footer moves to the empty band above the zero rule right of the
       crossing and the negative-end sign key to the bottom LEFT to free that
       gutter; G's ylim floor drops to -6.95 pp (drawn axis still -5 to 10).
   15. Panel B's squared term is parenthesised, `- 1/2 Sum (sigma_l)^2`: a
       7.0 pt token raised 2.2 pt straight after a lowered subscript still
       read as a separate numeral 2.  The exponent now follows a full-height
       `)`.  The raise stays 2.2 pt (deviation 7).
   16. The margin/gutter departure from plan section 0 (left 26 + 20 pt
       reserve, top 22, bottom 30, vgutter 44) and its CF-10 consequence
       (14.9 %, not 15.0 %) are now listed as a deviation to QA; plan section
       0 and section 9 check 2 are amended to match.
    """
    import build_main_figure_06 as depth
    from figure_canvas import enforce_tokens
    rows = []
    # Design pass 2026-09-14: 22 + 116 + 30 + 108 + 30 + 108 + 30 = 444 pt.
    # Row 0 keeps its 116 pt for the two schematics; rows 1 and 2 are sized
    # to their curves (the lock pass carves ~9 pt at the top of each for the
    # x labels of the row above and the letters, so their axes are ~99 pt;
    # G's and H's rotated y titles are 90 pt tall and must fit inside that).
    # Gutters 32 pt horizontally (E's y decorations are 23.4 pt wide and need
    # the 8 pt pad beside them, or D is carved 1.4 pt narrower than E) and
    # 30 pt vertically.  No data panel carries a title, and the
    # statistics prose in C, E, F, G and H is reduced to the marked values,
    # each of which the running text quotes.
    c = NativeCanvas(444 / 72, 3, row_weights=[116, 108, 108], hgutter_pt=32,
                     vgutter_pt=30,
                     margins=Margins(left=26, right=15, top=22, bottom=30))
    a = c.panel('A', 0, 0, 4, title='Same modules, more stages',
                schematic=True, lock=False)
    f6_style(a, 'Same modules, more stages')
    f6_architectures(a)
    b = c.panel('B', 0, 4, 4, title='Where the gain factors act',
                schematic=True, lock=False)
    f6_style(b, 'Where the gain factors act')
    f6_task_model(b)

    effects = read('task_family_alignment/architecture_effects.csv')
    seeds = read('task_family_alignment/seed_outcomes.csv')
    cc = c.panel('C', 0, 8, 4)
    f6_style(cc)
    f6_family_dose(cc, effects, seeds, rows)

    conf = read('nonlinear_physical_depth_confirmatory/condition_summary.csv')
    remaining = read('remaining_physical_experiments/condition_summary.csv')
    ceiling = read('point_dendrite_credit_controls/condition_summary.csv')
    contrast = read('physical_depth_h4_factorial/paired_contrasts.csv')
    d = c.panel('D', 1, 0, 4)
    f6_style(d)
    f6_depth_ladder(d, conf, remaining, ceiling, contrast, rows)

    curves = read('physical_depth_followup/condition_trajectory_summary.csv')
    stopping = read('physical_depth_followup/stopping_by_seed.csv')
    budget = read(
        'physical_depth_budget/canonical/extension_paired_contrasts.csv')
    e = c.panel('E', 1, 4, 4, sharey=d)
    f6_style(e)
    f6_accuracy_budget(e, curves, stopping, budget, rows)
    f = c.panel('F', 1, 8, 4, sharex=e)
    f6_style(f)
    f6_validation_loss(f, curves, stopping, rows)

    paired = read('physical_depth_followup/paired_trajectory_summary.csv')
    g = c.panel('G', 2, 0, 6)
    f6_style(g)
    # QA 2026-09-10: the floor drops from -6.6 to -6.95 so the two-line
    # 600-epoch note clears the 486 note above it by 0.7 pt and still ends
    # 0.3 pt clear of the x spine; the drawn axis stays -5 to 10 (`f6_trim`).
    f6_paired(g, paired, 'test_accuracy', rows, ylim=(-6.95, 13.5),
              yticks=[-5, 0, 5, 10],
              ylabel='Exact − shared soma (pp)', band=(300, 325),
              sign=('exact path ahead', 'shared soma ahead'),
              # QA 2026-09-10: 0.982, not the default 1.0 -- at the axes top
              # the key's box grazed the panel title by 0.4 pt.
              sign_top=(1.0, 0.982, 'right'),
              # QA 2026-09-10: the negative-end key moves to the bottom LEFT,
              # beside the -5 tick it belongs to, so the whole bottom-right
              # gutter is free for the 600-epoch note (below).
              sign_bottom=(0.0, 'left'),
              rule_span=(-1.2, 13.5), zero_dy=0.45, zero_at=612.0,
              # Design pass 2026-09-14: each marked epoch carries its value
              # and nothing else -- the seed counts, the interval and the
              # unresolved-crossing sentence are in the running text and the
              # caption (the shaded band IS the unresolved crossing).
              marks=((180, 205.0, 13.3, 'left', ('+10.86 pp',),
                      ((186.0, 11.5), (203.0, 12.6))),
                     (486, 300.0, -3.0, 'left', ('−2.94 pp',),
                      ((464.0, -3.6), (483.0, -3.05))),
                     (600, 600.0, -4.55, 'right', ('−1.52 pp at 600',),
                      None)),
              notes=())
    h = c.panel('H', 2, 6, 6, sharex=g)
    f6_style(h)
    f6_paired(h, paired, 'test_cross_entropy', rows, ylim=(-0.098, 0.026),
              yticks=[-0.08, -0.04, 0.0],
              ylabel='Exact − shared soma (nats)',
              # QA 2026-09-10 (blocker): H's ordinate is a LOSS difference, so
              # the arm that is ahead is the one BELOW zero.  Plan §4 H copies
              # G's gain-ordinate key and is wrong here; the two readings are
              # swapped (and their hues with them) so the key agrees with the
              # panel's own note, with both marked points and with the title.
              sign=('shared soma ahead', 'exact path ahead'),
              sign_hues=('amber', 'bp'),
              rule_span=(-0.0775, 0.0010),
              # The four items above the zero rule are dealt one per baseline:
              # the sign note keeps the top line, the endpoint tag folds to a
              # single line on the second, and the positive-sign key takes the
              # third, immediately above the rule it refers to.  The previous
              # build put the note and the key 5 pt apart on one baseline, so
              # they read as a run-on string.
              sign_top=(0.21, 0.8684, 'left'),
              zero_dy=-0.0035, zero_at=380.0, zero_va='top',
              # Design pass 2026-09-14: values only; the seed counts, the
              # 0.254 vs 0.276 endpoint pair and the decay factor are in the
              # running text, and the sign key names what `below zero` means.
              marks=((180, 188.0, -0.0785, 'left', ('−0.073 nats at 180',),
                      None),
                     (600, 612.0, 0.01759, 'right', ('−0.021 nats at 600',),
                      None)),
              notes=())

    # 22 pt, not 20: the letters sit 16 pt left of the slots (10 pt from the
    # page edge) and column 0's rotated y titles reach 26 pt out, so 22 pt
    # keeps them clear of the letter column.  H's y decorations (29.4 pt:
    # the '−0.08' tick labels plus the title) overrun the gutter, so the lock
    # pass carves G's right edge by the shortfall; H takes the same amount on
    # its left, which makes the two 6-module boxes equal without declaring a
    # right reserve on H's column boundary -- a declared right at c1 = 12
    # would leak onto C and F, which end on the same boundary.  save() is
    # told not to re-equalise, so this per-column split survives.
    for name in ('C', 'D', 'E', 'F', 'G', 'H'):
        c.declare_reserve(name, left=22.0)
    c.declare_reserve('H', left=22.0 + c.lock_reserves()['G'][1])
    enforce_tokens(c.fig)
    sources = ['task_family_alignment/architecture_effects.csv',
               'task_family_alignment/seed_outcomes.csv',
               'nonlinear_physical_depth_confirmatory/condition_summary.csv',
               'remaining_physical_experiments/condition_summary.csv',
               'point_dendrite_credit_controls/condition_summary.csv',
               'physical_depth_h4_factorial/paired_contrasts.csv',
               'physical_depth_h4_factorial/seed_outcomes.csv',
               'physical_depth_followup/condition_trajectory_summary.csv',
               'physical_depth_followup/validation_selected_seed_trajectories.csv',
               'physical_depth_followup/stopping_by_seed.csv',
               'physical_depth_followup/paired_trajectory_summary.csv',
               'physical_depth_budget/canonical/extension_paired_contrasts.csv']
    panels = {
        'A': 'Native architecture schematic (no data): D1 [8] and D3 [2,1,2] '
             'over the same eight nonsomatic compartments, the grouped-point '
             'control swatch, and the shared inventory printed from '
             'physical_depth_h4_factorial/seed_outcomes.csv (66,178 trainable '
             'parameters, 14,336 active synapses).',
        'B': 'Native generative-model schematic (no data): tier supports and '
             'family strips from configs/nonlinear_physical_depth/'
             'confirmatory/aligned_shunting_bp.yaml (n_levels 3, '
             'hierarchy_branching 2, n_flat_groups 8, e_signal_delta 0.8) and '
             'supplementary/curated/si_06_physical_depth.tex Eq. S1-S2.',
        'C': 'Frozen serial-minus-grouped-point effects under exact BP at 180 '
             'epochs, seeds 10500-10509; per-seed fan at alpha = 1 from the '
             'same cohort. No re-estimation.',
        'D': 'Frozen three-tier condition summaries, seeds 10200-10209, 180 '
             'epochs, with the parameter-matched point network and the '
             'four-tier D4-D3 paired contrast (seeds 10400-10409).',
        'E': 'Frozen 600-epoch restart trajectories for six arms, ten seeds '
             'each, with the archived pointwise intervals and the exact '
             'per-seed stopping records.',
        'F': 'Same restarts, best validation loss, with the eight exact-BP D1 '
             'stopping epochs marked; all fifty D3 fits reach the cap.',
        'G': 'Frozen paired exact-path-minus-shared-soma accuracy trajectory '
             'with archived pointwise intervals and sign counts.',
        'H': 'Same paired states, test cross-entropy difference, with the two '
             'absolute 600-epoch levels printed.'}
    for row in rows:
        row.setdefault('record', 'summary')
    live_w = 518.4 - 26 - 15
    live_h = 444.0 - 22 - 30
    frac = 100 * sum(w * h for _, _, w, h in (c.slot_pt(0, 0, 4),
                                              c.slot_pt(0, 4, 4))) \
        / (live_w * live_h)
    save(c, 6, sources, panels, F6_CAPTION, rows,
         {'helper_sha256': {str(Path(depth.__file__).relative_to(J)):
                            sha(depth.__file__)},
          'schematic_fraction_percent': round(frac, 1),
          'design_pass_2026_09_14':
              '444 pt on rows 116/108/108 at 32/30 pt gutters; no titles on '
              'C-H; C drops its tie note, E its restart sub-title, F its '
              'stopping-epoch note and two caveats, G and H every clause but '
              'the marked values (all quoted in the running text); G and H '
              'y titles shortened to fit a 99 pt axes; per-column reserves '
              'kept (save equalize=False)',
          'schematic_fraction_formula':
              'sum(schematic slot w_pt*h_pt) / (live_w_pt*live_h_pt), A and '
              'B slots over the 116 pt row against the live area',
          'no_forest': 'CF-6: Fig 6 uses no forest panel (recorded)',
          'in_axes_key_count': 0}, equalize=False)


def dictionary_cartoon(ax):
    ax.set(xlim=(0,1),ylim=(0,1));ax.set_axis_off()
    pos={0:(.18,.86),1:(.08,.61),2:(.28,.61),3:(.025,.35),4:(.13,.35),5:(.23,.35),6:(.335,.35)}
    for i in range(1,7):
        x,y=pos[i];xx,yy=pos[(i-1)//2]
        ax.plot([x,xx],[y,yy],color=COLORS['shunting']if i in(1,3,4)else COLORS['mute'],lw=LW_DATA)
    for i,(x,y)in pos.items():ax.scatter(x,y,s=25,color=COLORS['shunting']if i in(1,3,4)else COLORS['mute'],zorder=3)
    ax.text(.18,.17,'Illustrative arbor',ha='center',fontsize=PT_SMALL)
    matrix=np.column_stack([np.ones(6),[1,0,1,1,0,0],[0,1,0,0,1,1]])
    inner=ax.inset_axes([.51,.28,.35,.59]);inner.imshow(matrix,aspect='auto',cmap=ListedColormap(['white',COLORS['shunting']]),vmin=0,vmax=1)
    inner.set_xticks([0,1,2],['Common','Left','Right']);inner.set_yticks([]);inner.tick_params(length=0,labelsize=PT_SMALL)
    inner.set_ylabel('Nonsomatic sites',fontsize=PT_LABEL)
    ax.text(.5,.03,'One common signal + ancestry-defined spatial profiles',ha='center',fontsize=PT_SMALL)


def figure7():
    # Superseded 2026-09-08: main Figure 7 is drawn by
    # scripts/credit_first_figures/build_anatomy.py, which replaced the
    # commonmode.contrast_forest helper this body calls.
    raise SystemExit('main Figure 7 is built by scripts/credit_first_figures/'
                     'build_anatomy.py; this entry point is superseded')
    tables={q:read('anatomy_commonmode/'+q+'/cell_method_summary.csv')for q in ('original8','v661','pinky')}
    table=tables['v661'];report=json.loads((S/'anatomy_commonmode/v661/summary.json').read_text())
    rows=[];c=NativeCanvas(493/72,3,row_weights=[137,133,127],hgutter_pt=36,vgutter_pt=45,
                          margins=Margins(left=48,right=20,top=26,bottom=36))
    anatomy.panel_arbor(c.panel('A',0,0,6,title='Measured arbor and mapped contacts',schematic=True,lock=False))
    dictionary_cartoon(c.panel('B',0,6,6,title='The dictionary includes a shared signal',schematic=True,lock=False))
    cc=c.panel('C',1,0,6,title='Ancestry capacity across budgets',grid='y')
    specs=[(commonmode.METHODS[0],COLORS['shunting'],'o','Ancestry'),(commonmode.METHODS[1],COLORS['mute'],'s','Surrogate tree'),(commonmode.METHODS[2],COLORS['additive'],'^','Depth bins'),(commonmode.METHODS[-1],COLORS['ink'],'D','SVD oracle')]
    for method,color,marker,label in specs:
        p=table[table.method.eq(method)&table.channels.isin([1,2,4,8])].pivot(index='root_id',columns='channels',values='residual_capture').sort_index();assert p.shape==(47,4) and not p.isna().any().any()
        m,lo,hi=boot(p.to_numpy(),2609087);k=p.columns.to_numpy()
        cc.plot(k,m,color=color,marker=marker,ms=3,lw=LW_DATA,label=label);cc.fill_between(k,lo,hi,color=color,alpha=.075,lw=0)
        rows.extend(dict(panel='C',method=method,channels=int(kk),mean=float(mm),ci95_low=float(ll),ci95_high=float(hh))for kk,mm,ll,hh in zip(k,m,lo,hi))
    cc.set_xscale('log',base=2);cc.set(xlim=(.9,8.8),ylim=(-.02,1.02),xlabel='Profiles K (including the common signal)',ylabel='Spatial-residual energy captured')
    cc.set_xticks([1,2,4,8],['1','2','4','8']);cc.minorticks_off();cc.legend(frameon=False,fontsize=PT_SMALL,loc='upper left',handlelength=1.3,labelspacing=.15)
    d=c.panel('D',1,6,6,title='The spatial controls at K = 8')
    contrasts=commonmode.contrast_forest(d,report);d.set_xlabel('Ancestry advantage in residual capture (pp)');d.tick_params(axis='y',labelsize=PT_SMALL)
    rows.extend(dict(panel='D',**r)for r in contrasts.to_dict('records'))
    surrogate=commonmode.METHODS[1]
    paired_cells=focused.paired_residual_cells(table,surrogate,
        contrasts[contrasts.control.eq(surrogate)].iloc[0])
    jitter=np.random.default_rng(26090847).permutation(np.linspace(-.20,.20,len(paired_cells)))
    d.scatter(paired_cells.residual_difference_pp,3+jitter,s=8,color=COLORS['shunting'],
              alpha=.38,linewidths=0,zorder=1)
    d.set(xlim=(-27,36),xticks=[-20,0,20],ylim=(-.6,3.8))
    d.text(.99,.99,'Surrogate: 38 / 47 cells positive',transform=d.transAxes,
           ha='right',va='top',fontsize=PT_SMALL,color=COLORS['mute'])
    rows.extend(dict(panel='D',record_type='paired surrogate cell',**r)
                for r in paired_cells.to_dict('records'))
    e=c.panel('E',2,0,6,title='Capture and wiring at K = 8',grid='both')
    focus=table[table.channels.eq(8)].groupby('method').mean(numeric_only=True)
    label_positions={'Ancestry':(31,.64),'Surrogate tree':(40,.51),'Depth bins':(44,.41),
                     'Random routes':(31,.15),'Shuffled routes':(43,.29),'SVD oracle':(99,.79)}
    for method,label in zip(commonmode.METHODS,commonmode.LABELS):
        r=focus.loc[method];color=COLORS['shunting']if method==commonmode.METHODS[0]else COLORS['additive']if label=='Depth bins'else COLORS['ink']if label=='SVD oracle'else COLORS['mute']
        e.scatter(100*r.wiring_density,r.residual_capture,s=25,color=color,zorder=3)
        x,y=label_positions[label]
        e.annotate(label,(100*r.wiring_density,r.residual_capture),xytext=(x,y),
                   textcoords='data',fontsize=PT_SMALL,color=color,
                   ha='right'if label=='SVD oracle'else'left',va='center',
                   arrowprops=dict(arrowstyle='-',color=color,lw=LW_HAIR,shrinkA=2,shrinkB=3))
        rows.append(dict(panel='E',method=method,**r.to_dict()))
    e.set(xlim=(0,106),ylim=(0,1.05),xlabel='Nonzero coefficients (% of dense wiring)',ylabel='Spatial-residual energy captured')
    f=c.panel('F',2,6,6,title='Capacity across three cohorts')
    cohorts=commonmode.cohort_points(f,tables);rows.extend(dict(panel='F',**r)for r in cohorts.to_dict('records'))
    sources=['figure3/segment_metrics.csv','anatomy_commonmode/protocol_freeze.json']
    sources +=['anatomy_commonmode/'+q+'/'+n for q in tables for n in ('cell_method_summary.csv','summary.json')]
    panels={'A':'Original median-sized example actual PCA geometry and mapped E/I contacts, unchanged helper.',
      'B':'Explicitly illustrative ancestry dictionary with constant column; not an inferred measured-support matrix.',
      'C':'Corrected common-mode residual-capture atK1/2/4/8, the complete47-cell budgets, whole-cell descriptive bootstrap. K16is available only in a cell subset and is not displayed here; original source rows remain untouched.',
      'D':'Archived paired47-cell K8ancestry residual-capture differences against allfour spatial controls; original means and95%intervals unchanged. Small points show each cell for the surrogate comparison (38/47positive) in residual-energy percentage points, not total-energy capture. Other control rows retain summary intervals.',
      'E':'Same47cells,K8:mean residual capture versus actual nonzero density; no claim equalK fixes rank or wiring.',
      'F':'Original8,disjoint47 and second-mouse8 common+ancestry total/residual capture, archived helper fixed-seed cell bootstrap.'}
    caption='''**Anatomy supplies sparse spatial dictionaries beyond a shared broadcast. A,** A reconstructed arbor from the original eight-cell cohort, chosen by median segment count. Segment color reflects mapped excitatory/inhibitory contact area; line width encodes total contact area, and the scale bar is 50 μm. **B,** An explicitly illustrative dictionary combines a constant profile with ancestry-defined subtree profiles. **C,** Capture of the modeled field remaining after the weighted common projection, across the four complete profile budgets K=1,2,4,8 in 47 disjoint v661 cells. Every dictionary contains the same constant profile. Lines show cell means; shading gives descriptive whole-cell 95% intervals. The common-constrained SVD oracle is a representational ceiling. **D,** At K=8, paired ancestry advantages over surrogate-tree, depth-bin, random-site and shuffled-route controls, with the retained 95% cell-bootstrap intervals. Surrogates preserve segment depth and parent out-degree, providing the closest topology control. Small points show all 47 paired cell differences for this comparison in residual-energy coordinates, including nine negative differences; 38 are positive. Other controls retain their original mean intervals. **E,** The same K=8 dictionaries compared by average residual capture and nonzero coefficient density relative to dense eight-column wiring. Equal K does not equate rank or wiring; ancestry and its shuffled control have the same nonzero counts. **F,** Common-plus-ancestry total and residual capture across the original eight cells, 47 disjoint cells from the same mouse and eight eligible Pinky cells from a second mouse. Pinky's near-saturation involves only 9–13 excitatory-bearing sites per cell. Fields are modeled responses to focal passive shunts and therefore carry ancestry structure through cable physics; these comparisons establish representational capacity, not independent evidence of endogenous route use.'''
    save(c,7,sources,panels,caption,rows,{'helper_sha256':{str(Path(p.__file__).relative_to(J)):sha(p.__file__)for p in (anatomy,commonmode)}})


def figure9():
    # Superseded 2026-09-08: main Figure 9 is drawn by
    # scripts/credit_first_figures/build_measured.py, which replaced the
    # measured.forest helper this body calls.
    raise SystemExit('main Figure 9 is built by scripts/credit_first_figures/'
                     'build_measured.py; this entry point is superseded')
    rows=[];c=NativeCanvas(365/72,2,row_weights=[135,153],hgutter_pt=36,vgutter_pt=52,
                          margins=Margins(left=58,right=17,top=26,bottom=39))
    a=c.panel('A',0,0,12,title='Observed ancestry–response similarity in seven targets')
    sums=read('review_evidence_reanalysis/functional_native_contrasts.csv');vals=read('review_evidence_reanalysis/functional_native_target_effects.csv');data=[]
    for mode,label in [('selected_scans','Selected scans'),('scan_complete','All eligible scans')]:
        r=sums[sums.endpoint.eq('structure_function_partial_r')&sums.comparison.eq(mode)].iloc[0]
        v=vals[vals.endpoint.eq('structure_function_partial_r')&vals.comparison.eq(mode)].effect.to_numpy();assert len(v)==7
        data.append((label,v,(r['mean'],r.ci95_low,r.ci95_high),COLORS['shunting']))
        rows.append(dict(panel='A',**r.to_dict()))
    measured.forest(a,data,'Partial rank correlation',(-.65,.55));a.axvline(0,color=COLORS['mute'],lw=LW_REF,ls='--')
    b=c.panel('B',1,0,6,title='Sensitivity with the measured sampling',grid='y')
    power=read('measured_alignment_power/power_summary.csv')
    for scenario,color,ls,label in [('measured',COLORS['shunting'],'-','Measured reliability'),('perfect',COLORS['mute'],'--','Perfect reliability')]:
        p=power[power.reliability.eq(scenario)].sort_values('lambda')
        b.plot(p['lambda'],p.power,color=color,ls=ls,lw=LW_DATA,label=label)
        b.fill_between(p['lambda'],p.power_ci95_low,p.power_ci95_high,color=color,alpha=.12,lw=0)
        rows.extend(dict(panel='B',**r)for r in p.to_dict('records'))
    b.axhline(.8,color=COLORS['edge'],lw=LW_REF,ls=':');b.axvspan(.55,.60,ymin=.72,ymax=.82,color=COLORS['shunting'],alpha=.15,lw=0)
    b.set(xlim=(0,1),ylim=(0,1.05),xticks=[0,.5,1],yticks=[0,.5,.8,1],xlabel='Ancestry variance fraction λ (simulated)',ylabel='Positive-alignment detection probability')
    b.legend(frameon=False,fontsize=PT_SMALL,loc='lower right',handlelength=1.5)
    b.text(.97,.53,'80% detection corresponds to\npartial r ≈ 0.25\nin this simulation model',
           ha='right',va='center',transform=b.transAxes,fontsize=PT_SMALL,color=COLORS['mute'])
    cc=c.panel('C',1,6,6,title='Repeat reliability of the observed inputs',grid='y')
    reliability=read('measured_alignment_power/reliability_calibration_audit.csv')
    values=reliability.measured_split_half_spearman.to_numpy();assert len(values)==125
    bins=np.linspace(-.3,1,14);assert values.min()>=bins[0]and values.max()<=bins[-1]
    cc.hist(values,bins=bins,color=COLORS['mute'],edgecolor='white',lw=LW_HAIR)
    cc.axvline(0,color=COLORS['edge'],lw=LW_REF,ls=':')
    cc.set(xlim=(-.3,1),xticks=[0,.5,1],xlabel='Split-half Spearman correlation',ylabel='Partner–scan observations')
    cc.text(.97,.97,'125 observations\n102 partners · 13 scans',ha='right',va='top',transform=cc.transAxes,fontsize=PT_SMALL)
    rows.extend(dict(panel='C',**r)for r in reliability.to_dict('records'))
    sources=['review_evidence_reanalysis/functional_native_contrasts.csv','review_evidence_reanalysis/functional_native_target_effects.csv',
             'measured_alignment_power/power_summary.csv','measured_alignment_power/reliability_calibration_audit.csv',
             'measured_alignment_power/RESULTS.json','measured_alignment_power/protocol_freeze.json']
    panels={'A':'Original empirical7target partial-rank estimates; selected/all-eligible scan policies, originalCI.',
      'B':'Completed4000joint-dataset simulation; observed partners/shared-pair dependencies,13scans7targets,exact two-sided n7signrank plus positive effect; measured/perfect reliability.',
      'C':'Measured repeat Spearman reliabilities for all125partner-recording observations (102unique partners); no simulation outcomes in histogram. No independent-observation test.'}
    caption='''**Measured responses provide no evidence of preferential ancestry alignment at the available sensitivity. A,** Empirical ancestry–response partial-rank correlations after distance adjustment. Small points are seven target-level estimates; diamonds and bars show means and the archived descriptive 95% target-bootstrap intervals. Selected scans and all eligible scans are displayed separately. **B,** Conditional sensitivity simulations preserve the actual partners, repeated partners and shared-pair dependence across thirteen scans in seven targets. The horizontal axis λ is the simulated fraction of latent response variance allocated to the specified Gaussian ancestry kernel; it is not an observed correlation. Each point summarizes 4,000 complete simulated datasets tested by the exact two-sided seven-unit signed-rank test at 0.05, additionally requiring a positive mean effect. Shading is a 95% Monte Carlo interval. With measured repeat attenuation, sustained 80% detection is bracketed by λ=0.55–0.60; the corresponding interpolated mean partial-rank effect is approximately 0.249 within this model family. This is not a universal detectable biological correlation. The exact seven-unit test has minimum two-sided P=0.015625; no universal power ceiling below one follows from that lattice. **C,** The measured split-half response correlations used for attenuation, retaining all 125 partner–scan records from 102 unique partners and thirteen scans. Negative measured reliabilities are displayed here and clipped to zero only in the simulation's nonnegative signal-variance calibration. The measured-reliability null gives two-sided type-I error 0.0495. Actual route support, coverage, response-prediction comparisons with ridge and fixed-profile reconstruction remain in the Supplementary Information.'''
    save(c,9,sources,panels,caption,rows,{'helper_sha256':{str(Path(measured.__file__).relative_to(J)):sha(measured.__file__)}})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',nargs='+',type=int,default=[4,6],choices=[1,4,6,7,9])
    parser.add_argument('--emit-main',action='store_true')
    args=parser.parse_args()
    for number in args.figures:
        globals()[f'figure{number}']()
        if args.emit_main:
            public_number = {6: 7}.get(number, number)
            shutil.copyfile(OUT/f'restored_main_{number:02d}.pdf', J/f'figures/main/figure_{public_number:02d}.pdf')


if __name__=='__main__':main()
