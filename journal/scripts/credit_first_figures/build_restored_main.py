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


def save(canvas, number, sources, panels, caption, rows=(), extra=None):
    OUT.mkdir(exist_ok=True);REC.mkdir(parents=True,exist_ok=True)
    path=OUT/f'restored_main_{number:02d}.pdf'
    style_direct_color_labels(canvas.fig)
    # Equal module spans keep equal plotting widths despite long forest labels.
    locks=canvas.lock_reserves()
    groups={}
    for rec in canvas._records:
        if not rec.get('schematic'):
            groups.setdefault(rec['colspan'],[]).append(rec['name'])
    for group in groups.values():
        if len(group)<2:continue
        left=max(locks[name][0]for name in group)
        right=max(locks[name][1]for name in group)
        for name in group:canvas.declare_reserve(name,left=left,right=right)
    findings=canvas.save(path,name=f'restored_main_{number:02d}',dpi=180)
    plt.close(canvas.fig)
    pd.DataFrame(rows).to_csv(REC/f'figure_{number:02d}_plotted.csv',index=False)
    helpers=[Path(__file__),Path(focused.__file__),J/'scripts/figure_canvas.py',J/'scripts/journal_style.py']
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
    f.text((.5, 1.0), 'same tree, weights and examples; I8/4; signs ±0.5',
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
        f.error_in(nodes.soma, side='right', label='δ0')
    f.require_soma_lowest()
    f.require_delta0()
    return f


def f4_profile_bus(f, nodes, radii):
    """Private helper (DECISIONS G5): the calibrated-broadcast bus.

    The ``mode='neuron'`` bus of ``Frame.credit_delivery`` in
    ``COLORS['additive']``, with each drop terminated by a filled disc whose
    radius encodes the mean absolute calibrated weight at that site.
    Recommended upstream as a ``drop_radii`` keyword on ``credit_delivery``.
    """
    f.credit_delivery(nodes, mode='neuron', targets=list(radii),
                      rule_color='additive', label=None)
    for site, r_pt in radii.items():
        f.disc(nodes[site], r_pt, fill=COLORS['additive'], zorder=6.2)


def f4_delivery_panel(ax, radii):
    """Panel B — where each rule delivers credit."""
    from native_schematics import Frame
    f = Frame(ax)
    cells = f.split(3, axis='x', gap_pt=9.0)
    y0, hh = f.fy(13.0), 1.0 - f.fy(13.0)
    spec = [('Exact path', 'per-site q(x)'),
            ('Unit broadcast', 'same drop, six sites'),
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
            f.credit_delivery(nodes, mode='neuron', targets=list(FIG4_SITES),
                              rule_color='scalar', label=None)
        else:
            f4_profile_bus(f, nodes, radii)
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

    ``native_schematics.reference_line`` sets the label at the line's RIGHT
    end.  In C--E every curve has converged onto the noise floor there and
    the vertical 1,024 rule reaches the top of the axes, so the label is set
    at the free end of the same line instead -- still on the line, never in a
    key.
    """
    from journal_style import PT_BASE
    if axis == 'y':
        lo, hi = ax.get_xlim()
        line, = ax.plot([lo, hi], [value, value], color=COLORS['mute'],
                        lw=LW_REF, zorder=1.0, solid_capstyle='butt')
        if label:
            ax.annotate(label, xy=(lo, value), xytext=(2.0, -1.6),
                        textcoords='offset points', fontsize=PT_BASE,
                        color=COLORS['mute'], ha='left', va='top', zorder=5)
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
            for _, trace in w.iterrows():
                ax.plot(steps, trace.to_numpy(), color=COLORS['bp'],
                        lw=LW_HAIR, alpha=.28, zorder=2.0,
                        solid_capstyle='round')
        mean, lo, hi = boot(w.to_numpy())
        f4_band(ax, steps, lo, hi, RC[rule])
        line, = ax.plot(steps, mean, color=RC[rule], lw=LW_DATA, zorder=3.0,
                        solid_capstyle='round')
        if rule == 'unit_broadcast':
            line.set_dashes((2.6, 1.6))
        rows.extend(dict(panel=panel, task=FIG4_NAME[task], series=RN[rule],
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
        x = x0 + .02 * (x1 - x0)
        ax.plot([x], [y], marker='D', ms=MARKER_MS - .8, mfc=mfc,
                mec=COLORS['bp'] if mfc == 'white' else 'white',
                mew=LW_HAIR, ls='none', zorder=4.0, clip_on=False)
        ax.annotate(name, xy=(x, y), xycoords=('data', 'data'),
                    xytext=(4.5, 0.0), textcoords='offset points',
                    ha='left', va='center', fontsize=PT_BASE,
                    color=COLORS['mute'], annotation_clip=False)


def f4_row_counts(ax, ypos, label='n = 20'):
    """CF-6's right-aligned per-row n, set inside the axes."""
    from journal_style import PT_BASE
    for y in ypos:
        ax.annotate(label, xy=(.985, y), xycoords=('axes fraction', 'data'),
                    ha='right', va='center', fontsize=PT_BASE,
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
        entries.append(dict(label=f'{budget:,}', mean=float(end['mean']),
                            lo=float(end.ci95_low), hi=float(end.ci95_high),
                            seeds=list(map(float, draw)), n=20, marker='D'))
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
    out = forest_rows(ax, entries, value_label='Quartic − pairwise deficit\n'
                      '(calibrated − exact NMSE)', reference=0.0,
                      reference_label='no deficit', color='bp',
                      xlim=(-.06, 1.56), tag='')
    for y, (m, lo, hi) in zip(out['ypos'], second):
        yy = y + .22
        ax.plot([lo, hi], [yy, yy], color=COLORS['bp'], lw=LW_ERR, zorder=3.0,
                solid_capstyle='butt')
        for xb in (lo, hi):
            ax.plot([xb, xb], [yy - .09, yy + .09], color=COLORS['bp'],
                    lw=LW_ERR, zorder=3.0, solid_capstyle='butt')
        ax.plot([m], [yy], marker='D', ms=MARKER_MS - .8, mfc='white',
                mec=COLORS['bp'], mew=LW_HAIR, ls='none', zorder=4.0)
    f4_row_counts(ax, out['ypos'])
    f4_forest_strip(ax, extra_rows=1.15, n_rows=len(entries),
                    key=((COLORS['bp'], 'endpoint'),
                         ('white', 'validation-selected')))
    return out


def f4_energy(ax, eigen, diag, rows, ramp):
    """Panel G — cumulative captured path-field energy against k."""
    from journal_style import PT_BASE
    from matplotlib.ticker import FixedLocator
    field = eigen[eigen.field.eq('path_q') & eigen.rule.eq('exact')
                  & eigen.optimizer.eq('adam') & eigen.selected_rate]
    kk = np.arange(1, 7, dtype=float)
    style = {'matching': ('o', True), 'quartet': ('s', False),
             'nested': ('^', True)}
    uni_x = -0.95                 # the 'uniform' tick is 30 pt wide: at
    ax.set(xlim=(-1.62, 6.55), ylim=(0, 1.16))   # x = 0 it ran into '1'
    for task, colour in ramp.items():
        cut = field[field.family.eq(task) & field.step.eq(1024)]
        w = cut.pivot_table(index='seed', columns='index', values='fraction')
        assert len(w) == 20
        cum = w.cumsum(axis=1).to_numpy()[:, :6]
        mean, lo, hi = boot(cum)
        f4_band(ax, kk, lo, hi, colour)
        marker, filled = style[task]
        ax.plot(kk, mean, color=colour, lw=LW_DATA, marker=marker,
                ms=MARKER_MS - 1.2, mfc=colour if filled else 'white',
                mec=colour, mew=LW_HAIR, zorder=3.0)
        uni = diag[diag.task.eq(task) & diag.step.eq(1024)]
        assert len(uni) == 20
        um, ul, uh = boot(uni.path_uniform_oracle_capture.to_numpy(), 771009)
        ax.errorbar([uni_x], [um], yerr=[[um - ul], [uh - um]], color=colour,
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
    ax.set_ylabel('Captured path-field energy', fontsize=PT_LABEL,
                  color=COLORS['ink'])
    ax.xaxis.set_major_locator(FixedLocator([uni_x, 1, 2, 3, 4, 5, 6]))
    ax.set_xticklabels(['uniform\nnot fitted', '1', '2', '3', '4', '5', '6'])
    ax.yaxis.set_major_locator(FixedLocator([0, .5, 1]))
    ax.set_yticklabels(['0', '0.5', '1.0'])
    ax.plot([0.0, 0.0], [0, 1.16], color=COLORS['grid'], lw=LW_HAIR,
            zorder=.9, solid_capstyle='butt')
    f4_reference(ax, .95, axis='y', label='0.95')
    for task, y in (('matching', .70), ('quartet', .59), ('nested', .48)):
        ax.annotate(FIG4_NAME[task], xy=(5.95, y), ha='right', va='center',
                    fontsize=PT_BASE, color=ramp[task])
    ax.annotate('dashed grey: initial (shared)\n'
                '16,384 updates: k = 1 gives\n1.00 / 0.40 / 0.40',
                xy=(.62, .40), ha='left', va='top', fontsize=PT_BASE,
                color=COLORS['mute'], linespacing=1.3)
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
    """Panel H — the leaf-assignment control, three families and the pool."""
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
                          ('nested', 'Nested'), ('all', 'Pooled')):
        rec = contrast[contrast.optimizer.eq('adam')
                       & contrast.structure.eq('paired')
                       & contrast.contrast.eq(key)
                       & contrast.family.eq(family)]
        assert len(rec) == 1
        rec = rec.iloc[0]
        assert int(rec.positive_seeds) == 20 and int(rec.n_seeds) == 20
        assert abs(float(rec['mean']) - float(fans[family].mean())) < 1e-9
        entries.append(dict(label=label, mean=float(rec['mean']),
                            lo=float(rec.ci95_low), hi=float(rec.ci95_high),
                            seeds=list(map(float, fans[family])), n=20,
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
    ax.plot([-.04, 1.05], [2.5, 2.5], color=COLORS['grid'], lw=LW_HAIR,
            zorder=1.2, solid_capstyle='butt')
    f4_row_counts(ax, out['ypos'])
    f4_forest_strip(ax, extra_rows=.22, n_rows=len(entries))
    return out


FIG4_CAPTION = r"""\caption{\textbf{Input-spectrum-matched targets develop different credit geometry and different tolerance of calibrated broadcast credit.}
\textbf{A}, Pairwise and quartic targets on one shared balanced tree: soma lowest, inputs $x_1$--$x_8$, student output $y$, somatic error $\delta_0$, and only the junction badges differing; tree, initialization, examples and input-sensitivity second moment $I_8/4$ are shared. Schematic, no data.
\textbf{B}, The three credit rules over the six nonsomatic sites: exact path (per-site $\bm q(x)$), unit broadcast (amber bus) and calibrated broadcast (blue bus; drop-dot radii are the mean absolute calibrated weight over the twenty seeds, $0.133$--$0.386$, signs mixed, set at step $0$ from 256 unlabeled examples). Schematic, no data.
\textbf{C}, Held-out NMSE against updates on the pairwise target at the frozen rule-specific rates; label-noise floor dashed ($n=20$ paired seeds, 95\% percentile bootstrap bands, fixed checkpoints).
\textbf{D}, As \textbf{C} for the quartic target, all twenty exact trajectories drawn: three stall past the primary checkpoint and reach the floor by 4,096 updates ($n=20$ seeds, 95\% bands, fixed checkpoints).
\textbf{E}, A depth-four nested control on its own tree, not sharing $I_8/4$ ($n=20$ seeds, 95\% bands, fixed checkpoints); \textbf{C}--\textbf{E} denominators are the clean target variances ($1.0$, $0.5$, $1.0$), so panel heights are not comparable, and the teal ramp orders targets within this figure only.
\textbf{F}, Quartic minus pairwise difference in calibrated-broadcast minus exact NMSE at four budgets; dots the paired seeds, filled diamonds endpoint states, open diamonds validation-selected states ($n=20$ seeds, 95\% paired percentile intervals, 20 of 20 positive at every budget).
\textbf{G}, Cumulative exact-trained path-field energy captured by the best $k$ oracle directions, the uniform profile a separate unfitted category ($n=20$ seeds, 95\% percentile intervals, fixed checkpoint at 1,024 updates).
\textbf{H}, Exact-path NMSE rises when leaf input assignments are shuffled at fixed tree shape and parameter count ($n=20$ seeds per target, 95\% paired percentile intervals, 20 of 20 positive, endpoint state at 1,024 updates; copy in Supplementary Fig.~S13C).
Source Data: \texttt{source\_data/curated\_publication/figure\_04\_plotted.csv}.}"""


def figure4():
    """Main Figure 4, fig:prospective — eight panels A-H on one native canvas.

    CF-1 canvas 518.4 x 490.0 pt, aspect 1.058, height on the 340/415/490
    ladder.  CF-10 schematic_fraction = 25.3 % on the AMENDMENTS B12 formula
    (A 171.5 pt + B 254.9 pt over a 118 pt row against the live area
    463.4 x 430 pt); schematic waiver: none.  The left margin is 40 pt, not
    the plan's 58 pt: at 58 pt the strict audit fails `fill-width` (content
    fills 90.7 % of the canvas, gate 92 %).  The fraction is unchanged to a
    tenth of a point by that substitution (25.2 % at 58 pt).  waiver D3: row 2 (F seed strip / G energy curve /
    H shuffle forest) is three 4-module panels that share no axis; each is a
    different estimand and the row is column-locked.  CF-5: zero legend
    artists -- the only sanctioned key in the nine-figure set is the frameless
    rule key inside Fig 5C, and B's rule-key strip is a schematic panel's
    footer, not an in-axes key.  CF-4 DELTA0_EXEMPTIONS: one, panel E's
    morphology icon ("morphology icon in a data panel; credit delivery is
    drawn in panel B").  Private helpers under DECISIONS G5 (library
    follow-ups): f4_card, f4_operator_badges, f4_profile_bus, f4_badge,
    f4_title, f4_reference, f4_forest_tag, f4_row_counts.
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

    c = NativeCanvas(490 / 72, 3, row_weights=[118, 116, 116], hgutter_pt=37,
                     vgutter_pt=40,
                     margins=Margins(left=40, right=15, top=24, bottom=36))
    a = c.panel('A', 0, 0, 5, title=None, schematic=True, lock=False,
                inset_pt=(10, 10, 8, 10))
    f4_title(a, 'Two targets, one shared tree')
    f4_targets_panel(a)
    b = c.panel('B', 0, 5, 7, title=None, schematic=True, lock=False,
                inset_pt=(10, 10, 8, 10))
    f4_title(b, 'Where each rule delivers credit',
             right_sub=('six nonsomatic sites; radius = mean |weight| over '
                        '20 seeds, 0.133–0.386', COLORS['mute']))
    f4_delivery_panel(b, radii)

    spec = {'matching': ('Pairwise: broadcast fits', ('frozen Adam rates',),
                         ('n = 20 paired seeds', COLORS['mute'])),
            'quartet': ('Quartic: only exact fits',
                        ('3 of 20 exact seeds stall',
                         'all at floor by 4,096'),
                        ('unit and calibrated broadcast', COLORS['additive'])),
            'nested': ('Nested control: depth 4',
                       ('depth 4, unmatched spectrum',),
                       ('n = 20 paired seeds', COLORS['mute']))}
    curve_axes = {}
    for letter, col, task in (('C', 0, 'matching'), ('D', 4, 'quartet'),
                              ('E', 8, 'nested')):
        ax = c.panel(letter, 1, col, 4, title=None, grid='none')
        curve_axes[task] = ax
        if task == 'nested':
            tint_patch(ax, ('rect', FIG4_XLIM[0], FIG4_YLIM[0],
                            FIG4_XLIM[1] - FIG4_XLIM[0],
                            FIG4_YLIM[1] - FIG4_YLIM[0]), color='mute', pct=16,
                       edge=True, lw=LW_HAIR, radius_pt=2.0, zorder=0.0,
                       clip_on=True, transform=ax.transData)
        f4_curve(ax, data, task, letter, rows, seed_layer=(task == 'quartet'))
        f4_title(ax, spec[task][0], accent=ramp[task], sub=spec[task][1],
                 right_sub=spec[task][2], pad=34.0)
        if letter == 'C':
            ax.set_ylabel('Test NMSE', fontsize=PT_LABEL, color=COLORS['ink'])
            ax.set_yticklabels(['0.02', '0.1', '1'])
        else:
            ax.set_yticklabels([])
    d = curve_axes['quartet']
    d.yaxis.set_minor_locator(FixedLocator([.045]))
    d.annotate('Exact path', xy=(16000, .0295), ha='right', va='center',
               fontsize=PT_BASE, color=COLORS['bp'])
    e = curve_axes['nested']
    f4_badge(e, .965, .965, 'control')
    icon = e.inset_axes([.50, .255, .275, .400])
    fi = Frame(icon, labels=True)
    fi.text((.5, .02), 'depth 4', size=PT_BASE, color=COLORS['mute'],
            ha='center', va='bottom')
    nodes = fi.balanced_tree((0.0, fi.fy(10.0), 1.0, 1.0 - fi.fy(10.0)),
                             depth=2, mode='plain', trunk=True, labels=False)
    for name in nodes.terminals:                    # levels three and four
        x, y = nodes[name]
        for dx in (-1.0, 1.0):
            tip = (x + fi.fx(3.4 * dx), y + fi.fy(7.0))
            fi.dendrite((x, y), tip, level=2)
            for dx2 in (-1.0, 1.0):
                fi.dendrite(tip, (tip[0] + fi.fx(2.0 * dx2),
                                  tip[1] + fi.fy(5.0)), level=3)
    fi.require_soma_lowest()
    fi.require_delta0(allow_no_delta0=True,
                      reason='morphology icon in a data panel; credit '
                             'delivery is drawn in panel B')
    # The icon lives on an inset, which the canvas manifest does not walk, so
    # its declared CF-4 exemption is carried onto panel E's own note list.
    e._journal_schematic_notes = list(
        getattr(e, '_journal_schematic_notes', []) or []) + list(fi._notes)

    f_ax = c.panel('F', 2, 0, 4, title=None, grid='none')
    f4_title(f_ax, 'Larger quartic deficit', pad=22.5,
             sub=('20/20 positive at every budget',),
             right_sub=('n = 20; mean [95 % CI]', COLORS['mute']))
    f4_deficit(f_ax, contrast, seedfan, rows)
    g_ax = c.panel('G', 2, 4, 4, title=None, grid='none')
    f4_title(g_ax, 'One direction, or five', pad=22.5,
             sub=('oracle fit at 1,024 updates',),
             right_sub=('n = 20 seeds', COLORS['mute']))
    f4_energy(g_ax, eigen, diag, rows, ramp)
    f4_badge(g_ax, .965, .975, 'oracle')
    h_ax = c.panel('H', 2, 8, 4, title=None, grid='none')
    f4_title(h_ax, 'Leaf assignment matters', pad=22.5,
             sub=('tree shape, parameters fixed',),
             right_sub=('n = 20; mean [95 % CI]', COLORS['mute']))
    f4_shuffle(h_ax, morph, ends, rows, ramp)
    # The panel titles are set with loc='left', which NativeCanvas cannot
    # measure (it reads ax.get_title(), i.e. the centred title), so the title
    # block, its sub-lines and the panel letter are declared here: 8.5 pt of
    # clear gutter between every pair of rows (audit_row_separation) and a
    # 25 pt label column so no panel's ink enters the letter column
    # (audit_letter_alignment).
    for name in ('C', 'D', 'E'):
        c.declare_reserve(name, left=25.0, top=16.0)
    for name in ('F', 'G', 'H'):
        c.declare_reserve(name, left=25.0, top=26.0)

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
           for p in ('C', 'D', 'E')},
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
        'H': 'morphology_credit/summaries/fresh/paired_contrasts.csv '
             '(shuffled minus compatible: exact, paired, Adam) with the '
             'per-seed differences recomputed from selected_endpoints.csv; '
             'morphology_credit/protocol_freeze.json fixes the twenty fresh '
             'seeds.'}
    contract = {
        'CF-1 canvas': '518.4 x 490.0 pt, aspect 1.058, ladder 340/415/490',
        'CF-10 schematic_fraction':
            '25.3 % on the AMENDMENTS B12 formula (A 171.5 + B 254.9 pt over '
            '118 pt, live area 463.4 x 430 pt); waiver: none',
        'margin deviation': 'left margin 40 pt, not the plan\'s 58 pt: at '
                            '58 pt the strict audit fails fill-width '
                            '(90.7 % of the canvas width, gate 92 %)',
        'waiver D3': 'row 2 (F seed strip / G energy curve / H shuffle forest) '
                     'is three 4-module panels that share no axis; each is a '
                     'different estimand and the row is column-locked',
        'CF-5 legends': 'zero legend artists; B rule-key strip is a schematic '
                        'footer, not an in-axes key',
        'CF-4 delta0 exemptions': ['panel E morphology icon in a data panel; '
                                   'credit delivery is drawn in panel B'],
        'G5 private helpers': ['f4_card', 'f4_operator_badges',
                               'f4_profile_bus', 'f4_badge', 'f4_title',
                               'f4_reference', 'f4_forest_tag',
                               'f4_row_counts'],
        'rank95_at_1024': {k: round(float(v), 2) for k, v in rank95.items()}}
    save(c, 4, sources, panels, FIG4_CAPTION, rows,
         {'original_capture_rows_joined': len(joined),
          'original_capture_replay_max_abs_difference': replay,
          'csv_parse_mode': 'round_trip; replay equality checked to '
                            'floating-point precision',
          'canvas_contract': contract})


def f6_formula(f, xy, parts, *, size=None, color=None):
    """Private helper (errata #7): an expression as baseline-shifted runs.

    `build_main_figure_06.formula` measures its advance widths in points and
    lays them out in DATA units, which only works on an axes whose data
    coordinates are points; a `Frame` is a 0-1 frame, so the same idea is
    re-expressed here with annotation chaining (x from the previous run, y
    always from the first run) so no baseline drifts and no mathtext is
    emitted.  Recommended upstream as a `Frame.formula` method.
    """
    from journal_style import COLORS, PT_BASE
    size = PT_BASE if size is None else size
    color = COLORS['ink'] if color is None else color
    ax = f.ax
    base = ax.text(xy[0], xy[1], parts[0][0], fontsize=size, color=color,
                   ha='left', va='baseline', zorder=6)
    prev = base
    for text, dy in parts[1:]:
        prev = ax.annotate(text, xy=(1.0, 0.0), xycoords=(prev, base),
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
    for px, py in tree[-1]:                       # class-bearing distal input
        f.contact((px, py + f.fy(3.2)), kind='exc')
    right = core[0] + core[2]
    if depth == 1:
        px, py = tree[0][1]
        f.contact((base[0] + (px - base[0]) * 0.5,
                   base[1] + (py - base[1]) * 0.5), kind='inh')
        f.text((right, base[1] + f.fy(5.0)), 'all tiers', size=PT_BASE,
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
    lines = ('D2 [2,3] and D4 [1,1,2,2] keep', 'the same inventory: 66,178',
             'parameters and 14,336 active', 'synapses in every architecture')
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
    f.text((f.fx(1.0), rail_y - f.fy(5.0)), 'sensors report tier ℓ with '
           'fidelity α', size=PT_BASE, color=COLORS['mute'], ha='left',
           va='top')
    y_eq = rail_y - f.fy(17.0)
    f6_formula(f, (f.fx(1.0), y_eq),
               [('x', 0.0), ('E', -1.7), (' = (b', 0.0), ('E', -1.7),
                (' + y Δ + ε', 0.0), ('E', -1.7), (')', 0.0)])
    f6_formula(f, (f.fx(1.0), y_eq - f.fy(8.6)),
               [('× exp(Σ σ', 0.0), ('ℓ', -1.7), (' z', 0.0), ('ℓ', -1.7),
                (' − ½ Σ σ', 0.0), ('ℓ', -1.7), ('2', 2.2), ('),  Δ = 0.80,',
                                                             0.0)])
    f6_formula(f, (f.fx(1.0), y_eq - f.fy(17.2)),
               [('σ', 0.0), ('ℓ', -1.7), (' = 0.25', 0.0)])
    fam = (('nested', ORDINAL_RAMP[3], 'nested'),
           ('flat', ORDINAL_RAMP[2], 'flat'),
           ('local ratio', ORDINAL_RAMP[1], 'ratio'))
    strip_top = y_eq - f.fy(23.0)
    row_pt = 10.4
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
           'same product, different spatial supports', size=PT_BASE,
           color=COLORS['mute'], ha='left', va='top')
    f.text((f.fx(1.0), strip_top - f.fy(3 * row_pt + 10.0)),
           '÷ : E and I meet inside each module', size=PT_BASE,
           color=ORDINAL_RAMP[1], ha='left', va='top')
    f.require_soma_lowest()          # no tree in this card: a no-op assertion
    f.require_delta0(allow_no_delta0=True,
                     reason='input-side generative model; the tree, its soma '
                            'and δ0 are drawn in panel A')
    return f


def f6_style(ax, title):
    """Tighten the label bands so three rows clear 3 mm (audit_row_separation)."""
    from journal_style import COLORS, PT_EMPH
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


def f6_sign_legend(ax, top, bottom, *, top_color, bottom_color):
    """The paired panels' sign key, at the two ends of the y range.

    Plan §4 G: the sign of the difference is named on the panel, not by a
    second data artist.  The plan's rotated placement beside the y axis was
    measured against the tick column and the y label and collides with both
    (three text-collision violations), so the two readings sit at the top and
    the bottom of the panel's own right edge, outside every annotation block.
    """
    from journal_style import PT_BASE
    ax.text(1.0, 1.0, top, transform=ax.transAxes, fontsize=PT_BASE,
            color=top_color, ha='right', va='top', zorder=6, clip_on=False)
    ax.text(1.0, 0.0, bottom, transform=ax.transAxes, fontsize=PT_BASE,
            color=bottom_color, ha='right', va='bottom', zorder=6,
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
            rows.append(dict(panel='C', family=key, alignment_alpha=1.0,
                             credit='bp', estimand='serial_minus_grouped',
                             seed=int(seed), mean_pp=float(value)))
    for name, y, colour in (('nested', 30.32, ORDINAL_RAMP[3]),
                            ('flat', 22.20, ORDINAL_RAMP[2]),
                            ('local ratio', 9.0, ORDINAL_RAMP[1])):
        ax.text(2.13, y, name, fontsize=PT_BASE, color=colour, va='center',
                ha='left', zorder=6, clip_on=False)
    ax.plot([2.055, 2.105], [1.3, 7.6], color=ORDINAL_RAMP[1], lw=LW_HAIR,
            zorder=1.6)
    ax.plot([2.0, 2.0], [-0.95, -2.45], color=ORDINAL_RAMP[1], lw=LW_HAIR,
            zorder=1.6)
    f6_note_data(ax, 2.85, -2.6, ('exact tie at α = 1:',
                                  'all ten pairs identical'),
                 color=ORDINAL_RAMP[1], ha='right')
    f6_note_data(ax, -0.15, 36.5, ('fixed D3, exact BP,', '180 epochs',
                                   'n = 10 paired seeds',
                                   '95 % paired bootstrap',
                                   'intervals < markers'))
    return ax


def f6_depth_ladder(ax, conf, remaining, ceiling, contrast, rows):
    """Fig 6D: the three-tier depth ladder with its two controls."""
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
    ax.set(xlim=(0.80, 4.42), ylim=(32.0, 104.0), xticks=[1, 2, 3],
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
    for name, colour, marker, dash, regime, arch in (
            ('grouped', COLORS['point_mlp'], 'D', (0, (4.2, 2.0)), 'aligned',
             'grouped_point'),
            ('reversed', COLORS['highlight'], 'v', (0, (1.2, 1.6)),
             'rewired_tree', 'serial_tree')):
        sub = remaining[remaining.hierarchy.eq(3) & remaining.regime.eq(regime)
                        & remaining.architecture.eq(arch)
                        & remaining.credit.eq('full_bp')].sort_values('depth')
        x = sub.depth.to_numpy(float)
        m = 100 * sub.mean_test_accuracy.to_numpy(float)
        lo = 100 * sub.ci_low.to_numpy(float)
        hi = 100 * sub.ci_high.to_numpy(float)
        ax.plot(x, m, color=colour, lw=LW_DATA, ls=dash, zorder=2.2,
                solid_capstyle='round')
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt=marker, ms=MARKER_MS,
                    mfc='white', mec=colour, mew=LW_EDGE, ecolor=colour,
                    elinewidth=LW_ERR, capsize=2.0, zorder=3.0)
        label_y[name] = (float(m[-1]), colour)
        rows.extend(dict(panel='D', arm=name, depth=int(d),
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
    rows.append(dict(panel='D', arm='point network ceiling', depth=0,
                     mean_test_accuracy_pp=100 * float(top.mean_test_accuracy),
                     ci95_low_pp=lo, ci95_high_pp=hi, n_seeds=10))
    for name, y in (('exact BP', 91.0), ('exact path', 85.6),
                    ('shared soma', 77.8), ('grouped', 64.6),
                    ('reversed', 57.6), ('additive', 54.6)):
        value, colour = label_y[name]
        ax.text(3.16, y, name, fontsize=PT_BASE, color=colour, ha='left',
                va='center', zorder=6, clip_on=False)
        if abs(y - value) > 1.4:
            ax.plot([3.03, 3.13], [value, y], color=colour, lw=LW_HAIR,
                    zorder=1.8)
    row = contrast[contrast.contrast.eq(
        'depth__serial_bp__aligned__d4_d3')].iloc[0]
    f6_note_data(ax, 0.85, 97.5, ('four-tier cohort:', 'D4 − D3 = −1.46 pp',
                                  '[−1.74, −1.17],', '0 of 10 seeds'))
    f6_note_data(ax, 0.85, 47.5, ('n = 10 paired seeds;',
                                  '95 % bootstrap; 180 epochs',
                                  'seeds 10200–10209'))
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
    ('exact_autograd_bp_recipe', 1, 'ink', 'dashed', 'exact BP (D1)', True),
)
F6_DASH = {'solid': '-', 'dashed': (0, (4.2, 2.0)), 'dotted': (0, (1.2, 1.6))}
F6_XLIM = (0.0, 1071.0)
F6_XLAB = 622.0


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
            keep = x <= split
            ax.plot(x[keep], y[keep], color=colour, lw=lw, ls='-', zorder=2.4,
                    solid_capstyle='round')
            ax.plot(x[~keep], y[~keep], color=colour, lw=lw,
                    ls=F6_DASH['dotted'], zorder=2.4, solid_capstyle='round')
        else:
            ax.plot(x, y, color=colour, lw=lw, ls=F6_DASH[style], zorder=2.4,
                    solid_capstyle='round')
        ends[label] = (float(y[-1]), colour)
        rows.extend(dict(panel=panel, arm=arm, depth=depth, metric=metric,
                         epoch=int(e), mean=float(m), ci95_low=float(a),
                         ci95_high=float(b), n_seeds=10)
                    for e, m, a, b in zip(p.epoch, p['mean'], p.ci95_low,
                                          p.ci95_high)
                    if int(e) in (1, 60, 180, 300, 400, 486, 600))
    return ends


def f6_direct_ends(ax, ends, places, x, *, lead_from=None):
    """Direct labels in the panel's own right-hand label column, with leaders."""
    from journal_style import PT_BASE, LW_HAIR
    from credit_tree_schematics import mix
    for label, y in places.items():
        value, colour = ends[label]
        ax.text(x, y, label, fontsize=PT_BASE, color=colour, ha='left',
                va='center', zorder=6, clip_on=False)
        if lead_from is not None and abs(y - value) > 1e-9:
            ax.plot([lead_from, x - (x - lead_from) * 0.30], [value, y],
                    color=mix(colour, 55), lw=LW_HAIR, zorder=1.8)


def f6_accuracy_budget(ax, curves, stopping, budget, rows):
    """Fig 6E: validation-selected accuracy over the 600-epoch restarts."""
    ax.set(xlim=F6_XLIM, ylim=(32.0, 104.0), xticks=[0, 180, 400, 600],
           yticks=[50, 60, 70, 80, 90, 100], xlabel='Epoch')
    ax.tick_params(labelleft=False)
    f6_trim(ax, x=(0, 600), y=(50, 100))
    d1 = stopping[stopping.arm.eq('exact_autograd_bp_recipe')
                  & stopping.depth.eq(1)]
    runs = sorted(int(v) for v in d1.epochs_run)
    split = max(e for e in runs if sum(r >= e for r in runs) >= 8)
    ends = f6_trajectories(ax, curves, 'test_accuracy', 100.0, 'E', rows,
                           split=split)
    f6_ref(ax, 50.0, 'chance', at=1065.0, dy=-0.7, va='top', span=F6_XLIM)
    f6_budget_rule(ax, (50.0, 104.0))
    f6_direct_ends(ax, ends, {'exact BP (D3)': 99.4, 'broadcast (BP)': 93.8,
                              'shared soma': 88.2, 'broadcast (local)': 82.6,
                              'exact path': 77.0, 'exact BP (D1)': 61.2},
                   F6_XLAB, lead_from=600.0)
    gain180 = budget[budget.budget.eq(180)
                     & budget.contrast.eq('depth_gain_exact_bp')].iloc[0]
    gain600 = budget[budget.budget.eq(600)
                     & budget.contrast.eq('depth_gain_exact_bp')].iloc[0]
    bcast = budget[budget.budget.eq(600)
                   & budget.contrast.eq('bp_exact_minus_broadcast')].iloc[0]
    counts = ' / '.join(str(sum(r >= e for r in runs)) for e in (180, 400, 600))
    f6_note_data(ax, 10.0, 59.6,
                 ('D3 − D1: +30.82 pp at 180, +38.17 at 600',
                  'exact − broadcast (BP) at 600: +0.48 pp'))
    f6_note_data(ax, 10.0, 48.8,
                 ('[0.40, 0.55], 10 of 10 seeds',
                  'D1 fits still training at 180 / 400 / 600:',
                  '%s of 10;  n = 10 paired seeds' % counts))
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
    ax.set(xlim=F6_XLIM, ylim=(0.045, 0.84), xticks=[0, 180, 400, 600],
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
    ax.plot([330.0, 330.0], [0.700, 0.664], color=COLORS['point_mlp'],
            lw=LW_HAIR, zorder=1.8)
    rows.extend(dict(panel='F', arm='exact_autograd_bp_recipe', depth=1,
                     metric='stopping_epoch', epoch=int(e),
                     mean=float(track.loc[e]), n_seeds=10) for e in stops)
    f6_direct_ends(ax, ends, {'exact BP (D3)': 0.108, 'exact path': 0.214,
                              'shared soma': 0.320}, F6_XLAB, lead_from=600.0)
    f6_note_data(ax, 40.0, 0.835,
                 ('open circles: the eight D1 stopping',
                  'epochs, %d–%d; all 50 D3 fits reach' % (stops[0],
                                                           stops[-1]),
                  'the cap, loss still falling; n = 10'))
    f6_note_data(ax, 1065.0, 0.084, ('convergence not established',),
                 ha='right')
    return ax


def f6_paired(ax, paired, metric, rows, *, ylim, yticks, ylabel, marks,
              notes, sign, rule_span, band=None, zero_dy=0.0, zero_at=612.0,
              zero_va='bottom', zero_ha='right'):
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
    f6_sign_legend(ax, sign[0], sign[1], top_color=COLORS['bp'],
                   bottom_color=AMBER_TEXT)
    at = p.set_index('epoch')
    for epoch, tx, ty, ha, lines, leader in marks:
        value = float(at.loc[epoch, 'mean'])
        ax.plot([epoch], [value], marker='o', ms=MARKER_MS, mfc=COLORS['bp'],
                mec='white', mew=LW_EDGE, ls='none', zorder=3.4)
        if leader is not None:
            (lx0, ly0), (lx1, ly1) = leader
            ax.plot([lx0, lx1], [ly0, ly1], color=COLORS['bp'], lw=LW_HAIR,
                    zorder=1.8)
        f6_note_data(ax, tx, ty, lines, color=COLORS['ink'], ha=ha)
    for x, y, ha, lines in notes:
        f6_note_data(ax, x, y, lines, ha=ha)
    rows.extend(dict(panel='G' if metric == 'test_accuracy' else 'H',
                     metric=metric, epoch=int(e), mean=float(m),
                     ci95_low=float(a), ci95_high=float(b),
                     positive_seeds=int(ps), negative_seeds=int(ns),
                     n_seeds=10)
                for e, m, a, b, ps, ns in zip(p.epoch, p['mean'], p.ci95_low,
                                              p.ci95_high, p.positive_seeds,
                                              p.negative_seeds)
                if int(e) in (1, 60, 180, 300, 314, 315, 325, 400, 486, 600))
    return ax


F6_CAPTION = r'''\caption{\textbf{Task organization sets the forward benefit of serial dendrites, and the training budget sets the credit-rule ranking.}
\textbf{A}, The same eight nonsomatic compartments per soma in one stage (D1 $[8]$) or three (D3 $[2,1,2]$): blue, excitatory class-bearing contacts; carmine, inhibitory sensors, all tiers in D1, one tier per stage in D3; somatic error $\delta_0$; footer, the resource-identical grouped-point control. D2 $[2,3]$ and D4 $[1,1,2,2]$ keep that inventory: 66,178 parameters, 14,336 active synapses. Schematic, no data.
\textbf{B}, Generative model: nested gains act on 4, 2 and 1 blocks of the input stream, flat gains on eight blocks each, and local-ratio inputs carry the ratio inside every module (frozen task configuration, Methods); sensors report tier $\ell$ with fidelity $\alpha$. Schematic, no data.
\textbf{C}, Serial-minus-grouped-point test accuracy at fixed D3 under exact BP, by task family and fidelity (family colours are ordinal within this panel only); the local-ratio value at $\alpha=1$ is an exact tie in all ten pairs, not a sampled null (exact-path LocalCA counterpart, Supplementary Fig.~S22D).
\textbf{D}, Three-tier task: accuracy against serial depth for four credit arms, two resource-matched controls and the point-network ceiling, with the four-tier D4-minus-D3 contrast annotated; raw-additive means vary by up to 1.6 points across re-runs.
\textbf{E}, Validation-selected accuracy for six 600-epoch restarts of the same seeds, dotted for the autograd-broadcast variant of each rule; these are restarts, not checkpoint continuations, so the 180-epoch D3-minus-D1 gain reads 30.82 points here and 30.86 in the original budget. Counts of fits still training use $\text{epochs\_run}\ge$ epoch.
\textbf{F}, Best validation loss for the same six arms; open markers, the eight D1 stopping epochs.
\textbf{G,H}, Paired exact-path-minus-shared-soma differences in accuracy and cross-entropy; the band in \textbf{G} is where the pointwise interval straddles zero; the dotted rule in \textbf{E}--\textbf{H} marks the 180-epoch budget.
Points are means over ten paired training seeds; bars and shading are descriptive 95\% paired-seed bootstrap intervals, pointwise in \textbf{E}--\textbf{H}; accuracy endpoints are validation-selected states. Source Data: \texttt{source\_data/curated\_publication/figure\_06\_plotted.csv}.}'''


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
    """
    import build_main_figure_06 as depth
    from figure_canvas import enforce_tokens
    rows = []
    c = NativeCanvas(490 / 72, 3, row_weights=[116, 118, 116], hgutter_pt=37,
                     vgutter_pt=44,
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
    cc = c.panel('C', 0, 8, 4, title='Gains must be distributed')
    f6_style(cc, 'Gains must be distributed')
    f6_family_dose(cc, effects, seeds, rows)

    conf = read('nonlinear_physical_depth_confirmatory/condition_summary.csv')
    remaining = read('remaining_physical_experiments/condition_summary.csv')
    ceiling = read('point_dendrite_credit_controls/condition_summary.csv')
    contrast = read('physical_depth_h4_factorial/paired_contrasts.csv')
    d = c.panel('D', 1, 0, 4, title='Depth gain needs the tree')
    f6_style(d, 'Depth gain needs the tree')
    f6_depth_ladder(d, conf, remaining, ceiling, contrast, rows)

    curves = read('physical_depth_followup/condition_trajectory_summary.csv')
    stopping = read('physical_depth_followup/stopping_by_seed.csv')
    budget = read(
        'physical_depth_budget/canonical/extension_paired_contrasts.csv')
    e = c.panel('E', 1, 4, 4, title='D3 keeps its lead to 600', sharey=d)
    f6_style(e, 'D3 keeps its lead to 600')
    f6_accuracy_budget(e, curves, stopping, budget, rows)
    f = c.panel('F', 1, 8, 4, title='No plateau by 600 epochs', sharex=e)
    f6_style(f, 'No plateau by 600 epochs')
    f6_validation_loss(f, curves, stopping, rows)

    paired = read('physical_depth_followup/paired_trajectory_summary.csv')
    g = c.panel('G', 2, 0, 6, title='Accuracy ranking flips with budget')
    f6_style(g, 'Accuracy ranking flips with budget')
    f6_paired(g, paired, 'test_accuracy', rows, ylim=(-6.6, 13.5),
              yticks=[-5, 0, 5, 10],
              ylabel='Exact path − shared soma (pp)', band=(300, 330),
              sign=('exact path ahead', 'shared soma ahead'),
              rule_span=(-1.2, 13.5), zero_dy=0.45, zero_at=612.0,
              marks=((180, 205.0, 13.3, 'left',
                      ('at 180: +10.86 pp', '10 of 10 seeds positive'),
                      ((186.0, 11.5), (203.0, 12.6))),
                     (486, 300.0, -3.0, 'left', ('at 486: −2.94 pp',),
                      ((464.0, -3.6), (483.0, -3.05))),
                     (600, 612.0, 6.5, 'right',
                      ('at 600: −1.52 pp', '[−1.83, −1.23], 10 of 10'),
                      ((600.0, 3.8), (600.0, -1.1)))),
              notes=((250.0, 9.5, 'left',
                      ('crossing not resolved (300–330)',)),
                     (8.0, -1.6, 'left', ('n = 10 paired seeds;',
                                          'pointwise 95 % bootstrap;',
                                          'validation-selected states'))))
    h = c.panel('H', 2, 6, 6, title='Cross-entropy ordering does not flip',
                sharex=g)
    f6_style(h, 'Cross-entropy ordering does not flip')
    f6_paired(h, paired, 'test_cross_entropy', rows, ylim=(-0.098, 0.026),
              yticks=[-0.08, -0.04, 0.0],
              ylabel='Exact path − shared soma (nats)',
              sign=('exact path ahead', 'shared soma ahead'),
              rule_span=(-0.0775, 0.0010),
              zero_dy=-0.0020, zero_at=470.0, zero_va='top',
              marks=((180, 15.0, -0.0785, 'left',
                      ('−0.073 nats at 180, 10 of 10 seeds',),
                      ((180.0, -0.0757), (180.0, -0.0782))),
                     (600, 612.0, 0.0170, 'right',
                      ('−0.021 nats at 600 (0.254 vs 0.276), 9 of 10',
                       'advantage decays 3.4×; ordering does not reverse'),
                      ((600.0, 0.0010), (600.0, -0.0196)))),
              notes=((15.0, 0.0255, 'left',
                      ('below zero: lower loss with exact-path credit',)),
                     (15.0, -0.0862, 'left',
                      ('n = 10 paired seeds; 95 % bootstrap',))))

    for name in ('C', 'D', 'E', 'F', 'G', 'H'):
        c.declare_reserve(name, left=20.0)
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
    save(c, 6, sources, panels, F6_CAPTION, rows,
         {'helper_sha256': {str(Path(depth.__file__).relative_to(J)):
                            sha(depth.__file__)},
          'schematic_fraction_percent': 14.9,
          'schematic_fraction_formula':
              'sum(schematic slot w_pt*h_pt) / (live_w_pt*live_h_pt) = '
              '2*134.47*116 / (477.4*438)',
          'no_forest': 'CF-6: Fig 6 uses no forest panel (recorded)',
          'in_axes_key_count': 0})


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
    parser.add_argument('--figures',nargs='+',type=int,default=[1,4,6,7,9],choices=[1,4,6,7,9])
    parser.add_argument('--emit-main',action='store_true')
    args=parser.parse_args()
    for number in args.figures:
        globals()[f'figure{number}']()
        if args.emit_main:shutil.copyfile(OUT/f'restored_main_{number:02d}.pdf',J/f'figures/main/figure_{number:02d}.pdf')


if __name__=='__main__':main()
